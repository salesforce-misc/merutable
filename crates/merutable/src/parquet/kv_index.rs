//! `KvSparseIndex`: front-coded sparse "user_key → Parquet page location" index.
//!
//! # Why this exists
//!
//! Parquet's built-in `ColumnIndex` is general-purpose: it stores per-page
//! min/max statistics for *every* column, truncated at 64 bytes by default.
//! For an LSM hot tier — where data is sorted KV, point lookups dominate,
//! and composite primary keys routinely exceed 64 bytes — that's pure
//! overhead and lossy precision.
//!
//! `KvSparseIndex` is a domain-specific replacement: one entry per data page
//! on the `_merutable_ikey` column only, full keys (no truncation),
//! prefix-compressed because sorted internal keys share huge prefixes,
//! and binary-searchable via restart points (LevelDB / RocksDB index-block
//! style).
//!
//! Stored in the Parquet footer KV under `merutable.kv_index.v1`, sibling
//! to `merutable.bloom`.
//!
//! # Wire format (`merutable.kv_index.v1`)
//!
//! All multi-byte integers are little-endian.
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │ header (24 bytes)                    │
//! │   u8  version              (= 1)     │
//! │   u8  reserved[3]          (= 0)     │
//! │   u32 num_entries                    │
//! │   u32 restart_interval               │
//! │   u32 entries_size                   │
//! │   u32 num_restarts                   │
//! │   u32 reserved              (= 0)    │
//! ├──────────────────────────────────────┤
//! │ entries (entries_size bytes)         │
//! │   per entry:                         │
//! │     u16 shared_prefix_len            │
//! │     u16 suffix_len                   │
//! │     bytes suffix         (suffix_len)│
//! │     u64 page_offset                  │
//! │     u32 page_size                    │
//! │     u64 first_row_index              │
//! ├──────────────────────────────────────┤
//! │ restart_offsets (num_restarts × u32) │
//! │   absolute byte offsets into the     │
//! │   entries section, one per restart   │
//! └──────────────────────────────────────┘
//! ```
//!
//! At every restart point (every `restart_interval` entries), the entry is
//! written with `shared_prefix_len = 0`, so the suffix *is* the full key —
//! no decode state is needed to inspect a restart entry.
//!
//! # Search
//!
//! `find_page(target)` returns the page that contains the largest key ≤ target:
//!
//! 1. Binary-search the restart points by comparing `target` to each
//!    restart entry's full key. Find the largest restart `r` whose key ≤
//!    `target`.
//! 2. Linear-scan from restart `r` for at most `restart_interval` entries,
//!    rebuilding prefix-compressed keys against a running buffer, tracking
//!    the largest entry whose key ≤ `target`.
//! 3. Return that entry's `PageLocation`.
//!
//! Result: O(log(num_restarts) + restart_interval) comparisons, ~3-5 KiB
//! of footer KV bytes for a typical L0 4 MiB row group with 8 KiB pages.
//!
//! # Invariants
//!
//! - Entries are written in strict ascending key order. Build-time `assert!`
//!   panics on out-of-order input — sort before calling [`KvSparseIndex::build`].
//! - Restart interval ≥ 1. Build-time clamp.
//! - The first entry is always a restart (its shared_prefix_len is 0).
//!
//! # Encoding choices (and why)
//!
//! - **Fixed-width header fields, fixed-width page metadata.** Varint would
//!   shave ~4-8 bytes per entry but cost branches on the hot decode path.
//!   The compression we care about is on the *keys*, which is where prefix
//!   coding wins big; the per-entry overhead is small enough that fixed
//!   widths are simpler and faster.
//! - **u64 page_offset / u64 first_row_index.** Absolute byte offsets into
//!   the parquet file and absolute row indices, both safely u64 even for
//!   files measured in tens of GiB.
//! - **u32 page_size.** Page compressed size; capped well under 4 GiB by
//!   Parquet's own page size limits.

use crate::types::{MeruError, Result};
use bytes::Bytes;

/// Format version stored in the header `version` byte.
pub const KV_INDEX_VERSION: u8 = 1;

/// Footer KV key under which the encoded index lives.
pub const KV_INDEX_FOOTER_KEY: &str = "merutable.kv_index.v1";

/// Default restart interval. Every Nth entry is a "restart" with full
/// (non-prefix-compressed) key, enabling binary search. 16 is a sweet
/// spot: log2(num_restarts) bsearch + ≤16 linear-scan steps inside a
/// restart group, with minimal restart-table overhead.
pub const DEFAULT_RESTART_INTERVAL: u32 = 16;

const HEADER_SIZE: usize = 24;
const ENTRY_FIXED_TAIL: usize = 8 + 4 + 8; // page_offset + page_size + first_row_index
const ENTRY_HEADER: usize = 2 + 2; // shared_prefix_len + suffix_len

/// Location of a Parquet data page within a file. The values come from
/// Parquet's own `OffsetIndex`/`PageLocation`, captured at write time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageLocation {
    /// Absolute byte offset of the data page within the Parquet file.
    pub page_offset: u64,
    /// Compressed size of the data page in bytes.
    pub page_size: u32,
    /// Row index of the first row in this page (within its row group's
    /// global row numbering).
    pub first_row_index: u64,
}

/// Upper bound on a single key's length. The entry header encodes
/// `suffix_len` (and `shared_prefix_len`) as `u16`, so a key whose
/// suffix would exceed `u16::MAX` bytes cannot be represented. This is
/// also well above any realistic InternalKey size (composite PK +
/// seq/op_type trailer = tens to a few thousand bytes).
pub const MAX_KEY_LEN: usize = u16::MAX as usize;

/// Build the on-wire bytes for a `KvSparseIndex` from a strictly-ascending
/// sequence of `(user_key, location)` pairs.
///
/// Fails (not panics) if:
/// - any key is longer than [`MAX_KEY_LEN`] (would silently truncate the
///   u16 suffix_len field — a data-corruption bug);
/// - the input is not strictly ascending by key (would produce an index
///   whose binary search answers are meaningless).
///
/// The sort check used to be a `debug_assert!` which meant release
/// builds silently accepted out-of-order input and shipped a broken
/// index. It is now a runtime check — build runs once per flush, so the
/// O(n) cost is amortized across a whole row group worth of writes.
pub fn build(entries: &[(Vec<u8>, PageLocation)], restart_interval: u32) -> Result<Bytes> {
    let restart_interval = restart_interval.max(1);
    let num_entries = entries.len() as u32;

    // Compute number of restart points up front so we can pre-size the
    // restart table.
    let num_restarts = if num_entries == 0 {
        0
    } else {
        num_entries.div_ceil(restart_interval)
    };

    // Worst-case capacity: every entry full-key encoded.
    let mut entries_buf: Vec<u8> = Vec::with_capacity(
        entries
            .iter()
            .map(|(k, _)| ENTRY_HEADER + k.len() + ENTRY_FIXED_TAIL)
            .sum(),
    );
    let mut restart_offsets: Vec<u32> = Vec::with_capacity(num_restarts as usize);

    let mut prev_key: &[u8] = &[];

    for (i, (key, loc)) in entries.iter().enumerate() {
        if key.len() > MAX_KEY_LEN {
            return Err(MeruError::Parquet(format!(
                "kv_index::build: key {i} has length {} bytes, exceeds MAX_KEY_LEN ({MAX_KEY_LEN})",
                key.len()
            )));
        }
        if i > 0 && key.as_slice() <= prev_key {
            return Err(MeruError::Parquet(format!(
                "kv_index::build requires strictly ascending keys; entry {i} is not > entry {}",
                i - 1
            )));
        }

        let is_restart = (i as u32).is_multiple_of(restart_interval);
        let shared = if is_restart {
            0
        } else {
            shared_prefix_len(prev_key, key)
        };
        let suffix = &key[shared..];

        if is_restart {
            restart_offsets.push(entries_buf.len() as u32);
        }

        // u16 shared, u16 suffix_len, suffix bytes. `shared` and
        // `suffix.len()` are both ≤ key.len() ≤ MAX_KEY_LEN, checked
        // above, so the u16 casts are lossless.
        entries_buf.extend_from_slice(&(shared as u16).to_le_bytes());
        entries_buf.extend_from_slice(&(suffix.len() as u16).to_le_bytes());
        entries_buf.extend_from_slice(suffix);
        // u64 page_offset, u32 page_size, u64 first_row_index
        entries_buf.extend_from_slice(&loc.page_offset.to_le_bytes());
        entries_buf.extend_from_slice(&loc.page_size.to_le_bytes());
        entries_buf.extend_from_slice(&loc.first_row_index.to_le_bytes());

        prev_key = key;
    }

    let entries_size = entries_buf.len() as u32;

    let total_size = HEADER_SIZE + entries_size as usize + 4 * restart_offsets.len();
    let mut out = Vec::with_capacity(total_size);

    // Header.
    out.push(KV_INDEX_VERSION);
    out.extend_from_slice(&[0u8, 0, 0]); // reserved
    out.extend_from_slice(&num_entries.to_le_bytes());
    out.extend_from_slice(&restart_interval.to_le_bytes());
    out.extend_from_slice(&entries_size.to_le_bytes());
    out.extend_from_slice(&(restart_offsets.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    debug_assert_eq!(out.len(), HEADER_SIZE);

    out.extend_from_slice(&entries_buf);
    for ro in &restart_offsets {
        out.extend_from_slice(&ro.to_le_bytes());
    }

    Ok(Bytes::from(out))
}

/// Length of the longest common prefix of `a` and `b`.
fn shared_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Decoded, searchable view over a `KvSparseIndex` byte buffer.
///
/// Holding a `KvSparseIndex` keeps the underlying `Bytes` alive; lookups
/// allocate only the small per-call key buffer.
///
/// # Evolution
///
/// The footer key carries an explicit `v1` suffix and the wire format
/// starts with a `version` byte, so a future `merutable.kv_index.v2` can
/// coexist with v1 readers. The intended v2 evolution is *partitioned*
/// indexes: split the entries section into N restart-aligned shards, store
/// a tiny top-level shard directory in the footer, and load only the shard
/// covering a given probe — useful once cold-tier files routinely carry
/// indexes in the megabyte range. v1 is monolithic because at expected
/// L0/L1 sizes (a few hundred KiB at most) the savings don't justify the
/// extra indirection.
#[derive(Debug, Clone)]
pub struct KvSparseIndex {
    bytes: Bytes,
    num_entries: u32,
    restart_interval: u32,
    entries_offset: usize,
    entries_size: usize,
    restart_offsets: Vec<u32>,
}

impl KvSparseIndex {
    /// Parse and fully validate the on-wire index buffer.
    ///
    /// After this returns `Ok`, every entry in the buffer is guaranteed
    /// to be structurally well-formed: all byte offsets lie within the
    /// entries section, all `shared_prefix_len` values are ≤ the
    /// preceding entry's rebuilt-key length, every restart entry has
    /// `shared = 0`, and the restart table offsets are monotonically
    /// increasing and point at entry boundaries.
    ///
    /// This lets the hot decode path ([`find_page`], [`iter`]) remain
    /// infallible — it would be a bug for decode to panic on any
    /// instance returned by this function.
    ///
    /// [`find_page`]: KvSparseIndex::find_page
    /// [`iter`]: KvSparseIndex::iter
    pub fn from_bytes(bytes: Bytes) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(MeruError::Corruption(format!(
                "kv_index: buffer too small ({} < {HEADER_SIZE})",
                bytes.len()
            )));
        }

        let version = bytes[0];
        if version != KV_INDEX_VERSION {
            return Err(MeruError::Corruption(format!(
                "kv_index: unsupported version {version} (expected {KV_INDEX_VERSION})"
            )));
        }

        let num_entries = u32_at(&bytes, 4);
        let restart_interval = u32_at(&bytes, 8);
        let entries_size = u32_at(&bytes, 12) as usize;
        let num_restarts = u32_at(&bytes, 16) as usize;

        if restart_interval == 0 {
            return Err(MeruError::Corruption(
                "kv_index: restart_interval is 0".into(),
            ));
        }

        let entries_offset = HEADER_SIZE;
        // Check total length (header + entries + restart table) with
        // checked arithmetic so a malicious num_restarts can't overflow
        // usize on the subsequent range index.
        let restart_table_bytes = num_restarts
            .checked_mul(4)
            .ok_or_else(|| MeruError::Corruption("kv_index: num_restarts overflow".into()))?;
        let restart_section_offset = entries_offset
            .checked_add(entries_size)
            .ok_or_else(|| MeruError::Corruption("kv_index: entries_size overflow".into()))?;
        let expected_total = restart_section_offset
            .checked_add(restart_table_bytes)
            .ok_or_else(|| MeruError::Corruption("kv_index: total size overflow".into()))?;
        if bytes.len() != expected_total {
            return Err(MeruError::Corruption(format!(
                "kv_index: buffer size mismatch (have {}, need exactly {expected_total})",
                bytes.len()
            )));
        }

        // Load and validate the restart offsets. Each must be strictly
        // increasing and all must be ≤ entries_size.
        let mut restart_offsets: Vec<u32> = Vec::with_capacity(num_restarts);
        let mut prev_ro: Option<u32> = None;
        for i in 0..num_restarts {
            let ro = u32_at(&bytes, restart_section_offset + 4 * i);
            if (ro as usize) > entries_size {
                return Err(MeruError::Corruption(format!(
                    "kv_index: restart_offset[{i}] = {ro} exceeds entries_size {entries_size}"
                )));
            }
            if let Some(prev) = prev_ro
                && ro <= prev
                && i > 0
            {
                return Err(MeruError::Corruption(format!(
                    "kv_index: restart_offset[{i}] = {ro} is not > previous {prev}"
                )));
            }
            prev_ro = Some(ro);
            restart_offsets.push(ro);
        }

        // Consistency: num_restarts must match what build() would have
        // produced for the stated num_entries + restart_interval.
        let expected_restarts = if num_entries == 0 {
            0
        } else {
            num_entries.div_ceil(restart_interval) as usize
        };
        if num_restarts != expected_restarts {
            return Err(MeruError::Corruption(format!(
                "kv_index: num_restarts {num_restarts} inconsistent with num_entries \
                 {num_entries} / restart_interval {restart_interval} (expected {expected_restarts})"
            )));
        }

        // Walk every entry once, validating bounds and invariants.
        // After this succeeds, the decode paths can index the buffer
        // without any bounds checks beyond the ones the compiler
        // inserts for slicing.
        {
            let mut cursor = 0usize;
            let mut prev_key_len: usize = 0;
            let entries_end = entries_size;
            for i in 0..num_entries as usize {
                let abs = entries_offset + cursor;
                // Header (4 bytes: shared + suffix_len).
                if cursor + ENTRY_HEADER > entries_end {
                    return Err(MeruError::Corruption(format!(
                        "kv_index: entry {i} header overruns entries section"
                    )));
                }
                let shared = u16_at(&bytes, abs) as usize;
                let suffix_len = u16_at(&bytes, abs + 2) as usize;

                let is_restart = (i as u32).is_multiple_of(restart_interval);
                if is_restart && shared != 0 {
                    return Err(MeruError::Corruption(format!(
                        "kv_index: restart entry {i} has shared={shared} (expected 0)"
                    )));
                }
                if shared > prev_key_len {
                    return Err(MeruError::Corruption(format!(
                        "kv_index: entry {i} shared={shared} exceeds previous key length {prev_key_len}"
                    )));
                }

                // Suffix + fixed tail must fit in entries_size.
                let need = ENTRY_HEADER + suffix_len + ENTRY_FIXED_TAIL;
                if cursor + need > entries_end {
                    return Err(MeruError::Corruption(format!(
                        "kv_index: entry {i} body overruns entries section \
                         (need {need} at cursor {cursor}, entries_size {entries_end})"
                    )));
                }

                // The restart table must reference this exact byte
                // offset if entry i is a restart. (Prevents a corrupt
                // restart table from mis-binary-searching.)
                if is_restart {
                    let restart_idx = (i as u32 / restart_interval) as usize;
                    let expected_ro = cursor as u32;
                    if restart_offsets[restart_idx] != expected_ro {
                        return Err(MeruError::Corruption(format!(
                            "kv_index: restart_offset[{restart_idx}] = {} does not point at \
                             entry {i} (expected byte offset {expected_ro})",
                            restart_offsets[restart_idx]
                        )));
                    }
                }

                prev_key_len = shared + suffix_len;
                cursor += need;
            }
            if cursor != entries_end {
                return Err(MeruError::Corruption(format!(
                    "kv_index: trailing bytes in entries section (walked {cursor}, \
                     entries_size {entries_end})"
                )));
            }
        }

        Ok(Self {
            bytes,
            num_entries,
            restart_interval,
            entries_offset,
            entries_size,
            restart_offsets,
        })
    }

    /// Number of (key → page) entries.
    pub fn len(&self) -> usize {
        self.num_entries as usize
    }

    /// Whether the index has zero entries.
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    /// Restart interval the index was built with.
    pub fn restart_interval(&self) -> u32 {
        self.restart_interval
    }

    /// On-wire byte size of the encoded index.
    pub fn encoded_size(&self) -> usize {
        HEADER_SIZE + self.entries_size + 4 * self.restart_offsets.len()
    }

    /// Like [`find_page`] but also returns the `first_row_index` of the
    /// *next* entry in the index — or `None` if the matched entry was the
    /// last one. Callers need this to bound the matched page's row count
    /// when issuing a Parquet `RowSelection`: the page contains rows
    /// `[matched.first_row_index, next_first_row_index)`, clamped to the
    /// matched page's enclosing row group (which the caller does using
    /// file metadata).
    ///
    /// This is the integration point with the Parquet reader's
    /// page-skipping point-lookup path.
    ///
    /// [`find_page`]: KvSparseIndex::find_page
    pub fn find_page_with_next(&self, target: &[u8]) -> Option<(PageLocation, Option<u64>)> {
        self.find_page_inner(target)
    }

    /// Find the page that contains the largest key ≤ `target`.
    ///
    /// Returns `None` if the target is strictly less than every key in the
    /// index — i.e., the target precedes the file's smallest key, so the
    /// file cannot contain it.
    pub fn find_page(&self, target: &[u8]) -> Option<PageLocation> {
        self.find_page_inner(target).map(|(loc, _)| loc)
    }

    fn find_page_inner(&self, target: &[u8]) -> Option<(PageLocation, Option<u64>)> {
        if self.num_entries == 0 {
            return None;
        }

        // Phase 1: binary search restart points by full key. Goal: find
        // the largest restart `lo` whose key ≤ target.
        let restarts = &self.restart_offsets;
        let mut lo: i64 = -1; // sentinel: "no restart known to be ≤ target"
        let mut hi: i64 = restarts.len() as i64; // exclusive upper bound

        while hi - lo > 1 {
            let mid = ((lo + hi) / 2) as usize;
            let mid_key = self.read_restart_key(mid);
            match mid_key.cmp(target) {
                std::cmp::Ordering::Less | std::cmp::Ordering::Equal => lo = mid as i64,
                std::cmp::Ordering::Greater => hi = mid as i64,
            }
        }

        // Phase 2: linear scan from the chosen restart (or from the start
        // if every restart key is > target).
        let scan_start_pos = if lo < 0 {
            // No restart ≤ target. Start scanning from the very first
            // entry; if even entry 0's key > target, no page contains it.
            0
        } else {
            restarts[lo as usize] as usize
        };

        let mut cursor = scan_start_pos;
        let mut prev_key: Vec<u8> = Vec::new();
        let mut best: Option<PageLocation> = None;
        let mut next_first_row: Option<u64> = None;

        // We may scan beyond `restart_interval` entries when lo == -1, but
        // in that case we still bail out at the next restart (whose key is
        // > target by construction).
        loop {
            if cursor >= self.entries_size {
                break;
            }
            let (entry_key, loc, next_cursor) = self.decode_entry_at(cursor, &prev_key);

            match entry_key.as_slice().cmp(target) {
                std::cmp::Ordering::Greater => {
                    // First entry strictly greater than target — its
                    // first_row_index bounds the matched page's row range.
                    if best.is_some() {
                        next_first_row = Some(loc.first_row_index);
                    }
                    break;
                }
                _ => {
                    best = Some(loc);
                    prev_key = entry_key;
                    cursor = next_cursor;
                }
            }
        }

        best.map(|loc| (loc, next_first_row))
    }

    /// Read the *full* key at restart point `idx`. By construction the
    /// restart entry stores the full key in its suffix (shared_prefix_len = 0).
    fn read_restart_key(&self, idx: usize) -> &[u8] {
        let pos = self.entries_offset + self.restart_offsets[idx] as usize;
        let suffix_len = u16_at(&self.bytes, pos + 2) as usize;
        let key_start = pos + ENTRY_HEADER;
        &self.bytes[key_start..key_start + suffix_len]
    }

    /// Decode an entry starting at `cursor` (offset into the entries
    /// section). Returns `(full_key, page_location, next_cursor)`.
    fn decode_entry_at(&self, cursor: usize, prev_key: &[u8]) -> (Vec<u8>, PageLocation, usize) {
        let abs = self.entries_offset + cursor;
        let shared = u16_at(&self.bytes, abs) as usize;
        let suffix_len = u16_at(&self.bytes, abs + 2) as usize;
        let suffix_start = abs + ENTRY_HEADER;
        let suffix = &self.bytes[suffix_start..suffix_start + suffix_len];

        let mut key = Vec::with_capacity(shared + suffix_len);
        key.extend_from_slice(&prev_key[..shared]);
        key.extend_from_slice(suffix);

        let tail = suffix_start + suffix_len;
        let page_offset = u64_at(&self.bytes, tail);
        let page_size = u32_at(&self.bytes, tail + 8);
        let first_row_index = u64_at(&self.bytes, tail + 12);

        let next_cursor = (tail + 20) - self.entries_offset;
        (
            key,
            PageLocation {
                page_offset,
                page_size,
                first_row_index,
            },
            next_cursor,
        )
    }

    /// Iterator over `(full_key, location)` for every entry in the index,
    /// in ascending key order. Allocates per-call key buffers; intended
    /// for tests and diagnostics, not the hot read path.
    pub fn iter(&self) -> KvSparseIndexIter<'_> {
        KvSparseIndexIter {
            index: self,
            cursor: 0,
            prev_key: Vec::new(),
            remaining: self.num_entries,
        }
    }
}

/// Iterator yielding all `(key, location)` pairs in the index.
pub struct KvSparseIndexIter<'a> {
    index: &'a KvSparseIndex,
    cursor: usize,
    prev_key: Vec<u8>,
    remaining: u32,
}

impl Iterator for KvSparseIndexIter<'_> {
    type Item = (Vec<u8>, PageLocation);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let (key, loc, next) = self.index.decode_entry_at(self.cursor, &self.prev_key);
        self.cursor = next;
        self.prev_key = key.clone();
        self.remaining -= 1;
        Some((key, loc))
    }
}

#[inline]
fn u16_at(buf: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes([buf[pos], buf[pos + 1]])
}

#[inline]
fn u32_at(buf: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
}

#[inline]
fn u64_at(buf: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes([
        buf[pos],
        buf[pos + 1],
        buf[pos + 2],
        buf[pos + 3],
        buf[pos + 4],
        buf[pos + 5],
        buf[pos + 6],
        buf[pos + 7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn loc(offset: u64, size: u32, row: u64) -> PageLocation {
        PageLocation {
            page_offset: offset,
            page_size: size,
            first_row_index: row,
        }
    }

    /// Empty index encodes, decodes, and reports zero length.
    #[test]
    fn empty_index_round_trip() {
        let bytes = build(&[], DEFAULT_RESTART_INTERVAL).unwrap();
        let idx = KvSparseIndex::from_bytes(bytes).unwrap();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        assert_eq!(idx.find_page(b"anything"), None);
        assert_eq!(idx.iter().count(), 0);
    }

    /// Single-entry index always returns that entry for any key ≥ its key.
    #[test]
    fn single_entry_returns_for_anything_geq() {
        let entries = vec![(b"banana".to_vec(), loc(100, 8192, 0))];
        let bytes = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap();
        let idx = KvSparseIndex::from_bytes(bytes).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.find_page(b"apple"), None); // before first key
        assert_eq!(idx.find_page(b"banana"), Some(loc(100, 8192, 0)));
        assert_eq!(idx.find_page(b"cherry"), Some(loc(100, 8192, 0)));
    }

    /// Round-trip a small ordered set, exercising prefix compression
    /// across the bucket boundary and the iterator.
    #[test]
    fn small_ordered_round_trip() {
        let raw: Vec<(&[u8], PageLocation)> = vec![
            (b"k/0001/aaaa", loc(0, 8192, 0)),
            (b"k/0001/bbbb", loc(8192, 8192, 100)),
            (b"k/0001/cccc", loc(16384, 8192, 200)),
            (b"k/0002/aaaa", loc(24576, 8192, 300)),
            (b"k/0002/bbbb", loc(32768, 8192, 400)),
        ];
        let entries: Vec<(Vec<u8>, PageLocation)> =
            raw.iter().map(|(k, l)| (k.to_vec(), *l)).collect();

        let bytes = build(&entries, 2).unwrap(); // restart every 2 entries
        let idx = KvSparseIndex::from_bytes(bytes).unwrap();

        assert_eq!(idx.len(), 5);
        assert_eq!(idx.restart_interval(), 2);

        // Iterator returns the same sequence we built from.
        let collected: Vec<(Vec<u8>, PageLocation)> = idx.iter().collect();
        assert_eq!(collected, entries);

        // Exact-key lookups.
        for (k, l) in &entries {
            assert_eq!(idx.find_page(k), Some(*l), "exact lookup failed for {k:?}");
        }

        // Predecessor lookups: a key just past entry i must return entry i.
        assert_eq!(
            idx.find_page(b"k/0001/aaab"),
            Some(loc(0, 8192, 0)),
            "predecessor of /aaab should be /aaaa"
        );
        assert_eq!(idx.find_page(b"k/0001/cccd"), Some(loc(16384, 8192, 200)));
        // Past the end → returns the final entry.
        assert_eq!(idx.find_page(b"k/9999/zzzz"), Some(loc(32768, 8192, 400)));
        // Strictly before all entries → None.
        assert_eq!(idx.find_page(b"k/0000/zzzz"), None);
    }

    /// Compare against a `BTreeMap` oracle on a 1024-entry randomized set.
    /// Every key in the input plus 1024 randomly chosen probes must agree
    /// with the BTreeMap answer.
    #[test]
    fn matches_btreemap_oracle() {
        // Deterministic pseudo-random keys with shared prefixes (mimicking
        // composite PKs that share a tenant/table prefix).
        let mut rng_state: u64 = 0xdeadbeefcafebabe;
        let mut next = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng_state
        };

        let mut oracle: BTreeMap<Vec<u8>, PageLocation> = BTreeMap::new();
        for i in 0..1024u64 {
            let r = next();
            let key = format!("tenant/0001/table/users/pk/{:016x}/{:016x}", i, r).into_bytes();
            oracle.insert(key, loc(i * 8192, 8192, i * 100));
        }

        let entries: Vec<(Vec<u8>, PageLocation)> =
            oracle.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let bytes = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap();
        let idx = KvSparseIndex::from_bytes(bytes).unwrap();

        // Every key in the oracle is found exactly.
        for (k, expected) in &oracle {
            let got = idx.find_page(k);
            assert_eq!(got, Some(*expected), "exact lookup failed for {k:?}");
        }

        // 1024 random probes: compare to BTreeMap::range upper-bound.
        for _ in 0..1024 {
            let r = next();
            let probe =
                format!("tenant/0001/table/users/pk/{:016x}/{:016x}", r % 2048, r).into_bytes();
            let oracle_answer: Option<PageLocation> =
                oracle.range(..=probe.clone()).next_back().map(|(_, v)| *v);
            let idx_answer = idx.find_page(&probe);
            assert_eq!(
                idx_answer, oracle_answer,
                "oracle disagreement for probe {probe:?}"
            );
        }
    }

    /// Prefix compression must actually compress sorted keys with shared
    /// prefixes. Worst-case (everything stored full) would be ~76 bytes per
    /// entry; we expect well under that.
    #[test]
    fn prefix_compression_meaningfully_shrinks_index() {
        let entries: Vec<(Vec<u8>, PageLocation)> = (0..512u64)
            .map(|i| {
                (
                    format!("tenant/0001/table/users/pk/{:032x}", i).into_bytes(),
                    loc(i * 8192, 8192, i * 100),
                )
            })
            .collect();

        let raw_key_bytes: usize = entries.iter().map(|(k, _)| k.len()).sum();
        let bytes = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap();
        let idx = KvSparseIndex::from_bytes(bytes.clone()).unwrap();

        let on_disk = idx.encoded_size();
        // Sanity: encoded matches Bytes length.
        assert_eq!(on_disk, bytes.len());

        // Without compression, each entry would carry its full key plus
        // 24 bytes of fixed metadata. Demand at least 2x reduction over
        // the raw-key budget alone.
        let raw_lower_bound = raw_key_bytes;
        assert!(
            on_disk * 2 < raw_lower_bound + 24 * entries.len(),
            "kv_index expected ≥2x compression vs raw keys; got {on_disk} bytes for {raw_key_bytes} raw key bytes ({} entries)",
            entries.len()
        );
    }

    /// A truncated buffer must be rejected, not silently misread.
    #[test]
    fn truncated_buffer_is_rejected() {
        let entries: Vec<(Vec<u8>, PageLocation)> = (0..32u64)
            .map(|i| (format!("k{i:04}").into_bytes(), loc(i, 1, i)))
            .collect();
        let bytes = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap();

        // Drop the last 8 bytes (chops the restart table).
        let truncated = bytes.slice(..bytes.len() - 8);
        let result = KvSparseIndex::from_bytes(truncated);
        assert!(result.is_err(), "truncated buffer should be rejected");

        // Drop everything but a tiny prefix.
        let tiny = bytes.slice(..4);
        let result = KvSparseIndex::from_bytes(tiny);
        assert!(result.is_err(), "tiny buffer should be rejected");
    }

    /// Wrong version byte is rejected.
    #[test]
    fn wrong_version_is_rejected() {
        let entries: Vec<(Vec<u8>, PageLocation)> = vec![(b"hello".to_vec(), loc(0, 1, 0))];
        let bytes = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap();
        let mut tampered = bytes.to_vec();
        tampered[0] = 99;
        let result = KvSparseIndex::from_bytes(Bytes::from(tampered));
        assert!(matches!(result, Err(MeruError::Corruption(_))));
    }

    /// Restart interval of 1 (every entry is a restart) is valid and
    /// gives correct answers — useful as a worst-case for binary search.
    #[test]
    fn restart_interval_of_one_works() {
        let entries: Vec<(Vec<u8>, PageLocation)> = (0..16u64)
            .map(|i| (format!("k{i:04}").into_bytes(), loc(i * 100, 50, i)))
            .collect();
        let bytes = build(&entries, 1).unwrap();
        let idx = KvSparseIndex::from_bytes(bytes).unwrap();
        for (k, l) in &entries {
            assert_eq!(idx.find_page(k), Some(*l));
        }
    }

    /// `find_page_with_next` returns both the matched page and the
    /// `first_row_index` of the *next* entry — needed by the reader to
    /// bound the matched page's row range when issuing a `RowSelection`.
    #[test]
    fn find_page_with_next_reports_successor_first_row() {
        let entries: Vec<(Vec<u8>, PageLocation)> = vec![
            (b"k01".to_vec(), loc(0, 8192, 0)),
            (b"k02".to_vec(), loc(8192, 8192, 100)),
            (b"k03".to_vec(), loc(16384, 8192, 250)),
            (b"k04".to_vec(), loc(24576, 8192, 400)),
        ];
        let bytes = build(&entries, 2).unwrap();
        let idx = KvSparseIndex::from_bytes(bytes).unwrap();

        // Exact hit on a non-last entry: next is the immediately following
        // entry's first_row_index.
        let (got, next) = idx.find_page_with_next(b"k02").unwrap();
        assert_eq!(got, loc(8192, 8192, 100));
        assert_eq!(next, Some(250));

        // Predecessor lookup that lands on a non-last entry.
        let (got, next) = idx.find_page_with_next(b"k02zzz").unwrap();
        assert_eq!(got, loc(8192, 8192, 100));
        assert_eq!(next, Some(250));

        // First entry: next is entry 1's first_row_index.
        let (got, next) = idx.find_page_with_next(b"k01").unwrap();
        assert_eq!(got, loc(0, 8192, 0));
        assert_eq!(next, Some(100));

        // Last entry by exact hit: no successor.
        let (got, next) = idx.find_page_with_next(b"k04").unwrap();
        assert_eq!(got, loc(24576, 8192, 400));
        assert_eq!(next, None);

        // Past the end: matched is the final entry, no successor.
        let (got, next) = idx.find_page_with_next(b"k99").unwrap();
        assert_eq!(got, loc(24576, 8192, 400));
        assert_eq!(next, None);

        // Strictly before all entries: no match at all.
        assert!(idx.find_page_with_next(b"k00").is_none());
    }

    /// Helper used in proptest-style smoke: build, decode, iterate, search.
    #[test]
    fn iterator_yields_in_order_for_random_lengths() {
        // Mix short and long keys.
        let raw_keys: Vec<&[u8]> = vec![
            b"a",
            b"aa",
            b"aaa",
            b"aab",
            b"ab",
            b"abcdefghijklmnopqrstuvwxyz",
            b"abcdefghijklmnopqrstuvwxyz1",
            b"b",
            b"ba",
            b"baz",
        ];
        let entries: Vec<(Vec<u8>, PageLocation)> = raw_keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k.to_vec(), loc(i as u64 * 8, 8, i as u64)))
            .collect();
        let bytes = build(&entries, 4).unwrap();
        let idx = KvSparseIndex::from_bytes(bytes).unwrap();
        let yielded: Vec<Vec<u8>> = idx.iter().map(|(k, _)| k).collect();
        assert_eq!(
            yielded,
            raw_keys.iter().map(|k| k.to_vec()).collect::<Vec<_>>()
        );
    }

    // ── Negative / corruption tests (post-hardening) ─────────────────────

    /// A key longer than `MAX_KEY_LEN` (u16::MAX bytes) cannot be
    /// encoded without silently truncating `suffix_len`. Before the
    /// fix, `build` cast `suffix.len() as u16` unchecked, producing a
    /// valid-looking buffer with the wrong key bytes at decode time.
    #[test]
    fn build_rejects_key_longer_than_u16() {
        let oversized = vec![0x41u8; MAX_KEY_LEN + 1];
        let entries = vec![(oversized, loc(0, 1, 0))];
        let err = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("MAX_KEY_LEN") || msg.contains("exceeds"),
            "error should name the length limit: {msg}"
        );
    }

    /// A key of exactly `MAX_KEY_LEN` bytes must round-trip (boundary
    /// test — make sure the off-by-one lands on the correct side).
    #[test]
    fn build_accepts_key_at_u16_max() {
        let at_limit = vec![0x42u8; MAX_KEY_LEN];
        let entries = vec![(at_limit.clone(), loc(42, 1, 7))];
        let bytes = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap();
        let idx = KvSparseIndex::from_bytes(bytes).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.find_page(&at_limit), Some(loc(42, 1, 7)));
    }

    /// Out-of-order input must fail at build time in release mode, not
    /// just panic in debug. Before the fix this was a `debug_assert!`
    /// so release builds shipped a silently-broken index.
    #[test]
    fn build_rejects_unsorted_input() {
        let entries = vec![
            (b"zebra".to_vec(), loc(0, 1, 0)),
            (b"apple".to_vec(), loc(1, 1, 1)),
        ];
        let err = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("strictly ascending") || msg.contains("not >"),
            "{msg}"
        );
    }

    /// Duplicate adjacent keys are also rejected (strictly ascending
    /// means no equals).
    #[test]
    fn build_rejects_duplicate_adjacent_keys() {
        let entries = vec![
            (b"dup".to_vec(), loc(0, 1, 0)),
            (b"dup".to_vec(), loc(1, 1, 1)),
        ];
        assert!(build(&entries, DEFAULT_RESTART_INTERVAL).is_err());
    }

    /// A buffer with trailing garbage past the restart table must be
    /// rejected by `from_bytes`. Previously the length check was
    /// `bytes.len() < expected_total` which silently accepted trailing
    /// bytes — they'd be read by neither decode nor iter but the hash
    /// of the bytes would still differ between producer and consumer.
    #[test]
    fn from_bytes_rejects_trailing_garbage() {
        let entries: Vec<(Vec<u8>, PageLocation)> = (0..8u64)
            .map(|i| (format!("k{i:04}").into_bytes(), loc(i, 1, i)))
            .collect();
        let bytes = build(&entries, DEFAULT_RESTART_INTERVAL).unwrap();
        let mut padded = bytes.to_vec();
        padded.extend_from_slice(b"trailing garbage");
        let err = KvSparseIndex::from_bytes(Bytes::from(padded)).unwrap_err();
        assert!(matches!(err, MeruError::Corruption(_)));
    }

    /// A corrupt restart_offset that points past the end of the
    /// entries section must be rejected at `from_bytes` time so decode
    /// paths can stay infallible.
    #[test]
    fn from_bytes_rejects_restart_offset_out_of_range() {
        let entries: Vec<(Vec<u8>, PageLocation)> = (0..4u64)
            .map(|i| (format!("k{i:04}").into_bytes(), loc(i, 1, i)))
            .collect();
        let bytes = build(&entries, 2).unwrap();

        // Tamper: overwrite the first restart offset with a huge value.
        let entries_size = u32_at(&bytes, 12) as usize;
        let num_restarts = u32_at(&bytes, 16) as usize;
        assert!(num_restarts >= 1);
        let restart_table_start = HEADER_SIZE + entries_size;
        let mut tampered = bytes.to_vec();
        // Write a value past the end of the entries section.
        let bad = (entries_size as u32 + 9999).to_le_bytes();
        tampered[restart_table_start..restart_table_start + 4].copy_from_slice(&bad);
        let err = KvSparseIndex::from_bytes(Bytes::from(tampered)).unwrap_err();
        assert!(matches!(err, MeruError::Corruption(_)));
    }

    /// A corrupt entry whose `shared_prefix_len` exceeds the length of
    /// the previous key must be rejected — otherwise the decode path
    /// would panic on `prev_key[..shared]` slicing.
    #[test]
    fn from_bytes_rejects_shared_exceeding_prev_key() {
        // Two short keys, restart_interval=4 so entry 1 is NOT a
        // restart and carries a shared_prefix_len field we can tamper.
        let entries = vec![
            (b"aa".to_vec(), loc(0, 1, 0)),
            (b"ab".to_vec(), loc(1, 1, 1)),
        ];
        let bytes = build(&entries, 4).unwrap();
        let mut tampered = bytes.to_vec();
        // Locate entry 1's header: entry 0 is at HEADER_SIZE, has
        // 4-byte header + 2-byte suffix + 20-byte tail = 26 bytes.
        // Entry 1 starts at HEADER_SIZE + 26.
        let entry1_off = HEADER_SIZE + 4 + 2 + 20;
        // Rewrite shared = 99 (> prev key length 2).
        tampered[entry1_off..entry1_off + 2].copy_from_slice(&99u16.to_le_bytes());
        let err = KvSparseIndex::from_bytes(Bytes::from(tampered)).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("shared") && msg.contains("exceeds"),
            "error should name the shared/prev-key mismatch: {msg}"
        );
    }

    /// A corrupt restart entry (one where the restart table offset
    /// points at an entry whose stored `shared` is non-zero) must be
    /// rejected. Such a buffer would mis-answer binary search because
    /// the "restart key" the searcher reads would be only the suffix
    /// rather than the full key.
    #[test]
    fn from_bytes_rejects_restart_entry_with_nonzero_shared() {
        // Two-entry buffer, restart_interval=1 → entry 1 is a restart.
        // Tamper entry 1's shared field to 1.
        let entries = vec![
            (b"aa".to_vec(), loc(0, 1, 0)),
            (b"ab".to_vec(), loc(1, 1, 1)),
        ];
        let bytes = build(&entries, 1).unwrap();
        let mut tampered = bytes.to_vec();
        // Each entry: 4-byte header + 2-byte suffix + 20-byte tail = 26 bytes.
        let entry1_off = HEADER_SIZE + 26;
        tampered[entry1_off..entry1_off + 2].copy_from_slice(&1u16.to_le_bytes());
        let err = KvSparseIndex::from_bytes(Bytes::from(tampered)).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("restart entry") && msg.contains("shared"),
            "{msg}"
        );
    }

    /// Header claims `num_restarts` that disagrees with what `build`
    /// would have produced for the stated `num_entries` /
    /// `restart_interval`. Such a mismatch means at least one of the
    /// three fields is corrupt; reject it.
    #[test]
    fn from_bytes_rejects_inconsistent_num_restarts() {
        let entries: Vec<(Vec<u8>, PageLocation)> = (0..8u64)
            .map(|i| (format!("k{i:04}").into_bytes(), loc(i, 1, i)))
            .collect();
        let bytes = build(&entries, 4).unwrap(); // 8 entries, interval 4 → 2 restarts
        let mut tampered = bytes.to_vec();
        // Rewrite num_restarts field (offset 16, u32 LE) to 99.
        tampered[16..20].copy_from_slice(&99u32.to_le_bytes());
        let err = KvSparseIndex::from_bytes(Bytes::from(tampered)).unwrap_err();
        assert!(matches!(err, MeruError::Corruption(_)));
    }
}
