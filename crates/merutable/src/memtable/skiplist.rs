//! `MemtableSkipList`: lock-free sorted map keyed by encoded `InternalKey` bytes.
//!
//! We use `crossbeam_skiplist::SkipMap<bytes::Bytes, EntryValue>`.
//! Keys are the raw wire bytes of `InternalKey`, which sort correctly via
//! `bytes::Bytes`'s lexicographic `Ord` implementation (guaranteed by the
//! InternalKey encoding — see `merutable-types/src/key.rs`).
//!
//! # Point lookup algorithm
//!
//! Given `user_key_bytes` (the PK-encoded portion, without tag) and `read_seq`:
//!
//! 1. Build a `seek_bytes` = `user_key_bytes ++ seek_tag(SEQNUM_MAX)`.
//!    This is the lexicographically smallest InternalKey for that PK (tag = 0x0000000000000001).
//! 2. `map.lower_bound(Included(&seek_bytes))` → first entry ≥ seek key.
//! 3. Verify entry's PK prefix matches `user_key_bytes`.
//! 4. Decode `seq` from the entry's tag; reject if `seq > read_seq`.
//! 5. Return `Some(entry.value())` or `None`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::types::sequence::{OpType, SEQNUM_MAX, SeqNum};
use bytes::Bytes;
use crossbeam_skiplist::SkipMap;

/// Value stored in the skip list.
#[derive(Clone, Debug)]
pub struct EntryValue {
    pub op_type: OpType,
    /// The row value encoded as raw bytes (for Put), empty for Delete.
    /// The encoding format is determined by the engine layer (Arrow/Parquet row).
    pub value: Bytes,
}

/// The tag bytes that sort earliest for a given user-key prefix.
/// Computed as: `encode_tag(SEQNUM_MAX, Put)` = `(0 << 8) | 0x01` as 8-byte BE = `[0,0,0,0,0,0,0,1]`.
const SEEK_TAG: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];

pub struct MemtableSkipList {
    map: SkipMap<Bytes, EntryValue>,
    entry_count: AtomicU64,
    size_bytes: AtomicUsize,
}

impl MemtableSkipList {
    pub fn new() -> Self {
        Self {
            map: SkipMap::new(),
            entry_count: AtomicU64::new(0),
            size_bytes: AtomicUsize::new(0),
        }
    }

    /// Insert an entry. `ikey_bytes` is the full encoded `InternalKey` (PK + tag).
    /// `value_size` is the size of `entry.value` for memory accounting.
    pub fn insert(&self, ikey_bytes: Bytes, entry: EntryValue, value_size: usize) {
        let key_size = ikey_bytes.len();
        self.map.insert(ikey_bytes, entry);
        self.entry_count.fetch_add(1, Ordering::Relaxed);
        // Approximate: key bytes + value bytes + skip-list node overhead (~64B).
        self.size_bytes
            .fetch_add(key_size + value_size + 64, Ordering::Relaxed);
    }

    /// Point lookup. `user_key_bytes` is the PK-encoded portion (without tag).
    ///
    /// Returns `Some(&EntryValue)` if found with `seq ≤ read_seq`, else `None`.
    pub fn get(&self, user_key_bytes: &[u8], read_seq: SeqNum) -> Option<EntryValue> {
        let seek_key = build_seek_key(user_key_bytes);
        // Iterate from the seek position forward. Entries for the same PK are
        // sorted seq DESC (newest first), so we scan until we find one with
        // seq <= read_seq or the PK changes.
        let mut cursor = self.map.lower_bound(std::ops::Bound::Included(&seek_key));
        while let Some(entry) = cursor {
            let found_key = entry.key();
            // Verify PK prefix still matches.
            if found_key.len() < user_key_bytes.len() + 8 {
                return None;
            }
            if &found_key[..user_key_bytes.len()] != user_key_bytes {
                return None; // moved past this PK
            }

            let seq = decode_seq_from_key(found_key);
            if seq <= read_seq {
                return Some(entry.value().clone());
            }
            // This version is too new; try the next (older) version.
            cursor = entry.next();
        }
        None
    }

    /// Iterate all entries in sorted order (PK ASC, seq DESC within same PK).
    pub fn iter(&self) -> impl Iterator<Item = (Bytes, EntryValue)> + '_ {
        self.map
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
    }

    /// Iterate entries in the range `[start_user_key, end_user_key)`.
    /// `start_user_key` and `end_user_key` are PK-encoded bytes (without tag).
    pub fn range_iter(
        &self,
        start_user_key: &[u8],
        end_user_key: Option<&[u8]>,
    ) -> impl Iterator<Item = (Bytes, EntryValue)> + '_ {
        let start = std::ops::Bound::Included(build_seek_key(start_user_key));
        let end = match end_user_key {
            Some(ek) => std::ops::Bound::Excluded(build_seek_key(ek)),
            None => std::ops::Bound::Unbounded,
        };
        self.map
            .range((start, end))
            .map(|e| (e.key().clone(), e.value().clone()))
    }

    pub fn entry_count(&self) -> u64 {
        self.entry_count.load(Ordering::Relaxed)
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes.load(Ordering::Relaxed)
    }
}

impl Default for MemtableSkipList {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a seek key: `user_key_bytes ++ SEEK_TAG`.
/// This is the smallest InternalKey for the given PK (SEQNUM_MAX → tag = [0..0,1]).
fn build_seek_key(user_key_bytes: &[u8]) -> Bytes {
    let mut buf = Vec::with_capacity(user_key_bytes.len() + 8);
    buf.extend_from_slice(user_key_bytes);
    buf.extend_from_slice(&SEEK_TAG);
    Bytes::from(buf)
}

/// Decode the sequence number from a fully encoded InternalKey bytes slice.
/// The tag occupies the last 8 bytes in big-endian: `(inverted_seq << 8) | op_type`.
pub fn decode_seq_from_key(ikey_bytes: &[u8]) -> SeqNum {
    debug_assert!(ikey_bytes.len() >= 8);
    let tag_bytes: [u8; 8] = ikey_bytes[ikey_bytes.len() - 8..].try_into().unwrap();
    let tag = u64::from_be_bytes(tag_bytes);
    let inverted = tag >> 8;
    SeqNum(SEQNUM_MAX.0 - inverted)
}

/// Decode the `OpType` from a fully encoded InternalKey bytes slice.
pub fn decode_op_type_from_key(ikey_bytes: &[u8]) -> Option<OpType> {
    debug_assert!(ikey_bytes.len() >= 8);
    let op_byte = ikey_bytes[ikey_bytes.len() - 1];
    match op_byte {
        0x00 => Some(OpType::Delete),
        0x01 => Some(OpType::Put),
        _ => None,
    }
}

/// Extract user-key (PK) bytes from a fully encoded InternalKey bytes slice.
pub fn user_key_of(ikey_bytes: &[u8]) -> &[u8] {
    &ikey_bytes[..ikey_bytes.len() - 8]
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fake InternalKey: `user_key ++ tag`.
    fn make_ikey(user_key: &[u8], seq: u64, op: OpType) -> Bytes {
        let inverted = SEQNUM_MAX.0 - seq;
        let tag = (inverted << 8) | (op as u64);
        let mut buf = Vec::with_capacity(user_key.len() + 8);
        buf.extend_from_slice(user_key);
        buf.extend_from_slice(&tag.to_be_bytes());
        Bytes::from(buf)
    }

    #[test]
    fn insert_and_get_single() {
        let sl = MemtableSkipList::new();
        let ikey = make_ikey(b"hello", 1, OpType::Put);
        let entry = EntryValue {
            op_type: OpType::Put,
            value: Bytes::from("world"),
        };
        sl.insert(ikey, entry, 5);

        let found = sl.get(b"hello", SeqNum(1));
        assert!(found.is_some());
        assert_eq!(found.unwrap().value, Bytes::from("world"));
    }

    #[test]
    fn get_returns_none_for_missing() {
        let sl = MemtableSkipList::new();
        let ikey = make_ikey(b"exists", 1, OpType::Put);
        sl.insert(
            ikey,
            EntryValue {
                op_type: OpType::Put,
                value: Bytes::from("val"),
            },
            3,
        );
        assert!(sl.get(b"missing", SeqNum(1)).is_none());
    }

    #[test]
    fn get_respects_read_seq() {
        let sl = MemtableSkipList::new();
        let ikey = make_ikey(b"key", 10, OpType::Put);
        sl.insert(
            ikey,
            EntryValue {
                op_type: OpType::Put,
                value: Bytes::from("v10"),
            },
            3,
        );

        // Read at seq 9: should not see seq=10 write.
        assert!(sl.get(b"key", SeqNum(9)).is_none());
        // Read at seq 10: should see it.
        assert!(sl.get(b"key", SeqNum(10)).is_some());
        // Read at seq 100: should still see it.
        assert!(sl.get(b"key", SeqNum(100)).is_some());
    }

    #[test]
    fn newer_version_shadows_older() {
        let sl = MemtableSkipList::new();
        // Insert two versions of the same key.
        let ikey1 = make_ikey(b"key", 1, OpType::Put);
        sl.insert(
            ikey1,
            EntryValue {
                op_type: OpType::Put,
                value: Bytes::from("v1"),
            },
            2,
        );
        let ikey2 = make_ikey(b"key", 5, OpType::Put);
        sl.insert(
            ikey2,
            EntryValue {
                op_type: OpType::Put,
                value: Bytes::from("v5"),
            },
            2,
        );

        // Read at seq 5 should see v5 (newer version).
        let found = sl.get(b"key", SeqNum(5)).unwrap();
        assert_eq!(found.value, Bytes::from("v5"));

        // Read at seq 3 should see v1 (only version visible).
        let found = sl.get(b"key", SeqNum(3)).unwrap();
        assert_eq!(found.value, Bytes::from("v1"));
    }

    #[test]
    fn delete_entry_returned() {
        let sl = MemtableSkipList::new();
        let ikey = make_ikey(b"key", 1, OpType::Delete);
        sl.insert(
            ikey,
            EntryValue {
                op_type: OpType::Delete,
                value: Bytes::new(),
            },
            0,
        );
        let found = sl.get(b"key", SeqNum(1)).unwrap();
        assert_eq!(found.op_type, OpType::Delete);
    }

    #[test]
    fn entry_count_and_size() {
        let sl = MemtableSkipList::new();
        assert_eq!(sl.entry_count(), 0);
        assert_eq!(sl.size_bytes(), 0);

        let ikey = make_ikey(b"abc", 1, OpType::Put);
        sl.insert(
            ikey,
            EntryValue {
                op_type: OpType::Put,
                value: Bytes::from("xyz"),
            },
            3,
        );
        assert_eq!(sl.entry_count(), 1);
        assert!(sl.size_bytes() > 0);
    }

    #[test]
    fn iter_yields_all_entries_sorted() {
        let sl = MemtableSkipList::new();
        // Insert keys out of order.
        for (key, seq) in [("ccc", 1u64), ("aaa", 2), ("bbb", 3)] {
            let ikey = make_ikey(key.as_bytes(), seq, OpType::Put);
            sl.insert(
                ikey,
                EntryValue {
                    op_type: OpType::Put,
                    value: Bytes::from(key),
                },
                3,
            );
        }
        let entries: Vec<_> = sl.iter().collect();
        let keys: Vec<_> = entries
            .iter()
            .map(|(k, _)| user_key_of(k).to_vec())
            .collect();
        assert_eq!(
            keys,
            vec![b"aaa".to_vec(), b"bbb".to_vec(), b"ccc".to_vec()]
        );
    }

    #[test]
    fn decode_seq_and_op_roundtrip() {
        let ikey = make_ikey(b"test", 42, OpType::Put);
        assert_eq!(decode_seq_from_key(&ikey), SeqNum(42));
        assert_eq!(decode_op_type_from_key(&ikey), Some(OpType::Put));

        let ikey_del = make_ikey(b"test", 99, OpType::Delete);
        assert_eq!(decode_seq_from_key(&ikey_del), SeqNum(99));
        assert_eq!(decode_op_type_from_key(&ikey_del), Some(OpType::Delete));
    }

    #[test]
    fn user_key_of_extracts_prefix() {
        let ikey = make_ikey(b"my_pk_data", 1, OpType::Put);
        assert_eq!(user_key_of(&ikey), b"my_pk_data");
    }

    #[test]
    fn concurrent_inserts() {
        use std::sync::Arc;
        use std::thread;

        let sl = Arc::new(MemtableSkipList::new());
        let mut handles = Vec::new();

        for t in 0..8u64 {
            let sl = Arc::clone(&sl);
            handles.push(thread::spawn(move || {
                for i in 0..1000u64 {
                    let key = format!("t{t:02}_k{i:06}");
                    let seq = t * 1000 + i + 1;
                    let ikey = make_ikey(key.as_bytes(), seq, OpType::Put);
                    sl.insert(
                        ikey,
                        EntryValue {
                            op_type: OpType::Put,
                            value: Bytes::from(key),
                        },
                        10,
                    );
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(sl.entry_count(), 8000);
        // Spot check a few entries.
        assert!(sl.get(b"t00_k000000", SeqNum(1)).is_some());
        assert!(sl.get(b"t07_k000999", SeqNum(8000)).is_some());
    }
}
