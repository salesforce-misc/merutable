//! `Memtable`: the mutable in-memory write buffer.
//!
//! A single `Memtable` is alive from the moment it receives its first write
//! (associated `first_seq`) until it has been flushed to a Parquet file and
//! all references are dropped.
//!
//! Write path: `MeruEngine` calls `apply_batch` → returns `should_flush()`.
//! When `should_flush()` is true, the engine rotates this memtable to the
//! immutable queue and starts a `FlushJob`.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::types::{Result, sequence::SeqNum};
use bytes::Bytes;

use crate::memtable::{
    iterator::MemtableIterator,
    skiplist::{EntryValue, MemtableSkipList},
};
use crate::wal::batch::WriteBatch;

pub struct Memtable {
    skiplist: MemtableSkipList,
    /// Seq of the first record ever applied to this memtable.
    pub first_seq: SeqNum,
    /// Highest seq applied so far. Updated on every `apply_batch`.
    last_seq: AtomicU64,
    /// Approximate in-memory size (keys + values + skip-list overhead).
    size_bytes: AtomicUsize,
    /// Whether this memtable has been sealed (no more writes allowed).
    sealed: AtomicBool,
    /// Flush threshold in bytes. When `size_bytes >= flush_threshold`, the
    /// engine should rotate this memtable to the immutable queue.
    flush_threshold: usize,
}

impl Memtable {
    pub fn new(first_seq: SeqNum, flush_threshold: usize) -> Self {
        Self {
            skiplist: MemtableSkipList::new(),
            first_seq,
            last_seq: AtomicU64::new(first_seq.0),
            size_bytes: AtomicUsize::new(0),
            sealed: AtomicBool::new(false),
            flush_threshold,
        }
    }

    /// Apply a `WriteBatch` to this memtable.
    ///
    /// Returns `true` if the flush threshold has been crossed after this batch
    /// and the engine should rotate + schedule a flush.
    ///
    /// Panics if the memtable has been sealed.
    pub fn apply_batch(&self, batch: &WriteBatch) -> Result<bool> {
        assert!(
            !self.sealed.load(Ordering::Acquire),
            "apply_batch on sealed memtable"
        );

        let mut seq = batch.sequence;
        for rec in &batch.records {
            // Build InternalKey wire bytes directly (avoid schema dependency in memtable).
            // The engine ensures PK-encoded bytes are already in rec.user_key.
            let mut ikey_buf = Vec::with_capacity(rec.user_key.len() + 8);
            ikey_buf.extend_from_slice(&rec.user_key);
            // encode_tag: (SEQNUM_MAX.0 - seq.0) << 8 | op_type, big-endian
            let inverted = crate::types::sequence::SEQNUM_MAX.0 - seq.0;
            let tag = (inverted << 8) | (rec.op_type as u64);
            ikey_buf.extend_from_slice(&tag.to_be_bytes());
            let ikey_bytes = Bytes::from(ikey_buf);

            let value = rec.value.clone().unwrap_or(Bytes::new());
            let value_size = value.len();
            let entry = EntryValue {
                op_type: rec.op_type,
                value,
            };

            self.skiplist.insert(ikey_bytes, entry, value_size);
            self.size_bytes
                .fetch_add(rec.user_key.len() + 8 + value_size + 64, Ordering::Relaxed);

            // Update last_seq.
            let mut current = self.last_seq.load(Ordering::Relaxed);
            loop {
                if seq.0 <= current {
                    break;
                }
                match self.last_seq.compare_exchange_weak(
                    current,
                    seq.0,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(c) => current = c,
                }
            }
            seq = seq.next();
        }

        Ok(self.should_flush())
    }

    /// Returns `true` if accumulated size has crossed the flush threshold.
    #[inline]
    pub fn should_flush(&self) -> bool {
        self.size_bytes.load(Ordering::Relaxed) >= self.flush_threshold
    }

    /// Seal this memtable: no more writes will be accepted.
    pub fn seal(&self) {
        self.sealed.store(true, Ordering::Release);
    }

    pub fn last_seq(&self) -> SeqNum {
        SeqNum(self.last_seq.load(Ordering::Acquire))
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes.load(Ordering::Relaxed)
    }

    pub fn entry_count(&self) -> u64 {
        self.skiplist.entry_count()
    }

    /// Point lookup. `user_key_bytes` = PK-encoded bytes (without tag).
    pub fn get(&self, user_key_bytes: &[u8], read_seq: SeqNum) -> Option<EntryValue> {
        self.skiplist.get(user_key_bytes, read_seq)
    }

    /// Iterator over all entries at or before `read_seq`, deduplicated by user-key
    /// (only the latest version of each key is yielded).
    pub fn iter(&self, read_seq: SeqNum) -> MemtableIterator<'_> {
        MemtableIterator::new(self.skiplist.iter(), read_seq)
    }

    /// Issue #29 Phase 2a: change-feed iteration. Yields EVERY
    /// version in the skip list whose seq is in `(0, read_seq]`,
    /// including superseded versions that `iter()` would dedup away.
    /// Necessary for a change feed — a put+delete pair on the same
    /// key must surface both ops, not just the last.
    ///
    /// Output order is skip-list order: ascending user-key, then
    /// descending seq. Callers that need seq-ascending order (the
    /// change-feed contract) must sort after collecting.
    pub fn iter_all_versions(&self, read_seq: SeqNum) -> Vec<crate::memtable::iterator::MemEntry> {
        use crate::memtable::iterator::MemEntry;
        use crate::memtable::skiplist::{decode_seq_from_key, user_key_of};
        use bytes::Bytes;
        self.skiplist
            .iter()
            .filter_map(move |(ikey_bytes, entry)| {
                let seq = decode_seq_from_key(&ikey_bytes);
                if seq > read_seq {
                    return None;
                }
                let uk = Bytes::copy_from_slice(user_key_of(&ikey_bytes));
                Some(MemEntry {
                    user_key: uk,
                    seq,
                    entry,
                })
            })
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::sequence::OpType;
    use crate::wal::batch::WriteBatch;

    fn make_batch(seq: u64, key: &str, val: &str) -> WriteBatch {
        let mut b = WriteBatch::new(SeqNum(seq));
        b.put(Bytes::from(key.to_string()), Bytes::from(val.to_string()));
        b
    }

    fn make_delete_batch(seq: u64, key: &str) -> WriteBatch {
        let mut b = WriteBatch::new(SeqNum(seq));
        b.delete(Bytes::from(key.to_string()));
        b
    }

    #[test]
    fn basic_put_and_get() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.apply_batch(&make_batch(1, "hello", "world")).unwrap();

        let found = mt.get(b"hello", SeqNum(1));
        assert!(found.is_some());
        assert_eq!(found.unwrap().value, Bytes::from("world"));
        assert_eq!(mt.entry_count(), 1);
    }

    #[test]
    fn get_missing_key_returns_none() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.apply_batch(&make_batch(1, "exists", "val")).unwrap();
        assert!(mt.get(b"missing", SeqNum(1)).is_none());
    }

    #[test]
    fn multiple_keys() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.apply_batch(&make_batch(1, "a", "1")).unwrap();
        mt.apply_batch(&make_batch(2, "b", "2")).unwrap();
        mt.apply_batch(&make_batch(3, "c", "3")).unwrap();

        assert_eq!(mt.get(b"a", SeqNum(3)).unwrap().value, Bytes::from("1"));
        assert_eq!(mt.get(b"b", SeqNum(3)).unwrap().value, Bytes::from("2"));
        assert_eq!(mt.get(b"c", SeqNum(3)).unwrap().value, Bytes::from("3"));
        assert_eq!(mt.entry_count(), 3);
    }

    #[test]
    fn overwrite_same_key() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.apply_batch(&make_batch(1, "key", "v1")).unwrap();
        mt.apply_batch(&make_batch(2, "key", "v2")).unwrap();

        // Latest version visible.
        let found = mt.get(b"key", SeqNum(2)).unwrap();
        assert_eq!(found.value, Bytes::from("v2"));

        // Old snapshot still sees v1.
        let found = mt.get(b"key", SeqNum(1)).unwrap();
        assert_eq!(found.value, Bytes::from("v1"));
    }

    #[test]
    fn delete_shadows_put() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.apply_batch(&make_batch(1, "key", "val")).unwrap();
        mt.apply_batch(&make_delete_batch(2, "key")).unwrap();

        // At seq 2, should see delete.
        let found = mt.get(b"key", SeqNum(2)).unwrap();
        assert_eq!(found.op_type, OpType::Delete);

        // At seq 1, should still see the put.
        let found = mt.get(b"key", SeqNum(1)).unwrap();
        assert_eq!(found.op_type, OpType::Put);
    }

    #[test]
    fn seq_isolation() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.apply_batch(&make_batch(5, "key", "val")).unwrap();

        assert!(mt.get(b"key", SeqNum(4)).is_none());
        assert!(mt.get(b"key", SeqNum(5)).is_some());
    }

    #[test]
    fn should_flush_threshold() {
        // 100-byte threshold — triggers quickly.
        let mt = Memtable::new(SeqNum(1), 100);
        assert!(!mt.should_flush());

        // Write enough data to cross 100 bytes.
        let big_val = "x".repeat(200);
        let batch = make_batch(1, "key", &big_val);
        let crossed = mt.apply_batch(&batch).unwrap();
        assert!(crossed);
        assert!(mt.should_flush());
    }

    #[test]
    #[should_panic(expected = "apply_batch on sealed memtable")]
    fn sealed_memtable_rejects_writes() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.seal();
        mt.apply_batch(&make_batch(1, "key", "val")).unwrap();
    }

    #[test]
    fn last_seq_tracks_correctly() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        assert_eq!(mt.last_seq(), SeqNum(1));

        mt.apply_batch(&make_batch(5, "a", "1")).unwrap();
        assert_eq!(mt.last_seq(), SeqNum(5));

        mt.apply_batch(&make_batch(3, "b", "2")).unwrap();
        // last_seq should still be 5 (doesn't go backward).
        assert_eq!(mt.last_seq(), SeqNum(5));

        mt.apply_batch(&make_batch(10, "c", "3")).unwrap();
        assert_eq!(mt.last_seq(), SeqNum(10));
    }

    #[test]
    fn iterator_yields_all_keys_sorted() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.apply_batch(&make_batch(1, "ccc", "3")).unwrap();
        mt.apply_batch(&make_batch(2, "aaa", "1")).unwrap();
        mt.apply_batch(&make_batch(3, "bbb", "2")).unwrap();

        let entries: Vec<_> = mt.iter(SeqNum(10)).collect();
        let keys: Vec<_> = entries.iter().map(|e| e.user_key.to_vec()).collect();
        assert_eq!(
            keys,
            vec![b"aaa".to_vec(), b"bbb".to_vec(), b"ccc".to_vec()]
        );
    }

    #[test]
    fn iterator_dedup_keeps_latest() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        mt.apply_batch(&make_batch(1, "key", "v1")).unwrap();
        mt.apply_batch(&make_batch(2, "key", "v2")).unwrap();

        let entries: Vec<_> = mt.iter(SeqNum(10)).collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry.value, Bytes::from("v2"));
    }

    #[test]
    fn size_bytes_increases() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        assert_eq!(mt.size_bytes(), 0);

        mt.apply_batch(&make_batch(1, "key", "value")).unwrap();
        assert!(mt.size_bytes() > 0);

        let s1 = mt.size_bytes();
        mt.apply_batch(&make_batch(2, "another", "data")).unwrap();
        assert!(mt.size_bytes() > s1);
    }

    #[test]
    fn multi_record_batch() {
        let mt = Memtable::new(SeqNum(1), 64 * 1024 * 1024);
        let mut batch = WriteBatch::new(SeqNum(1));
        batch.put(Bytes::from("k1"), Bytes::from("v1"));
        batch.put(Bytes::from("k2"), Bytes::from("v2"));
        batch.put(Bytes::from("k3"), Bytes::from("v3"));
        mt.apply_batch(&batch).unwrap();

        assert_eq!(mt.entry_count(), 3);
        assert!(mt.get(b"k1", SeqNum(3)).is_some());
        assert!(mt.get(b"k2", SeqNum(3)).is_some());
        assert!(mt.get(b"k3", SeqNum(3)).is_some());
    }
}
