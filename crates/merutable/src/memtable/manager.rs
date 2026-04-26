//! `MemtableManager`: owns the active memtable and a queue of immutable ones.
//!
//! # Locking strategy
//!
//! - `RwLock<MemtableSet>`: writers take a **write lock only during `rotate()`**
//!   (rare). Normal `apply_batch` calls take a **read lock** — the skip list
//!   inside `Memtable` is itself lock-free, so concurrent writes within one
//!   memtable do not block each other.
//! - `drop_flushed()` takes a write lock to pop from the immutable queue.
//!
//! # Flow control
//!
//! When `immutable.len() >= max_immutable_count`, writers park on a
//! `tokio::sync::Notify`. The flush background task calls `notify()` after
//! each successful flush.

use std::{
    collections::VecDeque,
    sync::{Arc, RwLock},
};

use crate::types::{Result, sequence::SeqNum};
use crate::wal::batch::WriteBatch;
use tracing::debug;

use crate::memtable::{iterator::MemEntry, memtable::Memtable, skiplist::EntryValue};

struct MemtableSet {
    active: Arc<Memtable>,
    immutable: VecDeque<Arc<Memtable>>,
}

pub struct MemtableManager {
    inner: RwLock<MemtableSet>,
    flush_threshold: usize,
    max_immutable: usize,
    /// Notified when an immutable memtable is dropped (flush complete).
    pub flush_complete: Arc<tokio::sync::Notify>,
    /// Bug S fix: notified when a new immutable memtable becomes available
    /// (i.e., after `rotate()`). Background flush workers should wait on
    /// this signal, NOT on `flush_complete` which fires when a flush
    /// finishes — the old code created a chicken-and-egg problem where
    /// the flush worker only woke when a *prior* flush completed, but
    /// nothing triggered the *first* flush from the background worker.
    pub immutable_available: Arc<tokio::sync::Notify>,
}

impl MemtableManager {
    pub fn new(first_seq: SeqNum, flush_threshold: usize, max_immutable: usize) -> Self {
        let active = Arc::new(Memtable::new(first_seq, flush_threshold));
        Self {
            inner: RwLock::new(MemtableSet {
                active,
                immutable: VecDeque::new(),
            }),
            flush_threshold,
            max_immutable,
            flush_complete: Arc::new(tokio::sync::Notify::new()),
            immutable_available: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Apply a `WriteBatch` to the active memtable.
    /// Returns `true` if the flush threshold was crossed (engine should rotate).
    pub fn apply_batch(&self, batch: &WriteBatch) -> Result<bool> {
        let set = self.inner.read().unwrap();
        set.active.apply_batch(batch)
    }

    /// Seal the active memtable, move it to the immutable queue, and open a
    /// new active memtable starting at `new_first_seq`.
    /// Returns the sealed (now-immutable) memtable for the flush job.
    pub fn rotate(&self, new_first_seq: SeqNum) -> Arc<Memtable> {
        debug!(new_first_seq = new_first_seq.0, "rotating memtable");
        let mut set = self.inner.write().unwrap();
        set.active.seal();
        let sealed = set.active.clone();
        set.active = Arc::new(Memtable::new(new_first_seq, self.flush_threshold));
        set.immutable.push_back(sealed.clone());
        drop(set);
        // Issue #14 Phase-1 metrics. Low-churn (rotation rate ~= flush
        // rate), off the per-write hot path.
        metrics::counter!("merutable.memtable.rotations_total").increment(1);
        // Bug S fix: wake any background flush workers waiting for an
        // immutable memtable to become available.
        self.immutable_available.notify_waiters();
        sealed
    }

    /// Remove the oldest immutable memtable after its flush job succeeds.
    /// Notifies stalled writers.
    pub fn drop_flushed(&self, first_seq: SeqNum) {
        debug!(first_seq = first_seq.0, "dropping flushed memtable");
        let mut set = self.inner.write().unwrap();
        set.immutable.retain(|m| m.first_seq != first_seq);
        // Wake any writer that was stalled waiting for immutable queue space.
        self.flush_complete.notify_waiters();
    }

    /// Returns the oldest immutable memtable (if any) for flushing.
    pub fn oldest_immutable(&self) -> Option<Arc<Memtable>> {
        let set = self.inner.read().unwrap();
        set.immutable.front().cloned()
    }

    /// How many immutable memtables are waiting to be flushed.
    pub fn immutable_count(&self) -> usize {
        self.inner.read().unwrap().immutable.len()
    }

    /// Returns `true` if the write should stall (immutable queue full).
    pub fn should_stall(&self) -> bool {
        self.immutable_count() >= self.max_immutable
    }

    /// Returns `true` if the **current active** memtable has crossed its
    /// flush threshold. Used by the auto-flush path on the write side so
    /// that after a burst of concurrent writes all observed the same
    /// `apply_batch → should_flush=true`, only the task that actually gets
    /// to rotate re-checks and proceeds — later tasks find a freshly
    /// rotated (small) active memtable and bail out.
    pub fn active_should_flush(&self) -> bool {
        self.inner.read().unwrap().active.should_flush()
    }

    /// Approximate byte size of the active memtable.
    pub fn active_size_bytes(&self) -> usize {
        self.inner.read().unwrap().active.size_bytes()
    }

    /// Number of entries in the active memtable.
    pub fn active_entry_count(&self) -> u64 {
        self.inner.read().unwrap().active.entry_count()
    }

    /// Flush threshold in bytes.
    pub fn flush_threshold(&self) -> usize {
        self.flush_threshold
    }

    /// Multi-level point lookup: check active first, then immutable queue newest→oldest.
    pub fn get(&self, user_key_bytes: &[u8], read_seq: SeqNum) -> Option<EntryValue> {
        let set = self.inner.read().unwrap();
        if let Some(e) = set.active.get(user_key_bytes, read_seq) {
            return Some(e);
        }
        // Iterate immutable queue from newest (back) to oldest (front).
        for mem in set.immutable.iter().rev() {
            if let Some(e) = mem.get(user_key_bytes, read_seq) {
                return Some(e);
            }
        }
        None
    }

    /// Snapshot all memtable entries into owned Vecs, newest-source first.
    /// Callers feed these into the K-way merge heap (Phase 7).
    /// Collecting avoids holding the RwLock across async boundaries.
    pub fn snapshot_entries(&self, read_seq: SeqNum) -> Vec<Vec<MemEntry>> {
        let set = self.inner.read().unwrap();
        let mut snapshots: Vec<Vec<MemEntry>> = Vec::new();
        snapshots.push(set.active.iter(read_seq).collect());
        for mem in set.immutable.iter().rev() {
            snapshots.push(mem.iter(read_seq).collect());
        }
        snapshots
    }

    /// Issue #29 Phase 2a: change-feed snapshot. Yields EVERY
    /// version (not just the latest per key) across every memtable,
    /// for callers that need put+delete pairs to surface separately.
    /// Returned in a single flat vec — callers sort by seq to get
    /// the change-feed order.
    pub fn snapshot_all_versions(&self, read_seq: SeqNum) -> Vec<MemEntry> {
        let set = self.inner.read().unwrap();
        let mut out: Vec<MemEntry> = set.active.iter_all_versions(read_seq);
        for mem in set.immutable.iter().rev() {
            out.extend(mem.iter_all_versions(read_seq));
        }
        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::sequence::{OpType, SeqNum};
    use crate::wal::batch::WriteBatch;
    use bytes::Bytes;

    fn make_batch(seq: u64, key: &str, val: &str) -> WriteBatch {
        let mut b = WriteBatch::new(SeqNum(seq));
        // In real usage, user_key would be InternalKey::user_key_bytes (PK encoded).
        // For this test we use raw string bytes as the "pk encoded" key.
        let pk_bytes = encode_test_pk(key);
        b.put(pk_bytes, Bytes::from(val.to_string()));
        b
    }

    /// Minimal PK encoding for tests: [u8::MAX (type tag placeholder)][raw_bytes][0x00].
    /// Must produce bytes that sort correctly relative to each other.
    fn encode_test_pk(key: &str) -> Bytes {
        // Simple: just use UTF-8 bytes directly. Keys are ASCII in tests.
        Bytes::from(key.to_string())
    }

    #[test]
    fn apply_and_get() {
        let mgr = MemtableManager::new(SeqNum(1), 64 * 1024 * 1024, 4);
        let batch = make_batch(1, "hello", "world");
        mgr.apply_batch(&batch).unwrap();

        let pk = encode_test_pk("hello");
        let found = mgr.get(&pk, SeqNum(1));
        assert!(found.is_some());
        let e = found.unwrap();
        assert_eq!(e.op_type, OpType::Put);
        assert_eq!(e.value, Bytes::from("world"));
    }

    #[test]
    fn read_seq_isolation() {
        let mgr = MemtableManager::new(SeqNum(1), 64 * 1024 * 1024, 4);
        let batch = make_batch(5, "key", "value_at_5");
        mgr.apply_batch(&batch).unwrap();

        let pk = encode_test_pk("key");
        // Read at seq 4 — should not see seq=5 write.
        assert!(mgr.get(&pk, SeqNum(4)).is_none());
        // Read at seq 5 — should see it.
        assert!(mgr.get(&pk, SeqNum(5)).is_some());
    }

    #[test]
    fn rotate_and_lookup_in_immutable() {
        let mgr = MemtableManager::new(SeqNum(1), 64 * 1024 * 1024, 4);
        let batch = make_batch(1, "alpha", "val1");
        mgr.apply_batch(&batch).unwrap();

        // Rotate: alpha goes to immutable.
        let _sealed = mgr.rotate(SeqNum(2));

        // Write to new active memtable.
        let batch2 = make_batch(2, "beta", "val2");
        mgr.apply_batch(&batch2).unwrap();

        // alpha should still be findable (in immutable).
        let pk_alpha = encode_test_pk("alpha");
        assert!(mgr.get(&pk_alpha, SeqNum(1)).is_some());

        // beta should be findable (in active).
        let pk_beta = encode_test_pk("beta");
        assert!(mgr.get(&pk_beta, SeqNum(2)).is_some());
    }

    #[test]
    fn drop_flushed_removes_immutable() {
        let mgr = MemtableManager::new(SeqNum(1), 64 * 1024 * 1024, 4);
        let _sealed = mgr.rotate(SeqNum(2));
        assert_eq!(mgr.immutable_count(), 1);
        mgr.drop_flushed(SeqNum(1));
        assert_eq!(mgr.immutable_count(), 0);
    }
}
