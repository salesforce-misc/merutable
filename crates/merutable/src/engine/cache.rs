//! Row-level LRU buffer cache for the read path.
//!
//! Sits between the memtable check and Parquet file I/O in `point_lookup()`.
//! Cache entries are invalidated on every write (put/delete) and cleared on
//! compaction so the cache never serves stale data after a key leaves the
//! memtable.
//!
//! ## Insert-invalidate race (fix)
//!
//! The read path checks cache AFTER memtable. If a reader takes the path
//! `memtable miss → L0 hit V1`, it inserts V1 into the cache to accelerate
//! future reads. A concurrent writer of `V2` calls `invalidate(key)` on
//! the cache, but the write interleaving allowed the reader's insert to
//! land AFTER the writer's invalidate — leaving stale V1 in the cache.
//! Once the memtable is later flushed and the memtable check misses for
//! that key, the stale V1 wins indefinitely.
//!
//! Fix: a monotonic `generation` counter, bumped on every `invalidate`,
//! `invalidate_key`, or `clear`. Readers capture the generation BEFORE
//! reading from disk and pass it to `insert_if_fresh` — the insert is
//! dropped if any invalidation happened in between. This mirrors an
//! optimistic-concurrency-control pattern (version tag).
//!
//! Thread-safe: wraps `lru::LruCache` in a `std::sync::Mutex`. The lock is
//! held only for the duration of a hash-map get/put — no I/O under the lock.

use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::{sequence::OpType, value::Row};
use lru::LruCache;

/// A cached read result from a Parquet file lookup.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub op_type: OpType,
    pub row: Row,
}

/// Row-level LRU cache. Thread-safe.
pub struct RowCache {
    inner: Mutex<LruCache<Vec<u8>, CacheEntry>>,
    /// Monotonic counter; incremented on every mutating operation
    /// (`invalidate`, `clear`). Readers capture the generation before
    /// reading from disk; `insert_if_fresh` rejects the insert if the
    /// generation has advanced — some write invalidated the cache between
    /// the reader's disk read and its insert, so the insert might be stale.
    generation: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl RowCache {
    /// Create a new cache with the given capacity.
    /// Panics if `capacity` is 0 — use `Option<RowCache>` to represent "no cache".
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("row cache capacity must be > 0"),
            )),
            generation: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Look up a key. Returns `Some(entry)` on hit, `None` on miss.
    /// Promotes the entry to most-recently-used on hit.
    pub fn get(&self, user_key: &[u8]) -> Option<CacheEntry> {
        let mut guard = self.inner.lock().unwrap();
        match guard.get(user_key) {
            Some(entry) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                // Issue #14 Phase 2: surface the cache-hit signal
                // through the metrics facade in addition to the
                // internal AtomicU64. The AtomicU64 feeds the sync
                // `stats()` API; the metrics facade feeds operator
                // dashboards. Both are cheap-relaxed.
                crate::engine::metrics::inc(crate::engine::metrics::ROW_CACHE_HITS_TOTAL);
                Some(entry.clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                crate::engine::metrics::inc(crate::engine::metrics::ROW_CACHE_MISSES_TOTAL);
                None
            }
        }
    }

    /// Snapshot of the current generation. Readers call this BEFORE
    /// reading a key from disk; the returned value is passed to
    /// `insert_if_fresh` at insertion time.
    #[inline]
    pub fn snapshot_generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Insert or update a cache entry **only if** the generation is
    /// still `captured_gen`. If any `invalidate` or `clear` fired since
    /// the reader snapshotted the generation, the insert is dropped —
    /// the reader's disk-sourced value might now be stale relative to
    /// the write that triggered the invalidation.
    ///
    /// Slight cache-hit-rate cost (occasional dropped inserts under
    /// concurrent writes) in exchange for strict read-after-write
    /// correctness. The check is under the same lock as the insert, so
    /// there is no TOCTOU between check and put.
    pub fn insert_if_fresh(&self, user_key: Vec<u8>, entry: CacheEntry, captured_gen: u64) {
        let mut guard = self.inner.lock().unwrap();
        if self.generation.load(Ordering::Acquire) != captured_gen {
            return;
        }
        guard.put(user_key, entry);
    }

    /// Unconditional insert — retained for tests and for paths where the
    /// caller knows no concurrent writer can race. Prefer
    /// `insert_if_fresh` from any code path that takes a disk read
    /// across an await.
    pub fn insert(&self, user_key: Vec<u8>, entry: CacheEntry) {
        let mut guard = self.inner.lock().unwrap();
        guard.put(user_key, entry);
    }

    /// Invalidate a single key. Called on every write (put/delete).
    /// Bumps the generation so any concurrent reader's pending
    /// `insert_if_fresh` is rejected.
    pub fn invalidate(&self, user_key: &[u8]) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        let mut guard = self.inner.lock().unwrap();
        guard.pop(user_key);
    }

    /// Drop every cached entry. Called after compaction installs a new
    /// version — compaction rewrites files and may resolve MVCC versions,
    /// so any cached entry read from a now-obsolete file could be stale.
    /// Also bumps the generation.
    pub fn clear(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        let mut guard = self.inner.lock().unwrap();
        guard.clear();
    }

    /// Current number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cache capacity.
    pub fn cap(&self) -> usize {
        self.inner.lock().unwrap().cap().into()
    }

    /// Cumulative hit count.
    pub fn hit_count(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Cumulative miss count.
    pub fn miss_count(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::value::FieldValue;

    fn make_entry(val: i64) -> CacheEntry {
        CacheEntry {
            op_type: OpType::Put,
            row: Row::new(vec![Some(FieldValue::Int64(val))]),
        }
    }

    #[test]
    fn insert_and_get() {
        let cache = RowCache::new(10);
        cache.insert(b"key1".to_vec(), make_entry(1));
        let hit = cache.get(b"key1");
        assert!(hit.is_some());
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 0);
    }

    #[test]
    fn miss_returns_none() {
        let cache = RowCache::new(10);
        assert!(cache.get(b"missing").is_none());
        assert_eq!(cache.miss_count(), 1);
    }

    #[test]
    fn invalidate_removes_entry() {
        let cache = RowCache::new(10);
        cache.insert(b"key1".to_vec(), make_entry(1));
        cache.invalidate(b"key1");
        assert!(cache.get(b"key1").is_none());
    }

    #[test]
    fn lru_eviction() {
        let cache = RowCache::new(2);
        cache.insert(b"a".to_vec(), make_entry(1));
        cache.insert(b"b".to_vec(), make_entry(2));
        cache.insert(b"c".to_vec(), make_entry(3)); // evicts "a"
        assert!(cache.get(b"a").is_none());
        assert!(cache.get(b"b").is_some());
        assert!(cache.get(b"c").is_some());
    }

    /// Insert-invalidate race: if the cache is invalidated between a
    /// reader's generation snapshot and its insert, the insert must be
    /// dropped. Without this, a reader could successfully install a
    /// stale entry that a concurrent writer had just invalidated — and
    /// after a subsequent memtable flush, the stale entry would be
    /// served indefinitely.
    #[test]
    fn insert_if_fresh_rejects_after_invalidate() {
        let cache = RowCache::new(10);
        cache.insert(b"k".to_vec(), make_entry(1));

        // Reader captures generation, then the cache is invalidated.
        let generation = cache.snapshot_generation();
        cache.invalidate(b"k");
        assert!(cache.get(b"k").is_none());

        // Reader tries to reinstall the (now-stale) value. Must be dropped.
        cache.insert_if_fresh(b"k".to_vec(), make_entry(42), generation);
        assert!(
            cache.get(b"k").is_none(),
            "insert_if_fresh must drop when generation advanced"
        );
    }

    #[test]
    fn insert_if_fresh_rejects_after_clear() {
        let cache = RowCache::new(10);
        let generation = cache.snapshot_generation();
        cache.clear();
        cache.insert_if_fresh(b"k".to_vec(), make_entry(42), generation);
        assert!(cache.get(b"k").is_none());
    }

    #[test]
    fn insert_if_fresh_succeeds_without_races() {
        let cache = RowCache::new(10);
        let generation = cache.snapshot_generation();
        cache.insert_if_fresh(b"k".to_vec(), make_entry(42), generation);
        assert!(cache.get(b"k").is_some());
    }

    /// IMP-03 regression: `clear()` drops every cached entry so that
    /// compaction-invalidated data cannot be served.
    #[test]
    fn clear_drops_all_entries() {
        let cache = RowCache::new(10);
        cache.insert(b"a".to_vec(), make_entry(1));
        cache.insert(b"b".to_vec(), make_entry(2));
        cache.insert(b"c".to_vec(), make_entry(3));
        assert_eq!(cache.len(), 3);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get(b"a").is_none());
        assert!(cache.get(b"b").is_none());
        assert!(cache.get(b"c").is_none());
    }

    #[test]
    fn len_and_cap() {
        let cache = RowCache::new(5);
        assert_eq!(cache.cap(), 5);
        assert_eq!(cache.len(), 0);
        cache.insert(b"x".to_vec(), make_entry(1));
        assert_eq!(cache.len(), 1);
    }
}
