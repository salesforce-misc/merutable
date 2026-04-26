//! `MeruEngine`: central orchestrator. Owns WAL, memtable, version set, catalog,
//! and background workers. All public operations go through this struct.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::iceberg::{IcebergCatalog, VersionSet};
use crate::memtable::manager::MemtableManager;
use crate::types::{
    MeruError, Result,
    key::InternalKey,
    schema::TableSchema,
    sequence::{GlobalSeq, OpType, SeqNum},
    value::{FieldValue, Row},
};
use crate::wal::{batch::WriteBatch, manager::WalManager};
use tokio::sync::Mutex;
use tracing::{info, instrument, warn};

use crate::engine::config::EngineConfig;

/// Issue #29 Phase 2b: one tuple from the change-feed scan. Bundles
/// the seq + op-type + decoded row + PK bytes for a single op.
/// Kept as a struct (not a 4-tuple) so callers that only need the
/// PK — e.g. replica tails applying tombstones — access it by name.
#[derive(Clone, Debug)]
pub struct ChangeTuple {
    pub seq: u64,
    pub op_type: OpType,
    /// Post-state row for Put; `Row::default()` for Delete until
    /// Phase 2c reconstructs the pre-image.
    pub row: Row,
    /// PK-encoded bytes — the canonical way to address the mutation.
    pub pk_bytes: Vec<u8>,
}

/// The engine. Thread-safe via `Arc<MeruEngine>` — pass it across async tasks.
pub struct MeruEngine {
    pub(crate) config: EngineConfig,
    pub(crate) schema: Arc<TableSchema>,
    pub(crate) global_seq: GlobalSeq,
    pub(crate) wal: Mutex<WalManager>,
    pub(crate) memtable: MemtableManager,
    pub(crate) version_set: VersionSet,
    pub(crate) catalog: Arc<IcebergCatalog>,
    /// Serializes memtable rotation attempts from the auto-flush path so a
    /// burst of concurrent writes all seeing `should_flush=true` triggers
    /// at most one rotation. Bug F regression: without this, every task in
    /// a concurrent write burst would either (a) skip rotation entirely or
    /// (b) race and seal empty memtables. The fix is `try_lock` + double
    /// check under the lock: the loser drops through, the winner rotates.
    pub(crate) rotation_lock: Mutex<()>,
    /// Serializes `run_flush` so two concurrently-spawned auto-flush tasks
    /// don't both call `oldest_immutable()`, see the same sealed memtable,
    /// and write two L0 Parquet files containing identical rows (followed
    /// by two competing catalog commits). Bug G regression.
    pub(crate) flush_mutex: Mutex<()>,
    /// Per-level reservation set. Replaces the pre-existing single
    /// `compaction_mutex` so two background compaction workers can run
    /// concurrently on **disjoint level sets** — e.g. worker A does
    /// L0→L1 while worker B does L2→L3 in parallel. Before this was
    /// introduced, a single 45-minute L2→L3 compaction blocked all L0
    /// drainage (observed in stress test: L0 grew to 75 files with 2
    /// idle workers stalled on the mutex).
    ///
    /// Invariants (Bug T fix preserved):
    /// - A compaction reserves both its input_level and output_level
    ///   before executing. Two compactions never share any level.
    /// - Full-level picking (current picker model) means two compactions
    ///   on the same level would either share input files OR produce
    ///   overlapping output ranges at the output level. Either would
    ///   corrupt the L1+ non-overlap invariant or the manifest. Per-level
    ///   reservation prevents both at once with a single lock acquire.
    ///
    /// Follows Pebble's `compactionEnv.inProgressCompactions` and
    /// RocksDB's `level0_compactions_in_progress_` / `FileMetaData::
    /// being_compacted` pattern, scaled down to level granularity since
    /// the current picker is full-level. Can be refined to file-granular
    /// tracking if/when partial-level picking is introduced.
    pub(crate) compacting_levels: Mutex<std::collections::HashSet<crate::types::level::Level>>,
    /// Issue #30 observability: cached size of `compacting_levels`.
    /// Updated by the call sites that insert / remove entries under
    /// the tokio Mutex; `stats()` reads this counter synchronously
    /// so hot-path stats snapshots never block on the compaction
    /// scheduler. Eventually consistent with the HashSet — a read
    /// during the window between HashSet mutation and counter
    /// update sees a value off by one.
    pub(crate) compacting_levels_len: std::sync::atomic::AtomicUsize,
    /// Serializes just the catalog commit phase — brief (ms), distinct
    /// from the long merge phase. Two parallel compactions that finish
    /// their merges concurrently must linearize through `catalog.commit()`
    /// because each commit computes `next_ver` from the current manifest
    /// and writes `v{N+1}.metadata.json`; without serialization they'd
    /// race on the version number and overwrite each other's manifest.
    pub(crate) commit_lock: Mutex<()>,
    /// Row-level LRU cache for point lookups. `None` if disabled (capacity=0).
    pub(crate) row_cache: Option<crate::engine::cache::RowCache>,
    /// True if opened in read-only mode. Write ops will return `MeruError::ReadOnly`.
    pub(crate) read_only: bool,
    /// IMP-02: reader-visible sequence number. Only advanced AFTER memtable
    /// apply completes (inside the WAL lock), so `read_seq()` never returns a
    /// sequence whose data isn't in the memtable yet. `global_seq` remains
    /// the allocation counter; `visible_seq` is what readers snapshot.
    pub(crate) visible_seq: GlobalSeq,
    /// Files pending physical deletion after a compaction committed the
    /// manifest removal. Each entry records:
    /// - `path`: the file on disk.
    /// - `obsoleted_at`: wall-clock time of commit; used by the legacy
    ///   time-based grace period.
    /// - `obsoleted_after_snapshot`: the `snapshot_id` at which the file
    ///   was still live. Readers pinning any snapshot `<=` this value
    ///   may still try to read this file, so GC must wait for those
    ///   pins to release.
    ///
    /// GC deletes a file iff BOTH conditions hold:
    /// 1. No reader pin exists at a snapshot `<=` `obsoleted_after_snapshot`
    ///    (version-pinned safety — fixes BUG-0007..0013 where long
    ///    integrity scans hit `IO NotFound` because GC ran while the
    ///    reader still held the old `Version`).
    /// 2. `obsoleted_at.elapsed() >= gc_grace_period_secs` (time-based
    ///    grace for external external analytical readers, e.g. DuckDB, which don't
    ///    participate in the pin protocol).
    ///
    /// Both must hold: version-pin protects internal readers,
    /// time-grace protects external ones.
    pub(crate) pending_deletions: Mutex<Vec<PendingDelete>>,
    /// Issue #30 observability: cached size of `pending_deletions`,
    /// maintained by `enqueue_for_deletion` and `gc_pending_deletions`.
    /// Readable synchronously from `stats()` without touching the
    /// tokio Mutex (which holds across `remove_file().await` during
    /// GC). Stays eventually consistent with the Vec — a stats read
    /// in the middle of a GC sweep may be off by O(queue size).
    pub(crate) pending_deletions_len: std::sync::atomic::AtomicUsize,
    /// Issue #30 observability: Instant at which the oldest currently-
    /// pending deletion was enqueued. `None` when the queue is empty.
    /// Maintained alongside the Vec so `stats()` can compute the age
    /// without holding the tokio Mutex.
    pub(crate) pending_oldest_enqueue: std::sync::RwLock<Option<std::time::Instant>>,
    /// Notified by compaction after a successful commit that reduced
    /// the L0 file count. Writers stalled on the L0 stop trigger park
    /// on this notify; waking them on L0 drainage (rather than on the
    /// 1-second worker heartbeat) avoids unnecessary sleep latency on
    /// the hot write path. Fires on compaction commit regardless of
    /// input level — over-firing is harmless (waiters just re-check
    /// the condition and go back to sleep) and keeps the notifier
    /// logic simple.
    pub(crate) l0_drained: std::sync::Arc<tokio::sync::Notify>,
    /// Multiset of snapshot_ids pinned by active internal readers.
    /// `get()` / `scan()` pin the snapshot they capture from
    /// `version_set.current()` and unpin on return via `SnapshotPin`'s
    /// `Drop`. GC's `min_pinned_snapshot()` queries the smallest key;
    /// any file whose `obsoleted_after_snapshot >= min_pinned` is still
    /// reachable by a live reader and must not be deleted.
    pub(crate) live_snapshots: std::sync::Mutex<std::collections::BTreeMap<i64, usize>>,
    /// Set to `true` by `close()`. All subsequent write/flush/compact ops
    /// return `MeruError::Closed`. Reads remain available until the engine
    /// is dropped.
    closed: AtomicBool,
}

/// An entry in `pending_deletions` — see the field docs on `MeruEngine`.
#[derive(Debug)]
pub(crate) struct PendingDelete {
    pub path: std::path::PathBuf,
    pub obsoleted_at: std::time::Instant,
    pub obsoleted_after_snapshot: i64,
}

/// RAII guard returned by `pin_current_snapshot`. On drop, decrements
/// the snapshot's pin count; when the count hits zero, the snapshot_id
/// is removed from the pin map so `min_pinned_snapshot()` can advance.
///
/// Holding this guard prevents GC from deleting any file whose
/// `obsoleted_after_snapshot >= self.snapshot_id` — the reader is
/// guaranteed every file its `Version` snapshot references will remain
/// on disk until the guard drops.
pub struct SnapshotPin<'a> {
    engine: &'a MeruEngine,
    snapshot_id: i64,
}

impl Drop for SnapshotPin<'_> {
    fn drop(&mut self) {
        let mut pins = self.engine.live_snapshots.lock().unwrap();
        if let Some(count) = pins.get_mut(&self.snapshot_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                pins.remove(&self.snapshot_id);
            }
        }
    }
}

impl MeruEngine {
    /// Open (or create) an engine instance.
    ///
    /// 1. Open/recover WAL directory.
    /// 2. Replay recovered batches into a fresh memtable.
    /// 3. Open Iceberg catalog and load current version.
    /// 4. Initialize global seq to `max(wal_max_seq, iceberg_max_seq) + 1`.
    #[instrument(skip(config), fields(table = %config.schema.table_name))]
    pub async fn open(mut config: EngineConfig) -> Result<Arc<Self>> {
        // Bug SC2 fix: validate the schema upfront so misconfigured schemas
        // (out-of-bounds PK indices, nullable PKs, empty columns) produce a
        // clear error here instead of panicking deep inside encode/decode.
        //
        // Issue #25: validate() takes &mut because it auto-assigns Iceberg
        // field-ids. The mutation is safe here — the schema is owned by
        // this open call and the normalized form is what we want to persist.
        config.schema.validate()?;

        let schema = Arc::new(config.schema.clone());

        let read_only = config.read_only;

        // WAL recovery — in read-only mode, skip if WAL dir doesn't exist.
        let (recovered_batches, wal_max_seq, max_log_number) =
            if read_only && !config.wal_dir.exists() {
                (Vec::new(), SeqNum(0), 0u64)
            } else {
                WalManager::recover_from_dir(&config.wal_dir)?
            };
        info!(
            recovered = recovered_batches.len(),
            wal_max_seq = wal_max_seq.0,
            read_only,
            "WAL recovery complete"
        );

        // Open Iceberg catalog and load current version.
        let catalog = IcebergCatalog::open(&config.catalog_uri, config.schema.clone()).await?;
        let manifest = catalog.current_manifest().await;
        let version = manifest.to_version(schema.clone());
        let iceberg_max_seq = version
            .levels
            .values()
            .flat_map(|files| files.iter().map(|f| f.meta.seq_max))
            .max()
            .unwrap_or(0);

        let version_set = VersionSet::new(version);
        let catalog = Arc::new(catalog);

        // Global seq = max of WAL and Iceberg + 1.
        let init_seq = std::cmp::max(wal_max_seq.0, iceberg_max_seq) + 1;
        let global_seq = GlobalSeq::new(init_seq);

        // Memtable manager.
        let memtable = MemtableManager::new(
            SeqNum(init_seq),
            config.memtable_size_bytes,
            config.max_immutable_count,
        );

        // Replay recovered WAL batches into memtable.
        for batch in &recovered_batches {
            memtable.apply_batch(batch)?;
        }
        if !recovered_batches.is_empty() {
            info!(
                count = recovered_batches.len(),
                "replayed WAL batches into memtable"
            );
        }

        // Bug W fix: compute `next_log` from the highest WAL file number on
        // disk, NOT from the batch count. After partial WAL GC, the batch
        // count can be smaller than the highest surviving log number, causing
        // the new WAL file to collide with (and truncate) an existing file.
        // A second crash before flush would then lose the overwritten data.
        let next_log = max_log_number + 1;
        // In read-only mode, ensure WAL dir exists for WalManager::open
        // (it won't be used since write ops are guarded).
        if read_only {
            std::fs::create_dir_all(&config.wal_dir).map_err(MeruError::Io)?;
        }
        let wal = WalManager::open(&config.wal_dir, next_log)?;

        // Issue #22: register every recovered WAL file (log_num < next_log)
        // as a closed log so the first `mark_flushed_seq()` after
        // recovery GCs it. Without this, orphaned WAL files from prior
        // crashes persist indefinitely, are re-replayed on every
        // subsequent reopen, and — in the presence of racing
        // background compaction — create a window for stale-seq
        // memtable entries to shadow freshly-compacted L1 output.
        // Net effect on the original reproducer was 50 rows lost per
        // crash cycle. `max_log_number + 1 == next_log`, so any file
        // at `log_num < next_log` is an orphan from before the new
        // WalManager opened its own file.
        if !read_only {
            match WalManager::list_wal_files(&config.wal_dir) {
                Ok(files) => {
                    for (log_num, _path) in files {
                        if log_num < next_log {
                            wal.register_closed_log(log_num, wal_max_seq);
                        }
                    }
                }
                Err(e) => {
                    // Non-fatal — GC will still happen lazily via
                    // gc_logs_before after the first Iceberg commit,
                    // but the fast path won't fire. Log and continue.
                    tracing::warn!(
                        error = %e,
                        "failed to enumerate WAL dir for orphan registration"
                    );
                }
            }
        }

        let row_cache = if config.row_cache_capacity > 0 {
            Some(crate::engine::cache::RowCache::new(
                config.row_cache_capacity,
            ))
        } else {
            None
        };

        // Issue #8 + IMP-02: `visible_seq` is "the highest sequence
        // whose data is visible" (inclusive), not "the next sequence
        // to become visible" (exclusive). On a fresh DB the next
        // allocated seq is 1 and nothing is visible yet, so
        // `visible_seq = 0`. On a recovered DB with iceberg/WAL max
        // seq = M, the visible frontier is M (everything at M and
        // below is visible; M+1 is the next to allocate). The
        // inclusive semantic lets callers assert the natural invariant
        // `put_result > read_seq_before_put` without an off-by-one.
        let visible_seq = GlobalSeq::new(init_seq.saturating_sub(1));

        let engine = Arc::new(Self {
            config,
            schema,
            global_seq,
            wal: Mutex::new(wal),
            memtable,
            version_set,
            catalog,
            rotation_lock: Mutex::new(()),
            flush_mutex: Mutex::new(()),
            compacting_levels: Mutex::new(std::collections::HashSet::new()),
            compacting_levels_len: std::sync::atomic::AtomicUsize::new(0),
            commit_lock: Mutex::new(()),
            row_cache,
            read_only,
            visible_seq,
            pending_deletions: Mutex::new(Vec::new()),
            pending_deletions_len: std::sync::atomic::AtomicUsize::new(0),
            pending_oldest_enqueue: std::sync::RwLock::new(None),
            l0_drained: std::sync::Arc::new(tokio::sync::Notify::new()),
            live_snapshots: std::sync::Mutex::new(std::collections::BTreeMap::new()),
            closed: AtomicBool::new(false),
        });

        Ok(engine)
    }

    // ── Write path ──────────────────────────────────���────────────────────

    /// Insert a row. `pk_values` are the primary key fields; `row` is the full
    /// row (including PK columns).
    #[instrument(skip(self, row), fields(op = "put"))]
    pub async fn put(self: &Arc<Self>, pk_values: Vec<FieldValue>, row: Row) -> Result<SeqNum> {
        if self.read_only {
            return Err(MeruError::ReadOnly);
        }
        // Issue #12: validate row shape (arity, per-column type,
        // nullability) BEFORE anything reaches the WAL. A malformed
        // row was previously accepted and written, then on read
        // silently produced a phantom empty Row indistinguishable
        // from NULL.
        if let Err(e) = row.validate(&self.schema) {
            crate::engine::metrics::inc(crate::engine::metrics::SCHEMA_MISMATCH_TOTAL);
            return Err(e);
        }
        // Issue #14 Phase 2: hot-path counter. Bumped only on the
        // successful-validation path so schema errors don't inflate
        // the throughput signal.
        crate::engine::metrics::inc(crate::engine::metrics::PUTS_TOTAL);
        self.write_internal(pk_values, Some(row), OpType::Put).await
    }

    /// Delete by primary key.
    #[instrument(skip(self), fields(op = "delete"))]
    pub async fn delete(self: &Arc<Self>, pk_values: Vec<FieldValue>) -> Result<SeqNum> {
        if self.read_only {
            return Err(MeruError::ReadOnly);
        }
        crate::engine::metrics::inc(crate::engine::metrics::DELETES_TOTAL);
        self.write_internal(pk_values, None, OpType::Delete).await
    }

    #[instrument(skip(self, pk_values, row), fields(op_type = ?op_type))]
    async fn write_internal(
        self: &Arc<Self>,
        pk_values: Vec<FieldValue>,
        row: Option<Row>,
        op_type: OpType,
    ) -> Result<SeqNum> {
        if self.closed.load(Ordering::Acquire) {
            return Err(MeruError::Closed);
        }

        // Flow control #1: L0 file-count backpressure (Issue #5).
        //
        // `l0_stop_trigger` and `l0_slowdown_trigger` are configured
        // but were previously dead code — the write path only checked
        // the immutable memtable queue, so L0 could grow unbounded
        // during a compaction stall. Stress test observed L0 reaching
        // 44 files (8 past the stop trigger) with writes still
        // proceeding at full speed, directly enabling the Arrow i32
        // overflow crash (Issue #3).
        //
        // Hard stop: `L0 >= l0_stop_trigger` → park on `l0_drained`
        // until compaction reduces L0 below the trigger. Bug Z fix:
        // register `notified()` BEFORE the check to avoid lost-wakeup.
        {
            let mut first_iter = true;
            loop {
                let notify = self.l0_drained.notified();
                let l0 = self.version_set.current().l0_file_count();
                if l0 < self.config.l0_stop_trigger {
                    break;
                }
                if first_iter {
                    // Issue #14: emit exactly once per parked writer,
                    // not once per wake-up. Avoids inflating the
                    // counter when a writer goes to sleep multiple
                    // times during a single stall episode.
                    crate::engine::metrics::inc(crate::engine::metrics::STALL_EVENTS_TOTAL);
                    first_iter = false;
                }
                notify.await;
            }
        }
        // Graduated slowdown: `L0 >= l0_slowdown_trigger` → sleep
        // proportional to excess. Linear ramp from 0 µs at the
        // slowdown trigger to `L0_MAX_DELAY_MICROS` at the stop
        // trigger. Matches RocksDB `SetupDelay()` behaviour
        // (`db/column_family.cc` in upstream).
        {
            let l0 = self.version_set.current().l0_file_count();
            let slow = self.config.l0_slowdown_trigger;
            let stop = self.config.l0_stop_trigger;
            if l0 >= slow && stop > slow {
                const L0_MAX_DELAY_MICROS: u64 = 1000; // 1ms at the stop trigger.
                let excess = (l0 - slow) as u64;
                let range = (stop - slow) as u64;
                // Clamp ratio to [0, 1] — the stop loop above should
                // already keep L0 < stop, but guard anyway.
                let delay = (L0_MAX_DELAY_MICROS * excess.min(range)) / range;
                if delay > 0 {
                    crate::engine::metrics::inc(crate::engine::metrics::SLOWDOWN_EVENTS_TOTAL);
                    tokio::time::sleep(std::time::Duration::from_micros(delay)).await;
                }
            }
        }

        // Flow control #2: immutable-memtable queue backpressure.
        //
        // IMP-09: instead of a binary block at max_immutable, introduce a
        // graduated delay once the immutable queue is >50% full. This
        // prevents the sawtooth throughput pattern where all blocked
        // writers resume simultaneously and re-trigger the stall.
        //
        // Bug Z fix preserved: register `notified()` BEFORE checking the
        // condition to avoid lost-wakeup TOCTOU race.
        loop {
            let notify = self.memtable.flush_complete.notified();
            if !self.memtable.should_stall() {
                break;
            }
            notify.await;
        }

        // Graduated delay: slow writes proportionally as the immutable
        // queue fills, before the hard stall triggers.
        {
            let imm_count = self.memtable.immutable_count() as f64;
            let max_imm = self.config.max_immutable_count as f64;
            let ratio = imm_count / max_imm;
            if ratio > 0.5 {
                // Linear ramp: 0ms at 50%, 5ms at 100%.
                let delay_ms = ((ratio - 0.5) * 2.0 * 5.0) as u64;
                if delay_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
            }
        }

        // Encode only the user-key (PK) bytes — no full InternalKey struct,
        // no pk_values.to_vec() clone, no tag encoding. Hot-path optimization.
        let user_key_bytes = InternalKey::encode_user_key(&pk_values, &self.schema)?;

        // Encode value bytes (CPU work, outside the WAL lock).
        // Issue #33 fix: for Delete ops, the pre-image row IS the
        // value we encode — captured via point_lookup BEFORE the
        // WAL lock so the change feed gets a meaningful pre-image
        // on the resulting DELETE record. An empty bytes result
        // means the key had no prior live state (the delete is
        // idempotent over an already-tombstoned or never-existed
        // key).
        let value_bytes = match op_type {
            OpType::Put => match row {
                Some(r) => Some(bytes::Bytes::from(crate::engine::codec::encode_row(&r)?)),
                None => None,
            },
            OpType::Delete => {
                let pre_image = crate::engine::read_path::point_lookup(self, &pk_values)?;
                match pre_image {
                    Some(r) => Some(bytes::Bytes::from(crate::engine::codec::encode_row(&r)?)),
                    None => Some(bytes::Bytes::new()),
                }
            }
        };

        // IMP-02: allocate seq, WAL append, and memtable apply all happen
        // inside the WAL lock. This guarantees that when visible_seq is
        // advanced, all data for seqs <= visible_seq is in the memtable.
        // A concurrent reader can never observe a sequence number whose
        // data hasn't been applied yet.
        let (seq, should_flush) = {
            let mut wal = self.wal.lock().await;

            let seq = self.global_seq.allocate();
            let mut batch = WriteBatch::new(seq);
            match op_type {
                OpType::Put => batch.put(
                    bytes::Bytes::from(user_key_bytes.clone()),
                    value_bytes.unwrap_or_default(),
                ),
                OpType::Delete => batch.delete_with_pre_image(
                    bytes::Bytes::from(user_key_bytes.clone()),
                    value_bytes.clone().unwrap_or_default(),
                ),
            }

            wal.append(&batch)?;
            let should_flush = self.memtable.apply_batch(&batch)?;

            // Advance visible_seq now that the data is in the memtable.
            // visible_seq semantic is inclusive-latest-visible (Issue #8),
            // so the just-applied seq IS the new visible frontier.
            self.visible_seq.set_at_least(seq.0);

            (seq, should_flush)
        };

        // Invalidate row cache so post-flush reads don't serve stale data.
        if let Some(ref cache) = self.row_cache {
            cache.invalidate(&user_key_bytes);
        }

        // Trigger flush if threshold crossed. The flush requires a rotate
        // (active → immutable) so that `run_flush` has something to find in
        // `oldest_immutable()`; before Bug F was fixed, this path spawned
        // `run_flush` without rotating and the task returned a no-op,
        // leaving the memtable to grow unbounded. Concurrent writers all
        // see the same stale `should_flush=true` during a burst — serialize
        // rotation through `rotation_lock` and re-check under the lock so
        // only one task actually seals and spawns a flush.
        if should_flush {
            if let Ok(_guard) = self.rotation_lock.try_lock() {
                // Stale should_flush from another task's apply_batch? If the
                // active memtable was already rotated out from under us, the
                // new active is small and we have nothing to do.
                if self.memtable.active_should_flush() {
                    let next_seq = self.global_seq.current().next();
                    self.memtable.rotate(next_seq);
                    // Rotate the WAL as well so the sealed memtable's
                    // writes live in a closed log that GC can reclaim.
                    {
                        let mut wal = self.wal.lock().await;
                        wal.rotate()?;
                    }
                    let engine = Arc::clone(self);
                    tokio::spawn(async move {
                        if let Err(e) = crate::engine::flush::run_flush(&engine).await {
                            tracing::error!(error = %e, "auto-flush failed");
                        }
                    });
                }
            }
        }

        Ok(seq)
    }

    // ── Read path ────────────────────────────────────────────────────────

    /// Point lookup by primary key. Returns the row if found (not deleted).
    #[instrument(skip(self), fields(op = "get"))]
    pub fn get(&self, pk_values: &[FieldValue]) -> Result<Option<Row>> {
        // Issue #14 Phase 2: hot-path counters. When no metrics
        // recorder is registered, these compile to ~1 ns TLS-cached
        // null checks. When a recorder IS registered, operators can
        // compute hit ratio without guessing.
        crate::engine::metrics::inc(crate::engine::metrics::GETS_TOTAL);
        let result = crate::engine::read_path::point_lookup(self, pk_values)?;
        if result.is_some() {
            crate::engine::metrics::inc(crate::engine::metrics::GET_HITS_TOTAL);
        }
        Ok(result)
    }

    /// Range scan. Returns rows in PK order where `start_pk <= pk < end_pk`.
    /// If `start_pk` is `None`, scan from the beginning.
    /// If `end_pk` is `None`, scan to the end.
    #[instrument(skip(self), fields(op = "scan"))]
    pub fn scan(
        &self,
        start_pk: Option<&[FieldValue]>,
        end_pk: Option<&[FieldValue]>,
    ) -> Result<Vec<(InternalKey, Row)>> {
        crate::engine::metrics::inc(crate::engine::metrics::SCANS_TOTAL);
        let result = crate::engine::read_path::range_scan(self, start_pk, end_pk)?;
        crate::engine::metrics::inc_by(
            crate::engine::metrics::SCAN_ROWS_TOTAL,
            result.len() as u64,
        );
        Ok(result)
    }

    // ── Admin ────────────────────────────────────────────────────────────

    /// Force flush all immutable memtables and the active memtable.
    #[instrument(skip(self), fields(op = "flush"))]
    pub async fn flush(self: &Arc<Self>) -> Result<()> {
        if self.read_only {
            return Err(MeruError::ReadOnly);
        }
        if self.closed.load(Ordering::Acquire) {
            return Err(MeruError::Closed);
        }
        // Bug R fix: hold `rotation_lock` while rotating so a concurrent
        // auto-flush task from `write_internal` doesn't race on `rotate()`
        // and seal an empty (freshly-created) memtable.
        {
            let _rotation_guard = self.rotation_lock.lock().await;
            // Rotate active memtable AND the WAL together so that (a) the
            // sealed memtable's writes live in a closed WAL file that can be
            // GC'd once the flush commits, and (b) new writes after this call
            // land in a fresh WAL file. Bug D regression: before this, the WAL
            // never rotated under any flush path, so the log directory grew
            // without bound and recovery replayed already-flushed batches.
            let next_seq = self.global_seq.current().next();
            self.memtable.rotate(next_seq);
            {
                let mut wal = self.wal.lock().await;
                wal.rotate()?;
            }
        } // rotation_lock dropped
        // Flush all immutables. `run_flush` calls `mark_flushed_seq` which
        // GCs the matching closed WAL file as a side effect.
        while self.memtable.oldest_immutable().is_some() {
            crate::engine::flush::run_flush(self).await?;
        }
        Ok(())
    }

    /// Trigger a manual compaction. Picks the best level and runs one job.
    #[instrument(skip(self), fields(op = "compact"))]
    pub async fn compact(self: &Arc<Self>) -> Result<()> {
        if self.read_only {
            return Err(MeruError::ReadOnly);
        }
        if self.closed.load(Ordering::Acquire) {
            return Err(MeruError::Closed);
        }
        crate::engine::compaction::job::run_compaction(self).await
    }

    /// Current read sequence (snapshot for reads).
    ///
    /// IMP-02: returns `visible_seq` — the highest sequence number whose
    /// data is guaranteed to be in the memtable. This is strictly <=
    /// `global_seq.current()` and is only advanced after memtable apply
    /// completes inside the WAL lock.
    pub fn read_seq(&self) -> SeqNum {
        self.visible_seq.current()
    }

    pub fn schema(&self) -> &TableSchema {
        &self.schema
    }

    /// Issue #31 Phase 2a: synchronous read of the current
    /// committed snapshot id. O(1) — just an `ArcSwap` load. Used
    /// by the mirror worker to poll for snapshot advances without
    /// allocating a full `EngineStats` every tick.
    pub fn current_snapshot_id(&self) -> i64 {
        self.version_set.snapshot_id()
    }

    /// Issue #31 Phase 2b: a clone of the currently-committed
    /// manifest. Used by the mirror worker to enumerate live data
    /// files + serialize the manifest for upload. Takes a brief
    /// async lock on the catalog's current-manifest mutex.
    pub async fn current_manifest(&self) -> crate::iceberg::Manifest {
        self.catalog.current_manifest().await
    }

    /// Issue #29 Phase 2a: scan the memtable for ops with seq in
    /// `(since_seq, read_seq]` and return them in seq-ascending
    /// order as `(seq, op_type, decoded_row)` tuples. Phase 2a
    /// covers the un-flushed tail only; L0 and deeper levels are
    /// Phase 2b (needs a Parquet scan that filters by seq column).
    ///
    /// Pre-image reconstruction for `Delete` ops is NOT performed
    /// here — the tuple carries an empty `Row` on delete. The
    /// Phase 2c point-lookup-at-seq-minus-1 step will populate
    /// pre-images; callers assembling a `ChangeRecord` can treat
    /// an empty Row on a Delete as "pre-image pending".
    ///
    /// `read_seq` is typically `self.read_seq().0`; callers pass it
    /// explicitly so they can pin a consistent snapshot across
    /// multiple calls.
    pub fn scan_memtable_changes(
        &self,
        since_seq: u64,
        read_seq: SeqNum,
    ) -> Result<Vec<ChangeTuple>> {
        // Use `snapshot_all_versions` (not `snapshot_entries`) so
        // put+delete pairs on the same key both surface — the
        // point-lookup iterator dedups superseded versions, which
        // is wrong for a change feed.
        let entries = self.memtable.snapshot_all_versions(read_seq);
        let mut out: Vec<ChangeTuple> = Vec::new();
        for entry in entries {
            if entry.seq.0 <= since_seq {
                continue;
            }
            // Issue #33 fix: for Delete ops, the pre-image row is
            // encoded inline as the value (see apply_batch /
            // write_internal). Empty value → no pre-image was
            // available at delete time.
            let row = match entry.entry.op_type {
                OpType::Put => crate::engine::codec::decode_row(&entry.entry.value)?,
                OpType::Delete => {
                    if entry.entry.value.is_empty() {
                        crate::types::value::Row::default()
                    } else {
                        crate::engine::codec::decode_row(&entry.entry.value)?
                    }
                }
            };
            out.push(ChangeTuple {
                seq: entry.seq.0,
                op_type: entry.entry.op_type,
                row,
                pk_bytes: entry.user_key.to_vec(),
            });
        }
        out.sort_by_key(|t| t.seq);
        Ok(out)
    }

    /// Issue #29 Phase 2c: point-lookup by pre-encoded user_key at
    /// an explicit seq. Thin wrapper over
    /// `read_path::point_lookup_at_seq` so change-feed callers can
    /// probe a prior state without re-encoding PK values they already
    /// have as bytes.
    pub fn point_lookup_by_user_key_at_seq(
        &self,
        user_key_bytes: &[u8],
        read_seq: SeqNum,
    ) -> Result<Option<Row>> {
        crate::engine::read_path::point_lookup_at_seq(self, user_key_bytes, read_seq)
    }

    /// Issue #29 Phase 2c: variant of `scan_tail_changes` that also
    /// reconstructs delete pre-images. For every Delete op in the
    /// result, the `row` field carries the last live state of the
    /// key at `seq - 1` (not `Row::default()` as Phase 2a/b did).
    ///
    /// Costs one extra point-lookup per Delete op. Callers on the
    /// happy path (mostly Puts) pay nothing extra; delete-heavy
    /// workloads can opt out by calling `scan_tail_changes` instead.
    pub fn scan_tail_changes_with_pre_image(
        &self,
        since_seq: u64,
        read_seq: SeqNum,
    ) -> Result<Vec<ChangeTuple>> {
        let mut tuples = self.scan_tail_changes(since_seq, read_seq)?;
        for t in &mut tuples {
            if t.op_type != OpType::Delete {
                continue;
            }
            if t.seq == 0 {
                // Can't look up at seq-1 = u64::MAX sentinel;
                // leave the Row empty. seq=0 is never a real
                // committed op anyway.
                continue;
            }
            let pre_image = crate::engine::read_path::point_lookup_at_seq(
                self,
                &t.pk_bytes,
                SeqNum(t.seq - 1),
            )?;
            if let Some(row) = pre_image {
                t.row = row;
            }
            // If None, the key was already tombstoned or absent at
            // seq-1 — a delete-of-delete or a legitimately new
            // tombstone with no history. Leave `row` as
            // Row::default() to signal "no pre-image available."
        }
        Ok(tuples)
    }

    /// Issue #29 Phase 2b: extend the change-feed scan to include
    /// L0 Parquet files. Every file whose `seq_max > since_seq`
    /// AND `seq_min <= read_seq` is opened + scanned; rows whose
    /// internal seq falls in `(since_seq, read_seq]` are decoded
    /// and returned alongside the memtable ops.
    ///
    /// Result is seq-ascending across memtable + L0. Callers treat
    /// this as the "live-tail" view of the change feed. L1..LN scan
    /// is Phase 2c (in-progress); DELETE pre-image reconstruction is
    /// in `scan_tail_changes_with_pre_image`.
    pub fn scan_tail_changes(&self, since_seq: u64, read_seq: SeqNum) -> Result<Vec<ChangeTuple>> {
        use crate::iceberg::DeletionVector;
        use crate::parquet::reader::ParquetReader;
        use crate::types::level::Level;
        use bytes::Bytes;

        // Issue #37 fix: pin the snapshot before ANY memtable or
        // file work. Pre-#37, this scan read `version_set.current()`
        // with no pin at all — GC could (and did, under the chaos-
        // monkey concurrent-export workload) delete an L0 Parquet
        // file while the change-feed cursor was about to open it,
        // producing `IO error: No such file or directory`. Holding
        // `_pin` for the duration of the scan ties every L0 file
        // the captured `Version` references to disk.
        let (_pin, version) = self.pin_current_snapshot();

        // Memtable first so the L0 rows append after the memtable
        // rows before the final sort. Either order is correct
        // (sort is the invariant), ordering this way just keeps
        // memory contiguous on the happy path where the memtable
        // dominates.
        let mut out = self.scan_memtable_changes(since_seq, read_seq)?;

        let files = version.files_at(Level(0));
        let base = self.catalog.base_path();
        for file in files {
            // Prune files that can't possibly contribute rows in
            // the requested range. `seq_min/seq_max` are the tight
            // bounds the compaction iterator + external analytical readers already
            // rely on.
            if file.meta.seq_max <= since_seq || file.meta.seq_min > read_seq.0 {
                continue;
            }

            let abs_parquet = base.join(&file.path);
            let parquet_bytes = std::fs::read(&abs_parquet).map_err(MeruError::Io)?;
            let reader = ParquetReader::open(Bytes::from(parquet_bytes), self.schema.clone())?;

            // DV: skip logically-deleted rows. A row in the file's
            // DV means a later partial compaction removed it; the
            // feed aligns with what point lookups / range scans
            // actually observe.
            let dv_bitmap = match (&file.dv_path, file.dv_offset, file.dv_length) {
                (Some(dv_path), Some(offset), Some(length)) => {
                    let abs_dv = base.join(dv_path);
                    let puffin_bytes = std::fs::read(&abs_dv).map_err(MeruError::Io)?;
                    let start = offset as usize;
                    let end = start
                        .checked_add(length as usize)
                        .ok_or_else(|| MeruError::Corruption("DV offset+length overflow".into()))?;
                    if end > puffin_bytes.len() {
                        return Err(MeruError::Corruption(format!(
                            "DV blob out of range: path={dv_path} offset={offset} \
                             length={length} puffin_len={}",
                            puffin_bytes.len()
                        )));
                    }
                    Some(
                        DeletionVector::from_puffin_blob(&puffin_bytes[start..end])?
                            .bitmap()
                            .clone(),
                    )
                }
                (None, None, None) => None,
                _ => {
                    return Err(MeruError::Corruption(format!(
                        "inconsistent DV coords on file {}: dv_path={:?} dv_offset={:?} \
                         dv_length={:?}",
                        file.path, file.dv_path, file.dv_offset, file.dv_length
                    )));
                }
            };

            let rows = reader.read_physical_rows_with_positions(dv_bitmap.as_ref())?;
            for (ikey, row, _pos) in rows {
                let seq = ikey.seq.0;
                if seq <= since_seq || seq > read_seq.0 {
                    continue;
                }
                // Issue #33 fix: for Delete ops in L0 Parquet, the
                // pre-image row is the row data stored alongside
                // the tombstone (Parquet preserves the value column
                // through flush + compaction). If for any reason
                // the row decodes as empty (no pre-image captured
                // at write time, or legacy file), fall back to
                // Row::default().
                let op_row = match ikey.op_type {
                    OpType::Put => row,
                    OpType::Delete => {
                        if row.fields.is_empty() {
                            crate::types::value::Row::default()
                        } else {
                            row
                        }
                    }
                };
                // Encode pk_bytes from the InternalKey's decoded
                // pk_values so the tuple carries the same key
                // encoding the memtable path uses (both go through
                // `encode_pk_fields` internally).
                let pk_bytes = crate::types::key::InternalKey::encode_user_key(
                    ikey.pk_values(),
                    &self.schema,
                )?;
                out.push(ChangeTuple {
                    seq,
                    op_type: ikey.op_type,
                    row: op_row,
                    pk_bytes,
                });
            }
        }
        out.sort_by_key(|t| t.seq);
        Ok(out)
    }

    /// Catalog base directory (for external analytics: point DuckDB at Parquet files).
    pub fn catalog_path(&self) -> String {
        self.catalog.base_path().to_string_lossy().to_string()
    }

    /// Export the current catalog snapshot as an Apache Iceberg v2
    /// `metadata.json` under `target_dir`. Delegates to
    /// [`crate::iceberg::IcebergCatalog::export_to_iceberg`] — see that
    /// method's docs for the exact shape, field mapping, and limitations.
    ///
    /// Returns the absolute path of the emitted `v{N}.metadata.json`.
    pub async fn export_iceberg(
        &self,
        target_dir: impl AsRef<std::path::Path>,
    ) -> Result<std::path::PathBuf> {
        // Issue #24: pin the current snapshot for the lifetime of the
        // export. Without the pin, `gc_pending_deletions()` can run
        // concurrently (the background compaction worker's heartbeat
        // calls it) while the export is mid-write. If compaction has
        // obsoleted a file referenced by the exported manifest and the
        // wall-clock grace window has elapsed, GC deletes it — the
        // Iceberg JSON we just emitted points at a missing Parquet
        // file. Worse, a scan immediately after the export picks up
        // the new `Version` that still lists the deleted file and
        // fails with ENOENT.
        //
        // The read path (`get`/`scan`) has had this pin since
        // BUG-0007..0013; export_iceberg was a missed call-site in
        // that sweep. Hold the pin until the catalog write fully
        // returns — the Iceberg catalog also rereads the manifest
        // internally, and every file it might touch must remain
        // GC-ineligible under our snapshot refcount.
        let (_pin, _version) = self.pin_current_snapshot();
        self.catalog.export_to_iceberg(target_dir).await
    }

    /// Re-read the Iceberg manifest from disk and install a new version.
    /// Used by read-only replicas to pick up snapshots written by the primary.
    ///
    /// IMP-15: validates that all data files and Puffin files referenced by the
    /// new manifest exist on disk before swapping the version. If any file is
    /// missing (e.g. not yet replicated), the refresh is rejected and the
    /// current version stays in place.
    ///
    /// Issue #6: `visible_seq` is also advanced to the new manifest's
    /// max sequence number. Without this, the replica's `read_seq()`
    /// stays pinned at the open-time value and filters out any row
    /// whose seq was allocated after the replica opened — the replica
    /// would silently hide data that the primary wrote. The
    /// `set_at_least` semantic (monotonic, non-decreasing) guarantees
    /// we never regress `visible_seq` if a refresh happens to pick up
    /// an older snapshot (shouldn't happen in practice, but the guard
    /// is cheap).
    pub async fn refresh(&self) -> Result<()> {
        let new_version = self.catalog.refresh(self.schema.clone()).await?;

        // IMP-15: pre-flight check — every referenced file must exist.
        let base = self.catalog.base_path();
        for files in new_version.levels.values() {
            for file in files {
                let data_path = base.join(&file.path);
                if !data_path.exists() {
                    return Err(MeruError::ObjectStore(format!(
                        "refresh: referenced data file missing: {}",
                        file.path,
                    )));
                }
                if let Some(ref dv_path) = file.dv_path {
                    let puffin_path = base.join(dv_path);
                    if !puffin_path.exists() {
                        return Err(MeruError::ObjectStore(format!(
                            "refresh: referenced puffin file missing: {dv_path}",
                        )));
                    }
                }
            }
        }

        // Advance visible_seq (and global_seq) so the read path accepts
        // any row newly introduced by the primary. Must be done BEFORE
        // installing the new version — once a reader observes the new
        // version, it must see consistent read_seq/files.
        let new_max_seq: u64 = new_version
            .levels
            .values()
            .flat_map(|files| files.iter().map(|f| f.meta.seq_max))
            .max()
            .unwrap_or(0);
        if new_max_seq > 0 {
            self.visible_seq.set_at_least(new_max_seq);
            self.global_seq.set_at_least(new_max_seq + 1);
        }

        self.version_set.install(new_version);

        // Issue #10: clear the row cache after version install so the
        // replica doesn't serve stale values for keys the primary
        // overwrote or deleted. The cache is populated from Parquet
        // file reads during point lookups; on the primary, writes
        // invalidate per-key and compaction clears wholesale, but a
        // read-only replica has no writes and no compactions — so
        // without this call, a cached value from the old version
        // survives every subsequent refresh. Same pattern as the
        // post-compaction cache clear in `compaction/job.rs`.
        if let Some(ref cache) = self.row_cache {
            cache.clear();
        }

        info!(new_max_seq, "version refreshed from disk");
        Ok(())
    }

    /// Enqueue obsoleted files for deferred deletion.
    ///
    /// `obsoleted_after_snapshot` is the `snapshot_id` of the version in
    /// which these files were still live — i.e., the version the
    /// compaction was based on. Any reader pinning a snapshot <= this
    /// value may still dereference these files, so GC will keep them
    /// alive until the last such pin releases.
    pub(crate) async fn enqueue_for_deletion(
        &self,
        paths: Vec<std::path::PathBuf>,
        obsoleted_after_snapshot: i64,
    ) {
        if paths.is_empty() {
            return;
        }
        let now = std::time::Instant::now();
        let mut pending = self.pending_deletions.lock().await;
        let added = paths.len();
        for path in paths {
            pending.push(PendingDelete {
                path,
                obsoleted_at: now,
                obsoleted_after_snapshot,
            });
        }
        // #30 observability: bump the cached counter and record the
        // oldest-enqueue timestamp under the lock so stats() sees a
        // value consistent with the Vec until the next GC sweep.
        self.pending_deletions_len
            .store(pending.len(), std::sync::atomic::Ordering::Relaxed);
        let mut oldest = self.pending_oldest_enqueue.write().unwrap();
        if oldest.is_none() && added > 0 {
            *oldest = Some(now);
        }
    }

    /// Delete files that are both (a) no longer pinned by any live
    /// internal reader AND (b) past the `gc_grace_period_secs` time
    /// grace for external (external analytics) readers. Called after compaction
    /// commits and periodically by the optional background worker.
    ///
    /// Version-pinned safety (fixes BUG-0007..0013): even when the
    /// time-based grace elapsed, GC refuses to delete files still
    /// referenced by any `Version` held via `pin_current_snapshot`.
    /// A 40 GB integrity scan can legitimately exceed the default
    /// 5-minute grace period; without version pinning, GC would delete
    /// files the scan still needs to open, producing spurious
    /// `IO error: No such file or directory` failures.
    pub async fn gc_pending_deletions(&self) {
        let grace = std::time::Duration::from_secs(self.config.gc_grace_period_secs);
        let min_pinned = self.min_pinned_snapshot();
        let mut pending = self.pending_deletions.lock().await;
        let mut remaining = Vec::new();
        for entry in pending.drain(..) {
            // Keep if any live reader's snapshot could still see this
            // file. A pin at snapshot S means S was live for that
            // reader; the reader's `Version` references files visible
            // up to S. Our file was live through `obsoleted_after_snapshot`
            // inclusive, so pins at S <= obsoleted_after_snapshot still
            // reference it.
            let still_pinned = min_pinned.is_some_and(|m| m <= entry.obsoleted_after_snapshot);
            if still_pinned {
                crate::engine::metrics::inc(crate::engine::metrics::GC_FILES_DEFERRED_BY_PIN_TOTAL);
                remaining.push(entry);
                continue;
            }
            // Time-based grace for external readers (DuckDB, Spark,
            // etc.) which don't participate in the pin protocol.
            if entry.obsoleted_at.elapsed() >= grace {
                match tokio::fs::remove_file(&entry.path).await {
                    Ok(_) => {
                        crate::engine::metrics::inc(crate::engine::metrics::GC_FILES_DELETED_TOTAL);
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            crate::engine::metrics::inc(crate::engine::metrics::IO_ERRORS_TOTAL);
                            warn!(path = %entry.path.display(), error = %e,
                                  "failed to GC obsoleted file");
                        } else {
                            // Already gone — count as deleted so metrics
                            // don't undercount the set of removed files.
                            crate::engine::metrics::inc(
                                crate::engine::metrics::GC_FILES_DELETED_TOTAL,
                            );
                        }
                    }
                }
            } else {
                crate::engine::metrics::inc(
                    crate::engine::metrics::GC_FILES_DEFERRED_BY_GRACE_TOTAL,
                );
                remaining.push(entry);
            }
        }
        // #30: refresh the cached counter + oldest-enqueue timestamp
        // to match the trimmed Vec. The oldest is now the min over
        // what remains; empty Vec → None (queue fully drained).
        let new_oldest = remaining.iter().map(|e| e.obsoleted_at).min();
        *pending = remaining;
        self.pending_deletions_len
            .store(pending.len(), std::sync::atomic::Ordering::Relaxed);
        *self.pending_oldest_enqueue.write().unwrap() = new_oldest;
        crate::engine::metrics::inc(crate::engine::metrics::GC_SWEEPS_TOTAL);
    }

    /// Pin the current snapshot for the lifetime of the returned guard
    /// and return the pinned `Version` Arc. While the pin is held, GC
    /// will NOT delete any file whose `obsoleted_after_snapshot` is
    /// `>=` the pinned snapshot_id — the reader is guaranteed every
    /// file its `Version` references will remain on disk.
    ///
    /// Used by the read path (point_lookup, range_scan) to prevent
    /// file-GC races during long reads. Release by dropping the guard.
    ///
    /// Public so integration tests can exercise the pin contract; not
    /// meant to be called from user code.
    pub fn pin_current_snapshot(
        &self,
    ) -> (
        SnapshotPin<'_>,
        std::sync::Arc<crate::iceberg::version::Version>,
    ) {
        // Issue #37 fix: acquire `live_snapshots` BEFORE reading the
        // current version. Pre-#37 this read the version first, then
        // took the lock — the window between the two calls let GC
        // observe `min_pinned_snapshot == None`, decide a file was
        // deletable, and unlink it before our pin registered. A
        // subsequent read against our captured `Version` would then
        // fail with `ENOENT`.
        //
        // Lock-first makes the check-then-act atomic with respect
        // to GC's pin-status read (which takes the same lock):
        //   - If GC's pass runs BEFORE our lock acquisition, we
        //     subsequently observe the NEWER version (compaction
        //     installed before GC enqueued the delete), which does
        //     not reference the deleted file.
        //   - If GC's pass runs AFTER our lock acquisition, GC sees
        //     our pin in `min_pinned_snapshot` and defers any file
        //     whose `obsoleted_after_snapshot >= our_id`.
        // Either way, files in our pinned `Version` remain on disk.
        let mut pins = self.live_snapshots.lock().unwrap();
        let version_guard = self.version_set.current();
        let snapshot_id = version_guard.snapshot_id;
        let version: std::sync::Arc<crate::iceberg::version::Version> = (*version_guard).clone();
        drop(version_guard);
        *pins.entry(snapshot_id).or_insert(0) += 1;
        drop(pins);
        (
            SnapshotPin {
                engine: self,
                snapshot_id,
            },
            version,
        )
    }

    /// The smallest pinned snapshot_id across all live readers, or
    /// `None` if nothing is pinned. GC uses this as the watermark
    /// below which files can be safely deleted.
    pub fn min_pinned_snapshot(&self) -> Option<i64> {
        self.live_snapshots.lock().unwrap().keys().next().copied()
    }

    /// Graceful shutdown: flush all in-memory data to durable storage, fsync
    /// the WAL, and set the closed flag. After `close()` returns, all data
    /// written before this call is durable on disk. Subsequent write/flush/
    /// compact calls return `MeruError::Closed`. Reads remain available
    /// until the `MeruEngine` is dropped.
    ///
    /// Follows the RocksDB/sled pattern: the library provides the method,
    /// the host process calls it in its shutdown path. No signal handlers
    /// are installed.
    #[instrument(skip(self), fields(op = "close"))]
    pub async fn close(self: &Arc<Self>) -> Result<()> {
        if self.read_only {
            // Read-only instances have nothing to flush. Just set the flag.
            self.closed.store(true, Ordering::Release);
            info!("read-only engine closed");
            return Ok(());
        }

        // Prevent double-close.
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }

        info!("engine closing — flushing memtable to durable storage");

        // Rotate the active memtable so its contents become immutable and
        // can be flushed. Same pattern as the manual flush() path.
        {
            let _rotation_guard = self.rotation_lock.lock().await;
            if self.memtable.active_size_bytes() > 0 {
                let next_seq = self.global_seq.current().next();
                self.memtable.rotate(next_seq);
                {
                    let mut wal = self.wal.lock().await;
                    wal.rotate()?;
                }
            }
        }

        // Flush all immutable memtables.
        while self.memtable.oldest_immutable().is_some() {
            crate::engine::flush::run_flush(self).await?;
        }

        // Final WAL sync for any trailing writes that landed between
        // the last flush and the close flag.
        {
            let mut wal = self.wal.lock().await;
            wal.sync()?;
        }

        // Issue #11: drain any queued deletions whose grace period
        // has elapsed. Background workers are already being shut down
        // at the MeruDB layer, so this may be the last chance to
        // clean up obsoleted files before the engine goes away.
        self.gc_pending_deletions().await;

        info!("engine closed — all data flushed and synced");
        Ok(())
    }

    /// Returns `true` if `close()` has been called.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Snapshot of the per-level compaction reservations. Used only by
    /// regression tests to assert that `LevelReservation`'s Drop impl
    /// freed the levels after a compaction completes.
    #[doc(hidden)]
    pub async fn __compacting_levels_snapshot(
        &self,
    ) -> std::collections::HashSet<crate::types::level::Level> {
        self.compacting_levels.lock().await.clone()
    }

    /// Snapshot of engine statistics. Lock-free on the version side (ArcSwap),
    /// brief read lock on memtable. Zero overhead on the hot path — only runs
    /// when explicitly called.
    pub fn stats(&self) -> crate::engine::stats::EngineStats {
        let version = self.version_set.current();
        let max_level = version.max_level().0;

        let mut levels = Vec::new();
        for l in 0..=max_level {
            let level = crate::types::level::Level(l);
            let files = version.files_at(level);
            if files.is_empty() {
                continue;
            }
            let file_stats: Vec<crate::engine::stats::FileStats> = files
                .iter()
                .map(|f| crate::engine::stats::FileStats {
                    path: f.path.clone(),
                    file_size: f.meta.file_size,
                    num_rows: f.meta.num_rows,
                    seq_range: (f.meta.seq_min, f.meta.seq_max),
                    has_dv: f.has_dv(),
                })
                .collect();
            levels.push(crate::engine::stats::LevelStats {
                level: l,
                file_count: files.len(),
                total_bytes: version.level_bytes(level),
                total_rows: files.iter().map(|f| f.meta.num_rows).sum(),
                files: file_stats,
            });
        }

        let memtable = crate::engine::stats::MemtableStats {
            active_size_bytes: self.memtable.active_size_bytes(),
            active_entry_count: self.memtable.active_entry_count(),
            flush_threshold: self.memtable.flush_threshold(),
            immutable_count: self.memtable.immutable_count(),
        };

        let cache = match &self.row_cache {
            Some(c) => crate::engine::stats::CacheStats {
                capacity: c.cap(),
                size: c.len(),
                hit_count: c.hit_count(),
                miss_count: c.miss_count(),
            },
            None => crate::engine::stats::CacheStats {
                capacity: 0,
                size: 0,
                hit_count: 0,
                miss_count: 0,
            },
        };

        // #30 observability: synchronous read of the pending-deletions
        // counters — no await, no tokio Mutex access. Stale by at most
        // one enqueue/sweep transition, which is fine for a stats
        // snapshot.
        let pending_count = self
            .pending_deletions_len
            .load(std::sync::atomic::Ordering::Relaxed);
        let oldest_age_secs = self
            .pending_oldest_enqueue
            .read()
            .unwrap()
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0);

        crate::engine::stats::EngineStats {
            snapshot_id: version.snapshot_id,
            current_seq: self.global_seq.current().0,
            levels,
            memtable,
            cache,
            gc: crate::engine::stats::GcStats {
                pending_count,
                oldest_pending_age_secs: oldest_age_secs,
            },
            compaction: crate::engine::stats::CompactionStats {
                inflight_levels: self
                    .compacting_levels_len
                    .load(std::sync::atomic::Ordering::Relaxed),
            },
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        schema::{ColumnDef, ColumnType},
        value::Row,
    };

    fn test_schema() -> TableSchema {
        TableSchema {
            table_name: "test".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,

                    ..Default::default()
                },
                ColumnDef {
                    name: "val".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: true,

                    ..Default::default()
                },
            ],
            primary_key: vec![0],

            ..Default::default()
        }
    }

    fn test_config(tmp: &tempfile::TempDir) -> crate::engine::config::EngineConfig {
        crate::engine::config::EngineConfig {
            schema: test_schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn open_creates_fresh_engine() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
        // Issue #8: a fresh engine has seen no writes → visible frontier is 0.
        assert_eq!(engine.read_seq().0, 0);
        assert_eq!(engine.schema().table_name, "test");
    }

    /// Issue #30 observability: `stats().gc.pending_count` tracks the
    /// pending-deletions queue size without blocking on the tokio
    /// mutex that `gc_pending_deletions` holds across awaited file
    /// deletes. A synchronous `stats()` caller must never await.
    /// Issue #30 observability: `stats().compaction.inflight_levels`
    /// tracks the size of `compacting_levels` without blocking on
    /// its tokio Mutex. On a quiescent engine the counter is zero;
    /// after forcing a reservation it reflects the inserted levels.
    #[tokio::test]
    async fn stats_compaction_inflight_tracks_reserved_levels() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
        // Fresh engine: no compactions.
        assert_eq!(engine.stats().compaction.inflight_levels, 0);

        // Simulate a reservation by inserting levels directly under
        // the lock + bumping the counter (the real compaction-job
        // reservation path does exactly this).
        {
            let mut busy = engine.compacting_levels.lock().await;
            busy.insert(crate::types::level::Level(0));
            busy.insert(crate::types::level::Level(1));
            engine
                .compacting_levels_len
                .store(busy.len(), std::sync::atomic::Ordering::Relaxed);
        }
        assert_eq!(engine.stats().compaction.inflight_levels, 2);

        // Simulate release.
        {
            let mut busy = engine.compacting_levels.lock().await;
            busy.clear();
            engine
                .compacting_levels_len
                .store(busy.len(), std::sync::atomic::Ordering::Relaxed);
        }
        assert_eq!(engine.stats().compaction.inflight_levels, 0);
    }

    #[tokio::test]
    async fn stats_gc_tracks_pending_deletions() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        // Fresh engine: queue empty.
        let s0 = engine.stats();
        assert_eq!(s0.gc.pending_count, 0);
        assert_eq!(s0.gc.oldest_pending_age_secs, 0);

        // Enqueue three fake paths at snapshot=0 so the grace-period
        // check treats them as unpinned.
        engine
            .enqueue_for_deletion(
                vec![
                    tmp.path().join("ghost1.parquet"),
                    tmp.path().join("ghost2.parquet"),
                    tmp.path().join("ghost3.parquet"),
                ],
                /*obsoleted_after_snapshot=*/ 0,
            )
            .await;
        let s1 = engine.stats();
        assert_eq!(s1.gc.pending_count, 3);
        // oldest_age may be 0 in the same-tick case; just assert it
        // doesn't panic or overflow.
        let _ = s1.gc.oldest_pending_age_secs;

        // Run GC; with default grace period the entries stay pending
        // (time-based grace not yet elapsed) but the counter invariant
        // — count == Vec length — must still hold.
        engine.gc_pending_deletions().await;
        let s2 = engine.stats();
        assert_eq!(
            s2.gc.pending_count, 3,
            "still within grace, queue length unchanged"
        );
    }

    /// Issue #8 regression: the first put's returned seq must be
    /// strictly greater than the pre-put read_seq. With the old
    /// exclusive-upper-bound semantics for visible_seq, a fresh DB
    /// had read_seq = 1 AND the first allocate returned 1 — violating
    /// monotonicity at the first operation.
    #[tokio::test]
    async fn first_put_seq_greater_than_initial_read_seq() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
        let pre = engine.read_seq();
        let seq = engine
            .put(
                vec![FieldValue::Int64(1)],
                Row::new(vec![Some(FieldValue::Int64(1)), None]),
            )
            .await
            .unwrap();
        assert!(
            seq > pre,
            "Issue #8: first put seq {:?} must be > pre-put read_seq {:?}",
            seq,
            pre
        );
    }

    #[tokio::test]
    async fn open_after_recovery_has_correct_seq() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
        assert!(engine.read_seq().0 == 0); // fresh
        assert_eq!(engine.schema().table_name, "test");
    }

    #[tokio::test]
    async fn put_and_get() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        engine
            .put(
                vec![FieldValue::Int64(1)],
                Row::new(vec![
                    Some(FieldValue::Int64(1)),
                    Some(FieldValue::Bytes(bytes::Bytes::from("hello"))),
                ]),
            )
            .await
            .unwrap();

        let row = engine.get(&[FieldValue::Int64(1)]).unwrap();
        assert!(row.is_some());
    }

    #[tokio::test]
    async fn get_missing_key() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
        let row = engine.get(&[FieldValue::Int64(999)]).unwrap();
        assert!(row.is_none());
    }

    #[tokio::test]
    async fn delete_removes_key() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        engine
            .put(
                vec![FieldValue::Int64(1)],
                Row::new(vec![Some(FieldValue::Int64(1)), None]),
            )
            .await
            .unwrap();
        assert!(engine.get(&[FieldValue::Int64(1)]).unwrap().is_some());

        engine.delete(vec![FieldValue::Int64(1)]).await.unwrap();
        assert!(engine.get(&[FieldValue::Int64(1)]).unwrap().is_none());
    }

    #[tokio::test]
    async fn multiple_puts_and_scan() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        for i in 1..=10i64 {
            engine
                .put(
                    vec![FieldValue::Int64(i)],
                    Row::new(vec![
                        Some(FieldValue::Int64(i)),
                        Some(FieldValue::Bytes(bytes::Bytes::from(format!("val{i}")))),
                    ]),
                )
                .await
                .unwrap();
        }

        // Full scan.
        let results = engine.scan(None, None).unwrap();
        assert_eq!(results.len(), 10);

        // Range scan: keys 3..7 (exclusive end).
        let results = engine
            .scan(Some(&[FieldValue::Int64(3)]), Some(&[FieldValue::Int64(7)]))
            .unwrap();
        assert_eq!(results.len(), 4); // 3, 4, 5, 6
    }

    #[tokio::test]
    async fn overwrite_updates_value() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        engine
            .put(
                vec![FieldValue::Int64(1)],
                Row::new(vec![
                    Some(FieldValue::Int64(1)),
                    Some(FieldValue::Bytes(bytes::Bytes::from("v1"))),
                ]),
            )
            .await
            .unwrap();
        engine
            .put(
                vec![FieldValue::Int64(1)],
                Row::new(vec![
                    Some(FieldValue::Int64(1)),
                    Some(FieldValue::Bytes(bytes::Bytes::from("v2"))),
                ]),
            )
            .await
            .unwrap();

        let row = engine.get(&[FieldValue::Int64(1)]).unwrap().unwrap();
        // Should see the latest value.
        let val = row.get(1).unwrap();
        assert_eq!(*val, FieldValue::Bytes(bytes::Bytes::from("v2")));
    }

    #[tokio::test]
    async fn seq_increases_monotonically() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        let s1 = engine
            .put(
                vec![FieldValue::Int64(1)],
                Row::new(vec![Some(FieldValue::Int64(1)), None]),
            )
            .await
            .unwrap();
        let s2 = engine
            .put(
                vec![FieldValue::Int64(2)],
                Row::new(vec![Some(FieldValue::Int64(2)), None]),
            )
            .await
            .unwrap();
        let s3 = engine.delete(vec![FieldValue::Int64(1)]).await.unwrap();

        assert!(s1 < s2);
        assert!(s2 < s3);
    }

    #[tokio::test]
    async fn flush_and_scan() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        for i in 1..=5i64 {
            engine
                .put(
                    vec![FieldValue::Int64(i)],
                    Row::new(vec![Some(FieldValue::Int64(i)), None]),
                )
                .await
                .unwrap();
        }

        // Flush to Parquet.
        engine.flush().await.unwrap();

        // Data should still be scannable (from Parquet or re-read).
        // At minimum, the scan should not error.
        let _results = engine.scan(None, None);
    }

    #[tokio::test]
    async fn wal_recovery() {
        let tmp = tempfile::tempdir().unwrap();

        // Write some data.
        {
            let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
            engine
                .put(
                    vec![FieldValue::Int64(42)],
                    Row::new(vec![
                        Some(FieldValue::Int64(42)),
                        Some(FieldValue::Bytes(bytes::Bytes::from("persisted"))),
                    ]),
                )
                .await
                .unwrap();
            // Drop engine without explicit close — simulates crash.
        }

        // Reopen — WAL recovery should replay the write.
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
        let row = engine.get(&[FieldValue::Int64(42)]).unwrap();
        assert!(row.is_some(), "WAL recovery should restore the row");
    }

    #[tokio::test]
    async fn close_flushes_and_blocks_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        // Write data that sits in the memtable.
        engine
            .put(
                vec![FieldValue::Int64(1)],
                Row::new(vec![
                    Some(FieldValue::Int64(1)),
                    Some(FieldValue::Bytes(bytes::Bytes::from("before_close"))),
                ]),
            )
            .await
            .unwrap();

        assert!(!engine.is_closed());
        engine.close().await.unwrap();
        assert!(engine.is_closed());

        // Writes must fail after close.
        let err = engine
            .put(
                vec![FieldValue::Int64(2)],
                Row::new(vec![Some(FieldValue::Int64(2)), None]),
            )
            .await;
        assert!(
            matches!(err, Err(MeruError::Closed)),
            "put after close must return Closed"
        );

        // Delete must fail.
        let err = engine.delete(vec![FieldValue::Int64(1)]).await;
        assert!(matches!(err, Err(MeruError::Closed)));

        // Reads still work.
        let row = engine.get(&[FieldValue::Int64(1)]).unwrap();
        assert!(row.is_some(), "reads must still work after close");
    }

    #[tokio::test]
    async fn close_data_survives_reopen() {
        let tmp = tempfile::tempdir().unwrap();

        // Write, close, drop.
        {
            let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
            for i in 1..=10i64 {
                engine
                    .put(
                        vec![FieldValue::Int64(i)],
                        Row::new(vec![
                            Some(FieldValue::Int64(i)),
                            Some(FieldValue::Bytes(bytes::Bytes::from(format!("v{i}")))),
                        ]),
                    )
                    .await
                    .unwrap();
            }
            engine.close().await.unwrap();
        }

        // Reopen — data was flushed by close(), should be in Parquet.
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();
        for i in 1..=10i64 {
            let row = engine.get(&[FieldValue::Int64(i)]).unwrap();
            assert!(row.is_some(), "key {i} must survive close + reopen");
        }
    }

    #[tokio::test]
    async fn double_close_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(test_config(&tmp)).await.unwrap();

        engine.close().await.unwrap();
        // Second close should succeed silently.
        engine.close().await.unwrap();
        assert!(engine.is_closed());
    }
}
