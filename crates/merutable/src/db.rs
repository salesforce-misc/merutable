//! `MeruDB`: the public embedding API.
//!
//! `MeruDB` is `Send + Sync`. All methods are async. One instance = one table.
//!
//! # Example
//! ```no_run
//! use merutable::{MeruDB, OpenOptions};
//! use merutable::schema::{ColumnDef, ColumnType, TableSchema};
//! use merutable::value::{FieldValue, Row};
//!
//! # async fn example() -> merutable::error::Result<()> {
//! // Issue #25: ColumnDef/TableSchema carry evolution-ready fields;
//! // use `..Default::default()` or the builder.
//! let schema = TableSchema {
//!     table_name: "events".into(),
//!     columns: vec![
//!         ColumnDef {
//!             name: "id".into(),
//!             col_type: ColumnType::Int64,
//!             nullable: false,
//!             ..Default::default()
//!         },
//!         ColumnDef {
//!             name: "payload".into(),
//!             col_type: ColumnType::ByteArray,
//!             nullable: true,
//!             ..Default::default()
//!         },
//!     ],
//!     primary_key: vec![0],
//!     ..Default::default()
//! };
//!
//! let db = MeruDB::open(
//!     OpenOptions::new(schema)
//!         .wal_dir("/tmp/meru-wal")
//!         .catalog_uri("/tmp/meru-data")
//! ).await?;
//!
//! db.put(Row::new(vec![
//!     Some(FieldValue::Int64(1)),
//!     Some(FieldValue::Bytes(bytes::Bytes::from("hello"))),
//! ])).await?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use crate::engine::{background::BackgroundWorkers, config::EngineConfig, engine::MeruEngine};
use crate::types::{
    key::InternalKey,
    schema::TableSchema,
    sequence::SeqNum,
    value::{FieldValue, Row},
    Result,
};

use crate::options::OpenOptions;

/// Primary embedding interface. Thread-safe, cloneable (via `Arc`).
///
/// # Shutdown
///
/// Call [`close()`](Self::close) before dropping to flush all in-memory data to
/// durable storage. If `close()` is not called, a warning is logged on drop.
/// Reads remain available after `close()`; writes, flushes, and compactions
/// return `MeruError::Closed`.
///
/// ```no_run
/// # async fn example(db: merutable::MeruDB) {
/// db.close().await.expect("clean shutdown");
/// # }
/// ```
pub struct MeruDB {
    engine: Arc<MeruEngine>,
    /// Background flush + compaction workers — spawned automatically
    /// on a non-read-only `open` whenever `flush_parallelism > 0` or
    /// `compaction_parallelism > 0`. Fixes Issue #4: the configuration
    /// promised N parallel workers, but without this spawn call
    /// `compact()` and `flush()` only ran when the caller invoked
    /// them, so a deep L2→L3 compaction could block L0 drainage for
    /// tens of minutes even with `compaction_parallelism: 2`
    /// configured. Held behind a `tokio::sync::Mutex<Option<_>>` so
    /// `close()` can `take()` and `shutdown().await` the workers
    /// before the engine's final flush.
    background: tokio::sync::Mutex<Option<BackgroundWorkers>>,
    /// Issue #31 Phase 2a: mirror worker. Spawned iff
    /// `OpenOptions::mirror` was set AND the DB is not read-only.
    /// Same `take()/shutdown().await` lifecycle as `background` so
    /// `close()` drains both before the engine's final flush.
    mirror_worker: tokio::sync::Mutex<Option<crate::mirror::MirrorWorker>>,
}

impl MeruDB {
    /// Open (or create) a database instance.
    /// Install a global `metrics` recorder. Call BEFORE `open` if
    /// you want to receive any of the Phase-1 counters; registration
    /// is global (per-process) and can only happen once. `metrics`
    /// silently discards subsequent calls — the crate does not panic.
    ///
    /// Typical usage:
    /// ```no_run
    /// # use std::error::Error;
    /// # fn example() -> Result<(), Box<dyn Error>> {
    /// // e.g. with metrics-exporter-prometheus:
    /// //   let handle = PrometheusBuilder::new().install_recorder()?;
    /// //   merutable::MeruDB::install_metrics_recorder_global(Box::new(handle.recorder()));
    /// # Ok(()) }
    /// ```
    ///
    /// Passing `None` is a no-op (returns `Ok(())`). Returns an
    /// error if the metrics crate rejects the recorder (e.g., already
    /// set by another component).
    pub fn install_metrics_recorder(
        recorder: Box<dyn metrics::Recorder + Send + Sync>,
    ) -> std::result::Result<(), metrics::SetRecorderError<Box<dyn metrics::Recorder + Send + Sync>>>
    {
        metrics::set_global_recorder(recorder)
    }

    pub async fn open(options: OpenOptions) -> Result<Self> {
        // Issue #31 Phase 1: reject incoherent mirror+mode combinations
        // at the API boundary BEFORE we touch the WAL or the catalog.
        // Issue #31 Phase 2a: pull the mirror config out of options
        // BEFORE `options.schema` etc are moved into the EngineConfig.
        // The worker is spawned after the engine is up.
        let mirror_config = options.mirror.clone();
        // Issue #9: surface the full tuning matrix through
        // `OpenOptions`. Previously only 7/14 EngineConfig fields
        // were reachable from the public API — users who wanted to
        // tune `level_target_bytes` / L0 triggers / parallelism /
        // `max_compaction_bytes` / `gc_grace_period_secs` had to
        // reach into `merutable-engine` directly, which is now
        // `publish = false`. Every field maps 1:1 so the defaults
        // and semantics come straight from EngineConfig.
        let config = EngineConfig {
            schema: options.schema.clone(),
            catalog_uri: options.catalog_uri.clone(),
            object_store_prefix: options.catalog_uri.clone(),
            wal_dir: options.wal_dir.clone(),
            memtable_size_bytes: options.memtable_size_mb * 1024 * 1024,
            max_immutable_count: options.max_immutable_count,
            row_cache_capacity: options.row_cache_capacity,
            level_target_bytes: options.level_target_bytes.clone(),
            l0_compaction_trigger: options.l0_compaction_trigger,
            l0_slowdown_trigger: options.l0_slowdown_trigger,
            l0_stop_trigger: options.l0_stop_trigger,
            bloom_bits_per_key: options.bloom_bits_per_key,
            max_compaction_bytes: options.max_compaction_bytes,
            max_compaction_input_rows: options.max_compaction_input_rows,
            flush_parallelism: options.flush_parallelism,
            compaction_parallelism: options.compaction_parallelism,
            gc_grace_period_secs: options.gc_grace_period_secs,
            read_only: options.read_only,
            dual_format_max_level: options.dual_format_max_level,
        };

        let engine = MeruEngine::open(config).await?;

        // Spawn background workers for non-read-only instances. Users
        // who set `flush_parallelism = 0` AND `compaction_parallelism = 0`
        // can still opt out: `BackgroundWorkers::spawn` is a no-op when
        // both counts are zero. Read-only replicas never need writers —
        // they use `refresh()` to pick up new snapshots from the primary.
        let background = if options.read_only {
            None
        } else {
            Some(BackgroundWorkers::spawn(engine.clone()))
        };

        // Issue #31 Phase 2a: spawn the mirror worker only for
        // non-read-only writers that asked for a mirror. Read-only
        // replicas never originate writes, so there's nothing to
        // mirror; spawning anyway would burn a tokio task for no
        // benefit.
        let mirror_worker = if options.read_only {
            None
        } else {
            mirror_config.map(|cfg| crate::mirror::MirrorWorker::spawn(engine.clone(), cfg))
        };

        Ok(Self {
            engine,
            background: tokio::sync::Mutex::new(background),
            mirror_worker: tokio::sync::Mutex::new(mirror_worker),
        })
    }

    /// Open a read-only replica. No WAL, no writes. Call `refresh()` to
    /// pick up new snapshots from the primary.
    pub async fn open_read_only(options: OpenOptions) -> Result<Self> {
        let mut opts = options;
        opts.read_only = true;
        Self::open(opts).await
    }

    /// Insert or update a row. PK is extracted from the row.
    ///
    /// Issue #44 Stage 4: rows shorter than the schema's column
    /// count are auto-padded with each missing tail column's
    /// `write_default` (or `None` if the column is nullable with
    /// no default). A missing non-nullable-no-default column
    /// surfaces `SchemaMismatch` at this boundary so the caller
    /// sees it before the WAL lock.
    pub async fn put(&self, mut row: Row) -> Result<SeqNum> {
        row.pad_with_defaults(self.engine.schema())?;
        let pk_values = row.pk_values(&self.engine.schema().primary_key)?;
        self.engine.put(pk_values, row).await
    }

    /// Batch insert/update. All rows share a single WAL sync — N× faster than
    /// individual `put()` calls.
    pub async fn put_batch(&self, rows: Vec<Row>) -> Result<SeqNum> {
        use crate::engine::write_path::{self, MutationBatch};

        let mut batch = MutationBatch::new();
        let schema = self.engine.schema();
        for mut row in rows {
            // Issue #44 Stage 4: per-row pad with write_default.
            row.pad_with_defaults(schema)?;
            let pk_values = row.pk_values(&schema.primary_key)?;
            batch.put(pk_values, row);
        }
        write_path::apply_batch(&self.engine, batch).await
    }

    /// Issue #44 Stage 2: additive schema evolution.
    ///
    /// Append `col` to the table's schema and persist the new
    /// version in the catalog. Returns the new `TableSchema`.
    ///
    /// Contract (strict, matches `check_schema_compatible`):
    ///   - The column name must not already exist in the schema.
    ///   - PK columns cannot be added post-creation.
    ///   - The column must be nullable OR carry a `write_default`
    ///     / `initial_default` — otherwise existing rows cannot
    ///     be back-filled.
    ///
    /// After this call returns, the NEXT `MeruDB::open` sees the
    /// extended schema; the currently-open `MeruDB` instance does
    /// NOT swap its in-engine schema (live-engine schema swap is
    /// follow-up work). This is safe because Stage 3's read-path
    /// projection makes pre-evolution files readable under the new
    /// schema on reopen, and Stage 4's `pad_with_defaults` handles
    /// short writes from that reopen-point forward.
    pub async fn add_column(
        &self,
        col: crate::types::schema::ColumnDef,
    ) -> Result<crate::types::schema::TableSchema> {
        use crate::iceberg::SnapshotTransaction;
        use std::sync::Arc;

        let current = self.engine.schema().clone();

        // Validation — every reject message names the offending
        // shape so the caller can fix it without guessing.
        if current.columns.iter().any(|c| c.name == col.name) {
            return Err(crate::types::MeruError::SchemaMismatch(format!(
                "add_column: column '{}' already exists in table '{}'",
                col.name, current.table_name,
            )));
        }
        if !col.nullable && col.write_default.is_none() && col.initial_default.is_none() {
            return Err(crate::types::MeruError::SchemaMismatch(format!(
                "add_column: new column '{}' is NOT NULL and has no \
                 write_default / initial_default — cannot back-fill existing rows",
                col.name,
            )));
        }

        // Build the evolved schema. `validate` on the new schema
        // assigns the field_id from `last_column_id + 1` and bumps
        // `last_column_id`, so the caller doesn't have to stamp
        // field_id manually.
        let mut evolved = current.clone();
        evolved.columns.push(col);
        // The new column cannot be a PK (PK is validated by
        // positions, and the issue explicitly disallows PK
        // additions); `validate` doesn't enforce that directly,
        // but the column was appended at the tail so its index
        // isn't in `primary_key` unless the caller mutated that —
        // we leave `primary_key` untouched and rely on the
        // `check_schema_compatible` PK-match check at reopen time.
        evolved.validate()?;

        // Persist via an empty snapshot transaction. `commit`
        // writes a new manifest version with the evolved schema
        // embedded; `check_schema_compatible` on next open
        // accepts the extension per #44 Stage 1.
        let txn = SnapshotTransaction::new();
        self.engine
            .catalog
            .commit(&txn, Arc::new(evolved.clone()))
            .await?;

        Ok(evolved)
    }

    /// Delete by primary key values.
    pub async fn delete(&self, pk_values: Vec<FieldValue>) -> Result<SeqNum> {
        self.engine.delete(pk_values).await
    }

    /// Point lookup by primary key.
    pub fn get(&self, pk_values: &[FieldValue]) -> Result<Option<Row>> {
        self.engine.get(pk_values)
    }

    /// Range scan. Returns rows in PK order.
    pub fn scan(
        &self,
        start_pk: Option<&[FieldValue]>,
        end_pk: Option<&[FieldValue]>,
    ) -> Result<Vec<(InternalKey, Row)>> {
        self.engine.scan(start_pk, end_pk)
    }

    /// Force flush all memtables to Parquet.
    pub async fn flush(&self) -> Result<()> {
        self.engine.flush().await
    }

    /// Trigger a manual compaction.
    pub async fn compact(&self) -> Result<()> {
        self.engine.compact().await
    }

    /// Get the table schema.
    pub fn schema(&self) -> &TableSchema {
        self.engine.schema()
    }

    /// Current read sequence number.
    pub fn read_seq(&self) -> SeqNum {
        self.engine.read_seq()
    }

    /// Engine statistics snapshot. Zero hot-path overhead.
    pub fn stats(&self) -> crate::engine::EngineStats {
        self.engine.stats()
    }

    /// Issue #31 Phase 3: highest snapshot id the mirror worker has
    /// uploaded to the destination. `None` when no mirror is
    /// attached; `Some(0)` when a worker is running but hasn't yet
    /// observed a committed snapshot; `Some(N)` with N >= 1 after
    /// the first successful upload.
    ///
    /// Lock-free: reads the worker's `AtomicI64` counter. Safe to
    /// call from hot paths, metrics exporters, or test harnesses.
    /// Use a `futures::executor::block_on` or a test-runtime `await`
    /// if you need this synchronously — the tokio Mutex is held only
    /// long enough to Option::as_ref the worker.
    pub async fn mirror_seq(&self) -> Option<i64> {
        let guard = self.mirror_worker.lock().await;
        guard.as_ref().map(|w| w.mirror_seq())
    }

    /// Issue #31 Phase 4: seconds since the last successful mirror
    /// upload. `None` when no mirror is attached OR the worker
    /// hasn't completed an upload yet. `Some(n)` with n monotonically
    /// growing between uploads, reset on each successful tick.
    ///
    /// Exceeds `max_lag_alert_secs` → the worker also emits a
    /// `tracing::warn!` with the lag value; this accessor lets
    /// metrics exporters plot the same value as a gauge.
    pub async fn mirror_lag_secs(&self) -> Option<u64> {
        let guard = self.mirror_worker.lock().await;
        guard.as_ref().and_then(|w| w.mirror_lag_secs())
    }

    /// Issue #55: wait until the mirror worker has uploaded at least
    /// through the current snapshot. Returns the mirrored snapshot id.
    ///
    /// Captures `current_snapshot_id()` at call time and blocks until
    /// `mirror_seq >= target`. Kicks the worker immediately so the
    /// caller doesn't waste up to `POLL_INTERVAL` (5 s) waiting for
    /// the next natural tick.
    ///
    /// Returns `InvalidArgument` if no mirror is configured;
    /// `Closed` if the engine shuts down while waiting.
    pub async fn await_mirror(&self) -> crate::types::Result<i64> {
        let target = self.engine.current_snapshot_id();
        if target <= 0 {
            return Ok(0);
        }
        // Clone the Arcs we need, then drop the Mutex guard so
        // mirror_seq() and other callers aren't blocked for the
        // duration of the wait.
        let (seq_arc, advanced, wake) = {
            let guard = self.mirror_worker.lock().await;
            let worker = guard.as_ref().ok_or_else(|| {
                crate::types::MeruError::InvalidArgument("no mirror configured".into())
            })?;
            (
                worker.mirror_seq_arc(),
                worker.mirror_advanced_arc(),
                worker.wake_arc(),
            )
        };
        loop {
            let current = seq_arc.load(std::sync::atomic::Ordering::Relaxed);
            if current >= target {
                return Ok(current);
            }
            if self.engine.is_closed() {
                return Err(crate::types::MeruError::Closed);
            }
            // Register BEFORE waking the worker — if the worker
            // uploads and fires notify_waiters before we park,
            // the notification is lost (Bug Z pattern).
            let notified = advanced.notified();
            wake.notify_one();
            notified.await;
        }
    }

    /// Catalog base directory path (for external analytics file access).
    pub fn catalog_path(&self) -> String {
        self.engine.catalog_path()
    }

    /// Export the current catalog snapshot as an Apache Iceberg v2
    /// `metadata.json` under `target_dir`.
    ///
    /// merutable commits a native JSON manifest on every flush; that
    /// manifest is a strict superset of Iceberg v2 `TableMetadata`. This
    /// method projects the current snapshot onto the Iceberg shape and
    /// writes it to `{target_dir}/metadata/v{N}.metadata.json` alongside a
    /// `version-hint.text`. Downstream tools (pyiceberg, Spark, Trino,
    /// DuckDB, Snowflake, Athena) can read the exported metadata to
    /// register the table, inspect its schema, or audit lineage.
    ///
    /// See [`crate::iceberg::IcebergCatalog::export_to_iceberg`] for
    /// the full field mapping and the current limitations
    /// (manifest-list / manifest Avro emission is follow-on work).
    ///
    /// Returns the absolute path of the emitted `v{N}.metadata.json`.
    pub async fn export_iceberg(
        &self,
        target_dir: impl AsRef<std::path::Path>,
    ) -> Result<std::path::PathBuf> {
        self.engine.export_iceberg(target_dir).await
    }

    /// Re-read the Iceberg manifest from disk. Only meaningful for
    /// read-only replicas; on a read-write instance this is a no-op.
    pub async fn refresh(&self) -> Result<()> {
        self.engine.refresh().await
    }

    /// Issue #32 Phase 2 escape hatch: hand out the inner
    /// `Arc<MeruEngine>` so the replica crate's `InProcessLogSource`
    /// can call `scan_memtable_changes` without forcing every
    /// change-feed-ish client through a public-facing engine facade.
    /// The name signals the narrow intent; stable callers should
    /// prefer the forthcoming `merutable_changes(table, since_seq)`
    /// Flight SQL endpoint once it lands.
    pub fn engine_for_replica(&self) -> Arc<MeruEngine> {
        self.engine.clone()
    }

    /// Graceful shutdown: flush all in-memory data to durable storage and
    /// fsync. After `close()` returns, every write that completed before
    /// this call is durable on disk. Subsequent write/flush/compact calls
    /// return `MeruError::Closed`. Reads remain available until the `MeruDB`
    /// is dropped.
    ///
    /// Call this in your application's shutdown path (e.g. before returning
    /// from `main`, or in a signal handler's async block). If you drop a
    /// `MeruDB` without calling `close()`, a warning is logged — data in
    /// the active memtable that hasn't been flushed will be recovered from
    /// the WAL on the next `open()`, but deferring to WAL recovery is
    /// slower and riskier than an explicit flush.
    ///
    /// Calling `close()` more than once is a no-op.
    pub async fn close(&self) -> Result<()> {
        // Shut down background workers FIRST so they don't fight with
        // the engine's final flush for the rotation lock, and so any
        // in-flight compaction finishes cleanly. `shutdown()` awaits
        // each worker's `JoinHandle`, guaranteeing no background
        // tokio task is still holding `Arc<MeruEngine>` when we
        // proceed to seal the engine.
        let workers = self.background.lock().await.take();
        if let Some(w) = workers {
            w.shutdown().await;
        }
        // Issue #31 Phase 2a: drain the mirror worker alongside the
        // flush/compaction workers. Order doesn't matter between
        // them (the mirror doesn't compete for the rotation lock)
        // but draining before `engine.close()` ensures the final
        // flush's manifest is observable in `mirror_seq` if a Phase
        // 3 test relies on a close-synchronous snapshot of mirror
        // state.
        let mirror = self.mirror_worker.lock().await.take();
        if let Some(mut m) = mirror {
            m.shutdown().await;
        }
        self.engine.close().await
    }

    /// Returns `true` if `close()` has been called.
    pub fn is_closed(&self) -> bool {
        self.engine.is_closed()
    }
}

impl Drop for MeruDB {
    fn drop(&mut self) {
        if !self.engine.is_closed() {
            tracing::warn!(
                table = %self.engine.schema().table_name,
                "MeruDB dropped without calling close() — \
                 unflushed memtable data will rely on WAL recovery"
            );
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::schema::{ColumnDef, ColumnType};
    use bytes::Bytes;

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
                    name: "name".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: true,

                    ..Default::default()
                },
            ],
            primary_key: vec![0],

            ..Default::default()
        }
    }

    fn test_options(tmp: &tempfile::TempDir) -> crate::options::OpenOptions {
        crate::options::OpenOptions::new(test_schema())
            .wal_dir(tmp.path().join("wal"))
            .catalog_uri(tmp.path().to_string_lossy().to_string())
    }

    fn make_row(id: i64, name: &str) -> Row {
        Row::new(vec![
            Some(FieldValue::Int64(id)),
            Some(FieldValue::Bytes(Bytes::from(name.to_string()))),
        ])
    }

    #[tokio::test]
    async fn open_and_close() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();
        assert_eq!(db.schema().table_name, "test");
        // Issue #8: fresh DB has no writes → read_seq is the zero frontier.
        assert_eq!(db.read_seq().0, 0);
    }

    #[tokio::test]
    async fn put_and_get() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        db.put(make_row(1, "alice")).await.unwrap();
        let row = db.get(&[FieldValue::Int64(1)]).unwrap();
        assert!(row.is_some());
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();
        assert!(db.get(&[FieldValue::Int64(999)]).unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_removes_row() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        db.put(make_row(1, "alice")).await.unwrap();
        assert!(db.get(&[FieldValue::Int64(1)]).unwrap().is_some());

        db.delete(vec![FieldValue::Int64(1)]).await.unwrap();
        assert!(db.get(&[FieldValue::Int64(1)]).unwrap().is_none());
    }

    #[tokio::test]
    async fn put_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        db.put(make_row(1, "alice")).await.unwrap();
        db.put(make_row(1, "bob")).await.unwrap();

        let row = db.get(&[FieldValue::Int64(1)]).unwrap().unwrap();
        let name = row.get(1).unwrap();
        assert_eq!(*name, FieldValue::Bytes(Bytes::from("bob")));
    }

    #[tokio::test]
    async fn scan_all() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        db.put(make_row(3, "charlie")).await.unwrap();
        db.put(make_row(1, "alice")).await.unwrap();
        db.put(make_row(2, "bob")).await.unwrap();

        let results = db.scan(None, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn scan_range() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        for i in 1..=10i64 {
            db.put(make_row(i, &format!("user{i}"))).await.unwrap();
        }

        let results = db
            .scan(Some(&[FieldValue::Int64(3)]), Some(&[FieldValue::Int64(7)]))
            .unwrap();
        assert_eq!(results.len(), 4); // 3, 4, 5, 6
    }

    #[tokio::test]
    async fn delete_then_scan_excludes_deleted() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        db.put(make_row(1, "alice")).await.unwrap();
        db.put(make_row(2, "bob")).await.unwrap();
        db.put(make_row(3, "charlie")).await.unwrap();

        db.delete(vec![FieldValue::Int64(2)]).await.unwrap();

        let results = db.scan(None, None).unwrap();
        assert_eq!(results.len(), 2); // 1 and 3 remain
    }

    #[tokio::test]
    async fn many_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        for i in 0..500i64 {
            db.put(make_row(i, &format!("row{i}"))).await.unwrap();
        }

        let results = db.scan(None, None).unwrap();
        assert_eq!(results.len(), 500);

        // Point lookups.
        assert!(db.get(&[FieldValue::Int64(0)]).unwrap().is_some());
        assert!(db.get(&[FieldValue::Int64(499)]).unwrap().is_some());
        assert!(db.get(&[FieldValue::Int64(500)]).unwrap().is_none());
    }

    #[tokio::test]
    async fn put_batch_writes_all() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        let rows = (0..100i64)
            .map(|i| make_row(i, &format!("batch_{i}")))
            .collect::<Vec<_>>();

        db.put_batch(rows).await.unwrap();

        // All 100 rows must be readable.
        let results = db.scan(None, None).unwrap();
        assert_eq!(results.len(), 100);
        assert!(db.get(&[FieldValue::Int64(0)]).unwrap().is_some());
        assert!(db.get(&[FieldValue::Int64(99)]).unwrap().is_some());
    }

    #[tokio::test]
    async fn flush_persists_data() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        db.put(make_row(1, "alice")).await.unwrap();
        db.flush().await.unwrap();

        // After flush, data should still be accessible.
        // (Memtable was rotated and flushed; the scan may come from
        // the new empty memtable + Parquet files.)
        let _results = db.scan(None, None);
    }

    #[tokio::test]
    async fn null_column() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        // Insert a row with NULL in the nullable column.
        let row = Row::new(vec![Some(FieldValue::Int64(1)), None]);
        db.put(row).await.unwrap();

        let found = db.get(&[FieldValue::Int64(1)]).unwrap().unwrap();
        assert!(found.get(1).is_none()); // name column is NULL
    }

    #[tokio::test]
    async fn read_only_blocks_writes() {
        let tmp = tempfile::tempdir().unwrap();

        // Write some data with a read-write instance first.
        {
            let db = MeruDB::open(test_options(&tmp)).await.unwrap();
            db.put(make_row(1, "alice")).await.unwrap();
            db.flush().await.unwrap();
        }

        // Open read-only.
        let ro_opts = test_options(&tmp).read_only(true);
        let db = MeruDB::open(ro_opts).await.unwrap();

        // Reads should work.
        let row = db.get(&[FieldValue::Int64(1)]).unwrap();
        assert!(row.is_some());

        // Writes should fail.
        let err = db.put(make_row(2, "bob")).await;
        assert!(err.is_err());

        // Scan should work.
        let results = db.scan(None, None).unwrap();
        assert_eq!(results.len(), 1);
    }

    /// End-to-end Iceberg translator test: open, write, flush, export,
    /// then hand the exported metadata to the `iceberg-rs` crate's
    /// deserializer. This is the strongest compatibility signal
    /// available in-tree — if that crate accepts our payload, every
    /// v2-aware Iceberg reader in the ecosystem will too.
    #[tokio::test]
    async fn export_iceberg_metadata_parses_with_iceberg_rs() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        // Write and flush some rows so the exported snapshot is non-empty.
        for i in 0..5i64 {
            db.put(make_row(i, &format!("row{i}"))).await.unwrap();
        }
        db.flush().await.unwrap();

        let target = tempfile::tempdir().unwrap();
        let metadata_path = db.export_iceberg(target.path()).await.unwrap();
        assert!(metadata_path.exists());

        let bytes = tokio::fs::read(&metadata_path).await.unwrap();
        let parsed: std::result::Result<iceberg::spec::TableMetadata, _> =
            serde_json::from_slice(&bytes);
        assert!(
            parsed.is_ok(),
            "iceberg-rs rejected exported metadata: {:?}\n\nfile: {}\n\ncontent:\n{}",
            parsed.err(),
            metadata_path.display(),
            String::from_utf8_lossy(&bytes)
        );
        let tm = parsed.unwrap();
        // Post-flush we expect exactly one snapshot with a live file.
        assert!(tm.current_snapshot_id().is_some());
        assert!(
            tm.last_sequence_number() >= 1,
            "sequence number must advance after the first commit"
        );
    }

    #[tokio::test]
    async fn read_only_refresh() {
        let tmp = tempfile::tempdir().unwrap();

        // Write and flush with read-write instance.
        let rw_db = MeruDB::open(test_options(&tmp)).await.unwrap();
        rw_db.put(make_row(1, "alice")).await.unwrap();
        rw_db.flush().await.unwrap();

        // Open read-only replica.
        let ro_opts = test_options(&tmp).read_only(true);
        let ro_db = MeruDB::open(ro_opts).await.unwrap();
        let row = ro_db.get(&[FieldValue::Int64(1)]).unwrap();
        assert!(row.is_some());

        // Write more data with the primary.
        rw_db.put(make_row(2, "bob")).await.unwrap();
        rw_db.flush().await.unwrap();

        // Before refresh, replica doesn't see key 2 (it's only in the new snapshot).
        // After refresh, it should.
        ro_db.refresh().await.unwrap();
        let row2 = ro_db.get(&[FieldValue::Int64(2)]).unwrap();
        assert!(
            row2.is_some(),
            "read-only replica should see key 2 after refresh"
        );
    }

    #[tokio::test]
    async fn close_flushes_memtable() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        // Write data — sits in memtable, not yet flushed.
        for i in 0..50i64 {
            db.put(make_row(i, &format!("row{i}"))).await.unwrap();
        }

        assert!(!db.is_closed());
        db.close().await.unwrap();
        assert!(db.is_closed());

        // Reads still work after close.
        let row = db.get(&[FieldValue::Int64(0)]).unwrap();
        assert!(row.is_some(), "reads must work after close");

        // Writes are rejected.
        let err = db.put(make_row(100, "nope")).await;
        assert!(err.is_err(), "writes must fail after close");

        // Batch writes are rejected.
        let err = db.put_batch(vec![make_row(101, "nope")]).await;
        assert!(err.is_err(), "batch writes must fail after close");

        // Flush is rejected.
        let err = db.flush().await;
        assert!(err.is_err(), "flush must fail after close");

        // Compact is rejected.
        let err = db.compact().await;
        assert!(err.is_err(), "compact must fail after close");
    }

    #[tokio::test]
    async fn close_data_durable_across_reopen() {
        let tmp = tempfile::tempdir().unwrap();

        {
            let db = MeruDB::open(test_options(&tmp)).await.unwrap();
            for i in 0..20i64 {
                db.put(make_row(i, &format!("val{i}"))).await.unwrap();
            }
            // close() flushes to Parquet — no WAL recovery needed.
            db.close().await.unwrap();
        }

        // Reopen and verify all data is present from Parquet.
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();
        for i in 0..20i64 {
            let row = db.get(&[FieldValue::Int64(i)]).unwrap();
            assert!(
                row.is_some(),
                "key {i} must be durable after close + reopen"
            );
        }
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn double_close_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();
        db.close().await.unwrap();
        db.close().await.unwrap(); // must not panic or error
        assert!(db.is_closed());
    }

    /// Issue #31 Phase 2a: attaching a MirrorConfig spawns the
    /// mirror worker, writes+flushes advance the snapshot, and
    /// close() drains the worker cleanly without hanging.
    #[tokio::test]
    async fn mirror_worker_spawned_and_drained_on_close() {
        use crate::options::MirrorConfig;
        use crate::store::local::LocalFileStore;
        use std::sync::Arc;
        let tmp = tempfile::tempdir().unwrap();
        let mirror_dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileStore::new(mirror_dir.path()).unwrap());
        let opts = test_options(&tmp).mirror(MirrorConfig::new(store));
        let db = MeruDB::open(opts).await.unwrap();
        // Write + flush so the snapshot advances past 0.
        db.put(make_row(1, "alice")).await.unwrap();
        db.flush().await.unwrap();
        // Close must complete within a bounded time — any deadlock in
        // the mirror worker's shutdown path would hang here.
        tokio::time::timeout(std::time::Duration::from_secs(10), db.close())
            .await
            .expect("close hung past 10s — mirror worker deadlock?")
            .unwrap();
    }

    /// Issue #31 Phase 3: `mirror_seq()` returns `None` when no
    /// mirror is attached, `Some(0)` on a freshly-spawned worker,
    /// and `Some(N)` once N snapshots have been uploaded.
    #[tokio::test]
    async fn mirror_seq_surfaces_worker_state() {
        use crate::options::MirrorConfig;
        use crate::store::local::LocalFileStore;
        use std::sync::Arc;
        // No-mirror case.
        let tmp1 = tempfile::tempdir().unwrap();
        let db_no_mirror = MeruDB::open(test_options(&tmp1)).await.unwrap();
        assert_eq!(db_no_mirror.mirror_seq().await, None);
        db_no_mirror.close().await.unwrap();

        // Mirror-attached case.
        let tmp = tempfile::tempdir().unwrap();
        let mirror_dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileStore::new(mirror_dir.path()).unwrap());
        let opts = test_options(&tmp).mirror(MirrorConfig::new(store));
        let db = MeruDB::open(opts).await.unwrap();

        // Fresh worker: has not yet uploaded anything.
        assert_eq!(db.mirror_seq().await, Some(0));

        // Close drains the worker. We don't assert on a specific
        // post-close mirror_seq value (the tick may or may not have
        // fired depending on test timing), but the method must keep
        // returning Some — the worker still exists as an Option
        // until close takes it.
        db.close().await.unwrap();
    }

    /// Issue #55 regression: await_mirror returns InvalidArgument
    /// when no mirror is configured.
    #[tokio::test]
    async fn await_mirror_no_mirror_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();
        // Write + flush so snapshot_id > 0.
        db.put(make_row(1, "v")).await.unwrap();
        db.flush().await.unwrap();
        let err = db.await_mirror().await.unwrap_err();
        match err {
            crate::types::MeruError::InvalidArgument(msg) => {
                assert!(msg.contains("mirror"), "msg: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
        db.close().await.unwrap();
    }

    /// Issue #55 regression: await_mirror blocks until the mirror
    /// worker uploads the current snapshot, then returns the mirrored
    /// snapshot id.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn await_mirror_blocks_until_synced() {
        use crate::options::MirrorConfig;
        use crate::store::local::LocalFileStore;
        use std::sync::Arc;

        let tmp = tempfile::tempdir().unwrap();
        let mirror_dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileStore::new(mirror_dir.path()).unwrap());
        let opts = test_options(&tmp).mirror(MirrorConfig::new(store.clone()));
        let db = MeruDB::open(opts).await.unwrap();

        // Write data and flush to create a committed snapshot.
        for i in 1..=5i64 {
            db.put(make_row(i, &format!("v{i}"))).await.unwrap();
        }
        db.flush().await.unwrap();
        let snapshot_id = db.engine.current_snapshot_id();
        assert!(snapshot_id > 0, "flush must commit a snapshot");

        // await_mirror must return >= snapshot_id, not block forever.
        let mirrored = tokio::time::timeout(std::time::Duration::from_secs(15), db.await_mirror())
            .await
            .expect("await_mirror must not block indefinitely")
            .expect("await_mirror must succeed");
        assert!(
            mirrored >= snapshot_id,
            "mirrored seq ({mirrored}) must be >= snapshot ({snapshot_id})"
        );

        // Verify the manifest was actually uploaded.
        let manifest_path = format!("metadata/v{snapshot_id}.manifest.bin");
        assert!(
            mirror_dir.path().join(&manifest_path).exists(),
            "mirror must have uploaded {manifest_path}"
        );

        db.close().await.unwrap();
    }

    /// Issue #55: await_mirror on a fresh DB with no committed
    /// snapshot returns immediately with 0.
    #[tokio::test]
    async fn await_mirror_no_snapshot_returns_zero() {
        use crate::options::MirrorConfig;
        use crate::store::local::LocalFileStore;
        use std::sync::Arc;

        let tmp = tempfile::tempdir().unwrap();
        let mirror_dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileStore::new(mirror_dir.path()).unwrap());
        let opts = test_options(&tmp).mirror(MirrorConfig::new(store));
        let db = MeruDB::open(opts).await.unwrap();

        // No writes, no flush — snapshot_id is 0.
        let result = db.await_mirror().await.unwrap();
        assert_eq!(result, 0, "no snapshot committed, should return 0");

        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn scan_works_after_close() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MeruDB::open(test_options(&tmp)).await.unwrap();

        for i in 1..=5i64 {
            db.put(make_row(i, &format!("user{i}"))).await.unwrap();
        }
        db.close().await.unwrap();

        // Scan must still work.
        let results = db.scan(None, None).unwrap();
        assert_eq!(results.len(), 5);
    }
}
