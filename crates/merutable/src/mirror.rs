//! Issue #31 Phase 2b: mirror worker with commit-order-preserving uploads.
//!
//! Spawns a long-lived tokio task that polls the primary's version
//! set; on observing a new snapshot, it:
//!
//! 1. Enumerates the live data files (and DV puffins) referenced by
//!    the current manifest.
//! 2. `put_if_absent` each file to the mirror target. Shared files
//!    across snapshots are no-ops after the first upload, so catch-up
//!    is amortized over the steady-state tick cadence.
//! 3. Serializes the current manifest as protobuf (#28) and writes it
//!    at `metadata/v{N}.manifest.bin` via `put_if_absent`. The
//!    conditional PUT on the manifest is the single race-safety boundary.
//! 4. Writes/advances `metadata/low_water.txt = N` so readers
//!    mounting the mirror with `discover_head_from(low_water, ..)`
//!    find the uploaded manifest as HEAD.
//!
//! The order matters: data files BEFORE the manifest, always. A reader
//! opening the mirror must never observe a manifest pointing at files
//! that don't exist yet.
//!
//! # Scope (Phase 2b)
//!
//! - **Only the most recent observed snapshot is uploaded.** Operators
//!   running a hot primary against a cold mirror will see gaps in the
//!   mirror's backward-pointer chain — `v{N}.parent_snapshot_id` may
//!   reference a version that isn't present on the mirror. This is
//!   safe for HEAD-only reads (the dominant remote-reader case)
//!   because `discover_head_from(low_water)` probes version numbers,
//!   not the parent chain. Time-travel on the mirror below `low_water`
//!   is not available until Phase 2c fills in the historical chain.
//! - **Single-writer.** Two primaries mirroring to the same destination
//!   would race on `put_if_absent(manifest)`; one wins, the other
//!   logs a warning and skips that snapshot. Don't do this on purpose.
//!
//! # Shutdown
//!
//! Mirrors the `BackgroundWorkers` pattern: `AtomicBool` flag set
//! FIRST, then `Notify::notify_waiters()`. The worker checks the
//! flag at the top of every loop iteration, so a shutdown signal
//! arriving between `notified().await` registrations is not lost.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::engine::engine::MeruEngine;
use crate::iceberg::Manifest;
use crate::store::traits::MeruStore;
use crate::types::MeruError;
use bytes::Bytes;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::options::MirrorConfig;

/// The mirror destination's low-water marker path — matches what
/// `ObjectStoreCatalog::reclaim_old_manifests` and its HEAD discovery
/// both use. Keeping the exact same path means a remote reader
/// opening the mirror with the object-store layout probes from the
/// right position without any special coordination.
const LOW_WATER_PATH: &str = "metadata/low_water.txt";

fn manifest_path(v: i64) -> String {
    // Mirror the naming used by ObjectStoreCatalog so remote readers
    // opening the mirror via the object-store layout find HEAD in
    // the expected location.
    format!("metadata/v{v}.manifest.bin")
}

/// The mirror worker's cadence. Not exposed as a knob yet — 5
/// seconds is short enough to keep mirror_lag bounded to single-
/// digit seconds under sustained writes, long enough to avoid
/// burning CPU on a quiescent primary.
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Handle to the spawned mirror worker. Held by `MeruDB` behind a
/// `tokio::sync::Mutex<Option<MirrorWorker>>` so `close()` can
/// `take()` and `shutdown().await` before the engine's final
/// flush.
pub struct MirrorWorker {
    shutdown_flag: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    handle: Option<JoinHandle<()>>,
    /// Highest snapshot_id the worker has OBSERVED (Phase 2a) or
    /// UPLOADED (Phase 2b+). Exposed via `mirror_seq()` so
    /// integration tests + future Phase 3 `stats()` plumbing can
    /// read it without reaching into the worker's internals.
    mirror_seq: Arc<AtomicI64>,
    /// Issue #31 Phase 4: wall-clock seconds-since-UNIX-epoch at
    /// the last successful upload. Zero means "never uploaded"; any
    /// positive value means "last upload finished at t=value".
    /// `mirror_lag_secs()` subtracts from `now` to produce the lag.
    /// Stored as i64 in seconds (not Instant) so reading it from a
    /// non-async context — a metrics exporter thread — doesn't need
    /// access to the tokio runtime.
    last_upload_unix_secs: Arc<AtomicI64>,
    /// Issue #55: fired after every successful upload (mirror_seq
    /// advance). `await_mirror()` registers on this BEFORE kicking
    /// the worker, so the notification is never lost.
    mirror_advanced: Arc<Notify>,
    /// Issue #55: kick the worker to tick immediately instead of
    /// sleeping up to POLL_INTERVAL. `await_mirror()` fires this
    /// so the caller doesn't waste up to 5 seconds waiting for the
    /// next natural poll.
    wake: Arc<Notify>,
}

impl MirrorWorker {
    /// Spawn a mirror worker. Called by `MeruDB::open` when a
    /// `MirrorConfig` is attached. The worker lives until
    /// `shutdown()` is awaited.
    pub fn spawn(engine: Arc<MeruEngine>, config: MirrorConfig) -> Self {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_notify = Arc::new(Notify::new());
        let mirror_seq = Arc::new(AtomicI64::new(0));
        let last_upload = Arc::new(AtomicI64::new(0));
        let mirror_advanced = Arc::new(Notify::new());
        let wake = Arc::new(Notify::new());
        let state = MirrorLoopState {
            shutdown_flag: shutdown_flag.clone(),
            shutdown_notify: shutdown_notify.clone(),
            mirror_seq: mirror_seq.clone(),
            last_upload_unix_secs: last_upload.clone(),
            mirror_advanced: mirror_advanced.clone(),
            wake: wake.clone(),
        };
        let handle = tokio::spawn(async move {
            mirror_loop(engine, config, state).await;
        });
        Self {
            shutdown_flag,
            shutdown_notify,
            handle: Some(handle),
            mirror_seq,
            last_upload_unix_secs: last_upload,
            mirror_advanced,
            wake,
        }
    }

    /// Latest snapshot_id the worker has observed (Phase 2a) or
    /// mirrored (Phase 2b+). Synchronously readable from anywhere.
    /// Zero on a freshly-spawned worker that hasn't yet completed a
    /// tick.
    pub fn mirror_seq(&self) -> i64 {
        self.mirror_seq.load(Ordering::Relaxed)
    }

    /// Issue #31 Phase 4: seconds since the last successful upload.
    /// `None` when the worker has never successfully uploaded;
    /// `Some(n)` with n >= 0 otherwise. Computed from the current
    /// wall clock, so repeated calls without a new upload return
    /// monotonically-increasing values.
    pub fn mirror_lag_secs(&self) -> Option<u64> {
        let last = self.last_upload_unix_secs.load(Ordering::Relaxed);
        if last == 0 {
            return None;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(last);
        Some((now - last).max(0) as u64)
    }

    /// Issue #55: cloned Arc accessors so `MeruDB::await_mirror()`
    /// can drop the Mutex guard before entering the await loop.
    pub(crate) fn mirror_seq_arc(&self) -> Arc<AtomicI64> {
        self.mirror_seq.clone()
    }
    pub(crate) fn mirror_advanced_arc(&self) -> Arc<Notify> {
        self.mirror_advanced.clone()
    }
    pub(crate) fn wake_arc(&self) -> Arc<Notify> {
        self.wake.clone()
    }

    /// Signal the worker to shut down and await its exit.
    ///
    /// Ordering matches `BackgroundWorkers::shutdown`:
    /// 1. Set the flag (the loop checks it at the top).
    /// 2. Notify (wake any task parked in `notified().await`).
    /// 3. Await the `JoinHandle` (drain the final tick).
    pub async fn shutdown(&mut self) {
        self.shutdown_flag.store(true, Ordering::SeqCst);
        self.shutdown_notify.notify_waiters();
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

/// Shared state passed to the mirror loop. Bundled to avoid
/// exceeding clippy's `too_many_arguments` threshold.
struct MirrorLoopState {
    shutdown_flag: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    mirror_seq: Arc<AtomicI64>,
    last_upload_unix_secs: Arc<AtomicI64>,
    mirror_advanced: Arc<Notify>,
    wake: Arc<Notify>,
}

async fn mirror_loop(engine: Arc<MeruEngine>, config: MirrorConfig, state: MirrorLoopState) {
    let MirrorLoopState {
        shutdown_flag,
        shutdown_notify,
        mirror_seq,
        last_upload_unix_secs,
        mirror_advanced,
        wake,
    } = state;
    info!("mirror worker started (Issue #31 Phase 2b — observe + upload)");
    let catalog_path = PathBuf::from(engine.catalog_path());
    let mut last_uploaded: i64 = 0;
    // Phase 4: alert when lag exceeds `max_lag_alert_secs` AND at
    // least one upload has happened. Without the "at least one"
    // guard a never-written catalog would fire false alerts forever.
    let alert_threshold = config.max_lag_alert_secs;
    loop {
        if shutdown_flag.load(Ordering::SeqCst) {
            break;
        }
        let current = engine.current_snapshot_id();
        if current > last_uploaded && current > 0 {
            match mirror_snapshot(&engine, &catalog_path, &config, current).await {
                Ok(()) => {
                    info!(
                        snapshot_id = current,
                        previous_mirror_seq = last_uploaded,
                        "mirror worker uploaded snapshot"
                    );
                    last_uploaded = current;
                    mirror_seq.store(current, Ordering::Relaxed);
                    // Issue #55: wake any await_mirror() callers
                    // blocked on this advance.
                    mirror_advanced.notify_waiters();
                    let now_secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    last_upload_unix_secs.store(now_secs, Ordering::Relaxed);
                }
                Err(e) => {
                    // Don't update last_uploaded so the next tick
                    // retries. Orphans from a partial upload are
                    // reconciled by subsequent successful attempts
                    // (put_if_absent on already-uploaded files is a
                    // clean no-op).
                    warn!(
                        snapshot_id = current,
                        error = %e,
                        "mirror worker failed to upload snapshot — will retry next tick"
                    );
                }
            }
        } else {
            debug!(
                snapshot_id = current,
                "mirror worker tick — no new snapshot"
            );
        }

        // Phase 4: lag-alert check runs every tick, independent of
        // whether an upload just happened. Surfaces the common bad
        // case — primary committing, mirror lagging because the
        // destination is slow — without spamming on each tick by
        // only warning once per alert_threshold window.
        let last = last_upload_unix_secs.load(Ordering::Relaxed);
        if last > 0 && alert_threshold > 0 {
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(last);
            let lag = (now_secs - last).max(0) as u64;
            if lag >= alert_threshold {
                warn!(
                    mirror_lag_secs = lag,
                    max_lag_alert_secs = alert_threshold,
                    mirror_seq = last_uploaded,
                    primary_snapshot_id = current,
                    "mirror worker: upload lag exceeded alert threshold — no backpressure, \
                     destination may be slow or unreachable"
                );
            }
        }

        // Wait for either the poll interval, an explicit shutdown, or
        // a wake kick from await_mirror(). The wake branch lets
        // callers avoid the up-to-POLL_INTERVAL latency.
        tokio::select! {
            _ = tokio::time::sleep(POLL_INTERVAL) => {}
            _ = shutdown_notify.notified() => {}
            _ = wake.notified() => {}
        }
    }
    info!(last_uploaded_seq = last_uploaded, "mirror worker shut down");
}

/// Upload everything the mirror needs to serve snapshot `version`:
///
/// 1. Every live data file (and attached DV puffin) referenced by
///    the manifest — via `put_if_absent`, so repeated attempts and
///    shared-file catch-up are idempotent.
/// 2. The manifest itself at `metadata/v{version}.manifest.bin`.
/// 3. `metadata/low_water.txt` advanced to `version` (always
///    overwritten — low-water on the mirror tracks the latest
///    uploaded snapshot, not the earliest).
///
/// Order: files BEFORE manifest. A reader who observes the manifest
/// must find every file it references already present.
async fn mirror_snapshot(
    engine: &MeruEngine,
    catalog_path: &std::path::Path,
    config: &MirrorConfig,
    version: i64,
) -> Result<(), MeruError> {
    let manifest: Manifest = engine.current_manifest().await;

    // Step 1: upload data files + DV puffins. Parallelism bounded
    // by `mirror_parallelism`; each worker does its own put_if_absent.
    let semaphore = Arc::new(tokio::sync::Semaphore::new(
        config.mirror_parallelism.max(1),
    ));
    let mut join = tokio::task::JoinSet::new();
    for entry in &manifest.entries {
        if entry.status == "deleted" {
            continue;
        }
        spawn_upload(
            &mut join,
            semaphore.clone(),
            config.target.clone(),
            catalog_path.to_path_buf(),
            entry.path.clone(),
        );
        if let Some(dv_path) = entry.dv_path.clone() {
            spawn_upload(
                &mut join,
                semaphore.clone(),
                config.target.clone(),
                catalog_path.to_path_buf(),
                dv_path,
            );
        }
    }
    while let Some(res) = join.join_next().await {
        match res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(join_err) => {
                return Err(MeruError::ObjectStore(format!(
                    "mirror upload task panicked: {join_err}"
                )));
            }
        }
    }

    // Step 2: serialize + upload manifest. `put_if_absent` because
    // two primary processes mirroring to the same target would race
    // here; conditional PUT is the single serialization boundary.
    // `AlreadyExists` means the version was already mirrored —
    // idempotent no-op.
    let pb_bytes = manifest.to_protobuf()?;
    match config
        .target
        .put_if_absent(&manifest_path(version), Bytes::from(pb_bytes))
        .await
    {
        Ok(()) | Err(MeruError::AlreadyExists(_)) => {}
        Err(e) => return Err(e),
    }

    // Step 3: advance the low-water pointer. Always overwritten
    // (via `put`, not `put_if_absent`) so re-runs of
    // `mirror_snapshot` at a higher version correctly bump the
    // pointer forward.
    config
        .target
        .put(LOW_WATER_PATH, Bytes::from(version.to_string()))
        .await?;

    Ok(())
}

fn spawn_upload(
    join: &mut tokio::task::JoinSet<Result<(), MeruError>>,
    semaphore: Arc<tokio::sync::Semaphore>,
    target: Arc<dyn MeruStore>,
    catalog_path: PathBuf,
    rel_path: String,
) {
    join.spawn(async move {
        let _permit = semaphore
            .acquire_owned()
            .await
            .expect("semaphore never closed");
        let abs = catalog_path.join(&rel_path);
        let bytes = tokio::fs::read(&abs).await.map_err(MeruError::Io)?;
        match target.put_if_absent(&rel_path, Bytes::from(bytes)).await {
            Ok(()) | Err(MeruError::AlreadyExists(_)) => Ok(()),
            Err(e) => Err(e),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::EngineConfig;
    use crate::store::local::LocalFileStore;
    use crate::types::schema::{ColumnDef, ColumnType, TableSchema};

    fn schema() -> TableSchema {
        TableSchema {
            table_name: "mirror-worker-test".into(),
            columns: vec![ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,
                ..Default::default()
            }],
            primary_key: vec![0],
            ..Default::default()
        }
    }

    fn engine_config(tmp: &tempfile::TempDir) -> EngineConfig {
        EngineConfig {
            schema: schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn spawn_and_shutdown_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let mirror_dir = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(engine_config(&tmp)).await.unwrap();
        let store = Arc::new(LocalFileStore::new(mirror_dir.path()).unwrap());
        let cfg = MirrorConfig::new(store);

        let mut worker = MirrorWorker::spawn(engine, cfg);
        // Fresh engine: snapshot_id is 0. Worker's mirror_seq is
        // either 0 (hasn't ticked yet) or 0 (ticked and saw 0).
        assert_eq!(worker.mirror_seq(), 0);
        // Shutdown must return within a bounded wait; no deadlock.
        tokio::time::timeout(Duration::from_secs(5), worker.shutdown())
            .await
            .expect("mirror worker shutdown hung past 5s");
    }

    /// A second shutdown call after the first is a no-op (not a
    /// panic). Mirrors the `close()` contract on `MeruDB`.
    #[tokio::test]
    async fn double_shutdown_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let mirror_dir = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(engine_config(&tmp)).await.unwrap();
        let store = Arc::new(LocalFileStore::new(mirror_dir.path()).unwrap());
        let mut worker = MirrorWorker::spawn(engine, MirrorConfig::new(store));
        worker.shutdown().await;
        worker.shutdown().await; // must not panic
    }

    /// Phase 4: `mirror_lag_secs()` is None until the first upload,
    /// then Some(0..N) afterwards.
    #[tokio::test]
    async fn mirror_lag_transitions_from_none_to_some() {
        // Exercise the accessor math in isolation — no live
        // background tick, no catalog. The worker's real
        // upload+write path is covered by
        // `mirror_snapshot_uploads_files_manifest_and_low_water`
        // below. Driving both an engine-coupled worker AND a direct
        // upload in the same test flakes on CI: the worker's first
        // tick fires ~immediately, opens files in the source
        // tempdir, and races tempdir drop as the test exits.
        let tmp = tempfile::tempdir().unwrap();
        let mirror_dir = tempfile::tempdir().unwrap();
        let engine = MeruEngine::open(engine_config(&tmp)).await.unwrap();
        let store = Arc::new(LocalFileStore::new(mirror_dir.path()).unwrap());
        let mut worker = MirrorWorker::spawn(engine, MirrorConfig::new(store));
        // Shutdown IMMEDIATELY so the worker's loop sees the flag
        // and exits before calling current_snapshot_id on a dropped
        // engine. We're only testing the accessor math below.
        worker.shutdown().await;

        // No upload recorded yet — accessor returns None.
        assert_eq!(worker.mirror_lag_secs(), None);

        // Simulate a just-completed upload by writing the current
        // UNIX second directly into the atomic. Phase 4's loop does
        // exactly this on every successful tick.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        worker.last_upload_unix_secs.store(now, Ordering::Relaxed);

        let lag = worker.mirror_lag_secs().expect("lag is Some after upload");
        assert!(lag < 10, "lag should be near-zero on fresh upload: {lag}");
    }

    /// Phase 2b: `mirror_snapshot` uploads data files AND the
    /// protobuf manifest AND advances low_water.txt. Contract
    /// pinned at the function level so the integration test below
    /// doesn't need to race the worker's polling tick.
    #[tokio::test]
    async fn mirror_snapshot_uploads_files_manifest_and_low_water() {
        use crate::iceberg::{
            snapshot::{IcebergDataFile, SnapshotTransaction},
            IcebergCatalog,
        };
        use crate::types::level::{Level, ParquetFileMeta};
        let tmp = tempfile::tempdir().unwrap();
        let mirror_dir = tempfile::tempdir().unwrap();

        // Build a POSIX catalog at tmp with two data files.
        let schema = std::sync::Arc::new(schema());
        let catalog = IcebergCatalog::open(tmp.path(), schema.as_ref().clone())
            .await
            .unwrap();
        tokio::fs::create_dir_all(tmp.path().join("data/L0"))
            .await
            .unwrap();
        let mut txn = SnapshotTransaction::new();
        for i in 0..2 {
            let path = format!("data/L0/f{i}.parquet");
            tokio::fs::write(tmp.path().join(&path), format!("pq-body-{i}"))
                .await
                .unwrap();
            txn.add_file(IcebergDataFile {
                path,
                file_size: 9,
                num_rows: 100,
                meta: ParquetFileMeta {
                    level: Level(0),
                    seq_min: 1,
                    seq_max: 10,
                    key_min: vec![0x01],
                    key_max: vec![0xFF],
                    num_rows: 100,
                    file_size: 9,
                    dv_path: None,
                    dv_offset: None,
                    dv_length: None,
                    format: None,
                    column_stats: None,
                },
            });
        }
        catalog.commit(&txn, schema.clone()).await.unwrap();

        // Open the engine against the same catalog path. Engine
        // sees the committed v=1 manifest.
        let engine = MeruEngine::open(engine_config(&tmp)).await.unwrap();
        assert_eq!(engine.current_snapshot_id(), 1);

        // Set up the mirror target.
        let store = Arc::new(LocalFileStore::new(mirror_dir.path()).unwrap());
        let cfg = MirrorConfig::new(store.clone());
        let catalog_path = PathBuf::from(engine.catalog_path());

        // Directly invoke the upload path (bypassing the worker
        // tick) so the test is deterministic.
        super::mirror_snapshot(&engine, &catalog_path, &cfg, 1)
            .await
            .unwrap();

        // Data files are present at the mirror with matching bytes.
        for i in 0..2 {
            let path = format!("data/L0/f{i}.parquet");
            let got = store.get(&path).await.unwrap();
            assert_eq!(got.as_ref(), format!("pq-body-{i}").as_bytes());
        }
        // Manifest is present at the canonical ObjectStore path.
        assert!(store.exists("metadata/v1.manifest.bin").await.unwrap());
        // Low-water points at v=1.
        let lw = store.get("metadata/low_water.txt").await.unwrap();
        assert_eq!(lw.as_ref(), b"1");

        // Re-running the upload against the same destination is
        // idempotent — no errors, data bytes unchanged.
        super::mirror_snapshot(&engine, &catalog_path, &cfg, 1)
            .await
            .unwrap();
    }
}
