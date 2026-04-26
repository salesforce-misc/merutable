//! Regression for Issue #21: dropping `BackgroundWorkers` without first
//! awaiting `shutdown()` must terminate the spawned worker tasks.
//!
//! Before the fix, the derived `Drop` just released the `JoinHandle`s,
//! which **detaches** the underlying tokio tasks. Those tasks keep
//! their own `Arc<MeruEngine>` clone, so the engine stays live and the
//! workers spin forever trying to flush / compact against files that
//! the dropped DB has already removed.
//!
//! The fix gives `BackgroundWorkers` a custom `Drop` that sets the
//! shutdown flag, notifies parked waiters, and calls `JoinHandle::abort()`
//! on every handle. Aborted tasks drop their captured state — including
//! the `Arc<MeruEngine>` — at the next yield point, so a `Weak` held by
//! the test sees `strong_count() == 1` (the test's own Arc) within a
//! bounded window.

use std::{
    sync::{Arc, Weak},
    time::Duration,
};

use merutable::engine::{EngineConfig, MeruEngine, background::BackgroundWorkers};
use merutable::types::schema::{ColumnDef, ColumnType, TableSchema};
use tempfile::TempDir;

fn schema() -> TableSchema {
    TableSchema {
        table_name: "bgdrop".into(),
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

fn config(tmp: &TempDir) -> EngineConfig {
    EngineConfig {
        schema: schema(),
        catalog_uri: tmp.path().to_string_lossy().to_string(),
        object_store_prefix: tmp.path().to_string_lossy().to_string(),
        wal_dir: tmp.path().join("wal"),
        flush_parallelism: 1,
        compaction_parallelism: 2,
        ..Default::default()
    }
}

/// Drop `BackgroundWorkers` without awaiting `shutdown()`. The tasks
/// must release their `Arc<MeruEngine>` within a bounded window.
///
/// Uses `flavor = "multi_thread"` so the aborted workers have a
/// scheduler to land on — on `current_thread`, `abort()` just flags
/// the task and the drop can't complete until the runtime advances,
/// which in a single-threaded test is the current task itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_without_shutdown_releases_engine_arc() {
    let tmp = tempfile::tempdir().unwrap();
    let engine: Arc<MeruEngine> = MeruEngine::open(config(&tmp)).await.unwrap();
    let weak: Weak<MeruEngine> = Arc::downgrade(&engine);

    let workers = BackgroundWorkers::spawn(engine.clone());

    // Baseline: 1 (caller) + 1 (workers hold one clone that they clone
    // into each spawned task; the clones live inside the tokio task
    // futures, so they count toward strong_count).
    // With 1 flush worker + 2 compaction workers = 4 clones held by
    // tasks plus our `engine` Arc = 5 (approximate; exact count is
    // implementation detail — we just need it to drop to 1 after drop).
    assert!(
        weak.strong_count() > 1,
        "workers should hold strong refs while alive"
    );

    // Drop the workers. No .await, no shutdown() — simulates a
    // `MeruDB` dropping without `close()`.
    drop(workers);

    // Give the scheduler up to 2 seconds to advance aborted tasks to
    // their drop point. In practice this takes a few ms.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while weak.strong_count() > 1 && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
        // Yield to the runtime so the aborted tasks actually run their
        // drop code.
        tokio::task::yield_now().await;
    }

    assert_eq!(
        weak.strong_count(),
        1,
        "background workers must release Arc<MeruEngine> after drop; \
         lingering strong_count = {} implies tasks are still orphaned",
        weak.strong_count()
    );

    // Explicit close to flush the engine before the tempdir drops.
    // Without this, the tempdir cleanup races the engine's final
    // fsync and produces spurious warnings.
    engine.close().await.unwrap();
}
