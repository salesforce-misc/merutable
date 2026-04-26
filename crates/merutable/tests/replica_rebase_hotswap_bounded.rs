#![cfg(feature = "replica")]
//! Issue #34 regression: `Replica::rebase_hotswap` MUST complete
//! (or error cleanly via timeout) under concurrent primary writes.
//!
//! Pre-#34, `InProcessLogSource::stream` looped `cursor.next_batch`
//! until empty, but each `next_batch` re-sampled `engine.read_seq()`.
//! Under a live primary with steady writes, the target kept
//! advancing and the drain never terminated — rebase_hotswap hung
//! forever (observed 35+ min), tail RSS ballooned 10x disk size.
//!
//! Two properties pinned here:
//!   1. Snapshot-bounded drain: `stream()` caps at the primary's
//!      read_seq captured AT stream-open time. New ops committed
//!      after that are out of scope for this drain; the next
//!      advance/rebase picks them up.
//!   2. Timeout safety net: `rebase_hotswap` wraps warmup in
//!      `tokio::time::timeout` with a configurable deadline.
//!      Hangs from any future `LogSource` impl (Raft, Kafka,
//!      ObjectStore) produce a clean error instead of a deadlock.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use merutable::replica::{InProcessLogSource, Replica};
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};
use merutable::{MeruDB, OpenOptions};

fn schema() -> TableSchema {
    TableSchema {
        table_name: "issue34-rebase".into(),
        columns: vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,
                ..Default::default()
            },
            ColumnDef {
                name: "v".into(),
                col_type: ColumnType::Int64,
                nullable: true,
                ..Default::default()
            },
        ],
        primary_key: vec![0],
        ..Default::default()
    }
}

fn row(id: i64, v: i64) -> Row {
    Row::new(vec![
        Some(FieldValue::Int64(id)),
        Some(FieldValue::Int64(v)),
    ])
}

async fn open_primary(tmp: &tempfile::TempDir) -> Arc<MeruDB> {
    Arc::new(
        MeruDB::open(
            OpenOptions::new(schema())
                .wal_dir(tmp.path().join("wal"))
                .catalog_uri(tmp.path().to_string_lossy().to_string()),
        )
        .await
        .unwrap(),
    )
}

/// Core regression: primary under continuous write load while
/// rebase_hotswap runs. Before #34, this would hang indefinitely.
/// After #34, the snapshot-bounded drain guarantees termination
/// regardless of the writer's rate.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rebase_hotswap_completes_under_concurrent_writes() {
    let primary_tmp = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_tmp).await;

    // Seed so the base has committed data.
    for i in 1..=50i64 {
        primary.put(row(i, i * 10)).await.unwrap();
    }
    primary.flush().await.unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let writer_stop = stop.clone();
    let writer_db = primary.clone();
    let writer = tokio::spawn(async move {
        // Steady-state writer: keeps issuing puts until stopped.
        // This is the workload that hung rebase pre-#34.
        let mut next_id: i64 = 1000;
        while !writer_stop.load(Ordering::Relaxed) {
            // Best-effort put; if the DB is closing we stop.
            if writer_db.put(row(next_id, next_id)).await.is_err() {
                break;
            }
            next_id += 1;
            // Yield so the test task gets to run; without this
            // the writer can starve the single-pool runtime.
            tokio::task::yield_now().await;
        }
    });

    // Give the writer a head start so read_seq is moving when
    // rebase_hotswap opens its log stream.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(
        OpenOptions::new(schema())
            .wal_dir(primary_tmp.path().join("wal"))
            .catalog_uri(primary_tmp.path().to_string_lossy().to_string())
            .read_only(true),
        log,
    )
    .await
    .unwrap();

    // 30s outer budget — the snapshot-bounded drain should
    // finish in well under a second even with concurrent writes.
    // Pre-#34 this would time out; the explicit budget converts
    // a hang into a test failure instead of a hung CI job.
    let started = std::time::Instant::now();
    let out = tokio::time::timeout(Duration::from_secs(30), replica.rebase_hotswap()).await;
    stop.store(true, Ordering::Relaxed);
    let _ = writer.await;

    let elapsed = started.elapsed();
    let (new_base_seq, new_visible_seq) = out
        .expect("rebase_hotswap should not hang")
        .expect("rebase_hotswap should succeed");

    assert!(
        new_base_seq >= 50,
        "new base must include the seeded rows: got {new_base_seq}"
    );
    assert!(
        new_visible_seq >= new_base_seq,
        "visible_seq ({new_visible_seq}) >= base_seq ({new_base_seq})"
    );
    assert!(
        elapsed < Duration::from_secs(25),
        "rebase_hotswap took {elapsed:?} — expected << 25s under bounded drain"
    );

    // Verify metrics surface the warmup.
    let stats = replica.stats().await;
    assert_eq!(stats.rebase_count, 1);
    assert_eq!(stats.base_seq, new_base_seq);
}

/// The timeout wrapper IS plumbed: if the drain stalls, the
/// call errors cleanly instead of deadlocking. This belt-and-
/// braces path guards against any future LogSource impl that
/// could reintroduce unbounded drain (Raft, Kafka, HTTP).
///
/// Uses a deliberately slow custom `LogSource` — `stream()`
/// sleeps for an hour before yielding. The 100ms timeout fires
/// deterministically regardless of host speed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rebase_hotswap_respects_configured_timeout() {
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use merutable::replica::{LogSource, OpRecord};
    use merutable::types::{MeruError, Result};

    struct SleepyLogSource;

    #[async_trait]
    impl LogSource for SleepyLogSource {
        async fn stream(&self, _since: u64) -> Result<BoxStream<'static, Result<OpRecord>>> {
            // Simulate a hung drain — e.g., a network log source
            // blocked on a connection that never completes.
            tokio::time::sleep(Duration::from_secs(3600)).await;
            unreachable!("timeout must fire before we get here")
        }

        async fn latest_seq(&self) -> Result<u64> {
            Ok(0)
        }
    }

    let primary_tmp = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_tmp).await;
    for i in 1..=10i64 {
        primary.put(row(i, i)).await.unwrap();
    }

    let replica = Replica::open(
        OpenOptions::new(schema())
            .wal_dir(primary_tmp.path().join("wal"))
            .catalog_uri(primary_tmp.path().to_string_lossy().to_string())
            .read_only(true),
        Arc::new(SleepyLogSource),
    )
    .await
    .unwrap();

    replica.set_rebase_timeout(Some(Duration::from_millis(100)));
    assert_eq!(replica.rebase_timeout(), Some(Duration::from_millis(100)));

    let started = std::time::Instant::now();
    let err = replica
        .rebase_hotswap()
        .await
        .expect_err("rebase should time out against a hung log source");
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "timeout fired late: {elapsed:?} — should be ~100ms, not hours"
    );
    match err {
        MeruError::InvalidArgument(s) => {
            assert!(
                s.contains("timed out") && s.contains("rebase_hotswap"),
                "error should name rebase_hotswap + timeout: {s}"
            );
        }
        other => panic!("expected InvalidArgument(timeout), got {other:?}"),
    }

    // OLD state must still be live — a timed-out rebase must not
    // leave the replica in a half-swapped state. rebase_count
    // stays 0 (no completed rebase).
    let stats = replica.stats().await;
    assert_eq!(
        stats.rebase_count, 0,
        "timed-out rebase must not bump rebase_count"
    );
}

/// Disabling the timeout (`None`) is a first-class config — the
/// call runs to completion even if it takes long. Verifies the
/// `None` → 0-millis → "unbounded" mapping doesn't accidentally
/// wrap in `tokio::time::timeout(0)` and fire instantly.
#[tokio::test]
async fn rebase_hotswap_none_timeout_runs_unbounded() {
    let primary_tmp = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_tmp).await;
    primary.put(row(1, 1)).await.unwrap();

    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(
        OpenOptions::new(schema())
            .wal_dir(primary_tmp.path().join("wal"))
            .catalog_uri(primary_tmp.path().to_string_lossy().to_string())
            .read_only(true),
        log,
    )
    .await
    .unwrap();

    replica.set_rebase_timeout(None);
    assert_eq!(replica.rebase_timeout(), None);

    // With no primary under load this completes quickly — but
    // the path being exercised is the `else { work.await? }`
    // branch, not the `timeout().await` branch. A regression
    // that turned "no timeout" into `timeout(Duration::ZERO)`
    // would fail this.
    let (_base, _visible) = tokio::time::timeout(Duration::from_secs(5), replica.rebase_hotswap())
        .await
        .expect("no-timeout rebase should still complete quickly with idle primary")
        .expect("no-timeout rebase should succeed");
}

/// Snapshot-bound property at the LogSource layer: ops committed
/// AFTER `stream()` opens must not appear in that drain. Pre-#34
/// they did (until the writer stopped, which under live load
/// was never). This test pins the contract directly without
/// going through Replica so a regression in InProcessLogSource
/// surfaces with a narrow failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_process_log_source_drain_is_snapshot_bounded() {
    use futures::StreamExt;
    use merutable::replica::LogSource;

    let primary_tmp = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_tmp).await;
    for i in 1..=5i64 {
        primary.put(row(i, i)).await.unwrap();
    }
    let upper_at_open = primary.read_seq().0;

    let log = InProcessLogSource::new(primary.clone());
    let mut stream = log.stream(0).await.unwrap();

    // Commit MORE ops AFTER stream() returned. These must not
    // appear in this drain.
    for i in 100..=110i64 {
        primary.put(row(i, i)).await.unwrap();
    }

    let mut max_seq = 0u64;
    let mut count = 0;
    while let Some(r) = stream.next().await {
        let rec = r.unwrap();
        assert!(
            rec.seq <= upper_at_open,
            "stream yielded seq {} > snapshot upper {} — drain is NOT snapshot-bounded",
            rec.seq,
            upper_at_open
        );
        max_seq = max_seq.max(rec.seq);
        count += 1;
    }
    assert!(count >= 5, "should have drained seeded ops");
    assert!(
        max_seq <= upper_at_open,
        "max drained seq {max_seq} must be <= snapshot upper {upper_at_open}"
    );
}
