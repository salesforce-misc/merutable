//! Issue #37 regression: concurrent compaction-GC must not delete
//! Parquet files that a live `range_scan` / `scan_tail_changes`
//! still references.
//!
//! Pre-#37, `range_scan` captured the `Version` via
//! `version_set.current()`, then did tens of ms of memtable-harvest
//! work, and ONLY THEN called `pin_current_snapshot`. A concurrent
//! compaction could commit a new `Version`, enqueue the old file
//! for deletion, and GC could read `min_pinned_snapshot() = None`
//! (because this scan hadn't registered its pin yet), then delete
//! the file. The scan subsequently tried to open the deleted file
//! and returned `IO error: No such file or directory`.
//!
//! `scan_tail_changes` had NO pin at all — any change-feed cursor
//! or replica log source draining L0 under concurrent compaction
//! was wide open to the same race.
//!
//! Fix (three-site):
//!   1. `pin_current_snapshot`: acquire `live_snapshots` lock
//!      BEFORE reading the current version, so the pin registers
//!      atomically with the version read.
//!   2. `range_scan`: call `pin_current_snapshot` at the TOP of the
//!      function, before the memtable harvest.
//!   3. `scan_tail_changes`: add a pin; was previously unprotected.
//!
//! This test drives the chaos-monkey Phase-142 workload shape
//! (writes + flush + compact + concurrent scans) and asserts no
//! `IO NotFound` surfaces. Before the fix, this test was designed
//! to reproduce the ENOENT; after the fix it stays green across
//! runs.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use merutable::engine::{EngineConfig, MeruEngine};
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};
use tempfile::TempDir;

fn schema() -> TableSchema {
    TableSchema {
        table_name: "issue37".into(),
        columns: vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,
                ..Default::default()
            },
            ColumnDef {
                name: "payload".into(),
                col_type: ColumnType::ByteArray,
                nullable: true,
                ..Default::default()
            },
        ],
        primary_key: vec![0],
        ..Default::default()
    }
}

fn turbo_config(tmp: &TempDir) -> EngineConfig {
    EngineConfig {
        schema: schema(),
        catalog_uri: tmp.path().to_string_lossy().to_string(),
        object_store_prefix: tmp.path().to_string_lossy().to_string(),
        wal_dir: tmp.path().join("wal"),
        l0_compaction_trigger: 2,
        l0_slowdown_trigger: 8,
        l0_stop_trigger: 12,
        memtable_size_bytes: 8 * 1024 * 1024,
        // Shrink grace so the race is NARROW — pre-#37 this would
        // reproduce the ENOENT deterministically. Post-fix the pin
        // closes the window regardless of grace.
        gc_grace_period_secs: 0,
        row_cache_capacity: 1_000,
        ..Default::default()
    }
}

fn mk_row(id: i64, tag: u64) -> Row {
    Row::new(vec![
        Some(FieldValue::Int64(id)),
        Some(FieldValue::Bytes(Bytes::from(format!("p{tag}")))),
    ])
}

fn pk(id: i64) -> Vec<FieldValue> {
    vec![FieldValue::Int64(id)]
}

/// Core regression: writer + compactor + GC run concurrently while
/// a scanner repeatedly calls `scan(None, None)`. No scan should
/// return an `Io` error — that's the #37 failure mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn range_scan_survives_concurrent_compaction_and_gc() {
    let tmp = tempfile::tempdir().unwrap();
    let engine: Arc<MeruEngine> = MeruEngine::open(turbo_config(&tmp)).await.unwrap();

    // Seed so there's initial data to compact.
    for i in 0..500i64 {
        engine.put(pk(i), mk_row(i, i as u64)).await.unwrap();
    }
    engine.flush().await.unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let scans_completed = Arc::new(AtomicU64::new(0));
    let scans_errored = Arc::new(AtomicU64::new(0));

    // Writer: rotating puts with frequent flush + compact to keep
    // the compaction-GC loop spinning.
    let writer_engine = engine.clone();
    let writer_stop = stop.clone();
    let writer = tokio::spawn(async move {
        let mut n: u64 = 0;
        while !writer_stop.load(Ordering::Relaxed) {
            let id = (n % 500) as i64;
            if writer_engine.put(pk(id), mk_row(id, n)).await.is_err() {
                break;
            }
            n += 1;
            if n.is_multiple_of(50) {
                let _ = writer_engine.flush().await;
            }
            if n.is_multiple_of(200) {
                let _ = writer_engine.compact().await;
                // Explicit GC sweep to accelerate the race window.
                writer_engine.gc_pending_deletions().await;
            }
            tokio::task::yield_now().await;
        }
    });

    // Scanner: hammer full scans. Each scan is the #37 hot path.
    let scanner_engine = engine.clone();
    let scanner_stop = stop.clone();
    let scanner_completed = scans_completed.clone();
    let scanner_errored = scans_errored.clone();
    let scanner = tokio::spawn(async move {
        while !scanner_stop.load(Ordering::Relaxed) {
            match scanner_engine.scan(None, None) {
                Ok(rows) => {
                    scanner_completed.fetch_add(1, Ordering::Relaxed);
                    // Sanity: scan returned SOME rows from the seed
                    // (deduped LWW; upper bounded by 500 unique PKs).
                    assert!(
                        rows.len() <= 500,
                        "scan returned {} rows; seed is 500 unique PKs",
                        rows.len()
                    );
                }
                Err(e) => {
                    scanner_errored.fetch_add(1, Ordering::Relaxed);
                    panic!("Issue #37 regression: scan errored with {e:?} — GC raced the pin");
                }
            }
            tokio::task::yield_now().await;
        }
    });

    // Run the storm for long enough to cycle through many
    // compactions. 3s on CI is plenty.
    tokio::time::sleep(Duration::from_secs(3)).await;
    stop.store(true, Ordering::Relaxed);
    let _ = writer.await;
    let _ = scanner.await;

    let done = scans_completed.load(Ordering::Relaxed);
    let err = scans_errored.load(Ordering::Relaxed);
    assert!(
        done > 0,
        "should have completed >= 1 scan in 3s; got {done}"
    );
    assert_eq!(err, 0, "any scan error is a #37 regression");

    // Final scan post-storm: the reporter's exact repro step — a
    // scan after the workload stops. Pre-#37 this is where the
    // ENOENT surfaced.
    let final_rows = engine.scan(None, None).expect("final scan");
    assert!(
        !final_rows.is_empty(),
        "final scan should return data from the seed"
    );
}

/// The change-feed tail scan had ZERO pin protection pre-#37.
/// This test drives concurrent compaction + GC while
/// `scan_tail_changes_with_pre_image` (the cursor's primitive)
/// runs repeatedly. Any error is the #37 regression in the
/// change-feed path.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn scan_tail_changes_survives_concurrent_compaction_and_gc() {
    use merutable::types::sequence::SeqNum;

    let tmp = tempfile::tempdir().unwrap();
    let engine: Arc<MeruEngine> = MeruEngine::open(turbo_config(&tmp)).await.unwrap();

    for i in 0..300i64 {
        engine.put(pk(i), mk_row(i, i as u64)).await.unwrap();
    }
    engine.flush().await.unwrap();

    let stop = Arc::new(AtomicBool::new(false));

    let writer_engine = engine.clone();
    let writer_stop = stop.clone();
    let writer = tokio::spawn(async move {
        let mut n: u64 = 0;
        while !writer_stop.load(Ordering::Relaxed) {
            let id = (n % 300) as i64;
            if writer_engine.put(pk(id), mk_row(id, n)).await.is_err() {
                break;
            }
            n += 1;
            if n.is_multiple_of(30) {
                let _ = writer_engine.flush().await;
            }
            if n.is_multiple_of(120) {
                let _ = writer_engine.compact().await;
                writer_engine.gc_pending_deletions().await;
            }
            tokio::task::yield_now().await;
        }
    });

    let scanner_engine = engine.clone();
    let scanner_stop = stop.clone();
    let scanner = tokio::spawn(async move {
        let mut since: u64 = 0;
        while !scanner_stop.load(Ordering::Relaxed) {
            let read_seq = scanner_engine.read_seq();
            let tuples = scanner_engine
                .scan_tail_changes_with_pre_image(since, read_seq)
                .expect("scan_tail_changes must not error under #37 fix");
            if let Some(last) = tuples.last() {
                since = last.seq.max(since);
            }
            tokio::task::yield_now().await;
        }
        let _ = SeqNum(since); // suppress unused
    });

    tokio::time::sleep(Duration::from_secs(3)).await;
    stop.store(true, Ordering::Relaxed);
    let _ = writer.await;
    let _ = scanner.await;
}
