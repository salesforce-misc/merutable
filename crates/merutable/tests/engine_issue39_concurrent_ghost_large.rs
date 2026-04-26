//! Issue #39 — aggressive stress reproducer (mid-scale).
//!
//! More concurrent than `engine_issue39_concurrent_ghost.rs`:
//!   - 8 concurrent readers (matches the reporter's harness).
//!   - 40_000 ops across 4_000 PKs so the LSM reaches L2 under
//!     turbo's compressed `level_target_bytes`.
//!   - Mid-run *synchronous* integrity checkpoints: drain
//!     flush+compact to steady state, sweep every shadow entry
//!     from the write-loop task (serialized with writes). This
//!     is the authoritative check — shadow-update-lag cannot
//!     apply because the sweep runs INSIDE the write task, after
//!     the writer's own latest op returned.
//!   - Final post-quiescence sweep over every tracked key.
//!
//! Any ghost OR missing live key surfaced by a synchronous
//! checkpoint is a real engine bug. The test fails loudly with
//! the first 5 offending keys so the invariant violation is
//! bisectable to a specific op sequence.
//!
//! `#[ignore]` — the op count + concurrent readers take minutes.
//! Run explicitly:
//!   cargo test --test engine_issue39_concurrent_ghost_large \
//!     -- --ignored --nocapture

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use bytes::Bytes;
use merutable::engine::{EngineConfig, MeruEngine};
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};
use tempfile::TempDir;

type Shadow = Arc<Mutex<HashMap<(i64, Vec<u8>), bool>>>;

fn schema() -> TableSchema {
    TableSchema {
        table_name: "issue39-large".into(),
        columns: vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,
                ..Default::default()
            },
            ColumnDef {
                name: "key2".into(),
                col_type: ColumnType::ByteArray,
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
        primary_key: vec![0, 1],
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
        // Reporter's turbo config (reasonably accurate).
        memtable_size_bytes: 16 * 1024 * 1024,
        row_cache_capacity: 1_000,
        compaction_parallelism: 4,
        flush_parallelism: 2,
        // Small level targets so the tree reaches L2/L3 under the
        // 100k-op workload. Matches the reporter's "data reaching
        // L3" observation.
        level_target_bytes: vec![
            32 * 1024 * 1024,       // L1
            128 * 1024 * 1024,      // L2
            512 * 1024 * 1024,      // L3
            2 * 1024 * 1024 * 1024, // L4
        ],
        ..Default::default()
    }
}

fn mk_row(id: i64, key2: &[u8]) -> Row {
    Row::new(vec![
        Some(FieldValue::Int64(id)),
        Some(FieldValue::Bytes(Bytes::copy_from_slice(key2))),
        Some(FieldValue::Bytes(Bytes::from(vec![b'x'; 256]))),
    ])
}

fn pk(id: i64, key2: &[u8]) -> Vec<FieldValue> {
    vec![
        FieldValue::Int64(id),
        FieldValue::Bytes(Bytes::copy_from_slice(key2)),
    ]
}

/// Synchronous integrity checkpoint: pause the writer, flush +
/// compact to steady state, then sweep every shadow entry. Returns
/// the ghost list (should always be empty).
async fn synchronous_checkpoint(
    engine: &Arc<MeruEngine>,
    shadow: &Shadow,
    label: &str,
) -> (Vec<(i64, Vec<u8>)>, Vec<(i64, Vec<u8>)>) {
    engine.flush().await.unwrap();
    engine.compact().await.unwrap();
    // Let any background sweep settle.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let snap: Vec<((i64, Vec<u8>), bool)> = shadow
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let mut ghosts = Vec::new();
    let mut missing = Vec::new();
    for ((id, key2), expected_present) in &snap {
        let got = engine.get(&pk(*id, key2)).unwrap();
        let actual_present = got.is_some();
        if actual_present != *expected_present {
            if *expected_present {
                missing.push((*id, key2.clone()));
            } else {
                ghosts.push((*id, key2.clone()));
            }
        }
    }
    eprintln!(
        "[{}] shadow_size={} ghosts={} missing={}",
        label,
        snap.len(),
        ghosts.len(),
        missing.len()
    );
    (ghosts, missing)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 12)]
#[ignore]
async fn issue39_aggressive_stress_no_ghosts() {
    let tmp = tempfile::tempdir().unwrap();
    let engine: Arc<MeruEngine> = MeruEngine::open(turbo_config(&tmp)).await.unwrap();

    let shadow: Shadow = Arc::new(Mutex::new(HashMap::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let concurrent_ghosts = Arc::new(AtomicU64::new(0));

    // 8 readers racing get() against live writes.
    let mut reader_handles = Vec::new();
    for _ in 0..8 {
        let e = engine.clone();
        let s = shadow.clone();
        let stp = stop.clone();
        let g = concurrent_ghosts.clone();
        reader_handles.push(tokio::spawn(async move {
            while !stp.load(Ordering::Relaxed) {
                let samples: Vec<((i64, Vec<u8>), bool)> = {
                    let sh = s.lock().unwrap();
                    sh.iter().take(50).map(|(k, v)| (k.clone(), *v)).collect()
                };
                for ((id, k2), present) in &samples {
                    if let Ok(got) = e.get(&pk(*id, k2))
                        && got.is_some() != *present
                    {
                        g.fetch_add(1, Ordering::Relaxed);
                    }
                }
                tokio::task::yield_now().await;
            }
        }));
    }

    const TOTAL_OPS: i64 = 40_000;
    const PK_POOL: i64 = 4_000;
    let mut checkpoints_at = Vec::new();

    for i in 0..TOTAL_OPS {
        let id = i % PK_POOL;
        let key2 = format!("k{:05}", i % 257).into_bytes();
        let pk_vals = pk(id, &key2);

        let decision = i % 100;
        if decision < 5 {
            engine.delete(pk_vals.clone()).await.unwrap();
            shadow.lock().unwrap().insert((id, key2), false);
        } else {
            engine
                .put(pk_vals.clone(), mk_row(id, &key2))
                .await
                .unwrap();
            shadow.lock().unwrap().insert((id, key2), true);
        }

        // Synchronous integrity checkpoint every 10k ops. The
        // shadow sweep runs INSIDE this writer task, so the check
        // is linearized with the write loop's timeline —
        // shadow-update-lag is impossible because the writer's
        // latest op already returned before the sweep starts.
        if i > 0 && i % 10_000 == 0 {
            let label = format!("checkpoint@{i}ops");
            let (ghosts, missing) = synchronous_checkpoint(&engine, &shadow, &label).await;
            checkpoints_at.push((i, ghosts.len(), missing.len()));
            assert!(
                ghosts.is_empty(),
                "Issue #39 regression at {label}: {} ghost(s). First 5: {:?}",
                ghosts.len(),
                ghosts.iter().take(5).collect::<Vec<_>>()
            );
            assert!(
                missing.is_empty(),
                "Issue #39 regression at {label}: {} missing live key(s). First 5: {:?}",
                missing.len(),
                missing.iter().take(5).collect::<Vec<_>>()
            );
        }
    }

    // Stop readers. Final authoritative sweep.
    stop.store(true, Ordering::Relaxed);
    for h in reader_handles {
        let _ = h.await;
    }
    let (ghosts, missing) = synchronous_checkpoint(&engine, &shadow, "final").await;
    let cg = concurrent_ghosts.load(Ordering::Relaxed);
    eprintln!(
        "concurrent_ghosts={cg} (in-flight — these include shadow-update-lag from the #23 closure; \
         authoritative checks above/below are what matters); mid-run checkpoints: {checkpoints_at:?}"
    );
    assert!(
        ghosts.is_empty(),
        "Issue #39 regression (final sweep): {} ghost(s). First 5: {:?}",
        ghosts.len(),
        ghosts.iter().take(5).collect::<Vec<_>>()
    );
    assert!(
        missing.is_empty(),
        "Issue #39 regression (final sweep): {} missing live. First 5: {:?}",
        missing.len(),
        missing.iter().take(5).collect::<Vec<_>>()
    );
}
