//! Issue #23 repro attempt: under turbo-mode aggressive compaction
//! settings, a small percentage of deleted keys may still return
//! data on `get()`. This test drives the reported workload shape and
//! asserts that no deleted key resurrects.
//!
//! NOTE: this test is explicitly `#[ignore]` because it runs for tens
//! of seconds and intentionally stresses the engine with conditions
//! that might produce flaky-looking failures. Run explicitly with
//! `cargo test --test ghost_rows_turbo -- --ignored`.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

type Shadow = Arc<Mutex<HashMap<(i64, Vec<u8>), bool>>>;

use bytes::Bytes;
use merutable::engine::{EngineConfig, MeruEngine};
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};
use tempfile::TempDir;

fn schema() -> TableSchema {
    TableSchema {
        table_name: "ghost".into(),
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
        level_target_bytes: vec![
            64 * 1024 * 1024,
            512 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            32 * 1024 * 1024 * 1024,
        ],
        max_compaction_bytes: 64 * 1024 * 1024,
        compaction_parallelism: 4,
        flush_parallelism: 2,
        gc_grace_period_secs: 30,
        memtable_size_bytes: 16 * 1024 * 1024,
        row_cache_capacity: 1_000,
        ..Default::default()
    }
}

fn make_row(id: i64, key2: &[u8]) -> Row {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn turbo_mode_deletes_do_not_resurrect() {
    let tmp = tempfile::tempdir().unwrap();
    let engine: Arc<MeruEngine> = MeruEngine::open(turbo_config(&tmp)).await.unwrap();

    // Deterministic pseudo-random: each "batch" picks from a pool of
    // keys, either writes, overwrites, or deletes. Shadow state
    // tracks the authoritative final op per PK.
    let shadow: Shadow = Arc::new(Mutex::new(HashMap::new()));
    // true  = latest op is Put
    // false = latest op is Delete

    const TOTAL_OPS: i64 = 50_000;
    for i in 0..TOTAL_OPS {
        let id = i % 5_000; // reuse keys to create overwrites/deletes
        let key2 = format!("k{:04}", i % 113).into_bytes();
        let pk_vals = pk(id, &key2);

        // 15% overwrite, 5% delete, 80% fresh put.
        let decision = i % 100;
        if decision < 5 {
            engine.delete(pk_vals.clone()).await.unwrap();
            shadow.lock().unwrap().insert((id, key2.clone()), false);
        } else {
            engine
                .put(pk_vals.clone(), make_row(id, &key2))
                .await
                .unwrap();
            shadow.lock().unwrap().insert((id, key2.clone()), true);
        }

        // Every 2000 ops, verify sampled keys.
        if i > 0 && i % 2000 == 0 {
            let snap: Vec<((i64, Vec<u8>), bool)> = shadow
                .lock()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            // Sample every 50th key.
            for ((id, key2), expected_present) in snap.iter().step_by(50) {
                let pk_vals = pk(*id, key2);
                let got = engine.get(&pk_vals).unwrap();
                let actual_present = got.is_some();
                assert_eq!(
                    actual_present,
                    *expected_present,
                    "Issue #23 regression at op {i}: pk=({id}, {:?}): \
                     expected present={expected_present}, got={actual_present} \
                     (get returned {got:?})",
                    String::from_utf8_lossy(key2)
                );
            }
        }
    }

    // Final full-pass integrity check.
    engine.flush().await.unwrap();
    let final_shadow: Vec<((i64, Vec<u8>), bool)> = shadow
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    for ((id, key2), expected_present) in &final_shadow {
        let pk_vals = pk(*id, key2);
        let got = engine.get(&pk_vals).unwrap();
        let actual_present = got.is_some();
        assert_eq!(
            actual_present,
            *expected_present,
            "Issue #23 final check: pk=({id}, {:?}) expected={expected_present} got={actual_present}",
            String::from_utf8_lossy(key2)
        );
    }

    engine.close().await.unwrap();
}
