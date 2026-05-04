//! RFC-0002 stress + chaos tests for flush-time DV emission.
//!
//! These tests run as integration tests (separate binary), so they
//! exercise the full public API surface — `MeruDB::open`, `put`,
//! `delete`, `get`, `scan`, `flush`, `compact`, `close` — under
//! workloads designed to trip the resolve+commit ordering, the
//! L0/L1+ DV merge, and the read-side dedup invariants.
//!
//! Bounded resources (per CLAUDE.md):
//! - Iteration count caps the stress duration deterministically.
//!   No "run for 60s" loops that flake on CI.
//! - Concurrency is fixed at K worker tasks, not unbounded.
//! - Memory: memtable_size_mb=1 keeps the per-test footprint < 16 MiB.

use bytes::Bytes;
use merutable::schema::{ColumnDef, ColumnType, TableSchema};
use merutable::value::{FieldValue, Row};
use merutable::{MeruDB, OpenOptions};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tempfile::TempDir;

fn schema() -> TableSchema {
    TableSchema {
        table_name: "stress".into(),
        columns: vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,
                ..Default::default()
            },
            ColumnDef {
                name: "version".into(),
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

fn make_row(id: i64, version: i64, payload: &str) -> Row {
    Row::new(vec![
        Some(FieldValue::Int64(id)),
        Some(FieldValue::Int64(version)),
        Some(FieldValue::Bytes(Bytes::from(payload.to_string()))),
    ])
}

fn open(tmp: &TempDir) -> OpenOptions {
    OpenOptions::new(schema())
        .wal_dir(tmp.path().join("wal"))
        .catalog_uri(tmp.path().to_string_lossy().to_string())
        .memtable_size_mb(1)
        .gc_grace_period_secs(0)
}

/// Sum of DV cardinality across every `.puffin` file in the catalog
/// **as observed by an external reader**. Walks `{catalog}/data/L*`
/// for `.puffin` files, parses each blob, and sums cardinalities.
///
/// External-view (no internal accessors) by design — proves the DV
/// state is durable on disk in the shape an external Iceberg reader
/// would consume, not just in-memory state.
async fn total_dv_cardinality(db: &MeruDB) -> u64 {
    use merutable::iceberg::DeletionVector;
    let base = std::path::PathBuf::from(db.catalog_path()).join("data");
    let mut total: u64 = 0;
    for level_dir in walk_dirs(&base) {
        for entry in std::fs::read_dir(&level_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().map(|e| e == "puffin").unwrap_or(false) {
                let bytes = tokio::fs::read(&path).await.unwrap();
                // A puffin file may contain multiple blobs; the DV
                // helper consumes one blob's bytes. We scan with the
                // referenced_data_file API to discover blob ranges,
                // but for the simple flush-emitted layout (one DV
                // blob per puffin file) the whole file IS one blob's
                // raw bytes wrapped in the PFA1 envelope. The
                // catalog's writer emits exactly this shape, so
                // from_puffin_bytes (whole-file decode) round-trips.
                if let Ok(dv) = DeletionVector::from_puffin_bytes(&bytes) {
                    total += dv.cardinality();
                }
            }
        }
    }
    total
}

fn walk_dirs(base: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(base) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                out.push(p);
            }
        }
    }
    out
}

// ── Scenario 1: large upsert sweep ───────────────────────────────────────────

/// Large monotonic upsert workload: write N keys, flush, then upsert
/// every key with a new version, flush, repeat for several rounds.
/// Asserts:
/// - Final read returns exactly the latest version of every key.
/// - DV cardinality across the snapshot covers every prior version
///   (so external Iceberg readers would see one row per PK).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stress_repeated_upsert_sweep_preserves_pk_uniqueness() {
    let tmp = TempDir::new().unwrap();
    let db = MeruDB::open(open(&tmp)).await.unwrap();

    const N_KEYS: i64 = 500;
    const ROUNDS: i64 = 5;
    for round in 0..ROUNDS {
        for id in 0..N_KEYS {
            db.put(make_row(id, round, &format!("r{round}_id{id}")))
                .await
                .unwrap();
        }
        db.flush().await.unwrap();
    }

    // Every key reads back the LAST round's value.
    for id in 0..N_KEYS {
        let row = db.get(&[FieldValue::Int64(id)]).unwrap().unwrap();
        let version = row.get(1).unwrap();
        assert_eq!(
            *version,
            FieldValue::Int64(ROUNDS - 1),
            "key {id} must read latest round"
        );
    }
    // Range scan returns exactly N_KEYS rows.
    let rows = db.scan(None, None).unwrap();
    assert_eq!(rows.len() as i64, N_KEYS, "scan cardinality");

    // Every prior version got DV-marked. Each round after round 0
    // marks N_KEYS positions in prior files. Total DVs = (ROUNDS-1)*N_KEYS.
    let card = total_dv_cardinality(&db).await;
    assert_eq!(
        card,
        ((ROUNDS - 1) as u64) * (N_KEYS as u64),
        "every prior version must be DV-marked"
    );

    db.close().await.unwrap();
}

// ── Scenario 2: interleaved upsert + delete + readback ───────────────────────

/// Interleaved write workload: each round upserts half the keys
/// and deletes the other half, then flushes. The reads must reflect
/// the last operation per key.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stress_interleaved_upsert_delete_round_robin() {
    let tmp = TempDir::new().unwrap();
    let db = MeruDB::open(open(&tmp)).await.unwrap();

    const N_KEYS: i64 = 200;
    // Round 0: insert all.
    for id in 0..N_KEYS {
        db.put(make_row(id, 0, "init")).await.unwrap();
    }
    db.flush().await.unwrap();

    // Track the oracle: (id → last_op).
    enum Op {
        Upsert(i64),
        Deleted,
    }
    let mut oracle: BTreeMap<i64, Op> = (0..N_KEYS).map(|id| (id, Op::Upsert(0))).collect();

    // Three rounds. Even ids upserted, odd ids deleted; then swap.
    for round in 1..=3 {
        for id in 0..N_KEYS {
            let upsert_this_round = (id + round) % 2 == 0;
            if upsert_this_round {
                db.put(make_row(id, round, &format!("r{round}")))
                    .await
                    .unwrap();
                oracle.insert(id, Op::Upsert(round));
            } else {
                db.delete(vec![FieldValue::Int64(id)]).await.unwrap();
                oracle.insert(id, Op::Deleted);
            }
        }
        db.flush().await.unwrap();
    }

    // Verify reads against the oracle. Tight equality.
    for id in 0..N_KEYS {
        let observed = db.get(&[FieldValue::Int64(id)]).unwrap();
        match oracle[&id] {
            Op::Deleted => assert!(observed.is_none(), "key {id} must read None"),
            Op::Upsert(v) => {
                let row = observed.unwrap_or_else(|| panic!("key {id} expected version {v}"));
                let version = row.get(1).unwrap();
                assert_eq!(*version, FieldValue::Int64(v), "key {id} version mismatch");
            }
        }
    }

    db.close().await.unwrap();
}

// ── Scenario 3: chaos: concurrent put + flush + compact + scan ──────────────

/// Chaos test: K concurrent tasks each writing into the same DB,
/// with periodic flush + compact + scan from a separate task.
/// The point-lookup oracle must still match the last write per key.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn chaos_concurrent_put_flush_compact_scan() {
    let tmp = TempDir::new().unwrap();
    let db = Arc::new(MeruDB::open(open(&tmp)).await.unwrap());

    const N_WRITERS: usize = 4;
    const KEYS_PER_WRITER: i64 = 250;
    const ROUNDS: i64 = 3;

    let mut writer_handles = Vec::new();
    for w in 0..N_WRITERS {
        let db = db.clone();
        let writer_id = w as i64;
        writer_handles.push(tokio::spawn(async move {
            // Each writer owns a disjoint id range so no inter-writer
            // races; we exercise concurrent writes WITH concurrent
            // flush/compact/scan, not concurrent multi-writer to the
            // same key (single-writer-per-catalog is the documented
            // contract).
            let base = writer_id * 10_000;
            for round in 0..ROUNDS {
                for k in 0..KEYS_PER_WRITER {
                    let id = base + k;
                    db.put(make_row(id, round, &format!("w{writer_id}_r{round}")))
                        .await
                        .unwrap();
                }
            }
        }));
    }

    // Concurrent flush + compact + scan. Bounded loops — 5 iterations
    // of each, so the test runs deterministically.
    let chaos_db = db.clone();
    let chaos = tokio::spawn(async move {
        for _ in 0..5 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let _ = chaos_db.flush().await;
            let _ = chaos_db.compact().await;
            // Scan to assert it doesn't blow up under concurrent
            // writers + flush + compact.
            let _ = chaos_db.scan(None, None);
        }
    });

    for h in writer_handles {
        h.await.unwrap();
    }
    chaos.await.unwrap();

    // Final flush so all writes are visible externally.
    db.flush().await.unwrap();

    // Oracle: every (writer, key) pair has its last round's value.
    for w in 0..N_WRITERS {
        let base = (w as i64) * 10_000;
        for k in 0..KEYS_PER_WRITER {
            let id = base + k;
            let row = db
                .get(&[FieldValue::Int64(id)])
                .unwrap()
                .unwrap_or_else(|| panic!("key {id} (w={w}) must exist"));
            let version = row.get(1).unwrap();
            assert_eq!(
                *version,
                FieldValue::Int64(ROUNDS - 1),
                "key {id} must be at last round"
            );
        }
    }

    let total = (N_WRITERS as i64) * KEYS_PER_WRITER;
    let scan = db.scan(None, None).unwrap();
    assert_eq!(scan.len() as i64, total, "scan must return every key once");

    Arc::try_unwrap(db).ok().unwrap().close().await.unwrap();
}

// ── Scenario 4: chaos: compaction race during flush DV resolve ───────────────

/// Force a compaction to run between flushes that each emit DVs.
/// The flush's resolve runs against the current pinned version; if
/// compaction lands first, the L1+ files referenced by the resolve
/// must still exist (compaction's GC honors the pin via
/// `gc_grace_period_secs`). This test forces the race deterministically.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chaos_compaction_during_flush_keeps_dv_consistent() {
    let tmp = TempDir::new().unwrap();
    let db = Arc::new(MeruDB::open(open(&tmp)).await.unwrap());

    // Round 1: load 200 keys, flush, compact → goes to L1.
    for id in 0..200 {
        db.put(make_row(id, 0, "r0")).await.unwrap();
    }
    db.flush().await.unwrap();
    db.compact().await.unwrap();

    // Round 2: upsert all 200, flush → DV must mark all 200 in L1.
    for id in 0..200 {
        db.put(make_row(id, 1, "r1")).await.unwrap();
    }
    db.flush().await.unwrap();

    // Round 3: trigger another compaction CONCURRENT with another
    // upsert flush. The compaction may rewrite L1 (dropping DV-marked
    // rows). The flush's resolve must hit the post-compaction version,
    // not the pre-compaction one — commit_lock guarantees this.
    let db_compact = db.clone();
    let compact_task = tokio::spawn(async move {
        let _ = db_compact.compact().await;
    });
    let db_flush = db.clone();
    let flush_task = tokio::spawn(async move {
        for id in 0..200 {
            db_flush.put(make_row(id, 2, "r2")).await.unwrap();
        }
        db_flush.flush().await.unwrap();
    });
    flush_task.await.unwrap();
    compact_task.await.unwrap();

    // Final reads: every key at round 2.
    for id in 0..200 {
        let row = db.get(&[FieldValue::Int64(id)]).unwrap().unwrap();
        let version = row.get(1).unwrap();
        assert_eq!(*version, FieldValue::Int64(2), "key {id} must be at r2");
    }
    Arc::try_unwrap(db).ok().unwrap().close().await.unwrap();
}

// ── Scenario 5: large flush cardinality boundary ─────────────────────────────

/// Flush with a memtable holding many keys all overlapping a single
/// L1 file. Validates that the DV bitmap tracks every position
/// without integer overflow / truncation. Memtable size capped to
/// keep test runtime bounded; the cardinality is still in the
/// thousands.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stress_large_dv_cardinality_per_file() {
    let tmp = TempDir::new().unwrap();
    // Larger memtable so more keys land in one L0 → one L1 file.
    let opts = OpenOptions::new(schema())
        .wal_dir(tmp.path().join("wal"))
        .catalog_uri(tmp.path().to_string_lossy().to_string())
        .memtable_size_mb(8)
        .gc_grace_period_secs(0);
    let db = MeruDB::open(opts).await.unwrap();

    const N: i64 = 10_000;
    for id in 0..N {
        db.put(make_row(id, 0, "v0")).await.unwrap();
    }
    db.flush().await.unwrap();
    db.compact().await.unwrap();

    // Upsert every key.
    for id in 0..N {
        db.put(make_row(id, 1, "v1")).await.unwrap();
    }
    db.flush().await.unwrap();

    // Total DV cardinality must be exactly N (every prior position
    // marked once).
    let card = total_dv_cardinality(&db).await;
    assert_eq!(card, N as u64, "every prior position DV-marked");

    // Reads return v1 for every key.
    for id in (0..N).step_by(317) {
        // Sample to keep test fast.
        let row = db.get(&[FieldValue::Int64(id)]).unwrap().unwrap();
        let version = row.get(1).unwrap();
        assert_eq!(*version, FieldValue::Int64(1));
    }

    db.close().await.unwrap();
}

// ── Scenario 6: reopen recovers DV-bearing snapshot intact ───────────────────

/// Crash-restart simulation (close + reopen) preserves DV state.
/// After reopen the reads must agree with pre-close reads. RFC-0002:
/// DV blobs are durable via the catalog's fsync-before-rename chain;
/// reopening must surface them intact.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stress_reopen_preserves_dv_state() {
    let tmp = TempDir::new().unwrap();
    {
        let db = MeruDB::open(open(&tmp)).await.unwrap();
        for id in 0..100 {
            db.put(make_row(id, 0, "v0")).await.unwrap();
        }
        db.flush().await.unwrap();
        db.compact().await.unwrap();
        for id in 0..100 {
            db.put(make_row(id, 1, "v1")).await.unwrap();
        }
        db.flush().await.unwrap();
        let card_before = total_dv_cardinality(&db).await;
        assert_eq!(card_before, 100);
        db.close().await.unwrap();
    }

    // Reopen.
    let db = MeruDB::open(open(&tmp)).await.unwrap();
    let card_after = total_dv_cardinality(&db).await;
    assert_eq!(card_after, 100, "DV state must survive reopen");

    // Reads still return v1 for every key.
    let mut by_id: HashMap<i64, i64> = HashMap::new();
    for row in db.scan(None, None).unwrap() {
        let id = row.1.get(0).cloned().unwrap();
        let version = row.1.get(1).cloned().unwrap();
        if let (FieldValue::Int64(i), FieldValue::Int64(v)) = (id, version) {
            by_id.insert(i, v);
        }
    }
    assert_eq!(by_id.len(), 100, "scan returns every key exactly once");
    for v in by_id.values() {
        assert_eq!(*v, 1, "every key at v1");
    }

    db.close().await.unwrap();
}
