//! RFC-0002 Phase 4 bench: prove the load-bearing performance contract.
//!
//! Two benches:
//!
//! 1. `put_latency_dv_on_vs_off` — single-row put latency, measured on
//!    a fresh DB with one prior file. The flush DV emission MUST NOT
//!    regress the write path. We bench `put` on a steady state where
//!    the memtable is bounded; flush happens off-bench. RFC-0002
//!    non-negotiable: `db.put()` does no file I/O.
//!
//! 2. `flush_dv_emission_overhead` — wall time of a full upsert flush
//!    that DV-marks N L1+ priors, vs. the same flush with DV emission
//!    off. The cost is the resolve+commit chain.

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use merutable::schema::{ColumnDef, ColumnType, TableSchema};
use merutable::value::{FieldValue, Row};
use merutable::{MeruDB, OpenOptions};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

fn schema() -> TableSchema {
    TableSchema {
        table_name: "bench".into(),
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

fn make_row(id: i64, value: &str) -> Row {
    Row::new(vec![
        Some(FieldValue::Int64(id)),
        Some(FieldValue::Bytes(Bytes::from(value.to_string()))),
    ])
}

fn options(tmp: &TempDir, dv_on: bool) -> OpenOptions {
    OpenOptions::new(schema())
        .wal_dir(tmp.path().join("wal"))
        .catalog_uri(tmp.path().to_string_lossy().to_string())
        .memtable_size_mb(64) // big enough that put_latency bench never flushes
        .gc_grace_period_secs(0)
        .enable_flush_dv_emission(dv_on)
}

/// Bench `db.put()` per-row latency. Pre-load a prior L1+ file so
/// flush-time DV emission would have something to resolve against;
/// the bench itself only measures `put`. The memtable threshold is
/// 64 MiB so puts never trigger a flush mid-bench.
fn bench_put_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("put_latency");
    for &dv_on in &[false, true] {
        let label = if dv_on { "dv_on" } else { "dv_off" };
        group.bench_function(label, |b| {
            let rt = Runtime::new().unwrap();
            let tmp = TempDir::new().unwrap();
            let db = rt.block_on(async {
                let db = MeruDB::open(options(&tmp, dv_on)).await.unwrap();
                // Pre-populate one L1 file so flush-time DV emission
                // has prior versions to resolve against (worst case
                // for the dv_on series — the resolver actually has
                // work to do on the next flush).
                for i in 0..1_000i64 {
                    db.put(make_row(i, "init")).await.unwrap();
                }
                db.flush().await.unwrap();
                db.compact().await.unwrap();
                Arc::new(db)
            });

            // Bench: each iteration does one put. We use a counter
            // so each iter writes a distinct id (otherwise the
            // memtable's same-key dedup makes the bench measure
            // almost nothing).
            let counter = std::sync::atomic::AtomicI64::new(10_000);
            b.iter(|| {
                let id = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                rt.block_on(async {
                    db.put(make_row(id, "x")).await.unwrap();
                });
            });

            // Cleanup: avoid background flush spinning forever.
            rt.block_on(async {
                let _ = Arc::try_unwrap(db).map_err(|_| ()).unwrap().close().await;
            });
        });
    }
    group.finish();
}

/// Bench flush wall time: pre-populate an L1 file, upsert N rows,
/// flush. Measures the resolve + commit chain. dv_on vs dv_off.
fn bench_flush_overhead(c: &mut Criterion) {
    const N: i64 = 1_000;
    let mut group = c.benchmark_group("flush_overhead");
    group.throughput(Throughput::Elements(N as u64));
    for &dv_on in &[false, true] {
        let label = if dv_on { "dv_on" } else { "dv_off" };
        group.bench_with_input(BenchmarkId::from_parameter(label), &dv_on, |b, &dv_on| {
            b.iter_custom(|iters| {
                let rt = Runtime::new().unwrap();
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let tmp = TempDir::new().unwrap();
                    let db = rt.block_on(async {
                        let db = MeruDB::open(options(&tmp, dv_on)).await.unwrap();
                        for i in 0..N {
                            db.put(make_row(i, "v0")).await.unwrap();
                        }
                        db.flush().await.unwrap();
                        db.compact().await.unwrap();
                        // Upsert all N — these are the rows whose
                        // flush we measure.
                        for i in 0..N {
                            db.put(make_row(i, "v1")).await.unwrap();
                        }
                        db
                    });
                    let t0 = std::time::Instant::now();
                    rt.block_on(async {
                        db.flush().await.unwrap();
                    });
                    total += t0.elapsed();
                    rt.block_on(async {
                        db.close().await.unwrap();
                    });
                }
                total
            });
        });
    }
    group.finish();
}

/// Issue #91: range-scan throughput post-flush vs. clean-Parquet
/// baseline. The RFC's acceptance criterion is "within constant
/// factor of clean-Parquet"; this bench pins where we sit.
///
/// Workload:
/// - Pre-populate a snapshot with N rows, force-compact into L1.
/// - Upsert all N rows, force-flush. Now L1 carries a full DV and
///   L0 carries the new versions.
/// - Bench: full-table scan via `db.scan(None, None)`.
///
/// Comparator: same N rows materialized into a single in-memory
/// Vec<Row> via `db.scan` on a CLEAN snapshot (no upserts, no DV).
/// This is the closest apples-to-apples we get for "what does the
/// engine's scan path cost when there is nothing to reconcile."
///
/// We intentionally bench `db.scan` (the engine path) on both sides
/// so the comparison isolates the DV-handling cost — not the
/// pyarrow / DuckDB scan stack which would conflate codepaths.
fn bench_scan_throughput(c: &mut Criterion) {
    const N: i64 = 5_000;
    let mut group = c.benchmark_group("scan_throughput_post_flush");
    group.throughput(Throughput::Elements(N as u64));

    // Baseline: clean snapshot, no DV (no upsert ever happened).
    group.bench_function("clean", |b| {
        let rt = Runtime::new().unwrap();
        let tmp = TempDir::new().unwrap();
        let db = rt.block_on(async {
            let db = MeruDB::open(options(&tmp, true)).await.unwrap();
            for i in 0..N {
                db.put(make_row(i, "v")).await.unwrap();
            }
            db.flush().await.unwrap();
            db.compact().await.unwrap();
            Arc::new(db)
        });
        b.iter(|| {
            let scan = db.scan(None, None).unwrap();
            assert_eq!(scan.len(), N as usize);
        });
        rt.block_on(async {
            let _ = Arc::try_unwrap(db).map_err(|_| ()).unwrap().close().await;
        });
    });

    // Post-upsert: snapshot has L1 with full DV + L0 with new
    // versions. The scan path applies DV per file then merges.
    group.bench_function("post_upsert", |b| {
        let rt = Runtime::new().unwrap();
        let tmp = TempDir::new().unwrap();
        let db = rt.block_on(async {
            let db = MeruDB::open(options(&tmp, true)).await.unwrap();
            for i in 0..N {
                db.put(make_row(i, "v0")).await.unwrap();
            }
            db.flush().await.unwrap();
            db.compact().await.unwrap();
            for i in 0..N {
                db.put(make_row(i, "v1")).await.unwrap();
            }
            db.flush().await.unwrap();
            Arc::new(db)
        });
        b.iter(|| {
            let scan = db.scan(None, None).unwrap();
            assert_eq!(scan.len(), N as usize);
        });
        rt.block_on(async {
            let _ = Arc::try_unwrap(db).map_err(|_| ()).unwrap().close().await;
        });
    });

    group.finish();
}

/// Issue #92 measurement: scan latency DURING the upsert→flush
/// window vs. post-flush. The RFC defers a transient scan-time
/// bitmask "conditional on measurement showing the gap is meaningful."
/// This bench provides the measurement.
fn bench_window_scan(c: &mut Criterion) {
    const N: i64 = 5_000;
    let mut group = c.benchmark_group("scan_during_upsert_window");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("in_window", |b| {
        let rt = Runtime::new().unwrap();
        let tmp = TempDir::new().unwrap();
        let db = rt.block_on(async {
            let db = MeruDB::open(options(&tmp, true)).await.unwrap();
            for i in 0..N {
                db.put(make_row(i, "v0")).await.unwrap();
            }
            db.flush().await.unwrap();
            db.compact().await.unwrap();
            // Upsert WITHOUT flushing — this is the window.
            for i in 0..N {
                db.put(make_row(i, "v1")).await.unwrap();
            }
            Arc::new(db)
        });
        b.iter(|| {
            let scan = db.scan(None, None).unwrap();
            assert_eq!(scan.len(), N as usize);
        });
        rt.block_on(async {
            let _ = Arc::try_unwrap(db).map_err(|_| ()).unwrap().close().await;
        });
    });

    group.bench_function("post_flush", |b| {
        let rt = Runtime::new().unwrap();
        let tmp = TempDir::new().unwrap();
        let db = rt.block_on(async {
            let db = MeruDB::open(options(&tmp, true)).await.unwrap();
            for i in 0..N {
                db.put(make_row(i, "v0")).await.unwrap();
            }
            db.flush().await.unwrap();
            db.compact().await.unwrap();
            for i in 0..N {
                db.put(make_row(i, "v1")).await.unwrap();
            }
            db.flush().await.unwrap(); // close the window
            Arc::new(db)
        });
        b.iter(|| {
            let scan = db.scan(None, None).unwrap();
            assert_eq!(scan.len(), N as usize);
        });
        rt.block_on(async {
            let _ = Arc::try_unwrap(db).map_err(|_| ()).unwrap().close().await;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_put_latency,
    bench_flush_overhead,
    bench_scan_throughput,
    bench_window_scan
);
criterion_main!(benches);
