# merutable

[![Crates.io](https://img.shields.io/crates/v/merutable.svg)](https://crates.io/crates/merutable)
[![docs.rs](https://docs.rs/merutable/badge.svg)](https://docs.rs/merutable)
[![CI](https://github.com/merutable/merutable/actions/workflows/ci.yml/badge.svg)](https://github.com/merutable/merutable/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-stable-blue.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**An embeddable single-table engine where the data is both row and columnar and metadata is Iceberg-compatible.** One logical table backed by an LSM-tree. L0 SSTables are Parquet tuned as a rowstore (8 KiB pages, PLAIN encoding — optimized for point lookups). L1+ SSTables are Parquet tuned as a columnstore (128 KiB pages, per-column encoding — optimized for scans). Deletion vectors are Apache Iceberg v3 `deletion-vector-v1` Puffin blobs, and each commit writes a native JSON manifest that is a **strict superset of Apache Iceberg v2 `TableMetadata`** — losslessly projectable onto a spec-compliant Iceberg table via [`merutable::iceberg::translate`](crates/merutable/src/iceberg/translate.rs) and exportable on demand with `db.export_iceberg(target_dir)`. DuckDB, Spark, Trino, and pyiceberg read the exported view; the Parquet files themselves are the single source of truth — no ETL, no format conversion. **Analytical query execution happens in those external engines, not inside merutable** — see [docs/EXTERNAL_READS.md](docs/EXTERNAL_READS.md) for the reader contract.

Named after the [Meru Parvatha](https://en.wikipedia.org/wiki/Mount_Meru) from Indian mythology.

## Why merutable

- **One table, two workloads**: Write rows through a KV API, query them with SQL. Same Parquet files, zero ETL.
- **Open format, no lock-in**: Data is Parquet. Metadata is Iceberg. DuckDB, Spark, Trino, Snowflake read it natively via the exported Iceberg manifest — apply the MVCC dedup projection ([docs/EXTERNAL_READS.md](docs/EXTERNAL_READS.md)) once and any v2-aware reader sees consistent results.
- **Embed, don't deploy**: Link one crate. No server, no cluster, no JVM.

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="merutable architecture" width="900"/>
</p>

`IcebergCatalog` manages the single table's native manifest chain (`metadata/v{N}.metadata.json` + `version-hint.text`). The manifest is a superset of Iceberg v2 `TableMetadata` — every field Iceberg needs (`table_uuid`, `last_updated_ms`, `parent_snapshot_id`, `sequence_number`, `schemas[]`) is already stored, so projection is a pure function with no data loss. Catalog integration (Hive, Glue, REST, etc.) is an external layer on top — merutable provides the table artifacts, not the catalog service.

**Write path**: Under WAL lock: sequence allocate → WAL append → memtable insert → advance `visible_seq`. Writes back off when the L0 file count or immutable-memtable queue crosses configured thresholds (graduated slowdown → hard stop). Flush when threshold crossed. Each flush writes a new Parquet SST (fsynced), then commits a new `v{N}.metadata.json` and installs a new `Version` via `ArcSwap`. `MeruDB::open` spawns configurable background workers (`flush_parallelism`, `compaction_parallelism`) so flushes and compactions run continuously without needing caller intervention.

**Read path**: Memtable (active + immutable queue) → L0 files (bloom → `KvSparseIndex` page skip → scan) → L1..LN (bloom → `KvSparseIndex` → binary search).

**Compaction**: Leveled compaction, always full-rewrite — every selected input is fully read, merged, and removed from the manifest. Compactions on disjoint level sets run in parallel (L0→L1 and L2→L3 concurrently) via per-level reservation; the catalog commit phase is serialized by a separate brief lock. L0 is prioritized above the slowdown trigger so deep compactions can't starve flush drainage. Each job's input is capped by `max_compaction_bytes` to bound per-worker memory. Output SST is fsynced before manifest commit. Fully compacted files are removed from the manifest and queued for deferred deletion; GC waits on BOTH a time-based grace (for external analytical readers like DuckDB / Spark) AND version-pin refcounts (for internal `get`/`scan` holding an old `Version`) — a long integrity scan on tens of GB will never see a file it needs disappear mid-read. Row cache is cleared on every compaction commit. Snapshot-aware version dropping preserves MVCC versions needed by active readers.

**Deletion Vectors (read-side live, write-side dormant)**: merutable's format reads and applies any Puffin `deletion-vector-v1` blob it finds on a source file (Iceberg v3 compatibility — an external writer can stamp a DV on data merutable owns, and merutable will honor it on the next compaction merge). merutable itself does **not** produce DVs today, because every compaction is a full rewrite — there is no residual source file to stamp. The write-side API (`SnapshotTransaction::add_dv` + Puffin v3 encoder + post-union cardinality validation) is implemented and tested; it waits for a partial-compaction caller ([RFC #19](https://github.com/merutable/merutable/issues/19)). This is intentional simplicity, not a missing feature — see [docs/SEMANTICS.md](docs/SEMANTICS.md).

**Iceberg translation**: Call `db.export_iceberg(target_dir)` at any time to emit a spec-compliant Iceberg v2 `metadata.json` under `target_dir/metadata/v{N}.metadata.json` plus a `version-hint.text`. Spec compliance is pinned by a CI test that round-trips the emitted JSON through the `iceberg-rs` `TableMetadata` deserializer. Manifest-list / manifest Avro file emission is tracked as follow-on work; for inspection, catalog registration, and schema audit the `metadata.json` alone is sufficient.

## Crate map

Issue #38 collapsed the workspace into a single published crate.
Internal modules under `crates/merutable/src/` retain the same
responsibility split:

| Module | Responsibility |
|---|---|
| `merutable::types` | `InternalKey` encoding, `TableSchema`, `FieldValue`, `SeqNum`, `OpType`, `MeruError` |
| `merutable::wal` | 32 KiB block format WAL with CRC32, recovery, rotation |
| `merutable::memtable` | `crossbeam` skip-list memtable, `bumpalo` arena, rotation, flow control |
| `merutable::parquet` | Parquet SSTable writer/reader, `FastLocalBloom`, `KvSparseIndex`, footer KV metadata |
| `merutable::iceberg` | Native JSON manifest + `VersionSet` (ArcSwap) + `DeletionVector` (Puffin v3 `deletion-vector-v1`) + `translate` module projecting snapshots onto Apache Iceberg v2 `TableMetadata`. Not a catalog — catalog integration (Hive, Glue, REST) is external. |
| `merutable::store` | Pluggable object store: local FS, S3, LRU disk cache |
| `merutable::engine` | `FlushJob`, `CompactionJob`, `MergingIterator`, `RowCache`, read/write paths |
| `merutable::sql` | Change-feed cursor + DataFusion `TableProvider` (feature `sql`, on by default) |
| `merutable::replica` | Scale-out RO replica with hot-swap rebase (feature `replica`, depends on `sql`) |
| `merutable` (root) | Public embedding API: `MeruDB`, `OpenOptions`, `ScanIterator`, `MirrorConfig` |

The PyO3 bindings live in `crates/merutable-python/` (structurally
separate because Python extensions must be a `cdylib`).

## Storage tuning

The LSM tree uses level-aware Parquet tuning to serve both OLTP and OLAP workloads:

| Level | Row group | Page size | Encoding | Tuning biased for |
|-------|-----------|-----------|----------|-------------------|
| L0 | 4 MiB | 8 KiB | PLAIN (all columns) | Rowstore — point lookups, memtable flush |
| L1 | 32 MiB | 32 KiB | Per-column (see below) | Warm — transitional |
| L2+ | 128 MiB | 128 KiB | Per-column (see below) | Columnstore — analytics scans |

**Per-column encoding at L1+:**
- `_merutable_ikey` (lookup key): PLAIN — zero-overhead decode for point lookups
- `Int32`/`Int64`: DELTA_BINARY_PACKED — optimal for sorted integer columns
- `Float`/`Double`: BYTE_STREAM_SPLIT — IEEE 754 byte-transposition
- `ByteArray` (strings): RLE_DICTIONARY — high compression for categorical data
- `Boolean`: RLE

L0 files carry both `_merutable_ikey` + `_merutable_value` (postcard blob for KV fast-path) and typed columns. L1+ files drop the blob and store only `_merutable_ikey` + typed columns — the analytical format external engines read.

## Quick start

```rust
use merutable::{MeruDB, OpenOptions};
use merutable::schema::{TableSchema, ColumnDef, ColumnType};
use merutable::value::FieldValue;

#[tokio::main]
async fn main() {
    let schema = TableSchema {
        table_name: "events".into(),
        columns: vec![
            ColumnDef { name: "id".into(), col_type: ColumnType::Int64, nullable: false },
            ColumnDef { name: "payload".into(), col_type: ColumnType::ByteArray, nullable: true },
        ],
        primary_key: vec![0],
    };

    let db = MeruDB::open(OpenOptions::new(schema)).await.unwrap();
    db.put(&[FieldValue::Int64(1)], &[FieldValue::Int64(1), FieldValue::Null]).await.unwrap();
    let row = db.get(&[FieldValue::Int64(1)]).unwrap();
    println!("{row:?}");

    // Graceful shutdown — flushes memtable to Parquet, fsyncs, and
    // rejects further writes. Reads remain available until drop.
    db.close().await.unwrap();
}
```

## Interactive notebook

The [`lab/lab_merutable.ipynb`](lab/lab_merutable.ipynb) notebook is a live, runnable showcase — open it on GitHub to see pre-rendered outputs, or run it locally for the full interactive experience:

```bash
cd lab && bash setup.sh
```

The notebook covers: write/flush/inspect, compaction with Deletion Vectors, **external analytical reads from DuckDB** (SQL queries on merutable's Parquet files — zero ETL), acceleration structures (bloom filter + KvSparseIndex), and write/read performance benchmarks.

## Python bindings

merutable ships a PyO3 crate (`merutable-python`) that exposes the full API to Python:

```python
from merutable import MeruDB

db = MeruDB("/tmp/mydb", "events", [
    ("id",     "int64",  False),
    ("name",   "string", True),
    ("score",  "double", True),
    ("active", "bool",   True),
])

db.put({"id": 1, "name": "alice", "score": 95.5, "active": True})
row = db.get(1)         # {'id': 1, 'name': 'alice', 'score': 95.5, 'active': True}

# Batch writes — single WAL sync per batch, 100-1000× faster than individual puts
db.put_batch([
    {"id": 2, "name": "bob",   "score": 88.0, "active": True},
    {"id": 3, "name": "carol", "score": 92.1, "active": False},
])

db.flush()              # → L0 Parquet file + new v{N}.metadata.json
db.compact()            # → L1 columnstore + Deletion Vectors (Puffin v3)
print(db.stats())       # includes cache hit/miss counters

# External analytical reads: register merutable's Iceberg metadata with DuckDB.
# Always go through the Iceberg manifest — raw-glob `read_parquet` picks
# up files still in the GC grace window and skips the MVCC dedup
# projection, producing wrong answers on any non-trivial workload.
# See docs/EXTERNAL_READS.md for the canonical projection.
import duckdb
db.export_iceberg("/tmp/events-iceberg")  # metadata.json + version-hint
duckdb.sql("INSTALL iceberg; LOAD iceberg;")
duckdb.sql(f"""
    SELECT * EXCLUDE (_merutable_ikey, _merutable_seq, _merutable_op)
    FROM iceberg_scan('/tmp/events-iceberg/metadata/v1.metadata.json')
    -- MVCC dedup: pick newest seq per PK, drop tombstones.
    -- _merutable_seq (BIGINT) and _merutable_op (INT, 1=Put/0=Delete)
    -- are typed hidden columns every merutable-written file carries.
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY id
        ORDER BY _merutable_seq DESC
    ) = 1
       AND _merutable_op = 1;
""").show()

# Read-only replica — opens same catalog, no WAL, no writes
replica = MeruDB("/tmp/mydb", "events", [...], read_only=True)
replica.get(1)          # reads from Parquet files
replica.refresh()       # picks up new snapshots from the primary

# Hand the current snapshot to any Iceberg v2 reader
db.export_iceberg("/tmp/events-iceberg")   # writes metadata/v{N}.metadata.json
# Then, e.g.:
#   from pyiceberg.table import StaticTable
#   t = StaticTable.from_metadata("/tmp/events-iceberg/metadata/v1.metadata.json")
#   t.schema()              # ← full Iceberg v2 schema, table_uuid, snapshot chain

# Graceful shutdown — flush + fsync + seal
db.close()              # writes are rejected after this; reads still work
```

Build with [maturin](https://www.maturin.rs/):
```bash
cd crates/merutable-python && maturin develop --release
```

## License

Apache-2.0
