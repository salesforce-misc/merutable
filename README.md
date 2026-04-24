<p>
  <img src="docs/assets/wordmark.svg" alt="merutable" height="36"/>
</p>

<p>
  <a href="https://github.com/merutable/merutable/actions/workflows/ci.yml"><img src="https://github.com/merutable/merutable/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/merutable"><img src="https://img.shields.io/crates/v/merutable.svg" alt="crates.io"></a>
  <a href="https://docs.rs/merutable"><img src="https://docs.rs/merutable/badge.svg" alt="docs.rs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License"></a>
</p>

<p><b>An embeddable Rust table engine. LSM writes, Parquet storage, Iceberg-compatible metadata.</b></p>

The writes go through a WAL + skip-list memtable; flushes land as
Apache Parquet based SSTables. Invoke `db.export_iceberg(path)` when you need an
Iceberg v2 view — DuckDB, Spark, Trino, Snowflake, and pyiceberg read it with
no format conversion.

```rust
use merutable::{MeruDB, OpenOptions};
use merutable::schema::{ColumnDef, ColumnType, TableSchema};
use merutable::value::{FieldValue, Row};

#[tokio::main]
async fn main() -> merutable::error::Result<()> {
    let schema = TableSchema {
        table_name: "events".into(),
        columns: vec![
            ColumnDef { name: "id".into(),      col_type: ColumnType::Int64,     nullable: false, ..Default::default() },
            ColumnDef { name: "payload".into(), col_type: ColumnType::ByteArray, nullable: true,  ..Default::default() },
        ],
        primary_key: vec![0],
        ..Default::default()
    };

    let db = MeruDB::open(OpenOptions::new(schema)).await?;

    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Bytes(b"hello"[..].into())),
    ])).await?;

    let row = db.get(&[FieldValue::Int64(1)])?;
    println!("{row:?}");

    db.close().await?;   // flush + fsync + seal; reads remain until drop
    Ok(())
}
```

## When merutable fits

Structured data thats both **write-heavy** - agent memory, session state, audit logs, feature stores, embedded
time-series - and **readable by analytical engines** without an ETL job. An LSM
gives you the fast-writes; Iceberg compatible metadata layer gives you the analytics reads.

## What's in the box

- **Durable LSM write path.** Write-ahead log with 32 KiB block framing and
  CRC32, crossbeam skip-list memtable, graduated writer backpressure on
  L0-file buildup. `visible_seq` advances only after the memtable apply, so
  readers never observe a torn write.
- **Leveled compaction.** Full-rewrite, run in parallel on disjoint level
  sets, bounded per-job memory, fsync-before-commit, version-pinned GC so
  a long scan never sees a file disappear mid-read.
- **Iceberg export on demand.** `db.export_iceberg(path)` writes a
  spec-clean Iceberg v2 chain — `metadata.json` + manifest-list Avro +
  manifest Avro — that DuckDB `iceberg_scan`, pyiceberg, Spark, Trino, and
  Athena consume as-is. You call `export_iceberg` when you want the view.
  `merutable`'s metadata layer efficiency is not bound by the Iceberg spec.
- **Change feed.** Committed operations are exposed as a change feed table
  provider with `seq > N` predicate pushdown and per-DELETE pre-image
  reconstruction.
- **Read-only replica** *(opt-in).* Base + tail replayed from the change
  feed; rebase hot-swaps behind `ArcSwap` so in-flight readers never see a
  torn state.
- **Schema evolution.** `db.add_column(ColumnDef)` — reopen accepts the
  extension, reads of pre-evolution files fill defaults, writes pad short
  rows with `write_default`.
- **Python bindings** *(via PyO3).* `crates/merutable-python/`.

## Install

```toml
[dependencies]
merutable = "0.0.1"
```

## Architecture at a glance

```
          ┌──────── your process ────────┐
writes ──▶│ WAL → memtable → flush → SST │
reads  ◀──│   memtable  ∪  L0  ∪  L1…    │
          └─────────────┬────────────────┘
                        │  Parquet files on disk
                        ▼
              db.export_iceberg(path)
                        │
                        ▼
           DuckDB / Spark / Trino / pyiceberg
```

Deeper reads:
[`docs/architecture.svg`](docs/architecture.svg) ·
[`docs/SEMANTICS.md`](docs/SEMANTICS.md) ·
[`docs/EXTERNAL_READS.md`](docs/EXTERNAL_READS.md) ·
[`docs/MIRROR.md`](docs/MIRROR.md) ·
[`docs/SCALE_OUT_REPLICA.md`](docs/SCALE_OUT_REPLICA.md) ·
[`docs/TAXONOMY.md`](docs/TAXONOMY.md) ·
[`DEVELOPER.md`](DEVELOPER.md)

## Benchmarks

[`lab/lab_merutable.ipynb`](lab/lab_merutable.ipynb) — a live, runnable
showcase comparing merutable against DuckDB head-to-head, then demonstrating
the zero-ETL federated read (fresh memtable rows inside merutable, columnar
analytical reads from DuckDB against the same on-disk Parquet).

```bash
cd lab && bash setup.sh
```

## Status

| Area              | 0.0.1                                                               |
|-------------------|---------------------------------------------------------------------|
| Storage format    | LSM tree layout optimized for both row and columnar. Iceberg v2-compatible.  |
| Durability        | fsync on SST write, fsync on WAL, fsync on manifest commit.          |
| Concurrency       | Designed for one primary writer per catalog (not yet lock-enforced); many concurrent readers via version pinning. |


Named after [Mount Meru](https://en.wikipedia.org/wiki/Mount_Meru) — the axis
around which the cosmos is ordered in Indian cosmology.
