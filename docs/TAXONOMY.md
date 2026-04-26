# merutable taxonomy

Issue #40. merutable's tagline is deliberate. This document records
why each word earns its place and what the product is **not**.

## Tagline

> **Embeddable single-table engine where the data is both row and
> columnar and metadata is Iceberg-compatible.**

### Embeddable

Library, not server. You link one crate and use it. No JVM, no
cluster, no daemon. SQLite-shaped positioning.

### Single-table

That is what 0.1 ships. Not a database — no multi-table, no
cross-table joins, no transaction coordinator. Stating this upfront
sets correct expectations. The `IcebergCatalog` is single-table by
construction; the manifest format and the engine internals would
have to change to support multiple tables.

### Engine

Schema awareness, MVCC, change feed, compaction — more than a KV
store. But it is a building block, not a finished product. "Engine"
communicates that it is composed into higher-level systems (an
agent's memory, a service's audit log, a time-series ingest path)
rather than shipping end-to-end on its own.

### Row and columnar

Storage layout, not a runtime mode:

- **L0 SSTables** are Parquet tuned as a rowstore. 4 MiB row groups,
  8 KiB pages, PLAIN encoding across every column, plus a
  `_merutable_value` blob alongside the typed columns for the KV
  fast path. Optimized for point lookups right after a flush.
- **L1+ SSTables** are Parquet tuned as a columnstore. 32 MiB /
  128 MiB row groups, 32 KiB / 128 KiB pages, per-column encoding
  (DELTA_BINARY_PACKED for ints, BYTE_STREAM_SPLIT for floats,
  RLE_DICTIONARY for strings, RLE for bools). Optimized for
  analytical scans.

This is a **physical storage layout choice** that happens to serve
both access patterns when paired with the right reader. It is not
"hybrid transactional-analytical processing" — that phrase implies
a query planner that dispatches OLAP queries inside the engine.
merutable has no such planner. See "Not HTAP" below.

### Iceberg-compatible

Each commit writes a native JSON manifest. On demand,
`db.export_iceberg(target_dir)` projects it into a spec-clean
Iceberg v2 chain (`metadata.json` + manifest-list + manifest Avro)
any v2-aware reader (DuckDB, Spark, Trino, Snowflake, pyiceberg)
can open.

"Compatible" is deliberately weaker than "native":

- merutable writes its own manifest format first; Iceberg export is
  a projection, not the source of truth.
- merutable is not a catalog. Hive, Glue, REST catalog integration
  is an external layer callers add on top of the exported
  `metadata.json`.
- Export is an explicit, caller-invoked step. No Iceberg metadata is
  written on every commit; the commit path stays on merutable's own
  manifest.

## What merutable is NOT

### Not HTAP

HTAP (Hybrid Transactional-Analytical Processing) describes systems
like TiDB, SingleStore, and AlloyDB that run both OLTP and OLAP
workloads inside a single engine with a unified query planner.
merutable doesn't run analytical queries. DuckDB, Spark, and Trino
do — they open the Parquet files merutable writes and execute SQL
on them. The analytical capability is **emergent from the format
choice**, not an engineered feature inside merutable. Claiming HTAP
would be claiming credit for work those external engines do on
merutable's open-format files.

### Not a database

Not multi-table, not SQL-fronted, no transaction coordinator, no
role / auth / permissions surface. The change feed exposes the LSM
via DataFusion's `TableProvider` for the 0.1 preview, but the
subject is still one table.

### Not a storage engine

Too low-level. merutable carries schema, MVCC, change feed, and
compaction — more than a B-tree or LSM over opaque bytes. "Storage
engine" undersells the semantic surface.

### Not a runtime

Vague. Doesn't tell the reader what it runs. The tagline's
"embeddable single-table engine" is concrete about shape and scope;
"runtime" is a marketing word, not a contract.
