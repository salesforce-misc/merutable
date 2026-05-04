# Changelog

## 0.0.2 — 2026-05-03

RFC-0002: flush-time deletion vector emission. Brings external
Iceberg readers (DuckDB `iceberg_scan`, Spark, Trino, pyiceberg)
to one-row-per-PK without an MVCC dedup projection.

### Highlights

- **Flush-time deletion vectors.** Every committed snapshot now
  DV-marks every prior version of every upserted/deleted memtable
  key, across L0 + L1+ alike, atomically with the L0 SST and the
  manifest commit. External Iceberg readers see exactly one row
  per primary key — no `ROW_NUMBER() OVER PARTITION BY pk` dedup
  required.
- **Write path unchanged.** `db.put()` does no file I/O. The
  resolve+emit work piggybacks on the flush job's existing
  memtable iteration. Measured: put latency unchanged; flush
  overhead +18% for a 1K-upsert workload (resolve + Puffin + manifest).
- **Range merge-intersection.** The resolver does a single sorted-
  merge per overlapping prior file — `O(|f_overlap| + |m_overlap|)`
  per file — not a per-key bloom+sparse-index probe. Files outside
  the memtable range are skipped without I/O.
- **Configurable.** `OpenOptions::enable_flush_dv_emission(false)`
  opts out for workloads with no upserts. Default ON.
- New public reader API: `ParquetReader::iter_user_keys_in_range(lo, hi)`
  streams `(user_key, file-global position)` without decoding the
  row payload.
- New engine module: `engine::dv_resolve::resolve_dv_for_flush` —
  pure helper unit-testable in isolation.

### Tests

- 11 unit tests for `dv_resolve` (every-level probe, multi-version,
  partial overlap, missing file, etc.).
- 8 reader unit tests for `iter_user_keys_in_range` (lazy
  construction, zero-I/O on disjoint range, multi-version yield).
- 8 integration tests in `db::tests` for the wired flush path
  (cardinality equality, tombstone DV-marking, feature-off,
  internal-read correctness, subset upserts, disjoint ranges).
- 6 stress + chaos tests in `tests/stress_flush_dv.rs` (2500-write
  upsert sweep with cardinality equality, interleaved upsert/delete
  oracle, 4-writer concurrent chaos with flush/compact/scan,
  compaction-during-flush race, 10K-cardinality per-file DV bitmap,
  close+reopen DV durability).

### References

- RFC: `docs/rfc/0002-flush-time-deletion-vectors.md`
- Issue: #73

## 0.0.1 — 2026-04-22

Initial public release of the `merutable` crate.

### Highlights

- **Embedded single-table engine** with an LSM-tree architecture. L0
  SSTables are Parquet tuned as a rowstore (8 KiB pages); L1+ SSTables
  are Parquet tuned as a columnstore (128 KiB pages). One format, two
  access patterns.
- **KV API**: `put`, `get`, `delete`, `scan`, `put_batch` with MVCC
  sequence numbering. Configurable L0 backpressure via slowdown/stop
  triggers.
- **WAL + crash recovery**: CRC-checked write-ahead log with fragmented
  records. Orphaned WAL files are detected and replayed on reopen.
- **Compaction**: multi-level, parallel per-level reservation,
  snapshot-aware tombstone dropping, bounded memory via
  `max_compaction_input_rows` / `max_compaction_bytes`.
- **Row cache**: LRU cache for point lookups, invalidated on refresh.
- **Iceberg v2 export**: `export_iceberg(target_dir)` projects the
  native manifest into a spec-clean Iceberg v2 chain (`metadata.json`
  + Avro manifest-list + Avro manifest files). DuckDB `iceberg_scan()`,
  Spark, Trino, and pyiceberg read the export directly. Export is an
  explicit step; commits themselves stay on the native manifest.
- **Deletion vectors (read)**: Puffin `deletion-vector-v1` blobs
  produced by external Iceberg writers are honored during reads and
  compaction.
- **SQL change-feed**: `merutable_changes(table, since_seq)` DataFusion
  table provider (feature `sql`, enabled by default). Supports INSERT /
  UPDATE / DELETE discrimination and seq-filter pushdown.
- **Schema evolution**: additive `add_column` with Iceberg-compatible
  field IDs, initial defaults, and read-path projection for files
  written under older schemas.
- **Mirror worker**: async background upload of flushed files to an
  object store, with lag alerting and `await_mirror(seq)` for
  deterministic sync.
- **Scale-out read-only replica**: `Replica` combines a base `MeruDB`
  with an in-memory tail fed by `InProcessLogSource`, supporting
  hot-swap rebase and log-gap recovery.
- **Metrics facade**: opt-in counters, histograms, and gauges for write
  path, cache, flush, compaction, and mirror lag. No-op when no recorder
  is registered.
- **Graceful shutdown**: `MeruDB::close()` flushes the memtable, fsyncs
  the WAL, and joins all background workers.

### Bug fixes included in this release

- #46: propagate encode errors in batch write path (was silent data loss)
- #47: propagate fsync errors and use unique tmp paths in LocalFileStore
- #48: guard tombstone drop against snapshot-pinned readers (key resurrection)
- #49: reject NaN in Float/Double primary key encoding (non-deterministic ordering)
- #54: emit Avro manifest-list and manifest files so `iceberg_scan()` works end-to-end
- #55: add `await_mirror()` for deterministic sync after flush
