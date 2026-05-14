# external analytical reads contract for external engines

This document defines the correct read-time projection for external
SQL engines (DuckDB, Spark, Trino, Snowflake, Athena) consuming a
merutable table via its exported Iceberg v2 `metadata.json`. **Every
external reader MUST apply this projection.** A naive `SELECT *`
returns MVCC duplicates and tombstones as if they were valid distinct
rows — silent wrong answers.

**As of 0.0.2 (RFC-0002, issue #73), snapshots produced by merutable's
own flush emit per-file deletion vectors that DV-mark every prior
version of every upserted/deleted memtable key. External Iceberg
readers see one row per primary key without the projection below.**
The projection is still required for:

- Snapshots written by pre-0.0.2 merutable.
- Snapshots written by a 0.0.2+ writer running with
  `OpenOptions::enable_flush_dv_emission(false)`.
- Tombstone filtering (`_merutable_op = 1`), which the projection
  still handles since DVs cover prior versions but not the new
  L0 row representing the delete.

### How to tell which mode produced a snapshot

Issue #93: every flush commit stamps a snapshot summary property
`merutable.flush_dv_emission` set to `"true"` or `"false"`.
Snapshots written before 0.0.2 do not carry the property at all.

```sql
-- DuckDB: list the snapshot summary properties.
SELECT properties
FROM iceberg_snapshots('/path/to/exported/metadata.json');
```

If the property is missing or `"false"`, apply the full projection
below. If `"true"`, you can drop the `ROW_NUMBER()` deduplication
and keep only the `_merutable_op = 1` tombstone filter.

## Go through the Iceberg manifest, never glob raw Parquet

```python
# WRONG — picks up files still in GC grace window, bypasses status=deleted
duckdb.sql(f"SELECT * FROM read_parquet('{db.catalog_path()}/data/L*/*.parquet')")

# RIGHT — Iceberg-aware readers respect the manifest
db.export_iceberg("/tmp/events-iceberg")
duckdb.sql(f"SELECT * FROM iceberg_scan('/tmp/events-iceberg/metadata/v1.metadata.json')")
```

Raw `read_parquet` globs return files that:
- Compaction has already removed from the manifest but are still on
  disk during the `gc_grace_period_secs` window (default 300 s).
- Belong to a snapshot other than the one the user queried.

Iceberg-aware readers filter these out via the manifest's `status:
deleted` markers and snapshot boundaries. Raw globs do not.

## Apply the MVCC dedup projection

Every merutable-written file carries two typed hidden columns
(Issue #16):

- `_merutable_seq` — `BIGINT`, the sequence number of the row.
- `_merutable_op` — `INT`, the op-type: `1 = Put`, `0 = Delete`.

External analytics readers reference them directly:

```sql
SELECT * EXCLUDE (_merutable_ikey, _merutable_seq, _merutable_op)
FROM iceberg_table
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY <primary_key_columns>
    ORDER BY _merutable_seq DESC
) = 1
   AND _merutable_op = 1;
```

No UDF, no ikey-trailer arithmetic. The DuckDB `QUALIFY` above works
verbatim on Spark / Trino / Snowflake after the usual dialect swap.

Why this is mandatory:

1. **Cross-L0 duplicates** — two memtable flushes can both contain the
   same PK at different sequence numbers; both end up as distinct
   rows in the raw file union.
2. **Cross-level duplicates** — L0 may hold the newest version of a
   key while L1 still carries an older version from a prior compaction.
   Both are visible until the next L0→L1 merge collapses them.
3. **Tombstones** — `db.delete(pk)` writes a row with `op_type = Delete`
   and non-PK columns set to NULL or sentinels. Without the `_op =
   'Put'` filter, the tombstone surfaces as a valid row with all-NULL
   non-PK columns — indistinguishable from a user who wrote a real row
   with all-NULL non-PK columns.

Engines that recognize the sort-order metadata emitted by
`db.export_iceberg` (see [#20](https://github.com/salesforce-misc/merutable/issues/20))
apply this as a streaming "first row per partition" filter at O(N)
cost; engines that don't pay an O(N log N) sort. Either way the
projection is required — the cost difference is only in the plan.

## Legacy UDFs (pre-#16 files only)

Files written before Issue #16 landed carry only `_merutable_ikey`
and do not have the typed `_merutable_seq` / `_merutable_op`
columns. Those files are rare in practice (Issue #16 lands on
initial release), but if you encounter them the UDF-based
decoding is:

```python
import duckdb
duckdb.sql("""
CREATE OR REPLACE MACRO merutable_seq_from_ikey(ikey) AS
    (get_byte(ikey, length(ikey) - 8)::BIGINT << 48)
  | (get_byte(ikey, length(ikey) - 7)::BIGINT << 40)
  | (get_byte(ikey, length(ikey) - 6)::BIGINT << 32)
  | (get_byte(ikey, length(ikey) - 5)::BIGINT << 24)
  | (get_byte(ikey, length(ikey) - 4)::BIGINT << 16)
  | (get_byte(ikey, length(ikey) - 3)::BIGINT << 8)
  |  get_byte(ikey, length(ikey) - 2)::BIGINT;
""")
```

New files do not need this — use the typed columns directly.

## `primary_key_columns`

These are exactly the columns declared `primary_key: [i, j, ...]` in
the `TableSchema` passed to `MeruDB::open`. They're the natural
`PARTITION BY`. The Iceberg metadata exported by merutable includes
an `identifier-field-ids` entry that enumerates them; Iceberg-aware
engines can surface this to users.

## Snapshot isolation

Each `db.flush()` or `db.compact()` commit produces a new Iceberg
snapshot. `db.export_iceberg(target_dir)` writes a `v{N}.metadata.json`
and a `version-hint.text` pointing at it. External readers that want
stable, repeatable results should pin to a specific `metadata.json`
path rather than re-reading the `version-hint.text` on every query.

## Summary

- Go through the manifest.
- Apply the dedup projection.
- Engines that understand merutable's sort order get it for free (O(N));
  engines that don't pay the sort (O(N log N)).
- Typed `_seq` / `_op` columns are tracked in [#16](https://github.com/salesforce-misc/merutable/issues/16).
