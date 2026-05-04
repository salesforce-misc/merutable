# merutable semantics reference

Precise statements about what the engine does and does not guarantee.
This file exists to ground ambiguous README wording against the code
that actually runs in production. When the README and this file
disagree, this file wins and the README is wrong.

## Deletion Vectors: emitted at flush, honored on read

As of 0.0.2 (RFC-0002, issue #73), every flush emits per-file
deletion vectors that DV-mark every prior version of every
upserted/deleted memtable key, across L0 + L1+ alike. The flush
job, the manifest commit, and the puffin file write are one
atomic unit (same `commit_lock`-serialized chain that compaction
already used for partial-compaction DVs).

This is the contract that gives external Iceberg readers
(DuckDB `iceberg_scan`, Spark, Trino, pyiceberg) **one row per
primary key without an MVCC dedup projection**. Removing the
flush-side emission would re-introduce the duplicate-rows-per-PK
leak `docs/EXTERNAL_READS.md` describes for pre-0.0.2 snapshots.

Implementation:
- Read path opens each file alongside its DV
  (`crates/merutable/src/engine/read_path.rs`).
- Compaction honors input DVs and rewrites without DV-marked rows
  (`crates/merutable/src/engine/compaction/job.rs:open_source_file`).
- Flush computes per-file DV deltas via range merge-intersection
  (`crates/merutable/src/engine/dv_resolve.rs`), runs under
  `commit_lock` against a pinned version snapshot, and stamps the
  Puffin blobs into the same manifest commit as the new L0 SST
  (`crates/merutable/src/engine/flush.rs`).

`db.delete(pk)` still writes a logical tombstone row through the
normal LSM write path (`OpType::Delete`); the flush emits the DV
for the prior version's position, **and** keeps the L0 tombstone
row so snapshot-pinned older readers and future compactions both
see consistent state.

External writers (Spark, pyiceberg, Trino) can also stamp v3 DVs on
merutable-owned data files; merutable honors them uniformly on the
read path and the next compaction.

`OpenOptions::enable_flush_dv_emission(false)` opts out for
workloads with no upserts where the resolve+emit cost is pure
overhead. Default ON.

## MVCC semantics seen by external readers

Post-0.0.2 the dedup projection is **no longer required** for
snapshots produced by merutable's own flushes — flush-emitted DVs
guarantee one row per PK per snapshot. The projection's only
remaining role is back-compat with snapshots written by
pre-0.0.2 merutable (or by callers running with
`enable_flush_dv_emission(false)`). See
[docs/EXTERNAL_READS.md](EXTERNAL_READS.md) for the legacy
projection; new external integrations against 0.0.2+ snapshots
should read the data files directly with each file's DV applied
and skip the projection.

## Full-rewrite invariants

Every successful compaction commit satisfies:

- Every file in the picked input set is either in the new snapshot's
  `status: deleted` marker list, or is physically absent from the
  manifest. There is no "partially live" state.
- L1+ files are non-overlapping within each level (the picker pulls
  in every overlapping L(k+1) file in full; the invariant is
  structural).
- The output file set is fsynced before the manifest rename commits.
- The row cache is invalidated at commit.

See the compaction job in `crates/merutable/src/engine/compaction/job.rs`
for the authoritative sequence.
