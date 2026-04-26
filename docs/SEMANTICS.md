# merutable semantics reference

Precise statements about what the engine does and does not guarantee.
This file exists to ground ambiguous README wording against the code
that actually runs in production. When the README and this file
disagree, this file wins and the README is wrong.

## Deletion Vectors: honored on read, not emitted

merutable's own deletes are tombstone rows written through the normal
LSM write path (`db.delete(pk)` → `write_internal(.., OpType::Delete)`);
compaction reconciles them. No part of merutable's commit pipeline
produces a Puffin `deletion-vector-v1` blob.

What merutable *does* do is honor DVs if it encounters them. On every
compaction merge and every read path, each input file is opened with
its associated DV (if the manifest entry has a `dv_path`) and the
marked row positions are filtered out before the merge iterator sees
them. Implementation:
`crates/merutable/src/engine/compaction/job.rs:open_source_file`,
`crates/merutable/src/engine/read_path.rs`.

This matters for Apache Iceberg v3 interop. An external writer
(Spark, pyiceberg, Trino) can stamp a DV on a merutable-owned data
file, and merutable will honor it on the next merge that touches the
file. Removing the read-side would break v3 compatibility.

The Puffin v3 `deletion-vector-v1` encoder in
`crates/merutable/src/iceberg/deletion_vector.rs` and the
`SnapshotTransaction::add_dv` API exist so that external tooling or
future library-level commit flows can construct DV-stamped commits
against a merutable catalog. The engine itself does not call them.

## MVCC semantics seen by external readers

Short version: an external reader sees the *union* of every live
Parquet file — which includes cross-level duplicates and tombstones.
An `ORDER BY seq DESC, LIMIT 1` (or `ROW_NUMBER() QUALIFY …`)
projection is required to recover the `MeruDB::get`/`scan`-equivalent
view. Full treatment is in [docs/EXTERNAL_READS.md](EXTERNAL_READS.md) once
that file lands; until then see the `IcebergCatalog::commit` and
`read_path` modules.

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
