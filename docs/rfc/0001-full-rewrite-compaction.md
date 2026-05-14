# RFC-0001: Stay full-rewrite compaction only

**Status**: Accepted
**Tracks**: [#19](https://github.com/salesforce-misc/merutable/issues/19)
**Date**: 2026-04-18

## Decision

merutable stays full-rewrite compaction only. Partial compaction
(rewrite a subset of rows, leave the rest in place with a Puffin
`deletion-vector-v1` blob stamping the moved positions) is explicitly
out of scope until field data demonstrates write-amplification pain
at deep levels that cannot be mitigated within the full-rewrite model.

This decision is revisited no sooner than the first production
deployment reports sustained L(N-1)→L(N) write-amp above 8× with a
workload pattern that partial compaction could meaningfully reduce.

## Why

### The trade is real but asymmetric

Partial compaction trades ~10× lower write amplification at deep
levels for higher read amplification on every query that touches a
partially-rewritten file, higher compaction-picker complexity, loss
of the L1+ non-overlap structural invariant, and write-path code
that we do not currently carry.

Full rewrite loses the write-amp optimization but keeps:

- `find_file_for_key` O(log N) per level via binary search on
  non-overlapping ranges. Point lookups stay predictable.
- A simple picker (whole-file selection, `max_compaction_bytes` cap).
- A simple commit (`txn.remove_file(path)` for every input; no DV
  coordination between files; no residual-file lifetime extension).
- GC that's driven solely by snapshot pins + wall-clock grace.
  Partial compaction adds a new axis — a source file stays live
  across arbitrary many merges because it accumulates DVs — and
  deciding when to collapse it requires its own scheduler.

### The motivating use case does not need partial compaction

Partial compaction's pitch was usually "produce the Puffin DVs you
claim to produce." That rationale is voided by the [Issue #17 fix](../../docs/SEMANTICS.md)
(README corrected to state the read/write asymmetry; DV write-side
preserved but dormant). The external-reader dedup projection
([Issue #16](https://github.com/salesforce-misc/merutable/issues/16),
[docs/EXTERNAL_READS.md](../EXTERNAL_READS.md)) is mandatory whether or not
partial compaction is implemented — partial compaction addresses
write-amp only, not the duplicates external readers see.

### Complexity cost is front-loaded

The work to land partial compaction is not a single PR:

1. Picker changes — per-row or per-key-range selection logic.
2. Job changes — partial read path that knows to stamp a DV rather
   than remove the source file.
3. Read-path changes — `find_file_for_key` must cope with overlapping
   files at L1+, applying DVs per file.
4. Version layer — file metadata must carry accumulated DV state,
   and version drops must coordinate with DV lifetimes.
5. GC changes — a source file with an active DV is not deletable
   even if its snapshot is unpinned; lifetimes become transitive
   through the DV chain.
6. Commit path — the per-file DV union validation (already landed
   for the read-side) extends to multi-file coordinated commits.
7. Tests — every compaction invariant test has to re-prove itself
   under the partial-compaction model.

Each is individually tractable; collectively they replace a simple,
correct mental model with a more subtle one whose edge cases take
years to fully discover. We decline the trade.

### Iceberg v3 interop is already satisfied

The read-side applies any Puffin `deletion-vector-v1` blob it finds
on a source file — external writers (Spark, pyiceberg, Trino) can
stamp DVs on merutable-owned data and merutable will honor them on
the next merge. That is the v3 interop contract. Producing DVs as
output of merutable-owned writes is not required by the spec; it's a
write-amp optimization.

## Consequences

- The DV write-side API (`SnapshotTransaction::add_dv`, Puffin v3
  encoder, post-union cardinality validation) stays in the code
  base. It is preserved for (a) the partial-compaction possibility
  if this RFC is revisited, (b) external tooling paths (admin CLI
  for row-level deletes) that might stamp DVs without going through
  a compaction.
- `docs/SEMANTICS.md` pins the read/write asymmetry as the official
  semantics.
- The README's Architecture section states compaction is full-
  rewrite, no caveat or "future" language.
- [Issue #20](https://github.com/salesforce-misc/merutable/issues/20) (sort
  order + per-column stats into Iceberg) is independent and proceeds.
- [Issue #16](https://github.com/salesforce-misc/merutable/issues/16)
  (external-reader dedup projection + typed `_seq`/`_op`) is
  independent and proceeds.
- Chaos testing continues to exercise the full-rewrite path; no new
  partial-compaction fault models are added.

## Revisit criteria

Reopen this RFC only if:

1. A production deployment reports sustained write-amplification >
   8× at the deepest level, with a workload (large L(N) files,
   small trickle of L(N-1) updates) that partial compaction could
   mechanically reduce.
2. OR: the Iceberg v3 ecosystem evolves a convention for
   engine-native DV emission that merutable wants to match for
   compatibility (e.g. row-level deletes exposed to external SQL
   tooling as a first-class feature).

Neither is true today.
