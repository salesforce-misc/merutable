# RFC-0002: Flush-time deletion vector emission

**Issue:** #73
**Status:** Draft, awaiting sign-off

## Problem

Two related problems, one fix:

1. **External Iceberg readers see duplicate PKs.** A reader
   (DuckDB, Spark, Trino, pyiceberg) scans every data file in the
   snapshot. Without per-file deletion vectors, an upserted key's
   prior versions all surface, and the reader has to apply an MVCC
   dedup projection (`ROW_NUMBER() OVER (PARTITION BY pk ORDER BY
   seq DESC) = 1` or equivalent — see `docs/EXTERNAL_READS.md`).
   That projection is a leaky abstraction the Iceberg v3 spec was
   designed to eliminate.
2. **Range scans force cross-level LSM merge** to reconcile
   versions, serializing the read pipeline and defeating columnar
   SIMD at L1+. Industry measurement on similar designs (Hyper,
   Photon, DuckDB lineage) puts the penalty at up to ~100x vs. a
   clean-Parquet baseline.

Both problems share one fix: positional deletion vectors — the same
primitive Iceberg v3, Delta, and Hudi converged on — applied to
**every** prior version of every upserted key, regardless of level.
The question is **where in the write pipeline** to compute them.

## Non-negotiables

The write path is the LSM's reason to exist. It stays cheap.

1. **WAL record format unchanged.** No `(file, position)` references
   in WAL records. Recovery must work against any post-crash
   physical state.
2. **Memtable interface unchanged.** Skip-list of `(key → row | tombstone)`.
   No file/position state coupled to `VersionSet`.
3. **`db.put()` does no file I/O.** No bloom check. No sparse-index
   probe. No prior-version page read.
4. **`db.put()` p99 latency does not regress vs. main.** Asserted
   by bench with a 10% slack bound.

These are load-bearing. Every design decision below is downstream of
these four constraints.

## Decision: DV computation runs at flush, not at upsert

The flush job already iterates every memtable key to produce the L0
SST. We piggyback on that pass. The added cost is paid by the flush
job (already async, already amortized over a memtable's worth of
rows), not by `db.put()`.

### Flush job, post-RFC

```
1. Drain immutable memtable → entries.
2. Write L0 Parquet SST + fsync.
3. Acquire commit_lock.                       ← new ordering matters
4. Pin current version snapshot.              ← stable file set
5. For each entry:
       resolve_dv_target(user_key, version) → Option<(file, position)>
       aggregate (file → DeletionVector) by union.
6. Build SnapshotTransaction { adds: [L0 SST], dvs: by-file }.
7. catalog.commit(txn) — same atomic chain we already exercise for
   compaction-emitted DVs.
8. Install new version. GC WAL. Drop memtable. Notify writers.
```

Steps 4–6 run inside `commit_lock`. Compaction cannot commit between
step 4's pin and step 7's commit, so the `(file, position)`
references stay valid through the commit.

### Algorithm: range merge-intersection, not per-key probe

Naive per-key probing — for each of M memtable keys, bloom-check
and sparse-index-search every prior file — is `O(M × F)` work and
wastes the structure both sides already have. Both the memtable
**and** every parquet file are **sorted by user_key**. The right
algorithm is a single sorted-merge per file:

```
for each prior file F in version (L0 + L1+):
    if not ranges_overlap(F.key_range, memtable.key_range):
        continue                                       # cheap skip
    f_iter = F.iter_keys_in_range(memtable.min, memtable.max)
    m_iter = memtable.user_keys_in_range(F.key_min, F.key_max)
    while m_iter and f_iter:
        cmp = m_iter.peek().cmp(f_iter.peek().user_key)
        match cmp:
            Equal:   dv[F.path].insert(f_iter.peek().position)
                     advance f_iter            # may have more versions
            Less:    advance m_iter
            Greater: advance f_iter
```

Per file: `O(|f_overlap_pages| + |m_overlap_keys|)` — both linear
in the overlap, no bloom check, no per-key binary search.

The **range hint** to the reader (`memtable.min`, `memtable.max`)
lets the file skip pages outside the overlap window using its
existing `KvSparseIndex`. A file whose entire range is disjoint
from the memtable does zero page I/O — bailed out by the
`ranges_overlap` check.

A user_key with N versions in F (multi-version row) yields N
positions to DV-mark — the file iterator yields every InternalKey
occurrence (user_key ASC, seq DESC), and we mark every position
that user_key appears at. Correct without special handling.

### Phase 1 deliverable: streaming key+position API on the reader

```rust
impl ParquetReader {
    /// Yield `(user_key_bytes, file_global_row_position)` in sorted
    /// `(user_key ASC, seq DESC)` order for every InternalKey whose
    /// user_key falls within `[lo, hi]` (inclusive). Uses the
    /// sparse index to skip pages outside the range. Multi-version
    /// rows yield one tuple per version. Does NOT decode the row.
    pub fn iter_user_keys_in_range<'a>(
        &'a self,
        lo: &[u8],
        hi: &[u8],
    ) -> Result<impl Iterator<Item = Result<(Vec<u8>, u64)>> + 'a>;
}
```

This is the only new reader API. It composes naturally with the
merge-intersection above. There is no per-key `position_of` —
that approach was discarded.

Why every level, not just L1+:

> **Deletion vectors are how external Iceberg readers see one row
> per primary key without an MVCC dedup projection.** An external
> reader (DuckDB `iceberg_scan`, Spark, Trino, pyiceberg) opens
> every data file in the snapshot, applies each file's DV bitmap,
> and unions the surviving rows. If the prior L0 copy of an
> upserted key is not DV-marked, the external reader sees
> duplicates. The "scan parallelism for L1+" framing is a real
> benefit but it is not the load-bearing reason — **PK uniqueness
> for external readers is.** That guarantee requires DV-marking
> every prior version regardless of level.

L0 file count is bounded by the L0 backpressure thresholds
(`l0_slowdown_trigger`, `l0_stop_trigger`); per-level for L1+ a key
lives in at most one file (non-overlapping ranges per level). The
range merge-intersection (next subsection) traverses each
overlapping prior file once, not once per memtable key.

### Multiple prior versions of one key

A key can have versions in multiple L0 files (sequence of flushes
before any L0→L1 compaction) or across L0 and L1+ during transient
states between compactions. The probe finds **all** of them and
DV-marks each. Aggregation is per-file: the resolve pass produces a
`HashMap<String, RoaringBitmap>` keyed by parquet file path; each
position is unioned into the file's bitmap before being attached to
the transaction.

Idempotency falls out for free: if a probe re-discovers a position
already in an existing DV blob (e.g. a fully-DV'd L1 file the bloom
filter can't tell us about), the `catalog.commit()` path's existing
DV merge (`union_with`) is a no-op. The optimization to short-circuit
that case after the bloom hit but before the page read is a perf
follow-on, not a correctness concern.

### What about the upsert→flush window?

After flush commits, **external Iceberg readers see uniqueness
immediately** — every prior version is DV-marked at every level. No
MVCC projection required for the post-flush snapshot.

Between `db.put()` and the next flush, the new value lives only in
the memtable. An external reader sees the un-superseded prior
version (no DV yet) and is unaware of the new value (memtable not
exposed externally). This is the existing visibility model: external
readers see the most recent committed snapshot. The RFC doesn't
change that contract; it eliminates the duplicate-row problem within
each committed snapshot.

For merutable's **internal** range scan during the window, the
MergingIterator still reconciles memtable + L1+ rows for the keys
in flight. Same bound as before: L0/memtable size is configurable;
new-key inserts (the common case) don't overlap. A transient
in-memory scan-time bitmask is a follow-on if measurement justifies
it. **Not in this RFC's scope.**

## Invariants this preserves

I1. WAL records carry only logical operations.
I2. Memtable interface is logical-row-only.
I3. `db.put()` performs no file I/O.
I4. The manifest commit is the **linearization point** for "L0 SST
    + DV updates + new manifest" as one atomic unit (the existing
    `catalog.commit()` chain already enforces fsync-before-rename for
    puffin blobs and SST file alike).
I5. Reads against any committed version honor exactly that version's
    DVs. Snapshot-pinned readers on older versions see no DV that
    wasn't part of their pinned manifest.
I6. WAL recovery can replay against any post-crash physical state.
    No (file, position) references to validate post-replay.

## Anti-fixes guarded

| # | Failure mode                                                                 | Guard                                                                                                                                  |
|---|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| A1 | DV references a file compaction rewrote                                       | Resolve under `commit_lock` against pinned version. Compaction can't commit during resolve→commit.                                     |
| A2 | Two flushes race on the same file's DV                                        | `flush_mutex` serializes flushes (already today). Each flush's DV deltas merge with prior persisted DV inside `catalog.commit()` (already today, used by partial compaction). |
| A3 | WAL format leaks physical state                                              | WAL changes prohibited. Tests in `wal/` unchanged. Reviewer-enforced.                                                                  |
| A4 | Memtable holds (file, position) state                                         | Memtable interface unchanged. Tests in `memtable/` unchanged. Reviewer-enforced.                                                       |
| A5 | Probe latency unbounded under huge dataset                                    | Probe is O(memtable_keys × L1+_file_count) for bloom; O(memtable_keys × hits) for page reads. Bounded by configurable flush threshold + level fan-out. |
| A6 | Crash mid-DV-write before manifest commit                                    | Existing `catalog.commit()` cleanup path (IMP-06) deletes orphaned puffin blobs on rollback. Already exercised by partial-compaction path. |
| A7 | DV resolution decodes the row (wasted work)                                  | New `iter_user_keys_in_range` streams `(user_key, position)` only — no row decode. Sparse index restricts reads to overlapping pages.   |
| A8 | Tombstone in memtable resolves a stale prior position                         | Tombstones go through the same resolve path. Both L0 and L1+ prior versions of the deleted key get DV-marked. The new L0 SST still carries the tombstone row for snapshot-pinned readers and for compaction-time tombstone reaping under snapshot-aware drop. |
| A9 | Multiple prior versions of one key across L0/L1+                              | Probe returns all positions across all levels. Per-file aggregation in a `HashMap<path, RoaringBitmap>` ensures each prior copy is DV-marked. |
| A10 | Existing fully-DV'd L1+ file gets re-probed every flush (wasted work)         | Correctness-safe (DV union is idempotent). Perf follow-on: skip the page-read step when the file's existing DV already covers the bloom-hit positions. Not in this RFC's scope. |

## Cost model

For a flush of M memtable keys against a version with F0 L0 files
plus F_n total L1+ files spanning N levels:

- **Range-overlap filter:** O(F0 + F_n) cheap range-pair comparisons.
  Sub-millisecond at any reasonable file count.
- **Per-overlapping-file merge:** O(|f_overlap_pages| + |m_overlap_keys|).
  For a memtable that fully overlaps an L1+ file: a streaming page
  scan + memtable iterator walk. Page reads are sequential, so the
  parquet metadata cache + OS page cache amortize them.
- **No per-key bloom probe and no per-key sparse-index search.**
  The merge-intersection eliminates both — they were the dominant
  cost in the per-key-probe alternative.
- **Total flush latency increase:** dominated by the additional
  page reads of overlapping prior files. For a 64 MiB memtable
  spanning 1 L0 file (64 MiB) and 1 L1 file (64 MiB) with full
  overlap: ~128 MiB additional sequential read ≈ 100–500ms on SSD.
  Cache-warm: tens of ms.
- **No write-path latency change.** The `db.put()` cost remains
  WAL fsync + memtable insert, unchanged.

L0 file count is bounded by `l0_slowdown_trigger` /
`l0_stop_trigger` (defaults: 8 / 12). N is bounded by total dataset
size — `log_b(dataset_bytes / l0_size)` for level multiplier b.
Both fan-outs are configurable; flush probe cost is bounded by
configuration, not by adversarial inputs.

If `enable_flush_dv_emission` is set false (config flag, defaults
true), the entire resolve phase is skipped — for users who want
the pre-RFC behavior or who do not have upsert workloads.

## Tests (with tight bounds — 2x-regression-tight per the engineering bar)

| Test                                                  | Bound                                                                                                                          |
|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `put_latency_no_regression`                           | p99 latency post-RFC ≤ 1.10 × pre-RFC p99. Both measured under identical workload. Fails if write path was perturbed.          |
| `flush_emits_dv_for_l1_priors`                        | Write N rows, force-flush, force-compact (so all are in L1+). Upsert all N. Force-flush. Asserts: L1 file's DV cardinality == N. **Equality**. |
| `flush_emits_dv_for_l0_priors`                        | Write N rows, force-flush (rows in L0). Upsert all N. Force-flush. Asserts: prior L0 file's DV cardinality == N. **Equality**. This is the test the original RFC draft missed. |
| `flush_emits_dv_for_multi_level_priors`               | Write N → flush → compact (in L1). Upsert N → flush (in L0_b). Upsert N again → flush. Asserts: L0_b's DV cardinality == N AND L1 file's DV cardinality == N. Tight. |
| `external_iceberg_read_returns_unique_pks_post_flush` | Use the existing `iceberg-rs` round-trip path: write N → flush → compact → upsert N → flush → `db.export_iceberg(target)`. Have an external reader (or the Rust-side `iceberg-rs` reader) scan the table. Asserts: exactly N rows surfaced, every PK appears exactly once, every row is the upserted value. **No MVCC dedup projection applied.** This is the load-bearing acceptance test. |
| `flush_emits_no_dv_for_new_keys`                      | Write N entirely-new keys, force-flush. Asserts: zero DV blobs in committed snapshot. Tight. |
| `flush_emits_dv_for_tombstones`                       | Write N rows → flush → compact. Delete all N. Force-flush. Asserts: L1 file's DV cardinality == N. Tight. |
| `flush_emits_dv_for_l0_tombstones`                    | Write N rows → flush (in L0). Delete all N. Force-flush. Asserts: prior L0 file's DV cardinality == N. Tight. |
| `flush_resolves_against_pinned_version_under_compaction` | Concurrent compaction commits between flush start and flush commit. Asserts: flush's DV references the post-compaction file set, not stale (pre-compaction) paths. |
| `scan_post_flush_no_duplicates_or_old_versions`       | Write N rows → flush → compact. Upsert all N. Flush. Range-scan. Asserts: exactly N rows, every value is the upserted value. Equality. |
| `wal_format_unchanged`                                | Snapshot-test the WAL record bytes pre- and post-RFC. Asserts: byte-identical.                                                |
| `memtable_interface_unchanged`                        | Public surface of `MemtableManager` and `Memtable` unchanged. Compile-time test via doc-test or pub-API snapshot.            |
| `range_scan_throughput_post_flush`                    | After flush emits DVs, range-scan throughput ≥ 0.5 × clean-Parquet baseline. (The "small constant factor" target from the issue's acceptance criteria; ≥0.5x is "within 2x" — explicit and measurable.) |
| `concurrent_flush_compaction_dv_consistency`          | Long-running stress: continuous puts, periodic flush + compaction, periodic scans. Asserts: monotonic scan results match an in-process oracle (write-side state). 60s run. |

## Phases

Each phase is independently reviewable and lands as its own PR.

**Phase 1 — `ParquetReader::iter_user_keys_in_range`** (no behavior change yet)

- New streaming method on the existing reader.
- Walks the sparse index, streams `(user_key_bytes, position)` pairs
  in `(user_key ASC, seq DESC)` order across only the pages that
  overlap `[lo, hi]`.
- Multi-version rows yield one tuple per occurrence.
- Unit tests:
  - Empty range → empty iterator.
  - Range disjoint from file range → empty iterator, **zero page reads**
    (asserted via a counting wrapper).
  - Range fully containing file → every (user_key, position) yielded
    exactly once.
  - Multi-version file: every version yields its own position.
  - Iterator is **lazy**: opening + dropping the iterator without
    pulling does not read any data pages.

**Phase 2 — `engine::dv_resolve` module** (helper, no integration yet)

- `resolve_dv_for_flush(memtable_keys: &[InternalKey], version: &Version, base_path: &Path, schema: &Arc<TableSchema>) -> Result<HashMap<String, RoaringBitmap>>`.
- Pure function over `&Version`. Unit-testable with synthetic versions.
- For each prior file: range-overlap filter, then sorted merge against
  memtable keys, accumulating positions into the per-file bitmap.
- Tests:
  - Memtable keys disjoint from every file range → returns empty map.
  - Memtable keys fully overlap one L1 file → bitmap matches expected
    positions exactly (cardinality equality).
  - Multi-version L0 file: all versions of an upserted key DV-marked.
  - Mix of L0 + L1 priors for the same key: both files' bitmaps populated.
  - Tombstone in memtable: same resolution as upsert (the resolve API
    operates on user_keys, not op_types).

**Phase 3 — wire into `flush.rs` behind a feature flag**

- Default-on flag `EngineConfig::enable_flush_dv_emission = true`.
- Move pre-commit work inside `commit_lock`.
- Resolve every memtable entry; aggregate by file; `txn.add_dv`.
- Integration tests from the table above.

**Phase 4 — bench harness** (landed)

- Criterion bench at `crates/merutable/benches/flush_dv_emission.rs`:
  - `put_latency/{dv_off, dv_on}` — single-row `put` latency on a
    pre-populated DB (1K rows in L1 so the resolver has work on the
    next flush); memtable size 64 MiB so puts never trigger a flush
    mid-bench.
  - `flush_overhead/{dv_off, dv_on}` — flush wall time of 1K upserts
    against an L1 file holding 1K priors.
- Range-scan-throughput bench landed in issue #91; numbers below.

### Measured (macOS, criterion --quick)

| Metric | Value | Comparator | Δ |
|---|---|---|---|
| `put` mean (dv_off) | 3.88 ms | — | — |
| `put` mean (dv_on) | 3.68 ms | dv_off | within noise |
| 1K-upsert flush (dv_off) | 26.9 ms | — | — |
| 1K-upsert flush (dv_on) | 31.8 ms | dv_off | +4.9 ms (+18%) |
| 5K range-scan (clean) | 4.27 ms / 1.17 M elem/s | — | — |
| 5K range-scan (post-upsert) | 6.75 ms / 0.74 M elem/s | clean | **0.63× throughput (1.58× slower)** — within the 2× bound |

**The load-bearing contract holds:** dv_on ≈ dv_off on `put`. The
+18% on flush is the amortized resolve+puffin-write cost paid at the
flush boundary, not at the upsert. Per-row marginal: ~5 μs.

**Range-scan post-flush stays within constant factor of clean.** The
1.58× slowdown is the cost of opening DV blobs alongside data files
and applying the bitmap during the merge — bounded, predictable,
inside the 2× bar.

**Phase 5 — docs**

- `docs/SEMANTICS.md` — flush-time DV section. Update DV section to
  describe both compaction-emitted and flush-emitted paths.
- `CHANGELOG.md` — entry under next release.
- `docs/rfc/0002-…md` — mark Status: Implemented, append measurements.

## Out of scope (explicit non-goals)

- Synchronous DV emission on the upsert path. **Rejected.** Couples
  WAL + memtable to physical layout; puts file I/O on `db.put()`.
- Transient scan-time bitmask for the upsert→flush window. Follow-on,
  conditional on measurement.
- Equality / key-based deletes at read time. Rejected — incompatible
  with columnar parallelism.
- Bulk-load ingestion path skipping L0. Separate issue.
- Changes to L0 rowstore or L1+ columnstore encoding strategy.

## References

- Issue #73 — original proposal + revised body
- RFC-0001 — full-rewrite compaction (the architectural precedent
  for "amortize at the file boundary, not at the row boundary")
- Iceberg v3 `deletion-vector-v1` spec — Puffin envelope used by
  `crates/merutable/src/iceberg/deletion_vector.rs`
- Existing partial-compaction DV path in
  `crates/merutable/src/engine/compaction/job.rs` and
  `crates/merutable/src/iceberg/catalog.rs:302` — the substrate this
  RFC extends from compaction-only to flush+compaction
