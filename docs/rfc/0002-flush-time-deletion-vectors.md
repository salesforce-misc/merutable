# RFC-0002: Flush-time deletion vector emission

**Issue:** #73
**Status:** Draft, awaiting sign-off

## Problem

Range scans over upserting workloads currently force cross-level
LSM merge to reconcile versions. That merge serializes the read
pipeline and forces row materialization across L0..Ln, defeating
columnar SIMD execution at L1+. Industry measurement on similar
designs (Hyper, Photon, DuckDB lineage) puts the penalty at up to
~100x vs. clean-Parquet baseline.

The fix is positional deletion vectors — the same primitive Iceberg
v3, Delta, and Hudi converged on. The question is **where in the
write pipeline** to compute them.

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

### `resolve_dv_target` shape

```rust
fn resolve_dv_target(
    user_key: &[u8],
    version: &Version,
    base_path: &Path,
    schema: &Arc<TableSchema>,
) -> Result<Option<(String, u64)>>
```

For each level L1..Lmax, find the (at most one) covering file via the
existing non-overlapping range index, then call a new
`ParquetReader::position_of(user_key)` that walks the bloom + sparse
index → page lookup → returns the file-global row position WITHOUT
decoding the row. `get`'s existing logic already locates the row;
the new method skips the value materialization. Returns the first
hit by level (lowest level wins because L1+ are non-overlapping per
level and a key cannot live in multiple L1+ files at the same
sequence horizon).

L0 is **deliberately not probed.** L0 files are the recent flushes
themselves; an upsert of a key whose prior version is in L0 will be
reconciled by future compaction merging the two L0 files. Probing L0
during flush would create a self-referential dependency
(this-flush's L0 vs. last-flush's L0) and add cost without changing
the columnar-parallelism guarantee, which only matters at L1+.

### What about the upsert→flush window?

Between `db.put()` and the next flush, an upserted key's prior L1+
version is **not yet** DV-marked. A concurrent range scan during
that window sees both versions and pays the cross-level merge
penalty for those rows.

This is **bounded**, not eliminated:

- L0 size is bounded by `memtable_size_mb × max_immutable_memtables`,
  both configurable.
- The merge penalty applies only to the keys that overlap between
  L0/memtable and L1+. New-key inserts (the dominant upsert pattern
  for many workloads) do not overlap.
- For workloads where the window matters, the lever is "tune flush
  thresholds tighter," **not** "couple the upsert path to file
  layout."

A transient in-memory scan-time bitmask (computed per scan from
active memtable+L0 keys vs. L1+ file row positions) is a viable
follow-on if measurement shows the window is a real bottleneck. It
is **not** in this RFC's scope. We ship the flush-time path first,
measure, then decide.

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
| A7 | DV resolution decodes the row (wasted work)                                  | New `position_of` method skips row decode; bloom + sparse-index → page → InternalKey scan → return position only.                      |
| A8 | Tombstone in memtable resolves a stale L1+ position                          | Tombstones go through the same resolve path. The DV emission handles tombstone-of-L1+-row correctly: L1+ position gets DV-marked, tombstone in L0 still drives compaction-time deletion of any L0-resident prior versions. |
| A9 | Probing L0 creates a cycle                                                    | L0 explicitly skipped — see "What about the upsert→flush window?" above.                                                               |

## Cost model

For a flush of M memtable keys against a version with F L1+ files
holding K total rows:

- **Bloom probes:** `M × F`, ~50ns each. M=500K, F=50 → 25M ops ≈ 1.25s.
  This is the worst case; bloom rejects most keys at <1ns amortized
  with cache-hot bloom pages.
- **Page reads on bloom hits:** ≈ `M × hit_rate × log2(K/page_size)`.
  For 10% upsert hit rate, M=500K → 50K probes × ~10μs cache-warm
  = 500ms.
- **Total flush latency increase:** ~1–5s for a 64 MiB memtable on
  realistic data, acceptable on a background flush.
- **No write-path latency change.** The `db.put()` cost remains
  WAL fsync + memtable insert, unchanged.

If `enable_flush_dv_emission` is set false (config flag, defaults
true), the entire resolve phase is skipped — for users who want
the pre-RFC behavior or who do not have upsert workloads.

## Tests (with tight bounds — 2x-regression-tight per the engineering bar)

| Test                                                  | Bound                                                                                                                          |
|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `put_latency_no_regression`                           | p99 latency post-RFC ≤ 1.10 × pre-RFC p99. Both measured under identical workload. Fails if write path was perturbed.          |
| `flush_emits_dv_for_upserts`                          | Write N rows, force-flush, force-compact (so all are in L1+). Upsert all N. Force-flush. Asserts: L1 file's DV cardinality == N. **Equality**, not inequality. |
| `flush_emits_no_dv_for_new_keys`                      | Write N entirely-new keys, force-flush. Asserts: zero DV blobs in committed snapshot. Tight.                                  |
| `flush_emits_dv_for_tombstones`                       | Write N rows → flush → compact. Delete all N. Force-flush. Asserts: L1 file's DV cardinality == N. Tight.                     |
| `flush_resolves_against_pinned_version_under_compaction` | Concurrent compaction commits between flush start and flush commit. Asserts: flush's DV references the post-compaction file set, not stale (pre-compaction) paths. |
| `scan_post_flush_no_duplicates_or_old_versions`       | Write N rows → flush → compact. Upsert all N. Flush. Range-scan. Asserts: exactly N rows, every value is the upserted value. Equality. |
| `wal_format_unchanged`                                | Snapshot-test the WAL record bytes pre- and post-RFC. Asserts: byte-identical.                                                |
| `memtable_interface_unchanged`                        | Public surface of `MemtableManager` and `Memtable` unchanged. Compile-time test via doc-test or pub-API snapshot.            |
| `range_scan_throughput_post_flush`                    | After flush emits DVs, range-scan throughput ≥ 0.5 × clean-Parquet baseline. (The "small constant factor" target from the issue's acceptance criteria; ≥0.5x is "within 2x" — explicit and measurable.) |
| `concurrent_flush_compaction_dv_consistency`          | Long-running stress: continuous puts, periodic flush + compaction, periodic scans. Asserts: monotonic scan results match an in-process oracle (write-side state). 60s run. |

## Phases

Each phase is independently reviewable and lands as its own PR.

**Phase 1 — `ParquetReader::position_of`** (no behavior change yet)

- New method on the existing reader. Same probe path as `get`, returns
  position only.
- Unit tests: position consistency vs. `get`'s implicit position;
  None for absent keys; correct under multi-version files.

**Phase 2 — `engine::dv_resolve` module** (helper, no integration yet)

- `resolve_dv_target(user_key, version, base_path, schema)`.
- Pure function over `&Version`. Unit-testable with synthetic versions.
- Tests: hits L1+ correctly, returns None for new keys, skips L0,
  picks lowest L when present in multiple levels (which shouldn't
  happen in practice but the test pins the contract).

**Phase 3 — wire into `flush.rs` behind a feature flag**

- Default-on flag `EngineConfig::enable_flush_dv_emission = true`.
- Move pre-commit work inside `commit_lock`.
- Resolve every memtable entry; aggregate by file; `txn.add_dv`.
- Integration tests from the table above.

**Phase 4 — bench harness**

- Criterion bench:
  - `bench_put_latency` — p99 baseline + post.
  - `bench_flush_with_dv` — flush wall time vs. memtable size, dv ON vs. OFF.
  - `bench_range_scan_post_flush` — vs. clean-Parquet baseline.
- Numbers committed to `docs/rfc/0002-flush-time-deletion-vectors.md`
  as a "Measured" addendum.

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
