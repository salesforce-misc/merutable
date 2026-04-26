# Hybrid commit: POSIX primary with async object-store mirror

Issue #31. This document describes the architecture merutable uses to
combine local-disk write latency with object-store read scale-out.

## What problem this solves

the POSIX (atomic-rename) commit path (the default) is the right answer for sub-
millisecond commit latency — local atomic rename is faster than any
conditional-PUT roundtrip will ever be. But local disk has two
fundamental limits:

1. **Durability**: dies with the host.
2. **Read scaling**: only the host machine can read it.

a conditional-PUT object-store layout (Issue #26) goes the other way: every
commit is a conditional PUT against an object store. Multi-writer
safe, durable cross-host, but pays roundtrip latency per commit.

The deployment shape that matters in practice for external analytics is **neither
endpoint alone**: a single-writer primary with local POSIX commit
(for latency) AND a continuously-updated object-store copy (for
durability + read scaling).

## Surface

Enabled via a `MirrorConfig` attached to `OpenOptions`:

```rust
let mirror = MirrorConfig::new(Arc::new(S3Store::new(s3, "bucket/prefix")))
    .max_lag_alert_secs(60)
    .mirror_parallelism(4);

let db = MeruDB::open(
    OpenOptions::new(schema)
        .wal_dir("/local/wal")
        .catalog_uri("/local/data")
        .mirror(mirror),
).await?;
```

Issue #43 collapsed merutable to a single POSIX commit path, so
there is no longer a validation step that rejects mirror + commit-
mode combinations — the POSIX atomic-rename path is the only
commit path the primary uses, and the mirror always shadows it.

## Scope: flushed files only

**The mirror covers SSTs + manifests. Never the WAL.**

This is a hard boundary:

- The WAL is the un-flushed in-memory tail's durability primitive.
  It exists precisely because the memtable hasn't been flushed yet.
- Once a memtable flushes to an SST and the SST lands in a manifest,
  the WAL entries for those rows are discardable.
- The mirror picks up exactly that state — SSTs and manifests — and
  no earlier.

If you need the WAL's durability cross-host, use a conditional-PUT object-store layout
directly. That's what it's for. Going halfway — a mirror that covers
WAL on top of POSIX — is worse than either endpoint: more complexity
than the POSIX baseline, less consistency than the ObjectStore mode.

**Crash-loss model**: a primary crash before the mirror catches up
loses everything in the un-flushed tail at the time of crash. A
reader on the mirror sees the most recent fully-mirrored snapshot,
missing all post-mirror activity. RPO is bounded by
`max(mirror_lag, gc_grace_period)`.

## Mirror layout = a conditional-PUT object-store layout layout

The mirror writes to its target in the EXACT same shape that
a conditional-PUT object-store layout uses:

```
bucket/prefix/
├── metadata/
│   ├── v1.manifest.bin       # protobuf with MRUB magic + length framing (#28)
│   ├── v2.manifest.bin
│   ├── v3.manifest.bin
│   └── low_water.txt         # if reclaim has happened (#26 Phase 6)
└── data/
    ├── L0/*.parquet
    ├── L1/*.parquet
    └── L2/*.parquet
```

Same file paths, same manifest format, same backward-pointer chain.
This is the critical property.

**It means any reader that can open an `ObjectStore`-mode bucket can
open a mirror destination.** There is no "restore from mirror" step,
no mirror-format-vs-canonical-format discrimination, no special
reader path. The mirror IS a live, mountable layout.

Cross-region RO replica drops out as a byproduct: point a
`OpenOptions::read_only(true) + the object-store layout` at the
mirror destination from another region. The replica catches up by
reading manifests as the primary mirrors them. (See Issue #32 for
the replica's hot-swap-rebase architecture that builds on this.)

## Commit-order invariant

The mirror worker MUST upload data files BEFORE the manifest that
references them. Manifests MUST be uploaded in seq order. A reader
of the mirror must never observe a manifest pointing at files that
don't exist yet.

Per snapshot S:

1. Enumerate data files referenced in S that aren't yet on the mirror.
2. Upload them in parallel (`mirror_parallelism`). Use `put_if_absent`
   so retries are safe.
3. Once all data files for S are confirmed, `put_if_absent` the
   manifest for S.
4. On any failure mid-pattern, the next mirror tick re-runs the
   pattern. Idempotent throughout.

The conditional PUT on the manifest gives the same race-safety
guarantee as a conditional-PUT object-store layout: if a second process is
somehow also trying to mirror to the same destination, only one
wins. **Mirror destinations should not be shared targets**; this is
documented loudly but not enforced beyond the conditional-PUT
natural serialization.

## `mirror_seq` tracking

The primary surfaces two seq values:

- `visible_seq` — local commit watermark (existing).
- `mirror_seq` — last seq fully mirrored (new, exposed in Phase 3).

Invariant: `mirror_seq <= visible_seq`.

Derived: `mirror_lag_secs = clock - mirror_seq.commit_time`.

Both are exposed via `stats()` and the Issue #14 metrics surface.
Above `max_lag_alert_secs`, a `tracing::warn!` fires.

**Writes do NOT backpressure on mirror lag.** The whole point of the
hybrid is async. If users want backpressure, that's an explicit
follow-on with explicit semantics (block vs. slow vs. drop).

## Phases

Implementation is phased so each increment is independently useful
and reviewable:

- **Phase 1 (shipped)**: `MirrorConfig` type + `OpenOptions::mirror()`
  builder + validation. Accepting `MirrorConfig` compiles and
  round-trips through `OpenOptions`; the worker is not yet spawned.
- **Phase 2 (planned)**: mirror worker spawned alongside flush +
  compaction workers. Implements the commit-order-preserving upload
  loop above. Idempotent retries.
- **Phase 3 (planned)**: `mirror_seq` exposed via `stats()`;
  `mirror_lag_secs` available as a derived metric.
- **Phase 4 (planned)**: `max_lag_alert_secs` triggers
  `tracing::warn!`; no backpressure.

## Guarantees (v1)

- Mirror destination is byte-compatible with a conditional-PUT object-store layout.
  Remote readers open it via the standard read path. No special
  tools.
- Commit order on the mirror matches commit order on the primary.
  No dangling-manifest observations.
- Idempotent retries: killing the mirror worker mid-upload and
  restarting completes without re-uploading already-confirmed files.
- WAL is NEVER uploaded.

## Non-guarantees (by design)

- **Not a sync commit**. The whole point is async. Writes return to
  the caller as soon as the POSIX commit lands locally.
- **Not multi-writer**. One primary per mirror destination.
  Conditional PUT on the manifest protects against accidental
  misconfiguration; operators should not rely on this for
  deliberate multi-writer workloads (use a conditional-PUT object-store layout).
- **Not backpressured**. Mirror lag never blocks writes. Alert-only.
- **Not WAL durable**. An un-flushed tail is lost on primary crash.

## When to use what

| Need | Recommendation |
|---|---|
| Sub-ms commit latency, single host | the POSIX (atomic-rename) commit path |
| Cross-host durability, ms RPO | the POSIX (atomic-rename) commit path + mirror |
| Multi-writer, any RPO | a conditional-PUT object-store layout |
| Cross-region analytics reader | Mirror destination + `read_only(true) + the object-store layout` |
| Zero-RPO, any latency | a conditional-PUT object-store layout |
