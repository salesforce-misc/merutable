# Developer guide

## Prerequisites

- **Rust**: install via [rustup](https://rustup.rs/)
  ```
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
  The exact toolchain version is pinned in `rust-toolchain.toml` at the workspace root.
  `rustup` reads this file automatically — no manual version selection needed.
- **Git**

## Building

```bash
# Debug build (fast compile, slow runtime)
cargo build --workspace

# Release build (LTO enabled, optimized)
cargo build --workspace --release
```

## Running tests

```bash
# All tests, debug mode
cargo test --workspace

# All tests, release mode (catches release-only UB)
cargo test --workspace --release
```

## Linting

```bash
# Format check
cargo fmt --check --all

# Fix formatting
cargo fmt --all

# Clippy with deny warnings (matches CI)
cargo clippy --workspace --all-targets -- -D warnings
```

## Benchmarks

```bash
cargo bench --workspace
```

Benchmarks cover bloom filter probes, memtable insert throughput, and compaction iterator merge rate.

## CI

CI runs on every push and PR to `main` via GitHub Actions (`.github/workflows/ci.yml`):

1. `cargo fmt --check --all`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`
4. `cargo test --workspace --release`

Benchmarks run on PRs only.

## Workspace layout

Issue #38 collapsed every internal `merutable-*` crate into a single
`merutable` crate. Two crates remain — `merutable` (the published
library + `merutable-migrate` binary) and `merutable-python`
(PyO3 cdylib; structurally must be its own crate).

```
merutable/
├── Cargo.toml                          # Workspace root, shared dependency versions
├── crates/
│   ├── merutable/                      # Published library + migrate binary
│   │   ├── Cargo.toml                  # Single-crate manifest with [features]
│   │   ├── build.rs                    # prost-build for the manifest .proto
│   │   ├── proto/manifest.proto        # Catalog manifest schema
│   │   └── src/
│   │       ├── lib.rs                  # Public API + module declarations
│   │       ├── db.rs / options.rs / error.rs / iterator.rs / mirror.rs
│   │       ├── types/                  # InternalKey, schema, FieldValue, errors
│   │       ├── store/                  # Pluggable object store
│   │       ├── wal/                    # Write-ahead log
│   │       ├── memtable/               # Skip-list memtable + arena
│   │       ├── parquet/                # Parquet SSTable + bloom + KvSparseIndex
│   │       ├── iceberg/                # Iceberg catalog + manifest + deletion vectors
│   │       ├── engine/                 # Flush, compaction, read/write paths
│   │       ├── sql/                    # Change-feed (feature `sql`, on by default)
│   │       ├── replica/                # Scale-out RO replica (feature `replica`)
│   │       └── bin/merutable-migrate.rs
│   └── merutable-python/               # PyO3 bindings (cdylib)
└── .github/workflows/ci.yml            # CI pipeline
```

`merutable` features: `default = ["sql"]`, `sql` (DataFusion-backed
change feed), `replica = ["sql"]`. The replica module depends on
the change-feed cursor, so enabling `replica` automatically enables
`sql`.

Internal dependencies flow `db.rs` → `engine` → `{iceberg, parquet,
memtable, wal, store}` → `types`. After #38 these are intra-crate
modules with `pub` visibility (a follow-up sweep tightens to
`pub(crate)`).

## Adding a dependency

All dependency versions are pinned in the workspace root `Cargo.toml` under `[workspace.dependencies]`. Individual crates reference them with `{ workspace = true }`. Never add version specs in crate-level `Cargo.toml` files.
