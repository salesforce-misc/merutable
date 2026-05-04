//! `OpenOptions`: the stable tuning surface for `MeruDB::open`.
//!
//! Every field here is a deliberate, documented knob. The internal
//! `EngineConfig` has additional fields, but callers should not reach
//! into `merutable-engine` directly — that crate is `publish = false`
//! for a reason (Issue #9).
//!
//! All knobs default to sane production values; builder methods let
//! you override individually. Unset knobs pass `EngineConfig::default()`
//! through.

use crate::types::schema::TableSchema;
use std::path::PathBuf;
use std::sync::Arc;

/// Issue #31: async object-store mirror of flushed files + manifests.
/// The primary commits via the (only) POSIX atomic-rename path and an
/// async worker uploads each flushed SST + each new manifest version
/// to a remote object-store target — giving remote readers a
/// near-real-time copy without the primary having to block on object-
/// store latency per commit.
///
/// ## Crash-loss model
///
/// The WAL is NEVER mirrored. A primary crash loses the un-flushed
/// in-memory tail. Readers on the mirror see the most recent
/// fully-mirrored snapshot.
///
/// ## Phases
///
/// - Phase 1 (this type): struct, builder, validation. Creating a
///   `MirrorConfig` and attaching it compiles and round-trips through
///   `OpenOptions`; the mirror worker is not yet spawned.
/// - Phase 2 (planned): mirror worker spawned alongside flush +
///   compaction workers, commit-order-preserving upload loop.
/// - Phase 3 (planned): `mirror_seq` tracking via `stats()`.
/// - Phase 4 (planned): alert on `max_lag_alert_secs`.
#[derive(Clone)]
pub struct MirrorConfig {
    /// S3 / GCS / Azure destination. Must implement `MeruStore`.
    pub target: Arc<dyn crate::store::traits::MeruStore>,
    /// Warn above this lag (seconds between primary commit_time and
    /// last-mirrored commit_time). Alert-only in v1; writes never
    /// block on mirror lag.
    pub max_lag_alert_secs: u64,
    /// Concurrent uploads during a single mirror sweep. Higher =
    /// faster catch-up after a sustained primary burst; higher also
    /// = more in-flight object-store connections. Default: 4.
    pub mirror_parallelism: usize,
}

impl std::fmt::Debug for MirrorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MirrorConfig")
            .field("target", &"Arc<dyn MeruStore>")
            .field("max_lag_alert_secs", &self.max_lag_alert_secs)
            .field("mirror_parallelism", &self.mirror_parallelism)
            .finish()
    }
}

impl MirrorConfig {
    /// Production defaults for lag alert (60s) and parallelism (4).
    /// Callers must still provide `target`.
    pub fn new(target: Arc<dyn crate::store::traits::MeruStore>) -> Self {
        Self {
            target,
            max_lag_alert_secs: 60,
            mirror_parallelism: 4,
        }
    }

    pub fn max_lag_alert_secs(mut self, secs: u64) -> Self {
        self.max_lag_alert_secs = secs;
        self
    }

    pub fn mirror_parallelism(mut self, n: usize) -> Self {
        self.mirror_parallelism = n.max(1);
        self
    }
}

/// Builder for opening a `MeruDB` instance.
#[derive(Clone, Debug)]
pub struct OpenOptions {
    pub schema: TableSchema,
    pub catalog_uri: String,
    pub object_store_url: String,
    pub wal_dir: PathBuf,

    // Memtable
    pub memtable_size_mb: usize,
    pub max_immutable_count: usize,

    // Row cache
    pub row_cache_capacity: usize,

    // Compaction targets
    /// Per-level size targets for L1 and beyond, in bytes.
    /// `level_target_bytes[0]` is L1 target, `[1]` is L2, etc.
    /// Default: [256 MiB, 2 GiB, 16 GiB, 128 GiB].
    pub level_target_bytes: Vec<u64>,

    // L0 triggers
    pub l0_compaction_trigger: usize,
    pub l0_slowdown_trigger: usize,
    pub l0_stop_trigger: usize,

    // Parquet tuning
    pub bloom_bits_per_key: u8,

    // Compaction I/O cap
    pub max_compaction_bytes: u64,

    /// Issue #30: upper bound on total rows a single compaction
    /// ingests. `0` = unbounded (back-compat default). Operators
    /// seeing the RSS-2.6x symptom set this to cap decoded-row
    /// memory per compaction independent of byte size. See
    /// `EngineConfig::max_compaction_input_rows` for the full
    /// contract.
    pub max_compaction_input_rows: u64,

    // Background parallelism
    pub flush_parallelism: usize,
    pub compaction_parallelism: usize,

    // GC grace
    pub gc_grace_period_secs: u64,

    // Lifecycle
    pub read_only: bool,

    /// Issue #15: highest level (inclusive) that carries the row-blob
    /// fast path. `Some(0)` matches the default; `None` = columnar
    /// everywhere; `Some(N)` pushes fast-path deeper for OLTP-heavy
    /// workloads.
    pub dual_format_max_level: Option<u8>,

    /// Issue #31: optional async mirror to an object-store target.
    /// See [`MirrorConfig`].
    pub mirror: Option<MirrorConfig>,

    /// RFC-0002: emit per-file deletion vectors at flush time so
    /// every prior version of each upserted/deleted memtable key is
    /// DV-marked. Required for external Iceberg readers to see one
    /// row per primary key without an MVCC dedup projection.
    /// Default: `true`.
    pub enable_flush_dv_emission: bool,
}

impl OpenOptions {
    /// Construct a builder with the given table schema and
    /// production defaults for every other field. The defaults come
    /// from `crate::engine::config::EngineConfig::default()` — see
    /// that type for the authoritative values.
    pub fn new(schema: TableSchema) -> Self {
        // Pull defaults from EngineConfig so there's exactly one
        // place to change production constants.
        let ec = crate::engine::config::EngineConfig::default();
        Self {
            schema,
            catalog_uri: String::new(),
            object_store_url: String::new(),
            wal_dir: ec.wal_dir,
            memtable_size_mb: ec.memtable_size_bytes / (1024 * 1024),
            max_immutable_count: ec.max_immutable_count,
            row_cache_capacity: ec.row_cache_capacity,
            level_target_bytes: ec.level_target_bytes,
            l0_compaction_trigger: ec.l0_compaction_trigger,
            l0_slowdown_trigger: ec.l0_slowdown_trigger,
            l0_stop_trigger: ec.l0_stop_trigger,
            bloom_bits_per_key: ec.bloom_bits_per_key,
            max_compaction_bytes: ec.max_compaction_bytes,
            max_compaction_input_rows: ec.max_compaction_input_rows,
            flush_parallelism: ec.flush_parallelism,
            compaction_parallelism: ec.compaction_parallelism,
            gc_grace_period_secs: ec.gc_grace_period_secs,
            read_only: ec.read_only,
            dual_format_max_level: ec.dual_format_max_level,
            mirror: None,
            enable_flush_dv_emission: ec.enable_flush_dv_emission,
        }
    }

    /// Issue #31: attach an async mirror to an object-store target.
    pub fn mirror(mut self, cfg: MirrorConfig) -> Self {
        self.mirror = Some(cfg);
        self
    }

    /// Issue #15: highest LSM level whose SSTables carry the
    /// `_merutable_value` row-blob fast path. `Some(0)` (default)
    /// matches the pre-Issue-#15 hard boundary; `Some(N)` pushes the
    /// fast path to L0..=LN for OLTP-heavy workloads; `None` = every
    /// level columnar-only for OLAP / append-only.
    pub fn dual_format_max_level(mut self, max: Option<u8>) -> Self {
        self.dual_format_max_level = max;
        self
    }

    /// RFC-0002: toggle flush-time deletion-vector emission. Default
    /// is `true`; set `false` to skip the resolve+emit step (e.g. for
    /// workloads with no upserts where the cost is pure overhead).
    pub fn enable_flush_dv_emission(mut self, on: bool) -> Self {
        self.enable_flush_dv_emission = on;
        self
    }

    pub fn catalog_uri(mut self, uri: impl Into<String>) -> Self {
        self.catalog_uri = uri.into();
        self
    }

    pub fn object_store(mut self, url: impl Into<String>) -> Self {
        self.object_store_url = url.into();
        self
    }

    pub fn wal_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.wal_dir = dir.into();
        self
    }

    pub fn memtable_size_mb(mut self, mb: usize) -> Self {
        self.memtable_size_mb = mb;
        self
    }

    /// Maximum number of rotated memtables waiting to be flushed
    /// before writes hard-stall. Default: 4.
    pub fn max_immutable_count(mut self, n: usize) -> Self {
        self.max_immutable_count = n;
        self
    }

    pub fn row_cache_capacity(mut self, capacity: usize) -> Self {
        self.row_cache_capacity = capacity;
        self
    }

    /// Per-level byte targets for L1..LN. Index 0 = L1. Default:
    /// `[256 MiB, 2 GiB, 16 GiB, 128 GiB]`.
    pub fn level_target_bytes(mut self, targets: Vec<u64>) -> Self {
        self.level_target_bytes = targets;
        self
    }

    /// L0 file count that triggers a compaction. Default: 4.
    pub fn l0_compaction_trigger(mut self, n: usize) -> Self {
        self.l0_compaction_trigger = n;
        self
    }

    /// L0 file count at which writes begin graduated slowdown.
    /// Default: 20.
    pub fn l0_slowdown_trigger(mut self, n: usize) -> Self {
        self.l0_slowdown_trigger = n;
        self
    }

    /// L0 file count at which writes hard-stop until compaction
    /// drains L0. Default: 36.
    pub fn l0_stop_trigger(mut self, n: usize) -> Self {
        self.l0_stop_trigger = n;
        self
    }

    /// Bits per key for the SIMD bloom filter stored in Parquet footer
    /// KV metadata. Higher = smaller false-positive rate, more bytes.
    /// Default: 10 (~1% FPR).
    pub fn bloom_bits_per_key(mut self, bits: u8) -> Self {
        self.bloom_bits_per_key = bits;
        self
    }

    /// Upper bound on per-compaction input bytes. Prevents a single
    /// deep-level compaction from pulling multi-GiB into memory.
    /// Default: 256 MiB. See Issue #2.
    /// Issue #30: cap the total rows a single compaction ingests.
    /// `0` disables. Set to bound decoded-row memory per
    /// compaction; good starting point: `max_compaction_bytes /
    /// avg_row_bytes` for your workload.
    pub fn max_compaction_input_rows(mut self, rows: u64) -> Self {
        self.max_compaction_input_rows = rows;
        self
    }

    pub fn max_compaction_bytes(mut self, bytes: u64) -> Self {
        self.max_compaction_bytes = bytes;
        self
    }

    /// Number of background flush workers. Default: 1.
    /// `0` disables the auto-flush background loop (manual
    /// `flush()` calls still work).
    pub fn flush_parallelism(mut self, n: usize) -> Self {
        self.flush_parallelism = n;
        self
    }

    /// Number of background compaction workers. Default: 2.
    /// Workers run on disjoint level sets in parallel. `0`
    /// disables the auto-compaction background loop.
    pub fn compaction_parallelism(mut self, n: usize) -> Self {
        self.compaction_parallelism = n;
        self
    }

    /// Seconds to retain compaction-obsoleted files before GC. Gives
    /// external external analytical readers (DuckDB, Spark) time to finish mid-read.
    /// Default: 300 (5 minutes). Internal readers use version-pin
    /// refcounting and are NOT bounded by this timer.
    pub fn gc_grace_period_secs(mut self, secs: u64) -> Self {
        self.gc_grace_period_secs = secs;
        self
    }

    pub fn read_only(mut self, enabled: bool) -> Self {
        self.read_only = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::local::LocalFileStore;

    // Issue #43: removed the `schema()` helper + `CommitMode`-vs-
    // MirrorConfig validation tests. With ObjectStore mode gone,
    // there is no combination left to validate — the `MirrorConfig`
    // defaults and the parallelism floor are the only invariants
    // the type surface still pins.

    #[test]
    fn mirror_defaults_are_production_sane() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileStore::new(tmp.path()).unwrap());
        let cfg = MirrorConfig::new(store);
        assert_eq!(cfg.max_lag_alert_secs, 60);
        assert_eq!(cfg.mirror_parallelism, 4);
    }

    #[test]
    fn mirror_parallelism_floored_at_one() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileStore::new(tmp.path()).unwrap());
        let cfg = MirrorConfig::new(store).mirror_parallelism(0);
        assert_eq!(cfg.mirror_parallelism, 1, "zero coerced to one");
    }
}
