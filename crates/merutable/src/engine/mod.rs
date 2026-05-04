pub mod background;
pub mod cache;
pub mod codec;
pub mod compaction;
pub mod config;
pub mod dv_resolve;
// Issue #38: workspace collapsed; the former crate's `lib.rs` →
// `mod.rs`, and the former `engine.rs` retains its name to keep
// the public path `crate::engine::engine::MeruEngine` stable for
// any downstream code (already re-exported as
// `crate::engine::MeruEngine` below).
#[allow(clippy::module_inception)]
pub mod engine;
pub mod flush;
pub mod metrics;
pub mod read_path;
pub mod stats;
pub mod write_path;

pub use config::EngineConfig;
pub use engine::MeruEngine;
pub use stats::{CacheStats, DvStats, EngineStats, FileStats, LevelStats, MemtableStats};
