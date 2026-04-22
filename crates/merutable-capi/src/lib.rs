//! C ABI bindings for merutable.
//!
//! Use `meru_open` / `meru_open_existing` to obtain a handle, then call
//! `meru_put` / `meru_get` / `meru_scan` etc. Finish with `meru_close_free`.
//!
//! The generated C header is at `include/merutable.h`.

mod db_fns;
mod error;
mod handle;
mod types;

// Re-export all public symbols so they appear in the dylib and are picked up
// by cbindgen when it traverses this crate root.
pub use db_fns::{
    meru_catalog_path, meru_close, meru_close_free, meru_compact, meru_delete, meru_export_iceberg,
    meru_flush, meru_free, meru_get, meru_is_closed, meru_open, meru_open_existing, meru_put,
    meru_put_batch, meru_refresh, meru_row_free, meru_scan, meru_scan_result_free, meru_stats,
};
pub use error::meru_free_string;
pub use types::{
    MerucacheStats, MeruBytesView, MeruColumnDef, MeruColumnType, MeruMemtableStats,
    MeruOpenOptions, MeruRow, MeruScanResult, MeruSchema, MeruStats, MeruValue, MeruValueInner,
};
pub use error::MeruStatus;
