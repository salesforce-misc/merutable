//! Shareable tokio runtime handle.
//!
//! Multiple database handles opened on the same MeruRuntime share one thread
//! pool instead of each spinning up independent multi_thread runtimes.
//!
//! Usage:
//!   MeruRuntime *rt = meru_runtime_new(0, &err);   // 0 = default thread count
//!   MeruHandle  *db = meru_open(&opts, rt, &err);  // rt=NULL also works (creates a private one)
//!   ...
//!   meru_close_free(db, &err);
//!   meru_runtime_free(rt);

use std::sync::Arc;

/// Opaque shareable tokio runtime. Pass to meru_open() so multiple databases
/// share one thread pool. NULL is accepted everywhere and creates a
/// per-handle runtime instead.
pub struct MeruRuntime {
    pub inner: Arc<tokio::runtime::Runtime>,
}

/// Create a new multi-thread tokio runtime.
///
/// `worker_threads = 0` uses the tokio default (one thread per logical CPU).
///
/// Returns NULL on failure; `*err_out` is set when err_out is non-null.
///
/// # Safety
/// `err_out` must be null or a valid pointer to a `char *`.
#[no_mangle]
pub unsafe extern "C" fn meru_runtime_new(
    worker_threads: usize,
    err_out: *mut *mut std::ffi::c_char,
) -> *mut MeruRuntime {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    if worker_threads > 0 {
        builder.worker_threads(worker_threads);
    }
    match builder.build() {
        Ok(rt) => Box::into_raw(Box::new(MeruRuntime {
            inner: Arc::new(rt),
        })),
        Err(e) => {
            crate::error::set_err(err_out, &format!("failed to create runtime: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free a MeruRuntime. Safe to call with NULL.
///
/// The underlying thread pool is shut down when the last handle that shares
/// this runtime is also freed.
///
/// # Safety
/// `rt` must have been returned by `meru_runtime_new`.
#[no_mangle]
pub unsafe extern "C" fn meru_runtime_free(rt: *mut MeruRuntime) {
    if !rt.is_null() {
        drop(Box::from_raw(rt));
    }
}

