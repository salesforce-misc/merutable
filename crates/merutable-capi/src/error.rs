use std::ffi::{c_char, c_int, CString};

use merutable::types::MeruError;

/// Status codes returned by all fallible C API functions.
#[repr(C)]
pub enum MeruStatus {
    Ok = 0,
    ErrIo = 1,
    ErrCorruption = 2,
    ErrSchemaMismatch = 3,
    ErrNotFound = 4,
    ErrInvalidArg = 5,
    ErrReadOnly = 6,
    ErrClosed = 7,
    ErrObjectStore = 8,
    ErrParquet = 9,
    ErrAlreadyExists = 10,
    ErrUnknown = 99,
}

pub fn error_to_status(e: &MeruError) -> MeruStatus {
    match e {
        MeruError::Io(_) => MeruStatus::ErrIo,
        MeruError::Corruption(_) => MeruStatus::ErrCorruption,
        MeruError::SchemaMismatch(_) => MeruStatus::ErrSchemaMismatch,
        MeruError::NotFound => MeruStatus::ErrNotFound,
        MeruError::InvalidArgument(_) => MeruStatus::ErrInvalidArg,
        MeruError::ReadOnly => MeruStatus::ErrReadOnly,
        MeruError::Closed => MeruStatus::ErrClosed,
        MeruError::ObjectStore(_) => MeruStatus::ErrObjectStore,
        MeruError::Parquet(_) => MeruStatus::ErrParquet,
        MeruError::AlreadyExists(_) => MeruStatus::ErrAlreadyExists,
        _ => MeruStatus::ErrUnknown,
    }
}

/// Write an error message string into *err_out when err_out is non-null.
/// The caller must free the string with meru_free_string().
pub unsafe fn set_err(err_out: *mut *mut c_char, msg: &str) {
    if err_out.is_null() {
        return;
    }
    let s = CString::new(msg).unwrap_or_else(|_| CString::new("(error message contained nul)").unwrap());
    *err_out = s.into_raw();
}

/// Free a C string returned by the API (catalog_path, error messages).
///
/// # Safety
/// `s` must have been allocated by the merutable C API. Passing NULL is safe.
#[no_mangle]
pub unsafe extern "C" fn meru_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    drop(CString::from_raw(s));
}

/// Convenience: convert a Rust Result into a C status code + optional error string.
pub unsafe fn result_to_c<T>(
    result: Result<T, MeruError>,
    err_out: *mut *mut c_char,
) -> (c_int, Option<T>) {
    match result {
        Ok(v) => (MeruStatus::Ok as c_int, Some(v)),
        Err(e) => {
            set_err(err_out, &e.to_string());
            (error_to_status(&e) as c_int, None)
        }
    }
}
