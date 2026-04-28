//! All extern "C" database functions.

use std::ffi::{c_char, c_int, CStr, CString};
use std::sync::Arc;

use merutable::{
    iceberg::manifest::Manifest,
    parquet::codec::{IKEY_COLUMN_NAME, OP_COLUMN_NAME, SEQ_COLUMN_NAME, VALUE_BLOB_COLUMN_NAME},
    types::schema::TableSchema,
    MeruDB, OpenOptions,
};

use crate::{
    error::{result_to_c, set_err, MeruStatus},
    handle::MeruHandle,
    runtime::MeruRuntime,
    types::{
        c_pk_to_rust, c_row_to_rust, c_schema_to_rust, free_c_row_fields, rust_col_def_to_c,
        rust_row_to_c, MerucacheStats, MeruColumnDef, MeruManifestInfo, MeruMemtableStats,
        MeruOpenOptions, MeruRow, MeruScanResult, MeruStats, MeruValue,
    },
};

/// Borrow the runtime from a required `*mut MeruRuntime`.
/// Returns None (and sets err) if rt_ptr is null.
unsafe fn require_runtime(
    rt_ptr: *mut MeruRuntime,
    err_out: *mut *mut c_char,
) -> Option<Arc<tokio::runtime::Runtime>> {
    if rt_ptr.is_null() {
        set_err(err_out, "rt must be non-null — create one with meru_runtime_new()");
        None
    } else {
        Some(Arc::clone(&(*rt_ptr).inner))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

unsafe fn handle_ref<'a>(db: *const MeruHandle) -> &'a MeruHandle {
    &*db
}

unsafe fn handle_mut<'a>(db: *mut MeruHandle) -> &'a MeruHandle {
    &*db
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

/// Open (or create) a database.
///
/// `rt` must be a non-null runtime from `meru_runtime_new()`. Multiple databases
/// can share one runtime — they will share its thread pool.
///
/// Calling meru_* functions from a coroutine/async task: dispatch to a
/// blockable thread (e.g. ASIO post to a thread_pool_executor, or
/// std::thread) rather than calling directly from your executor's event loop
/// thread, which may deadlock.
///
/// `opts->wal_dir = NULL` defaults to `"{catalog_uri}/wal"`.
///
/// # Safety
/// All pointer fields in `opts` must be valid non-null C strings (except `wal_dir`).
#[no_mangle]
pub unsafe extern "C" fn meru_open(
    opts: *const MeruOpenOptions,
    rt: *mut MeruRuntime,
    db_out: *mut *mut MeruHandle,
    err_out: *mut *mut c_char,
) -> c_int {
    if opts.is_null() || db_out.is_null() {
        set_err(err_out, "opts and db_out must be non-null");
        return MeruStatus::ErrInvalidArg as c_int;
    }
    let opts = &*opts;

    let rt = match require_runtime(rt, err_out) {
        Some(r) => r,
        None => return MeruStatus::ErrIo as c_int,
    };

    let schema = match c_schema_to_rust(&opts.schema) {
        Ok(s) => s,
        Err(e) => {
            set_err(err_out, &e.to_string());
            return MeruStatus::ErrInvalidArg as c_int;
        }
    };

    let catalog_uri = match CStr::from_ptr(opts.catalog_uri).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_err(err_out, "catalog_uri is not valid UTF-8");
            return MeruStatus::ErrInvalidArg as c_int;
        }
    };

    let wal_dir = if opts.wal_dir.is_null() {
        std::path::PathBuf::from(&catalog_uri).join("wal")
    } else {
        match CStr::from_ptr(opts.wal_dir).to_str() {
            Ok(s) => std::path::PathBuf::from(s),
            Err(_) => {
                set_err(err_out, "wal_dir is not valid UTF-8");
                return MeruStatus::ErrInvalidArg as c_int;
            }
        }
    };

    let memtable_mb = if opts.memtable_size_mb == 0 { 64 } else { opts.memtable_size_mb };

    let open_opts = OpenOptions::new(schema.clone())
        .catalog_uri(&catalog_uri)
        .wal_dir(wal_dir)
        .memtable_size_mb(memtable_mb)
        .read_only(opts.read_only != 0);

    let (status, maybe_db) = result_to_c(rt.block_on(MeruDB::open(open_opts)), err_out);

    if let Some(db) = maybe_db {
        *db_out = Box::into_raw(MeruHandle::new(db, schema, rt));
        MeruStatus::Ok as c_int
    } else {
        status
    }
}

/// Open an existing database, reading the schema from the manifest on disk.
///
/// Use this when the schema is not known ahead of time (e.g. DuckDB extension).
/// `rt` must be non-null — same rules as `meru_open`.
///
/// Returns `MERU_ERR_NOT_FOUND` when no catalog exists at `path`.
///
/// # Safety
/// `path` must be a valid non-null UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn meru_open_existing(
    path: *const c_char,
    read_only: u8,
    rt: *mut MeruRuntime,
    db_out: *mut *mut MeruHandle,
    err_out: *mut *mut c_char,
) -> c_int {
    if path.is_null() || db_out.is_null() {
        set_err(err_out, "path and db_out must be non-null");
        return MeruStatus::ErrInvalidArg as c_int;
    }
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_err(err_out, "path is not valid UTF-8");
            return MeruStatus::ErrInvalidArg as c_int;
        }
    };

    let rt = match require_runtime(rt, err_out) {
        Some(r) => r,
        None => return MeruStatus::ErrIo as c_int,
    };

    let schema: TableSchema = match rt
        .block_on(merutable::iceberg::catalog::load_persisted_schema(&path_str))
    {
        Ok(Some(s)) => s,
        Ok(None) => {
            set_err(err_out, &format!("no catalog found at '{path_str}'"));
            return MeruStatus::ErrNotFound as c_int;
        }
        Err(e) => {
            set_err(err_out, &e.to_string());
            return crate::error::error_to_status(&e) as c_int;
        }
    };

    let wal_dir = std::path::PathBuf::from(&path_str).join("wal");
    let open_opts = OpenOptions::new(schema.clone())
        .catalog_uri(&path_str)
        .wal_dir(wal_dir)
        .read_only(read_only != 0);

    let (status, maybe_db) = result_to_c(rt.block_on(MeruDB::open(open_opts)), err_out);

    if let Some(db) = maybe_db {
        *db_out = Box::into_raw(MeruHandle::new(db, schema, rt));
        MeruStatus::Ok as c_int
    } else {
        status
    }
}

/// Graceful shutdown: flushes memtable, fsyncs, and seals.
///
/// After this returns MERU_OK, writes return MERU_ERR_CLOSED; reads still work.
/// Call `meru_free` afterwards, or use `meru_close_free` to do both in one call.
///
/// To also export Iceberg metadata before closing (so DuckDB can read the latest
/// data without an explicit meru_export_iceberg call), call meru_export_iceberg(db, NULL, err)
/// before meru_close().
///
/// # Safety
/// `db` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn meru_close(db: *mut MeruHandle, err_out: *mut *mut c_char) -> c_int {
    let h = handle_mut(db);
    let db_arc = std::sync::Arc::clone(&h.db);
    let (status, _) = result_to_c(h.rt.block_on(db_arc.close()), err_out);
    status
}

/// Free the handle. Does NOT flush — call `meru_close` first.
///
/// # Safety
/// `db` must be a valid non-null handle that was returned by `meru_open` or
/// `meru_open_existing`. After this call `db` is a dangling pointer.
#[no_mangle]
pub unsafe extern "C" fn meru_free(db: *mut MeruHandle) {
    if !db.is_null() {
        drop(Box::from_raw(db));
    }
}

/// Convenience: close then free. The handle is always freed even if close fails.
///
/// # Safety
/// `db` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn meru_close_free(
    db: *mut MeruHandle,
    err_out: *mut *mut c_char,
) -> c_int {
    let status = meru_close(db, err_out);
    meru_free(db);
    status
}

/// Returns 1 if `meru_close` has been called, 0 otherwise.
///
/// # Safety
/// `db` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn meru_is_closed(db: *const MeruHandle) -> c_int {
    let h = handle_ref(db);
    h.db.is_closed() as c_int
}

// ── Writes ────────────────────────────────────────────────────────────────────

/// Insert or update a single row. Returns the sequence number in `*seq_out`.
///
/// # Safety
/// `row` must be a valid MeruRow with `field_count` == schema column count.
#[no_mangle]
pub unsafe extern "C" fn meru_put(
    db: *mut MeruHandle,
    row: *const MeruRow,
    seq_out: *mut u64,
    err_out: *mut *mut c_char,
) -> c_int {
    let h = handle_mut(db);
    let rust_row = match c_row_to_rust(row, &h.schema) {
        Ok(r) => r,
        Err(e) => {
            set_err(err_out, &e.to_string());
            return MeruStatus::ErrInvalidArg as c_int;
        }
    };
    let db_arc = std::sync::Arc::clone(&h.db);
    let (status, seq) = result_to_c(h.rt.block_on(db_arc.put(rust_row)), err_out);
    if let Some(s) = seq {
        if !seq_out.is_null() {
            *seq_out = s.0;
        }
    }
    status
}

/// Batch insert/update. All rows share a single WAL sync.
///
/// # Safety
/// `rows` must point to `row_count` valid MeruRow entries.
#[no_mangle]
pub unsafe extern "C" fn meru_put_batch(
    db: *mut MeruHandle,
    rows: *const MeruRow,
    row_count: usize,
    seq_out: *mut u64,
    err_out: *mut *mut c_char,
) -> c_int {
    let h = handle_mut(db);
    let c_rows = std::slice::from_raw_parts(rows, row_count);
    let mut rust_rows = Vec::with_capacity(row_count);
    for c_row in c_rows {
        match c_row_to_rust(c_row as *const MeruRow, &h.schema) {
            Ok(r) => rust_rows.push(r),
            Err(e) => {
                set_err(err_out, &e.to_string());
                return MeruStatus::ErrInvalidArg as c_int;
            }
        }
    }
    let db_arc = std::sync::Arc::clone(&h.db);
    let (status, seq) = result_to_c(h.rt.block_on(db_arc.put_batch(rust_rows)), err_out);
    if let Some(s) = seq {
        if !seq_out.is_null() {
            *seq_out = s.0;
        }
    }
    status
}

/// Delete a row by primary key. `pk_values` must have `pk_count` entries in PK order.
///
/// # Safety
/// `pk_values` must point to `pk_count` valid MeruValue entries.
#[no_mangle]
pub unsafe extern "C" fn meru_delete(
    db: *mut MeruHandle,
    pk_values: *const MeruValue,
    pk_count: usize,
    seq_out: *mut u64,
    err_out: *mut *mut c_char,
) -> c_int {
    let h = handle_mut(db);
    let fvs = match c_pk_to_rust(pk_values, pk_count, &h.schema) {
        Ok(v) => v,
        Err(e) => {
            set_err(err_out, &e.to_string());
            return MeruStatus::ErrInvalidArg as c_int;
        }
    };
    let db_arc = std::sync::Arc::clone(&h.db);
    let (status, seq) = result_to_c(h.rt.block_on(db_arc.delete(fvs)), err_out);
    if let Some(s) = seq {
        if !seq_out.is_null() {
            *seq_out = s.0;
        }
    }
    status
}

// ── Reads ─────────────────────────────────────────────────────────────────────

/// Point lookup by primary key.
///
/// On success: `*found = 1` and `*row_out` points to a heap-allocated MeruRow
/// (free with `meru_row_free`); or `*found = 0` and `*row_out = NULL` if the
/// key does not exist.
///
/// Returns MERU_OK for both found and not-found; non-zero only on error.
///
/// # Safety
/// `pk_values` must point to `pk_count` valid MeruValue entries.
#[no_mangle]
pub unsafe extern "C" fn meru_get(
    db: *const MeruHandle,
    pk_values: *const MeruValue,
    pk_count: usize,
    found: *mut c_int,
    row_out: *mut *mut MeruRow,
    err_out: *mut *mut c_char,
) -> c_int {
    let h = handle_ref(db);
    let fvs = match c_pk_to_rust(pk_values, pk_count, &h.schema) {
        Ok(v) => v,
        Err(e) => {
            set_err(err_out, &e.to_string());
            return MeruStatus::ErrInvalidArg as c_int;
        }
    };
    match h.db.get(&fvs) {
        Ok(Some(row)) => {
            let c_row = Box::new(rust_row_to_c(&row, &h.schema));
            if !found.is_null() {
                *found = 1;
            }
            if !row_out.is_null() {
                *row_out = Box::into_raw(c_row);
            }
            MeruStatus::Ok as c_int
        }
        Ok(None) => {
            if !found.is_null() {
                *found = 0;
            }
            if !row_out.is_null() {
                *row_out = std::ptr::null_mut();
            }
            MeruStatus::Ok as c_int
        }
        Err(e) => {
            set_err(err_out, &e.to_string());
            crate::error::error_to_status(&e) as c_int
        }
    }
}

/// Range scan. Both bounds optional (pass NULL for open-ended).
///
/// On success `*result_out` points to a heap-allocated MeruScanResult
/// (free with `meru_scan_result_free`).
///
/// # Safety
/// `start_pk` / `end_pk` must each point to their respective count entries, or be NULL.
#[no_mangle]
pub unsafe extern "C" fn meru_scan(
    db: *const MeruHandle,
    start_pk: *const MeruValue,
    start_count: usize,
    end_pk: *const MeruValue,
    end_count: usize,
    result_out: *mut *mut MeruScanResult,
    err_out: *mut *mut c_char,
) -> c_int {
    let h = handle_ref(db);

    let start_fvs = if start_pk.is_null() {
        None
    } else {
        match c_pk_to_rust(start_pk, start_count, &h.schema) {
            Ok(v) => Some(v),
            Err(e) => {
                set_err(err_out, &e.to_string());
                return MeruStatus::ErrInvalidArg as c_int;
            }
        }
    };
    let end_fvs = if end_pk.is_null() {
        None
    } else {
        match c_pk_to_rust(end_pk, end_count, &h.schema) {
            Ok(v) => Some(v),
            Err(e) => {
                set_err(err_out, &e.to_string());
                return MeruStatus::ErrInvalidArg as c_int;
            }
        }
    };

    match h
        .db
        .scan(start_fvs.as_deref(), end_fvs.as_deref())
    {
        Ok(pairs) => {
            let mut entries: Vec<MeruRow> = pairs
                .iter()
                .map(|(_, row)| rust_row_to_c(row, &h.schema))
                .collect();
            let count = entries.len();
            let ptr = entries.as_mut_ptr();
            std::mem::forget(entries);
            let scan_result = Box::new(MeruScanResult {
                entries: ptr,
                count,
            });
            if !result_out.is_null() {
                *result_out = Box::into_raw(scan_result);
            }
            MeruStatus::Ok as c_int
        }
        Err(e) => {
            set_err(err_out, &e.to_string());
            crate::error::error_to_status(&e) as c_int
        }
    }
}

// ── Maintenance ───────────────────────────────────────────────────────────────

/// Force flush in-memory data to Parquet.
///
/// # Safety
/// `db` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn meru_flush(db: *mut MeruHandle, err_out: *mut *mut c_char) -> c_int {
    let h = handle_mut(db);
    let db_arc = std::sync::Arc::clone(&h.db);
    let (status, _) = result_to_c(h.rt.block_on(db_arc.flush()), err_out);
    status
}

/// Trigger manual compaction.
///
/// # Safety
/// `db` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn meru_compact(db: *mut MeruHandle, err_out: *mut *mut c_char) -> c_int {
    let h = handle_mut(db);
    let db_arc = std::sync::Arc::clone(&h.db);
    let (status, _) = result_to_c(h.rt.block_on(db_arc.compact()), err_out);
    status
}

/// Re-read the Iceberg manifest from disk. Only meaningful for read-only replicas.
///
/// # Safety
/// `db` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn meru_refresh(db: *mut MeruHandle, err_out: *mut *mut c_char) -> c_int {
    let h = handle_mut(db);
    let db_arc = std::sync::Arc::clone(&h.db);
    let (status, _) = result_to_c(h.rt.block_on(db_arc.refresh()), err_out);
    status
}

/// Write Iceberg metadata.json to `target_dir` (or in-place when `target_dir = NULL`).
///
/// `meru_close` calls this automatically. Use explicitly to export mid-session for
/// a DuckDB extension to pick up new data without closing the database.
///
/// # Safety
/// `target_dir` must be a valid UTF-8 C string or NULL.
#[no_mangle]
pub unsafe extern "C" fn meru_export_iceberg(
    db: *mut MeruHandle,
    target_dir: *const c_char,
    err_out: *mut *mut c_char,
) -> c_int {
    let h = handle_mut(db);
    let dir = if target_dir.is_null() {
        None
    } else {
        match CStr::from_ptr(target_dir).to_str() {
            Ok(s) => Some(std::path::PathBuf::from(s)),
            Err(_) => {
                set_err(err_out, "target_dir is not valid UTF-8");
                return MeruStatus::ErrInvalidArg as c_int;
            }
        }
    };

    let db_arc = std::sync::Arc::clone(&h.db);
    let (status, _) = result_to_c(
        h.rt.block_on(async {
            db_arc.export_iceberg(dir.as_deref().unwrap_or_else(|| std::path::Path::new("."))).await
        }),
        err_out,
    );
    status
}

// ── Introspection ─────────────────────────────────────────────────────────────

/// Fill `*stats_out` with an engine statistics snapshot.
///
/// # Safety
/// `db` and `stats_out` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn meru_stats(
    db: *const MeruHandle,
    stats_out: *mut MeruStats,
    _err_out: *mut *mut c_char,
) -> c_int {
    let h = handle_ref(db);
    let s = h.db.stats();
    *stats_out = MeruStats {
        snapshot_id: s.snapshot_id,
        current_seq: s.current_seq,
        memtable: MeruMemtableStats {
            active_size_bytes: s.memtable.active_size_bytes,
            active_entry_count: s.memtable.active_entry_count,
            flush_threshold: s.memtable.flush_threshold,
            immutable_count: s.memtable.immutable_count,
        },
        cache: MerucacheStats {
            capacity: s.cache.capacity,
            size: s.cache.size,
            hit_count: s.cache.hit_count,
            miss_count: s.cache.miss_count,
        },
    };
    MeruStatus::Ok as c_int
}

/// Return the catalog base directory path as a heap-allocated C string.
/// Free with `meru_free_string`.
///
/// # Safety
/// `db` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn meru_catalog_path(db: *const MeruHandle) -> *mut c_char {
    let h = handle_ref(db);
    let s = h.db.catalog_path();
    CString::new(s)
        .map(|cs| cs.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

// ── Memory helpers ────────────────────────────────────────────────────────────

/// Free a MeruRow returned by `meru_get`. Safe to call with NULL.
///
/// # Safety
/// `row` must have been allocated by the merutable C API.
#[no_mangle]
pub unsafe extern "C" fn meru_row_free(row: *mut MeruRow) {
    if row.is_null() {
        return;
    }
    free_c_row_fields(&mut *row);
    drop(Box::from_raw(row));
}

/// Free a MeruScanResult returned by `meru_scan`. Safe to call with NULL.
///
/// # Safety
/// `result` must have been allocated by the merutable C API.
#[no_mangle]
pub unsafe extern "C" fn meru_scan_result_free(result: *mut MeruScanResult) {
    if result.is_null() {
        return;
    }
    let result = &mut *result;
    if !result.entries.is_null() && result.count > 0 {
        let rows = std::slice::from_raw_parts_mut(result.entries, result.count);
        for row in rows.iter_mut() {
            free_c_row_fields(row);
        }
        let _owned = Vec::from_raw_parts(result.entries, result.count, result.count);
    }
    drop(Box::from_raw(result as *mut MeruScanResult));
}

// ── Manifest inspection ───────────────────────────────────────────────────────

/// Read the manifest from a catalog at `path` without opening a write handle.
///
/// On success fills `*out` with a heap-allocated `MeruManifestInfo` containing
/// the table schema and absolute paths of all live (non-deleted) Parquet files.
/// Returns `MERU_STATUS_ERR_NOT_FOUND` when no catalog exists at `path`.
/// The caller must free the result with `meru_manifest_info_free`.
///
/// # Safety
/// `rt` must be non-null. `path`, `out`, and `err_out` follow the same
/// pointer rules as `meru_open`.
#[no_mangle]
pub unsafe extern "C" fn meru_manifest_info(
    rt: *mut MeruRuntime,
    path: *const c_char,
    out: *mut *mut MeruManifestInfo,
    err_out: *mut *mut c_char,
) -> c_int {
    if path.is_null() || out.is_null() {
        set_err(err_out, "path and out must be non-null");
        return MeruStatus::ErrInvalidArg as c_int;
    }
    let rt = match require_runtime(rt, err_out) {
        Some(r) => r,
        None => return MeruStatus::ErrIo as c_int,
    };
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => {
            set_err(err_out, "path is not valid UTF-8");
            return MeruStatus::ErrInvalidArg as c_int;
        }
    };

    let result = rt.block_on(async {
        let base = std::path::Path::new(&path_str);
        let hint_path = base.join("version-hint.text");
        if !hint_path.exists() {
            return Err(merutable::types::MeruError::NotFound);
        }
        let hint = tokio::fs::read_to_string(&hint_path)
            .await
            .map_err(merutable::types::MeruError::Io)?;
        let ver: i64 = hint.trim().parse().map_err(|_| {
            merutable::types::MeruError::Corruption("bad version-hint".into())
        })?;

        let metadata_dir = base.join("metadata");
        let pb_path = metadata_dir.join(format!("v{ver}.metadata.pb"));
        let data = if pb_path.exists() {
            tokio::fs::read(&pb_path).await.map_err(merutable::types::MeruError::Io)?
        } else {
            let json_path = metadata_dir.join(format!("v{ver}.metadata.json"));
            tokio::fs::read(&json_path).await.map_err(merutable::types::MeruError::Io)?
        };

        let manifest = if data.len() >= 4 && &data[0..4] == b"MRUB" {
            Manifest::from_protobuf(&data)?
        } else {
            Manifest::from_json(&data)?
        };

        // Canonicalize base path once so all parquet paths are absolute.
        let abs_base = tokio::fs::canonicalize(base)
            .await
            .unwrap_or_else(|_| base.to_path_buf());
        let abs_base_str = abs_base.to_string_lossy().into_owned();

        Ok((manifest, abs_base_str))
    });

    let (manifest, abs_base) = match result {
        Ok(v) => v,
        Err(e) => {
            set_err(err_out, &e.to_string());
            return crate::error::error_to_status(&e) as c_int;
        }
    };

    // Build the C struct.
    let table_name = CString::new(manifest.schema.table_name.as_str())
        .unwrap_or_default()
        .into_raw();

    let mut cols: Vec<MeruColumnDef> = manifest
        .schema
        .columns
        .iter()
        .filter(|c| {
            !matches!(
                c.name.as_str(),
                IKEY_COLUMN_NAME | SEQ_COLUMN_NAME | OP_COLUMN_NAME | VALUE_BLOB_COLUMN_NAME
            )
        })
        .map(rust_col_def_to_c)
        .collect();
    let column_count = cols.len();
    let columns = cols.as_mut_ptr();
    std::mem::forget(cols);

    let mut pk: Vec<usize> = manifest.schema.primary_key.clone();
    let pk_count = pk.len();
    let primary_key = pk.as_mut_ptr();
    std::mem::forget(pk);

    let live_paths: Vec<*mut c_char> = manifest
        .entries
        .iter()
        .filter(|e| e.status != "deleted")
        .map(|e| {
            let abs = format!("{abs_base}/{}", e.path);
            CString::new(abs).unwrap_or_default().into_raw()
        })
        .collect();
    let parquet_count = live_paths.len();
    let mut paths_vec = live_paths;
    let parquet_paths = paths_vec.as_mut_ptr();
    std::mem::forget(paths_vec);

    let info = Box::new(MeruManifestInfo {
        table_name,
        columns,
        column_count,
        primary_key,
        pk_count,
        parquet_paths,
        parquet_count,
    });
    *out = Box::into_raw(info);
    MeruStatus::Ok as c_int
}

/// Free a `MeruManifestInfo` returned by `meru_manifest_info`. Safe to call with NULL.
///
/// # Safety
/// `info` must have been returned by `meru_manifest_info`.
#[no_mangle]
pub unsafe extern "C" fn meru_manifest_info_free(info: *mut MeruManifestInfo) {
    if info.is_null() {
        return;
    }
    let info = &mut *info;

    drop(CString::from_raw(info.table_name));

    if !info.columns.is_null() && info.column_count > 0 {
        let cols = std::slice::from_raw_parts_mut(info.columns, info.column_count);
        for col in cols.iter_mut() {
            if !col.name.is_null() {
                drop(CString::from_raw(col.name as *mut c_char));
            }
        }
        drop(Box::from_raw(std::slice::from_raw_parts_mut(
            info.columns,
            info.column_count,
        )));
    }

    if !info.primary_key.is_null() && info.pk_count > 0 {
        drop(Box::from_raw(std::slice::from_raw_parts_mut(
            info.primary_key,
            info.pk_count,
        )));
    }

    if !info.parquet_paths.is_null() && info.parquet_count > 0 {
        let paths = std::slice::from_raw_parts_mut(info.parquet_paths, info.parquet_count);
        for p in paths.iter_mut() {
            if !p.is_null() {
                drop(CString::from_raw(*p));
            }
        }
        drop(Box::from_raw(std::slice::from_raw_parts_mut(
            info.parquet_paths,
            info.parquet_count,
        )));
    }

    drop(Box::from_raw(info as *mut MeruManifestInfo));
}
