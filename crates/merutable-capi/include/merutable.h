#ifndef MERUTABLE_H
#define MERUTABLE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Column type tag. Maps to merutable's internal ColumnType.
 */
typedef enum {
    MeruColumnType_Boolean = 0,
    MeruColumnType_Int32 = 1,
    MeruColumnType_Int64 = 2,
    MeruColumnType_Float = 3,
    MeruColumnType_Double = 4,
    MeruColumnType_ByteArray = 5,
    MeruColumnType_FixedLenBytes = 6,
} MeruColumnType;

/**
 * Status codes returned by all fallible C API functions.
 */
typedef enum {
    MeruStatus_Ok = 0,
    MeruStatus_ErrIo = 1,
    MeruStatus_ErrCorruption = 2,
    MeruStatus_ErrSchemaMismatch = 3,
    MeruStatus_ErrNotFound = 4,
    MeruStatus_ErrInvalidArg = 5,
    MeruStatus_ErrReadOnly = 6,
    MeruStatus_ErrClosed = 7,
    MeruStatus_ErrObjectStore = 8,
    MeruStatus_ErrParquet = 9,
    MeruStatus_ErrAlreadyExists = 10,
    MeruStatus_ErrUnknown = 99,
} MeruStatus;

/**
 * Opaque handle. C callers hold `*mut MeruHandle`.
 *
 * The runtime is Arc-shared so multiple handles can be opened on a single
 * MeruRuntime (one thread pool serving all of them).
 */
typedef struct MeruHandle MeruHandle;

/**
 * Opaque shareable tokio runtime. Pass to meru_open() so multiple databases
 * share one thread pool. NULL is accepted everywhere and creates a
 * per-handle runtime instead.
 */
typedef struct MeruRuntime MeruRuntime;

/**
 * Pointer + length into a byte buffer. Used inside MeruValue.
 */
typedef struct {
    const uint8_t *data;
    uintptr_t len;
} MeruBytesView;

/**
 * The inner union of MeruValue. Do not use directly; use MeruValue.
 */
typedef union {
    uint8_t v_bool;
    int32_t v_int32;
    int64_t v_int64;
    float v_float;
    double v_double;
    MeruBytesView v_bytes;
} MeruValueInner;

/**
 * A single nullable field value.
 *
 * `is_null = 1` means SQL NULL; the union contents are undefined.
 * For BYTE_ARRAY / FIXED_LEN_BYTES: when this value is returned by the API
 * (meru_get, meru_scan), `v_bytes.data` is owned by the enclosing MeruRow
 * and is freed when `meru_row_free` / `meru_scan_result_free` is called.
 * Do NOT cache `v_bytes.data` past that call.
 *
 * When passing a value *into* the API (meru_put, meru_delete), the byte
 * buffer is read and copied before the call returns; the caller retains
 * ownership.
 */
typedef struct {
    MeruColumnType tag;
    uint8_t is_null;
    uint8_t _pad[3];
    MeruValueInner inner;
} MeruValue;

/**
 * Column definition used to declare a schema in MeruOpenOptions.
 *
 * `initial_default` and `write_default` may be NULL (meaning no default).
 */
typedef struct {
    const char *name;
    MeruColumnType col_type;
    /**
     * Only meaningful when col_type == MERU_COLUMN_FIXED_LEN_BYTES.
     */
    int32_t fixed_byte_len;
    /**
     * 0 = NOT NULL, 1 = nullable.
     */
    uint8_t nullable;
    const MeruValue *initial_default;
    const MeruValue *write_default;
} MeruColumnDef;

/**
 * Table schema: column definitions + primary key column indices.
 */
typedef struct {
    const char *table_name;
    const MeruColumnDef *columns;
    uintptr_t column_count;
    /**
     * Array of column indices (into `columns`) that form the primary key.
     */
    const uintptr_t *primary_key;
    uintptr_t primary_key_len;
} MeruSchema;

/**
 * Options for meru_open(). Pass wal_dir = NULL to default to "{catalog_uri}/wal".
 */
typedef struct {
    MeruSchema schema;
    const char *catalog_uri;
    const char *wal_dir;
    /**
     * 0 = use engine default (64 MiB).
     */
    uintptr_t memtable_size_mb;
    /**
     * 1 = open read-only.
     */
    uint8_t read_only;
} MeruOpenOptions;

/**
 * A database row: an array of `field_count` MeruValue entries.
 * Field order matches the schema column order.
 */
typedef struct {
    MeruValue *fields;
    uintptr_t field_count;
} MeruRow;

/**
 * Array of rows returned by meru_scan(). Free with meru_scan_result_free().
 */
typedef struct {
    MeruRow *entries;
    uintptr_t count;
} MeruScanResult;

/**
 * Memtable statistics snapshot.
 */
typedef struct {
    uintptr_t active_size_bytes;
    uint64_t active_entry_count;
    uintptr_t flush_threshold;
    uintptr_t immutable_count;
} MeruMemtableStats;

/**
 * Row cache statistics snapshot.
 */
typedef struct {
    uintptr_t capacity;
    uintptr_t size;
    uint64_t hit_count;
    uint64_t miss_count;
} MeruCacheStats;

/**
 * Engine statistics snapshot. Filled by meru_stats().
 */
typedef struct {
    int64_t snapshot_id;
    uint64_t current_seq;
    MeruMemtableStats memtable;
    MeruCacheStats cache;
} MeruStats;

/**
 * Manifest snapshot returned by `meru_manifest_info`.
 * Free with `meru_manifest_info_free`.
 */
typedef struct {
    /**
     * Heap-allocated table name. Free via `meru_free_string`.
     */
    char *table_name;
    /**
     * Heap-allocated array of column definitions (`column_count` entries).
     */
    MeruColumnDef *columns;
    uintptr_t column_count;
    /**
     * Heap-allocated array of primary-key column indices (`pk_count` entries).
     */
    uintptr_t *primary_key;
    uintptr_t pk_count;
    /**
     * Heap-allocated array of heap-allocated absolute path strings (`parquet_count` entries).
     */
    char **parquet_paths;
    uintptr_t parquet_count;
} MeruManifestInfo;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Open (or create) a database.
 *
 * `rt` must be a non-null runtime from `meru_runtime_new()`. Multiple databases
 * can share one runtime — they will share its thread pool.
 *
 * Calling meru_* functions from a coroutine/async task: dispatch to a
 * blockable thread (e.g. ASIO post to a thread_pool_executor, or
 * std::thread) rather than calling directly from your executor's event loop
 * thread, which may deadlock.
 *
 * `opts->wal_dir = NULL` defaults to `"{catalog_uri}/wal"`.
 *
 * # Safety
 * All pointer fields in `opts` must be valid non-null C strings (except `wal_dir`).
 */
extern
int meru_open(const MeruOpenOptions *opts,
              MeruRuntime *rt,
              MeruHandle **db_out,
              char **err_out);

/**
 * Open an existing database, reading the schema from the manifest on disk.
 *
 * Use this when the schema is not known ahead of time (e.g. DuckDB extension).
 * `rt` must be non-null — same rules as `meru_open`.
 *
 * Returns `MERU_ERR_NOT_FOUND` when no catalog exists at `path`.
 *
 * # Safety
 * `path` must be a valid non-null UTF-8 C string.
 */
extern
int meru_open_existing(const char *path,
                       uint8_t read_only,
                       MeruRuntime *rt,
                       MeruHandle **db_out,
                       char **err_out);

/**
 * Graceful shutdown: flushes memtable, fsyncs, and seals.
 *
 * After this returns MERU_OK, writes return MERU_ERR_CLOSED; reads still work.
 * Call `meru_free` afterwards, or use `meru_close_free` to do both in one call.
 *
 * To also export Iceberg metadata before closing (so DuckDB can read the latest
 * data without an explicit meru_export_iceberg call), call meru_export_iceberg(db, NULL, err)
 * before meru_close().
 *
 * # Safety
 * `db` must be a valid non-null handle.
 */
extern int meru_close(MeruHandle *db, char **err_out);

/**
 * Free the handle. Does NOT flush — call `meru_close` first.
 *
 * # Safety
 * `db` must be a valid non-null handle that was returned by `meru_open` or
 * `meru_open_existing`. After this call `db` is a dangling pointer.
 */
extern void meru_free(MeruHandle *db);

/**
 * Convenience: close then free. The handle is always freed even if close fails.
 *
 * # Safety
 * `db` must be a valid non-null handle.
 */
extern int meru_close_free(MeruHandle *db, char **err_out);

/**
 * Returns 1 if `meru_close` has been called, 0 otherwise.
 *
 * # Safety
 * `db` must be a valid non-null handle.
 */
extern int meru_is_closed(const MeruHandle *db);

/**
 * Insert or update a single row. Returns the sequence number in `*seq_out`.
 *
 * # Safety
 * `row` must be a valid MeruRow with `field_count` == schema column count.
 */
extern int meru_put(MeruHandle *db, const MeruRow *row, uint64_t *seq_out, char **err_out);

/**
 * Batch insert/update. All rows share a single WAL sync.
 *
 * # Safety
 * `rows` must point to `row_count` valid MeruRow entries.
 */
extern
int meru_put_batch(MeruHandle *db,
                   const MeruRow *rows,
                   uintptr_t row_count,
                   uint64_t *seq_out,
                   char **err_out);

/**
 * Delete a row by primary key. `pk_values` must have `pk_count` entries in PK order.
 *
 * # Safety
 * `pk_values` must point to `pk_count` valid MeruValue entries.
 */
extern
int meru_delete(MeruHandle *db,
                const MeruValue *pk_values,
                uintptr_t pk_count,
                uint64_t *seq_out,
                char **err_out);

/**
 * Point lookup by primary key.
 *
 * On success: `*found = 1` and `*row_out` points to a heap-allocated MeruRow
 * (free with `meru_row_free`); or `*found = 0` and `*row_out = NULL` if the
 * key does not exist.
 *
 * Returns MERU_OK for both found and not-found; non-zero only on error.
 *
 * # Safety
 * `pk_values` must point to `pk_count` valid MeruValue entries.
 */
extern
int meru_get(const MeruHandle *db,
             const MeruValue *pk_values,
             uintptr_t pk_count,
             int *found,
             MeruRow **row_out,
             char **err_out);

/**
 * Range scan. Both bounds optional (pass NULL for open-ended).
 *
 * On success `*result_out` points to a heap-allocated MeruScanResult
 * (free with `meru_scan_result_free`).
 *
 * # Safety
 * `start_pk` / `end_pk` must each point to their respective count entries, or be NULL.
 */
extern
int meru_scan(const MeruHandle *db,
              const MeruValue *start_pk,
              uintptr_t start_count,
              const MeruValue *end_pk,
              uintptr_t end_count,
              MeruScanResult **result_out,
              char **err_out);

/**
 * Force flush in-memory data to Parquet.
 *
 * # Safety
 * `db` must be a valid non-null handle.
 */
extern int meru_flush(MeruHandle *db, char **err_out);

/**
 * Trigger manual compaction.
 *
 * # Safety
 * `db` must be a valid non-null handle.
 */
extern int meru_compact(MeruHandle *db, char **err_out);

/**
 * Re-read the Iceberg manifest from disk. Only meaningful for read-only replicas.
 *
 * # Safety
 * `db` must be a valid non-null handle.
 */
extern int meru_refresh(MeruHandle *db, char **err_out);

/**
 * Write Iceberg metadata.json to `target_dir` (or in-place when `target_dir = NULL`).
 *
 * `meru_close` calls this automatically. Use explicitly to export mid-session for
 * a DuckDB extension to pick up new data without closing the database.
 *
 * # Safety
 * `target_dir` must be a valid UTF-8 C string or NULL.
 */
extern int meru_export_iceberg(MeruHandle *db, const char *target_dir, char **err_out);

/**
 * Fill `*stats_out` with an engine statistics snapshot.
 *
 * # Safety
 * `db` and `stats_out` must be valid non-null pointers.
 */
extern int meru_stats(const MeruHandle *db, MeruStats *stats_out, char **_err_out);

/**
 * Return the catalog base directory path as a heap-allocated C string.
 * Free with `meru_free_string`.
 *
 * # Safety
 * `db` must be a valid non-null handle.
 */
extern char *meru_catalog_path(const MeruHandle *db);

/**
 * Free a MeruRow returned by `meru_get`. Safe to call with NULL.
 *
 * # Safety
 * `row` must have been allocated by the merutable C API.
 */
extern void meru_row_free(MeruRow *row);

/**
 * Free a MeruScanResult returned by `meru_scan`. Safe to call with NULL.
 *
 * # Safety
 * `result` must have been allocated by the merutable C API.
 */
extern void meru_scan_result_free(MeruScanResult *result);

/**
 * Read the manifest from a catalog at `path` without opening a write handle.
 *
 * On success fills `*out` with a heap-allocated `MeruManifestInfo` containing
 * the table schema and absolute paths of all live (non-deleted) Parquet files.
 * Returns `MERU_STATUS_ERR_NOT_FOUND` when no catalog exists at `path`.
 * The caller must free the result with `meru_manifest_info_free`.
 *
 * # Safety
 * `rt` must be non-null. `path`, `out`, and `err_out` follow the same
 * pointer rules as `meru_open`.
 */
extern
int meru_manifest_info(MeruRuntime *rt,
                       const char *path,
                       MeruManifestInfo **out,
                       char **err_out);

/**
 * Free a `MeruManifestInfo` returned by `meru_manifest_info`. Safe to call with NULL.
 *
 * # Safety
 * `info` must have been returned by `meru_manifest_info`.
 */
extern void meru_manifest_info_free(MeruManifestInfo *info);

/**
 * Free a C string returned by the API (catalog_path, error messages).
 *
 * # Safety
 * `s` must have been allocated by the merutable C API. Passing NULL is safe.
 */
extern void meru_free_string(char *s);

/**
 * Create a new multi-thread tokio runtime.
 *
 * `worker_threads = 0` uses the tokio default (one thread per logical CPU).
 *
 * Returns NULL on failure; `*err_out` is set when err_out is non-null.
 *
 * # Safety
 * `err_out` must be null or a valid pointer to a `char *`.
 */
extern MeruRuntime *meru_runtime_new(uintptr_t worker_threads, char **err_out);

/**
 * Free a MeruRuntime. Safe to call with NULL.
 *
 * The underlying thread pool is shut down when the last handle that shares
 * this runtime is also freed.
 *
 * # Safety
 * `rt` must have been returned by `meru_runtime_new`.
 */
extern void meru_runtime_free(MeruRuntime *rt);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* MERUTABLE_H */
