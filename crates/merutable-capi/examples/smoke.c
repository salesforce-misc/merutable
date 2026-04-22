/*
 * Smoke test for the merutable C API.
 *
 * Exercises the full lifecycle: open → put → get → scan → stats →
 * catalog_path → close → reopen (read-only) → close.
 *
 * The database directory is read from the MERU_SMOKE_DB environment variable,
 * defaulting to /tmp/meru_smoke_test when unset.
 *
 * Manual compile + run (from this directory):
 *   REPO=../../..
 *   cc smoke.c -I../include -L${REPO}/target/debug -lmerutable_capi \
 *       -Wl,-rpath,${REPO}/target/debug -o smoke
 *   MERU_SMOKE_DB=/tmp/my_test_db ./smoke
 *
 * Via cargo nextest (runs automatically as part of the crate's test suite):
 *   cargo nextest run -p merutable-capi --test c_smoke
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/merutable.h"

/* Print a failure message and exit non-zero. Frees err if non-NULL. */
static void die(const char *ctx, int status, char *err) {
    fprintf(stderr, "FAIL %s: status=%d err=%s\n", ctx, status,
            err ? err : "(none)");
    if (err) meru_free_string(err);
    exit(1);
}

int main(void) {
    char *err = NULL;
    int status;

    /* ── Determine database path ─────────────────────────────────────── */
    const char *db_path = getenv("MERU_SMOKE_DB");
    if (!db_path) db_path = "/tmp/meru_smoke_test";

    /* ── Schema ──────────────────────────────────────────────────────── */
    /*
     * Three-column table: id (int64, PK), name (ByteArray, nullable),
     * score (double, nullable).
     *
     * MeruColumnDef fields: name, col_type, fixed_byte_len, nullable,
     *                        initial_default, write_default.
     * initial_default / write_default = NULL means "no default".
     */
    MeruColumnDef cols[] = {
        { "id",    MeruColumnType_Int64,     0, 0, NULL, NULL },
        { "name",  MeruColumnType_ByteArray, 0, 1, NULL, NULL },
        { "score", MeruColumnType_Double,    0, 1, NULL, NULL },
    };
    size_t pk[] = { 0 };   /* column index 0 ("id") is the primary key */

    MeruOpenOptions opts = {
        .schema = {
            .table_name      = "events",
            .columns         = cols,
            .column_count    = 3,
            .primary_key     = pk,
            .primary_key_len = 1,
        },
        .catalog_uri      = db_path,
        .wal_dir          = NULL,  /* NULL → default: "{db_path}/wal" */
        .memtable_size_mb = 0,     /* 0 → engine default (64 MiB)     */
        .read_only        = 0,
    };

    /* ── Open ────────────────────────────────────────────────────────── */
    MeruHandle *db = NULL;
    status = meru_open(&opts, &db, &err);
    if (status != MeruStatus_Ok) die("meru_open", status, err);
    printf("opened db at %s\n", db_path);

    /* ── Put a row ───────────────────────────────────────────────────── */
    /*
     * Build a 3-element MeruValue array in column order.
     *
     * For byte-array input values (name), v_bytes.data points into a
     * stack string — the API copies the bytes before returning so this
     * is safe.
     */
    const char *name_str = "alice";
    MeruValue fields[3];

    /* id = 42 (NOT NULL, so is_null must be 0) */
    fields[0].tag    = MeruColumnType_Int64;
    fields[0].is_null = 0;
    memset(fields[0]._pad, 0, 3);
    fields[0].inner.v_int64 = 42;

    /* name = "alice" */
    fields[1].tag    = MeruColumnType_ByteArray;
    fields[1].is_null = 0;
    memset(fields[1]._pad, 0, 3);
    fields[1].inner.v_bytes.data = (const uint8_t *)name_str;
    fields[1].inner.v_bytes.len  = strlen(name_str);

    /* score = 0.95 */
    fields[2].tag    = MeruColumnType_Double;
    fields[2].is_null = 0;
    memset(fields[2]._pad, 0, 3);
    fields[2].inner.v_double = 0.95;

    MeruRow row = { fields, 3 };
    uint64_t seq = 0;
    status = meru_put(db, &row, &seq, &err);
    if (status != MeruStatus_Ok) die("meru_put", status, err);
    printf("put row seq=%llu\n", (unsigned long long)seq);

    /* ── Get it back ─────────────────────────────────────────────────── */
    /*
     * meru_get takes PK values (one per PK column).
     * On success: *found=1 and *row_out is a heap-allocated MeruRow.
     * Caller must free with meru_row_free().
     */
    MeruValue pk_val;
    pk_val.tag    = MeruColumnType_Int64;
    pk_val.is_null = 0;
    memset(pk_val._pad, 0, 3);
    pk_val.inner.v_int64 = 42;

    int found = 0;
    MeruRow *got = NULL;
    status = meru_get(db, &pk_val, 1, &found, &got, &err);
    if (status != MeruStatus_Ok) die("meru_get", status, err);
    if (!found) { fprintf(stderr, "FAIL: row not found after put\n"); exit(1); }

    /* Verify returned values */
    if (got->fields[0].inner.v_int64 != 42) {
        fprintf(stderr, "FAIL: id mismatch (got %lld)\n",
                (long long)got->fields[0].inner.v_int64);
        exit(1);
    }
    if (got->fields[2].inner.v_double != 0.95) {
        fprintf(stderr, "FAIL: score mismatch (got %g)\n",
                got->fields[2].inner.v_double);
        exit(1);
    }
    MeruBytesView bv = got->fields[1].inner.v_bytes;
    printf("got row: id=%lld name=%.*s score=%g\n",
           (long long)got->fields[0].inner.v_int64,
           (int)bv.len, (const char *)bv.data,
           got->fields[2].inner.v_double);
    meru_row_free(got);   /* frees row + all owned byte buffers */

    /* ── Scan (open-ended) ───────────────────────────────────────────── */
    /*
     * NULL start/end means unbounded range. Returns all rows in PK order.
     * Result must be freed with meru_scan_result_free().
     */
    MeruScanResult *scan = NULL;
    status = meru_scan(db, NULL, 0, NULL, 0, &scan, &err);
    if (status != MeruStatus_Ok) die("meru_scan", status, err);
    if (scan->count != 1) {
        fprintf(stderr, "FAIL: expected 1 row from scan, got %zu\n", scan->count);
        exit(1);
    }
    printf("scan returned %zu row(s)\n", scan->count);
    meru_scan_result_free(scan);

    /* ── Stats ───────────────────────────────────────────────────────── */
    MeruStats stats;
    meru_stats(db, &stats, NULL);
    printf("stats: seq=%llu snapshot=%lld memtable_entries=%llu\n",
           (unsigned long long)stats.current_seq,
           (long long)stats.snapshot_id,
           (unsigned long long)stats.memtable.active_entry_count);

    /* ── Catalog path ────────────────────────────────────────────────── */
    /*
     * Returns a heap-allocated C string. Free with meru_free_string().
     * DuckDB extensions can point at {catalog_path}/data/L1/*.parquet.
     */
    char *cpath = meru_catalog_path(db);
    printf("catalog path: %s\n", cpath);
    meru_free_string(cpath);

    /* ── Close + free ────────────────────────────────────────────────── */
    /*
     * meru_close_free is the normal teardown path: flushes, fsyncs, seals,
     * then frees the handle. The handle pointer is invalid after this call.
     *
     * To export Iceberg metadata.json before closing (so DuckDB can read
     * the latest data), call meru_export_iceberg(db, NULL, &err) first.
     */
    status = meru_close_free(db, &err);
    if (status != MeruStatus_Ok) die("meru_close_free", status, err);
    printf("closed ok\n");

    /* ── Reopen with schema inferred from disk ───────────────────────── */
    /*
     * meru_open_existing reads the TableSchema from the manifest on disk
     * so the caller does not need to re-supply it. Useful for tooling and
     * the DuckDB extension where the schema is not known ahead of time.
     */
    MeruHandle *db2 = NULL;
    status = meru_open_existing(db_path, /*read_only=*/1, &db2, &err);
    if (status != MeruStatus_Ok) die("meru_open_existing", status, err);
    printf("reopened (read-only) with meru_open_existing\n");
    meru_close_free(db2, NULL);

    printf("smoke test PASSED\n");
    return 0;
}
