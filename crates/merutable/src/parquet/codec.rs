//! Arrow `RecordBatch` ↔ `(InternalKey, Row)` conversion.
//!
//! Used in the flush pipeline:
//!   memtable entries → `rows_to_record_batch` → Parquet writer
//! And in the read pipeline:
//!   Parquet reader → `record_batch_to_rows` → engine merge iterator
//!
//! The Parquet schema produced here is:
//!   - Column `_merutable_ikey` (Binary, required): full encoded InternalKey bytes
//!   - One column per `TableSchema::columns` entry, in definition order
//!
//! The `_merutable_ikey` column is hidden from the public API but is the
//! sort key and bloom filter target for every Parquet file.

use std::sync::Arc;

use crate::types::{
    key::InternalKey,
    level::{FileFormat, Level},
    schema::{ColumnType, TableSchema},
    value::{FieldValue, Row},
    MeruError, Result,
};
use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, FixedSizeBinaryArray, Float32Array, Float64Array,
    Int32Array, Int64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;

/// Hidden column carrying the encoded `InternalKey` for every row. Always
/// present, always column 0. Sort key + bloom filter target + kv_index target.
pub const IKEY_COLUMN_NAME: &str = "_merutable_ikey";

/// Hidden column carrying a postcard-encoded `Row` blob. Present at L0 only
/// (the hot tier where point lookups dominate and want a single
/// column-chunk decode), absent at L1+ (cold analytics tier — typed columns
/// are sufficient and reading a redundant blob would waste bytes).
pub const VALUE_BLOB_COLUMN_NAME: &str = "_merutable_value";

/// Issue #16: typed Int64 column carrying the sequence number of each
/// row. Always present. External analytics readers use this directly
/// in the mandatory MVCC dedup projection
/// (`ROW_NUMBER() OVER (PARTITION BY pk ORDER BY _merutable_seq DESC)`)
/// without decoding the `_merutable_ikey` trailer by hand. Encoded as
/// DELTA_BINARY_PACKED at the Parquet layer (near-zero on-disk cost
/// because writes are sequential-ish).
pub const SEQ_COLUMN_NAME: &str = "_merutable_seq";

/// Issue #16: typed Int32 column carrying the op-type of each row
/// (`0 = Delete`, `1 = Put` — matches the on-disk tag byte in
/// InternalKey). Always present. External analytics readers filter
/// on `_merutable_op = 1` to drop tombstones. RLE-encoded at the
/// Parquet layer (effectively free; one byte per run of consecutive
/// same-op rows).
pub const OP_COLUMN_NAME: &str = "_merutable_op";

/// Whether files at the given level carry a `_merutable_value` blob column.
/// Legacy helper — retained for callers that haven't plumbed the
/// explicit `FileFormat` through yet. Delegates to
/// `FileFormat::default_for_level`, so the per-level default matches
/// the pre-Issue-#15 hard-coded behavior.
///
/// Prefer `FileFormat::has_value_blob` directly at new call sites.
#[inline]
pub fn level_has_value_blob(level: Level) -> bool {
    FileFormat::default_for_level(level).has_value_blob()
}

/// Build the Arrow schema for a given `TableSchema` and target format.
///
/// Layout:
/// - Column 0: `_merutable_ikey` (always)
/// - Column 1: `_merutable_value` (FileFormat::Dual only)
/// - Remaining: one typed column per `TableSchema::columns` entry, in
///   schema order. These are the columns external external analytical readers see.
pub fn arrow_schema(schema: &TableSchema, format: FileFormat) -> Arc<Schema> {
    let mut fields = vec![
        Field::new(IKEY_COLUMN_NAME, DataType::Binary, false),
        // Issue #16: typed _seq + _op columns so external analytics
        // readers can apply the MVCC dedup projection without decoding
        // the ikey trailer. Always present in every merutable-written
        // file; internal reads skip them via the projection mask.
        Field::new(SEQ_COLUMN_NAME, DataType::Int64, false),
        Field::new(OP_COLUMN_NAME, DataType::Int32, false),
    ];
    if format.has_value_blob() {
        fields.push(Field::new(VALUE_BLOB_COLUMN_NAME, DataType::Binary, false));
    }
    for col in &schema.columns {
        let dtype = column_type_to_arrow(&col.col_type);
        fields.push(Field::new(&col.name, dtype, col.nullable));
    }
    Arc::new(Schema::new(fields))
}

fn column_type_to_arrow(ct: &ColumnType) -> DataType {
    match ct {
        ColumnType::Boolean => DataType::Boolean,
        ColumnType::Int32 => DataType::Int32,
        ColumnType::Int64 => DataType::Int64,
        ColumnType::Float => DataType::Float32,
        ColumnType::Double => DataType::Float64,
        ColumnType::ByteArray => DataType::Binary,
        ColumnType::FixedLenByteArray(n) => DataType::FixedSizeBinary(*n),
    }
}

/// Convert a slice of `(InternalKey, Row)` pairs into an Arrow `RecordBatch`
/// laid out for the given LSM level.
///
/// At L0 the batch carries the `_merutable_value` blob column in addition to
/// the typed user columns; at L1+ only typed user columns. The `_merutable_ikey`
/// column is always column 0.
pub fn rows_to_record_batch(
    rows: &[(InternalKey, Row)],
    schema: &TableSchema,
    format: FileFormat,
) -> Result<RecordBatch> {
    let arrow_sch = arrow_schema(schema, format);
    if rows.is_empty() {
        return Ok(RecordBatch::new_empty(arrow_sch));
    }

    // Build _merutable_ikey column.
    let ikey_col: ArrayRef = Arc::new(BinaryArray::from_iter_values(
        rows.iter().map(|(ik, _)| ik.as_bytes()),
    ));
    // Issue #16: typed _seq / _op columns. We populate these from the
    // InternalKey fields so the external-reader contract matches the
    // engine's own seq+op semantics exactly. No additional per-row
    // decoding cost — these are already in the InternalKey struct.
    let seq_col: ArrayRef = Arc::new(Int64Array::from_iter_values(
        rows.iter().map(|(ik, _)| ik.seq.0 as i64),
    ));
    let op_col: ArrayRef = Arc::new(Int32Array::from_iter_values(
        rows.iter().map(|(ik, _)| ik.op_type as u8 as i32),
    ));
    let mut col_arrays: Vec<ArrayRef> = vec![ikey_col, seq_col, op_col];

    // Optional _merutable_value blob column for Dual format.
    if format.has_value_blob() {
        let blobs: Vec<Vec<u8>> = rows
            .iter()
            .map(|(_, row)| {
                postcard::to_allocvec(row)
                    .map_err(|e| MeruError::Parquet(format!("postcard encode row: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let blob_col: ArrayRef = Arc::new(BinaryArray::from_iter_values(
            blobs.iter().map(|b| b.as_slice()),
        ));
        col_arrays.push(blob_col);
    }

    // Typed user columns, in schema order. Non-nullable columns need
    // type-appropriate defaults for tombstone rows whose `Row` is empty
    // (Bug K: `Row::default()` produces zero fields, which `build_column`
    // maps to `None` → Arrow NULL → rejected by `RecordBatch::try_new`
    // when the column is declared non-nullable).
    for (col_idx, col_def) in schema.columns.iter().enumerate() {
        let arr = build_column(rows, col_idx, &col_def.col_type, col_def.nullable)?;
        col_arrays.push(arr);
    }

    RecordBatch::try_new(arrow_sch, col_arrays).map_err(|e| MeruError::Parquet(e.to_string()))
}

/// Build one Arrow column from the `col_idx`-th field of every row.
///
/// # Nullability
///
/// When `nullable` is `false` and a row's field at `col_idx` is absent
/// (tombstone rows carry `Row::default()` with zero fields), the column
/// emits a type-appropriate default value instead of NULL. This prevents
/// Arrow from rejecting the batch for nulls in a non-nullable column
/// (Bug K regression).
///
/// # Type-mismatch policy
///
/// If a row contains a `FieldValue` variant that doesn't match `col_type`
/// this returns `MeruError::Parquet` with the row index and expected /
/// actual variant names. Previously the writer silently coerced
/// mismatches to `0` / `false` / empty bytes, which produced a valid
/// Parquet file containing garbage values — a silent data-corruption
/// bug that survived every round-trip test (which uses correct types).
fn build_column(
    rows: &[(InternalKey, Row)],
    col_idx: usize,
    col_type: &ColumnType,
    nullable: bool,
) -> Result<ArrayRef> {
    /// Shorthand for "type mismatch at row `i`, expected X, got Y".
    fn mismatch(col_idx: usize, row_idx: usize, expected: &str, got: &FieldValue) -> MeruError {
        MeruError::Parquet(format!(
            "codec::build_column: type mismatch at column {col_idx} row {row_idx}: \
             expected {expected}, got {}",
            field_variant_name(got)
        ))
    }

    match col_type {
        ColumnType::Boolean => {
            let mut vals: Vec<Option<bool>> = Vec::with_capacity(rows.len());
            for (row_idx, (_, row)) in rows.iter().enumerate() {
                match row.get(col_idx) {
                    None if !nullable => vals.push(Some(false)),
                    None => vals.push(None),
                    Some(FieldValue::Boolean(b)) => vals.push(Some(*b)),
                    Some(other) => return Err(mismatch(col_idx, row_idx, "Boolean", other)),
                }
            }
            Ok(Arc::new(BooleanArray::from(vals)))
        }
        ColumnType::Int32 => {
            let mut vals: Vec<Option<i32>> = Vec::with_capacity(rows.len());
            for (row_idx, (_, row)) in rows.iter().enumerate() {
                match row.get(col_idx) {
                    None if !nullable => vals.push(Some(0)),
                    None => vals.push(None),
                    Some(FieldValue::Int32(i)) => vals.push(Some(*i)),
                    Some(other) => return Err(mismatch(col_idx, row_idx, "Int32", other)),
                }
            }
            Ok(Arc::new(Int32Array::from(vals)))
        }
        ColumnType::Int64 => {
            let mut vals: Vec<Option<i64>> = Vec::with_capacity(rows.len());
            for (row_idx, (_, row)) in rows.iter().enumerate() {
                match row.get(col_idx) {
                    None if !nullable => vals.push(Some(0)),
                    None => vals.push(None),
                    Some(FieldValue::Int64(i)) => vals.push(Some(*i)),
                    Some(other) => return Err(mismatch(col_idx, row_idx, "Int64", other)),
                }
            }
            Ok(Arc::new(Int64Array::from(vals)))
        }
        ColumnType::Float => {
            let mut vals: Vec<Option<f32>> = Vec::with_capacity(rows.len());
            for (row_idx, (_, row)) in rows.iter().enumerate() {
                match row.get(col_idx) {
                    None if !nullable => vals.push(Some(0.0)),
                    None => vals.push(None),
                    Some(FieldValue::Float(f)) => vals.push(Some(*f)),
                    Some(other) => return Err(mismatch(col_idx, row_idx, "Float", other)),
                }
            }
            Ok(Arc::new(Float32Array::from(vals)))
        }
        ColumnType::Double => {
            let mut vals: Vec<Option<f64>> = Vec::with_capacity(rows.len());
            for (row_idx, (_, row)) in rows.iter().enumerate() {
                match row.get(col_idx) {
                    None if !nullable => vals.push(Some(0.0)),
                    None => vals.push(None),
                    Some(FieldValue::Double(d)) => vals.push(Some(*d)),
                    Some(other) => return Err(mismatch(col_idx, row_idx, "Double", other)),
                }
            }
            Ok(Arc::new(Float64Array::from(vals)))
        }
        ColumnType::ByteArray => {
            // For non-nullable ByteArray, use a static empty sentinel.
            static EMPTY: &[u8] = &[];
            let mut vals: Vec<Option<&[u8]>> = Vec::with_capacity(rows.len());
            for (row_idx, (_, row)) in rows.iter().enumerate() {
                match row.get(col_idx) {
                    None if !nullable => vals.push(Some(EMPTY)),
                    None => vals.push(None),
                    Some(FieldValue::Bytes(b)) => vals.push(Some(b.as_ref())),
                    Some(other) => return Err(mismatch(col_idx, row_idx, "Bytes", other)),
                }
            }
            Ok(Arc::new(BinaryArray::from_iter(vals)))
        }
        ColumnType::FixedLenByteArray(n) => {
            // Every non-null row at this column must be exactly `n` bytes.
            // The Arrow schema (see `arrow_schema`) declares this column
            // as `FixedSizeBinary(n)`, so the `ArrayRef` we produce here
            // must be a `FixedSizeBinaryArray` of the same width —
            // otherwise `RecordBatch::try_new` would reject the batch.
            let expected_len = *n as usize;
            let mut vals: Vec<Option<Vec<u8>>> = Vec::with_capacity(rows.len());
            for (row_idx, (_, row)) in rows.iter().enumerate() {
                match row.get(col_idx) {
                    None if !nullable => vals.push(Some(vec![0u8; expected_len])),
                    None => vals.push(None),
                    Some(FieldValue::Bytes(b)) => {
                        if b.len() != expected_len {
                            return Err(MeruError::Parquet(format!(
                                "codec::build_column: FixedLenByteArray({expected_len}) at column \
                                 {col_idx} row {row_idx} has wrong length {} (expected {expected_len})",
                                b.len()
                            )));
                        }
                        vals.push(Some(b.to_vec()));
                    }
                    Some(other) => return Err(mismatch(col_idx, row_idx, "Bytes", other)),
                }
            }
            let arr = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vals.into_iter(),
                *n,
            )
            .map_err(|e| {
                MeruError::Parquet(format!(
                    "codec::build_column: FixedSizeBinaryArray::try_from_sparse_iter_with_size({n}): {e}"
                ))
            })?;
            Ok(Arc::new(arr))
        }
    }
}

/// Name of a `FieldValue` variant, for error messages. Kept in sync
/// with [`FieldValue`] by construction.
fn field_variant_name(v: &FieldValue) -> &'static str {
    match v {
        FieldValue::Boolean(_) => "Boolean",
        FieldValue::Int32(_) => "Int32",
        FieldValue::Int64(_) => "Int64",
        FieldValue::Float(_) => "Float",
        FieldValue::Double(_) => "Double",
        FieldValue::Bytes(_) => "Bytes",
    }
}

/// Decode the `_merutable_ikey` column from a `RecordBatch` into a
/// `Vec<InternalKey>`, **without** materializing per-column field
/// values or the postcard blob.
///
/// Used by the flush-time deletion-vector resolve path
/// (`ParquetReader::iter_user_keys_in_range`), where the consumer
/// only needs `(user_key, file_position)` and pays nothing for the
/// row payload it would discard. The batch must include
/// `_merutable_ikey`; other columns are ignored.
///
/// The `_merutable_ikey` column is encoded as `BinaryArray` and is
/// `not nullable` (asserted by the writer's `arrow_schema`), so a
/// missing/null entry is corruption.
pub fn record_batch_to_ikeys(
    batch: &RecordBatch,
    schema: &TableSchema,
) -> Result<Vec<InternalKey>> {
    let n = batch.num_rows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let arrow_schema = batch.schema();
    let ikey_idx = arrow_schema
        .index_of(IKEY_COLUMN_NAME)
        .map_err(|_| MeruError::Parquet(format!("missing {IKEY_COLUMN_NAME} column")))?;
    let ikey_col = batch
        .column(ikey_idx)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| MeruError::Parquet(format!("{IKEY_COLUMN_NAME} not BinaryArray")))?;

    let mut out = Vec::with_capacity(n);
    for row_idx in 0..n {
        let ikey_bytes = ikey_col.value(row_idx);
        out.push(InternalKey::decode(ikey_bytes, schema)?);
    }
    Ok(out)
}

/// Convert an Arrow `RecordBatch` (from a Parquet read) back into
/// `(InternalKey, Row)` pairs.
///
/// Column lookup is **by name**, not position, so this function works
/// uniformly against:
/// - L0 batches (with `_merutable_value` blob present) — fast path:
///   decode each `Row` from postcard bytes; the typed columns are ignored.
/// - L1+ batches (typed columns only) — materialize each `Row` field by
///   field from the per-column Arrow arrays.
/// - Projected batches that include only a subset of columns, as long as
///   `_merutable_ikey` is present.
pub fn record_batch_to_rows(
    batch: &RecordBatch,
    schema: &TableSchema,
) -> Result<Vec<(InternalKey, Row)>> {
    let n = batch.num_rows();
    if n == 0 {
        return Ok(vec![]);
    }

    let arrow_schema = batch.schema();
    let ikey_idx = arrow_schema
        .index_of(IKEY_COLUMN_NAME)
        .map_err(|_| MeruError::Parquet(format!("missing {IKEY_COLUMN_NAME} column")))?;
    let ikey_col = batch
        .column(ikey_idx)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| MeruError::Parquet(format!("{IKEY_COLUMN_NAME} not BinaryArray")))?;

    // L0 fast path: if the blob column is present, decode rows directly
    // from postcard bytes and skip per-column field extraction entirely.
    if let Ok(blob_idx) = arrow_schema.index_of(VALUE_BLOB_COLUMN_NAME) {
        let blob_col = batch
            .column(blob_idx)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                MeruError::Parquet(format!("{VALUE_BLOB_COLUMN_NAME} not BinaryArray"))
            })?;

        let mut result = Vec::with_capacity(n);
        for row_idx in 0..n {
            let ikey_bytes = ikey_col.value(row_idx);
            let ikey = InternalKey::decode(ikey_bytes, schema)?;
            let blob = blob_col.value(row_idx);
            let mut row: Row = postcard::from_bytes(blob)
                .map_err(|e| MeruError::Parquet(format!("postcard decode row: {e}")))?;
            // Issue #44 Stage 3: if the blob was written under an
            // older schema (fewer columns), pad the decoded Row
            // with `initial_default` / `None` so every downstream
            // code path sees a Row matching the current schema's
            // arity. Blobs written under the CURRENT schema are
            // already the right length — this loop is a no-op for
            // them.
            while row.fields.len() < schema.columns.len() {
                let missing_idx = row.fields.len();
                let col = &schema.columns[missing_idx];
                row.fields.push(col.initial_default.clone());
            }
            result.push((ikey, row));
        }
        return Ok(result);
    }

    // L1+ path: rebuild Row from typed user columns by name.
    //
    // Issue #44 Stage 3 — additive-evolution tolerance. A Parquet
    // file written under an older schema_id may legitimately be
    // missing one or more of the current schema's user columns
    // (the column was added after this file was flushed). For
    // those columns the batch has no Arrow array at all; we fill
    // the missing cell with the column's `initial_default` (or
    // `None` if the column is nullable and no default is set).
    //
    // `check_schema_compatible` (iceberg/catalog.rs) has already
    // guaranteed that any missing column is either nullable or
    // carries a default — so the None fallback here can never
    // violate a NOT NULL constraint.
    let mut user_col_indices: Vec<Option<usize>> = Vec::with_capacity(schema.columns.len());
    for col_def in &schema.columns {
        user_col_indices.push(arrow_schema.index_of(&col_def.name).ok());
    }

    let mut result = Vec::with_capacity(n);
    for row_idx in 0..n {
        let ikey_bytes = ikey_col.value(row_idx);
        let ikey = InternalKey::decode(ikey_bytes, schema)?;

        let mut fields = Vec::with_capacity(schema.columns.len());
        for (col_def, slot) in schema.columns.iter().zip(&user_col_indices) {
            let fv = match slot {
                Some(arrow_col_idx) => {
                    extract_field(batch.column(*arrow_col_idx), row_idx, &col_def.col_type)?
                }
                None => col_def.initial_default.clone(),
            };
            fields.push(fv);
        }
        result.push((ikey, Row::new(fields)));
    }
    Ok(result)
}

/// Extract one `FieldValue` from an Arrow array cell. Returns an error
/// (rather than panicking) on downcast mismatch — this can happen when
/// a Parquet file's physical column type drifted away from the table
/// schema it's being decoded against, e.g. after a schema-evolution
/// rename or a corrupted footer. Panicking in that path would take down
/// the read path on a diagnosable condition.
fn extract_field(
    arr: &dyn arrow::array::Array,
    row: usize,
    col_type: &ColumnType,
) -> Result<Option<FieldValue>> {
    if arr.is_null(row) {
        return Ok(None);
    }

    fn downcast_err(expected: &str, actual: &arrow::datatypes::DataType) -> MeruError {
        MeruError::Parquet(format!(
            "codec::extract_field: Arrow array type mismatch — expected {expected}, got {actual:?}"
        ))
    }

    let val = match col_type {
        ColumnType::Boolean => {
            let a = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| downcast_err("BooleanArray", arr.data_type()))?;
            FieldValue::Boolean(a.value(row))
        }
        ColumnType::Int32 => {
            let a = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err("Int32Array", arr.data_type()))?;
            FieldValue::Int32(a.value(row))
        }
        ColumnType::Int64 => {
            let a = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err("Int64Array", arr.data_type()))?;
            FieldValue::Int64(a.value(row))
        }
        ColumnType::Float => {
            let a = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err("Float32Array", arr.data_type()))?;
            FieldValue::Float(a.value(row))
        }
        ColumnType::Double => {
            let a = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err("Float64Array", arr.data_type()))?;
            FieldValue::Double(a.value(row))
        }
        ColumnType::ByteArray => {
            let a = arr
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| downcast_err("BinaryArray", arr.data_type()))?;
            FieldValue::Bytes(Bytes::copy_from_slice(a.value(row)))
        }
        ColumnType::FixedLenByteArray(_) => {
            // The Arrow schema declares this as `FixedSizeBinary(n)` (see
            // `column_type_to_arrow`) and `build_column` constructs a
            // `FixedSizeBinaryArray`. Decode must downcast to the same
            // type — previously this arm shared `BinaryArray` with
            // `ByteArray` and would panic/error on FixedSizeBinary input.
            let a = arr
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| downcast_err("FixedSizeBinaryArray", arr.data_type()))?;
            FieldValue::Bytes(Bytes::copy_from_slice(a.value(row)))
        }
    };
    Ok(Some(val))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        key::InternalKey,
        schema::{ColumnDef, ColumnType, TableSchema},
        sequence::{OpType, SeqNum},
    };

    fn scalar_schema() -> TableSchema {
        TableSchema {
            table_name: "t".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,

                    ..Default::default()
                },
                ColumnDef {
                    name: "flag".into(),
                    col_type: ColumnType::Boolean,
                    nullable: true,

                    ..Default::default()
                },
                ColumnDef {
                    name: "score".into(),
                    col_type: ColumnType::Double,
                    nullable: false,

                    ..Default::default()
                },
            ],
            primary_key: vec![0],

            ..Default::default()
        }
    }

    fn make_ikey(id: i64, seq: u64, schema: &TableSchema) -> InternalKey {
        InternalKey::encode(&[FieldValue::Int64(id)], SeqNum(seq), OpType::Put, schema).unwrap()
    }

    /// Round-trip a small set through `rows_to_record_batch` →
    /// `record_batch_to_rows` at L1 (typed-only, no blob fast path).
    /// This pins the happy-path field-for-field equality contract.
    #[test]
    fn l1_roundtrip_typed_columns_match_input() {
        let schema = scalar_schema();
        let rows: Vec<(InternalKey, Row)> = (1..=5i64)
            .map(|i| {
                (
                    make_ikey(i, i as u64, &schema),
                    Row::new(vec![
                        Some(FieldValue::Int64(i)),
                        if i % 2 == 0 {
                            Some(FieldValue::Boolean(true))
                        } else {
                            None
                        },
                        Some(FieldValue::Double(i as f64 * 1.5)),
                    ]),
                )
            })
            .collect();

        let batch =
            rows_to_record_batch(&rows, &schema, FileFormat::default_for_level(Level(1))).unwrap();
        let decoded = record_batch_to_rows(&batch, &schema).unwrap();
        assert_eq!(decoded.len(), rows.len());
        for ((orig_ik, orig_row), (got_ik, got_row)) in rows.iter().zip(decoded.iter()) {
            assert_eq!(orig_ik.as_bytes(), got_ik.as_bytes());
            assert_eq!(orig_row, got_row);
        }
    }

    /// Silent-coercion regression: passing a row whose `FieldValue`
    /// variant doesn't match the schema's `ColumnType` used to store
    /// `0` / `false` / empty bytes in the Parquet file without any
    /// warning. It must now error with a precise row + column index.
    #[test]
    fn build_column_rejects_type_mismatch_int32_vs_int64() {
        let schema = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "n".into(),
                col_type: ColumnType::Int32,
                nullable: false,

                ..Default::default()
            }],
            primary_key: vec![0],

            ..Default::default()
        };
        let ikey =
            InternalKey::encode(&[FieldValue::Int32(1)], SeqNum(1), OpType::Put, &schema).unwrap();
        // Schema says Int32 but we hand it an Int64 — classic drift.
        let rows = vec![(ikey, Row::new(vec![Some(FieldValue::Int64(0x1_0000_0000))]))];
        let err = rows_to_record_batch(&rows, &schema, FileFormat::default_for_level(Level(1)))
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("type mismatch") && msg.contains("Int32") && msg.contains("Int64"),
            "error should name expected + actual variants: {msg}"
        );
    }

    #[test]
    fn build_column_rejects_type_mismatch_bytes_vs_bool() {
        let schema = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "flag".into(),
                col_type: ColumnType::Boolean,
                nullable: false,

                ..Default::default()
            }],
            primary_key: vec![],

            ..Default::default()
        };
        let ikey = InternalKey::encode(&[], SeqNum(1), OpType::Put, &schema).unwrap();
        let rows = vec![(
            ikey,
            Row::new(vec![Some(FieldValue::Bytes(Bytes::from("nope")))]),
        )];
        let err = rows_to_record_batch(&rows, &schema, FileFormat::default_for_level(Level(1)))
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("Boolean") && msg.contains("Bytes"), "{msg}");
    }

    /// FixedLenByteArray(n) now validates length at write time. Previously
    /// the declared length was completely ignored and any bytes were
    /// accepted, silently breaking the schema contract downstream.
    #[test]
    fn fixed_len_byte_array_rejects_wrong_length() {
        let schema = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "fb".into(),
                col_type: ColumnType::FixedLenByteArray(4),
                nullable: false,

                ..Default::default()
            }],
            primary_key: vec![],

            ..Default::default()
        };
        let ikey = InternalKey::encode(&[], SeqNum(1), OpType::Put, &schema).unwrap();
        let rows = vec![(
            ikey,
            Row::new(vec![Some(FieldValue::Bytes(Bytes::from("too_long_bytes")))]),
        )];
        let err = rows_to_record_batch(&rows, &schema, FileFormat::default_for_level(Level(1)))
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("FixedLenByteArray") && msg.contains("wrong length"),
            "{msg}"
        );
    }

    /// FixedLenByteArray(n) accepts exactly-n-byte values.
    #[test]
    fn fixed_len_byte_array_accepts_correct_length() {
        let schema = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "fb".into(),
                col_type: ColumnType::FixedLenByteArray(4),
                nullable: false,

                ..Default::default()
            }],
            primary_key: vec![],

            ..Default::default()
        };
        let ikey = InternalKey::encode(&[], SeqNum(1), OpType::Put, &schema).unwrap();
        let rows = vec![(
            ikey,
            Row::new(vec![Some(FieldValue::Bytes(Bytes::from("abcd")))]),
        )];
        let batch =
            rows_to_record_batch(&rows, &schema, FileFormat::default_for_level(Level(1))).unwrap();
        let decoded = record_batch_to_rows(&batch, &schema).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].1, rows[0].1);
    }

    /// Null fields at nullable columns must round-trip as `None`.
    #[test]
    fn null_fields_round_trip_as_none() {
        let schema = scalar_schema();
        let ikey = make_ikey(42, 1, &schema);
        let row = Row::new(vec![
            Some(FieldValue::Int64(42)),
            None, // flag is nullable
            Some(FieldValue::Double(123.456)),
        ]);
        let batch = rows_to_record_batch(
            &[(ikey, row.clone())],
            &schema,
            crate::types::level::FileFormat::Columnar,
        )
        .unwrap();
        let decoded = record_batch_to_rows(&batch, &schema).unwrap();
        assert_eq!(decoded[0].1, row);
        assert_eq!(decoded[0].1.get(1), None);
    }

    /// Empty-input batches must not round-trip via the typed decode
    /// path (they early-return an empty Vec). This pins the zero-row
    /// fast path so a future refactor can't accidentally dereference a
    /// non-existent ikey column on an empty batch.
    #[test]
    fn empty_batch_decodes_to_empty_vec() {
        let schema = scalar_schema();
        let batch =
            rows_to_record_batch(&[], &schema, FileFormat::default_for_level(Level(1))).unwrap();
        assert_eq!(batch.num_rows(), 0);
        let decoded = record_batch_to_rows(&batch, &schema).unwrap();
        assert!(decoded.is_empty());
    }

    /// An Arrow array whose physical type disagrees with the declared
    /// `ColumnType` must produce a `MeruError::Parquet` instead of
    /// panicking. Previously `downcast_ref().unwrap()` would take down
    /// the read path on a diagnosable condition.
    #[test]
    fn extract_field_returns_error_on_downcast_mismatch() {
        // Hand-build a Float64Array and ask `extract_field` to decode
        // it as Int64 — type mismatch must surface as Err, not panic.
        let arr = arrow::array::Float64Array::from(vec![1.5_f64]);
        let result = extract_field(&arr, 0, &ColumnType::Int64);
        assert!(matches!(result, Err(MeruError::Parquet(_))));
    }
}
