//! C-ABI type definitions and C↔Rust conversion helpers.

use std::ffi::{c_char, CStr};

use bytes::Bytes;
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
    MeruError,
};

// ── Column type enum ──────────────────────────────────────────────────────────

/// Column type tag. Maps to merutable's internal ColumnType.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MeruColumnType {
    Boolean = 0,
    Int32 = 1,
    Int64 = 2,
    Float = 3,
    Double = 4,
    ByteArray = 5,
    FixedLenBytes = 6,
}

// ── Value ─────────────────────────────────────────────────────────────────────

/// Pointer + length into a byte buffer. Used inside MeruValue.
#[repr(C)]
pub struct MeruBytesView {
    pub data: *const u8,
    pub len: usize,
}

/// The inner union of MeruValue. Do not use directly; use MeruValue.
#[repr(C)]
pub union MeruValueInner {
    pub v_bool: u8,
    pub v_int32: i32,
    pub v_int64: i64,
    pub v_float: f32,
    pub v_double: f64,
    pub v_bytes: std::mem::ManuallyDrop<MeruBytesView>,
}

/// A single nullable field value.
///
/// `is_null = 1` means SQL NULL; the union contents are undefined.
/// For BYTE_ARRAY / FIXED_LEN_BYTES: when this value is returned by the API
/// (meru_get, meru_scan), `v_bytes.data` is owned by the enclosing MeruRow
/// and is freed when `meru_row_free` / `meru_scan_result_free` is called.
/// Do NOT cache `v_bytes.data` past that call.
///
/// When passing a value *into* the API (meru_put, meru_delete), the byte
/// buffer is read and copied before the call returns; the caller retains
/// ownership.
#[repr(C)]
pub struct MeruValue {
    pub tag: MeruColumnType,
    pub is_null: u8,
    pub _pad: [u8; 3],
    pub inner: MeruValueInner,
}

// ── Row ───────────────────────────────────────────────────────────────────────

/// A database row: an array of `field_count` MeruValue entries.
/// Field order matches the schema column order.
#[repr(C)]
pub struct MeruRow {
    pub fields: *mut MeruValue,
    pub field_count: usize,
}

// ── Schema ────────────────────────────────────────────────────────────────────

/// Column definition used to declare a schema in MeruOpenOptions.
///
/// `initial_default` and `write_default` may be NULL (meaning no default).
#[repr(C)]
pub struct MeruColumnDef {
    pub name: *const c_char,
    pub col_type: MeruColumnType,
    /// Only meaningful when col_type == MERU_COLUMN_FIXED_LEN_BYTES.
    pub fixed_byte_len: i32,
    /// 0 = NOT NULL, 1 = nullable.
    pub nullable: u8,
    pub initial_default: *const MeruValue,
    pub write_default: *const MeruValue,
}

/// Table schema: column definitions + primary key column indices.
#[repr(C)]
pub struct MeruSchema {
    pub table_name: *const c_char,
    pub columns: *const MeruColumnDef,
    pub column_count: usize,
    /// Array of column indices (into `columns`) that form the primary key.
    pub primary_key: *const usize,
    pub primary_key_len: usize,
}

// ── Open options ──────────────────────────────────────────────────────────────

/// Options for meru_open(). Pass wal_dir = NULL to default to "{catalog_uri}/wal".
#[repr(C)]
pub struct MeruOpenOptions {
    pub schema: MeruSchema,
    pub catalog_uri: *const c_char,
    pub wal_dir: *const c_char,
    /// 0 = use engine default (64 MiB).
    pub memtable_size_mb: usize,
    /// 1 = open read-only.
    pub read_only: u8,
}

// ── Scan result ───────────────────────────────────────────────────────────────

/// Array of rows returned by meru_scan(). Free with meru_scan_result_free().
#[repr(C)]
pub struct MeruScanResult {
    pub entries: *mut MeruRow,
    pub count: usize,
}

// ── Stats ─────────────────────────────────────────────────────────────────────

/// Memtable statistics snapshot.
#[repr(C)]
pub struct MeruMemtableStats {
    pub active_size_bytes: usize,
    pub active_entry_count: u64,
    pub flush_threshold: usize,
    pub immutable_count: usize,
}

/// Row cache statistics snapshot.
#[repr(C)]
pub struct MerucacheStats {
    pub capacity: usize,
    pub size: usize,
    pub hit_count: u64,
    pub miss_count: u64,
}

/// Engine statistics snapshot. Filled by meru_stats().
#[repr(C)]
pub struct MeruStats {
    pub snapshot_id: i64,
    pub current_seq: u64,
    pub memtable: MeruMemtableStats,
    pub cache: MerucacheStats,
}

// ── C → Rust conversions ─────────────────────────────────────────────────────

fn c_col_type_to_rust(tag: MeruColumnType, fixed_byte_len: i32) -> ColumnType {
    match tag {
        MeruColumnType::Boolean => ColumnType::Boolean,
        MeruColumnType::Int32 => ColumnType::Int32,
        MeruColumnType::Int64 => ColumnType::Int64,
        MeruColumnType::Float => ColumnType::Float,
        MeruColumnType::Double => ColumnType::Double,
        MeruColumnType::ByteArray => ColumnType::ByteArray,
        MeruColumnType::FixedLenBytes => ColumnType::FixedLenByteArray(fixed_byte_len),
    }
}

unsafe fn c_value_to_field_value(
    v: &MeruValue,
    col_type: &ColumnType,
) -> Result<Option<FieldValue>, MeruError> {
    if v.is_null != 0 {
        return Ok(None);
    }
    let fv = match col_type {
        ColumnType::Boolean => FieldValue::Boolean(v.inner.v_bool != 0),
        ColumnType::Int32 => FieldValue::Int32(v.inner.v_int32),
        ColumnType::Int64 => FieldValue::Int64(v.inner.v_int64),
        ColumnType::Float => FieldValue::Float(v.inner.v_float),
        ColumnType::Double => FieldValue::Double(v.inner.v_double),
        ColumnType::ByteArray | ColumnType::FixedLenByteArray(_) => {
            let bv = &*v.inner.v_bytes;
            if bv.data.is_null() && bv.len > 0 {
                return Err(MeruError::InvalidArgument(
                    "v_bytes.data is null but len > 0".into(),
                ));
            }
            let slice = if bv.len == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(bv.data, bv.len)
            };
            FieldValue::Bytes(Bytes::copy_from_slice(slice))
        }
    };
    Ok(Some(fv))
}

/// Convert a C MeruRow to a Rust Row, using the schema for type context.
///
/// # Safety
/// `c_row` must be a valid pointer to a MeruRow with `field_count` == schema column count.
pub unsafe fn c_row_to_rust(
    c_row: *const MeruRow,
    schema: &TableSchema,
) -> Result<Row, MeruError> {
    let c_row = &*c_row;
    if c_row.field_count != schema.columns.len() {
        return Err(MeruError::InvalidArgument(format!(
            "row field_count {} != schema column count {}",
            c_row.field_count,
            schema.columns.len()
        )));
    }
    let c_fields = std::slice::from_raw_parts(c_row.fields, c_row.field_count);
    let mut fields = Vec::with_capacity(schema.columns.len());
    for (c_val, col) in c_fields.iter().zip(&schema.columns) {
        fields.push(c_value_to_field_value(c_val, &col.col_type)?);
    }
    Ok(Row::new(fields))
}

/// Convert a slice of C MeruValues (PK values) to Rust FieldValues.
///
/// # Safety
/// `pk` must point to `pk_count` valid MeruValue entries.
pub unsafe fn c_pk_to_rust(
    pk: *const MeruValue,
    pk_count: usize,
    schema: &TableSchema,
) -> Result<Vec<FieldValue>, MeruError> {
    if pk_count != schema.primary_key.len() {
        return Err(MeruError::InvalidArgument(format!(
            "pk_count {} != schema pk len {}",
            pk_count,
            schema.primary_key.len()
        )));
    }
    let c_pks = std::slice::from_raw_parts(pk, pk_count);
    let mut fvs = Vec::with_capacity(pk_count);
    for (i, &col_idx) in schema.primary_key.iter().enumerate() {
        let col_type = &schema.columns[col_idx].col_type;
        match c_value_to_field_value(&c_pks[i], col_type)? {
            Some(fv) => fvs.push(fv),
            None => {
                return Err(MeruError::InvalidArgument(format!(
                    "PK column {} cannot be NULL",
                    schema.columns[col_idx].name
                )));
            }
        }
    }
    Ok(fvs)
}

/// Convert a C MeruSchema to a Rust TableSchema.
///
/// # Safety
/// All pointer fields in `cs` and its column array must be valid.
pub unsafe fn c_schema_to_rust(cs: &MeruSchema) -> Result<TableSchema, MeruError> {
    let invalid = |msg: &str| MeruError::InvalidArgument(msg.to_string());

    let table_name = CStr::from_ptr(cs.table_name)
        .to_str()
        .map_err(|_| invalid("table_name is not valid UTF-8"))?
        .to_string();

    let c_cols = std::slice::from_raw_parts(cs.columns, cs.column_count);
    let mut columns = Vec::with_capacity(cs.column_count);
    for cc in c_cols {
        let name = CStr::from_ptr(cc.name)
            .to_str()
            .map_err(|_| invalid("column name is not valid UTF-8"))?
            .to_string();
        let col_type = c_col_type_to_rust(cc.col_type, cc.fixed_byte_len);
        let initial_default = if cc.initial_default.is_null() {
            None
        } else {
            c_value_to_field_value(&*cc.initial_default, &col_type)?
        };
        let write_default = if cc.write_default.is_null() {
            None
        } else {
            c_value_to_field_value(&*cc.write_default, &col_type)?
        };
        columns.push(ColumnDef {
            name,
            col_type,
            nullable: cc.nullable != 0,
            initial_default,
            write_default,
            ..Default::default()
        });
    }

    let pk_slice = std::slice::from_raw_parts(cs.primary_key, cs.primary_key_len);
    let primary_key = pk_slice.to_vec();

    let mut schema = TableSchema {
        table_name,
        columns,
        primary_key,
        ..Default::default()
    };
    schema.validate().map_err(|e| MeruError::InvalidArgument(e.to_string()))?;
    Ok(schema)
}

// ── Rust → C conversions ─────────────────────────────────────────────────────

fn rust_col_type_to_c(ct: &ColumnType) -> MeruColumnType {
    match ct {
        ColumnType::Boolean => MeruColumnType::Boolean,
        ColumnType::Int32 => MeruColumnType::Int32,
        ColumnType::Int64 => MeruColumnType::Int64,
        ColumnType::Float => MeruColumnType::Float,
        ColumnType::Double => MeruColumnType::Double,
        ColumnType::ByteArray | ColumnType::FixedLenByteArray(_) => MeruColumnType::ByteArray,
    }
}

/// Convert a Rust FieldValue to a heap-allocated C MeruValue.
/// Byte buffers are heap-allocated and owned by the returned value.
/// Free with `free_c_value`.
pub fn rust_field_value_to_c(fv: Option<&FieldValue>, col_type: &ColumnType) -> MeruValue {
    let tag = rust_col_type_to_c(col_type);
    match fv {
        None => MeruValue {
            tag,
            is_null: 1,
            _pad: [0; 3],
            inner: MeruValueInner { v_int64: 0 },
        },
        Some(fv) => {
            let inner = match fv {
                FieldValue::Boolean(b) => MeruValueInner { v_bool: *b as u8 },
                FieldValue::Int32(v) => MeruValueInner { v_int32: *v },
                FieldValue::Int64(v) => MeruValueInner { v_int64: *v },
                FieldValue::Float(v) => MeruValueInner { v_float: *v },
                FieldValue::Double(v) => MeruValueInner { v_double: *v },
                FieldValue::Bytes(b) => {
                    let mut owned: Vec<u8> = b.to_vec();
                    let data = owned.as_mut_ptr();
                    let len = owned.len();
                    std::mem::forget(owned);
                    MeruValueInner {
                        v_bytes: std::mem::ManuallyDrop::new(MeruBytesView { data, len }),
                    }
                }
            };
            MeruValue {
                tag,
                is_null: 0,
                _pad: [0; 3],
                inner,
            }
        }
    }
}

/// Frees the byte buffer inside a MeruValue (if it is a bytes-type with is_null=0).
///
/// # Safety
/// `v` must have been produced by `rust_field_value_to_c`.
pub unsafe fn free_c_value(v: &mut MeruValue) {
    if v.is_null != 0 {
        return;
    }
    if v.tag == MeruColumnType::ByteArray || v.tag == MeruColumnType::FixedLenBytes {
        let bv = &*v.inner.v_bytes;
        if !bv.data.is_null() {
            let _owned =
                Vec::from_raw_parts(bv.data as *mut u8, bv.len, bv.len);
            // dropped here
        }
    }
}

/// Convert a Rust Row to a heap-allocated C MeruRow.
pub fn rust_row_to_c(row: &Row, schema: &TableSchema) -> MeruRow {
    let field_count = schema.columns.len();
    let mut fields: Vec<MeruValue> = schema
        .columns
        .iter()
        .enumerate()
        .map(|(i, col)| rust_field_value_to_c(row.get(i), &col.col_type))
        .collect();
    let ptr = fields.as_mut_ptr();
    std::mem::forget(fields);
    MeruRow {
        fields: ptr,
        field_count,
    }
}

/// Free all memory owned by a MeruRow (byte buffers + the fields array).
///
/// # Safety
/// `row` must have been allocated by `rust_row_to_c`.
pub unsafe fn free_c_row_fields(row: &mut MeruRow) {
    if row.fields.is_null() || row.field_count == 0 {
        return;
    }
    let fields = std::slice::from_raw_parts_mut(row.fields, row.field_count);
    for v in fields.iter_mut() {
        free_c_value(v);
    }
    let _owned = Vec::from_raw_parts(row.fields, row.field_count, row.field_count);
    row.fields = std::ptr::null_mut();
    row.field_count = 0;
}
