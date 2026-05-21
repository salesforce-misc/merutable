//! Python ↔ Rust type conversion.

use ::merutable::types::{
    schema::{ColumnType, TableSchema},
    value::{FieldValue, Row},
};
use bytes::Bytes;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict, PyString},
};

/// Convert a Python dict to a `Row`, using the schema for type dispatch.
///
/// Errors if the dict contains keys not in the schema (catches typos like
/// `"nmae"` instead of `"name"` that would otherwise silently become NULL).
pub fn dict_to_row(dict: &Bound<'_, PyDict>, schema: &TableSchema) -> PyResult<Row> {
    // Reject extra keys early — catches typos that would silently NULL a column.
    for key_obj in dict.keys() {
        let key: String = key_obj.extract()?;
        if schema.column_by_name(&key).is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown column '{key}' in row dict (not in schema)"
            )));
        }
    }

    let mut fields: Vec<Option<FieldValue>> = Vec::with_capacity(schema.columns.len());
    for col in &schema.columns {
        let key = &col.name;
        match dict.get_item(key)? {
            None => {
                if !col.nullable {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "non-nullable column '{key}' missing from row dict"
                    )));
                }
                fields.push(None);
            }
            Some(val) => {
                if val.is_none() {
                    if !col.nullable {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "non-nullable column '{key}' cannot be None"
                        )));
                    }
                    fields.push(None);
                } else {
                    fields.push(Some(py_to_field_value(&val, &col.col_type, key)?));
                }
            }
        }
    }
    Ok(Row::new(fields))
}

/// Convert a single Python object to a `FieldValue` given the expected column type.
fn py_to_field_value(
    obj: &Bound<'_, PyAny>,
    col_type: &ColumnType,
    col_name: &str,
) -> PyResult<FieldValue> {
    match col_type {
        ColumnType::Boolean => {
            let v: bool = obj.extract()?;
            Ok(FieldValue::Boolean(v))
        }
        ColumnType::Int32 => {
            let v: i32 = obj.extract()?;
            Ok(FieldValue::Int32(v))
        }
        ColumnType::Int64 => {
            let v: i64 = obj.extract()?;
            Ok(FieldValue::Int64(v))
        }
        ColumnType::Float => {
            let v: f32 = obj.extract()?;
            Ok(FieldValue::Float(v))
        }
        ColumnType::Double => {
            let v: f64 = obj.extract()?;
            Ok(FieldValue::Double(v))
        }
        ColumnType::ByteArray => {
            // Accept both str and bytes from Python.
            if let Ok(s) = obj.downcast::<PyString>() {
                let s: String = s.extract()?;
                Ok(FieldValue::Bytes(Bytes::from(s)))
            } else if let Ok(b) = obj.downcast::<PyBytes>() {
                let b: Vec<u8> = b.extract()?;
                Ok(FieldValue::Bytes(Bytes::from(b)))
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "column '{col_name}' (ByteArray) expects str or bytes, got {}",
                    obj.get_type().name()?
                )))
            }
        }
        ColumnType::FixedLenByteArray(expected_len) => {
            let b: Vec<u8> = obj.extract()?;
            if b.len() != *expected_len as usize {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "column '{col_name}' (FixedLenByteArray({expected_len})) expects {expected_len} bytes, got {}",
                    b.len()
                )));
            }
            Ok(FieldValue::Bytes(Bytes::from(b)))
        }
    }
}

/// Convert a `Row` to a Python dict, using the schema for column names.
pub fn row_to_dict<'py>(
    py: Python<'py>,
    row: &Row,
    schema: &TableSchema,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for (i, col) in schema.columns.iter().enumerate() {
        match row.get(i) {
            None => {
                dict.set_item(&col.name, py.None())?;
            }
            Some(fv) => {
                dict.set_item(&col.name, field_value_to_py(py, fv)?)?;
            }
        }
    }
    Ok(dict)
}

/// Convert a `FieldValue` to a Python object.
fn field_value_to_py(py: Python<'_>, fv: &FieldValue) -> PyResult<PyObject> {
    match fv {
        FieldValue::Boolean(v) => Ok(v.into_py(py)),
        FieldValue::Int32(v) => Ok(v.into_py(py)),
        FieldValue::Int64(v) => Ok(v.into_py(py)),
        FieldValue::Float(v) => Ok(v.into_py(py)),
        FieldValue::Double(v) => Ok(v.into_py(py)),
        FieldValue::Bytes(b) => Ok(PyBytes::new_bound(py, b).into_py(py)),
    }
}

/// Convert Python PK values (positional tuple) to `Vec<FieldValue>`.
pub fn pk_args_to_field_values(
    args: &Bound<'_, pyo3::types::PyTuple>,
    schema: &TableSchema,
) -> PyResult<Vec<FieldValue>> {
    if args.len() != schema.primary_key.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {} PK value(s), got {}",
            schema.primary_key.len(),
            args.len()
        )));
    }
    let mut pk_values = Vec::with_capacity(schema.primary_key.len());
    for (i, &col_idx) in schema.primary_key.iter().enumerate() {
        let col = &schema.columns[col_idx];
        let obj = args.get_item(i)?;
        pk_values.push(py_to_field_value(&obj, &col.col_type, &col.name)?);
    }
    Ok(pk_values)
}

/// Public wrapper for `py_to_field_value` (used by `lib.rs` for scan bounds).
pub fn py_to_field_value_pub(
    obj: &Bound<'_, PyAny>,
    col_type: &ColumnType,
    col_name: &str,
) -> PyResult<FieldValue> {
    py_to_field_value(obj, col_type, col_name)
}

/// Parse a column type string from Python to `ColumnType`.
pub fn parse_column_type(type_str: &str) -> PyResult<ColumnType> {
    match type_str.to_lowercase().as_str() {
        "bool" | "boolean" => Ok(ColumnType::Boolean),
        "int32" | "i32" => Ok(ColumnType::Int32),
        "int64" | "i64" | "int" => Ok(ColumnType::Int64),
        "float" | "f32" | "float32" => Ok(ColumnType::Float),
        "double" | "f64" | "float64" => Ok(ColumnType::Double),
        "bytes" | "bytearray" | "string" | "str" => Ok(ColumnType::ByteArray),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown column type '{type_str}'. valid: bool, int32, int64, float, double, bytes"
        ))),
    }
}
