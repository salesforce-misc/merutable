use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// A typed field value. `Bytes` covers both `ByteArray` and `FixedLenByteArray`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    Float(f32),
    Double(f64),
    Bytes(Bytes),
}

/// A complete table row. Fields are parallel to `TableSchema::columns`.
/// `None` = SQL NULL (only valid for nullable columns).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub fields: Vec<Option<FieldValue>>,
}

impl Row {
    pub fn new(fields: Vec<Option<FieldValue>>) -> Self {
        Self { fields }
    }

    /// Access a field by column index.
    pub fn get(&self, col_idx: usize) -> Option<&FieldValue> {
        self.fields.get(col_idx).and_then(Option::as_ref)
    }

    /// Extract PK field values in `primary_key` index order.
    /// Returns `Err` if any PK field is NULL.
    pub fn pk_values(&self, primary_key: &[usize]) -> crate::types::Result<Vec<FieldValue>> {
        primary_key
            .iter()
            .map(|&idx| {
                self.fields
                    .get(idx)
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or_else(|| {
                        crate::types::MeruError::InvalidArgument(format!(
                            "PK column at index {idx} is NULL"
                        ))
                    })
            })
            .collect()
    }

    /// Issue #12: validate a row against a schema before it enters the
    /// write path. Three checks in order, each with a column-pointed
    /// error message:
    ///
    /// 1. **Arity**: `fields.len()` must equal `schema.columns.len()`.
    ///    A too-short or too-long row corrupts the output Parquet
    ///    schema and makes "is this NULL or missing?" ambiguous on
    ///    read.
    /// 2. **Type compatibility**: each present field's `FieldValue`
    ///    discriminant must match the column's `ColumnType`. E.g.,
    ///    you cannot put a `Bytes` value into an `Int64` column.
    /// 3. **NOT NULL**: a `None` in a non-nullable column is rejected.
    ///
    /// Called at every write entry point (put, put_batch, apply_batch,
    /// internal engine.put) so malformed rows never reach the WAL or
    /// memtable. Cheap — just field iteration; no allocations.
    /// Issue #44 Stage 4: pad a row that was built under an older
    /// schema arity up to the current schema's column count by
    /// appending each missing tail column's `write_default` (or
    /// `None` if the column is nullable and no default is set).
    ///
    /// Called at every write entry point BEFORE `validate` so that
    /// a caller who omits newly-added columns doesn't get a row-
    /// arity mismatch. Non-evolution writes (where the row already
    /// matches the schema's arity) pay a single length check and
    /// no extra work.
    ///
    /// Errors if a missing column is non-nullable AND has no
    /// `write_default` — that's the one case the caller MUST fix
    /// by supplying the value, and it mirrors the constraint
    /// `check_schema_compatible` applies at reopen time (Stage 1).
    pub fn pad_with_defaults(
        &mut self,
        schema: &crate::types::schema::TableSchema,
    ) -> crate::types::Result<()> {
        if self.fields.len() >= schema.columns.len() {
            return Ok(());
        }
        for idx in self.fields.len()..schema.columns.len() {
            let col = &schema.columns[idx];
            let fill = col
                .write_default
                .clone()
                .or_else(|| col.initial_default.clone());
            if fill.is_none() && !col.nullable {
                return Err(crate::types::MeruError::SchemaMismatch(format!(
                    "row omits column {idx} '{}' which is NOT NULL and has no write_default — \
                     caller must provide a value",
                    col.name,
                )));
            }
            self.fields.push(fill);
        }
        Ok(())
    }

    pub fn validate(&self, schema: &crate::types::schema::TableSchema) -> crate::types::Result<()> {
        use crate::types::schema::ColumnType;
        if self.fields.len() != schema.columns.len() {
            return Err(crate::types::MeruError::SchemaMismatch(format!(
                "row arity mismatch: got {} fields, schema has {} columns ({})",
                self.fields.len(),
                schema.columns.len(),
                schema.table_name,
            )));
        }
        for (idx, (field_opt, col)) in self.fields.iter().zip(schema.columns.iter()).enumerate() {
            match field_opt {
                None => {
                    if !col.nullable {
                        return Err(crate::types::MeruError::SchemaMismatch(format!(
                            "column {idx} '{}' is NOT NULL but row field is None",
                            col.name,
                        )));
                    }
                }
                Some(fv) => {
                    let ok = matches!(
                        (fv, &col.col_type),
                        (FieldValue::Boolean(_), ColumnType::Boolean)
                            | (FieldValue::Int32(_), ColumnType::Int32)
                            | (FieldValue::Int64(_), ColumnType::Int64)
                            | (FieldValue::Float(_), ColumnType::Float)
                            | (FieldValue::Double(_), ColumnType::Double)
                            | (
                                FieldValue::Bytes(_),
                                ColumnType::ByteArray | ColumnType::FixedLenByteArray(_)
                            )
                    );
                    if !ok {
                        return Err(crate::types::MeruError::SchemaMismatch(format!(
                            "column {idx} '{}': field value type {} incompatible with column type {:?}",
                            col.name,
                            field_value_kind(fv),
                            col.col_type,
                        )));
                    }
                    // FixedLenByteArray length check: the PK-encoding
                    // path also checks this, but validate here so
                    // non-PK fixed-length columns get the same
                    // guarantee.
                    if let (FieldValue::Bytes(b), ColumnType::FixedLenByteArray(n)) =
                        (fv, &col.col_type)
                        && b.len() != *n as usize
                    {
                        return Err(crate::types::MeruError::SchemaMismatch(format!(
                            "column {idx} '{}': FixedLenByteArray({n}) got {} bytes",
                            col.name,
                            b.len(),
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

fn field_value_kind(fv: &FieldValue) -> &'static str {
    match fv {
        FieldValue::Boolean(_) => "Boolean",
        FieldValue::Int32(_) => "Int32",
        FieldValue::Int64(_) => "Int64",
        FieldValue::Float(_) => "Float",
        FieldValue::Double(_) => "Double",
        FieldValue::Bytes(_) => "Bytes",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_new_and_get() {
        let row = Row::new(vec![
            Some(FieldValue::Int64(42)),
            Some(FieldValue::Boolean(true)),
            None,
        ]);
        assert_eq!(row.get(0), Some(&FieldValue::Int64(42)));
        assert_eq!(row.get(1), Some(&FieldValue::Boolean(true)));
        assert_eq!(row.get(2), None); // NULL field
        assert_eq!(row.get(3), None); // out of bounds
    }

    #[test]
    fn row_default_is_empty() {
        let row = Row::default();
        assert!(row.fields.is_empty());
        assert_eq!(row.get(0), None);
    }

    #[test]
    fn pk_values_extracts_correctly() {
        let row = Row::new(vec![
            Some(FieldValue::Int64(1)),
            Some(FieldValue::Bytes(Bytes::from("hello"))),
            Some(FieldValue::Boolean(false)),
        ]);
        let pk = row.pk_values(&[0, 2]).unwrap();
        assert_eq!(pk, vec![FieldValue::Int64(1), FieldValue::Boolean(false)]);
    }

    #[test]
    fn pk_values_errors_on_null() {
        let row = Row::new(vec![Some(FieldValue::Int64(1)), None]);
        let result = row.pk_values(&[1]);
        assert!(result.is_err());
    }

    #[test]
    fn field_value_serde_roundtrip() {
        let values = vec![
            FieldValue::Boolean(true),
            FieldValue::Int32(42),
            FieldValue::Int64(-100),
            FieldValue::Float(1.23),
            FieldValue::Double(4.56789),
            FieldValue::Bytes(Bytes::from("test")),
        ];
        for v in &values {
            let json = serde_json::to_string(v).unwrap();
            let back: FieldValue = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    #[test]
    fn row_serde_roundtrip() {
        let row = Row::new(vec![
            Some(FieldValue::Int64(1)),
            None,
            Some(FieldValue::Bytes(Bytes::from("data"))),
        ]);
        let json = serde_json::to_vec(&row).unwrap();
        let back: Row = serde_json::from_slice(&json).unwrap();
        assert_eq!(back.fields.len(), 3);
        assert_eq!(back.get(0), Some(&FieldValue::Int64(1)));
        assert_eq!(back.get(1), None);
        assert_eq!(back.get(2), Some(&FieldValue::Bytes(Bytes::from("data"))));
    }
}
