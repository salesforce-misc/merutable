//! Issue #29 Phase 2d: Arrow RecordBatch adapter for change-feed records.
//!
//! Converts a `Vec<ChangeRecord>` into the Arrow columnar form
//! DataFusion's TableProvider consumes. Landed ahead of the full
//! TableProvider wiring (Phase 2e) so the schema contract is
//! stable + testable without pulling DataFusion into the dep
//! graph yet.
//!
//! # Output schema
//!
//! Every change-feed query returns columns:
//!
//! ```text
//! seq:       UInt64   NOT NULL
//! op:        Utf8     NOT NULL   ("INSERT" | "UPDATE" | "DELETE")
//! pk_bytes:  Binary   NOT NULL
//! <user columns from TableSchema...>
//! ```
//!
//! User columns are nullable because a Delete record's pre-image
//! `Row` may be empty (when the key had no prior live state), in
//! which case every user-column field is Null. Insert/Update
//! records carry the post-state; its field_values follow the
//! schema's nullability.
//!
//! # Why seq + op + pk_bytes as explicit columns
//!
//! - Queries filter by `seq > N` directly (filter pushdown to the
//!   `since_seq` parameter happens in Phase 2e's TableProvider).
//! - `op` is the human-readable SQL label consumers branch on in
//!   WHERE clauses (`WHERE op = 'DELETE'`).
//! - `pk_bytes` lets downstream tools address the mutated key
//!   without decoding user columns — useful for replicas applying
//!   tombstones where user columns would be empty anyway.

use std::sync::Arc;

use crate::types::{
    MeruError, Result,
    schema::{ColumnType, TableSchema},
    value::FieldValue,
};
use arrow_array::{
    ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};

use crate::sql::ChangeRecord;

/// Arrow schema for the change feed over `table`. Stable across
/// `merutable_changes(...)` invocations — consumers can cache it.
pub fn change_feed_schema(table: &TableSchema) -> Arc<Schema> {
    let mut fields: Vec<Field> = Vec::with_capacity(3 + table.columns.len());
    fields.push(Field::new("seq", DataType::UInt64, false));
    fields.push(Field::new("op", DataType::Utf8, false));
    fields.push(Field::new("pk_bytes", DataType::Binary, false));
    for col in &table.columns {
        fields.push(Field::new(
            &col.name,
            column_type_to_arrow(&col.col_type),
            // User columns are always nullable at the feed level,
            // regardless of the table's nullability, because
            // Delete records with no pre-image produce Null rows.
            true,
        ));
    }
    Arc::new(Schema::new(fields))
}

fn column_type_to_arrow(t: &ColumnType) -> DataType {
    match t {
        ColumnType::Boolean => DataType::Boolean,
        ColumnType::Int32 => DataType::Int32,
        ColumnType::Int64 => DataType::Int64,
        ColumnType::Float => DataType::Float32,
        ColumnType::Double => DataType::Float64,
        ColumnType::ByteArray | ColumnType::FixedLenByteArray(_) => DataType::Binary,
    }
}

/// Materialize `records` as an Arrow RecordBatch. Schema matches
/// `change_feed_schema(table)`.
///
/// Returns an empty RecordBatch with the schema intact when
/// `records.is_empty()` — DataFusion prefers a zero-row batch to
/// None for "no rows yet" so the schema stays pinned.
pub fn records_to_record_batch(
    records: &[ChangeRecord],
    table: &TableSchema,
) -> Result<RecordBatch> {
    let schema = change_feed_schema(table);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

    // seq
    let seqs: Vec<u64> = records.iter().map(|r| r.seq).collect();
    columns.push(Arc::new(UInt64Array::from(seqs)));
    // op — the SQL-stable label from ChangeOp.as_sql_str.
    let ops: Vec<&str> = records.iter().map(|r| r.op.as_sql_str()).collect();
    columns.push(Arc::new(StringArray::from(ops)));
    // pk_bytes
    let pks: Vec<&[u8]> = records.iter().map(|r| r.pk_bytes.as_slice()).collect();
    columns.push(Arc::new(BinaryArray::from(pks)));

    // User columns. Each column is built independently; a row whose
    // `fields.len()` is zero (Delete with no pre-image) produces
    // Null across every user column.
    for (col_idx, col) in table.columns.iter().enumerate() {
        columns.push(build_user_column(&col.col_type, records, col_idx)?);
    }

    RecordBatch::try_new(schema, columns).map_err(|e| {
        MeruError::InvalidArgument(format!("change-feed RecordBatch assembly failed: {e}"))
    })
}

fn build_user_column(
    col_type: &ColumnType,
    records: &[ChangeRecord],
    col_idx: usize,
) -> Result<ArrayRef> {
    // Helper: extract FieldValue at col_idx for one record, or
    // None if the row is empty / column is NULL. Delete records
    // with no pre-image pass through as None uniformly.
    fn field_at(r: &ChangeRecord, col_idx: usize) -> Option<&FieldValue> {
        r.row.fields.get(col_idx).and_then(|f| f.as_ref())
    }
    match col_type {
        ColumnType::Boolean => Ok(Arc::new(BooleanArray::from(
            records
                .iter()
                .map(|r| match field_at(r, col_idx) {
                    Some(FieldValue::Boolean(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ColumnType::Int32 => Ok(Arc::new(Int32Array::from(
            records
                .iter()
                .map(|r| match field_at(r, col_idx) {
                    Some(FieldValue::Int32(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ColumnType::Int64 => Ok(Arc::new(Int64Array::from(
            records
                .iter()
                .map(|r| match field_at(r, col_idx) {
                    Some(FieldValue::Int64(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ColumnType::Float => Ok(Arc::new(Float32Array::from(
            records
                .iter()
                .map(|r| match field_at(r, col_idx) {
                    Some(FieldValue::Float(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ColumnType::Double => Ok(Arc::new(Float64Array::from(
            records
                .iter()
                .map(|r| match field_at(r, col_idx) {
                    Some(FieldValue::Double(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ColumnType::ByteArray | ColumnType::FixedLenByteArray(_) => {
            // BinaryArray needs Option<&[u8]>. Collect the owned
            // bytes to keep the reference chain straight.
            let owned: Vec<Option<bytes::Bytes>> = records
                .iter()
                .map(|r| match field_at(r, col_idx) {
                    Some(FieldValue::Bytes(b)) => Some(b.clone()),
                    _ => None,
                })
                .collect();
            let refs: Vec<Option<&[u8]>> = owned.iter().map(|o| o.as_deref()).collect();
            Ok(Arc::new(BinaryArray::from_opt_vec(refs)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::ChangeOp;
    use crate::types::{
        schema::ColumnDef,
        value::{FieldValue, Row},
    };
    use arrow_array::Array;

    fn two_col_schema() -> TableSchema {
        TableSchema {
            table_name: "arrow-test".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                    ..Default::default()
                },
                ColumnDef {
                    name: "v".into(),
                    col_type: ColumnType::Int64,
                    nullable: true,
                    ..Default::default()
                },
            ],
            primary_key: vec![0],
            ..Default::default()
        }
    }

    #[test]
    fn schema_includes_meta_and_user_columns() {
        let sch = change_feed_schema(&two_col_schema());
        assert_eq!(sch.fields().len(), 5);
        assert_eq!(sch.field(0).name(), "seq");
        assert_eq!(sch.field(0).data_type(), &DataType::UInt64);
        assert_eq!(sch.field(1).name(), "op");
        assert_eq!(sch.field(2).name(), "pk_bytes");
        assert_eq!(sch.field(3).name(), "id");
        assert_eq!(sch.field(4).name(), "v");
    }

    #[test]
    fn records_to_batch_populates_all_columns() {
        let table = two_col_schema();
        let records = vec![
            ChangeRecord {
                seq: 1,
                op: ChangeOp::Insert,
                row: Row::new(vec![
                    Some(FieldValue::Int64(1)),
                    Some(FieldValue::Int64(100)),
                ]),
                pk_bytes: vec![0xAA, 0xBB],
            },
            ChangeRecord {
                seq: 2,
                op: ChangeOp::Update,
                row: Row::new(vec![
                    Some(FieldValue::Int64(1)),
                    Some(FieldValue::Int64(200)),
                ]),
                pk_bytes: vec![0xAA, 0xBB],
            },
        ];
        let batch = records_to_record_batch(&records, &table).unwrap();
        assert_eq!(batch.num_rows(), 2);
        let seqs = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(seqs.value(0), 1);
        assert_eq!(seqs.value(1), 2);
        let ops = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ops.value(0), "INSERT");
        assert_eq!(ops.value(1), "UPDATE");
        let v = batch
            .column(4)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(v.value(0), 100);
        assert_eq!(v.value(1), 200);
    }

    #[test]
    fn delete_with_empty_row_produces_null_user_columns() {
        let table = two_col_schema();
        let records = vec![ChangeRecord {
            seq: 10,
            op: ChangeOp::Delete,
            row: Row::default(), // no pre-image
            pk_bytes: vec![0xFF],
        }];
        let batch = records_to_record_batch(&records, &table).unwrap();
        assert_eq!(batch.num_rows(), 1);
        // User columns must be Null for a Delete without pre-image.
        let id = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(id.is_null(0));
        let v = batch
            .column(4)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(v.is_null(0));
        // Meta columns are populated.
        let seqs = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(seqs.value(0), 10);
        let ops = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ops.value(0), "DELETE");
    }

    #[test]
    fn empty_records_produces_schema_with_zero_rows() {
        let batch = records_to_record_batch(&[], &two_col_schema()).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 5);
    }
}
