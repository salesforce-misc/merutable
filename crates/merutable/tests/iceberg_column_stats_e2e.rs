//! Issue #20 Part 2b end-to-end: Parquet row-group statistics emitted
//! at write time hoist through `ParquetFileMeta::column_stats` and
//! surface in the exported Iceberg `metadata.json`'s per-column maps.

use std::sync::Arc;

use bytes::Bytes;
use merutable::iceberg::{manifest::ManifestEntry, to_iceberg_data_file_v2_with_schema};
use merutable::parquet::writer::write_sorted_rows;
use merutable::types::{
    key::InternalKey,
    level::{FileFormat, Level},
    schema::{ColumnDef, ColumnType, TableSchema},
    sequence::{OpType, SeqNum},
    value::{FieldValue, Row},
};

fn schema() -> TableSchema {
    TableSchema {
        table_name: "stats".into(),
        columns: vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,

                ..Default::default()
            },
            ColumnDef {
                name: "val".into(),
                col_type: ColumnType::ByteArray,
                nullable: true,

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

/// Writer-side: writing rows must populate `ParquetFileMeta::column_stats`
/// with non-empty entries for every user column. Iceberg emission side:
/// projected Iceberg JSON must carry `column_sizes`, `value_counts`,
/// `lower_bounds`, and `upper_bounds` keyed by field id.
#[test]
fn writer_stats_propagate_through_iceberg_projection() {
    let schema = schema();
    let schema_arc = Arc::new(schema.clone());

    // Deterministic rows spanning a known id range.
    let mut rows: Vec<(InternalKey, Row)> = Vec::new();
    for id in 1..=100i64 {
        let ikey = make_ikey(id, id as u64, &schema);
        let row = Row::new(vec![
            Some(FieldValue::Int64(id)),
            Some(FieldValue::Bytes(Bytes::from(format!("val{id}")))),
        ]);
        rows.push((ikey, row));
    }

    let (_parquet_bytes, _bloom, meta) =
        write_sorted_rows(rows, schema_arc, Level(1), FileFormat::Columnar, 10).unwrap();

    // column_stats must be populated (writer reads back its own
    // row-group metadata).
    let stats = meta.column_stats.as_ref().expect("column_stats populated");
    assert!(
        stats.iter().any(|cs| cs.field_id == 1),
        "id column (field_id=1) stats expected"
    );
    let id_stats = stats.iter().find(|cs| cs.field_id == 1).unwrap();
    assert_eq!(id_stats.value_count, 100, "100 non-null ids");
    assert_eq!(id_stats.null_count, 0, "id column is non-nullable");
    assert!(id_stats.compressed_bytes > 0, "id column non-empty on disk");

    // Int64 min/max are LE bytes: lower=1, upper=100.
    let lb = id_stats
        .lower_bound
        .as_ref()
        .expect("id lower_bound present");
    let ub = id_stats
        .upper_bound
        .as_ref()
        .expect("id upper_bound present");
    assert_eq!(lb, &1i64.to_le_bytes().to_vec());
    assert_eq!(ub, &100i64.to_le_bytes().to_vec());

    // Iceberg projection: entry carries the stats forward to the
    // Iceberg data-file JSON.
    let entry = ManifestEntry {
        path: "data/L1/test.parquet".into(),
        meta,
        dv_path: None,
        dv_offset: None,
        dv_length: None,
        status: "added".into(),
        first_row_id: None,
    };
    let v = to_iceberg_data_file_v2_with_schema(&entry, Some(&schema));
    let df = &v["data_file"];

    // Non-empty stat maps.
    let col_sizes = df["column_sizes"].as_object().unwrap();
    let value_counts = df["value_counts"].as_object().unwrap();
    let null_counts = df["null_value_counts"].as_object().unwrap();
    let lower = df["lower_bounds"].as_object().unwrap();
    let upper = df["upper_bounds"].as_object().unwrap();

    assert!(col_sizes.contains_key("1"), "column_sizes[1] (id)");
    assert_eq!(value_counts["1"], 100);
    assert_eq!(null_counts["1"], 0);
    // Iceberg bounds are hex-encoded byte sequences. Decode and check.
    let lb_hex = lower["1"].as_str().unwrap();
    let ub_hex = upper["1"].as_str().unwrap();
    assert_eq!(hex::decode(lb_hex).unwrap(), 1i64.to_le_bytes().to_vec());
    assert_eq!(hex::decode(ub_hex).unwrap(), 100i64.to_le_bytes().to_vec());
}
