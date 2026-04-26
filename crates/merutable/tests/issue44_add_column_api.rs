//! Issue #44 Stages 2 + 4: runtime `MeruDB::add_column` API +
//! write-path `pad_with_defaults`.
//!
//! Stage 2 pins `add_column`'s contract: valid additive changes
//! persist cleanly (visible after reopen), invalid ones are
//! rejected at the API boundary.
//!
//! Stage 4 pins that `put` / `put_batch` auto-pad rows that were
//! built under an older arity when the new column has a
//! `write_default` or is nullable — and reject rows that omit a
//! NOT-NULL-no-default column.

use merutable::types::{
    MeruError,
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};
use merutable::{MeruDB, OpenOptions};

fn base_schema(name: &str) -> TableSchema {
    TableSchema {
        table_name: name.into(),
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

async fn open_db(path: &std::path::Path, schema: TableSchema) -> MeruDB {
    MeruDB::open(
        OpenOptions::new(schema)
            .wal_dir(path.join("wal"))
            .catalog_uri(path.to_string_lossy().to_string()),
    )
    .await
    .unwrap()
}

// ── Stage 2: add_column API ────────────────────────────────────

#[tokio::test]
async fn add_column_persists_additive_nullable() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(tmp.path(), base_schema("add-null")).await;
    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Int64(100)),
    ]))
    .await
    .unwrap();

    let new_col = ColumnDef {
        name: "extra".into(),
        col_type: ColumnType::Int64,
        nullable: true,
        ..Default::default()
    };
    let evolved = db.add_column(new_col).await.unwrap();
    assert_eq!(evolved.columns.len(), 3);
    assert_eq!(evolved.columns[2].name, "extra");
    assert!(evolved.columns[2].nullable);
    db.close().await.unwrap();

    // Reopen under the evolved schema — Stage 1 accepts; Stage 3
    // fills the missing column as None for pre-evolution rows.
    let db2 = open_db(tmp.path(), evolved.clone()).await;
    let got = db2.get(&[FieldValue::Int64(1)]).unwrap().unwrap();
    assert_eq!(got.fields.len(), 3);
    assert!(got.fields[2].is_none());
    db2.close().await.unwrap();
}

#[tokio::test]
async fn add_column_rejects_duplicate_name() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(tmp.path(), base_schema("add-dup")).await;
    let dup = ColumnDef {
        name: "v".into(), // already exists
        col_type: ColumnType::Int64,
        nullable: true,
        ..Default::default()
    };
    let err = db.add_column(dup).await.unwrap_err();
    match err {
        MeruError::SchemaMismatch(s) => {
            assert!(
                s.contains("already exists") && s.contains("'v'"),
                "msg: {s}"
            )
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
    db.close().await.unwrap();
}

#[tokio::test]
async fn add_column_rejects_non_nullable_without_default() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(tmp.path(), base_schema("add-noreq")).await;
    let bad = ColumnDef {
        name: "mandatory".into(),
        col_type: ColumnType::Int64,
        nullable: false,
        // no write_default / initial_default
        ..Default::default()
    };
    let err = db.add_column(bad).await.unwrap_err();
    match err {
        MeruError::SchemaMismatch(s) => {
            assert!(
                s.contains("NOT NULL") && s.contains("back-fill"),
                "msg: {s}"
            )
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
    db.close().await.unwrap();
}

#[tokio::test]
async fn add_column_accepts_non_nullable_with_default() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(tmp.path(), base_schema("add-def")).await;
    let withdef = ColumnDef {
        name: "filled".into(),
        col_type: ColumnType::Int64,
        nullable: false,
        write_default: Some(FieldValue::Int64(42)),
        initial_default: Some(FieldValue::Int64(42)),
        ..Default::default()
    };
    let evolved = db.add_column(withdef).await.unwrap();
    assert_eq!(
        evolved.columns[2].write_default,
        Some(FieldValue::Int64(42))
    );
    db.close().await.unwrap();
}

// ── Stage 4: pad_with_defaults on put ─────────────────────────

#[tokio::test]
async fn put_pads_short_row_with_write_default() {
    let tmp = tempfile::tempdir().unwrap();
    let mut schema = base_schema("pad-def");
    // Inline-declare the evolved schema on first open so Stage 1
    // treats it as the table's persisted shape.
    schema.columns.push(ColumnDef {
        name: "extra".into(),
        col_type: ColumnType::Int64,
        nullable: false,
        write_default: Some(FieldValue::Int64(-1)),
        initial_default: Some(FieldValue::Int64(-1)),
        ..Default::default()
    });
    let db = open_db(tmp.path(), schema).await;

    // Caller supplies a row with only 2 fields (old arity); Stage 4
    // pads the tail with write_default.
    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Int64(10)),
    ]))
    .await
    .unwrap();

    let got = db.get(&[FieldValue::Int64(1)]).unwrap().unwrap();
    assert_eq!(got.fields.len(), 3);
    assert_eq!(got.fields[2], Some(FieldValue::Int64(-1)));
    db.close().await.unwrap();
}

#[tokio::test]
async fn put_pads_short_row_nullable_as_none() {
    let tmp = tempfile::tempdir().unwrap();
    let mut schema = base_schema("pad-null");
    schema.columns.push(ColumnDef {
        name: "extra".into(),
        col_type: ColumnType::Int64,
        nullable: true,
        ..Default::default()
    });
    let db = open_db(tmp.path(), schema).await;

    db.put(Row::new(vec![
        Some(FieldValue::Int64(2)),
        Some(FieldValue::Int64(20)),
    ]))
    .await
    .unwrap();

    let got = db.get(&[FieldValue::Int64(2)]).unwrap().unwrap();
    assert_eq!(got.fields.len(), 3);
    assert!(got.fields[2].is_none());
    db.close().await.unwrap();
}

#[tokio::test]
async fn put_rejects_short_row_when_tail_column_is_required() {
    let tmp = tempfile::tempdir().unwrap();
    let mut schema = base_schema("pad-fail");
    schema.columns.push(ColumnDef {
        name: "mandatory".into(),
        col_type: ColumnType::Int64,
        nullable: false,
        // no defaults — caller MUST supply
        ..Default::default()
    });
    let db = open_db(tmp.path(), schema).await;

    let err = db
        .put(Row::new(vec![
            Some(FieldValue::Int64(3)),
            Some(FieldValue::Int64(30)),
        ]))
        .await
        .unwrap_err();
    match err {
        MeruError::SchemaMismatch(s) => {
            assert!(
                s.contains("mandatory") && s.contains("NOT NULL"),
                "msg: {s}"
            )
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
    db.close().await.unwrap();
}
