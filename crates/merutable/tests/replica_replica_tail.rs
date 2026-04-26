#![cfg(feature = "replica")]
//! Issue #32 Phase 3a: ReplicaTail + append-only advance.
//!
//! End-to-end: a primary MeruDB writes, an InProcessLogSource
//! bridges the change feed, a ReplicaTail absorbs and serves
//! point-lookups.

use std::sync::Arc;

use merutable::MeruDB;
use merutable::replica::{InProcessLogSource, ReplicaTail};
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};

fn schema() -> TableSchema {
    TableSchema {
        table_name: "replica-tail-test".into(),
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

async fn open_db(tmp: &tempfile::TempDir) -> Arc<MeruDB> {
    Arc::new(
        MeruDB::open(
            merutable::OpenOptions::new(schema())
                .wal_dir(tmp.path().join("wal"))
                .catalog_uri(tmp.path().to_string_lossy().to_string()),
        )
        .await
        .unwrap(),
    )
}

/// Phase 2b change records carry pk_bytes produced via
/// `InternalKey::encode_user_key`. Test lookups must use the same
/// encoding so the hash-map keys match.
fn pk_bytes(id: i64) -> Vec<u8> {
    merutable::types::key::InternalKey::encode_user_key(
        &[merutable::types::value::FieldValue::Int64(id)],
        &schema(),
    )
    .unwrap()
}

#[tokio::test]
async fn advance_absorbs_puts_and_resolves_get() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(&tmp).await;
    for i in 1..=5i64 {
        db.put(Row::new(vec![
            Some(FieldValue::Int64(i)),
            Some(FieldValue::Int64(i * 10)),
        ]))
        .await
        .unwrap();
    }

    let src = InProcessLogSource::new(db.clone());
    let mut tail = ReplicaTail::new();
    tail.advance(&src).await.unwrap();

    assert_eq!(tail.ops_applied(), 5);
    assert!(tail.visible_seq() >= 5);

    for i in 1..=5i64 {
        let pk = pk_bytes(i);
        let row = tail.get(&pk).expect("key present");
        match row.fields.get(1).and_then(|f| f.as_ref()) {
            Some(FieldValue::Int64(n)) => assert_eq!(*n, i * 10),
            other => panic!("unexpected v field: {other:?}"),
        }
    }
    // A never-written key is None.
    assert!(tail.get(&pk_bytes(99)).is_none());
}

#[tokio::test]
async fn advance_is_idempotent_across_repeated_calls() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(&tmp).await;
    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Int64(111)),
    ]))
    .await
    .unwrap();

    let src = InProcessLogSource::new(db);
    let mut tail = ReplicaTail::new();
    tail.advance(&src).await.unwrap();
    let seq_after_first = tail.visible_seq();
    let ops_after_first = tail.ops_applied();

    // Second advance sees nothing new (since_seq = visible_seq
    // already > boundary). ops_applied unchanged, visible_seq
    // unchanged.
    tail.advance(&src).await.unwrap();
    assert_eq!(tail.ops_applied(), ops_after_first);
    assert_eq!(tail.visible_seq(), seq_after_first);
}

#[tokio::test]
async fn advance_picks_up_new_writes_incrementally() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(&tmp).await;
    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Int64(10)),
    ]))
    .await
    .unwrap();

    let src = InProcessLogSource::new(db.clone());
    let mut tail = ReplicaTail::new();
    tail.advance(&src).await.unwrap();
    assert_eq!(tail.ops_applied(), 1);

    // Primary writes more; replica advances and picks up only
    // the new ops.
    db.put(Row::new(vec![
        Some(FieldValue::Int64(2)),
        Some(FieldValue::Int64(20)),
    ]))
    .await
    .unwrap();
    db.put(Row::new(vec![
        Some(FieldValue::Int64(3)),
        Some(FieldValue::Int64(30)),
    ]))
    .await
    .unwrap();

    tail.advance(&src).await.unwrap();
    assert_eq!(tail.ops_applied(), 3, "two new ops absorbed");
    assert!(tail.visible_seq() >= 3);

    for i in 1..=3i64 {
        assert!(tail.get(&pk_bytes(i)).is_some());
    }
}

#[tokio::test]
async fn put_then_update_resolves_to_latest() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(&tmp).await;
    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Int64(100)),
    ]))
    .await
    .unwrap();
    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Int64(999)),
    ]))
    .await
    .unwrap();

    let src = InProcessLogSource::new(db);
    let mut tail = ReplicaTail::new();
    tail.advance(&src).await.unwrap();

    let row = tail.get(&pk_bytes(1)).unwrap();
    match row.fields.get(1).and_then(|f| f.as_ref()) {
        Some(FieldValue::Int64(n)) => assert_eq!(*n, 999, "latest seq wins"),
        other => panic!("unexpected v: {other:?}"),
    }
}
