#![cfg(feature = "replica")]
//! Issue #32 Phase 2: InProcessLogSource drives real ops out of a
//! running MeruDB primary for a co-located replica.

use std::sync::Arc;

use futures::StreamExt;
use merutable::MeruDB;
use merutable::replica::{InProcessLogSource, LogSource};
use merutable::sql::ChangeOp;
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};

fn schema() -> TableSchema {
    TableSchema {
        table_name: "replica-test".into(),
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

#[tokio::test]
async fn stream_yields_ops_above_since_in_seq_order() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(&tmp).await;
    for i in 1..=3i64 {
        db.put(Row::new(vec![
            Some(FieldValue::Int64(i)),
            Some(FieldValue::Int64(i * 10)),
        ]))
        .await
        .unwrap();
    }

    let src = InProcessLogSource::new(db.clone());
    let mut stream = src.stream(0).await.unwrap();
    let mut seqs = Vec::new();
    let mut ops = Vec::new();
    while let Some(r) = stream.next().await {
        let rec = r.unwrap();
        seqs.push(rec.seq);
        ops.push(rec.op);
    }
    assert_eq!(seqs.len(), 3);
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "ascending seq");
    assert_eq!(ops, vec![ChangeOp::Insert; 3]);
}

#[tokio::test]
async fn stream_skips_ops_at_or_below_since() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(&tmp).await;
    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Int64(10)),
    ]))
    .await
    .unwrap();
    let boundary = db.read_seq().0;
    db.put(Row::new(vec![
        Some(FieldValue::Int64(2)),
        Some(FieldValue::Int64(20)),
    ]))
    .await
    .unwrap();

    let src = InProcessLogSource::new(db);
    let mut stream = src.stream(boundary).await.unwrap();
    let mut count = 0;
    while let Some(r) = stream.next().await {
        let rec = r.unwrap();
        assert!(rec.seq > boundary);
        count += 1;
    }
    assert_eq!(count, 1, "only the op strictly above boundary shows");
}

#[tokio::test]
async fn latest_seq_tracks_primary_read_seq() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(&tmp).await;
    db.put(Row::new(vec![
        Some(FieldValue::Int64(1)),
        Some(FieldValue::Int64(1)),
    ]))
    .await
    .unwrap();
    db.put(Row::new(vec![
        Some(FieldValue::Int64(2)),
        Some(FieldValue::Int64(2)),
    ]))
    .await
    .unwrap();
    let src = InProcessLogSource::new(db.clone());
    let latest = src.latest_seq().await.unwrap();
    assert_eq!(latest, db.read_seq().0);
    assert!(latest >= 2);
}

#[tokio::test]
async fn batch_size_coerces_zero_to_one() {
    let tmp = tempfile::tempdir().unwrap();
    let db = open_db(&tmp).await;
    db.put(Row::new(vec![
        Some(FieldValue::Int64(42)),
        Some(FieldValue::Int64(0)),
    ]))
    .await
    .unwrap();
    // Explicit zero must not hang or panic. The source coerces to
    // 1 so each stream() call emits rows one at a time but still
    // converges.
    let src = InProcessLogSource::new(db).with_batch_size(0);
    let mut stream = src.stream(0).await.unwrap();
    let rec = stream.next().await.unwrap().unwrap();
    assert_eq!(rec.seq, 1);
}
