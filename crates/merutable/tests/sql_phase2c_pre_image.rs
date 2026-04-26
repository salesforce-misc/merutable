#![cfg(feature = "sql")]
//! Issue #29 Phase 2c: delete pre-image reconstruction + INSERT vs
//! UPDATE discrimination.

use std::sync::Arc;

use merutable::engine::{config::EngineConfig, engine::MeruEngine};
use merutable::sql::{ChangeFeedCursor, ChangeOp};
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};

fn test_schema() -> TableSchema {
    TableSchema {
        table_name: "cf-phase2c".into(),
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

async fn open_engine(tmp: &tempfile::TempDir) -> Arc<MeruEngine> {
    open_engine_with_l0_trigger(tmp, 4).await
}

async fn open_engine_with_l0_trigger(
    tmp: &tempfile::TempDir,
    l0_compaction_trigger: usize,
) -> Arc<MeruEngine> {
    let cfg = EngineConfig {
        schema: test_schema(),
        catalog_uri: tmp.path().to_string_lossy().to_string(),
        object_store_prefix: tmp.path().to_string_lossy().to_string(),
        wal_dir: tmp.path().join("wal"),
        l0_compaction_trigger,
        ..Default::default()
    };
    MeruEngine::open(cfg).await.unwrap()
}

fn row(id: i64, v: i64) -> Row {
    Row::new(vec![
        Some(FieldValue::Int64(id)),
        Some(FieldValue::Int64(v)),
    ])
}

#[tokio::test]
async fn insert_vs_update_distinguished_by_prior_state() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = open_engine(&tmp).await;
    // seq 1: put id=1, v=1 (Insert — no prior state).
    engine
        .put(vec![FieldValue::Int64(1)], row(1, 1))
        .await
        .unwrap();
    // seq 2: put id=1, v=2 (Update — prior state exists).
    engine
        .put(vec![FieldValue::Int64(1)], row(1, 2))
        .await
        .unwrap();
    // seq 3: put id=2, v=20 (Insert — no prior state for id=2).
    engine
        .put(vec![FieldValue::Int64(2)], row(2, 20))
        .await
        .unwrap();

    let mut cur = ChangeFeedCursor::from_engine(engine, 0);
    let batch = cur.next_batch(100).unwrap();
    assert_eq!(batch.len(), 3);
    assert_eq!(batch[0].op, ChangeOp::Insert, "first put of id=1 is Insert");
    assert_eq!(
        batch[1].op,
        ChangeOp::Update,
        "second put of id=1 is Update"
    );
    assert_eq!(batch[2].op, ChangeOp::Insert, "first put of id=2 is Insert");
}

#[tokio::test]
async fn delete_pre_image_reconstructed_from_memtable() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = open_engine(&tmp).await;
    engine
        .put(vec![FieldValue::Int64(42)], row(42, 9999))
        .await
        .unwrap();
    engine.delete(vec![FieldValue::Int64(42)]).await.unwrap();

    let mut cur = ChangeFeedCursor::from_engine(engine, 0);
    let batch = cur.next_batch(10).unwrap();
    let del = batch
        .iter()
        .find(|r| matches!(r.op, ChangeOp::Delete))
        .unwrap();
    // Pre-image carries id=42, v=9999.
    assert_eq!(del.row.fields.len(), 2);
    match del.row.fields.first().and_then(|f| f.as_ref()) {
        Some(FieldValue::Int64(n)) => assert_eq!(*n, 42),
        other => panic!("pre-image id: {other:?}"),
    }
    match del.row.fields.get(1).and_then(|f| f.as_ref()) {
        Some(FieldValue::Int64(n)) => assert_eq!(*n, 9999),
        other => panic!("pre-image v: {other:?}"),
    }
}

#[tokio::test]
async fn delete_pre_image_reconstructed_across_l0_flush() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = open_engine(&tmp).await;
    // Put + flush so the pre-image lives in L0 now.
    engine
        .put(vec![FieldValue::Int64(5)], row(5, 55))
        .await
        .unwrap();
    engine.flush().await.unwrap();
    // Delete the row from memtable (post-flush).
    engine.delete(vec![FieldValue::Int64(5)]).await.unwrap();

    let mut cur = ChangeFeedCursor::from_engine(engine, 0);
    let batch = cur.next_batch(10).unwrap();
    let del = batch
        .iter()
        .find(|r| matches!(r.op, ChangeOp::Delete))
        .unwrap();
    assert_eq!(
        del.row.fields.len(),
        2,
        "pre-image reconstructed across flush boundary"
    );
    match del.row.fields.get(1).and_then(|f| f.as_ref()) {
        Some(FieldValue::Int64(n)) => assert_eq!(*n, 55),
        other => panic!("pre-image v: {other:?}"),
    }
}

#[tokio::test]
async fn skip_update_discrimination_tags_every_put_as_insert() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = open_engine(&tmp).await;
    engine
        .put(vec![FieldValue::Int64(1)], row(1, 1))
        .await
        .unwrap();
    engine
        .put(vec![FieldValue::Int64(1)], row(1, 2))
        .await
        .unwrap();

    let mut cur = ChangeFeedCursor::from_engine(engine, 0).skip_update_discrimination(true);
    let batch = cur.next_batch(10).unwrap();
    for rec in &batch {
        assert_eq!(
            rec.op,
            ChangeOp::Insert,
            "skip_update_discrimination forces Insert for every Put"
        );
    }
}

/// Issue #33 regression: the DELETE pre-image MUST survive
/// compaction. Under stress, a Put→Delete pair on the same key
/// can be compacted to L1 before the change feed drains; if
/// pre-image reconstruction relied on looking up the prior Put
/// at `seq-1`, the Put would be gone (compaction merged it away
/// behind the tombstone). The fix (#33) captures the pre-image
/// inline on the tombstone at write time.
/// Issue #33: the DELETE pre-image survives the flush → L0
/// transition (the hot path the stress test exercises). Before
/// this fix, pre-image capture happened at write time into the
/// memtable, but flush discarded `value` for Delete entries and
/// wrote a PK-only row. The post-hoc `point_lookup_at(seq-1)`
/// worked only until the Put file got merged into L1 with the
/// tombstone dropping it. Once the Put was gone from every level,
/// the feed emitted empty pre-images under the steady state the
/// stress test hit at 2 GB.
///
/// The fix spans THREE call sites:
/// 1. Write-time capture (engine.rs / write_path.rs): delete value
///    carries the pre-image bytes.
/// 2. Flush preservation (flush.rs): value bytes are decoded as
///    the Row and written to the tombstone's typed columns +
///    `_merutable_value` blob, so post-flush reads observe the
///    pre-image directly.
/// 3. Scan decode (engine.rs scan_*): Delete entries with
///    non-empty values decode as the pre-image row.
///
/// This test covers the flush transition — the DELETE is served
/// from L0 (not memtable, not L1). Compaction-away-into-L1 is a
/// separate concern: the change feed's Phase 2b scope is
/// memtable+L0; once compacted into L1 the op is no longer a
/// "tail" record. A user who requires post-compaction visibility
/// of every historical op needs a different durability tier
/// (future work; outside #33).
#[tokio::test]
async fn issue_33_delete_pre_image_survives_flush() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = open_engine(&tmp).await;

    // Put → flush (Put now in L0).
    engine
        .put(vec![FieldValue::Int64(7)], row(7, 12345))
        .await
        .unwrap();
    engine.flush().await.unwrap();

    // Delete → flush (Delete now in L0). Critical: the tombstone
    // must carry its pre-image through the flush, so reads from
    // L0 after this point decode the row as (id=7, v=12345).
    engine.delete(vec![FieldValue::Int64(7)]).await.unwrap();
    engine.flush().await.unwrap();

    // Memtable should now be empty — feed must surface the DELETE
    // from L0 and the pre-image must come from the flushed value
    // bytes, not from a post-hoc lookup.
    let mut cur = ChangeFeedCursor::from_engine(engine, 0);
    let batch = cur.next_batch(100).unwrap();
    let del = batch
        .iter()
        .find(|r| matches!(r.op, ChangeOp::Delete))
        .expect("delete op present in feed");
    assert!(
        !del.row.fields.is_empty(),
        "#33 regression: DELETE at seq {} returned empty pre-image",
        del.seq
    );
    match del.row.fields.first().and_then(|f| f.as_ref()) {
        Some(FieldValue::Int64(n)) => assert_eq!(*n, 7, "pre-image id"),
        other => panic!("pre-image id: {other:?}"),
    }
    match del.row.fields.get(1).and_then(|f| f.as_ref()) {
        Some(FieldValue::Int64(n)) => assert_eq!(*n, 12345, "pre-image v"),
        other => panic!("pre-image v: {other:?}"),
    }
}

/// Issue #33 corollary: same-batch Put+Delete on the same key.
/// The delete's pre-image correctly reflects the Put that
/// preceded it in the batch — not via write-time capture (which
/// sees pre-batch state = None), but via Phase 2c's post-hoc
/// point_lookup_at_seq which finds the Put@seq-1 still in the
/// memtable. The two mechanisms COMPOSE:
///   - Write-time capture → pre-image in the tombstone's value.
///   - Post-hoc lookup → overrides with the authoritative state
///     at `seq-1`, preferred when findable.
///   - Falls through to the stored value when the Put has been
///     compacted away.
/// This test pins the same-batch behavior so a future refactor
/// that touches either mechanism doesn't silently lose it.
#[tokio::test]
async fn issue_33_same_batch_put_then_delete_surfaces_put_as_pre_image() {
    use merutable::engine::write_path::{MutationBatch, apply_batch};
    let tmp = tempfile::tempdir().unwrap();
    let engine = open_engine(&tmp).await;

    let mut batch = MutationBatch::new();
    batch.put(vec![FieldValue::Int64(9)], row(9, 999));
    batch.delete(vec![FieldValue::Int64(9)]);
    apply_batch(&engine, batch).await.unwrap();

    let mut cur = ChangeFeedCursor::from_engine(engine, 0);
    let recs = cur.next_batch(100).unwrap();
    let del = recs
        .iter()
        .find(|r| matches!(r.op, ChangeOp::Delete))
        .expect("delete in batch");
    match del.row.fields.get(1).and_then(|f| f.as_ref()) {
        Some(FieldValue::Int64(n)) => assert_eq!(*n, 999),
        other => panic!("same-batch delete pre-image v: {other:?}"),
    }
}

#[tokio::test]
async fn delete_without_prior_state_returns_empty_pre_image() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = open_engine(&tmp).await;
    // Delete a key that was never written — a legitimately weird
    // but valid op (idempotent tombstone insertion).
    engine.delete(vec![FieldValue::Int64(999)]).await.unwrap();

    let mut cur = ChangeFeedCursor::from_engine(engine, 0);
    let batch = cur.next_batch(10).unwrap();
    let del = batch
        .iter()
        .find(|r| matches!(r.op, ChangeOp::Delete))
        .unwrap();
    // No prior state → empty Row (Phase 2c contract: "pre-image
    // available or Row::default()").
    assert_eq!(del.row.fields.len(), 0);
}
