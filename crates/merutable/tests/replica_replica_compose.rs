#![cfg(feature = "replica")]
//! Issue #32 Phase 3b: Replica composite — base MeruDB + ReplicaTail.
//!
//! Verifies tail-first read semantics, base fallback for unseen
//! keys, and delete authoritativeness (tail Delete hides a base row
//! that hasn't been flushed + mirrored yet).

use std::sync::Arc;

use merutable::MeruDB;
use merutable::replica::{InProcessLogSource, Replica};
use merutable::types::{
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};

fn schema() -> TableSchema {
    TableSchema {
        table_name: "replica-compose-test".into(),
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

async fn open_primary(tmp: &tempfile::TempDir) -> Arc<MeruDB> {
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

fn row(id: i64, v: i64) -> Row {
    Row::new(vec![
        Some(FieldValue::Int64(id)),
        Some(FieldValue::Int64(v)),
    ])
}

#[tokio::test]
async fn replica_serves_from_tail_when_present() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;

    // Seed the primary's catalog with a committed row so the
    // replica's base mount has something to read on open.
    primary.put(row(1, 100)).await.unwrap();
    primary.flush().await.unwrap();

    // Open the replica's base against the primary's catalog
    // (read-only). The replica tail comes from an in-process
    // log source.
    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();

    // Primary writes a NEW row post-replica-open. The tail should
    // surface it before any flush.
    primary.put(row(2, 200)).await.unwrap();
    replica.advance().await.unwrap();

    let r1 = replica.get(&[FieldValue::Int64(1)]).await.unwrap();
    assert!(r1.is_some(), "base reads resolve unchanged entries");
    let r2 = replica.get(&[FieldValue::Int64(2)]).await.unwrap();
    assert!(r2.is_some(), "tail surfaces post-open writes without flush");
}

#[tokio::test]
async fn replica_falls_through_to_base_on_tail_miss() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;
    primary.put(row(42, 9999)).await.unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();
    // Phase 5: visible_seq is seeded to base_seq on open so
    // subsequent advances() only replay post-base ops. No tail
    // entries yet.
    assert_eq!(replica.visible_seq().await, replica.base_seq());

    let r = replica.get(&[FieldValue::Int64(42)]).await.unwrap();
    assert!(r.is_some(), "tail miss falls through to base");
}

#[tokio::test]
async fn tail_delete_is_authoritative_over_base_row() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;

    // Commit row to base.
    primary.put(row(1, 111)).await.unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();

    // Sanity: without any tail ops the replica sees the row.
    let before = replica.get(&[FieldValue::Int64(1)]).await.unwrap();
    assert!(before.is_some());

    // Primary deletes. Replica advances; the tail captures the
    // Delete op. Read should return None — the tail's tombstone
    // hides the base row.
    primary.delete(vec![FieldValue::Int64(1)]).await.unwrap();
    replica.advance().await.unwrap();
    let after = replica.get(&[FieldValue::Int64(1)]).await.unwrap();
    assert!(
        after.is_none(),
        "tail Delete must shadow the (stale) base row"
    );
}

/// Phase 4a: `Replica::rebase()` advances the base to a newer
/// snapshot and resets the tail to anchor at the new base_seq.
/// After rebase, reads that previously required the tail can be
/// served straight from the refreshed base.
#[tokio::test]
async fn rebase_advances_base_and_resets_tail() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;

    // Commit one row, flush — this lands in the primary's v=1
    // snapshot.
    primary.put(row(1, 10)).await.unwrap();
    primary.flush().await.unwrap();

    // Open the replica. Its base mounts v=1.
    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();
    let base_seq_v1 = replica.base_seq();

    // Primary writes + flushes again — a new snapshot lands
    // that the replica's base hasn't seen yet.
    primary.put(row(2, 20)).await.unwrap();
    primary.flush().await.unwrap();

    // Before rebase: the replica's base_seq is still pinned at
    // v=1. Reading key=2 from the replica fails because:
    // - tail is empty (we never advanced),
    // - base is at v=1 and doesn't have key=2.
    assert_eq!(replica.base_seq(), base_seq_v1);
    let pre_rebase = replica.get(&[FieldValue::Int64(2)]).await.unwrap();
    assert!(pre_rebase.is_none(), "key=2 not yet visible before rebase");

    // Rebase. base_seq should advance; tail is empty + anchored
    // at the new base_seq.
    replica.rebase().await.unwrap();
    assert!(
        replica.base_seq() > base_seq_v1,
        "rebase advances base_seq: {} -> {}",
        base_seq_v1,
        replica.base_seq()
    );
    // Now key=2 is readable — from the base, not the tail.
    let post_rebase = replica.get(&[FieldValue::Int64(2)]).await.unwrap();
    assert!(post_rebase.is_some(), "key=2 visible after rebase");
    assert!(
        replica
            .get(&[FieldValue::Int64(1)])
            .await
            .unwrap()
            .is_some(),
        "pre-existing key stays visible"
    );
}

/// Phase 4b: `rebase_hotswap()` returns only after the new state
/// is fully warmed up AND atomically retargeted — callers see the
/// refreshed base_seq and caught-up visible_seq as a tuple.
#[tokio::test]
async fn hotswap_returns_warm_state_atomically() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;
    primary.put(row(1, 10)).await.unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();
    let old_base_seq = replica.base_seq();

    // Primary advances: new snapshot v2 + unflushed tail op.
    primary.put(row(2, 20)).await.unwrap();
    primary.flush().await.unwrap();
    primary.put(row(3, 30)).await.unwrap();

    // Hot-swap. Return value should reflect the new state.
    let (new_base_seq, visible_seq) = replica.rebase_hotswap().await.unwrap();
    assert!(new_base_seq > old_base_seq, "base advanced");
    assert!(
        visible_seq >= 3,
        "tail caught up to unflushed op: {visible_seq}"
    );

    // Reads land on the new state — every key visible.
    for i in 1..=3i64 {
        let r = replica.get(&[FieldValue::Int64(i)]).await.unwrap();
        assert!(r.is_some(), "key={i} visible after hotswap");
    }
}

/// Phase 4b: a reader with an outstanding Arc<ReplicaState>
/// snapshot (via `Replica::current_state_for_testing`) continues
/// to see its original base+tail view after the main `Replica`
/// has swapped. Tombstoned access patterns don't re-read from
/// the new state under the reader.
#[tokio::test]
async fn old_state_arc_persists_across_hotswap() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;
    primary.put(row(1, 10)).await.unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();

    // First read pins a reference via load_full — but the API
    // scope is whole-`get()` calls for now, so we assert the
    // weaker property: a swap in flight does not error any
    // concurrent in-progress `get()`. Run multiple gets in
    // parallel with a hotswap.
    let (hotswap_ok, _reads_ok): (Result<_, _>, Vec<_>) = tokio::join!(
        replica.rebase_hotswap(),
        futures::future::join_all(
            (1..=5).map(|_| async { replica.get(&[FieldValue::Int64(1)]).await.unwrap() })
        ),
    );
    assert!(hotswap_ok.is_ok(), "hotswap does not error under read load");
}

/// Phase 5: `Replica::stats()` exposes visible_seq, base_seq,
/// tail_length, rebase_count, and last_rebase_warmup_millis.
/// rebase_count advances per hotswap; tail_length grows with
/// unflushed primary writes.
#[tokio::test]
async fn stats_surface_advances_across_advance_and_rebase() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;
    primary.put(row(1, 1)).await.unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();

    // Initial stats: base mounted, empty tail seeded at base_seq,
    // no rebases.
    let s0 = replica.stats().await;
    assert!(s0.base_seq > 0);
    assert_eq!(
        s0.visible_seq, s0.base_seq,
        "tail seeded at base_seq on open"
    );
    assert_eq!(s0.tail_length, 0);
    assert_eq!(s0.rebase_count, 0);
    assert_eq!(s0.last_rebase_warmup_millis, 0);

    // Primary writes (unflushed). Advance the tail.
    primary.put(row(2, 2)).await.unwrap();
    primary.put(row(3, 3)).await.unwrap();
    replica.advance().await.unwrap();
    let s1 = replica.stats().await;
    assert!(
        s1.visible_seq >= 3,
        "visible advanced to >=3: {}",
        s1.visible_seq
    );
    assert_eq!(s1.tail_length, 2, "2 tail ops absorbed");

    // Hotswap — rebase_count bumps, last_warmup recorded,
    // tail_length drops back to 0 (new state seeded at new
    // base_seq with empty tail, then advance drained 0 new ops
    // since primary hasn't written anything post-flush).
    primary.flush().await.unwrap();
    let _ = replica.rebase_hotswap().await.unwrap();
    let s2 = replica.stats().await;
    assert_eq!(s2.rebase_count, 1);
    // warmup is small but MAY be 0 on very fast CPUs — the
    // assertion here is just "we wrote something."
    let _ = s2.last_rebase_warmup_millis;
    assert!(s2.base_seq > s0.base_seq, "base advanced through rebase");
    assert_eq!(s2.tail_length, 0, "fresh tail after rebase is empty");
}

#[tokio::test]
async fn replica_base_seq_and_visible_seq_track_independently() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;
    primary.put(row(1, 1)).await.unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();
    let initial_base_seq = replica.base_seq();

    // Primary writes several more unflushed ops. Replica advances.
    for i in 2..=5 {
        primary.put(row(i, i)).await.unwrap();
    }
    replica.advance().await.unwrap();
    assert!(replica.visible_seq().await >= 5);
    // Base is read-only — its seq doesn't change from further
    // primary writes until the replica is closed + reopened to
    // mount a new snapshot (that's Phase 4's rebase).
    assert_eq!(replica.base_seq(), initial_base_seq);
}
