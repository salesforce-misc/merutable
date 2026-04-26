#![cfg(feature = "replica")]
//! Issue #32 Phase 6: log-gap recovery.
//!
//! When the log source returns `ChangeFeedBelowRetention`,
//! `advance_or_recover()` hard-resets the replica to the latest
//! mirrored base snapshot with a fresh empty tail.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use merutable::MeruDB;
use merutable::replica::{AdvanceOutcome, InProcessLogSource, LogSource, OpRecord, Replica};
use merutable::types::{
    MeruError, Result,
    schema::{ColumnDef, ColumnType, TableSchema},
    value::{FieldValue, Row},
};

fn schema() -> TableSchema {
    TableSchema {
        table_name: "log-gap-test".into(),
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

/// Test double: wraps an InProcessLogSource and can be flipped
/// into "retention gap" mode via a flag. When gap=true, `stream`
/// returns `ChangeFeedBelowRetention`; when gap=false it delegates
/// to the inner source.
struct FlappyLogSource {
    inner: InProcessLogSource,
    gap_mode: Arc<AtomicBool>,
    low_water: u64,
}

impl FlappyLogSource {
    fn new(inner: InProcessLogSource, low_water: u64) -> (Self, Arc<AtomicBool>) {
        let gap_mode = Arc::new(AtomicBool::new(false));
        let flag = gap_mode.clone();
        (
            Self {
                inner,
                gap_mode,
                low_water,
            },
            flag,
        )
    }
}

#[async_trait]
impl LogSource for FlappyLogSource {
    async fn stream(&self, since: u64) -> Result<BoxStream<'static, Result<OpRecord>>> {
        if self.gap_mode.load(Ordering::SeqCst) {
            return Err(MeruError::ChangeFeedBelowRetention {
                requested: since,
                low_water: self.low_water,
            });
        }
        self.inner.stream(since).await
    }
    async fn latest_seq(&self) -> Result<u64> {
        self.inner.latest_seq().await
    }
}

#[tokio::test]
async fn advance_or_recover_returns_advanced_on_happy_path() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;
    primary
        .put(Row::new(vec![
            Some(FieldValue::Int64(1)),
            Some(FieldValue::Int64(1)),
        ]))
        .await
        .unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let inner = InProcessLogSource::new(primary.clone());
    let (flappy, _flag) = FlappyLogSource::new(inner, 0);
    let replica = Replica::open(base_opts, Arc::new(flappy)).await.unwrap();

    let outcome = replica.advance_or_recover().await.unwrap();
    assert_eq!(outcome, AdvanceOutcome::Advanced);
}

#[tokio::test]
async fn advance_or_recover_triggers_hotswap_on_retention_gap() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;
    primary
        .put(Row::new(vec![
            Some(FieldValue::Int64(1)),
            Some(FieldValue::Int64(10)),
        ]))
        .await
        .unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let inner = InProcessLogSource::new(primary.clone());
    let (flappy, flag) = FlappyLogSource::new(inner, 500);
    let replica = Replica::open(base_opts, Arc::new(flappy)).await.unwrap();

    // Primary advances (new flush adds a snapshot the mirror now
    // reflects). Then we flip the log source into gap mode so the
    // next advance pretends the subscriber has fallen off retention.
    primary
        .put(Row::new(vec![
            Some(FieldValue::Int64(2)),
            Some(FieldValue::Int64(20)),
        ]))
        .await
        .unwrap();
    primary.flush().await.unwrap();
    flag.store(true, Ordering::SeqCst);

    let stats_before = replica.stats().await;
    let outcome = replica.advance_or_recover().await.unwrap();
    match outcome {
        AdvanceOutcome::Recovered { new_base_seq } => {
            assert!(
                new_base_seq > stats_before.base_seq,
                "recovery advanced base_seq"
            );
        }
        AdvanceOutcome::Advanced => panic!("expected Recovered"),
    }
    // rebase_count advanced by exactly 1.
    let stats_after = replica.stats().await;
    assert_eq!(stats_after.rebase_count, stats_before.rebase_count + 1);
}

#[tokio::test]
async fn recover_from_log_gap_returns_new_base_seq() {
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let primary = open_primary(&primary_dir).await;
    primary
        .put(Row::new(vec![
            Some(FieldValue::Int64(1)),
            Some(FieldValue::Int64(1)),
        ]))
        .await
        .unwrap();
    primary.flush().await.unwrap();

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let log = Arc::new(InProcessLogSource::new(primary.clone()));
    let replica = Replica::open(base_opts, log).await.unwrap();
    let before = replica.base_seq();

    primary
        .put(Row::new(vec![
            Some(FieldValue::Int64(2)),
            Some(FieldValue::Int64(2)),
        ]))
        .await
        .unwrap();
    primary.flush().await.unwrap();

    let after = replica.recover_from_log_gap().await.unwrap();
    assert!(after > before);
    assert_eq!(replica.base_seq(), after);
}

#[tokio::test]
async fn non_retention_error_propagates_without_recovery() {
    // A LogSource that always returns an IO error — advance_or_recover
    // must propagate, not treat as retention gap.
    struct BrokenLogSource;
    #[async_trait]
    impl LogSource for BrokenLogSource {
        async fn stream(&self, _since: u64) -> Result<BoxStream<'static, Result<OpRecord>>> {
            // A lone error item in the stream — the tail's advance
            // loop short-circuits on the first Err.
            let items: Vec<Result<OpRecord>> =
                vec![Err(MeruError::ObjectStore("simulated I/O failure".into()))];
            Ok(Box::pin(futures::stream::iter(items)))
        }
        async fn latest_seq(&self) -> Result<u64> {
            Ok(0)
        }
    }
    let primary_dir = tempfile::tempdir().unwrap();
    let replica_dir = tempfile::tempdir().unwrap();
    let _primary = open_primary(&primary_dir).await;

    let base_opts = merutable::OpenOptions::new(schema())
        .wal_dir(replica_dir.path().join("wal"))
        .catalog_uri(primary_dir.path().to_string_lossy().to_string());
    let replica = Replica::open(base_opts, Arc::new(BrokenLogSource))
        .await
        .unwrap();
    let before = replica.stats().await.rebase_count;

    let err = replica.advance_or_recover().await.unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("simulated I/O failure"),
        "non-retention error surfaces verbatim: {msg}"
    );
    // rebase_count unchanged — non-retention errors do NOT trigger
    // recovery.
    assert_eq!(replica.stats().await.rebase_count, before);
}

/// Reference `StreamExt::boxed` in a dummy fn so the import isn't
/// flagged as unused by the compiler on the BrokenLogSource test
/// path (boxed() is used inside the trait impl above). This is a
/// cheaper alternative than an `#[allow(unused_imports)]` that
/// hides real future mistakes.
#[allow(dead_code)]
fn _keep_stream_ext_in_scope() {
    let _ = futures::stream::iter(std::iter::empty::<()>()).boxed();
}
