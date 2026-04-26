use thiserror::Error;

#[derive(Error, Debug)]
pub enum MeruError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("corruption: {0}")]
    Corruption(String),

    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),

    #[error("key not found")]
    NotFound,

    #[error("object store error: {0}")]
    ObjectStore(String),

    #[error("parquet error: {0}")]
    Parquet(String),

    #[error("iceberg error: {0}")]
    Iceberg(String),

    #[error("WAL error: {0}")]
    Wal(String),

    #[error("compaction error: {0}")]
    Compaction(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("operation not permitted: database is read-only")]
    ReadOnly,

    #[error("database is closed")]
    Closed,

    /// Issue #26: create-only PUT lost a race.
    ///
    /// Returned by `MeruStore::put_if_absent` when the target path
    /// already exists. Callers handle this by refetching HEAD,
    /// rebuilding their manifest on top, and retrying. Not an error
    /// condition — the *expected* non-error outcome of losing a race.
    #[error("object already exists: {0}")]
    AlreadyExists(String),

    /// Issue #29: change-feed caller requested a `since_seq` below
    /// the engine's retention low-water.
    ///
    /// The change feed is bounded by `[low_water, visible_seq]`
    /// where `low_water` is the oldest retained seq in the LSM
    /// (driven by #11 snapshot-pin GC policy). A caller with stale
    /// bookmarks must escalate to an Iceberg snapshot scan and
    /// restart the feed from the seq embedded in that snapshot —
    /// this matches Debezium + pg_logical's escalation pattern.
    /// Critically, the change feed does NOT hold back retention;
    /// stale consumers don't pin the LSM.
    #[error(
        "change feed below retention: requested {requested}, low_water {low_water} — escalate to Iceberg snapshot scan"
    )]
    ChangeFeedBelowRetention { requested: u64, low_water: u64 },
}

pub type Result<T> = std::result::Result<T, MeruError>;
