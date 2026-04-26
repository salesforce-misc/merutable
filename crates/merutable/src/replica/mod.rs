//! Issue #32: scale-out RO replica for merutable.
//!
//! The replica composes two sources to serve a fresh view of a
//! primary:
//!
//! - **Base**: an object-store layout (#26 / #31 mirror target).
//!   Read via `crate::OpenOptions::read_only(true) +
//!   the object-store layout`.
//! - **Tail**: a streamed log of ops newer than the base. The log
//!   source is pluggable; see [`LogSource`] for the contract.
//!
//! The replica advances in two modes:
//!
//! 1. **Append-only tail advance** (common case): new log ops
//!    replay into the in-memory tail, `visible_seq` advances.
//!    Cheap; monotonic; no rebase.
//! 2. **Hot-swap rebase** (on mirror advance): a new `ReplicaState`
//!    is warmed up against the new base snapshot in parallel with
//!    the old one continuing to serve reads. When the new state
//!    catches up, an `ArcSwap` pointer atomically retargets.
//!
//! # Phases
//!
//! - **Phase 1 (shipped)**: `LogSource` trait, `OpRecord` type shape,
//!   `LogGap` error, placeholder `ChangeFeedLogSource` stub.
//! - **Phase 2 (shipped)**: `InProcessLogSource` — a `LogSource`
//!   implementation that pulls from a co-located `Arc<MeruDB>` via
//!   #29 Phase 2a's `ChangeFeedCursor`.
//! - **Phase 3a (shipped)**: `ReplicaTail` — in-memory tail state +
//!   `advance()` that replays ops from any `LogSource`. PK-indexed
//!   `get()` resolves last-writer-wins.
//! - **Phase 3b (shipped)**: `Replica` — composite type pairing a
//!   read-only base `MeruDB` with a `ReplicaTail`. Reads go
//!   tail-first, fall through to the base on miss.
//! - **Phase 4a (shipped)**: `Replica::rebase()` — manual
//!   stop-the-world rebase that refreshes the base to the latest
//!   mirrored snapshot, resets the tail, and catches up.
//! - **Phase 4b (shipped)**: zero-gap hot-swap via parallel warmup
//!   + `ArcSwap<ReplicaState>` retarget.
//! - **Phase 5 (shipped)**: metrics surface — `Replica::stats()`
//!   returns visible_seq / base_seq / tail_length / rebase_count /
//!   last_rebase_warmup_millis.
//! - **Phase 6 (this commit)**: log-gap recovery. When the log
//!   source returns `ChangeFeedBelowRetention`, the replica can
//!   hard-reset by opening a fresh base at the latest-mirrored
//!   snapshot with an empty tail (via `advance_or_recover()` or
//!   the explicit `recover_from_log_gap()` call). Stress-test
//!   harness is a future, outside-the-crate effort.

use crate::sql::ChangeOp;
use crate::types::{MeruError, Result, value::Row};
use async_trait::async_trait;
use futures::stream::BoxStream;

/// A single log op visible to the replica. Same shape as a
/// change-feed record except that the replica needs `op_type`
/// explicitly (so it can apply tombstones) and consumes `row` as
/// owned data (the stream hands it off).
///
/// The seq defines ordering; replicas MUST observe ops in seq-ascending
/// order and MUST reject out-of-order delivery as a corrupted source
/// (the `LogSource` contract guarantees ordering).
#[derive(Clone, Debug)]
pub struct OpRecord {
    pub seq: u64,
    pub op: ChangeOp,
    pub row: Row,
    /// PK-encoded bytes of the affected key. Populated for every
    /// op, including deletes where `row` is empty. Replicas key
    /// their tail index on these bytes.
    pub pk_bytes: Vec<u8>,
}

/// A log source's view of the replica's starting point was below
/// the source's earliest retained seq. The replica's only recourse
/// is a hard reset: pick a new base snapshot from the object store
/// and rebuild the tail from `mirror_seq` forward.
///
/// Separate from `MeruError::ChangeFeedBelowRetention` (which is
/// the primary's retention-bound error for `merutable_changes`);
/// the replica layer surfaces its own variant so callers can
/// distinguish "primary said below retention" from "my tail source
/// timed out".
#[derive(Clone, Debug)]
pub struct LogGap {
    /// The seq the replica asked the source to stream from.
    pub requested: u64,
    /// The source's earliest available seq (if it can report one).
    pub earliest_available: Option<u64>,
    /// Human-readable reason for the gap (e.g. "Kafka offset
    /// retention exceeded", "change-feed low-water advanced").
    pub reason: String,
}

impl std::fmt::Display for LogGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "log gap: requested={}, earliest_available={:?}, reason={}",
            self.requested, self.earliest_available, self.reason
        )
    }
}

impl std::error::Error for LogGap {}

/// Pluggable log source. v1 implementation will be
/// [`ChangeFeedLogSource`] consuming `merutable_changes` over
/// Flight SQL. Future implementations — `RaftLogSource`,
/// `KafkaLogSource`, `ObjectStoreLogSource` — plug in by
/// implementing this trait without touching the replica core.
#[async_trait]
pub trait LogSource: Send + Sync + 'static {
    /// Stream ops with `seq > since` in seq-ascending order.
    /// Source-defined retention; returns `Err(LogGap)` if `since`
    /// is below the source's earliest available seq.
    async fn stream(&self, since: u64) -> Result<BoxStream<'static, Result<OpRecord>>>;

    /// Source-side latest-known seq. Best-effort; the stream is
    /// the source of truth. Used by the replica to decide when to
    /// kick a refresh.
    async fn latest_seq(&self) -> Result<u64>;
}

/// Phase 1 placeholder. Returns `LogGap` on every call so the v1
/// impl contract is pinned and callers can exercise their
/// hard-reset recovery path today. Real impl in Phase 2 consumes
/// `merutable_changes(table, since_seq)` over the primary's Flight
/// SQL endpoint.
pub struct ChangeFeedLogSource {
    /// The primary's retention low-water as of the last probe.
    /// Included in `LogGap::earliest_available` so the replica
    /// knows where to restart from.
    pub primary_low_water: u64,
}

impl ChangeFeedLogSource {
    pub fn new(primary_low_water: u64) -> Self {
        Self { primary_low_water }
    }
}

#[async_trait]
impl LogSource for ChangeFeedLogSource {
    async fn stream(&self, since: u64) -> Result<BoxStream<'static, Result<OpRecord>>> {
        Err(MeruError::ChangeFeedBelowRetention {
            requested: since,
            low_water: self.primary_low_water,
        })
    }

    async fn latest_seq(&self) -> Result<u64> {
        Err(MeruError::InvalidArgument(
            "ChangeFeedLogSource::latest_seq: Phase 2 pending (requires #29 Phase 2 \
             Flight SQL endpoint)"
                .into(),
        ))
    }
}

/// Issue #32 Phase 2: co-located log source for single-process
/// replica setups (tests, benchmarks, single-host external analytics demos).
///
/// Pulls ops directly from an `Arc<MeruDB>` via the change-feed
/// cursor (#29 Phase 2a). No network, no serialization, no
/// retention handshake — stream just ends when the cursor returns
/// an empty batch.
///
/// Limitations (by design — Phase 2 scope, lifted in Phase 2b/3):
/// - Memtable-only visibility inherited from #29 Phase 2a. Flushed
///   ops don't surface until #29 Phase 2b adds L0 scan.
/// - `latest_seq()` returns the primary's current read_seq; under
///   contention it can race ahead of what a subsequent `stream()`
///   call actually yields (an op could be committed between the
///   two calls). Callers use it as a wake-up hint, not a pinned
///   upper bound.
pub struct InProcessLogSource {
    db: std::sync::Arc<crate::MeruDB>,
    /// Batch size passed to `ChangeFeedCursor::next_batch`. Default
    /// 4096 — large enough that a single stream() call amortizes
    /// lock-acquisition overhead, small enough to keep memory
    /// bounded.
    batch_size: usize,
}

impl InProcessLogSource {
    pub fn new(db: std::sync::Arc<crate::MeruDB>) -> Self {
        Self {
            db,
            batch_size: 4096,
        }
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }
}

#[async_trait]
impl LogSource for InProcessLogSource {
    async fn stream(&self, since: u64) -> Result<BoxStream<'static, Result<OpRecord>>> {
        use futures::stream::StreamExt;
        // Issue #34 fix: snapshot the primary's latest seq AT
        // stream-open time and drain only up to that bound.
        //
        // Pre-#34, this loop called `cursor.next_batch` until it
        // returned empty. But `next_batch` captures a FRESH
        // `engine.read_seq()` on each call — so under a live
        // primary with steady writes, the target kept advancing
        // and the drain never terminated. `rebase_hotswap()` and
        // `advance()` both drain the stream and both hung
        // indefinitely (observed: 35+ min against a 1.7 MB/s
        // primary, replica RSS ballooned 10x disk size as the
        // tail accumulated every op since seq 0).
        //
        // The replica contract is "catch up to the primary's NOW
        // at drain-open time." Ops committed AFTER we started are
        // out of scope for this drain; the next advance/rebase
        // will pick them up. This makes stream() snapshot-
        // semantic and guaranteed to terminate regardless of
        // primary write rate.
        let engine = self.db.engine_for_replica();
        let upper = engine.read_seq().0;
        let records: Vec<Result<OpRecord>> = if since >= upper {
            Vec::new()
        } else {
            let mut cursor = crate::sql::ChangeFeedCursor::from_engine(engine, since);
            let mut out: Vec<Result<OpRecord>> = Vec::new();
            loop {
                let batch = cursor.next_batch(self.batch_size)?;
                if batch.is_empty() {
                    break;
                }
                let mut reached_upper = false;
                for r in batch {
                    if r.seq > upper {
                        // Cursor's internal read_seq raced past
                        // our snapshot bound. Drop — next drain
                        // will pick it up.
                        reached_upper = true;
                        break;
                    }
                    out.push(Ok(OpRecord {
                        seq: r.seq,
                        op: r.op,
                        row: r.row,
                        pk_bytes: r.pk_bytes,
                    }));
                }
                if reached_upper || cursor.since_seq() >= upper {
                    break;
                }
            }
            out
        };
        Ok(futures::stream::iter(records).boxed())
    }

    async fn latest_seq(&self) -> Result<u64> {
        Ok(self.db.read_seq().0)
    }
}

/// Issue #32 Phase 3a: in-memory tail state.
///
/// Maintains every op the replica has replayed from its `LogSource`,
/// indexed by primary-key bytes for last-writer-wins point lookups
/// and by seq for feed inspection. `advance()` drains a log-source
/// stream into the tail and bumps `visible_seq`.
///
/// Scope: tail-only. `get()` returns `None` both for "deleted" and
/// for "never seen in the tail." Phase 3b lifts the latter to a
/// fall-through read of the object-store base; until then, the
/// tail is the whole world and keys below `since_seq` at
/// construction time are invisible.
///
/// Not thread-safe — callers wrap in `RwLock<ReplicaTail>` or
/// equivalent. Phase 4's hot-swap machinery does this via
/// `ArcSwap<ReplicaTail>` so a reader's snapshot isn't torn by a
/// concurrent advance.
pub struct ReplicaTail {
    /// Latest op per user-key. Last-writer-wins by seq; a later
    /// Delete overwrites an earlier Put.
    index: std::collections::HashMap<Vec<u8>, TailEntry>,
    /// Highest seq the tail has absorbed. Zero on a fresh tail.
    visible_seq: u64,
    /// Count of ops ever applied — monotonic, for observability.
    /// Doesn't shrink when deletes overwrite earlier puts.
    ops_applied: u64,
}

/// Internal tail entry — the op-type + payload Row, tagged with seq
/// for debug/LWW resolution. `row` is meaningful only for
/// `op == ChangeOp::Insert` / `Update`; for `Delete` it's stored
/// as an empty `Row` (pre-image reconstruction is #29 Phase 2c).
#[derive(Clone, Debug)]
struct TailEntry {
    seq: u64,
    op: ChangeOp,
    row: crate::types::value::Row,
}

impl Default for ReplicaTail {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplicaTail {
    pub fn new() -> Self {
        Self {
            index: std::collections::HashMap::new(),
            visible_seq: 0,
            ops_applied: 0,
        }
    }

    /// Reader-visible seq: the highest seq the tail has applied.
    pub fn visible_seq(&self) -> u64 {
        self.visible_seq
    }

    /// Issue #32 Phase 4a: seed the tail's `visible_seq` to a
    /// starting floor. Used by `Replica::rebase` to anchor a
    /// freshly-reset tail at the new base's seq so the next
    /// `advance()` only streams ops strictly newer than the base.
    /// No-op if `seq <= self.visible_seq`.
    pub fn seed_visible_seq(&mut self, seq: u64) {
        if seq > self.visible_seq {
            self.visible_seq = seq;
        }
    }

    /// Lifetime count of ops absorbed. Useful for metrics +
    /// regression tests that want to see work actually happened.
    pub fn ops_applied(&self) -> u64 {
        self.ops_applied
    }

    /// Point-lookup by PK-encoded bytes. Returns the most recent
    /// op the tail has for this key:
    /// - `Some(Row)` for an Insert/Update.
    /// - `None` for a Delete OR a key the tail has never seen.
    ///
    /// Phase 3b will add a fall-through to the object-store base
    /// so "never seen in tail" becomes a proper cache miss instead
    /// of masquerading as a Delete.
    pub fn get(&self, pk_bytes: &[u8]) -> Option<&crate::types::value::Row> {
        let entry = self.index.get(pk_bytes)?;
        match entry.op {
            ChangeOp::Insert | ChangeOp::Update => Some(&entry.row),
            ChangeOp::Delete => None,
        }
    }

    /// Apply a single op into the tail. Last-writer-wins — an op
    /// with `seq <= existing.seq` for the same key is silently
    /// dropped (pins the contract against an out-of-order log
    /// source). Advances `visible_seq` to the max seq seen.
    ///
    /// Primary-key derivation: the `OpRecord` carries a full `Row`,
    /// from which we extract PK bytes via `schema.primary_key` on
    /// the index columns. The replica doesn't yet carry a schema
    /// (Phase 3b wires it), so Phase 3a uses the caller-supplied
    /// `pk_bytes` directly.
    pub fn apply(&mut self, pk_bytes: Vec<u8>, op_record: OpRecord) {
        if let Some(existing) = self.index.get(&pk_bytes) {
            if op_record.seq <= existing.seq {
                // Out-of-order or duplicate — silent drop. A real
                // log source (#32 v1's InProcessLogSource) never
                // produces this, but bad actors shouldn't corrupt
                // state.
                return;
            }
        }
        let seq = op_record.seq;
        self.index.insert(
            pk_bytes,
            TailEntry {
                seq,
                op: op_record.op,
                row: op_record.row,
            },
        );
        if seq > self.visible_seq {
            self.visible_seq = seq;
        }
        self.ops_applied += 1;
    }

    /// Drain a log-source stream into the tail. Each `OpRecord`
    /// now carries its `pk_bytes` directly (#29 Phase 2b), so the
    /// tail no longer needs a caller-supplied extractor — it
    /// simply uses `rec.pk_bytes`.
    pub async fn advance<S>(&mut self, source: &S) -> Result<()>
    where
        S: LogSource + ?Sized,
    {
        use futures::stream::StreamExt;
        let since = self.visible_seq;
        let mut stream = source.stream(since).await?;
        while let Some(rec) = stream.next().await {
            let rec = rec?;
            let pk = rec.pk_bytes.clone();
            self.apply(pk, rec);
        }
        Ok(())
    }
}

/// Phase 4b atomic read target: one base + one tail, swapped as a
/// unit when `rebase_hotswap()` retargets. The tail is held behind
/// a `RwLock` so `advance()` can mutate without cloning the entire
/// state.
pub struct ReplicaState {
    base: std::sync::Arc<crate::MeruDB>,
    tail: tokio::sync::RwLock<ReplicaTail>,
}

/// Issue #32 Phase 3b: composite replica — base `MeruDB` + tail.
///
/// Reads go tail-first: if the tail has a live entry for the PK,
/// return it. If the tail has a tombstone, return None (the tail
/// records the delete). If the tail has nothing for the PK, fall
/// through to the base `MeruDB::get` (which reads the object-store
/// snapshot the replica is mounted against).
///
/// `advance()` drains the `LogSource` into the tail. Callers
/// typically run this on a timer or driven by
/// `LogSource::latest_seq` polls.
///
/// Phase 4b wraps the (base, tail) pair behind `ArcSwap<State>` so
/// `rebase_hotswap()` can retarget new readers atomically without
/// interrupting in-flight ones.
/// Phase 5: replica observability snapshot. Every field is an
/// instantaneous value; no running totals that drift across calls.
/// Cheap to compute — reads two AtomicUsize/u64 fields + one
/// RwLock::read on the live tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaStats {
    /// The reader-visible seq (tail's visible_seq). Lag relative
    /// to the primary is `primary.read_seq - this`.
    pub visible_seq: u64,
    /// The base MeruDB's read_seq — the snapshot the current
    /// state is mounted against. Advances on `rebase*()` calls.
    pub base_seq: u64,
    /// Number of ops currently held in the tail index. Grows
    /// between rebases, drops to ~0 after a rebase (the new
    /// tail is seeded at base_seq and only holds post-mirror ops).
    pub tail_length: usize,
    /// Lifetime count of completed hot-swap rebases. Useful as
    /// a sanity check: flat → rebase worker stalled; climbing
    /// linearly → healthy mirror cadence.
    pub rebase_count: u64,
    /// Wall-clock milliseconds the most recent `rebase_hotswap`
    /// took to open the new base + drain the log. Zero before any
    /// hotswap has run. Callers build histograms off this.
    pub last_rebase_warmup_millis: u64,
}

pub struct Replica {
    state: arc_swap::ArcSwap<ReplicaState>,
    log_source: std::sync::Arc<dyn LogSource>,
    schema: std::sync::Arc<crate::types::schema::TableSchema>,
    /// Phase 5 counters. Atomic so `stats()` is lock-free at the
    /// counter layer; the tail snapshot still needs a brief
    /// `RwLock::read`.
    rebase_count: std::sync::atomic::AtomicU64,
    last_rebase_warmup_millis: std::sync::atomic::AtomicU64,
    /// Issue #34: `rebase_hotswap` deadline in milliseconds. Zero
    /// disables the timeout (unbounded). Default 60_000 — 60s is
    /// generous for warm catalog reopens + log-source drain;
    /// hangs beyond that indicate the primary's write rate
    /// exceeds replica drain throughput and the caller should
    /// escalate (raise timeout, throttle writes, or rebuild the
    /// replica). Before #34, there was no timeout — a hang meant
    /// the caller await'd forever. See
    /// `Replica::set_rebase_timeout`.
    rebase_timeout_millis: std::sync::atomic::AtomicU64,
    /// OpenOptions used to spawn the initial base; cloned + used
    /// again by `rebase_hotswap` to open a fresh read-only MeruDB
    /// for the new state. The new MeruDB reads the latest
    /// committed manifest (via the POSIX commit path's
    /// version-hint.text or ObjectStore's HEAD discovery), so
    /// re-open is equivalent to `base.refresh()` on a shared base
    /// — but the two states get INDEPENDENT Version snapshots,
    /// which is what keeps in-flight readers on the old state
    /// from seeing post-rebase data.
    base_opts: crate::OpenOptions,
}

impl Replica {
    /// Open a replica. `base_options` must set `read_only(true)` —
    /// the replica never originates writes. The `log_source` supplies
    /// the tail stream; construction does NOT drain it, so the caller
    /// chooses when to catch up.
    pub async fn open(
        mut base_options: crate::OpenOptions,
        log_source: std::sync::Arc<dyn LogSource>,
    ) -> Result<Self> {
        base_options = base_options.read_only(true);
        let schema = std::sync::Arc::new(base_options.schema.clone());
        let base_opts = base_options.clone();
        let base = std::sync::Arc::new(crate::MeruDB::open(base_options).await?);
        // Phase 5 refinement: seed the initial tail's visible_seq
        // to base_seq so an immediate advance() only pulls ops
        // strictly newer than the base snapshot. Mirrors the
        // rebase_hotswap path where the new tail is always
        // anchored at new_base_seq. Before this refinement, a
        // replica opening against a flushed-primary would replay
        // every op that was also in the base — correct (LWW
        // dedups) but burns memory proportional to catalog size.
        let mut initial_tail = ReplicaTail::new();
        let base_seq = base.read_seq().0;
        if base_seq > 0 {
            initial_tail.seed_visible_seq(base_seq);
        }
        let state = std::sync::Arc::new(ReplicaState {
            base,
            tail: tokio::sync::RwLock::new(initial_tail),
        });
        Ok(Self {
            state: arc_swap::ArcSwap::new(state),
            log_source,
            schema,
            base_opts,
            rebase_count: std::sync::atomic::AtomicU64::new(0),
            last_rebase_warmup_millis: std::sync::atomic::AtomicU64::new(0),
            rebase_timeout_millis: std::sync::atomic::AtomicU64::new(60_000),
        })
    }

    /// Issue #34: tune `rebase_hotswap`'s deadline. `None`
    /// disables the timeout; `Some(d)` sets it to `d`. Storage
    /// is milliseconds; values are clamped to `u64::MAX ms`.
    /// Default (set in `open`) is 60s.
    pub fn set_rebase_timeout(&self, timeout: Option<std::time::Duration>) {
        let millis = timeout.map(|d| d.as_millis() as u64).unwrap_or(0);
        self.rebase_timeout_millis
            .store(millis, std::sync::atomic::Ordering::Relaxed);
    }

    /// Currently configured `rebase_hotswap` deadline. `None` if
    /// timeouts are disabled.
    pub fn rebase_timeout(&self) -> Option<std::time::Duration> {
        let m = self
            .rebase_timeout_millis
            .load(std::sync::atomic::Ordering::Relaxed);
        if m == 0 {
            None
        } else {
            Some(std::time::Duration::from_millis(m))
        }
    }

    /// Phase 5: observability snapshot. Cheap — atomic loads +
    /// one brief RwLock::read on the live tail. No allocations
    /// beyond the returned struct.
    pub async fn stats(&self) -> ReplicaStats {
        let state = self.state.load_full();
        let base_seq = state.base.read_seq().0;
        let (visible_seq, tail_length) = {
            let tail = state.tail.read().await;
            (tail.visible_seq(), tail.ops_applied() as usize)
        };
        ReplicaStats {
            visible_seq,
            base_seq,
            tail_length,
            rebase_count: self.rebase_count.load(std::sync::atomic::Ordering::Relaxed),
            last_rebase_warmup_millis: self
                .last_rebase_warmup_millis
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Drain new ops from the log source into the tail of the
    /// currently-live state. Idempotent across repeated calls.
    pub async fn advance(&self) -> Result<()> {
        let state = self.state.load_full();
        let mut tail = state.tail.write().await;
        tail.advance(self.log_source.as_ref()).await
    }

    /// Point lookup. Tail-first, base-fallback.
    ///
    /// Semantics for the tail-first walk:
    /// - Tail has a Put/Update for the key → return the row.
    /// - Tail has a Delete for the key → return `None` (delete is
    ///   authoritative; tail is newer than the base's snapshot).
    /// - Tail has no entry → fall through to `base.get()`.
    ///
    /// The second bullet is the crucial distinction from a naïve
    /// `tail.get().or_else(base.get)`. A Delete in the tail means
    /// "the primary observed a delete at seq > base_seq"; the base
    /// may still have the row because its snapshot predates the
    /// delete. Returning `None` here matches the semantics a reader
    /// on the primary would observe.
    pub async fn get(
        &self,
        pk_values: &[crate::types::value::FieldValue],
    ) -> Result<Option<crate::types::value::Row>> {
        let encoded = crate::types::key::InternalKey::encode_user_key(pk_values, &self.schema)?;
        let state = self.state.load_full();
        {
            let tail = state.tail.read().await;
            if tail.index.contains_key(&encoded) {
                // Authoritative: Put returns the row, Delete returns None.
                return Ok(tail.get(&encoded).cloned());
            }
        }
        state.base.get(pk_values)
    }

    /// Reader-visible seq. Equals the tail's visible_seq of the
    /// currently-live state — the base's snapshot id is tracked
    /// separately via `base_seq()` because the two advance at
    /// different rates.
    pub async fn visible_seq(&self) -> u64 {
        let state = self.state.load_full();
        let seq = state.tail.read().await.visible_seq();
        seq
    }

    /// Base snapshot's current seq for the currently-live state.
    pub fn base_seq(&self) -> u64 {
        self.state.load_full().base.read_seq().0
    }

    /// Issue #32 Phase 4a: rebase the replica onto a newer base
    /// snapshot. Calls `MeruDB::refresh()` to pick up commits that
    /// have landed on the mirror since open, then resets the tail
    /// and advances it from the new `base_seq` forward.
    ///
    /// This is a stop-the-world rebase: the tail is briefly empty
    /// between the reset and the advance, during which point
    /// lookups that would have hit the tail cache miss and fall
    /// through to the base. The gap is bounded by the drain time
    /// of the log source's `stream` call; for an in-process
    /// source that's microseconds.
    ///
    /// Phase 4b will make this zero-gap via `ArcSwap<State>`:
    /// warm a new state in parallel, swap on completion, drain
    /// old state on a TTL.
    ///
    /// Callers typically poll `LogSource::latest_seq` on a timer
    /// and compare against `visible_seq` / the mirror's published
    /// `mirror_seq` to decide when to invoke this.
    pub async fn rebase(&self) -> Result<()> {
        // Advance the base MeruDB to the newest manifest on its
        // underlying catalog. For a POSIX-mounted base this
        // re-reads version-hint.text and reloads the current
        // manifest; for the object-store layout (once wired) it
        // re-discovers HEAD and reloads.
        let state = self.state.load_full();
        state.base.refresh().await?;

        let new_base_seq = state.base.read_seq().0;
        {
            let mut tail = state.tail.write().await;
            // Reset the tail to an empty state anchored at the
            // new base_seq. The tail's internal visible_seq
            // becomes the new starting point for log-source
            // streaming.
            *tail = ReplicaTail::new();
            // We could load a starting visible_seq by taking the
            // max of (old visible_seq, new base_seq) to avoid
            // re-replaying ops the old tail already absorbed. Use
            // new_base_seq as the floor instead: ops <= base_seq
            // are guaranteed to be in the base by construction
            // (mirror uploaded the snapshot before advancing
            // low_water), so starting there is always correct.
            // The log source's range filter prunes anyway.
            if new_base_seq > 0 {
                tail.seed_visible_seq(new_base_seq);
            }
            tail.advance(self.log_source.as_ref()).await?;
        }
        Ok(())
    }

    /// Issue #32 Phase 4b: zero-gap rebase via parallel warmup +
    /// ArcSwap retarget.
    ///
    /// 1. Open a FRESH read-only `MeruDB` at the latest-committed
    ///    manifest (the stored `OpenOptions` are the same ones the
    ///    initial state was opened with; reopening picks up any
    ///    new snapshot that has landed since).
    /// 2. Build a new `ReplicaState` around that base with an
    ///    empty tail seeded at the new `base_seq`.
    /// 3. Warm the new tail by draining the log source from
    ///    `new_base_seq` forward. All this time, the OLD state is
    ///    still serving reads — in-flight readers hold an `Arc`
    ///    to it and are unaffected.
    /// 4. Atomically swap the `ArcSwap<State>` pointer. New
    ///    incoming reads land on the warm new state. The old
    ///    state drops when the last in-flight reader releases it.
    ///
    /// Returns the (new_base_seq, new_visible_seq) tuple as a
    /// confirmation signal for caller timers / metrics exporters.
    pub async fn rebase_hotswap(&self) -> Result<(u64, u64)> {
        let start = std::time::Instant::now();
        let timeout_millis = self
            .rebase_timeout_millis
            .load(std::sync::atomic::Ordering::Relaxed);
        let base_opts = self.base_opts.clone();
        let log_source = self.log_source.clone();

        // All the slow work — open a fresh MeruDB, drain the log
        // source into a warm tail — is wrapped in one future so
        // #34's timeout applies to the combined chain. Any single
        // step (reopen, drain) that runs long enough to exceed
        // the budget produces a clean `Timeout` error instead of
        // an indefinite `.await`.
        //
        // Pre-#34, a drain-loop hang (InProcessLogSource chasing
        // a moving target under live writes) wedged this call
        // forever. The source-side snapshot bound (this commit)
        // is the root-cause fix; this timeout is the belt-and-
        // braces safety net for any future `LogSource` impl that
        // could reintroduce unbounded drain (network, Raft,
        // Kafka) or for warm-up I/O that genuinely takes too long
        // (degraded disk, large catalog refresh).
        let work = async move {
            // Step 1: open the new base.
            let new_base = std::sync::Arc::new(crate::MeruDB::open(base_opts).await?);
            let new_base_seq = new_base.read_seq().0;

            // Step 2: empty tail anchored at new_base_seq.
            let mut fresh_tail = ReplicaTail::new();
            if new_base_seq > 0 {
                fresh_tail.seed_visible_seq(new_base_seq);
            }

            // Step 3: warm up by draining the log source.
            // Phase 6: tolerate `ChangeFeedBelowRetention` — it's
            // the recovery path, and an empty tail at
            // new_base_seq is the correct end state. Every other
            // error propagates.
            match fresh_tail.advance(log_source.as_ref()).await {
                Ok(()) => {}
                Err(MeruError::ChangeFeedBelowRetention { .. }) => {}
                Err(other) => return Err(other),
            }
            let new_visible_seq = fresh_tail.visible_seq();

            let new_state = std::sync::Arc::new(ReplicaState {
                base: new_base,
                tail: tokio::sync::RwLock::new(fresh_tail),
            });
            Ok::<_, MeruError>((new_state, new_base_seq, new_visible_seq))
        };

        let (new_state, new_base_seq, new_visible_seq) = if timeout_millis > 0 {
            match tokio::time::timeout(std::time::Duration::from_millis(timeout_millis), work).await
            {
                Ok(inner) => inner?,
                Err(_) => {
                    return Err(MeruError::InvalidArgument(format!(
                        "rebase_hotswap timed out after {timeout_millis}ms — \
                         primary's write rate likely exceeds replica drain \
                         throughput; raise `Replica::set_rebase_timeout` or \
                         reduce primary load. The replica's old state is still \
                         serving reads (this call left it untouched)."
                    )));
                }
            }
        } else {
            work.await?
        };

        // Step 4: atomic swap. Moved outside the timed block so
        // that a caller who asks for "no timeout" still gets an
        // `ArcSwap::store` that cannot itself block indefinitely.
        self.state.store(new_state);

        // Phase 5 counters. Record BEFORE returning so a caller
        // that immediately calls `stats()` sees the fresh values.
        let warmup_millis = start.elapsed().as_millis() as u64;
        self.last_rebase_warmup_millis
            .store(warmup_millis, std::sync::atomic::Ordering::Relaxed);
        self.rebase_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok((new_base_seq, new_visible_seq))
    }

    /// Issue #32 Phase 6: advance-or-recover. Call `advance()`;
    /// on `ChangeFeedBelowRetention` (the retention gap signal),
    /// fall back to `rebase_hotswap` for a hard reset to the
    /// latest mirrored snapshot with an empty tail. The replica
    /// resumes serving reads at the new base_seq.
    ///
    /// Returns:
    /// - `Ok(AdvanceOutcome::Advanced)` — normal path, tail picked
    ///   up new ops (possibly zero).
    /// - `Ok(AdvanceOutcome::Recovered { new_base_seq })` — the
    ///   log source told us we're below retention, we reset.
    ///   Callers typically log this and reset any downstream
    ///   consumer state that depended on continuity of the tail
    ///   stream.
    /// - `Err(_)` — the error was NOT a retention gap (I/O,
    ///   corruption, etc.); caller's usual error-handling path.
    ///
    /// Note: operationally, repeated Recovered outcomes indicate
    /// the replica is chronically behind — size the mirror
    /// cadence or bump `gc_grace_period_secs` on the primary.
    pub async fn advance_or_recover(&self) -> Result<AdvanceOutcome> {
        match self.advance().await {
            Ok(()) => Ok(AdvanceOutcome::Advanced),
            Err(MeruError::ChangeFeedBelowRetention { .. }) => {
                let (new_base_seq, _) = self.rebase_hotswap().await?;
                Ok(AdvanceOutcome::Recovered { new_base_seq })
            }
            Err(other) => Err(other),
        }
    }

    /// Issue #32 Phase 6: explicit hard-reset for log-gap
    /// recovery. Alias for `rebase_hotswap` with an
    /// operator-facing name. Callers who detect a retention gap
    /// out-of-band (e.g., their monitoring told them so) use
    /// this to pre-emptively reset without waiting for the next
    /// `advance()` to discover the gap.
    pub async fn recover_from_log_gap(&self) -> Result<u64> {
        let (new_base_seq, _) = self.rebase_hotswap().await?;
        Ok(new_base_seq)
    }
}

/// Phase 6 return shape for `advance_or_recover`. Operators log
/// on `Recovered` variants; automated pipelines treat both as
/// success but may want to reset their downstream state on
/// `Recovered`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdvanceOutcome {
    /// Normal append-only advance: tail picked up zero or more ops.
    Advanced,
    /// Log source said we're below retention; we hard-reset via
    /// `rebase_hotswap`. `new_base_seq` is the replica's base
    /// after recovery.
    Recovered { new_base_seq: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn phase1_stream_returns_below_retention() {
        let src = ChangeFeedLogSource::new(1000);
        let err = src.stream(500).await.err().unwrap();
        match err {
            MeruError::ChangeFeedBelowRetention {
                requested,
                low_water,
            } => {
                assert_eq!(requested, 500);
                assert_eq!(low_water, 1000);
            }
            other => panic!("unexpected error shape: {other:?}"),
        }
    }

    #[tokio::test]
    async fn phase1_latest_seq_errors_with_pointer() {
        let src = ChangeFeedLogSource::new(0);
        let err = src.latest_seq().await.err().unwrap();
        assert!(format!("{err:?}").contains("Phase 2"));
    }

    #[test]
    fn log_gap_display_is_informative() {
        let gap = LogGap {
            requested: 42,
            earliest_available: Some(100),
            reason: "change-feed retention".into(),
        };
        let s = format!("{gap}");
        assert!(s.contains("42"));
        assert!(s.contains("100"));
        assert!(s.contains("retention"));
    }
}
