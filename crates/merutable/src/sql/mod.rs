//! Issue #29 (0.1-preview blocker): SQL surface for merutable.
//!
//! This crate is the home of the `merutable_changes(table, since_seq)`
//! table function — the transaction-granularity change feed served by
//! the RO replica via an embedded DataFusion `SessionContext`.
//!
//! # Phase 1 (this module)
//!
//! API surface only:
//! - [`ChangeOp`] — the three operation kinds a change-feed row can
//!   carry: Insert / Update / Delete.
//! - [`ChangeRecord`] — one row of the feed: (seq, op, row).
//! - [`ChangeFeedCursor`] — type-shape placeholder for the polling
//!   reader contract. Returns `MeruError::ChangeFeedBelowRetention`
//!   today; real iteration plumbing lands in Phase 2.
//!
//! # Phase 2a (shipped): memtable-only change scan
//!
//! The [`ChangeFeedCursor`] in-retention path pulls real records
//! from `MeruEngine::scan_memtable_changes`. Sufficient for
//! low-latency subscribers (RO replicas, audit log tailers) as
//! long as they keep up with the flush cadence.
//!
//! # Phase 2b (shipped): memtable + L0 scan
//!
//! The cursor calls `MeruEngine::scan_tail_changes` and sees ops
//! across memtable + L0. Subscribers can fall multiple snapshots
//! behind without escalating.
//!
//! # Phase 2e (this commit): DataFusion TableProvider
//!
//! `datafusion_provider::ChangeFeedTableProvider` wraps a running
//! `MeruEngine` + a `since_seq` watermark behind DataFusion's
//! `TableProvider` trait. Register it on a `SessionContext` and
//! the 0.1-preview headline query works:
//!
//! ```sql
//! SELECT * FROM merutable_changes WHERE op = 'DELETE'
//! ```
//!
//! Phase 2e scope is one-shot scan (drain cursor → `MemoryExec`).
//! Streaming exec + seq-range filter pushdown are follow-on.
//!
//! # Phase 2d (shipped): Arrow RecordBatch adapter
//!
//! New `crate::sql::arrow` module converts `Vec<ChangeRecord>` into the
//! Arrow columnar form DataFusion expects. Schema shape:
//! `seq UInt64, op Utf8, pk_bytes Binary, <user columns>`. Phase
//! 2e wires a `TableProvider` that produces these batches on
//! demand from a `ChangeFeedCursor`; landing the adapter
//! separately keeps the dep graph free of DataFusion until the
//! TableProvider actually ships.
//!
//! # Phase 2c (shipped): pre-image reconstruction + INSERT vs UPDATE
//!
//! The cursor now calls `scan_tail_changes_with_pre_image`, which
//! resolves each Delete op's pre-image via a point lookup at
//! `seq - 1`. Records for Delete ops now carry the row that was
//! live immediately before the delete — sufficient for a change
//! consumer to invalidate its own derived state or replay against
//! a sibling system.
//!
//! INSERT vs UPDATE is now distinguished by a pre-image lookup on
//! Puts: a Put with no prior live state at `seq - 1` is an Insert;
//! one with prior state is an Update. The `op` field of the
//! `ChangeRecord` now reflects that distinction. Workloads
//! dominated by pure Inserts pay one extra point-lookup per op;
//! callers that don't need the distinction can set
//! `ChangeFeedCursor::skip_update_discrimination(true)` to keep
//! every Put tagged Insert (the Phase 2a behavior) — cheaper by
//! 1 lookup per op.
//!
//! # Phase 2c (planned)
//!
//! - L1..LN scan (seq-range-filtered) + DELETE pre-image
//!   reconstruction via LSM point lookup at `seq - 1`.
//!
//! # Phase 2d (planned)
//!
//! - DataFusion `TableProvider` wrapper exposing the iterator as
//!   `merutable_changes(table, since_seq)` with tight statistics,
//!   seq-ordered output, and filter-pushdown (`Exact`).
//!
//! # Phase 3 (0.5-beta)
//!
//! - Arrow Flight SQL server binary wrapping the in-process
//!   SessionContext.
//! - Merged overlay view that UNIONs an Iceberg base scan with the
//!   change feed and resolves last-writer-wins per PK.
//! - Streaming subscription API (pushes new ops vs. polling).

use std::sync::Arc;

use crate::engine::engine::MeruEngine;
use crate::types::{
    MeruError, Result,
    sequence::{OpType, SeqNum},
    value::Row,
};

pub mod arrow;
pub mod datafusion_provider;

/// The kind of mutation a change-feed row represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChangeOp {
    /// New key — no prior row existed at `since_seq`.
    Insert,
    /// Existing key re-written.
    Update,
    /// Key deleted. The accompanying row is the pre-image
    /// reconstructed via LSM point lookup at `seq - 1`.
    Delete,
}

impl ChangeOp {
    /// SQL-compatible text label: `'INSERT' | 'UPDATE' | 'DELETE'`.
    pub fn as_sql_str(self) -> &'static str {
        match self {
            ChangeOp::Insert => "INSERT",
            ChangeOp::Update => "UPDATE",
            ChangeOp::Delete => "DELETE",
        }
    }
}

/// One row of the change feed.
#[derive(Clone, Debug)]
pub struct ChangeRecord {
    /// Sequence number the mutation was committed at.
    pub seq: u64,
    /// Kind of mutation.
    pub op: ChangeOp,
    /// Full row payload. For `Insert`/`Update` this is the post-state;
    /// for `Delete` this is the pre-image at `seq - 1` reconstructed
    /// via a point lookup on the LSM. Phase 2a/2b still return an
    /// empty `Row` for deletes — the pre-image reconstruction is
    /// Phase 2c. Consumers that need only the PK (e.g. replica
    /// tails applying tombstones) should use `pk_bytes`.
    pub row: Row,
    /// PK-encoded bytes of the affected key. Populated for every
    /// op (Insert, Update, Delete) — this is the canonical way to
    /// address the mutation across the memtable + SSTable scan
    /// boundary. Replicas key their tail index on these bytes;
    /// tombstones without a pre-image still carry the PK.
    pub pk_bytes: Vec<u8>,
}

/// Polling cursor over the change feed.
///
/// Phase 2a: `next_batch` scans the memtable for ops in
/// `(since_seq, read_seq]` and returns them. Phase 1's retention-
/// bound stub is preserved — constructing a cursor with
/// [`ChangeFeedCursor::new_below_retention`] still returns the
/// stable error shape on every call.
pub struct ChangeFeedCursor {
    inner: CursorInner,
    /// When true, every Put op is tagged `Insert` without the
    /// pre-image lookup that would distinguish Insert from Update.
    /// Defaults to false (full Phase 2c discrimination).
    skip_update_discrimination: bool,
}

enum CursorInner {
    Engine {
        engine: Arc<MeruEngine>,
        since_seq: u64,
        /// Issue #36 fix: pre-resolved records waiting to be
        /// served. `next_batch` drains up to `max_rows` from the
        /// head on each call; when the buffer empties, the next
        /// call refills it by scanning from the cursor's current
        /// `since_seq` to the engine's current `read_seq`.
        ///
        /// Pre-#36, every call re-scanned the ENTIRE
        /// `(since_seq, read_seq]` range and returned only the
        /// first `max_rows`. For a 3M-record drain at
        /// `batch_size=1024`, that's ~3k batches × ~3M scan =
        /// O(N²) work; the 30s budget blew out at ~1M records.
        /// Buffering collapses the drain to a single O(N) scan
        /// plus O(1) per `next_batch` pop.
        ///
        /// Records are stored in reverse order (newest at the
        /// front of `Vec::pop`) so `next_batch` can use O(1)
        /// `pop` from the tail. But we want seq-ascending output,
        /// so we store them in ascending order and drain from
        /// the front via `split_off`. `split_off` is O(k) where k
        /// is the drained count — amortized O(1) per record.
        buffer: Vec<ChangeRecord>,
    },
    BelowRetention {
        requested: u64,
        low_water: u64,
    },
}

impl ChangeFeedCursor {
    /// Open a cursor that pulls from the running engine's memtable.
    /// Phase 2a scope: memtable only. Rows with seq in
    /// `(since_seq, engine.read_seq()]` are returned.
    pub fn from_engine(engine: Arc<MeruEngine>, since_seq: u64) -> Self {
        Self {
            inner: CursorInner::Engine {
                engine,
                since_seq,
                buffer: Vec::new(),
            },
            skip_update_discrimination: false,
        }
    }

    /// Legacy Phase 1 shape — returns `ChangeFeedBelowRetention` on
    /// every `next_batch` so callers wiring escalation paths can
    /// keep exercising them.
    pub fn new_below_retention(requested: u64, low_water: u64) -> Self {
        Self {
            inner: CursorInner::BelowRetention {
                requested,
                low_water,
            },
            skip_update_discrimination: false,
        }
    }

    /// Opt out of Insert/Update discrimination. Every Put op is
    /// tagged Insert. Saves one point-lookup per op — useful for
    /// subscribers that only care about the key + op kind at a
    /// coarse level (replicas applying LWW, audit tailers that
    /// don't branch on INSERT vs UPDATE).
    pub fn skip_update_discrimination(mut self, skip: bool) -> Self {
        self.skip_update_discrimination = skip;
        self
    }

    /// Pull up to `max_rows` records from the feed.
    ///
    /// Phase 2a scope:
    /// - Engine-backed cursor walks `scan_memtable_changes`, takes
    ///   the first `max_rows` ops by seq, and advances `since_seq`
    ///   past the highest returned seq so the next call continues
    ///   from there.
    /// - Below-retention cursor returns the stable error on every
    ///   call until the caller resets.
    pub fn next_batch(&mut self, max_rows: usize) -> Result<Vec<ChangeRecord>> {
        match &mut self.inner {
            CursorInner::BelowRetention {
                requested,
                low_water,
            } => Err(MeruError::ChangeFeedBelowRetention {
                requested: *requested,
                low_water: *low_water,
            }),
            CursorInner::Engine {
                engine,
                since_seq,
                buffer,
            } => {
                // Issue #36 fix: refill the buffer in ONE scan when
                // it empties, then serve O(1) batches from the
                // buffer. Pre-#36, every `next_batch` call re-scanned
                // the entire `(since_seq, read_seq]` range and took
                // only the first `max_rows` — quadratic total work
                // for a bootstrap drain from seq=0. A 3M-record drain
                // at batch_size=1024 blew past the 30s budget at
                // ~1M records. Buffering makes the total work
                // O(N) in the tail size (one scan) plus O(1) per
                // returned record.
                if buffer.is_empty() {
                    let read_seq = engine.read_seq();
                    if SeqNum(*since_seq) >= read_seq {
                        return Ok(Vec::new());
                    }
                    // Phase 2c: pre-image reconstruction. Walks the
                    // tail AND resolves each Delete's prior live state.
                    let raw = engine.scan_tail_changes_with_pre_image(*since_seq, read_seq)?;
                    buffer.reserve(raw.len());
                    for tuple in raw {
                        let op = match tuple.op_type {
                            OpType::Put => {
                                if self.skip_update_discrimination {
                                    ChangeOp::Insert
                                } else {
                                    // Distinguish Insert from Update
                                    // by probing whether a live row
                                    // existed at `seq - 1`.
                                    let had_prior = if tuple.seq == 0 {
                                        false
                                    } else {
                                        engine
                                            .point_lookup_by_user_key_at_seq(
                                                &tuple.pk_bytes,
                                                SeqNum(tuple.seq - 1),
                                            )?
                                            .is_some()
                                    };
                                    if had_prior {
                                        ChangeOp::Update
                                    } else {
                                        ChangeOp::Insert
                                    }
                                }
                            }
                            OpType::Delete => ChangeOp::Delete,
                        };
                        buffer.push(ChangeRecord {
                            seq: tuple.seq,
                            op,
                            row: tuple.row,
                            pk_bytes: tuple.pk_bytes,
                        });
                    }
                }

                // Serve up to `max_rows` from the head of the buffer.
                let drain_n = buffer.len().min(max_rows);
                if drain_n == 0 {
                    return Ok(Vec::new());
                }
                // `split_off(drain_n)` keeps the tail in the buffer
                // and returns the head-drained chunk. We want the
                // HEAD out; so swap.
                let remainder = buffer.split_off(drain_n);
                let out = std::mem::replace(buffer, remainder);
                if let Some(last) = out.last() {
                    *since_seq = last.seq;
                }
                Ok(out)
            }
        }
    }

    /// Current `since_seq` — advances past each batch. Readers
    /// persisting a resume point read this after `next_batch`.
    pub fn since_seq(&self) -> u64 {
        match &self.inner {
            CursorInner::Engine { since_seq, .. } => *since_seq,
            CursorInner::BelowRetention { requested, .. } => *requested,
        }
    }

    /// Records prefetched but not yet drained to the caller. Zero
    /// on a fresh cursor and between the drain and the next scan;
    /// non-zero after a scan fills the buffer and before subsequent
    /// `next_batch` calls empty it. Useful for tests and operators
    /// reasoning about cursor memory footprint.
    pub fn buffered_len(&self) -> usize {
        match &self.inner {
            CursorInner::Engine { buffer, .. } => buffer.len(),
            CursorInner::BelowRetention { .. } => 0,
        }
    }
}
