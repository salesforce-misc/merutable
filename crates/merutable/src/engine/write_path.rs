//! Write path helpers.
//!
//! The primary write path is `MeruEngine::put()` and `MeruEngine::delete()`
//! (implemented in `engine.rs`). This module provides batch-write support
//! and helper utilities.

use std::sync::Arc;

use crate::types::{
    key::InternalKey,
    sequence::{OpType, SeqNum},
    value::{FieldValue, Row},
    Result,
};
use crate::wal::batch::WriteBatch;
use bytes::Bytes;
use tracing::{debug, instrument};

use crate::engine::codec;

use crate::engine::engine::MeruEngine;

/// A batch of mutations to apply atomically to the engine.
/// Assigns sequential `SeqNum`s starting from the allocated base.
pub struct MutationBatch {
    pub(crate) ops: Vec<Mutation>,
}

pub struct Mutation {
    pub pk_values: Vec<FieldValue>,
    pub row: Option<Row>,
    pub op_type: OpType,
}

impl MutationBatch {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn put(&mut self, pk_values: Vec<FieldValue>, row: Row) {
        self.ops.push(Mutation {
            pk_values,
            row: Some(row),
            op_type: OpType::Put,
        });
    }

    pub fn delete(&mut self, pk_values: Vec<FieldValue>) {
        self.ops.push(Mutation {
            pk_values,
            row: None,
            op_type: OpType::Delete,
        });
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl Default for MutationBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a `MutationBatch` to the engine atomically.
/// All mutations share the same WAL record and are applied to the memtable
/// in a single `WriteBatch`.
#[instrument(skip(engine, batch), fields(op = "apply_batch", batch_size = batch.ops.len()))]
pub async fn apply_batch(engine: &Arc<MeruEngine>, batch: MutationBatch) -> Result<SeqNum> {
    if batch.is_empty() {
        return Ok(engine.visible_seq.current());
    }

    if engine.read_only {
        return Err(crate::types::MeruError::ReadOnly);
    }

    if engine.is_closed() {
        return Err(crate::types::MeruError::Closed);
    }

    // Issue #12: validate every Put row in the batch BEFORE any I/O.
    // One invalid row aborts the whole batch with a column-pointed
    // error — atomic-batch semantics preserved.
    for (i, mutation) in batch.ops.iter().enumerate() {
        if let (OpType::Put, Some(row)) = (mutation.op_type, mutation.row.as_ref()) {
            row.validate(&engine.schema).map_err(|e| {
                crate::types::MeruError::SchemaMismatch(format!("batch entry {i}: {e}",))
            })?;
        }
    }

    // Issue #14 Phase 2: batch-entry counters (Put/Delete distinguished
    // from single-op counters so operators can see the batch-vs-single
    // mix). Bumped AFTER validation so schema errors don't inflate.
    let batch_len = batch.ops.len() as u64;
    crate::engine::metrics::inc(crate::engine::metrics::PUT_BATCHES_TOTAL);
    crate::engine::metrics::inc_by(crate::engine::metrics::PUT_BATCH_ROWS_TOTAL, batch_len);

    // Flow control #1: L0 file-count stop (Issue #5). Same contract as
    // `write_internal` — batch writes go through the same triggers so
    // a bulk load can't evade the stall by using the batch API.
    loop {
        let notify = engine.l0_drained.notified();
        let l0 = engine.version_set.current().l0_file_count();
        if l0 < engine.config.l0_stop_trigger {
            break;
        }
        notify.await;
    }

    // Flow control #2: immutable-memtable queue stop.
    loop {
        let notify = engine.memtable.flush_complete.notified();
        if !engine.memtable.should_stall() {
            break;
        }
        notify.await;
    }

    // Encode all mutations (CPU work) outside the WAL lock.
    let has_cache = engine.row_cache.is_some();
    let mut user_keys: Vec<Vec<u8>> = if has_cache {
        Vec::with_capacity(batch.ops.len())
    } else {
        Vec::new()
    };

    struct EncodedOp {
        user_key_bytes: Vec<u8>,
        value_bytes: Bytes,
        op_type: OpType,
    }
    let mut encoded_ops: Vec<EncodedOp> = Vec::with_capacity(batch.ops.len());

    for mutation in &batch.ops {
        let user_key_bytes = InternalKey::encode_user_key(&mutation.pk_values, &engine.schema)?;
        if has_cache {
            user_keys.push(user_key_bytes.clone());
        }
        let value_bytes = match mutation.op_type {
            OpType::Put => match mutation.row.as_ref() {
                Some(r) => Bytes::from(codec::encode_row(r)?),
                None => Bytes::new(),
            },
            // Issue #33 fix: capture the delete pre-image at write
            // time instead of reconstructing post-hoc via
            // point_lookup_at(seq-1). The reconstruction approach
            // (Phase 2c) silently returned empty rows whenever
            // compaction had already merged the superseded Put into
            // L1+ and dropped it behind the tombstone — exactly
            // what happens under a hot stress workload. Capturing
            // the pre-image inline on the tombstone means the
            // change feed reads the pre-image directly from the
            // memtable/L0/L1+ without a lookup. Compaction
            // preserves the tombstone's value bytes, so the
            // pre-image survives to wherever the feed drains.
            //
            // Cost: one point_lookup per delete op at write time.
            // Amortized into the WAL-fsync critical section by
            // doing the lookup BEFORE acquiring the WAL lock.
            // Stale-window risk: between the lookup and the WAL
            // append, another writer could update the key; the
            // captured pre-image would then be "one op behind."
            // For a change feed this is acceptable — the
            // semantics are "last-known state at delete seq,"
            // not "causally-immediate pre-image."
            OpType::Delete => {
                let pre_image =
                    crate::engine::read_path::point_lookup(engine, &mutation.pk_values)?;
                match pre_image {
                    Some(row) => Bytes::from(codec::encode_row(&row)?),
                    None => Bytes::new(),
                }
            }
        };
        encoded_ops.push(EncodedOp {
            user_key_bytes,
            value_bytes,
            op_type: mutation.op_type,
        });
    }

    let batch_len = encoded_ops.len() as u64;

    // IMP-02: allocate, WAL append, and memtable apply all inside the WAL
    // lock. This ensures visible_seq is only advanced after data is in the
    // memtable — no torn-read window for concurrent readers.
    let (base_seq, should_flush) = {
        let mut wal = engine.wal.lock().await;

        let base_seq = engine.global_seq.allocate_n(batch_len);
        debug!(
            base_seq = base_seq.0,
            count = batch_len,
            "batch seq allocated"
        );

        let mut wal_batch = WriteBatch::new(base_seq);
        for op in &encoded_ops {
            match op.op_type {
                OpType::Put => {
                    wal_batch.put(
                        Bytes::from(op.user_key_bytes.clone()),
                        op.value_bytes.clone(),
                    );
                }
                OpType::Delete => {
                    // Issue #33 fix: delete carries its pre-image
                    // bytes (possibly empty). The value was
                    // populated by the earlier encoding loop via
                    // `point_lookup(&mutation.pk_values)`.
                    wal_batch.delete_with_pre_image(
                        Bytes::from(op.user_key_bytes.clone()),
                        op.value_bytes.clone(),
                    );
                }
            }
        }

        wal.append(&wal_batch)?;
        let should_flush = engine.memtable.apply_batch(&wal_batch)?;

        // Advance visible_seq now that the data is in the memtable.
        // Issue #8: visible_seq is the highest visible seq (inclusive).
        // A batch of N ops uses seqs [base, base+1, ..., base+N-1];
        // the highest (= new frontier) is base + N - 1.
        engine.visible_seq.set_at_least(base_seq.0 + batch_len - 1);

        (base_seq, should_flush)
    };

    // Invalidate row cache for every key in the batch.
    if let Some(ref cache) = engine.row_cache {
        for uk in &user_keys {
            cache.invalidate(uk);
        }
    }

    // Auto-flush when the threshold is crossed. Must rotate the active
    // memtable to the immutable queue BEFORE spawning `run_flush`, or the
    // spawned task sees an empty immutable queue and silently no-ops (Bug
    // F regression). `rotation_lock` serializes bursts so only one writer
    // actually rotates; later writers re-check under the lock and find the
    // fresh (small) active memtable.
    if should_flush {
        if let Ok(_guard) = engine.rotation_lock.try_lock() {
            if engine.memtable.active_should_flush() {
                let next_seq = engine.global_seq.current().next();
                engine.memtable.rotate(next_seq);
                {
                    let mut wal = engine.wal.lock().await;
                    wal.rotate()?;
                }
                let engine = Arc::clone(engine);
                tokio::spawn(async move {
                    if let Err(e) = crate::engine::flush::run_flush(&engine).await {
                        tracing::error!(error = %e, "auto-flush failed");
                    }
                });
            }
        }
    }

    Ok(base_seq)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        schema::{ColumnDef, ColumnType, TableSchema},
        value::Row,
    };

    #[test]
    fn mutation_batch_builder() {
        let mut batch = MutationBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        batch.put(
            vec![FieldValue::Int64(1)],
            Row::new(vec![Some(FieldValue::Int64(1))]),
        );
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());

        batch.delete(vec![FieldValue::Int64(2)]);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn mutation_batch_default() {
        let batch = MutationBatch::default();
        assert!(batch.is_empty());
    }

    fn test_schema() -> TableSchema {
        TableSchema {
            table_name: "test".into(),
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

    #[tokio::test]
    async fn apply_empty_batch_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::engine::config::EngineConfig {
            schema: test_schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        };
        let engine = crate::engine::engine::MeruEngine::open(config)
            .await
            .unwrap();
        let pre = engine.read_seq();
        let batch = MutationBatch::new();
        let seq = apply_batch(&engine, batch).await.unwrap();
        // Should return the current visible seq WITHOUT advancing (Issue #8:
        // visible_seq is the inclusive-latest-visible frontier; for a fresh
        // engine that's 0).
        assert_eq!(seq, pre);
    }

    #[tokio::test]
    async fn apply_batch_writes_and_reads() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::engine::config::EngineConfig {
            schema: test_schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        };
        let engine = crate::engine::engine::MeruEngine::open(config)
            .await
            .unwrap();

        let mut batch = MutationBatch::new();
        batch.put(
            vec![FieldValue::Int64(42)],
            Row::new(vec![
                Some(FieldValue::Int64(42)),
                Some(FieldValue::Bytes(bytes::Bytes::from("hello"))),
            ]),
        );
        batch.put(
            vec![FieldValue::Int64(99)],
            Row::new(vec![
                Some(FieldValue::Int64(99)),
                Some(FieldValue::Bytes(bytes::Bytes::from("world"))),
            ]),
        );

        apply_batch(&engine, batch).await.unwrap();

        // Issue #46 regression: verify actual field VALUES, not just
        // existence. The old code used unwrap_or_default() on
        // encode_row, which would silently produce empty bytes on
        // failure. decode_row("") returns Row::default() — an empty
        // row that is_some(). Checking only is_some() would not catch
        // the data loss.
        let row1 = engine
            .get(&[FieldValue::Int64(42)])
            .unwrap()
            .expect("key 42 must exist");
        assert_eq!(
            row1.get(1).cloned(),
            Some(FieldValue::Bytes(bytes::Bytes::from("hello"))),
            "key 42 value must round-trip through batch encode path"
        );
        let row2 = engine
            .get(&[FieldValue::Int64(99)])
            .unwrap()
            .expect("key 99 must exist");
        assert_eq!(
            row2.get(1).cloned(),
            Some(FieldValue::Bytes(bytes::Bytes::from("world"))),
            "key 99 value must round-trip through batch encode path"
        );
    }

    /// Regression: a multi-record `MutationBatch` must advance `global_seq`
    /// by exactly `batch.len()` — not by 1. The memtable's own `apply_batch`
    /// stamps each record at `base, base+1, …, base+N-1`, so if the write
    /// path only calls `allocate()` once, the next writer collides with the
    /// tail of the batch in the skiplist. crossbeam_skiplist's `insert`
    /// silently overwrites on duplicate-key collision, dropping a record
    /// without any error — this is a data-loss bug.
    #[tokio::test]
    async fn apply_batch_advances_seq_by_record_count() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::engine::config::EngineConfig {
            schema: test_schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        };
        let engine = crate::engine::engine::MeruEngine::open(config)
            .await
            .unwrap();

        let seq_before = engine.read_seq().0;

        // 4-record batch. All distinct keys.
        let mut batch = MutationBatch::new();
        for i in 1..=4i64 {
            batch.put(
                vec![FieldValue::Int64(i)],
                Row::new(vec![
                    Some(FieldValue::Int64(i)),
                    Some(FieldValue::Bytes(bytes::Bytes::from(format!("batch_{i}")))),
                ]),
            );
        }
        let base_seq = apply_batch(&engine, batch).await.unwrap();
        // Issue #8: visible_seq is the highest visible seq (inclusive).
        // The first NEW seq = visible_seq + 1.
        assert_eq!(
            base_seq.0,
            seq_before + 1,
            "batch's first seq must be one greater than the pre-batch visible frontier"
        );

        // After a 4-record batch, global_seq MUST have advanced by 4.
        let seq_after = engine.read_seq().0;
        assert_eq!(
            seq_after,
            seq_before + 4,
            "4-record batch must advance global_seq by 4, not 1; before={seq_before} after={seq_after}"
        );

        // And the next single put MUST get a seq strictly greater than the
        // tail of the batch (base_seq + 3).
        let s5 = engine
            .put(
                vec![FieldValue::Int64(5)],
                Row::new(vec![
                    Some(FieldValue::Int64(5)),
                    Some(FieldValue::Bytes(bytes::Bytes::from("single_5"))),
                ]),
            )
            .await
            .unwrap();
        assert!(
            s5.0 > base_seq.0 + 3,
            "single put after 4-record batch must have seq > base+3: base={} s5={}",
            base_seq.0,
            s5.0
        );
    }

    /// Regression: after a multi-record batch, a subsequent put of a NEW
    /// key must not silently overwrite any of the batch's records. This is
    /// the user-observable face of Bug A — with the old `allocate()` code,
    /// the single put collided in the skiplist at (different user_key, same
    /// seq)... and while that specific collision doesn't shadow data, the
    /// invariant "every batch record is readable at the correct value" must
    /// hold regardless of the allocation strategy.
    #[tokio::test]
    async fn apply_batch_then_single_put_all_values_survive() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::engine::config::EngineConfig {
            schema: test_schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        };
        let engine = crate::engine::engine::MeruEngine::open(config)
            .await
            .unwrap();

        let mut batch = MutationBatch::new();
        for i in 1..=3i64 {
            batch.put(
                vec![FieldValue::Int64(i)],
                Row::new(vec![
                    Some(FieldValue::Int64(i)),
                    Some(FieldValue::Bytes(bytes::Bytes::from(format!("batch_v{i}")))),
                ]),
            );
        }
        apply_batch(&engine, batch).await.unwrap();

        engine
            .put(
                vec![FieldValue::Int64(4)],
                Row::new(vec![
                    Some(FieldValue::Int64(4)),
                    Some(FieldValue::Bytes(bytes::Bytes::from("single_v4"))),
                ]),
            )
            .await
            .unwrap();

        for (id, expected) in [
            (1i64, "batch_v1"),
            (2, "batch_v2"),
            (3, "batch_v3"),
            (4, "single_v4"),
        ] {
            let row = engine
                .get(&[FieldValue::Int64(id)])
                .unwrap()
                .unwrap_or_else(|| panic!("id {id} missing after batch+single put"));
            let got = row.get(1).cloned();
            assert_eq!(
                got,
                Some(FieldValue::Bytes(bytes::Bytes::from(expected))),
                "value mismatch for id {id}"
            );
        }
    }

    /// Regression: two sequential multi-record batches must leave the
    /// skiplist with exactly `batch1.len() + batch2.len()` logical entries
    /// (no silent overwrite where the second batch's base seq collides with
    /// the first batch's tail). This is the scenario that triggered the
    /// original bug in production workloads that mix batches and singles.
    #[tokio::test]
    async fn two_consecutive_batches_preserve_all_records() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::engine::config::EngineConfig {
            schema: test_schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        };
        let engine = crate::engine::engine::MeruEngine::open(config)
            .await
            .unwrap();

        // Batch 1: keys 1..=5, values "b1_N".
        let mut b1 = MutationBatch::new();
        for i in 1..=5i64 {
            b1.put(
                vec![FieldValue::Int64(i)],
                Row::new(vec![
                    Some(FieldValue::Int64(i)),
                    Some(FieldValue::Bytes(bytes::Bytes::from(format!("b1_{i}")))),
                ]),
            );
        }
        apply_batch(&engine, b1).await.unwrap();

        // Batch 2: keys 6..=10, values "b2_N".
        let mut b2 = MutationBatch::new();
        for i in 6..=10i64 {
            b2.put(
                vec![FieldValue::Int64(i)],
                Row::new(vec![
                    Some(FieldValue::Int64(i)),
                    Some(FieldValue::Bytes(bytes::Bytes::from(format!("b2_{i}")))),
                ]),
            );
        }
        apply_batch(&engine, b2).await.unwrap();

        // All 10 keys must be readable with their correct values.
        for i in 1..=5i64 {
            let row = engine.get(&[FieldValue::Int64(i)]).unwrap().unwrap();
            assert_eq!(
                row.get(1).cloned(),
                Some(FieldValue::Bytes(bytes::Bytes::from(format!("b1_{i}")))),
                "batch 1 value missing/corrupt at id {i}"
            );
        }
        for i in 6..=10i64 {
            let row = engine.get(&[FieldValue::Int64(i)]).unwrap().unwrap();
            assert_eq!(
                row.get(1).cloned(),
                Some(FieldValue::Bytes(bytes::Bytes::from(format!("b2_{i}")))),
                "batch 2 value missing/corrupt at id {i}"
            );
        }

        // Full scan must return exactly 10 distinct rows.
        let scan = engine.scan(None, None).unwrap();
        assert_eq!(scan.len(), 10, "scan should see all 10 distinct keys");
    }

    #[tokio::test]
    async fn apply_batch_with_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::engine::config::EngineConfig {
            schema: test_schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        };
        let engine = crate::engine::engine::MeruEngine::open(config)
            .await
            .unwrap();

        // Put then delete via batch.
        engine
            .put(
                vec![FieldValue::Int64(1)],
                Row::new(vec![Some(FieldValue::Int64(1)), None]),
            )
            .await
            .unwrap();

        let mut batch = MutationBatch::new();
        batch.delete(vec![FieldValue::Int64(1)]);
        apply_batch(&engine, batch).await.unwrap();

        // Should not be visible (deleted).
        let row = engine.get(&[FieldValue::Int64(1)]).unwrap();
        assert!(row.is_none());
    }
}
