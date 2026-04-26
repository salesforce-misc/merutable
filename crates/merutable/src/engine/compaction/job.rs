//! `CompactionJob`: orchestrates one compaction run with DV bookkeeping.
//!
//! Sequence:
//! 1. Pick compaction (level, input files).
//! 2. Read input files via `ParquetReader::read_physical_rows_with_positions`,
//!    applying each source file's existing Deletion Vector loaded from its
//!    Puffin blob at the manifest-recorded byte range.
//! 3. Merge via `CompactionIterator` (dedup, tombstone drop).
//! 4. Write output files at the target level.
//! 5. Track promoted row positions for DV updates.
//! 6. Build `SnapshotTransaction` (adds + removes + dvs).
//! 7. Commit Iceberg snapshot.
//! 8. Install new version.

use std::{path::Path, sync::Arc};

use crate::iceberg::{DeletionVector, IcebergDataFile, SnapshotTransaction, version::DataFileMeta};
use crate::parquet::reader::ParquetReader;
use crate::types::{
    MeruError, Result,
    level::Level,
    level::ParquetFileMeta,
    schema::TableSchema,
    value::{FieldValue, Row},
};
use bytes::Bytes;
use roaring::RoaringBitmap;
use tracing::{debug, info, instrument, warn};

use crate::engine::{
    compaction::{
        iterator::{CompactionIterator, FileEntries},
        picker,
    },
    engine::MeruEngine,
};

/// Open a source Parquet file and load its Deletion Vector (if any) from
/// the companion Puffin blob at the exact manifest-recorded byte range.
/// Kept local to the compaction module so the read path can keep its own
/// equivalent helper without risking circular visibility; both are thin
/// wrappers over `ParquetReader::open` + `DeletionVector::from_puffin_blob`.
fn open_source_file(
    base: &Path,
    file: &DataFileMeta,
    schema: Arc<TableSchema>,
) -> Result<(ParquetReader<Bytes>, Option<RoaringBitmap>)> {
    let abs_parquet = base.join(&file.path);
    let parquet_bytes = std::fs::read(&abs_parquet).map_err(MeruError::Io)?;
    let reader = ParquetReader::open(Bytes::from(parquet_bytes), schema)?;

    let dv = match (&file.dv_path, file.dv_offset, file.dv_length) {
        (Some(dv_path), Some(offset), Some(length)) => {
            let abs_dv = base.join(dv_path);
            let puffin_bytes = std::fs::read(&abs_dv).map_err(MeruError::Io)?;
            if offset < 0 || length < 0 {
                return Err(MeruError::Corruption(format!(
                    "DV has negative offset ({offset}) or length ({length}) on file {}",
                    file.path,
                )));
            }
            let start = offset as usize;
            let end = start
                .checked_add(length as usize)
                .ok_or_else(|| MeruError::Corruption("DV offset+length overflow".into()))?;
            if end > puffin_bytes.len() {
                return Err(MeruError::Corruption(format!(
                    "DV blob out of range: path={dv_path} offset={offset} length={length} puffin_len={}",
                    puffin_bytes.len()
                )));
            }
            let dv = DeletionVector::from_puffin_blob(&puffin_bytes[start..end])?;
            Some(dv.bitmap().clone())
        }
        (None, None, None) => None,
        _ => {
            return Err(MeruError::Corruption(format!(
                "inconsistent DV coords on file {}: dv_path={:?} dv_offset={:?} dv_length={:?}",
                file.path, file.dv_path, file.dv_offset, file.dv_length
            )));
        }
    };

    Ok((reader, dv))
}

/// Estimate the in-memory byte footprint of a row — sum of field byte
/// sizes across the schema's columns. Used to bound compaction output
/// file size (Issue #3: Arrow's `BinaryArray` uses i32 offsets, so a
/// single `ByteArray` column aggregated across all rows cannot exceed
/// `i32::MAX` ≈ 2.14 GiB, and exceeding it panics the process). The
/// estimate is conservative: the actual Parquet encoding is typically
/// smaller due to dictionary + RLE + compression, so capping on this
/// estimate leaves a comfortable safety margin versus the hard Arrow
/// limit.
fn estimate_row_bytes(row: &Row) -> u64 {
    let mut total: u64 = 0;
    for fv in row.fields.iter().flatten() {
        total += match fv {
            FieldValue::Boolean(_) => 1,
            FieldValue::Int32(_) | FieldValue::Float(_) => 4,
            FieldValue::Int64(_) | FieldValue::Double(_) => 8,
            // +8 accounts for the i32 offset + dictionary overhead
            // that inflates the accumulated in-memory representation
            // before compression.
            FieldValue::Bytes(b) => b.len() as u64 + 8,
        };
    }
    total
}

/// Target uncompressed bytes per compaction output file. Arrow's
/// `BinaryArray` caps a single column at i32::MAX (~2.14 GiB); keep
/// output files well under that so even a pathological skewed payload
/// distribution cannot overflow a single column. 512 MiB gives a 4×
/// safety margin versus the hard limit.
const TARGET_OUTPUT_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Per-chunk accumulator while splitting compaction output into
/// multiple files. A new chunk starts when the current one exceeds
/// `TARGET_OUTPUT_FILE_BYTES` AND the next entry has a different
/// `user_key` — splitting inside a user_key would produce two L1+
/// files both containing the same user_key, breaking the non-overlap
/// invariant that point-lookup binary search relies on.
struct OutputChunk {
    rows: Vec<(crate::types::key::InternalKey, Row)>,
    seq_min: u64,
    seq_max: u64,
    key_min: Vec<u8>,
    key_max: Vec<u8>,
    approx_bytes: u64,
    last_user_key: Vec<u8>,
}

impl OutputChunk {
    fn empty() -> Self {
        Self {
            rows: Vec::new(),
            seq_min: u64::MAX,
            seq_max: 0,
            key_min: Vec::new(),
            key_max: Vec::new(),
            approx_bytes: 0,
            last_user_key: Vec::new(),
        }
    }

    fn push(&mut self, ikey: crate::types::key::InternalKey, row: Row, est_bytes: u64) {
        let uk = ikey.user_key_bytes().to_vec();
        let s = ikey.seq.0;
        if s < self.seq_min {
            self.seq_min = s;
        }
        if s > self.seq_max {
            self.seq_max = s;
        }
        if self.key_min.is_empty() || uk.as_slice() < self.key_min.as_slice() {
            self.key_min = uk.clone();
        }
        if uk.as_slice() > self.key_max.as_slice() {
            self.key_max.clone_from(&uk);
        }
        self.approx_bytes = self.approx_bytes.saturating_add(est_bytes);
        self.last_user_key = uk;
        self.rows.push((ikey, row));
    }
}

/// Compute the union `[key_min, key_max]` range across a set of files.
/// Returns `(None, None)` if the set is empty. Files whose own `key_min`/
/// `key_max` are empty (unbounded) are treated as such — `key_min == []`
/// contributes the lexicographically smallest possible value (shrinks
/// `union_min` to `[]`), and `key_max == []` is treated as
/// "no upper bound known" and expands `union_max` to `[0xFF; ...]` by
/// clearing it. In practice every real file carries concrete bounds.
fn compute_union_range(files: &[DataFileMeta]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut union_min: Option<Vec<u8>> = None;
    let mut union_max: Option<Vec<u8>> = None;
    for f in files {
        let km = &f.meta.key_min;
        let kx = &f.meta.key_max;
        match &union_min {
            None => union_min = Some(km.clone()),
            Some(cur) if km.as_slice() < cur.as_slice() => union_min = Some(km.clone()),
            _ => {}
        }
        match &union_max {
            None => union_max = Some(kx.clone()),
            Some(cur) if kx.as_slice() > cur.as_slice() => union_max = Some(kx.clone()),
            _ => {}
        }
    }
    (union_min, union_max)
}

/// RAII guard that releases a level reservation when dropped. Ensures
/// reserved levels are always freed even on error paths — without this,
/// a mid-compaction error would permanently wedge the levels and no
/// future compaction could touch them.
struct LevelReservation {
    engine: Arc<MeruEngine>,
    levels: Vec<Level>,
}

impl Drop for LevelReservation {
    fn drop(&mut self) {
        // We're in a sync drop; `Mutex::blocking_lock` is wrong inside a
        // tokio runtime (deadlocks if the runtime has only one worker).
        // Use `try_lock` — the contention window is microseconds
        // (pick/execute/commit all release naturally), so try_lock
        // virtually always succeeds. If it fails, spawn an async task
        // to free the reservation so levels can't leak permanently.
        let levels = std::mem::take(&mut self.levels);
        if levels.is_empty() {
            return;
        }
        let acquired = {
            let try_result = self.engine.compacting_levels.try_lock();
            match try_result {
                Ok(mut guard) => {
                    for l in &levels {
                        guard.remove(l);
                    }
                    // #30 observability counter mirror: update
                    // under the same lock acquisition so stats()
                    // sees a consistent view.
                    self.engine
                        .compacting_levels_len
                        .store(guard.len(), std::sync::atomic::Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            }
        };
        if !acquired {
            let engine = self.engine.clone();
            tokio::spawn(async move {
                let mut guard = engine.compacting_levels.lock().await;
                for l in &levels {
                    guard.remove(l);
                }
                engine
                    .compacting_levels_len
                    .store(guard.len(), std::sync::atomic::Ordering::Relaxed);
            });
        }
    }
}

/// Run compaction until the LSM tree is healthy or all eligible work is
/// owned by other workers.
///
/// Loops: acquire the state lock → pick a compaction on levels not
/// currently reserved by another worker → reserve those levels
/// (input + output) → release the state lock → execute the merge (no
/// lock held, so concurrent workers can run on disjoint levels) →
/// acquire `commit_lock` for the brief catalog commit → release all
/// reservations via RAII.
///
/// Returns when `pick_compaction()` returns `None`, either because no
/// level needs compaction or because every eligible level is currently
/// being compacted by another worker. The other worker's loop will
/// handle any remaining work as its levels free up.
///
/// Follows Pebble's `inProgressCompactions` / BadgerDB's `compactStatus`
/// pattern. Scaled to per-level granularity because the current picker
/// always picks full levels; refinable to per-file tracking if the
/// picker learns to select subranges.
#[instrument(skip(engine), fields(op = "compaction"))]
pub async fn run_compaction(engine: &Arc<MeruEngine>) -> Result<()> {
    const MAX_ITERATIONS: usize = 128;
    for iter in 0..MAX_ITERATIONS {
        let did_work = run_one_compaction_job(engine).await?;
        if !did_work {
            if iter > 0 {
                debug!(iterations = iter, "compaction drained all pressure");
            }
            return Ok(());
        }
    }
    warn!(
        max = MAX_ITERATIONS,
        "compaction loop hit iteration cap — will resume on next trigger"
    );
    Ok(())
}

/// Reserve the input + output levels for a new compaction. Returns the
/// pick, the version snapshot the pick was made from, and an RAII guard.
/// Returning the version preserves Bug Q's invariant: the compaction
/// reads files from the same version the picker scored — a later
/// `version_set.current()` call could see a different version after a
/// concurrent flush committed.
async fn reserve_next_compaction(
    engine: &Arc<MeruEngine>,
) -> Option<(
    picker::CompactionPick,
    Arc<crate::iceberg::version::Version>,
    LevelReservation,
)> {
    let mut busy = engine.compacting_levels.lock().await;
    let version_guard = engine.version_set.current();
    let pick = picker::pick_compaction(&version_guard, &engine.config, &busy)?;
    // Clone the Arc out of the ArcSwap guard so the version outlives the
    // reservation (and the caller can read files from it without holding
    // the guard across await points).
    let version: Arc<crate::iceberg::version::Version> = (*version_guard).clone();
    drop(version_guard);

    // Reserve the picked levels. Inserts must succeed because the
    // picker guaranteed neither is already in `busy`.
    let input_level = pick.input_level;
    let output_level = pick.output_level;
    busy.insert(input_level);
    busy.insert(output_level);
    // #30 observability counter mirror: update before releasing
    // the lock so stats() can never observe an "inflight" count
    // less than what the HashSet actually holds.
    engine
        .compacting_levels_len
        .store(busy.len(), std::sync::atomic::Ordering::Relaxed);
    drop(busy);

    let reservation = LevelReservation {
        engine: engine.clone(),
        levels: vec![input_level, output_level],
    };
    Some((pick, version, reservation))
}

/// Run one compaction job. Returns `true` if a compaction was executed,
/// `false` if no eligible level needed compaction (or all were busy).
async fn run_one_compaction_job(engine: &Arc<MeruEngine>) -> Result<bool> {
    // Phase 1: pick + reserve (brief lock).
    let (pick, version, _reservation) = match reserve_next_compaction(engine).await {
        Some(r) => r,
        None => {
            debug!("no compaction needed (or all candidates busy)");
            return Ok(false);
        }
    };
    // Issue #14 Phase 3: compaction-job wall-clock duration. Includes
    // source-file read, merge, output write, and commit. Sampled once
    // per job — never per row. Labeled by output level so operators
    // see which tier is slow without per-file cardinality.
    let job_started_at = std::time::Instant::now();
    let output_level_str = format!("L{}", pick.output_level.0);

    info!(
        input_level = pick.input_level.0,
        output_level = pick.output_level.0,
        score = pick.score,
        input_files = pick.input_files.len(),
        "starting compaction"
    );

    // Bug Q fix: reuse the same version snapshot used for picking. The old
    // code re-snapshotted `version_set.current()` here, which could return
    // a different version if a concurrent flush committed between pick and
    // this point — causing the compaction to read a file set inconsistent
    // with what the picker evaluated.
    //
    // Honor the picker's `input_files` selection (not `version.files_at`):
    // the picker may have chosen a subset of the level to bound per-job
    // memory via `max_compaction_bytes`. Taking all level files here
    // would defeat the cap and re-introduce OOM under stress (BUG-0002).
    let input_selection: std::collections::HashSet<&str> =
        pick.input_files.iter().map(|s| s.as_str()).collect();
    let input_file_metas: Vec<DataFileMeta> = version
        .files_at(pick.input_level)
        .iter()
        .filter(|f| input_selection.contains(f.path.as_str()))
        .cloned()
        .collect();
    if input_file_metas.is_empty() {
        debug!("compaction picked an empty input level");
        return Ok(false);
    }

    // ── Overlap pull-in: include every output-level file whose key range
    // intersects the union key range of `input_file_metas`. ──
    //
    // Classic leveled compaction invariant: L1+ must be non-overlapping
    // within each level so `find_file_for_key` can binary-search to a
    // single covering file. If a compaction writes a new L(k+1) file from
    // L(k) inputs WITHOUT rewriting the existing L(k+1) files that cover
    // the same keys, the new file overlaps the old ones and the point
    // lookup path silently returns stale data from the wrong file
    // (regression: `l1_overlap_regression::overlapping_l1_files_serve_correct_version_on_get`).
    //
    // Fix: compute the union key range of the primary inputs, then pull
    // every output-level file whose `[key_min, key_max]` intersects that
    // union into the compaction as additional inputs. They are read,
    // merged with full MVCC+dedup, and rewritten as part of the single
    // output file — leaving L(k+1) non-overlapping by construction.
    let (union_min, union_max) = compute_union_range(&input_file_metas);
    let overlap_output_metas: Vec<DataFileMeta> = if let (Some(umin), Some(umax)) =
        (union_min.as_ref(), union_max.as_ref())
    {
        version
            .files_at(pick.output_level)
            .iter()
            .filter(|f| {
                // Overlap iff `f.key_min <= umax && f.key_max >= umin`.
                (f.meta.key_min.is_empty() || f.meta.key_min.as_slice() <= umax.as_slice())
                    && (f.meta.key_max.is_empty() || f.meta.key_max.as_slice() >= umin.as_slice())
            })
            .cloned()
            .collect()
    } else {
        Vec::new()
    };

    if !overlap_output_metas.is_empty() {
        info!(
            output_level = pick.output_level.0,
            overlap_count = overlap_output_metas.len(),
            "pulling overlapping output-level files into compaction to preserve non-overlap invariant"
        );
    }

    // Read every source file (primary inputs + overlap pull-ins), applying
    // each file's current Deletion Vector so already-promoted rows don't
    // re-enter the output. Row positions are file-global physical positions
    // (u32) that DV stamping expects. `file_idx` is a dense index into the
    // combined list so `CompactionIterator` can disambiguate rows.
    let base = engine.catalog.base_path();
    let all_source_metas: Vec<&DataFileMeta> = input_file_metas
        .iter()
        .chain(overlap_output_metas.iter())
        .collect();
    let mut file_entries: Vec<FileEntries> = Vec::with_capacity(all_source_metas.len());
    for (file_idx, file_meta) in all_source_metas.iter().enumerate() {
        let (reader, dv) = open_source_file(base, file_meta, engine.schema.clone())?;
        let physical = reader.read_physical_rows_with_positions(dv.as_ref())?;
        file_entries.push(FileEntries {
            file_idx,
            entries: physical,
        });
    }

    // Build compaction iterator.
    let read_seq = engine.read_seq();
    let drop_tombstones =
        picker::should_drop_tombstones(pick.output_level, engine.config.level_target_bytes.len());
    let iter = CompactionIterator::new(file_entries, read_seq, drop_tombstones);

    if iter.is_empty() {
        // All inputs were tombstones dropped at the bottom level, or every
        // row was already DV-masked. Still install an empty transaction so
        // the source files can be removed.
        debug!(
            input_level = pick.input_level.0,
            "compaction produced no output rows"
        );
    }

    // Collect surviving entries and split into multiple output chunks
    // so no single file's aggregate-column-bytes can overflow Arrow's
    // i32 byte-array offset limit (~2.14 GiB per column). Issue #3:
    // a 10 GiB external analytics stress test panicked in
    // `arrow_array::builder::GenericBytesBuilder::append_value` when
    // a single ByteArray column in the output exceeded 2 GiB. Cap at
    // `TARGET_OUTPUT_FILE_BYTES` per chunk with 4× safety margin.
    //
    // Chunk boundaries MUST fall between different user_keys: two
    // L1+ files both containing user_key `X` would violate the
    // non-overlap invariant that `find_file_for_key` binary search
    // relies on. The iterator emits entries sorted by (user_key ASC,
    // seq DESC), so consecutive entries with the same user_key stay
    // in the same chunk even if the byte budget is exceeded.
    let mut chunks: Vec<OutputChunk> = Vec::new();
    let mut current = OutputChunk::empty();

    for entry in iter {
        let est = estimate_row_bytes(&entry.row) + entry.ikey.user_key_bytes().len() as u64 + 16;
        let uk = entry.ikey.user_key_bytes();
        // Start a new chunk when:
        //   (a) the current chunk has content,
        //   (b) adding this entry would exceed the target size,
        //   (c) this entry's user_key differs from the last — safe
        //       split point (otherwise two files would share a key).
        if !current.rows.is_empty()
            && current.approx_bytes.saturating_add(est) > TARGET_OUTPUT_FILE_BYTES
            && current.last_user_key.as_slice() != uk
        {
            chunks.push(std::mem::replace(&mut current, OutputChunk::empty()));
        }
        current.push(entry.ikey, entry.row, est);
    }
    if !current.rows.is_empty() {
        chunks.push(current);
    }

    let total_output_rows: u64 = chunks.iter().map(|c| c.rows.len() as u64).sum();

    // Build the snapshot transaction up front so we can commit even when
    // the compaction output is empty (pure tombstone drop at the bottom).
    let mut txn = SnapshotTransaction::new();

    // Write each chunk as its own Parquet file. Each gets its own
    // UUID, its own bloom filter, its own KvSparseIndex. Output L1+
    // non-overlap is preserved because chunk boundaries are between
    // different user_keys (see iterator contract above).
    if !chunks.is_empty() {
        engine.catalog.ensure_level_dir(pick.output_level).await?;
    }
    for chunk in chunks {
        if chunk.rows.is_empty() {
            continue;
        }
        let chunk_rows = chunk.rows.len() as u64;
        let seq_min = if chunk.seq_min == u64::MAX {
            0
        } else {
            chunk.seq_min
        };
        let seq_max = chunk.seq_max;
        let key_min = chunk.key_min;
        let key_max = chunk.key_max;

        let file_id = uuid::Uuid::new_v4().to_string();
        let output_path = format!("data/L{}/{file_id}.parquet", pick.output_level.0);

        // Issue #15: per-output-level format. A compaction moving from
        // Dual-at-L0 to Columnar-at-L1 drops the `_merutable_value`
        // blob; moving Dual-deeper retains it.
        let format = engine.config.file_format_for(pick.output_level);
        let (parquet_bytes, _, writer_meta) = crate::parquet::writer::write_sorted_rows(
            chunk.rows,
            engine.schema.clone(),
            pick.output_level,
            format,
            engine.config.bloom_bits_per_key,
        )?;

        if parquet_bytes.is_empty() {
            return Err(MeruError::Parquet(
                "writer returned empty bytes for non-empty row set".into(),
            ));
        }
        let file_size = parquet_bytes.len() as u64;

        let full_path = engine.catalog.data_file_path(pick.output_level, &file_id);
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(MeruError::Io)?;
        }
        tokio::fs::write(&full_path, &parquet_bytes)
            .await
            .map_err(MeruError::Io)?;

        // IMP-01: fsync the output SST before the manifest references it.
        tokio::fs::File::open(&full_path)
            .await
            .map_err(MeruError::Io)?
            .sync_all()
            .await
            .map_err(MeruError::Io)?;

        // IMP-19: fsync the data directory so the new file's directory
        // entry is durable before the manifest commit.
        if let Some(parent) = full_path.parent() {
            if let Ok(dir) = tokio::fs::File::open(parent).await {
                let _ = dir.sync_all().await;
            }
        }

        let meta = ParquetFileMeta {
            level: pick.output_level,
            seq_min,
            seq_max,
            key_min,
            key_max,
            num_rows: chunk_rows,
            file_size,
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            format: Some(format),
            column_stats: writer_meta.column_stats,
        };

        txn.add_file(IcebergDataFile {
            path: output_path,
            file_size,
            num_rows: chunk_rows,
            meta,
        });
    }

    // Remove every fully-consumed input file — primary inputs AND the
    // overlap pull-ins from the output level, which were fully read and
    // rewritten above.
    for file_meta in &input_file_metas {
        txn.remove_file(file_meta.path.clone());
    }
    for file_meta in &overlap_output_metas {
        txn.remove_file(file_meta.path.clone());
    }

    txn.set_prop("merutable.job", "compaction");
    txn.set_prop("merutable.input_level", pick.input_level.0.to_string());
    txn.set_prop("merutable.output_level", pick.output_level.0.to_string());

    // Commit. Two concurrent compactions on disjoint levels finish
    // their merges in parallel; their commits must linearize because
    // each computes `next_ver` from the current manifest — without
    // serialization they'd race on the version number and overwrite
    // each other's `v{N+1}.metadata.json`. The commit itself is brief
    // (single fsync chain), so this lock doesn't serialize the hot
    // merge work.
    let new_version = {
        let _commit_guard = engine.commit_lock.lock().await;
        let commit_started = std::time::Instant::now();
        let v = engine.catalog.commit(&txn, engine.schema.clone()).await?;
        crate::engine::metrics::record(
            crate::engine::metrics::COMMIT_DURATION_SECONDS,
            commit_started.elapsed().as_secs_f64(),
        );
        v
    };
    engine.version_set.install(new_version);

    // Issue #5: wake any writer parked on the L0 stop trigger.
    // Over-firing is harmless (waiters re-check L0 count before
    // proceeding); under-firing would hang the writer for up to the
    // background worker's 1-second heartbeat.
    engine.l0_drained.notify_waiters();

    // Issue #14: Phase-1 metrics. Per-level counter with a static
    // label (`level={0..5}`) — no dynamic per-key labels, matching
    // the issue's label policy.
    crate::engine::metrics::inc_labeled(
        crate::engine::metrics::COMPACTIONS_TOTAL,
        "input_level",
        pick.input_level.0.to_string(),
    );
    crate::engine::metrics::inc(crate::engine::metrics::SNAPSHOTS_COMMITTED_TOTAL);
    if !overlap_output_metas.is_empty() {
        crate::engine::metrics::inc_by(
            crate::engine::metrics::OVERLAP_PULLINS_TOTAL,
            overlap_output_metas.len() as u64,
        );
    }

    // Issue #14 Phase 3: compaction duration + output bytes, labeled
    // by output level. Sampled once per job (not per row) so per-op
    // overhead is zero.
    let output_bytes_total: u64 = txn.adds.iter().map(|f| f.file_size).sum();
    crate::engine::metrics::record_labeled(
        crate::engine::metrics::COMPACTION_DURATION_SECONDS,
        "output_level",
        output_level_str,
        job_started_at.elapsed().as_secs_f64(),
    );
    crate::engine::metrics::record(
        crate::engine::metrics::COMPACTION_OUTPUT_BYTES,
        output_bytes_total as f64,
    );

    // IMP-03: clear the row cache after compaction. Compaction rewrites
    // files and resolves MVCC versions — any entry cached from a now-obsolete
    // file could be stale. A full clear is simple and correct; the cache
    // refills on the next read wave.
    if let Some(ref cache) = engine.row_cache {
        cache.clear();
    }

    // IMP-12: enqueue obsoleted files for deferred deletion. External
    // readers (DuckDB, Spark) may still be mid-read of old files; immediate
    // deletion would cause read failures. Files are physically deleted
    // after gc_grace_period_secs has elapsed.
    let mut obsoleted_paths: Vec<std::path::PathBuf> = Vec::new();
    for file_meta in input_file_metas.iter().chain(overlap_output_metas.iter()) {
        obsoleted_paths.push(base.join(&file_meta.path));
        if let Some(ref dv) = file_meta.dv_path {
            obsoleted_paths.push(base.join(dv));
        }
    }
    // `version.snapshot_id` is the snapshot the compaction was based on
    // — the last snapshot in which the obsoleted files were still live.
    // GC keeps files alive while any reader pins a snapshot <= this
    // value (version-pinned safety for long reads).
    engine
        .enqueue_for_deletion(obsoleted_paths, version.snapshot_id)
        .await;

    // Run GC to clean up any files whose grace period has expired.
    engine.gc_pending_deletions().await;

    info!(
        input_level = pick.input_level.0,
        output_level = pick.output_level.0,
        output_rows = total_output_rows,
        "compaction committed"
    );

    Ok(true)
}
