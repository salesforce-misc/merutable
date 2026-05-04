//! Flush-time deletion-vector resolver. RFC-0002 Phase 2.
//!
//! Pure helper that consumes a sorted list of memtable user_keys and
//! returns the per-file `RoaringBitmap` of file-global row positions
//! to mark deleted in the next manifest commit.
//!
//! The resolver runs in the flush job under `commit_lock` against a
//! pinned `Version` snapshot. It performs **no** mutation of any
//! engine state — the caller threads the returned map into the
//! `SnapshotTransaction` via `add_dv()`.
//!
//! Algorithm: per-file range merge-intersection, NOT per-key probe.
//! For every prior file (L0 + L1+) whose key range overlaps the
//! memtable's range, the file's `iter_user_keys_in_range` stream is
//! sorted-merged against the memtable's sorted user_keys. Each
//! match contributes one position to that file's bitmap. Multi-version
//! rows for the same user_key all match the same memtable entry and
//! all positions are recorded (so an upsert DV-marks every prior
//! version, not just the latest).
//!
//! See `docs/rfc/0002-flush-time-deletion-vectors.md` for the full
//! design contract.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use roaring::RoaringBitmap;

use crate::iceberg::version::Version;
use crate::parquet::reader::ParquetReader;
use crate::types::{level::Level, schema::TableSchema, MeruError, Result};

/// Per-file DV deltas produced by one flush. Key = parquet file path
/// (object-store-relative, matches `DataFileMeta::path`); value =
/// the set of file-global row positions to mark deleted.
///
/// Empty bitmaps are NOT inserted — the map only contains files that
/// have at least one position to DV-mark. Callers that want to merge
/// with existing DVs should consult `SnapshotTransaction::add_dv`,
/// which handles the union.
pub type DvDeltas = HashMap<String, RoaringBitmap>;

/// Resolve flush-time DV deltas for one memtable's worth of upserts.
///
/// `memtable_user_keys` MUST be:
/// - sorted ASCending by byte order (the natural InternalKey
///   user_key ordering),
/// - deduplicated by user_key (no repeated user_keys — multiple
///   memtable versions of the same user_key collapse to one entry
///   for resolution purposes).
///
/// Empty input returns an empty map without opening any file.
///
/// `version` is the pinned snapshot the flush will commit on top of.
/// The resolver iterates every level (L0 included — see RFC-0002 on
/// why external Iceberg PK uniqueness requires L0 DV-marking too).
///
/// On any read error against any file, the resolver fails — the flush
/// caller decides whether to retry the whole job. Partial maps are
/// not returned (no half-commit risk).
pub fn resolve_dv_for_flush(
    memtable_user_keys: &[Vec<u8>],
    version: &Version,
    base_path: &Path,
    schema: &Arc<TableSchema>,
) -> Result<DvDeltas> {
    let mut out: DvDeltas = HashMap::new();
    if memtable_user_keys.is_empty() {
        return Ok(out);
    }
    // Cheap upper-bound for per-file disjoint check: the memtable's
    // own min/max user_key. Files outside this range are skipped
    // before any I/O.
    let m_lo = memtable_user_keys.first().unwrap().as_slice();
    let m_hi = memtable_user_keys.last().unwrap().as_slice();

    // Iterate every level from 0 upwards. RFC-0002: external Iceberg
    // readers must see one row per PK; that requires DV-marking
    // every prior version regardless of level.
    let max_level = version.max_level();
    for lvl in 0..=max_level.0 {
        let level = Level(lvl);
        for file in version.files_at(level) {
            if !file_overlaps_memtable(file, m_lo, m_hi) {
                continue;
            }
            // Compute the tight per-file range to iterate: the
            // intersection of the file's range and the memtable's
            // range. Saves work when only a slice of the file
            // overlaps the memtable.
            let f_min = file.meta.key_min.as_slice();
            let f_max = file.meta.key_max.as_slice();
            let lo = std::cmp::max(m_lo, f_min);
            let hi = std::cmp::min(m_hi, f_max);

            // Issue #90: load the file's existing DV (if any) so we
            // can short-circuit work that's already covered.
            let existing_dv = load_existing_dv(file, base_path)?;

            // Issue #90 fast-path: if the file is fully DV-marked
            // already (existing.cardinality == num_rows), no flush
            // can add a new position. Skip the page reads entirely.
            if let Some(ref existing) = existing_dv {
                if existing.len() == file.meta.num_rows {
                    continue;
                }
            }

            // Read + open the file. The parquet crate's
            // SerializedFileReader needs a `ChunkReader` — we hold
            // the bytes and clone-by-Arc internally via `Bytes`.
            let abs = base_path.join(&file.path);
            let bytes = std::fs::read(&abs).map_err(MeruError::Io)?;
            let reader = ParquetReader::open(Bytes::from(bytes), schema.clone())?;

            // Sorted merge: walk the file's (uk, position) stream,
            // advancing the memtable cursor on each step. Every match
            // writes a position into the file's bitmap. Memtable
            // keys greater than every yielded user_key are simply
            // not in this file (which is fine — they may match a
            // different file or be entirely new keys).
            let mut bitmap = RoaringBitmap::new();
            // partition_point: first index whose key >= lo. The
            // memtable_user_keys cursor we keep is the smallest
            // index whose key is >= the current file user_key (the
            // merge invariant).
            let mut m_idx = memtable_user_keys.partition_point(|k| k.as_slice() < lo);

            for item in reader.iter_user_keys_in_range(lo, hi)? {
                let (uk_f, pos) = item?;
                // Advance memtable cursor past every key strictly < uk_f.
                while m_idx < memtable_user_keys.len()
                    && memtable_user_keys[m_idx].as_slice() < uk_f.as_slice()
                {
                    m_idx += 1;
                }
                if m_idx >= memtable_user_keys.len() {
                    break; // memtable exhausted within this file's window
                }
                if memtable_user_keys[m_idx].as_slice() == uk_f.as_slice() {
                    // Match. Record this file-global position.
                    // RoaringBitmap is u32-indexed; positions
                    // exceeding u32::MAX are corruption (a single
                    // SST cannot hold > 4G rows).
                    if pos > u32::MAX as u64 {
                        return Err(MeruError::Corruption(format!(
                            "row position {pos} exceeds u32::MAX in file {}",
                            file.path
                        )));
                    }
                    let pos_u32 = pos as u32;
                    // Issue #90 per-position dedup: skip positions
                    // already in the existing DV. The catalog merge
                    // would no-op them anyway, but skipping here
                    // avoids enlarging the in-flight bitmap and
                    // (when ALL matches are covered) lets us drop
                    // the file from the txn entirely below.
                    if existing_dv.as_ref().is_some_and(|e| e.contains(pos_u32)) {
                        continue;
                    }
                    bitmap.insert(pos_u32);
                    // Do NOT advance m_idx: a multi-version key
                    // yields multiple positions (same uk_f) and we
                    // must mark every one.
                }
                // else memtable[m_idx] > uk_f: file's row not in
                // memtable, leave it alone.
            }

            if !bitmap.is_empty() {
                out.entry(file.path.clone())
                    .and_modify(|existing| {
                        *existing |= &bitmap;
                    })
                    .or_insert(bitmap);
            }
        }
    }
    Ok(out)
}

/// Issue #90: load the on-disk DV for a file if its manifest entry
/// has DV coords, returning the underlying `RoaringBitmap`. Returns
/// `Ok(None)` when the file has no DV. Defensive: inconsistent
/// coords (some Some, some None) are surfaced as `MeruError::Corruption`,
/// matching the catalog/read-path validation contract.
fn load_existing_dv(
    file: &crate::iceberg::version::DataFileMeta,
    base_path: &Path,
) -> Result<Option<RoaringBitmap>> {
    use crate::iceberg::DeletionVector;
    match (&file.dv_path, file.dv_offset, file.dv_length) {
        (Some(p), Some(o), Some(l)) => {
            if o < 0 || l < 0 {
                return Err(MeruError::Corruption(format!(
                    "DV has negative offset ({o}) or length ({l}) on file {}",
                    file.path
                )));
            }
            let abs = base_path.join(p);
            let bytes = std::fs::read(&abs).map_err(MeruError::Io)?;
            let off = o as usize;
            let len = l as usize;
            let end = off
                .checked_add(len)
                .ok_or_else(|| MeruError::Corruption("DV offset+length overflow".into()))?;
            if end > bytes.len() {
                return Err(MeruError::Corruption(format!(
                    "DV blob out of range on file {}: end={end} puffin_len={}",
                    file.path,
                    bytes.len()
                )));
            }
            let dv = DeletionVector::from_puffin_blob(&bytes[off..end])?;
            Ok(Some(dv.bitmap().clone()))
        }
        (None, None, None) => Ok(None),
        _ => Err(MeruError::Corruption(format!(
            "inconsistent DV coords on file {}: dv_path={:?} dv_offset={:?} dv_length={:?}",
            file.path, file.dv_path, file.dv_offset, file.dv_length
        ))),
    }
}

/// Range overlap on user_key bytes. Empty key bounds (a freshly
/// constructed file with no rows) are treated as "no overlap" so an
/// empty file never gets opened for DV resolution.
fn file_overlaps_memtable(
    file: &crate::iceberg::version::DataFileMeta,
    m_lo: &[u8],
    m_hi: &[u8],
) -> bool {
    let f_min = file.meta.key_min.as_slice();
    let f_max = file.meta.key_max.as_slice();
    if f_min.is_empty() || f_max.is_empty() {
        return false;
    }
    !(f_max < m_lo || f_min > m_hi)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iceberg::version::{DataFileMeta, Version};
    use crate::types::{
        key::InternalKey,
        level::{FileFormat, Level, ParquetFileMeta},
        schema::{ColumnDef, ColumnType, TableSchema},
        sequence::{OpType, SeqNum},
        value::{FieldValue, Row},
    };
    use bytes::Bytes as BBytes;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn test_schema() -> TableSchema {
        TableSchema {
            table_name: "t".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                    ..Default::default()
                },
                ColumnDef {
                    name: "v".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: true,
                    ..Default::default()
                },
            ],
            primary_key: vec![0],
            ..Default::default()
        }
    }

    fn ikey(id: i64, seq: u64) -> InternalKey {
        InternalKey::encode(
            &[FieldValue::Int64(id)],
            SeqNum(seq),
            OpType::Put,
            &test_schema(),
        )
        .unwrap()
    }

    fn user_key(id: i64) -> Vec<u8> {
        ikey(id, 0).user_key_bytes().to_vec()
    }

    /// Write a Parquet file with the given (id, seq) sequence and
    /// return its `DataFileMeta` ready to be installed into a
    /// `Version`. Writes the file at `base_path/relative` and
    /// returns the relative path inside the meta.
    fn write_file(
        base_path: &Path,
        relative: &str,
        rows: Vec<(i64, u64)>,
        level: Level,
    ) -> DataFileMeta {
        let schema = test_schema();
        let pq_rows: Vec<(InternalKey, Row)> = rows
            .iter()
            .map(|(id, seq)| {
                (
                    ikey(*id, *seq),
                    Row::new(vec![
                        Some(FieldValue::Int64(*id)),
                        Some(FieldValue::Bytes(BBytes::from(format!("v{id}_{seq}")))),
                    ]),
                )
            })
            .collect();
        let (bytes, _bloom, mut meta) = crate::parquet::writer::write_sorted_rows(
            pq_rows,
            Arc::new(schema),
            level,
            FileFormat::default_for_level(level),
            10,
        )
        .unwrap();
        meta.level = level;
        let abs = base_path.join(relative);
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&abs, &bytes).unwrap();
        DataFileMeta {
            path: relative.to_string(),
            meta,
            dv_path: None,
            dv_offset: None,
            dv_length: None,
        }
    }

    fn version_with_files(files_per_level: HashMap<Level, Vec<DataFileMeta>>) -> Version {
        Version {
            snapshot_id: 1,
            levels: files_per_level,
            schema: Arc::new(test_schema()),
        }
    }

    #[test]
    fn empty_memtable_returns_empty_map_no_io() {
        // No files necessary — empty input must short-circuit.
        let tmp = TempDir::new().unwrap();
        let v = Version::empty(Arc::new(test_schema()));
        let result = resolve_dv_for_flush(&[], &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn empty_version_returns_empty_map() {
        // Memtable has keys, but no prior files exist. Nothing to
        // resolve, but the call must not panic.
        let tmp = TempDir::new().unwrap();
        let keys: Vec<Vec<u8>> = (0..10).map(user_key).collect();
        let v = Version::empty(Arc::new(test_schema()));
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn memtable_disjoint_from_every_file_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let f0 = write_file(
            tmp.path(),
            "data/L1/a.parquet",
            (0..32).map(|id| (id, id as u64 + 1)).collect(),
            Level(1),
        );
        let v = version_with_files(HashMap::from([(Level(1), vec![f0])]));
        // Memtable keys at id ∈ [1000..1010] — disjoint from [0..32).
        let keys: Vec<Vec<u8>> = (1000..1010).map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn memtable_fully_overlaps_l1_file_marks_every_position() {
        let tmp = TempDir::new().unwrap();
        // File: id ∈ [0..32), one version each at seq=id+1.
        let f0 = write_file(
            tmp.path(),
            "data/L1/a.parquet",
            (0..32).map(|id| (id, id as u64 + 1)).collect(),
            Level(1),
        );
        let v = version_with_files(HashMap::from([(Level(1), vec![f0])]));
        // Memtable: every id in the file gets upserted.
        let keys: Vec<Vec<u8>> = (0..32).map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert_eq!(result.len(), 1);
        let bm = result
            .get("data/L1/a.parquet")
            .expect("file must be in map");
        // EXACT cardinality + EXACT positions: the file is sorted
        // by user_key ASC so positions 0..32 align with ids 0..32.
        assert_eq!(bm.len(), 32, "every prior position must be DV-marked");
        for pos in 0..32u32 {
            assert!(bm.contains(pos), "position {pos} missing from DV bitmap");
        }
    }

    #[test]
    fn partial_overlap_marks_only_intersecting_positions() {
        let tmp = TempDir::new().unwrap();
        // File: ids 0..32.
        let f0 = write_file(
            tmp.path(),
            "data/L1/a.parquet",
            (0..32).map(|id| (id, id as u64 + 1)).collect(),
            Level(1),
        );
        let v = version_with_files(HashMap::from([(Level(1), vec![f0])]));
        // Memtable: only ids 5..10 are upserts (others are new keys).
        let keys: Vec<Vec<u8>> = (5..10).map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert_eq!(result.len(), 1);
        let bm = &result["data/L1/a.parquet"];
        assert_eq!(bm.len(), 5, "exactly 5 positions");
        for pos in 5u32..10 {
            assert!(bm.contains(pos), "position {pos}");
        }
    }

    #[test]
    fn l0_file_priors_get_dv_marked() {
        // RFC-0002 load-bearing: L0 priors MUST be DV-marked for
        // external Iceberg PK uniqueness, not just L1+ priors.
        let tmp = TempDir::new().unwrap();
        let f0 = write_file(
            tmp.path(),
            "data/L0/a.parquet",
            (0..16).map(|id| (id, id as u64 + 1)).collect(),
            Level(0),
        );
        let v = version_with_files(HashMap::from([(Level(0), vec![f0])]));
        let keys: Vec<Vec<u8>> = (0..16).map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        let bm = &result["data/L0/a.parquet"];
        assert_eq!(bm.len(), 16, "every L0 prior must be DV-marked");
    }

    #[test]
    fn multi_version_l0_marks_every_version_position() {
        // Same id appears at three positions (three seqs) in one
        // L0 file. An upsert in the memtable must DV-mark all three.
        let tmp = TempDir::new().unwrap();
        let id: i64 = 7;
        // Three versions at seqs 30, 20, 10. Writer requires the
        // input rows to already be sorted ASC by InternalKey
        // (user_key ASC, seq DESC), which `(7, 30) (7, 20) (7, 10)`
        // satisfies.
        let f0 = write_file(
            tmp.path(),
            "data/L0/a.parquet",
            vec![(id, 30), (id, 20), (id, 10)],
            Level(0),
        );
        let v = version_with_files(HashMap::from([(Level(0), vec![f0])]));
        let keys = vec![user_key(id)];
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        let bm = &result["data/L0/a.parquet"];
        assert_eq!(bm.len(), 3, "all three versions must be marked");
        assert!(bm.contains(0));
        assert!(bm.contains(1));
        assert!(bm.contains(2));
    }

    #[test]
    fn priors_in_both_l0_and_l1_both_get_marked() {
        // The same key has versions in an L0 file AND an L1 file
        // (transient state between compactions). Both positions
        // must be DV-marked — RFC-0002 anti-fix A9.
        let tmp = TempDir::new().unwrap();
        let id_l0: i64 = 5;
        let id_l1: i64 = 5;
        let f_l0 = write_file(tmp.path(), "data/L0/a.parquet", vec![(id_l0, 50)], Level(0));
        let f_l1 = write_file(tmp.path(), "data/L1/b.parquet", vec![(id_l1, 10)], Level(1));
        let v = version_with_files(HashMap::from([
            (Level(0), vec![f_l0]),
            (Level(1), vec![f_l1]),
        ]));
        let keys = vec![user_key(id_l0)];
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert_eq!(result.len(), 2, "both files must have DV deltas");
        assert_eq!(result["data/L0/a.parquet"].len(), 1);
        assert_eq!(result["data/L1/b.parquet"].len(), 1);
        assert!(result["data/L0/a.parquet"].contains(0));
        assert!(result["data/L1/b.parquet"].contains(0));
    }

    #[test]
    fn new_keys_without_priors_emit_no_dv() {
        // Memtable keys are entirely new — none have priors in any
        // file. Result must be empty.
        let tmp = TempDir::new().unwrap();
        let f0 = write_file(
            tmp.path(),
            "data/L1/a.parquet",
            (0..16).map(|id| (id, id as u64 + 1)).collect(),
            Level(1),
        );
        let v = version_with_files(HashMap::from([(Level(1), vec![f0])]));
        // Memtable: ids 100..110 — none in the file's range.
        let keys: Vec<Vec<u8>> = (100..110).map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert!(
            result.is_empty(),
            "new keys must not emit DV deltas, got {result:?}"
        );
    }

    #[test]
    fn mixed_new_and_upsert_keys_only_priors_marked() {
        let tmp = TempDir::new().unwrap();
        let f0 = write_file(
            tmp.path(),
            "data/L1/a.parquet",
            vec![(1, 1), (3, 3), (5, 5), (7, 7)],
            Level(1),
        );
        let v = version_with_files(HashMap::from([(Level(1), vec![f0])]));
        // Memtable: 1 (upsert), 2 (new), 3 (upsert), 4 (new), 5 (upsert).
        let keys: Vec<Vec<u8>> = vec![1, 2, 3, 4, 5].into_iter().map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        let bm = &result["data/L1/a.parquet"];
        // File rows: [1@pos0, 3@pos1, 5@pos2, 7@pos3]. Upserts of
        // 1, 3, 5 mark positions 0, 1, 2.
        assert_eq!(bm.len(), 3);
        assert!(bm.contains(0));
        assert!(bm.contains(1));
        assert!(bm.contains(2));
        // 7@pos3 was NOT upserted, must NOT be marked.
        assert!(!bm.contains(3));
    }

    #[test]
    fn resolver_errors_on_missing_file() {
        let tmp = TempDir::new().unwrap();
        // Build the meta for a file that does NOT exist on disk.
        let bogus = DataFileMeta {
            path: "data/L1/missing.parquet".to_string(),
            meta: ParquetFileMeta {
                level: Level(1),
                seq_min: 1,
                seq_max: 10,
                key_min: user_key(0),
                key_max: user_key(31),
                num_rows: 32,
                file_size: 1024,
                dv_path: None,
                dv_offset: None,
                dv_length: None,
                format: Some(FileFormat::default_for_level(Level(1))),
                column_stats: None,
            },
            dv_path: None,
            dv_offset: None,
            dv_length: None,
        };
        let v = version_with_files(HashMap::from([(Level(1), vec![bogus])]));
        let keys: Vec<Vec<u8>> = (0..16).map(user_key).collect();
        let err = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema()))
            .expect_err("missing file must fail the resolver");
        // Surface as Io — we propagate fs::read errors verbatim.
        assert!(
            matches!(err, MeruError::Io(_)),
            "expected Io error, got {err:?}"
        );
    }

    // ── Issue #90: existing-DV optimizations ────────────────────────────────

    /// Helper: write a Puffin DV blob next to a parquet file and
    /// stamp the (path, offset, length) coords back into the meta.
    /// Mirrors what catalog::commit does on a real DV emission.
    fn stamp_dv_on_file(base: &Path, meta: &mut DataFileMeta, positions: &[u32], snapshot_id: i64) {
        use crate::iceberg::DeletionVector;
        let mut dv = DeletionVector::new();
        for &p in positions {
            dv.mark_deleted(p);
        }
        let encoded = dv
            .encode_puffin(&meta.path, snapshot_id, snapshot_id)
            .unwrap();
        let stem = std::path::Path::new(&meta.path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap();
        let parent = std::path::Path::new(&meta.path).parent().unwrap();
        let rel = parent.join(format!("{stem}.dv-{snapshot_id}.puffin"));
        let abs = base.join(&rel);
        std::fs::write(&abs, &encoded.bytes).unwrap();
        meta.dv_path = Some(rel.to_string_lossy().into_owned());
        meta.dv_offset = Some(encoded.blob_offset);
        meta.dv_length = Some(encoded.blob_length);
    }

    /// Files already FULLY DV-marked must be skipped: no page reads,
    /// no entry in the result. The fast-path is hit when an existing
    /// DV's cardinality equals the file's num_rows.
    #[test]
    fn fully_dv_marked_file_is_skipped() {
        let tmp = TempDir::new().unwrap();
        let mut f = write_file(
            tmp.path(),
            "data/L1/a.parquet",
            (0..10).map(|id| (id, id as u64 + 1)).collect(),
            Level(1),
        );
        // Stamp a DV that covers ALL 10 positions.
        let all: Vec<u32> = (0..10).collect();
        stamp_dv_on_file(tmp.path(), &mut f, &all, 5);
        let v = version_with_files(HashMap::from([(Level(1), vec![f])]));

        // Memtable upserts every key — the resolver MUST return
        // an empty map because the prior file is already fully marked.
        let keys: Vec<Vec<u8>> = (0..10).map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert!(
            result.is_empty(),
            "fully-marked file must produce no DV delta, got {result:?}"
        );
    }

    /// Per-position dedup: positions already in the existing DV are
    /// not re-added. The result bitmap holds ONLY the new positions.
    #[test]
    fn partial_dv_skips_already_covered_positions() {
        let tmp = TempDir::new().unwrap();
        let mut f = write_file(
            tmp.path(),
            "data/L1/a.parquet",
            (0..10).map(|id| (id, id as u64 + 1)).collect(),
            Level(1),
        );
        // Existing DV covers positions {0, 1, 2}; positions 3..10 are
        // still live.
        stamp_dv_on_file(tmp.path(), &mut f, &[0u32, 1, 2], 5);
        let v = version_with_files(HashMap::from([(Level(1), vec![f])]));

        // Memtable upserts ids 0..5 — positions 0,1,2 already
        // covered, only 3,4 are new.
        let keys: Vec<Vec<u8>> = (0..5).map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert_eq!(result.len(), 1);
        let bm = &result["data/L1/a.parquet"];
        assert_eq!(
            bm.len(),
            2,
            "only NEW positions (3, 4) should be in the delta, got cardinality {}",
            bm.len()
        );
        assert!(bm.contains(3));
        assert!(bm.contains(4));
        assert!(
            !bm.contains(0) && !bm.contains(1) && !bm.contains(2),
            "covered positions must NOT be re-added"
        );
    }

    /// If every match is already covered, the file must NOT appear
    /// in the result map (else we'd emit a redundant DV update).
    #[test]
    fn all_matches_covered_excludes_file_from_result() {
        let tmp = TempDir::new().unwrap();
        let mut f = write_file(
            tmp.path(),
            "data/L1/a.parquet",
            (0..10).map(|id| (id, id as u64 + 1)).collect(),
            Level(1),
        );
        // Covers positions 0..5.
        stamp_dv_on_file(tmp.path(), &mut f, &(0u32..5).collect::<Vec<_>>(), 5);
        let v = version_with_files(HashMap::from([(Level(1), vec![f])]));

        // Memtable upserts ids 0..5 — every match already covered.
        let keys: Vec<Vec<u8>> = (0..5).map(user_key).collect();
        let result = resolve_dv_for_flush(&keys, &v, tmp.path(), &Arc::new(test_schema())).unwrap();
        assert!(
            result.is_empty(),
            "every match covered → no entry, got {result:?}"
        );
    }
}
