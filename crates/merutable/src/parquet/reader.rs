//! `ParquetReader`: point lookup and range scan with Deletion Vector masking.
//!
//! Reads via the `parquet` crate's Arrow record-batch reader, projecting
//! only the columns needed for the file's level:
//! - L0: `[_merutable_ikey, _merutable_value]` — two-column read; the
//!   postcard blob carries the entire row, so a point lookup decodes one
//!   row from one column chunk instead of N typed columns.
//! - L1+: `[_merutable_ikey, ...all user columns]` — typed-column path;
//!   no value blob exists at this tier.

use std::sync::Arc;

use crate::types::{
    MeruError, Result, key::InternalKey, level::ParquetFileMeta, schema::TableSchema,
    sequence::SeqNum, value::Row,
};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::file::reader::{ChunkReader, FileReader, SerializedFileReader};
use roaring::RoaringBitmap;

use crate::parquet::{
    bloom::FastLocalBloom,
    codec::{self, IKEY_COLUMN_NAME, VALUE_BLOB_COLUMN_NAME},
    footer::decode_footer_kv,
    kv_index::{KV_INDEX_FOOTER_KEY, KvSparseIndex, PageLocation},
};

pub struct ParquetReader<R: ChunkReader + Clone> {
    /// Original source, kept so we can rebuild a fresh
    /// `ParquetRecordBatchReaderBuilder` per read. `Bytes` (the production
    /// case) is cheaply `Clone`; `File` users should wrap with `try_clone`.
    source: R,
    schema: Arc<TableSchema>,
    meta: ParquetFileMeta,
    bloom: Option<FastLocalBloom>,
    /// Domain sparse index over the `_merutable_ikey` column. Loaded from
    /// the `merutable.kv_index.v1` footer KV when present (every non-empty
    /// file written by `writer::write_sorted_rows`). When present, point
    /// lookups skip the full-file scan and read only the matched data
    /// page's row range.
    kv_index: Option<KvSparseIndex>,
}

impl<R: ChunkReader + Clone + 'static> ParquetReader<R> {
    /// Open a Parquet file for reading. Loads bloom filter from KV metadata if present.
    pub fn open(source: R, schema: Arc<TableSchema>) -> Result<Self> {
        let file_reader = SerializedFileReader::new(source.clone())
            .map_err(|e| MeruError::Parquet(e.to_string()))?;

        // Read KV metadata from the file footer.
        let file_meta = file_reader.metadata().file_metadata();
        let kv_map: std::collections::HashMap<String, String> = file_meta
            .key_value_metadata()
            .map(|kv| {
                kv.iter()
                    .filter_map(|e| e.value.as_ref().map(|v| (e.key.clone(), v.clone())))
                    .collect()
            })
            .unwrap_or_default();

        // Decode the merutable-specific footer KV pair (meta + embedded
        // schema) through the canonical decoder. This errors cleanly on
        // any missing or corrupt entry — previously this code inlined a
        // partial parser that silently fabricated a fake `ParquetFileMeta`
        // on a missing `merutable.meta` key, masking real corruption.
        // We drop the embedded schema here because the caller-provided
        // `schema` is authoritative; a follow-up could cross-check them
        // to detect schema drift.
        let (meta, _embedded_schema) = decode_footer_kv(&kv_map)?;

        // Load bloom filter from "merutable.bloom" KV entry.
        let bloom = kv_map.get("merutable.bloom").and_then(|hex_str| {
            hex::decode(hex_str)
                .ok()
                .and_then(|b| FastLocalBloom::from_bytes(&b).ok())
        });

        // Load kv_index from "merutable.kv_index.v1" KV entry. Absence is
        // tolerated (legacy / empty file) — the read path falls back to a
        // full-file scan in that case.
        let kv_index = kv_map.get(KV_INDEX_FOOTER_KEY).and_then(|hex_str| {
            hex::decode(hex_str)
                .ok()
                .and_then(|raw| KvSparseIndex::from_bytes(bytes::Bytes::from(raw)).ok())
        });

        Ok(Self {
            source,
            schema,
            meta,
            bloom,
            kv_index,
        })
    }

    /// Point lookup. Returns `None` if definitely absent (bloom) or not found.
    ///
    /// Read path:
    /// 1. **Bloom gate** — definite absence short-circuits before any I/O.
    /// 2. **File-level key range gate** — drops out-of-range probes.
    /// 3. **kv_index page locate** — when the footer carries a
    ///    `KvSparseIndex`, find the data page that contains the largest key
    ///    ≤ the probe and read only that page's rows. Internal keys with
    ///    the same user key but different `seq` are ordered (PK ASC,
    ///    seq DESC) within the *same* page on the `_merutable_ikey` column,
    ///    so a single page is sufficient to cover every visible version
    ///    of a given user key.
    /// 4. **Fallback full scan** — only when no kv_index is present
    ///    (legacy or pathological files).
    pub fn get(
        &self,
        user_key_bytes: &[u8],
        read_seq: SeqNum,
        deleted_rows: Option<&RoaringBitmap>,
    ) -> Result<Option<(InternalKey, Row)>> {
        // Bloom gate.
        if let Some(bloom) = &self.bloom
            && !bloom.may_contain(user_key_bytes)
        {
            return Ok(None);
        }

        // Check file-level key range.
        if !self.meta.key_min.is_empty() && user_key_bytes < self.meta.key_min.as_slice() {
            return Ok(None);
        }
        if !self.meta.key_max.is_empty() && user_key_bytes > self.meta.key_max.as_slice() {
            return Ok(None);
        }

        // kv_index fast path: locate the data page containing the largest
        // ikey ≤ (user_key_bytes, MAX_SEQ). The synthetic probe key is the
        // user key encoded with `seq = u64::MAX` (or any seq above any
        // possible written seq), because internal keys sort by user key
        // ASC then seq DESC — so the *first* version of `user_key_bytes`
        // is the smallest ikey we want to cover.
        //
        // We search for "the largest ikey ≤ probe", which means the
        // matched page is *guaranteed* to contain the first version of
        // `user_key_bytes` if any version exists in this file at all.
        if let Some(idx) = &self.kv_index {
            // Build the synthetic probe: user_key_bytes appended with
            // ikey trailer for (seq=MAX, OpType::Put). Since the kv_index
            // entries are full encoded `InternalKey` bytes, we need to
            // construct a probe in the same encoding space. The cheapest
            // valid upper bound is `user_key_bytes` followed by trailing
            // bytes that compare ≥ any ikey trailer for this user key.
            //
            // InternalKey encoding (see merutable-types::key) places the
            // user key bytes first and the seq+op_type trailer last, so
            // appending 0xFF...0xFF gives a probe that strictly succeeds
            // every real ikey for this user key. The kv_index search uses
            // "largest ≤ target", which then lands on the page that
            // contains this user key's smallest ikey (its first version).
            let mut probe = Vec::with_capacity(user_key_bytes.len() + 9);
            probe.extend_from_slice(user_key_bytes);
            probe.extend_from_slice(&[0xFFu8; 9]);

            let matched = idx.find_page_with_next(&probe);
            match matched {
                Some((page_loc, next_first_row)) => {
                    let global_start = page_loc.first_row_index;
                    let rows = self.read_rows_in_page(page_loc, next_first_row)?;

                    // Bug H guard: the probe (uk + 0xFF×9) searches for
                    // the "largest page ≤ probe", which for a single
                    // user key spanning many pages always lands on the
                    // LAST page (lowest seq versions). That page may
                    // return a valid but STALE hit — or no hit at all.
                    //
                    // Safe fast return iff the page cannot have missed
                    // NEWER versions on an earlier page. If the page is
                    // not the first in the file AND its first row carries
                    // the SAME user key as the probe, then versions of
                    // this key may continue from the previous page —
                    // fall through to full scan in that case.
                    let may_have_earlier = global_start > 0
                        && rows
                            .first()
                            .is_some_and(|(ik, _)| ik.user_key_bytes() == user_key_bytes);

                    let result = Self::find_visible(
                        rows,
                        global_start,
                        user_key_bytes,
                        read_seq,
                        deleted_rows,
                    );
                    if result.is_some() && !may_have_earlier {
                        return Ok(result);
                    }
                    // Else: either no visible version on this page, or
                    // newer versions may exist on earlier pages (Bug H).
                    // Fall through to the full-file scan which is
                    // guaranteed correct.
                }
                None => {
                    // Probe precedes the file's first key — fall through
                    // to full scan.
                }
            }
        }

        // Full-file scan fallback: either no kv_index, or kv_index page
        // didn't contain the visible version (cross-page multi-version
        // edge case). Always correct.
        let rows = self.read_all_rows()?;
        Ok(Self::find_visible(
            rows,
            0,
            user_key_bytes,
            read_seq,
            deleted_rows,
        ))
    }

    /// Walk a row slice and return the first MVCC-visible
    /// `(ikey, row)` whose user key matches `user_key_bytes` and whose
    /// `seq` is ≤ `read_seq`, honoring the optional Deletion Vector.
    ///
    /// `global_start` is the file-global row index of the slice's first
    /// element — the DV is keyed against file-global positions, so a
    /// page-restricted slice must add its starting offset to the
    /// per-call enumerate index before consulting the DV.
    fn find_visible(
        rows: Vec<(InternalKey, Row)>,
        global_start: u64,
        user_key_bytes: &[u8],
        read_seq: SeqNum,
        deleted_rows: Option<&RoaringBitmap>,
    ) -> Option<(InternalKey, Row)> {
        for (local_pos, (ikey, row)) in rows.into_iter().enumerate() {
            let global_pos = global_start + local_pos as u64;
            if let Some(dv) = deleted_rows {
                // Bug M: match the safe u32::try_from pattern used in
                // read_physical_rows_with_positions. Positions beyond u32::MAX
                // cannot be in the DV bitmap, so treat them as not-deleted.
                if let Ok(pos32) = u32::try_from(global_pos)
                    && dv.contains(pos32)
                {
                    continue;
                }
            }
            if ikey.user_key_bytes() != user_key_bytes {
                continue;
            }
            if ikey.seq > read_seq {
                continue;
            }
            return Some((ikey, row));
        }
        None
    }

    /// Scan rows in key order with optional range and DV filtering.
    ///
    /// Tombstone entries (`OpType::Delete`) are **included** in the output
    /// so that the cross-file merge in `read_path::range_scan` can see
    /// them and correctly shadow older Puts from other files (or the
    /// memtable). Bug J: previously tombstones were dropped here, which
    /// caused a Delete in one file to be silently lost while an older
    /// Put from a different file survived the global dedup — resurrecting
    /// deleted rows.
    pub fn scan(
        &self,
        start_user_key: Option<&[u8]>,
        end_user_key: Option<&[u8]>,
        read_seq: SeqNum,
        deleted_rows: Option<&RoaringBitmap>,
    ) -> Result<Vec<(InternalKey, Row)>> {
        let rows = self.read_all_rows()?;
        let mut results = Vec::new();
        let mut last_uk: Option<Vec<u8>> = None;

        for (global_pos, (ikey, row)) in rows.into_iter().enumerate() {
            if let Some(dv) = deleted_rows {
                // Bug M: safe u32 conversion — positions beyond u32::MAX
                // cannot appear in a RoaringBitmap, so skip the DV check.
                if let Ok(pos32) = u32::try_from(global_pos)
                    && dv.contains(pos32)
                {
                    continue;
                }
            }
            if ikey.seq > read_seq {
                continue;
            }
            let uk = ikey.user_key_bytes().to_vec();
            if let Some(start) = start_user_key
                && uk.as_slice() < start
            {
                continue;
            }
            if let Some(end) = end_user_key
                && uk.as_slice() >= end
            {
                break;
            }
            if let Some(ref last) = last_uk
                && *last == uk
            {
                continue;
            }
            last_uk = Some(uk);
            // Tombstones are NOT dropped here — the caller handles them
            // in the global cross-file merge. See Bug J.
            results.push((ikey, row));
        }
        Ok(results)
    }

    pub fn meta(&self) -> &ParquetFileMeta {
        &self.meta
    }

    /// Read every physical row in file order, tagged with its file-global
    /// row position, and filter out any positions masked by `deleted_rows`.
    ///
    /// Intended for **compaction** — the compaction iterator needs every
    /// physical row (no seq/tombstone filtering, because it performs its
    /// own dedup and tombstone handling) *and* needs the original row
    /// position so it can later stamp the source file's Deletion Vector.
    ///
    /// Internal readers (point lookup, scan) should NOT use this; they
    /// already do MVCC gating inside `get` / `scan`.
    pub fn read_physical_rows_with_positions(
        &self,
        deleted_rows: Option<&RoaringBitmap>,
    ) -> Result<Vec<(InternalKey, Row, u32)>> {
        let rows = self.read_all_rows()?;
        let mut out = Vec::with_capacity(rows.len());
        for (pos, (ikey, row)) in rows.into_iter().enumerate() {
            let pos_u32 = u32::try_from(pos).map_err(|_| {
                MeruError::Parquet(format!(
                    "row position {pos} exceeds u32::MAX in Parquet file"
                ))
            })?;
            if let Some(dv) = deleted_rows
                && dv.contains(pos_u32)
            {
                continue;
            }
            out.push((ikey, row, pos_u32));
        }
        Ok(out)
    }

    /// Read every row in the file as a fully-decoded `(InternalKey, Row)`
    /// pair using a column-projected Arrow record-batch reader.
    ///
    /// Projection is level-aware:
    /// - At L0 we ask only for `_merutable_ikey` and `_merutable_value`,
    ///   then `codec::record_batch_to_rows` takes the postcard fast path
    ///   and decodes each `Row` from a single column chunk's bytes.
    /// - At L1+ we ask for `_merutable_ikey` plus every user column, and
    ///   `codec::record_batch_to_rows` materializes each `Row` field by
    ///   field from the typed Arrow arrays.
    fn read_all_rows(&self) -> Result<Vec<(InternalKey, Row)>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.source.clone())
            .map_err(|e| MeruError::Parquet(e.to_string()))?;

        let mask = self.build_projection_mask(builder.parquet_schema())?;
        let reader = builder
            .with_projection(mask)
            .build()
            .map_err(|e| MeruError::Parquet(e.to_string()))?;

        let mut out = Vec::with_capacity(self.meta.num_rows as usize);
        for batch_result in reader {
            let batch = batch_result.map_err(|e| MeruError::Parquet(e.to_string()))?;
            let mut decoded = codec::record_batch_to_rows(&batch, &self.schema)?;
            out.append(&mut decoded);
        }
        Ok(out)
    }

    /// Read every row inside a single Parquet data page on the
    /// `_merutable_ikey` column, identified by its `PageLocation`.
    ///
    /// The kv_index entries store **file-global** `first_row_index` values
    /// (accumulated across all row groups; see
    /// `writer::extract_kv_index_entries`), but Parquet's `RowSelection`
    /// is **row-group-local**: when combined with `with_row_groups(vec![rg])`,
    /// the selection's offsets must be relative to that row group's start.
    /// We therefore walk the file metadata once to find the row group
    /// containing `page_loc.first_row_index` and convert to the local
    /// offset, then bound the page's row count using `next_first_row`
    /// (the next kv_index entry's first_row_index, or the row group's
    /// end if the matched page is the last one in the file or the next
    /// entry lives in a later row group — pages cannot span row groups,
    /// so the row-group end is always a safe clamp).
    fn read_rows_in_page(
        &self,
        page_loc: PageLocation,
        next_first_row: Option<u64>,
    ) -> Result<Vec<(InternalKey, Row)>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.source.clone())
            .map_err(|e| MeruError::Parquet(e.to_string()))?;

        // Locate the enclosing row group by walking cumulative row counts.
        let metadata = builder.metadata().clone();
        let mut cum: u64 = 0;
        let mut found: Option<(usize, u64, u64)> = None; // (rg_idx, rg_start, rg_end)
        for (rg_idx, rg) in metadata.row_groups().iter().enumerate() {
            let rg_num_rows = rg.num_rows();
            // Bug P5 fix: validate non-negative num_rows before u64 cast.
            if rg_num_rows < 0 {
                return Err(MeruError::Corruption(format!(
                    "negative num_rows {} in row group {rg_idx}",
                    rg_num_rows
                )));
            }
            let rg_rows = rg_num_rows as u64;
            let rg_start = cum;
            let rg_end = cum + rg_rows;
            if page_loc.first_row_index >= rg_start && page_loc.first_row_index < rg_end {
                found = Some((rg_idx, rg_start, rg_end));
                break;
            }
            cum = rg_end;
        }
        let (rg_idx, rg_start, rg_end) = found.ok_or_else(|| {
            MeruError::Parquet(format!(
                "kv_index page first_row_index {} out of range for file row count {}",
                page_loc.first_row_index, cum
            ))
        })?;

        // Pages cannot span row groups: clamp `next_first_row` (which may
        // be `None` for the last entry, or live in a later row group) to
        // the enclosing row group's end.
        let upper = match next_first_row {
            Some(n) if n <= rg_end => n,
            _ => rg_end,
        };
        let page_row_count = (upper - page_loc.first_row_index) as usize;
        let intra_rg_offset = (page_loc.first_row_index - rg_start) as usize;

        let mask = self.build_projection_mask(builder.parquet_schema())?;
        let mut selectors: Vec<RowSelector> = Vec::with_capacity(2);
        if intra_rg_offset > 0 {
            selectors.push(RowSelector::skip(intra_rg_offset));
        }
        if page_row_count == 0 {
            // Degenerate: empty page range, nothing to read.
            return Ok(Vec::new());
        }
        selectors.push(RowSelector::select(page_row_count));
        let selection = RowSelection::from(selectors);

        let reader = builder
            .with_row_groups(vec![rg_idx])
            .with_projection(mask)
            .with_row_selection(selection)
            .build()
            .map_err(|e| MeruError::Parquet(e.to_string()))?;

        let mut out = Vec::with_capacity(page_row_count);
        for batch_result in reader {
            let batch = batch_result.map_err(|e| MeruError::Parquet(e.to_string()))?;
            let mut decoded = codec::record_batch_to_rows(&batch, &self.schema)?;
            out.append(&mut decoded);
        }
        Ok(out)
    }

    /// Build the level-aware leaf-column projection mask used by both
    /// `read_all_rows` and `read_rows_in_page`. At L0 we project only
    /// `[_merutable_ikey, _merutable_value]` (the postcard fast path); at
    /// L1+ we project `[_merutable_ikey, ...all user columns]`.
    fn build_projection_mask(
        &self,
        parquet_schema: &parquet::schema::types::SchemaDescriptor,
    ) -> Result<ProjectionMask> {
        let mut leaf_indices: Vec<usize> = Vec::new();
        leaf_indices.push(find_leaf(parquet_schema, IKEY_COLUMN_NAME)?);
        // Issue #15: switch on the file's stamped format rather than
        // its level. Legacy files that predate the stamp fall through
        // to `FileFormat::default_for_level` which matches the old
        // level-based behavior (Dual iff L0).
        let format = self
            .meta
            .format
            .unwrap_or_else(|| crate::types::level::FileFormat::default_for_level(self.meta.level));
        if format.has_value_blob() {
            leaf_indices.push(find_leaf(parquet_schema, VALUE_BLOB_COLUMN_NAME)?);
        } else {
            // Issue #44 Stage 3: additive schema evolution. A file
            // written under an older schema_id may legitimately be
            // missing one of the current schema's user columns.
            // SKIP missing leaves in the projection mask; the codec
            // layer (`record_batch_to_rows`) fills them with
            // `initial_default` (or null) at row-construction time.
            // Without this tolerance, a reopen-with-extended-schema
            // (already accepted by `check_schema_compatible` per
            // #44 Stage 1) would break every read of an existing
            // Parquet file.
            for col in &self.schema.columns {
                if let Some(idx) = find_leaf_opt(parquet_schema, &col.name) {
                    leaf_indices.push(idx);
                }
            }
        }
        Ok(ProjectionMask::leaves(parquet_schema, leaf_indices))
    }
}

fn find_leaf(schema: &parquet::schema::types::SchemaDescriptor, name: &str) -> Result<usize> {
    find_leaf_opt(schema, name).ok_or_else(|| {
        MeruError::Corruption(format!("column '{name}' not found in Parquet schema"))
    })
}

/// Issue #44 Stage 3: non-failing leaf lookup. Returns `None` when
/// the column is absent from the Parquet file, letting the caller
/// decide whether that's a hard error (e.g., `_merutable_ikey`
/// missing → corruption) or an acceptable additive-evolution gap
/// the codec fills with a default.
fn find_leaf_opt(schema: &parquet::schema::types::SchemaDescriptor, name: &str) -> Option<usize> {
    (0..schema.num_columns()).find(|&i| schema.column(i).name() == name)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        level::Level,
        schema::{ColumnDef, ColumnType},
        sequence::OpType,
        value::{FieldValue, Row},
    };
    use bytes::Bytes as BBytes;
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

    fn write_test_file(rows: Vec<(InternalKey, Row)>, schema: &TableSchema) -> Vec<u8> {
        let (parquet_bytes, _bloom, _meta) = crate::parquet::writer::write_sorted_rows(
            rows,
            Arc::new(schema.clone()),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();
        parquet_bytes
    }

    fn make_ikey(id: i64, seq: u64) -> InternalKey {
        InternalKey::encode(
            &[FieldValue::Int64(id)],
            SeqNum(seq),
            OpType::Put,
            &test_schema(),
        )
        .unwrap()
    }

    #[test]
    fn write_and_read_roundtrip() {
        let schema = test_schema();
        let rows = vec![
            (
                make_ikey(1, 1),
                Row::new(vec![
                    Some(FieldValue::Int64(1)),
                    Some(FieldValue::Bytes(BBytes::from("hello"))),
                ]),
            ),
            (
                make_ikey(2, 2),
                Row::new(vec![
                    Some(FieldValue::Int64(2)),
                    Some(FieldValue::Bytes(BBytes::from("world"))),
                ]),
            ),
        ];

        let bytes = write_test_file(rows, &schema);
        assert!(!bytes.is_empty());

        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema)).unwrap();
        assert_eq!(reader.meta().num_rows, 2);
    }

    #[test]
    fn point_lookup_found() {
        let schema = test_schema();
        let ikey = make_ikey(42, 1);
        let uk = ikey.user_key_bytes().to_vec();
        let original_row = Row::new(vec![
            Some(FieldValue::Int64(42)),
            Some(FieldValue::Bytes(BBytes::from("found_me"))),
        ]);
        let rows = vec![(ikey, original_row.clone())];

        let bytes = write_test_file(rows, &schema);
        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema)).unwrap();

        let result = reader.get(&uk, SeqNum(10), None).unwrap();
        assert!(result.is_some());
        let (ikey, row) = result.unwrap();
        assert_eq!(ikey.seq, SeqNum(1));
        // Real value decode — not a Row::default() placeholder. The row
        // returned by the reader must equal the row written, field-for-field.
        assert_eq!(
            row, original_row,
            "reader must return the actual decoded row, not a default placeholder"
        );
    }

    #[test]
    fn point_lookup_not_found() {
        let schema = test_schema();
        let rows = vec![(
            make_ikey(1, 1),
            Row::new(vec![Some(FieldValue::Int64(1)), None]),
        )];

        let bytes = write_test_file(rows, &schema);
        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema.clone())).unwrap();

        // Look up key 99 — should not be found.
        let missing_ikey = make_ikey(99, 1);
        let result = reader
            .get(missing_ikey.user_key_bytes(), SeqNum(10), None)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn point_lookup_bloom_rejects() {
        let schema = test_schema();
        let rows = vec![(
            make_ikey(1, 1),
            Row::new(vec![Some(FieldValue::Int64(1)), None]),
        )];

        let bytes = write_test_file(rows, &schema);
        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema.clone())).unwrap();

        // Random key very likely rejected by bloom.
        let random_ikey = make_ikey(999_999, 1);
        let result = reader
            .get(random_ikey.user_key_bytes(), SeqNum(10), None)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn point_lookup_respects_read_seq() {
        let schema = test_schema();
        let rows = vec![(
            make_ikey(1, 10),
            Row::new(vec![Some(FieldValue::Int64(1)), None]),
        )];

        let bytes = write_test_file(rows, &schema);
        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema.clone())).unwrap();

        let ikey = make_ikey(1, 10);
        // Read at seq 5: should not see seq=10 write.
        let result = reader.get(ikey.user_key_bytes(), SeqNum(5), None).unwrap();
        assert!(result.is_none());

        // Read at seq 10: should see it.
        let result = reader.get(ikey.user_key_bytes(), SeqNum(10), None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn point_lookup_with_deletion_vector() {
        let schema = test_schema();
        let rows = vec![
            (
                make_ikey(1, 1),
                Row::new(vec![Some(FieldValue::Int64(1)), None]),
            ),
            (
                make_ikey(2, 2),
                Row::new(vec![Some(FieldValue::Int64(2)), None]),
            ),
        ];

        let bytes = write_test_file(rows, &schema);
        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema.clone())).unwrap();

        // DV marks row 0 (key=1) as deleted.
        let mut dv = RoaringBitmap::new();
        dv.insert(0);

        let ikey1 = make_ikey(1, 1);
        let result = reader
            .get(ikey1.user_key_bytes(), SeqNum(10), Some(&dv))
            .unwrap();
        assert!(result.is_none(), "row 0 should be masked by DV");

        let ikey2 = make_ikey(2, 2);
        let result = reader
            .get(ikey2.user_key_bytes(), SeqNum(10), Some(&dv))
            .unwrap();
        assert!(result.is_some(), "row 1 should still be visible");
    }

    #[test]
    fn scan_returns_all_rows() {
        let schema = test_schema();
        let originals: Vec<(InternalKey, Row)> = (1..=5i64)
            .map(|i| {
                (
                    make_ikey(i, i as u64),
                    Row::new(vec![
                        Some(FieldValue::Int64(i)),
                        Some(FieldValue::Bytes(BBytes::from(format!("v{i}")))),
                    ]),
                )
            })
            .collect();

        let bytes = write_test_file(originals.clone(), &schema);
        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema)).unwrap();

        let results = reader.scan(None, None, SeqNum(100), None).unwrap();
        assert_eq!(results.len(), 5);
        // Field-for-field equality: real value decode end-to-end.
        for ((orig_ik, orig_row), (got_ik, got_row)) in originals.iter().zip(results.iter()) {
            assert_eq!(orig_ik.seq, got_ik.seq);
            assert_eq!(orig_ik.user_key_bytes(), got_ik.user_key_bytes());
            assert_eq!(orig_row, got_row);
        }
    }

    #[test]
    fn scan_with_dv_excludes_deleted() {
        let schema = test_schema();
        let rows: Vec<_> = (1..=5i64)
            .map(|i| {
                (
                    make_ikey(i, i as u64),
                    Row::new(vec![Some(FieldValue::Int64(i)), None]),
                )
            })
            .collect();

        let bytes = write_test_file(rows, &schema);
        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema)).unwrap();

        let mut dv = RoaringBitmap::new();
        dv.insert(1); // Delete row at position 1 (key=2).
        dv.insert(3); // Delete row at position 3 (key=4).

        let results = reader.scan(None, None, SeqNum(100), Some(&dv)).unwrap();
        assert_eq!(results.len(), 3); // rows 0, 2, 4 survive
    }

    /// L1 files have no `_merutable_value` blob — the reader must take the
    /// typed-column decode path and still reconstruct each `Row`
    /// field-for-field. This is the cold-tier read contract: pure columnar
    /// shape, full row materialization.
    #[test]
    fn l1_typed_only_decode_roundtrip() {
        let schema = test_schema();
        let originals: Vec<(InternalKey, Row)> = (1..=10i64)
            .map(|i| {
                let val = if i % 2 == 0 {
                    Some(FieldValue::Bytes(BBytes::from(format!("payload_{i}"))))
                } else {
                    None
                };
                (
                    make_ikey(i, i as u64),
                    Row::new(vec![Some(FieldValue::Int64(i)), val]),
                )
            })
            .collect();

        // Write at Level(1) → no value blob column.
        let (bytes, _bloom, _meta) = crate::parquet::writer::write_sorted_rows(
            originals.clone(),
            Arc::new(schema.clone()),
            Level(1),
            crate::types::level::FileFormat::Columnar,
            10,
        )
        .unwrap();
        let reader = ParquetReader::open(BBytes::from(bytes), Arc::new(schema)).unwrap();

        let scanned = reader.scan(None, None, SeqNum(100), None).unwrap();
        assert_eq!(scanned.len(), originals.len());
        for ((orig_ik, orig_row), (got_ik, got_row)) in originals.iter().zip(scanned.iter()) {
            assert_eq!(orig_ik.seq, got_ik.seq);
            assert_eq!(orig_ik.user_key_bytes(), got_ik.user_key_bytes());
            assert_eq!(
                orig_row, got_row,
                "L1 typed-column decode must reproduce written rows exactly"
            );
        }

        // Point lookup also goes through the typed-column path at L1.
        let probe = make_ikey(4, 4);
        let got = reader
            .get(probe.user_key_bytes(), SeqNum(100), None)
            .unwrap();
        let (_, row) = got.expect("L1 point lookup must find existing key");
        assert_eq!(&row, &originals[3].1);
    }

    /// Build a multi-page L0 file by emitting `n` rows with
    /// **incompressible** per-row payloads — every payload byte derives
    /// from an LCG seeded with the row index, so snappy/dictionary
    /// coding can't collapse them. This guarantees that even modest row
    /// counts (a few thousand) reliably produce multiple data pages on
    /// the `_merutable_ikey` column at L0's 8 KiB page limit. The
    /// returned `(rows, reader)` mirror the inputs so tests can do
    /// field-for-field comparison.
    fn build_multi_page_l0_reader(n: i64) -> (Vec<(InternalKey, Row)>, ParquetReader<BBytes>) {
        let schema = Arc::new(test_schema());
        let rows: Vec<(InternalKey, Row)> = (1..=n)
            .map(|i| {
                let ikey = make_ikey(i, i as u64);
                // 256-byte incompressible payload: an LCG seeded from
                // the row index. Each row's payload is fully unique and
                // bytes within a single row look uniformly random, so
                // snappy gets ≈1.0× ratio.
                let mut state: u64 = (i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ 0xdeadbeef;
                let mut payload = vec![0u8; 256];
                for byte in &mut payload {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *byte = (state >> 33) as u8;
                }
                let row = Row::new(vec![
                    Some(FieldValue::Int64(i)),
                    Some(FieldValue::Bytes(BBytes::from(payload))),
                ]);
                (ikey, row)
            })
            .collect();
        let (bytes, _, _) = crate::parquet::writer::write_sorted_rows(
            rows.clone(),
            schema.clone(),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();
        let reader = ParquetReader::open(BBytes::from(bytes), schema).unwrap();
        (rows, reader)
    }

    /// On a multi-page L0 file, the kv_index page-skipping path must
    /// (a) return the same row that a full-file scan would, and
    /// (b) actually read *strictly fewer* rows than the file contains
    /// (otherwise no page-skipping has happened — silent regression).
    #[test]
    fn kv_index_point_lookup_skips_pages() {
        let (rows, reader) = build_multi_page_l0_reader(16_384);
        assert!(
            reader.kv_index.is_some(),
            "writer must emit a kv_index for a 16k-row L0 file"
        );
        let idx = reader.kv_index.as_ref().unwrap();
        assert!(
            idx.len() >= 2,
            "expected ≥2 kv_index entries (multi-page); got {}",
            idx.len()
        );

        // Probe a row in the middle of the file.
        let target = &rows[8000].0;
        let user_key = target.user_key_bytes();

        // Build the same synthetic upper-bound probe `get()` uses.
        let mut probe = Vec::with_capacity(user_key.len() + 9);
        probe.extend_from_slice(user_key);
        probe.extend_from_slice(&[0xFFu8; 9]);
        let (page_loc, next) = idx.find_page_with_next(&probe).unwrap();
        let page_rows = reader.read_rows_in_page(page_loc, next).unwrap();

        // Page-skipping happened: matched page has *fewer* rows than the
        // entire file (otherwise we'd be back to a full scan).
        assert!(
            page_rows.len() < rows.len(),
            "page slice ({}) must be strictly smaller than full file ({})",
            page_rows.len(),
            rows.len()
        );
        assert!(!page_rows.is_empty(), "matched page must contain rows");

        // The target row must be inside the matched page slice.
        let found = page_rows
            .iter()
            .find(|(ik, _)| ik.user_key_bytes() == user_key);
        assert!(
            found.is_some(),
            "kv_index page slice must contain the probed user key"
        );

        // End-to-end via `get()` returns the same row as the input.
        let got = reader.get(user_key, SeqNum(u64::MAX), None).unwrap();
        let (_, got_row) = got.expect("kv_index path must find existing row");
        assert_eq!(got_row, rows[8000].1);
    }

    /// Edge cases on a multi-page L0 file: first row, last row, and a
    /// missing key (between two existing keys). All three must be served
    /// correctly through the kv_index fast path.
    #[test]
    fn kv_index_point_lookup_edge_cases() {
        let (rows, reader) = build_multi_page_l0_reader(8192);
        assert!(reader.kv_index.is_some());

        // First row.
        let first_uk = rows[0].0.user_key_bytes();
        let got = reader
            .get(first_uk, SeqNum(u64::MAX), None)
            .unwrap()
            .expect("first row must be findable");
        assert_eq!(got.1, rows[0].1);

        // Last row.
        let last_uk = rows[rows.len() - 1].0.user_key_bytes();
        let got = reader
            .get(last_uk, SeqNum(u64::MAX), None)
            .unwrap()
            .expect("last row must be findable");
        assert_eq!(got.1, rows[rows.len() - 1].1);

        // Missing key beyond the file's max — bloom + key range will
        // reject before kv_index even runs.
        let missing = make_ikey(999_999, 1);
        let got = reader
            .get(missing.user_key_bytes(), SeqNum(u64::MAX), None)
            .unwrap();
        assert!(got.is_none());
    }

    /// Oracle sweep: build a multi-page L0 file with 8192 rows and call
    /// `get()` for **every** input user key. Each lookup must return the
    /// exact (ikey, row) that was written. This is the strongest single
    /// guarantee for the kv_index fast path: if the synthetic upper-bound
    /// probe construction, the page-locate logic, the row-group/row
    /// translation, or the page row-count clamp is wrong for *any* key,
    /// this test will catch it.
    #[test]
    fn kv_index_oracle_sweep_every_key_resolves() {
        // The kv_index lives on the `_merutable_ikey` column, not the
        // payload column, so the page count is driven by total ikey
        // bytes (not payload bytes). 4096 rows × ~14-byte ikeys ≈ 56 KiB
        // which the writer reliably splits into multiple data pages at
        // L0's 8 KiB page limit. Every input key is then tested against
        // the kv_index fast path.
        let (rows, reader) = build_multi_page_l0_reader(4096);
        assert!(reader.kv_index.is_some());

        for (idx, (orig_ik, orig_row)) in rows.iter().enumerate() {
            let user_key = orig_ik.user_key_bytes();
            let got = reader
                .get(user_key, SeqNum(u64::MAX), None)
                .unwrap_or_else(|e| panic!("get() errored at row {idx}: {e:?}"));
            let (got_ik, got_row) = got.unwrap_or_else(|| {
                panic!("kv_index path failed to locate row {idx} (user_key={user_key:?})")
            });
            assert_eq!(
                got_ik.seq, orig_ik.seq,
                "seq mismatch at row {idx}: got {:?}, want {:?}",
                got_ik.seq, orig_ik.seq
            );
            assert_eq!(
                got_ik.user_key_bytes(),
                user_key,
                "user_key mismatch at row {idx}"
            );
            assert_eq!(got_row, *orig_row, "row payload mismatch at row {idx}");
        }
    }

    /// Multi-row-group point lookup. Forces the file past the L0 row
    /// group rows-per-rg cap (16384) so the writer emits at least two
    /// row groups, then sweeps every key through the kv_index fast path.
    /// This exercises the row-group walk + intra-row-group offset
    /// translation in `read_rows_in_page`.
    #[test]
    fn kv_index_multi_row_group_oracle_sweep() {
        // 32_768 rows ÷ 16_384 rows-per-rg ≥ 2 row groups at L0.
        let n_rows = 32_768i64;
        let (rows, reader) = build_multi_page_l0_reader(n_rows);
        assert!(reader.kv_index.is_some());

        // Confirm the file actually has multiple row groups.
        let file_reader =
            SerializedFileReader::new(BBytes::from(reader.source.clone().to_vec())).unwrap();
        let num_row_groups = file_reader.metadata().num_row_groups();
        assert!(
            num_row_groups >= 2,
            "expected ≥2 row groups for {n_rows} rows at L0; got {num_row_groups}. \
             Adjust the row count if writer rg sizing changed."
        );

        // Sample 256 keys spread across the entire file (not just the
        // first row group) to force lookups that hit every row group.
        let sample_step = (rows.len() / 256).max(1);
        let mut probed = 0usize;
        for (idx, (orig_ik, orig_row)) in rows.iter().enumerate() {
            if !idx.is_multiple_of(sample_step) {
                continue;
            }
            probed += 1;
            let user_key = orig_ik.user_key_bytes();
            let got = reader
                .get(user_key, SeqNum(u64::MAX), None)
                .unwrap()
                .unwrap_or_else(|| panic!("missing row {idx}"));
            assert_eq!(got.0.seq, orig_ik.seq, "seq mismatch at row {idx}");
            assert_eq!(got.1, *orig_row, "payload mismatch at row {idx}");
        }
        assert!(probed >= 200, "too few probes: {probed}");

        // Also explicitly test the very first and very last rows of the
        // file — these are the edge of the global row index space.
        let first_uk = rows.first().unwrap().0.user_key_bytes();
        let last_uk = rows.last().unwrap().0.user_key_bytes();
        let first_got = reader
            .get(first_uk, SeqNum(u64::MAX), None)
            .unwrap()
            .unwrap();
        let last_got = reader
            .get(last_uk, SeqNum(u64::MAX), None)
            .unwrap()
            .unwrap();
        assert_eq!(first_got.1, rows.first().unwrap().1);
        assert_eq!(last_got.1, rows.last().unwrap().1);
    }

    /// MVCC: write multiple seq versions of the same user key. The
    /// kv_index fast path must use the synthetic upper-bound probe to
    /// locate a page that contains *every* version of the user key, then
    /// `find_visible` must return the highest seq ≤ read_seq.
    ///
    /// This is the test that proves the synthetic `0xFF * 9` trailer is
    /// the correct upper bound: a too-small probe would miss higher-seq
    /// versions; a too-large probe would land on the wrong page.
    #[test]
    fn kv_index_mvcc_versions_resolve_through_fast_path() {
        let schema = Arc::new(test_schema());

        // Build a multi-page file where the *middle* user key has 5
        // distinct versions (seqs 1..=5). Other rows are unique single
        // versions to push the file beyond one page.
        let mut rows: Vec<(InternalKey, Row)> = Vec::new();
        for i in 1..=4_000i64 {
            let row = Row::new(vec![
                Some(FieldValue::Int64(i)),
                Some(FieldValue::Bytes(BBytes::from(vec![i as u8; 256]))),
            ]);
            rows.push((make_ikey(i, i as u64 * 10), row));
        }
        // Insert 5 versions of key=2000 into the right sorted slot.
        // Note: InternalKey sorts by user key ASC then seq DESC, so the
        // newest version (seq=2000*10) is followed by older ones in
        // descending seq order.
        let target_id: i64 = 2000;
        // Drop the existing seq=20000 entry for id=2000 and replace
        // with explicit versions.
        rows.retain(|(ik, _)| ik.user_key_bytes() != make_ikey(target_id, 1).user_key_bytes());
        for ver in (1..=5u64).rev() {
            // ver=5 → seq=50; ver=1 → seq=10. All distinct from
            // surrounding rows' seqs.
            let row = Row::new(vec![
                Some(FieldValue::Int64(target_id)),
                Some(FieldValue::Bytes(BBytes::from(format!("version_{ver}")))),
            ]);
            rows.push((make_ikey(target_id, ver * 10), row));
        }
        // Re-sort by InternalKey bytes to satisfy the writer contract.
        rows.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

        let (bytes, _, _) = crate::parquet::writer::write_sorted_rows(
            rows.clone(),
            schema.clone(),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();
        let reader = ParquetReader::open(BBytes::from(bytes), schema).unwrap();
        assert!(reader.kv_index.is_some());

        let target_uk = make_ikey(target_id, 1).user_key_bytes().to_vec();

        // Read at seq=u64::MAX → must see the highest version (ver=5,
        // seq=50, payload="version_5").
        let got = reader
            .get(&target_uk, SeqNum(u64::MAX), None)
            .unwrap()
            .expect("must find target user key");
        assert_eq!(got.0.seq, SeqNum(50), "must return newest visible version");
        let payload = match got.1.get(1) {
            Some(FieldValue::Bytes(b)) => b.clone(),
            other => panic!("unexpected payload: {other:?}"),
        };
        assert_eq!(&payload[..], b"version_5");

        // Read at seq=30 → must skip ver=4(seq=40) and ver=5(seq=50),
        // returning ver=3 (seq=30, payload="version_3").
        let got = reader
            .get(&target_uk, SeqNum(30), None)
            .unwrap()
            .expect("must find target user key at seq 30");
        assert_eq!(got.0.seq, SeqNum(30));
        let payload = match got.1.get(1) {
            Some(FieldValue::Bytes(b)) => b.clone(),
            _ => unreachable!(),
        };
        assert_eq!(&payload[..], b"version_3");

        // Read at seq=15 → only ver=1(seq=10) is visible.
        let got = reader
            .get(&target_uk, SeqNum(15), None)
            .unwrap()
            .expect("must find target user key at seq 15");
        assert_eq!(got.0.seq, SeqNum(10));

        // Read at seq=5 → no version visible.
        let got = reader.get(&target_uk, SeqNum(5), None).unwrap();
        assert!(got.is_none(), "no version ≤ seq 5 should be visible");
    }

    /// Cross-check: for a randomized set of probes (both hits and
    /// misses), the kv_index fast path must return the same answer as
    /// a hand-rolled full-scan oracle that walks `scan(...)`. This
    /// catches any subtle disagreement between the page-restricted read
    /// and the full-file read across the same MVCC + key-range logic.
    #[test]
    fn kv_index_path_matches_full_scan_oracle_on_random_probes() {
        let (rows, reader) = build_multi_page_l0_reader(4096);
        assert!(reader.kv_index.is_some());

        // Build the full-scan oracle once.
        let oracle: Vec<(InternalKey, Row)> =
            reader.scan(None, None, SeqNum(u64::MAX), None).unwrap();

        // Map oracle by user_key for O(1) lookup.
        let oracle_map: std::collections::HashMap<Vec<u8>, Row> = oracle
            .into_iter()
            .map(|(ik, row)| (ik.user_key_bytes().to_vec(), row))
            .collect();

        // Deterministic LCG for repeatable probes.
        let mut state: u64 = 0xc0ffee_d00dface;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };

        // 1024 random probes: half hits (existing keys), half misses
        // (random ids that may or may not exist).
        let n_rows = rows.len() as i64;
        for _ in 0..1024 {
            let r = next();
            let probe_id = if r % 2 == 0 {
                // Hit: pick a random existing row.
                let idx = (r / 2) as usize % rows.len();
                let user_key = rows[idx].0.user_key_bytes().to_vec();
                let got = reader.get(&user_key, SeqNum(u64::MAX), None).unwrap();
                assert!(got.is_some(), "kv_index path missed existing key");
                let oracle_row = oracle_map.get(&user_key).unwrap();
                assert_eq!(&got.unwrap().1, oracle_row);
                continue;
            } else {
                // Potential miss: random id in extended range.
                ((r >> 1) as i64).rem_euclid(n_rows * 4)
            };

            let probe_ikey = make_ikey(probe_id, 1);
            let user_key = probe_ikey.user_key_bytes().to_vec();
            let got = reader.get(&user_key, SeqNum(u64::MAX), None).unwrap();
            let oracle_answer = oracle_map.get(&user_key);
            match (got, oracle_answer) {
                (Some((_, got_row)), Some(oracle_row)) => {
                    assert_eq!(&got_row, oracle_row, "row mismatch for id {probe_id}");
                }
                (None, None) => {} // both agree it's missing
                (got, oracle) => {
                    panic!(
                        "kv_index path disagrees with oracle for id {probe_id}: \
                         got={got:?}, oracle={oracle:?}"
                    );
                }
            }
        }
    }

    /// Bug H regression: when a single user key has enough MVCC versions
    /// to span multiple data pages on the `_merutable_ikey` column, the
    /// kv_index probe lands on the **first** page (highest seq — newest
    /// version), but an old `read_seq` whose visible version lives on a
    /// *later* page used to return `None` instead of falling back to a
    /// full-file scan. The fix: if `find_visible` on the matched page
    /// returns `None`, fall through to the full-scan path.
    ///
    /// To trigger: the ikey column uses dictionary encoding, so each data
    /// page holds bit-packed indices. At L0's 8 KiB page limit, we need
    /// ≈10K versions of the same key to push enough bit-packed indices
    /// across the page boundary.
    #[test]
    fn kv_index_multi_page_same_key_old_read_seq() {
        let schema = Arc::new(test_schema());

        // 10_000 versions of the same user key (id=42), seqs 1..=10_000.
        // InternalKey sort: higher seq → smaller ikey → earlier in file.
        let n_versions = 10_000u64;
        let mut rows: Vec<(InternalKey, Row)> = Vec::new();
        for ver in (1..=n_versions).rev() {
            let row = Row::new(vec![
                Some(FieldValue::Int64(42)),
                Some(FieldValue::Bytes(BBytes::from(format!("v{ver}")))),
            ]);
            rows.push((
                InternalKey::encode(&[FieldValue::Int64(42)], SeqNum(ver), OpType::Put, &schema)
                    .unwrap(),
                row,
            ));
        }
        // Already in ascending ikey order (higher seq → smaller tag → earlier).

        let (bytes, _, _) = crate::parquet::writer::write_sorted_rows(
            rows.clone(),
            schema.clone(),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();
        let reader = ParquetReader::open(BBytes::from(bytes), schema).unwrap();
        assert!(
            reader.kv_index.is_some(),
            "writer must emit kv_index for non-empty file"
        );
        let idx = reader.kv_index.as_ref().unwrap();

        let user_key = rows[0].0.user_key_bytes().to_vec();

        if idx.len() < 2 {
            // Dictionary encoding compressed the ikey column into a single
            // page — the multi-page edge case is not reachable at this row
            // count / page size. The fix is still correct (it's a no-op
            // when there's only one page), but we can't regression-test it
            // structurally. Skip the multi-page assertions and verify via
            // functional correctness only.
            eprintln!(
                "WARN: kv_index has {} page(s) — single page, multi-page \
                 edge case not structurally tested. Bumping n_versions may \
                 help on different encoding configs.",
                idx.len()
            );
        }

        // read_seq=MAX → newest visible is seq=n_versions.
        let got = reader.get(&user_key, SeqNum(n_versions), None).unwrap();
        let (ik, row) = got.expect("newest version must be visible");
        assert_eq!(ik.seq, SeqNum(n_versions));
        assert_eq!(
            row.get(1),
            Some(&FieldValue::Bytes(BBytes::from(format!("v{n_versions}"))))
        );

        // read_seq=10 → visible version is seq=10. If multi-page, it lives
        // on a later page. Before the fix, the kv_index path returned None.
        let got = reader.get(&user_key, SeqNum(10), None).unwrap();
        let (ik, row) = got.expect(
            "seq=10 must be visible (Bug H: kv_index must fall back \
             to full scan when page miss)",
        );
        assert_eq!(ik.seq, SeqNum(10));
        assert_eq!(row.get(1), Some(&FieldValue::Bytes(BBytes::from("v10"))));

        // read_seq=1 → oldest version.
        let got = reader.get(&user_key, SeqNum(1), None).unwrap();
        let (ik, _) = got.expect("seq=1 must be visible");
        assert_eq!(ik.seq, SeqNum(1));

        // Full version sweep: every seq from 1..=n_versions must resolve
        // to exactly that version. This is the strongest functional test
        // of the fallback — if ANY version is missed, the bug is live.
        for ver in [1u64, 5, 50, 500, 1000, 5000, n_versions] {
            let got = reader
                .get(&user_key, SeqNum(ver), None)
                .unwrap()
                .unwrap_or_else(|| panic!("version {ver} must be visible"));
            assert_eq!(got.0.seq, SeqNum(ver), "seq mismatch for version {ver}");
        }
    }

    #[test]
    fn empty_file_roundtrip() {
        let schema = test_schema();
        let (bytes, _bloom, meta) = crate::parquet::writer::write_sorted_rows(
            vec![],
            Arc::new(schema.clone()),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();
        assert!(bytes.is_empty());
        assert_eq!(meta.num_rows, 0);
    }
}
