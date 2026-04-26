//! `ParquetWriter`: writes a sorted stream of `(InternalKey, Row)` to a Parquet file.
//!
//! Output file is a valid Apache Parquet file that:
//! - Sorts rows by `InternalKey` (PK ASC, seq DESC) within each row group
//! - Stores bloom filter in KV metadata (`merutable.bloom`)
//! - Stores `ParquetFileMeta` + `TableSchema` in Parquet KV footer
//! - Uses row group sizing tuned to the target LSM level

use std::sync::Arc;

use crate::types::{
    MeruError, Result,
    key::InternalKey,
    level::{ColumnStats, FileFormat, Level, ParquetFileMeta},
    schema::{ColumnType, TableSchema},
    value::Row,
};
use bytes::Bytes;
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, Encoding},
    file::{
        properties::{WriterProperties, WriterPropertiesBuilder},
        reader::FileReader,
        serialized_reader::{ReadOptionsBuilder, SerializedFileReader},
    },
    schema::types::ColumnPath,
};

use crate::parquet::{
    bloom::FastLocalBloom,
    codec,
    kv_index::{self, PageLocation},
};

/// Assumed average row width in bytes. Used to convert byte budgets into
/// row counts for `set_max_row_group_size` (which takes rows, not bytes).
/// This is a rough heuristic; production tuning should measure actual widths.
const ASSUMED_ROW_BYTES: usize = 256;

/// Target row-group **byte** budget per LSM level.
///
/// The LSM tree treats L0 as the *hot* tier (just flushed from the memtable,
/// point-lookup heavy, write-amplification sensitive) and L2+ as the *cold*
/// tier (compacted, scan-heavy, analytics target). We therefore size row
/// groups *small* at the hot tier (row-store-like — favors selective reads
/// and reduces flush latency) and *large* at the cold tier (columnar
/// analytics — maximizes scan throughput and compression ratio).
pub fn target_row_group_bytes(level: Level) -> usize {
    match level.0 {
        0 => 4 * 1024 * 1024,   // 4 MiB  — hot
        1 => 32 * 1024 * 1024,  // 32 MiB — warm
        _ => 128 * 1024 * 1024, // 128 MiB — cold (analytics)
    }
}

/// Target data-page **byte** size per LSM level. Same hot→cold rationale
/// as `target_row_group_bytes`: small pages favor selective reads at L0,
/// large pages favor scan throughput at L2+.
///
/// ## Per-column page-size rationale
///
/// Parquet-rs (v53) does NOT support per-column `data_page_size_limit` —
/// the setting is global across all columns within a file. To approximate
/// per-column tuning:
///
/// - **L0** (rowstore): 8 KiB globally. All columns (`_merutable_ikey`,
///   `_merutable_value`, typed cols) use small pages for fast point lookups.
///   The `_merutable_ikey` and `_merutable_value` columns carry the KV
///   store's hot-path data; small pages minimize decompression per lookup.
///
/// - **L1+** (columnstore): 128 KiB globally for analytical scan throughput
///   on typed columns. The `_merutable_ikey` column uses PLAIN encoding
///   (set by `build_column_encoding_props`) so its pages remain direct-access
///   friendly even at the larger page size — PLAIN-encoded binary keys
///   decompress with zero overhead vs. dictionary/RLE.
///
/// The per-column ENCODING settings (`build_column_encoding_props`) provide
/// the differentiation that per-column page sizes cannot:
///   - `_merutable_ikey`: PLAIN — O(1) decode per key, ideal for lookups
///   - `_merutable_value` (L0 only): PLAIN — opaque postcard blobs
///   - Int32/Int64 typed cols: DELTA_BINARY_PACKED — optimal for sorted ints
///   - Float/Double typed cols: BYTE_STREAM_SPLIT — IEEE 754 byte-transposition
///   - ByteArray typed cols: RLE_DICTIONARY — high compression for strings
pub fn target_data_page_bytes(level: Level) -> usize {
    match level.0 {
        0 => 8 * 1024,   // 8 KiB   — hot, OS-page-aligned, point-lookup tier
        1 => 32 * 1024,  // 32 KiB  — warm
        _ => 128 * 1024, // 128 KiB — cold (analytics-friendly without 1 MiB bloat)
    }
}

/// Apply per-column encoding + dictionary settings to a `WriterPropertiesBuilder`.
///
/// At L0 (rowstore): all columns use PLAIN encoding — no dictionary overhead,
/// fastest decode for point lookups via `_merutable_ikey` and `_merutable_value`.
///
/// At L1+ (columnstore): `_merutable_ikey` stays PLAIN (key-lookup column),
/// typed columns get analytics-optimized encodings.
fn build_column_encoding_props(
    mut builder: WriterPropertiesBuilder,
    schema: &TableSchema,
    format: FileFormat,
) -> WriterPropertiesBuilder {
    let ikey_col = ColumnPath::new(vec!["_merutable_ikey".to_string()]);
    builder = builder
        .set_column_encoding(ikey_col.clone(), Encoding::PLAIN)
        .set_column_dictionary_enabled(ikey_col, false);

    // Issue #16: _merutable_seq is monotonic-ish (flush writes a
    // contiguous range; compaction writes a shuffled but still
    // delta-friendly range). DELTA_BINARY_PACKED is near-optimal.
    let seq_col = ColumnPath::new(vec![crate::parquet::codec::SEQ_COLUMN_NAME.to_string()]);
    builder = builder
        .set_column_encoding(seq_col.clone(), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(seq_col, false);

    // Issue #16: _merutable_op is effectively a two-value enum
    // (Put=1, Delete=0) dominated by long runs of Put. Parquet's RLE
    // physical encoding only supports Boolean data, so for this Int32
    // column we enable dictionary encoding — parquet-rs will emit
    // RLE_DICTIONARY on the dictionary-index stream, which achieves
    // the same collapse (2-entry dict + RLE-encoded indices).
    let op_col = ColumnPath::new(vec![crate::parquet::codec::OP_COLUMN_NAME.to_string()]);
    builder = builder.set_column_dictionary_enabled(op_col, true);

    if format.has_value_blob() {
        // Dual format: postcard value blob — PLAIN, no dictionary
        let value_col = ColumnPath::new(vec!["_merutable_value".to_string()]);
        builder = builder
            .set_column_encoding(value_col.clone(), Encoding::PLAIN)
            .set_column_dictionary_enabled(value_col, false);

        // Dual typed columns: PLAIN for rowstore fast decode
        for col_def in &schema.columns {
            let col_path = ColumnPath::new(vec![col_def.name.clone()]);
            builder = builder
                .set_column_encoding(col_path.clone(), Encoding::PLAIN)
                .set_column_dictionary_enabled(col_path, false);
        }
    } else {
        // L1+: typed columns get analytics-optimized encodings
        for col_def in &schema.columns {
            let col_path = ColumnPath::new(vec![col_def.name.clone()]);
            match col_def.col_type {
                ColumnType::Int32 | ColumnType::Int64 => {
                    builder = builder
                        .set_column_encoding(col_path.clone(), Encoding::DELTA_BINARY_PACKED)
                        .set_column_dictionary_enabled(col_path, false);
                }
                ColumnType::Float | ColumnType::Double => {
                    builder = builder
                        .set_column_encoding(col_path.clone(), Encoding::BYTE_STREAM_SPLIT)
                        .set_column_dictionary_enabled(col_path, false);
                }
                ColumnType::ByteArray => {
                    // RLE_DICTIONARY is parquet-rs's default and optimal for
                    // string/categorical columns; just make sure it's enabled.
                    builder = builder.set_column_dictionary_enabled(col_path, true);
                }
                ColumnType::Boolean => {
                    builder = builder.set_column_encoding(col_path, Encoding::RLE);
                }
                ColumnType::FixedLenByteArray(_) => {
                    builder = builder
                        .set_column_encoding(col_path.clone(), Encoding::PLAIN)
                        .set_column_dictionary_enabled(col_path, false);
                }
            }
        }
    }
    builder
}

/// Row count to pass to `set_max_row_group_size` for a given level,
/// derived from `target_row_group_bytes` and `ASSUMED_ROW_BYTES`. Floored
/// at 1024 rows so very narrow heuristics never produce a degenerate group.
pub fn target_rows_per_row_group(level: Level) -> usize {
    (target_row_group_bytes(level) / ASSUMED_ROW_BYTES).max(1024)
}

pub struct WriterStats {
    pub num_rows: u64,
    pub file_size: u64,
    pub key_min: Vec<u8>,
    pub key_max: Vec<u8>,
    pub seq_min: u64,
    pub seq_max: u64,
}

/// Convenience: write all rows at once into a `Vec<u8>` buffer.
/// Returns `(parquet_bytes, bloom_bytes, meta)`.
pub fn write_sorted_rows(
    rows: Vec<(InternalKey, Row)>,
    schema: Arc<TableSchema>,
    level: Level,
    format: FileFormat,
    bloom_bits_per_key: u8,
) -> Result<(Vec<u8>, Bytes, ParquetFileMeta)> {
    if rows.is_empty() {
        let meta = ParquetFileMeta {
            level,
            seq_min: 0,
            seq_max: 0,
            key_min: Vec::new(),
            key_max: Vec::new(),
            num_rows: 0,
            file_size: 0,
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            format: Some(format),
            column_stats: None,
        };
        return Ok((Vec::new(), Bytes::new(), meta));
    }

    let estimated = rows.len();
    let arrow_schema = codec::arrow_schema(&schema, format);

    // Track stats.
    let mut bloom = FastLocalBloom::new(estimated.max(1000), bloom_bits_per_key);
    let mut key_min: Option<Vec<u8>> = None;
    let mut key_max: Option<Vec<u8>> = None;
    let mut seq_min = u64::MAX;
    let mut seq_max = 0u64;

    for (ikey, _) in &rows {
        let uk = ikey.user_key_bytes().to_vec();
        bloom.add(&uk);
        if key_min.is_none() {
            key_min = Some(uk.clone());
        }
        key_max = Some(uk);
        if ikey.seq.0 < seq_min {
            seq_min = ikey.seq.0;
        }
        if ikey.seq.0 > seq_max {
            seq_max = ikey.seq.0;
        }
    }

    // Build KV metadata.
    let meta = ParquetFileMeta {
        level,
        seq_min: if seq_min == u64::MAX { 0 } else { seq_min },
        seq_max,
        key_min: key_min.unwrap_or_default(),
        key_max: key_max.unwrap_or_default(),
        num_rows: rows.len() as u64,
        file_size: 0, // filled after writing
        dv_path: None,
        dv_offset: None,
        dv_length: None,
        format: Some(format),
        column_stats: None, // filled after extract_column_stats below
    };

    let bloom_bytes = bloom.to_bytes();
    let footer_kv = crate::parquet::footer::encode_footer_kv(&meta, &schema)?;

    // Base KV: file meta + table schema + bloom (no kv_index yet — that
    // requires knowing page offsets, which only exist after a write).
    let mut base_kv: Vec<(String, String)> = footer_kv;
    base_kv.push(("merutable.bloom".to_string(), hex::encode(&bloom_bytes)));

    // Pass 1: write the file with base KV. Pages land at their final byte
    // offsets in the data section, which is invariant across passes — only
    // the trailing footer KV changes.
    let pass1_bytes = arrow_write_pass(&rows, &arrow_schema, &schema, level, format, &base_kv)?;

    // Inspect pass-1's OffsetIndex to learn page boundaries on the
    // `_merutable_ikey` column, then build a `(first_key_on_page → location)`
    // sparse index over those boundaries.
    let kv_index_entries = extract_kv_index_entries(&rows, &pass1_bytes)?;
    let kv_index_bytes = kv_index::build(&kv_index_entries, kv_index::DEFAULT_RESTART_INTERVAL)?;

    // Pass 2: same input, same properties, plus the kv_index footer KV.
    // Determinism of `ArrowWriter` guarantees identical page layout and
    // thus identical offsets to those captured in pass 1.
    let mut full_kv = base_kv;
    full_kv.push((
        kv_index::KV_INDEX_FOOTER_KEY.to_string(),
        hex::encode(&kv_index_bytes),
    ));
    let pass2_bytes = arrow_write_pass(&rows, &arrow_schema, &schema, level, format, &full_kv)?;

    let mut final_meta = meta;
    final_meta.file_size = pass2_bytes.len() as u64;
    // Issue #20 Part 2b: hoist per-column stats from the final
    // file's row-group metadata. We reduce across row groups so the
    // file-level stats are usable by external readers without having
    // to walk every row group themselves.
    final_meta.column_stats = extract_column_stats(&pass2_bytes, &schema).ok();

    Ok((pass2_bytes, bloom_bytes, final_meta))
}

/// Issue #20 Part 2b: reduce Parquet row-group statistics to per-file
/// per-column stats for every user column. Hidden `_merutable_*`
/// columns are skipped because they're not part of the Iceberg schema
/// and external readers should not reference them directly.
///
/// Errors here are non-fatal — the caller falls back to `None`, which
/// projects to empty Iceberg stat maps (spec-valid).
fn extract_column_stats(bytes: &[u8], schema: &TableSchema) -> Result<Vec<ColumnStats>> {
    let reader = SerializedFileReader::new(Bytes::copy_from_slice(bytes))
        .map_err(|e| MeruError::Parquet(e.to_string()))?;
    let file_meta = reader.metadata();
    let parquet_schema = file_meta.file_metadata().schema_descr();

    // Map Iceberg field_id → Parquet column index. The Iceberg schema
    // is `schema.columns` in order; the Parquet schema adds hidden
    // merutable columns. Look up each user column by name.
    let mut name_to_col: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for i in 0..parquet_schema.num_columns() {
        let name = parquet_schema.column(i).name().to_string();
        name_to_col.insert(name, i);
    }

    // Per-column accumulators keyed by field_id (1-based).
    struct Acc {
        field_id: i32,
        col_type: ColumnType,
        compressed_bytes: u64,
        value_count: u64,
        null_count: u64,
        min_bytes: Option<Vec<u8>>,
        max_bytes: Option<Vec<u8>>,
    }
    let mut accs: Vec<(usize, Acc)> = Vec::new();
    for (idx, col_def) in schema.columns.iter().enumerate() {
        let field_id = (idx + 1) as i32;
        if let Some(&parquet_col_idx) = name_to_col.get(&col_def.name) {
            accs.push((
                parquet_col_idx,
                Acc {
                    field_id,
                    col_type: col_def.col_type.clone(),
                    compressed_bytes: 0,
                    value_count: 0,
                    null_count: 0,
                    min_bytes: None,
                    max_bytes: None,
                },
            ));
        }
    }

    // Walk every row group and reduce per column.
    for rg_idx in 0..file_meta.num_row_groups() {
        let rg = file_meta.row_group(rg_idx);
        for (parquet_col_idx, acc) in accs.iter_mut() {
            let chunk = rg.column(*parquet_col_idx);
            acc.compressed_bytes = acc
                .compressed_bytes
                .saturating_add(chunk.compressed_size().max(0) as u64);
            if let Some(stats) = chunk.statistics() {
                // Parquet Statistics carries `null_count_opt()` and the
                // typed min/max. Value count = total rows in row group
                // minus null count (the chunk reports row-group row
                // count).
                let rg_rows = rg.num_rows().max(0) as u64;
                let null_count = stats.null_count_opt().unwrap_or(0);
                let values = rg_rows.saturating_sub(null_count);
                acc.value_count = acc.value_count.saturating_add(values);
                acc.null_count = acc.null_count.saturating_add(null_count);

                // Reduce min/max using Iceberg single-value byte
                // encoding. Parquet's numeric statistics already use
                // the little-endian on-disk form we want.
                use parquet::file::statistics::Statistics as PqStats;
                let (min_b, max_b): (Option<Vec<u8>>, Option<Vec<u8>>) = match stats {
                    PqStats::Boolean(s) => {
                        let b2b = |b: &bool| vec![if *b { 1u8 } else { 0u8 }];
                        (s.min_opt().map(b2b), s.max_opt().map(b2b))
                    }
                    PqStats::Int32(s) => (
                        s.min_opt().map(|v| v.to_le_bytes().to_vec()),
                        s.max_opt().map(|v| v.to_le_bytes().to_vec()),
                    ),
                    PqStats::Int64(s) => (
                        s.min_opt().map(|v| v.to_le_bytes().to_vec()),
                        s.max_opt().map(|v| v.to_le_bytes().to_vec()),
                    ),
                    PqStats::Float(s) => (
                        s.min_opt().map(|v| v.to_le_bytes().to_vec()),
                        s.max_opt().map(|v| v.to_le_bytes().to_vec()),
                    ),
                    PqStats::Double(s) => (
                        s.min_opt().map(|v| v.to_le_bytes().to_vec()),
                        s.max_opt().map(|v| v.to_le_bytes().to_vec()),
                    ),
                    PqStats::ByteArray(s) => (
                        s.min_opt().map(|v| v.data().to_vec()),
                        s.max_opt().map(|v| v.data().to_vec()),
                    ),
                    PqStats::FixedLenByteArray(s) => (
                        s.min_opt().map(|v| v.data().to_vec()),
                        s.max_opt().map(|v| v.data().to_vec()),
                    ),
                    _ => (None, None),
                };
                match (&mut acc.min_bytes, min_b) {
                    (slot @ None, Some(v)) => *slot = Some(v),
                    (Some(cur), Some(v)) if bound_is_less(&acc.col_type, &v, cur) => {
                        *cur = v;
                    }
                    _ => {}
                }
                match (&mut acc.max_bytes, max_b) {
                    (slot @ None, Some(v)) => *slot = Some(v),
                    (Some(cur), Some(v)) if bound_is_less(&acc.col_type, cur, &v) => {
                        *cur = v;
                    }
                    _ => {}
                }
            } else {
                // No statistics on this row group → we can still
                // count values (chunk row count minus whatever we
                // know; absent stats means we don't know null count
                // either, so leave the counters alone).
            }
        }
    }

    Ok(accs
        .into_iter()
        .map(|(_, a)| ColumnStats {
            field_id: a.field_id,
            compressed_bytes: a.compressed_bytes,
            value_count: a.value_count,
            null_count: a.null_count,
            lower_bound: a.min_bytes,
            upper_bound: a.max_bytes,
        })
        .collect())
}

/// Iceberg single-value byte comparison for reducing min/max across
/// row groups. Numeric bounds are LE-encoded; raw byte comparison
/// would give wrong answers for signed ints. Booleans and raw bytes
/// compare lexicographically, which coincides with value order.
fn bound_is_less(col_type: &ColumnType, a: &[u8], b: &[u8]) -> bool {
    match col_type {
        ColumnType::Int32 => {
            let a = i32::from_le_bytes(a.try_into().unwrap_or([0u8; 4]));
            let b = i32::from_le_bytes(b.try_into().unwrap_or([0u8; 4]));
            a < b
        }
        ColumnType::Int64 => {
            let a = i64::from_le_bytes(a.try_into().unwrap_or([0u8; 8]));
            let b = i64::from_le_bytes(b.try_into().unwrap_or([0u8; 8]));
            a < b
        }
        ColumnType::Float => {
            let a = f32::from_le_bytes(a.try_into().unwrap_or([0u8; 4]));
            let b = f32::from_le_bytes(b.try_into().unwrap_or([0u8; 4]));
            a < b
        }
        ColumnType::Double => {
            let a = f64::from_le_bytes(a.try_into().unwrap_or([0u8; 8]));
            let b = f64::from_le_bytes(b.try_into().unwrap_or([0u8; 8]));
            a < b
        }
        ColumnType::Boolean | ColumnType::ByteArray | ColumnType::FixedLenByteArray(_) => a < b,
    }
}

/// Single Parquet write of `rows` with the given footer KV. Used twice by
/// `write_sorted_rows`: first to discover page offsets, then to embed the
/// `kv_index` that references them.
///
/// KV pairs are passed via `WriterProperties::set_key_value_metadata`,
/// which writes them into the Parquet thrift `FileMetaData.key_value_metadata`
/// section. (Stuffing them on the Arrow schema metadata does *not* propagate
/// them to the Parquet footer — that path only round-trips through
/// `ARROW:schema`, which is opaque to non-Arrow Parquet readers.)
fn arrow_write_pass(
    rows: &[(InternalKey, Row)],
    arrow_schema: &Arc<arrow::datatypes::Schema>,
    schema: &TableSchema,
    level: Level,
    format: FileFormat,
    kv: &[(String, String)],
) -> Result<Vec<u8>> {
    let kv_meta: Vec<parquet::format::KeyValue> = kv
        .iter()
        .map(|(k, v)| parquet::format::KeyValue {
            key: k.clone(),
            value: Some(v.clone()),
        })
        .collect();

    let builder = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_size(target_rows_per_row_group(level))
        .set_data_page_size_limit(target_data_page_bytes(level))
        .set_key_value_metadata(Some(kv_meta));
    let builder = build_column_encoding_props(builder, schema, format);
    let props = builder.build();

    let buf: Vec<u8> = Vec::new();
    let mut writer = ArrowWriter::try_new(buf, arrow_schema.clone(), Some(props))
        .map_err(|e| MeruError::Parquet(e.to_string()))?;

    let batch = codec::rows_to_record_batch(rows, schema, format)?;
    writer
        .write(&batch)
        .map_err(|e| MeruError::Parquet(e.to_string()))?;

    writer
        .into_inner()
        .map_err(|e| MeruError::Parquet(e.to_string()))
}

/// Walk pass-1's `OffsetIndex` for the `_merutable_ikey` column and emit
/// one `kv_index` entry per data page: the page's first key (full encoded
/// `InternalKey`) plus its absolute file offset, compressed size, and
/// global row index.
///
/// Global row indices are computed by accumulating the row count of each
/// preceding row group, since `OffsetIndex.first_row_index` is row-group
/// local.
fn extract_kv_index_entries(
    rows: &[(InternalKey, Row)],
    pass1_bytes: &[u8],
) -> Result<Vec<(Vec<u8>, PageLocation)>> {
    let bytes = Bytes::copy_from_slice(pass1_bytes);
    let opts = ReadOptionsBuilder::new().with_page_index().build();
    let reader = SerializedFileReader::new_with_options(bytes, opts)
        .map_err(|e| MeruError::Parquet(e.to_string()))?;
    let metadata = reader.metadata();

    let offset_index = metadata.offset_index().ok_or_else(|| {
        MeruError::Parquet(
            "OffsetIndex missing from pass-1 Parquet file (expected ArrowWriter to emit it)".into(),
        )
    })?;

    // The `_merutable_ikey` column is always column 0 in both L0 and L1+
    // schemas (see `codec::arrow_schema`). At L0 the `_merutable_value`
    // postcard blob column is inserted *after* it, never before, so this
    // index is invariant across the per-level schema shape.
    const IKEY_COL_IDX: usize = 0;

    let mut entries: Vec<(Vec<u8>, PageLocation)> = Vec::new();
    let mut row_group_start: u64 = 0;

    for (rg_idx, rg_offset_indexes) in offset_index.iter().enumerate() {
        let ikey_offsets = rg_offset_indexes.get(IKEY_COL_IDX).ok_or_else(|| {
            MeruError::Parquet(format!(
                "OffsetIndex missing _merutable_ikey column for row group {rg_idx}"
            ))
        })?;

        for page_loc in ikey_offsets.page_locations() {
            // Bug P1-P4 fix: validate sign of Parquet thrift i64/i32 fields
            // before casting to u64/u32. Same class as Bug L/M — negative
            // values from corrupt files wrap to huge unsigned values.
            if page_loc.first_row_index < 0 {
                return Err(MeruError::Corruption(format!(
                    "negative first_row_index {} in row group {rg_idx}",
                    page_loc.first_row_index
                )));
            }
            if page_loc.offset < 0 {
                return Err(MeruError::Corruption(format!(
                    "negative page offset {} in row group {rg_idx}",
                    page_loc.offset
                )));
            }
            if page_loc.compressed_page_size < 0 {
                return Err(MeruError::Corruption(format!(
                    "negative compressed_page_size {} in row group {rg_idx}",
                    page_loc.compressed_page_size
                )));
            }
            let global_first_row = row_group_start + page_loc.first_row_index as u64;
            let row = rows.get(global_first_row as usize).ok_or_else(|| {
                MeruError::Parquet(format!(
                    "OffsetIndex first_row_index {global_first_row} \
                     out of bounds for input row count {}",
                    rows.len()
                ))
            })?;
            entries.push((
                row.0.as_bytes().to_vec(),
                PageLocation {
                    page_offset: page_loc.offset as u64,
                    page_size: page_loc.compressed_page_size as u32,
                    first_row_index: global_first_row,
                },
            ));
        }

        let rg_num_rows = metadata.row_group(rg_idx).num_rows();
        if rg_num_rows < 0 {
            return Err(MeruError::Corruption(format!(
                "negative num_rows {} in row group {rg_idx}",
                rg_num_rows
            )));
        }
        let rg_num_rows = rg_num_rows as u64;
        row_group_start += rg_num_rows;
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L0 is the *hot* tier (just flushed, point-lookup heavy). L2+ is the
    /// *cold* analytics tier. Row-group byte budget must grow with level so
    /// that hot data behaves like a row store and cold data like a columnar
    /// analytics store. A regression here silently destroys read latency at
    /// L0 or scan throughput at L2+.
    #[test]
    fn row_group_bytes_grow_from_hot_to_cold() {
        let l0 = target_row_group_bytes(Level(0));
        let l1 = target_row_group_bytes(Level(1));
        let l2 = target_row_group_bytes(Level(2));
        let l3 = target_row_group_bytes(Level(3));
        assert!(l0 < l1, "L0 ({l0}) must be smaller than L1 ({l1})");
        assert!(l1 < l2, "L1 ({l1}) must be smaller than L2 ({l2})");
        assert_eq!(l2, l3, "L2+ should plateau at the cold-tier size");
    }

    /// Same hot→cold rationale for data pages: small pages at L0 favor
    /// selective decoding for point lookups; large pages at L2+ favor
    /// scan throughput and compression.
    #[test]
    fn data_page_bytes_grow_from_hot_to_cold() {
        let l0 = target_data_page_bytes(Level(0));
        let l1 = target_data_page_bytes(Level(1));
        let l2 = target_data_page_bytes(Level(2));
        let l3 = target_data_page_bytes(Level(3));
        assert!(l0 < l1, "L0 ({l0}) must be smaller than L1 ({l1})");
        assert!(l1 < l2, "L1 ({l1}) must be smaller than L2 ({l2})");
        assert_eq!(l2, l3, "L2+ should plateau at the cold-tier size");
    }

    /// The cold tier should be *significantly* larger than the hot tier —
    /// not just nominally larger. If the ratio collapses below 8x, the
    /// external analytics tradeoff has been weakened and the level differentiation is
    /// no longer meaningful.
    #[test]
    fn cold_tier_significantly_larger_than_hot() {
        let rg_ratio = target_row_group_bytes(Level(2)) / target_row_group_bytes(Level(0));
        let pg_ratio = target_data_page_bytes(Level(2)) / target_data_page_bytes(Level(0));
        assert!(
            rg_ratio >= 8,
            "row-group cold/hot ratio {rg_ratio}x is too small (need ≥8x)"
        );
        assert!(
            pg_ratio >= 8,
            "data-page cold/hot ratio {pg_ratio}x is too small (need ≥8x)"
        );
    }

    /// L0 specifically must stay row-store-sized so flushes are fast and
    /// point lookups are cheap. Anchor the absolute upper bound so that
    /// future "tuning" doesn't drift L0 back into analytics territory.
    /// The hot-tier page must align to (or be a small multiple of) the
    /// 4 KiB OS page; we land at exactly 8 KiB.
    #[test]
    fn l0_stays_row_store_sized() {
        assert!(
            target_row_group_bytes(Level(0)) <= 8 * 1024 * 1024,
            "L0 row group must stay ≤ 8 MiB to behave like a row store"
        );
        assert!(
            target_data_page_bytes(Level(0)) <= 16 * 1024,
            "L0 data page must stay ≤ 16 KiB to behave like a row store"
        );
    }

    /// L2+ must remain analytics-friendly *enough*: large enough that
    /// snappy/dictionary compression on composite-PK data has working set,
    /// I/O is amortized over object-store request overhead, and per-page
    /// header overhead is negligible. We anchor the floor at 64 KiB —
    /// well above the compression-effectiveness break-even — and the row
    /// group floor at 64 MiB. This explicitly forbids regressing back to
    /// the L0 hot-tier shape.
    #[test]
    fn cold_tier_stays_analytics_sized() {
        assert!(
            target_row_group_bytes(Level(2)) >= 64 * 1024 * 1024,
            "Cold tier row group must stay ≥ 64 MiB for analytics scans"
        );
        assert!(
            target_data_page_bytes(Level(2)) >= 64 * 1024,
            "Cold tier data page must stay ≥ 64 KiB for compression + I/O amortization"
        );
    }

    /// `set_max_row_group_size` takes a row count, not a byte count. The
    /// converted row count must be both nonzero and at least the floor
    /// (1024 rows), and must still grow monotonically with level.
    #[test]
    fn rows_per_row_group_monotonic_and_floored() {
        let r0 = target_rows_per_row_group(Level(0));
        let r1 = target_rows_per_row_group(Level(1));
        let r2 = target_rows_per_row_group(Level(2));
        assert!(r0 >= 1024, "row floor violated: {r0}");
        assert!(r0 < r1, "L0 rows ({r0}) must be < L1 rows ({r1})");
        assert!(r1 < r2, "L1 rows ({r1}) must be < L2 rows ({r2})");
    }

    // ── kv_index integration tests ──────────────────────────────────────

    use crate::types::{
        schema::{ColumnDef, ColumnType, TableSchema},
        sequence::{OpType, SeqNum},
        value::{FieldValue, Row},
    };
    use bytes::Bytes as BBytes;

    use crate::parquet::kv_index::{KV_INDEX_FOOTER_KEY, KvSparseIndex};

    fn kv_index_test_schema() -> TableSchema {
        TableSchema {
            table_name: "kv_index_test".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,

                    ..Default::default()
                },
                ColumnDef {
                    name: "payload".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: false,

                    ..Default::default()
                },
            ],
            primary_key: vec![0],

            ..Default::default()
        }
    }

    fn make_test_rows(n: usize, schema: &TableSchema) -> Vec<(InternalKey, Row)> {
        // Per-row unique payload defeats Parquet's dictionary encoding so
        // that the configured page size limit (8 KiB at L0) is what
        // actually drives page boundaries — without this, identical
        // payloads dictionary-compress to a single page.
        (0..n as i64)
            .map(|i| {
                let ikey = InternalKey::encode(
                    &[FieldValue::Int64(i)],
                    SeqNum(i as u64 + 1),
                    OpType::Put,
                    schema,
                )
                .unwrap();
                // 256-byte payload, unique per row.
                let mut payload_bytes = vec![0u8; 256];
                let stamp = i.to_le_bytes();
                for chunk in payload_bytes.chunks_mut(8) {
                    let n = chunk.len().min(8);
                    chunk[..n].copy_from_slice(&stamp[..n]);
                }
                let row = Row::new(vec![
                    Some(FieldValue::Int64(i)),
                    Some(FieldValue::Bytes(BBytes::from(payload_bytes))),
                ]);
                (ikey, row)
            })
            .collect()
    }

    /// Read the `merutable.kv_index.v1` footer KV out of a written file.
    /// Returns `None` if the writer didn't emit one (which would indicate
    /// the two-pass integration regressed).
    fn read_kv_index_from_file(file_bytes: &[u8]) -> Option<KvSparseIndex> {
        let bytes = BBytes::copy_from_slice(file_bytes);
        let reader = SerializedFileReader::new(bytes).ok()?;
        let kv = reader.metadata().file_metadata().key_value_metadata()?;
        let entry = kv.iter().find(|e| e.key == KV_INDEX_FOOTER_KEY)?;
        let hex_str = entry.value.as_ref()?;
        let raw = hex::decode(hex_str).ok()?;
        KvSparseIndex::from_bytes(BBytes::from(raw)).ok()
    }

    /// Writer must emit a kv_index footer KV, and decoding it must yield
    /// at least one entry per data page on the `_merutable_ikey` column.
    #[test]
    fn writer_emits_kv_index_in_footer() {
        let schema = kv_index_test_schema();
        let rows = make_test_rows(16_384, &schema);
        let (file_bytes, _bloom, _meta) = write_sorted_rows(
            rows.clone(),
            Arc::new(schema),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();

        let idx =
            read_kv_index_from_file(&file_bytes).expect("writer must emit merutable.kv_index.v1");
        assert!(
            !idx.is_empty(),
            "kv_index must contain at least one entry for a non-empty file"
        );
        // Multi-page sanity: 16k × 256B payload ≫ 8 KiB page limit so the
        // _merutable_ikey column must page-split at least a few times.
        // We deliberately don't assert a tight bound — Parquet's page
        // sizing is "best effort" based on `write_batch_size` checks and
        // depends on column compressibility, so we just demand "more
        // than one page" as the regression guard.
        assert!(
            idx.len() >= 2,
            "expected ≥2 page entries for 16k rows at L0; got {}",
            idx.len()
        );
    }

    /// Every input row's encoded `InternalKey` must resolve via the
    /// kv_index to a page whose first key is ≤ the probe — i.e., the
    /// predecessor-search invariant holds against the actual written
    /// page boundaries. The `first_row_index` must also fall within the
    /// input row count.
    #[test]
    fn kv_index_predecessor_holds_for_every_input_key() {
        let schema = kv_index_test_schema();
        let rows = make_test_rows(16_384, &schema);
        let (file_bytes, _bloom, _meta) = write_sorted_rows(
            rows.clone(),
            Arc::new(schema),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();

        let idx = read_kv_index_from_file(&file_bytes).unwrap();

        // Collect entries to verify each probe lands on a real page entry.
        let entries: Vec<_> = idx.iter().collect();
        assert!(!entries.is_empty());

        for (row_idx, (ikey, _)) in rows.iter().enumerate() {
            let probe = ikey.as_bytes();
            let loc = idx.find_page(probe).unwrap_or_else(|| {
                panic!("kv_index find_page returned None for input row {row_idx}")
            });
            assert!(
                (loc.first_row_index as usize) <= row_idx,
                "page first_row_index {} must be ≤ probe row {}",
                loc.first_row_index,
                row_idx
            );
            assert!(
                (loc.first_row_index as usize) < rows.len(),
                "page first_row_index {} out of bounds for row count {}",
                loc.first_row_index,
                rows.len()
            );
        }
    }

    /// `kv_index` entries must be in strictly ascending key order — that's
    /// the invariant the binary search and prefix coding both rely on.
    /// Pages on a sorted `_merutable_ikey` column should naturally satisfy
    /// this; this test pins the contract so a future writer change can't
    /// silently break the sort assumption.
    #[test]
    fn kv_index_entries_are_strictly_ascending() {
        let schema = kv_index_test_schema();
        let rows = make_test_rows(1500, &schema);
        let (file_bytes, _bloom, _meta) = write_sorted_rows(
            rows,
            Arc::new(schema),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();

        let idx = read_kv_index_from_file(&file_bytes).unwrap();
        let mut prev: Option<Vec<u8>> = None;
        for (k, _) in idx.iter() {
            if let Some(ref p) = prev {
                assert!(
                    k.as_slice() > p.as_slice(),
                    "kv_index entries not strictly ascending: {p:?} ≮ {k:?}"
                );
            }
            prev = Some(k);
        }
    }

    /// L0 files must carry the `_merutable_value` postcard blob column so
    /// hot-tier point lookups can decode an entire row from a single
    /// column-chunk read; L1+ files must NOT carry it (cold tier is pure
    /// typed-column analytics shape, redundant blobs would inflate scans).
    /// This pins the per-level schema contract end-to-end through the
    /// writer.
    #[test]
    fn l0_has_value_blob_column_l1_does_not() {
        let schema = kv_index_test_schema();
        let rows = make_test_rows(64, &schema);

        let (l0_bytes, _, _) = write_sorted_rows(
            rows.clone(),
            Arc::new(schema.clone()),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();
        let (l1_bytes, _, _) = write_sorted_rows(
            rows,
            Arc::new(schema),
            Level(1),
            crate::types::level::FileFormat::Columnar,
            10,
        )
        .unwrap();

        let l0_reader = SerializedFileReader::new(BBytes::from(l0_bytes)).unwrap();
        let l1_reader = SerializedFileReader::new(BBytes::from(l1_bytes)).unwrap();

        let l0_descr = l0_reader.metadata().file_metadata().schema_descr_ptr();
        let l1_descr = l1_reader.metadata().file_metadata().schema_descr_ptr();
        let l0_cols: Vec<String> = (0..l0_descr.num_columns())
            .map(|i| l0_descr.column(i).name().to_string())
            .collect();
        let l1_cols: Vec<String> = (0..l1_descr.num_columns())
            .map(|i| l1_descr.column(i).name().to_string())
            .collect();

        assert!(
            l0_cols
                .iter()
                .any(|c| c == crate::parquet::codec::IKEY_COLUMN_NAME),
            "L0 must contain ikey column; got {l0_cols:?}"
        );
        assert!(
            l0_cols
                .iter()
                .any(|c| c == crate::parquet::codec::VALUE_BLOB_COLUMN_NAME),
            "L0 must contain value blob column; got {l0_cols:?}"
        );
        assert!(
            l1_cols
                .iter()
                .any(|c| c == crate::parquet::codec::IKEY_COLUMN_NAME),
            "L1 must contain ikey column; got {l1_cols:?}"
        );
        assert!(
            !l1_cols
                .iter()
                .any(|c| c == crate::parquet::codec::VALUE_BLOB_COLUMN_NAME),
            "L1 must NOT contain value blob column; got {l1_cols:?}"
        );
        // Both levels must expose user-defined typed columns for external analytics visibility.
        assert!(l0_cols.iter().any(|c| c == "id") && l0_cols.iter().any(|c| c == "payload"));
        assert!(l1_cols.iter().any(|c| c == "id") && l1_cols.iter().any(|c| c == "payload"));
    }

    /// The empty-input fast path must NOT emit a kv_index entry — there
    /// are no pages to index, and the writer short-circuits before any
    /// Parquet bytes are produced.
    #[test]
    fn empty_input_emits_no_kv_index() {
        let schema = kv_index_test_schema();
        let (file_bytes, _bloom, _meta) = write_sorted_rows(
            vec![],
            Arc::new(schema),
            Level(0),
            crate::types::level::FileFormat::Dual,
            10,
        )
        .unwrap();
        assert!(file_bytes.is_empty());
        // Nothing to read; just confirm we don't crash trying.
        assert!(read_kv_index_from_file(&file_bytes).is_none());
    }
}
