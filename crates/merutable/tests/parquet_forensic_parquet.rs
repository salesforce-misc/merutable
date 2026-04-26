//! Forensic inspection of Parquet files produced by merutable's writer.
//!
//! This test suite complements the unit tests: instead of exercising one
//! code path at a time, it produces a real file, opens it with **raw**
//! upstream `parquet` APIs (no merutable reader in the loop), and walks
//! every section of the file to validate that:
//!
//! 1. **Every footer KV we claim to write is present** and nothing else is
//!    there that we didn't put there. The schema of the footer is the
//!    cross-system contract with external analytical readers — it must be stable.
//! 2. **The column layout matches the per-level contract**: L0 carries
//!    `_merutable_ikey`, `_merutable_value`, then typed user columns in
//!    order; L1+ carries `_merutable_ikey` then typed user columns in
//!    order — no postcard blob.
//! 3. **Every data page referenced by `merutable.kv_index.v1` actually
//!    exists in the file** at the stated (offset, size) and the first
//!    value on that page decodes to the key the index entry claims.
//!    This is the cross-consistency check that a latent off-by-one
//!    between page sizing and index construction would trigger.
//! 4. **The bloom filter in `merutable.bloom` answers `may_contain` true
//!    for every written user key** (zero false negatives).
//! 5. **`ParquetFileMeta.num_rows` equals the sum of row-group num_rows**.
//! 6. **Every column chunk uses SNAPPY** — the writer's configured codec.
//! 7. **`OffsetIndex` and `ColumnIndex` are populated** on every column
//!    chunk (the writer enables them via `ArrowWriter`'s defaults, and
//!    the kv_index extractor relies on them).
//!
//! Any discrepancy surfaced here is a real, user-visible bug: either the
//! file structure diverged from the spec or the footer metadata does not
//! describe the actual byte layout.

use std::sync::Arc;

use bytes::Bytes;
use merutable::parquet::{
    bloom::FastLocalBloom,
    codec::{IKEY_COLUMN_NAME, OP_COLUMN_NAME, SEQ_COLUMN_NAME, VALUE_BLOB_COLUMN_NAME},
    kv_index::{KV_INDEX_FOOTER_KEY, KvSparseIndex},
    writer::write_sorted_rows,
};
use merutable::types::{
    key::InternalKey,
    level::{Level, ParquetFileMeta},
    schema::{ColumnDef, ColumnType, TableSchema},
    sequence::{OpType, SeqNum},
    value::{FieldValue, Row},
};
use parquet::basic::Compression;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::{ReadOptionsBuilder, SerializedFileReader};

// ── Test schema / fixtures ───────────────────────────────────────────────

fn forensic_schema() -> TableSchema {
    TableSchema {
        table_name: "forensic".into(),
        columns: vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,

                ..Default::default()
            },
            ColumnDef {
                name: "name".into(),
                col_type: ColumnType::ByteArray,
                nullable: true,

                ..Default::default()
            },
            ColumnDef {
                name: "active".into(),
                col_type: ColumnType::Boolean,
                nullable: false,

                ..Default::default()
            },
            ColumnDef {
                name: "score".into(),
                col_type: ColumnType::Double,
                nullable: true,

                ..Default::default()
            },
        ],
        primary_key: vec![0],

        ..Default::default()
    }
}

fn build_forensic_rows(n: i64, schema: &TableSchema) -> Vec<(InternalKey, Row)> {
    (1..=n)
        .map(|i| {
            // Unique 128-byte payload so Parquet's dictionary encoding
            // doesn't collapse into a single page — we need multi-page
            // output to exercise the kv_index page enumeration.
            let mut pad = vec![0u8; 120];
            for (j, b) in pad.iter_mut().enumerate() {
                *b = ((i as usize + j) & 0xFF) as u8;
            }
            let name = FieldValue::Bytes(Bytes::from(pad));
            let row = Row::new(vec![
                Some(FieldValue::Int64(i)),
                Some(name),
                Some(FieldValue::Boolean(i % 2 == 0)),
                Some(FieldValue::Double(i as f64 * 0.5)),
            ]);
            let ikey = InternalKey::encode(
                &[FieldValue::Int64(i)],
                SeqNum(i as u64 + 1),
                OpType::Put,
                schema,
            )
            .unwrap();
            (ikey, row)
        })
        .collect()
}

/// Read the full on-disk key-value footer as a `(key, value)` list. Unlike
/// the engine's own helper this does not filter by prefix so we can
/// catch surprising entries.
fn all_footer_kv(file_bytes: &[u8]) -> Vec<(String, String)> {
    let reader = SerializedFileReader::new(Bytes::copy_from_slice(file_bytes)).unwrap();
    let kv = reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default();
    kv.into_iter()
        .map(|e| (e.key, e.value.unwrap_or_default()))
        .collect()
}

// ── Structural forensics ─────────────────────────────────────────────────

/// An L0 file produced by the writer must carry exactly this set of
/// merutable footer KV keys, plus whatever Arrow/Parquet-internal keys
/// the upstream ArrowWriter adds (e.g. ARROW:schema). The merutable set
/// must be exhaustive — an unexpected extra key, or a missing one,
/// means the cross-system footer contract has drifted.
#[test]
fn l0_footer_kv_contains_every_expected_merutable_key() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(2_000, &schema);
    let (file_bytes, _bloom, _meta) = write_sorted_rows(
        rows,
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();

    let kv = all_footer_kv(&file_bytes);
    let keys: Vec<&str> = kv.iter().map(|(k, _)| k.as_str()).collect();

    // Required merutable keys.
    for required in [
        ParquetFileMeta::FOOTER_KEY,
        ParquetFileMeta::SCHEMA_KEY,
        "merutable.bloom",
        KV_INDEX_FOOTER_KEY,
    ] {
        assert!(
            keys.contains(&required),
            "L0 footer missing required key '{required}'; got: {keys:?}"
        );
    }

    // Every `merutable.*` key must be in the known set. An unknown
    // merutable-prefixed key means we wrote something and forgot to
    // update this allow-list — which also means any reader that
    // rejects unknown keys (not merutable's own, but a hypothetical
    // strict consumer) would break.
    let allowed_merutable: &[&str] = &[
        ParquetFileMeta::FOOTER_KEY,
        ParquetFileMeta::SCHEMA_KEY,
        "merutable.bloom",
        KV_INDEX_FOOTER_KEY,
    ];
    for (k, _) in &kv {
        if k.starts_with("merutable.") {
            assert!(
                allowed_merutable.contains(&k.as_str()),
                "unexpected merutable.* footer key '{k}' — update the allow-list or stop writing it"
            );
        }
    }
}

/// Each footer KV value must be non-empty and must decode through the
/// type's constructor without error. This catches corrupted or empty
/// values that pass the raw Parquet KV schema check but fail the
/// downstream consumer.
#[test]
fn every_footer_kv_value_decodes_cleanly() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(2_000, &schema);
    let (file_bytes, _, _) = write_sorted_rows(
        rows,
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();
    let kv = all_footer_kv(&file_bytes);

    let meta_json = kv
        .iter()
        .find(|(k, _)| k == ParquetFileMeta::FOOTER_KEY)
        .map(|(_, v)| v.clone())
        .unwrap();
    let meta = ParquetFileMeta::deserialize(&meta_json).expect("merutable.meta must decode");
    assert_eq!(meta.level, Level(0));
    assert_eq!(meta.num_rows, 2_000);

    let schema_json = kv
        .iter()
        .find(|(k, _)| k == ParquetFileMeta::SCHEMA_KEY)
        .map(|(_, v)| v.clone())
        .unwrap();
    let decoded_schema: TableSchema =
        serde_json::from_str(&schema_json).expect("merutable.schema must decode");
    assert_eq!(decoded_schema.columns.len(), schema.columns.len());
    for (a, b) in decoded_schema.columns.iter().zip(schema.columns.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.col_type, b.col_type);
        assert_eq!(a.nullable, b.nullable);
    }

    let bloom_hex = kv
        .iter()
        .find(|(k, _)| k == "merutable.bloom")
        .map(|(_, v)| v.clone())
        .unwrap();
    let bloom_bytes = hex::decode(&bloom_hex).expect("bloom hex must decode");
    let _bloom = FastLocalBloom::from_bytes(&bloom_bytes)
        .expect("bloom bytes must parse via FastLocalBloom");

    let kv_hex = kv
        .iter()
        .find(|(k, _)| k == KV_INDEX_FOOTER_KEY)
        .map(|(_, v)| v.clone())
        .unwrap();
    let kv_raw = hex::decode(&kv_hex).expect("kv_index hex must decode");
    let _idx = KvSparseIndex::from_bytes(Bytes::from(kv_raw))
        .expect("kv_index bytes must parse via KvSparseIndex");
}

/// The file's physical column layout must match the per-level contract:
/// L0 → `[_merutable_ikey, _merutable_value, id, name, active, score]`
/// L1 → `[_merutable_ikey, id, name, active, score]`
/// in *that* order. Column index stability is what lets the kv_index
/// writer use `IKEY_COL_IDX = 0` as an invariant across levels.
#[test]
fn physical_column_layout_matches_level_contract() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(64, &schema);

    let (l0, _, _) = write_sorted_rows(
        rows.clone(),
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();
    let (l1, _, _) = write_sorted_rows(
        rows,
        Arc::new(schema.clone()),
        Level(1),
        merutable::types::level::FileFormat::Columnar,
        10,
    )
    .unwrap();

    let l0_reader = SerializedFileReader::new(Bytes::from(l0)).unwrap();
    let l1_reader = SerializedFileReader::new(Bytes::from(l1)).unwrap();

    let l0_descr = l0_reader.metadata().file_metadata().schema_descr_ptr();
    let l1_descr = l1_reader.metadata().file_metadata().schema_descr_ptr();

    let l0_names: Vec<String> = (0..l0_descr.num_columns())
        .map(|i| l0_descr.column(i).name().to_string())
        .collect();
    let l1_names: Vec<String> = (0..l1_descr.num_columns())
        .map(|i| l1_descr.column(i).name().to_string())
        .collect();

    // Issue #16: every merutable-written file carries _merutable_seq
    // (Int64) and _merutable_op (Int32) after _merutable_ikey, so
    // external analytics readers can apply the MVCC dedup projection
    // without decoding the ikey trailer.
    let expected_l0: Vec<String> = vec![
        IKEY_COLUMN_NAME.to_string(),
        SEQ_COLUMN_NAME.to_string(),
        OP_COLUMN_NAME.to_string(),
        VALUE_BLOB_COLUMN_NAME.to_string(),
        "id".into(),
        "name".into(),
        "active".into(),
        "score".into(),
    ];
    let expected_l1: Vec<String> = vec![
        IKEY_COLUMN_NAME.to_string(),
        SEQ_COLUMN_NAME.to_string(),
        OP_COLUMN_NAME.to_string(),
        "id".into(),
        "name".into(),
        "active".into(),
        "score".into(),
    ];

    assert_eq!(
        l0_names, expected_l0,
        "L0 column order is part of the file contract"
    );
    assert_eq!(
        l1_names, expected_l1,
        "L1 column order is part of the file contract"
    );
}

/// `ParquetFileMeta.num_rows` must equal the sum of row-group num_rows
/// (as reported by Parquet). A drift here means the writer's bookkeeping
/// disagrees with what actually got written — the kind of bug that
/// makes an engine `get()` fall off a page boundary.
#[test]
fn footer_num_rows_matches_row_group_total() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(2_000, &schema);
    let (file_bytes, _, meta) = write_sorted_rows(
        rows,
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();

    let reader = SerializedFileReader::new(Bytes::from(file_bytes)).unwrap();
    let rg_total: i64 = (0..reader.metadata().num_row_groups())
        .map(|i| reader.metadata().row_group(i).num_rows())
        .sum();
    assert_eq!(
        rg_total as u64, meta.num_rows,
        "sum of row-group num_rows ({rg_total}) disagrees with ParquetFileMeta.num_rows ({})",
        meta.num_rows
    );
}

/// Every column chunk in every row group must use the writer's
/// configured compression (SNAPPY). A drift here means the writer's
/// property builder lost a call or someone flipped the default — both
/// regressions we catch once and then pin.
#[test]
fn every_column_chunk_uses_snappy_compression() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(2_000, &schema);
    let (file_bytes, _, _) = write_sorted_rows(
        rows,
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();

    let reader = SerializedFileReader::new(Bytes::from(file_bytes)).unwrap();
    let md = reader.metadata();
    for rg in 0..md.num_row_groups() {
        let rg_md = md.row_group(rg);
        for col in 0..rg_md.num_columns() {
            let ccm = rg_md.column(col);
            assert_eq!(
                ccm.compression(),
                Compression::SNAPPY,
                "row group {rg} column {col} ({}) uses {:?}, expected SNAPPY",
                ccm.column_path().string(),
                ccm.compression()
            );
        }
    }
}

/// Every column chunk must have an `OffsetIndex` and a `ColumnIndex`.
/// The writer relies on the OffsetIndex for kv_index extraction; missing
/// it silently disables the kv_index entirely. ColumnIndex gives
/// external external analytical readers page-level min/max pruning, which is part of
/// the external analytics performance story.
#[test]
fn offset_and_column_indexes_exist_on_every_column_chunk() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(2_000, &schema);
    let (file_bytes, _, _) = write_sorted_rows(
        rows,
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();

    let opts = ReadOptionsBuilder::new().with_page_index().build();
    let reader = SerializedFileReader::new_with_options(Bytes::from(file_bytes), opts).unwrap();
    let md = reader.metadata();

    let offset_index = md.offset_index().expect("OffsetIndex must be present");
    let column_index = md.column_index().expect("ColumnIndex must be present");

    assert_eq!(offset_index.len(), md.num_row_groups());
    assert_eq!(column_index.len(), md.num_row_groups());
    for rg in 0..md.num_row_groups() {
        let rg_md = md.row_group(rg);
        assert_eq!(
            offset_index[rg].len(),
            rg_md.num_columns(),
            "row group {rg} OffsetIndex column count"
        );
        assert_eq!(
            column_index[rg].len(),
            rg_md.num_columns(),
            "row group {rg} ColumnIndex column count"
        );
    }
}

/// Each kv_index entry must point at a page that actually exists in
/// the Parquet file, and the first key on that page must equal the
/// kv_index entry's key. This is the cross-layer invariant that lets
/// the reader skip straight to the right page for point lookups —
/// any drift here corrupts point-lookup answers silently.
#[test]
fn kv_index_entries_point_at_real_pages_with_matching_first_key() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(8_000, &schema);
    let (file_bytes, _, _) = write_sorted_rows(
        rows.clone(),
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();

    // Load the kv_index from the footer.
    let kv = all_footer_kv(&file_bytes);
    let kv_hex = kv
        .iter()
        .find(|(k, _)| k == KV_INDEX_FOOTER_KEY)
        .map(|(_, v)| v.clone())
        .unwrap();
    let kv_raw = hex::decode(&kv_hex).unwrap();
    let idx = KvSparseIndex::from_bytes(Bytes::from(kv_raw)).unwrap();
    assert!(!idx.is_empty());

    // Load the OffsetIndex so we can enumerate the actual pages.
    let opts = ReadOptionsBuilder::new().with_page_index().build();
    let reader =
        SerializedFileReader::new_with_options(Bytes::copy_from_slice(&file_bytes), opts).unwrap();
    let md = reader.metadata();
    let offset_index = md.offset_index().unwrap();

    // Build a `(page_offset → (rg_idx, page_idx, global_first_row, compressed_size))` map.
    let mut page_by_offset: std::collections::HashMap<i64, (usize, usize, u64, i32)> =
        std::collections::HashMap::new();
    let mut row_group_start: u64 = 0;
    for (rg_idx, rg_offsets) in offset_index.iter().enumerate() {
        for page_loc in rg_offsets[0].page_locations() {
            let global_first = row_group_start + page_loc.first_row_index as u64;
            page_by_offset.insert(
                page_loc.offset,
                (
                    rg_idx,
                    0, // col idx (we only use ikey = 0)
                    global_first,
                    page_loc.compressed_page_size,
                ),
            );
        }
        row_group_start += md.row_group(rg_idx).num_rows() as u64;
    }

    // Every kv_index entry must match exactly one page at the claimed
    // offset, the stated compressed size, and a global first_row that
    // equals the one in the kv_index entry.
    for (entry_key, loc) in idx.iter() {
        let (_rg_idx, _col_idx, actual_first_row, actual_size) = page_by_offset
            .get(&(loc.page_offset as i64))
            .unwrap_or_else(|| {
                panic!(
                    "kv_index entry at page_offset {} does not correspond to any real page",
                    loc.page_offset
                )
            });

        assert_eq!(
            *actual_first_row, loc.first_row_index,
            "kv_index first_row_index {} does not match actual first_row {actual_first_row} \
             at page_offset {}",
            loc.first_row_index, loc.page_offset
        );
        assert_eq!(
            *actual_size as u32, loc.page_size,
            "kv_index page_size disagrees with OffsetIndex at page_offset {}",
            loc.page_offset
        );

        // Cross-check: the input row at `first_row_index` should have an
        // InternalKey that matches the kv_index entry key.
        let input_key = rows[*actual_first_row as usize].0.as_bytes();
        assert_eq!(
            entry_key.as_slice(),
            input_key,
            "kv_index entry key at global_first_row {actual_first_row} disagrees with input"
        );
    }
}

/// Every user key we wrote must be reported as "may contain" by the
/// bloom filter stored in the footer. Zero false negatives is the
/// hard contract — false positives are fine.
#[test]
fn every_written_user_key_passes_the_footer_bloom() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(2_000, &schema);
    let (file_bytes, _, _) = write_sorted_rows(
        rows.clone(),
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();

    let kv = all_footer_kv(&file_bytes);
    let bloom_hex = kv
        .iter()
        .find(|(k, _)| k == "merutable.bloom")
        .map(|(_, v)| v.clone())
        .unwrap();
    let bloom_bytes = hex::decode(&bloom_hex).unwrap();
    let bloom = FastLocalBloom::from_bytes(&bloom_bytes).unwrap();

    for (ikey, _) in &rows {
        let uk = ikey.user_key_bytes();
        assert!(
            bloom.may_contain(uk),
            "bloom false negative for user key {:?} (length {})",
            uk,
            uk.len()
        );
    }
}

/// A completely non-matching probe should be rejected by the bloom
/// filter the vast majority of the time. This isn't a correctness
/// invariant (false positives are allowed) but it's a coarse sanity
/// check that the bloom was actually populated — a broken bloom that
/// returns true for everything would pass the previous test but fail
/// this one with very high probability.
#[test]
fn bloom_rejects_most_random_non_matching_probes() {
    let schema = forensic_schema();
    let rows = build_forensic_rows(2_000, &schema);
    let (file_bytes, _, _) = write_sorted_rows(
        rows,
        Arc::new(schema.clone()),
        Level(0),
        merutable::types::level::FileFormat::Dual,
        10,
    )
    .unwrap();

    let kv = all_footer_kv(&file_bytes);
    let bloom_hex = kv
        .iter()
        .find(|(k, _)| k == "merutable.bloom")
        .map(|(_, v)| v.clone())
        .unwrap();
    let bloom = FastLocalBloom::from_bytes(&hex::decode(&bloom_hex).unwrap()).unwrap();

    // Probe with IDs well outside the written range [1..=2000].
    let mut fp = 0usize;
    let total = 5_000usize;
    for i in 10_000_000i64..10_000_000 + total as i64 {
        // Encode a user key in the same shape as real writes so the
        // hash space is comparable.
        let uk = InternalKey::encode(
            &[FieldValue::Int64(i)],
            SeqNum(1),
            OpType::Put,
            &forensic_schema(),
        )
        .unwrap();
        if bloom.may_contain(uk.user_key_bytes()) {
            fp += 1;
        }
    }
    let fpr = fp as f64 / total as f64;
    assert!(
        fpr < 0.05,
        "bloom FPR {fpr:.4} too high — expected <5% for a populated bloom at 10 bits/key"
    );
}
