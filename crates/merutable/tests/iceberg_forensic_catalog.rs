//! Forensic inspection of the Iceberg catalog's on-disk artifacts.
//!
//! Every test spins up a real `IcebergCatalog` against a `TempDir`,
//! performs real commits, then opens the raw files on disk and
//! validates their structure section-by-section without using the
//! catalog's own reader. If the writer and reader drift (for example
//! by silently stamping placeholder zeros into DV pointers, as an
//! earlier bug did), the forensic walker catches it.

use std::{collections::HashMap, fs, sync::Arc};

use bytes::Bytes;
use merutable::iceberg::{
    IcebergCatalog,
    deletion_vector::DeletionVector,
    manifest::{DvLocation, Manifest},
    snapshot::{IcebergDataFile, SnapshotTransaction},
};
use merutable::types::{
    level::{Level, ParquetFileMeta},
    schema::{ColumnDef, ColumnType, TableSchema},
};

fn sample_schema() -> TableSchema {
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

fn sample_meta(level: u8) -> ParquetFileMeta {
    ParquetFileMeta {
        level: Level(level),
        seq_min: 1,
        seq_max: 10,
        key_min: vec![0x01],
        key_max: vec![0xFF],
        num_rows: 100,
        file_size: 1024,
        dv_path: None,
        dv_offset: None,
        dv_length: None,
        format: None,
        column_stats: None,
    }
}

/// A freshly opened catalog must create `metadata/` and `data/` dirs
/// *but no manifest file* — nothing has been committed yet.
#[tokio::test]
async fn open_creates_dirs_without_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let _c = IcebergCatalog::open(tmp.path(), sample_schema())
        .await
        .unwrap();

    assert!(tmp.path().join("metadata").is_dir());
    assert!(tmp.path().join("data").is_dir());
    assert!(!tmp.path().join("version-hint.text").exists());
    // No v*.metadata.json files yet.
    let files: Vec<_> = fs::read_dir(tmp.path().join("metadata"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(files.is_empty(), "no metadata files expected: {files:?}");
}

/// Issue #28 Phase 5: after a single flush commit, exactly one
/// `v1.metadata.pb` file exists (JSON writes were dropped), and
/// the protobuf decodes to a manifest with the right shape.
#[tokio::test]
async fn single_flush_commit_writes_v1_metadata_and_hint() {
    let tmp = tempfile::tempdir().unwrap();
    let schema = Arc::new(sample_schema());
    let catalog = IcebergCatalog::open(tmp.path(), sample_schema())
        .await
        .unwrap();

    let mut txn = SnapshotTransaction::new();
    txn.add_file(IcebergDataFile {
        path: "data/L0/abc.parquet".into(),
        file_size: 1024,
        num_rows: 100,
        meta: sample_meta(0),
    });
    txn.set_prop("merutable.job", "flush");
    catalog.commit(&txn, schema).await.unwrap();

    // version-hint.text contains exactly "1".
    let hint = fs::read_to_string(tmp.path().join("version-hint.text")).unwrap();
    assert_eq!(hint.trim(), "1");

    // Issue #28 Phase 5: commit emits ONLY v1.metadata.pb. JSON
    // is no longer written. Legacy JSON-only catalogs still read
    // correctly (covered by catalog::tests), but the canonical
    // write path is protobuf-only.
    let mut names: Vec<_> = fs::read_dir(tmp.path().join("metadata"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["v1.metadata.pb".to_string()]);

    // Decode the protobuf manifest and verify the fields flushed
    // into the commit round-tripped.
    let pb = fs::read(tmp.path().join("metadata").join("v1.metadata.pb")).unwrap();
    let m = Manifest::from_protobuf(&pb).unwrap();
    assert_eq!(m.snapshot_id, 1);
    assert_eq!(m.entries.len(), 1);
    assert_eq!(m.entries[0].path, "data/L0/abc.parquet");
    assert_eq!(m.entries[0].meta.level, Level(0));
    assert_eq!(m.entries[0].dv_path, None);
    assert_eq!(m.entries[0].dv_offset, None);
    assert_eq!(m.entries[0].dv_length, None);
    assert_eq!(m.properties.get("merutable.job").unwrap(), "flush");
}

/// Issue #28 Phase 5: every commit produces a new `v{N}.metadata.pb`
/// file; older ones must be preserved so snapshot isolation / time
/// travel works. The hint must always point at the newest version.
#[tokio::test]
async fn sequential_commits_preserve_every_metadata_version() {
    let tmp = tempfile::tempdir().unwrap();
    let schema = Arc::new(sample_schema());
    let catalog = IcebergCatalog::open(tmp.path(), sample_schema())
        .await
        .unwrap();

    for i in 0..5 {
        let mut txn = SnapshotTransaction::new();
        txn.add_file(IcebergDataFile {
            path: format!("data/L0/f{i}.parquet"),
            file_size: 1024,
            num_rows: 100,
            meta: sample_meta(0),
        });
        catalog.commit(&txn, schema.clone()).await.unwrap();
    }

    // All 5 protobuf metadata files must exist.
    for i in 1..=5 {
        let p = tmp
            .path()
            .join("metadata")
            .join(format!("v{i}.metadata.pb"));
        assert!(p.exists(), "missing {p:?}");
        let m = Manifest::from_protobuf(&fs::read(&p).unwrap()).unwrap();
        assert_eq!(m.snapshot_id, i as i64);
        assert_eq!(m.entries.len(), i);
    }

    // version-hint.text points at the newest version.
    assert_eq!(
        fs::read_to_string(tmp.path().join("version-hint.text"))
            .unwrap()
            .trim(),
        "5"
    );
}

/// Regression: after committing a DV update, the on-disk manifest
/// must carry REAL dv_offset/dv_length pointing at the actual blob
/// bytes inside the on-disk Puffin file. A prior bug stamped
/// `(Some(0), Some(0))` placeholders into the manifest; on reload
/// the reader would read zero bytes → empty bitmap → deleted rows
/// silently reappeared.
#[tokio::test]
async fn commit_with_dv_writes_real_offset_and_length_to_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let schema = Arc::new(sample_schema());
    let catalog = IcebergCatalog::open(tmp.path(), sample_schema())
        .await
        .unwrap();

    // Flush an L0 file.
    let mut txn1 = SnapshotTransaction::new();
    txn1.add_file(IcebergDataFile {
        path: "data/L0/abc.parquet".into(),
        file_size: 1024,
        num_rows: 100,
        meta: sample_meta(0),
    });
    catalog.commit(&txn1, schema.clone()).await.unwrap();

    // Partial compaction: attach a DV to the L0 file.
    let mut txn2 = SnapshotTransaction::new();
    let mut dv = DeletionVector::new();
    for i in 0..50u32 {
        dv.mark_deleted(i);
    }
    txn2.add_dv("data/L0/abc.parquet".into(), dv.clone());
    catalog.commit(&txn2, schema.clone()).await.unwrap();

    // Open v2.metadata.pb as raw bytes (Phase 5: pb is the only
    // write format). Previously this test read the JSON variant.
    let v2 = fs::read(tmp.path().join("metadata").join("v2.metadata.pb")).unwrap();
    let m = Manifest::from_protobuf(&v2).unwrap();
    let l0_entry = m
        .entries
        .iter()
        .find(|e| e.path == "data/L0/abc.parquet")
        .expect("L0 entry must still be present after DV commit");

    // The DV pointer must be non-null AND non-zero. (Placeholder zeros
    // were the bug.)
    let dv_path = l0_entry
        .dv_path
        .as_deref()
        .expect("dv_path must be set after DV commit");
    let dv_offset = l0_entry
        .dv_offset
        .expect("dv_offset must be set after DV commit");
    let dv_length = l0_entry
        .dv_length
        .expect("dv_length must be set after DV commit");
    assert_ne!(dv_offset, 0, "dv_offset was 0 — placeholder bug regression");
    assert!(
        dv_length > 0,
        "dv_length was {dv_length} — placeholder bug regression"
    );

    // Mirrored into the embedded ParquetFileMeta.
    assert_eq!(l0_entry.meta.dv_path.as_deref(), Some(dv_path));
    assert_eq!(l0_entry.meta.dv_offset, Some(dv_offset));
    assert_eq!(l0_entry.meta.dv_length, Some(dv_length));

    // The puffin file must exist at that path on disk.
    let puffin_abs = tmp.path().join(dv_path);
    assert!(
        puffin_abs.exists(),
        "puffin file not found at {puffin_abs:?}"
    );
    let puffin_bytes = fs::read(&puffin_abs).unwrap();

    // The [offset..offset+length] slice must decode as a roaring bitmap
    // that matches the DV we committed — byte-for-byte.
    let off = dv_offset as usize;
    let len = dv_length as usize;
    assert!(
        off + len <= puffin_bytes.len(),
        "manifest pointer ({off}..{}) exceeds puffin file size {}",
        off + len,
        puffin_bytes.len()
    );
    let blob = &puffin_bytes[off..off + len];
    let decoded = DeletionVector::from_puffin_blob(blob).unwrap();
    assert_eq!(decoded.cardinality(), dv.cardinality());
    for i in 0..50u32 {
        assert!(
            decoded.is_deleted(i),
            "row {i} should be deleted in reloaded DV"
        );
    }
}

/// After a DV commit, reopening the catalog from disk must preserve
/// the real DV pointers — no silent loss of deletion state across a
/// restart.
#[tokio::test]
async fn reopen_catalog_preserves_dv_pointers() {
    let tmp = tempfile::tempdir().unwrap();
    let schema = Arc::new(sample_schema());

    // Phase 1: flush + DV commit.
    {
        let catalog = IcebergCatalog::open(tmp.path(), sample_schema())
            .await
            .unwrap();
        let mut txn1 = SnapshotTransaction::new();
        txn1.add_file(IcebergDataFile {
            path: "data/L0/abc.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: sample_meta(0),
        });
        catalog.commit(&txn1, schema.clone()).await.unwrap();

        let mut txn2 = SnapshotTransaction::new();
        let mut dv = DeletionVector::new();
        dv.mark_deleted(42);
        dv.mark_deleted(99);
        txn2.add_dv("data/L0/abc.parquet".into(), dv);
        catalog.commit(&txn2, schema.clone()).await.unwrap();
    }

    // Phase 2: reopen and read the manifest via the catalog.
    let catalog = IcebergCatalog::open(tmp.path(), sample_schema())
        .await
        .unwrap();
    let m = catalog.current_manifest().await;
    assert_eq!(m.snapshot_id, 2);
    let e = m
        .entries
        .iter()
        .find(|e| e.path == "data/L0/abc.parquet")
        .unwrap();
    let dv_path = e.dv_path.as_deref().unwrap();
    let dv_offset = e.dv_offset.unwrap();
    let dv_length = e.dv_length.unwrap();
    assert_ne!(dv_offset, 0);
    assert!(dv_length > 0);

    // Load the puffin file at the recorded coordinates and verify.
    let puffin = fs::read(tmp.path().join(dv_path)).unwrap();
    let blob = &puffin[dv_offset as usize..(dv_offset + dv_length) as usize];
    let decoded = DeletionVector::from_puffin_blob(blob).unwrap();
    assert_eq!(decoded.cardinality(), 2);
    assert!(decoded.is_deleted(42));
    assert!(decoded.is_deleted(99));
}

/// The on-disk Puffin file written by `catalog::commit` must itself
/// be forensically valid: starts with "PFA1", ends with "PFA1", has a
/// well-formed footer with a `deletion-vector-v1` blob whose declared
/// offset/length match the manifest's pointer and the roaring bitmap
/// at that slice.
#[tokio::test]
async fn puffin_file_structure_is_byte_accurate_on_disk() {
    let tmp = tempfile::tempdir().unwrap();
    let schema = Arc::new(sample_schema());
    let catalog = IcebergCatalog::open(tmp.path(), sample_schema())
        .await
        .unwrap();

    let mut txn1 = SnapshotTransaction::new();
    txn1.add_file(IcebergDataFile {
        path: "data/L0/forensic.parquet".into(),
        file_size: 1024,
        num_rows: 100,
        meta: sample_meta(0),
    });
    catalog.commit(&txn1, schema.clone()).await.unwrap();

    let mut txn2 = SnapshotTransaction::new();
    let mut dv = DeletionVector::new();
    for i in (0..200u32).step_by(3) {
        dv.mark_deleted(i);
    }
    txn2.add_dv("data/L0/forensic.parquet".into(), dv.clone());
    catalog.commit(&txn2, schema).await.unwrap();

    // Find the puffin file on disk.
    let l0_dir = tmp.path().join("data").join("L0");
    let puffin_path = fs::read_dir(&l0_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some("puffin"))
        .expect("no puffin file written to data/L0/");
    let bytes = fs::read(&puffin_path).unwrap();

    // Magic bytes at both ends.
    assert_eq!(&bytes[..4], b"PFA1", "opening magic must be PFA1");
    assert_eq!(
        &bytes[bytes.len() - 4..],
        b"PFA1",
        "closing magic must be PFA1"
    );

    // Footer length + footer JSON.
    let fl_start = bytes.len() - 8;
    let footer_len = i32::from_le_bytes(bytes[fl_start..fl_start + 4].try_into().unwrap()) as usize;
    assert!(footer_len > 0 && footer_len < bytes.len());

    let footer_start = fl_start - footer_len;
    let footer_json = std::str::from_utf8(&bytes[footer_start..fl_start]).unwrap();
    let footer: serde_json::Value = serde_json::from_str(footer_json).unwrap();
    let blob = &footer["blobs"][0];
    assert_eq!(blob["type"], "deletion-vector-v1");
    assert_eq!(
        blob["fields"]["referenced-data-file"],
        "data/L0/forensic.parquet"
    );
    let blob_offset = blob["offset"].as_i64().unwrap();
    let blob_length = blob["length"].as_i64().unwrap();
    assert_eq!(
        blob_offset, 4,
        "DV blob must sit immediately after opening magic"
    );
    assert!(blob_length > 0);

    // The [offset..offset+length] bytes must reassemble into a
    // roaring bitmap equal to what we committed.
    let slice = &bytes[blob_offset as usize..(blob_offset as usize) + (blob_length as usize)];
    let decoded = DeletionVector::from_puffin_blob(slice).unwrap();
    assert_eq!(decoded.cardinality(), dv.cardinality());
    for i in (0..200u32).step_by(3) {
        assert!(decoded.is_deleted(i));
    }
}

/// Direct unit pin on `Manifest::apply`: supplying a `DvLocation` for
/// an entry in `txn.dvs` must result in the manifest carrying those
/// exact coordinates (no reconstruction, no placeholders).
#[test]
fn apply_threads_real_dv_location_into_manifest() {
    let schema = sample_schema();
    let mut m = Manifest::empty(schema);
    m.entries.push(merutable::iceberg::ManifestEntry {
        path: "data/L0/a.parquet".into(),
        meta: sample_meta(0),
        dv_path: None,
        dv_offset: None,
        dv_length: None,
        status: "existing".into(),
    });

    let mut txn = SnapshotTransaction::new();
    let mut dv = DeletionVector::new();
    dv.mark_deleted(0);
    txn.add_dv("data/L0/a.parquet".into(), dv);

    let mut locs = HashMap::new();
    locs.insert(
        "data/L0/a.parquet".into(),
        DvLocation {
            dv_path: "data/L0/a.dv-99.puffin".into(),
            dv_offset: 4,
            dv_length: 137,
        },
    );

    let m2 = m.apply(&txn, 99, &locs).unwrap();
    let e = m2
        .entries
        .iter()
        .find(|e| e.path == "data/L0/a.parquet")
        .unwrap();
    assert_eq!(e.dv_path.as_deref(), Some("data/L0/a.dv-99.puffin"));
    assert_eq!(e.dv_offset, Some(4));
    assert_eq!(e.dv_length, Some(137));

    // Shut up unused-import lint for Bytes.
    let _ = Bytes::new();
}
