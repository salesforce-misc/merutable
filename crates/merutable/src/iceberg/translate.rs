//! Iceberg translator: projects merutable's native `Manifest` onto Apache
//! Iceberg v2 `TableMetadata` JSON.
//!
//! # Why this module exists
//!
//! merutable's commit path writes a native JSON manifest (see [`Manifest`])
//! rather than Iceberg's four-file (`metadata.json` + manifest-list Avro +
//! manifest Avro + data Parquet) layout. The native format is chosen for
//! efficiency — one fsyncable JSON per commit, zero Avro dependency on the
//! hot path — but it is designed as a strict **superset** of Iceberg v2
//! `TableMetadata` so that any merutable snapshot can be projected onto a
//! spec-compliant Iceberg `metadata.json` with no loss of information.
//!
//! This module is that projection.
//!
//! # What the translator does NOT do (yet)
//!
//! - **No deletion-vector projection.** merutable writes V3-style Puffin
//!   `deletion-vector-v1` blobs for partial compactions. V3 is not yet
//!   implemented by the `iceberg-rs` crate we depend on, so the emitted
//!   `metadata.json` is shaped as v2 and the DV files are listed under
//!   a merutable-specific property (`merutable.deletion-vectors`). The
//!   Puffin files themselves on disk are already Iceberg v3 spec-compliant.
//!
//! # Field mapping
//!
//! | merutable `Manifest`          | Iceberg `TableMetadata`      |
//! |-------------------------------|------------------------------|
//! | `format_version`              | `format-version` (pinned 2)  |
//! | `table_uuid`                  | `table-uuid`                 |
//! | `last_updated_ms`             | `last-updated-ms`            |
//! | `sequence_number`             | `last-sequence-number`       |
//! | `snapshot_id`                 | `current-snapshot-id`        |
//! | `parent_snapshot_id`          | `snapshot.parent-snapshot-id`|
//! | `schema` (merutable)          | `schemas[0]` (Iceberg types) |
//! | `entries[]`                   | referenced via manifest-list |
//! | `properties`                  | `properties` (passed through)|

use std::sync::Arc;

use crate::types::{
    level::ParquetFileMeta,
    schema::{ColumnType, TableSchema},
};
use serde_json::{json, Value};

use crate::iceberg::manifest::{Manifest, ManifestEntry};

// ── Top-level projection ─────────────────────────────────────────────────────

/// Project a merutable [`Manifest`] onto an Iceberg v2 `TableMetadata` JSON
/// value.
///
/// `table_location` is the filesystem URI (e.g. `file:///tmp/db` or
/// `s3://bucket/path`) that Iceberg catalogs record in the `location`
/// field. The manifest-list path is derived from `snapshot_id` and
/// embedded in `snapshots[0].manifest-list`.
///
/// The returned `Value` can be written directly to
/// `{target}/metadata/v{N}.metadata.json` for consumption by pyiceberg,
/// Spark, Trino, DuckDB, Snowflake, Athena, etc.
pub fn to_iceberg_v2_table_metadata(manifest: &Manifest, table_location: &str) -> Value {
    let schema_json = to_iceberg_schema_v2(&manifest.schema, 0);
    let last_column_id = manifest.schema.columns.len() as i32;
    let sort_order = to_iceberg_sort_order_v2(&manifest.schema, 1);

    // The manifest-list path is conventional: every Iceberg snapshot
    // references one manifest-list file. We emit a deterministic path
    // under the table location so that a downstream Avro emitter can
    // fill in the file.
    let manifest_list_path = format!(
        "{}/metadata/snap-{}-0-{}.avro",
        table_location.trim_end_matches('/'),
        manifest.snapshot_id,
        manifest.table_uuid
    );

    // One snapshot entry for the current commit. merutable does not
    // retain a full chain in memory (only the live version), so the
    // snapshots array starts with the current one; callers who want
    // the full history should translate each v{N}.metadata.json in
    // turn and concatenate the snapshot entries.
    let mut snapshot = json!({
        "snapshot-id": manifest.snapshot_id,
        "timestamp-ms": manifest.last_updated_ms,
        "sequence-number": manifest.sequence_number,
        "summary": snapshot_summary(manifest),
        "manifest-list": manifest_list_path,
        "schema-id": 0i32,
    });
    if let Some(parent) = manifest.parent_snapshot_id {
        snapshot
            .as_object_mut()
            .unwrap()
            .insert("parent-snapshot-id".to_string(), json!(parent));
    }

    json!({
        "format-version": 2,
        "table-uuid": manifest.table_uuid,
        "location": table_location,
        "last-sequence-number": manifest.sequence_number,
        "last-updated-ms": manifest.last_updated_ms,
        "last-column-id": last_column_id,
        "current-schema-id": 0i32,
        "schemas": [schema_json],
        "default-spec-id": 0i32,
        // merutable doesn't partition; emit the unpartitioned spec.
        "partition-specs": [{
            "spec-id": 0i32,
            "fields": []
        }],
        "last-partition-id": 999i32,
        // merutable writes every Parquet file in strict (_merutable_ikey
        // ASC) order, which encodes (PK ASC, seq DESC). The user-visible
        // effect — and the part an Iceberg sort-order can express — is
        // PK ASC. Issue #20: projecting this lets Iceberg-aware engines
        // (DuckDB, Spark, Trino) apply streaming "first row per partition"
        // for the MVCC dedup projection (docs/EXTERNAL_READS.md) instead of
        // a full sort, turning an O(N log N) scan into O(N).
        "default-sort-order-id": 1i32,
        "sort-orders": [
            // Iceberg requires order-id 0 to exist as the "unsorted"
            // sentinel; table-metadata validators reject the file
            // without it. order-id 1 is our real sort.
            {"order-id": 0i32, "fields": []},
            sort_order,
        ],
        "properties": all_properties(manifest),
        "current-snapshot-id": manifest.snapshot_id,
        "snapshots": [snapshot],
        "snapshot-log": [{
            "snapshot-id": manifest.snapshot_id,
            "timestamp-ms": manifest.last_updated_ms
        }],
        "metadata-log": [],
        "refs": {
            "main": {
                "snapshot-id": manifest.snapshot_id,
                "type": "branch"
            }
        }
    })
}

// ── Schema projection ────────────────────────────────────────────────────────

/// Project a merutable [`TableSchema`] onto an Iceberg v2 schema JSON.
///
/// Iceberg uses typed field IDs starting at 1. merutable's columns map
/// as follows:
///
/// | `ColumnType`              | Iceberg type       |
/// |---------------------------|--------------------|
/// | `Boolean`                 | `boolean`          |
/// | `Int32`                   | `int`              |
/// | `Int64`                   | `long`             |
/// | `Float`                   | `float`            |
/// | `Double`                  | `double`           |
/// | `ByteArray`               | `binary`           |
/// | `FixedLenByteArray(n)`    | `fixed[n]`         |
///
/// merutable `ByteArray` stores both opaque bytes and UTF-8 strings — we
/// project conservatively as `binary` since we cannot know which at the
/// schema layer. Callers that know a particular column holds UTF-8 can
/// post-process the returned JSON to rewrite the type to `string`.
pub fn to_iceberg_schema_v2(schema: &TableSchema, schema_id: i32) -> Value {
    let mut fields = Vec::with_capacity(schema.columns.len());
    for (idx, col) in schema.columns.iter().enumerate() {
        let field_id = (idx + 1) as i32;
        fields.push(json!({
            "id": field_id,
            "name": col.name,
            // Iceberg: required=true means NOT NULL.
            "required": !col.nullable,
            "type": column_type_to_iceberg(&col.col_type),
        }));
    }

    // Iceberg's identifier-field-ids corresponds to merutable's primary
    // key: a list of field ids that together identify a row uniquely.
    let identifier_field_ids: Vec<i32> = schema
        .primary_key
        .iter()
        .map(|&idx| (idx + 1) as i32)
        .collect();

    json!({
        "type": "struct",
        "schema-id": schema_id,
        "identifier-field-ids": identifier_field_ids,
        "fields": fields,
    })
}

/// Project merutable's sort discipline (`_merutable_ikey` ASC, which
/// encodes `(PK ASC, seq DESC)`) onto an Iceberg v2 sort-order entry.
///
/// External engines that recognize the sort order apply streaming
/// partition-aware reductions — specifically, the mandatory MVCC
/// dedup projection in `docs/EXTERNAL_READS.md` collapses from an
/// O(N log N) full sort to an O(N) streaming pass.
///
/// We express only the PK ASC part. The seq DESC tail is embedded in
/// `_merutable_ikey` but `_merutable_ikey` is not in the public
/// Iceberg schema — we cannot reference it from a sort-order field.
/// External engines don't need the seq dimension to do streaming
/// dedup; they just need to know the PK is monotonic within a file.
pub fn to_iceberg_sort_order_v2(schema: &TableSchema, order_id: i32) -> Value {
    let fields: Vec<Value> = schema
        .primary_key
        .iter()
        .map(|&col_idx| {
            let source_id = (col_idx + 1) as i32;
            json!({
                "transform": "identity",
                "source-id": source_id,
                "direction": "asc",
                // Iceberg v2 requires one of "nulls-first" / "nulls-last".
                // merutable PK columns are always non-null (validated at
                // write time), so either is acceptable; "nulls-first" is
                // the spec's default for ascending.
                "null-order": "nulls-first"
            })
        })
        .collect();
    json!({
        "order-id": order_id,
        "fields": fields,
    })
}

fn column_type_to_iceberg(ct: &ColumnType) -> Value {
    match ct {
        ColumnType::Boolean => json!("boolean"),
        ColumnType::Int32 => json!("int"),
        ColumnType::Int64 => json!("long"),
        ColumnType::Float => json!("float"),
        ColumnType::Double => json!("double"),
        ColumnType::ByteArray => json!("binary"),
        ColumnType::FixedLenByteArray(n) => json!(format!("fixed[{n}]")),
    }
}

// ── Data-file projection ─────────────────────────────────────────────────────

/// Project one merutable [`ManifestEntry`] onto an Iceberg v2 `DataFile`
/// entry in the shape an Avro manifest writer expects.
///
/// The returned JSON is **not** a full Avro record (it lacks the
/// `snapshot_id` / `sequence_number` fields the manifest writer would
/// stamp), but it contains all per-file information in Iceberg-shaped
/// keys. A downstream Avro emitter can consume it directly.
pub fn to_iceberg_data_file_v2(entry: &ManifestEntry) -> Value {
    to_iceberg_data_file_v2_with_schema(entry, None)
}

/// Issue #20 Part 2b: schema-aware variant that projects
/// `ParquetFileMeta::column_stats` (hoisted from the Parquet writer's
/// own row-group metadata at write time) onto Iceberg's five
/// per-column stat maps. When stats are present, emits per field id:
/// - `column_sizes` = compressed bytes per column
/// - `value_counts` = non-null row count
/// - `null_value_counts` = null row count
/// - `lower_bounds` = min value (hex-encoded Iceberg single-value bytes)
/// - `upper_bounds` = max value
///
/// When the file predates #20 Part 2b (no `column_stats` stamped), the
/// projection falls back to the Part 2 behavior: `value_counts` /
/// `null_value_counts` for non-nullable columns only, the other maps
/// empty. Legacy-safe by design.
pub fn to_iceberg_data_file_v2_with_schema(
    entry: &ManifestEntry,
    schema: Option<&TableSchema>,
) -> Value {
    let meta = &entry.meta;

    let mut column_sizes = serde_json::Map::new();
    let mut value_counts = serde_json::Map::new();
    let mut null_value_counts = serde_json::Map::new();
    let mut lower_bounds = serde_json::Map::new();
    let mut upper_bounds = serde_json::Map::new();

    if let Some(stats) = meta.column_stats.as_ref() {
        // Part 2b path: use real per-column stats from the writer.
        for cs in stats {
            let key = cs.field_id.to_string();
            column_sizes.insert(key.clone(), json!(cs.compressed_bytes));
            value_counts.insert(key.clone(), json!(cs.value_count));
            null_value_counts.insert(key.clone(), json!(cs.null_count));
            if let Some(lb) = &cs.lower_bound {
                lower_bounds.insert(key.clone(), json!(hex::encode(lb)));
            }
            if let Some(ub) = &cs.upper_bound {
                upper_bounds.insert(key, json!(hex::encode(ub)));
            }
        }
    } else if let Some(schema) = schema {
        // Legacy / Part 2 path: non-nullable columns are known to be
        // fully populated by merutable's write contract.
        for (idx, col) in schema.columns.iter().enumerate() {
            if col.nullable {
                continue;
            }
            let field_id = (idx + 1) as i32;
            value_counts.insert(field_id.to_string(), json!(meta.num_rows));
            null_value_counts.insert(field_id.to_string(), json!(0));
        }
    }

    // Iceberg `content`: 0=data, 1=position deletes, 2=equality deletes.
    // merutable Parquet files are always data. DVs are carried separately
    // as a Puffin pointer under properties.
    json!({
        "status": manifest_status_code(&entry.status),
        "data_file": {
            "content": 0,
            "file_path": entry.path,
            "file_format": "PARQUET",
            "partition": {},
            "record_count": meta.num_rows,
            "file_size_in_bytes": meta.file_size,
            "column_sizes": column_sizes,
            "value_counts": value_counts,
            "null_value_counts": null_value_counts,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "split_offsets": [],
            // Every data file written by merutable adheres to the
            // PK-ASC sort order (order-id 1). order-id 0 is Iceberg's
            // "unsorted" sentinel.
            "sort_order_id": 1,
        }
    })
}

fn manifest_status_code(status: &str) -> i32 {
    // Iceberg manifest entry status codes: 0=existing, 1=added, 2=deleted.
    match status {
        "added" => 1,
        "deleted" => 2,
        _ => 0,
    }
}

// ── Summary helpers ──────────────────────────────────────────────────────────

fn snapshot_summary(manifest: &Manifest) -> Value {
    let live_files = manifest.live_file_count();
    let total_rows: u64 = manifest
        .entries
        .iter()
        .filter(|e| e.status != "deleted")
        .map(|e| e.meta.num_rows)
        .sum();
    let total_bytes: u64 = manifest
        .entries
        .iter()
        .filter(|e| e.status != "deleted")
        .map(|e| e.meta.file_size)
        .sum();
    json!({
        "operation": "append",
        "total-data-files": live_files.to_string(),
        "total-records": total_rows.to_string(),
        "total-files-size": total_bytes.to_string(),
    })
}

/// Merge the manifest's `properties` map with merutable-specific keys
/// that describe how Iceberg readers should interpret the table.
fn all_properties(manifest: &Manifest) -> Value {
    let mut props = manifest.properties.clone();
    // Source-of-truth marker so a downstream auditor can tell the table
    // was produced by merutable's translator, not hand-crafted.
    props.insert(
        "merutable.translator.version".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    );
    props.insert(
        "merutable.format-version".to_string(),
        manifest.format_version.to_string(),
    );
    // List deletion-vector files so a V3-aware tool can locate them even
    // though the v2 `metadata.json` has no native DV pointer.
    let dv_paths: Vec<String> = manifest
        .entries
        .iter()
        .filter(|e| e.status != "deleted")
        .filter_map(|e| e.dv_path.clone())
        .collect();
    if !dv_paths.is_empty() {
        props.insert("merutable.deletion-vectors".to_string(), dv_paths.join(","));
    }
    serde_json::to_value(props).unwrap_or(json!({}))
}

// ── Convenience helpers ──────────────────────────────────────────────────────

/// Convenience: serialize [`to_iceberg_v2_table_metadata`] to pretty JSON
/// bytes ready to write to disk.
pub fn to_iceberg_v2_table_metadata_bytes(
    manifest: &Manifest,
    table_location: &str,
) -> crate::types::Result<Vec<u8>> {
    let v = to_iceberg_v2_table_metadata(manifest, table_location);
    serde_json::to_vec_pretty(&v)
        .map_err(|e| crate::types::MeruError::Iceberg(format!("iceberg metadata serialize: {e}")))
}

/// Unused summary sanity — kept for type-inferred access to `ParquetFileMeta`.
/// Prevents the compiler from deciding the import is dead if upstream code
/// moves bound fields around.
#[allow(dead_code)]
fn _meta_touch(_m: &ParquetFileMeta) {}

// ── iceberg-rs Schema construction (Issue #54) ──────────────────────────────

/// Convert a merutable [`TableSchema`] into an `iceberg::spec::Schema`.
///
/// Used by [`crate::iceberg::catalog::IcebergCatalog::export_to_iceberg`] to
/// feed the iceberg-rs `ManifestWriter` / `ManifestListWriter` with a
/// Schema object that has the correct field IDs, types, and identifier
/// (primary-key) columns. The mapping is identical to
/// [`to_iceberg_schema_v2`] — same field IDs (1-based), same type
/// projection — so the metadata.json and Avro manifests agree on column
/// identity.
pub(crate) fn to_iceberg_rs_schema(
    schema: &TableSchema,
) -> crate::types::Result<iceberg::spec::Schema> {
    let fields: Vec<Arc<iceberg::spec::NestedField>> = schema
        .columns
        .iter()
        .enumerate()
        .map(|(i, col)| {
            let field_id = (i + 1) as i32;
            let iceberg_type = match col.col_type {
                ColumnType::Boolean => {
                    iceberg::spec::Type::Primitive(iceberg::spec::PrimitiveType::Boolean)
                }
                ColumnType::Int32 => {
                    iceberg::spec::Type::Primitive(iceberg::spec::PrimitiveType::Int)
                }
                ColumnType::Int64 => {
                    iceberg::spec::Type::Primitive(iceberg::spec::PrimitiveType::Long)
                }
                ColumnType::Float => {
                    iceberg::spec::Type::Primitive(iceberg::spec::PrimitiveType::Float)
                }
                ColumnType::Double => {
                    iceberg::spec::Type::Primitive(iceberg::spec::PrimitiveType::Double)
                }
                ColumnType::ByteArray => {
                    iceberg::spec::Type::Primitive(iceberg::spec::PrimitiveType::Binary)
                }
                ColumnType::FixedLenByteArray(n) => {
                    iceberg::spec::Type::Primitive(iceberg::spec::PrimitiveType::Fixed(n as u64))
                }
            };
            if col.nullable {
                Arc::new(iceberg::spec::NestedField::optional(
                    field_id,
                    &col.name,
                    iceberg_type,
                ))
            } else {
                Arc::new(iceberg::spec::NestedField::required(
                    field_id,
                    &col.name,
                    iceberg_type,
                ))
            }
        })
        .collect();

    let identifier_field_ids: Vec<i32> = schema
        .primary_key
        .iter()
        .map(|&idx| (idx + 1) as i32)
        .collect();

    iceberg::spec::Schema::builder()
        .with_schema_id(0)
        .with_fields(fields)
        .with_identifier_field_ids(identifier_field_ids)
        .build()
        .map_err(|e| crate::types::MeruError::Iceberg(format!("iceberg schema build: {e}")))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iceberg::manifest::{Manifest, ManifestEntry};
    use crate::types::{
        level::{Level, ParquetFileMeta},
        schema::{ColumnDef, ColumnType, TableSchema},
    };

    fn schema() -> TableSchema {
        TableSchema {
            table_name: "events".into(),
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
                    name: "score".into(),
                    col_type: ColumnType::Double,
                    nullable: true,

                    ..Default::default()
                },
                ColumnDef {
                    name: "active".into(),
                    col_type: ColumnType::Boolean,
                    nullable: true,

                    ..Default::default()
                },
            ],
            primary_key: vec![0],

            ..Default::default()
        }
    }

    fn file_meta(level: u8, rows: u64, bytes: u64) -> ParquetFileMeta {
        ParquetFileMeta {
            level: Level(level),
            seq_min: 1,
            seq_max: 10,
            key_min: vec![0x01],
            key_max: vec![0xFF],
            num_rows: rows,
            file_size: bytes,
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            format: None,
            column_stats: None,
        }
    }

    fn sample_manifest() -> Manifest {
        let mut m = Manifest::empty(schema());
        // Simulate two commits' worth of state.
        m.snapshot_id = 7;
        m.parent_snapshot_id = Some(6);
        m.sequence_number = 7;
        m.last_updated_ms = 1_700_000_000_000;
        m.table_uuid = "deadbeef-1234-5678-9abc-0123456789ab".to_string();
        m.entries.push(ManifestEntry {
            path: "data/L0/a.parquet".into(),
            meta: file_meta(0, 100, 4096),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "added".into(),
        });
        m.entries.push(ManifestEntry {
            path: "data/L1/b.parquet".into(),
            meta: file_meta(1, 500, 20480),
            dv_path: Some("data/L1/b.dv-5.puffin".into()),
            dv_offset: Some(4),
            dv_length: Some(24),
            status: "existing".into(),
        });
        m.properties
            .insert("merutable.job".into(), "compaction".into());
        m
    }

    #[test]
    fn schema_projection_types() {
        let s = to_iceberg_schema_v2(&schema(), 0);
        let fields = s["fields"].as_array().unwrap();
        assert_eq!(fields.len(), 4);
        assert_eq!(fields[0]["name"], "id");
        assert_eq!(fields[0]["type"], "long");
        assert_eq!(fields[0]["required"], true);
        // pk is field id 1
        assert_eq!(s["identifier-field-ids"], json!([1]));
        assert_eq!(fields[1]["type"], "binary"); // ByteArray
        assert_eq!(fields[1]["required"], false);
        assert_eq!(fields[2]["type"], "double");
        assert_eq!(fields[3]["type"], "boolean");
    }

    #[test]
    fn top_level_fields_match_spec() {
        let m = sample_manifest();
        let v = to_iceberg_v2_table_metadata(&m, "file:///tmp/events");
        assert_eq!(v["format-version"], 2);
        assert_eq!(v["table-uuid"], "deadbeef-1234-5678-9abc-0123456789ab");
        assert_eq!(v["location"], "file:///tmp/events");
        assert_eq!(v["last-sequence-number"], 7);
        assert_eq!(v["last-updated-ms"], 1_700_000_000_000i64);
        assert_eq!(v["last-column-id"], 4);
        assert_eq!(v["current-schema-id"], 0);
        assert_eq!(v["current-snapshot-id"], 7);
        // schemas[]
        assert_eq!(v["schemas"].as_array().unwrap().len(), 1);
        // snapshots[]
        let snaps = v["snapshots"].as_array().unwrap();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0]["snapshot-id"], 7);
        assert_eq!(snaps[0]["parent-snapshot-id"], 6);
        assert_eq!(snaps[0]["sequence-number"], 7);
        // refs.main
        assert_eq!(v["refs"]["main"]["snapshot-id"], 7);
        assert_eq!(v["refs"]["main"]["type"], "branch");
    }

    #[test]
    fn parent_snapshot_absent_for_first_commit() {
        let mut m = sample_manifest();
        m.parent_snapshot_id = None;
        m.snapshot_id = 1;
        let v = to_iceberg_v2_table_metadata(&m, "file:///tmp/events");
        let snap = &v["snapshots"][0];
        assert!(
            snap.get("parent-snapshot-id").is_none(),
            "first commit must omit parent-snapshot-id, got {snap:#?}"
        );
    }

    #[test]
    fn properties_include_merutable_markers() {
        let m = sample_manifest();
        let v = to_iceberg_v2_table_metadata(&m, "file:///tmp/events");
        let props = &v["properties"];
        assert_eq!(props["merutable.job"], "compaction");
        assert!(props["merutable.translator.version"].is_string());
        assert_eq!(props["merutable.format-version"], "3");
        // DV file path surfaces under a merutable-specific key because v2
        // has no native DV representation.
        assert_eq!(props["merutable.deletion-vectors"], "data/L1/b.dv-5.puffin");
    }

    #[test]
    fn summary_counts_live_entries_only() {
        let mut m = sample_manifest();
        // Add a deleted entry that must be excluded from the summary.
        m.entries.push(ManifestEntry {
            path: "data/L0/old.parquet".into(),
            meta: file_meta(0, 99999, 99999),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "deleted".into(),
        });
        let v = to_iceberg_v2_table_metadata(&m, "file:///tmp/events");
        let summary = &v["snapshots"][0]["summary"];
        assert_eq!(summary["operation"], "append");
        // Live: 100 + 500 = 600 rows, 4096 + 20480 = 24576 bytes
        assert_eq!(summary["total-records"], "600");
        assert_eq!(summary["total-files-size"], "24576");
        assert_eq!(summary["total-data-files"], "2");
    }

    #[test]
    fn data_file_projection_preserves_size_and_rows() {
        let entry = ManifestEntry {
            path: "data/L1/foo.parquet".into(),
            meta: file_meta(1, 123, 4567),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "added".into(),
        };
        let v = to_iceberg_data_file_v2(&entry);
        assert_eq!(v["status"], 1); // added
        assert_eq!(v["data_file"]["content"], 0); // data
        assert_eq!(v["data_file"]["file_format"], "PARQUET");
        assert_eq!(v["data_file"]["file_path"], "data/L1/foo.parquet");
        assert_eq!(v["data_file"]["record_count"], 123);
        assert_eq!(v["data_file"]["file_size_in_bytes"], 4567);
    }

    /// Issue #20 Part 1: the emitted sort-order references the PK
    /// columns by Iceberg field id, in the declared PK order, all ASC.
    /// Iceberg-aware engines use this to apply streaming "first row
    /// per partition" for the MVCC dedup projection.
    #[test]
    fn sort_order_projects_primary_key_asc() {
        let m = sample_manifest();
        let v = to_iceberg_v2_table_metadata(&m, "file:///tmp/events");

        // Two sort-orders emitted: 0 (unsorted sentinel) + 1 (real).
        let orders = v["sort-orders"].as_array().unwrap();
        assert_eq!(orders.len(), 2);
        assert_eq!(orders[0]["order-id"], 0);
        assert!(orders[0]["fields"].as_array().unwrap().is_empty());

        let real = &orders[1];
        assert_eq!(real["order-id"], 1);
        let fields = real["fields"].as_array().unwrap();
        // sample_manifest uses primary_key: vec![0] → one PK column,
        // Iceberg field id 1.
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0]["source-id"], 1);
        assert_eq!(fields[0]["direction"], "asc");
        assert_eq!(fields[0]["transform"], "identity");
        // Table-level default-sort-order-id points at our real order.
        assert_eq!(v["default-sort-order-id"], 1);
    }

    /// Issue #20 Part 2 (partial): schema-aware projection emits
    /// `value_counts` and `null_value_counts` for every non-nullable
    /// user column, keyed by Iceberg field id (1..=N). Nullable
    /// columns are omitted pending Part 2b (Parquet row-group stat
    /// plumbing). Hidden merutable columns (`_merutable_ikey`,
    /// `_merutable_seq`, `_merutable_op`, `_merutable_value`) never
    /// appear in Iceberg stats because they're not in the Iceberg
    /// schema.
    #[test]
    fn data_file_projection_emits_value_counts_for_non_nullable_columns() {
        let entry = ManifestEntry {
            path: "data/L1/foo.parquet".into(),
            meta: file_meta(1, 777, 1000),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "added".into(),
        };
        // schema() defines:
        //   col 0 "id"     Int64    required (non-nullable)
        //   col 1 "name"   Binary   nullable
        //   col 2 "score"  Double   nullable
        //   col 3 "active" Boolean  nullable
        // Only "id" (field id 1) should appear in the per-column maps.
        let s = schema();
        let v = to_iceberg_data_file_v2_with_schema(&entry, Some(&s));
        let vc = v["data_file"]["value_counts"].as_object().unwrap();
        let nc = v["data_file"]["null_value_counts"].as_object().unwrap();
        assert_eq!(vc.len(), 1, "only non-nullable columns emit stats");
        assert_eq!(vc["1"], 777, "non-nullable column count = num_rows");
        assert_eq!(nc.len(), 1);
        assert_eq!(nc["1"], 0, "non-nullable column null count = 0");
        // Nullable columns (field ids 2..4) omitted — pending Part 2b.
        assert!(vc.get("2").is_none());
        assert!(vc.get("3").is_none());
        assert!(vc.get("4").is_none());
    }

    /// The schema-less overload remains available for pre-#20 callers;
    /// it emits empty stat maps.
    #[test]
    fn data_file_projection_without_schema_emits_empty_stats() {
        let entry = ManifestEntry {
            path: "data/L1/foo.parquet".into(),
            meta: file_meta(1, 100, 1000),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "added".into(),
        };
        let v = to_iceberg_data_file_v2(&entry);
        assert_eq!(v["data_file"]["value_counts"].as_object().unwrap().len(), 0);
        assert_eq!(
            v["data_file"]["null_value_counts"]
                .as_object()
                .unwrap()
                .len(),
            0
        );
    }

    /// Per-file manifest entries must declare adherence to the real
    /// sort order (id 1), not the unsorted sentinel (id 0).
    #[test]
    fn data_file_adheres_to_pk_sort_order() {
        let entry = ManifestEntry {
            path: "data/L1/foo.parquet".into(),
            meta: file_meta(1, 10, 1000),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "added".into(),
        };
        let v = to_iceberg_data_file_v2(&entry);
        assert_eq!(v["data_file"]["sort_order_id"], 1);
    }

    #[test]
    fn bytes_helper_round_trips_through_serde() {
        let m = sample_manifest();
        let bytes = to_iceberg_v2_table_metadata_bytes(&m, "file:///tmp/events").unwrap();
        // Round-trip through serde to confirm the output is valid JSON.
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["format-version"], 2);
        assert_eq!(parsed["table-uuid"], "deadbeef-1234-5678-9abc-0123456789ab");
    }

    /// The iceberg-rs crate's `TableMetadata` deserializer is the most
    /// faithful compatibility check available in this workspace: if our
    /// emitted JSON parses into its struct without error, we are
    /// compatible at the on-wire level with every V2-aware reader
    /// (pyiceberg, Spark, Trino, DuckDB, Snowflake, Athena).
    #[test]
    fn emitted_json_parses_with_iceberg_crate() {
        let m = sample_manifest();
        let bytes = to_iceberg_v2_table_metadata_bytes(&m, "file:///tmp/events").unwrap();
        let parsed: Result<iceberg::spec::TableMetadata, _> = serde_json::from_slice(&bytes);
        assert!(
            parsed.is_ok(),
            "iceberg-rs rejected our TableMetadata JSON: {:?}\n\npayload:\n{}",
            parsed.err(),
            String::from_utf8_lossy(&bytes)
        );
        let tm = parsed.unwrap();
        // iceberg-rs keeps most fields private — probe only the public
        // accessors. This is still a strong compatibility signal: if
        // the struct constructed at all, every field passed the crate's
        // validation (format version, schema shape, snapshot chain,
        // refs, sort orders, etc.).
        assert_eq!(tm.last_sequence_number(), 7);
        assert_eq!(tm.last_updated_ms(), 1_700_000_000_000i64);
        assert_eq!(tm.current_snapshot_id(), Some(7));
    }
}
