//! `ManifestReader`: reads an Iceberg snapshot and reconstructs the LSM
//! level-file map. Each `DataFile` in the Iceberg manifest carries the
//! `"merutable.meta"` KV footer from which we extract the `Level`.
//!
//! For the embedded (file-system catalog) case, the manifest is a JSON file
//! on disk rather than a full Iceberg catalog scan. We keep the interface
//! generic enough for both paths.

use std::{collections::HashMap, sync::Arc};

use crate::types::{
    level::{Level, ParquetFileMeta},
    schema::TableSchema,
    MeruError, Result,
};

use crate::iceberg::version::{DataFileMeta, Version};

// ── DvLocation ───────────────────────────────────────────────────────────────

/// Real on-storage coordinates of a DV's Puffin blob. Produced by the
/// catalog commit path after writing the Puffin file and passed into
/// [`Manifest::apply`] so that DV pointers in the manifest point at
/// actual byte ranges. Placeholder zeros are NOT acceptable — the
/// earlier design stamped `(0, 0)` and caused deleted rows to reappear
/// on reload.
#[derive(Clone, Debug)]
pub struct DvLocation {
    /// Object-store path of the Puffin file.
    pub dv_path: String,
    /// Byte offset of the roaring-bitmap blob inside the Puffin file.
    pub dv_offset: i64,
    /// Byte length of the roaring-bitmap blob.
    pub dv_length: i64,
}

// ── ManifestEntry ────────────────────────────────────────────────────────────

/// A single file entry as stored in our manifest (simplified Iceberg manifest
/// subset). Full Iceberg catalogs use `DataFile` from the iceberg crate;
/// for the embedded FS catalog we use this lightweight representation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ManifestEntry {
    /// Object-store path of the Parquet file.
    pub path: String,
    /// Serialized `ParquetFileMeta` (same as the Parquet KV footer).
    pub meta: ParquetFileMeta,
    /// `.puffin` DV file path, if any.
    pub dv_path: Option<String>,
    /// Byte offset of the DV blob within the `.puffin` file.
    pub dv_offset: Option<i64>,
    /// Byte length of the DV blob.
    pub dv_length: Option<i64>,
    /// Status: "existing", "added", or "deleted".
    #[serde(default = "default_status")]
    pub status: String,
    /// Iceberg v3 row lineage: first `_row_id` assigned to the first row in
    /// this file. Rows in this file have IDs in `[first_row_id, first_row_id + num_rows)`.
    /// `None` for files written before row lineage was enabled (pre-upgrade).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_row_id: Option<i64>,
}

fn default_status() -> String {
    "existing".to_string()
}

impl ManifestEntry {
    /// Convert to a `DataFileMeta` for the version layer.
    pub fn to_data_file_meta(&self) -> DataFileMeta {
        DataFileMeta {
            path: self.path.clone(),
            meta: self.meta.clone(),
            dv_path: self.dv_path.clone(),
            dv_offset: self.dv_offset,
            dv_length: self.dv_length,
        }
    }
}

// ── Manifest ─────────────────────────────────────────────────────────────────

/// Serializable manifest — a list of file entries plus snapshot metadata.
///
/// This is merutable's native manifest format. It is intentionally a superset
/// of what Apache Iceberg v2 `TableMetadata` carries so that
/// [`crate::iceberg::translate`] can project this struct onto a real Iceberg v2
/// `metadata.json` without loss of information.
///
/// ## Iceberg-facing fields
///
/// The following fields exist purely to make Iceberg translation lossless:
///
/// - `table_uuid`          — Iceberg `table-uuid` (persists across snapshots)
/// - `last_updated_ms`     — Iceberg `last-updated-ms` (set on every commit)
/// - `parent_snapshot_id`  — Iceberg `snapshot.parent-snapshot-id`
/// - `sequence_number`     — Iceberg `last-sequence-number` (bumped per commit)
///
/// All four are `#[serde(default)]` so older merutable deployments whose
/// metadata files don't carry them still load cleanly; the catalog fills in
/// sane defaults on the next commit.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    /// Iceberg format version. Must be 3 for v3 with deletion vectors.
    #[serde(default = "default_format_version")]
    pub format_version: i32,
    /// Stable, per-table identifier. Generated once when the catalog is
    /// first created; every subsequent commit carries the same value.
    /// Maps 1:1 to Iceberg's `table-uuid`.
    #[serde(default)]
    pub table_uuid: String,
    /// Wall-clock epoch-millis at which this manifest was written.
    /// Maps 1:1 to Iceberg's `last-updated-ms`.
    #[serde(default)]
    pub last_updated_ms: i64,
    /// Monotonically increasing snapshot ID.
    pub snapshot_id: i64,
    /// Previous `snapshot_id`, or `None` for the very first commit.
    /// Maps 1:1 to Iceberg snapshot `parent-snapshot-id`.
    #[serde(default)]
    pub parent_snapshot_id: Option<i64>,
    /// Iceberg-style monotonic sequence number. Incremented on every commit.
    /// Maps 1:1 to Iceberg's `last-sequence-number`. This is distinct from
    /// merutable's per-row `seq_num` (which lives inside `ParquetFileMeta`).
    #[serde(default)]
    pub sequence_number: i64,
    /// Iceberg v3 row lineage: the next `_row_id` to allocate. Each commit
    /// claims `[next_row_id, next_row_id + added_rows)` and advances this
    /// counter. Initialized to 0 for new tables; monotonically increasing.
    ///
    /// NOTE: compaction currently allocates fresh row IDs rather than
    /// preserving original assignments. This is a known Phase-1
    /// simplification acceptable for export-only use cases.
    #[serde(default)]
    pub next_row_id: i64,
    /// Iceberg v3 snapshot-level `first-row-id`: the first `_row_id` assigned
    /// in this snapshot's commit. Equal to the predecessor's `next_row_id`.
    /// Required on every v3 snapshot, even if no ID space was allocated.
    #[serde(default)]
    pub first_row_id: i64,
    /// Schema of the table at this snapshot.
    pub schema: TableSchema,
    /// All live file entries (status != "deleted").
    pub entries: Vec<ManifestEntry>,
    /// Snapshot summary properties.
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

fn default_format_version() -> i32 {
    3
}

/// Current wall clock in epoch milliseconds. Kept private and used by
/// [`Manifest::empty`] / [`Manifest::apply`] so every commit stamps a
/// fresh `last_updated_ms` without the caller having to thread a clock
/// through.
pub(crate) fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl Manifest {
    /// Build a `Version` from this manifest.
    pub fn to_version(&self, schema: Arc<TableSchema>) -> Version {
        let mut levels: HashMap<Level, Vec<DataFileMeta>> = HashMap::new();
        for entry in &self.entries {
            if entry.status == "deleted" {
                continue;
            }
            levels
                .entry(entry.meta.level)
                .or_default()
                .push(entry.to_data_file_meta());
        }

        // Sort L0 by seq_max descending (newest first).
        if let Some(l0_files) = levels.get_mut(&Level(0)) {
            l0_files.sort_by_key(|f| std::cmp::Reverse(f.meta.seq_max));
        }
        // Sort L1+ by key_min ascending (for binary search).
        for (level, files) in levels.iter_mut() {
            if level.0 >= 1 {
                files.sort_by(|a, b| a.meta.key_min.cmp(&b.meta.key_min));
            }
        }

        Version {
            snapshot_id: self.snapshot_id,
            levels,
            schema,
        }
    }

    /// Serialize manifest to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self)
            .map_err(|e| MeruError::Iceberg(format!("manifest serialize: {e}")))
    }

    /// Deserialize manifest from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| MeruError::Iceberg(format!("manifest deserialize: {e}")))
    }

    /// Issue #28 Phase 2: serialize this native manifest to the
    /// wire-format protobuf bytes (magic + version + length +
    /// prost-encoded Manifest). The resulting bytes decode back via
    /// [`Manifest::from_protobuf`] with every field preserved.
    ///
    /// The JSON path is still authoritative today; protobuf is an
    /// additional encoding that the catalog will start writing once
    /// Phase 3 wires it into the commit step.
    pub fn to_protobuf(&self) -> Result<Vec<u8>> {
        let pb = self.to_pb()?;
        Ok(crate::iceberg::manifest_pb::encode(&pb))
    }

    /// Issue #28 Phase 2: deserialize from protobuf wire format.
    /// Validates the framing header and then maps every prost field
    /// back onto the native `Manifest` shape.
    pub fn from_protobuf(bytes: &[u8]) -> Result<Self> {
        let pb = crate::iceberg::manifest_pb::decode(bytes)?;
        Self::from_pb(pb)
    }

    /// Internal: project onto the prost `pb::Manifest` message.
    /// Every field in the native struct has a corresponding prost
    /// field number; nothing is silently dropped. `TableSchema` rides
    /// inline as JSON bytes under a well-known property key until
    /// schema evolution #25 promotes schema to a native protobuf
    /// submessage (reserved field in the proto).
    fn to_pb(&self) -> Result<crate::iceberg::manifest_pb::pb::Manifest> {
        let data_files: Vec<crate::iceberg::manifest_pb::pb::DataFileRef> = self
            .entries
            .iter()
            .map(|e| {
                let format_i = e.meta.format.map(|f| match f {
                    crate::types::level::FileFormat::Columnar => 0,
                    crate::types::level::FileFormat::Dual => 1,
                });
                let status_code = match e.status.as_str() {
                    "added" => 1,
                    "deleted" => 2,
                    _ => 0, // existing
                };
                crate::iceberg::manifest_pb::pb::DataFileRef {
                    path: e.path.clone(),
                    file_size: e.meta.file_size as i64,
                    num_rows: e.meta.num_rows as i64,
                    level: e.meta.level.0 as i32,
                    seq_min: e.meta.seq_min as i64,
                    seq_max: e.meta.seq_max as i64,
                    key_min: e.meta.key_min.clone(),
                    key_max: e.meta.key_max.clone(),
                    dv_path: e.dv_path.clone(),
                    dv_offset: e.dv_offset,
                    dv_length: e.dv_length,
                    status: status_code,
                    format: format_i,
                    first_row_id: e.first_row_id,
                }
            })
            .collect();

        // Schema rides as a JSON string under a reserved property
        // key so the prost schema doesn't need a full `TableSchema`
        // submessage yet. Issue #25 will promote schema to a native
        // protobuf message in a later phase.
        let mut properties = self.properties.clone();
        let schema_json = serde_json::to_string(&self.schema)
            .map_err(|e| MeruError::Iceberg(format!("manifest.schema json: {e}")))?;
        properties.insert("merutable.schema.json".to_string(), schema_json);
        properties.insert(
            "merutable.format_version".to_string(),
            self.format_version.to_string(),
        );

        Ok(crate::iceberg::manifest_pb::pb::Manifest {
            snapshot_id: self.snapshot_id,
            sequence_number: self.sequence_number,
            schema_id: self.schema.schema_id as i32,
            partition_spec_id: 0,
            data_files,
            delete_files: Vec::new(),
            previous_snapshot_id: self.parent_snapshot_id,
            table_uuid: self.table_uuid.clone(),
            last_updated_ms: self.last_updated_ms,
            properties,
            last_column_id: self.schema.last_column_id as i32,
            next_row_id: self.next_row_id,
            first_row_id: self.first_row_id,
        })
    }

    /// Internal: reverse direction of `to_pb`.
    fn from_pb(pb: crate::iceberg::manifest_pb::pb::Manifest) -> Result<Self> {
        let mut properties = pb.properties;

        // Pull the schema back out of the reserved property key.
        let schema_json = properties.remove("merutable.schema.json").ok_or_else(|| {
            MeruError::Iceberg(
                "protobuf manifest missing 'merutable.schema.json' property — \
                     cannot reconstruct TableSchema"
                    .into(),
            )
        })?;
        let schema: TableSchema = serde_json::from_str(&schema_json)
            .map_err(|e| MeruError::Iceberg(format!("schema.json parse: {e}")))?;
        let format_version = properties
            .remove("merutable.format_version")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or_else(default_format_version);

        let entries: Result<Vec<ManifestEntry>> = pb
            .data_files
            .into_iter()
            .map(|d| {
                let format = d.format.map(|f| match f {
                    1 => crate::types::level::FileFormat::Dual,
                    _ => crate::types::level::FileFormat::Columnar,
                });
                let status = match d.status {
                    1 => "added".to_string(),
                    2 => "deleted".to_string(),
                    _ => "existing".to_string(),
                };
                Ok(ManifestEntry {
                    path: d.path,
                    meta: crate::types::level::ParquetFileMeta {
                        level: crate::types::level::Level(d.level as u8),
                        seq_min: d.seq_min as u64,
                        seq_max: d.seq_max as u64,
                        key_min: d.key_min,
                        key_max: d.key_max,
                        num_rows: d.num_rows as u64,
                        file_size: d.file_size as u64,
                        dv_path: d.dv_path.clone(),
                        dv_offset: d.dv_offset,
                        dv_length: d.dv_length,
                        format,
                        // Per-column Parquet stats (Issue #20 Part 2b)
                        // are NOT carried through the protobuf path
                        // yet — reserved field numbers in the proto
                        // schema will hold them once wired.
                        column_stats: None,
                    },
                    dv_path: d.dv_path,
                    dv_offset: d.dv_offset,
                    dv_length: d.dv_length,
                    status,
                    first_row_id: d.first_row_id,
                })
            })
            .collect();

        Ok(Self {
            format_version,
            table_uuid: pb.table_uuid,
            last_updated_ms: pb.last_updated_ms,
            snapshot_id: pb.snapshot_id,
            parent_snapshot_id: pb.previous_snapshot_id,
            sequence_number: pb.sequence_number,
            next_row_id: pb.next_row_id,
            first_row_id: pb.first_row_id,
            schema,
            entries: entries?,
            properties,
        })
    }

    /// Create an empty initial manifest with a fresh `table_uuid`.
    /// Snapshot IDs start at 0 (the initial empty snapshot has no parent
    /// and sequence number 0).
    pub fn empty(schema: TableSchema) -> Self {
        Self {
            format_version: 3,
            table_uuid: uuid::Uuid::new_v4().to_string(),
            last_updated_ms: now_ms(),
            snapshot_id: 0,
            parent_snapshot_id: None,
            sequence_number: 0,
            next_row_id: 0,
            first_row_id: 0,
            schema,
            entries: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Create an empty manifest with a caller-supplied `table_uuid`. Used
    /// by the catalog when reopening a legacy on-disk manifest that has
    /// no `table_uuid` field — we want the first post-upgrade commit to
    /// mint one deterministically so all subsequent snapshots agree.
    pub fn empty_with_uuid(schema: TableSchema, table_uuid: String) -> Self {
        Self {
            format_version: 3,
            table_uuid,
            last_updated_ms: now_ms(),
            snapshot_id: 0,
            parent_snapshot_id: None,
            sequence_number: 0,
            next_row_id: 0,
            first_row_id: 0,
            schema,
            entries: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Apply a `SnapshotTransaction` to produce a new manifest.
    /// This is the core commit logic for the embedded FS catalog.
    ///
    /// `dv_locations` carries the real `(dv_path, dv_offset, dv_length)`
    /// for every DV in `txn.dvs`. The catalog computes these after
    /// writing the Puffin file to object storage and must pass them in
    /// here. If `txn.dvs` contains a key that is not present in
    /// `dv_locations`, this is a programmer bug and returns an error
    /// (previously the manifest silently stamped zeros, which caused
    /// every DV to be invisible on reload).
    pub fn apply(
        &self,
        txn: &crate::iceberg::snapshot::SnapshotTransaction,
        new_snapshot_id: i64,
        dv_locations: &HashMap<String, DvLocation>,
    ) -> Result<Self> {
        // Sanity-check: every DV in the transaction must have a real
        // location recorded. A missing entry means the caller forgot to
        // upload the puffin file or forgot to thread the offsets back.
        for path in txn.dvs.keys() {
            if !dv_locations.contains_key(path) {
                return Err(MeruError::Iceberg(format!(
                    "apply: DV for '{path}' missing from dv_locations — \
                     commit path must upload the puffin file and record \
                     its blob_offset/blob_length before applying"
                )));
            }
        }

        // IMP-04: conflict detection — a transaction must not both add a DV
        // for a file and remove that same file.  This is a logical error in
        // the compaction/flush path (the DV would be silently discarded).
        for dv_path in txn.dvs.keys() {
            if txn.removes.contains(dv_path) {
                return Err(MeruError::Iceberg(format!(
                    "apply: transaction both adds DV and removes file '{dv_path}' — \
                     this is a conflict; retry against the current manifest"
                )));
            }
        }

        let remove_set: std::collections::HashSet<&str> =
            txn.removes.iter().map(|s| s.as_str()).collect();

        let mut new_entries: Vec<ManifestEntry> = Vec::new();

        // Carry forward existing entries that aren't removed.
        for entry in &self.entries {
            if entry.status == "deleted" {
                continue;
            }
            if remove_set.contains(entry.path.as_str()) {
                continue; // fully compacted — drop
            }
            let mut e = entry.clone();
            // Apply DV update if present. The location map is the
            // source of truth for dv_path/offset/length.
            if txn.dvs.contains_key(&entry.path) {
                let loc = &dv_locations[&entry.path];
                e.dv_path = Some(loc.dv_path.clone());
                e.dv_offset = Some(loc.dv_offset);
                e.dv_length = Some(loc.dv_length);
                // Mirror into the embedded ParquetFileMeta so readers
                // that consult the file-level metadata see the same
                // coordinates as readers that consult the manifest.
                e.meta.dv_path = e.dv_path.clone();
                e.meta.dv_offset = e.dv_offset;
                e.meta.dv_length = e.dv_length;
            }
            new_entries.push(e);
        }

        // Iceberg v3 row lineage: assign first_row_id to existing files
        // that predate row lineage (upgraded tables). Per spec, "any null
        // first_row_id must be assigned via inheritance" in the first commit
        // after upgrade.
        let mut cursor = self.next_row_id;
        for entry in &mut new_entries {
            if entry.first_row_id.is_none() {
                entry.first_row_id = Some(cursor);
                cursor += entry.meta.num_rows as i64;
            }
        }

        // Add new files with row ID allocation.
        for add in &txn.adds {
            let first_row_id = cursor;
            cursor += add.num_rows as i64;
            new_entries.push(ManifestEntry {
                path: add.path.clone(),
                meta: add.meta.clone(),
                dv_path: None,
                dv_offset: None,
                dv_length: None,
                status: "added".to_string(),
                first_row_id: Some(first_row_id),
            });
        }

        let mut props = self.properties.clone();
        props.extend(txn.props.iter().map(|(k, v)| (k.clone(), v.clone())));

        // Preserve table_uuid across snapshots. If the predecessor manifest
        // was a legacy one with no uuid (all zeros or empty string), mint a
        // fresh v4 — this path runs at most once per table upgrade.
        let table_uuid = if self.table_uuid.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            self.table_uuid.clone()
        };
        // parent_snapshot_id is the predecessor's snapshot_id, UNLESS the
        // predecessor was the initial empty manifest (snapshot_id=0) in
        // which case Iceberg expects None on the first real snapshot.
        let parent_snapshot_id = if self.snapshot_id == 0 {
            None
        } else {
            Some(self.snapshot_id)
        };

        Ok(Manifest {
            format_version: self.format_version,
            table_uuid,
            last_updated_ms: now_ms(),
            snapshot_id: new_snapshot_id,
            parent_snapshot_id,
            // Iceberg sequence number is monotonic — bump by one on every
            // commit, regardless of how many files the transaction touched.
            sequence_number: self.sequence_number + 1,
            next_row_id: cursor,
            first_row_id: self.next_row_id,
            schema: self.schema.clone(),
            entries: new_entries,
            properties: props,
        })
    }

    /// Number of live (non-deleted) file entries.
    pub fn live_file_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.status != "deleted")
            .count()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iceberg::deletion_vector::DeletionVector;
    use crate::iceberg::snapshot::{IcebergDataFile, SnapshotTransaction};
    use crate::types::{
        level::{Level, ParquetFileMeta},
        schema::{ColumnDef, ColumnType, TableSchema},
    };

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

    fn test_meta(
        level: u8,
        seq_min: u64,
        seq_max: u64,
        key_min: &[u8],
        key_max: &[u8],
    ) -> ParquetFileMeta {
        ParquetFileMeta {
            level: Level(level),
            seq_min,
            seq_max,
            key_min: key_min.to_vec(),
            key_max: key_max.to_vec(),
            num_rows: 100,
            file_size: 1024,
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            format: None,
            column_stats: None,
        }
    }

    #[test]
    fn empty_manifest_roundtrip() {
        let m = Manifest::empty(test_schema());
        assert_eq!(
            m.format_version, 3,
            "new manifests must be format-version 3"
        );
        let json = m.to_json().unwrap();
        let decoded = Manifest::from_json(&json).unwrap();
        assert_eq!(decoded.format_version, 3);
        assert_eq!(decoded.snapshot_id, 0);
        assert_eq!(decoded.entries.len(), 0);
        assert_eq!(decoded.schema.table_name, "test");
    }

    /// Issue #28 Phase 2: protobuf round-trip preserves every field
    /// the JSON path does — table uuid, timestamps, snapshot chain,
    /// entries, properties, and schema.
    #[test]
    fn protobuf_roundtrip_empty_manifest() {
        let m = Manifest::empty(test_schema());
        let bytes = m.to_protobuf().unwrap();
        let decoded = Manifest::from_protobuf(&bytes).unwrap();
        assert_eq!(decoded.format_version, m.format_version);
        assert_eq!(decoded.table_uuid, m.table_uuid);
        assert_eq!(decoded.snapshot_id, m.snapshot_id);
        assert_eq!(decoded.sequence_number, m.sequence_number);
        assert_eq!(decoded.parent_snapshot_id, m.parent_snapshot_id);
        assert_eq!(decoded.entries.len(), 0);
        assert_eq!(decoded.schema.table_name, m.schema.table_name);
    }

    /// Protobuf round-trip with a populated entry — exercises the
    /// DataFileRef mapping (level, seqs, keys, DV pointers, format
    /// stamp, status).
    #[test]
    fn protobuf_roundtrip_with_entries() {
        let mut m = Manifest::empty(test_schema());
        m.snapshot_id = 42;
        m.sequence_number = 7;
        m.parent_snapshot_id = Some(41);
        m.entries.push(ManifestEntry {
            path: "data/L0/abc.parquet".into(),
            meta: crate::types::level::ParquetFileMeta {
                level: crate::types::level::Level(0),
                seq_min: 1,
                seq_max: 100,
                key_min: vec![0x01, 0x02],
                key_max: vec![0xFE, 0xFF],
                num_rows: 500,
                file_size: 8192,
                dv_path: Some("data/L0/abc.dv-1.puffin".into()),
                dv_offset: Some(16),
                dv_length: Some(64),
                format: Some(crate::types::level::FileFormat::Dual),
                column_stats: None,
            },
            dv_path: Some("data/L0/abc.dv-1.puffin".into()),
            dv_offset: Some(16),
            dv_length: Some(64),
            status: "added".into(),
            first_row_id: None,
        });
        m.properties.insert("merutable.job".into(), "flush".into());

        let bytes = m.to_protobuf().unwrap();
        let decoded = Manifest::from_protobuf(&bytes).unwrap();

        assert_eq!(decoded.snapshot_id, 42);
        assert_eq!(decoded.sequence_number, 7);
        assert_eq!(decoded.parent_snapshot_id, Some(41));
        assert_eq!(decoded.entries.len(), 1);
        let e = &decoded.entries[0];
        assert_eq!(e.path, "data/L0/abc.parquet");
        assert_eq!(e.status, "added");
        assert_eq!(e.meta.level, crate::types::level::Level(0));
        assert_eq!(e.meta.seq_min, 1);
        assert_eq!(e.meta.seq_max, 100);
        assert_eq!(e.meta.key_min, vec![0x01, 0x02]);
        assert_eq!(e.meta.key_max, vec![0xFE, 0xFF]);
        assert_eq!(e.meta.num_rows, 500);
        assert_eq!(e.meta.file_size, 8192);
        assert_eq!(e.dv_path.as_deref(), Some("data/L0/abc.dv-1.puffin"));
        assert_eq!(e.dv_offset, Some(16));
        assert_eq!(e.dv_length, Some(64));
        assert_eq!(e.meta.format, Some(crate::types::level::FileFormat::Dual));
        assert_eq!(
            decoded.properties.get("merutable.job").map(|s| s.as_str()),
            Some("flush")
        );
        // Internal-only reserved property keys do NOT leak through
        // to the public properties map after deserialize.
        assert!(!decoded.properties.contains_key("merutable.schema.json"));
        assert!(!decoded.properties.contains_key("merutable.format_version"));
    }

    /// Every protobuf-encoded manifest starts with the "MRUB" magic,
    /// distinguishing it unambiguously from JSON — a dual-read
    /// catalog in Phase 3 will sniff this byte to pick the decoder.
    #[test]
    fn protobuf_bytes_have_mrub_magic() {
        let m = Manifest::empty(test_schema());
        let bytes = m.to_protobuf().unwrap();
        assert_eq!(&bytes[0..4], b"MRUB");
    }

    /// Deserializing a legacy manifest (no format_version field) defaults to 3.
    #[test]
    fn legacy_manifest_defaults_to_v3() {
        let json = r#"{"snapshot_id":5,"schema":{"table_name":"t","columns":[],"primary_key":[]},"entries":[]}"#;
        let m = Manifest::from_json(json.as_bytes()).unwrap();
        assert_eq!(m.format_version, 3);
    }

    /// `apply` carries forward format_version.
    #[test]
    fn apply_preserves_format_version() {
        let m = Manifest::empty(test_schema());
        assert_eq!(m.format_version, 3);
        let txn = SnapshotTransaction::new();
        let m2 = m.apply(&txn, 1, &HashMap::new()).unwrap();
        assert_eq!(m2.format_version, 3);
    }

    #[test]
    fn apply_flush_txn() {
        let m = Manifest::empty(test_schema());
        let mut txn = SnapshotTransaction::new();
        txn.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
        });
        txn.set_prop("merutable.job", "flush");

        let m2 = m.apply(&txn, 1, &HashMap::new()).unwrap();
        assert_eq!(m2.snapshot_id, 1);
        assert_eq!(m2.live_file_count(), 1);
        assert_eq!(m2.entries[0].path, "data/L0/a.parquet");
        assert_eq!(m2.properties.get("merutable.job").unwrap(), "flush");
    }

    #[test]
    fn apply_compaction_with_remove() {
        // Start with 2 L0 files.
        let mut m = Manifest::empty(test_schema());
        m.snapshot_id = 1;
        m.entries.push(ManifestEntry {
            path: "data/L0/a.parquet".into(),
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });
        m.entries.push(ManifestEntry {
            path: "data/L0/b.parquet".into(),
            meta: test_meta(0, 11, 20, b"\x03", b"\x08"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });

        // Compact both into one L1 file.
        let mut txn = SnapshotTransaction::new();
        txn.remove_file("data/L0/a.parquet".into());
        txn.remove_file("data/L0/b.parquet".into());
        txn.add_file(IcebergDataFile {
            path: "data/L1/merged.parquet".into(),
            file_size: 2048,
            num_rows: 200,
            meta: test_meta(1, 1, 20, b"\x01", b"\x08"),
        });

        let m2 = m.apply(&txn, 2, &HashMap::new()).unwrap();
        assert_eq!(m2.snapshot_id, 2);
        assert_eq!(m2.live_file_count(), 1);
        assert_eq!(m2.entries[0].path, "data/L1/merged.parquet");
    }

    #[test]
    fn apply_partial_compaction_with_dv() {
        let mut m = Manifest::empty(test_schema());
        m.snapshot_id = 1;
        m.entries.push(ManifestEntry {
            path: "data/L0/a.parquet".into(),
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });

        let mut txn = SnapshotTransaction::new();
        let mut dv = DeletionVector::new();
        dv.mark_deleted(0);
        dv.mark_deleted(5);
        dv.mark_deleted(10);
        txn.add_dv("data/L0/a.parquet".into(), dv);
        txn.add_file(IcebergDataFile {
            path: "data/L1/promoted.parquet".into(),
            file_size: 512,
            num_rows: 3,
            meta: test_meta(1, 1, 10, b"\x01", b"\x03"),
        });

        // Supply real on-storage coordinates for the DV, matching what
        // the catalog commit path would produce after writing the
        // puffin file.
        let mut dv_locs = HashMap::new();
        dv_locs.insert(
            "data/L0/a.parquet".to_string(),
            DvLocation {
                dv_path: "data/L0/a.dv-2.puffin".to_string(),
                dv_offset: 4,
                dv_length: 24,
            },
        );
        let m2 = m.apply(&txn, 2, &dv_locs).unwrap();
        assert_eq!(m2.live_file_count(), 2);
        // L0 file still exists and carries the REAL DV coordinates
        // (not placeholder zeros — regression for the bug where apply
        // stamped (0, 0) and every deleted row reappeared on reload).
        let l0_entry = m2
            .entries
            .iter()
            .find(|e| e.path == "data/L0/a.parquet")
            .unwrap();
        assert_eq!(
            l0_entry.dv_path.as_deref(),
            Some("data/L0/a.dv-2.puffin"),
            "dv_path must come from the location map, not be reconstructed"
        );
        assert_eq!(l0_entry.dv_offset, Some(4));
        assert_eq!(l0_entry.dv_length, Some(24));
        // Mirrored into the embedded ParquetFileMeta so both the
        // manifest view and the file-level view agree.
        assert_eq!(l0_entry.meta.dv_offset, Some(4));
        assert_eq!(l0_entry.meta.dv_length, Some(24));
    }

    /// `apply` must refuse to stamp a DV whose on-storage location has
    /// not been provided. Previously it silently filled in zeros; the
    /// zeros then made it to disk and every deleted row came back
    /// after reload. This test pins the refusal.
    #[test]
    fn apply_errors_when_dv_location_missing() {
        let mut m = Manifest::empty(test_schema());
        m.entries.push(ManifestEntry {
            path: "data/L0/a.parquet".into(),
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });

        let mut txn = SnapshotTransaction::new();
        let mut dv = DeletionVector::new();
        dv.mark_deleted(0);
        txn.add_dv("data/L0/a.parquet".into(), dv);

        let err = m.apply(&txn, 2, &HashMap::new()).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("data/L0/a.parquet") && msg.contains("dv_locations"),
            "error must name the missing file and the missing map: {msg}"
        );
    }

    #[test]
    fn to_version_sort_order() {
        let mut m = Manifest::empty(test_schema());
        m.snapshot_id = 5;
        // L0 files with different seq_max — should be sorted DESC.
        m.entries.push(ManifestEntry {
            path: "l0_old.parquet".into(),
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });
        m.entries.push(ManifestEntry {
            path: "l0_new.parquet".into(),
            meta: test_meta(0, 11, 20, b"\x03", b"\x08"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });
        // L1 files — should be sorted by key_min ASC.
        m.entries.push(ManifestEntry {
            path: "l1_b.parquet".into(),
            meta: test_meta(1, 1, 20, b"\x05", b"\x0A"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });
        m.entries.push(ManifestEntry {
            path: "l1_a.parquet".into(),
            meta: test_meta(1, 1, 20, b"\x01", b"\x04"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });

        let v = m.to_version(Arc::new(test_schema()));
        // L0: newest first (seq_max=20 before seq_max=10).
        let l0 = v.files_at(Level(0));
        assert_eq!(l0[0].path, "l0_new.parquet");
        assert_eq!(l0[1].path, "l0_old.parquet");
        // L1: sorted by key_min ASC.
        let l1 = v.files_at(Level(1));
        assert_eq!(l1[0].path, "l1_a.parquet");
        assert_eq!(l1[1].path, "l1_b.parquet");
    }

    /// IMP-04 regression: a transaction that both adds a DV for a file and
    /// removes that same file must be rejected — the DV would be silently
    /// discarded. This catches logic errors in the compaction path.
    #[test]
    fn apply_rejects_dv_add_and_file_remove_conflict() {
        let mut m = Manifest::empty(test_schema());
        m.entries.push(ManifestEntry {
            path: "data/L0/a.parquet".into(),
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });

        let mut txn = SnapshotTransaction::new();
        // Both add DV and remove the same file — conflict.
        let mut dv = DeletionVector::new();
        dv.mark_deleted(0);
        txn.add_dv("data/L0/a.parquet".into(), dv);
        txn.remove_file("data/L0/a.parquet".into());

        let mut dv_locs = HashMap::new();
        dv_locs.insert(
            "data/L0/a.parquet".to_string(),
            DvLocation {
                dv_path: "data/L0/a.dv-2.puffin".to_string(),
                dv_offset: 4,
                dv_length: 24,
            },
        );

        let err = m.apply(&txn, 2, &dv_locs).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("conflict"),
            "error must indicate a conflict: {msg}"
        );
    }

    /// Iceberg v3 row lineage: `apply` must assign monotonically increasing
    /// `first_row_id` to every file and advance `next_row_id` by the total
    /// number of rows added across all files in the transaction.
    #[test]
    fn apply_assigns_row_lineage() {
        let m = Manifest::empty(test_schema());
        assert_eq!(m.next_row_id, 0);

        // Commit 1: add a 100-row file.
        let mut txn1 = SnapshotTransaction::new();
        txn1.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
        });
        let m2 = m.apply(&txn1, 1, &HashMap::new()).unwrap();
        assert_eq!(m2.next_row_id, 100);
        assert_eq!(m2.first_row_id, 0);
        assert_eq!(m2.entries[0].first_row_id, Some(0));

        // Commit 2: add a 200-row file. Row IDs continue from 100.
        let mut txn2 = SnapshotTransaction::new();
        txn2.add_file(IcebergDataFile {
            path: "data/L0/b.parquet".into(),
            file_size: 2048,
            num_rows: 200,
            meta: test_meta(0, 11, 20, b"\x03", b"\x08"),
        });
        let m3 = m2.apply(&txn2, 2, &HashMap::new()).unwrap();
        assert_eq!(m3.next_row_id, 300);
        // Snapshot-level first_row_id = predecessor's next_row_id.
        assert_eq!(m3.first_row_id, 100);
        let b_entry = m3.entries.iter().find(|e| e.path == "data/L0/b.parquet").unwrap();
        assert_eq!(b_entry.first_row_id, Some(100));
        // Existing file retains its original row ID.
        let a_entry = m3.entries.iter().find(|e| e.path == "data/L0/a.parquet").unwrap();
        assert_eq!(a_entry.first_row_id, Some(0));
    }

    /// Row lineage: compaction removes old files and adds new ones. The new
    /// files get fresh row IDs (since the rows are physically rewritten).
    /// `next_row_id` never goes backward.
    #[test]
    fn apply_row_lineage_through_compaction() {
        let m = Manifest::empty(test_schema());

        // Add two L0 files (100 + 200 rows).
        let mut txn1 = SnapshotTransaction::new();
        txn1.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
        });
        txn1.add_file(IcebergDataFile {
            path: "data/L0/b.parquet".into(),
            file_size: 2048,
            num_rows: 200,
            meta: test_meta(0, 11, 20, b"\x03", b"\x08"),
        });
        let m2 = m.apply(&txn1, 1, &HashMap::new()).unwrap();
        assert_eq!(m2.next_row_id, 300);

        // Compact both into one L1 file with 300 rows.
        let mut txn2 = SnapshotTransaction::new();
        txn2.remove_file("data/L0/a.parquet".into());
        txn2.remove_file("data/L0/b.parquet".into());
        txn2.add_file(IcebergDataFile {
            path: "data/L1/merged.parquet".into(),
            file_size: 3072,
            num_rows: 300,
            meta: test_meta(1, 1, 20, b"\x01", b"\x08"),
        });
        let m3 = m2.apply(&txn2, 2, &HashMap::new()).unwrap();
        // New file gets row IDs starting from where we left off (300).
        assert_eq!(m3.next_row_id, 600);
        assert_eq!(m3.entries[0].first_row_id, Some(300));
        assert_eq!(m3.entries[0].path, "data/L1/merged.parquet");
    }

    /// Row lineage: upgrading a legacy manifest (next_row_id=0, entries
    /// with first_row_id=None) must assign row IDs to all existing files
    /// on the first commit.
    #[test]
    fn apply_assigns_row_ids_to_legacy_files_on_upgrade() {
        let mut m = Manifest::empty(test_schema());
        m.snapshot_id = 5;
        m.next_row_id = 0;
        // Simulate legacy files that have no row ID.
        m.entries.push(ManifestEntry {
            path: "data/L0/legacy1.parquet".into(),
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });
        m.entries.push(ManifestEntry {
            path: "data/L1/legacy2.parquet".into(),
            meta: test_meta(1, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: None,
        });

        // First commit after upgrade adds a new file.
        let mut txn = SnapshotTransaction::new();
        txn.add_file(IcebergDataFile {
            path: "data/L0/new.parquet".into(),
            file_size: 512,
            num_rows: 50,
            meta: test_meta(0, 11, 15, b"\x06", b"\x09"),
        });
        let m2 = m.apply(&txn, 6, &HashMap::new()).unwrap();

        // Legacy files got row IDs assigned (100 rows each from test_meta).
        let legacy1 = m2.entries.iter().find(|e| e.path == "data/L0/legacy1.parquet").unwrap();
        assert_eq!(legacy1.first_row_id, Some(0));
        let legacy2 = m2.entries.iter().find(|e| e.path == "data/L1/legacy2.parquet").unwrap();
        assert_eq!(legacy2.first_row_id, Some(100));
        // New file gets IDs after both legacy files.
        let new_entry = m2.entries.iter().find(|e| e.path == "data/L0/new.parquet").unwrap();
        assert_eq!(new_entry.first_row_id, Some(200));
        // next_row_id = 200 (legacy) + 50 (new) = 250.
        assert_eq!(m2.next_row_id, 250);
    }

    /// Row lineage round-trips through protobuf serialization.
    #[test]
    fn row_lineage_survives_protobuf_roundtrip() {
        let mut m = Manifest::empty(test_schema());
        m.snapshot_id = 3;
        m.next_row_id = 500;
        m.first_row_id = 300;
        m.entries.push(ManifestEntry {
            path: "data/L0/a.parquet".into(),
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: Some(200),
        });

        let bytes = m.to_protobuf().unwrap();
        let decoded = Manifest::from_protobuf(&bytes).unwrap();
        assert_eq!(decoded.next_row_id, 500);
        assert_eq!(decoded.first_row_id, 300);
        assert_eq!(decoded.entries[0].first_row_id, Some(200));
    }

    /// Row lineage round-trips through JSON serialization.
    #[test]
    fn row_lineage_survives_json_roundtrip() {
        let mut m = Manifest::empty(test_schema());
        m.snapshot_id = 3;
        m.next_row_id = 500;
        m.first_row_id = 300;
        m.entries.push(ManifestEntry {
            path: "data/L0/a.parquet".into(),
            meta: test_meta(0, 1, 10, b"\x01", b"\x05"),
            dv_path: None,
            dv_offset: None,
            dv_length: None,
            status: "existing".into(),
            first_row_id: Some(200),
        });

        let json = m.to_json().unwrap();
        let decoded = Manifest::from_json(&json).unwrap();
        assert_eq!(decoded.next_row_id, 500);
        assert_eq!(decoded.first_row_id, 300);
        assert_eq!(decoded.entries[0].first_row_id, Some(200));
    }

    /// Legacy JSON manifests with no row lineage fields default to 0/None.
    #[test]
    fn legacy_json_without_row_lineage_loads_cleanly() {
        let json = r#"{
            "snapshot_id": 5,
            "schema": {"table_name":"t","columns":[],"primary_key":[]},
            "entries": [{
                "path": "data/L0/old.parquet",
                "meta": {"level":0,"seq_min":1,"seq_max":10,"key_min":"01","key_max":"ff","num_rows":100,"file_size":1024},
                "status": "existing"
            }]
        }"#;
        let m = Manifest::from_json(json.as_bytes()).unwrap();
        assert_eq!(m.next_row_id, 0);
        assert_eq!(m.entries[0].first_row_id, None);
    }
}
