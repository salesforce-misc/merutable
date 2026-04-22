//! `IcebergCatalog`: file-system-based Iceberg catalog for embedded use.
//!
//! Layout on disk:
//! ```text
//! {base_path}/
//! ├── metadata/
//! │   ├── v1.metadata.json      # manifest snapshots
//! │   ├── v2.metadata.json
//! │   └── ...
//! ├── data/
//! │   ├── L0/
//! │   │   ├── {uuid}.parquet
//! │   │   └── {uuid}.dv-{snap}.puffin
//! │   ├── L1/
//! │   │   └── ...
//! │   └── ...
//! └── version-hint.text         # current metadata version pointer
//! ```
//!
//! This is a file-system catalog that writes merutable's **native manifest
//! format** — a JSON superset of Iceberg's `TableMetadata` — rather than
//! Iceberg's on-wire format (`TableMetadata` JSON + Avro manifest-list +
//! Avro manifest files). The native format is chosen for efficiency: one
//! fsyncable JSON file per commit instead of four, no Avro dependency on
//! the hot path.
//!
//! The on-disk layout is **losslessly translatable** to a real Apache
//! Iceberg v2 table via [`crate::iceberg::translate`] and the
//! [`IcebergCatalog::export_to_iceberg`] method. The export produces a
//! complete Iceberg v2 table — `metadata.json` + manifest-list Avro +
//! manifest Avro — that DuckDB, pyiceberg, Spark, Trino, Snowflake,
//! and Athena can read directly. See the translate module's docs for
//! the mapping from merutable fields to Iceberg spec fields.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::types::{level::Level, schema::TableSchema, MeruError, Result};
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::iceberg::{
    deletion_vector::DeletionVector,
    manifest::{DvLocation, Manifest},
    snapshot::SnapshotTransaction,
    version::Version,
};

// ── IcebergCatalog ───────────────────────────────────────────────────────────

/// File-system Iceberg catalog. Manages manifest JSON files and the
/// version pointer. Thread-safe: commit serialized via `Mutex`.
pub struct IcebergCatalog {
    base_path: PathBuf,
    /// Current manifest (the source of truth on disk).
    current: Mutex<Manifest>,
    /// Next metadata version number.
    next_version: Mutex<i64>,
}

/// Issue #42 / #44: compatibility check for reopen.
///
/// Two shapes are accepted:
///   1. Identical schema (the #42 baseline).
///   2. Additive extension (#44 Stage 1): provided is a strict
///      superset of persisted — the first `persisted.columns.len()`
///      entries match byte-for-byte, and each newly-appended column
///      is either nullable or carries a `write_default` /
///      `initial_default` so existing data can be back-filled.
///
/// Every other shape (rename, reorder, type change, PK change,
/// column removal) is rejected.
pub(crate) fn check_schema_compatible(
    persisted: &TableSchema,
    provided: &TableSchema,
) -> Result<()> {
    if persisted.table_name != provided.table_name {
        return Err(MeruError::SchemaMismatch(format!(
            "catalog at this path was created for table `{}`; reopen provided \
             `{}`. A merutable catalog is single-table — refusing to reopen \
             under a conflicting name.",
            persisted.table_name, provided.table_name,
        )));
    }
    // Provided must cover at least every persisted column (can only
    // grow, never shrink or replace).
    if provided.columns.len() < persisted.columns.len() {
        return Err(MeruError::SchemaMismatch(format!(
            "schema mismatch on reopen of table `{}`: persisted has {} \
             columns, provided has {}. Column removal is not supported \
             (additive-only evolution — #44).",
            persisted.table_name,
            persisted.columns.len(),
            provided.columns.len(),
        )));
    }
    // Prefix must match exactly. Columns are positional — reorder or
    // type change breaks existing files.
    for (i, (pc, vc)) in persisted
        .columns
        .iter()
        .zip(provided.columns.iter())
        .enumerate()
    {
        if pc.name != vc.name {
            return Err(MeruError::SchemaMismatch(format!(
                "schema mismatch on reopen of table `{}`: column {} is named \
                 `{}` in the persisted schema, but `{}` in the provided schema. \
                 Columns are positional and cannot be renamed or reordered.",
                persisted.table_name, i, pc.name, vc.name,
            )));
        }
        if pc.col_type != vc.col_type {
            return Err(MeruError::SchemaMismatch(format!(
                "schema mismatch on reopen of table `{}`: column `{}` has \
                 type {:?} persisted, but {:?} provided.",
                persisted.table_name, pc.name, pc.col_type, vc.col_type,
            )));
        }
        if pc.nullable != vc.nullable {
            return Err(MeruError::SchemaMismatch(format!(
                "schema mismatch on reopen of table `{}`: column `{}` has \
                 nullable={} persisted, but {} provided.",
                persisted.table_name, pc.name, pc.nullable, vc.nullable,
            )));
        }
    }
    // Newly-appended columns must be back-fillable against the
    // existing data — nullable, or carry a non-null default.
    for new_col in &provided.columns[persisted.columns.len()..] {
        if !new_col.nullable && new_col.write_default.is_none() && new_col.initial_default.is_none()
        {
            return Err(MeruError::SchemaMismatch(format!(
                "schema evolution on reopen of table `{}`: new column `{}` \
                 is non-nullable and has no write_default / initial_default. \
                 Cannot back-fill existing rows — additive evolution requires \
                 nullable OR a default (#44).",
                persisted.table_name, new_col.name,
            )));
        }
    }
    if persisted.primary_key != provided.primary_key {
        return Err(MeruError::SchemaMismatch(format!(
            "schema mismatch on reopen of table `{}`: primary key was {:?} \
             persisted, provided as {:?}. PK cannot change post-creation.",
            persisted.table_name, persisted.primary_key, provided.primary_key,
        )));
    }
    Ok(())
}

/// Issue #42: read-only helper that loads the `TableSchema`
/// persisted in an existing catalog's latest manifest, without
/// running the reopen-compatibility check. Used by admin/migration
/// tools that need to discover the schema before calling the
/// ordinary `IcebergCatalog::open(path, schema)` — such tools
/// inspect catalog contents without knowing the original schema
/// a priori and would otherwise need a placeholder that the #42
/// check would (correctly) reject.
///
/// Returns `Ok(None)` when the directory has no catalog yet
/// (i.e., no `version-hint.text`). Returns `Err` on a corrupted
/// manifest or I/O failure.
pub async fn load_persisted_schema(base_path: impl AsRef<Path>) -> Result<Option<TableSchema>> {
    let base = base_path.as_ref();
    let hint_path = base.join("version-hint.text");
    if !hint_path.exists() {
        return Ok(None);
    }
    let hint = tokio::fs::read_to_string(&hint_path)
        .await
        .map_err(MeruError::Io)?;
    let ver: i64 = hint
        .trim()
        .parse()
        .map_err(|_| MeruError::Corruption("bad version-hint".into()))?;
    let data = read_manifest_payload(&base.join("metadata"), ver).await?;
    let manifest = decode_manifest(&data)?;
    Ok(Some(manifest.schema))
}

impl IcebergCatalog {
    /// Open or create a catalog at `base_path`.
    /// If the directory already has metadata, loads the latest manifest.
    /// Otherwise, creates an empty initial manifest.
    pub async fn open(base_path: impl AsRef<Path>, schema: TableSchema) -> Result<Self> {
        let base = base_path.as_ref().to_path_buf();

        // Ensure directory structure exists.
        let metadata_dir = base.join("metadata");
        let data_dir = base.join("data");
        tokio::fs::create_dir_all(&metadata_dir)
            .await
            .map_err(MeruError::Io)?;
        tokio::fs::create_dir_all(&data_dir)
            .await
            .map_err(MeruError::Io)?;

        // Crash-recovery housekeeping: remove any leftover
        // `version-hint.text.tmp` from a commit that crashed after the
        // write but before the rename. Leaving it around is harmless
        // (it's never read), but cleaning up keeps the base dir tidy
        // and prevents ambiguity for human operators.
        let tmp_hint = base.join("version-hint.text.tmp");
        if tmp_hint.exists() {
            let _ = tokio::fs::remove_file(&tmp_hint).await;
        }

        // Try to load existing manifest from version-hint.
        let hint_path = base.join("version-hint.text");
        let (manifest, next_ver) = if hint_path.exists() {
            let hint = tokio::fs::read_to_string(&hint_path)
                .await
                .map_err(MeruError::Io)?;
            let ver: i64 = hint
                .trim()
                .parse()
                .map_err(|_| MeruError::Corruption("bad version-hint".into()))?;
            // Issue #28 Phase 3: dual-read. Prefer .pb (protobuf)
            // if present; fall back to .json. Both decoders then
            // produce the same native `Manifest` struct.
            let data = read_manifest_payload(&metadata_dir, ver).await?;
            let mut manifest = decode_manifest(&data)?;
            // Legacy-upgrade path: manifests written by pre-Iceberg-enrichment
            // merutable carry neither a `table_uuid` nor `last_updated_ms`.
            // Mint the uuid here — `apply()` will carry it forward to every
            // future snapshot. `last_updated_ms` stays at whatever the old
            // manifest had (0 by default) until the next commit stamps a
            // real clock.
            if manifest.table_uuid.is_empty() {
                manifest.table_uuid = uuid::Uuid::new_v4().to_string();
            }
            // Issue #42: enforce the single-table invariant at the
            // API boundary on reopen. Before this check, a caller
            // could open `./db` with schema `{table_name:"events"}`
            // and reopen the same path with
            // `{table_name:"logs"}` — the engine silently accepted
            // the mismatch and subsequent commits overwrote the
            // persisted schema, corrupting reads that depended on
            // the original schema.
            //
            // Contract for 0.1: the provided schema must match the
            // persisted schema exactly. Full evolution rules are
            // #44's scope; here we reject every mismatch so a stale
            // caller cannot write under a conflicting shape.
            check_schema_compatible(&manifest.schema, &schema)?;
            (manifest, ver + 1)
        } else {
            // No version-hint.text. Detect silent data loss: if the
            // metadata/ directory already contains snapshot files, this
            // means version-hint was lost (manual deletion, bad restore,
            // filesystem corruption) and initializing to an empty
            // manifest would orphan the existing snapshots. Error out
            // so the operator can recover explicitly rather than
            // silently clobbering data.
            let mut has_existing_metadata = false;
            if let Ok(mut entries) = tokio::fs::read_dir(&metadata_dir).await {
                while let Ok(Some(e)) = entries.next_entry().await {
                    let name = e.file_name();
                    let n = name.to_string_lossy();
                    // Issue #28 Phase 3: recognise both the legacy
                    // `.metadata.json` and the new `.metadata.pb`.
                    if n.starts_with('v')
                        && (n.ends_with(".metadata.json") || n.ends_with(".metadata.pb"))
                    {
                        has_existing_metadata = true;
                        break;
                    }
                }
            }
            if has_existing_metadata {
                return Err(MeruError::Corruption(format!(
                    "version-hint.text is missing but {}/ contains snapshot \
                     metadata files — refusing to initialize a fresh catalog \
                     over existing data. Restore version-hint.text from backup \
                     or move the existing metadata/ directory aside.",
                    metadata_dir.display(),
                )));
            }
            (Manifest::empty(schema), 1)
        };

        info!(base_path = %base.display(), "catalog opened");
        Ok(Self {
            base_path: base,
            current: Mutex::new(manifest),
            next_version: Mutex::new(next_ver),
        })
    }

    /// Commit a `SnapshotTransaction`. This is the linearization point
    /// for every flush and compaction.
    ///
    /// 1. Upload DV puffin files for partial compactions.
    /// 2. Apply transaction to current manifest → new manifest.
    /// 3. Write new manifest JSON to `metadata/v{N}.metadata.json`.
    /// 4. Atomically update `version-hint.text`.
    /// 5. Return the new `Version`.
    pub async fn commit(
        &self,
        txn: &SnapshotTransaction,
        schema: Arc<TableSchema>,
    ) -> Result<Version> {
        let mut current = self.current.lock().await;
        let mut next_ver = self.next_version.lock().await;

        let new_snapshot_id = *next_ver;
        debug!(snapshot_id = new_snapshot_id, "committing snapshot");

        // IMP-06: track puffin files written during this commit so we can
        // delete them if any later step fails (prevents orphaned blobs).
        let mut pending_puffin_files: Vec<std::path::PathBuf> = Vec::new();

        // Upload puffin files for DV updates AND record their real
        // on-storage blob coordinates so the manifest can point at the
        // exact byte range of each roaring-bitmap blob. Skipping this
        // step was a real bug: the manifest used to stamp (0, 0)
        // placeholders and every deleted row reappeared on reload.
        let mut dv_locations: HashMap<String, DvLocation> = HashMap::new();
        for (parquet_path, new_dv) in &txn.dvs {
            // If the file already has a DV from a prior partial compaction,
            // load it and union with the new DV. Without this, the second
            // partial compaction's DV replaces the first and rows deleted
            // in the first compaction silently reappear.
            let merged_dv = match current
                .entries
                .iter()
                .find(|e| e.status != "deleted" && e.path == *parquet_path)
            {
                Some(entry)
                    if entry.dv_path.is_some()
                        && entry.dv_offset.is_some()
                        && entry.dv_length.is_some() =>
                {
                    let existing_puffin_path = self.base_path.join(entry.dv_path.as_ref().unwrap());
                    let puffin_data = tokio::fs::read(&existing_puffin_path)
                        .await
                        .map_err(MeruError::Io)?;
                    let offset_raw = entry.dv_offset.unwrap();
                    let length_raw = entry.dv_length.unwrap();
                    if offset_raw < 0 || length_raw < 0 {
                        return Err(MeruError::Corruption(format!(
                            "existing DV for '{}' has negative offset ({offset_raw}) or length ({length_raw})",
                            parquet_path,
                        )));
                    }
                    let offset = offset_raw as usize;
                    let length = length_raw as usize;
                    let end = offset.checked_add(length).ok_or_else(|| {
                        MeruError::Corruption(format!(
                            "existing DV for '{}' offset {offset} + length {length} overflows usize",
                            parquet_path,
                        ))
                    })?;
                    if end > puffin_data.len() {
                        return Err(MeruError::Corruption(format!(
                            "existing DV blob for '{}' at offset {offset} length {length} \
                             exceeds puffin file size {}",
                            parquet_path,
                            puffin_data.len()
                        )));
                    }
                    let existing_dv =
                        DeletionVector::from_puffin_blob(&puffin_data[offset..offset + length])?;
                    let existing_card = existing_dv.cardinality();
                    let new_card = new_dv.cardinality();
                    let mut merged = existing_dv;
                    merged.union_with(new_dv);

                    // IMP-17: a union can never shrink — if it does, the
                    // bitmap library dropped bits and deleted rows will
                    // silently reappear.
                    let merged_card = merged.cardinality();
                    let min_expected = existing_card.max(new_card);
                    if merged_card < min_expected {
                        return Err(MeruError::Corruption(format!(
                            "DV union for '{}' shrank: existing={existing_card} new={new_card} \
                             merged={merged_card}",
                            parquet_path,
                        )));
                    }

                    merged
                }
                _ => new_dv.clone(),
            };

            let encoded =
                merged_dv.encode_puffin(parquet_path, new_snapshot_id, new_snapshot_id)?;
            let puffin_filename = format!(
                "{}.dv-{new_snapshot_id}.puffin",
                Path::new(parquet_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
            );
            // Determine the level directory from the parquet path.
            let level_dir = Path::new(parquet_path)
                .parent()
                .unwrap_or(Path::new("data/L0"));
            let rel_puffin_path = level_dir.join(&puffin_filename);
            let abs_puffin_path = self.base_path.join(&rel_puffin_path);
            // Ensure parent dir exists.
            if let Some(parent) = abs_puffin_path.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(MeruError::Io)?;
            }
            tokio::fs::write(&abs_puffin_path, &encoded.bytes)
                .await
                .map_err(MeruError::Io)?;
            pending_puffin_files.push(abs_puffin_path.clone());
            // fsync the puffin file so deleted-row bitmaps survive a crash.
            tokio::fs::File::open(&abs_puffin_path)
                .await
                .map_err(MeruError::Io)?
                .sync_all()
                .await
                .map_err(MeruError::Io)?;
            // fsync the parent directory so the new directory entry is durable.
            if let Some(parent) = abs_puffin_path.parent() {
                tokio::fs::File::open(parent)
                    .await
                    .map_err(MeruError::Io)?
                    .sync_all()
                    .await
                    .map_err(MeruError::Io)?;
            }

            dv_locations.insert(
                parquet_path.clone(),
                DvLocation {
                    dv_path: rel_puffin_path.to_string_lossy().into_owned(),
                    dv_offset: encoded.blob_offset,
                    dv_length: encoded.blob_length,
                },
            );
        }

        // Apply transaction to produce new manifest. `apply` validates
        // that every DV in `txn.dvs` has a matching entry in
        // `dv_locations`; it errors out if the caller forgot any, so
        // the zero-placeholder bug cannot recur silently.
        //
        // IMP-06: everything below is wrapped so that on any error the
        // puffin files written above are cleaned up (best-effort).
        let result: Result<Version> = async {
            let new_manifest = current.apply(txn, new_snapshot_id, &dv_locations)?;
            let metadata_dir = self.base_path.join("metadata");

            // Issue #28 Phase 5: protobuf is the only canonical
            // write format. JSON reads remain supported in
            // `read_manifest_payload` for back-compat with
            // catalogs committed before #28 Phase 4 (when JSON
            // was the only format) or with Phase 4 dual-write
            // catalogs that are still in transition. New commits
            // never write JSON.
            let meta_path_pb = metadata_dir.join(format!("v{new_snapshot_id}.metadata.pb"));
            let pb_bytes = new_manifest.to_protobuf()?;
            tokio::fs::write(&meta_path_pb, &pb_bytes)
                .await
                .map_err(MeruError::Io)?;
            tokio::fs::File::open(&meta_path_pb)
                .await
                .map_err(MeruError::Io)?
                .sync_all()
                .await
                .map_err(MeruError::Io)?;

            // fsync the metadata DIRECTORY so both files' directory
            // entries are durably linked BEFORE the version-hint
            // points at them. Without this, a crash between the
            // version-hint rename and the metadata dir fsync can
            // leave version-hint pointing at filenames that aren't
            // yet in the directory listing (ext4/btrfs journal it
            // separately), and recovery fails with "metadata not
            // found".
            tokio::fs::File::open(&metadata_dir)
                .await
                .map_err(MeruError::Io)?
                .sync_all()
                .await
                .map_err(MeruError::Io)?;

            // Update version hint (atomic: write tmp + rename).
            let hint_path = self.base_path.join("version-hint.text");
            let tmp_hint = self.base_path.join("version-hint.text.tmp");
            tokio::fs::write(&tmp_hint, new_snapshot_id.to_string())
                .await
                .map_err(MeruError::Io)?;
            // fsync the tmp hint file before rename so the content is durable.
            tokio::fs::File::open(&tmp_hint)
                .await
                .map_err(MeruError::Io)?
                .sync_all()
                .await
                .map_err(MeruError::Io)?;
            tokio::fs::rename(&tmp_hint, &hint_path)
                .await
                .map_err(MeruError::Io)?;
            // fsync the base directory so the version-hint rename is
            // durably linked — without this, the rename can "roll back"
            // on a crash and readers see the old version.
            tokio::fs::File::open(&self.base_path)
                .await
                .map_err(MeruError::Io)?
                .sync_all()
                .await
                .map_err(MeruError::Io)?;

            // Build Version from the new manifest.
            let version = new_manifest.to_version(schema);

            // Update in-memory state.
            *current = new_manifest;
            *next_ver = new_snapshot_id + 1;

            Ok(version)
        }
        .await;

        // IMP-06: on commit failure, best-effort cleanup of orphaned puffin
        // files that were written but never referenced by a manifest.
        if result.is_err() {
            for puffin_path in &pending_puffin_files {
                let _ = tokio::fs::remove_file(puffin_path).await;
            }
        }

        result
    }

    /// Re-read the current manifest from disk. Used by read-only replicas
    /// to pick up new snapshots written by the primary.
    ///
    /// Reads `version-hint.text` → loads the corresponding manifest via
    /// `read_manifest_payload` (prefers `.pb`, falls back to legacy
    /// `.json`) → updates the in-memory manifest. Returns the new
    /// `Version`.
    pub async fn refresh(&self, schema: Arc<TableSchema>) -> Result<Version> {
        let hint_path = self.base_path.join("version-hint.text");
        let hint = tokio::fs::read_to_string(&hint_path)
            .await
            .map_err(MeruError::Io)?;
        let ver: i64 = hint
            .trim()
            .parse()
            .map_err(|_| MeruError::Corruption("bad version-hint on refresh".into()))?;

        // Issue #28 Phase 5: prefer the protobuf payload, fall back
        // to JSON for legacy catalogs. Matches the open-path
        // dual-read behavior; before Phase 5 this path would
        // hard-code `.json` and miss pb-only commits.
        let metadata_dir = self.base_path.join("metadata");
        let data = read_manifest_payload(&metadata_dir, ver).await?;
        let manifest = decode_manifest(&data)?;
        let version = manifest.to_version(schema);

        let mut current = self.current.lock().await;
        *current = manifest;
        let mut next_ver = self.next_version.lock().await;
        *next_ver = ver + 1;

        Ok(version)
    }

    /// Get the current manifest (for inspection/debugging).
    pub async fn current_manifest(&self) -> Manifest {
        self.current.lock().await.clone()
    }

    /// Get the base path.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Construct the data file path for a new Parquet file.
    pub fn data_file_path(&self, level: Level, file_id: &str) -> PathBuf {
        self.base_path
            .join("data")
            .join(format!("L{}", level.0))
            .join(format!("{file_id}.parquet"))
    }

    /// Ensure the data directory for a level exists.
    pub async fn ensure_level_dir(&self, level: Level) -> Result<()> {
        let dir = self.base_path.join("data").join(format!("L{}", level.0));
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(MeruError::Io)?;
        Ok(())
    }

    /// Export the current catalog snapshot as a complete Apache Iceberg v2
    /// table: `metadata.json` + manifest-list Avro + manifest Avro.
    ///
    /// This is the **enabler artifact** for Iceberg interop: it projects
    /// the in-memory `Manifest` onto the Iceberg v2 `TableMetadata` shape
    /// (see [`crate::iceberg::translate`]) and writes:
    ///
    /// 1. `{target_dir}/metadata/v{N}.metadata.json` — Iceberg v2 table
    ///    metadata referencing the manifest-list.
    /// 2. `{target_dir}/metadata/snap-{N}-0-{uuid}.avro` — manifest-list
    ///    Avro file containing one entry per manifest.
    /// 3. `{target_dir}/metadata/{uuid}-m0.avro` — manifest Avro file
    ///    listing every live data file.
    /// 4. `{target_dir}/version-hint.text` — pointer to the latest
    ///    metadata version.
    ///
    /// After this call, a DuckDB `iceberg_scan('{target_dir}')`, pyiceberg,
    /// Spark, Trino, Snowflake, or Athena client can read the table
    /// (read-only; data files live under the original `base_path`).
    ///
    /// # Path resolution
    ///
    /// Data file paths in the manifest Avro are absolute URIs rooted at
    /// the catalog's `base_path` (e.g. `file:///db/data/L0/a.parquet`).
    /// Manifest and manifest-list paths are absolute URIs under
    /// `target_dir`. This allows `target_dir != base_path` — the export
    /// directory holds Iceberg metadata while data files remain in-place.
    pub async fn export_to_iceberg(&self, target_dir: impl AsRef<Path>) -> Result<PathBuf> {
        let target = target_dir.as_ref().to_path_buf();
        let meta_dir = target.join("metadata");
        tokio::fs::create_dir_all(&meta_dir)
            .await
            .map_err(MeruError::Io)?;

        // Snapshot the current manifest under the lock, then release it
        // so the projection work runs off the critical path.
        let manifest = {
            let guard = self.current.lock().await;
            guard.clone()
        };

        // Use an absolute `file://` URI so downstream Iceberg readers
        // resolve the table location consistently regardless of cwd.
        let canonical = tokio::fs::canonicalize(&self.base_path)
            .await
            .unwrap_or_else(|_| self.base_path.clone());
        let table_location = format!("file://{}", canonical.display());

        // Canonical target for Avro file URIs.
        let canonical_target = tokio::fs::canonicalize(&target)
            .await
            .unwrap_or_else(|_| target.clone());
        let target_uri = format!("file://{}", canonical_target.display());

        // ── Issue #54: Avro manifest + manifest-list ────────────────

        // Write manifest Avro and manifest-list Avro so that downstream
        // engines (DuckDB iceberg_scan, pyiceberg, Spark) can discover
        // data files through the standard Iceberg read path.
        let manifest_list_avro_path = self
            .write_iceberg_avro_manifests(&manifest, &table_location, &target_uri)
            .await?;

        // ── metadata.json ───────────────────────────────────────────

        // Build the metadata JSON. The manifest-list path inside the
        // snapshot must point to the Avro file we just wrote (under
        // target_dir), not to the default path derived from
        // table_location.
        let mut metadata_value =
            crate::iceberg::translate::to_iceberg_v2_table_metadata(&manifest, &table_location);
        // Patch the manifest-list path in the snapshot to point at the
        // actual Avro file under target_dir.
        if let Some(snap) = metadata_value
            .get_mut("snapshots")
            .and_then(|s| s.as_array_mut())
            .and_then(|a| a.first_mut())
        {
            snap["manifest-list"] = serde_json::json!(manifest_list_avro_path);
        }

        let json = serde_json::to_vec_pretty(&metadata_value)
            .map_err(|e| MeruError::Iceberg(format!("iceberg metadata serialize: {e}")))?;

        let out_path = meta_dir.join(format!("v{}.metadata.json", manifest.snapshot_id));
        tokio::fs::write(&out_path, &json)
            .await
            .map_err(MeruError::Io)?;
        // fsync before updating version-hint.
        tokio::fs::File::open(&out_path)
            .await
            .map_err(MeruError::Io)?
            .sync_all()
            .await
            .map_err(MeruError::Io)?;

        let hint_path = target.join("version-hint.text");
        tokio::fs::write(&hint_path, manifest.snapshot_id.to_string())
            .await
            .map_err(MeruError::Io)?;
        tokio::fs::File::open(&hint_path)
            .await
            .map_err(MeruError::Io)?
            .sync_all()
            .await
            .map_err(MeruError::Io)?;

        info!(
            target = %target.display(),
            snapshot_id = manifest.snapshot_id,
            table_uuid = %manifest.table_uuid,
            "exported Iceberg v2 table (metadata.json + Avro manifests)"
        );
        Ok(out_path)
    }

    /// Issue #54: write the manifest Avro and manifest-list Avro files
    /// using the iceberg-rs crate's `ManifestWriter` and
    /// `ManifestListWriter`. Returns the absolute URI of the
    /// manifest-list Avro file (for embedding in metadata.json).
    async fn write_iceberg_avro_manifests(
        &self,
        manifest: &Manifest,
        table_location: &str,
        target_uri: &str,
    ) -> Result<String> {
        use std::sync::Arc;

        // Convert merutable schema → iceberg-rs Schema.
        let iceberg_schema = crate::iceberg::translate::to_iceberg_rs_schema(&manifest.schema)?;
        let iceberg_schema_ref = Arc::new(iceberg_schema);

        // Unpartitioned partition spec.
        let partition_spec = iceberg::spec::PartitionSpec::builder(iceberg_schema_ref.clone())
            .with_spec_id(0)
            .build()
            .map_err(|e| MeruError::Iceberg(format!("partition spec build: {e}")))?;

        // Manifest metadata.
        let metadata = iceberg::spec::ManifestMetadata::builder()
            .schema(iceberg_schema_ref)
            .schema_id(0)
            .partition_spec(partition_spec)
            .format_version(iceberg::spec::FormatVersion::V2)
            .content(iceberg::spec::ManifestContentType::Data)
            .build();

        // Build manifest entries from merutable's live files.
        let entries: Vec<iceberg::spec::ManifestEntry> = manifest
            .entries
            .iter()
            .filter(|e| e.status != "deleted")
            .map(|e| {
                let status = match e.status.as_str() {
                    "added" => iceberg::spec::ManifestStatus::Added,
                    _ => iceberg::spec::ManifestStatus::Existing,
                };
                // Data file paths are absolute URIs rooted at the
                // catalog's base_path so DuckDB / pyiceberg can
                // resolve them without knowing the original base.
                let file_path = format!("{}/{}", table_location.trim_end_matches('/'), e.path);
                let data_file = iceberg::spec::DataFileBuilder::default()
                    .content(iceberg::spec::DataContentType::Data)
                    .file_path(file_path)
                    .file_format(iceberg::spec::DataFileFormat::Parquet)
                    .file_size_in_bytes(e.meta.file_size)
                    .record_count(e.meta.num_rows)
                    .partition(iceberg::spec::Struct::from_iter(std::iter::empty::<
                        Option<iceberg::spec::Literal>,
                    >()))
                    .build()
                    .map_err(|e| MeruError::Iceberg(format!("data file build: {e}")))?;
                Ok(iceberg::spec::ManifestEntry::builder()
                    .status(status)
                    .snapshot_id(manifest.snapshot_id)
                    .sequence_number(manifest.sequence_number)
                    .file_sequence_number(manifest.sequence_number)
                    .data_file(data_file)
                    .build())
            })
            .collect::<Result<Vec<_>>>()?;

        let iceberg_manifest = iceberg::spec::Manifest::new(metadata, entries);

        // Local filesystem I/O for writing Avro files.
        let file_io = iceberg::io::FileIOBuilder::new_fs_io()
            .build()
            .map_err(|e| MeruError::Iceberg(format!("FileIO build: {e}")))?;

        // ── Write manifest Avro ─────────────────────────────────────
        let manifest_uuid = uuid::Uuid::new_v4().to_string();
        let manifest_filename = format!("{manifest_uuid}-m0.avro");
        let manifest_avro_uri = format!(
            "{}/metadata/{manifest_filename}",
            target_uri.trim_end_matches('/')
        );
        let manifest_output = file_io
            .new_output(&manifest_avro_uri)
            .map_err(|e| MeruError::Iceberg(format!("manifest output: {e}")))?;
        let manifest_writer =
            iceberg::spec::ManifestWriter::new(manifest_output, manifest.snapshot_id, vec![]);
        let manifest_file = manifest_writer
            .write(iceberg_manifest)
            .await
            .map_err(|e| MeruError::Iceberg(format!("manifest Avro write: {e}")))?;

        // ── Write manifest-list Avro ────────────────────────────────
        let manifest_list_filename = format!(
            "snap-{}-0-{}.avro",
            manifest.snapshot_id, manifest.table_uuid
        );
        let manifest_list_uri = format!(
            "{}/metadata/{manifest_list_filename}",
            target_uri.trim_end_matches('/')
        );
        let manifest_list_output = file_io
            .new_output(&manifest_list_uri)
            .map_err(|e| MeruError::Iceberg(format!("manifest-list output: {e}")))?;
        let mut manifest_list_writer = iceberg::spec::ManifestListWriter::v2(
            manifest_list_output,
            manifest.snapshot_id,
            manifest.parent_snapshot_id,
            manifest.sequence_number,
        );
        manifest_list_writer
            .add_manifests(std::iter::once(manifest_file))
            .map_err(|e| MeruError::Iceberg(format!("manifest-list add: {e}")))?;
        manifest_list_writer
            .close()
            .await
            .map_err(|e| MeruError::Iceberg(format!("manifest-list close: {e}")))?;

        Ok(manifest_list_uri)
    }
}

// ── Dual-read helpers (Issue #28 Phase 3) ────────────────────────────────────

/// Locate the on-disk bytes for version `ver`. Prefers
/// `v{ver}.metadata.pb` (the Issue #28 Phase-3+ protobuf format).
/// Falls back to `v{ver}.metadata.json` (legacy) when the `.pb`
/// file is absent. Returns the raw bytes so the caller can dispatch
/// on magic — this keeps the I/O step separate from the decode step
/// for cleaner error attribution.
async fn read_manifest_payload(metadata_dir: &std::path::Path, ver: i64) -> Result<Vec<u8>> {
    let pb_path = metadata_dir.join(format!("v{ver}.metadata.pb"));
    if pb_path.exists() {
        return tokio::fs::read(&pb_path).await.map_err(MeruError::Io);
    }
    let json_path = metadata_dir.join(format!("v{ver}.metadata.json"));
    tokio::fs::read(&json_path).await.map_err(MeruError::Io)
}

/// Decode a manifest payload by sniffing the leading bytes. The
/// protobuf wire format always starts with the "MRUB" magic; JSON
/// always starts with `{`. Anything else is a corrupted header and
/// surfaces as `MeruError::Corruption` rather than a misleading
/// parse error deep inside the decoder.
fn decode_manifest(data: &[u8]) -> Result<Manifest> {
    if data.len() >= 4 && &data[0..4] == b"MRUB" {
        Manifest::from_protobuf(data)
    } else if data.first().copied() == Some(b'{') {
        Manifest::from_json(data)
    } else {
        let head: Vec<u8> = data.iter().take(8).copied().collect();
        Err(MeruError::Corruption(format!(
            "manifest payload has neither \"MRUB\" magic nor leading '{{': head={head:02X?}"
        )))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iceberg::deletion_vector::DeletionVector;
    use crate::iceberg::snapshot::IcebergDataFile;
    use crate::types::{
        level::{Level, ParquetFileMeta},
        schema::{ColumnDef, ColumnType},
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

    fn test_meta(level: u8) -> ParquetFileMeta {
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

    #[tokio::test]
    async fn open_creates_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let _catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();
        assert!(tmp.path().join("metadata").exists());
        assert!(tmp.path().join("data").exists());
    }

    #[tokio::test]
    async fn commit_flush_and_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let schema = Arc::new(test_schema());

        // Open, commit a flush.
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();
        let mut txn = SnapshotTransaction::new();
        txn.add_file(IcebergDataFile {
            path: "data/L0/abc.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0),
        });
        txn.set_prop("merutable.job", "flush");
        let v = catalog.commit(&txn, schema.clone()).await.unwrap();
        assert_eq!(v.snapshot_id, 1);
        assert_eq!(v.l0_file_count(), 1);

        // Reopen from disk.
        let catalog2 = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();
        let m = catalog2.current_manifest().await;
        assert_eq!(m.snapshot_id, 1);
        assert_eq!(m.live_file_count(), 1);
    }

    #[tokio::test]
    async fn multiple_commits() {
        let tmp = tempfile::tempdir().unwrap();
        let schema = Arc::new(test_schema());
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();

        // Commit 1: flush file A.
        let mut txn1 = SnapshotTransaction::new();
        txn1.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0),
        });
        let v1 = catalog.commit(&txn1, schema.clone()).await.unwrap();
        assert_eq!(v1.snapshot_id, 1);

        // Commit 2: flush file B.
        let mut txn2 = SnapshotTransaction::new();
        txn2.add_file(IcebergDataFile {
            path: "data/L0/b.parquet".into(),
            file_size: 2048,
            num_rows: 200,
            meta: {
                let mut m = test_meta(0);
                m.seq_min = 11;
                m.seq_max = 20;
                m
            },
        });
        let v2 = catalog.commit(&txn2, schema.clone()).await.unwrap();
        assert_eq!(v2.snapshot_id, 2);
        assert_eq!(v2.l0_file_count(), 2);

        // Commit 3: compact both into L1.
        let mut txn3 = SnapshotTransaction::new();
        txn3.remove_file("data/L0/a.parquet".into());
        txn3.remove_file("data/L0/b.parquet".into());
        txn3.add_file(IcebergDataFile {
            path: "data/L1/merged.parquet".into(),
            file_size: 3072,
            num_rows: 300,
            meta: test_meta(1),
        });
        let v3 = catalog.commit(&txn3, schema.clone()).await.unwrap();
        assert_eq!(v3.snapshot_id, 3);
        assert_eq!(v3.l0_file_count(), 0);
        assert_eq!(v3.files_at(Level(1)).len(), 1);
    }

    #[tokio::test]
    async fn commit_with_dv() {
        let tmp = tempfile::tempdir().unwrap();
        let schema = Arc::new(test_schema());
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();

        // Flush a file.
        let mut txn1 = SnapshotTransaction::new();
        txn1.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0),
        });
        catalog.commit(&txn1, schema.clone()).await.unwrap();

        // Partial compaction — promote some rows, add DV to source.
        let mut txn2 = SnapshotTransaction::new();
        let mut dv = DeletionVector::new();
        for i in 0..50u32 {
            dv.mark_deleted(i);
        }
        txn2.add_dv("data/L0/a.parquet".into(), dv);
        txn2.add_file(IcebergDataFile {
            path: "data/L1/promoted.parquet".into(),
            file_size: 512,
            num_rows: 50,
            meta: test_meta(1),
        });

        let v = catalog.commit(&txn2, schema.clone()).await.unwrap();
        assert_eq!(v.l0_file_count(), 1); // L0 file still present
        let l0 = &v.files_at(Level(0))[0];
        assert!(l0.dv_path.is_some()); // but now has a DV
        assert_eq!(v.files_at(Level(1)).len(), 1);
    }

    /// Regression test: two successive partial compactions on the same
    /// file must produce a DV that is the union of both. Before the fix,
    /// the second DV replaced the first and rows deleted in the first
    /// partial compaction silently reappeared.
    #[tokio::test]
    async fn successive_dv_updates_are_unioned() {
        let tmp = tempfile::tempdir().unwrap();
        let schema = Arc::new(test_schema());
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();

        // Flush a file with 100 rows.
        let mut txn1 = SnapshotTransaction::new();
        txn1.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0),
        });
        catalog.commit(&txn1, schema.clone()).await.unwrap();

        // First partial compaction: delete rows 0..10.
        let mut txn2 = SnapshotTransaction::new();
        let mut dv1 = DeletionVector::new();
        for i in 0..10u32 {
            dv1.mark_deleted(i);
        }
        txn2.add_dv("data/L0/a.parquet".into(), dv1);
        txn2.add_file(IcebergDataFile {
            path: "data/L1/batch1.parquet".into(),
            file_size: 256,
            num_rows: 10,
            meta: test_meta(1),
        });
        let v2 = catalog.commit(&txn2, schema.clone()).await.unwrap();
        let l0_after_first = &v2.files_at(Level(0))[0];
        assert!(l0_after_first.dv_path.is_some());

        // Read back the DV puffin written by the first partial compaction
        // to confirm it has exactly 10 deleted rows.
        let puffin_path_1 = tmp.path().join(l0_after_first.dv_path.as_ref().unwrap());
        let puffin_data_1 = tokio::fs::read(&puffin_path_1).await.unwrap();
        let dv_after_first = DeletionVector::from_puffin_bytes(&puffin_data_1).unwrap();
        assert_eq!(dv_after_first.cardinality(), 10);

        // Second partial compaction: delete rows 50..60 (disjoint from
        // the first set). The commit must union with the existing DV.
        let mut txn3 = SnapshotTransaction::new();
        let mut dv2 = DeletionVector::new();
        for i in 50..60u32 {
            dv2.mark_deleted(i);
        }
        txn3.add_dv("data/L0/a.parquet".into(), dv2);
        txn3.add_file(IcebergDataFile {
            path: "data/L1/batch2.parquet".into(),
            file_size: 256,
            num_rows: 10,
            meta: {
                let mut m = test_meta(1);
                m.seq_min = 11;
                m.seq_max = 20;
                m
            },
        });
        let v3 = catalog.commit(&txn3, schema.clone()).await.unwrap();

        // The L0 file must still exist with a DV.
        assert_eq!(v3.l0_file_count(), 1);
        let l0_after_second = &v3.files_at(Level(0))[0];
        assert!(l0_after_second.dv_path.is_some());

        // Read back the DV puffin and verify it is the UNION of both
        // partial compactions: rows 0..10 AND 50..60 = 20 deleted rows.
        let puffin_path_2 = tmp.path().join(l0_after_second.dv_path.as_ref().unwrap());
        let puffin_data_2 = tokio::fs::read(&puffin_path_2).await.unwrap();
        let dv_merged = DeletionVector::from_puffin_bytes(&puffin_data_2).unwrap();
        assert_eq!(
            dv_merged.cardinality(),
            20,
            "DV must be union of both partial compactions (10 + 10 = 20)"
        );
        // Verify specific row positions from both compactions.
        for i in 0..10u32 {
            assert!(
                dv_merged.is_deleted(i),
                "row {i} from first compaction missing"
            );
        }
        for i in 50..60u32 {
            assert!(
                dv_merged.is_deleted(i),
                "row {i} from second compaction missing"
            );
        }
        // Rows outside both ranges must NOT be deleted.
        for i in 10..50u32 {
            assert!(!dv_merged.is_deleted(i), "row {i} should not be deleted");
        }
    }

    /// Every commit must stamp a non-zero `table_uuid`, bump
    /// `sequence_number`, set `parent_snapshot_id`, and roll
    /// `last_updated_ms` forward. These are the fields `crate::iceberg::translate`
    /// relies on for lossless Iceberg v2 projection.
    #[tokio::test]
    async fn commit_enriches_iceberg_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let schema = Arc::new(test_schema());
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();

        // Commit #1.
        let mut txn1 = SnapshotTransaction::new();
        txn1.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0),
        });
        catalog.commit(&txn1, schema.clone()).await.unwrap();
        let m1 = catalog.current_manifest().await;

        assert!(
            !m1.table_uuid.is_empty(),
            "first commit must mint a table_uuid"
        );
        assert!(m1.last_updated_ms > 0, "last_updated_ms must be set");
        assert_eq!(m1.sequence_number, 1);
        assert_eq!(m1.parent_snapshot_id, None, "first commit has no parent");

        // Commit #2 — must carry uuid, bump sequence, point parent at #1.
        let mut txn2 = SnapshotTransaction::new();
        txn2.add_file(IcebergDataFile {
            path: "data/L0/b.parquet".into(),
            file_size: 2048,
            num_rows: 200,
            meta: {
                let mut m = test_meta(0);
                m.seq_min = 11;
                m.seq_max = 20;
                m
            },
        });
        catalog.commit(&txn2, schema.clone()).await.unwrap();
        let m2 = catalog.current_manifest().await;

        assert_eq!(
            m2.table_uuid, m1.table_uuid,
            "table_uuid must persist across commits"
        );
        assert_eq!(m2.sequence_number, 2);
        assert_eq!(m2.parent_snapshot_id, Some(1));
        assert!(m2.last_updated_ms >= m1.last_updated_ms);
    }

    /// `export_to_iceberg` must produce a `metadata.json` that parses
    /// cleanly with the `iceberg` crate's `TableMetadata` deserializer.
    /// That struct runs every spec validation the crate knows about —
    /// if it accepts the payload, every V2-aware reader (pyiceberg,
    /// Spark, Trino, DuckDB, Snowflake, Athena) will too.
    #[tokio::test]
    async fn export_produces_iceberg_spec_compliant_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let schema_arc = Arc::new(test_schema());
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();

        // Commit a file so the exported snapshot isn't empty.
        let mut txn = SnapshotTransaction::new();
        txn.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0),
        });
        catalog.commit(&txn, schema_arc.clone()).await.unwrap();

        let target = tempfile::tempdir().unwrap();
        let out = catalog.export_to_iceberg(target.path()).await.unwrap();

        // File exists under metadata/ with the expected name.
        assert!(out.exists());
        assert!(out.starts_with(target.path().join("metadata")));
        assert!(out
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".metadata.json"));

        // version-hint.text must point at the emitted snapshot.
        let hint = tokio::fs::read_to_string(target.path().join("version-hint.text"))
            .await
            .unwrap();
        assert_eq!(hint.trim(), "1");

        // Parse with the iceberg crate to enforce spec compliance.
        let bytes = tokio::fs::read(&out).await.unwrap();
        let parsed: std::result::Result<iceberg::spec::TableMetadata, _> =
            serde_json::from_slice(&bytes);
        assert!(
            parsed.is_ok(),
            "iceberg-rs rejected exported metadata: {:?}\n\nfile: {}\n\ncontent:\n{}",
            parsed.err(),
            out.display(),
            String::from_utf8_lossy(&bytes)
        );
        let tm = parsed.unwrap();
        assert_eq!(tm.last_sequence_number(), 1);
        assert_eq!(tm.current_snapshot_id(), Some(1));

        // Issue #54: Avro manifest files must exist alongside
        // metadata.json. Count .avro files in the metadata dir —
        // one manifest-list and one manifest.
        let mut avro_count = 0usize;
        let mut entries = tokio::fs::read_dir(target.path().join("metadata"))
            .await
            .unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            if entry.file_name().to_string_lossy().ends_with(".avro") {
                avro_count += 1;
            }
        }
        assert_eq!(
            avro_count, 2,
            "export must produce exactly 2 Avro files (manifest-list + manifest)"
        );
    }

    /// Issue #54 regression: the full Iceberg read path — metadata.json
    /// → manifest-list Avro → manifest Avro → data file list — must
    /// resolve end-to-end. This is the exact sequence DuckDB's
    /// `iceberg_scan()`, pyiceberg, Spark, and Trino execute.
    #[tokio::test]
    async fn export_avro_manifest_chain_resolves_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let schema_arc = Arc::new(test_schema());
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();

        // Commit two files so the manifest has multiple entries.
        let mut txn = SnapshotTransaction::new();
        txn.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0),
        });
        txn.add_file(IcebergDataFile {
            path: "data/L0/b.parquet".into(),
            file_size: 2048,
            num_rows: 200,
            meta: {
                let mut m = test_meta(0);
                m.seq_min = 11;
                m.seq_max = 20;
                m.num_rows = 200;
                m
            },
        });
        catalog.commit(&txn, schema_arc.clone()).await.unwrap();

        let target = tempfile::tempdir().unwrap();
        catalog.export_to_iceberg(target.path()).await.unwrap();

        // Step 1: Read metadata.json → get manifest-list path.
        let meta_path = target.path().join("metadata/v1.metadata.json");
        let meta_bytes = tokio::fs::read(&meta_path).await.unwrap();
        let tm: iceberg::spec::TableMetadata = serde_json::from_slice(&meta_bytes).unwrap();
        let snapshot = tm.current_snapshot().expect("must have a snapshot");
        let manifest_list_path = snapshot.manifest_list();

        // The manifest-list path must be a file:// URI pointing into
        // the target directory.
        assert!(
            manifest_list_path.starts_with("file://"),
            "manifest-list must be an absolute file:// URI, got: {manifest_list_path}"
        );
        let manifest_list_local = manifest_list_path.strip_prefix("file://").unwrap();
        assert!(
            std::path::Path::new(manifest_list_local).exists(),
            "manifest-list Avro must exist at {manifest_list_local}"
        );

        // Step 2: Read manifest-list Avro → get manifest paths.
        let file_io = iceberg::io::FileIOBuilder::new_fs_io().build().unwrap();
        let manifest_list = snapshot.load_manifest_list(&file_io, &tm).await.unwrap();
        let manifest_entries = manifest_list.entries();
        assert_eq!(manifest_entries.len(), 1, "one manifest file expected");

        // Step 3: Read manifest Avro → verify data file list.
        let manifest_file = &manifest_entries[0];
        let manifest_path = &manifest_file.manifest_path;
        assert!(
            manifest_path.starts_with("file://"),
            "manifest path must be absolute file:// URI, got: {manifest_path}"
        );
        let manifest_local = manifest_path.strip_prefix("file://").unwrap();
        assert!(
            std::path::Path::new(manifest_local).exists(),
            "manifest Avro must exist at {manifest_local}"
        );

        // Parse the manifest Avro bytes.
        let manifest_bytes = tokio::fs::read(manifest_local).await.unwrap();
        let manifest = iceberg::spec::Manifest::parse_avro(&manifest_bytes).unwrap();
        let data_entries = manifest.entries();
        assert_eq!(data_entries.len(), 2, "manifest must list both data files");

        // Verify data file paths point to the catalog's base_path.
        let canonical_base = tokio::fs::canonicalize(tmp.path()).await.unwrap();
        let base_uri = format!("file://{}", canonical_base.display());
        for entry in data_entries {
            let fp = entry.file_path();
            assert!(
                fp.starts_with(&base_uri),
                "data file path must be under the catalog base_path, got: {fp}"
            );
        }

        // Verify record counts.
        let total_rows: u64 = data_entries.iter().map(|e| e.record_count()).sum();
        assert_eq!(total_rows, 300, "100 + 200 = 300 rows");
    }

    /// Issue #28 Phase 3: dual-read. A catalog directory that was
    /// seeded with a `v1.metadata.pb` (protobuf, "MRUB" magic) opens
    /// cleanly and the loaded manifest round-trips every field.
    #[tokio::test]
    async fn open_reads_protobuf_manifest_when_present() {
        let tmp = tempfile::tempdir().unwrap();
        // Seed the catalog dir with a protobuf-encoded v1 manifest.
        let metadata = tmp.path().join("metadata");
        tokio::fs::create_dir_all(&metadata).await.unwrap();
        let mut m = Manifest::empty(test_schema());
        m.snapshot_id = 1;
        let pb_bytes = m.to_protobuf().unwrap();
        tokio::fs::write(metadata.join("v1.metadata.pb"), &pb_bytes)
            .await
            .unwrap();
        tokio::fs::write(tmp.path().join("version-hint.text"), "1")
            .await
            .unwrap();

        // Reopen. The catalog must pick up the .pb file, decode it
        // via protobuf (not JSON), and expose the same snapshot_id.
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();
        let current = catalog.current_manifest().await;
        assert_eq!(current.snapshot_id, 1);
        assert_eq!(current.schema.table_name, "test");
    }

    /// Dual-read: when BOTH v1.metadata.pb and v1.metadata.json
    /// exist for the same version, the protobuf wins. Rationale:
    /// the dual-write path (Phase 4, future) will emit both; if they
    /// ever disagree it's a bug, and we pick the canonical new
    /// format. Document this tiebreak as an explicit test.
    #[tokio::test]
    async fn dual_read_prefers_protobuf_over_json() {
        let tmp = tempfile::tempdir().unwrap();
        let metadata = tmp.path().join("metadata");
        tokio::fs::create_dir_all(&metadata).await.unwrap();

        // Write conflicting snapshot_ids to show which one wins:
        // JSON claims 999, protobuf claims 7. Protobuf must win.
        let mut json_m = Manifest::empty(test_schema());
        json_m.snapshot_id = 999;
        tokio::fs::write(metadata.join("v1.metadata.json"), json_m.to_json().unwrap())
            .await
            .unwrap();
        let mut pb_m = Manifest::empty(test_schema());
        pb_m.snapshot_id = 7;
        tokio::fs::write(metadata.join("v1.metadata.pb"), pb_m.to_protobuf().unwrap())
            .await
            .unwrap();
        tokio::fs::write(tmp.path().join("version-hint.text"), "1")
            .await
            .unwrap();

        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();
        assert_eq!(catalog.current_manifest().await.snapshot_id, 7);
    }

    /// Issue #28 Phase 5: every commit writes ONLY protobuf.
    /// JSON is no longer written on commit; reads still accept
    /// legacy JSON-only catalogs for back-compat (see
    /// `back_compat_json_only_catalog_still_reads`).
    #[tokio::test]
    async fn commit_writes_protobuf_only_phase5() {
        let tmp = tempfile::tempdir().unwrap();
        let schema_arc = Arc::new(test_schema());
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();

        let mut txn = SnapshotTransaction::new();
        txn.add_file(IcebergDataFile {
            path: "data/L0/a.parquet".into(),
            file_size: 1024,
            num_rows: 100,
            meta: test_meta(0),
        });
        catalog.commit(&txn, schema_arc.clone()).await.unwrap();

        let metadata = tmp.path().join("metadata");
        let json_path = metadata.join("v1.metadata.json");
        let pb_path = metadata.join("v1.metadata.pb");
        assert!(
            !json_path.exists(),
            "Phase 5: JSON manifest must NOT be written on commit"
        );
        assert!(
            pb_path.exists(),
            "protobuf manifest is the canonical write path"
        );

        let pb_m = Manifest::from_protobuf(&std::fs::read(&pb_path).unwrap()).unwrap();
        assert_eq!(pb_m.snapshot_id, 1);
        assert_eq!(pb_m.entries.len(), 1);
        assert_eq!(pb_m.entries[0].path, "data/L0/a.parquet");
    }

    /// Reopening after a protobuf-only commit goes through the
    /// protobuf path unchanged — this is the steady-state path
    /// post-Phase-5.
    #[tokio::test]
    async fn commit_roundtrip_reopen_through_protobuf() {
        let tmp = tempfile::tempdir().unwrap();
        let schema_arc = Arc::new(test_schema());
        {
            let catalog = IcebergCatalog::open(tmp.path(), test_schema())
                .await
                .unwrap();
            let mut txn = SnapshotTransaction::new();
            txn.add_file(IcebergDataFile {
                path: "data/L0/x.parquet".into(),
                file_size: 2048,
                num_rows: 200,
                meta: test_meta(0),
            });
            catalog.commit(&txn, schema_arc.clone()).await.unwrap();
        }
        // Phase 5: JSON is no longer written; this assertion
        // replaces the Phase-4 "remove JSON to prove pb works"
        // gate with "assert JSON absent from the start."
        assert!(!tmp.path().join("metadata/v1.metadata.json").exists());

        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();
        let m = catalog.current_manifest().await;
        assert_eq!(m.snapshot_id, 1);
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0].path, "data/L0/x.parquet");
    }

    /// Issue #28 Phase 5: back-compat read. A catalog committed
    /// before #28 Phase 4 (JSON-only) still opens cleanly. We
    /// simulate by writing a JSON manifest manually and checking
    /// that open + current_manifest surface it.
    #[tokio::test]
    async fn back_compat_json_only_catalog_still_reads() {
        let tmp = tempfile::tempdir().unwrap();
        let schema_arc = Arc::new(test_schema());
        // Open creates v0 (genesis) — then commit writes v1 via pb.
        // To simulate a pre-Phase-4 catalog, do the commit, then
        // REPLACE the .pb with a .json file on disk.
        {
            let catalog = IcebergCatalog::open(tmp.path(), test_schema())
                .await
                .unwrap();
            let mut txn = SnapshotTransaction::new();
            txn.add_file(IcebergDataFile {
                path: "data/L0/legacy.parquet".into(),
                file_size: 512,
                num_rows: 50,
                meta: test_meta(0),
            });
            catalog.commit(&txn, schema_arc.clone()).await.unwrap();
        }
        // Convert pb → json on disk.
        let pb_bytes = std::fs::read(tmp.path().join("metadata/v1.metadata.pb")).unwrap();
        let manifest = Manifest::from_protobuf(&pb_bytes).unwrap();
        std::fs::remove_file(tmp.path().join("metadata/v1.metadata.pb")).unwrap();
        let json_bytes = manifest.to_json().unwrap();
        std::fs::write(tmp.path().join("metadata/v1.metadata.json"), json_bytes).unwrap();

        // Reopen — must surface the JSON manifest unchanged.
        let catalog = IcebergCatalog::open(tmp.path(), test_schema())
            .await
            .unwrap();
        let m = catalog.current_manifest().await;
        assert_eq!(m.snapshot_id, 1);
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0].path, "data/L0/legacy.parquet");
    }

    /// Orphan-metadata detection recognises both .pb and .json.
    /// Without this the orphan check misses pre-existing protobuf
    /// snapshots and silently initializes a fresh catalog over them.
    #[tokio::test]
    async fn open_rejects_initializing_fresh_over_existing_protobuf() {
        let tmp = tempfile::tempdir().unwrap();
        let metadata = tmp.path().join("metadata");
        tokio::fs::create_dir_all(&metadata).await.unwrap();
        // Create a .pb file WITHOUT a version-hint to simulate
        // lost-hint recovery scenario.
        let m = Manifest::empty(test_schema());
        tokio::fs::write(metadata.join("v42.metadata.pb"), m.to_protobuf().unwrap())
            .await
            .unwrap();

        let err = match IcebergCatalog::open(tmp.path(), test_schema()).await {
            Err(e) => e,
            Ok(_) => panic!("expected open to refuse orphan-metadata scenario"),
        };
        let msg = format!("{err:?}");
        assert!(
            msg.contains("version-hint.text is missing"),
            "expected orphan detection to fire, got: {msg}"
        );
    }
}
