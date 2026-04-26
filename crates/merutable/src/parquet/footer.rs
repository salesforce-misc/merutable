//! Parquet KV metadata footer encode/decode for merutable-specific fields.
//!
//! Two keys are written into every merutable-managed Parquet file's footer:
//! - `"merutable.meta"` → JSON-serialized `ParquetFileMeta`
//! - `"merutable.schema"` → JSON-serialized `TableSchema`
//!
//! External external analytical readers (DuckDB, Spark, iceberg-rust) can pick up the
//! schema from the footer without any merutable imports — the JSON shape
//! is stable and human-readable.

use crate::types::{MeruError, Result, level::ParquetFileMeta, schema::TableSchema};
use std::collections::HashMap;

/// Encode merutable metadata as Parquet KV footer entries.
///
/// The returned pairs are independent — order is deterministic (`meta`
/// first, `schema` second) so callers that concatenate multiple KV
/// sources get a stable layout.
pub fn encode_footer_kv(
    meta: &ParquetFileMeta,
    schema: &TableSchema,
) -> Result<Vec<(String, String)>> {
    let meta_json = meta.serialize()?;
    let schema_json =
        serde_json::to_string(schema).map_err(|e| MeruError::Parquet(e.to_string()))?;
    Ok(vec![
        (ParquetFileMeta::FOOTER_KEY.to_string(), meta_json),
        (ParquetFileMeta::SCHEMA_KEY.to_string(), schema_json),
    ])
}

/// Decode merutable metadata from the Parquet KV footer entries.
///
/// Both `merutable.meta` and `merutable.schema` must be present; a
/// missing or unparseable entry surfaces as `MeruError::Corruption` with
/// the key name in the message. Extra unknown keys are ignored for
/// forward compatibility.
pub fn decode_footer_kv(kv: &HashMap<String, String>) -> Result<(ParquetFileMeta, TableSchema)> {
    let meta_json = kv.get(ParquetFileMeta::FOOTER_KEY).ok_or_else(|| {
        MeruError::Corruption(format!(
            "missing '{}' in Parquet KV footer",
            ParquetFileMeta::FOOTER_KEY
        ))
    })?;
    let schema_json = kv.get(ParquetFileMeta::SCHEMA_KEY).ok_or_else(|| {
        MeruError::Corruption(format!(
            "missing '{}' in Parquet KV footer",
            ParquetFileMeta::SCHEMA_KEY
        ))
    })?;

    let meta: ParquetFileMeta = ParquetFileMeta::deserialize(meta_json).map_err(|e| {
        MeruError::Corruption(format!(
            "failed to parse '{}': {e}",
            ParquetFileMeta::FOOTER_KEY
        ))
    })?;
    let schema: TableSchema = serde_json::from_str(schema_json).map_err(|e| {
        MeruError::Corruption(format!(
            "failed to parse '{}': {e}",
            ParquetFileMeta::SCHEMA_KEY
        ))
    })?;

    Ok((meta, schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        level::{Level, ParquetFileMeta},
        schema::{ColumnDef, ColumnType, TableSchema},
    };

    fn sample_meta() -> ParquetFileMeta {
        ParquetFileMeta {
            level: Level(2),
            seq_min: 1,
            seq_max: 100,
            // Non-ASCII bytes to catch any codec that assumes valid UTF-8.
            key_min: vec![0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF],
            key_max: vec![0xFF; 16],
            num_rows: 500,
            file_size: 1024 * 1024,
            dv_path: Some("s3://bucket/path.puffin".into()),
            dv_offset: Some(42),
            dv_length: Some(128),
            format: None,
            column_stats: None,
        }
    }

    fn sample_schema() -> TableSchema {
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
                    name: "val".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: true,

                    ..Default::default()
                },
                ColumnDef {
                    name: "fb".into(),
                    col_type: ColumnType::FixedLenByteArray(8),
                    nullable: true,

                    ..Default::default()
                },
            ],
            primary_key: vec![0],

            ..Default::default()
        }
    }

    /// Full field-level round-trip. Previously the assertion compared
    /// only three fields, so key_min/key_max/dv_* could have silently
    /// drifted between encode and decode without failing.
    #[test]
    fn roundtrip_preserves_every_field() {
        let meta = sample_meta();
        let schema = sample_schema();
        let kv = encode_footer_kv(&meta, &schema).unwrap();
        let map: HashMap<_, _> = kv.into_iter().collect();
        let (got_meta, got_schema) = decode_footer_kv(&map).unwrap();

        // ParquetFileMeta: every field.
        assert_eq!(got_meta.level, meta.level);
        assert_eq!(got_meta.seq_min, meta.seq_min);
        assert_eq!(got_meta.seq_max, meta.seq_max);
        assert_eq!(
            got_meta.key_min, meta.key_min,
            "key_min must round-trip byte-for-byte"
        );
        assert_eq!(
            got_meta.key_max, meta.key_max,
            "key_max must round-trip byte-for-byte"
        );
        assert_eq!(got_meta.num_rows, meta.num_rows);
        assert_eq!(got_meta.file_size, meta.file_size);
        assert_eq!(got_meta.dv_path, meta.dv_path);
        assert_eq!(got_meta.dv_offset, meta.dv_offset);
        assert_eq!(got_meta.dv_length, meta.dv_length);

        // TableSchema: every column + primary key.
        assert_eq!(got_schema.table_name, schema.table_name);
        assert_eq!(got_schema.primary_key, schema.primary_key);
        assert_eq!(got_schema.columns.len(), schema.columns.len());
        for (i, (orig, got)) in schema.columns.iter().zip(&got_schema.columns).enumerate() {
            assert_eq!(orig.name, got.name, "column {i} name");
            assert_eq!(orig.col_type, got.col_type, "column {i} col_type");
            assert_eq!(orig.nullable, got.nullable, "column {i} nullable");
        }
    }

    /// `encode_footer_kv` must return exactly two pairs, in deterministic
    /// order (meta first, schema second). Callers rely on this shape.
    #[test]
    fn encode_returns_two_keys_in_stable_order() {
        let meta = sample_meta();
        let schema = sample_schema();
        let kv = encode_footer_kv(&meta, &schema).unwrap();
        assert_eq!(kv.len(), 2);
        assert_eq!(kv[0].0, ParquetFileMeta::FOOTER_KEY);
        assert_eq!(kv[1].0, ParquetFileMeta::SCHEMA_KEY);
    }

    /// Missing `merutable.meta`: decode must return a Corruption error
    /// whose message names the missing key so operators can tell which
    /// half of the pair is absent.
    #[test]
    fn missing_meta_key_errors_with_key_name() {
        let schema = sample_schema();
        let mut map = HashMap::new();
        map.insert(
            ParquetFileMeta::SCHEMA_KEY.to_string(),
            serde_json::to_string(&schema).unwrap(),
        );
        let err = decode_footer_kv(&map).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains(ParquetFileMeta::FOOTER_KEY) && msg.contains("missing"),
            "error should name the missing meta key: {msg}"
        );
    }

    /// Missing `merutable.schema`: same error shape as missing meta.
    #[test]
    fn missing_schema_key_errors_with_key_name() {
        let meta = sample_meta();
        let mut map = HashMap::new();
        map.insert(
            ParquetFileMeta::FOOTER_KEY.to_string(),
            meta.serialize().unwrap(),
        );
        let err = decode_footer_kv(&map).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains(ParquetFileMeta::SCHEMA_KEY) && msg.contains("missing"),
            "error should name the missing schema key: {msg}"
        );
    }

    /// Both keys missing: must still error cleanly (not panic, not
    /// return a default).
    #[test]
    fn empty_kv_errors() {
        let kv: HashMap<String, String> = HashMap::new();
        assert!(decode_footer_kv(&kv).is_err());
    }

    /// Corrupt JSON in the meta slot must surface as Corruption with
    /// the key name attached, not as a generic serde error swallowed
    /// silently or surfaced as a different variant.
    #[test]
    fn corrupt_meta_json_errors_with_key_name() {
        let schema = sample_schema();
        let mut map = HashMap::new();
        map.insert(
            ParquetFileMeta::FOOTER_KEY.to_string(),
            "not-json-at-all {[".into(),
        );
        map.insert(
            ParquetFileMeta::SCHEMA_KEY.to_string(),
            serde_json::to_string(&schema).unwrap(),
        );
        let err = decode_footer_kv(&map).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains(ParquetFileMeta::FOOTER_KEY) && msg.contains("parse"),
            "error should name the failing key: {msg}"
        );
        assert!(matches!(err, MeruError::Corruption(_)));
    }

    /// Corrupt JSON in the schema slot: same shape as meta corruption.
    #[test]
    fn corrupt_schema_json_errors_with_key_name() {
        let meta = sample_meta();
        let mut map = HashMap::new();
        map.insert(
            ParquetFileMeta::FOOTER_KEY.to_string(),
            meta.serialize().unwrap(),
        );
        map.insert(
            ParquetFileMeta::SCHEMA_KEY.to_string(),
            "{\"table_name\": not-a-string}".into(),
        );
        let err = decode_footer_kv(&map).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains(ParquetFileMeta::SCHEMA_KEY) && msg.contains("parse"),
            "error should name the failing key: {msg}"
        );
        assert!(matches!(err, MeruError::Corruption(_)));
    }

    /// Forward compatibility: unknown extra keys in the footer must be
    /// silently ignored. A future merutable version may add new footer
    /// entries; older readers must tolerate them.
    #[test]
    fn unknown_extra_keys_are_ignored() {
        let meta = sample_meta();
        let schema = sample_schema();
        let mut map: HashMap<String, String> = encode_footer_kv(&meta, &schema)
            .unwrap()
            .into_iter()
            .collect();
        map.insert(
            "merutable.future_feature".into(),
            "some opaque value".into(),
        );
        map.insert("third_party.random".into(), "{\"nested\":true}".into());
        let (got_meta, got_schema) = decode_footer_kv(&map).unwrap();
        assert_eq!(got_meta.num_rows, meta.num_rows);
        assert_eq!(got_schema.table_name, schema.table_name);
    }

    /// `key_min` / `key_max` can contain arbitrary (non-UTF-8) bytes.
    /// The `hex_bytes` serde adapter must round-trip those bytes exactly
    /// — an earlier draft used `String::from_utf8_lossy` which would have
    /// silently corrupted any non-ASCII byte.
    #[test]
    fn key_bytes_roundtrip_arbitrary_non_utf8() {
        let mut meta = sample_meta();
        // Every possible byte value, twice, including interior zeros.
        meta.key_min = (0u8..=255).collect();
        meta.key_max = (0u8..=255).rev().collect();
        let schema = sample_schema();
        let map: HashMap<_, _> = encode_footer_kv(&meta, &schema)
            .unwrap()
            .into_iter()
            .collect();
        let (got, _) = decode_footer_kv(&map).unwrap();
        assert_eq!(got.key_min, meta.key_min);
        assert_eq!(got.key_max, meta.key_max);
    }
}
