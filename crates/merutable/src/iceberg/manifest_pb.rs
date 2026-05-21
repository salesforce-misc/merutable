//! Issue #28 Phase 1: protobuf manifest format (on-disk wire format).
//!
//! Wraps the prost-generated `merutable.catalog.v1.Manifest` type in a
//! stable framing header so the on-disk bytes carry a magic number,
//! a format-version byte, and a length prefix. The header lets the
//! decoder reject garbage immediately and gives us room to evolve
//! the on-disk shape (e.g., switch to a different codec later)
//! without changing the Rust type.
//!
//! # Wire format
//!
//! ```text
//! +----------------+--------------+----------------+--------------------+
//! | 4-byte magic   | 1-byte fmt   | 4-byte length  | protobuf payload   |
//! | "MRUB"         | version (=1) | (LE u32)       | (prost-encoded)    |
//! +----------------+--------------+----------------+--------------------+
//! ```
//!
//! # Status
//!
//! Phase 1: types + encode/decode round-trip. NOT wired into the
//! catalog commit path. JSON (`Manifest::to_json` / `from_json`)
//! remains the on-disk format today.
//!
//! Phase 2 (planned): add `manifest.pb` emission alongside JSON,
//! read both, prefer protobuf on decode.
//!
//! Phase 3 (planned): JSON is reader-only (legacy), all new commits
//! write protobuf.
//!
//! This phasing means an in-place upgrade: run new binary, it writes
//! protobuf manifests; old binaries still read the JSON chain before
//! the upgrade point but not new commits.

use crate::types::{MeruError, Result};
use prost::Message as _;

/// Generated protobuf types. See `build.rs` + `proto/manifest.proto`.
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/merutable.catalog.v1.rs"));
}

/// Magic bytes: "MRUB" (Merutable Binary).
pub const MAGIC: [u8; 4] = *b"MRUB";

/// Current format-version byte. Decoder rejects unknown versions —
/// a newer writer talking to an older reader surfaces as a crisp
/// error rather than a silent malformed parse.
pub const FORMAT_VERSION: u8 = 1;

/// Minimum valid header length: 4 (magic) + 1 (version) + 4 (length).
const HEADER_LEN: usize = 4 + 1 + 4;

/// Encode a `pb::Manifest` onto the wire format (magic + version +
/// length-prefix + prost bytes). Infallible besides the protobuf
/// encoder itself, which cannot fail on a well-formed Rust value.
pub fn encode(manifest: &pb::Manifest) -> Vec<u8> {
    let body = manifest.encode_to_vec();
    let mut buf = Vec::with_capacity(HEADER_LEN + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(FORMAT_VERSION);
    buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
    buf.extend_from_slice(&body);
    buf
}

/// Decode a wire-format manifest. Validates magic + version + length
/// before handing the payload to prost; a corrupt or mismatched
/// frame surfaces as `MeruError::Corruption` with a descriptive
/// message naming the specific check that failed. No silent fallback.
pub fn decode(bytes: &[u8]) -> Result<pb::Manifest> {
    if bytes.len() < HEADER_LEN {
        return Err(MeruError::Corruption(format!(
            "manifest wire frame too short: {} bytes (need at least {})",
            bytes.len(),
            HEADER_LEN
        )));
    }
    if bytes[0..4] != MAGIC {
        return Err(MeruError::Corruption(format!(
            "manifest magic mismatch: expected {:02X?}, got {:02X?}",
            MAGIC,
            &bytes[0..4]
        )));
    }
    let version = bytes[4];
    if version != FORMAT_VERSION {
        return Err(MeruError::Corruption(format!(
            "manifest format version {version} not supported by this binary \
             (supported: {FORMAT_VERSION})"
        )));
    }
    let len = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
    if bytes.len() < HEADER_LEN + len {
        return Err(MeruError::Corruption(format!(
            "manifest frame truncated: header claims {len} bytes of payload, \
             have {}",
            bytes.len() - HEADER_LEN
        )));
    }
    let payload = &bytes[HEADER_LEN..HEADER_LEN + len];
    pb::Manifest::decode(payload)
        .map_err(|e| MeruError::Corruption(format!("manifest protobuf decode failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> pb::Manifest {
        pb::Manifest {
            snapshot_id: 42,
            sequence_number: 7,
            schema_id: 0,
            partition_spec_id: 0,
            data_files: vec![pb::DataFileRef {
                path: "data/L0/abc.parquet".into(),
                file_size: 1024,
                num_rows: 100,
                level: 0,
                seq_min: 1,
                seq_max: 100,
                key_min: vec![0x00, 0x01],
                key_max: vec![0xFF, 0xFE],
                dv_path: None,
                dv_offset: None,
                dv_length: None,
                status: 1,       // added
                format: Some(1), // Dual
                first_row_id: Some(0),
            }],
            delete_files: vec![],
            previous_snapshot_id: Some(41),
            table_uuid: "deadbeef-1234-5678-9abc-0123456789ab".into(),
            last_updated_ms: 1_700_000_000_000,
            properties: [("merutable.job".to_string(), "flush".to_string())]
                .into_iter()
                .collect(),
            last_column_id: 2,
            next_row_id: 100,
        }
    }

    #[test]
    fn roundtrip_preserves_every_field() {
        let m = sample_manifest();
        let bytes = encode(&m);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded, m);
    }

    #[test]
    fn decode_rejects_short_frame() {
        let err = decode(b"MRU").unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("too short"), "msg: {msg}");
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = encode(&sample_manifest());
        bytes[0] = b'X';
        let err = decode(&bytes).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("magic mismatch"), "msg: {msg}");
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let mut bytes = encode(&sample_manifest());
        bytes[4] = 99;
        let err = decode(&bytes).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("format version 99"), "msg: {msg}");
    }

    #[test]
    fn decode_rejects_truncated_payload() {
        let bytes = encode(&sample_manifest());
        let truncated = &bytes[..bytes.len() - 10];
        let err = decode(truncated).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("truncated"), "msg: {msg}");
    }

    /// Write-once contract: the encoded frame's layout is stable.
    /// Byte 0..4 = "MRUB", byte 4 = 1, bytes 5..9 = LE length,
    /// payload follows.
    #[test]
    fn wire_header_is_layout_stable() {
        let bytes = encode(&sample_manifest());
        assert_eq!(&bytes[0..4], b"MRUB");
        assert_eq!(bytes[4], 1);
        let len = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
        assert_eq!(bytes.len(), HEADER_LEN + len);
    }
}
