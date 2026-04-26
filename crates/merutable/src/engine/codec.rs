//! Row serialization codec for the WAL value path.
//!
//! Uses `postcard` (compact binary serde) for all new writes. Falls back to
//! `serde_json` for data written before the migration — WAL replay can
//! encounter old JSON-encoded rows during crash recovery.
//!
//! **Discrimination**: JSON-encoded `Row` always starts with `{` (0x7B).
//! `postcard`-encoded `Row` starts with a varint for the `fields` vec length,
//! which for rows with < 123 columns will never be 0x7B. To make the boundary
//! unambiguous regardless of column count, we prepend a single `0x01` marker
//! byte to postcard output.

use crate::types::{MeruError, Result, value::Row};

/// Marker byte prepended to postcard-encoded rows.
/// JSON never starts with 0x01, so this is a reliable discriminator.
const POSTCARD_MARKER: u8 = 0x01;

/// Serialize a `Row` to bytes using postcard (binary, ~10× faster than JSON).
/// Prepends a 1-byte format marker for backward-compatible decoding.
#[inline]
pub fn encode_row(row: &Row) -> Result<Vec<u8>> {
    let raw = postcard::to_allocvec(row)
        .map_err(|e| MeruError::InvalidArgument(format!("row postcard serialize failed: {e}")))?;
    let mut out = Vec::with_capacity(1 + raw.len());
    out.push(POSTCARD_MARKER);
    out.extend_from_slice(&raw);
    Ok(out)
}

/// Deserialize a `Row` from bytes.
///
/// - **Empty input** → `Ok(Row::default())`. This is the legitimate
///   tombstone signal: `OpType::Delete` writes encode empty value
///   bytes, and we need to preserve that path.
/// - First byte `POSTCARD_MARKER` (0x01) → strip + postcard-decode.
/// - Otherwise → `serde_json` (legacy WAL data).
///
/// Issue #12: any decode failure now returns `MeruError::Corruption`
/// with a diagnostic message. The old signature returned `Row` and
/// used `unwrap_or_default()`, silently swallowing WAL corruption
/// and format-mismatch errors into phantom empty rows that callers
/// treated identically to NULL — data loss without a trail. Callers
/// now know exactly when bytes were unreadable and can surface the
/// failure (abort recovery, alert, quarantine the file).
#[inline]
pub fn decode_row(bytes: &[u8]) -> Result<Row> {
    if bytes.is_empty() {
        return Ok(Row::default());
    }
    if bytes[0] == POSTCARD_MARKER {
        postcard::from_bytes(&bytes[1..]).map_err(|e| {
            MeruError::Corruption(format!(
                "row postcard decode failed ({} bytes after marker): {e}",
                bytes.len() - 1
            ))
        })
    } else {
        serde_json::from_slice(bytes).map_err(|e| {
            MeruError::Corruption(format!(
                "row legacy-JSON decode failed ({} bytes): {e}",
                bytes.len()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::value::FieldValue;
    use bytes::Bytes;

    #[test]
    fn roundtrip_postcard() {
        let row = Row::new(vec![
            Some(FieldValue::Int64(42)),
            None,
            Some(FieldValue::Bytes(Bytes::from("hello"))),
        ]);
        let encoded = encode_row(&row).unwrap();
        assert_eq!(encoded[0], POSTCARD_MARKER);
        let decoded = decode_row(&encoded).unwrap();
        assert_eq!(decoded, row);
    }

    #[test]
    fn decode_legacy_json() {
        let row = Row::new(vec![
            Some(FieldValue::Int64(1)),
            Some(FieldValue::Boolean(true)),
        ]);
        let json = serde_json::to_vec(&row).unwrap();
        // First byte of JSON is '{' (0x7B), not 0x01.
        assert_ne!(json[0], POSTCARD_MARKER);
        let decoded = decode_row(&json).unwrap();
        assert_eq!(decoded, row);
    }

    #[test]
    fn decode_empty() {
        // Empty bytes = tombstone marker. Must decode to default row,
        // NOT an error — `OpType::Delete` writes empty value bytes.
        let decoded = decode_row(&[]).unwrap();
        assert_eq!(decoded, Row::default());
    }

    /// Issue #12 regression: decode failures must surface as
    /// `MeruError::Corruption`, not silently collapse to empty rows.
    #[test]
    fn decode_corrupt_postcard_returns_corruption_error() {
        // Marker byte + garbage postcard payload.
        let bad = [POSTCARD_MARKER, 0xFF, 0xFF, 0xFF, 0xFF];
        let err = decode_row(&bad).unwrap_err();
        match err {
            MeruError::Corruption(msg) => {
                assert!(
                    msg.contains("postcard decode failed"),
                    "error message should identify postcard decode failure; got: {msg}"
                );
            }
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    /// Issue #12 regression: legacy-JSON path also returns
    /// Corruption on malformed input (no silent default).
    #[test]
    fn decode_corrupt_legacy_json_returns_corruption_error() {
        // Starts with '{' (JSON fallback) but is not valid JSON.
        let bad = b"{not valid json}";
        let err = decode_row(bad).unwrap_err();
        match err {
            MeruError::Corruption(msg) => {
                assert!(
                    msg.contains("legacy-JSON decode failed"),
                    "error should identify JSON decode failure; got: {msg}"
                );
            }
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    #[test]
    fn postcard_smaller_than_json() {
        let row = Row::new(vec![
            Some(FieldValue::Int64(123456789)),
            Some(FieldValue::Double(98.765)),
            Some(FieldValue::Bytes(Bytes::from("test data"))),
            None,
        ]);
        let postcard_bytes = encode_row(&row).unwrap();
        let json_bytes = serde_json::to_vec(&row).unwrap();
        // postcard should be significantly smaller.
        assert!(
            postcard_bytes.len() < json_bytes.len(),
            "postcard={} json={}",
            postcard_bytes.len(),
            json_bytes.len()
        );
    }
}
