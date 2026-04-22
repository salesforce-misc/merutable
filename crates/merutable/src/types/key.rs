//! Internal key encoding — the most correctness-critical module in merutable.
//!
//! # Wire format
//!
//! ```text
//! [ encoded_pk_fields ... ][ tag: 8 bytes BE ]
//! ```
//!
//! ## PK field encoding (one per primary-key column, in schema order)
//!
//! | Column type          | Encoded as                                          |
//! |----------------------|-----------------------------------------------------|
//! | Boolean              | 1 byte: 0x00 = false, 0x01 = true                  |
//! | Int32                | 4 bytes BE, sign-bit-flipped (`v ^ 0x8000_0000`)    |
//! | Int64                | 8 bytes BE, sign-bit-flipped (`v ^ 0x8000…0000`)    |
//! | Float                | 4 bytes, IEEE order-preserving (see below)          |
//! | Double               | 8 bytes, IEEE order-preserving (see below)          |
//! | FixedLenByteArray(n) | n bytes raw (fixed length, no terminator needed)    |
//! | ByteArray            | escape(bytes) + [0x00, 0x00] terminator             |
//!
//! **ByteArray escape**: replace each `0x00` in data with `[0x00, 0xFF]`;
//! terminate with a two-byte `[0x00, 0x00]`. Issue #7 regression: a
//! single-byte `0x00` terminator allowed a shorter PK's terminator to
//! collide with a longer PK's escape-continuation byte at the same
//! position. Because the 8-byte tag that follows the encoded user-key
//! always begins with `0xFF` for sequence numbers in normal range
//! (`SEQNUM_MAX - seq` has `0xFF` as its high byte), a shorter
//! user-key's `encode(A) + tag_A` could sort AFTER a longer
//! `encode(B) + tag_B` when A was a raw-byte prefix of B. The two-byte
//! `[0x00, 0x00]` terminator is distinguishable from an escape
//! continuation (`0x00, 0xFF`) — the second byte of the terminator
//! (`0x00`) sorts strictly before `0xFF`, so the comparison resolves
//! at the terminator itself, before any tag byte comes into play.
//! CockroachDB hit the same class of bug and uses an equivalent
//! two-byte-terminator scheme.
//!
//! **Float order preservation**:
//! - Positive (or +0): flip sign bit → `bits ^ 0x8000_0000`
//! - Negative: flip all bits → `!bits`
//!
//! ## Tag encoding
//!
//! ```text
//! tag = ((SEQNUM_MAX.0 - seq.0) << 8) | (op_type as u64)
//! stored as 8 bytes big-endian.
//! ```
//!
//! Higher real `seq` → smaller inverted `seq` → smaller tag → sorts **earlier**
//! for the same PK. A seek with `SEQNUM_MAX` therefore lands before all real
//! entries for a given PK (newest-first semantics on skip-list iteration).
//!
//! # Invariant
//!
//! Lexicographic byte comparison (`memcmp`) of two encoded `InternalKey`s gives:
//! PK **ascending**, then seq **descending** (newest first).

use std::cmp::Ordering;

use bytes::Bytes;

use crate::types::{
    schema::{ColumnType, TableSchema},
    sequence::{OpType, SeqNum, SEQNUM_MAX},
    value::FieldValue,
    MeruError, Result,
};

/// An internal key: PK values + sequence number + operation type, pre-encoded
/// into a byte string that sorts correctly with `memcmp`.
#[derive(Clone, Debug)]
pub struct InternalKey {
    /// Pre-encoded wire bytes. Used directly as the skip-list key.
    encoded: Bytes,
    pub seq: SeqNum,
    pub op_type: OpType,
    pk_values: Vec<FieldValue>,
}

impl InternalKey {
    /// Encode from PK field values + seq + op_type given the table schema.
    pub fn encode(
        pk_values: &[FieldValue],
        seq: SeqNum,
        op_type: OpType,
        schema: &TableSchema,
    ) -> Result<Self> {
        let mut buf = Vec::with_capacity(64);
        encode_pk_fields(pk_values, schema, &mut buf)?;
        encode_tag(seq, op_type, &mut buf)?;
        Ok(Self {
            encoded: Bytes::from(buf),
            seq,
            op_type,
            pk_values: pk_values.to_vec(),
        })
    }

    /// Seek-sentinel: encodes a key that sorts before all real entries for the given PK.
    /// Use for skip-list seeks: `seek(InternalKey::seek_latest(pk, schema))` then advance
    /// to the first entry with matching PK and `entry.seq <= read_seq`.
    pub fn seek_latest(pk_values: &[FieldValue], schema: &TableSchema) -> Result<Self> {
        Self::encode(pk_values, SEQNUM_MAX, OpType::Put, schema)
    }

    /// Raw wire bytes (used as the `crossbeam_skiplist::SkipMap` key).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.encoded
    }

    pub fn pk_values(&self) -> &[FieldValue] {
        &self.pk_values
    }

    /// Decode from raw wire bytes + schema. Inverse of `encode`.
    pub fn decode(raw: &[u8], schema: &TableSchema) -> Result<Self> {
        if raw.len() < 8 {
            return Err(MeruError::Corruption("internal key too short".into()));
        }
        let (pk_bytes, tag_bytes) = raw.split_at(raw.len() - 8);
        let tag = u64::from_be_bytes(tag_bytes.try_into().unwrap());
        let inverted_seq = tag >> 8;
        let op_byte = (tag & 0xFF) as u8;
        // Bug K3 fix: validate inverted_seq doesn't exceed SEQNUM_MAX.
        // On corrupt data, this subtraction would underflow, producing a
        // garbage SeqNum that breaks MVCC ordering.
        if inverted_seq > SEQNUM_MAX.0 {
            return Err(MeruError::Corruption(format!(
                "inverted_seq {inverted_seq} exceeds SEQNUM_MAX ({})",
                SEQNUM_MAX.0
            )));
        }
        let seq = SeqNum(SEQNUM_MAX.0 - inverted_seq);
        let op_type = match op_byte {
            0x00 => OpType::Delete,
            0x01 => OpType::Put,
            _ => {
                return Err(MeruError::Corruption(format!(
                    "unknown op_type {op_byte:#x}"
                )))
            }
        };
        let pk_values = decode_pk_fields(pk_bytes, schema)?;
        Ok(Self {
            encoded: Bytes::copy_from_slice(raw),
            seq,
            op_type,
            pk_values,
        })
    }

    /// Extract only the user (PK) portion of the encoded key (without the tag).
    /// Used for bloom filter probing: hash the PK bytes, not the full internal key.
    pub fn user_key_bytes(&self) -> &[u8] {
        &self.encoded[..self.encoded.len() - 8]
    }

    /// Encode only the user-key (PK) bytes — no tag, no struct allocation.
    ///
    /// This is the hot-path version for writes: the write path only needs
    /// `user_key_bytes` for the WAL record key and cache invalidation. It
    /// does NOT need `seq`, `op_type`, or `pk_values` stored in a struct.
    /// Avoids the `pk_values.to_vec()` clone and the 8-byte tag encode that
    /// `InternalKey::encode()` performs.
    pub fn encode_user_key(pk_values: &[FieldValue], schema: &TableSchema) -> Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(64);
        encode_pk_fields(pk_values, schema, &mut buf)?;
        Ok(buf)
    }
}

impl PartialEq for InternalKey {
    fn eq(&self, other: &Self) -> bool {
        self.encoded == other.encoded
    }
}
impl Eq for InternalKey {}

impl PartialOrd for InternalKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalKey {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.encoded.cmp(&other.encoded)
    }
}

// ── Encoding ─────────────────────────────────────────────────────────────────

fn encode_pk_fields(values: &[FieldValue], schema: &TableSchema, buf: &mut Vec<u8>) -> Result<()> {
    if values.len() != schema.primary_key.len() {
        return Err(MeruError::InvalidArgument(format!(
            "expected {} PK values, got {}",
            schema.primary_key.len(),
            values.len()
        )));
    }
    for (val, &col_idx) in values.iter().zip(schema.primary_key.iter()) {
        encode_field(val, &schema.columns[col_idx].col_type, buf)?;
    }
    Ok(())
}

fn encode_field(val: &FieldValue, col_type: &ColumnType, buf: &mut Vec<u8>) -> Result<()> {
    match (val, col_type) {
        (FieldValue::Boolean(b), ColumnType::Boolean) => {
            buf.push(u8::from(*b));
        }
        (FieldValue::Int32(v), ColumnType::Int32) => {
            buf.extend_from_slice(&((*v as u32) ^ 0x8000_0000_u32).to_be_bytes());
        }
        (FieldValue::Int64(v), ColumnType::Int64) => {
            buf.extend_from_slice(&((*v as u64) ^ 0x8000_0000_0000_0000_u64).to_be_bytes());
        }
        (FieldValue::Float(v), ColumnType::Float) => {
            if v.is_nan() {
                return Err(MeruError::InvalidArgument(
                    "NaN is not allowed in primary key columns (Float): \
                     IEEE 754 NaN has multiple bit representations, \
                     producing non-deterministic key encoding"
                        .into(),
                ));
            }
            buf.extend_from_slice(&order_preserving_f32(*v));
        }
        (FieldValue::Double(v), ColumnType::Double) => {
            if v.is_nan() {
                return Err(MeruError::InvalidArgument(
                    "NaN is not allowed in primary key columns (Double): \
                     IEEE 754 NaN has multiple bit representations, \
                     producing non-deterministic key encoding"
                        .into(),
                ));
            }
            buf.extend_from_slice(&order_preserving_f64(*v));
        }
        (FieldValue::Bytes(b), ColumnType::FixedLenByteArray(n)) => {
            if b.len() != *n as usize {
                return Err(MeruError::SchemaMismatch(format!(
                    "FixedLenByteArray({n}): got {} bytes",
                    b.len()
                )));
            }
            buf.extend_from_slice(b);
        }
        (FieldValue::Bytes(b), ColumnType::ByteArray) => {
            escape_byte_array(b, buf);
        }
        _ => {
            return Err(MeruError::SchemaMismatch(format!(
                "field value type mismatch with column type {col_type:?}"
            )));
        }
    }
    Ok(())
}

/// ByteArray escape encoding for sort-safe composite key embedding.
/// Each `0x00` byte in data → `[0x00, 0xFF]`. Terminated by a two-byte
/// `[0x00, 0x00]`. See the module-level comment on terminator design:
/// the second terminator byte `0x00` must be strictly less than any
/// escape-continuation byte (`0xFF`), so `encode(A) + any_tag` sorts
/// strictly before `encode(B) + any_tag` whenever A is a raw-byte
/// prefix of B — independent of the tag contents.
#[inline]
fn escape_byte_array(bytes: &[u8], buf: &mut Vec<u8>) {
    for &b in bytes {
        if b == 0x00 {
            buf.push(0x00);
            buf.push(0xFF);
        } else {
            buf.push(b);
        }
    }
    buf.push(0x00); // terminator byte 1
    buf.push(0x00); // terminator byte 2
}

/// IEEE 754 order-preserving encoding for f32.
/// Negative: flip all bits. Non-negative (incl. +0): flip sign bit only.
/// Result sorts correctly with unsigned byte comparison.
#[inline]
fn order_preserving_f32(v: f32) -> [u8; 4] {
    let bits = v.to_bits();
    let encoded = if bits >> 31 == 1 {
        !bits
    } else {
        bits ^ 0x8000_0000
    };
    encoded.to_be_bytes()
}

/// IEEE 754 order-preserving encoding for f64.
#[inline]
fn order_preserving_f64(v: f64) -> [u8; 8] {
    let bits = v.to_bits();
    let encoded = if bits >> 63 == 1 {
        !bits
    } else {
        bits ^ 0x8000_0000_0000_0000
    };
    encoded.to_be_bytes()
}

fn encode_tag(seq: SeqNum, op_type: OpType, buf: &mut Vec<u8>) -> Result<()> {
    // Bug K2 fix: guard against seq > SEQNUM_MAX. Without this check,
    // the subtraction wraps (u64 underflow in release, panic in debug),
    // producing a corrupted tag that breaks sort-order invariants.
    if seq.0 > SEQNUM_MAX.0 {
        return Err(MeruError::InvalidArgument(format!(
            "sequence number {} exceeds SEQNUM_MAX ({})",
            seq.0, SEQNUM_MAX.0
        )));
    }
    let inverted = SEQNUM_MAX.0 - seq.0;
    let tag = (inverted << 8) | (op_type as u64);
    buf.extend_from_slice(&tag.to_be_bytes());
    Ok(())
}

// ── Decoding ─────────────────────────────────────────────────────────────────

fn decode_pk_fields(pk_bytes: &[u8], schema: &TableSchema) -> Result<Vec<FieldValue>> {
    let mut pos = 0usize;
    let mut values = Vec::with_capacity(schema.primary_key.len());
    for &col_idx in &schema.primary_key {
        let col_type = &schema.columns[col_idx].col_type;
        let (val, consumed) = decode_field(&pk_bytes[pos..], col_type)?;
        values.push(val);
        pos += consumed;
    }
    if pos != pk_bytes.len() {
        return Err(MeruError::Corruption(format!(
            "{} leftover bytes after decoding all PK fields",
            pk_bytes.len() - pos
        )));
    }
    Ok(values)
}

fn decode_field(bytes: &[u8], col_type: &ColumnType) -> Result<(FieldValue, usize)> {
    match col_type {
        ColumnType::Boolean => {
            ensure_len(bytes, 1, "boolean")?;
            Ok((FieldValue::Boolean(bytes[0] != 0x00), 1))
        }
        ColumnType::Int32 => {
            ensure_len(bytes, 4, "int32")?;
            let u = u32::from_be_bytes(bytes[..4].try_into().unwrap()) ^ 0x8000_0000;
            Ok((FieldValue::Int32(u as i32), 4))
        }
        ColumnType::Int64 => {
            ensure_len(bytes, 8, "int64")?;
            let u = u64::from_be_bytes(bytes[..8].try_into().unwrap()) ^ 0x8000_0000_0000_0000;
            Ok((FieldValue::Int64(u as i64), 8))
        }
        ColumnType::Float => {
            ensure_len(bytes, 4, "float")?;
            let bits = u32::from_be_bytes(bytes[..4].try_into().unwrap());
            let orig = if bits >> 31 == 0 {
                bits ^ 0x8000_0000
            } else {
                !bits
            };
            Ok((FieldValue::Float(f32::from_bits(orig)), 4))
        }
        ColumnType::Double => {
            ensure_len(bytes, 8, "double")?;
            let bits = u64::from_be_bytes(bytes[..8].try_into().unwrap());
            let orig = if bits >> 63 == 0 {
                bits ^ 0x8000_0000_0000_0000
            } else {
                !bits
            };
            Ok((FieldValue::Double(f64::from_bits(orig)), 8))
        }
        ColumnType::FixedLenByteArray(n) => {
            let n = *n as usize;
            ensure_len(bytes, n, "fixed-len byte array")?;
            Ok((FieldValue::Bytes(Bytes::copy_from_slice(&bytes[..n])), n))
        }
        ColumnType::ByteArray => {
            let (val, consumed) = unescape_byte_array(bytes)?;
            Ok((FieldValue::Bytes(Bytes::from(val)), consumed))
        }
    }
}

fn ensure_len(bytes: &[u8], required: usize, field: &str) -> Result<()> {
    if bytes.len() < required {
        Err(MeruError::Corruption(format!(
            "truncated {field} field: need {required}, have {}",
            bytes.len()
        )))
    } else {
        Ok(())
    }
}

/// Inverse of `escape_byte_array`. Returns `(decoded_bytes, bytes_consumed_including_terminator)`.
///
/// Terminator is `[0x00, 0x00]`. Escape continuation is `[0x00, 0xFF]`
/// (represents a raw `0x00` byte). Any other `[0x00, X]` where
/// `X ∉ {0x00, 0xFF}` is invalid and returns `Corruption`.
fn unescape_byte_array(bytes: &[u8]) -> Result<(Vec<u8>, usize)> {
    let mut result = Vec::new();
    let mut i = 0;
    loop {
        if i >= bytes.len() {
            return Err(MeruError::Corruption(
                "unterminated escaped byte array".into(),
            ));
        }
        if bytes[i] == 0x00 {
            if i + 1 >= bytes.len() {
                return Err(MeruError::Corruption(
                    "truncated escape/terminator sequence".into(),
                ));
            }
            match bytes[i + 1] {
                0xFF => {
                    // Escape continuation: emit a raw 0x00.
                    result.push(0x00);
                    i += 2;
                }
                0x00 => {
                    // Two-byte terminator.
                    return Ok((result, i + 2));
                }
                other => {
                    return Err(MeruError::Corruption(format!(
                        "invalid byte sequence 0x00 followed by 0x{other:02X} \
                         (expected 0xFF for escape or 0x00 for terminator)"
                    )));
                }
            }
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        schema::{ColumnDef, ColumnType, TableSchema},
        sequence::{OpType, SeqNum, SEQNUM_MAX},
        value::FieldValue,
    };

    fn int64_schema() -> TableSchema {
        TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,

                ..Default::default()
            }],
            primary_key: vec![0],

            ..Default::default()
        }
    }

    fn bytearray_schema() -> TableSchema {
        TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "k".into(),
                col_type: ColumnType::ByteArray,
                nullable: false,

                ..Default::default()
            }],
            primary_key: vec![0],

            ..Default::default()
        }
    }

    fn composite_schema() -> TableSchema {
        TableSchema {
            table_name: "t".into(),
            columns: vec![
                ColumnDef {
                    name: "a".into(),
                    col_type: ColumnType::Int32,
                    nullable: false,

                    ..Default::default()
                },
                ColumnDef {
                    name: "b".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: false,

                    ..Default::default()
                },
                ColumnDef {
                    name: "v".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: true,

                    ..Default::default()
                },
            ],
            primary_key: vec![0, 1],

            ..Default::default()
        }
    }

    // ── Roundtrip tests ───────────────────────────────────────────────────────

    #[test]
    fn roundtrip_int64() {
        let s = int64_schema();
        let pk = vec![FieldValue::Int64(42)];
        let k = InternalKey::encode(&pk, SeqNum(100), OpType::Put, &s).unwrap();
        let d = InternalKey::decode(k.as_bytes(), &s).unwrap();
        assert_eq!(d.seq, SeqNum(100));
        assert_eq!(d.op_type, OpType::Put);
        assert_eq!(d.pk_values()[0], FieldValue::Int64(42));
    }

    #[test]
    fn roundtrip_negative_int64() {
        let s = int64_schema();
        let pk = vec![FieldValue::Int64(-1_000_000)];
        let k = InternalKey::encode(&pk, SeqNum(1), OpType::Delete, &s).unwrap();
        let d = InternalKey::decode(k.as_bytes(), &s).unwrap();
        assert_eq!(d.pk_values()[0], FieldValue::Int64(-1_000_000));
        assert_eq!(d.op_type, OpType::Delete);
    }

    #[test]
    fn roundtrip_bytearray_with_nulls() {
        let s = bytearray_schema();
        let raw = Bytes::from(vec![0x61u8, 0x00, 0xFF, 0x00, 0x62]);
        let pk = vec![FieldValue::Bytes(raw.clone())];
        let k = InternalKey::encode(&pk, SeqNum(7), OpType::Put, &s).unwrap();
        let d = InternalKey::decode(k.as_bytes(), &s).unwrap();
        match &d.pk_values()[0] {
            FieldValue::Bytes(b) => assert_eq!(&b[..], &raw[..]),
            _ => panic!("expected Bytes"),
        }
    }

    #[test]
    fn roundtrip_composite() {
        let s = composite_schema();
        let pk = vec![
            FieldValue::Int32(-5),
            FieldValue::Bytes(Bytes::from("hello\x00world")),
        ];
        let k = InternalKey::encode(&pk, SeqNum(99), OpType::Put, &s).unwrap();
        let d = InternalKey::decode(k.as_bytes(), &s).unwrap();
        assert_eq!(d.pk_values()[0], FieldValue::Int32(-5));
        match &d.pk_values()[1] {
            FieldValue::Bytes(b) => assert_eq!(b.as_ref(), b"hello\x00world"),
            _ => panic!("expected Bytes"),
        }
    }

    // ── Sort-order tests ──────────────────────────────────────────────────────

    #[test]
    fn newer_seq_sorts_first() {
        let s = int64_schema();
        let pk = vec![FieldValue::Int64(1)];
        let k_old = InternalKey::encode(&pk, SeqNum(1), OpType::Put, &s).unwrap();
        let k_new = InternalKey::encode(&pk, SeqNum(100), OpType::Put, &s).unwrap();
        assert!(
            k_new < k_old,
            "newer seq must sort before older for same PK"
        );
    }

    #[test]
    fn seek_latest_sorts_first() {
        let s = int64_schema();
        let pk = vec![FieldValue::Int64(1)];
        let seek = InternalKey::seek_latest(&pk, &s).unwrap();
        let real = InternalKey::encode(&pk, SeqNum(999_999), OpType::Put, &s).unwrap();
        assert!(seek <= real);
    }

    #[test]
    fn pk_ascending_order() {
        let s = int64_schema();
        let k1 = InternalKey::encode(&[FieldValue::Int64(1)], SeqNum(0), OpType::Put, &s).unwrap();
        let k2 = InternalKey::encode(&[FieldValue::Int64(2)], SeqNum(0), OpType::Put, &s).unwrap();
        assert!(k1 < k2);
    }

    #[test]
    fn negative_before_positive_int64() {
        let s = int64_schema();
        let neg =
            InternalKey::encode(&[FieldValue::Int64(-1)], SeqNum(0), OpType::Put, &s).unwrap();
        let pos = InternalKey::encode(&[FieldValue::Int64(1)], SeqNum(0), OpType::Put, &s).unwrap();
        assert!(neg < pos);
    }

    #[test]
    fn i64_min_before_zero_before_max() {
        let s = int64_schema();
        let kmin = InternalKey::encode(&[FieldValue::Int64(i64::MIN)], SeqNum(0), OpType::Put, &s)
            .unwrap();
        let kzero =
            InternalKey::encode(&[FieldValue::Int64(0)], SeqNum(0), OpType::Put, &s).unwrap();
        let kmax = InternalKey::encode(&[FieldValue::Int64(i64::MAX)], SeqNum(0), OpType::Put, &s)
            .unwrap();
        assert!(kmin < kzero && kzero < kmax);
    }

    #[test]
    fn bytearray_lexicographic_order() {
        let s = bytearray_schema();
        let ka = InternalKey::encode(
            &[FieldValue::Bytes(Bytes::from("abc"))],
            SeqNum(0),
            OpType::Put,
            &s,
        )
        .unwrap();
        let kb = InternalKey::encode(
            &[FieldValue::Bytes(Bytes::from("abd"))],
            SeqNum(0),
            OpType::Put,
            &s,
        )
        .unwrap();
        let kc = InternalKey::encode(
            &[FieldValue::Bytes(Bytes::from("abcd"))],
            SeqNum(0),
            OpType::Put,
            &s,
        )
        .unwrap();
        assert!(ka < kb);
        assert!(ka < kc);
    }

    /// Issue #7 regression: empty ByteArray, single-null ByteArray, and
    /// multi-null ByteArray all must produce distinct encodings AND
    /// sort in ascending order by their original bytewise comparison.
    /// The escape function is `0x00 → [0x00, 0xFF]`, terminator `0x00`.
    ///
    /// This test pins down the exact expected encodings and sort order
    /// so any regression in `escape_byte_array` / `unescape_byte_array`
    /// is caught here rather than manifesting as silent data loss in a
    /// stress test.
    #[test]
    fn bytearray_empty_and_null_keys_distinct_and_ordered() {
        let s = bytearray_schema();
        let k_empty = InternalKey::encode_user_key(&[FieldValue::Bytes(Bytes::new())], &s).unwrap();
        let k_null1 =
            InternalKey::encode_user_key(&[FieldValue::Bytes(Bytes::from_static(&[0u8]))], &s)
                .unwrap();
        let k_null2 =
            InternalKey::encode_user_key(&[FieldValue::Bytes(Bytes::from_static(&[0u8, 0u8]))], &s)
                .unwrap();
        let k_one =
            InternalKey::encode_user_key(&[FieldValue::Bytes(Bytes::from_static(&[0x01u8]))], &s)
                .unwrap();
        let k_null1_one = InternalKey::encode_user_key(
            &[FieldValue::Bytes(Bytes::from_static(&[0u8, 0x01u8]))],
            &s,
        )
        .unwrap();

        // Pinned exact encodings — shape is load-bearing for the
        // unescape logic and for the ordering invariants below.
        // Terminator is [0x00, 0x00] (two bytes).
        assert_eq!(k_empty, vec![0x00, 0x00]);
        assert_eq!(k_null1, vec![0x00, 0xFF, 0x00, 0x00]);
        assert_eq!(k_null2, vec![0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00]);
        assert_eq!(k_one, vec![0x01, 0x00, 0x00]);
        assert_eq!(k_null1_one, vec![0x00, 0xFF, 0x01, 0x00, 0x00]);

        // Distinct.
        assert_ne!(k_empty, k_null1);
        assert_ne!(k_null1, k_null2);
        assert_ne!(k_null1, k_null1_one);

        // Sort order: [] < [0x00] < [0x00, 0x00] < [0x00, 0x01] < [0x01].
        assert!(k_empty < k_null1);
        assert!(k_null1 < k_null2);
        assert!(k_null2 < k_null1_one);
        assert!(k_null1_one < k_one);
    }

    /// Round-trip every edge-case key through escape + unescape: the
    /// decoded bytes must equal the input. Regression guard for
    /// Issue #7's "data loss on persistence" scenario.
    #[test]
    fn bytearray_escape_unescape_roundtrip() {
        let cases: &[&[u8]] = &[
            &[],
            &[0x00],
            &[0x00, 0x00],
            &[0x00, 0x01],
            &[0x01, 0x00],
            &[0xFF],
            &[0x00, 0xFF],
            &[0xFF, 0x00],
            &[0x00, 0xFF, 0x00],
            b"hello",
            b"hello\0world",
        ];
        for case in cases {
            let mut buf = Vec::new();
            escape_byte_array(case, &mut buf);
            let (decoded, consumed) = unescape_byte_array(&buf).unwrap();
            assert_eq!(
                decoded.as_slice(),
                *case,
                "roundtrip failed for {case:?} (encoded={buf:?})"
            );
            assert_eq!(
                consumed,
                buf.len(),
                "consumed={consumed} but buf.len()={} for {case:?}",
                buf.len()
            );
        }
    }

    #[test]
    fn float_order_neg_before_pos() {
        let s = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "f".into(),
                col_type: ColumnType::Float,
                nullable: false,

                ..Default::default()
            }],
            primary_key: vec![0],

            ..Default::default()
        };
        let neg =
            InternalKey::encode(&[FieldValue::Float(-1.0)], SeqNum(0), OpType::Put, &s).unwrap();
        let pos =
            InternalKey::encode(&[FieldValue::Float(1.0)], SeqNum(0), OpType::Put, &s).unwrap();
        assert!(neg < pos);
    }

    /// Issue #49 regression: NaN must be rejected in Float PK columns.
    /// IEEE 754 defines multiple NaN bit patterns; the order-preserving
    /// encoding faithfully maps them to different byte sequences,
    /// producing non-deterministic keys for the same semantic value.
    #[test]
    fn nan_float_pk_rejected() {
        let s = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "f".into(),
                col_type: ColumnType::Float,
                nullable: false,
                ..Default::default()
            }],
            primary_key: vec![0],
            ..Default::default()
        };
        let err = InternalKey::encode(&[FieldValue::Float(f32::NAN)], SeqNum(1), OpType::Put, &s)
            .unwrap_err();
        match err {
            MeruError::InvalidArgument(msg) => assert!(msg.contains("NaN"), "msg: {msg}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Issue #49 regression: NaN must be rejected in Double PK columns.
    #[test]
    fn nan_double_pk_rejected() {
        let s = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "d".into(),
                col_type: ColumnType::Double,
                nullable: false,
                ..Default::default()
            }],
            primary_key: vec![0],
            ..Default::default()
        };
        let err = InternalKey::encode(&[FieldValue::Double(f64::NAN)], SeqNum(1), OpType::Put, &s)
            .unwrap_err();
        match err {
            MeruError::InvalidArgument(msg) => assert!(msg.contains("NaN"), "msg: {msg}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Non-NaN floats (including ±0, ±inf, subnormals) must encode fine.
    #[test]
    fn non_nan_floats_encode_fine() {
        let s = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "d".into(),
                col_type: ColumnType::Double,
                nullable: false,
                ..Default::default()
            }],
            primary_key: vec![0],
            ..Default::default()
        };
        for v in [
            0.0_f64,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
        ] {
            InternalKey::encode(&[FieldValue::Double(v)], SeqNum(1), OpType::Put, &s)
                .unwrap_or_else(|e| panic!("encoding {v} should succeed: {e}"));
        }
    }

    #[test]
    fn seqnum_max_roundtrip() {
        let s = int64_schema();
        let pk = vec![FieldValue::Int64(0)];
        let k = InternalKey::encode(&pk, SEQNUM_MAX, OpType::Put, &s).unwrap();
        let d = InternalKey::decode(k.as_bytes(), &s).unwrap();
        assert_eq!(d.seq, SEQNUM_MAX);
    }
}
