//! `WriteBatch` — the atomic unit written to the WAL and applied to the memtable.
//!
//! # Wire format
//!
//! ```text
//! [sequence_number: u64 LE][record_count: u32 LE][record…]*
//!
//! record:
//!   [op_type: u8]
//!   [key_len: varint u64]
//!   [key: key_len bytes]
//!   if op_type == Put:
//!     [val_len: varint u64]
//!     [val: val_len bytes]
//! ```
//!
//! Varint encoding: little-endian base-128 (standard LEB128).

use crate::types::{
    MeruError, Result,
    sequence::{OpType, SeqNum},
};
use bytes::{BufMut, Bytes, BytesMut};

#[derive(Clone, Debug)]
pub struct BatchRecord {
    pub op_type: OpType,
    pub user_key: Bytes,
    /// `None` for `Delete`.
    pub value: Option<Bytes>,
}

#[derive(Clone, Debug)]
pub struct WriteBatch {
    pub sequence: SeqNum,
    pub records: Vec<BatchRecord>,
}

impl WriteBatch {
    pub fn new(sequence: SeqNum) -> Self {
        Self {
            sequence,
            records: Vec::new(),
        }
    }

    pub fn put(&mut self, key: Bytes, value: Bytes) {
        self.records.push(BatchRecord {
            op_type: OpType::Put,
            user_key: key,
            value: Some(value),
        });
    }

    pub fn delete(&mut self, key: Bytes) {
        // Back-compat shim. Issue #33 fix: prefer
        // `delete_with_pre_image` so the change feed surfaces
        // a meaningful pre-image on the resulting DELETE record.
        self.delete_with_pre_image(key, Bytes::new());
    }

    /// Issue #33: delete carrying a pre-image payload for the
    /// change feed. `pre_image` is the encoded row state at the
    /// instant before the delete; empty bytes signal "no prior
    /// live state" (the key was already tombstoned or never
    /// existed). Persists through compaction because the memtable
    /// + SST formats keep the full `EntryValue::value`.
    pub fn delete_with_pre_image(&mut self, key: Bytes, pre_image: Bytes) {
        self.records.push(BatchRecord {
            op_type: OpType::Delete,
            user_key: key,
            // Store as `Some` even when empty so the encode/decode
            // path treats Delete records identically to Put: a
            // varint length prefix always follows the user key.
            value: Some(pre_image),
        });
    }

    /// Encode to wire bytes for WAL append.
    ///
    /// Issue #33 breaking change: Delete records now ALWAYS carry
    /// a varint-prefixed value segment (possibly zero-length).
    /// Pre-#33 WAL files encoded Delete records with no value
    /// segment at all; they are not readable by the post-#33
    /// decoder. merutable is pre-0.1 — WAL-format incompat is
    /// acceptable at this phase, and fresh catalogs are
    /// unaffected.
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(16 + self.records.len() * 32);
        buf.put_u64_le(self.sequence.0);
        buf.put_u32_le(self.records.len() as u32);
        for rec in &self.records {
            buf.put_u8(rec.op_type as u8);
            put_varint(&mut buf, rec.user_key.len() as u64);
            buf.put_slice(&rec.user_key);
            // Both Put and Delete records carry a value segment.
            // Delete with no pre-image → empty bytes (varint(0)).
            let val = rec.value.as_ref().cloned().unwrap_or_default();
            put_varint(&mut buf, val.len() as u64);
            buf.put_slice(&val);
        }
        buf.freeze()
    }

    /// Decode from wire bytes.
    pub fn decode(mut data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            return Err(MeruError::Corruption("WriteBatch too short".into()));
        }
        let sequence = SeqNum(read_u64_le(&mut data)?);
        let count = read_u32_le(&mut data)? as usize;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let op_byte = read_byte(&mut data)?;
            let op_type = match op_byte {
                0x00 => OpType::Delete,
                0x01 => OpType::Put,
                b => return Err(MeruError::Corruption(format!("unknown op_type {b:#x}"))),
            };
            let key_len = read_varint(&mut data)?;
            let user_key = read_bytes(&mut data, key_len as usize)?;
            // Issue #33: both Put and Delete records carry a
            // varint-prefixed value segment. Delete records use
            // the value to store the pre-image row at delete
            // time; empty bytes (varint(0)) mean "no prior
            // state."
            let val_len = read_varint(&mut data)?;
            let value = Some(read_bytes(&mut data, val_len as usize)?);
            records.push(BatchRecord {
                op_type,
                user_key,
                value,
            });
        }
        Ok(Self { sequence, records })
    }

    /// Maximum sequence number assigned across all records in this batch.
    pub fn last_seq(&self) -> SeqNum {
        SeqNum(self.sequence.0 + self.records.len().saturating_sub(1) as u64)
    }
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

fn put_varint(buf: &mut BytesMut, mut val: u64) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.put_u8(byte);
            break;
        }
        buf.put_u8(byte | 0x80);
    }
}

// ── Decoding helpers ──────────────────────────────────────────────────────────

fn read_byte(data: &mut &[u8]) -> Result<u8> {
    if data.is_empty() {
        return Err(MeruError::Corruption("unexpected end of WriteBatch".into()));
    }
    let b = data[0];
    *data = &data[1..];
    Ok(b)
}

fn read_u64_le(data: &mut &[u8]) -> Result<u64> {
    if data.len() < 8 {
        return Err(MeruError::Corruption("truncated u64".into()));
    }
    let val = u64::from_le_bytes(data[..8].try_into().unwrap());
    *data = &data[8..];
    Ok(val)
}

fn read_u32_le(data: &mut &[u8]) -> Result<u32> {
    if data.len() < 4 {
        return Err(MeruError::Corruption("truncated u32".into()));
    }
    let val = u32::from_le_bytes(data[..4].try_into().unwrap());
    *data = &data[4..];
    Ok(val)
}

fn read_varint(data: &mut &[u8]) -> Result<u64> {
    let mut val = 0u64;
    let mut shift = 0u32;
    loop {
        let b = read_byte(data)?;
        val |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(val);
        }
        shift += 7;
        if shift >= 64 {
            return Err(MeruError::Corruption("varint overflow".into()));
        }
    }
}

fn read_bytes(data: &mut &[u8], len: usize) -> Result<Bytes> {
    if data.len() < len {
        return Err(MeruError::Corruption(format!(
            "need {len} bytes, have {}",
            data.len()
        )));
    }
    let out = Bytes::copy_from_slice(&data[..len]);
    *data = &data[len..];
    Ok(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::sequence::SeqNum;

    #[test]
    fn roundtrip_put_and_delete() {
        let mut batch = WriteBatch::new(SeqNum(42));
        batch.put(Bytes::from("key1"), Bytes::from("value1"));
        batch.delete(Bytes::from("key2"));
        batch.put(Bytes::from("key3"), Bytes::from_static(b"\x00\xFF\x00"));

        let encoded = batch.encode();
        let decoded = WriteBatch::decode(&encoded).unwrap();

        assert_eq!(decoded.sequence, SeqNum(42));
        assert_eq!(decoded.records.len(), 3);
        assert_eq!(decoded.records[0].op_type, OpType::Put);
        assert_eq!(decoded.records[0].user_key, Bytes::from("key1"));
        assert_eq!(decoded.records[0].value.as_deref().unwrap(), b"value1");
        assert_eq!(decoded.records[1].op_type, OpType::Delete);
        // Issue #33: Delete records now carry an (empty-for-no-
        // pre-image) value segment. Empty bytes → Some(0-len).
        assert_eq!(decoded.records[1].value.as_deref(), Some(&[][..]));
        assert_eq!(
            decoded.records[2].value.as_deref().unwrap(),
            b"\x00\xFF\x00"
        );
    }

    #[test]
    fn empty_batch_roundtrip() {
        let batch = WriteBatch::new(SeqNum(0));
        let decoded = WriteBatch::decode(&batch.encode()).unwrap();
        assert_eq!(decoded.records.len(), 0);
    }

    #[test]
    fn large_value_roundtrip() {
        let mut batch = WriteBatch::new(SeqNum(1));
        let big_val = Bytes::from(vec![0xABu8; 128 * 1024]);
        batch.put(Bytes::from("k"), big_val.clone());
        let decoded = WriteBatch::decode(&batch.encode()).unwrap();
        assert_eq!(decoded.records[0].value.as_deref().unwrap(), &big_val[..]);
    }

    #[test]
    fn last_seq() {
        let mut batch = WriteBatch::new(SeqNum(10));
        batch.put(Bytes::from("a"), Bytes::new());
        batch.put(Bytes::from("b"), Bytes::new());
        assert_eq!(batch.last_seq(), SeqNum(11));
    }

    #[test]
    fn corruption_detected() {
        let batch = WriteBatch::new(SeqNum(1));
        let mut encoded = batch.encode().to_vec();
        if !encoded.is_empty() {
            encoded[0] ^= 0xFF;
        }
        // seq is still decodeable but record_count would be mangled
        // Just verify decode doesn't panic on short input
        assert!(WriteBatch::decode(&encoded[..3]).is_err());
    }
}
