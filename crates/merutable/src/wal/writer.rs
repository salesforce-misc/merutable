//! WAL writer: appends `WriteBatch`es as physical records into 32 KiB blocks.
//!
//! A `WriteBatch` payload is fragmented across block boundaries using the
//! standard First/Middle/Last record types (same as RocksDB).
//!
//! CRC32c checksum covers `[record_type_byte ++ payload_fragment]`.

use std::io::Write;

use crate::types::{MeruError, Result};
use crc32fast::Hasher as Crc32;

use crate::wal::format::{RecordType, BLOCK_SIZE, HEADER_SIZE, RECYCLABLE_HEADER_SIZE};

/// Pluggable WAL sink. Local filesystem: `std::fs::File`.
/// Distributed/consensus WAL: implement this trait (e.g., Raft append log).
pub trait WalSink: Send + Sync {
    /// Append bytes to the log. Must be durable after `sync()`.
    fn append(&mut self, data: &[u8]) -> Result<()>;
    /// Fsync (fdatasync) — persist data only. Used on the hot append path.
    fn sync(&mut self) -> Result<()>;
    /// Full fsync — persist data + metadata. Used on rotate so the
    /// closed file's size/stat info is durable before the fd is dropped.
    /// Default delegates to `sync()` for implementations that don't
    /// distinguish (e.g., in-memory / network sinks).
    fn sync_all(&mut self) -> Result<()> {
        self.sync()
    }
    /// Close the sink cleanly.
    fn close(self: Box<Self>) -> Result<()>;
}

/// `WalSink` backed by a local file.
pub struct FileSink {
    file: std::fs::File,
}

impl FileSink {
    pub fn create(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self { file })
    }
}

impl WalSink for FileSink {
    fn append(&mut self, data: &[u8]) -> Result<()> {
        self.file.write_all(data).map_err(MeruError::Io)
    }
    fn sync(&mut self) -> Result<()> {
        // Hot-path fsync — `sync_data` (fdatasync on Unix) skips metadata
        // (mtime, etc.) but DOES persist file size, which is the only
        // metadata we actually need for durability. Correct and faster
        // than a full `sync_all` per append.
        self.file.sync_data().map_err(MeruError::Io)
    }
    fn sync_all(&mut self) -> Result<()> {
        // Full fsync including metadata. Called on rotate so the file's
        // size + timestamps are durably committed before its fd is
        // dropped — `sync_data` alone can leave stat metadata stale in
        // some filesystems (e.g., older ext4 journaled modes), which
        // would make recovery misread the file length on the closed log.
        self.file.sync_all().map_err(MeruError::Io)
    }
    fn close(self: Box<Self>) -> Result<()> {
        // `file` dropped on return — OS closes fd.
        Ok(())
    }
}

/// Writes `WriteBatch` payloads as WAL records.
pub struct WalWriter {
    sink: Box<dyn WalSink>,
    /// Byte offset within the current 32 KiB block.
    block_offset: usize,
    log_number: u64,
    recyclable: bool,
}

impl WalWriter {
    pub fn new(sink: Box<dyn WalSink>, log_number: u64, recyclable: bool) -> Self {
        Self {
            sink,
            block_offset: 0,
            log_number,
            recyclable,
        }
    }

    /// Append one encoded `WriteBatch` payload. Fragments across block boundaries.
    /// Caller must have encoded the batch via `WriteBatch::encode()`.
    pub fn add_record(&mut self, payload: &[u8]) -> Result<()> {
        let mut remaining = payload;
        let mut is_first = true;

        let header_size = self.effective_header_size();

        while !remaining.is_empty() {
            let available = self.available_in_block();

            // If the block can't even fit a header, pad with zeros and start a new one.
            // Note: the threshold must be the *effective* header size — recyclable
            // format needs 11 bytes, non-recyclable 7. A 7-byte threshold on a
            // recyclable writer with 8..=10 bytes remaining would underflow
            // `capacity = available - header_size` below.
            if available < header_size {
                let pad = vec![0u8; available];
                self.sink.append(&pad)?;
                self.block_offset = 0;
            }

            let available = self.available_in_block();
            let capacity = available - header_size;
            let fragment = &remaining[..remaining.len().min(capacity)];
            remaining = &remaining[fragment.len()..];

            let rtype = if is_first && remaining.is_empty() {
                if self.recyclable {
                    RecordType::RecyclableFull
                } else {
                    RecordType::Full
                }
            } else if is_first {
                if self.recyclable {
                    RecordType::RecyclableFirst
                } else {
                    RecordType::First
                }
            } else if remaining.is_empty() {
                if self.recyclable {
                    RecordType::RecyclableLast
                } else {
                    RecordType::Last
                }
            } else {
                if self.recyclable {
                    RecordType::RecyclableMiddle
                } else {
                    RecordType::Middle
                }
            };

            self.emit_physical_record(rtype, fragment)?;
            is_first = false;
        }
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.sink.sync()
    }

    pub fn sync_all(&mut self) -> Result<()> {
        self.sink.sync_all()
    }

    pub fn close(self) -> Result<()> {
        self.sink.close()
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn available_in_block(&self) -> usize {
        BLOCK_SIZE - self.block_offset
    }

    fn effective_header_size(&self) -> usize {
        if self.recyclable {
            RECYCLABLE_HEADER_SIZE
        } else {
            HEADER_SIZE
        }
    }

    fn emit_physical_record(&mut self, rtype: RecordType, payload: &[u8]) -> Result<()> {
        debug_assert!(payload.len() <= u16::MAX as usize);
        let header_size = self.effective_header_size();

        // CRC covers: [type_byte ++ payload].
        let mut hasher = Crc32::new();
        hasher.update(&[rtype as u8]);
        hasher.update(payload);
        let crc = hasher.finalize();

        // Build header.
        let mut header = [0u8; 11]; // max recyclable header
        header[..4].copy_from_slice(&crc.to_le_bytes());
        header[4..6].copy_from_slice(&(payload.len() as u16).to_le_bytes());
        header[6] = rtype as u8;
        if self.recyclable {
            header[7..11].copy_from_slice(&(self.log_number as u32).to_le_bytes());
        }

        self.sink.append(&header[..header_size])?;
        self.sink.append(payload)?;
        self.block_offset += header_size + payload.len();
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::reader::{VecSource, WalReader};

    fn make_writer(recyclable: bool) -> (WalWriter, std::sync::Arc<std::sync::Mutex<Vec<u8>>>) {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = VecSink { buf: buf.clone() };
        let writer = WalWriter::new(Box::new(sink), 1, recyclable);
        (writer, buf)
    }

    struct VecSink {
        buf: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    }
    impl WalSink for VecSink {
        fn append(&mut self, data: &[u8]) -> Result<()> {
            self.buf.lock().unwrap().extend_from_slice(data);
            Ok(())
        }
        fn sync(&mut self) -> Result<()> {
            Ok(())
        }
        fn close(self: Box<Self>) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn write_small_record_and_read_back() {
        let (mut writer, buf) = make_writer(false);
        let payload = b"hello merutable";
        writer.add_record(payload).unwrap();
        let data = buf.lock().unwrap().clone();
        let mut reader = WalReader::new(VecSource::new(data));
        let records: Vec<_> = reader.records().collect();
        assert_eq!(records.len(), 1);
        assert_eq!(&records[0].as_ref().unwrap()[..], payload);
    }

    #[test]
    fn write_multiple_records() {
        let (mut writer, buf) = make_writer(false);
        let payloads: &[&[u8]] = &[b"alpha", b"beta", b"gamma delta"];
        for p in payloads {
            writer.add_record(p).unwrap();
        }
        let data = buf.lock().unwrap().clone();
        let mut reader = WalReader::new(VecSource::new(data));
        let records: Vec<_> = reader.records().map(|r| r.unwrap()).collect();
        assert_eq!(records.len(), payloads.len());
        for (got, expected) in records.iter().zip(payloads.iter()) {
            assert_eq!(&got[..], *expected);
        }
    }

    #[test]
    fn fragmentation_across_block_boundary() {
        let (mut writer, buf) = make_writer(false);
        // Payload larger than one block to force fragmentation.
        let payload = vec![0xAAu8; BLOCK_SIZE + 100];
        writer.add_record(&payload).unwrap();
        let data = buf.lock().unwrap().clone();
        let mut reader = WalReader::new(VecSource::new(data));
        let records: Vec<_> = reader.records().map(|r| r.unwrap()).collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0], payload);
    }

    #[test]
    fn recyclable_format_roundtrip() {
        let (mut writer, buf) = make_writer(true);
        writer.add_record(b"recyclable test").unwrap();
        let data = buf.lock().unwrap().clone();
        // make_writer uses log_number = 1; the reader must assert the
        // same value to validate the embedded log_number check.
        let mut reader = WalReader::new_recyclable(VecSource::new(data), 1);
        let records: Vec<_> = reader.records().map(|r| r.unwrap()).collect();
        assert_eq!(records.len(), 1);
        assert_eq!(&records[0][..], b"recyclable test");
    }

    /// Regression for a writer bug where `MIN_REMAINING = HEADER_SIZE = 7`
    /// was used unconditionally — when a recyclable writer had 7..=10 bytes
    /// remaining in the current block, padding was skipped and then
    /// `capacity = available - 11` underflowed (panic in debug, wrap in
    /// release). The fix pads when `available < effective_header_size`.
    #[test]
    fn recyclable_padding_respects_11_byte_header() {
        // Craft a payload that lands the writer with exactly 8 bytes
        // remaining in the current block. An 11-byte recyclable header
        // then cannot fit, so the writer must pad.
        //
        // First record consumes BLOCK_SIZE - 8 bytes on disk. With an
        // 11-byte header, the payload must be BLOCK_SIZE - 8 - 11 = 32749.
        let (mut writer, buf) = make_writer(true);
        let first = vec![0x5Au8; BLOCK_SIZE - 8 - 11];
        writer.add_record(&first).unwrap();
        // Second record is small and would have underflown capacity on
        // the buggy path. On the fixed path it pads the 8-byte tail with
        // zeros, starts a new block, and writes a RecyclableFull record.
        let second = b"after-tail";
        writer.add_record(second).unwrap();

        let data = buf.lock().unwrap().clone();
        // File must contain at least one full block of padding (first
        // record + 8 zero tail bytes) plus the second record in block 2.
        assert!(data.len() >= BLOCK_SIZE + 11 + second.len());
        // The 8 bytes immediately before block 2 must be zero padding.
        let pad_start = BLOCK_SIZE - 8;
        assert!(
            data[pad_start..BLOCK_SIZE].iter().all(|&b| b == 0),
            "block 1 tail must be zero-padded; got {:x?}",
            &data[pad_start..BLOCK_SIZE]
        );

        let mut reader = WalReader::new_recyclable(VecSource::new(data), 1);
        let records: Vec<_> = reader.records().map(|r| r.unwrap()).collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0], first);
        assert_eq!(&records[1][..], second);
    }

    /// A recyclable reader configured with `expected_log_number = 7` must
    /// treat records written with `log_number = 1` as stale (clean EOF),
    /// NOT silently return their payloads as if they were live. This is
    /// the entire point of the recyclable format.
    #[test]
    fn recyclable_reader_rejects_stale_log_number() {
        let (mut writer, buf) = make_writer(true); // writer uses log_number=1
        writer.add_record(b"stale bytes").unwrap();
        let data = buf.lock().unwrap().clone();
        let mut reader = WalReader::new_recyclable(VecSource::new(data), 7);
        let records: Vec<_> = reader.records().collect();
        assert!(
            records.is_empty(),
            "stale recyclable records must not be surfaced"
        );
    }

    /// A non-recyclable reader fed a recyclable-format log must refuse
    /// the type byte rather than silently mis-parsing the 11-byte header
    /// as a 7-byte header.
    #[test]
    fn non_recyclable_reader_rejects_recyclable_records() {
        let (mut writer, buf) = make_writer(true); // recyclable format
        writer.add_record(b"recyc payload").unwrap();
        let data = buf.lock().unwrap().clone();
        let mut reader = WalReader::new(VecSource::new(data));
        let records: Vec<_> = reader.records().collect();
        // Reader stops cleanly (no panic, no surfacing of wrong bytes).
        assert!(records.is_empty());
    }
}
