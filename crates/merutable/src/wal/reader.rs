//! WAL reader: recovers `WriteBatch` payloads from the on-disk record stream.
//!
//! Recovery semantics:
//! - CRC mismatch on any record → stop and return all records seen so far (partial
//!   tail record is NOT a hard error — it's a common crash scenario).
//! - Zero-padded tail blocks (written during block pre-allocation) are skipped.
//! - Truncated data at end of file → same treatment as CRC mismatch.
//!
//! The reader reassembles fragmented records (First/Middle/Last) before returning.

use crate::types::{MeruError, Result};
use crc32fast::Hasher as Crc32;

use crate::wal::format::{BLOCK_SIZE, HEADER_SIZE, RECYCLABLE_HEADER_SIZE, RecordType};

/// Pluggable WAL source.
pub trait WalSource: Send {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<usize>;
    fn size(&self) -> Result<u64>;
}

/// In-memory `WalSource` for testing.
pub struct VecSource {
    data: Vec<u8>,
}

impl VecSource {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }
}

impl WalSource for VecSource {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        let start = offset as usize;
        if start >= self.data.len() {
            return Ok(0);
        }
        let end = (start + buf.len()).min(self.data.len());
        let n = end - start;
        buf[..n].copy_from_slice(&self.data[start..end]);
        Ok(n)
    }
    fn size(&self) -> Result<u64> {
        Ok(self.data.len() as u64)
    }
}

/// File-backed `WalSource`.
pub struct FileSource {
    file: std::fs::File,
    size: u64,
}

impl FileSource {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let size = file.metadata()?.len();
        Ok(Self { file, size })
    }
}

impl WalSource for FileSource {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        use std::io::{Read, Seek, SeekFrom};
        self.file.seek(SeekFrom::Start(offset))?;
        let n = self.file.read(buf)?;
        Ok(n)
    }
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }
}

/// Reads WAL records, reassembling fragments.
pub struct WalReader {
    source: Box<dyn WalSource>,
    file_offset: u64,
    block_offset: usize,
    recyclable: bool,
    /// Expected log number for the *recyclable* format. When `recyclable`
    /// is true, every physical record's embedded `log_number` must equal
    /// this value; a mismatch means the record belongs to a previous
    /// generation of a recycled file and the reader treats it as a clean
    /// EOF. Ignored when `recyclable` is false.
    expected_log_number: u32,
    eof: bool,
}

impl WalReader {
    /// Construct a non-recyclable reader. The reader will emit standard
    /// 7-byte headers and will not validate any log_number field.
    pub fn new<S: WalSource + 'static>(source: S) -> Self {
        Self {
            source: Box::new(source),
            file_offset: 0,
            block_offset: 0,
            recyclable: false,
            expected_log_number: 0,
            eof: false,
        }
    }

    /// Construct a recyclable reader. Every physical record's embedded
    /// `log_number` must equal `expected_log_number`; records from an
    /// older generation (stale bytes left over in a recycled file) are
    /// detected via the mismatch and the reader stops cleanly at that
    /// point, exactly as RocksDB does.
    pub fn new_recyclable<S: WalSource + 'static>(source: S, expected_log_number: u32) -> Self {
        Self {
            source: Box::new(source),
            file_offset: 0,
            block_offset: 0,
            recyclable: true,
            expected_log_number,
            eof: false,
        }
    }

    /// Iterator over reassembled record payloads.
    /// On CRC error or truncation, the iterator stops (does not yield an error item
    /// for partial tail records, which are expected crash artifacts).
    pub fn records(&mut self) -> RecordIter<'_> {
        RecordIter { reader: self }
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Read the next complete record payload (reassembled). Returns `None` at EOF
    /// or on CRC error (treated as clean stop).
    fn next_record(&mut self) -> Option<Result<Vec<u8>>> {
        if self.eof {
            return None;
        }
        let mut assembled: Vec<u8> = Vec::new();
        let mut in_fragment = false;

        loop {
            match self.read_physical_record() {
                Ok(None) => {
                    self.eof = true;
                    if in_fragment {
                        // Truncated mid-record — treat as clean stop.
                        return None;
                    }
                    return None;
                }
                Ok(Some((rtype, payload))) => match rtype {
                    RecordType::Full | RecordType::RecyclableFull => {
                        if in_fragment {
                            return Some(Err(MeruError::Corruption(
                                "Full record while assembling fragment".into(),
                            )));
                        }
                        return Some(Ok(payload));
                    }
                    RecordType::First | RecordType::RecyclableFirst => {
                        if in_fragment {
                            return Some(Err(MeruError::Corruption(
                                "First record while already in fragment".into(),
                            )));
                        }
                        assembled = payload;
                        in_fragment = true;
                    }
                    RecordType::Middle | RecordType::RecyclableMiddle => {
                        if !in_fragment {
                            return Some(Err(MeruError::Corruption(
                                "Middle record without First".into(),
                            )));
                        }
                        assembled.extend_from_slice(&payload);
                    }
                    RecordType::Last | RecordType::RecyclableLast => {
                        if !in_fragment {
                            return Some(Err(MeruError::Corruption(
                                "Last record without First".into(),
                            )));
                        }
                        assembled.extend_from_slice(&payload);
                        return Some(Ok(assembled));
                    }
                },
                Err(_) => {
                    // CRC mismatch or truncation → clean stop (not panic).
                    self.eof = true;
                    return None;
                }
            }
        }
    }

    /// Read one physical record from the current position.
    /// Returns `Ok(None)` at EOF. Returns `Err` on CRC mismatch or truncation.
    fn read_physical_record(&mut self) -> Result<Option<(RecordType, Vec<u8>)>> {
        let header_size = if self.recyclable {
            RECYCLABLE_HEADER_SIZE
        } else {
            HEADER_SIZE
        };

        // Bug F17 fix: the old code recursed on zero-pad sentinels, which
        // stack-overflows on a large all-zero WAL file (~32K recursive calls
        // for a 1 GB file). Loop instead.
        loop {
            // Skip zero-padded block tail if not enough space for a header.
            if BLOCK_SIZE - self.block_offset < header_size {
                let skip = BLOCK_SIZE - self.block_offset;
                self.file_offset += skip as u64;
                self.block_offset = 0;
            }

            // Read header.
            let mut header = [0u8; RECYCLABLE_HEADER_SIZE];
            let n = self
                .source
                .read_at(self.file_offset, &mut header[..header_size])?;
            if n == 0 {
                return Ok(None);
            }
            if n < header_size {
                return Err(MeruError::Corruption("truncated WAL header".into()));
            }

            let stored_crc = u32::from_le_bytes(header[..4].try_into().unwrap());
            let length = u16::from_le_bytes(header[4..6].try_into().unwrap()) as usize;
            let type_byte = header[6];

            // Check for zero-pad sentinel — skip to next block and retry.
            if stored_crc == 0 && length == 0 && type_byte == 0 {
                let skip = BLOCK_SIZE - self.block_offset;
                self.file_offset += skip as u64;
                self.block_offset = 0;
                continue;
            }

            let rtype = RecordType::from_byte(type_byte).ok_or_else(|| {
                MeruError::Corruption(format!("unknown record type {type_byte:#x}"))
            })?;

            // Recyclable format carries a 4-byte log number after the type
            // byte. A mismatch means the record is left over from a previous
            // generation of a recycled log file; signal as clean EOF so the
            // caller stops recovery here (matches RocksDB's semantics).
            if self.recyclable {
                if !rtype.is_recyclable() {
                    return Err(MeruError::Corruption(format!(
                        "non-recyclable record type {type_byte:#x} in recyclable log"
                    )));
                }
                let embedded_log =
                    u32::from_le_bytes(header[7..RECYCLABLE_HEADER_SIZE].try_into().unwrap());
                if embedded_log != self.expected_log_number {
                    return Ok(None);
                }
            } else if rtype.is_recyclable() {
                return Err(MeruError::Corruption(format!(
                    "recyclable record type {type_byte:#x} in non-recyclable log"
                )));
            }

            // Read payload.
            let payload_offset = self.file_offset + header_size as u64;
            let mut payload = vec![0u8; length];
            if length > 0 {
                let n = self.source.read_at(payload_offset, &mut payload)?;
                if n < length {
                    return Err(MeruError::Corruption(format!(
                        "truncated WAL payload: need {length}, got {n}"
                    )));
                }
            }

            // Verify CRC.
            let mut hasher = Crc32::new();
            hasher.update(&[type_byte]);
            hasher.update(&payload);
            let computed_crc = hasher.finalize();
            if computed_crc != stored_crc {
                return Err(MeruError::Corruption(format!(
                    "WAL CRC mismatch: stored {stored_crc:#x}, computed {computed_crc:#x}"
                )));
            }

            let total = header_size + length;
            self.file_offset += total as u64;
            self.block_offset = (self.block_offset + total) % BLOCK_SIZE;

            return Ok(Some((rtype, payload)));
        } // end loop (Bug F17)
    }
}

pub struct RecordIter<'a> {
    reader: &'a mut WalReader,
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = Result<Vec<u8>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.reader.next_record()
    }
}
