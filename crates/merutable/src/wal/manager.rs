//! `WalManager`: lifecycle management for WAL files in a directory.
//!
//! Responsibilities:
//! - Create the initial WAL file on open.
//! - Rotate to a new WAL file when the memtable flushes (old log can be reclaimed).
//! - GC WAL files whose max sequence has been persisted to Parquet.
//! - Recover all valid `WriteBatch`es from a WAL directory on startup.
//!
//! File naming: `{log_number:06}.wal` (e.g., `000001.wal`).

use std::{
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::types::{MeruError, Result, sequence::SeqNum};
use tracing::{debug, info, trace, warn};

use crate::wal::{
    batch::WriteBatch,
    reader::{FileSource, WalReader},
    writer::{FileSink, WalWriter},
};

/// A closed (rotated-out) WAL file that is a candidate for GC once its
/// `max_seq` has been durably flushed to Parquet.
#[derive(Debug, Clone, Copy)]
struct ClosedLog {
    log_number: u64,
    max_seq: SeqNum,
}

pub struct WalManager {
    dir: PathBuf,
    current: WalWriter,
    log_number: u64,
    next_log: AtomicU64,
    /// Max sequence number observed in the currently-open log. Tracked so
    /// that on rotate we can remember "log N holds seqs up to M" and GC
    /// the file once the engine confirms M has been flushed.
    current_log_max_seq: AtomicU64,
    /// FIFO of closed logs awaiting GC. Append-on-rotate, drain-on-flush
    /// via `mark_flushed_seq`.
    closed_logs: Mutex<Vec<ClosedLog>>,
    /// Max sequence number that has been durably flushed to Parquet.
    /// WAL files with `max_seq <= flushed_seq` are safe to delete.
    flushed_seq: AtomicU64,
}

impl WalManager {
    /// Open (or create) a WAL directory. Starts writing to a new log file.
    ///
    /// The parent directory is fsynced after creating the new log file so
    /// the file's directory entry is durable. Without this fsync, a crash
    /// between file creation and the first append can leave the WAL file
    /// unlinked on recovery — on reopen, `recover_from_dir` computes
    /// `max_log_number` from the surviving files, so the next log number
    /// can collide with the unlinked one (or the unlinked file's writes
    /// are silently lost).
    pub fn open(dir: &Path, initial_log_number: u64) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let log_number = initial_log_number;
        let path = log_path(dir, log_number);
        info!(log_number, path = %path.display(), "opening WAL");
        let sink = FileSink::create(&path)?;
        // fsync the parent directory so the new WAL file's directory
        // entry survives a crash before any append completes.
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
        let writer = WalWriter::new(Box::new(sink), log_number, false);
        Ok(Self {
            dir: dir.to_path_buf(),
            current: writer,
            log_number,
            next_log: AtomicU64::new(log_number + 1),
            current_log_max_seq: AtomicU64::new(0),
            closed_logs: Mutex::new(Vec::new()),
            flushed_seq: AtomicU64::new(0),
        })
    }

    /// Append a `WriteBatch` to the current WAL and fsync. Tracks the
    /// highest sequence number observed in this log file so a later
    /// `rotate()` can record (log, max_seq) as a GC candidate.
    pub fn append(&mut self, batch: &WriteBatch) -> Result<()> {
        trace!(
            seq = batch.sequence.0,
            records = batch.records.len(),
            "WAL append"
        );
        let encoded = batch.encode();
        let bytes = encoded.len() as u64;
        self.current.add_record(&encoded)?;
        self.current.sync()?;
        // Issue #14 Phase-1 metrics. Off the single-record hot path
        // (engine.put goes through write_internal, not directly
        // through WAL.append), but WAL append/sync are themselves
        // the durability contract; operators need visibility on
        // sync rate and byte volume to reason about fsync overhead.
        metrics::counter!("merutable.wal.appends_total").increment(1);
        metrics::counter!("merutable.wal.syncs_total").increment(1);
        metrics::counter!("merutable.wal.bytes_total").increment(bytes);
        let last = batch.last_seq().0;
        // fetch_max via CAS loop since AtomicU64 doesn't have fetch_max
        // until Rust 1.45+ is confirmed; use compare_exchange_weak for
        // portability with MSRV targets.
        let mut cur = self.current_log_max_seq.load(Ordering::Relaxed);
        while last > cur {
            match self.current_log_max_seq.compare_exchange_weak(
                cur,
                last,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        Ok(())
    }

    /// Fsync the current WAL file to ensure all appended records are durable.
    pub fn sync(&mut self) -> Result<()> {
        self.current.sync()
    }

    /// Rotate: close the current WAL and open a new one.
    /// Returns the log number of the file that was closed (the caller
    /// associates it with the immutable memtable that needs flushing).
    /// The closed log is pushed onto the internal GC candidate list along
    /// with the max sequence number observed in it — a subsequent
    /// `mark_flushed_seq` will delete the file once that seq is durably
    /// flushed.
    pub fn rotate(&mut self) -> Result<u64> {
        let old_log = self.log_number;
        let new_log = self.next_log.fetch_add(1, Ordering::Relaxed);
        debug!(old_log, new_log, "rotating WAL");

        // Full fsync of the old file BEFORE closing it. `sync_data`
        // persists the last appended bytes but can leave file-size
        // metadata stale on some filesystem configurations; recovery
        // reads `file.metadata()?.len()` to bound iteration, so a stale
        // size could truncate recovery short of the last record.
        self.current.sync_all()?;
        // Replace writer — old WalWriter dropped here (file closed by OS).
        let path = log_path(&self.dir, new_log);
        let sink = FileSink::create(&path)?;
        // fsync the parent directory so both (a) the old file's final
        // size is durable and (b) the new file's directory entry is
        // durable before the first append to it.
        if let Ok(d) = std::fs::File::open(&self.dir) {
            let _ = d.sync_all();
        }
        self.current = WalWriter::new(Box::new(sink), new_log, false);
        self.log_number = new_log;

        metrics::counter!("merutable.wal.rotations_total").increment(1);

        // Snapshot and reset the per-log max_seq counter. A log that never
        // received any writes is still a real file on disk and should be
        // GC'd eventually — record it with max_seq = 0, which will be
        // GC'd at the first `mark_flushed_seq` since the flushed seq is
        // always ≥ 0.
        let old_max = SeqNum(self.current_log_max_seq.swap(0, Ordering::AcqRel));
        let mut closed = self.closed_logs.lock().unwrap();
        closed.push(ClosedLog {
            log_number: old_log,
            max_seq: old_max,
        });
        drop(closed);

        Ok(old_log)
    }

    /// Mark that all sequences up to (and including) `seq` have been
    /// persisted to Parquet. Bumps the flushed high-water mark AND
    /// immediately GCs any closed WAL files whose `max_seq <= seq` — this
    /// is the only place WAL GC happens. Before Bug D was fixed the
    /// matching `gc_logs_before` was defined but never called by the
    /// engine, so the WAL directory grew without bound.
    pub fn mark_flushed_seq(&self, seq: SeqNum) {
        debug!(seq = seq.0, "WAL mark flushed");
        let mut current = self.flushed_seq.load(Ordering::Acquire);
        loop {
            if seq.0 <= current {
                break;
            }
            match self.flushed_seq.compare_exchange_weak(
                current,
                seq.0,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(now) => current = now,
            }
        }

        // GC any closed log whose contents are now durable.
        let flushed_seq = seq.0;
        let mut closed = self.closed_logs.lock().unwrap();
        let (to_delete, keep): (Vec<ClosedLog>, Vec<ClosedLog>) =
            closed.drain(..).partition(|c| c.max_seq.0 <= flushed_seq);
        *closed = keep;
        drop(closed);

        for entry in to_delete {
            let path = log_path(&self.dir, entry.log_number);
            debug!(
                log = entry.log_number,
                max_seq = entry.max_seq.0,
                path = %path.display(),
                "GC WAL file"
            );
            match std::fs::remove_file(&path) {
                Ok(_) => {
                    metrics::counter!("merutable.wal.files_gcd_total").increment(1);
                }
                Err(e) => {
                    // Treat "not found" as success — the file was already
                    // removed (possibly by a prior flush). Log the real errors
                    // at warn level; we don't fail the flush over a GC miss.
                    if e.kind() == std::io::ErrorKind::NotFound {
                        metrics::counter!("merutable.wal.files_gcd_total").increment(1);
                    } else {
                        tracing::warn!(
                            log = entry.log_number,
                            error = %e,
                            "failed to GC WAL file; will retry on next flush"
                        );
                        metrics::counter!("merutable.errors.io_total").increment(1);
                        // Re-queue so a later flush can retry.
                        self.closed_logs.lock().unwrap().push(entry);
                    }
                }
            }
        }
    }

    /// Delete WAL files whose log number is strictly less than `before_log`.
    /// Called after a successful Iceberg snapshot commit.
    pub fn gc_logs_before(&self, before_log: u64) -> Result<()> {
        let entries = std::fs::read_dir(&self.dir)?;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(log_num) = parse_log_number(&name) {
                if log_num < before_log {
                    let path = entry.path();
                    debug!(log_num, path = %path.display(), "GC WAL file");
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        Ok(())
    }

    /// Issue #22: register a previously-recovered WAL file as a closed
    /// log eligible for GC. Called by the engine's open path after
    /// `recover_from_dir()` so that the first `mark_flushed_seq()`
    /// after recovery sweeps the orphaned log files off disk. Without
    /// this registration, recovered WAL files persist forever and are
    /// re-replayed on every subsequent reopen — eventually racing
    /// background compaction into a data-loss window (see #22).
    ///
    /// `max_seq` should be the highest sequence number that appears in
    /// the recovered file; the engine plumbs `wal_max_seq` from
    /// `recover_from_dir` here, which is a conservative upper bound
    /// (ok: the first flush commit carries all recovered seqs too).
    pub fn register_closed_log(&self, log_number: u64, max_seq: SeqNum) {
        let mut closed = self.closed_logs.lock().unwrap();
        closed.push(ClosedLog {
            log_number,
            max_seq,
        });
    }

    /// Iterate the WAL files on disk. Returns `(log_number, path)` pairs
    /// so the engine can discover which files recovery produced. Exposed
    /// publicly for Issue #22's orphan-registration path.
    pub fn list_wal_files(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
        wal_files_in_dir(dir)
    }

    /// Recover all valid `WriteBatch`es from a WAL directory in log-number order.
    /// Returns `(batches, max_seq, max_log_number)`:
    /// - `batches`: all successfully decoded `WriteBatch`es in log order.
    /// - `max_seq`: the highest sequence number seen across all batches.
    /// - `max_log_number`: the highest WAL file number on disk (used by
    ///   the engine to compute a collision-free `next_log`; Bug W fix).
    pub fn recover_from_dir(dir: &Path) -> Result<(Vec<WriteBatch>, SeqNum, u64)> {
        let mut log_files = wal_files_in_dir(dir)?;
        log_files.sort_by_key(|(num, _)| *num);

        let mut batches = Vec::new();
        let mut max_seq = SeqNum(0);
        let mut max_log_number: u64 = 0;

        for (log_num, path) in log_files {
            if log_num > max_log_number {
                max_log_number = log_num;
            }
            info!(log_num, path = %path.display(), "recovering WAL");
            let source = FileSource::open(&path)?;
            let mut reader = WalReader::new(source);
            for record in reader.records() {
                match record {
                    Ok(payload) => {
                        match WriteBatch::decode(&payload) {
                            Ok(batch) => {
                                let last = batch.last_seq();
                                if last > max_seq {
                                    max_seq = last;
                                }
                                batches.push(batch);
                            }
                            Err(e) => {
                                // Corrupt batch in an otherwise valid record —
                                // stop reading THIS file (tail may be torn) but
                                // continue to the next WAL file, which belongs
                                // to an independent memtable generation.
                                warn!(
                                    log_num,
                                    error = %e,
                                    "corrupt WriteBatch in WAL file, skipping remainder of this file"
                                );
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        // CRC mismatch / truncation — stop reading this file
                        // (partial tail is expected on crash) but continue to
                        // the next WAL file.
                        warn!(
                            log_num,
                            error = %e,
                            "WAL read error, skipping remainder of this file"
                        );
                        break;
                    }
                }
            }
        }

        Ok((batches, max_seq, max_log_number))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn log_path(dir: &Path, log_number: u64) -> PathBuf {
    dir.join(format!("{log_number:06}.wal"))
}

fn parse_log_number(name: &str) -> Option<u64> {
    name.strip_suffix(".wal")?.parse().ok()
}

fn wal_files_in_dir(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    for entry in std::fs::read_dir(dir).map_err(MeruError::Io)?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(log_num) = parse_log_number(&name) {
            files.push((log_num, entry.path()));
        }
    }
    Ok(files)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::sequence::SeqNum;
    use bytes::Bytes;

    fn tmp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn open_write_recover() {
        let dir = tmp_dir();
        {
            let mut mgr = WalManager::open(dir.path(), 1).unwrap();
            let mut batch = WriteBatch::new(SeqNum(1));
            batch.put(Bytes::from("hello"), Bytes::from("world"));
            mgr.append(&batch).unwrap();

            let mut batch2 = WriteBatch::new(SeqNum(2));
            batch2.delete(Bytes::from("gone"));
            mgr.append(&batch2).unwrap();
        }

        let (batches, max_seq, max_log) = WalManager::recover_from_dir(dir.path()).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].sequence, SeqNum(1));
        assert_eq!(batches[1].sequence, SeqNum(2));
        assert_eq!(max_seq, SeqNum(2));
        assert_eq!(max_log, 1);
    }

    #[test]
    fn rotate_creates_new_file() {
        let dir = tmp_dir();
        let mut mgr = WalManager::open(dir.path(), 1).unwrap();

        let mut b = WriteBatch::new(SeqNum(1));
        b.put(Bytes::from("k"), Bytes::from("v"));
        mgr.append(&b).unwrap();

        let old = mgr.rotate().unwrap();
        assert_eq!(old, 1);
        assert_eq!(mgr.log_number, 2);

        let mut b2 = WriteBatch::new(SeqNum(2));
        b2.put(Bytes::from("k2"), Bytes::from("v2"));
        mgr.append(&b2).unwrap();

        let (batches, _, max_log) = WalManager::recover_from_dir(dir.path()).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(max_log, 2);
    }

    #[test]
    fn recovery_stops_on_truncation() {
        let dir = tmp_dir();
        {
            let mut mgr = WalManager::open(dir.path(), 1).unwrap();
            let mut b = WriteBatch::new(SeqNum(1));
            b.put(Bytes::from("key"), Bytes::from("val"));
            mgr.append(&b).unwrap();
        }
        // Truncate the WAL file to simulate a crash.
        let wal_path = dir.path().join("000001.wal");
        let meta = std::fs::metadata(&wal_path).unwrap();
        let truncated_len = meta.len() / 2;
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&wal_path)
            .unwrap();
        f.set_len(truncated_len).unwrap();

        // Recovery should not panic; it returns whatever it successfully read.
        let (batches, _, _) = WalManager::recover_from_dir(dir.path()).unwrap();
        // May get 0 or 1 batches depending on where truncation fell.
        let _ = batches;
    }

    #[test]
    fn corrupt_file_does_not_block_later_files() {
        let dir = tmp_dir();
        {
            let mut mgr = WalManager::open(dir.path(), 1).unwrap();

            // Write a batch to file 1.
            let mut b1 = WriteBatch::new(SeqNum(1));
            b1.put(Bytes::from("k1"), Bytes::from("v1"));
            mgr.append(&b1).unwrap();

            // Rotate → file 2.
            mgr.rotate().unwrap();

            // Write a batch to file 2.
            let mut b2 = WriteBatch::new(SeqNum(2));
            b2.put(Bytes::from("k2"), Bytes::from("v2"));
            mgr.append(&b2).unwrap();
        }

        // Corrupt the middle of file 1 so the CRC check fails.
        let wal1 = dir.path().join("000001.wal");
        let mut bytes = std::fs::read(&wal1).unwrap();
        assert!(bytes.len() > 10, "WAL file too short to corrupt");
        // Flip bytes in the payload region (past the 7-byte header).
        for i in 8..bytes.len().min(16) {
            bytes[i] ^= 0xFF;
        }
        std::fs::write(&wal1, &bytes).unwrap();

        // Recovery must still return the batch from file 2.
        let (batches, max_seq, max_log) = WalManager::recover_from_dir(dir.path()).unwrap();

        // File 1's batch may or may not survive depending on where
        // corruption landed, but file 2's batch MUST be present.
        let has_seq2 = batches.iter().any(|b| b.sequence == SeqNum(2));
        assert!(
            has_seq2,
            "batch from file 2 must be recovered despite corruption in file 1; got sequences: {:?}",
            batches.iter().map(|b| b.sequence).collect::<Vec<_>>()
        );
        assert_eq!(max_seq, SeqNum(2));
        assert_eq!(max_log, 2);
    }

    #[test]
    fn gc_removes_old_logs() {
        let dir = tmp_dir();
        let mut mgr = WalManager::open(dir.path(), 1).unwrap();
        mgr.rotate().unwrap(); // log 1 → log 2
        mgr.rotate().unwrap(); // log 2 → log 3

        mgr.gc_logs_before(3).unwrap();

        let files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        // Only log 3 (current) should remain.
        assert!(files.iter().all(|f| f.starts_with("000003")));
    }
}
