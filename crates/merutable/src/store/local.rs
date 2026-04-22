//! `LocalFileStore`: file-system-backed object store using atomic write-then-rename.

use std::path::{Path, PathBuf};

use crate::types::{MeruError, Result};
use async_trait::async_trait;
use bytes::Bytes;

use crate::store::traits::MeruStore;

/// Object store backed by a local directory.
pub struct LocalFileStore {
    root: PathBuf,
}

impl LocalFileStore {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn full_path(&self, path: &str) -> PathBuf {
        self.root.join(path)
    }
}

#[async_trait]
impl MeruStore for LocalFileStore {
    async fn put(&self, path: &str, data: Bytes) -> Result<()> {
        let full = self.full_path(path);
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        // Atomic: write to tmp, fsync tmp, rename, fsync parent dir.
        //
        // Without the fsync-before-rename, rename might be applied to
        // non-durable data and a crash loses the content. Without the
        // fsync-after-rename, the directory entry itself (the link
        // from `path` to the inode) is not durable on ext4/btrfs and
        // a crash can "roll back" the rename — leaving the caller
        // believing the write succeeded when the file is gone on reboot.
        // Temp file uses a unique suffix (PID + counter) to prevent
        // races when two callers put() the same path concurrently.
        use std::sync::atomic::{AtomicU64, Ordering};
        static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = full.with_extension(format!("tmp.{}.{seq}", std::process::id()));
        tokio::fs::write(&tmp, &data).await?;
        // fsync the file contents before the rename so the data is
        // durable under whatever name we link it to. Errors here are
        // hard failures — proceeding with the rename after a failed
        // fsync would leave non-durable data under the final name.
        let f = tokio::fs::File::open(&tmp).await.map_err(MeruError::Io)?;
        f.sync_all().await.map_err(MeruError::Io)?;
        drop(f);
        tokio::fs::rename(&tmp, &full).await?;
        // fsync the parent directory so the rename's directory-entry
        // change is durably committed.
        if let Some(parent) = full.parent() {
            let dir = tokio::fs::File::open(parent).await.map_err(MeruError::Io)?;
            dir.sync_all().await.map_err(MeruError::Io)?;
        }
        Ok(())
    }

    async fn get(&self, path: &str) -> Result<Bytes> {
        let full = self.full_path(path);
        let data = tokio::fs::read(&full)
            .await
            .map_err(|e| MeruError::ObjectStore(format!("{}: {e}", full.display())))?;
        Ok(Bytes::from(data))
    }

    async fn get_range(&self, path: &str, offset: usize, length: usize) -> Result<Bytes> {
        let data = self.get(path).await?;
        if offset + length > data.len() {
            return Err(MeruError::ObjectStore(format!(
                "range [{offset}, {}) exceeds file size {}",
                offset + length,
                data.len()
            )));
        }
        Ok(data.slice(offset..offset + length))
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let full = self.full_path(path);
        match tokio::fs::remove_file(&full).await {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(MeruError::Io(e)),
        }
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        let full = self.full_path(path);
        Ok(full.exists())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let dir = self.full_path(prefix);
        let mut results = Vec::new();
        if !dir.exists() {
            return Ok(results);
        }
        let mut entries = tokio::fs::read_dir(&dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if let Ok(rel) = entry.path().strip_prefix(&self.root) {
                results.push(rel.to_string_lossy().to_string());
            }
        }
        Ok(results)
    }

    /// Issue #26: race-free create-only write via `O_CREAT | O_EXCL`.
    ///
    /// The default `MeruStore::put_if_absent` does an `exists` +
    /// `put` two-step that's a TOCTOU race — two writers can both
    /// observe "doesn't exist" and both write. On POSIX, the kernel
    /// gives us a single atomic system call (`open(O_CREAT|O_EXCL)`)
    /// that either creates the file or fails with `EEXIST`. Map
    /// `EEXIST` → `MeruError::AlreadyExists`, everything else →
    /// `MeruError::Io`.
    ///
    /// The fsync chain matches `put`: write the body to the newly-
    /// created file, fsync the file, fsync the parent directory.
    /// Without the parent fsync, the directory entry itself is not
    /// durable under ext4/btrfs and a crash can resurrect the path
    /// as "does not exist" — which would let a subsequent
    /// `put_if_absent` succeed AGAIN on the same path. That's a
    /// silent-commit-duplication hazard we refuse by fsync'ing the
    /// directory entry on the create path.
    async fn put_if_absent(&self, path: &str, data: Bytes) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;
        let full = self.full_path(path);
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        // Do the O_CREAT|O_EXCL on the blocking pool — tokio::fs
        // wraps std::fs which opens in blocking mode; using OpenOptions
        // keeps the atomicity contract explicit.
        let full_cloned = full.clone();
        let data_cloned = data.clone();
        let res = tokio::task::spawn_blocking(move || -> Result<()> {
            let mut f = match OpenOptions::new()
                .write(true)
                .create_new(true) // == O_CREAT | O_EXCL
                .open(&full_cloned)
            {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    return Err(MeruError::AlreadyExists(
                        full_cloned.to_string_lossy().into_owned(),
                    ));
                }
                Err(e) => return Err(MeruError::Io(e)),
            };
            f.write_all(&data_cloned).map_err(MeruError::Io)?;
            f.sync_all().map_err(MeruError::Io)?;
            Ok(())
        })
        .await
        .map_err(|e| MeruError::ObjectStore(format!("spawn_blocking join: {e}")))?;
        res?;
        // fsync the parent directory so the new directory entry is
        // durable. Errors are hard failures — without the dir fsync,
        // a crash can "roll back" the create, letting a subsequent
        // put_if_absent succeed again on the same path.
        if let Some(parent) = full.parent() {
            let dir = tokio::fs::File::open(parent).await.map_err(MeruError::Io)?;
            dir.sync_all().await.map_err(MeruError::Io)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn put_get_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalFileStore::new(tmp.path()).unwrap();

        store
            .put("data/test.dat", Bytes::from("hello"))
            .await
            .unwrap();
        let data = store.get("data/test.dat").await.unwrap();
        assert_eq!(data.as_ref(), b"hello");

        assert!(store.exists("data/test.dat").await.unwrap());
        store.delete("data/test.dat").await.unwrap();
        assert!(!store.exists("data/test.dat").await.unwrap());
    }

    #[tokio::test]
    async fn get_range() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalFileStore::new(tmp.path()).unwrap();
        store
            .put("range.dat", Bytes::from("abcdefghij"))
            .await
            .unwrap();
        let slice = store.get_range("range.dat", 3, 4).await.unwrap();
        assert_eq!(slice.as_ref(), b"defg");
    }

    #[tokio::test]
    async fn delete_nonexistent_is_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalFileStore::new(tmp.path()).unwrap();
        store.delete("does-not-exist").await.unwrap();
    }

    /// Issue #26: `put_if_absent` is atomic via O_CREAT|O_EXCL. A
    /// second call against the same path returns `AlreadyExists`
    /// and does NOT clobber the first call's bytes.
    #[tokio::test]
    async fn put_if_absent_rejects_second_writer() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalFileStore::new(tmp.path()).unwrap();

        // First writer wins.
        store
            .put_if_absent("ver/v1.bin", Bytes::from_static(b"first"))
            .await
            .unwrap();

        // Second writer loses with AlreadyExists — the bytes must
        // stay the first writer's.
        let err = store
            .put_if_absent("ver/v1.bin", Bytes::from_static(b"second"))
            .await
            .unwrap_err();
        match err {
            MeruError::AlreadyExists(p) => assert!(p.contains("ver/v1.bin"), "path: {p}"),
            other => panic!("expected AlreadyExists, got {other:?}"),
        }

        let got = store.get("ver/v1.bin").await.unwrap();
        assert_eq!(got.as_ref(), b"first", "losing writer must not overwrite");
    }

    /// Issue #47 regression: concurrent put() to the same path must
    /// not corrupt data. With the old deterministic `.tmp` suffix,
    /// two writers would race on the same temp file and the loser's
    /// data could end up under the final name. With unique tmp paths
    /// (PID + counter), each writer has its own temp file and the
    /// last rename wins atomically — the final content is always a
    /// complete payload from one writer, never a torn mix.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_put_no_corruption() {
        use std::sync::Arc;
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileStore::new(tmp.path()).unwrap());

        const N: usize = 16;
        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                store
                    .put("contested/file.bin", Bytes::from(format!("writer-{i:04}")))
                    .await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }

        // The file must contain exactly one complete writer payload —
        // not a torn mix, not empty, not truncated.
        let body = store.get("contested/file.bin").await.unwrap();
        let s = std::str::from_utf8(&body).expect("body must be valid UTF-8");
        assert!(
            s.starts_with("writer-") && s.len() == 11,
            "body must be a complete 'writer-NNNN' payload, got: {s:?}"
        );
    }

    /// Concurrent-contention variant: N workers race on the same
    /// path; exactly one wins, all others see AlreadyExists. Runs
    /// in a Tokio multi-thread runtime so the race is real, not
    /// task-scheduled-serial.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn put_if_absent_concurrent_contention_single_winner() {
        use std::sync::Arc;
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileStore::new(tmp.path()).unwrap());

        const N: usize = 16;
        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                store
                    .put_if_absent("race/commit.bin", Bytes::from(format!("w{i}")))
                    .await
            }));
        }

        let mut wins = 0;
        let mut loses = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(()) => wins += 1,
                Err(MeruError::AlreadyExists(_)) => loses += 1,
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        assert_eq!(wins, 1, "exactly one writer must win the race");
        assert_eq!(loses, N - 1, "all others must see AlreadyExists");

        // Body must match the winner's content. We don't know
        // *which* worker won, but the content must be a valid "w{i}"
        // — not a torn mix.
        let body = store.get("race/commit.bin").await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(
            s.starts_with('w') && s[1..].parse::<usize>().is_ok() && body.len() >= 2,
            "body must be a complete w{{i}} payload, got: {s:?}"
        );
    }
}
