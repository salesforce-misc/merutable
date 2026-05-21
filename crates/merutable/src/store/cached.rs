//! `CachedStore`: LRU local disk cache fronting a remote object store.
//!
//! Cache key = blake3 hash of the object path.
//! Used in production to front S3 with a local SSD cache for hot Parquet reads.

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

use crate::types::{MeruError, Result};
use async_trait::async_trait;
use bytes::Bytes;
use lru::LruCache;

use crate::store::traits::MeruStore;

/// LRU disk cache wrapping any `MeruStore` backend.
pub struct CachedStore<S: MeruStore> {
    inner: S,
    cache_dir: PathBuf,
    lru: Mutex<LruCache<String, PathBuf>>,
}

impl<S: MeruStore> CachedStore<S> {
    /// Create a cached store with a maximum number of cached entries.
    pub fn new(inner: S, cache_dir: impl AsRef<Path>, max_entries: usize) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            inner,
            cache_dir,
            lru: Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(max_entries)
                    .unwrap_or(std::num::NonZeroUsize::new(1024).unwrap()),
            )),
        })
    }

    fn cache_path(&self, path: &str) -> PathBuf {
        let hash = blake3::hash(path.as_bytes());
        self.cache_dir.join(hash.to_hex().as_str())
    }
}

#[async_trait]
impl<S: MeruStore> MeruStore for CachedStore<S> {
    async fn put(&self, path: &str, data: Bytes) -> Result<()> {
        // Write through: put to backend, then cache locally.
        self.inner.put(path, data.clone()).await?;
        let cache_file = self.cache_path(path);
        tokio::fs::write(&cache_file, &data)
            .await
            .map_err(MeruError::Io)?;
        self.lru.lock().unwrap().put(path.to_string(), cache_file);
        Ok(())
    }

    async fn get(&self, path: &str) -> Result<Bytes> {
        // Try local cache first; tolerate ENOENT from races with eviction.
        let cache_file = self.cache_path(path);
        match tokio::fs::read(&cache_file).await {
            Ok(data) => {
                self.lru.lock().unwrap().promote(path);
                return Ok(Bytes::from(data));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(MeruError::Io(e)),
        }

        // Cache miss: fetch from backend.
        let data = self.inner.get(path).await?;
        tokio::fs::write(&cache_file, &data)
            .await
            .map_err(MeruError::Io)?;

        // Track in LRU; evict oldest if at capacity.
        let mut lru = self.lru.lock().unwrap();
        if let Some((_, evicted_path)) = lru.push(path.to_string(), cache_file) {
            // Best-effort delete of evicted cache file.
            let _ = std::fs::remove_file(&evicted_path);
        }

        Ok(data)
    }

    async fn get_range(&self, path: &str, offset: usize, length: usize) -> Result<Bytes> {
        // For simplicity, fetch the full object into cache, then slice.
        let full = self.get(path).await?;
        if offset + length > full.len() {
            return Err(MeruError::ObjectStore(format!(
                "range [{offset}, {}) exceeds object size {}",
                offset + length,
                full.len()
            )));
        }
        Ok(full.slice(offset..offset + length))
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await?;
        let cache_file = self.cache_path(path);
        let _ = tokio::fs::remove_file(&cache_file).await;
        self.lru.lock().unwrap().pop(path);
        Ok(())
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix).await
    }
}
