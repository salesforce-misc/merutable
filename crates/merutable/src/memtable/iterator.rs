//! `MemtableIterator`: ordered scan over a memtable with read-seq filtering.
//!
//! Semantics:
//! - Yields entries in PK ascending order.
//! - For duplicate user-keys (multiple versions), yields only the entry with
//!   the highest `seq ≤ read_seq` (i.e., the first one encountered in sort order,
//!   since our encoding puts higher seq first for the same PK).
//! - Skips entries with `seq > read_seq`.
//!
//! The iterator DOES yield `OpType::Delete` entries (tombstones) so
//! the merge layer and range-scan dedup can decide tombstone semantics.

use crate::types::sequence::SeqNum;
use bytes::Bytes;

use crate::memtable::skiplist::{EntryValue, decode_seq_from_key, user_key_of};

/// A snapshot of one memtable entry, ready for the merge layer.
#[derive(Clone, Debug)]
pub struct MemEntry {
    /// The user-key (PK-encoded) bytes, without tag.
    pub user_key: Bytes,
    pub seq: SeqNum,
    pub entry: EntryValue,
}

pub struct MemtableIterator<'a> {
    inner: Box<dyn Iterator<Item = (Bytes, EntryValue)> + 'a>,
    read_seq: SeqNum,
    last_user_key: Option<Bytes>,
}

impl<'a> MemtableIterator<'a> {
    pub fn new(inner: impl Iterator<Item = (Bytes, EntryValue)> + 'a, read_seq: SeqNum) -> Self {
        Self {
            inner: Box::new(inner),
            read_seq,
            last_user_key: None,
        }
    }
}

impl<'a> Iterator for MemtableIterator<'a> {
    type Item = MemEntry;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (ikey_bytes, entry) = self.inner.next()?;
            let seq = decode_seq_from_key(&ikey_bytes);

            // Skip entries newer than the read snapshot.
            if seq > self.read_seq {
                continue;
            }

            let uk = Bytes::copy_from_slice(user_key_of(&ikey_bytes));

            // Skip duplicate user-keys (older versions). Since the skip list sorts
            // newer seq first for the same PK, the first entry we see for a given
            // user_key is always the most recent. Subsequent entries for the same
            // user_key are older versions — skip them.
            if let Some(ref last) = self.last_user_key
                && *last == uk
            {
                continue;
            }

            self.last_user_key = Some(uk.clone());
            return Some(MemEntry {
                user_key: uk,
                seq,
                entry,
            });
        }
    }
}
