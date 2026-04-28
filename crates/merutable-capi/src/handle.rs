//! Opaque database handle.

use std::sync::Arc;

use merutable::{types::schema::TableSchema, MeruDB};

/// Opaque handle. C callers hold `*mut MeruHandle`.
///
/// The runtime is Arc-shared so multiple handles can be opened on a single
/// MeruRuntime (one thread pool serving all of them).
pub struct MeruHandle {
    pub db: Arc<MeruDB>,
    pub rt: Arc<tokio::runtime::Runtime>,
    pub schema: Arc<TableSchema>,
}

impl MeruHandle {
    pub fn new(
        db: MeruDB,
        schema: TableSchema,
        rt: Arc<tokio::runtime::Runtime>,
    ) -> Box<Self> {
        Box::new(Self {
            db: Arc::new(db),
            rt,
            schema: Arc::new(schema),
        })
    }
}
