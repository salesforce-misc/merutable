//! Opaque database handle wrapping MeruDB + an embedded tokio runtime.

use std::sync::Arc;

use merutable::{types::schema::TableSchema, MeruDB};

/// Opaque handle. C callers hold `*mut MeruHandle` (typedef'd as `MeruDB *` in the header).
pub struct MeruHandle {
    pub db: Arc<MeruDB>,
    pub rt: tokio::runtime::Runtime,
    pub schema: Arc<TableSchema>,
}

impl MeruHandle {
    pub fn new(db: MeruDB, schema: TableSchema) -> Result<Box<Self>, String> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("failed to create tokio runtime: {e}"))?;
        Ok(Box::new(Self {
            db: Arc::new(db),
            rt,
            schema: Arc::new(schema),
        }))
    }
}
