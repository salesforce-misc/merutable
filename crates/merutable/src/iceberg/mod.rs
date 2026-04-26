pub mod catalog;
pub mod deletion_vector;
pub mod manifest;
pub mod manifest_pb;
pub mod snapshot;
pub mod translate;
pub mod version;

pub use catalog::{IcebergCatalog, load_persisted_schema};
pub use deletion_vector::{DeletionVector, PuffinEncoded};
pub use manifest::{DvLocation, Manifest, ManifestEntry};
pub use snapshot::{IcebergDataFile, SnapshotTransaction};
pub use translate::{
    to_iceberg_data_file_v2, to_iceberg_data_file_v2_with_schema, to_iceberg_schema_v2,
    to_iceberg_v2_table_metadata, to_iceberg_v2_table_metadata_bytes,
};
pub use version::{DataFileMeta, Version, VersionSet};

// Re-export the common `Result` alias so this crate's modules can write
// `crate::types::Result<T>` instead of importing `merutable-types` in every file.
pub use crate::types::Result;
