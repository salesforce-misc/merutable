use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::types::{MeruError, Result, value::FieldValue};

/// Parquet-native column types. Schema is immutable after table creation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColumnType {
    Boolean,
    Int32,
    Int64,
    Float,
    Double,
    /// Variable-length binary or UTF-8 string.
    ByteArray,
    /// Fixed-length binary; `i32` is the byte length.
    FixedLenByteArray(i32),
}

/// One column in a `TableSchema`.
///
/// Issue #25: carries the evolution-ready fields (`field_id`,
/// `initial_default`, `write_default`) pre-0.1-preview. New fields
/// default to `None` via `#[serde(default)]` so existing serialized
/// schemas round-trip cleanly. `#[non_exhaustive]` is NOT set because
/// it blocks even `..Default::default()` from external crates;
/// the ~30 internal test sites rely on struct-literal construction
/// with `..Default::default()`. Future field additions before 1.0
/// remain a breaking change — callers who want forward-compat should
/// use [`ColumnDef::builder`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub col_type: ColumnType,
    /// PK columns must be non-nullable.
    pub nullable: bool,

    /// Apache Iceberg v2 `field-id` — immutable once assigned. Assigned
    /// by `TableSchema::validate` from `last_column_id + 1`.
    #[serde(default)]
    pub field_id: Option<u32>,

    /// Iceberg v2 `initial-default`: value to project for this column
    /// when reading a file written before this column existed.
    #[serde(default)]
    pub initial_default: Option<FieldValue>,

    /// Iceberg v2 `write-default`: default value at INSERT time.
    #[serde(default)]
    pub write_default: Option<FieldValue>,
}

impl Default for ColumnDef {
    fn default() -> Self {
        Self {
            name: String::new(),
            col_type: ColumnType::Int64,
            nullable: false,
            field_id: None,
            initial_default: None,
            write_default: None,
        }
    }
}

impl ColumnDef {
    /// Issue #25: construct via builder. Required from outside this
    /// crate because the struct is `#[non_exhaustive]`.
    pub fn builder(name: impl Into<String>, col_type: ColumnType) -> ColumnDefBuilder {
        ColumnDefBuilder {
            name: name.into(),
            col_type,
            nullable: false,
            field_id: None,
            initial_default: None,
            write_default: None,
        }
    }
}

pub struct ColumnDefBuilder {
    name: String,
    col_type: ColumnType,
    nullable: bool,
    field_id: Option<u32>,
    initial_default: Option<FieldValue>,
    write_default: Option<FieldValue>,
}

impl ColumnDefBuilder {
    pub fn nullable(mut self, v: bool) -> Self {
        self.nullable = v;
        self
    }
    pub fn field_id(mut self, id: u32) -> Self {
        self.field_id = Some(id);
        self
    }
    pub fn initial_default(mut self, v: FieldValue) -> Self {
        self.initial_default = Some(v);
        self
    }
    pub fn write_default(mut self, v: FieldValue) -> Self {
        self.write_default = Some(v);
        self
    }
    pub fn build(self) -> ColumnDef {
        ColumnDef {
            name: self.name,
            col_type: self.col_type,
            nullable: self.nullable,
            field_id: self.field_id,
            initial_default: self.initial_default,
            write_default: self.write_default,
        }
    }
}

/// The single logical table schema for a merutable instance.
///
/// Issue #25: carries `schema_id` and `last_column_id` pre-0.1-preview.
/// See [`ColumnDef`] for notes on `#[non_exhaustive]`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TableSchema {
    pub table_name: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Vec<usize>,

    /// Iceberg v2 `current-schema-id`.
    #[serde(default)]
    pub schema_id: u32,

    /// Iceberg v2 `last-column-id`. Monotonic; never reused.
    #[serde(default)]
    pub last_column_id: u32,
}

impl TableSchema {
    /// Issue #25: builder for external construction.
    pub fn builder(table_name: impl Into<String>) -> TableSchemaBuilder {
        TableSchemaBuilder {
            table_name: table_name.into(),
            columns: Vec::new(),
            primary_key: Vec::new(),
            schema_id: 0,
            last_column_id: 0,
        }
    }

    /// Validate schema invariants AND normalize auto-assignable fields.
    ///
    /// Issue #25: `&mut self` because validation auto-populates missing
    /// `field_id` values from `last_column_id + 1`.
    pub fn validate(&mut self) -> Result<()> {
        self.normalize_field_ids();
        self.validate_readonly()
    }

    /// Pure validation without mutation.
    pub fn validate_readonly(&self) -> Result<()> {
        if self.columns.is_empty() {
            return Err(MeruError::InvalidArgument(
                "schema must have at least one column".into(),
            ));
        }
        if self.primary_key.is_empty() {
            return Err(MeruError::InvalidArgument(
                "primary key must have at least one column".into(),
            ));
        }
        let mut seen = HashSet::new();
        for &idx in &self.primary_key {
            if idx >= self.columns.len() {
                return Err(MeruError::InvalidArgument(format!(
                    "primary_key index {idx} out of bounds (columns len={})",
                    self.columns.len()
                )));
            }
            if !seen.insert(idx) {
                return Err(MeruError::InvalidArgument(format!(
                    "duplicate primary_key column index {idx}"
                )));
            }
            if self.columns[idx].nullable {
                return Err(MeruError::InvalidArgument(format!(
                    "primary key column '{}' must be non-nullable",
                    self.columns[idx].name
                )));
            }
        }

        // Issue #25: field-id uniqueness + bound.
        let mut field_ids_seen: HashSet<u32> = HashSet::new();
        for col in &self.columns {
            if let Some(id) = col.field_id {
                if !field_ids_seen.insert(id) {
                    return Err(MeruError::InvalidArgument(format!(
                        "duplicate field_id {id} on column '{}'",
                        col.name
                    )));
                }
                if id > self.last_column_id {
                    return Err(MeruError::InvalidArgument(format!(
                        "field_id {id} on column '{}' exceeds last_column_id {}",
                        col.name, self.last_column_id
                    )));
                }
            }
        }
        Ok(())
    }

    /// Auto-assign `field_id` to any column that lacks one. Idempotent.
    pub fn normalize_field_ids(&mut self) {
        if self.columns.iter().all(|c| c.field_id.is_some()) {
            return;
        }
        let mut next_id = self.last_column_id.saturating_add(1);
        for col in &mut self.columns {
            if col.field_id.is_none() {
                col.field_id = Some(next_id);
                next_id = next_id.saturating_add(1);
            }
        }
        self.last_column_id = next_id.saturating_sub(1);
    }

    pub fn column_by_name(&self, name: &str) -> Option<(usize, &ColumnDef)> {
        self.columns
            .iter()
            .enumerate()
            .find(|(_, c)| c.name == name)
    }

    pub fn pk_len(&self) -> usize {
        self.primary_key.len()
    }
}

pub struct TableSchemaBuilder {
    table_name: String,
    columns: Vec<ColumnDef>,
    primary_key: Vec<usize>,
    schema_id: u32,
    last_column_id: u32,
}

impl TableSchemaBuilder {
    pub fn add_column(mut self, col: ColumnDef) -> Self {
        self.columns.push(col);
        self
    }
    pub fn primary_key(mut self, pk: Vec<usize>) -> Self {
        self.primary_key = pk;
        self
    }
    pub fn schema_id(mut self, id: u32) -> Self {
        self.schema_id = id;
        self
    }
    pub fn last_column_id(mut self, id: u32) -> Self {
        self.last_column_id = id;
        self
    }
    pub fn build(self) -> TableSchema {
        TableSchema {
            table_name: self.table_name,
            columns: self.columns,
            primary_key: self.primary_key,
            schema_id: self.schema_id,
            last_column_id: self.last_column_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema(pk_nullable: bool) -> TableSchema {
        TableSchema {
            table_name: "t".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: pk_nullable,
                    ..Default::default()
                },
                ColumnDef {
                    name: "val".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: true,
                    ..Default::default()
                },
            ],
            primary_key: vec![0],
            ..Default::default()
        }
    }

    #[test]
    fn valid_schema_passes() {
        make_schema(false).validate().unwrap();
    }

    #[test]
    fn nullable_pk_rejected() {
        assert!(make_schema(true).validate().is_err());
    }

    #[test]
    fn empty_pk_rejected() {
        let mut s = make_schema(false);
        s.primary_key.clear();
        assert!(s.validate().is_err());
    }

    #[test]
    fn out_of_bounds_pk_rejected() {
        let mut s = make_schema(false);
        s.primary_key = vec![99];
        assert!(s.validate().is_err());
    }

    #[test]
    fn duplicate_pk_col_rejected() {
        let mut s = make_schema(false);
        s.primary_key = vec![0, 0];
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_normalizes_field_ids() {
        let mut s = make_schema(false);
        assert!(s.columns.iter().all(|c| c.field_id.is_none()));
        s.validate().unwrap();
        assert_eq!(s.columns[0].field_id, Some(1));
        assert_eq!(s.columns[1].field_id, Some(2));
        assert_eq!(s.last_column_id, 2);
    }

    #[test]
    fn validate_is_idempotent() {
        let mut s = make_schema(false);
        s.validate().unwrap();
        let snapshot = s.clone();
        s.validate().unwrap();
        assert_eq!(s, snapshot);
    }

    #[test]
    fn builder_round_trips() {
        let schema = TableSchema::builder("events")
            .add_column(
                ColumnDef::builder("id", ColumnType::Int64)
                    .nullable(false)
                    .build(),
            )
            .add_column(
                ColumnDef::builder("payload", ColumnType::ByteArray)
                    .nullable(true)
                    .build(),
            )
            .primary_key(vec![0])
            .build();
        let mut s = schema;
        s.validate().unwrap();
        assert_eq!(s.columns.len(), 2);
        assert_eq!(s.columns[0].field_id, Some(1));
    }

    #[test]
    fn duplicate_field_ids_rejected() {
        let mut s = make_schema(false);
        s.columns[0].field_id = Some(1);
        s.columns[1].field_id = Some(1);
        s.last_column_id = 1;
        assert!(s.validate_readonly().is_err());
    }

    #[test]
    fn serde_legacy_json_defaults_new_fields() {
        let legacy = r#"{
            "table_name": "t",
            "columns": [
                {"name": "id", "col_type": "Int64", "nullable": false},
                {"name": "val", "col_type": "ByteArray", "nullable": true}
            ],
            "primary_key": [0]
        }"#;
        let s: TableSchema = serde_json::from_str(legacy).unwrap();
        assert_eq!(s.columns[0].field_id, None);
        assert_eq!(s.columns[1].field_id, None);
        assert_eq!(s.schema_id, 0);
        assert_eq!(s.last_column_id, 0);
    }
}
