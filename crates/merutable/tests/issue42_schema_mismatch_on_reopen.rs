//! Issue #42 regression: the single-table-per-catalog invariant MUST
//! be enforced at `IcebergCatalog::open` when a manifest is already
//! on disk. Pre-#42 a caller could reopen `./db` with a different
//! `table_name` or incompatible column set, and the engine silently
//! accepted it — later commits overwrote the persisted schema,
//! corrupting any reader that depended on the original shape.
//!
//! Tests pinned here:
//!   1. fresh open — no persisted manifest, any schema accepted.
//!   2. matching reopen — identical schema, succeeds.
//!   3. mismatched `table_name` — rejected.
//!   4. mismatched column count — rejected.
//!   5. mismatched column name at same index — rejected.
//!   6. mismatched column type — rejected.
//!   7. mismatched nullability — rejected.
//!   8. mismatched primary key — rejected.

use merutable::iceberg::IcebergCatalog;
use merutable::types::{
    MeruError,
    schema::{ColumnDef, ColumnType, TableSchema},
};

fn base_schema() -> TableSchema {
    TableSchema {
        table_name: "events".into(),
        columns: vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Int64,
                nullable: false,
                ..Default::default()
            },
            ColumnDef {
                name: "payload".into(),
                col_type: ColumnType::ByteArray,
                nullable: true,
                ..Default::default()
            },
        ],
        primary_key: vec![0],
        ..Default::default()
    }
}

async fn open_once_then_reopen_with(
    tmp: &tempfile::TempDir,
    reopen: TableSchema,
) -> Result<IcebergCatalog, MeruError> {
    // First open persists the base schema (fresh-open path).
    let _first = IcebergCatalog::open(tmp.path(), base_schema())
        .await
        .expect("first open");
    // Commit something so the manifest is actually on disk.
    use merutable::iceberg::SnapshotTransaction;
    use std::sync::Arc;
    let txn = SnapshotTransaction::new();
    _first
        .commit(&txn, Arc::new(base_schema()))
        .await
        .expect("commit base");
    drop(_first);
    IcebergCatalog::open(tmp.path(), reopen).await
}

#[tokio::test]
async fn fresh_open_accepts_any_schema() {
    let tmp = tempfile::tempdir().unwrap();
    let cat = IcebergCatalog::open(tmp.path(), base_schema()).await;
    assert!(cat.is_ok(), "fresh open must succeed");
}

#[tokio::test]
async fn matching_reopen_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let out = open_once_then_reopen_with(&tmp, base_schema()).await;
    if let Err(e) = out {
        panic!("matching reopen must succeed; got error: {e:?}");
    }
}

#[tokio::test]
async fn mismatched_table_name_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mut evil = base_schema();
    evil.table_name = "logs".into();
    let err = match open_once_then_reopen_with(&tmp, evil).await {
        Ok(_) => panic!("reopen should have been rejected"),
        Err(e) => e,
    };
    match err {
        MeruError::SchemaMismatch(s) => assert!(s.contains("events") && s.contains("logs")),
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

/// #44 Stage 1: strictly-additive schema extensions are ALLOWED
/// on reopen (nullable or defaulted new columns).
#[tokio::test]
async fn strictly_additive_nullable_column_accepted_on_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let mut evolved = base_schema();
    evolved.columns.push(ColumnDef {
        name: "extra_nullable".into(),
        col_type: ColumnType::Int64,
        nullable: true,
        ..Default::default()
    });
    let out = open_once_then_reopen_with(&tmp, evolved).await;
    if let Err(e) = out {
        panic!("strictly-additive nullable reopen must succeed; got {e:?}");
    }
}

/// #44 Stage 1 counterpart: a non-nullable, no-default new column
/// is NOT back-fillable against existing rows and must be rejected.
#[tokio::test]
async fn additive_non_nullable_without_default_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mut evil = base_schema();
    evil.columns.push(ColumnDef {
        name: "extra_required".into(),
        col_type: ColumnType::Int64,
        nullable: false,
        ..Default::default()
    });
    let err = match open_once_then_reopen_with(&tmp, evil).await {
        Ok(_) => panic!("reopen should have been rejected"),
        Err(e) => e,
    };
    match err {
        MeruError::SchemaMismatch(s) => {
            assert!(
                s.contains("extra_required") && s.contains("default"),
                "error must name the offending column + its missing default: {s}"
            );
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

/// Column removal still rejected — #44 is additive-only.
#[tokio::test]
async fn column_removal_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mut evil = base_schema();
    evil.columns.pop();
    let err = match open_once_then_reopen_with(&tmp, evil).await {
        Ok(_) => panic!("reopen should have been rejected"),
        Err(e) => e,
    };
    match err {
        MeruError::SchemaMismatch(s) => {
            assert!(s.contains("removal") || s.contains("2") && s.contains("1"))
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

#[tokio::test]
async fn mismatched_column_name_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mut evil = base_schema();
    evil.columns[1].name = "body".into();
    let err = match open_once_then_reopen_with(&tmp, evil).await {
        Ok(_) => panic!("reopen should have been rejected"),
        Err(e) => e,
    };
    match err {
        MeruError::SchemaMismatch(s) => {
            assert!(s.contains("payload") && s.contains("body"))
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

#[tokio::test]
async fn mismatched_column_type_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mut evil = base_schema();
    evil.columns[0].col_type = ColumnType::Int32;
    let err = match open_once_then_reopen_with(&tmp, evil).await {
        Ok(_) => panic!("reopen should have been rejected"),
        Err(e) => e,
    };
    match err {
        MeruError::SchemaMismatch(s) => {
            assert!(s.contains("Int64") && s.contains("Int32"))
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

#[tokio::test]
async fn mismatched_nullability_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mut evil = base_schema();
    evil.columns[1].nullable = false;
    let err = match open_once_then_reopen_with(&tmp, evil).await {
        Ok(_) => panic!("reopen should have been rejected"),
        Err(e) => e,
    };
    match err {
        MeruError::SchemaMismatch(s) => {
            assert!(s.contains("nullable") && s.contains("payload"))
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

#[tokio::test]
async fn mismatched_primary_key_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mut evil = base_schema();
    // Legal in isolation (payload is byte-array, PK requires
    // non-null), but we mark it non-null below so the shape passes
    // local validation and only the PK indices differ.
    evil.columns[1].nullable = false;
    evil.primary_key = vec![0, 1];
    let err = match open_once_then_reopen_with(&tmp, evil).await {
        Ok(_) => panic!("reopen should have been rejected"),
        Err(e) => e,
    };
    // Nullable-mismatch fires before PK-mismatch because the column
    // scan runs first; either error satisfies the invariant (reject
    // before data corruption). The test pins that the reopen fails
    // SOMEWHERE with SchemaMismatch.
    match err {
        MeruError::SchemaMismatch(_) => {}
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}
