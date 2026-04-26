//! Issue #29 Phase 2e: DataFusion `TableProvider` for the change
//! feed.
//!
//! Wraps a running `MeruEngine` + a `since_seq` watermark behind
//! the standard `datafusion::catalog::TableProvider` trait so
//! analytical consumers run the 0.1-preview headline query:
//!
//! ```sql
//! SELECT * FROM merutable_changes WHERE op = 'DELETE'
//! ```
//!
//! # Scope
//!
//! - **One-shot scan**: `scan()` drains the cursor once and
//!   materializes every record into a single RecordBatch wrapped
//!   in `MemoryExec`. Works well for the blocker's bounded-
//!   watermark query pattern.
//! - **No filter pushdown** in v1 — DataFusion applies
//!   `op = 'DELETE'` / `seq > N` after the scan. Push-down on
//!   `seq > since_seq` is a follow-on (the provider holds the
//!   watermark, so pushing the filter down would just bump
//!   `since_seq` before draining).

use std::any::Any;
use std::sync::Arc;

use crate::engine::engine::MeruEngine;
use crate::types::schema::TableSchema;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::TableProvider;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{
    BinaryExpr, Expr, Operator, TableProviderFilterPushDown, TableType,
};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::memory::MemoryExec;

use crate::sql::ChangeFeedCursor;
use crate::sql::arrow::{change_feed_schema, records_to_record_batch};

/// A DataFusion-shaped view of the change feed. Register with a
/// `SessionContext::register_table("merutable_changes", ..)` and
/// consumers can query the feed with plain SQL.
pub struct ChangeFeedTableProvider {
    engine: Arc<MeruEngine>,
    table_schema: TableSchema,
    since_seq: u64,
    arrow_schema: SchemaRef,
}

impl std::fmt::Debug for ChangeFeedTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChangeFeedTableProvider")
            .field("table", &self.table_schema.table_name)
            .field("since_seq", &self.since_seq)
            .field("arrow_schema", &self.arrow_schema)
            .finish()
    }
}

impl ChangeFeedTableProvider {
    pub fn new(engine: Arc<MeruEngine>, table_schema: TableSchema, since_seq: u64) -> Self {
        let arrow_schema = change_feed_schema(&table_schema);
        Self {
            engine,
            table_schema,
            since_seq,
            arrow_schema,
        }
    }
}

#[async_trait]
impl TableProvider for ChangeFeedTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.arrow_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Issue #29 Phase 2f: seq filter pushdown.
    /// A `WHERE seq > N` (or `>= N+1`) predicate can be pushed
    /// into the provider's own `since_seq` watermark — we simply
    /// bump `since_seq` to `max(since_seq, effective_bound)`
    /// before draining the cursor. Since the cursor already
    /// filters at the engine level, this is **exact** pushdown
    /// (DataFusion doesn't need to re-apply the predicate).
    /// Filters we can't push are `Inexact` so DataFusion still
    /// applies them post-scan.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|f| {
                if extract_seq_lower_bound(f).is_some() {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Inexact
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Phase 2f: derive the effective `since_seq` from any
        // pushed-down seq > N / seq >= N predicates. The watermark
        // moves FORWARD only — a filter can't ever lower the
        // provider's own baseline since_seq (set at construction).
        let mut effective_since = self.since_seq;
        for f in filters {
            if let Some(lb) = extract_seq_lower_bound(f) {
                if lb > effective_since {
                    effective_since = lb;
                }
            }
        }

        let mut cursor = ChangeFeedCursor::from_engine(self.engine.clone(), effective_since);
        let records = cursor
            .next_batch(usize::MAX)
            .map_err(|e| DataFusionError::Execution(format!("change-feed drain: {e}")))?;
        let batch = records_to_record_batch(&records, &self.table_schema)
            .map_err(|e| DataFusionError::Execution(format!("RecordBatch assembly: {e}")))?;
        let partitions = vec![vec![batch]];
        let exec =
            MemoryExec::try_new(&partitions, self.arrow_schema.clone(), projection.cloned())?;
        Ok(Arc::new(exec))
    }
}

/// Phase 2f helper. If `expr` is a predicate of the form
/// `seq > N` / `seq >= N` (or the mirrored `N < seq` / `N <= seq`)
/// where N is a non-negative integer literal, return the effective
/// `since_seq` lower bound (exclusive — i.e. the greatest seq that
/// should be EXCLUDED from the feed). Otherwise return None.
///
/// `seq > N` and `N < seq` → exclude N, include N+1+ → since = N.
/// `seq >= N` and `N <= seq` → exclude N-1 and below → since = N-1.
/// Anything else (AND/OR trees, non-literal comparisons, other
/// columns) → None, DataFusion applies post-scan.
fn extract_seq_lower_bound(expr: &Expr) -> Option<u64> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = expr else {
        return None;
    };
    let (col_side, lit_side, op) = match (left.as_ref(), right.as_ref(), op) {
        (Expr::Column(c), Expr::Literal(v), op) => (c, v, *op),
        (Expr::Literal(v), Expr::Column(c), Operator::Lt) => (c, v, Operator::Gt),
        (Expr::Literal(v), Expr::Column(c), Operator::LtEq) => (c, v, Operator::GtEq),
        _ => return None,
    };
    if col_side.name != "seq" {
        return None;
    }
    let n: i128 = match lit_side {
        datafusion::scalar::ScalarValue::UInt64(Some(v)) => *v as i128,
        datafusion::scalar::ScalarValue::UInt32(Some(v)) => *v as i128,
        datafusion::scalar::ScalarValue::Int64(Some(v)) => *v as i128,
        datafusion::scalar::ScalarValue::Int32(Some(v)) => *v as i128,
        _ => return None,
    };
    if n < 0 {
        return None;
    }
    let n_u64 = n as u64;
    match op {
        // `seq > N` → exclude ≤ N → since = N.
        Operator::Gt => Some(n_u64),
        // `seq >= N` → exclude ≤ N-1 → since = N-1. Guard against
        // `seq >= 0` collapsing to u64 underflow.
        Operator::GtEq => {
            if n_u64 == 0 {
                Some(0)
            } else {
                Some(n_u64 - 1)
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::EngineConfig;
    use crate::types::{
        schema::{ColumnDef, ColumnType, TableSchema},
        value::{FieldValue, Row},
    };
    use datafusion::execution::context::SessionContext;

    fn test_schema() -> TableSchema {
        TableSchema {
            table_name: "df-cf-test".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                    ..Default::default()
                },
                ColumnDef {
                    name: "v".into(),
                    col_type: ColumnType::Int64,
                    nullable: true,
                    ..Default::default()
                },
            ],
            primary_key: vec![0],
            ..Default::default()
        }
    }

    async fn open_engine(tmp: &tempfile::TempDir) -> Arc<MeruEngine> {
        let cfg = EngineConfig {
            schema: test_schema(),
            catalog_uri: tmp.path().to_string_lossy().to_string(),
            object_store_prefix: tmp.path().to_string_lossy().to_string(),
            wal_dir: tmp.path().join("wal"),
            ..Default::default()
        };
        MeruEngine::open(cfg).await.unwrap()
    }

    fn row(id: i64, v: i64) -> Row {
        Row::new(vec![
            Some(FieldValue::Int64(id)),
            Some(FieldValue::Int64(v)),
        ])
    }

    #[tokio::test]
    async fn select_star_returns_all_ops() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = open_engine(&tmp).await;
        for i in 1..=3i64 {
            engine
                .put(vec![FieldValue::Int64(i)], row(i, i * 10))
                .await
                .unwrap();
        }
        let ctx = SessionContext::new();
        let provider = Arc::new(ChangeFeedTableProvider::new(engine, test_schema(), 0));
        ctx.register_table("merutable_changes", provider).unwrap();
        let df = ctx
            .sql("SELECT seq, op FROM merutable_changes ORDER BY seq")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
    }

    #[tokio::test]
    async fn where_clause_on_op_filters_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = open_engine(&tmp).await;
        engine
            .put(vec![FieldValue::Int64(1)], row(1, 10))
            .await
            .unwrap();
        engine
            .put(vec![FieldValue::Int64(2)], row(2, 20))
            .await
            .unwrap();
        engine.delete(vec![FieldValue::Int64(1)]).await.unwrap();
        let ctx = SessionContext::new();
        let provider = Arc::new(ChangeFeedTableProvider::new(engine, test_schema(), 0));
        ctx.register_table("merutable_changes", provider).unwrap();
        let df = ctx
            .sql("SELECT seq FROM merutable_changes WHERE op = 'DELETE'")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
    }

    #[tokio::test]
    async fn seq_filter_pushdown_bumps_watermark() {
        // Phase 2f: a WHERE seq > N predicate pushes into the
        // provider's own `since_seq`, so the engine scan only
        // materializes the relevant rows. The outcome is
        // identical to no-pushdown but cheaper; here we verify
        // the SQL-level correctness.
        let tmp = tempfile::tempdir().unwrap();
        let engine = open_engine(&tmp).await;
        for i in 1..=5i64 {
            engine
                .put(vec![FieldValue::Int64(i)], row(i, i))
                .await
                .unwrap();
        }
        let ctx = SessionContext::new();
        let provider = Arc::new(ChangeFeedTableProvider::new(engine, test_schema(), 0));
        ctx.register_table("merutable_changes", provider).unwrap();

        // seq > 2 — rows with seq in (2, read_seq] surface (3, 4, 5).
        let df = ctx
            .sql("SELECT seq FROM merutable_changes WHERE seq > 2 ORDER BY seq")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3, "only 3 ops above seq=2");

        // seq >= 4 — rows with seq >= 4 surface (4, 5).
        let df = ctx
            .sql("SELECT seq FROM merutable_changes WHERE seq >= 4 ORDER BY seq")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }

    #[tokio::test]
    async fn since_seq_watermark_hides_earlier_ops() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = open_engine(&tmp).await;
        engine
            .put(vec![FieldValue::Int64(1)], row(1, 10))
            .await
            .unwrap();
        let boundary = engine.read_seq().0;
        engine
            .put(vec![FieldValue::Int64(2)], row(2, 20))
            .await
            .unwrap();
        engine
            .put(vec![FieldValue::Int64(3)], row(3, 30))
            .await
            .unwrap();
        let ctx = SessionContext::new();
        let provider = Arc::new(ChangeFeedTableProvider::new(
            engine,
            test_schema(),
            boundary,
        ));
        ctx.register_table("merutable_changes", provider).unwrap();
        let df = ctx.sql("SELECT seq FROM merutable_changes").await.unwrap();
        let batches = df.collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }
}
