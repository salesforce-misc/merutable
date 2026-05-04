//! Python bindings for merutable via PyO3.
//!
//! Exposes `MeruDB` as a Python class with put/get/delete/scan/flush/compact/stats.

// PyO3's #[pymethods] proc macro generates internal `Into<PyErr>` conversions
// that clippy flags as useless. These are not in our code.
#![allow(clippy::useless_conversion)]

mod convert;

use std::sync::Arc;

use ::merutable::types::schema::{ColumnDef, TableSchema};
use ::merutable::MeruDB as RustMeruDB;
use pyo3::{prelude::*, types::PyDict};

/// Python-visible MeruDB wrapper.
///
/// # Example
/// ```python
/// from merutable import MeruDB
///
/// db = MeruDB("/tmp/mydb", "events", [
///     ("id",      "int64",  False),
///     ("name",    "string", True),
///     ("score",   "double", True),
///     ("active",  "bool",   True),
/// ])
///
/// db.put({"id": 1, "name": "alice", "score": 0.95, "active": True})
/// row = db.get(1)
/// db.flush()
/// db.compact()
/// print(db.stats())
/// ```
#[pyclass(name = "MeruDB")]
struct PyMeruDB {
    inner: Arc<RustMeruDB>,
    schema: Arc<TableSchema>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyMeruDB {
    /// Create and open a MeruDB instance.
    ///
    /// Args:
    ///     path: Base directory for data and WAL.
    ///     table_name: Logical table name.
    ///     columns: List of (name, type, nullable) tuples.
    ///         Types: "bool", "int32", "int64", "float", "double", "bytes"/"string"
    ///     primary_key: List of PK column names. Defaults to [first column].
    ///     memtable_size_mb: Memtable flush threshold in MiB. Default 64.
    #[new]
    #[pyo3(signature = (path, table_name, columns, primary_key=None, memtable_size_mb=64, read_only=false))]
    fn new(
        path: &str,
        table_name: &str,
        columns: Vec<(String, String, bool)>,
        primary_key: Option<Vec<String>>,
        memtable_size_mb: usize,
        read_only: bool,
    ) -> PyResult<Self> {
        let col_defs: Vec<ColumnDef> = columns
            .iter()
            .map(|(name, type_str, nullable)| {
                Ok(ColumnDef {
                    name: name.clone(),
                    col_type: convert::parse_column_type(type_str)?,
                    nullable: *nullable,
                    ..Default::default()
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Resolve PK indices.
        let pk_indices: Vec<usize> = match primary_key {
            Some(pk_names) => pk_names
                .iter()
                .map(|name| {
                    col_defs
                        .iter()
                        .position(|c| &c.name == name)
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "primary_key column '{name}' not found in columns"
                            ))
                        })
                })
                .collect::<PyResult<Vec<_>>>()?,
            None => vec![0],
        };

        let mut schema = TableSchema {
            table_name: table_name.to_string(),
            columns: col_defs,
            primary_key: pk_indices,
            ..Default::default()
        };
        // Validate schema upfront — clear Python errors instead of cryptic engine panics.
        schema
            .validate()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid schema: {e}")))?;
        let schema = Arc::new(schema);

        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "failed to create tokio runtime: {e}"
                    ))
                })?,
        );

        let wal_dir = std::path::Path::new(path).join("wal");
        let options = ::merutable::OpenOptions::new((*schema).clone())
            .catalog_uri(path)
            .wal_dir(wal_dir)
            .memtable_size_mb(memtable_size_mb)
            .read_only(read_only);

        let inner = runtime
            .block_on(async { RustMeruDB::open(options).await })
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("failed to open MeruDB: {e}"))
            })?;

        Ok(Self {
            inner: Arc::new(inner),
            schema,
            runtime,
        })
    }

    /// Insert or update a row. Returns the sequence number.
    ///
    /// Args:
    ///     row: dict mapping column names to values.
    fn put(&self, py: Python<'_>, row: &Bound<'_, PyDict>) -> PyResult<u64> {
        let row = convert::dict_to_row(row, &self.schema)?;
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.runtime);
        let seq = py
            .allow_threads(move || rt.block_on(async { inner.put(row).await }))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        Ok(seq.0)
    }

    /// Batch insert/update. All rows share a single WAL sync — N× faster than
    /// individual put() calls. Returns the base sequence number.
    ///
    /// Args:
    ///     rows: list of dicts, each mapping column names to values.
    fn put_batch(&self, py: Python<'_>, rows: Vec<Bound<'_, PyDict>>) -> PyResult<u64> {
        let mut rust_rows = Vec::with_capacity(rows.len());
        for dict in &rows {
            rust_rows.push(convert::dict_to_row(dict, &self.schema)?);
        }
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.runtime);
        let seq = py
            .allow_threads(move || rt.block_on(async { inner.put_batch(rust_rows).await }))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        Ok(seq.0)
    }

    /// Point lookup by primary key value(s). Returns dict or None.
    ///
    /// Args:
    ///     *pk_values: PK column values in primary key order.
    #[pyo3(signature = (*pk_values))]
    fn get(
        &self,
        py: Python<'_>,
        pk_values: &Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<Option<PyObject>> {
        let fvs = convert::pk_args_to_field_values(pk_values, &self.schema)?;
        let inner = Arc::clone(&self.inner);
        // Release GIL during point lookup (may involve Parquet file I/O).
        let row = py
            .allow_threads(move || inner.get(&fvs))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        match row {
            None => Ok(None),
            Some(r) => {
                let dict = convert::row_to_dict(py, &r, &self.schema)?;
                Ok(Some(dict.unbind().into()))
            }
        }
    }

    /// Delete by primary key value(s). Returns the sequence number.
    #[pyo3(signature = (*pk_values))]
    fn delete(&self, py: Python<'_>, pk_values: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<u64> {
        let fvs = convert::pk_args_to_field_values(pk_values, &self.schema)?;
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.runtime);
        let seq = py
            .allow_threads(move || rt.block_on(async { inner.delete(fvs).await }))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        Ok(seq.0)
    }

    /// Range scan. Returns list of dicts in PK order.
    ///
    /// Args:
    ///     start: Optional start PK values (inclusive). List or tuple.
    ///     end: Optional end PK values (exclusive). List or tuple.
    #[pyo3(signature = (start=None, end=None))]
    fn scan(
        &self,
        py: Python<'_>,
        start: Option<Vec<PyObject>>,
        end: Option<Vec<PyObject>>,
    ) -> PyResult<Vec<PyObject>> {
        let start_fvs = match &start {
            Some(vals) => Some(self.py_list_to_pk(py, vals)?),
            None => None,
        };
        let end_fvs = match &end {
            Some(vals) => Some(self.py_list_to_pk(py, vals)?),
            None => None,
        };

        let inner = Arc::clone(&self.inner);
        // Release GIL during scan (may involve Parquet file I/O).
        let results = py
            .allow_threads(move || inner.scan(start_fvs.as_deref(), end_fvs.as_deref()))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        let mut out = Vec::with_capacity(results.len());
        for (_ikey, row) in &results {
            let dict = convert::row_to_dict(py, row, &self.schema)?;
            out.push(dict.unbind().into());
        }
        Ok(out)
    }

    /// Force flush memtable to Parquet.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.runtime);
        py.allow_threads(move || rt.block_on(async { inner.flush().await }))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Trigger manual compaction.
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.runtime);
        py.allow_threads(move || rt.block_on(async { inner.compact().await }))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Engine statistics as a nested dict.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.inner.stats();
        let dict = PyDict::new_bound(py);
        dict.set_item("snapshot_id", s.snapshot_id)?;
        dict.set_item("current_seq", s.current_seq)?;

        // Memtable.
        let mem = PyDict::new_bound(py);
        mem.set_item("active_size_bytes", s.memtable.active_size_bytes)?;
        mem.set_item("active_entry_count", s.memtable.active_entry_count)?;
        mem.set_item("flush_threshold", s.memtable.flush_threshold)?;
        mem.set_item("immutable_count", s.memtable.immutable_count)?;
        dict.set_item("memtable", &mem)?;

        // Levels.
        let levels_list = pyo3::types::PyList::empty_bound(py);
        for l in &s.levels {
            let ld = PyDict::new_bound(py);
            ld.set_item("level", l.level)?;
            ld.set_item("file_count", l.file_count)?;
            ld.set_item("total_bytes", l.total_bytes)?;
            ld.set_item("total_rows", l.total_rows)?;

            let files_list = pyo3::types::PyList::empty_bound(py);
            for f in &l.files {
                let fd = PyDict::new_bound(py);
                fd.set_item("path", &f.path)?;
                fd.set_item("file_size", f.file_size)?;
                fd.set_item("num_rows", f.num_rows)?;
                fd.set_item("seq_range", (f.seq_range.0, f.seq_range.1))?;
                fd.set_item("has_dv", f.has_dv)?;
                // Issue #89: surface DV coords as a nested dict so
                // Python callers can audit DV state without poking
                // the filesystem. None when the file has no DV.
                match &f.dv {
                    Some(dv) => {
                        let dv_dict = PyDict::new_bound(py);
                        dv_dict.set_item("path", &dv.path)?;
                        dv_dict.set_item("offset", dv.offset)?;
                        dv_dict.set_item("length", dv.length)?;
                        fd.set_item("dv", &dv_dict)?;
                    }
                    None => fd.set_item("dv", py.None())?,
                }
                files_list.append(&fd)?;
            }
            ld.set_item("files", &files_list)?;
            levels_list.append(&ld)?;
        }
        dict.set_item("levels", &levels_list)?;

        // Cache.
        let cache_dict = PyDict::new_bound(py);
        cache_dict.set_item("capacity", s.cache.capacity)?;
        cache_dict.set_item("size", s.cache.size)?;
        cache_dict.set_item("hit_count", s.cache.hit_count)?;
        cache_dict.set_item("miss_count", s.cache.miss_count)?;
        dict.set_item("cache", &cache_dict)?;

        Ok(dict.unbind().into())
    }

    /// Re-read Iceberg manifest from disk. For read-only replicas.
    fn refresh(&self, py: Python<'_>) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.runtime);
        py.allow_threads(move || rt.block_on(async { inner.refresh().await }))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Graceful shutdown: flush memtable, fsync, and seal the database.
    /// Writes are rejected after close(); reads still work until the
    /// object is garbage-collected.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.runtime);
        py.allow_threads(move || rt.block_on(async { inner.close().await }))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Catalog base directory path. Point DuckDB at `{catalog_path}/data/L1/*.parquet`.
    fn catalog_path(&self) -> String {
        self.inner.catalog_path()
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "MeruDB(table='{}', seq={}, files={}, levels={})",
            self.schema.table_name,
            s.current_seq,
            s.levels.iter().map(|l| l.file_count).sum::<usize>(),
            s.levels.len(),
        )
    }
}

impl PyMeruDB {
    /// Helper: convert a Python list of values to PK FieldValues.
    fn py_list_to_pk(
        &self,
        py: Python<'_>,
        vals: &[PyObject],
    ) -> PyResult<Vec<::merutable::types::value::FieldValue>> {
        if vals.len() != self.schema.primary_key.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} PK value(s), got {}",
                self.schema.primary_key.len(),
                vals.len()
            )));
        }
        let mut fvs = Vec::with_capacity(vals.len());
        for (i, &col_idx) in self.schema.primary_key.iter().enumerate() {
            let col = &self.schema.columns[col_idx];
            let obj = vals[i].bind(py);
            fvs.push(convert::py_to_field_value_pub(
                obj,
                &col.col_type,
                &col.name,
            )?);
        }
        Ok(fvs)
    }
}

/// Module definition.
#[pymodule]
fn merutable(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMeruDB>()?;
    Ok(())
}
