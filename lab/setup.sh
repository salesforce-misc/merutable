#!/bin/bash
# Setup script for the merutable Jupyter notebook lab.
#
# Creates a Python venv, installs dependencies, builds the merutable
# Python bindings via maturin, and launches JupyterLab.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Require Python 3.11+ (3.9 is EOL, 3.10 EOL upcoming).
PYVER=$(python3 -c 'import sys; print(sys.version_info[:2])')
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)'; then
    echo "==> Python version: $PYVER"
else
    echo "ERROR: Python 3.11+ is required (found $PYVER)" >&2
    exit 1
fi

echo "==> Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install maturin jupyterlab graphviz matplotlib duckdb pyarrow numpy

# RocksDB is optional — skip gracefully if it fails.
echo "==> Installing python-rocksdb (optional)..."
pip install python-rocksdb 2>/dev/null || echo "    python-rocksdb not available (benchmark will use SQLite only)"

echo "==> Building merutable Python bindings..."
cd crates/merutable-python
maturin develop --release
cd "$REPO_ROOT"

echo "==> Launching JupyterLab..."
jupyter lab lab/lab_merutable.ipynb
