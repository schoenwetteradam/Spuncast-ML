# Third-Party Dependency Register

This register documents the pinned runtime dependencies used by `Spuncast-ML`
for ISO 9001:2015 traceability and rebuild repeatability.

## Runtime dependencies

| Package | Version | License | Purpose |
|---|---:|---|---|
| scikit-learn | 1.4.2 | BSD-3-Clause | Core model training and inference APIs |
| numpy | 1.26.4 | BSD-3-Clause | Numerical arrays and vectorized operations |
| scipy | 1.13.1 | BSD-3-Clause | Scientific computing dependencies used by sklearn |
| joblib | 1.4.2 | BSD-3-Clause | Model serialization (`.joblib`) and caching |
| pyarrow | 16.1.0 | Apache-2.0 | Parquet read/write for dataset snapshots |
| threadpoolctl | 3.5.0 | BSD-3-Clause | Native threadpool controls used by sklearn/scipy |
| pandas | 2.2.3 | BSD-3-Clause | Tabular data processing and ETL transformations |
| psycopg2-binary | 2.9.10 | LGPL with exceptions | PostgreSQL connectivity |
| python-dotenv | 1.0.1 | BSD-3-Clause | Environment variable configuration loading |

## Control points

- Package pins are maintained in both `pyproject.toml` and `requirements.txt`.
- Model metadata captures training-time toolchain versions (including
  `scikit-learn` and `joblib`) for compatibility checks during operations.
