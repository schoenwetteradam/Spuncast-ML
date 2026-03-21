from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.environ.get("PGDATABASE", "spuncast"),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD"),
    "host": os.environ.get("PGHOST", os.environ.get("PG_HOST", "localhost")),
    "port": int(os.environ.get("PG_PORT", 5432)),
}


def source_view() -> str:
    return os.environ.get("SPUNCAST_ML_SOURCE_VIEW", "v_ml_heat_dataset_v1")


@contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def fetch_dataframe(sql: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

