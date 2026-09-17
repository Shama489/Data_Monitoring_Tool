# db.py
import os
import re

import pandas as pd
from sqlalchemy import create_engine, text

# DATABASE CONFIG 
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "monitoring_db")

# SQLAlchemy 2.x correct URL format
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ENGINE SETUP
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True
)

# TEST CONNECTION
def test_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


# SAFE QUERY EXECUTION
def get_data(query: str, params: dict | None = None) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return pd.DataFrame()

# GET TABLE NAMES
def get_tables() -> pd.DataFrame:
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """
    return get_data(query)


def _validate_table_name(table_name: str) -> str:
    if not isinstance(table_name, str):
        raise ValueError("table name must be a string")

    candidate = table_name.strip()
    if not candidate:
        raise ValueError("table name cannot be empty")

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
        raise ValueError(f"Invalid table name: {table_name!r}")

    return candidate


# GET SAMPLE DATA
def get_table_data(table_name: str, limit: int = 100) -> pd.DataFrame:
    safe_name = _validate_table_name(table_name)
    tables = get_tables()

    if tables.empty or "table_name" not in tables.columns:
        raise ValueError("Could not resolve valid table names from the database")

    if safe_name not in tables["table_name"].tolist():
        raise ValueError(f"Invalid table name: {safe_name}")

    query = text(f"SELECT * FROM {safe_name} LIMIT :limit")
    return get_data(query, {"limit": limit})