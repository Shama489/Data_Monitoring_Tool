# db.py
from sqlalchemy import create_engine, text
import pandas as pd
import os

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

# GET SAMPLE DATA 
def get_table_data(table_name: str, limit: int = 100) -> pd.DataFrame:
    # Validate table name (very important!)
    tables = get_tables()["table_name"].tolist()
    if table_name not in tables:
        print("❌ Invalid table name")
        return pd.DataFrame()

    query = text(f"SELECT * FROM {table_name} LIMIT :limit")
    return get_data(query, {"limit": limit})