"""Load monitoring datasets from files, databases, and cloud storage."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text


class DataSourceError(RuntimeError):
    """Raised when a monitoring source cannot be loaded."""


def _read_file_bytes(content: bytes, file_type: str, source_name: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    extension = file_type.lower().lstrip(".") or Path(source_name).suffix.lower().lstrip(".")
    stream = io.BytesIO(content)
    try:
        if extension == "csv":
            return pd.read_csv(stream)
        if extension in {"xls", "xlsx"}:
            return pd.read_excel(stream, sheet_name=sheet_name or 0)
        if extension == "json":
            return pd.read_json(stream)
        if extension == "parquet":
            return pd.read_parquet(stream)
        if extension == "tsv":
            return pd.read_csv(stream, sep="\t")
        raise DataSourceError(f"Unsupported file type: {extension or 'unknown'}")
    except DataSourceError:
        raise
    except Exception as error:
        raise DataSourceError(f"Could not read {source_name}: {error}") from error


def load_local_file(path: str, file_type: str = "", sheet_name: str | int | None = None) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file():
        raise DataSourceError(f"File was not found: {path}")
    return _read_file_bytes(file_path.read_bytes(), file_type, file_path.name, sheet_name)


def load_sql_query(database_url: str, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    if not database_url.strip():
        raise DataSourceError("database_url is required")
    normalized_query = query.strip().rstrip(";").strip()
    if (
        not normalized_query
        or not normalized_query.lower().startswith("select ")
        or ";" in normalized_query
    ):
        raise DataSourceError("Only SELECT queries are allowed")
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(text(query), connection, params=params)
    except Exception as error:
        raise DataSourceError(f"Database query failed: {error}") from error
    finally:
        engine.dispose()


def load_mongodb(uri: str, database: str, collection: str, query: dict[str, Any] | None = None, limit: int = 10000) -> pd.DataFrame:
    try:
        from pymongo import MongoClient
    except ImportError as error:
        raise DataSourceError("MongoDB support requires pymongo") from error
    if not uri or not database or not collection:
        raise DataSourceError("uri, database, and collection are required for MongoDB")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        documents = list(client[database][collection].find(query or {}, limit=max(1, min(limit, 100000))))
        for document in documents:
            document.pop("_id", None)
        return pd.DataFrame(documents)
    except Exception as error:
        raise DataSourceError(f"MongoDB query failed: {error}") from error
    finally:
        client.close()


def load_s3(bucket: str, key: str, file_type: str = "", sheet_name: str | int | None = None) -> pd.DataFrame:
    try:
        import boto3
    except ImportError as error:
        raise DataSourceError("Amazon S3 support requires boto3") from error
    try:
        response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return _read_file_bytes(response["Body"].read(), file_type, key, sheet_name)
    except DataSourceError:
        raise
    except Exception as error:
        raise DataSourceError(f"S3 object could not be loaded: {error}") from error


def load_dropbox(path: str, file_type: str = "", sheet_name: str | int | None = None) -> pd.DataFrame:
    try:
        import dropbox
    except ImportError as error:
        raise DataSourceError("Dropbox support requires dropbox") from error
    token = os.getenv("DROPBOX_ACCESS_TOKEN")
    if not token:
        raise DataSourceError("DROPBOX_ACCESS_TOKEN is required")
    try:
        _, response = dropbox.Dropbox(token).files_download(path)
        return _read_file_bytes(response.content, file_type, path, sheet_name)
    except DataSourceError:
        raise
    except Exception as error:
        raise DataSourceError(f"Dropbox file could not be loaded: {error}") from error


def load_google_drive(file_id: str, file_type: str = "", sheet_name: str | int | None = None) -> pd.DataFrame:
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
    except ImportError as error:
        raise DataSourceError("Google Drive support requires google-api-python-client and google-auth") from error
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise DataSourceError("GOOGLE_APPLICATION_CREDENTIALS is required")
    try:
        credentials = Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        metadata = build("drive", "v3", credentials=credentials).files().get(fileId=file_id, fields="name,mimeType").execute()
        response = build("drive", "v3", credentials=credentials).files().get_media(fileId=file_id).execute()
        return _read_file_bytes(response, file_type, metadata.get("name", file_id), sheet_name)
    except DataSourceError:
        raise
    except Exception as error:
        raise DataSourceError(f"Google Drive file could not be loaded: {error}") from error


def load_source(source: dict[str, Any]) -> pd.DataFrame:
    source_type = str(source.get("type", "")).lower().strip()
    file_type = str(source.get("file_type", ""))
    sheet_name = source.get("sheet_name")
    if source_type in {"csv", "excel", "xlsx", "xls", "json", "parquet", "tsv"}:
        path = source.get("path") or source.get("location")
        if not path:
            raise DataSourceError("path is required for file sources")
        normalized_type = file_type or (Path(str(path)).suffix.lower().lstrip(".")) or source_type
        return load_local_file(str(path), file_type or normalized_type, sheet_name)
    if source_type in {"postgresql", "mysql", "sql"}:
        return load_sql_query(str(source.get("url", "")), str(source.get("query", "")), source.get("params"))
    if source_type == "mongodb":
        return load_mongodb(str(source.get("uri", "")), str(source.get("database", "")), str(source.get("collection", "")), source.get("query"), int(source.get("limit", 10000)))
    if source_type == "s3":
        return load_s3(str(source.get("bucket", "")), str(source.get("key", "")), file_type, sheet_name)
    if source_type == "dropbox":
        return load_dropbox(str(source.get("path", "")), file_type, sheet_name)
    if source_type in {"google_drive", "gdrive"}:
        return load_google_drive(str(source.get("file_id", "")), file_type, sheet_name)
    raise DataSourceError(f"Unsupported source type: {source_type or 'unknown'}")


def summarize_source(source: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    return {
        "source": {key: value for key, value in source.items() if key not in {"url", "uri"}},
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "quality": {
            "nulls": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
        },
    }
