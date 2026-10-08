"""Load monitoring datasets from files, databases, and cloud storage."""

from __future__ import annotations

import io
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text


class DataSourceError(RuntimeError):
    """Raised when a monitoring source cannot be loaded."""


_LOCAL_FILE_EXTENSIONS = {"csv", "xls", "xlsx", "json", "parquet", "tsv", "txt"}
_EXTENSION_ALIASES = {"excel": {"xls", "xlsx"}}


def _configured_local_roots() -> list[Path]:
    configured = os.getenv("DATASET_ALLOWED_DIRS", "")
    if not configured.strip():
        raise DataSourceError(
            "Local file sources are disabled; configure DATASET_ALLOWED_DIRS"
        )

    roots = []
    for raw_root in configured.split(os.pathsep):
        if not raw_root.strip():
            continue
        try:
            root = Path(raw_root.strip()).expanduser().resolve(strict=True)
        except OSError as error:
            raise DataSourceError(
                "A configured DATASET_ALLOWED_DIRS entry does not exist"
            ) from error
        if not root.is_dir():
            raise DataSourceError("DATASET_ALLOWED_DIRS entries must be directories")
        roots.append(root)
    if not roots:
        raise DataSourceError(
            "Local file sources are disabled; configure DATASET_ALLOWED_DIRS"
        )
    return roots


def _validate_local_file_path(
    path: str | os.PathLike[str], file_type: str = ""
) -> tuple[Path, str]:
    raw_path = os.fspath(path)
    if any(part == ".." for part in raw_path.replace("\\", "/").split("/")):
        raise DataSourceError("Local file paths must not contain traversal components")

    candidate = Path(raw_path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise DataSourceError("Local source file was not found or could not be resolved") from error
    if not resolved.is_file():
        raise DataSourceError("Local source path must refer to a file")

    roots = _configured_local_roots()
    if not any(resolved.is_relative_to(root) for root in roots):
        raise DataSourceError("Local source file is outside DATASET_ALLOWED_DIRS")

    actual_extension = resolved.suffix.lower().lstrip(".")
    requested_extension = file_type.lower().lstrip(".")
    if actual_extension:
        if actual_extension not in _LOCAL_FILE_EXTENSIONS:
            raise DataSourceError(f"Unsupported file type: {actual_extension}")
        if requested_extension:
            accepted_extensions = _EXTENSION_ALIASES.get(
                requested_extension, {requested_extension}
            )
            if actual_extension not in accepted_extensions:
                raise DataSourceError(
                    "Declared file type does not match the local file extension"
                )
        extension = actual_extension
    else:
        extension = requested_extension
        if extension not in _LOCAL_FILE_EXTENSIONS | set(_EXTENSION_ALIASES):
            raise DataSourceError(f"Unsupported file type: {extension or 'unknown'}")

    return resolved, extension


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
        if extension == "txt":
            return pd.read_csv(stream, sep=None, engine="python")
        raise DataSourceError(f"Unsupported file type: {extension or 'unknown'}")
    except DataSourceError:
        raise
    except Exception as error:
        raise DataSourceError(f"Could not read {source_name}: {error}") from error


def load_local_file(path: str, file_type: str = "", sheet_name: str | int | None = None) -> pd.DataFrame:
    file_path, extension = _validate_local_file_path(path, file_type)
    return _read_file_bytes(file_path.read_bytes(), extension, file_path.name, sheet_name)


def _strip_sql_leading_comments(query: str) -> str:
    cleaned = query.lstrip()
    while True:
        if cleaned.startswith("--"):
            newline_index = cleaned.find("\n")
            if newline_index == -1:
                return ""
            cleaned = cleaned[newline_index + 1 :].lstrip()
            continue
        if cleaned.startswith("/*"):
            comment_end = cleaned.find("*/")
            if comment_end == -1:
                return ""
            cleaned = cleaned[comment_end + 2 :].lstrip()
            continue
        return cleaned


def _find_sql_statement_terminator(query: str) -> int | None:
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    index = 0
    while index < len(query):
        character = query[index]
        next_character = query[index + 1] if index + 1 < len(query) else ""

        if in_line_comment:
            if character == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            if character == "*" and next_character == "/":
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if in_single_quote:
            if character == "'" and next_character == "'":
                index += 2
                continue
            if character == "'":
                in_single_quote = False
            index += 1
            continue

        if in_double_quote:
            if character == '"' and next_character == '"':
                index += 2
                continue
            if character == '"':
                in_double_quote = False
            index += 1
            continue

        if character == "-" and next_character == "-":
            in_line_comment = True
            index += 2
            continue
        if character == "/" and next_character == "*":
            in_block_comment = True
            index += 2
            continue
        if character == "'":
            in_single_quote = True
            index += 1
            continue
        if character == '"':
            in_double_quote = True
            index += 1
            continue
        if character == ";":
            return index
        index += 1
    return None


def load_sql_query(database_url: str, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    if not database_url.strip():
        raise DataSourceError("database_url is required")

    normalized_query = _strip_sql_leading_comments(query).strip()
    lowered_query = normalized_query.lower()
    if not normalized_query:
        raise DataSourceError("Only SELECT queries are allowed")
    if not (lowered_query.startswith("select") or lowered_query.startswith("with")):
        raise DataSourceError("Only SELECT queries are allowed")

    statement_terminator = _find_sql_statement_terminator(normalized_query)
    if statement_terminator is not None:
        suffix = normalized_query[statement_terminator + 1 :].lstrip()
        if suffix:
            raise DataSourceError("Only SELECT queries are allowed")
        normalized_query = normalized_query[:statement_terminator].rstrip()
        if not normalized_query:
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
    if source_type in {"csv", "excel", "xlsx", "xls", "json", "parquet", "tsv", "txt"}:
        path = source.get("path") or source.get("location")
        if not path:
            raise DataSourceError("path is required for file sources")
        declared_extensions = _EXTENSION_ALIASES.get(source_type, {source_type})
        path_extension = Path(str(path)).suffix.lower().lstrip(".")
        if path_extension and path_extension not in declared_extensions:
            raise DataSourceError(
                "Source type does not match the local file extension"
            )
        normalized_type = file_type or (Path(str(path)).suffix.lower().lstrip(".")) or source_type
        if file_type:
            requested_extensions = _EXTENSION_ALIASES.get(
                file_type.lower().lstrip("."), {file_type.lower().lstrip(".")}
            )
            if not requested_extensions.issubset(declared_extensions):
                raise DataSourceError(
                    "Declared file type does not match the source type"
                )
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


def source_capabilities() -> dict[str, Any]:
    """Report supported sources, optional SDK availability, and setup requirements."""
    providers = [
        {
            "type": "local_files",
            "formats": ["csv", "excel", "json", "parquet", "tsv", "txt"],
            "dependencies": {"pandas": "pandas", "excel": "openpyxl", "parquet": "pyarrow"},
            "required_fields": ["path"],
            "credential_setup": "No credentials required; file must be readable by the API process.",
        },
        {
            "type": "sql",
            "aliases": ["postgresql", "mysql"],
            "dependencies": {"sqlalchemy": "sqlalchemy", "postgresql": "psycopg2", "mysql": "pymysql"},
            "required_fields": ["url", "query"],
            "credential_setup": "Supply a database URL and a read-only SELECT query.",
        },
        {
            "type": "mongodb",
            "dependencies": {"pymongo": "pymongo"},
            "required_fields": ["uri", "database", "collection"],
            "credential_setup": "Supply a MongoDB URI with read access to the target collection.",
        },
        {
            "type": "s3",
            "dependencies": {"boto3": "boto3"},
            "required_fields": ["bucket", "key"],
            "credential_setup": "Configure the AWS SDK credential chain (environment variables, profile, or execution role).",
        },
        {
            "type": "dropbox",
            "dependencies": {"dropbox": "dropbox"},
            "required_fields": ["path"],
            "credential_env": ["DROPBOX_ACCESS_TOKEN"],
        },
        {
            "type": "google_drive",
            "dependencies": {"google-api-python-client": "googleapiclient", "google-auth": "google.auth"},
            "required_fields": ["file_id"],
            "credential_env": ["GOOGLE_APPLICATION_CREDENTIALS"],
        },
    ]

    for provider in providers:
        missing_dependencies = []
        for package, module in provider["dependencies"].items():
            try:
                installed = importlib.util.find_spec(module) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                installed = False
            if not installed:
                missing_dependencies.append(package)

        missing_configuration = [
            variable for variable in provider.get("credential_env", []) if not os.getenv(variable)
        ]
        provider["missing_dependencies"] = missing_dependencies
        provider["missing_configuration"] = missing_configuration
        if missing_dependencies:
            provider["status"] = "missing_dependency"
        elif missing_configuration:
            provider["status"] = "configuration_required"
        else:
            provider["status"] = "ready"

    return {"providers": providers}


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
