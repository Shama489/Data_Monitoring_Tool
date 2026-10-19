"""Load monitoring datasets from files, databases, and cloud storage."""

from __future__ import annotations

import io
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import requests
from sqlalchemy import create_engine, text


class DataSourceError(RuntimeError):
    """Raised when a monitoring source cannot be loaded."""


logger = logging.getLogger(__name__)
MAX_SOURCE_ROWS = 100_000
MAX_SOURCE_COLUMNS = 1_000
MAX_SOURCE_BYTES = 50 * 1024 * 1024
SQL_READ_CHUNK_SIZE = 5_000
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


def _validate_frame_dimensions(
    frame: pd.DataFrame,
    max_rows: int | None = None,
    max_columns: int | None = None,
) -> pd.DataFrame:
    if max_columns is not None and len(frame.columns) > max_columns:
        raise DataSourceError(f"Dataset must not exceed {max_columns:,} columns")
    if max_rows is not None and len(frame.index) > max_rows:
        raise DataSourceError(f"Dataset must not exceed {max_rows:,} rows")
    return frame


def _is_credential_name(name: Any) -> bool:
    normalized = "".join(
        character for character in str(name).lower() if character.isalnum()
    )
    return any(
        marker in normalized
        for marker in (
            "apikey",
            "accesskey",
            "accesstoken",
            "authorization",
            "credential",
            "password",
            "privatekey",
            "secret",
            "token",
        )
    )


def _read_file_bytes(
    content: bytes,
    file_type: str,
    source_name: str,
    sheet_name: str | int | None = None,
    max_rows: int | None = None,
    max_columns: int | None = None,
) -> pd.DataFrame:
    extension = file_type.lower().lstrip(".") or Path(source_name).suffix.lower().lstrip(".")
    stream = io.BytesIO(content)
    try:
        if extension == "csv":
            frame = pd.read_csv(stream, nrows=max_rows + 1 if max_rows is not None else None)
            return _validate_frame_dimensions(frame, max_rows, max_columns)
        if extension in {"xls", "xlsx"}:
            frame = pd.read_excel(
                stream,
                sheet_name=sheet_name or 0,
                nrows=max_rows + 1 if max_rows is not None else None,
            )
            return _validate_frame_dimensions(frame, max_rows, max_columns)
        if extension == "json":
            frame = pd.read_json(stream)
            return _validate_frame_dimensions(frame, max_rows, max_columns)
        if extension == "parquet":
            frame = pd.read_parquet(stream)
            return _validate_frame_dimensions(frame, max_rows, max_columns)
        if extension == "tsv":
            frame = pd.read_csv(
                stream,
                sep="\t",
                nrows=max_rows + 1 if max_rows is not None else None,
            )
            return _validate_frame_dimensions(frame, max_rows, max_columns)
        if extension == "txt":
            frame = pd.read_csv(
                stream,
                sep=None,
                engine="python",
                nrows=max_rows + 1 if max_rows is not None else None,
            )
            return _validate_frame_dimensions(frame, max_rows, max_columns)
        raise DataSourceError(f"Unsupported file type: {extension or 'unknown'}")
    except DataSourceError:
        raise
    except Exception as error:
        raise DataSourceError(f"Could not read {source_name}: {error}") from error


def load_uploaded_file(
    content: bytes,
    file_type: str,
    source_name: str,
    max_rows: int | None = None,
    max_columns: int | None = None,
) -> pd.DataFrame:
    """Parse an uploaded tabular file using the same readers as configured sources."""
    extension = file_type.lower().lstrip(".")
    if extension not in _LOCAL_FILE_EXTENSIONS:
        raise DataSourceError(f"Unsupported file type: {extension or 'unknown'}")
    try:
        if extension in {"csv", "tsv", "txt"}:
            separator: str | None = {
                "csv": ",",
                "tsv": "\t",
                "txt": None,
            }[extension]
            header = pd.read_csv(
                io.BytesIO(content),
                sep=separator,
                engine="python" if separator is None else "c",
                nrows=0,
            )
            if max_columns is not None and len(header.columns) > max_columns:
                raise DataSourceError(f"Dataset must not exceed {max_columns:,} columns")
            frame = pd.read_csv(
                io.BytesIO(content),
                sep=separator,
                engine="python" if separator is None else "c",
                nrows=max_rows + 1 if max_rows is not None else None,
            )
            return _validate_frame_dimensions(frame, max_rows, max_columns)

        return _read_file_bytes(
            content,
            extension,
            source_name,
            max_rows=max_rows,
            max_columns=max_columns,
        )
    except DataSourceError:
        raise
    except Exception as error:
        raise DataSourceError(f"Could not read {source_name}: {error}") from error


def load_local_file(
    path: str,
    file_type: str = "",
    sheet_name: str | int | None = None,
    max_rows: int | None = None,
    max_columns: int | None = None,
) -> pd.DataFrame:
    file_path, extension = _validate_local_file_path(path, file_type)
    try:
        if file_path.stat().st_size > MAX_SOURCE_BYTES:
            raise DataSourceError(
                f"Local source file must not exceed {MAX_SOURCE_BYTES // (1024 * 1024)} MiB"
            )
        content = file_path.read_bytes()
    except DataSourceError:
        raise
    except OSError as error:
        logger.error("Local source file could not be read (%s)", type(error).__name__)
        raise DataSourceError("Local source file could not be read") from error
    return _read_file_bytes(content, extension, file_path.name, sheet_name, max_rows, max_columns)


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


def _read_sql_bounded(
    connection: Any,
    query: str,
    params: dict[str, Any] | None,
    max_rows: int | None,
    max_columns: int | None,
) -> pd.DataFrame:
    chunks = pd.read_sql_query(
        text(query),
        connection,
        params=params,
        chunksize=SQL_READ_CHUNK_SIZE,
    )
    frames = []
    row_count = 0
    columns: list[Any] = []
    for chunk in chunks:
        if not columns:
            columns = list(chunk.columns)
        _validate_frame_dimensions(chunk, max_columns=max_columns)
        row_count += len(chunk.index)
        if max_rows is not None and row_count > max_rows:
            raise DataSourceError(f"Dataset must not exceed {max_rows:,} rows")
        frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def load_sql_query(
    database_url: str,
    query: str,
    params: dict[str, Any] | None = None,
    max_rows: int | None = MAX_SOURCE_ROWS,
    max_columns: int | None = MAX_SOURCE_COLUMNS,
) -> pd.DataFrame:
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

    engine = None
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as connection:
            return _read_sql_bounded(
                connection,
                query,
                params,
                max_rows,
                max_columns,
            )
    except Exception as error:
        if isinstance(error, DataSourceError):
            raise
        logger.error("SQL source query failed (%s)", type(error).__name__)
        raise DataSourceError("Database query failed; check the source configuration and query") from error
    finally:
        if engine is not None:
            engine.dispose()


def load_mongodb(
    uri: str,
    database: str,
    collection: str,
    query: dict[str, Any] | None = None,
    limit: int = 10000,
    max_rows: int | None = MAX_SOURCE_ROWS,
    max_columns: int | None = MAX_SOURCE_COLUMNS,
) -> pd.DataFrame:
    try:
        from pymongo import MongoClient
    except ImportError as error:
        raise DataSourceError("MongoDB support requires pymongo") from error
    if not uri or not database or not collection:
        raise DataSourceError("uri, database, and collection are required for MongoDB")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise DataSourceError("MongoDB limit must be a positive integer")
    bounded_limit = min(limit, max_rows + 1) if max_rows is not None else limit
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        documents = list(
            client[database][collection].find(query or {}, limit=bounded_limit)
        )
        for document in documents:
            document.pop("_id", None)
        return _validate_frame_dimensions(
            pd.DataFrame(documents),
            max_rows,
            max_columns,
        )
    except Exception as error:
        if isinstance(error, DataSourceError):
            raise
        logger.error("MongoDB source query failed (%s)", type(error).__name__)
        raise DataSourceError("MongoDB query failed; check the source configuration") from error
    finally:
        client.close()


def load_s3(bucket: str, key: str, file_type: str = "", sheet_name: str | int | None = None) -> pd.DataFrame:
    try:
        import boto3
    except ImportError as error:
        raise DataSourceError("Amazon S3 support requires boto3") from error
    try:
        response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            content = body.read(MAX_SOURCE_BYTES + 1)
        finally:
            body.close()
        if len(content) > MAX_SOURCE_BYTES:
            raise DataSourceError(
                f"S3 object must not exceed {MAX_SOURCE_BYTES // (1024 * 1024)} MiB"
            )
        return _read_file_bytes(content, file_type, key, sheet_name)
    except DataSourceError:
        raise
    except Exception as error:
        logger.error("S3 source object could not be loaded (%s)", type(error).__name__)
        raise DataSourceError("S3 object could not be loaded; check the source configuration") from error


def load_dropbox(path: str, file_type: str = "", sheet_name: str | int | None = None) -> pd.DataFrame:
    try:
        import dropbox
    except ImportError as error:
        raise DataSourceError("Dropbox support requires dropbox") from error
    token = os.getenv("DROPBOX_ACCESS_TOKEN")
    if not token:
        raise DataSourceError("DROPBOX_ACCESS_TOKEN is required")
    try:
        client = dropbox.Dropbox(token)
        metadata = client.files_get_metadata(path)
        if getattr(metadata, "size", 0) > MAX_SOURCE_BYTES:
            raise DataSourceError(
                f"Dropbox file must not exceed {MAX_SOURCE_BYTES // (1024 * 1024)} MiB"
            )
        _, response = client.files_download(path)
        if len(response.content) > MAX_SOURCE_BYTES:
            raise DataSourceError(
                f"Dropbox file must not exceed {MAX_SOURCE_BYTES // (1024 * 1024)} MiB"
            )
        return _read_file_bytes(response.content, file_type, path, sheet_name)
    except DataSourceError:
        raise
    except Exception as error:
        logger.error("Dropbox source file could not be loaded (%s)", type(error).__name__)
        raise DataSourceError("Dropbox file could not be loaded; check the source configuration") from error


def load_google_drive(file_id: str, file_type: str = "", sheet_name: str | int | None = None) -> pd.DataFrame:
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
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
        drive = build("drive", "v3", credentials=credentials)
        metadata = drive.files().get(fileId=file_id, fields="name,mimeType,size").execute()
        if int(metadata.get("size", 0)) > MAX_SOURCE_BYTES:
            raise DataSourceError(
                f"Google Drive file must not exceed {MAX_SOURCE_BYTES // (1024 * 1024)} MiB"
            )
        output = io.BytesIO()
        downloader = MediaIoBaseDownload(
            output,
            drive.files().get_media(fileId=file_id),
            chunksize=256 * 1024,
        )
        complete = False
        while not complete:
            _, complete = downloader.next_chunk()
            if output.tell() > MAX_SOURCE_BYTES:
                raise DataSourceError(
                    f"Google Drive file must not exceed {MAX_SOURCE_BYTES // (1024 * 1024)} MiB"
                )
        response = output.getvalue()
        return _read_file_bytes(response, file_type, metadata.get("name", file_id), sheet_name)
    except DataSourceError:
        raise
    except Exception as error:
        logger.error("Google Drive source could not be loaded (%s)", type(error).__name__)
        raise DataSourceError("Google Drive file could not be loaded; check the source configuration") from error


def _environment_value(source: dict[str, Any], setting: str) -> str:
    environment_name = source.get(setting)
    if (
        not isinstance(environment_name, str)
        or not environment_name
        or not environment_name.isidentifier()
    ):
        raise DataSourceError(f"{setting} must name a server environment variable")
    value = os.getenv(environment_name)
    if not value:
        raise DataSourceError(f"The server environment variable {environment_name} is not configured")
    return value


def load_http_json(
    url: str,
    token_env: str | None = None,
    records_path: str | None = None,
    params: dict[str, Any] | None = None,
    max_rows: int | None = MAX_SOURCE_ROWS,
    max_columns: int | None = MAX_SOURCE_COLUMNS,
) -> pd.DataFrame:
    """Load tabular records from an allowlisted HTTPS JSON API."""
    if records_path is not None and (
        not isinstance(records_path, str)
        or not records_path
        or any(not part for part in records_path.split("."))
    ):
        raise DataSourceError("records_path must be a non-empty dotted path")
    if params is not None and not isinstance(params, dict):
        raise DataSourceError("HTTP API params must be an object")
    if params and any(_is_credential_name(key) for key in params):
        raise DataSourceError(
            "Do not put credentials in HTTP query parameters; use token_env"
        )
    parsed_url = urlsplit(url)
    if (
        parsed_url.scheme.lower() != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise DataSourceError("HTTP JSON sources require an HTTPS URL without embedded credentials")
    if any(
        _is_credential_name(key)
        for key, _ in parse_qsl(parsed_url.query, keep_blank_values=True)
    ):
        raise DataSourceError(
            "Do not put credentials in HTTP query parameters; use token_env"
        )

    configured_hosts = {
        item.strip().lower()
        for item in os.getenv("DATA_SOURCE_HTTP_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    }
    host_with_port = parsed_url.netloc.lower()
    if not configured_hosts:
        raise DataSourceError("DATA_SOURCE_HTTP_ALLOWED_HOSTS is not configured")
    try:
        port = parsed_url.port
    except ValueError as error:
        raise DataSourceError("HTTP API URL contains an invalid port") from error
    hostname = parsed_url.hostname.lower()
    allowed_by_hostname = hostname in configured_hosts
    if port not in (None, 443) and host_with_port not in configured_hosts:
        allowed_by_hostname = False
    if host_with_port not in configured_hosts and not allowed_by_hostname:
        raise DataSourceError("HTTP API host is not listed in DATA_SOURCE_HTTP_ALLOWED_HOSTS")

    headers = {"Accept": "application/json"}
    if token_env:
        headers["Authorization"] = (
            f"Bearer {_environment_value({'token_env': token_env}, 'token_env')}"
        )

    response = None
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=(5, 30),
            allow_redirects=False,
            stream=True,
        )
        if response.is_redirect:
            raise DataSourceError("HTTP API redirects are not supported")
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            content.extend(chunk)
            if len(content) > MAX_SOURCE_BYTES:
                raise DataSourceError(
                    f"HTTP API response must not exceed {MAX_SOURCE_BYTES // (1024 * 1024)} MiB"
                )
        payload = json.loads(content)
    except DataSourceError:
        raise
    except requests.RequestException as error:
        logger.error("HTTP JSON source request failed (%s)", type(error).__name__)
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            raise DataSourceError(f"HTTP API request failed with status {status_code}") from error
        raise DataSourceError("HTTP API request failed; check the endpoint and connectivity") from error
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        logger.error("HTTP JSON source returned invalid JSON (%s)", type(error).__name__)
        raise DataSourceError("HTTP API response was not valid JSON") from error
    finally:
        if response is not None:
            response.close()

    records: Any = payload
    if records_path:
        for key in records_path.split("."):
            if not isinstance(records, dict) or key not in records:
                raise DataSourceError("records_path does not exist in the HTTP API response")
            records = records[key]
    elif isinstance(records, dict):
        records = records.get("data", records.get("results"))
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise DataSourceError("HTTP API response must contain a list of record objects")
    return _validate_frame_dimensions(
        pd.DataFrame(records),
        max_rows,
        max_columns,
    )


def load_source(
    source: dict[str, Any],
    max_rows: int | None = MAX_SOURCE_ROWS,
    max_columns: int | None = MAX_SOURCE_COLUMNS,
) -> pd.DataFrame:
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
        frame = load_local_file(
            str(path),
            file_type or normalized_type,
            sheet_name,
            max_rows,
            max_columns,
        )
        return _validate_frame_dimensions(frame, max_rows, max_columns)
    if source_type in {"postgresql", "mysql", "sql"}:
        if "url" in source and "url_env" in source:
            raise DataSourceError("Specify either url or url_env for SQL sources, not both")
        database_url = (
            _environment_value(source, "url_env")
            if "url_env" in source
            else str(source.get("url", ""))
        )
        return load_sql_query(
            database_url,
            str(source.get("query", "")),
            source.get("params"),
            max_rows,
            max_columns,
        )
    if source_type == "mongodb":
        if "uri" in source and "uri_env" in source:
            raise DataSourceError("Specify either uri or uri_env for MongoDB sources, not both")
        uri = (
            _environment_value(source, "uri_env")
            if "uri_env" in source
            else str(source.get("uri", ""))
        )
        return load_mongodb(
            uri,
            str(source.get("database", "")),
            str(source.get("collection", "")),
            source.get("query"),
            int(source.get("limit", 10000)),
            max_rows,
            max_columns,
        )
    if source_type == "s3":
        frame = load_s3(str(source.get("bucket", "")), str(source.get("key", "")), file_type, sheet_name)
    elif source_type == "dropbox":
        frame = load_dropbox(str(source.get("path", "")), file_type, sheet_name)
    elif source_type in {"google_drive", "gdrive"}:
        frame = load_google_drive(str(source.get("file_id", "")), file_type, sheet_name)
    elif source_type in {"http_json", "rest_json"}:
        if "url" not in source:
            raise DataSourceError("url is required for HTTP JSON sources")
        frame = load_http_json(
            str(source["url"]),
            source.get("token_env"),
            source.get("records_path"),
            source.get("params"),
            max_rows,
            max_columns,
        )
    else:
        raise DataSourceError(f"Unsupported source type: {source_type or 'unknown'}")
    return _validate_frame_dimensions(frame, max_rows, max_columns)


def source_capabilities() -> dict[str, Any]:
    """Report supported sources, optional SDK availability, and setup requirements."""
    providers = [
        {
            "type": "local_files",
            "formats": ["csv", "excel", "json", "parquet", "tsv", "txt"],
            "dependencies": {"pandas": "pandas", "excel": "openpyxl", "parquet": "pyarrow"},
            "required_fields": ["path"],
            "credential_env": ["DATASET_ALLOWED_DIRS"],
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
        {
            "type": "http_json",
            "aliases": ["rest_json"],
            "dependencies": {"requests": "requests"},
            "required_fields": ["url"],
            "credential_env": ["DATA_SOURCE_HTTP_ALLOWED_HOSTS"],
            "credential_setup": "Allowlist API hostnames in DATA_SOURCE_HTTP_ALLOWED_HOSTS; use token_env for optional bearer credentials.",
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
        "source": {
            key: value
            for key, value in source.items()
            if key.lower()
            not in {
                "url",
                "uri",
                "query",
                "params",
                "headers",
                "authorization",
                "password",
                "secret",
                "token",
                "credentials",
            }
        },
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "quality": {
            "nulls": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
        },
    }
