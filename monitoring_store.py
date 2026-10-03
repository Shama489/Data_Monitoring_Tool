"""Durable SQLite storage for datasets and monitoring state."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def database_path() -> str:
    if os.getenv("MONITORING_DB_PATH"):
        return os.environ["MONITORING_DB_PATH"]
    data_root = Path(
        os.getenv("LOCALAPPDATA", str(Path.home() / ".local" / "share"))
    )
    return str(data_root / "DataMonitoringTool" / "monitoring.db")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default)


def _decode(value: str | None, fallback: Any = None) -> Any:
    return fallback if value is None else json.loads(value)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS monitoring_results (
            id TEXT PRIMARY KEY,
            dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
            check_type TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_monitoring_results_dataset
            ON monitoring_results(dataset_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS alert_rules (
            rule_order INTEGER PRIMARY KEY,
            rule TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS configurations (
            config_key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT,
            details TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_records_timestamp
            ON audit_records(timestamp DESC);
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            status TEXT NOT NULL,
            message TEXT NOT NULL,
            subject TEXT NOT NULL,
            channels TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            sent TEXT DEFAULT '[]',
            failed TEXT DEFAULT '[]',
            event_type TEXT DEFAULT '',
            severity TEXT DEFAULT 'warning'
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
            ON alerts(timestamp DESC);
        """
    )


def _migrate_legacy_alerts(connection: sqlite3.Connection, target: Path) -> None:
    if os.getenv("MONITORING_DB_PATH"):
        return
    legacy_path = Path(__file__).with_name("alerts.db")
    if not legacy_path.is_file() or legacy_path.resolve() == target.resolve():
        return
    legacy = sqlite3.connect(
        f"{legacy_path.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    try:
        table = legacy.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'alerts'"
        ).fetchone()
        if table is None:
            return
        rows = legacy.execute(
            "SELECT id, timestamp, status, message, subject, channels, dedupe_key, "
            "sent, failed, event_type, severity FROM alerts ORDER BY id"
        ).fetchall()
    finally:
        legacy.close()
    if rows:
        connection.executemany(
            "INSERT OR IGNORE INTO alerts "
            "(id, timestamp, status, message, subject, channels, dedupe_key, sent, "
            "failed, event_type, severity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def open_database(database: str | None = None) -> sqlite3.Connection:
    path = Path(database or database_path()).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_database = not path.exists()
    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        if is_new_database:
            connection.execute("PRAGMA journal_mode = WAL")
        _create_schema(connection)
        if (
            is_new_database
            and not os.getenv("MONITORING_DB_PATH")
            and path.resolve() == Path(database_path()).expanduser().resolve()
        ):
            _migrate_legacy_alerts(connection, path)
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        raise
    return connection


@contextmanager
def connect_database(database: str | None = None) -> Iterator[sqlite3.Connection]:
    connection = open_database(database)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _audit(
    connection: sqlite3.Connection,
    action: str,
    entity_type: str,
    entity_id: str | None,
    details: Any,
) -> None:
    connection.execute(
        "INSERT INTO audit_records (timestamp, action, entity_type, entity_id, details) "
        "VALUES (?, ?, ?, ?, ?)",
        (time.time(), action, entity_type, entity_id, _encode(details)),
    )


def _insert_dataset(
    connection: sqlite3.Connection,
    name: str,
    data: Any,
    source: Any = None,
) -> str:
    dataset_id = uuid.uuid4().hex
    connection.execute(
        "INSERT INTO datasets (id, name, source, data, created_at) VALUES (?, ?, ?, ?, ?)",
        (dataset_id, name, _encode(source or {}), _encode(data), time.time()),
    )
    _audit(connection, "created", "dataset", dataset_id, {"name": name})
    return dataset_id


def save_dataset(name: str, data: Any, source: Any = None) -> str:
    with connect_database() as connection:
        return _insert_dataset(connection, name, data, source)


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    with connect_database() as connection:
        row = connection.execute(
            "SELECT id, name, source, data, created_at FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "source": _decode(row["source"], {}),
        "data": _decode(row["data"], []),
        "created_at": row["created_at"],
    }


def list_datasets(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    with connect_database() as connection:
        rows = connection.execute(
            "SELECT id, name, source, created_at FROM datasets "
            "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "source": _decode(row["source"], {}),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def delete_dataset(dataset_id: str) -> bool:
    with connect_database() as connection:
        row = connection.execute(
            "SELECT name FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        if row is None:
            return False
        connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        _audit(connection, "deleted", "dataset", dataset_id, {"name": row["name"]})
    return True


def persist_monitoring_run(
    datasets: list[dict[str, Any]],
    check_type: str,
    result: Any,
) -> tuple[dict[str, str], str]:
    """Atomically persist the datasets, result, and matching audit events."""
    ids: dict[str, str] = {}
    with connect_database() as connection:
        for item in datasets:
            dataset_id = item.get("id")
            if dataset_id:
                exists = connection.execute(
                    "SELECT 1 FROM datasets WHERE id = ?", (dataset_id,)
                ).fetchone()
                if exists is None:
                    raise ValueError(f"Stored dataset was not found: {dataset_id}")
            else:
                dataset_id = _insert_dataset(
                    connection,
                    item["name"],
                    item["data"],
                    item.get("source"),
                )
            ids[item["key"]] = dataset_id

        current_dataset_id = ids.get("current")
        result_id = uuid.uuid4().hex
        connection.execute(
            "INSERT INTO monitoring_results (id, dataset_id, check_type, result, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (result_id, current_dataset_id, check_type, _encode(result), time.time()),
        )
        _audit(
            connection,
            "created",
            "monitoring_result",
            result_id,
            {"check_type": check_type, "dataset_id": current_dataset_id},
        )
    return ids, result_id


def get_monitoring_result(result_id: str) -> dict[str, Any] | None:
    with connect_database() as connection:
        row = connection.execute(
            "SELECT id, dataset_id, check_type, result, created_at "
            "FROM monitoring_results WHERE id = ?",
            (result_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "check_type": row["check_type"],
        "result": _decode(row["result"], {}),
        "created_at": row["created_at"],
    }


def list_monitoring_results(
    limit: int = 100,
    offset: int = 0,
    dataset_id: str | None = None,
) -> list[dict[str, Any]]:
    query = (
        "SELECT id, dataset_id, check_type, result, created_at "
        "FROM monitoring_results"
    )
    params: list[Any] = []
    if dataset_id is not None:
        query += " WHERE dataset_id = ?"
        params.append(dataset_id)
    query += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
    params.extend((limit, offset))
    with connect_database() as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "check_type": row["check_type"],
            "result": _decode(row["result"], {}),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_alert_rules(database: str | None = None) -> list[dict[str, Any]]:
    with connect_database(database) as connection:
        rows = connection.execute(
            "SELECT rule FROM alert_rules ORDER BY rule_order"
        ).fetchall()
    return [_decode(row["rule"], {}) for row in rows]


def replace_alert_rules(
    rules: list[dict[str, Any]],
    database: str | None = None,
) -> None:
    with connect_database(database) as connection:
        connection.execute("DELETE FROM alert_rules")
        now = time.time()
        connection.executemany(
            "INSERT INTO alert_rules (rule_order, rule, updated_at) VALUES (?, ?, ?)",
            [(index, _encode(rule), now) for index, rule in enumerate(rules)],
        )
        _audit(connection, "replaced", "alert_rules", None, {"count": len(rules)})


def clear_alert_data(database: str | None = None) -> None:
    with connect_database(database) as connection:
        connection.execute("DELETE FROM alerts")
        connection.execute("DELETE FROM alert_rules")


def get_configurations() -> dict[str, Any]:
    with connect_database() as connection:
        rows = connection.execute(
            "SELECT config_key, value FROM configurations ORDER BY config_key"
        ).fetchall()
    return {row["config_key"]: _decode(row["value"]) for row in rows}


def set_configuration(key: str, value: Any) -> None:
    set_configurations({key: value})


def set_configurations(configurations: dict[str, Any]) -> None:
    encoded = [(key, _encode(value)) for key, value in configurations.items()]
    with connect_database() as connection:
        now = time.time()
        connection.executemany(
            "INSERT INTO configurations (config_key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(config_key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            [(key, value, now) for key, value in encoded],
        )
        for key, _ in encoded:
            _audit(connection, "updated", "configuration", key, {"key": key})


def delete_configuration(key: str) -> bool:
    with connect_database() as connection:
        cursor = connection.execute(
            "DELETE FROM configurations WHERE config_key = ?", (key,)
        )
        if cursor.rowcount:
            _audit(connection, "deleted", "configuration", key, {"key": key})
        return cursor.rowcount > 0


def get_audit_records(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    with connect_database() as connection:
        rows = connection.execute(
            "SELECT id, timestamp, action, entity_type, entity_id, details "
            "FROM audit_records ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "action": row["action"],
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "details": _decode(row["details"], {}),
        }
        for row in rows
    ]
