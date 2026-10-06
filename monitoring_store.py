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
            created_at REAL NOT NULL,
            owner_id TEXT
        );
        CREATE TABLE IF NOT EXISTS monitoring_results (
            id TEXT PRIMARY KEY,
            dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
            check_type TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            owner_id TEXT
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
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'analyst', 'viewer')),
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS security_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS login_attempts (
            client_key TEXT PRIMARY KEY,
            attempts INTEGER NOT NULL,
            window_started REAL NOT NULL,
            blocked_until REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS monitoring_schedules (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            schedule TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            next_run_at REAL NOT NULL,
            last_started_at REAL,
            last_run_at REAL,
            last_status TEXT NOT NULL DEFAULT 'pending',
            last_error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_monitoring_schedules_due
            ON monitoring_schedules(enabled, next_run_at);
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
    connection.execute("BEGIN IMMEDIATE")
    for table in ("datasets", "monitoring_results", "audit_records"):
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if "owner_id" not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN owner_id TEXT")


def _migrate_stored_content(connection: sqlite3.Connection) -> None:
    if not os.getenv("AUTH_SECRET_KEY"):
        return
    migration_id = "encrypt_dataset_and_result_content_v1"
    if connection.execute(
        "SELECT 1 FROM security_migrations WHERE migration_id = ?",
        (migration_id,),
    ).fetchone():
        return

    from auth_security import encrypt_legacy_value

    for table, content_column in (
        ("datasets", "data"),
        ("monitoring_results", "result"),
    ):
        rows = connection.execute(
            f"SELECT id, {content_column} FROM {table}"
        ).fetchall()
        connection.executemany(
            f"UPDATE {table} SET {content_column} = ? WHERE id = ?",
            [
                (encrypt_legacy_value(row[content_column]), row["id"])
                for row in rows
            ],
        )
    connection.execute(
        "INSERT INTO security_migrations (migration_id, applied_at) VALUES (?, ?)",
        (migration_id, time.time()),
    )


def create_user(username: str, password_hash: str, role: str) -> dict[str, Any]:
    user_id = uuid.uuid4().hex
    try:
        with connect_database() as connection:
            connection.execute(
                "INSERT INTO users (id, username, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, username, password_hash, role, time.time()),
            )
            _audit(connection, "created", "user", user_id, {"username": username, "role": role})
    except sqlite3.IntegrityError as error:
        raise ValueError("username is already registered") from error
    return {"id": user_id, "username": username, "role": role, "is_active": True}


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with connect_database() as connection:
        row = connection.execute(
            "SELECT id, username, password_hash, role, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row is not None else None


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    with connect_database() as connection:
        row = connection.execute(
            "SELECT id, username, role, is_active FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def list_users(limit: int = 500) -> list[dict[str, Any]]:
    with connect_database() as connection:
        rows = connection.execute(
            "SELECT id, username, role, is_active, created_at FROM users "
            "ORDER BY created_at, username LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_user(user_id: str, role: str, is_active: bool) -> dict[str, Any] | None:
    with connect_database() as connection:
        cursor = connection.execute(
            "UPDATE users SET role = ?, is_active = ? WHERE id = ?",
            (role, int(is_active), user_id),
        )
        if cursor.rowcount == 0:
            return None
        row = connection.execute(
            "SELECT id, username, role, is_active FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        _audit(
            connection,
            "updated",
            "user",
            user_id,
            {"username": row["username"], "role": role, "is_active": bool(is_active)},
        )
    return dict(row)


def login_retry_after(client_key: str) -> int:
    now = time.time()
    with connect_database() as connection:
        row = connection.execute(
            "SELECT blocked_until FROM login_attempts WHERE client_key = ?",
            (client_key,),
        ).fetchone()
    return max(0, int(row["blocked_until"] - now + 0.999)) if row else 0


def record_login_failure(
    client_key: str, max_attempts: int = 5, window_seconds: int = 900
) -> int:
    now = time.time()
    with connect_database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT attempts, window_started, blocked_until FROM login_attempts "
            "WHERE client_key = ?",
            (client_key,),
        ).fetchone()
        if row and row["blocked_until"] > now:
            return int(row["blocked_until"] - now + 0.999)
        attempts = row["attempts"] if row and row["window_started"] > now - window_seconds else 0
        window_started = row["window_started"] if attempts else now
        attempts += 1
        blocked_until = now + window_seconds if attempts >= max_attempts else 0
        connection.execute(
            "INSERT INTO login_attempts (client_key, attempts, window_started, blocked_until) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(client_key) DO UPDATE SET "
            "attempts = excluded.attempts, window_started = excluded.window_started, "
            "blocked_until = excluded.blocked_until",
            (client_key, attempts, window_started, blocked_until),
        )
    return max(0, int(blocked_until - now + 0.999))


def clear_login_failures(client_key: str) -> None:
    with connect_database() as connection:
        connection.execute(
            "DELETE FROM login_attempts WHERE client_key = ?", (client_key,)
        )


def create_monitoring_schedule(
    owner_id: str,
    dataset_id: str,
    schedule: dict[str, Any],
    interval_seconds: int,
) -> dict[str, Any]:
    now = time.time()
    schedule_id = uuid.uuid4().hex
    persisted_schedule = {**schedule, "interval_seconds": interval_seconds}
    with connect_database() as connection:
        connection.execute(
            "INSERT INTO monitoring_schedules "
            "(id, owner_id, dataset_id, schedule, enabled, next_run_at, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
            (
                schedule_id,
                owner_id,
                dataset_id,
                _encode(persisted_schedule),
                now + interval_seconds,
                now,
                now,
            ),
        )
        _audit(
            connection,
            "created",
            "monitoring_schedule",
            schedule_id,
            {"dataset_id": dataset_id, "interval_seconds": interval_seconds},
            owner_id,
        )
    return get_monitoring_schedule(schedule_id, owner_id)


def get_monitoring_schedule(
    schedule_id: str, owner_id: str | None = None
) -> dict[str, Any] | None:
    with connect_database() as connection:
        if owner_id is None:
            row = connection.execute(
                "SELECT * FROM monitoring_schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM monitoring_schedules WHERE id = ? AND owner_id = ?",
                (schedule_id, owner_id),
            ).fetchone()
    return _schedule_record(row) if row is not None else None


def _schedule_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "owner_id": row["owner_id"],
        "dataset_id": row["dataset_id"],
        "schedule": _decode(row["schedule"], {}),
        "enabled": bool(row["enabled"]),
        "next_run_at": row["next_run_at"],
        "last_started_at": row["last_started_at"],
        "last_run_at": row["last_run_at"],
        "last_status": row["last_status"],
        "last_error": row["last_error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_monitoring_schedules(owner_id: str | None = None) -> list[dict[str, Any]]:
    with connect_database() as connection:
        if owner_id is None:
            rows = connection.execute(
                "SELECT * FROM monitoring_schedules ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM monitoring_schedules WHERE owner_id = ? "
                "ORDER BY created_at DESC",
                (owner_id,),
            ).fetchall()
    return [_schedule_record(row) for row in rows]


def set_monitoring_schedule_enabled(
    schedule_id: str, enabled: bool, owner_id: str | None = None
) -> bool:
    now = time.time()
    with connect_database() as connection:
        if owner_id is None:
            cursor = connection.execute(
                "UPDATE monitoring_schedules SET enabled = ?, "
                "last_status = CASE WHEN ? THEN 'pending' ELSE 'paused' END, "
                "updated_at = ? WHERE id = ?",
                (int(enabled), int(enabled), now, schedule_id),
            )
        else:
            cursor = connection.execute(
                "UPDATE monitoring_schedules SET enabled = ?, "
                "last_status = CASE WHEN ? THEN 'pending' ELSE 'paused' END, "
                "updated_at = ? WHERE id = ? AND owner_id = ?",
                (int(enabled), int(enabled), now, schedule_id, owner_id),
            )
        if cursor.rowcount:
            _audit(
                connection,
                "enabled" if enabled else "paused",
                "monitoring_schedule",
                schedule_id,
                {},
                owner_id,
            )
    return cursor.rowcount > 0


def delete_monitoring_schedule(
    schedule_id: str, owner_id: str | None = None
) -> bool:
    with connect_database() as connection:
        if owner_id is None:
            cursor = connection.execute(
                "DELETE FROM monitoring_schedules WHERE id = ?", (schedule_id,)
            )
        else:
            cursor = connection.execute(
                "DELETE FROM monitoring_schedules WHERE id = ? AND owner_id = ?",
                (schedule_id, owner_id),
            )
        if cursor.rowcount:
            _audit(
                connection,
                "deleted",
                "monitoring_schedule",
                schedule_id,
                {},
                owner_id,
            )
    return cursor.rowcount > 0


def claim_due_monitoring_schedules(
    now: float | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    current_time = time.time() if now is None else now
    claimed = []
    with connect_database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            "SELECT * FROM monitoring_schedules "
            "WHERE enabled = 1 AND next_run_at <= ? "
            "AND (last_status != 'running' OR last_started_at <= ?) "
            "ORDER BY next_run_at LIMIT ?",
            (current_time, current_time - 3600, limit),
        ).fetchall()
        for row in rows:
            schedule = _schedule_record(row)
            interval = int(schedule["schedule"]["interval_seconds"])
            next_run = max(float(row["next_run_at"]), current_time) + interval
            connection.execute(
                "UPDATE monitoring_schedules SET next_run_at = ?, "
                "last_started_at = ?, last_status = 'running', last_error = NULL, "
                "updated_at = ? WHERE id = ? AND enabled = 1 AND next_run_at <= ? "
                "AND (last_status != 'running' OR last_started_at <= ?)",
                (
                    next_run,
                    current_time,
                    current_time,
                    row["id"],
                    current_time,
                    current_time - 3600,
                ),
            )
            if connection.execute("SELECT changes()").fetchone()[0]:
                claimed.append(schedule)
    return claimed


def finish_monitoring_schedule(
    schedule_id: str,
    succeeded: bool,
    error: str | None = None,
    finished_at: float | None = None,
) -> None:
    now = time.time() if finished_at is None else finished_at
    status = "succeeded" if succeeded else "failed"
    with connect_database() as connection:
        row = connection.execute(
            "SELECT owner_id FROM monitoring_schedules WHERE id = ?",
            (schedule_id,),
        ).fetchone()
        if row is None:
            return
        connection.execute(
            "UPDATE monitoring_schedules SET last_run_at = ?, last_status = ?, "
            "last_error = ?, updated_at = ? WHERE id = ?",
            (now, status, error[:1000] if error else None, now, schedule_id),
        )
        _audit(
            connection,
            "run_succeeded" if succeeded else "run_failed",
            "monitoring_schedule",
            schedule_id,
            {"error": error[:1000] if error else None},
            row["owner_id"],
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
        _migrate_stored_content(connection)
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
    owner_id: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO audit_records "
        "(timestamp, action, entity_type, entity_id, details, owner_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (time.time(), action, entity_type, entity_id, _encode(details), owner_id),
    )


def _insert_dataset(
    connection: sqlite3.Connection,
    name: str,
    data: Any,
    source: Any = None,
    owner_id: str | None = None,
) -> str:
    from auth_security import encrypt_data

    dataset_id = uuid.uuid4().hex
    connection.execute(
        "INSERT INTO datasets (id, name, source, data, created_at, owner_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (dataset_id, name, _encode(source or {}), encrypt_data(data), time.time(), owner_id),
    )
    _audit(connection, "created", "dataset", dataset_id, {"name": name}, owner_id)
    return dataset_id


def save_dataset(
    name: str, data: Any, source: Any = None, owner_id: str | None = None
) -> str:
    with connect_database() as connection:
        return _insert_dataset(connection, name, data, source, owner_id)


def get_dataset(dataset_id: str, owner_id: str | None = None) -> dict[str, Any] | None:
    from auth_security import decrypt_data

    with connect_database() as connection:
        if owner_id is None:
            row = connection.execute(
                "SELECT id, name, source, data, created_at, owner_id "
                "FROM datasets WHERE id = ?",
                (dataset_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT id, name, source, data, created_at, owner_id FROM datasets "
                "WHERE id = ? AND owner_id = ?",
                (dataset_id, owner_id),
            ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "source": _decode(row["source"], {}),
        "data": decrypt_data(row["data"]),
        "created_at": row["created_at"],
        "owner_id": row["owner_id"],
    }


def list_datasets(
    limit: int = 100, offset: int = 0, owner_id: str | None = None
) -> list[dict[str, Any]]:
    with connect_database() as connection:
        if owner_id is None:
            rows = connection.execute(
                "SELECT id, name, source, created_at FROM datasets "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id, name, source, created_at FROM datasets WHERE owner_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (owner_id, limit, offset),
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


def delete_dataset(dataset_id: str, owner_id: str | None = None) -> bool:
    with connect_database() as connection:
        if owner_id is None:
            row = connection.execute(
                "SELECT name FROM datasets WHERE id = ?", (dataset_id,)
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT name FROM datasets WHERE id = ? AND owner_id = ?",
                (dataset_id, owner_id),
            ).fetchone()
        if row is None:
            return False
        connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        _audit(connection, "deleted", "dataset", dataset_id, {"name": row["name"]}, owner_id)
    return True


def persist_monitoring_run(
    datasets: list[dict[str, Any]],
    check_type: str,
    result: Any,
    owner_id: str | None = None,
) -> tuple[dict[str, str], str]:
    """Atomically persist the datasets, result, and matching audit events."""
    from auth_security import encrypt_data

    ids: dict[str, str] = {}
    with connect_database() as connection:
        for item in datasets:
            dataset_id = item.get("id")
            if dataset_id:
                exists = connection.execute(
                    "SELECT 1 FROM datasets WHERE id = ? AND (? IS NULL OR owner_id = ?)",
                    (dataset_id, owner_id, owner_id),
                ).fetchone()
                if exists is None:
                    raise ValueError(f"Stored dataset was not found: {dataset_id}")
            else:
                dataset_id = _insert_dataset(
                    connection,
                    item["name"],
                    item["data"],
                    item.get("source"),
                    owner_id,
                )
            ids[item["key"]] = dataset_id

        current_dataset_id = ids.get("current")
        result_id = uuid.uuid4().hex
        connection.execute(
            "INSERT INTO monitoring_results "
            "(id, dataset_id, check_type, result, created_at, owner_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                result_id,
                current_dataset_id,
                check_type,
                encrypt_data(result),
                time.time(),
                owner_id,
            ),
        )
        _audit(
            connection,
            "created",
            "monitoring_result",
            result_id,
            {"check_type": check_type, "dataset_id": current_dataset_id},
            owner_id,
        )
    return ids, result_id


def get_monitoring_result(
    result_id: str, owner_id: str | None = None
) -> dict[str, Any] | None:
    from auth_security import decrypt_data

    with connect_database() as connection:
        if owner_id is None:
            row = connection.execute(
                "SELECT id, dataset_id, check_type, result, created_at "
                "FROM monitoring_results WHERE id = ?",
                (result_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT id, dataset_id, check_type, result, created_at "
                "FROM monitoring_results WHERE id = ? AND owner_id = ?",
                (result_id, owner_id),
            ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "check_type": row["check_type"],
        "result": decrypt_data(row["result"]),
        "created_at": row["created_at"],
    }


def list_monitoring_results(
    limit: int = 100,
    offset: int = 0,
    dataset_id: str | None = None,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    from auth_security import decrypt_data

    query = (
        "SELECT id, dataset_id, check_type, result, created_at "
        "FROM monitoring_results"
    )
    params: list[Any] = []
    if owner_id is not None:
        query += " WHERE owner_id = ?"
        params.append(owner_id)
    if dataset_id is not None:
        query += " AND dataset_id = ?" if owner_id is not None else " WHERE dataset_id = ?"
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
            "result": decrypt_data(row["result"]),
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


def get_audit_records(
    limit: int = 100, offset: int = 0, owner_id: str | None = None
) -> list[dict[str, Any]]:
    with connect_database() as connection:
        if owner_id is None:
            rows = connection.execute(
                "SELECT id, timestamp, action, entity_type, entity_id, details "
                "FROM audit_records ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id, timestamp, action, entity_type, entity_id, details "
                "FROM audit_records WHERE owner_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (owner_id, limit, offset),
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
