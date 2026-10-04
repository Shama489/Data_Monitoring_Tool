import os
import time

import pytest

os.environ["AUTH_SECRET_KEY"] = "test-only-auth-secret-key-for-data-monitoring"


def admin_headers():
    from auth_security import issue_token

    token = issue_token({"id": "test-admin-user"}, lifetime_seconds=86400)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def isolate_monitoring_database(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "monitoring.db"))
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-only-auth-secret-key-for-data-monitoring")
    from auth_security import hash_password
    from monitoring_store import open_database

    connection = open_database()
    try:
        connection.execute(
            "INSERT INTO users (id, username, password_hash, role, is_active, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (
                "test-admin-user",
                "test-admin",
                hash_password("test-password-123"),
                "admin",
                time.time(),
            ),
        )
        connection.commit()
    finally:
        connection.close()
