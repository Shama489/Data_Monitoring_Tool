import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import admin_headers
from main import app
import monitoring_store
from monitoring_store import database_path, get_dataset, get_monitoring_result


def test_api_requires_authentication_except_health_and_login():
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/datasets").status_code == 401
    response = client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "test-password-123"},
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


def test_login_is_rate_limited_after_five_failed_attempts():
    client = TestClient(app)
    for attempt in range(4):
        response = client.post(
            "/api/auth/login",
            json={"username": "test-admin", "password": f"wrong-password-{attempt}"},
        )
        assert response.status_code == 401

    blocked = client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "wrong-password"},
    )
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0
    assert client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "test-password-123"},
    ).status_code == 429


def test_users_have_role_permissions_and_isolated_encrypted_datasets():
    admin = TestClient(app, headers=admin_headers())
    analyst_response = admin.post(
        "/api/auth/users",
        json={
            "username": "Analyst-One",
            "password": "analyst-password-123",
            "role": "analyst",
        },
    )
    assert analyst_response.status_code == 201
    analyst_id = analyst_response.json()["id"]

    login = admin.post(
        "/api/auth/login",
        json={"username": "analyst-one", "password": "analyst-password-123"},
    )
    assert login.status_code == 200
    analyst = TestClient(
        app,
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    saved = analyst.post(
        "/api/datasets",
        json={"name": "private", "data": [{"sensitive": "customer-secret"}]},
    )
    assert saved.status_code == 201
    dataset_id = saved.json()["id"]
    advanced_analysis = analyst.post(
        "/api/analytics/advanced-ml",
        json={
            "analysis": "clustering",
            "n_clusters": 2,
            "data": [{"value": 0}, {"value": 1}, {"value": 10}, {"value": 11}],
        },
    )
    assert advanced_analysis.status_code == 200

    connection = sqlite3.connect(database_path())
    try:
        ciphertext = connection.execute(
            "SELECT data FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert "customer-secret" not in ciphertext

    viewer_created = admin.post(
        "/api/auth/users",
        json={
            "username": "viewer-one",
            "password": "viewer-password-123",
            "role": "viewer",
        },
    )
    assert viewer_created.status_code == 201
    viewer_login = admin.post(
        "/api/auth/login",
        json={"username": "viewer-one", "password": "viewer-password-123"},
    )
    viewer = TestClient(
        app,
        headers={"Authorization": f"Bearer {viewer_login.json()['access_token']}"},
    )
    assert viewer.get(f"/api/datasets/{dataset_id}").status_code == 404
    assert viewer.post(
        "/api/datasets", json={"name": "denied", "data": [{"x": 1}]}
    ).status_code == 403
    assert viewer.post(
        "/api/analytics/advanced-ml",
        json={"analysis": "clustering", "data": [{"value": 1}, {"value": 2}, {"value": 3}]},
    ).status_code == 403
    assert analyst.post("/api/auth/users", json={}).status_code == 403
    analyst_dataset_ids = {
        dataset["id"] for dataset in analyst.get("/api/datasets").json()["datasets"]
    }
    assert dataset_id in analyst_dataset_ids

    uploaded = analyst.post(
        "/api/datasets/upload",
        params={"name": "..\\private.csv"},
        content=b"value\nsensitive-upload\n",
        headers={"Content-Type": "text/csv"},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["name"] == "private.csv"

    updated = admin.patch(
        f"/api/auth/users/{analyst_id}",
        json={"role": "viewer", "is_active": False},
    )
    assert updated.status_code == 200
    assert analyst.get("/api/auth/me").status_code == 401


def test_dataset_upload_rejects_oversized_requests():
    client = TestClient(app, headers=admin_headers())
    response = client.post(
        "/api/datasets",
        content=b" " * (10 * 1024 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413

    chunked_response = client.post(
        "/api/datasets/upload",
        content=iter([b" " * (10 * 1024 * 1024 + 1)]),
        headers={"Content-Type": "text/csv"},
    )
    assert chunked_response.status_code == 413, chunked_response.text


def test_csv_upload_enforces_row_and_column_limits():
    client = TestClient(app, headers=admin_headers())
    too_many_rows = client.post(
        "/api/datasets/upload",
        params={"name": "many-rows.csv"},
        content=("value\n" + "1\n" * 100_001).encode(),
        headers={"Content-Type": "text/csv"},
    )
    assert too_many_rows.status_code == 400
    assert "rows" in too_many_rows.json()["detail"]

    too_many_columns = client.post(
        "/api/datasets/upload",
        params={"name": "many-columns.csv"},
        content=(",".join(f"column_{index}" for index in range(1001))).encode(),
        headers={"Content-Type": "text/csv"},
    )
    assert too_many_columns.status_code == 400
    assert "columns" in too_many_columns.json()["detail"]


def test_preexisting_plaintext_monitoring_content_is_encrypted_on_upgrade(
    tmp_path, monkeypatch
):
    legacy_path = tmp_path / "legacy-monitoring.db"
    connection = sqlite3.connect(legacy_path)
    connection.executescript(
        """
        CREATE TABLE datasets (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL,
            data TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE monitoring_results (
            id TEXT PRIMARY KEY, dataset_id TEXT, check_type TEXT NOT NULL,
            result TEXT NOT NULL, created_at REAL NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO datasets VALUES (?, ?, ?, ?, ?)",
        ("legacy-dataset", "legacy", "{}", '[{"value":"old-secret"}]', 1),
    )
    connection.execute(
        "INSERT INTO monitoring_results VALUES (?, ?, ?, ?, ?)",
        ("legacy-result", "legacy-dataset", "quality", '{"report":"old-report"}', 1),
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("MONITORING_DB_PATH", str(legacy_path))
    connection = monitoring_store.open_database()
    try:
        raw_dataset = connection.execute(
            "SELECT data FROM datasets WHERE id = 'legacy-dataset'"
        ).fetchone()["data"]
        raw_result = connection.execute(
            "SELECT result FROM monitoring_results WHERE id = 'legacy-result'"
        ).fetchone()["result"]
    finally:
        connection.close()

    assert "old-secret" not in raw_dataset
    assert "old-report" not in raw_result
    assert get_dataset("legacy-dataset")["data"] == [{"value": "old-secret"}]
    assert get_monitoring_result("legacy-result")["result"] == {"report": "old-report"}


def test_legacy_api_modules_reuse_the_secured_application():
    import importlib.util

    import API.main

    backend_path = Path(__file__).parents[1] / "backend" / "backend.py"
    spec = importlib.util.spec_from_file_location("backend_api_entry", backend_path)
    assert spec is not None and spec.loader is not None
    backend_entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_entry)
    assert API.main.app is app
    assert backend_entry.app is app
