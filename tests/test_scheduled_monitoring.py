import time

from fastapi.testclient import TestClient

from conftest import admin_headers
from main import app
from monitoring_store import create_user, open_database
from scheduled_monitoring import run_due_schedules


client = TestClient(app, headers=admin_headers())


def test_schedule_runs_recurring_freshness_check_and_alerts(monkeypatch):
    created_dataset = client.post(
        "/api/datasets",
        json={
            "name": "stale-events",
            "data": [{"observed_at": "2020-01-01T00:00:00Z", "value": 2}],
        },
    )
    assert created_dataset.status_code == 201
    dataset_id = created_dataset.json()["id"]

    sent = []
    monkeypatch.setattr(
        "agents.monitoring_agent.send_monitoring_notification",
        lambda message, channels, severity="warning": (
            sent.append({"message": message, "channels": channels, "severity": severity})
            or {"status": "sent"}
        ),
    )
    created_schedule = client.post(
        "/api/schedules",
        json={
            "dataset_id": dataset_id,
            "interval_seconds": 60,
            "checks": ["quality"],
            "timestamp_column": "observed_at",
            "max_age_hours": 24,
            "alert_channels": [{"channel": "email", "recipient": "ops@example.com"}],
        },
    )
    assert created_schedule.status_code == 201
    schedule_id = created_schedule.json()["id"]

    connection = open_database()
    try:
        connection.execute(
            "UPDATE monitoring_schedules SET next_run_at = ? WHERE id = ?",
            (time.time() - 1, schedule_id),
        )
        connection.commit()
    finally:
        connection.close()

    assert run_due_schedules() == 1
    schedules = client.get("/api/schedules").json()["schedules"]
    schedule = next(item for item in schedules if item["id"] == schedule_id)
    assert schedule["last_status"] == "succeeded"
    assert schedule["last_run_at"] is not None
    assert schedule["next_run_at"] > schedule["last_run_at"]
    assert sent

    results = client.get(
        "/api/monitoring/results",
        params={"dataset_id": dataset_id, "limit": 10},
    ).json()["results"]
    scheduled_result = next(
        item for item in results
        if item["result"]["schedule_id"] == schedule_id
    )
    freshness = scheduled_result["result"]["results"]["quality"]["report"]["data_freshness"]
    assert freshness["is_fresh"] is False
    assert scheduled_result["result"]["notifications"]["status"] == "sent"
    assert run_due_schedules() == 0


def test_source_schedule_creates_versions_and_monitors_each_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ALLOWED_DIRS", str(tmp_path))
    source_path = tmp_path / "events.csv"
    source_path.write_text("value\n1\n2\n", encoding="utf-8")
    response = client.post(
        "/api/schedules",
        json={
            "source": {"type": "csv", "path": str(source_path)},
            "interval_seconds": 60,
            "checks": ["quality"],
        },
    )

    assert response.status_code == 201, response.text
    schedule = response.json()
    root_dataset_id = schedule["dataset_id"]
    source_path.write_text("value\n10\n20\n", encoding="utf-8")
    connection = open_database()
    try:
        connection.execute(
            "UPDATE monitoring_schedules SET next_run_at = ? WHERE id = ?",
            (time.time() - 1, schedule["id"]),
        )
        connection.commit()
    finally:
        connection.close()

    assert run_due_schedules() == 1
    updated_schedule = next(
        item for item in client.get("/api/schedules").json()["schedules"]
        if item["id"] == schedule["id"]
    )
    latest_dataset_id = updated_schedule["schedule"]["latest_dataset_id"]
    versions = client.get(
        f"/api/datasets/{root_dataset_id}/versions"
    ).json()["versions"]
    latest_dataset = client.get(f"/api/datasets/{latest_dataset_id}").json()
    results = client.get(
        "/api/monitoring/results",
        params={"dataset_id": latest_dataset_id},
    ).json()["results"]

    assert len(versions) == 2
    assert versions[1]["parent_dataset_id"] == root_dataset_id
    assert latest_dataset["data"] == [{"value": 10}, {"value": 20}]
    assert latest_dataset["version_number"] == 2
    assert any(
        item["result"]["schedule_id"] == schedule["id"]
        for item in results
    )


def test_failed_source_poll_creates_no_version_and_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ALLOWED_DIRS", str(tmp_path))
    source_path = tmp_path / "events.csv"
    source_path.write_text("value\n1\n", encoding="utf-8")
    created = client.post(
        "/api/schedules",
        json={
            "source": {"type": "csv", "path": str(source_path)},
            "interval_seconds": 60,
        },
    )
    assert created.status_code == 201, created.text
    schedule = created.json()
    source_path.unlink()

    connection = open_database()
    try:
        connection.execute(
            "UPDATE monitoring_schedules SET next_run_at = ? WHERE id = ?",
            (time.time() - 1, schedule["id"]),
        )
        connection.commit()
    finally:
        connection.close()

    assert run_due_schedules() == 1
    failed = next(
        item for item in client.get("/api/schedules").json()["schedules"]
        if item["id"] == schedule["id"]
    )
    versions = client.get(
        f"/api/datasets/{schedule['dataset_id']}/versions"
    ).json()["versions"]
    assert failed["last_status"] == "failed"
    assert failed["retry_count"] == 1
    assert "source refresh failed" in failed["last_error"]
    assert len(versions) == 1

    source_path.write_text("value\n2\n", encoding="utf-8")
    connection = open_database()
    try:
        connection.execute(
            "UPDATE monitoring_schedules SET next_run_at = ? WHERE id = ?",
            (time.time() - 1, schedule["id"]),
        )
        connection.commit()
    finally:
        connection.close()

    assert run_due_schedules() == 1
    recovered = next(
        item for item in client.get("/api/schedules").json()["schedules"]
        if item["id"] == schedule["id"]
    )
    versions = client.get(
        f"/api/datasets/{schedule['dataset_id']}/versions"
    ).json()["versions"]
    assert recovered["last_status"] == "succeeded"
    assert recovered["retry_count"] == 0
    assert len(versions) == 2


def test_scheduled_http_source_uses_environment_credentials_without_persisting_them(monkeypatch):
    import json

    calls = []

    class FakeResponse:
        status_code = 200
        is_redirect = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield json.dumps({"data": [{"value": 1}]}).encode()

        def close(self):
            return None

    def fake_get(url, **kwargs):
        calls.append(kwargs["headers"].get("Authorization"))
        return FakeResponse()

    secret = "test-scheduled-api-secret"
    monkeypatch.setenv("DATA_SOURCE_HTTP_ALLOWED_HOSTS", "api.example.test")
    monkeypatch.setenv("SCHEDULED_API_TOKEN", secret)
    monkeypatch.setattr("data_sources.requests.get", fake_get)

    response = client.post(
        "/api/schedules",
        json={
            "source": {
                "type": "http_json",
                "url": "https://api.example.test/events",
                "token_env": "SCHEDULED_API_TOKEN",
            },
            "interval_seconds": 60,
        },
    )

    assert response.status_code == 201, response.text
    assert response.json()["schedule"]["source"]["token_env"] == "SCHEDULED_API_TOKEN"
    assert secret not in response.text
    assert calls == [f"Bearer {secret}"]
    schedule_id = response.json()["id"]
    connection = open_database()
    try:
        connection.execute(
            "UPDATE monitoring_schedules SET next_run_at = ? WHERE id = ?",
            (time.time() - 1, schedule_id),
        )
        connection.commit()
    finally:
        connection.close()

    assert run_due_schedules() == 1
    assert calls == [f"Bearer {secret}", f"Bearer {secret}"]


def test_source_schedule_rejects_inline_credentials(tmp_path, monkeypatch):
    response = client.post(
        "/api/schedules",
        json={
            "source": {
                "type": "http_json",
                "url": "https://api.example.test/events?api_key=private",
                "token": "private",
            },
            "interval_seconds": 60,
        },
    )

    assert response.status_code == 400
    assert "credentials" in response.json()["detail"]


def test_schedule_validation_and_owner_scoping():
    dataset_response = client.post(
        "/api/datasets",
        json={"name": "events", "data": [{"value": 1}]},
    )
    dataset_id = dataset_response.json()["id"]

    missing_column = client.post(
        "/api/schedules",
        json={
            "dataset_id": dataset_id,
            "timestamp_column": "missing",
            "max_age_hours": 1,
        },
    )
    assert missing_column.status_code == 400

    too_frequent = client.post(
        "/api/schedules",
        json={"dataset_id": dataset_id, "interval_seconds": 30},
    )
    assert too_frequent.status_code == 400

    created = client.post(
        "/api/schedules",
        json={"dataset_id": dataset_id, "interval_seconds": 60},
    )
    assert created.status_code == 201
    schedule_id = created.json()["id"]
    paused = client.patch(f"/api/schedules/{schedule_id}", json={"enabled": False})
    assert paused.status_code == 200
    interval_update = client.patch(
        f"/api/schedules/{schedule_id}",
        json={"interval_seconds": 120},
    )
    assert interval_update.status_code == 200
    assert interval_update.json()["schedule"]["interval_seconds"] == 120
    invalid_interval = client.patch(
        f"/api/schedules/{schedule_id}",
        json={"interval_seconds": 30},
    )
    assert invalid_interval.status_code == 400
    schedule = next(
        item for item in client.get("/api/schedules").json()["schedules"]
        if item["id"] == schedule_id
    )
    assert schedule["enabled"] is False
    assert schedule["last_status"] == "paused"
    assert client.delete(f"/api/schedules/{schedule_id}").json()["deleted"] is True


def test_viewer_cannot_edit_or_delete_existing_schedules():
    from auth_security import hash_password, issue_token
    from monitoring_store import open_database

    client = TestClient(app, headers=admin_headers())
    dataset = client.post(
        "/api/datasets",
        json={"name": "viewer-schedule", "data": [{"value": 1}]},
    ).json()
    schedule = client.post(
        "/api/schedules",
        json={"dataset_id": dataset["id"], "interval_seconds": 60},
    ).json()
    connection = open_database()
    try:
        connection.execute(
            "INSERT INTO users (id, username, password_hash, role, is_active, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (
                "schedule-viewer-user",
                "schedule-viewer",
                hash_password("viewer-password-123"),
                "viewer",
                time.time(),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    viewer_token = issue_token({"id": "schedule-viewer-user"}, lifetime_seconds=3600)
    viewer = TestClient(
        app,
        headers={"Authorization": f"Bearer {viewer_token}"},
    )

    assert viewer.patch(
        f"/api/schedules/{schedule['id']}",
        json={"enabled": False},
    ).status_code == 403
    assert viewer.delete(f"/api/schedules/{schedule['id']}").status_code == 403


def test_analyst_can_only_schedule_and_manage_owned_datasets():
    analyst = create_user(
        "schedule-analyst",
        "unused-password-hash",
        "analyst",
    )
    # Use the application password hasher so this test exercises the regular login route.
    from auth_security import hash_password

    connection = open_database()
    try:
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password("analyst-password-123"), analyst["id"]),
        )
        connection.commit()
    finally:
        connection.close()
    login = client.post(
        "/api/auth/login",
        json={"username": "schedule-analyst", "password": "analyst-password-123"},
    )
    analyst_client = TestClient(
        app,
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )

    admin_dataset = client.post(
        "/api/datasets",
        json={"name": "admin-data", "data": [{"value": 1}]},
    )
    rejected = analyst_client.post(
        "/api/schedules",
        json={"dataset_id": admin_dataset.json()["id"]},
    )
    assert rejected.status_code == 404

    analyst_dataset = analyst_client.post(
        "/api/datasets",
        json={"name": "analyst-data", "data": [{"value": 1}]},
    )
    own_schedule = analyst_client.post(
        "/api/schedules",
        json={"dataset_id": analyst_dataset.json()["id"], "interval_seconds": 60},
    )
    assert own_schedule.status_code == 201
    schedule_id = own_schedule.json()["id"]
    assert analyst_client.patch(
        f"/api/schedules/{schedule_id}", json={"enabled": False}
    ).status_code == 200
    assert analyst_client.delete(f"/api/schedules/{schedule_id}").status_code == 200


def test_scheduler_marks_failed_runs_and_records_error(monkeypatch):
    dataset_response = client.post(
        "/api/datasets",
        json={"name": "events", "data": [{"value": 1}]},
    )
    schedule_response = client.post(
        "/api/schedules",
        json={"dataset_id": dataset_response.json()["id"], "interval_seconds": 60},
    )
    schedule_id = schedule_response.json()["id"]

    connection = open_database()
    try:
        connection.execute(
            "UPDATE monitoring_schedules SET next_run_at = ? WHERE id = ?",
            (time.time() - 1, schedule_id),
        )
        connection.commit()
    finally:
        connection.close()
    def fail_run(*_args, **_kwargs):
        raise ValueError(
            f"bad scheduled input ({len(_args)} positional, {len(_kwargs)} keyword)"
        )

    monkeypatch.setattr("scheduled_monitoring.run_monitoring", fail_run)

    assert run_due_schedules() == 1
    schedule = next(
        item for item in client.get("/api/schedules").json()["schedules"]
        if item["id"] == schedule_id
    )
    assert schedule["last_status"] == "failed"
    assert "bad scheduled input" in schedule["last_error"]
