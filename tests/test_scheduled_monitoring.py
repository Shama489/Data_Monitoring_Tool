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
    schedule = next(
        item for item in client.get("/api/schedules").json()["schedules"]
        if item["id"] == schedule_id
    )
    assert schedule["enabled"] is False
    assert schedule["last_status"] == "paused"
    assert client.delete(f"/api/schedules/{schedule_id}").json()["deleted"] is True


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
