import sqlite3
import time

from fastapi.testclient import TestClient

from conftest import admin_headers
from main import app
from monitoring_store import database_path, open_database
from scheduled_monitoring import run_due_schedules


client = TestClient(app, headers=admin_headers())


def _create_monitoring_result():
    dataset = client.post(
        "/api/datasets",
        json={
            "name": "report-sample",
            "data": [
                {"value": 1, "label": "a"},
                {"value": None, "label": "b"},
                {"value": 100, "label": "c"},
                {"value": 2, "label": "d"},
            ],
        },
    )
    assert dataset.status_code == 201
    run = client.post(
        "/api/monitoring/analyze",
        json={
            "dataset_id": dataset.json()["id"],
            "checks": ["quality", "anomalies"],
        },
    )
    assert run.status_code == 200
    return dataset.json(), run.json()


def test_monitoring_reports_generate_download_and_remain_encrypted():
    _, run = _create_monitoring_result()
    created = client.post(
        "/api/reports",
        json={
            "result_id": run["result_id"],
            "formats": ["pdf", "xlsx", "pptx"],
        },
    )

    assert created.status_code == 201
    report = created.json()
    assert set(report["formats"]) == {"pdf", "xlsx", "pptx"}
    signatures = {
        "pdf": b"%PDF",
        "xlsx": b"PK",
        "pptx": b"PK",
    }
    for report_format, signature in signatures.items():
        download = client.get(
            f"/api/reports/{report['id']}/{report_format}"
        )
        assert download.status_code == 200
        assert download.content.startswith(signature)

    listed = client.get("/api/reports").json()["reports"]
    assert listed[0]["id"] == report["id"]

    connection = sqlite3.connect(database_path())
    try:
        stored_content = connection.execute(
            "SELECT content FROM monitoring_report_files WHERE report_id = ? LIMIT 1",
            (report["id"],),
        ).fetchone()[0]
    finally:
        connection.close()
    assert "%PDF" not in stored_content
    assert "UEsDB" not in stored_content


def test_report_generation_rejects_invalid_formats_and_foreign_results():
    _, run = _create_monitoring_result()
    invalid = client.post(
        "/api/reports",
        json={"result_id": run["result_id"], "formats": ["html"]},
    )
    assert invalid.status_code == 400
    missing = client.post(
        "/api/reports",
        json={"result_id": "missing-result", "formats": ["pdf"]},
    )
    assert missing.status_code == 404


def test_scheduled_monitoring_generates_requested_report_formats():
    dataset = client.post(
        "/api/datasets",
        json={"name": "scheduled-report", "data": [{"value": 1}, {"value": 4}]},
    ).json()
    schedule = client.post(
        "/api/schedules",
        json={
            "dataset_id": dataset["id"],
            "interval_seconds": 60,
            "checks": ["quality"],
            "report_formats": ["pdf", "xlsx"],
        },
    )
    assert schedule.status_code == 201
    schedule_id = schedule.json()["id"]

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
    reports = client.get("/api/reports").json()["reports"]
    scheduled_report = next(
        item for item in reports if item["schedule_id"] == schedule_id
    )
    assert set(scheduled_report["formats"]) == {"pdf", "xlsx"}


def test_dashboard_preferences_health_and_metrics_endpoints():
    saved = client.post(
        "/api/preferences",
        json={
            "preferences": {
                "language": "fr",
                "theme": "high_contrast",
                "widgets": ["quality_score", "anomalies"],
            }
        },
    )
    assert saved.status_code == 200
    assert client.get("/api/preferences").json()["preferences"] == saved.json()["preferences"]

    assert client.get("/api/health/ready").json() == {
        "status": "ready",
        "database": "ok",
    }
    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["requests_total"] >= 1
    assert metrics.json()["request_duration_seconds_total"] >= 0


def test_api_rate_limit_returns_retry_after(monkeypatch):
    monkeypatch.setenv("API_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("API_RATE_LIMIT_WINDOW_SECONDS", "60")

    first = client.get("/api/auth/me")
    second = client.get("/api/auth/me")

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) > 0


def test_failed_schedules_record_retry_count_and_backoff(monkeypatch):
    dataset = client.post(
        "/api/datasets",
        json={"name": "retry-sample", "data": [{"value": 1}]},
    ).json()
    schedule = client.post(
        "/api/schedules",
        json={"dataset_id": dataset["id"], "interval_seconds": 60},
    ).json()
    connection = open_database()
    try:
        connection.execute(
            "UPDATE monitoring_schedules SET next_run_at = ? WHERE id = ?",
            (time.time() - 1, schedule["id"]),
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(
        "scheduled_monitoring.run_monitoring",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("temporary failure")),
    )
    assert run_due_schedules() == 1
    failed = next(
        item for item in client.get("/api/schedules").json()["schedules"]
        if item["id"] == schedule["id"]
    )
    assert failed["last_status"] == "failed"
    assert failed["retry_count"] == 1
    assert failed["next_run_at"] > time.time()
