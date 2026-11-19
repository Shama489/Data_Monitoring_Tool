from datetime import datetime, timezone

from fastapi.testclient import TestClient

import main
from conftest import admin_headers
from main import app
from monitoring_store import create_user
from auth_security import hash_password


client = TestClient(app, headers=admin_headers())


def test_external_pipeline_runs_report_latency_throughput_and_status_updates():
    started = client.post(
        "/api/pipelines/runs",
        json={
            "run_id": "warehouse-2026-11-10-001",
            "pipeline": "warehouse-load",
            "job": "orders",
            "status": "RUNNING",
            "started_at": "2026-11-10T10:00:00Z",
            "details": {"provider": "external-orchestrator"},
        },
    )
    assert started.status_code == 201, started.text
    assert started.json()["duration_seconds"] is None

    finished = client.post(
        "/api/pipelines/runs",
        json={
            "run_id": "warehouse-2026-11-10-001",
            "pipeline": "warehouse-load",
            "job": "orders",
            "status": "COMPLETED_WITH_WARNINGS",
            "finished_at": "2026-11-10T10:00:10Z",
            "records_processed": 25,
        },
    )
    assert finished.status_code == 201, finished.text
    run = finished.json()
    assert run["id"] == started.json()["id"]
    assert run["status"] == "COMPLETED_WITH_WARNINGS"
    assert run["duration_seconds"] == 10
    assert run["records_processed"] == 25
    assert run["throughput_records_per_second"] == 2.5
    assert run["details"] == {"provider": "external-orchestrator"}

    response = client.get(
        "/api/pipelines/runs",
        params={"pipeline": "warehouse-load", "status": "COMPLETED_WITH_WARNINGS"},
    )
    assert response.status_code == 200
    body = response.json()
    assert [item["run_id"] for item in body["runs"]] == [
        "warehouse-2026-11-10-001"
    ]
    assert body["summary"] == {
        "runs": 1,
        "failed_runs": 0,
        "average_latency_seconds": 10,
        "records_processed": 25,
        "throughput_records_per_second": 2.5,
    }


def test_pipeline_run_payload_validation_rejects_invalid_metrics():
    response = client.post(
        "/api/pipelines/runs",
        json={
            "run_id": "invalid-timing",
            "pipeline": "extract",
            "job": "orders",
            "status": "failed",
            "started_at": 200,
            "finished_at": 100,
            "records_processed": -1,
        },
    )
    assert response.status_code == 400
    assert "finished_at" in response.json()["detail"]


def test_pipeline_runs_are_scoped_for_non_admin_users():
    create_user(
        "pipeline-analyst",
        hash_password("pipeline-analyst-password"),
        "analyst",
    )
    token = client.post(
        "/api/auth/login",
        json={
            "username": "pipeline-analyst",
            "password": "pipeline-analyst-password",
        },
    ).json()["access_token"]
    analyst_client = TestClient(
        app,
        headers={"Authorization": f"Bearer {token}"},
    )
    recorded = analyst_client.post(
        "/api/pipelines/runs",
        json={
            "run_id": "private-run",
            "pipeline": "private-pipeline",
            "job": "private-job",
            "status": "queued",
        },
    )
    assert recorded.status_code == 201
    assert analyst_client.get("/api/pipelines/runs").json()["summary"]["runs"] == 1
    assert client.get(
        "/api/pipelines/runs",
        params={"pipeline": "private-pipeline"},
    ).json()["summary"]["runs"] == 1


def test_pipeline_summary_ignores_runs_missing_records_for_throughput():
    recorded = client.post(
        "/api/pipelines/runs",
        json={
            "run_id": "known-volume",
            "pipeline": "summary-check",
            "job": "orders",
            "status": "COMPLETED_WITH_WARNINGS",
            "started_at": 100,
            "finished_at": 110,
            "records_processed": 20,
        },
    )
    assert recorded.status_code == 201
    recorded_without_count = client.post(
        "/api/pipelines/runs",
        json={
            "run_id": "unknown-volume",
            "pipeline": "summary-check",
            "job": "customers",
            "status": "SUCCEEDED",
            "started_at": 100,
            "finished_at": 115,
        },
    )
    assert recorded_without_count.status_code == 201
    recorded_timeout = client.post(
        "/api/pipelines/runs",
        json={
            "run_id": "timed-out",
            "pipeline": "summary-check",
            "job": "invoices",
            "status": "TIMED_OUT",
        },
    )
    assert recorded_timeout.status_code == 201

    response = client.get(
        "/api/pipelines/runs",
        params={"pipeline": "summary-check"},
    )
    assert response.status_code == 200
    assert response.json()["summary"] == {
        "runs": 3,
        "failed_runs": 1,
        "average_latency_seconds": 12.5,
        "records_processed": 20,
        "throughput_records_per_second": 2,
    }


def test_pipeline_failure_forecast_uses_owner_scoped_weekly_history(monkeypatch):
    pipeline = "forecast-orders"
    statuses = ["SUCCEEDED", "FAILED", "COMPLETED", "TIMED_OUT"]
    for week, status in enumerate(statuses):
        started = datetime(2026, 1, 5 + week * 7, tzinfo=timezone.utc)
        recorded = client.post(
            "/api/pipelines/runs",
            json={
                "run_id": f"forecast-run-{week}",
                "pipeline": pipeline,
                "job": "orders",
                "status": status,
                "started_at": started.isoformat(),
                "finished_at": datetime.fromtimestamp(
                    started.timestamp() + 10, tz=timezone.utc
                ).isoformat(),
                "records_processed": 100,
            },
        )
        assert recorded.status_code == 201, recorded.text

    client.post(
        "/api/pipelines/runs",
        json={
            "run_id": "forecast-unclassified",
            "pipeline": pipeline,
            "job": "orders",
            "status": "RUNNING",
            "started_at": "2026-02-05T00:00:00Z",
        },
    )

    captured = {}

    def fake_forecast(frame, date_column, value_column, **kwargs):
        captured["frame"] = frame
        captured["date_column"] = date_column
        captured["value_column"] = value_column
        captured["kwargs"] = kwargs
        return {
            "metric": "failed_runs",
            "history": [{"date": "2026-02-01T00:00:00", "value": 1}],
            "forecast": [{"date": "2026-02-08T00:00:00", "value": 1.5}],
        }

    monkeypatch.setattr(main, "forecast_metric", fake_forecast)
    response = client.get(
        "/api/pipelines/failure-forecast",
        params={"pipeline": pipeline, "periods": 1},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["forecast_unit"] == "failed_runs_per_week"
    assert body["history_runs_analyzed"] == 4
    assert body["unclassified_runs_ignored"] == 1
    assert [item["failure_rate"] for item in body["weekly_history"]] == [
        0,
        1,
        0,
        1,
    ]
    assert body["forecast"]["forecast"][0]["value"] == 1.5
    assert captured["date_column"] == "date"
    assert captured["value_column"] == "failed_runs"
    assert captured["kwargs"] == {
        "periods": 1,
        "frequency": "W",
        "method": "linear",
    }


def test_pipeline_failure_forecast_returns_trend_from_three_weeks_of_history():
    for week, status in enumerate(["SUCCEEDED", "FAILED", "FAILED"]):
        recorded = client.post(
            "/api/pipelines/runs",
            json={
                "run_id": f"trend-history-{week}",
                "pipeline": "trend-history",
                "job": "orders",
                "status": status,
                "started_at": datetime(
                    2026, 2, 2 + week * 7, tzinfo=timezone.utc
                ).isoformat(),
            },
        )
        assert recorded.status_code == 201, recorded.text

    response = client.get(
        "/api/pipelines/failure-forecast",
        params={"pipeline": "trend-history", "periods": 2},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["weekly_history"]) == 3
    assert [item["failure_rate"] for item in body["weekly_history"]] == [
        0,
        1,
        1,
    ]
    assert body["forecast"]["method"] == "linear"
    assert len(body["forecast"]["forecast"]) == 2
    assert all(
        item["value"] >= 0
        for item in body["forecast"]["forecast"]
    )


def test_pipeline_failure_forecast_requires_three_weeks_of_completed_history():
    for week, status in enumerate(["SUCCEEDED", "FAILED"]):
        recorded = client.post(
            "/api/pipelines/runs",
            json={
                "run_id": f"short-history-{week}",
                "pipeline": "short-history",
                "job": "orders",
                "status": status,
                "started_at": datetime(
                    2026, 1, 5 + week * 7, tzinfo=timezone.utc
                ).isoformat(),
            },
        )
        assert recorded.status_code == 201, recorded.text

    response = client.get(
        "/api/pipelines/failure-forecast",
        params={"pipeline": "short-history"},
    )

    assert response.status_code == 422
    assert "At least three time periods" in response.json()["detail"]


def test_pipeline_failure_forecast_does_not_expose_another_owners_runs():
    for week, status in enumerate(["SUCCEEDED", "FAILED", "SUCCEEDED"]):
        recorded = client.post(
            "/api/pipelines/runs",
            json={
                "run_id": f"private-forecast-{week}",
                "pipeline": "private-forecast",
                "job": "orders",
                "status": status,
                "started_at": datetime(
                    2026, 3, 2 + week * 7, tzinfo=timezone.utc
                ).isoformat(),
            },
        )
        assert recorded.status_code == 201, recorded.text

    create_user(
        "forecast-analyst",
        hash_password("forecast-analyst-password"),
        "analyst",
    )
    token = client.post(
        "/api/auth/login",
        json={
            "username": "forecast-analyst",
            "password": "forecast-analyst-password",
        },
    ).json()["access_token"]
    analyst_client = TestClient(
        app,
        headers={"Authorization": f"Bearer {token}"},
    )

    response = analyst_client.get(
        "/api/pipelines/failure-forecast",
        params={"pipeline": "private-forecast"},
    )

    assert response.status_code == 422
    assert "No completed runs" in response.json()["detail"]
