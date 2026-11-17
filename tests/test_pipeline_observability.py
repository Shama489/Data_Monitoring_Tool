from fastapi.testclient import TestClient

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
