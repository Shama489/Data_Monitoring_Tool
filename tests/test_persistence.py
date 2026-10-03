import sqlite3

import pandas as pd
from fastapi.testclient import TestClient

import main
from main import app
from notifications import ALERT_HISTORY, send_notifications
from profiler import check_data_quality
import monitoring_store
from agents.monitoring_agent import run_monitoring


client = TestClient(app)


def test_datasets_and_monitoring_results_are_persisted_and_reusable():
    created = client.post(
        "/api/datasets",
        json={"name": "customers", "data": [{"age": 20}, {"age": None}]},
    )

    assert created.status_code == 201
    dataset_id = created.json()["id"]
    assert client.get(f"/api/datasets/{dataset_id}").json()["data"] == [
        {"age": 20.0},
        {"age": None},
    ]

    run = client.post(
        "/api/monitoring/analyze",
        json={"dataset_id": dataset_id, "checks": ["quality"]},
    )

    assert run.status_code == 200
    result_id = run.json()["result_id"]
    assert run.json()["dataset_id"] == dataset_id
    assert client.get(f"/api/monitoring/results/{result_id}").json()["result"]["checks"] == [
        "quality"
    ]
    assert client.get(
        "/api/monitoring/results", params={"dataset_id": dataset_id}
    ).json()["results"][0]["id"] == result_id


def test_analytics_reports_are_saved_as_monitoring_results():
    dates = pd.date_range("2025-01-01", periods=8, freq="W")
    response = client.post(
        "/api/analytics/trends",
        json={
            "date_column": "date",
            "data": [
                {"date": date.isoformat(), "value": index}
                for index, date in enumerate(dates)
            ],
        },
    )

    assert response.status_code == 200
    stored = client.get(
        f"/api/monitoring/results/{response.json()['result_id']}"
    ).json()
    assert stored["check_type"] == "trends"
    assert stored["dataset_id"] == response.json()["dataset_id"]


def test_alert_rules_configurations_and_audit_records_are_persisted():
    rules = [{"metric": "quality_score", "operator": "<", "value": 70}]
    response = client.post("/api/alerts/rules", json={"rules": rules})

    assert response.status_code == 200
    restored_rules = client.get("/api/alerts/rules").json()["rules"]
    assert len(restored_rules) == 1
    assert {
        key: restored_rules[0][key] for key in ("metric", "operator", "value")
    } == rules[0]

    config = client.put(
        "/api/configurations",
        json={"configurations": {"retention_days": 90, "notifications_enabled": True}},
    )
    assert config.status_code == 200
    assert client.get("/api/configurations").json()["configurations"] == {
        "retention_days": 90,
        "notifications_enabled": True,
    }

    audit = client.get("/api/audit").json()["records"]
    assert {record["entity_type"] for record in audit} >= {
        "alert_rules",
        "configuration",
    }


def test_alert_rule_management_rejects_unknown_operators():
    response = client.post(
        "/api/alerts/rules",
        json={"metric": "quality_score", "operator": "approximately", "value": 70},
    )

    assert response.status_code == 400
    assert "unsupported alert rule operator" in response.json()["detail"]


def test_alert_cooldown_uses_persisted_history_after_process_state_is_cleared():
    first = send_notifications({
        "message": "persistent cooldown",
        "dry_run": True,
        "channels": ["email"],
    })
    ALERT_HISTORY.clear()
    second = send_notifications({
        "message": "persistent cooldown",
        "dry_run": True,
        "channels": ["email"],
    })

    assert first["success"] is True
    assert second["status"] == "cooldown"
    assert client.get("/api/audit").json()["records"][0]["entity_type"] == "alert"


def test_rule_notifications_keep_the_highest_severity(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        main,
        "send_notifications",
        lambda alert: seen.update(alert) or {"status": "sent"},
    )

    result = main._send_rule_based_notifications(
        {
            "alert_rules": [{
                "metric": "drift_score",
                "operator": ">=",
                "value": 80,
                "channel": "email",
            }]
        },
        "data_drift",
        {"drift_score": 95, "severity": "critical"},
        "Drift alert",
    )

    assert result["status"] == "sent"
    assert seen["severity"] == "critical"


def test_monitoring_workflow_does_not_downgrade_critical_drift(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        "agents.monitoring_agent.investigate",
        lambda _results: {
            "findings": [{
                "signal": "distribution_drift",
                "evidence": {"overall_severity": "critical"},
            }]
        },
    )
    monkeypatch.setattr(
        "agents.monitoring_agent.send_monitoring_notification",
        lambda message, channels, severity="warning": (
            sent.update({"severity": severity}) or {"status": "sent"}
        ),
    )

    result = run_monitoring(
        {"data": [{"value": 1}], "notify": True, "channels": ["email"]},
        quality_options={},
    )

    assert result["notifications"]["status"] == "sent"
    assert sent["severity"] == "critical"


def test_timezone_aware_freshness_check_uses_matching_timezone():
    frame = pd.DataFrame({
        "seen_at": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)]
    })

    freshness = check_data_quality(
        frame,
        timestamp_column="seen_at",
        max_age_hours=2,
    )["data_freshness"]

    assert freshness["is_fresh"] is True
    assert 0.9 <= freshness["age_hours"] <= 1.1


def test_existing_alert_database_is_migrated_on_first_use(tmp_path, monkeypatch):
    legacy_path = tmp_path / "alerts.db"
    legacy = sqlite3.connect(legacy_path)
    legacy.execute(
        "CREATE TABLE alerts (id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, "
        "status TEXT NOT NULL, message TEXT NOT NULL, subject TEXT NOT NULL, "
        "channels TEXT NOT NULL, dedupe_key TEXT NOT NULL, sent TEXT, failed TEXT, "
        "event_type TEXT, severity TEXT)"
    )
    legacy.execute(
        "INSERT INTO alerts VALUES (1, 1, 'success', 'legacy', 'subject', '[]', "
        "'[]', '[]', '[]', 'test', 'low')"
    )
    legacy.commit()
    legacy.close()

    monkeypatch.setattr(monitoring_store, "__file__", str(tmp_path / "monitoring_store.py"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.delenv("MONITORING_DB_PATH")
    connection = monitoring_store.open_database()
    try:
        migrated = connection.execute(
            "SELECT message FROM alerts WHERE id = 1"
        ).fetchone()
    finally:
        connection.close()

    assert migrated["message"] == "legacy"
