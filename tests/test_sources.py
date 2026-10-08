import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data_sources import DataSourceError, load_local_file, load_sql_query
from main import app
from notifications import ALERT_HISTORY, reset_alert_state
from conftest import admin_headers


client = TestClient(app, headers=admin_headers())


def test_local_file_loader_supports_csv_and_json(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ALLOWED_DIRS", str(tmp_path))
    frame = pd.DataFrame({"id": [1, 2], "status": ["ok", "warn"]})
    csv_path = tmp_path / "events.csv"
    json_path = tmp_path / "events.json"
    txt_path = tmp_path / "events.txt"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(frame.to_dict(orient="records")), encoding="utf-8")
    frame.to_csv(txt_path, index=False, sep="\t")

    pd.testing.assert_frame_equal(load_local_file(str(csv_path)), frame)
    pd.testing.assert_frame_equal(load_local_file(str(json_path)), frame)
    pd.testing.assert_frame_equal(load_local_file(str(txt_path)), frame)


def test_source_capabilities_report_provider_setup():
    response = client.get("/api/sources/capabilities")

    assert response.status_code == 200
    providers = {provider["type"]: provider for provider in response.json()["providers"]}
    assert "txt" in providers["local_files"]["formats"]
    assert providers["dropbox"]["credential_env"] == ["DROPBOX_ACCESS_TOKEN"]
    assert "missing_dependencies" in providers["s3"]


def test_sql_loader_rejects_non_select_queries():
    with pytest.raises(DataSourceError, match="Only SELECT queries"):
        load_sql_query("sqlite://", "DELETE FROM events")


def test_sql_loader_rejects_multi_statement_select_queries():
    with pytest.raises(DataSourceError, match="Only SELECT queries"):
        load_sql_query("sqlite://", "SELECT 1; DELETE FROM events")


def test_sql_loader_accepts_multiline_select_queries():
    frame = load_sql_query(
        "sqlite://",
        "SELECT\n  1 AS value\nUNION ALL\nSELECT\n  2 AS value",
    )
    pd.testing.assert_frame_equal(frame, pd.DataFrame({"value": [1, 2]}))


def test_sql_loader_accepts_commented_select_queries_with_embedded_semicolons():
    frame = load_sql_query(
        "sqlite://",
        "-- leading comment\nSELECT 'alpha;beta' AS value UNION ALL SELECT 'gamma;delta' AS value",
    )
    pd.testing.assert_frame_equal(
        frame,
        pd.DataFrame({"value": ["alpha;beta", "gamma;delta"]}),
    )


def test_sources_endpoint_reports_success_and_partial_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ALLOWED_DIRS", str(tmp_path))
    path = tmp_path / "events.csv"
    pd.DataFrame({"id": [1], "status": ["ok"]}).to_csv(path, index=False)

    response = client.post(
        "/api/sources/analyze",
        json={
            "sources": [
                {"type": "csv", "path": str(path)},
                {"type": "unsupported", "path": "unused"},
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["sources"][0]["rows"] == 1
    assert body["failed"][0]["type"] == "unsupported"


def test_sources_endpoint_restricts_local_paths_and_file_types(tmp_path, monkeypatch):
    allowed_dir = tmp_path / "allowed"
    outside_dir = tmp_path / "outside"
    allowed_dir.mkdir()
    outside_dir.mkdir()
    allowed_csv = allowed_dir / "events.csv"
    outside_csv = outside_dir / "private.csv"
    unsupported_file = allowed_dir / "private.py"
    allowed_csv.write_text("value\n1\n", encoding="utf-8")
    outside_csv.write_text("secret\nprivate-data\n", encoding="utf-8")
    unsupported_file.write_text("private-data", encoding="utf-8")
    monkeypatch.setenv("DATASET_ALLOWED_DIRS", str(allowed_dir))

    valid = client.post(
        "/api/sources/analyze",
        json={"sources": [{"type": "csv", "path": str(allowed_csv)}]},
    )
    assert valid.status_code == 200
    assert valid.json()["sources"][0]["rows"] == 1

    traversal = client.post(
        "/api/sources/analyze",
        json={
            "sources": [{
                "type": "csv",
                "path": str(allowed_dir / ".." / "outside" / "private.csv"),
            }]
        },
    )
    assert traversal.status_code == 400

    absolute_outside = client.post(
        "/api/sources/analyze",
        json={"sources": [{"type": "csv", "path": str(outside_csv)}]},
    )
    assert absolute_outside.status_code == 400

    wrong_extension = client.post(
        "/api/sources/analyze",
        json={"sources": [{"type": "csv", "path": str(unsupported_file)}]},
    )
    assert wrong_extension.status_code == 400

    mismatched_type = client.post(
        "/api/sources/analyze",
        json={
            "sources": [{
                "type": "json",
                "file_type": "json",
                "path": str(allowed_csv),
            }]
        },
    )
    assert mismatched_type.status_code == 400


def test_sources_endpoint_rejects_symlink_escape(tmp_path, monkeypatch):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_file = tmp_path / "private.csv"
    outside_file.write_text("secret\nprivate-data\n", encoding="utf-8")
    link = allowed_dir / "linked.csv"
    try:
        link.symlink_to(outside_file)
    except OSError as error:
        pytest.skip(f"Symlinks are unavailable in this test environment: {error}")
    monkeypatch.setenv("DATASET_ALLOWED_DIRS", str(allowed_dir))

    response = client.post(
        "/api/sources/analyze",
        json={"sources": [{"type": "csv", "path": str(link)}]},
    )

    assert response.status_code == 400


def test_sources_endpoint_rejects_empty_source_list():
    response = client.post("/api/sources/analyze", json={"sources": []})

    assert response.status_code == 400
    assert response.json()["detail"] == "sources must be a non-empty list"


def test_monitoring_agent_runs_quality_by_default_and_returns_evidence():
    response = client.post(
        "/api/monitoring/analyze",
        json={"data": [{"age": 20}, {"age": None}, {"age": 20}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["checks"] == ["quality"]
    assert [item["tool"] for item in body["tool_trace"]] == ["data.load", "quality.analyze"]
    assert body["root_cause"]["status"] == "attention_required"
    assert {finding["signal"] for finding in body["root_cause"]["findings"]} >= {
        "missing_values",
        "duplicate_rows",
    }
    assert "drift" not in body["results"]


def test_monitoring_agent_selects_drift_when_baseline_is_present():
    response = client.post(
        "/api/monitoring/analyze",
        json={
            "baseline": [{"value": 0}, {"value": 0}, {"value": 1}, {"value": 1}],
            "current": [{"value": 10}, {"value": 10}, {"value": 11}, {"value": 11}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["checks"] == ["quality", "drift"]
    assert "drift.analyze" in [item["tool"] for item in body["tool_trace"]]
    assert body["results"]["drift"]["drift_detected"] is True
    drift_finding = next(
        finding for finding in body["root_cause"]["findings"]
        if finding["signal"] == "distribution_drift"
    )
    assert drift_finding["evidence"]["overall_drift_score"] == body["results"]["drift"]["overall_drift_score"]


def test_monitoring_agent_rejects_unknown_checks():
    response = client.post(
        "/api/monitoring/analyze",
        json={"data": [{"value": 1}], "checks": ["unknown"]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "checks may contain only quality, drift, anomalies, forecast, and explain"
    )


def test_monitoring_agent_runs_selected_forecast_tool():
    dates = pd.date_range("2025-01-01", periods=8, freq="W")
    response = client.post(
        "/api/monitoring/analyze",
        json={
            "data": [{"event_date": date.isoformat(), "amount": index} for index, date in enumerate(dates)],
            "checks": ["forecast"],
            "date_column": "event_date",
            "value_column": "amount",
            "periods": 2,
            "method": "linear",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["checks"] == ["forecast"]
    assert [item["tool"] for item in body["tool_trace"]] == ["data.load", "forecast.run"]
    assert len(body["results"]["forecast"]["forecast"]) == 2


def test_monitoring_agent_runs_anomaly_check_and_persists_result():
    response = client.post(
        "/api/monitoring/analyze",
        json={
            "data": [{"value": index if index < 19 else 1000} for index in range(20)],
            "checks": ["anomalies"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    anomaly_report = body["results"]["anomalies"]
    assert anomaly_report["available"] is True
    assert anomaly_report["total_anomalies"] >= 1
    assert "anomalies.analyze" in [
        item["tool"] for item in body["tool_trace"]
    ]
    stored = client.get(
        f"/api/monitoring/results/{body['result_id']}"
    ).json()
    assert stored["result"]["results"]["anomalies"] == anomaly_report


def test_monitoring_agent_loads_source_and_sends_requested_notification(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASET_ALLOWED_DIRS", str(tmp_path))
    source_path = tmp_path / "events.csv"
    pd.DataFrame({"value": [1, None]}).to_csv(source_path, index=False)
    sent_alerts = []
    monkeypatch.setattr(
        "tools.notification_tools.send_notifications",
        lambda alert: sent_alerts.append(alert) or {"status": "sent"},
    )

    response = client.post(
        "/api/monitoring/analyze",
        json={
            "source": {"type": "csv", "path": str(source_path)},
            "notify": True,
            "channels": ["slack"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_trace"][0]["tool"] == "data.load"
    assert body["notifications"]["status"] == "sent"
    assert sent_alerts[0]["event_type"] == "monitoring_workflow"
    assert sent_alerts[0]["channels"] == ["slack"]


def test_quality_endpoint_forwards_optional_llm_flag(monkeypatch):
    calls = []

    def fake_summary(df, report, quality_score, use_llm=False):
        calls.append(use_llm)
        return {
            "summary": "built-in",
            "risk_level": "low",
            "key_findings": [],
            "recommended_actions": [],
        }

    monkeypatch.setattr("main.generate_ai_quality_summary", fake_summary)

    default_response = client.post(
        "/api/data-quality/analyze",
        json={"data": [{"value": 1}]},
    )
    llm_response = client.post(
        "/api/data-quality/analyze",
        json={"data": [{"value": 1}], "use_llm": True},
    )

    assert default_response.status_code == 200
    assert llm_response.status_code == 200
    assert calls == [False, True]


def test_quality_endpoint_rejects_non_boolean_llm_flag():
    response = client.post(
        "/api/data-quality/analyze",
        json={"data": [{"value": 1}], "use_llm": "true"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "use_llm must be a boolean"


def test_quality_endpoint_triggers_notifications_when_requested(monkeypatch):
    seen = {}

    def fake_send_notifications(alert):
        seen["alert"] = alert
        return {"status": "sent", "sent": [{"channel": "email", "status": "dry_run"}]}

    monkeypatch.setattr("main.send_notifications", fake_send_notifications)

    response = client.post(
        "/api/data-quality/analyze",
        json={
            "data": [{"value": 1}, {"value": 50}],
            "notify": True,
            "channels": [{"channel": "email", "recipient": "alerts@example.com"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["notifications"]["status"] == "sent"
    assert seen["alert"]["event_type"] == "data_quality"
    assert "Data quality alert" in seen["alert"]["message"]


def test_drift_endpoint_triggers_notifications_when_requested(monkeypatch):
    seen = {}

    def fake_send_notifications(alert):
        seen["alert"] = alert
        return {"status": "sent", "sent": [{"channel": "slack", "status": "dry_run"}]}

    monkeypatch.setattr("main.send_notifications", fake_send_notifications)

    response = client.post(
        "/api/drift/analyze",
        json={
            "baseline": [{"value": 1}, {"value": 2}, {"value": 3}],
            "current": [{"value": 1}, {"value": 10}, {"value": 11}],
            "notify": True,
            "channels": [{"channel": "slack", "recipient": "https://example.com/webhook"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["notifications"]["status"] == "sent"
    assert seen["alert"]["event_type"] == "data_drift"
    assert "Drift alert" in seen["alert"]["message"]


def test_saved_alert_rules_trigger_quality_notifications(monkeypatch, tmp_path):
    monkeypatch.setattr("notifications.ALERT_DB_PATH", str(tmp_path / "alerts.db"))
    reset_alert_state()
    posted_rules = client.post(
        "/api/alerts/rules",
        json={"rules": [{
            "metric": "quality_score",
            "operator": ">=",
            "value": 0,
            "channel": "email",
            "recipient": "alerts@example.com",
        }]},
    )
    assert posted_rules.status_code == 200

    sent_alerts = []
    monkeypatch.setattr("main.send_notifications", lambda alert: sent_alerts.append(alert) or {"status": "sent"})
    response = client.post("/api/data-quality/analyze", json={"data": [{"value": 1}]})

    assert response.status_code == 200
    assert response.json()["notifications"]["status"] == "sent"
    assert sent_alerts[0]["channels"] == [{"channel": "email", "recipient": "alerts@example.com"}]


def test_alert_history_endpoint_reads_persisted_history(monkeypatch, tmp_path):
    monkeypatch.setattr("notifications.ALERT_DB_PATH", str(tmp_path / "alerts.db"))
    reset_alert_state()
    response = client.post(
        "/api/notifications/send",
        json={"message": "persisted through API", "dry_run": True, "channels": ["email"]},
    )
    assert response.status_code == 200

    ALERT_HISTORY.clear()
    history_response = client.get("/api/alerts/history")

    assert history_response.status_code == 200
    assert any(item["message"] == "persisted through API" for item in history_response.json()["history"])


def test_clearing_alert_rules_preserves_persisted_history(monkeypatch, tmp_path):
    monkeypatch.setattr("notifications.ALERT_DB_PATH", str(tmp_path / "alerts.db"))
    reset_alert_state()
    client.post(
        "/api/notifications/send",
        json={"message": "keep this history", "dry_run": True, "channels": ["email"]},
    )
    client.post(
        "/api/alerts/rules",
        json={"rules": [{"metric": "quality_score", "operator": "<", "value": 50}]},
    )

    response = client.delete("/api/alerts/rules")

    assert response.status_code == 200
    ALERT_HISTORY.clear()
    history_response = client.get("/api/alerts/history")
    assert any(item["message"] == "keep this history" for item in history_response.json()["history"])


def test_explain_api_returns_holdout_metrics_and_accepts_test_size(monkeypatch):
    monkeypatch.setattr("profiler.shap", None)
    response = client.post(
        "/api/analytics/explain",
        json={
            "data": [
                {"signal": value, "target": value % 2}
                for value in range(20)
            ],
            "target_column": "target",
            "task": "classification",
            "test_size": 0.3,
        },
    )

    assert response.status_code == 200
    body = response.json()["report"]
    assert body["test_size"] == 0.3
    assert body["holdout"]["rows"] == 6
    assert "accuracy" in body["holdout"]["metrics"]
