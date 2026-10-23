import json
from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data_sources import (
    DataSourceError,
    load_http_json,
    load_local_file,
    load_source,
    load_uploaded_file,
    load_sql_query,
    source_capabilities,
)
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
    assert providers["http_json"]["aliases"] == ["rest_json"]
    assert providers["http_json"]["credential_env"] == [
        "DATA_SOURCE_HTTP_ALLOWED_HOSTS"
    ]


def test_local_file_source_is_not_reported_ready_without_allowed_roots(monkeypatch):
    monkeypatch.delenv("DATASET_ALLOWED_DIRS", raising=False)

    local_files = {
        provider["type"]: provider
        for provider in source_capabilities()["providers"]
    }["local_files"]

    assert local_files["status"] == "configuration_required"
    assert "DATASET_ALLOWED_DIRS" in local_files["missing_configuration"]


def test_sql_loader_rejects_non_select_queries():
    with pytest.raises(DataSourceError, match="Only SELECT queries"):
        load_sql_query("sqlite://", "DELETE FROM events")


def test_sql_loader_rejects_multi_statement_select_queries():
    with pytest.raises(DataSourceError, match="Only SELECT queries"):
        load_sql_query("sqlite://", "SELECT 1; DELETE FROM events")


@pytest.mark.parametrize(
    "query",
    [
        "WITH changed AS (DELETE FROM events RETURNING *) SELECT * FROM changed",
        "WITH changed AS (UPDATE events SET value = 1 RETURNING *) SELECT * FROM changed",
        "WITH changed AS (INSERT INTO events (value) VALUES (1) RETURNING *) SELECT * FROM changed",
        "WITH changed AS (SELECT 1) DELETE FROM events",
        "WITH changed AS (SELECT 1) UPDATE events SET value = 1",
        "WITH changed AS (SELECT 1) INSERT INTO events (value) SELECT * FROM changed",
    ],
)
def test_sql_loader_rejects_data_modifying_ctes_and_with_statements(query):
    with pytest.raises(DataSourceError, match="Only SELECT queries"):
        load_sql_query("sqlite://", query)


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


def test_sql_loader_rejects_write_attempts_on_read_only_connection(monkeypatch, tmp_path):
    from sqlalchemy import create_engine

    database_path = tmp_path / "readonly.db"
    engine = create_engine(f"sqlite:///{database_path}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE events (value INTEGER)")
    engine.dispose()

    def attempt_write(connection, *_args, **_kwargs):
        connection.exec_driver_sql("INSERT INTO events (value) VALUES (1)")
        return pd.DataFrame()

    monkeypatch.setattr("data_sources._read_sql_bounded", attempt_write)
    with pytest.raises(DataSourceError, match="Database query failed"):
        load_sql_query(f"sqlite:///{database_path}", "SELECT 1")

    engine = create_engine(f"sqlite:///{database_path}")
    with engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT COUNT(*) FROM events").scalar_one() == 0
    engine.dispose()


def test_sql_loader_rejects_results_above_the_configured_row_limit():
    with pytest.raises(DataSourceError, match="must not exceed 2 rows"):
        load_sql_query(
            "sqlite://",
            "SELECT 1 AS value UNION ALL SELECT 2 UNION ALL SELECT 3",
            max_rows=2,
        )


def test_sql_loader_does_not_expose_connection_errors(monkeypatch):
    def fail_engine(*_args, **_kwargs):
        raise RuntimeError("connection failed for db://alice:private-password@host")

    monkeypatch.setattr("data_sources.create_engine", fail_engine)

    with pytest.raises(DataSourceError) as raised:
        load_sql_query("db://alice:private-password@host", "SELECT 1")

    assert "private-password" not in str(raised.value)
    assert str(raised.value) == (
        "Database query failed; check the source configuration and query"
    )


def test_source_loader_enforces_row_limit_for_local_files(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ALLOWED_DIRS", str(tmp_path))
    path = tmp_path / "events.csv"
    path.write_text("value\n1\n2\n3\n", encoding="utf-8")

    with pytest.raises(DataSourceError, match="must not exceed 2 rows"):
        load_source({"type": "csv", "path": str(path)}, max_rows=2)


def test_uploaded_file_loader_rejects_instead_of_truncating_excess_rows():
    with pytest.raises(DataSourceError, match="must not exceed 1 rows"):
        load_uploaded_file(
            b"value\n1\n2\n",
            "csv",
            "events.csv",
            max_rows=1,
        )


def test_http_json_source_uses_environment_token_and_extracts_records(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200
        is_redirect = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size > 0
            yield b'{"payload":{"items":[{"value":1},{"value":2}]}}'

        def close(self):
            return None

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setenv("DATA_SOURCE_HTTP_ALLOWED_HOSTS", "api.example.test")
    monkeypatch.setenv("TEST_API_TOKEN", "test-only-token")
    monkeypatch.setattr("data_sources.requests.get", fake_get)

    frame = load_http_json(
        "https://api.example.test/records",
        token_env="TEST_API_TOKEN",
        records_path="payload.items",
    )

    assert frame.to_dict(orient="records") == [{"value": 1}, {"value": 2}]
    assert calls[0][1]["headers"]["Authorization"] == "Bearer test-only-token"
    assert calls[0][1]["allow_redirects"] is False


def test_http_json_source_rejects_unallowlisted_hosts_and_oversized_results(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE_HTTP_ALLOWED_HOSTS", "trusted.example.test")
    with pytest.raises(DataSourceError, match="not listed"):
        load_http_json("https://other.example.test/data")
    with pytest.raises(DataSourceError, match="not listed"):
        load_http_json("https://trusted.example.test:8443/data")

    class FakeResponse:
        status_code = 200
        is_redirect = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b'{"data":[{"value":1},{"value":2}]}'

        def close(self):
            return None

    monkeypatch.setattr("data_sources.requests.get", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(DataSourceError, match="must not exceed 1 rows"):
        load_http_json(
            "https://trusted.example.test/data",
            max_rows=1,
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


def test_http_source_runs_quality_anomaly_and_drift_monitoring(monkeypatch):
    class FakeResponse:
        status_code = 200
        is_redirect = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            records = [{"value": value} for value in range(10, 29)]
            records.append({"value": 1000})
            yield json.dumps({"data": records}).encode()

        def close(self):
            return None

    monkeypatch.setenv("DATA_SOURCE_HTTP_ALLOWED_HOSTS", "api.example.test")
    monkeypatch.setattr(
        "data_sources.requests.get",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    response = client.post(
        "/api/monitoring/analyze",
        json={
            "source": {
                "type": "http_json",
                "url": "https://api.example.test/events",
            },
            "checks": ["quality", "anomalies", "drift"],
            "baseline": [{"value": value} for value in range(20)],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["results"]) == {"quality", "anomalies", "drift"}
    assert body["results"]["anomalies"]["available"] is True
    assert body["results"]["drift"]["drift_detected"] is True
    assert client.get(f"/api/datasets/{body['dataset_id']}").status_code == 200


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


def test_assistant_endpoint_returns_answer_in_requested_language():
    response = client.post(
        "/api/assistant/ask",
        json={
            "question": "¿Cuál es la puntuación de calidad de datos de hoy?",
            "language": "es",
            "data": [{"value": 1}, {"value": 2}],
        },
    )

    assert response.status_code == 200
    assert response.json()["category"] == "quality"
    assert "/100" in response.json()["answer"]


def test_assistant_uses_date_scoped_monitoring_history_for_quality_score():
    dataset_response = client.post(
        "/api/datasets",
        json={"name": "quality-history", "data": [{"value": 1}, {"value": 2}]},
    )
    assert dataset_response.status_code == 201
    dataset_id = dataset_response.json()["id"]

    from monitoring_store import persist_monitoring_run

    _, result_id = persist_monitoring_run(
        [{"key": "current", "id": dataset_id}],
        "monitoring",
        {
            "results": {
                "quality": {
                    "quality_score": {"overall_score": 91},
                    "report": {"rows": 2, "columns": 1, "total_nulls": 0},
                }
            }
        },
        owner_id="test-admin-user",
    )

    response = client.post(
        "/api/assistant/ask",
        json={
            "question": "What is today's quality score?",
            "dataset_id": dataset_id,
            "monitoring_date": date.today().isoformat(),
            "data": [{"value": 1}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "91.0/100" in body["answer"]
    assert date.today().isoformat() in body["answer"]
    assert dataset_id in body["answer"]
    assert body["context"]["source"] == {
        "type": "monitoring_history",
        "date": date.today().isoformat(),
        "dataset_id": dataset_id,
        "dataset_name": "quality-history",
        "result_id": result_id,
    }


def test_assistant_does_not_substitute_snapshot_when_date_has_no_quality_run():
    dataset_response = client.post(
        "/api/datasets",
        json={"name": "no-quality-run", "data": [{"value": 1}, {"value": 2}]},
    )
    dataset_id = dataset_response.json()["id"]
    requested_date = (date.today() + timedelta(days=1)).isoformat()

    response = client.post(
        "/api/assistant/ask",
        json={
            "question": "What is today's quality score?",
            "dataset_id": dataset_id,
            "monitoring_date": requested_date,
            "data": [{"value": 1}, {"value": 2}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "not available" in body["answer"]
    assert requested_date in body["answer"]
    assert "saved snapshot score was not substituted" in body["answer"]
    assert body["context"]["quality_score"] is None
    assert body["context"]["source"]["result_id"] is None


def test_assistant_rejects_invalid_monitoring_date():
    response = client.post(
        "/api/assistant/ask",
        json={
            "question": "today's quality score",
            "dataset_id": "dataset-a",
            "monitoring_date": "not-a-date",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "monitoring_date must be an ISO date"


def test_assistant_endpoint_rejects_invalid_language_type():
    response = client.post(
        "/api/assistant/ask",
        json={"question": "show anomalies", "language": [], "data": [{"value": 1}]},
    )

    assert response.status_code == 400


def test_advanced_ml_endpoint_runs_clustering_and_persists_report():
    response = client.post(
        "/api/analytics/advanced-ml",
        json={
            "analysis": "clustering",
            "n_clusters": 2,
            "data": [
                {"temperature": 10.0},
                {"temperature": 10.2},
                {"temperature": 90.0},
                {"temperature": 90.2},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["analysis"] == "clustering"
    assert len(body["report"]["assignments"]) == 4
    stored = client.get(f"/api/monitoring/results/{body['result_id']}")
    assert stored.status_code == 200
    assert stored.json()["result"]["report"] == body["report"]


def test_advanced_ml_endpoint_rejects_unknown_analysis():
    response = client.post(
        "/api/analytics/advanced-ml",
        json={"analysis": "unknown", "data": [{"value": 1}]},
    )

    assert response.status_code == 400


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
