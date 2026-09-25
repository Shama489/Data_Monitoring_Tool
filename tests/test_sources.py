import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data_sources import DataSourceError, load_local_file, load_sql_query
from main import app


client = TestClient(app)


def test_local_file_loader_supports_csv_and_json(tmp_path):
    frame = pd.DataFrame({"id": [1, 2], "status": ["ok", "warn"]})
    csv_path = tmp_path / "events.csv"
    json_path = tmp_path / "events.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(frame.to_dict(orient="records")), encoding="utf-8")

    pd.testing.assert_frame_equal(load_local_file(str(csv_path)), frame)
    pd.testing.assert_frame_equal(load_local_file(str(json_path)), frame)


def test_sql_loader_rejects_non_select_queries():
    with pytest.raises(DataSourceError, match="Only SELECT queries"):
        load_sql_query("sqlite://", "DELETE FROM events")


def test_sql_loader_rejects_multi_statement_select_queries():
    with pytest.raises(DataSourceError, match="Only SELECT queries"):
        load_sql_query("sqlite://", "SELECT 1; DELETE FROM events")


def test_sources_endpoint_reports_success_and_partial_failure(tmp_path):
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


def test_sources_endpoint_rejects_empty_source_list():
    response = client.post("/api/sources/analyze", json={"sources": []})

    assert response.status_code == 400
    assert response.json()["detail"] == "sources must be a non-empty list"


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
