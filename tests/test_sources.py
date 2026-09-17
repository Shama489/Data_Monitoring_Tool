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
