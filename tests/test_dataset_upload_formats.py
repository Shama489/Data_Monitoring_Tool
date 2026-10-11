from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

from conftest import admin_headers
from main import app


def test_dataset_upload_accepts_supported_json_and_tsv_formats():
    client = TestClient(app, headers=admin_headers())

    json_upload = client.post(
        "/api/datasets/upload",
        params={"name": "events.json"},
        content=b'[{"value":1},{"value":2}]',
        headers={"Content-Type": "application/json"},
    )
    assert json_upload.status_code == 201

    tsv_upload = client.post(
        "/api/datasets/upload",
        params={"name": "events.tsv"},
        content=b"value\tcategory\n3\tlow\n4\thigh\n",
        headers={"Content-Type": "text/tab-separated-values"},
    )
    assert tsv_upload.status_code == 201

    json_dataset = client.get(f"/api/datasets/{json_upload.json()['id']}")
    tsv_dataset = client.get(f"/api/datasets/{tsv_upload.json()['id']}")
    assert json_dataset.status_code == 200
    assert json_dataset.json()["data"] == [{"value": 1}, {"value": 2}]
    assert json_dataset.json()["source"]["file_type"] == "json"
    assert tsv_dataset.status_code == 200
    assert tsv_dataset.json()["data"][0] == {"value": 3, "category": "low"}
    assert tsv_dataset.json()["source"]["file_type"] == "tsv"


def test_dataset_upload_accepts_xlsx_and_parquet():
    client = TestClient(app, headers=admin_headers())
    frame = pd.DataFrame({"value": [5, 6], "group": ["a", "b"]})
    excel_content = BytesIO()
    frame.to_excel(excel_content, index=False)
    parquet_content = frame.to_parquet(index=False)

    excel_upload = client.post(
        "/api/datasets/upload",
        params={"name": "events.xlsx"},
        content=excel_content.getvalue(),
        headers={
            "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        },
    )
    parquet_upload = client.post(
        "/api/datasets/upload",
        params={"name": "events.parquet"},
        content=parquet_content,
        headers={"Content-Type": "application/vnd.apache.parquet"},
    )

    assert excel_upload.status_code == 201, excel_upload.text
    assert parquet_upload.status_code == 201, parquet_upload.text
    excel_dataset = client.get(f"/api/datasets/{excel_upload.json()['id']}")
    parquet_dataset = client.get(f"/api/datasets/{parquet_upload.json()['id']}")
    assert excel_dataset.json()["source"]["file_type"] == "xlsx"
    assert parquet_dataset.json()["source"]["file_type"] == "parquet"


def test_dataset_upload_rejects_mismatched_content_type_and_too_many_rows():
    client = TestClient(app, headers=admin_headers())
    mismatch = client.post(
        "/api/datasets/upload",
        params={"name": "events.json"},
        content=b'[{"value":1}]',
        headers={"Content-Type": "text/csv"},
    )
    assert mismatch.status_code == 400

    too_many_rows = client.post(
        "/api/datasets/upload",
        params={"name": "too-many.csv"},
        content=("value\n" + "1\n" * 100_001).encode(),
        headers={"Content-Type": "text/csv"},
    )
    assert too_many_rows.status_code == 400
    assert "rows" in too_many_rows.json()["detail"]
