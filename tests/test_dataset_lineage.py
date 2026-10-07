from fastapi.testclient import TestClient

from auth_security import hash_password
from conftest import admin_headers
from main import app
from monitoring_store import create_user


client = TestClient(app, headers=admin_headers())


def test_dataset_versions_are_immutable_and_lineage_tracks_monitoring(monkeypatch):
    created = client.post(
        "/api/datasets",
        json={
            "name": "orders",
            "data": [
                {"order_id": 1, "amount": 10, "region": "north", "legacy": "x"},
                {"order_id": 2, "amount": 10, "region": "north", "legacy": "x"},
                {"order_id": 2, "amount": 10, "region": "north", "legacy": "x"},
                {"order_id": None, "amount": None, "region": "south", "legacy": "y"},
            ],
        },
    )
    assert created.status_code == 201
    original_id = created.json()["id"]

    updated = client.post(
        f"/api/datasets/{original_id}/versions",
        json={
            "change_summary": "Add source label and normalize amount",
            "data": [
                {"order_id": 1, "amount": "100", "region": "west", "source": "web"},
                {"order_id": 1, "amount": "100", "region": "west", "source": "web"},
                {"order_id": 3, "amount": "250", "region": "east", "source": "store"},
            ],
        },
    )
    assert updated.status_code == 201
    version_id = updated.json()["id"]
    assert updated.json()["version_number"] == 2
    assert updated.json()["parent_dataset_id"] == original_id
    assert client.get(f"/api/datasets/{original_id}").json()["data"][0]["amount"] == 10

    versions = client.get(f"/api/datasets/{version_id}/versions").json()["versions"]
    assert [item["version_number"] for item in versions] == [1, 2]
    assert versions[1]["change_summary"] == "Add source label and normalize amount"

    comparison = client.post(
        "/api/datasets/compare",
        json={
            "baseline_dataset_id": original_id,
            "current_dataset_id": version_id,
        },
    )
    assert comparison.status_code == 200
    report = comparison.json()
    assert report["schema"]["added_columns"] == ["source"]
    assert report["schema"]["removed_columns"] == ["legacy"]
    assert report["schema"]["type_changes"] == [
        {"column": "order_id", "baseline": "float64", "current": "int64"},
        {"column": "amount", "baseline": "float64", "current": "object"}
    ]
    assert report["quality"]["baseline"]["missing_values"]["total"] == 2
    assert report["quality"]["baseline"]["duplicate_rows"] == 1
    assert report["quality"]["current"]["duplicate_rows"] == 1
    assert report["distribution_drift"]["drift_detected"] is True
    assert report["result_id"]

    run = client.post(
        "/api/monitoring/analyze",
        json={"dataset_id": version_id, "checks": ["quality"]},
    )
    assert run.status_code == 200

    def fail_quality(*_args, **_kwargs):
        raise ValueError(
            f"quality stage failed with {len(_args)} args and {len(_kwargs)} kwargs"
        )

    monkeypatch.setattr("agents.monitoring_agent.analyze_quality", fail_quality)
    stage_failure = client.post(
        "/api/monitoring/analyze",
        json={"dataset_id": version_id, "checks": ["quality"]},
    )
    assert stage_failure.status_code == 400

    failed_run = client.post(
        "/api/monitoring/analyze",
        json={"dataset_id": version_id, "checks": ["not-a-check"]},
    )
    assert failed_run.status_code == 400

    lineage = client.get(f"/api/datasets/{version_id}/lineage")
    assert lineage.status_code == 200
    events = lineage.json()["events"]
    assert any(
        event["event_type"] == "dataset_version_created"
        and event["input_dataset_ids"] == [original_id]
        for event in events
    )
    assert any(
        event["event_type"] == "dataset_comparison"
        and event["result_id"] == report["result_id"]
        for event in events
    )
    assert any(event["status"] == "succeeded" for event in events)
    assert any(
        event["status"] == "failed" and "checks" in event["error"]
        for event in events
    )
    assert any(
        event["details"].get("failed_stage") == "quality.analyze"
        for event in events
    )


def test_dataset_version_access_is_scoped_and_read_only_roles_cannot_write():
    create_user(
        "lineage-analyst",
        hash_password("lineage-analyst-password"),
        "analyst",
    )
    create_user(
        "lineage-viewer",
        hash_password("lineage-viewer-password"),
        "viewer",
    )

    analyst_login = client.post(
        "/api/auth/login",
        json={
            "username": "lineage-analyst",
            "password": "lineage-analyst-password",
        },
    )
    viewer_login = client.post(
        "/api/auth/login",
        json={
            "username": "lineage-viewer",
            "password": "lineage-viewer-password",
        },
    )
    assert analyst_login.status_code == 200
    assert viewer_login.status_code == 200
    analyst = TestClient(
        app,
        headers={"Authorization": f"Bearer {analyst_login.json()['access_token']}"},
    )
    viewer = TestClient(
        app,
        headers={"Authorization": f"Bearer {viewer_login.json()['access_token']}"},
    )

    dataset = analyst.post(
        "/api/datasets",
        json={"name": "private", "data": [{"value": 1}]},
    ).json()
    version = analyst.post(
        f"/api/datasets/{dataset['id']}/versions",
        json={"data": [{"value": 2}]},
    )
    assert version.status_code == 201
    assert viewer.get(f"/api/datasets/{dataset['id']}/lineage").status_code == 404
    assert viewer.post(
        f"/api/datasets/{dataset['id']}/versions",
        json={"data": [{"value": 3}]},
    ).status_code == 403


def test_deleting_a_base_dataset_keeps_its_versions_accessible():
    base = client.post(
        "/api/datasets",
        json={"name": "retained", "data": [{"value": 1}]},
    ).json()
    version = client.post(
        f"/api/datasets/{base['id']}/versions",
        json={"data": [{"value": 2}]},
    ).json()

    assert client.delete(f"/api/datasets/{base['id']}").status_code == 200
    lineage = client.get(f"/api/datasets/{version['id']}/lineage")
    assert lineage.status_code == 200
    assert [item["id"] for item in lineage.json()["versions"]] == [version["id"]]
