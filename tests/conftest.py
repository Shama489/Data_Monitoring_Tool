import pytest


@pytest.fixture(autouse=True)
def isolate_monitoring_database(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "monitoring.db"))
