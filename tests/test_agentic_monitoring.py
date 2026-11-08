from datetime import date, datetime, timedelta
from types import SimpleNamespace
import time

import pytest
from fastapi.testclient import TestClient

import agents.monitoring_agent as monitoring_agent
import main
from auth_security import hash_password, issue_token
from conftest import admin_headers
from main import app
from monitoring_store import open_database


client = TestClient(app, headers=admin_headers())


def test_agentic_planner_selects_approved_tools_and_uses_actual_results(monkeypatch):
    monkeypatch.setattr(
        monitoring_agent,
        "_call_agentic_planner",
        lambda *_args: [
            {"name": "find_missing_values", "arguments": {}},
            {"name": "profile_dataset", "arguments": {}},
        ],
    )

    response = client.post(
        "/api/monitoring/analyze",
        json={
            "agentic": True,
            "question": "Find missing values and profile this dataset",
            "data": [{"amount": 10}, {"amount": None}],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent"]["mode"] == "llm"
    assert body["agent"]["tools"] == ["find_missing_values", "profile_dataset"]
    assert body["results"]["missing_values"] == {
        "total": 1,
        "by_column": {"amount": 1},
    }
    assert body["results"]["profile"]["rows"] == 2
    assert body["agent"]["answer"].startswith("Performed: missing values, dataset profile.")
    assert {
        item["tool"] for item in body["tool_trace"]
        if item["tool"].startswith("agent.tool.")
    } == {
        "agent.tool.find_missing_values",
        "agent.tool.profile_dataset",
    }


def test_agentic_planner_can_select_drift_when_baseline_is_available(monkeypatch):
    monkeypatch.setattr(
        monitoring_agent,
        "_call_agentic_planner",
        lambda *_args: [{"name": "compare_drift", "arguments": {}}],
    )

    response = client.post(
        "/api/monitoring/analyze",
        json={
            "agentic": True,
            "question": "Did the data distribution drift?",
            "baseline": [{"value": value} for value in [0, 0, 1, 1]],
            "data": [{"value": value} for value in [10, 10, 11, 11]],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["checks"] == ["drift"]
    assert body["results"]["drift"]["drift_detected"] is True
    assert "drift score" in body["agent"]["answer"]


def test_agentic_execution_failure_is_reported_without_aborting_run(monkeypatch):
    monkeypatch.setattr(
        monitoring_agent,
        "_call_agentic_planner",
        lambda *_args: [{"name": "detect_anomalies", "arguments": {}}],
    )
    monkeypatch.setattr(
        monitoring_agent,
        "analyze_anomalies",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("analysis failed")),
    )

    response = client.post(
        "/api/monitoring/analyze",
        json={
            "agentic": True,
            "question": "Find anomalies",
            "data": [{"value": 1}, {"value": 2}],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent"]["status"] == "failed"
    failure = next(
        item for item in body["tool_trace"]
        if item["tool"] == "agent.tool.detect_anomalies"
    )
    assert failure["status"] == "failed"
    assert failure["error"] == "RuntimeError"


def test_agentic_history_tool_is_owner_scoped(monkeypatch):
    analyst_id = "test-agent-analyst"
    with open_database() as connection:
        connection.execute(
            "INSERT INTO users (id, username, password_hash, role, is_active, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (
                analyst_id,
                "test-agent-analyst",
                hash_password("test-password-123"),
                "analyst",
                time.time(),
            ),
        )
    analyst_token = issue_token({"id": analyst_id}, lifetime_seconds=3600)
    analyst_client = TestClient(
        app,
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    created = analyst_client.post(
        "/api/datasets",
        json={"name": "agent-history", "data": [{"value": 1}]},
    )
    assert created.status_code == 201
    dataset_id = created.json()["id"]
    calls = []
    monkeypatch.setattr(
        monitoring_agent,
        "_call_agentic_planner",
        lambda *_args: [
            {"name": "get_historical_results", "arguments": {"limit": 3}}
        ],
    )
    monkeypatch.setattr(
        monitoring_agent,
        "list_monitoring_results",
        lambda **kwargs: calls.append(kwargs) or [
            {
                "id": "previous-run",
                "created_at": 1,
                "check_type": "monitoring",
                "result": {"checks": ["quality"]},
            }
        ],
    )

    response = analyst_client.post(
        "/api/monitoring/analyze",
        json={
            "agentic": True,
            "question": "Show previous monitoring results",
            "dataset_id": dataset_id,
        },
    )

    assert response.status_code == 200, response.text
    assert calls == [{
        "limit": 3,
        "dataset_id": dataset_id,
        "owner_id": analyst_id,
    }]
    assert response.json()["results"]["history"]["results"][0]["id"] == "previous-run"


def test_agentic_mode_falls_back_without_openai_and_default_mode_stays_deterministic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agent_response = client.post(
        "/api/monitoring/analyze",
        json={
            "agentic": True,
            "question": "What is the quality score?",
            "data": [{"value": 1}, {"value": None}],
        },
    )
    default_response = client.post(
        "/api/monitoring/analyze",
        json={"data": [{"value": 1}, {"value": None}]},
    )

    assert agent_response.status_code == 200
    assert agent_response.json()["agent"]["mode"] == "deterministic_fallback"
    assert agent_response.json()["agent"]["fallback_reason"] == "agent_not_configured"
    assert "quality" in agent_response.json()["results"]
    assert default_response.status_code == 200
    assert "agent" not in default_response.json()
    assert default_response.json()["checks"] == ["quality"]


def test_agentic_tool_arguments_and_tool_count_are_bounded(monkeypatch):
    with pytest.raises(ValueError, match="limit must be an integer from 1 to 10"):
        monitoring_agent._validate_agent_tool_call(
            {
                "name": "get_historical_results",
                "arguments": {"limit": 1000},
            },
            {"get_historical_results"},
        )
    with pytest.raises(ValueError, match="does not accept arguments"):
        monitoring_agent._validate_agent_tool_call(
            {"name": "profile_dataset", "arguments": {"sql": "SELECT 1"}},
            {"profile_dataset"},
        )

    def too_many_calls(**_kwargs):
        calls = [
            SimpleNamespace(
                function=SimpleNamespace(name="profile_dataset", arguments="{}")
            )
            for _ in range(monitoring_agent.AGENT_MAX_TOOL_CALLS + 1)
        ]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))]
        )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=too_many_calls)
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(monitoring_agent, "OpenAI", FakeOpenAI)
    with pytest.raises(ValueError, match="agent_tool_limit_exceeded"):
        monitoring_agent._call_agentic_planner(
            "profile this dataset",
            monitoring_agent._AGENT_TOOL_DEFINITIONS,
        )


def test_today_question_uses_latest_run_on_current_date_not_selected_date(monkeypatch):
    today = date.today()
    previous_date = today - timedelta(days=1)
    dataset_id = "dataset-for-today"
    today_latest = {
        "id": "today-latest",
        "created_at": datetime.combine(today, datetime.min.time()).timestamp() + 10,
        "result": {
            "results": {"quality": {"quality_score": {"overall_score": 91}}}
        },
    }
    previous = {
        "id": "yesterday",
        "created_at": datetime.combine(previous_date, datetime.min.time()).timestamp() + 10,
        "result": {
            "results": {"quality": {"quality_score": {"overall_score": 42}}}
        },
    }
    captured = {}
    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda *_args, **_kwargs: {
            "id": dataset_id,
            "name": "Events",
            "data": [{"value": 1}],
        },
    )
    monkeypatch.setattr(
        main,
        "list_monitoring_results",
        lambda **kwargs: [today_latest, previous],
    )
    monkeypatch.setattr(
        main,
        "answer_monitoring_question",
        lambda _question, **kwargs: captured.update(kwargs) or {
            "answer": "The score is 91.",
            "category": "quality",
            "confidence": "high",
            "context": {"quality_score": 91},
        },
    )

    response = client.post(
        "/api/assistant/ask",
        json={
            "question": "What is today's score?",
            "dataset_id": dataset_id,
            "monitoring_date": previous_date.isoformat(),
            "data": [{"value": None}],
        },
    )

    assert response.status_code == 200, response.text
    assert captured["monitoring_results"]["quality_score"]["overall_score"] == 91
    assert response.json()["context"]["source"]["date"] == today.isoformat()
    assert response.json()["context"]["source"]["result_id"] == "today-latest"


def test_today_question_reports_no_run_without_calculating_snapshot_score(monkeypatch):
    today = date.today()
    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda *_args, **_kwargs: {
            "id": "dataset-no-run",
            "name": "Events",
            "data": [{"value": 100}],
        },
    )
    monkeypatch.setattr(main, "list_monitoring_results", lambda **_kwargs: [])

    response = client.post(
        "/api/assistant/ask",
        json={
            "question": "What is today's quality score?",
            "dataset_id": "dataset-no-run",
            "monitoring_date": (today - timedelta(days=1)).isoformat(),
            "data": [{"value": 100}],
        },
    )

    assert response.status_code == 200, response.text
    assert "No quality monitoring run was recorded" in response.json()["answer"]
    assert response.json()["context"]["source"]["result_id"] is None
    assert response.json()["context"]["quality_score"] is None
