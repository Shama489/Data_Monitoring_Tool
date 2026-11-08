import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from agents.root_cause_agent import investigate
from monitoring_store import list_monitoring_results, persist_monitoring_run
from profiler import OpenAI, answer_monitoring_question
from tools.data_tools import dataframe_from_value, load_dataset
from tools.anomaly_tools import analyze_anomalies
from tools.drift_tools import analyze_drift
from tools.ml_tools import forecast_data
from tools.notification_tools import send_monitoring_notification
from tools.quality_tools import analyze_quality
from tools.registry import ToolRegistry
from tools.xai_tools import explain_model

logger = logging.getLogger(__name__)
AGENT_MAX_TOOL_CALLS = 5


_AGENT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_quality",
            "description": "Analyze quality score, schema, duplicates, and data-quality rules.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_missing_values",
            "description": "Count missing values and identify affected columns.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect anomalous records in numeric columns.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_drift",
            "description": "Compare the selected dataset with its supplied baseline.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_results",
            "description": "Read recent completed monitoring results for the selected saved dataset.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 10}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_dataset",
            "description": "Summarize dataset shape, column types, missing counts, and numeric ranges.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

_AGENT_TOOL_TO_CHECK = {
    "analyze_quality": "quality",
    "find_missing_values": "missing_values",
    "detect_anomalies": "anomalies",
    "compare_drift": "drift",
    "get_historical_results": "history",
    "profile_dataset": "profile",
}


class MonitoringStageError(Exception):
    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(f"Monitoring stage '{stage}' failed ({type(cause).__name__})")


def _run_stage(
    trace: list[dict[str, str]],
    stage: str,
    operation: Callable[[], Any],
    include_success: bool = True,
) -> Any:
    try:
        result = operation()
    except Exception as error:
        trace.append({
            "tool": stage,
            "status": "failed",
            "error": type(error).__name__,
        })
        raise MonitoringStageError(stage, error) from error
    if include_success:
        trace.append({"tool": stage, "status": "completed"})
    return result


def _dataset_spec(
    key: str,
    frame,
    name: str,
    source: dict[str, Any] | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    sanitized_source = {}
    if source:
        for field in ("type", "file_type", "name", "key", "collection"):
            if source.get(field) is not None:
                sanitized_source[field] = source[field]
        source_path = source.get("path") or source.get("location")
        if source_path:
            sanitized_source["name"] = Path(str(source_path)).name
    return {
        "key": key,
        "id": dataset_id,
        "name": name,
        "data": json.loads(frame.to_json(orient="records", date_format="iso")),
        "source": sanitized_source or {"type": "inline"},
    }


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("data.load", "Load a dataset from inline records or a configured source.", load_dataset)
    registry.register("quality.analyze", "Run deterministic data quality, schema, freshness, and rule checks.", analyze_quality)
    registry.register("drift.analyze", "Compare current data with a baseline and calculate feature drift.", analyze_drift)
    registry.register("anomalies.analyze", "Detect anomalous rows in numeric features.", analyze_anomalies)
    registry.register("forecast.run", "Forecast a selected metric using the existing forecasting methods.", forecast_data)
    registry.register("xai.explain", "Train and explain a model with holdout evaluation and optional SHAP output.", explain_model)
    registry.register("assistant.answer", "Answer natural-language questions about data quality, missing values, anomalies, drift, and monitoring results.", answer_monitoring_question)
    registry.register("missing_values.analyze", "Count missing values and identify affected columns.", _missing_values)
    registry.register("dataset.profile", "Summarize dataset shape, column types, missing counts, and numeric ranges.", _profile_dataset)
    registry.register("history.latest", "Read recent completed monitoring results for a saved dataset.", _historical_summary)
    registry.register("notification.send", "Send an explicitly requested monitoring notification.", send_monitoring_notification)
    return registry


def _call_agentic_planner(question: str, available_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("agent_not_configured")
    client = OpenAI(api_key=api_key, timeout=30, max_retries=0)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Choose only the approved monitoring tools needed to answer the user's "
                    "question. Never request data, code, SQL, credentials, or notifications. "
                    "You are selecting tools only; analysis results are summarized locally."
                ),
            },
            {"role": "user", "content": question},
        ],
        tools=available_tools,
        tool_choice="required",
        parallel_tool_calls=True,
        temperature=0,
    )
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise ValueError("agent_no_tool_calls")
    if len(tool_calls) > AGENT_MAX_TOOL_CALLS:
        raise ValueError("agent_tool_limit_exceeded")

    calls = []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None)
        raw_arguments = getattr(function, "arguments", None)
        if not isinstance(name, str) or not isinstance(raw_arguments, str):
            raise ValueError("agent_invalid_tool_call")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            raise ValueError("agent_invalid_tool_arguments") from error
        if not isinstance(arguments, dict):
            raise ValueError("agent_invalid_tool_arguments")
        calls.append({"name": name, "arguments": arguments})
    return calls


def _validate_agent_tool_call(
    call: dict[str, Any],
    allowed_names: set[str],
) -> tuple[str, dict[str, Any]]:
    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or name not in allowed_names:
        raise ValueError("Unknown or unavailable agent tool")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")

    if name == "get_historical_results":
        if set(arguments) - {"limit"}:
            raise ValueError("Historical-results arguments may contain only limit")
        limit = arguments.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
            raise ValueError("Historical-results limit must be an integer from 1 to 10")
        return name, {"limit": limit}

    if arguments:
        raise ValueError(f"{name} does not accept arguments")
    return name, {}


def _deterministic_agent_calls(
    question: str,
    checks: list[str] | None,
    has_baseline: bool,
    has_dataset_id: bool,
) -> list[dict[str, Any]]:
    if checks is not None:
        requested = list(checks)
        calls = [
            {"name": name, "arguments": {}}
            for name, check in _AGENT_TOOL_TO_CHECK.items()
            if check in requested and check in {"quality", "anomalies", "drift"}
        ]
        if requested and not calls:
            calls.append({"name": "analyze_quality", "arguments": {}})
    else:
        normalized = question.casefold()
        calls = []
        if any(term in normalized for term in ("missing", "null", "blank", "empty")):
            calls.append({"name": "find_missing_values", "arguments": {}})
        if any(term in normalized for term in ("anomal", "outlier")):
            calls.append({"name": "detect_anomalies", "arguments": {}})
        if any(term in normalized for term in ("drift", "distribution shift")) and has_baseline:
            calls.append({"name": "compare_drift", "arguments": {}})
        if any(term in normalized for term in ("history", "historical", "previous run", "past run")) and has_dataset_id:
            calls.append({"name": "get_historical_results", "arguments": {"limit": 5}})
        if any(term in normalized for term in ("profile", "columns", "schema", "dataset")):
            calls.append({"name": "profile_dataset", "arguments": {}})
        if any(term in normalized for term in ("quality", "score", "duplicate")) or not calls:
            calls.append({"name": "analyze_quality", "arguments": {}})
    return calls[:AGENT_MAX_TOOL_CALLS]


def _profile_dataset(frame) -> dict[str, Any]:
    columns = []
    for column in frame.columns:
        series = frame[column]
        item: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = series.describe()
            item["numeric_summary"] = {
                key: float(numeric[key])
                for key in ("min", "mean", "max")
                if key in numeric and pd.notna(numeric[key]) and math.isfinite(float(numeric[key]))
            }
        columns.append(item)
    return {"rows": int(len(frame)), "column_count": int(len(frame.columns)), "columns": columns}


def _missing_values(frame) -> dict[str, Any]:
    counts = frame.isna().sum()
    return {
        "total": int(counts.sum()),
        "by_column": {str(column): int(count) for column, count in counts.items() if count},
    }


def _historical_summary(dataset_id: str | None, owner_id: str | None, limit: int) -> dict[str, Any]:
    if not dataset_id:
        return {"available": False, "reason": "A saved dataset is required for history."}
    records = list_monitoring_results(
        limit=limit,
        dataset_id=dataset_id,
        owner_id=owner_id,
    )
    return {
        "available": True,
        "results": [
            {
                "id": item["id"],
                "created_at": item["created_at"],
                "check_type": item["check_type"],
                "checks": (item.get("result") or {}).get("checks", []),
            }
            for item in records
        ],
    }


def _agent_answer(results: dict[str, Any], calls: list[dict[str, Any]]) -> str:
    labels = {
        "quality": "data quality",
        "missing_values": "missing values",
        "anomalies": "anomaly detection",
        "drift": "data drift",
        "history": "monitoring history",
        "profile": "dataset profile",
    }
    performed = [
        labels[_AGENT_TOOL_TO_CHECK[call["name"]]]
        for call in calls
        if call["name"] in _AGENT_TOOL_TO_CHECK
        and _AGENT_TOOL_TO_CHECK[call["name"]] in results
    ]
    details = []
    quality = results.get("quality")
    if isinstance(quality, dict):
        score = (quality.get("quality_score") or {}).get("overall_score")
        if score is not None:
            details.append(f"quality score {float(score):.1f}/100")
    missing = results.get("missing_values")
    if isinstance(missing, dict):
        details.append(f"{missing.get('total', 0)} missing values")
    anomalies = results.get("anomalies")
    if isinstance(anomalies, dict) and anomalies.get("available"):
        details.append(f"{anomalies.get('total_anomalies', 0)} anomalies")
    drift = results.get("drift")
    if isinstance(drift, dict):
        details.append(
            f"drift score {drift.get('overall_drift_score', 0)} "
            f"({ 'detected' if drift.get('drift_detected') else 'not detected' })"
        )
    profile = results.get("profile")
    if isinstance(profile, dict):
        details.append(f"{profile.get('rows', 0)} rows and {profile.get('column_count', 0)} columns")
    history = results.get("history")
    if isinstance(history, dict) and history.get("available"):
        details.append(f"{len(history.get('results', []))} recent monitoring runs")
    summary = f"Performed: {', '.join(performed) or 'no checks'}."
    if details:
        summary += f" Findings: {'; '.join(details)}."
    return summary


def run_monitoring(
    payload: dict[str, Any],
    quality_options: dict[str, Any],
    use_llm: bool = False,
    owner_id: str | None = None,
) -> dict[str, Any]:
    agentic = payload.get("agentic", False)
    if not isinstance(agentic, bool):
        raise ValueError("agentic must be a boolean")

    question = payload.get("question", "")
    if agentic and (not isinstance(question, str) or not question.strip()):
        raise ValueError("question must be a non-empty string when agentic is enabled")
    if agentic and len(question) > 2000:
        raise ValueError("question must be at most 2,000 characters")

    checks = payload.get("checks")
    if agentic and checks == []:
        checks = None
    if checks is None and not agentic:
        checks = ["quality"]
        if (
            payload.get("baseline") is not None
            or payload.get("baseline_dataset") is not None
            or payload.get("baseline_dataset_id") is not None
        ):
            checks.append("drift")
    if checks is not None and (
        not isinstance(checks, list)
        or not checks
        or not all(isinstance(check, str) for check in checks)
    ):
        raise ValueError("checks must be a non-empty list containing quality and/or drift")
    if checks is not None and any(
        check not in {"quality", "drift", "anomalies", "forecast", "explain"}
        for check in checks
    ):
        raise ValueError("checks may contain only quality, drift, anomalies, forecast, and explain")
    if checks is not None:
        checks = list(dict.fromkeys(checks))

    registry = build_tool_registry()
    trace: list[dict[str, str]] = []
    frame = _run_stage(
        trace,
        "data.load",
        lambda: load_dataset(payload, owner_id=owner_id),
    )

    agent = None
    if agentic:
        has_baseline = any(
            payload.get(key) is not None
            for key in ("baseline", "baseline_dataset", "baseline_dataset_id")
        )
        has_dataset_id = isinstance(payload.get("dataset_id"), str)
        baseline_tool_definitions = [
            tool for tool in _AGENT_TOOL_DEFINITIONS
            if (
                (tool["function"]["name"] != "compare_drift" or has_baseline)
                and (tool["function"]["name"] != "get_historical_results" or has_dataset_id)
            )
        ]
        allowed_tool_names = {
            tool["function"]["name"] for tool in baseline_tool_definitions
        }
        try:
            calls = _call_agentic_planner(question.strip(), baseline_tool_definitions)
            for call in calls:
                _validate_agent_tool_call(call, allowed_tool_names)
            agent = {"mode": "llm", "status": "active", "fallback_reason": None}
            trace.append({"tool": "agent.plan", "status": "completed"})
        except Exception as error:
            fallback_reason = (
                str(error) if str(error) in {
                    "agent_not_configured",
                    "agent_no_tool_calls",
                    "agent_tool_limit_exceeded",
                    "agent_invalid_tool_call",
                    "agent_invalid_tool_arguments",
                }
                else type(error).__name__
            )
            logger.warning("Agentic planning unavailable; using deterministic selection (%s)", fallback_reason)
            calls = _deterministic_agent_calls(
                question.strip(),
                checks,
                has_baseline,
                isinstance(payload.get("dataset_id"), str),
            )
            agent = {
                "mode": "deterministic_fallback",
                "status": "fallback",
                "fallback_reason": fallback_reason,
            }
            trace.append({
                "tool": "agent.plan",
                "status": "fallback",
                "error": fallback_reason,
            })

        calls = calls[:AGENT_MAX_TOOL_CALLS]
        selected_names = {call["name"] for call in calls}
        checks = [
            check for tool_name, check in _AGENT_TOOL_TO_CHECK.items()
            if tool_name in selected_names
        ]
        checks = list(dict.fromkeys(checks))
        if not checks:
            calls = [{"name": "analyze_quality", "arguments": {}}]
            checks = ["quality"]

    results = {}
    baseline = None
    if not agentic and "quality" in checks:
        results["quality"] = _run_stage(
            trace,
            "quality.analyze",
            lambda: registry.run(
                "quality.analyze",
                frame=frame,
                options=quality_options,
                use_llm=use_llm,
            ),
        )

    if not agentic and "drift" in checks:
        baseline_data = payload.get("baseline")
        if baseline_data is None:
            baseline_data = payload.get("baseline_dataset")
        baseline_id = payload.get("baseline_dataset_id")
        if baseline_data is None and baseline_id is None:
            raise ValueError("A baseline dataset is required when drift is selected.")
        if baseline_id is not None:
            if not isinstance(baseline_id, str) or not baseline_id.strip():
                raise ValueError("baseline_dataset_id must be a non-empty string")
            baseline = _run_stage(
                trace,
                "drift.baseline.load",
                lambda: load_dataset({"dataset_id": baseline_id}, owner_id=owner_id),
            )
        else:
            baseline = _run_stage(
                trace,
                "drift.baseline.load",
                lambda: dataframe_from_value(baseline_data, "Baseline dataset"),
            )
        results["drift"] = _run_stage(
            trace,
            "drift.analyze",
            lambda: registry.run("drift.analyze", baseline=baseline, current=frame),
        )

    if not agentic and "anomalies" in checks:
        results["anomalies"] = _run_stage(
            trace,
            "anomalies.analyze",
            lambda: registry.run("anomalies.analyze", frame=frame),
        )

    if not agentic and "forecast" in checks:
        results["forecast"] = _run_stage(
            trace,
            "forecast.run",
            lambda: registry.run(
                "forecast.run",
                frame=frame,
                date_column=payload.get("date_column"),
                value_column=payload.get("value_column"),
                periods=int(payload.get("periods", 4)),
                frequency=str(payload.get("frequency", "W")),
                method=str(payload.get("method", "auto")),
                metric=payload.get("metric"),
            ),
        )

    if not agentic and "explain" in checks:
        results["explanation"] = _run_stage(
            trace,
            "xai.explain",
            lambda: registry.run(
                "xai.explain",
                frame=frame,
                target_column=payload.get("target_column"),
                task=payload.get("task", "classification"),
                test_size=payload.get("test_size", 0.2),
            ),
        )

    if agentic:
        action_failures = 0
        executed_calls = []
        for call in calls:
            try:
                name, arguments = _validate_agent_tool_call(call, allowed_tool_names)
                action = _AGENT_TOOL_TO_CHECK[name]
                if action == "quality":
                    if "quality" not in results:
                        results["quality"] = registry.run(
                            "quality.analyze",
                            frame=frame,
                            options=quality_options,
                            use_llm=use_llm,
                        )
                elif action == "missing_values":
                    results["missing_values"] = registry.run(
                        "missing_values.analyze", frame=frame
                    )
                elif action == "anomalies":
                    if "anomalies" not in results:
                        results["anomalies"] = registry.run("anomalies.analyze", frame=frame)
                elif action == "drift":
                    if baseline is None:
                        baseline_data = payload.get("baseline")
                        if baseline_data is None:
                            baseline_data = payload.get("baseline_dataset")
                        baseline_id = payload.get("baseline_dataset_id")
                        if baseline_id is not None:
                            baseline = load_dataset({"dataset_id": baseline_id}, owner_id=owner_id)
                        elif baseline_data is not None:
                            baseline = dataframe_from_value(baseline_data, "Baseline dataset")
                        else:
                            raise ValueError("A baseline dataset is required for drift analysis.")
                    if "drift" not in results:
                        results["drift"] = registry.run(
                            "drift.analyze", baseline=baseline, current=frame
                        )
                elif action == "history":
                    results["history"] = registry.run(
                        "history.latest",
                        dataset_id=payload.get("dataset_id"),
                        owner_id=owner_id,
                        limit=arguments["limit"],
                    )
                elif action == "profile":
                    results["profile"] = registry.run("dataset.profile", frame=frame)
                executed_calls.append({"name": name, "arguments": arguments})
                trace.append({"tool": f"agent.tool.{name}", "status": "completed"})
            except Exception as error:
                action_failures += 1
                trace.append({
                    "tool": f"agent.tool.{call.get('name', 'unknown')}",
                    "status": "failed",
                    "error": type(error).__name__,
                })
                logger.warning("Agentic tool execution failed (%s)", type(error).__name__)
        if agent is not None:
            if action_failures and executed_calls:
                agent["status"] = "partial"
            elif action_failures or not executed_calls:
                agent["status"] = "failed"
            agent["tools"] = [call["name"] for call in executed_calls]
            agent["answer"] = _agent_answer(results, executed_calls)
            agent["tool_call_limit"] = AGENT_MAX_TOOL_CALLS

    root_cause = _run_stage(
        trace,
        "root_cause.investigate",
        lambda: investigate(results),
        include_success=False,
    )
    notifications = {"status": "skipped", "reason": "notifications not requested"}
    notify = payload.get("notify", False)
    if not isinstance(notify, bool):
        raise ValueError("notify must be a boolean")
    if notify:
        channels = payload.get("channels", payload.get("notifications", []))
        if isinstance(channels, (str, dict)):
            channels = [channels]
        if not isinstance(channels, list):
            raise ValueError("channels must be a list, string, or channel object")
        if root_cause["findings"]:
            drift_severities = [
                finding["evidence"].get("overall_severity")
                for finding in root_cause["findings"]
                if finding["signal"] == "distribution_drift"
            ]
            severity = (
                "critical" if "critical" in drift_severities
                else "high" if "high" in drift_severities
                else "warning"
            )
            summary = "; ".join(finding["signal"] for finding in root_cause["findings"])
            notifications = _run_stage(
                trace,
                "notification.send",
                lambda: registry.run(
                    "notification.send",
                    message=f"Monitoring findings detected: {summary}",
                    channels=channels,
                    severity=severity,
                ),
            )
            trace[-1]["status"] = notifications.get("status", "unknown")
        else:
            notifications = {"status": "skipped", "reason": "no monitoring findings"}

    response = {
        "workflow": "monitoring",
        "checks": checks,
        "tool_trace": trace,
        "results": results,
        "root_cause": root_cause,
        "notifications": notifications,
    }
    if agent is not None:
        response["agent"] = agent
    if payload.get("scheduled") is True and isinstance(payload.get("schedule_id"), str):
        response["schedule_id"] = payload["schedule_id"]
        response["scheduled"] = True
    dataset_name = payload.get("dataset_name") or payload.get("name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        source = payload.get("source")
        source_path = source.get("path") or source.get("location") if isinstance(source, dict) else None
        dataset_name = Path(str(source_path)).name if source_path else "monitoring_dataset"

    datasets = [
        _dataset_spec(
            "current",
            frame,
            dataset_name,
            payload.get("source") if isinstance(payload.get("source"), dict) else None,
            payload.get("dataset_id"),
        )
    ]
    if baseline is not None and payload.get("baseline_dataset_id") is None:
        datasets.append(
            _dataset_spec(
                "baseline",
                baseline,
                f"{dataset_name}_baseline",
                {"type": "inline"},
            )
        )

    dataset_ids, result_id = _run_stage(
        trace,
        "monitoring.persist",
        lambda: persist_monitoring_run(
            datasets, "monitoring", response, owner_id=owner_id
        ),
        include_success=False,
    )
    response["dataset_id"] = dataset_ids.get("current")
    response["result_id"] = result_id
    if "baseline" in dataset_ids:
        response["baseline_dataset_id"] = dataset_ids["baseline"]
    return response