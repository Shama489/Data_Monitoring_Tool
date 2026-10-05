import json
from pathlib import Path
from typing import Any

from agents.root_cause_agent import investigate
from monitoring_store import persist_monitoring_run
from profiler import answer_monitoring_question
from tools.data_tools import dataframe_from_value, load_dataset
from tools.anomaly_tools import analyze_anomalies
from tools.drift_tools import analyze_drift
from tools.ml_tools import forecast_data
from tools.notification_tools import send_monitoring_notification
from tools.quality_tools import analyze_quality
from tools.registry import ToolRegistry
from tools.xai_tools import explain_model


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
    registry.register("notification.send", "Send an explicitly requested monitoring notification.", send_monitoring_notification)
    return registry


def run_monitoring(
    payload: dict[str, Any],
    quality_options: dict[str, Any],
    use_llm: bool = False,
    owner_id: str | None = None,
) -> dict[str, Any]:
    checks = payload.get("checks")
    if checks is None:
        checks = ["quality"]
        if (
            payload.get("baseline") is not None
            or payload.get("baseline_dataset") is not None
            or payload.get("baseline_dataset_id") is not None
        ):
            checks.append("drift")
    if not isinstance(checks, list) or not checks or not all(isinstance(check, str) for check in checks):
        raise ValueError("checks must be a non-empty list containing quality and/or drift")
    if any(check not in {"quality", "drift", "anomalies", "forecast", "explain"} for check in checks):
        raise ValueError("checks may contain only quality, drift, anomalies, forecast, and explain")
    checks = list(dict.fromkeys(checks))

    registry = build_tool_registry()
    trace = []
    frame = load_dataset(payload, owner_id=owner_id)
    trace.append({"tool": "data.load", "status": "completed"})

    results = {}
    baseline = None
    if "quality" in checks:
        results["quality"] = registry.run(
            "quality.analyze",
            frame=frame,
            options=quality_options,
            use_llm=use_llm,
        )
        trace.append({"tool": "quality.analyze", "status": "completed"})

    if "drift" in checks:
        baseline_data = payload.get("baseline")
        if baseline_data is None:
            baseline_data = payload.get("baseline_dataset")
        baseline_id = payload.get("baseline_dataset_id")
        if baseline_data is None and baseline_id is None:
            raise ValueError("A baseline dataset is required when drift is selected.")
        if baseline_id is not None:
            if not isinstance(baseline_id, str) or not baseline_id.strip():
                raise ValueError("baseline_dataset_id must be a non-empty string")
            baseline = load_dataset({"dataset_id": baseline_id}, owner_id=owner_id)
        else:
            baseline = dataframe_from_value(baseline_data, "Baseline dataset")
        results["drift"] = registry.run("drift.analyze", baseline=baseline, current=frame)
        trace.append({"tool": "drift.analyze", "status": "completed"})

    if "anomalies" in checks:
        results["anomalies"] = registry.run("anomalies.analyze", frame=frame)
        trace.append({"tool": "anomalies.analyze", "status": "completed"})

    if "forecast" in checks:
        results["forecast"] = registry.run(
            "forecast.run",
            frame=frame,
            date_column=payload.get("date_column"),
            value_column=payload.get("value_column"),
            periods=int(payload.get("periods", 4)),
            frequency=str(payload.get("frequency", "W")),
            method=str(payload.get("method", "auto")),
            metric=payload.get("metric"),
        )
        trace.append({"tool": "forecast.run", "status": "completed"})

    if "explain" in checks:
        results["explanation"] = registry.run(
            "xai.explain",
            frame=frame,
            target_column=payload.get("target_column"),
            task=payload.get("task", "classification"),
            test_size=payload.get("test_size", 0.2),
        )
        trace.append({"tool": "xai.explain", "status": "completed"})

    root_cause = investigate(results)
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
            notifications = registry.run(
                "notification.send",
                message=f"Monitoring findings detected: {summary}",
                channels=channels,
                severity=severity,
            )
            trace.append({"tool": "notification.send", "status": notifications.get("status", "unknown")})
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

    dataset_ids, result_id = persist_monitoring_run(
        datasets, "monitoring", response, owner_id=owner_id
    )
    response["dataset_id"] = dataset_ids.get("current")
    response["result_id"] = result_id
    if "baseline" in dataset_ids:
        response["baseline_dataset_id"] = dataset_ids["baseline"]
    return response