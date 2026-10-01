from typing import Any

from agents.root_cause_agent import investigate
from tools.data_tools import dataframe_from_value, load_dataset
from tools.drift_tools import analyze_drift
from tools.notification_tools import send_monitoring_notification
from tools.quality_tools import analyze_quality
from tools.registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("data.load", "Load a dataset from inline records or a configured source.", load_dataset)
    registry.register("quality.analyze", "Run deterministic data quality, schema, freshness, and rule checks.", analyze_quality)
    registry.register("drift.analyze", "Compare current data with a baseline and calculate feature drift.", analyze_drift)
    registry.register("notification.send", "Send an explicitly requested monitoring notification.", send_monitoring_notification)
    return registry


def run_monitoring(payload: dict[str, Any], quality_options: dict[str, Any], use_llm: bool = False) -> dict[str, Any]:
    checks = payload.get("checks")
    if checks is None:
        checks = ["quality"]
        if payload.get("baseline") is not None or payload.get("baseline_dataset") is not None:
            checks.append("drift")
    if not isinstance(checks, list) or not checks or not all(isinstance(check, str) for check in checks):
        raise ValueError("checks must be a non-empty list containing quality and/or drift")
    if any(check not in {"quality", "drift"} for check in checks):
        raise ValueError("checks may contain only quality and drift")
    checks = list(dict.fromkeys(checks))

    registry = build_tool_registry()
    trace = []
    frame = registry.run("data.load", payload=payload)
    trace.append({"tool": "data.load", "status": "completed"})

    results = {}
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
        if baseline_data is None:
            raise ValueError("A baseline dataset is required when drift is selected.")
        baseline = dataframe_from_value(baseline_data, "Baseline dataset")
        results["drift"] = registry.run("drift.analyze", baseline=baseline, current=frame)
        trace.append({"tool": "drift.analyze", "status": "completed"})

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
            severity = "high" if any(
                finding["signal"] == "distribution_drift"
                and finding["evidence"].get("overall_severity") in {"high", "critical"}
                for finding in root_cause["findings"]
            ) else "warning"
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

    return {
        "workflow": "monitoring",
        "checks": checks,
        "tool_trace": trace,
        "results": results,
        "root_cause": root_cause,
        "notifications": notifications,
    }