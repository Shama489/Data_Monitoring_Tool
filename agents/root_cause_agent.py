from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def investigate(results: dict[str, Any]) -> dict[str, Any]:
    findings = []
    quality = _as_dict(results.get("quality"))
    quality_report = _as_dict(quality.get("report"))

    total_nulls = _as_int(quality_report.get("total_nulls"), 0)
    if total_nulls:
        total_cells = max(
            _as_int(quality_report.get("rows"), 0) * _as_int(quality_report.get("columns"), 0),
            1,
        )
        findings.append({
            "signal": "missing_values",
            "evidence": {
                "total_nulls": total_nulls,
                "rate_percent": round(total_nulls / total_cells * 100, 2),
            },
            "recommended_investigation": "Inspect the affected columns and the latest ingestion or transformation step.",
        })

    duplicates = _as_int(quality_report.get("duplicates"), 0)
    if duplicates:
        findings.append({
            "signal": "duplicate_rows",
            "evidence": {"exact_duplicates": duplicates},
            "recommended_investigation": "Check source identifiers and deduplication behavior in the ingestion pipeline.",
        })

    schema = _as_dict(quality_report.get("schema_validation"))
    if schema.get("status") == "warning":
        findings.append({
            "signal": "schema_mismatch",
            "evidence": {
                "missing_columns": schema.get("missing_columns", []),
                "extra_columns": schema.get("extra_columns", []),
                "type_issues": schema.get("type_issues", []),
                "renamed_columns": schema.get("renamed_columns", []),
            },
            "recommended_investigation": "Compare the incoming schema with the expected contract.",
        })

    freshness = _as_dict(quality_report.get("data_freshness"))
    if freshness.get("is_fresh") is False:
        findings.append({
            "signal": "stale_data",
            "evidence": {
                "latest_timestamp": freshness.get("latest_timestamp"),
                "age_hours": freshness.get("age_hours"),
                "max_age_hours": freshness.get("max_age_hours"),
            },
            "recommended_investigation": "Check the upstream job schedule and the most recent successful ingestion.",
        })

    rules = _as_dict(quality_report.get("business_rule_validation"))
    violations = _as_int(rules.get("violations"), 0)
    if violations:
        findings.append({
            "signal": "business_rule_violations",
            "evidence": {
                "violations": violations,
                "rules": [
                    {"id": rule.get("id"), "violations": _as_int(rule.get("violations"), 0)}
                    for rule in rules.get("rules", [])
                    if isinstance(rule, dict) and _as_int(rule.get("violations"), 0)
                ],
            },
            "recommended_investigation": "Review the failing rule definitions and the corresponding source records.",
        })

    drift = _as_dict(results.get("drift"))
    if drift.get("drift_detected"):
        features = _as_dict(drift.get("feature_metrics"))
        affected_features = []
        for name, metric in features.items():
            metric = _as_dict(metric)
            if _as_int(metric.get("drift_score"), 0) >= 25:
                affected_features.append({
                    "feature": name,
                    "drift_score": metric.get("drift_score"),
                    "severity": metric.get("severity"),
                    "psi": metric.get("psi"),
                })
        findings.append({
            "signal": "distribution_drift",
            "evidence": {
                "overall_drift_score": drift.get("overall_drift_score"),
                "overall_severity": drift.get("overall_severity"),
                "affected_features": affected_features,
            },
            "recommended_investigation": "Compare the baseline and current ingestion sources, transformations, and affected feature distributions.",
        })

    anomalies = _as_dict(results.get("anomalies"))
    if anomalies.get("available") and _as_int(anomalies.get("total_anomalies"), 0):
        findings.append({
            "signal": "anomaly_spike",
            "evidence": {
                "total_anomalies": _as_int(anomalies.get("total_anomalies"), 0),
                "anomaly_percentage": anomalies.get("anomaly_percentage"),
                "indices": anomalies.get("indices", [])[:10],
            },
            "recommended_investigation": "Inspect the anomalous rows and the upstream conditions that changed during that period.",
        })

    forecast = _as_dict(results.get("forecast"))
    if forecast.get("forecast"):
        history = forecast.get("history", [])
        future = forecast.get("forecast", [])
        if history and future:
            recent_history = [float(entry.get("value", 0)) for entry in history[-3:]]
            future_values = [float(entry.get("value", 0)) for entry in future]
            if recent_history and future_values:
                recent_mean = sum(recent_history) / len(recent_history)
                forecast_mean = sum(future_values) / len(future_values)
                if forecast_mean > recent_mean * 1.15 or forecast_mean < recent_mean * 0.85:
                    findings.append({
                        "signal": "forecast_shift",
                        "evidence": {
                            "metric": forecast.get("metric", "value"),
                            "recent_mean": round(recent_mean, 4),
                            "forecast_mean": round(forecast_mean, 4),
                            "method": forecast.get("method"),
                            "fallback_reason": forecast.get("fallback_reason"),
                        },
                        "recommended_investigation": "Review recent operational or seasonal changes that would explain the forecast deviation.",
                    })

    explanation = _as_dict(results.get("explanation"))
    if explanation.get("feature_importance"):
        top_features = explanation.get("feature_importance", [])[:3]
        if top_features:
            top_feature_names = [
                feature.get("feature") if isinstance(feature, dict) else str(feature)
                for feature in top_features
            ]
            findings.append({
                "signal": "feature_dominance",
                "evidence": {
                    "top_features": top_feature_names,
                    "feature_importance": top_features,
                },
                "recommended_investigation": "Check whether a single driver or recent feature change is influencing the metric swing.",
            })

    likely_causes = []
    for signal in (finding.get("signal") for finding in findings):
        if signal and signal not in likely_causes:
            likely_causes.append(signal)

    return {
        "status": "attention_required" if findings else "healthy",
        "findings": findings,
        "likely_causes": likely_causes,
        "basis": "Findings and evidence are derived only from deterministic monitoring tool outputs.",
    }