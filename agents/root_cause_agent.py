from typing import Any


def investigate(results: dict[str, Any]) -> dict[str, Any]:
    findings = []
    quality = results.get("quality", {})
    quality_report = quality.get("report", {})

    total_nulls = int(quality_report.get("total_nulls", 0) or 0)
    if total_nulls:
        total_cells = max(int(quality_report.get("rows", 0)) * int(quality_report.get("columns", 0)), 1)
        findings.append({
            "signal": "missing_values",
            "evidence": {
                "total_nulls": total_nulls,
                "rate_percent": round(total_nulls / total_cells * 100, 2),
            },
            "recommended_investigation": "Inspect the affected columns and the latest ingestion or transformation step.",
        })

    duplicates = int(quality_report.get("duplicates", 0) or 0)
    if duplicates:
        findings.append({
            "signal": "duplicate_rows",
            "evidence": {"exact_duplicates": duplicates},
            "recommended_investigation": "Check source identifiers and deduplication behavior in the ingestion pipeline.",
        })

    schema = quality_report.get("schema_validation", {})
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

    freshness = quality_report.get("data_freshness", {})
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

    rules = quality_report.get("business_rule_validation", {})
    violations = int(rules.get("violations", 0) or 0)
    if violations:
        findings.append({
            "signal": "business_rule_violations",
            "evidence": {
                "violations": violations,
                "rules": [
                    {"id": rule.get("id"), "violations": rule.get("violations", 0)}
                    for rule in rules.get("rules", [])
                    if rule.get("violations", 0)
                ],
            },
            "recommended_investigation": "Review the failing rule definitions and the corresponding source records.",
        })

    drift = results.get("drift", {})
    if drift.get("drift_detected"):
        features = drift.get("feature_metrics", {})
        affected_features = []
        for name, metric in features.items():
            if metric.get("drift_score", 0) >= 25:
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

    return {
        "status": "attention_required" if findings else "healthy",
        "findings": findings,
        "basis": "Findings and evidence are derived only from deterministic monitoring tool outputs.",
    }