from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from profiler import (
    analyze_dataset_drift,
    analyze_trends,
    calculate_data_quality_score,
    check_data_quality,
    forecast_data_health,
    forecast_metric,
    generate_ai_quality_summary,
    train_and_explain_model,
)
from notifications import (
    ALERT_RULES,
    NotificationError,
    evaluate_alert_rules,
    get_alert_history,
    send_notifications,
)
from data_sources import DataSourceError, load_source, summarize_source

app = FastAPI(title="Data Monitoring Tool")


def _bad_request(message: str) -> None:
    raise HTTPException(status_code=400, detail=message)


def _quality_options(payload: dict[str, Any]) -> dict[str, Any]:
    rules = payload.get("rules")
    if rules is not None and not isinstance(rules, (dict, list)):
        _bad_request("rules must be an object or list of rule objects")
    if isinstance(rules, list) and not all(isinstance(rule, dict) for rule in rules):
        _bad_request("rules list entries must be objects")

    expected_columns = payload.get("expected_columns")
    if expected_columns is not None and not isinstance(expected_columns, list):
        _bad_request("expected_columns must be a list")

    expected_dtypes = payload.get("expected_dtypes")
    if expected_dtypes is not None and not isinstance(expected_dtypes, dict):
        _bad_request("expected_dtypes must be an object")

    return {
        "expected_columns": expected_columns,
        "expected_dtypes": expected_dtypes,
        "timestamp_column": payload.get("timestamp_column"),
        "max_age_hours": payload.get("max_age_hours"),
        "similarity_threshold": payload.get("similarity_threshold", 0.8),
        "rules": rules,
    }


def _use_llm_quality_summary(payload: dict[str, Any]) -> bool:
    use_llm = payload.get("use_llm", False)
    if not isinstance(use_llm, bool):
        _bad_request("use_llm must be a boolean")
    return use_llm


def _notification_channels(payload: dict[str, Any]) -> list[Any]:
    for key in ("notifications", "alert_channels", "channels"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return value
        if isinstance(value, (str, dict)):
            return [value]
    return []


def _should_send_notifications(payload: dict[str, Any]) -> bool:
    return bool(payload.get("notify", payload.get("send_notifications", False))) or bool(_notification_channels(payload))


def _trigger_monitoring_notifications(
    payload: dict[str, Any],
    event_type: str,
    message: str,
    subject: str,
    severity: str = "warning",
) -> dict[str, Any]:
    channels = _notification_channels(payload)
    if not channels:
        return {"status": "skipped", "reason": "no notification channels configured"}

    alert = {
        "message": message,
        "subject": subject,
        "severity": severity,
        "event_type": event_type,
        "channels": channels,
    }
    try:
        return send_notifications(alert)
    except NotificationError as error:
        return {"status": "failed", "error": str(error)}


def _send_rule_based_notifications(payload: dict[str, Any], event_type: str, metrics: dict[str, Any], subject: str) -> dict[str, Any]:
    alert_rules = payload.get("alert_rules", ALERT_RULES)
    if not isinstance(alert_rules, list) or not alert_rules:
        return {"status": "skipped", "reason": "no alert rules configured"}

    rule_result = evaluate_alert_rules(metrics, alert_rules)
    if not rule_result["triggered"]:
        return {"status": "skipped", "reason": "no alert rules matched"}

    channels = []
    for match in rule_result["matches"]:
        channel = match.get("channel")
        recipient = match.get("recipient", "")
        if channel:
            channels.append({"channel": channel, "recipient": recipient})

    if not channels:
        return {"status": "skipped", "reason": "matched rules had no delivery channels"}

    summary = "; ".join(f"{m['metric']}={m['actual']}" for m in rule_result["matches"])
    try:
        return send_notifications({
            "message": f"{event_type.replace('_', ' ').title()} alert: {summary}",
            "subject": subject,
            "severity": max((metrics.get("risk_level", "warning"), metrics.get("severity", "warning")), key=lambda value: str(value)),
            "event_type": event_type,
            "channels": channels,
        })
    except NotificationError as error:
        return {"status": "failed", "error": str(error)}


@app.get("/")
def home():
    return {"message": "Data Monitoring Tool Running"}


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "data-monitoring-tool"}


@app.post("/api/sources/analyze")
def analyze_sources_endpoint(payload: dict[str, Any]):
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        _bad_request("sources must be a non-empty list")

    reports = []
    failures = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            failures.append({"index": index, "error": "Each source must be an object"})
            continue
        try:
            frame = load_source(source)
            if frame.empty:
                raise DataSourceError("Source dataset must not be empty")
            reports.append(summarize_source(source, frame))
        except (DataSourceError, TypeError, ValueError) as error:
            failures.append({"index": index, "type": source.get("type"), "error": str(error)})

    if not reports and failures:
        _bad_request(failures[0]["error"])
    return {"sources": reports, "failed": failures, "success": not failures}


@app.post("/api/notifications/send")
def send_notifications_endpoint(payload: dict[str, Any]):
    """Send one monitoring alert to email, SMS, WhatsApp, Slack, or Teams."""
    try:
        return send_notifications(payload)
    except NotificationError as error:
        _bad_request(str(error))


@app.get("/api/alerts/rules")
def list_alert_rules_endpoint():
    return {"rules": ALERT_RULES}


@app.post("/api/alerts/rules")
def upsert_alert_rules_endpoint(payload: dict[str, Any]):
    if "rules" in payload:
        incoming = payload["rules"]
    else:
        incoming = [payload]

    if not isinstance(incoming, list):
        _bad_request("rules must be a list")

    valid_rules = []
    for rule in incoming:
        if not isinstance(rule, dict):
            _bad_request("each rule must be an object")
        metric = str(rule.get("metric", "")).strip()
        if not metric:
            _bad_request("rule metric is required")
        valid_rules.append({
            "metric": metric,
            "operator": str(rule.get("operator", "==")),
            "value": rule.get("value"),
            "channel": rule.get("channel"),
            "recipient": rule.get("recipient", ""),
            "subject": rule.get("subject", "Data monitoring alert"),
            "message": rule.get("message", f"{metric} matched alert rule"),
        })

    ALERT_RULES[:] = valid_rules
    return {"rules": ALERT_RULES, "count": len(ALERT_RULES)}


@app.delete("/api/alerts/rules")
def clear_alert_rules_endpoint():
    ALERT_RULES.clear()
    return {"rules": ALERT_RULES}


@app.get("/api/alerts/history")
def list_alert_history_endpoint(
    limit: int = Query(default=50, ge=1, le=500),
    status: str | None = None,
    event_type: str | None = None,
):
    return {"history": get_alert_history(limit=limit, status=status, event_type=event_type)}


@app.post("/api/data-quality/analyze")
def analyze_data_quality_endpoint(payload: dict[str, Any]):
    dataset = payload.get("data") or payload.get("dataset") or payload.get("records")
    if dataset is None:
        _bad_request("Dataset payload is required.")

    try:
        df = pd.DataFrame(dataset)
    except Exception:
        _bad_request("Dataset payload must be list-of-records or column-oriented JSON.")

    if df.empty:
        _bad_request("Dataset must not be empty.")

    report = check_data_quality(df, **_quality_options(payload))
    quality_score = calculate_data_quality_score(df)
    ai_summary = generate_ai_quality_summary(
        df, report, quality_score, use_llm=_use_llm_quality_summary(payload)
    )
    notification_result = None
    if _should_send_notifications(payload):
        risk_level = ai_summary.get("risk_level", "unknown")
        summary_text = ai_summary.get("summary", "Data quality review completed.")
        notification_result = _trigger_monitoring_notifications(
            payload,
            "data_quality",
            f"Data quality alert: score={quality_score.get('overall_score', 0)}; risk={risk_level}; details={summary_text}",
            "Data Monitoring Alert",
            severity=risk_level,
        )
    rule_result = _send_rule_based_notifications(
        payload,
        "data_quality",
        {
            "quality_score": quality_score.get("overall_score", 0),
            "risk_level": ai_summary.get("risk_level", "unknown"),
            "severity": ai_summary.get("risk_level", "unknown"),
            "total_nulls": report.get("total_nulls", 0),
            "duplicates": report.get("duplicates", 0),
        },
        "Data Monitoring Alert",
    )
    if rule_result.get("status") != "skipped":
        notification_result = rule_result
    return {
        "report": report,
        "quality_score": quality_score,
        "ai_summary": ai_summary,
        "notifications": notification_result or {"status": "skipped", "reason": "notifications not requested"},
        "message": "Data quality analysis completed successfully.",
    }


@app.post("/api/data-quality/analyze-csv")
def analyze_data_quality_csv_endpoint(payload: dict[str, Any]):
    csv_content = payload.get("csv") or payload.get("dataset_csv")
    if csv_content is None:
        _bad_request("CSV payload is required.")

    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_content))
    except Exception:
        _bad_request("Invalid CSV content.")

    if df.empty:
        _bad_request("CSV dataset must not be empty.")

    report = check_data_quality(df, **_quality_options(payload))
    quality_score = calculate_data_quality_score(df)
    ai_summary = generate_ai_quality_summary(
        df, report, quality_score, use_llm=_use_llm_quality_summary(payload)
    )
    notification_result = None
    if _should_send_notifications(payload):
        risk_level = ai_summary.get("risk_level", "unknown")
        summary_text = ai_summary.get("summary", "Data quality review completed.")
        notification_result = _trigger_monitoring_notifications(
            payload,
            "data_quality_csv",
            f"CSV data quality alert: score={quality_score.get('overall_score', 0)}; risk={risk_level}; details={summary_text}",
            "CSV Data Monitoring Alert",
            severity=risk_level,
        )
    rule_result = _send_rule_based_notifications(
        payload,
        "data_quality_csv",
        {
            "quality_score": quality_score.get("overall_score", 0),
            "risk_level": ai_summary.get("risk_level", "unknown"),
            "severity": ai_summary.get("risk_level", "unknown"),
            "total_nulls": report.get("total_nulls", 0),
            "duplicates": report.get("duplicates", 0),
        },
        "CSV Data Monitoring Alert",
    )
    if rule_result.get("status") != "skipped":
        notification_result = rule_result
    return {
        "report": report,
        "quality_score": quality_score,
        "ai_summary": ai_summary,
        "notifications": notification_result or {"status": "skipped", "reason": "notifications not requested"},
    }


@app.post("/api/drift/analyze")
def analyze_drift_endpoint(payload: dict[str, Any]):
    baseline_data = payload.get("baseline") or payload.get("baseline_dataset")
    current_data = payload.get("current") or payload.get("current_dataset")

    if baseline_data is None or current_data is None:
        _bad_request("Both baseline and current datasets are required.")

    try:
        baseline_df = pd.DataFrame(baseline_data)
        current_df = pd.DataFrame(current_data)
    except Exception:
        _bad_request("Dataset payloads must be list-of-records or column-oriented JSON.")

    if baseline_df.empty or current_df.empty:
        _bad_request("Baseline and current datasets must not be empty.")

    report = analyze_dataset_drift(baseline_df, current_df)
    notification_result = None
    if _should_send_notifications(payload):
        drift_score = report.get("overall_drift_score", 0)
        message = (
            f"Drift alert: overall drift score={drift_score}; "
            f"severity={report.get('overall_severity', 'low')}; "
            f"detected={report.get('drift_detected', False)}"
        )
        notification_result = _trigger_monitoring_notifications(
            payload,
            "data_drift",
            message,
            "Drift Detection Alert",
            severity=report.get("overall_severity", "low"),
        )
    rule_result = _send_rule_based_notifications(
        payload,
        "data_drift",
        {
            "drift_score": report.get("overall_drift_score", 0),
            "severity": report.get("overall_severity", "low"),
            "drift_detected": report.get("drift_detected", False),
        },
        "Drift Detection Alert",
    )
    if rule_result.get("status") != "skipped":
        notification_result = rule_result
    return {
        "report": report,
        "notifications": notification_result or {"status": "skipped", "reason": "notifications not requested"},
        "message": "Drift analysis completed successfully.",
    }


@app.post("/api/drift/analyze-csv")
def analyze_drift_csv_endpoint(payload: dict[str, Any]):
    baseline_csv = payload.get("baseline_csv")
    current_csv = payload.get("current_csv")

    if baseline_csv is None or current_csv is None:
        _bad_request("CSV payloads are required for both datasets.")

    try:
        baseline_df = pd.read_csv(pd.io.common.StringIO(baseline_csv))
        current_df = pd.read_csv(pd.io.common.StringIO(current_csv))
    except Exception:
        _bad_request("Invalid CSV content for one or both datasets.")

    if baseline_df.empty or current_df.empty:
        _bad_request("Baseline and current CSV datasets must not be empty.")

    try:
        report = analyze_dataset_drift(baseline_df, current_df)
    except (TypeError, ValueError) as error:
        _bad_request(str(error))

    notification_result = None
    if _should_send_notifications(payload):
        drift_score = report.get("overall_drift_score", 0)
        message = (
            f"CSV drift alert: overall drift score={drift_score}; "
            f"severity={report.get('overall_severity', 'low')}; "
            f"detected={report.get('drift_detected', False)}"
        )
        notification_result = _trigger_monitoring_notifications(
            payload,
            "data_drift_csv",
            message,
            "CSV Drift Detection Alert",
            severity=report.get("overall_severity", "low"),
        )
    rule_result = _send_rule_based_notifications(
        payload,
        "data_drift_csv",
        {
            "drift_score": report.get("overall_drift_score", 0),
            "severity": report.get("overall_severity", "low"),
            "drift_detected": report.get("drift_detected", False),
        },
        "CSV Drift Detection Alert",
    )
    if rule_result.get("status") != "skipped":
        notification_result = rule_result
    return {
        "report": report,
        "notifications": notification_result or {"status": "skipped", "reason": "notifications not requested"},
    }


@app.post("/api/analytics/trends")
def analyze_trends_endpoint(payload: dict[str, Any]):
    dataset = payload.get("data") or payload.get("dataset") or payload.get("records")
    date_column = payload.get("date_column")
    if dataset is None or not date_column:
        _bad_request("Dataset and date_column are required.")
    try:
        return {"report": analyze_trends(pd.DataFrame(dataset), date_column, payload.get("value_column"))}
    except (TypeError, ValueError) as error:
        _bad_request(str(error))


@app.post("/api/analytics/forecast")
def forecast_endpoint(payload: dict[str, Any]):
    dataset = payload.get("data") or payload.get("dataset") or payload.get("records")
    date_column = payload.get("date_column")
    if dataset is None or not date_column:
        _bad_request("Dataset and date_column are required.")
    try:
        df = pd.DataFrame(dataset)
        periods = int(payload.get("periods", 4))
        if periods < 1 or periods > 365:
            _bad_request("periods must be between 1 and 365.")

        method = str(payload.get("method", "auto")).lower()
        allowed_methods = {"auto", "arima", "linear", "prophet", "lstm"}
        if method not in allowed_methods:
            _bad_request("method must be one of: auto, arima, linear, prophet, lstm.")

        frequency = str(payload.get("frequency", "W"))
        if not frequency:
            _bad_request("frequency is required.")

        if payload.get("metric") == "data_health":
            report = forecast_data_health(df, date_column, periods, frequency)
        else:
            report = forecast_metric(df, date_column, payload.get("value_column"), periods, frequency, method)
        return {"report": report}
    except (TypeError, ValueError) as error:
        _bad_request(str(error))


@app.post("/api/analytics/explain")
def explain_model_endpoint(payload: dict[str, Any]):
    dataset = payload.get("data") or payload.get("dataset") or payload.get("records")
    target_column = payload.get("target_column")
    if dataset is None or not target_column:
        _bad_request("Dataset and target_column are required.")
    try:
        report = train_and_explain_model(pd.DataFrame(dataset), target_column, payload.get("task", "classification"))
        return {"report": report}
    except (TypeError, ValueError) as error:
        _bad_request(str(error))
