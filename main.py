from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

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
from notifications import NotificationError, send_notifications
from data_sources import DataSourceError, load_source, summarize_source

app = FastAPI(title="Data Monitoring Tool")


def _bad_request(message: str) -> None:
    raise HTTPException(status_code=400, detail=message)


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

    report = check_data_quality(
        df,
        expected_columns=payload.get("expected_columns"),
        expected_dtypes=payload.get("expected_dtypes"),
        timestamp_column=payload.get("timestamp_column"),
        max_age_hours=payload.get("max_age_hours"),
        similarity_threshold=payload.get("similarity_threshold", 0.8),
        rules=payload.get("rules"),
    )
    quality_score = calculate_data_quality_score(df)
    ai_summary = generate_ai_quality_summary(df, report, quality_score)
    return {
        "report": report,
        "quality_score": quality_score,
        "ai_summary": ai_summary,
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

    report = check_data_quality(
        df,
        expected_columns=payload.get("expected_columns"),
        expected_dtypes=payload.get("expected_dtypes"),
        timestamp_column=payload.get("timestamp_column"),
        max_age_hours=payload.get("max_age_hours"),
        similarity_threshold=payload.get("similarity_threshold", 0.8),
        rules=payload.get("rules"),
    )
    quality_score = calculate_data_quality_score(df)
    ai_summary = generate_ai_quality_summary(df, report, quality_score)
    return {
        "report": report,
        "quality_score": quality_score,
        "ai_summary": ai_summary,
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
    return {
        "report": report,
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
    return {"report": report}


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
        if payload.get("metric") == "data_health":
            report = forecast_data_health(df, date_column, periods, payload.get("frequency", "W"))
        else:
            report = forecast_metric(df, date_column, payload.get("value_column"), periods, payload.get("frequency", "W"), payload.get("method", "auto"))
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
