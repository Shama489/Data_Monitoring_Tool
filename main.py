from typing import Any

import pandas as pd
from fastapi import FastAPI

from profiler import analyze_dataset_drift

app = FastAPI(title="Data Monitoring Tool")


@app.get("/")
def home():
    return {"message": "Data Monitoring Tool Running"}


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "data-monitoring-tool"}


@app.post("/api/drift/analyze")
def analyze_drift_endpoint(payload: dict[str, Any]):
    baseline_data = payload.get("baseline") or payload.get("baseline_dataset")
    current_data = payload.get("current") or payload.get("current_dataset")

    if baseline_data is None or current_data is None:
        return {"error": "Both baseline and current datasets are required."}

    try:
        baseline_df = pd.DataFrame(baseline_data)
        current_df = pd.DataFrame(current_data)
    except Exception:
        return {"error": "Dataset payloads must be list-of-records or column-oriented JSON."}

    if baseline_df.empty or current_df.empty:
        return {"error": "Baseline and current datasets must not be empty."}

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
        return {"error": "CSV payloads are required for both datasets."}

    try:
        baseline_df = pd.read_csv(pd.io.common.StringIO(baseline_csv))
        current_df = pd.read_csv(pd.io.common.StringIO(current_csv))
    except Exception:
        return {"error": "Invalid CSV content for one or both datasets."}

    report = analyze_dataset_drift(baseline_df, current_df)
    return {"report": report}
