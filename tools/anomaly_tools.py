"""Anomaly-analysis tool used by the monitoring workflow."""

import pandas as pd

from profiler import detect_anomalies_isolation_forest


def analyze_anomalies(frame: pd.DataFrame) -> dict:
    report = detect_anomalies_isolation_forest(frame)
    if report is None:
        return {
            "available": False,
            "total_anomalies": 0,
            "anomaly_percentage": 0.0,
            "indices": [],
            "reason": "At least ten rows and one numeric column are required.",
        }
    return {"available": True, **report}
