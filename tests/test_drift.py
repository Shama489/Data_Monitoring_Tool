import pandas as pd

from profiler import analyze_dataset_drift, classify_drift_severity


def test_drift_detection_returns_feature_metrics_and_severity():
    baseline = pd.DataFrame(
        {
            "age": [20, 22, 24, 26, 28, 30, 32, 34],
            "segment": ["A", "B", "A", "C", "B", "A", "C", "B"],
        }
    )
    current = pd.DataFrame(
        {
            "age": [28, 29, 31, 33, 35, 36, 38, 40],
            "segment": ["A", "A", "C", "C", "B", "B", "A", "C"],
        }
    )

    report = analyze_dataset_drift(baseline, current)

    assert report["baseline_rows"] == 8
    assert report["current_rows"] == 8
    assert "age" in report["feature_metrics"]
    assert "segment" in report["feature_metrics"]
    assert 0 <= report["feature_metrics"]["age"]["drift_score"] <= 100
    assert report["feature_metrics"]["age"]["severity"] in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert classify_drift_severity(0) == "low"
    assert classify_drift_severity(35) == "medium"
    assert classify_drift_severity(75) == "high"
    assert classify_drift_severity(95) == "critical"
