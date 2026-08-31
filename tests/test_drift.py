import pandas as pd

from profiler import (
    analyze_dataset_drift,
    calculate_data_quality_score,
    check_data_quality,
    classify_drift_severity,
    generate_ai_quality_summary,
)


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


def test_data_quality_and_ai_summary_are_generated():
    df = pd.DataFrame(
        {
            "age": [20, None, 22, 22],
            "segment": ["A", "A", "B", "B"],
            "score": [10, 11, None, 13],
        }
    )

    report = check_data_quality(df)
    quality_score = calculate_data_quality_score(df)
    ai_summary = generate_ai_quality_summary(df, report, quality_score)

    assert report["total_nulls"] >= 1
    assert 0 <= quality_score["overall_score"] <= 100
    assert "summary" in ai_summary
    assert "key_findings" in ai_summary
    assert "recommended_actions" in ai_summary


def test_generate_ai_quality_summary_uses_llm_when_available(monkeypatch):
    df = pd.DataFrame({"age": [20, None, 30], "score": [10, 12, None]})
    report = check_data_quality(df)
    quality_score = calculate_data_quality_score(df)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_llm_call(metrics):
        return {
            "summary": "AI-driven summary",
            "risk_level": "medium",
            "key_findings": ["LLM found data issues"],
            "recommended_actions": ["Use AI recommendations"],
        }

    monkeypatch.setattr("profiler._call_openai_quality_summary", fake_llm_call)

    ai_summary = generate_ai_quality_summary(df, report, quality_score, use_llm=True)

    assert ai_summary["summary"] == "AI-driven summary"
    assert ai_summary["risk_level"] == "medium"
    assert ai_summary["key_findings"] == ["LLM found data issues"]
