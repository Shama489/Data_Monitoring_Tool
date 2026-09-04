import pandas as pd

from profiler import (
    analyze_dataset_drift,
    analyze_trends,
    calculate_data_quality_score,
    check_data_quality,
    classify_drift_severity,
    forecast_data_health,
    generate_ai_quality_summary,
    train_and_explain_model,
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


def test_drift_detection_flags_new_categories():
    baseline = pd.DataFrame({"segment": ["A", "A", "B", "B"]})
    current = pd.DataFrame({"segment": ["C", "C", "C", "C"]})

    report = analyze_dataset_drift(baseline, current)

    metric = report["feature_metrics"]["segment"]
    assert metric["psi"] > 0
    assert metric["drift_score"] >= 25
    assert report["drift_detected"] is True


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


def test_trends_and_health_forecasts_are_generated():
    dates = pd.date_range("2025-01-01", periods=12, freq="W")
    df = pd.DataFrame(
        {
            "event_date": dates,
            "amount": range(12),
            "status": ["ok", "bad"] * 6,
        }
    )

    trends = analyze_trends(df, "event_date", "amount")
    health = forecast_data_health(df, "event_date", periods=2)

    assert trends["weekly"]
    assert trends["monthly"]
    assert trends["seasonal_patterns"]
    assert set(health) == {"volume", "missing_values", "quality_score"}
    assert all(len(item["forecast"]) == 2 for item in health.values())


def test_model_explanations_return_ranked_feature_importance():
    df = pd.DataFrame(
        {
            "age": [20, 21, 30, 31, 40, 41],
            "segment": ["A", "A", "B", "B", "C", "C"],
            "target": [0, 0, 1, 1, 1, 1],
        }
    )

    explanation = train_and_explain_model(df, "target")

    assert explanation["target"] == "target"
    assert explanation["feature_importance"]
    importances = [item["importance"] for item in explanation["feature_importance"]]
    assert importances == sorted(importances, reverse=True)
    assert "shap_available" in explanation
