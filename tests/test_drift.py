import warnings

import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from backend import get_table_data
from profiler import (
    analyze_dataset_drift,
    analyze_trends,
    calculate_data_quality_score,
    check_data_quality,
    classify_drift_severity,
    forecast_data_health,
    forecast_metric,
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


def test_numeric_psi_includes_baseline_empty_bins():
    baseline = pd.DataFrame({"value": [0, 0, 0, 1, 1, 1]})
    current = pd.DataFrame({"value": [0.5, 0.5, 0.5, 0.5]})

    report = analyze_dataset_drift(baseline, current)

    assert report["feature_metrics"]["value"]["psi"] > 0


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


def test_trend_analysis_includes_direction_growth_and_moving_average():
    dates = pd.date_range("2025-01-01", periods=8, freq="W")
    df = pd.DataFrame({"event_date": dates, "amount": range(8)})

    report = analyze_trends(df, "event_date", "amount")

    assert report["trend_direction"] == "increasing"
    assert report["growth_rate_percent"] > 0
    assert report["moving_average"]
    assert "volatility" in report


def test_forecast_metric_ignores_nonfatal_arima_convergence_warnings():
    dates = pd.date_range("2025-01-01", periods=12, freq="W")
    df = pd.DataFrame({"event_date": dates, "amount": range(12)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = forecast_metric(df, "event_date", "amount", periods=2, method="auto")

    assert report["forecast"]
    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)


def test_forecast_metric_supports_advanced_method_names():
    dates = pd.date_range("2025-01-01", periods=12, freq="W")
    df = pd.DataFrame({"event_date": dates, "amount": range(12)})

    report = forecast_metric(df, "event_date", "amount", periods=2, method="prophet")

    assert report["forecast"]
    assert report["requested_method"] == "prophet"
    assert report["method"] in {"Prophet", "linear", "ARIMA"}


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
    assert "explanation_summary" in explanation
    assert explanation["explanation_summary"]["top_features"]


def test_get_table_data_rejects_invalid_table_names():
    with pytest.raises(ValueError, match="table name"):
        get_table_data("users; DROP TABLE accounts;")


def test_generate_ai_quality_summary_warns_when_llm_fails(monkeypatch):
    df = pd.DataFrame({"age": [20, None, 30], "score": [10, 12, None]})
    report = check_data_quality(df)
    quality_score = calculate_data_quality_score(df)

    def fake_failure(_metrics):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("profiler._call_openai_quality_summary", fake_failure)

    with pytest.warns(RuntimeWarning, match="falling back"):
        summary = generate_ai_quality_summary(df, report, quality_score, use_llm=True)

    assert "summary" in summary
    assert "risk_level" in summary


def test_data_quality_report_includes_schema_freshness_and_rule_validation():
    df = pd.DataFrame(
        {
            "name": ["Alice", "Alice", "Bob", "Charlie"],
            "age": [30, 30, -1, 45],
            "email": ["alice@example.com", "alice@example.com", "bob@x", "bad-email"],
            "created_at": [
                pd.Timestamp("2025-01-01T00:00:00"),
                pd.Timestamp("2025-01-01T00:00:00"),
                pd.Timestamp("2025-01-01T00:00:00"),
                pd.Timestamp("2025-01-01T00:00:00"),
            ],
        }
    )

    report = check_data_quality(
        df,
        expected_columns=["name", "age", "email", "created_at"],
        timestamp_column="created_at",
        max_age_hours=1,
        rules={"age": ">= 0", "email": "contains @"},
    )

    assert report["schema_validation"]["status"] in {"ok", "warning"}
    assert report["duplicate_detection"]["exact_duplicates"] >= 1
    assert report["duplicate_detection"]["near_duplicate_pairs"] >= 0
    assert report["data_freshness"]["is_fresh"] is False
    assert report["business_rule_validation"]["violations"] >= 2


def test_near_duplicate_detection_flags_similar_rows():
    df = pd.DataFrame(
        {
            "name": ["Alice Johnson", "Alice Jhonson", "Bob Smith", "Carol Jones"],
            "email": ["alice@x.com", "alice@x.com", "bob@x.com", "carol@x.com"],
            "age": [30, 30, 25, 40],
        }
    )

    report = check_data_quality(df, similarity_threshold=0.8)

    assert report["duplicate_detection"]["near_duplicate_pairs"] >= 1


def test_schema_validation_checks_types_and_suggests_renamed_columns():
    df = pd.DataFrame({"customer_nme": ["Alice", "Bob"], "age": [30, 31]})

    report = check_data_quality(
        df,
        expected_columns=["customer_name", "age"],
        expected_dtypes={"customer_name": "object", "age": "int64"},
    )

    schema = report["schema_validation"]
    assert schema["type_issues"] == []
    assert schema["renamed_columns"] == [
        {"expected": "customer_name", "actual": "customer_nme", "confidence": 0.96}
    ]
    assert schema["status"] == "warning"


def test_schema_validation_reports_datatype_mismatch():
    df = pd.DataFrame({"age": ["30", "31"]})

    report = check_data_quality(df, expected_dtypes={"age": "int64"})

    assert report["schema_validation"]["type_issues"] == [
        {"column": "age", "expected": "int64", "actual": "object"}
    ]
    assert report["schema_validation"]["status"] == "warning"


def test_business_rule_engine_supports_multiple_operators_and_row_details():
    df = pd.DataFrame(
        {
            "age": [20, -1, 42],
            "email": ["valid@example.com", "invalid", "also@example.com"],
            "status": ["active", "deleted", "pending"],
        }
    )

    report = check_data_quality(
        df,
        rules=[
            {"id": "age_range", "column": "age", "operator": "between", "value": [0, 120]},
            {"id": "email_format", "column": "email", "operator": "regex", "value": r"^[^@]+@[^@]+\.[^@]+$"},
            {"id": "allowed_status", "column": "status", "operator": "in", "value": ["active", "pending"], "severity": "warning"},
        ],
    )

    rules_by_id = {item["id"]: item for item in report["business_rule_validation"]["rules"]}
    assert report["business_rule_validation"]["violations"] == 3
    assert rules_by_id["age_range"]["failed_rows"] == [1]
    assert rules_by_id["email_format"]["failed_rows"] == [1]
    assert rules_by_id["allowed_status"]["severity"] == "warning"
