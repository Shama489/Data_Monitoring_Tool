import hashlib
import warnings

import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from backend import get_table_data
from profiler import (
    analyze_dataset_drift,
    analyze_trends,
    answer_monitoring_question,
    calculate_data_quality_score,
    check_data_quality,
    classify_drift_severity,
    analyze_clusters,
    analyze_predictive_maintenance,
    compare_models,
    detect_fraud_patterns,
    detect_anomalies_isolation_forest,
    forecast_data_health,
    forecast_metric,
    generate_ai_quality_summary,
    recommend_items,
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


def test_monitoring_assistant_answers_quality_and_drift_questions():
    df = pd.DataFrame(
        {
            "age": [20, None, 22, 24, 26, 30, 32, 35],
            "region": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "score": [10, 12, None, 14, 15, 16, 17, 18],
        }
    )
    baseline = pd.DataFrame({"age": [20, 22, 24, 26, 28, 30], "region": ["A", "B", "A", "B", "A", "B"]})
    quality_report = check_data_quality(df)
    drift_report = analyze_dataset_drift(baseline, df)
    anomalies = detect_anomalies_isolation_forest(df)

    quality_answer = answer_monitoring_question(
        "How many missing values do we have in the dataset?",
        df=df,
        quality_report=quality_report,
        drift_report=drift_report,
        anomaly_report=anomalies,
    )
    drift_answer = answer_monitoring_question(
        "Is there drift in the monitoring data?",
        df=df,
        quality_report=quality_report,
        drift_report=drift_report,
        anomaly_report=anomalies,
    )

    assert quality_answer["category"] == "quality"
    assert "2" in quality_answer["answer"] or "two" in quality_answer["answer"].lower()
    assert drift_answer["category"] == "drift"
    assert "yes" in drift_answer["answer"].lower() or "drift detected" in drift_answer["answer"].lower()


@pytest.mark.parametrize(
    ("language", "question", "intent", "expected"),
    [
        ("es", "¿Cuál es la puntuación de calidad de datos de hoy?", "quality", "/100"),
        ("es", "Muestra las columnas con valores faltantes", "quality", "faltantes"),
        ("fr", "Montre le rapport d'anomalies", "anomaly", "anomalies"),
        ("fr", "Affiche les colonnes avec des valeurs manquantes", "quality", "manquantes"),
        ("hi", "आज डेटा गुणवत्ता स्कोर क्या है?", "quality", "/100"),
        ("hi", "खाली मान वाले कॉलम दिखाएं", "quality", "कॉलम"),
    ],
)
def test_monitoring_assistant_answers_supported_languages(language, question, intent, expected):
    df = pd.DataFrame({
        "sensor": [1, None, 3],
        "status": ["ok", "bad", None],
    })
    result = answer_monitoring_question(
        question,
        df=df,
        anomaly_report={"total_anomalies": 1, "anomaly_percentage": 33.3},
        language=language,
    )

    assert result["category"] == intent
    assert expected in result["answer"]


def test_monitoring_assistant_rejects_unsupported_language():
    with pytest.raises(ValueError, match="language must be one of"):
        answer_monitoring_question("show anomalies", language="de")


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


def test_forecast_metric_uses_lstm_forecast_when_available(monkeypatch):
    dates = pd.date_range("2025-01-01", periods=12, freq="W")
    df = pd.DataFrame({"event_date": dates, "amount": range(12)})
    monkeypatch.setattr("profiler._forecast_lstm", lambda _series, periods: [42.0] * periods)

    report = forecast_metric(df, "event_date", "amount", periods=2, method="lstm")

    assert report["method"] == "LSTM"
    assert report["fallback_reason"] is None
    assert [item["value"] for item in report["forecast"]] == [42.0, 42.0]


def test_forecast_metric_reports_lstm_runtime_fallback(monkeypatch):
    dates = pd.date_range("2025-01-01", periods=12, freq="W")
    df = pd.DataFrame({"event_date": dates, "amount": range(12)})

    def missing_runtime(_series, _periods):
        raise ImportError("TensorFlow is unavailable")

    monkeypatch.setattr("profiler._forecast_lstm", missing_runtime)
    report = forecast_metric(df, "event_date", "amount", periods=2, method="lstm")

    assert report["method"] == "linear"
    assert "TensorFlow is unavailable" in report["fallback_reason"]


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


def test_model_explanations_include_classification_holdout_metrics(monkeypatch):
    monkeypatch.setattr("profiler.shap", None)
    df = pd.DataFrame({
        "signal": list(range(30)),
        "segment": ["A", "B", "C"] * 10,
        "target": [0, 1] * 15,
    })

    report = train_and_explain_model(df, "target", test_size=0.3)

    assert report["test_size"] == 0.3
    assert report["holdout"]["rows"] == 9
    assert set(report["holdout"]["metrics"]) >= {
        "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "confusion_matrix"
    }
    assert report["holdout"]["predictions"]


def test_model_explanations_include_regression_holdout_metrics(monkeypatch):
    monkeypatch.setattr("profiler.shap", None)
    df = pd.DataFrame({
        "signal": list(range(20)),
        "target": [value * 2 + 1 for value in range(20)],
    })

    report = train_and_explain_model(df, "target", task="regression")

    assert set(report["holdout"]["metrics"]) == {"mae", "rmse", "r2"}
    assert report["holdout"]["metrics"]["mae"] >= 0


def test_model_explanations_reject_invalid_holdout_share():
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 0, 1, 1]})

    with pytest.raises(ValueError, match="test_size"):
        train_and_explain_model(df, "target", test_size=0.8)


def test_automl_regression_supports_current_sklearn_rmse_api():
    df = pd.DataFrame({
        "signal": list(range(20)),
        "target": [value * 2 + 1 for value in range(20)],
    })

    report = compare_models(
        df,
        "target",
        task="regression",
        metric="mean_squared_error",
        model_names=["linear_regression"],
    )

    assert report["direction"] == "lower"
    assert report["models"][0]["metrics"]["rmse"] >= 0
    assert report["model_failures"] == []


def test_automl_rejects_unknown_metric():
    df = pd.DataFrame({"signal": list(range(10)), "target": [0, 1] * 5})

    with pytest.raises(ValueError, match="Unsupported classification metric"):
        compare_models(df, "target", metric="unknown")


def test_multiclass_linear_feature_importance_matches_feature_names():
    from sklearn.linear_model import LogisticRegression

    from profiler import _feature_importance_for_model

    model = LogisticRegression(max_iter=1000).fit(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
        ["a", "b", "c", "a", "b", "c"],
    )

    importance = _feature_importance_for_model(model, ["first", "second"])

    assert {item["feature"] for item in importance} == {"first", "second"}


def test_advanced_clustering_reports_assignments_and_cluster_profiles():
    df = pd.DataFrame({
        "temperature": [10, 10.2, 9.8, 10.1, 10.3, 9.9, 90, 90.2, 89.8, 90.1, 90.3, 89.9],
        "machine": ["A"] * 6 + ["B"] * 6,
    })

    report = analyze_clusters(df, n_clusters=2)

    assert report["analysis"] == "clustering"
    assert report["actual_clusters"] == 2
    assert sum(cluster["rows"] for cluster in report["clusters"]) == len(df)
    assert len(report["assignments"]) == len(df)
    assert report["silhouette_score"] > 0


def test_fraud_detection_flags_outlier_and_describes_caveat():
    df = pd.DataFrame({
        "amount": list(range(39)) + [10000],
        "duration": [1] * 39 + [100],
    })

    report = detect_fraud_patterns(df, contamination=0.05)

    assert report["analysis"] == "fraud_detection"
    assert report["total_flagged"] >= 1
    assert any(row["row_index"] == "39" for row in report["flagged_rows"])
    assert "not confirmed fraud" in report["interpretation"]


def test_predictive_maintenance_uses_automl_model_selection():
    df = pd.DataFrame({
        "temperature": list(range(30)),
        "failure": [int(value >= 20) for value in range(30)],
    })

    report = analyze_predictive_maintenance(
        df,
        "failure",
        model_names=["decision_tree"],
    )

    assert report["analysis"] == "predictive_maintenance"
    assert report["selected_model"] == "decision_tree"
    assert report["target_interpretation"] == "Failure/event classification"
    assert report["models"][0]["metrics"]["accuracy"] >= 0


def test_recommendations_exclude_seen_items_and_use_item_similarity():
    df = pd.DataFrame({
        "user": ["u1", "u1", "u2", "u2", "u2", "u3", "u3"],
        "item": ["book-a", "book-b", "book-a", "book-b", "book-c", "book-b", "book-c"],
        "rating": [5, 4, 5, 4, 5, 5, 5],
    })

    report = recommend_items(
        df,
        "user",
        "item",
        rating_column="rating",
        user_id="u1",
    )

    assert report["method"] == "item_similarity"
    assert report["recommendations"]
    assert all(item["item"] not in {"book-a", "book-b"} for item in report["recommendations"])
    assert report["recommendations"][0]["item"] == "book-c"


def test_recommendations_use_popularity_for_cold_start():
    df = pd.DataFrame({
        "user": ["u1", "u2", "u3"],
        "item": ["popular", "popular", "other"],
    })

    report = recommend_items(df, "user", "item", user_id="u-new")

    assert report["method"] == "popularity"
    assert report["recommendations"][0]["item"] == "popular"


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


def test_near_duplicate_detection_scales_candidate_search_for_large_frames():
    def token(prefix, index):
        return hashlib.sha256(f"{prefix}:{index}".encode()).hexdigest()[:16]

    names = [token("name", index) for index in range(510)]
    emails = [token("email", index) for index in range(510)]
    regions = [token("region", index) for index in range(510)]
    names[1] = names[0][:-1] + ("0" if names[0][-1] != "0" else "1")
    emails[1] = emails[0]
    regions[1] = regions[0]
    df = pd.DataFrame({"name": names, "email": emails, "region": regions})

    report = check_data_quality(df)
    duplicate_details = report["duplicate_detection"]

    assert duplicate_details["candidate_generation"] == "minhash_lsh"
    assert duplicate_details["candidate_pairs_checked"] < duplicate_details["candidate_pairs_total"]
    assert duplicate_details["near_duplicate_pairs"] >= 1


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
