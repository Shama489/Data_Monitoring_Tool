from agents.root_cause_agent import investigate


def test_central_recommendations_prioritize_findings_across_monitoring_modules():
    root_cause = investigate({
        "quality": {
            "report": {
                "rows": 10,
                "columns": 2,
                "total_nulls": 2,
                "duplicates": 1,
                "schema_validation": {
                    "status": "warning",
                    "missing_columns": ["source_id"],
                },
                "data_freshness": {
                    "is_fresh": False,
                    "age_hours": 48,
                    "max_age_hours": 24,
                },
                "business_rule_validation": {
                    "violations": 1,
                    "rules": [{"id": "positive-value", "violations": 1}],
                },
            },
        },
        "drift": {
            "drift_detected": True,
            "overall_drift_score": 90,
            "overall_severity": "critical",
            "feature_metrics": {},
        },
        "anomalies": {
            "available": True,
            "total_anomalies": 2,
            "indices": [8, 9],
        },
        "forecast": {
            "history": [{"value": 10}],
            "forecast": [{"value": 20}],
            "metric": "volume",
        },
        "explanation": {
            "feature_importance": [{"feature": "volume", "importance": 1.0}],
        },
    })

    recommendations = root_cause["recommendations"]
    signals = [item["signal"] for item in recommendations]

    assert root_cause["status"] == "attention_required"
    assert signals[0] == "distribution_drift"
    assert recommendations[0]["priority"] == "critical"
    assert "anomaly_spike" in signals
    assert "missing_values" in signals
    assert "forecast_shift" in signals
    assert "feature_dominance" in signals
    assert [
        item["priority_score"] for item in recommendations
    ] == sorted(
        (item["priority_score"] for item in recommendations),
        reverse=True,
    )
    assert all(item["action"] and item["evidence"] for item in recommendations)


def test_central_recommendations_include_agentic_missing_value_results():
    root_cause = investigate({
        "missing_values": {
            "total": 3,
            "by_column": {"amount": 3},
        },
    })

    assert root_cause["recommendations"] == [{
        "signal": "missing_values",
        "priority": "medium",
        "priority_score": 50,
        "action": "Inspect the affected columns and the latest ingestion or transformation step.",
        "evidence": {
            "total_nulls": 3,
            "by_column": {"amount": 3},
        },
    }]


def test_central_recommendations_are_empty_for_healthy_results():
    root_cause = investigate({})

    assert root_cause["status"] == "healthy"
    assert root_cause["recommendations"] == []
