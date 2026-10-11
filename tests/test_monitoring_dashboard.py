from datetime import datetime, timezone

import pandas as pd

from monitoring_dashboard import _filter_monitoring_results, _result_rows


def test_filter_monitoring_results_uses_dataset_date_and_search_filters():
    recent_timestamp = datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp()
    earlier_timestamp = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
    results = [
        {
            "id": "quality-run",
            "dataset_id": "dataset-a",
            "check_type": "quality",
            "created_at": recent_timestamp,
        },
        {
            "id": "drift-run",
            "dataset_id": "dataset-a",
            "check_type": "drift",
            "created_at": recent_timestamp,
        },
        {
            "id": "other-dataset-run",
            "dataset_id": "dataset-b",
            "check_type": "quality",
            "created_at": recent_timestamp,
        },
        {
            "id": "old-run",
            "dataset_id": "dataset-a",
            "check_type": "quality",
            "created_at": earlier_timestamp,
        },
    ]

    selected = _filter_monitoring_results(
        results,
        {"dataset-a": "Events", "dataset-b": "Archive"},
        "dataset-a",
        "quality",
        pd.Timestamp("2026-10-01").date(),
        pd.Timestamp("2026-10-11").date(),
    )

    assert [item["id"] for item in selected] == ["quality-run"]


def test_result_rows_calculates_missing_and_duplicate_rates_from_saved_quality():
    results = [{
        "id": "run-1",
        "dataset_id": "dataset-a",
        "check_type": "quality",
        "created_at": datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp(),
        "result": {
            "results": {
                "quality": {
                    "report": {
                        "rows": 2,
                        "columns": 2,
                        "total_nulls": 1,
                        "duplicates": 1,
                    },
                    "quality_score": {"overall_score": 75},
                },
                "drift": {"overall_drift_score": 0.5, "drift_detected": False},
                "anomalies": {"total_anomalies": 1},
            },
            "root_cause": {"status": "warning"},
        },
    }]

    row = _result_rows(results)[0]

    assert row["quality_score"] == 75
    assert row["missing_percentage"] == 25
    assert row["duplicate_percentage"] == 50
    assert row["drift_score"] == 0.5
    assert row["anomalies"] == 1
