"""Unified Streamlit monitoring dashboard views."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


def _get(
    api_base_url: str,
    headers: dict[str, str],
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        response = requests.get(
            f"{api_base_url}{path}",
            headers=headers,
            params=params,
            timeout=20,
        )
    except requests.RequestException as error:
        st.error(f"Could not load {path}: {error}")
        return None
    if response.status_code != 200:
        st.error(response.json().get("detail", f"Could not load {path}."))
        return None
    return response.json()


def _run_monitoring(
    api_base_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> None:
    try:
        response = requests.post(
            f"{api_base_url}/api/monitoring/analyze",
            headers=headers,
            json=payload,
            timeout=300,
        )
    except requests.RequestException as error:
        st.error(f"Monitoring request failed: {error}")
        return
    if response.status_code != 200:
        st.error(response.json().get("detail", "Monitoring run failed."))
        return
    result = response.json()
    st.session_state["dashboard_latest_run_id"] = result["result_id"]
    st.success(f"Monitoring completed · result `{result['result_id']}`")


def _result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        result = item.get("result") or {}
        checks = result.get("results") or {}
        quality = checks.get("quality") or {}
        score = (quality.get("quality_score") or {}).get("overall_score")
        drift = checks.get("drift") or {}
        anomaly = checks.get("anomalies") or {}
        rows.append(
            {
                "created_at": pd.to_datetime(item.get("created_at"), unit="s", utc=True),
                "result_id": item.get("id"),
                "dataset_id": item.get("dataset_id"),
                "check_type": item.get("check_type"),
                "quality_score": score,
                "drift_score": drift.get("overall_drift_score"),
                "drift_detected": drift.get("drift_detected"),
                "anomalies": anomaly.get("total_anomalies"),
                "status": (result.get("root_cause") or {}).get("status"),
            }
        )
    return rows


def _render_overview(results: list[dict[str, Any]]) -> None:
    st.subheader("Monitoring overview")
    if not results:
        st.info("No monitoring runs yet. Start a run below to populate your dashboard.")
        return

    rows = _result_rows(results)
    history = pd.DataFrame(rows).sort_values("created_at")
    latest = history.iloc[-1]
    first, second, third, fourth = st.columns(4)
    score = latest["quality_score"]
    first.metric("Latest quality score", "—" if pd.isna(score) else f"{score:.1f}/100")
    drift = latest["drift_score"]
    second.metric("Latest drift score", "—" if pd.isna(drift) else f"{drift:.1f}")
    third.metric(
        "Latest anomalies",
        "—" if pd.isna(latest["anomalies"]) else int(latest["anomalies"]),
    )
    fourth.metric("Historical runs", len(history))

    quality_history = history.dropna(subset=["quality_score"])
    drift_history = history.dropna(subset=["drift_score"])
    quality_tab, drift_tab, anomaly_tab = st.tabs(
        ["Quality trend", "Drift trend", "Anomaly trend"]
    )
    with quality_tab:
        if quality_history.empty:
            st.info("No persisted quality scores are available.")
        else:
            figure = px.line(
                quality_history,
                x="created_at",
                y="quality_score",
                markers=True,
                title="Data quality score over time",
                hover_data=["dataset_id", "result_id"],
            )
            figure.update_yaxes(range=[0, 100], title="Score")
            st.plotly_chart(figure, use_container_width=True)
    with drift_tab:
        if drift_history.empty:
            st.info("No persisted drift checks are available.")
        else:
            figure = px.line(
                drift_history,
                x="created_at",
                y="drift_score",
                markers=True,
                title="Overall drift score over time",
                hover_data=["dataset_id", "result_id", "drift_detected"],
            )
            st.plotly_chart(figure, use_container_width=True)
    with anomaly_tab:
        anomaly_history = history.dropna(subset=["anomalies"])
        if anomaly_history.empty:
            st.info("No persisted anomaly checks are available.")
        else:
            figure = px.bar(
                anomaly_history,
                x="created_at",
                y="anomalies",
                title="Detected anomalous rows by monitoring run",
                hover_data=["dataset_id", "result_id"],
            )
            st.plotly_chart(figure, use_container_width=True)


def _render_history(
    api_base_url: str,
    headers: dict[str, str],
    results: list[dict[str, Any]],
    role: str,
) -> None:
    st.subheader("Historical monitoring results")
    if results:
        history = pd.DataFrame(_result_rows(results)).sort_values(
            "created_at", ascending=False
        )
        st.dataframe(history, use_container_width=True, hide_index=True)
        options = {f"{row['created_at']} · {row['id']}": row["id"] for row in results}
        selected = st.selectbox("Inspect a result", list(options))
        detail = _get(
            api_base_url,
            headers,
            f"/api/monitoring/results/{options[selected]}",
        )
        if detail is not None:
            report = detail.get("result") or {}
            checks = report.get("results") or {}
            quality, drift, anomalies, forecasts, xai = st.tabs(
                ["Quality", "Drift", "Anomalies", "Forecasts", "XAI"]
            )
            with quality:
                st.json(checks.get("quality", {}))
            with drift:
                st.json(checks.get("drift", {}))
            with anomalies:
                st.json(checks.get("anomalies", {}))
            with forecasts:
                forecast = checks.get("forecast")
                if isinstance(forecast, dict) and "forecast" in forecast:
                    frame = pd.DataFrame(forecast["history"] + forecast["forecast"])
                    frame["kind"] = ["History"] * len(forecast["history"]) + ["Forecast"] * len(forecast["forecast"])
                    figure = px.line(
                        frame,
                        x="date",
                        y="value",
                        color="kind",
                        markers=True,
                        title=f"Forecast · {forecast.get('metric', 'row_volume')}",
                    )
                    st.plotly_chart(figure, use_container_width=True)
                    if forecast.get("fallback_reason"):
                        st.warning(forecast["fallback_reason"])
                else:
                    st.info("This result has no forecast output.")
                st.json(forecast or {})
            with xai:
                st.json(checks.get("explanation", {}))
            st.json(report.get("root_cause", {}))
    else:
        st.info("No saved results are available yet.")

    st.divider()
    st.subheader("Alert history")
    if role != "admin":
        st.info("Alert history is administrator-only because it can contain delivery recipients.")
        return
    alerts = _get(api_base_url, headers, "/api/alerts/history", {"limit": 100})
    if alerts is not None:
        alert_rows = alerts.get("history") or []
        if alert_rows:
            st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No alerts have been recorded.")


def _render_schedules(
    api_base_url: str,
    headers: dict[str, str],
    role: str,
    datasets: list[dict[str, Any]],
    schedules: list[dict[str, Any]],
) -> None:
    st.subheader("Recurring monitoring schedules")
    if schedules:
        schedule_rows = [
            {
                "schedule_id": item["id"],
                "dataset_id": item["dataset_id"],
                "checks": ", ".join(item["schedule"].get("checks", [])),
                "interval_minutes": item["schedule"].get("interval_seconds", 0) // 60,
                "enabled": item["enabled"],
                "next_run": pd.to_datetime(item["next_run_at"], unit="s", utc=True),
                "last_run": (
                    pd.to_datetime(item["last_run_at"], unit="s", utc=True)
                    if item["last_run_at"] is not None else None
                ),
                "last_status": item["last_status"],
                "last_error": item["last_error"],
            }
            for item in schedules
        ]
        st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True, hide_index=True)
        for item in schedules:
            with st.expander(f"{item['id']} · {item['last_status']}"):
                left, right = st.columns(2)
                action = "Pause" if item["enabled"] else "Resume"
                if left.button(action, key=f"schedule-toggle-{item['id']}"):
                    try:
                        response = requests.patch(
                            f"{api_base_url}/api/schedules/{item['id']}",
                            headers=headers,
                            json={"enabled": not item["enabled"]},
                            timeout=20,
                        )
                    except requests.RequestException as error:
                        st.error(f"Could not update schedule: {error}")
                    else:
                        if response.status_code == 200:
                            st.rerun()
                        else:
                            st.error(response.json().get("detail", "Could not update schedule."))
                if right.button("Delete", key=f"schedule-delete-{item['id']}"):
                    try:
                        response = requests.delete(
                            f"{api_base_url}/api/schedules/{item['id']}",
                            headers=headers,
                            timeout=20,
                        )
                    except requests.RequestException as error:
                        st.error(f"Could not delete schedule: {error}")
                    else:
                        if response.status_code == 200:
                            st.rerun()
                        else:
                            st.error(response.json().get("detail", "Could not delete schedule."))
    else:
        st.info("No recurring schedules are configured.")

    if role == "viewer":
        st.info("Viewer access is read-only; an analyst or administrator can create schedules.")
        return
    if not datasets:
        st.info("Save a dataset before creating a recurring schedule.")
        return

    with st.expander("Create a recurring monitoring schedule", expanded=not schedules):
        dataset_options = {
            f"{item['name']} · {item['id']}": item for item in datasets
        }
        selected_label = st.selectbox(
            "Scheduled dataset", list(dataset_options), key="schedule-dataset"
        )
        selected_dataset = dataset_options[selected_label]
        dataset_response = _get(
            api_base_url,
            headers,
            f"/api/datasets/{selected_dataset['id']}",
        )
        frame = (
            pd.DataFrame(dataset_response.get("data", []))
            if dataset_response is not None else pd.DataFrame()
        )
        columns = [str(column) for column in frame.columns]
        date_columns = [
            column for column in columns
            if "date" in column.lower() or "time" in column.lower()
        ]
        numeric_columns = list(frame.select_dtypes(include="number").columns)
        selected_checks = st.multiselect(
            "Recurring checks",
            ["quality", "drift", "anomalies", "forecast", "explain"],
            default=["quality"],
            key="schedule-checks",
        )
        interval_minutes = st.number_input(
            "Run every (minutes)",
            min_value=1,
            max_value=44_640,
            value=60,
            step=1,
            key="schedule-interval",
        )
        timestamp_column = st.selectbox(
            "Freshness timestamp column",
            ["None"] + columns,
            key="schedule-timestamp",
        )
        max_age_hours = None
        if timestamp_column != "None":
            max_age_hours = st.number_input(
                "Maximum data age (hours)",
                min_value=0.01,
                value=24.0,
                step=1.0,
                key="schedule-max-age",
            )
        baseline_id = None
        if "drift" in selected_checks:
            baselines = {
                f"{item['name']} · {item['id']}": item
                for item in datasets if item["id"] != selected_dataset["id"]
            }
            if baselines:
                baseline_label = st.selectbox(
                    "Drift baseline", list(baselines), key="schedule-baseline"
                )
                baseline_id = baselines[baseline_label]["id"]
            else:
                st.warning("Add a second dataset to schedule drift checks.")
        date_column = None
        value_column = None
        if "forecast" in selected_checks and date_columns:
            date_column = st.selectbox(
                "Forecast date column", date_columns, key="schedule-date-column"
            )
            value_column = st.selectbox(
                "Forecast metric",
                ["Row volume"] + numeric_columns,
                key="schedule-value-column",
            )
        elif "forecast" in selected_checks:
            st.warning("This dataset has no detected date/time column.")
        target_column = None
        if "explain" in selected_checks and columns:
            target_column = st.selectbox(
                "XAI target column", columns, key="schedule-target-column"
            )
        channels_json = st.text_input(
            "Alert channels JSON (optional)",
            value="[]",
            help='Examples: ["email"] or [{"channel":"email","recipient":"alerts@example.com"}]',
            key="schedule-alert-channels",
        )
        st.caption(
            "Alerts are sent only when at least one channel is configured and the run detects a finding."
        )
        if st.button("Save schedule", type="primary", key="save-schedule"):
            if not selected_checks:
                st.error("Select at least one monitoring check.")
            elif "drift" in selected_checks and baseline_id is None:
                st.error("Select a baseline dataset for drift monitoring.")
            elif "forecast" in selected_checks and date_column is None:
                st.error("Select a date column for forecast monitoring.")
            else:
                try:
                    channels = json.loads(channels_json)
                    if not isinstance(channels, list):
                        raise ValueError("Alert channels must be a JSON list.")
                    payload: dict[str, Any] = {
                        "dataset_id": selected_dataset["id"],
                        "interval_seconds": int(interval_minutes * 60),
                        "checks": selected_checks,
                        "alert_channels": channels,
                    }
                    if timestamp_column != "None":
                        payload["timestamp_column"] = timestamp_column
                        payload["max_age_hours"] = max_age_hours
                    if baseline_id is not None:
                        payload["baseline_dataset_id"] = baseline_id
                    if date_column:
                        payload["date_column"] = date_column
                    if value_column:
                        payload["value_column"] = None if value_column == "Row volume" else value_column
                    if target_column:
                        payload["target_column"] = target_column
                    response = requests.post(
                        f"{api_base_url}/api/schedules",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
                except (requests.RequestException, ValueError) as error:
                    st.error(f"Could not create schedule: {error}")
                else:
                    if response.status_code == 201:
                        st.success(f"Schedule created: `{response.json()['id']}`")
                        st.rerun()
                    else:
                        st.error(response.json().get("detail", "Could not create schedule."))


def render_monitoring_dashboard(
    api_base_url: str,
    headers: dict[str, str],
    role: str,
) -> None:
    st.title("Advanced Monitoring Dashboard")
    st.caption(
        "Unified quality, drift, anomaly, alert, forecast, XAI, and historical monitoring."
    )

    datasets_payload = _get(api_base_url, headers, "/api/datasets", {"limit": 500})
    if datasets_payload is None:
        return
    datasets = datasets_payload.get("datasets") or []
    results_payload = _get(api_base_url, headers, "/api/monitoring/results", {"limit": 500})
    if results_payload is None:
        return
    results = results_payload.get("results") or []
    schedules_payload = _get(api_base_url, headers, "/api/schedules")
    if schedules_payload is None:
        return
    schedules = schedules_payload.get("schedules") or []

    overview_tab, run_tab, schedules_tab, history_tab = st.tabs(
        ["Overview", "Run monitoring", "Schedules", "History & alerts"]
    )
    with overview_tab:
        _render_overview(results)

    with run_tab:
        st.subheader("Create a monitoring run")
        if role == "viewer":
            st.info("Viewer access is read-only. An analyst or administrator can start monitoring runs.")
        else:
            uploaded = st.file_uploader(
                "Upload a CSV dataset (maximum 10 MiB)",
                type=["csv"],
                key="dashboard_csv_upload",
            )
            if uploaded is not None:
                try:
                    upload_response = requests.post(
                        f"{api_base_url}/api/datasets/upload",
                        params={"name": uploaded.name},
                        data=uploaded.getvalue(),
                        headers={**headers, "Content-Type": "text/csv"},
                        timeout=60,
                    )
                except requests.RequestException as error:
                    st.error(f"Dataset upload failed: {error}")
                else:
                    if upload_response.status_code != 201:
                        st.error(upload_response.json().get("detail", "Dataset upload failed."))
                    else:
                        st.success(f"Dataset saved: {upload_response.json()['name']}")
                        st.rerun()

            if not datasets:
                st.info("Upload or save a dataset before starting a monitoring run.")
            else:
                dataset_by_label = {
                    f"{item['name']} · {item['id']}": item for item in datasets
                }
                selected_label = st.selectbox("Current dataset", list(dataset_by_label))
                selected_dataset = dataset_by_label[selected_label]
                baseline_label = st.selectbox(
                    "Baseline dataset (used when drift is selected)",
                    ["None"] + [
                        label for label in dataset_by_label if label != selected_label
                    ],
                )
                selected_checks = st.multiselect(
                    "Checks",
                    ["quality", "drift", "anomalies", "forecast", "explain"],
                    default=["quality", "anomalies"],
                )

                source_payload = _get(
                    api_base_url,
                    headers,
                    f"/api/datasets/{selected_dataset['id']}",
                )
                frame = pd.DataFrame(source_payload.get("data", [])) if source_payload else pd.DataFrame()
                date_columns = [
                    column for column in frame.columns
                    if pd.api.types.is_datetime64_any_dtype(frame[column])
                    or "date" in str(column).lower()
                    or "time" in str(column).lower()
                ]
                numeric_columns = list(frame.select_dtypes(include="number").columns)
                date_column = None
                value_column = None
                target_column = None
                if "forecast" in selected_checks:
                    if not date_columns:
                        st.warning("Forecasting requires a date/time column.")
                    else:
                        date_column = st.selectbox("Forecast date column", date_columns)
                        value_column = st.selectbox(
                            "Forecast metric",
                            ["Row volume"] + numeric_columns,
                        )
                if "explain" in selected_checks and len(frame.columns):
                    target_column = st.selectbox("XAI target column", list(frame.columns))
                notify = st.checkbox("Send configured alerts for findings", value=False)

                if st.button("Run selected checks", type="primary", disabled=not selected_checks):
                    if "drift" in selected_checks and baseline_label == "None":
                        st.error("Select a baseline dataset before running drift analysis.")
                    elif "forecast" in selected_checks and date_column is None:
                        st.error("Select or provide a date/time column for forecasting.")
                    else:
                        payload: dict[str, Any] = {
                            "dataset_id": selected_dataset["id"],
                            "checks": selected_checks,
                            "notify": notify,
                        }
                        if "drift" in selected_checks:
                            payload["baseline_dataset_id"] = dataset_by_label[baseline_label]["id"]
                        if "forecast" in selected_checks:
                            payload["date_column"] = date_column
                            payload["value_column"] = None if value_column == "Row volume" else value_column
                        if "explain" in selected_checks:
                            payload["target_column"] = target_column
                        _run_monitoring(api_base_url, headers, payload)
                        st.rerun()
        if st.session_state.get("dashboard_latest_run_id"):
            st.caption(f"Most recent run: `{st.session_state['dashboard_latest_run_id']}`")

    with schedules_tab:
        _render_schedules(
            api_base_url,
            headers,
            role,
            datasets,
            schedules,
        )

    with history_tab:
        _render_history(api_base_url, headers, results, role)
