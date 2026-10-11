"""Unified Streamlit monitoring dashboard views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

_voice_command_component = components.declare_component(
    "monitoring_voice_commands",
    path=str(Path(__file__).parent / "voice_component"),
)


def _get(
    api_base_url: str,
    headers: dict[str, str],
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        with st.spinner("Loading monitoring data…"):
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
        st.error(_response_error(response, f"Could not load {path}."))
        return None
    try:
        payload = response.json()
    except ValueError:
        st.error(f"The API returned an invalid response while loading {path}.")
        return None
    if not isinstance(payload, dict):
        st.error(f"The API returned an unexpected response while loading {path}.")
        return None
    return payload


def _response_error(response: requests.Response, fallback: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback
    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        return payload["detail"]
    return fallback


def _response_payload(
    response: requests.Response,
    context: str,
) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except ValueError:
        st.error(f"The API returned an invalid response for {context}.")
        return None
    if not isinstance(payload, dict):
        st.error(f"The API returned an unexpected response for {context}.")
        return None
    return payload


def _post(
    api_base_url: str,
    headers: dict[str, str],
    path: str,
    payload: dict[str, Any],
    *,
    timeout: int = 60,
    expected_status: int = 200,
) -> dict[str, Any] | None:
    try:
        with st.spinner("Processing request…"):
            response = requests.post(
                f"{api_base_url}{path}",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
    except requests.RequestException as error:
        st.error(f"Request to {path} failed: {error}")
        return None
    if response.status_code != expected_status:
        st.error(_response_error(response, f"Request to {path} failed."))
        return None
    try:
        result = response.json()
    except ValueError:
        st.error(f"The API returned an invalid response while processing {path}.")
        return None
    if not isinstance(result, dict):
        st.error(f"The API returned an unexpected response while processing {path}.")
        return None
    return result


def _run_monitoring(
    api_base_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> bool:
    try:
        response = requests.post(
            f"{api_base_url}/api/monitoring/analyze",
            headers=headers,
            json=payload,
            timeout=300,
        )
    except requests.RequestException as error:
        st.error(f"Monitoring request failed: {error}")
        return False
    if response.status_code != 200:
        st.error(_response_error(response, "Monitoring run failed."))
        return False
    try:
        result = response.json()
    except ValueError:
        st.error("The API returned an invalid response for the monitoring run.")
        return False
    if not isinstance(result, dict) or not isinstance(result.get("result_id"), str):
        st.error("The API returned an unexpected response for the monitoring run.")
        return False
    st.session_state["dashboard_latest_run_id"] = result["result_id"]
    st.success(f"Monitoring completed · result `{result['result_id']}`")
    return True


def _render_voice_assistant(
    api_base_url: str,
    headers: dict[str, str],
    datasets: list[dict[str, Any]],
) -> None:
    st.subheader("Ask monitoring by voice or text")
    language_labels = {
        "English": ("en", "en-US"),
        "Español": ("es", "es-ES"),
        "Français": ("fr", "fr-FR"),
        "हिन्दी": ("hi", "hi-IN"),
    }
    selected_language = st.selectbox(
        "Query and response language",
        list(language_labels),
        key="assistant_language",
    )
    language_code, speech_locale = language_labels[selected_language]

    if not datasets:
        st.info("Save or upload a dataset before asking monitoring questions.")
        return

    dataset_by_label = {
        f"{item['name']} · {item['id']}": item for item in datasets
    }
    selected_label = st.selectbox(
        "Dataset snapshot",
        list(dataset_by_label),
        key="assistant_dataset",
    )
    st.caption(
        "Quality questions, including “today’s score,” use the selected saved dataset snapshot."
    )

    voice_value = _voice_command_component(
        language=speech_locale,
        speech_text=st.session_state.get("assistant_last_answer", ""),
        key="monitoring_voice_command",
        default=None,
    )
    new_voice_command = False
    if isinstance(voice_value, dict):
        nonce = voice_value.get("nonce")
        transcript = voice_value.get("transcript")
        if (
            nonce is not None
            and nonce != st.session_state.get("assistant_voice_nonce")
            and isinstance(transcript, str)
            and transcript.strip()
        ):
            st.session_state["assistant_voice_nonce"] = nonce
            st.session_state["assistant_question"] = transcript.strip()
            new_voice_command = True

    question = st.text_input(
        "Monitoring question",
        placeholder="Show anomaly report, today's quality score, or columns with missing values",
        key="assistant_question",
    )
    previous_answer = st.session_state.get("assistant_last_response")
    if isinstance(previous_answer, dict):
        st.markdown(previous_answer.get("answer", ""))
        st.caption(
            f"Category: {previous_answer.get('category', 'unknown')} · "
            f"Confidence: {previous_answer.get('confidence', 'unknown')}"
        )
        if isinstance(previous_answer.get("context"), dict):
            st.json(previous_answer["context"])

    submit = st.button("Ask", type="primary", key="assistant_ask_button")
    if not (submit or new_voice_command):
        return
    if not question.strip():
        st.warning("Enter a question or use the microphone to speak one.")
        return

    dataset_id = dataset_by_label[selected_label]["id"]
    dataset_response = _get(
        api_base_url,
        headers,
        f"/api/datasets/{dataset_id}",
    )
    if dataset_response is None:
        return

    answer = _post(
        api_base_url,
        headers,
        "/api/assistant/ask",
        {
            "question": question,
            "language": language_code,
            "data": dataset_response.get("data", []),
        },
    )
    if answer is None:
        return
    answer_text = answer.get("answer")
    if not isinstance(answer_text, str):
        st.error("The assistant returned an unexpected response.")
        return
    st.session_state["assistant_last_answer"] = answer_text
    st.session_state["assistant_last_response"] = answer
    st.rerun()


def _result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        result = item.get("result") or {}
        checks = result.get("results") or {}
        quality = checks.get("quality") or {}
        comparison_quality = (result.get("quality") or {}).get("current") or {}
        score = (quality.get("quality_score") or {}).get("overall_score")
        if score is None:
            score = comparison_quality.get("quality_score")
        drift = checks.get("drift") or {}
        if not drift and isinstance(result.get("distribution_drift"), dict):
            drift = result["distribution_drift"]
        anomaly = checks.get("anomalies") or {}
        quality_report = quality.get("report") or {}
        row_count = quality_report.get("rows")
        total_cells = (row_count or 0) * (quality_report.get("columns") or 0)
        total_nulls = quality_report.get("total_nulls")
        duplicate_rows = quality_report.get("duplicates")
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
                "missing_percentage": (
                    total_nulls / total_cells * 100
                    if total_cells and total_nulls is not None else None
                ),
                "duplicate_percentage": (
                    duplicate_rows / row_count * 100
                    if row_count and duplicate_rows is not None else None
                ),
                "duplicate_rows": duplicate_rows,
                "status": (result.get("root_cause") or {}).get("status"),
            }
        )
    return rows


def _filter_monitoring_results(
    results: list[dict[str, Any]],
    dataset_name_by_id: dict[str, str],
    dataset_id: str | None,
    search: str,
    start_date: Any,
    end_date: Any,
) -> list[dict[str, Any]]:
    filtered = []
    for item in results:
        item_dataset_id = item.get("dataset_id")
        if dataset_id and item_dataset_id != dataset_id:
            continue
        created_at = pd.to_datetime(item.get("created_at"), unit="s", utc=True, errors="coerce")
        if pd.isna(created_at) or not start_date <= created_at.date() <= end_date:
            continue
        searchable = " ".join((
            str(item.get("id", "")),
            str(item.get("check_type", "")),
            str(item_dataset_id or ""),
            dataset_name_by_id.get(item_dataset_id, ""),
        )).lower()
        if search and search not in searchable:
            continue
        filtered.append(item)
    return filtered


def _render_overview(
    results: list[dict[str, Any]],
    widgets: list[str] | None = None,
    copy: dict[str, str] | None = None,
    datasets: list[dict[str, Any]] | None = None,
    schedules: list[dict[str, Any]] | None = None,
    previous_results: list[dict[str, Any]] | None = None,
) -> None:
    copy = copy or _DASHBOARD_COPY["en"]
    st.subheader(copy["overview"])
    rows = _result_rows(results)
    history = (
        pd.DataFrame(rows).sort_values("created_at")
        if rows else pd.DataFrame()
    )
    latest = history.iloc[-1] if not history.empty else {}
    previous_rows = _result_rows(previous_results or [])
    previous_history = (
        pd.DataFrame(previous_rows).sort_values("created_at")
        if previous_rows else pd.DataFrame()
    )
    previous = previous_history.iloc[-1] if not previous_history.empty else {}
    quality_value = latest.get("quality_score")
    drift_value = latest.get("drift_score")
    anomaly_value = latest.get("anomalies")
    missing_value = latest.get("missing_percentage")
    duplicate_value = latest.get("duplicate_percentage")
    metrics = {
        "quality_score": (
            copy["quality"],
            "—" if pd.isna(quality_value) else f"{quality_value:.1f}/100",
            quality_value - previous.get("quality_score")
            if not pd.isna(quality_value)
            and not pd.isna(previous.get("quality_score")) else None,
        ),
        "drift_score": (
            copy["drift"],
            "—" if pd.isna(drift_value) else f"{drift_value:.1f}",
            drift_value - previous.get("drift_score")
            if not pd.isna(drift_value)
            and not pd.isna(previous.get("drift_score")) else None,
        ),
        "anomalies": (
            copy["anomalies"],
            "—" if pd.isna(anomaly_value) else int(anomaly_value),
            anomaly_value - previous.get("anomalies")
            if not pd.isna(anomaly_value)
            and not pd.isna(previous.get("anomalies")) else None,
        ),
        "historical_runs": (
            copy["runs"],
            len(history),
            len(history) - len(previous_history) if not previous_history.empty else None,
        ),
        "datasets": ("Datasets monitored", len(datasets or []), None),
        "active_schedules": (
            "Active schedules",
            sum(bool(item.get("enabled")) for item in (schedules or [])),
            None,
        ),
        "missing_values": (
            "Missing values",
            "—" if pd.isna(missing_value) else f"{missing_value:.1f}%",
            missing_value - previous.get("missing_percentage")
            if not pd.isna(missing_value)
            and not pd.isna(previous.get("missing_percentage")) else None,
        ),
        "duplicate_rows": (
            "Duplicate rows",
            "—" if pd.isna(duplicate_value) else f"{duplicate_value:.1f}%",
            duplicate_value - previous.get("duplicate_percentage")
            if not pd.isna(duplicate_value)
            and not pd.isna(previous.get("duplicate_percentage")) else None,
        ),
    }
    chosen_widgets = [key for key in (widgets or list(metrics)) if key in metrics]
    for start in range(0, len(chosen_widgets), 4):
        row_widgets = chosen_widgets[start:start + 4]
        columns = st.columns(len(row_widgets))
        for column, key in zip(columns, row_widgets):
            column.metric(*metrics[key])

    if history.empty:
        st.info("No monitoring runs yet. Start a run to populate trend charts and monitoring KPIs.")
        return

    quality_history = history.dropna(subset=["quality_score"])
    drift_history = history.dropna(subset=["drift_score"])
    quality_tab, drift_tab, anomaly_tab, completeness_tab = st.tabs(
        [
            copy["quality_trend"],
            copy["drift_trend"],
            copy["anomaly_trend"],
            "Missing values and duplicates",
        ]
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
    with completeness_tab:
        completeness = history.dropna(
            subset=["missing_percentage", "duplicate_percentage"],
            how="all",
        )
        if completeness.empty:
            st.info("Missing-value and duplicate trends are unavailable for these runs.")
        else:
            series = completeness.melt(
                id_vars=["created_at", "result_id"],
                value_vars=["missing_percentage", "duplicate_percentage"],
                var_name="measure",
                value_name="percentage",
            ).dropna(subset=["percentage"])
            st.plotly_chart(
                px.line(
                    series,
                    x="created_at",
                    y="percentage",
                    color="measure",
                    markers=True,
                    hover_data=["result_id"],
                    title="Missing cells and duplicate rows over time",
                ),
                use_container_width=True,
            )


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
            if "schema" in report and "quality" in report:
                st.json(report)
            else:
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


def _render_lineage_versions(
    api_base_url: str,
    headers: dict[str, str],
    datasets: list[dict[str, Any]],
    role: str,
) -> None:
    st.subheader("Dataset lineage and versions")
    if not datasets:
        st.info("Save a dataset to inspect its lineage or create versions.")
        return
    dataset_by_label = {
        f"{item['name']} · v{item.get('version_number', 1)} · {item['id']}": item
        for item in datasets
    }
    selected_label = st.selectbox(
        "Dataset",
        list(dataset_by_label),
        key="lineage_dataset",
    )
    selected = dataset_by_label[selected_label]
    lineage = _get(
        api_base_url,
        headers,
        f"/api/datasets/{selected['id']}/lineage",
    )
    if lineage is None:
        return
    versions = lineage.get("versions") or []
    events = lineage.get("events") or []
    if events:
        st.caption("Source, transformations, monitoring runs, and recorded failures")
        st.dataframe(
            pd.DataFrame([
                {
                    "time": pd.to_datetime(event.get("created_at"), unit="s", utc=True),
                    "dataset": event.get("dataset_id"),
                    "pipeline": event.get("pipeline"),
                    "event": event.get("event_type"),
                    "status": event.get("status"),
                    "inputs": ", ".join(event.get("input_dataset_ids") or []),
                    "result": event.get("result_id"),
                    "error": event.get("error"),
                }
                for event in events
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No lineage events are recorded for this dataset.")

    st.caption("Immutable dataset version history")
    st.dataframe(pd.DataFrame(versions), use_container_width=True, hide_index=True)
    if len(versions) >= 2:
        version_by_label = {
            f"v{version['version_number']} · {version['id']}": version
            for version in versions
        }
        version_labels = list(version_by_label)
        baseline_label = st.selectbox(
            "Baseline version",
            version_labels,
            index=0,
            key="comparison_baseline_version",
        )
        current_label = st.selectbox(
            "Compare with version",
            version_labels,
            index=len(version_labels) - 1,
            key="comparison_current_version",
        )
        if st.button(
            "Compare selected versions",
            disabled=baseline_label == current_label or role == "viewer",
        ):
            try:
                response = requests.post(
                    f"{api_base_url}/api/datasets/compare",
                    headers=headers,
                    json={
                        "baseline_dataset_id": version_by_label[baseline_label]["id"],
                        "current_dataset_id": version_by_label[current_label]["id"],
                    },
                    timeout=60,
                )
            except requests.RequestException as error:
                st.error(f"Dataset comparison failed: {error}")
            else:
                if response.status_code != 200:
                    st.error(_response_error(response, "Dataset comparison failed."))
                else:
                    comparison = _response_payload(response, "dataset comparison")
                    if comparison is not None:
                        st.json(comparison)

    if role != "viewer":
        with st.expander("Create a new version"):
            detail = _get(
                api_base_url,
                headers,
                f"/api/datasets/{selected['id']}",
            )
            if detail is not None:
                version_data = st.text_area(
                    "New version data (JSON records)",
                    value=json.dumps(detail.get("data", []), indent=2, default=str),
                    height=220,
                    key=f"version_data_{selected['id']}",
                )
                change_summary = st.text_input(
                    "Change summary",
                    max_chars=1000,
                    key=f"version_summary_{selected['id']}",
                )
                if st.button("Save dataset version"):
                    try:
                        parsed_data = json.loads(version_data)
                    except json.JSONDecodeError as error:
                        st.error(f"Version data must be valid JSON: {error}")
                    else:
                        try:
                            response = requests.post(
                                f"{api_base_url}/api/datasets/{selected['id']}/versions",
                                headers=headers,
                                json={
                                    "data": parsed_data,
                                    "change_summary": change_summary,
                                },
                                timeout=60,
                            )
                        except requests.RequestException as error:
                            st.error(f"Could not create dataset version: {error}")
                        else:
                            if response.status_code == 201:
                                version = _response_payload(response, "dataset version creation")
                                if version is not None:
                                    st.success(
                                        f"Created version v{version.get('version_number', '?')}."
                                    )
                                    st.rerun()
                            else:
                                st.error(_response_error(response, "Could not create dataset version."))


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
                "report_formats": ", ".join(item["schedule"].get("report_formats", [])),
                "interval_minutes": item["schedule"].get("interval_seconds", 0) // 60,
                "enabled": item["enabled"],
                "next_run": pd.to_datetime(item["next_run_at"], unit="s", utc=True),
                "last_run": (
                    pd.to_datetime(item["last_run_at"], unit="s", utc=True)
                    if item["last_run_at"] is not None else None
                ),
                "last_status": item["last_status"],
                "last_error": item["last_error"],
                "retry_count": item.get("retry_count", 0),
            }
            for item in schedules
        ]
        st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True, hide_index=True)
        for item in schedules:
            with st.expander(f"{item['id']} · {item['last_status']}"):
                if role == "viewer":
                    st.info("This account can inspect schedules but cannot change them.")
                else:
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
                                st.error(_response_error(response, "Could not update schedule."))
                    confirm_delete = st.checkbox(
                        "Confirm schedule deletion",
                        key=f"schedule-confirm-delete-{item['id']}",
                    )
                    if right.button(
                        "Delete",
                        key=f"schedule-delete-{item['id']}",
                        disabled=not confirm_delete,
                    ):
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
                                st.error(_response_error(response, "Could not delete schedule."))
                    with st.form(f"schedule-edit-{item['id']}"):
                        interval_minutes = st.number_input(
                            "Run interval (minutes)",
                            min_value=1,
                            max_value=44_640,
                            value=max(
                                1,
                                int(item["schedule"].get("interval_seconds", 60) // 60),
                            ),
                        )
                        save_interval = st.form_submit_button("Save schedule interval")
                    if save_interval:
                        try:
                            response = requests.patch(
                                f"{api_base_url}/api/schedules/{item['id']}",
                                headers=headers,
                                json={"interval_seconds": int(interval_minutes * 60)},
                                timeout=20,
                            )
                        except requests.RequestException as error:
                            st.error(f"Could not edit schedule: {error}")
                        else:
                            if response.status_code == 200:
                                st.success("Schedule interval updated.")
                                st.rerun()
                            else:
                                st.error(_response_error(response, "Could not edit schedule."))
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
        report_formats = st.multiselect(
            "Generate a report after each run",
            ["pdf", "xlsx", "pptx"],
            key="schedule-report-formats",
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
                        "report_formats": report_formats,
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
                        schedule = _response_payload(response, "schedule creation")
                        if schedule is not None:
                            st.success(f"Schedule created: `{schedule.get('id', 'unknown')}`")
                            st.rerun()
                    else:
                        st.error(_response_error(response, "Could not create schedule."))


_DASHBOARD_COPY = {
    "en": {
        "title": "Advanced Monitoring Dashboard",
        "caption": "Unified quality, drift, anomaly, alert, forecast, XAI, and historical monitoring.",
        "overview": "Overview",
        "run": "Run monitoring",
        "lineage": "Lineage & versions",
        "schedules": "Schedules",
        "history": "History & alerts",
        "assistant": "Voice assistant",
        "reports": "Reports",
        "quality": "Latest quality score",
        "drift": "Latest drift score",
        "anomalies": "Latest anomalies",
        "runs": "Historical runs",
        "quality_trend": "Quality trend",
        "drift_trend": "Drift trend",
        "anomaly_trend": "Anomaly trend",
        "pages": {
            "Overview": "Overview",
            "Data sources": "Data sources",
            "Data quality": "Data quality",
            "Anomalies": "Anomalies",
            "Data drift": "Data drift",
            "Trends & forecasting": "Trends & forecasting",
            "Explainable AI": "Explainable AI",
            "Run monitoring": "Run monitoring",
            "Scheduled monitoring": "Scheduled monitoring",
            "Automated reports": "Automated reports",
            "AI assistant": "AI assistant",
            "Alerts & notifications": "Alerts & notifications",
            "Compare datasets": "Compare datasets",
            "Settings & security": "Settings & security",
            "Monitoring history": "Monitoring history",
        },
    },
    "es": {
        "title": "Panel avanzado de supervisión",
        "caption": "Calidad, deriva, anomalías, alertas, pronósticos, XAI e historial en un solo lugar.",
        "overview": "Resumen",
        "run": "Ejecutar supervisión",
        "lineage": "Linaje y versiones",
        "schedules": "Programaciones",
        "history": "Historial y alertas",
        "assistant": "Asistente de voz",
        "reports": "Informes",
        "quality": "Puntuación de calidad",
        "drift": "Puntuación de deriva",
        "anomalies": "Anomalías recientes",
        "runs": "Ejecuciones históricas",
        "quality_trend": "Tendencia de calidad",
        "drift_trend": "Tendencia de deriva",
        "anomaly_trend": "Tendencia de anomalías",
        "pages": {
            "Overview": "Resumen",
            "Data sources": "Fuentes de datos",
            "Data quality": "Calidad de datos",
            "Anomalies": "Anomalías",
            "Data drift": "Deriva de datos",
            "Trends & forecasting": "Tendencias y pronósticos",
            "Explainable AI": "IA explicable",
            "Run monitoring": "Ejecutar supervisión",
            "Scheduled monitoring": "Supervisión programada",
            "Automated reports": "Informes automáticos",
            "AI assistant": "Asistente de IA",
            "Alerts & notifications": "Alertas y notificaciones",
            "Compare datasets": "Comparar conjuntos",
            "Settings & security": "Configuración y seguridad",
            "Monitoring history": "Historial de supervisión",
        },
    },
    "fr": {
        "title": "Tableau de surveillance avancée",
        "caption": "Qualité, dérive, anomalies, alertes, prévisions, XAI et historique réunis.",
        "overview": "Vue d’ensemble",
        "run": "Lancer la surveillance",
        "lineage": "Lignage et versions",
        "schedules": "Planifications",
        "history": "Historique et alertes",
        "assistant": "Assistant vocal",
        "reports": "Rapports",
        "quality": "Score de qualité",
        "drift": "Score de dérive",
        "anomalies": "Anomalies récentes",
        "runs": "Exécutions historiques",
        "quality_trend": "Tendance qualité",
        "drift_trend": "Tendance de dérive",
        "anomaly_trend": "Tendance des anomalies",
        "pages": {
            "Overview": "Vue d’ensemble",
            "Data sources": "Sources de données",
            "Data quality": "Qualité des données",
            "Anomalies": "Anomalies",
            "Data drift": "Dérive des données",
            "Trends & forecasting": "Tendances et prévisions",
            "Explainable AI": "IA explicable",
            "Run monitoring": "Lancer la surveillance",
            "Scheduled monitoring": "Surveillance planifiée",
            "Automated reports": "Rapports automatisés",
            "AI assistant": "Assistant IA",
            "Alerts & notifications": "Alertes et notifications",
            "Compare datasets": "Comparer les jeux de données",
            "Settings & security": "Paramètres et sécurité",
            "Monitoring history": "Historique de surveillance",
        },
    },
    "hi": {
        "title": "उन्नत निगरानी डैशबोर्ड",
        "caption": "गुणवत्ता, ड्रिफ्ट, विसंगतियां, अलर्ट, पूर्वानुमान, XAI और इतिहास।",
        "overview": "अवलोकन",
        "run": "निगरानी चलाएं",
        "lineage": "डेटा इतिहास और संस्करण",
        "schedules": "समय-सारणी",
        "history": "इतिहास और अलर्ट",
        "assistant": "वॉइस सहायक",
        "reports": "रिपोर्ट",
        "quality": "नवीनतम गुणवत्ता स्कोर",
        "drift": "नवीनतम ड्रिफ्ट स्कोर",
        "anomalies": "नवीनतम विसंगतियां",
        "runs": "ऐतिहासिक रन",
        "quality_trend": "गुणवत्ता रुझान",
        "drift_trend": "ड्रिफ्ट रुझान",
        "anomaly_trend": "विसंगति रुझान",
        "pages": {
            "Overview": "अवलोकन",
            "Data sources": "डेटा स्रोत",
            "Data quality": "डेटा गुणवत्ता",
            "Anomalies": "विसंगतियां",
            "Data drift": "डेटा ड्रिफ्ट",
            "Trends & forecasting": "रुझान और पूर्वानुमान",
            "Explainable AI": "व्याख्येय AI",
            "Run monitoring": "निगरानी चलाएं",
            "Scheduled monitoring": "निर्धारित निगरानी",
            "Automated reports": "स्वचालित रिपोर्ट",
            "AI assistant": "AI सहायक",
            "Alerts & notifications": "अलर्ट और सूचनाएं",
            "Compare datasets": "डेटासेट तुलना",
            "Settings & security": "सेटिंग और सुरक्षा",
            "Monitoring history": "निगरानी इतिहास",
        },
    },
}


def _render_dashboard_preferences(
    api_base_url: str,
    headers: dict[str, str],
) -> dict[str, Any]:
    response = _get(api_base_url, headers, "/api/preferences")
    preferences = response.get("preferences", {}) if response else {}
    if not isinstance(preferences, dict):
        preferences = {}
    defaults = {
        "language": "en",
        "theme": "dark",
        "widgets": [
            "quality_score",
            "drift_score",
            "anomalies",
            "historical_runs",
            "datasets",
            "active_schedules",
            "missing_values",
            "duplicate_rows",
        ],
    }
    preferences = {**defaults, **preferences}
    supported_languages = {"en", "es", "fr", "hi"}
    if (
        not isinstance(preferences.get("language"), str)
        or preferences["language"] not in supported_languages
    ):
        preferences["language"] = "en"
    supported_themes = {"system", "light", "dark", "high_contrast"}
    if (
        not isinstance(preferences.get("theme"), str)
        or preferences["theme"] not in supported_themes
    ):
        preferences["theme"] = "dark"
    if not isinstance(preferences.get("widgets"), list):
        preferences["widgets"] = defaults["widgets"]
    with st.expander("Dashboard settings", expanded=False):
        with st.form("dashboard-preferences"):
            language_labels = {
                "en": "English",
                "es": "Español",
                "fr": "Français",
                "hi": "हिन्दी",
            }
            language = st.selectbox(
                "Dashboard language",
                list(language_labels),
                format_func=language_labels.get,
                index=list(language_labels).index(preferences["language"])
                if preferences["language"] in language_labels else 0,
            )
            themes = ["system", "light", "dark", "high_contrast"]
            theme = st.selectbox(
                "Theme",
                themes,
                index=themes.index(preferences["theme"])
                if preferences["theme"] in themes else 0,
            )
            widget_labels = {
                "quality_score": "Quality score",
                "drift_score": "Drift score",
                "anomalies": "Anomaly count",
                "historical_runs": "Historical runs",
                "datasets": "Datasets monitored",
                "active_schedules": "Active schedules",
                "missing_values": "Missing-value rate",
                "duplicate_rows": "Duplicate-row rate",
            }
            widgets = st.multiselect(
                "Overview widgets",
                list(widget_labels),
                default=[
                    item for item in preferences["widgets"]
                    if item in widget_labels
                ] or list(widget_labels),
                format_func=widget_labels.get,
            )
            submitted = st.form_submit_button("Save dashboard settings")
        if submitted:
            if not widgets:
                st.error("Select at least one overview widget.")
            else:
                try:
                    save_response = requests.post(
                        f"{api_base_url}/api/preferences",
                        headers=headers,
                        json={
                            "preferences": {
                                "language": language,
                                "theme": theme,
                                "widgets": widgets,
                            }
                        },
                        timeout=20,
                    )
                except requests.RequestException as error:
                    st.error(f"Could not save dashboard settings: {error}")
                else:
                    if save_response.status_code != 200:
                        st.error(_response_error(save_response, "Could not save dashboard settings."))
                    else:
                        st.success("Dashboard settings saved.")
                        st.rerun()

    theme = preferences["theme"]
    palette = {
        "dark": ("#0b1220", "#e8eef8", "#111b2d", "#24344b"),
        "light": ("#f3f6fb", "#182538", "#ffffff", "#dbe3ef"),
        "high_contrast": ("#000000", "#ffffff", "#080808", "#ffffff"),
        "system": ("#f3f6fb", "#182538", "#ffffff", "#dbe3ef"),
    }
    background, foreground, surface, border = palette.get(theme, palette["dark"])
    system_dark_palette = (
        theme == "system"
        and """:root {
          color-scheme: dark;
          --monitor-bg: #0b1220;
          --monitor-fg: #e8eef8;
          --monitor-surface: #111b2d;
          --monitor-border: #24344b;
          --monitor-muted: #9eacc1;
        }"""
    )
    st.markdown(
        f"""<style>
        :root {{
          color-scheme: {"dark" if theme == "dark" else "light"};
          --monitor-bg: {background};
          --monitor-fg: {foreground};
          --monitor-surface: {surface};
          --monitor-border: {border};
          --monitor-accent: #168f86;
          --monitor-muted: {"#9eacc1" if theme == "dark" else "#607086"};
        }}
        [data-testid="stAppViewContainer"] {{
          background: var(--monitor-bg);
          color: var(--monitor-fg);
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
          background: var(--monitor-surface);
          border-right: 1px solid var(--monitor-border);
        }}
        [data-testid="stMainBlockContainer"] {{
          max-width: 1600px;
          padding-top: 1.2rem;
        }}
        [data-testid="stMetric"], [data-testid="stExpander"], div[data-testid="stForm"] {{
          background: var(--monitor-surface);
          border: 1px solid var(--monitor-border);
          border-radius: 12px;
          padding: 0.85rem 1rem;
        }}
        [data-testid="stMetricLabel"] p {{ color: var(--monitor-muted); }}
        button[kind="primary"] {{
          background: var(--monitor-accent);
          border-color: var(--monitor-accent);
          transition: filter 120ms ease, transform 120ms ease;
        }}
        button[kind="primary"]:hover {{ filter: brightness(1.08); transform: translateY(-1px); }}
        :focus-visible {{ outline: 3px solid #f0ad4e !important; outline-offset: 2px; }}
        .monitoring-kicker {{
          color: var(--monitor-accent);
          font-size: 0.76rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }}
        @media (max-width: 760px) {{
          [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: 0.55rem !important; }}
          [data-testid="column"] {{ min-width: min(100%, 220px) !important; flex: 1 1 100% !important; }}
          [data-testid="stMetric"] {{ padding: 0.6rem !important; }}
          [data-testid="stMainBlockContainer"] {{ padding: 0.8rem 0.8rem 2rem; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
          *, *::before, *::after {{ animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }}
        }}
        {"@media (prefers-color-scheme: dark) { " + system_dark_palette + " }" if system_dark_palette else ""}
        </style>""",
        unsafe_allow_html=True,
    )
    return preferences


def _render_reports(
    api_base_url: str,
    headers: dict[str, str],
    results: list[dict[str, Any]],
    role: str = "analyst",
) -> None:
    st.subheader("Automated monitoring reports")
    if results:
        reportable = [result for result in results if result.get("id")]
        labels = {
            f"{item.get('created_at', 'Run')} · {item.get('check_type', 'monitoring')} · {item['id']}": item
            for item in reportable
        }
        selected = st.selectbox("Monitoring run", list(labels), key="report-run")
        formats = st.multiselect(
            "Report formats",
            ["pdf", "xlsx", "pptx"],
            default=["pdf", "xlsx", "pptx"],
            key="report-formats",
        )
        if st.button(
            "Generate report",
            type="primary",
            key="generate-report",
            disabled=role == "viewer" or not formats,
        ):
            report = _post(
                api_base_url,
                headers,
                "/api/reports",
                {"result_id": labels[selected]["id"], "formats": formats},
                timeout=120,
                expected_status=201,
            )
            if report is not None:
                st.success(f"Generated report `{report['id']}`.")
                st.rerun()
    else:
        st.info("Run monitoring before generating a report.")

    saved_payload = _get(api_base_url, headers, "/api/reports", {"limit": 25})
    if saved_payload is None:
        return
    saved_reports = saved_payload.get("reports", [])
    st.markdown("#### Saved reports")
    if not saved_reports:
        st.info("No reports have been generated yet.")
        return
    for report in saved_reports:
        with st.expander(
            f"{pd.to_datetime(report['created_at'], unit='s', utc=True)} · {report['id']}"
        ):
            st.caption(f"Monitoring result: {report['result_id']}")
            for report_format in report.get("formats", []):
                download_key = f"report-download-{report['id']}-{report_format}"
                content = st.session_state.get(download_key)
                if content is None:
                    if st.button(
                        f"Prepare {report_format.upper()} download",
                        key=f"prepare-{report['id']}-{report_format}",
                    ):
                        try:
                            download = requests.get(
                                f"{api_base_url}/api/reports/{report['id']}/{report_format}",
                                headers=headers,
                                timeout=60,
                            )
                        except requests.RequestException as error:
                            st.error(f"Could not load {report_format.upper()} report: {error}")
                        else:
                            if download.status_code != 200:
                                st.error(_response_error(download, "Could not load report."))
                            else:
                                st.session_state[download_key] = download.content
                                st.rerun()
                else:
                    st.download_button(
                        f"Download {report_format.upper()}",
                        data=content,
                        file_name=f"monitoring_report_{report['id']}.{report_format}",
                        mime={
                            "pdf": "application/pdf",
                            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        }[report_format],
                        key=f"download-{report['id']}-{report_format}",
                    )


def _dataset_frame(
    api_base_url: str,
    headers: dict[str, str],
    dataset_id: str | None,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    if not dataset_id:
        return None, pd.DataFrame()
    dataset = _get(api_base_url, headers, f"/api/datasets/{dataset_id}")
    if dataset is None:
        return None, pd.DataFrame()
    return dataset, pd.DataFrame(dataset.get("data", []))


def _render_source_connection(
    api_base_url: str,
    headers: dict[str, str],
    providers: list[dict[str, Any]],
    role: str,
) -> None:
    if role == "viewer" or not providers:
        return
    labels = {
        "local_files": "Local file on API host",
        "sql": "SQL database",
        "mongodb": "MongoDB",
        "s3": "Amazon S3",
        "dropbox": "Dropbox",
        "google_drive": "Google Drive",
    }
    with st.expander("Connect and analyze a supported source"):
        provider_by_type = {item.get("type"): item for item in providers}
        source_type = st.selectbox(
            "Source type",
            list(provider_by_type),
            format_func=lambda value: labels.get(value, value),
            key="source-connector-type",
        )
        provider = provider_by_type[source_type]
        status = provider.get("status")
        if status != "ready":
            missing = (
                provider.get("missing_dependencies", [])
                + provider.get("missing_configuration", [])
            )
            st.warning(
                f"This connector is not ready on the API host: "
                f"{', '.join(missing) or status}."
            )
        with st.form("source-connection-form"):
            source: dict[str, Any] = {"type": source_type}
            file_type = ""
            if source_type == "local_files":
                source["path"] = st.text_input(
                    "File path allowed by DATASET_ALLOWED_DIRS",
                    placeholder=r"C:\data\incoming\events.csv",
                )
                file_type = st.selectbox(
                    "File format",
                    ["csv", "xlsx", "json", "parquet", "tsv", "txt"],
                )
            elif source_type == "sql":
                source["url"] = st.text_input(
                    "Database URL (masked)",
                    type="password",
                    help="Used only for this request; use a read-only database account.",
                )
                source["query"] = st.text_area(
                    "Read-only SELECT query",
                    placeholder="SELECT * FROM events",
                )
            elif source_type == "mongodb":
                source["uri"] = st.text_input(
                    "MongoDB URI (masked)",
                    type="password",
                )
                source["database"] = st.text_input("Database")
                source["collection"] = st.text_input("Collection")
            elif source_type == "s3":
                source["bucket"] = st.text_input("Bucket")
                source["key"] = st.text_input("Object key")
                file_type = st.selectbox(
                    "Object format",
                    ["csv", "xlsx", "json", "parquet", "tsv", "txt"],
                )
            elif source_type == "dropbox":
                source["path"] = st.text_input("Dropbox path")
                file_type = st.selectbox(
                    "File format",
                    ["csv", "xlsx", "json", "parquet", "tsv", "txt"],
                )
            elif source_type == "google_drive":
                source["file_id"] = st.text_input("Google Drive file ID")
                file_type = st.selectbox(
                    "File format",
                    ["csv", "xlsx", "json", "parquet", "tsv", "txt"],
                )
            if file_type:
                source["file_type"] = file_type
            submitted = st.form_submit_button(
                "Connect and analyze",
                type="primary",
                disabled=status != "ready",
            )
        if not submitted:
            return
        response = _post(
            api_base_url,
            headers,
            "/api/sources/analyze",
            {"sources": [source]},
            timeout=180,
        )
        if response is None:
            return
        reports = response.get("sources", [])
        failures = response.get("failed", [])
        for report in reports:
            st.success(
                f"Source analyzed: {report.get('rows', 0):,} rows × "
                f"{report.get('columns', 0):,} columns."
            )
            if report.get("result_id"):
                st.caption(f"Saved monitoring result: `{report['result_id']}`")
        for failure in failures:
            st.error(f"Source {failure.get('index', '')}: {failure.get('error', 'analysis failed')}")


def _latest_result(
    results: list[dict[str, Any]],
    dataset_id: str | None,
) -> dict[str, Any] | None:
    matching = [
        item for item in results
        if dataset_id is None or item.get("dataset_id") == dataset_id
    ]
    return max(matching, key=lambda item: item.get("created_at", 0), default=None)


def _check_results(item: dict[str, Any] | None) -> dict[str, Any]:
    result = item.get("result", {}) if item else {}
    return result.get("results", {}) if isinstance(result, dict) else {}


def _date_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        name = str(column)
        if "date" in name.lower() or "time" in name.lower():
            columns.append(name)
        elif not frame.empty and not pd.api.types.is_numeric_dtype(frame[column]):
            sample = frame[column].dropna().head(100)
            if sample.empty:
                continue
            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                utc=True,
                format="mixed",
            )
            if parsed.notna().mean() >= 0.8:
                columns.append(name)
    return list(dict.fromkeys(columns))


def _render_data_explorer(
    api_base_url: str,
    headers: dict[str, str],
    role: str,
    datasets: list[dict[str, Any]],
    selected_dataset_id: str | None,
) -> None:
    st.subheader("Data sources and dataset explorer")
    st.caption("Upload a supported tabular file, inspect saved snapshots, and review available source connectors.")
    if role == "viewer":
        st.info("Viewer access is read-only. An analyst or administrator can upload or connect datasets.")
    else:
        uploaded = st.file_uploader(
            "Drop a tabular file here or browse",
            type=["csv", "xlsx", "json", "parquet", "tsv", "txt"],
            help="Maximum upload size: 10 MiB. The file is parsed by the monitoring API.",
            key="explorer-file-upload",
        )
        upload_clicked = st.button(
            "Upload selected file",
            disabled=uploaded is None,
            type="primary",
            key="explorer-upload-button",
        )
        if uploaded is not None and upload_clicked:
            file_content = uploaded.getvalue()
            if len(file_content) > 10 * 1024 * 1024:
                st.error("The selected file exceeds the 10 MiB upload limit.")
            elif len(uploaded.name) > 255:
                st.error("The uploaded filename must be 255 characters or fewer.")
            else:
                extension = Path(uploaded.name).suffix.lower()
                content_type = {
                    ".csv": "text/csv",
                    ".xls": "application/vnd.ms-excel",
                    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ".json": "application/json",
                    ".parquet": "application/vnd.apache.parquet",
                    ".tsv": "text/tab-separated-values",
                    ".txt": "text/plain",
                }.get(extension, "application/octet-stream")
                try:
                    response = requests.post(
                        f"{api_base_url}/api/datasets/upload",
                        params={"name": uploaded.name},
                        data=file_content,
                        headers={**headers, "Content-Type": content_type},
                        timeout=90,
                    )
                except requests.RequestException as error:
                    st.error(f"Dataset upload failed: {error}")
                else:
                    if response.status_code != 201:
                        st.error(_response_error(response, "Dataset upload failed."))
                    else:
                        st.success(f"Saved dataset: {uploaded.name}")
                        st.rerun()

        with st.expander("Available database and cloud connectors"):
            capabilities = _get(api_base_url, headers, "/api/sources/capabilities")
            providers = capabilities.get("providers", []) if capabilities else []
            if providers:
                rows = [
                    {
                        "source": provider.get("type"),
                        "status": provider.get("status"),
                        "formats": ", ".join(provider.get("formats", [])),
                        "requirements": ", ".join(
                            provider.get("missing_dependencies", [])
                            + provider.get("missing_configuration", [])
                        ) or "Ready",
                    }
                    for provider in providers
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(
                    "Connector readiness comes from the API runtime. Credentials are configured server-side; "
                    "the dashboard does not store or display secret values."
                )
                _render_source_connection(
                    api_base_url, headers, providers, role
                )
            else:
                st.info("Source connector capabilities are unavailable.")

    if not datasets:
        st.info("No saved datasets yet.")
        return
    dataset_options = {
        f"{item.get('name', 'Dataset')} · {item['id']}": item
        for item in datasets
    }
    labels = list(dataset_options)
    selected_label = next(
        (label for label, item in dataset_options.items() if item["id"] == selected_dataset_id),
        labels[0],
    )
    selected = st.selectbox(
        "Saved dataset",
        labels,
        index=labels.index(selected_label),
        key="explorer-saved-dataset",
    )
    dataset = dataset_options[selected]
    full_dataset, frame = _dataset_frame(api_base_url, headers, dataset["id"])
    if full_dataset is None:
        return
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(frame):,}")
    metric_columns[1].metric("Columns", f"{len(frame.columns):,}")
    metric_columns[2].metric("Missing values", f"{int(frame.isna().sum().sum()):,}")
    metric_columns[3].metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
    st.markdown("#### Dataset preview")
    if frame.empty:
        st.info("This saved dataset contains no preview rows.")
    else:
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        profile = pd.DataFrame({
            "column": [str(column) for column in frame.columns],
            "data_type": [str(frame[column].dtype) for column in frame.columns],
            "missing": [int(frame[column].isna().sum()) for column in frame.columns],
            "missing_%": [
                round(float(frame[column].isna().mean() * 100), 2)
                for column in frame.columns
            ],
            "unique": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        })
        st.markdown("#### Column profile")
        st.dataframe(profile, use_container_width=True, hide_index=True)
    with st.expander("Lineage and dataset versions"):
        _render_lineage_versions(api_base_url, headers, datasets, role)


def _render_quality_page(
    results: list[dict[str, Any]],
    dataset_id: str | None,
    frame: pd.DataFrame,
) -> None:
    st.subheader("Data quality")
    latest = _latest_result(results, dataset_id)
    quality = _check_results(latest).get("quality", {})
    report = quality.get("report", {}) if isinstance(quality, dict) else {}
    if not report:
        st.info("No quality analysis is available for this dataset and time range. Run a quality check to populate this page.")
        return
    score = (quality.get("quality_score") or {}).get("overall_score")
    columns = st.columns(4)
    columns[0].metric("Quality score", "Unavailable" if score is None else f"{score:.1f}/100")
    columns[1].metric("Rows", f"{report.get('rows', 0):,}")
    columns[2].metric("Missing values", f"{report.get('total_nulls', 0):,}")
    columns[3].metric("Duplicate rows", f"{report.get('duplicates', 0):,}")
    if not frame.empty:
        missing_mask = frame.isna()
        if missing_mask.to_numpy().any():
            sample = missing_mask.iloc[:100, :40]
            heatmap = go.Figure(data=go.Heatmap(
                z=sample.astype(int).to_numpy(),
                x=[str(item) for item in sample.columns],
                y=[str(item) for item in sample.index],
                colorscale=[[0, "#168f86"], [1, "#d95d5d"]],
                zmin=0,
                zmax=1,
                colorbar={"title": "Missing"},
                hovertemplate="Row %{y}<br>%{x}<br>Missing: %{z}<extra></extra>",
            ))
            heatmap.update_layout(
                title="Missing-value map (sample of up to 100 rows and 40 columns)",
                xaxis_title="Column",
                yaxis_title="Source row",
            )
            st.plotly_chart(heatmap, use_container_width=True)
    null_counts = report.get("null_per_column", {})
    if isinstance(null_counts, dict) and null_counts:
        missing = pd.DataFrame(
            [{"Column": name, "Missing values": count} for name, count in null_counts.items()]
        ).sort_values("Missing values", ascending=False)
        missing = missing[missing["Missing values"] > 0]
        if not missing.empty:
            st.markdown("#### Missing values by column")
            st.plotly_chart(
                px.bar(missing, x="Column", y="Missing values", color="Missing values"),
                use_container_width=True,
            )
            st.dataframe(missing, use_container_width=True, hide_index=True)
        else:
            st.success("No missing values were reported in this run.")
    with st.expander("Schema, freshness, duplicate detection, and business rules"):
        st.json({
            "schema_validation": report.get("schema_validation"),
            "data_freshness": report.get("data_freshness"),
            "duplicate_detection": report.get("duplicate_detection"),
            "business_rule_validation": report.get("business_rule_validation"),
        })


def _render_anomalies_page(
    results: list[dict[str, Any]],
    dataset_id: str | None,
    frame: pd.DataFrame,
) -> None:
    st.subheader("Anomaly detection")
    latest = _latest_result(results, dataset_id)
    anomaly = _check_results(latest).get("anomalies")
    if not isinstance(anomaly, dict):
        st.info("No anomaly analysis is available for this dataset and time range. Run the anomaly check to populate this page.")
        return
    if not anomaly.get("available", True):
        st.info(anomaly.get("reason", "Anomaly analysis is unavailable for this dataset."))
        return
    columns = st.columns(3)
    columns[0].metric("Detected records", f"{anomaly.get('total_anomalies', 0):,}")
    columns[1].metric("Dataset share", f"{anomaly.get('anomaly_percentage', 0):.2f}%")
    columns[2].metric("Analysis", "Isolation Forest")
    indices = anomaly.get("indices", [])
    if indices:
        st.caption("Rows are the zero-based source indices returned by the anomaly detector.")
        valid_indices = [index for index in indices if index in frame.index]
        if valid_indices:
            anomalies = frame.loc[valid_indices].copy()
            anomalies.insert(0, "source_index", valid_indices)
            st.dataframe(anomalies, use_container_width=True, hide_index=True)
        else:
            st.dataframe(
                pd.DataFrame({"source_index": indices}),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            "This detector returns outlier row indices, not calibrated severity scores; "
            "the interface does not invent per-row severity."
        )
    else:
        st.success("No anomalous rows were returned by this analysis.")


def _render_drift_page(
    results: list[dict[str, Any]],
    dataset_id: str | None,
) -> None:
    st.subheader("Data drift")
    latest = _latest_result(results, dataset_id)
    drift = _check_results(latest).get("drift")
    if not isinstance(drift, dict):
        st.info("No drift analysis is available for this dataset and time range. Select a baseline and run drift monitoring.")
        return
    columns = st.columns(3)
    columns[0].metric("Drift detected", "Yes" if drift.get("drift_detected") else "No")
    columns[1].metric(
        "Overall drift score",
        "Unavailable" if drift.get("overall_drift_score") is None
        else f"{drift['overall_drift_score']:.3f}",
    )
    columns[2].metric("Severity", drift.get("overall_severity", drift.get("severity", "Unavailable")))
    feature_drift = drift.get(
        "feature_metrics",
        drift.get("features", drift.get("column_drift", {})),
    )
    if isinstance(feature_drift, dict) and feature_drift:
        st.dataframe(pd.json_normalize(feature_drift, sep=".").T, use_container_width=True)
    else:
        st.info(drift.get("reason", "This drift result has no feature-level detail."))
    with st.expander("Full drift evidence"):
        st.json(drift)


def _render_trends_forecast(
    api_base_url: str,
    headers: dict[str, str],
    frame: pd.DataFrame,
    dataset_id: str | None,
    role: str,
) -> None:
    st.subheader("Trends and forecasting")
    if frame.empty:
        st.info("Select a non-empty dataset to analyze trends or create a forecast.")
        return
    date_columns = _date_columns(frame)
    numeric_columns = [str(item) for item in frame.select_dtypes(include="number").columns]
    if not date_columns:
        st.warning("No date/time column was detected in this dataset.")
        return
    with st.form("trends-forecast-form"):
        date_column = st.selectbox("Date/time column", date_columns)
        value_column = st.selectbox(
            "Metric",
            ["Row volume"] + numeric_columns,
        )
        action = st.radio("Analysis", ["Trend analysis", "Forecast"], horizontal=True)
        periods = st.number_input("Forecast horizon (periods)", min_value=1, max_value=365, value=4)
        frequency = st.text_input("Forecast frequency", value="W", help="Pandas frequency such as D, W, or MS.")
        method = st.selectbox("Forecast method", ["auto", "arima", "linear", "prophet", "lstm"])
        submitted = st.form_submit_button(
            "Run analysis",
            type="primary",
            disabled=role == "viewer",
        )
    if not submitted:
        return
    request_payload = {
        "data": frame.to_dict(orient="records"),
        "dataset_id": dataset_id,
        "date_column": date_column,
        "value_column": None if value_column == "Row volume" else value_column,
    }
    endpoint = "/api/analytics/trends"
    if action == "Forecast":
        request_payload.update({
            "periods": int(periods),
            "frequency": frequency,
            "method": method,
        })
        endpoint = "/api/analytics/forecast"
    result = _post(api_base_url, headers, endpoint, request_payload, timeout=180)
    if result is None:
        return
    report = result.get("report", {})
    if not isinstance(report, dict):
        st.error("The analytics service returned an invalid report.")
        return
    values = report.get("history", []) + report.get("forecast", [])
    if action == "Trend analysis":
        weekly = report.get("weekly", [])
        moving_average = report.get("moving_average", [])
        if weekly:
            trend_frame = pd.DataFrame(weekly).assign(series="Observed")
            if moving_average:
                average_frame = pd.DataFrame(moving_average).assign(series="Moving average")
                trend_frame = pd.concat([trend_frame, average_frame], ignore_index=True)
            st.plotly_chart(
                px.line(
                    trend_frame,
                    x="date",
                    y="value",
                    color="series",
                    markers=True,
                    title=f"Weekly trend · {report.get('trend_direction', 'unknown')}",
                ),
                use_container_width=True,
            )
    if isinstance(values, list) and values and all(isinstance(row, dict) for row in values):
        chart_frame = pd.DataFrame(values)
        if report.get("forecast"):
            chart_frame = pd.concat(
                [
                    pd.DataFrame(report.get("history", [])).assign(kind="History"),
                    pd.DataFrame(report.get("forecast", [])).assign(kind="Forecast"),
                ],
                ignore_index=True,
            )
        figure = px.line(
            chart_frame,
            x="date",
            y="value",
            color="kind" if "kind" in chart_frame.columns else None,
            markers=True,
            title="Historical values and forecast",
        )
        st.plotly_chart(figure, use_container_width=True)
    if report.get("fallback_reason"):
        st.warning(report["fallback_reason"])
    with st.expander("Analysis details", expanded=True):
        st.json(report)


def _render_explainability(
    api_base_url: str,
    headers: dict[str, str],
    frame: pd.DataFrame,
    dataset_id: str | None,
    role: str,
) -> None:
    st.subheader("Explainable AI and model evaluation")
    if role == "viewer":
        st.info("Viewer access is read-only. An analyst or administrator can train an explanation model.")
    if frame.empty:
        st.info("Select a non-empty dataset to compare candidate models.")
        return
    with st.form("explainability-form"):
        target = st.selectbox("Target column", list(frame.columns))
        task = st.selectbox("Prediction task", ["classification", "regression"])
        test_size = st.slider("Test data fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        submitted = st.form_submit_button(
            "Compare models",
            type="primary",
            disabled=role == "viewer",
        )
    if not submitted:
        return
    report = _post(
        api_base_url,
        headers,
        "/api/analytics/explain",
        {
            "data": frame.to_dict(orient="records"),
            "dataset_id": dataset_id,
            "target_column": target,
            "task": task,
            "test_size": test_size,
        },
        timeout=300,
    )
    if report is None:
        return
    result = report.get("report", {})
    if not isinstance(result, dict):
        st.error("The explainability service returned an invalid report.")
        return
    summary = st.columns(3)
    summary[0].metric("Selected model", result.get("selected_model", "Unavailable"))
    summary[1].metric("Selection metric", result.get("metric", "Unavailable"))
    summary[2].metric("Best score", result.get("best_score", "Unavailable"))
    models = result.get("models", [])
    if models:
        model_frame = pd.DataFrame([
            {"model": item.get("model"), "score": item.get("score"), **item.get("metrics", {})}
            for item in models
        ])
        st.dataframe(model_frame, use_container_width=True, hide_index=True)
        importance = result.get("best_model_summary", {}).get("feature_importance", [])
        if importance:
            importance_frame = pd.DataFrame(importance)
            if "value" in importance_frame.columns and "importance" not in importance_frame.columns:
                importance_frame = importance_frame.rename(columns={"value": "importance"})
            if {"feature", "importance"}.issubset(importance_frame.columns):
                st.plotly_chart(
                    px.bar(
                        importance_frame.sort_values("importance"),
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Feature importance for selected model",
                    ),
                    use_container_width=True,
                )
    if result.get("model_failures"):
        with st.expander("Models that could not be evaluated"):
            st.dataframe(pd.DataFrame(result["model_failures"]), use_container_width=True, hide_index=True)
    with st.expander("Explanation details"):
        st.json(result)


def _render_monitoring_run(
    api_base_url: str,
    headers: dict[str, str],
    role: str,
    datasets: list[dict[str, Any]],
    selected_dataset_id: str | None,
) -> None:
    st.subheader("Run monitoring")
    if role == "viewer":
        st.info("Viewer access is read-only. An analyst or administrator can start monitoring runs.")
        return
    if not datasets:
        st.info("Upload or save a dataset before starting a monitoring run.")
        return
    options = {f"{item['name']} · {item['id']}": item for item in datasets}
    labels = list(options)
    default_label = next(
        (label for label, item in options.items() if item["id"] == selected_dataset_id),
        labels[0],
    )
    selected_label = st.selectbox(
        "Dataset",
        labels,
        index=labels.index(default_label),
        key="run-monitoring-dataset",
    )
    dataset = options[selected_label]
    payload, frame = _dataset_frame(api_base_url, headers, dataset["id"])
    if payload is None:
        return
    all_labels = {f"{item['name']} · {item['id']}": item for item in datasets}
    baseline_labels = [
        label for label, item in all_labels.items() if item["id"] != dataset["id"]
    ]
    checks = st.multiselect(
        "Monitoring checks",
        ["quality", "drift", "anomalies", "forecast", "explain"],
        default=["quality", "anomalies"],
        key="run-monitoring-checks",
    )
    baseline = None
    if "drift" in checks:
        if baseline_labels:
            baseline = st.selectbox("Baseline dataset", baseline_labels, key="run-drift-baseline")
        else:
            st.warning("A second saved dataset is required for drift analysis.")
    date_column = None
    value_column = None
    if "forecast" in checks:
        date_columns = _date_columns(frame)
        numeric_columns = [str(item) for item in frame.select_dtypes(include="number").columns]
        if not date_columns:
            st.warning("Forecasting requires a date/time column.")
        else:
            date_column = st.selectbox("Forecast date/time column", date_columns)
            value_column = st.selectbox("Forecast metric", ["Row volume"] + numeric_columns)
    target_column = None
    if "explain" in checks and not frame.empty:
        target_column = st.selectbox("Explainability target column", list(frame.columns))
    notify = st.checkbox("Send configured alerts when findings are detected", value=False)
    if st.button("Run selected checks", type="primary", disabled=not checks, key="run-monitoring"):
        if "drift" in checks and baseline is None:
            st.error("Select a baseline dataset before running drift analysis.")
            return
        if "forecast" in checks and date_column is None:
            st.error("Select or provide a date/time column for forecasting.")
            return
        if "explain" in checks and target_column is None:
            st.error("Select a target column for explainability.")
            return
        request_payload: dict[str, Any] = {
            "dataset_id": dataset["id"],
            "checks": checks,
            "notify": notify,
        }
        if baseline:
            request_payload["baseline_dataset_id"] = all_labels[baseline]["id"]
        if "forecast" in checks:
            request_payload["date_column"] = date_column
            request_payload["value_column"] = (
                None if value_column == "Row volume" else value_column
            )
        if target_column:
            request_payload["target_column"] = target_column
        if _run_monitoring(api_base_url, headers, request_payload):
            st.rerun()


def _render_alert_workspace(
    api_base_url: str,
    headers: dict[str, str],
    role: str,
) -> None:
    st.subheader("Alerts and notifications")
    if role != "admin":
        st.info("Alert history and rule configuration are restricted to administrators because they may include recipients.")
        return
    response = _get(api_base_url, headers, "/api/alerts/history", {"limit": 500})
    if response is None:
        return
    history = response.get("history", [])
    if history:
        severity_counts = (
            pd.DataFrame(history)["severity"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("severity")
            .reset_index(name="alerts")
        )
        st.plotly_chart(
            px.bar(
                severity_counts,
                x="severity",
                y="alerts",
                color="severity",
                title="Recent alerts by severity",
            ),
            use_container_width=True,
        )
        severities = sorted({
            str(item.get("severity", "unknown")) for item in history
        })
        chosen_severities = st.multiselect(
            "Filter by severity",
            severities,
            default=severities,
            key="alert-severity-filter",
        )
        query = st.text_input("Search alert history", key="alert-search").strip().lower()
        filtered = [
            item for item in history
            if str(item.get("severity", "unknown")) in chosen_severities
            and (
                not query
                or query in str(item.get("message", "")).lower()
                or query in str(item.get("event_type", "")).lower()
            )
        ]
        st.dataframe(
            pd.DataFrame(filtered),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No alerts have been recorded.")
    st.caption("Alert acknowledgement and resolution are not available in the current API.")

    rules_response = _get(api_base_url, headers, "/api/alerts/rules")
    if rules_response is None:
        return
    rules = rules_response.get("rules", [])
    st.markdown("#### Alert rules")
    if rules:
        st.dataframe(pd.DataFrame(rules), use_container_width=True, hide_index=True)
    else:
        st.info("No alert rules are configured.")
    with st.form("add-alert-rule"):
        metric = st.text_input(
            "Metric",
            value="quality_score",
            help="Examples: quality_score, risk_level, severity, total_nulls, duplicates, drift_score, drift_detected.",
        )
        operator = st.selectbox("Condition", [">", ">=", "<", "<=", "==", "!="])
        expected_value = st.text_input("Threshold value")
        channel = st.selectbox("Notification channel", ["None", "email", "slack", "teams", "sms", "whatsapp"])
        recipient = st.text_input("Recipient (if required)")
        add_rule = st.form_submit_button("Add alert rule")
    if add_rule:
        try:
            parsed_value: Any = float(expected_value)
        except ValueError:
            parsed_value = expected_value
        if not metric.strip() or not expected_value.strip():
            st.error("Enter both a metric name and a threshold value.")
            return
        new_rule = {
            "metric": metric.strip(),
            "operator": operator,
            "value": parsed_value,
            "channel": None if channel == "None" else channel,
            "recipient": recipient,
        }
        saved = _post(
            api_base_url,
            headers,
            "/api/alerts/rules",
            {"rules": [*rules, new_rule]},
        )
        if saved is not None:
            st.success("Alert rule added.")
            st.rerun()
    if rules:
        confirm_clear = st.checkbox("I understand this removes all configured alert rules.")
        if st.button("Clear all alert rules", disabled=not confirm_clear, key="clear-alert-rules"):
            try:
                cleared = requests.delete(
                    f"{api_base_url}/api/alerts/rules",
                    headers=headers,
                    timeout=20,
                )
            except requests.RequestException as error:
                st.error(f"Could not clear alert rules: {error}")
            else:
                if cleared.status_code == 200:
                    st.success("Alert rules cleared.")
                    st.rerun()
                else:
                    st.error(_response_error(cleared, "Could not clear alert rules."))


def _render_dataset_comparison(
    api_base_url: str,
    headers: dict[str, str],
    datasets: list[dict[str, Any]],
    role: str,
) -> None:
    st.subheader("Dataset comparison")
    if len(datasets) < 2:
        st.info("Save at least two datasets or versions to compare their schema, quality, and distributions.")
        return
    options = {f"{item['name']} · {item['id']}": item for item in datasets}
    first, second = st.columns(2)
    baseline_label = first.selectbox("Baseline", list(options), key="compare-baseline")
    current_options = [
        label for label, item in options.items()
        if item["id"] != options[baseline_label]["id"]
    ]
    current_label = second.selectbox("Current", current_options, key="compare-current")
    if st.button(
        "Compare datasets",
        type="primary",
        key="compare-datasets",
        disabled=role == "viewer",
    ):
        response = _post(
            api_base_url,
            headers,
            "/api/datasets/compare",
            {
                "baseline_dataset_id": options[baseline_label]["id"],
                "current_dataset_id": options[current_label]["id"],
            },
        )
        if response is not None:
            st.session_state["dashboard_comparison"] = response
    comparison = st.session_state.get("dashboard_comparison")
    if not isinstance(comparison, dict):
        return
    quality = comparison.get("quality", {})
    baseline_quality = quality.get("baseline", {})
    current_quality = quality.get("current", {})
    metrics = st.columns(3)
    metrics[0].metric(
        "Quality score",
        f"{current_quality.get('quality_score', 'Unavailable')}",
        delta=(
            round(
                current_quality["quality_score"] - baseline_quality["quality_score"],
                2,
            )
            if isinstance(current_quality.get("quality_score"), (int, float))
            and isinstance(baseline_quality.get("quality_score"), (int, float))
            else None
        ),
    )
    metrics[1].metric("Current rows", current_quality.get("rows", "Unavailable"))
    metrics[2].metric("Current duplicate rows", current_quality.get("duplicate_rows", "Unavailable"))
    schema = comparison.get("schema", {})
    st.markdown("#### Schema changes")
    st.json({
        "added_columns": schema.get("added_columns", []),
        "removed_columns": schema.get("removed_columns", []),
        "type_changes": schema.get("type_changes", []),
        "shared_columns": schema.get("shared_columns", []),
    })
    st.markdown("#### Distribution drift")
    st.json(comparison.get("distribution_drift", {}))
    if comparison.get("result_id"):
        st.caption(f"Comparison saved to monitoring history: {comparison['result_id']}")


def _render_settings(role: str, health_ok: bool) -> None:
    st.subheader("Settings and security")
    account, service = st.columns(2)
    account.metric("Current role", role)
    service.metric("API health", "Ready" if health_ok else "Unavailable")
    st.info(
        "Language, theme, and overview widgets are managed in Dashboard settings in the sidebar. "
        "Account and access controls are enforced by the authenticated API."
    )
    st.caption(
        "Secrets and connector credentials are configured in the API environment and are not displayed here."
    )


def render_monitoring_dashboard(
    api_base_url: str,
    headers: dict[str, str],
    role: str,
) -> None:
    with st.sidebar:
        st.markdown("### DATA OBSERVABILITY")
        preferences = _render_dashboard_preferences(api_base_url, headers)
        page_keys = [
            "Overview",
            "Data sources",
            "Data quality",
            "Anomalies",
            "Data drift",
            "Trends & forecasting",
            "Explainable AI",
            "Run monitoring",
            "Scheduled monitoring",
            "Automated reports",
            "AI assistant",
            "Alerts & notifications",
            "Compare datasets",
            "Settings & security",
            "Monitoring history",
        ]
        page_labels = _DASHBOARD_COPY.get(
            preferences.get("language", "en"),
            _DASHBOARD_COPY["en"],
        ).get("pages", {})
        selected_page = st.radio(
            "Workspace",
            page_keys,
            format_func=lambda page: page_labels.get(page, page),
            key="monitoring-workspace-navigation",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Signed in · **{role.title()}**")

    language = preferences.get("language", "en")
    copy = _DASHBOARD_COPY.get(language, _DASHBOARD_COPY["en"])
    page_title = copy.get("pages", {}).get(selected_page, selected_page)
    datasets_payload = _get(api_base_url, headers, "/api/datasets", {"limit": 500})
    if datasets_payload is None:
        return
    datasets = datasets_payload.get("datasets") or []
    results_payload = _get(api_base_url, headers, "/api/monitoring/results", {"limit": 500})
    if results_payload is None:
        return
    all_results = results_payload.get("results") or []
    schedules_payload = _get(api_base_url, headers, "/api/schedules")
    if schedules_payload is None:
        return
    schedules = schedules_payload.get("schedules") or []

    health_ok = False
    try:
        health_response = requests.get(f"{api_base_url}/api/health/ready", timeout=5)
        health_ok = health_response.status_code == 200
    except requests.RequestException:
        health_ok = False

    today = pd.Timestamp.now().date()
    header_columns = st.columns([2.2, 1.1, 1.8, 0.9])
    with header_columns[0]:
        st.markdown('<div class="monitoring-kicker">Data observability platform</div>', unsafe_allow_html=True)
        st.title(page_title)
    with header_columns[1]:
        dataset_options = {
            "All datasets": None,
            **{
                f"{item.get('name', 'Dataset')} · {item['id']}": item["id"]
                for item in datasets
            },
        }
        selected_dataset_label = st.selectbox(
            "Dataset",
            list(dataset_options),
            key="global-dataset-filter",
        )
        selected_dataset_id = dataset_options[selected_dataset_label]
    with header_columns[2]:
        search = st.text_input(
            "Search datasets and monitoring runs",
            placeholder="Search names, checks, or result IDs",
            key="global-monitoring-search",
        ).strip().lower()
        date_range = st.date_input(
            "Monitoring date range",
            value=(today - pd.Timedelta(days=30), today),
            key="global-monitoring-date-range",
        )
    with header_columns[3]:
        if health_ok:
            st.success("API ready", icon="✅")
        else:
            st.error("API unavailable", icon="⚠️")

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, tuple) and len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date = end_date = today
    dataset_name_by_id = {
        item["id"]: str(item.get("name", ""))
        for item in datasets
    }
    results = _filter_monitoring_results(
        all_results,
        dataset_name_by_id,
        selected_dataset_id,
        search,
        start_date,
        end_date,
    )
    window_days = max((end_date - start_date).days + 1, 1)
    previous_end = start_date - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=window_days - 1)
    previous_results = _filter_monitoring_results(
        all_results,
        dataset_name_by_id,
        selected_dataset_id,
        search,
        previous_start,
        previous_end,
    )

    latest_alert_count: int | None = None
    recent_alerts: list[dict[str, Any]] = []
    if role == "admin":
        alerts_response = _get(api_base_url, headers, "/api/alerts/history", {"limit": 100})
        if alerts_response is not None:
            recent_alerts = alerts_response.get("history", [])
            latest_alert_count = len(recent_alerts)

    data_view_dataset_id = selected_dataset_id
    if (
        data_view_dataset_id is None
        and selected_page in {
            "Data quality",
            "Anomalies",
            "Trends & forecasting",
            "Explainable AI",
            "Run monitoring",
        }
    ):
        latest_result = _latest_result(results, None)
        if latest_result is not None:
            data_view_dataset_id = latest_result.get("dataset_id")
    dataset_payload, frame = _dataset_frame(
        api_base_url,
        headers,
        data_view_dataset_id,
    )
    if data_view_dataset_id and dataset_payload is None:
        return
    st.caption(
        f"{len(datasets):,} saved datasets · {len(results):,} matching monitoring runs · "
        f"{sum(bool(item.get('enabled')) for item in schedules):,} active schedules"
        + (f" · {latest_alert_count:,} recent alerts" if latest_alert_count is not None else "")
    )

    if selected_page == "Overview":
        _render_overview(
            results,
            preferences.get("widgets"),
            copy,
            datasets,
            schedules,
            previous_results,
        )
        if results:
            st.markdown("#### Recent monitoring activity")
            history = pd.DataFrame(_result_rows(results)).sort_values(
                "created_at", ascending=False
            )
            st.dataframe(history.head(10), use_container_width=True, hide_index=True)
        if recent_alerts:
            severity_counts = (
                pd.DataFrame(recent_alerts)["severity"]
                .fillna("unknown")
                .value_counts()
                .rename_axis("severity")
                .reset_index(name="alerts")
            )
            st.plotly_chart(
                px.bar(
                    severity_counts,
                    x="severity",
                    y="alerts",
                    color="severity",
                    title="Recent alert severity",
                ),
                use_container_width=True,
            )
        st.markdown("#### Priority attention")
        if not results:
            st.info("No monitoring findings in the selected date range.")
        else:
            latest = _latest_result(results, selected_dataset_id)
            checks = _check_results(latest)
            quality = checks.get("quality", {}).get("report", {})
            anomalies = checks.get("anomalies", {})
            drift = checks.get("drift", {})
            issues = []
            if quality.get("total_nulls", 0):
                issues.append(f"{quality['total_nulls']:,} missing values were found in the latest quality check.")
            if anomalies.get("total_anomalies", 0):
                issues.append(f"{anomalies['total_anomalies']:,} anomalous rows were detected.")
            if drift.get("drift_detected"):
                issues.append(
                    f"Drift was detected (severity: {drift.get('overall_severity', drift.get('severity', 'unknown'))})."
                )
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("No quality, anomaly, or drift issues were flagged by the latest available checks.")
    elif selected_page == "Data sources":
        _render_data_explorer(
            api_base_url, headers, role, datasets, selected_dataset_id
        )
    elif selected_page == "Data quality":
        _render_quality_page(
            results, data_view_dataset_id, frame
        )
    elif selected_page == "Anomalies":
        _render_anomalies_page(results, data_view_dataset_id, frame)
    elif selected_page == "Data drift":
        _render_drift_page(results, selected_dataset_id)
    elif selected_page == "Trends & forecasting":
        _render_trends_forecast(
            api_base_url, headers, frame, data_view_dataset_id, role
        )
    elif selected_page == "Explainable AI":
        _render_explainability(
            api_base_url, headers, frame, data_view_dataset_id, role
        )
    elif selected_page == "Run monitoring":
        _render_monitoring_run(
            api_base_url, headers, role, datasets, selected_dataset_id
        )
        if st.session_state.get("dashboard_latest_run_id"):
            st.caption(f"Most recent run: `{st.session_state['dashboard_latest_run_id']}`")
    elif selected_page == "Scheduled monitoring":
        _render_schedules(api_base_url, headers, role, datasets, schedules)
    elif selected_page == "Automated reports":
        _render_reports(api_base_url, headers, results, role)
    elif selected_page == "AI assistant":
        _render_voice_assistant(api_base_url, headers, datasets)
    elif selected_page == "Alerts & notifications":
        _render_alert_workspace(api_base_url, headers, role)
    elif selected_page == "Compare datasets":
        _render_dataset_comparison(api_base_url, headers, datasets, role)
    elif selected_page == "Settings & security":
        _render_settings(role, health_ok)
    elif selected_page == "Monitoring history":
        _render_history(api_base_url, headers, results, role)
