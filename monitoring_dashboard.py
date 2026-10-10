"""Unified Streamlit monitoring dashboard views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
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

    try:
        response = requests.post(
            f"{api_base_url}/api/assistant/ask",
            headers=headers,
            json={
                "question": question,
                "language": language_code,
                "data": dataset_response.get("data", []),
            },
            timeout=60,
        )
    except requests.RequestException as error:
        st.error(f"Assistant request failed: {error}")
        return
    if response.status_code != 200:
        st.error(response.json().get("detail", "The assistant could not answer."))
        return
    answer = response.json()
    st.session_state["assistant_last_answer"] = answer["answer"]
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


def _render_overview(
    results: list[dict[str, Any]],
    widgets: list[str] | None = None,
    copy: dict[str, str] | None = None,
) -> None:
    copy = copy or _DASHBOARD_COPY["en"]
    st.subheader(copy["overview"])
    if not results:
        st.info("No monitoring runs yet. Start a run below to populate your dashboard.")
        return

    rows = _result_rows(results)
    history = pd.DataFrame(rows).sort_values("created_at")
    latest = history.iloc[-1]
    metrics = {
        "quality_score": (
            copy["quality"],
            "—" if pd.isna(latest["quality_score"]) else f"{latest['quality_score']:.1f}/100",
        ),
        "drift_score": (
            copy["drift"],
            "—" if pd.isna(latest["drift_score"]) else f"{latest['drift_score']:.1f}",
        ),
        "anomalies": (
            copy["anomalies"],
            "—" if pd.isna(latest["anomalies"]) else int(latest["anomalies"]),
        ),
        "historical_runs": (copy["runs"], len(history)),
    }
    chosen_widgets = [key for key in (widgets or list(metrics)) if key in metrics]
    columns = st.columns(max(len(chosen_widgets), 1))
    for column, key in zip(columns, chosen_widgets):
        column.metric(*metrics[key])

    quality_history = history.dropna(subset=["quality_score"])
    drift_history = history.dropna(subset=["drift_score"])
    quality_tab, drift_tab, anomaly_tab = st.tabs(
        [copy["quality_trend"], copy["drift_trend"], copy["anomaly_trend"]]
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
            disabled=baseline_label == current_label,
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
                    st.error(response.json().get("detail", "Dataset comparison failed."))
                else:
                    st.json(response.json())

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
                                st.success(
                                    f"Created version v{response.json()['version_number']}."
                                )
                                st.rerun()
                            else:
                                st.error(
                                    response.json().get(
                                        "detail", "Could not create dataset version."
                                    )
                                )


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
                        st.success(f"Schedule created: `{response.json()['id']}`")
                        st.rerun()
                    else:
                        st.error(response.json().get("detail", "Could not create schedule."))


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
        "widgets": ["quality_score", "drift_score", "anomalies", "historical_runs"],
    }
    preferences = {**defaults, **preferences}
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
                        st.error(save_response.json().get("detail", "Could not save dashboard settings."))
                    else:
                        st.success("Dashboard settings saved.")
                        st.rerun()

    theme = preferences["theme"]
    palette = {
        "dark": ("#0e1117", "#f4f6f8"),
        "light": ("#f8fafc", "#17212b"),
        "high_contrast": ("#000000", "#ffffff"),
        "system": ("transparent", "inherit"),
    }
    background, foreground = palette.get(theme, palette["dark"])
    st.markdown(
        f"""<style>
        [data-testid="stAppViewContainer"] {{ background: {background}; color: {foreground}; }}
        @media (max-width: 760px) {{
          [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: 0.5rem !important; }}
          [data-testid="column"] {{ min-width: min(100%, 260px) !important; flex: 1 1 100% !important; }}
          [data-testid="stMetric"] {{ padding: 0.55rem !important; }}
        }}
        </style>""",
        unsafe_allow_html=True,
    )
    return preferences


def _render_reports(
    api_base_url: str,
    headers: dict[str, str],
    results: list[dict[str, Any]],
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
        if st.button("Generate report", type="primary", key="generate-report"):
            try:
                response = requests.post(
                    f"{api_base_url}/api/reports",
                    headers=headers,
                    json={"result_id": labels[selected]["id"], "formats": formats},
                    timeout=120,
                )
            except requests.RequestException as error:
                st.error(f"Report generation failed: {error}")
            else:
                if response.status_code != 201:
                    st.error(response.json().get("detail", "Report generation failed."))
                else:
                    st.success(f"Generated report `{response.json()['id']}`.")
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
                                st.error(download.json().get("detail", "Could not load report."))
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


def render_monitoring_dashboard(
    api_base_url: str,
    headers: dict[str, str],
    role: str,
) -> None:
    preferences = _render_dashboard_preferences(api_base_url, headers)
    language = preferences["language"]
    copy = _DASHBOARD_COPY.get(language, _DASHBOARD_COPY["en"])
    st.title(copy["title"])
    st.caption(copy["caption"])

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

    overview_tab, run_tab, lineage_tab, schedules_tab, history_tab, assistant_tab, reports_tab = st.tabs(
        [
            copy["overview"],
            copy["run"],
            copy["lineage"],
            copy["schedules"],
            copy["history"],
            copy["assistant"],
            copy["reports"],
        ]
    )
    with overview_tab:
        _render_overview(results, preferences["widgets"], copy)

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

    with lineage_tab:
        _render_lineage_versions(api_base_url, headers, datasets, role)

    with history_tab:
        _render_history(api_base_url, headers, results, role)

    with assistant_tab:
        _render_voice_assistant(api_base_url, headers, datasets)

    with reports_tab:
        _render_reports(api_base_url, headers, results)
