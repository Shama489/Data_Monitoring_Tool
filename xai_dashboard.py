"""Interactive explainability dashboard for the data monitoring tool."""

import io
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from profiler import train_and_explain_model
from monitoring_dashboard import render_monitoring_dashboard


st.set_page_config(page_title="Explainable AI Monitor", page_icon="XAI", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _api_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state['access_token']}"}


def _render_login() -> None:
    st.title("Explainable AI Monitor")
    st.caption("Sign in with your Data Monitoring Tool account to continue.")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/auth/login",
                json={"username": username, "password": password},
                timeout=10,
            )
        except requests.RequestException as error:
            st.error(f"Could not reach the authentication service: {error}")
            return
        if response.status_code != 200:
            st.error(response.json().get("detail", "Sign in failed."))
            return
        session = response.json()
        st.session_state["access_token"] = session["access_token"]
        st.session_state["user"] = session["user"]
        st.session_state["login_password"] = ""
        st.rerun()


if "access_token" not in st.session_state:
    _render_login()
    st.stop()

try:
    identity_response = requests.get(
        f"{API_BASE_URL}/api/auth/me",
        headers=_api_headers(),
        timeout=10,
    )
except requests.RequestException as error:
    st.error(f"Could not verify your session with the API: {error}")
    st.stop()
if identity_response.status_code != 200:
    st.session_state.pop("access_token", None)
    st.session_state.pop("user", None)
    st.warning("Your session expired. Please sign in again.")
    _render_login()
    st.stop()
st.session_state["user"] = identity_response.json()

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state['user']['username']}**")
    st.caption(f"Role: `{st.session_state['user']['role']}`")
    selected_page = st.radio(
        "Workspace",
        ["Monitoring Dashboard", "Explainable AI"],
        index=0,
    )
    if st.button("Sign out"):
        st.session_state.pop("access_token", None)
        st.session_state.pop("user", None)
        st.rerun()

    if st.session_state["user"]["role"] == "admin":
        with st.expander("Manage users"):
            with st.form("create-user"):
                new_username = st.text_input("New username")
                new_password = st.text_input("Temporary password (12+ chars)", type="password")
                new_role = st.selectbox("Role", ["analyst", "viewer"])
                create_user = st.form_submit_button("Create user")
            if create_user:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/auth/users",
                        headers=_api_headers(),
                        json={
                            "username": new_username,
                            "password": new_password,
                            "role": new_role,
                        },
                        timeout=10,
                    )
                except requests.RequestException as error:
                    st.error(f"Could not create the user: {error}")
                else:
                    if response.status_code == 201:
                        st.success(f"Created {new_role} account {new_username}.")
                    else:
                        st.error(response.json().get("detail", "Could not create the user."))
            try:
                users_response = requests.get(
                    f"{API_BASE_URL}/api/auth/users",
                    headers=_api_headers(),
                    timeout=10,
                )
            except requests.RequestException as error:
                st.error(f"Could not load user accounts: {error}")
            else:
                if users_response.status_code != 200:
                    st.error(users_response.json().get("detail", "Could not load user accounts."))
                else:
                    users = users_response.json()["users"]
                    if users:
                        with st.form("update-user"):
                            selected_user = st.selectbox(
                                "Account",
                                users,
                                format_func=lambda user: f"{user['username']} · {user['role']}",
                            )
                            updated_role = st.selectbox(
                                "Role",
                                ["admin", "analyst", "viewer"],
                                index=["admin", "analyst", "viewer"].index(selected_user["role"]),
                            )
                            updated_active = st.checkbox(
                                "Account active",
                                value=bool(selected_user["is_active"]),
                            )
                            update_submitted = st.form_submit_button("Update account")
                        if update_submitted:
                            try:
                                update_response = requests.patch(
                                    f"{API_BASE_URL}/api/auth/users/{selected_user['id']}",
                                    headers=_api_headers(),
                                    json={"role": updated_role, "is_active": updated_active},
                                    timeout=10,
                                )
                            except requests.RequestException as error:
                                st.error(f"Could not update the user: {error}")
                            else:
                                if update_response.status_code == 200:
                                    st.success(f"Updated {selected_user['username']}.")
                                else:
                                    st.error(update_response.json().get("detail", "Could not update the user."))

if selected_page == "Monitoring Dashboard":
    render_monitoring_dashboard(
        API_BASE_URL,
        _api_headers(),
        st.session_state["user"]["role"],
    )
    st.stop()


def _importance_frame(report: dict, key: str, value_name: str) -> pd.DataFrame:
    values = report.get(key) or []
    if not values:
        return pd.DataFrame(columns=["feature", value_name])
    frame = pd.DataFrame(values)
    return frame.rename(columns={frame.columns[-1]: value_name})


def _render_feature_chart(frame: pd.DataFrame, value_name: str, title: str, color: str) -> None:
    if frame.empty:
        st.info("No explanation values are available for this view.")
        return

    chart_frame = frame.sort_values(value_name, ascending=True).tail(20)
    figure = px.bar(
        chart_frame,
        x=value_name,
        y="feature",
        orientation="h",
        title=title,
        color=value_name,
        color_continuous_scale=color,
    )
    figure.update_layout(height=max(420, len(chart_frame) * 28), margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(figure, use_container_width=True)


st.title("Explainable AI Monitor")
st.caption("Train a baseline model, inspect global drivers, and review SHAP explanations.")

with st.sidebar:
    st.header("Model setup")
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset (max 10 MiB)",
        type=["csv"],
        disabled=st.session_state["user"]["role"] == "viewer",
    )
    task = st.selectbox("Task", ["classification", "regression"])
    test_size = st.slider("Holdout share", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

if uploaded_file is None:
    try:
        datasets_response = requests.get(
            f"{API_BASE_URL}/api/datasets",
            headers=_api_headers(),
            timeout=10,
        )
    except requests.RequestException as error:
        st.error(f"Could not load your saved datasets: {error}")
        st.stop()
    if datasets_response.status_code != 200:
        st.error(datasets_response.json().get("detail", "Could not load your saved datasets."))
        st.stop()
    saved_datasets = datasets_response.json()["datasets"]
    if saved_datasets:
        selected_dataset = st.selectbox(
            "Saved dataset",
            saved_datasets,
            format_func=lambda item: f"{item['name']} · {item['id']}",
        )
        if st.button("View saved dataset"):
            try:
                saved_response = requests.get(
                    f"{API_BASE_URL}/api/datasets/{selected_dataset['id']}",
                    headers=_api_headers(),
                    timeout=10,
                )
            except requests.RequestException as error:
                st.error(f"Could not load the saved dataset: {error}")
            else:
                if saved_response.status_code == 200:
                    st.dataframe(
                        pd.DataFrame(saved_response.json()["data"]).head(100),
                        use_container_width=True,
                    )
                else:
                    st.error(saved_response.json().get("detail", "Could not load the saved dataset."))
    elif st.session_state["user"]["role"] == "viewer":
        st.info("No datasets are available to your account. Ask an analyst to upload one.")
    else:
        st.info("Upload a CSV file to begin an explainability analysis.")
    st.stop()

if uploaded_file.size > MAX_UPLOAD_BYTES:
    st.error("The uploaded CSV exceeds the 10 MiB limit.")
    st.stop()

try:
    dataset = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if dataset.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

with st.expander(f"Dataset preview · {len(dataset):,} rows · {len(dataset.columns)} columns"):
    st.dataframe(dataset.head(100), use_container_width=True)

with st.sidebar:
    target_column = st.selectbox("Target column", list(dataset.columns))
    run_analysis = st.button(
        "Run explanation",
        type="primary",
        use_container_width=True,
        disabled=st.session_state["user"]["role"] == "viewer",
    )

if not run_analysis:
    if st.session_state["user"]["role"] == "viewer":
        st.info("Your viewer role is read-only; ask an administrator for analyst access to run explanations.")
    else:
        st.info("Choose a target column and run the explanation.")
    st.stop()

try:
    upload_response = requests.post(
        f"{API_BASE_URL}/api/datasets/upload",
        params={"name": uploaded_file.name},
        data=uploaded_file.getvalue(),
        headers={**_api_headers(), "Content-Type": "text/csv"},
        timeout=30,
    )
except requests.RequestException as error:
    st.error(f"The dataset could not be stored securely: {error}")
    st.stop()
if upload_response.status_code != 201:
    st.error(upload_response.json().get("detail", "The dataset could not be stored securely."))
    st.stop()
st.caption(f"Encrypted dataset saved as `{upload_response.json()['id']}` for your account.")

try:
    report = train_and_explain_model(dataset, target_column, task, test_size)
except (TypeError, ValueError) as error:
    st.error(str(error))
    st.stop()

metric_one, metric_two, metric_three, metric_four = st.columns(4)
metric_one.metric("Model", report["explanation_summary"]["model_type"])
metric_two.metric("Rows", report["rows"])
metric_three.metric("Features", report["features"])
metric_four.metric("Holdout rows", report["holdout"]["rows"])

importance_frame = _importance_frame(report, "feature_importance", "importance")
shap_frame = _importance_frame(report, "shap_values", "mean_abs_shap")

importance_tab, evaluation_tab, local_tab, shap_tab, details_tab = st.tabs(
    ["Feature importance", "Model evaluation", "Row explanation", "SHAP impact", "Details"]
)
with importance_tab:
    _render_feature_chart(importance_frame, "importance", "Global model feature importance", "Teal")
    st.dataframe(importance_frame, use_container_width=True, hide_index=True)

with evaluation_tab:
    holdout = report["holdout"]
    metrics = holdout["metrics"]
    if task == "classification":
        metric_columns = st.columns(4)
        metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
        metric_columns[1].metric("Precision", f"{metrics['precision_weighted']:.1%}")
        metric_columns[2].metric("Recall", f"{metrics['recall_weighted']:.1%}")
        metric_columns[3].metric("F1", f"{metrics['f1_weighted']:.1%}")
        matrix = pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics["class_labels"],
            columns=metrics["class_labels"],
        )
        st.caption("Confusion matrix · rows are actual classes; columns are predicted classes")
        st.dataframe(matrix, use_container_width=True)
    else:
        metric_columns = st.columns(3)
        metric_columns[0].metric("MAE", f"{metrics['mae']:.4f}")
        metric_columns[1].metric("RMSE", f"{metrics['rmse']:.4f}")
        metric_columns[2].metric("R²", "Unavailable" if metrics["r2"] is None else f"{metrics['r2']:.4f}")
    st.subheader("Holdout predictions")
    st.dataframe(pd.DataFrame(holdout["predictions"]), use_container_width=True, hide_index=True)

with local_tab:
    local_explanations = report.get("local_explanations", [])
    if not report["shap_available"] or not local_explanations:
        st.info("Per-row SHAP explanations are unavailable for this model run.")
    else:
        selected_row = st.selectbox(
            "Holdout row",
            local_explanations,
            format_func=lambda item: f"Row {item['row_index']} · actual {item['actual']} · predicted {item['predicted']}",
        )
        st.write(f"Actual: **{selected_row['actual']}** · Predicted: **{selected_row['predicted']}**")
        local_frame = pd.DataFrame(selected_row["features"])
        if not local_frame.empty:
            local_frame = local_frame.rename(columns={"shap_value": "impact"})
            local_figure = px.bar(
                local_frame.sort_values("impact"),
                x="impact",
                y="feature",
                orientation="h",
                color="impact",
                color_continuous_scale="RdBu",
                title="Feature contributions for this prediction",
            )
            local_figure.update_layout(height=max(360, len(local_frame) * 30), margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(local_figure, use_container_width=True)
            st.dataframe(local_frame, use_container_width=True, hide_index=True)

with shap_tab:
    if not report["shap_available"] or shap_frame.empty:
        st.warning("SHAP values are unavailable. Install the SHAP dependency to enable this view.")
    else:
        _render_feature_chart(shap_frame, "mean_abs_shap", "Mean absolute SHAP impact", "Sunset")
        st.dataframe(shap_frame, use_container_width=True, hide_index=True)

with details_tab:
    st.json(report["explanation_summary"])
    st.caption(f"Holdout share: {report['test_size']:.0%}")
    st.download_button(
        "Download explanation JSON",
        data=pd.Series(report).to_json(default_handler=str),
        file_name="explanation_report.json",
        mime="application/json",
    )
