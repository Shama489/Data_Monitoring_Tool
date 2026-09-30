"""Interactive explainability dashboard for the data monitoring tool."""

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from profiler import train_and_explain_model


st.set_page_config(page_title="Explainable AI Monitor", page_icon="XAI", layout="wide")


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
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    task = st.selectbox("Task", ["classification", "regression"])
    test_size = st.slider("Holdout share", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

if uploaded_file is None:
    st.info("Upload a CSV file to begin an explainability analysis.")
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
    run_analysis = st.button("Run explanation", type="primary", use_container_width=True)

if not run_analysis:
    st.info("Choose a target column and run the explanation.")
    st.stop()

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
