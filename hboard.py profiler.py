[1mdiff --git a/backend.py b/backend.py[m
[1mindex 1ebfb54..2b0d2cd 100644[m
[1m--- a/backend.py[m
[1m+++ b/backend.py[m
[36m@@ -42,10 +42,11 @@[m [mdef test_connection() -> bool:[m
 [m
 [m
 # SAFE QUERY EXECUTION[m
[31m-def get_data(query: str, params: dict | None = None) -> pd.DataFrame:[m
[32m+[m[32mdef get_data(query: str | text, params: dict | None = None) -> pd.DataFrame:[m[41m[m
     try:[m
         with engine.connect() as conn:[m
[31m-            df = pd.read_sql(text(query), conn, params=params)[m
[32m+[m[32m            stmt = query if hasattr(query, "compile") else text(str(query))[m[41m[m
[32m+[m[32m            df = pd.read_sql(stmt, conn, params=params)[m[41m[m
         return df[m
     except Exception as e:[m
         print(f"❌ Database Error: {e}")[m
[36m@@ -87,5 +88,5 @@[m [mdef get_table_data(table_name: str, limit: int = 100) -> pd.DataFrame:[m
     if safe_name not in tables["table_name"].tolist():[m
         raise ValueError(f"Invalid table name: {safe_name}")[m
 [m
[31m-    query = text(f"SELECT * FROM {safe_name} LIMIT :limit")[m
[32m+[m[32m    query = f"SELECT * FROM {safe_name} LIMIT :limit"[m[41m[m
     return get_data(query, {"limit": limit})[m
\ No newline at end of file[m
[1mdiff --git a/main.py b/main.py[m
[1mindex 069c35a..c805b26 100644[m
[1m--- a/main.py[m
[1m+++ b/main.py[m
[36m@@ -29,6 +29,7 @@[m [mfrom profiler import ([m
     answer_monitoring_question,[m
     calculate_data_quality_score,[m
     check_data_quality,[m
[32m+[m[32m    compare_models,[m
     detect_fraud_patterns,[m
     forecast_data_health,[m
     forecast_metric,[m
[36m@@ -49,34 +50,39 @@[m [mfrom notifications import ([m
 )[m
 from data_sources import DataSourceError, load_source, load_uploaded_file, summarize_source[m
 from monitoring_store import ([m
[32m+[m[32m    clear_login_failures,[m
[32m+[m[32m    create_dataset_version,[m
[32m+[m[32m    create_monitoring_report,[m
[32m+[m[32m    create_monitoring_schedule,[m
     delete_configuration,[m
     delete_dataset,[m
[31m-    create_dataset_version,[m
[31m-    get_dataset_lineage,[m
[32m+[m[32m    delete_monitoring_schedule,[m
     get_audit_records,[m
     get_configurations,[m
     get_dataset,[m
[32m+[m[32m    get_dataset_lineage,[m
[32m+[m[32m    get_monitoring_report_file,[m
     get_monitoring_result,[m
[32m+[m[32m    get_monitoring_schedule,[m
[32m+[m[32m    get_user_preferences,[m
     list_datasets,[m
     list_dataset_versions,[m
[32m+[m[32m    list_monitoring_reports,[m
     list_monitoring_results,[m
[32m+[m[32m    list_monitoring_schedules,[m
[32m+[m[32m    list_users,[m
[32m+[m[32m    login_retry_after,[m
[32m+[m[32m    open_database,[m
     persist_monitoring_run,[m
     record_lineage_event,[m
[32m+[m[32m    record_login_failure,[m
     save_dataset,[m
     set_configuration,[m
     set_configurations,[m
[31m-    create_monitoring_schedule,[m
[31m-    create_monitoring_report,[m
[31m-    delete_monitoring_schedule,[m
[31m-    get_monitoring_report_file,[m
[31m-    get_monitoring_schedule,[m
[31m-    get_user_preferences,[m
[31m-    list_monitoring_schedules,[m
[31m-    list_monitoring_reports,[m
[31m-    open_database,[m
     set_monitoring_schedule_enabled,[m
[31m-    update_monitoring_schedule_interval,[m
     set_user_preferences,[m
[32m+[m[32m    update_monitoring_schedule_interval,[m
[32m+[m[32m    update_user,[m
 )[m
 from tools.data_tools import load_dataset[m
 from auth_security import ([m
[36m@@ -91,13 +97,6 @@[m [mfrom auth_security import ([m
     login_attempt_key,[m
     RequestBodyLimitMiddleware,[m
 )[m
[31m-from monitoring_store import ([m
[31m-    clear_login_failures,[m
[31m-    list_users,[m
[31m-    login_retry_after,[m
[31m-    record_login_failure,[m
[31m-    update_user,[m
[31m-)[m
 from scheduled_monitoring import scheduler_loop[m
 from reporting import REPORT_FORMATS, generate_monitoring_reports[m
 [m
[36m@@ -1431,7 +1430,11 @@[m [mdef analyze_drift_endpoint(payload: dict[str, Any]):[m
     if baseline_df.empty or current_df.empty:[m
         _bad_request("Baseline and current datasets must not be empty.")[m
 [m
[31m-    report = analyze_dataset_drift(baseline_df, current_df)[m
[32m+[m[32m    try:[m
[32m+[m[32m        report = analyze_dataset_drift(baseline_df, current_df)[m
[32m+[m[32m    except (TypeError, ValueError) as error:[m
[32m+[m[32m        _bad_request(str(error))[m
[32m+[m
     notification_result = None[m
     if _should_send_notifications(payload):[m
         drift_score = report.get("overall_drift_score", 0)[m
[36m@@ -1582,7 +1585,7 @@[m [mdef forecast_endpoint(payload: dict[str, Any]):[m
             report = forecast_data_health(df, date_column, periods, frequency)[m
         else:[m
             report = forecast_metric(df, date_column, payload.get("value_column"), periods, frequency, method)[m
[31m-    except (TypeError, ValueError) as error:[m
[32m+[m[32m    except (TypeError, ValueError, RuntimeError) as error:[m
         _bad_request(str(error))[m
     response = {"report": report}[m
     response.update(_persist_endpoint_result(payload, df, "forecast", response))[m
[36m@@ -1595,14 +1598,25 @@[m [mdef explain_model_endpoint(payload: dict[str, Any]):[m
     target_column = payload.get("target_column")[m
     if dataset is None or not target_column:[m
         _bad_request("Dataset and target_column are required.")[m
[32m+[m[32m    mode = str(payload.get("mode", "explain")).strip().lower()[m
     try:[m
         frame = pd.DataFrame(dataset)[m
[31m-        report = train_and_explain_model([m
[31m-            frame,[m
[31m-            target_column,[m
[31m-            payload.get("task", "classification"),[m
[31m-            payload.get("test_size", 0.2),[m
[31m-        )[m
[32m+[m[32m        if mode == "compare":[m
[32m+[m[32m            report = compare_models([m
[32m+[m[32m                frame,[m
[32m+[m[32m                target_column,[m
[32m+[m[32m                task=payload.get("task", "classification"),[m
[32m+[m[32m                metric=payload.get("metric"),[m
[32m+[m[32m                test_size=payload.get("test_size", 0.2),[m
[32m+[m[32m                model_names=payload.get("model_names"),[m
[32m+[m[32m            )[m
[32m+[m[32m        else:[m
[32m+[m[32m            report = train_and_explain_model([m
[32m+[m[32m                frame,[m
[32m+[m[32m                target_column,[m
[32m+[m[32m                payload.get("task", "classification"),[m
[32m+[m[32m                payload.get("test_size", 0.2),[m
[32m+[m[32m            )[m
     except (TypeError, ValueError) as error:[m
         _bad_request(str(error))[m
     response = {"report": report}[m
[1mdiff --git a/monitoring_dashboard.py b/monitoring_dashboard.py[m
[1mindex a324f5a..110ca05 100644[m
[1m--- a/monitoring_dashboard.py[m
[1m+++ b/monitoring_dashboard.py[m
[36m@@ -253,6 +253,11 @@[m [mdef _result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:[m
         result = item.get("result") or {}[m
         checks = result.get("results") or {}[m
         quality = checks.get("quality") or {}[m
[32m+[m[32m        if not quality and "quality_score" in result:[m
[32m+[m[32m            quality = {[m
[32m+[m[32m                "quality_score": result.get("quality_score", {}),[m
[32m+[m[32m                "report": result.get("report", {}),[m
[32m+[m[32m            }[m
         comparison_quality = (result.get("quality") or {}).get("current") or {}[m
         score = (quality.get("quality_score") or {}).get("overall_score")[m
         if score is None:[m
[36m@@ -260,7 +265,15 @@[m [mdef _result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:[m
         drift = checks.get("drift") or {}[m
         if not drift and isinstance(result.get("distribution_drift"), dict):[m
             drift = result["distribution_drift"][m
[32m+[m[32m        if not drift and isinstance(result.get("report"), dict) and "overall_drift_score" in result["report"]:[m
[32m+[m[32m            drift = result["report"][m
[32m+[m[32m        elif not drift and "overall_drift_score" in result:[m
[32m+[m[32m            drift = result[m
         anomaly = checks.get("anomalies") or {}[m
[32m+[m[32m        if not anomaly and isinstance(result.get("report"), dict) and "total_anomalies" in result["report"]:[m
[32m+[m[32m            anomaly = result["report"][m
[32m+[m[32m        elif not anomaly and "total_anomalies" in result:[m
[32m+[m[32m            anomaly = result[m
         quality_report = quality.get("report") or {}[m
         row_count = quality_report.get("rows")[m
         total_cells = (row_count or 0) * (quality_report.get("columns") or 0)[m
[36m@@ -1511,8 +1524,34 @@[m [mdef _latest_result([m
 [m
 [m
 def _check_results(item: dict[str, Any] | None) -> dict[str, Any]:[m
[31m-    result = item.get("result", {}) if item else {}[m
[31m-    return result.get("results", {}) if isinstance(result, dict) else {}[m
[32m+[m[32m    if not item:[m
[32m+[m[32m        return {}[m
[32m+[m[32m    result = item.get("result", {})[m
[32m+[m[32m    if not isinstance(result, dict):[m
[32m+[m[32m        return {}[m
[32m+[m[32m    if "results" in result and isinstance(result["results"], dict):[m
[32m+[m[32m        return result["results"][m
[32m+[m
[32m+[m[32m    checks: dict[str, Any] = {}[m
[32m+[m[32m    check_type = str(item.get("check_type", ""))[m
[32m+[m
[32m+[m[32m    if "quality_score" in result or check_type in {"data_quality", "data_quality_csv"}:[m
[32m+[m[32m        checks["quality"] = {[m
[32m+[m[32m            "quality_score": result.get("quality_score", {}),[m
[32m+[m[32m            "report": result.get("report", {}),[m
[32m+[m[32m            "ai_summary": result.get("ai_summary", {}),[m
[32m+[m[32m        }[m
[32m+[m[32m    if "overall_drift_score" in result or check_type in {"data_drift", "data_drift_csv"} or "distribution_drift" in result:[m
[32m+[m[32m        drift_data = result.get("distribution_drift") or result.get("report") or result[m
[32m+[m[32m        checks["drift"] = drift_data[m
[32m+[m[32m    if "total_anomalies" in result or (isinstance(result.get("report"), dict) and "total_anomalies" in result["report"]):[m
[32m+[m[32m        checks["anomalies"] = result.get("report") if "total_anomalies" in result.get("report", {}) else result[m
[32m+[m[32m    if check_type == "forecast" or "forecast" in result or (isinstance(result.get("report"), dict) and "forecast" in result["report"]):[m
[32m+[m[32m        checks["forecast"] = result.get("report") or result[m
[32m+[m[32m    if check_type == "explanation" or "feature_importance" in result or (isinstance(result.get("report"), dict) and "feature_importance" in result["report"]):[m
[32m+[m[32m        checks["explanation"] = result.get("report") or result[m
[32m+[m
[32m+[m[32m    return checks[m
 [m
 [m
 def _date_columns(frame: pd.DataFrame) -> list[str]:[m
[36m@@ -1727,6 +1766,37 @@[m [mdef _render_quality_page([m
             st.dataframe(missing, use_container_width=True, hide_index=True)[m
         else:[m
             st.success("No missing values were reported in this run.")[m
[32m+[m
[32m+[m[32m    ai_summary = quality.get("ai_summary", {})[m
[32m+[m[32m    if isinstance(ai_summary, dict) and ai_summary.get("summary"):[m
[32m+[m[32m        st.markdown("#### 🧠 AI Quality Insights & Recommendations")[m
[32m+[m[32m        risk = str(ai_summary.get("risk_level", "unknown")).upper()[m
[32m+[m[32m        risk_color = "#168f86" if risk == "LOW" else ("#e09f3e" if risk == "MEDIUM" else "#d95d5d")[m
[32m+[m[32m        st.markdown([m
[32m+[m[32m            f"""<div style="background: rgba(22, 143, 134, 0.08); border-left: 4px solid {risk_color}; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">[m
[32m+[m[32m            <strong>Risk Profile:</strong> <span style="color: {risk_color}; font-weight: bold;">{risk}</span><br>[m
[32m+[m[32m            <p style="margin-top: 6px; margin-bottom: 0;">{ai_summary.get('summary')}</p>[m
[32m+[m[32m            </div>""",[m
[32m+[m[32m            unsafe_allow_html=True,[m
[32m+[m[32m        )[m
[32m+[m[32m        col_findings, col_actions = st.columns(2)[m
[32m+[m[32m        with col_findings:[m
[32m+[m[32m            st.markdown("**Key Findings:**")[m
[32m+[m[32m            findings = ai_summary.get("key_findings", [])[m
[32m+[m[32m            if findings:[m
[32m+[m[32m                for f in findings:[m
[32m+[m[32m                    st.markdown(f"- {f}")[m
[32m+[m[32m            else:[m
[32m+[m[32m                st.caption("No issues detected.")[m
[32m+[m[32m        with col_actions:[m
[32m+[m[32m            st.markdown("**Recommended Actions:**")[m
[32m+[m[32m            actions = ai_summary.get("recommended_actions", [])[m
[32m+[m[32m            if actions:[m
[32m+[m[32m                for a in actions:[m
[32m+[m[32m                    st.markdown(f"- {a}")[m
[32m+[m[32m            else:[m
[32m+[m[32m                st.caption("No action required.")[m
[32m+[m
     with st.expander("Schema, freshness, duplicate detection, and business rules"):[m
         st.json({[m
             "schema_validation": report.get("schema_validation"),[m
[36m@@ -1921,25 +1991,28 @@[m [mdef _render_explainability([m
     with st.form("explainability-form"):[m
         target = st.selectbox("Target column", list(frame.columns))[m
         task = st.selectbox("Prediction task", ["classification", "regression"])[m
[32m+[m[32m        mode = st.radio("Analysis type", ["AutoML Model Comparison", "Model Training & Feature Explanation"], horizontal=True)[m
         test_size = st.slider("Test data fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05)[m
         submitted = st.form_submit_button([m
[31m-            "Compare models",[m
[32m+[m[32m            "Run analysis",[m
             type="primary",[m
             disabled=role == "viewer",[m
         )[m
     if not submitted:[m
         return[m
[32m+[m[32m    request_payload = {[m
[32m+[m[32m        "data": frame.to_dict(orient="records"),[m
[32m+[m[32m        "dataset_id": dataset_id,[m
[32m+[m[32m        "target_column": target,[m
[32m+[m[32m        "task": task,[m
[32m+[m[32m        "test_size": test_size,[m
[32m+[m[32m        "mode": "compare" if mode == "AutoML Model Comparison" else "explain",[m
[32m+[m[32m    }[m
     report = _post([m
         api_base_url,[m
         headers,[m
         "/api/analytics/explain",[m
[31m-        {[m
[31m-            "data": frame.to_dict(orient="records"),[m
[31m-            "dataset_id": dataset_id,[m
[31m-            "target_column": target,[m
[31m-            "task": task,[m
[31m-            "test_size": test_size,[m
[31m-        },[m
[32m+[m[32m        request_payload,[m
         timeout=300,[m
     )[m
     if report is None:[m
[36m@@ -1948,36 +2021,84 @@[m [mdef _render_explainability([m
     if not isinstance(result, dict):[m
         st.error("The explainability service returned an invalid report.")[m
         return[m
[31m-    summary = st.columns(3)[m
[31m-    summary[0].metric("Selected model", result.get("selected_model", "Unavailable"))[m
[31m-    summary[1].metric("Selection metric", result.get("metric", "Unavailable"))[m
[31m-    summary[2].metric("Best score", result.get("best_score", "Unavailable"))[m
[31m-    models = result.get("models", [])[m
[31m-    if models:[m
[31m-        model_frame = pd.DataFrame([[m
[31m-            {"model": item.get("model"), "score": item.get("score"), **item.get("metrics", {})}[m
[31m-            for item in models[m
[31m-        ])[m
[31m-        st.dataframe(model_frame, use_container_width=True, hide_index=True)[m
[31m-        importance = result.get("best_model_summary", {}).get("feature_importance", [])[m
[32m+[m
[32m+[m[32m    if "models" in result:[m
[32m+[m[32m        summary = st.columns(3)[m
[32m+[m[32m        summary[0].metric("Selected model", result.get("selected_model", "Unavailable"))[m
[32m+[m[32m        summary[1].metric("Selection metric", result.get("metric", "Unavailable"))[m
[32m+[m[32m        summary[2].metric("Best score", str(result.get("best_score", "Unavailable")))[m
[32m+[m[32m        models = result.get("models", [])[m
[32m+[m[32m        if models:[m
[32m+[m[32m            model_frame = pd.DataFrame([[m
[32m+[m[32m                {"model": item.get("model"), "score": item.get("score"), **item.get("metrics", {})}[m
[32m+[m[32m                for item in models[m
[32m+[m[32m            ])[m
[32m+[m[32m            st.markdown("#### Candidate models comparison")[m
[32m+[m[32m            st.dataframe(model_frame, use_container_width=True, hide_index=True)[m
[32m+[m[32m            importance = result.get("best_model_summary", {}).get("feature_importance", [])[m
[32m+[m[32m            if importance:[m
[32m+[m[32m                importance_frame = pd.DataFrame(importance)[m
[32m+[m[32m                if "value" in importance_frame.columns and "importance" not in importance_frame.columns:[m
[32m+[m[32m                    importance_frame = importance_frame.rename(columns={"value": "importance"})[m
[32m+[m[32m                if {"feature", "importance"}.issubset(importance_frame.columns):[m
[32m+[m[32m                    st.plotly_chart([m
[32m+[m[32m                        px.bar([m
[32m+[m[32m                            importance_frame.sort_values("importance"),[m
[32m+[m[32m                            x="importance",[m
[32m+[m[32m                            y="feature",[m
[32m+[m[32m                            orientation="h",[m
[32m+[m[32m                            title="Feature importance for selected model",[m
[32m+[m[32m                        ),[m
[32m+[m[32m                        use_container_width=True,[m
[32m+[m[32m                    )[m
[32m+[m[32m        if result.get("model_failures"):[m
[32m+[m[32m            with st.expander("Models that could not be evaluated"):[m
[32m+[m[32m                st.dataframe(pd.DataFrame(result["model_failures"]), use_container_width=True, hide_index=True)[m
[32m+[m[32m    else:[m
[32m+[m[32m        holdout = result.get("holdout", {})[m
[32m+[m[32m        metrics = holdout.get("metrics", {})[m
[32m+[m[32m        model_name = result.get("explanation_summary", {}).get("model_type", "Random Forest")[m
[32m+[m[32m        summary = st.columns(4)[m
[32m+[m[32m        summary[0].metric("Model", model_name)[m
[32m+[m[32m        if task == "classification":[m
[32m+[m[32m            summary[1].metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}" if "accuracy" in metrics else "N/A")[m
[32m+[m[32m            summary[2].metric("Precision", f"{metrics.get('precision_weighted', 0):.1%}" if "precision_weighted" in metrics else "N/A")[m
[32m+[m[32m            summary[3].metric("F1 Score", f"{metrics.get('f1_weighted', 0):.1%}" if "f1_weighted" in metrics else "N/A")[m
[32m+[m[32m            if "confusion_matrix" in metrics and "class_labels" in metrics:[m
[32m+[m[32m                cm = pd.DataFrame([m
[32m+[m[32m                    metrics["confusion_matrix"],[m
[32m+[m[32m                    index=metrics["class_labels"],[m
[32m+[m[32m                    columns=metrics["class_labels"],[m
[32m+[m[32m                )[m
[32m+[m[32m                st.markdown("#### Confusion Matrix")[m
[32m+[m[32m                st.dataframe(cm, use_container_width=True)[m
[32m+[m[32m        else:[m
[32m+[m[32m            summary[1].metric("MAE", f"{metrics.get('mae', 0):.4f}" if "mae" in metrics else "N/A")[m
[32m+[m[32m            summary[2].metric("RMSE", f"{metrics.get('rmse', 0):.4f}" if "rmse" in metrics else "N/A")[m
[32m+[m[32m            summary[3].metric("R² Score", f"{metrics.get('r2', 0):.4f}" if metrics.get("r2") is not None else "N/A")[m
[32m+[m
[32m+[m[32m        importance = result.get("feature_importance", [])[m
         if importance:[m
             importance_frame = pd.DataFrame(importance)[m
             if "value" in importance_frame.columns and "importance" not in importance_frame.columns:[m
                 importance_frame = importance_frame.rename(columns={"value": "importance"})[m
             if {"feature", "importance"}.issubset(importance_frame.columns):[m
[32m+[m[32m                st.markdown("#### Global Feature Importance")[m
                 st.plotly_chart([m
                     px.bar([m
                         importance_frame.sort_values("importance"),[m
                         x="importance",[m
                         y="feature",[m
                         orientation="h",[m
[31m-                        title="Feature importance for selected model",[m
[32m+[m[32m                        title=f"Feature importance ({model_name})",[m
                     ),[m
                     use_container_width=True,[m
                 )[m
[31m-    if result.get("model_failures"):[m
[31m-        with st.expander("Models that could not be evaluated"):[m
[31m-            st.dataframe(pd.DataFrame(result["model_failures"]), use_container_width=True, hide_index=True)[m
[32m+[m
[32m+[m[32m        if holdout.get("predictions"):[m
[32m+[m[32m            with st.expander("Holdout Predictions Preview"):[m
[32m+[m[32m                st.dataframe(pd.DataFrame(holdout["predictions"]), use_container_width=True, hide_index=True)[m
[32m+[m
     with st.expander("Explanation details"):[m
         st.json(result)[m
 [m
[1mdiff --git a/profiler.py b/profiler.py[m
[1mindex dd0c1e3..9d13115 100644[m
[1m--- a/profiler.py[m
[1m+++ b/profiler.py[m
[36m@@ -279,17 +279,46 @@[m [mdef check_data_quality([m
 [m
         operator = rule.get("operator")[m
         expected = rule.get("value")[m
[31m-        if rule.get("expression") == ">= 0":[m
[31m-            operator, expected = ">=", 0[m
[31m-        elif rule.get("expression") == "contains @":[m
[31m-            operator, expected = "contains", "@"[m
[32m+[m[32m        expression = str(rule.get("expression", "")).strip()[m[41m[m
[32m+[m[41m[m
[32m+[m[32m        if expression:[m[41m[m
[32m+[m[32m            if expression in {">= 0", "non_negative", ">=0"}:[m[41m[m
[32m+[m[32m                operator, expected = ">=", 0[m[41m[m
[32m+[m[32m            elif expression in {"> 0", "positive", ">0"}:[m[41m[m
[32m+[m[32m                operator, expected = ">", 0[m[41m[m
[32m+[m[32m            elif expression in {"<= 0", "<=0"}:[m[41m[m
[32m+[m[32m                operator, expected = "<=", 0[m[41m[m
[32m+[m[32m            elif expression in {"< 0", "negative", "<0"}:[m[41m[m
[32m+[m[32m                operator, expected = "<", 0[m[41m[m
[32m+[m[32m            elif expression in {"contains @", "email"}:[m[41m[m
[32m+[m[32m                operator, expected = "contains", "@"[m[41m[m
[32m+[m[32m            elif expression in {"not_null", "is_not_null", "required"}:[m[41m[m
[32m+[m[32m                operator = "is_not_null"[m[41m[m
[32m+[m[32m            elif expression == "is_null":[m[41m[m
[32m+[m[32m                operator = "is_null"[m[41m[m
[32m+[m[32m            else:[m[41m[m
[32m+[m[32m                match = re.match(r"^([><=!]=?)\s*(-?\d+(?:\.\d+)?)$", expression)[m[41m[m
[32m+[m[32m                if match:[m[41m[m
[32m+[m[32m                    operator = match.group(1)[m[41m[m
[32m+[m[32m                    val_str = match.group(2)[m[41m[m
[32m+[m[32m                    expected = float(val_str) if "." in val_str else int(val_str)[m[41m[m
 [m
         try:[m
             if operator in {">", ">=", "<", "<=", "==", "!="}:[m
[31m-                actual = pd.to_numeric(value, errors="coerce") if isinstance(expected, (int, float)) else value[m
[31m-                return {">": actual > expected, ">=": actual >= expected, "<": actual < expected, "<=": actual <= expected, "==": actual == expected, "!=": actual != expected}[operator][m
[32m+[m[32m                if isinstance(expected, (int, float)):[m[41m[m
[32m+[m[32m                    actual = pd.to_numeric(value, errors="coerce")[m[41m[m
[32m+[m[32m                    if pd.isna(actual):[m[41m[m
[32m+[m[32m                        return False[m[41m[m
[32m+[m[32m                    return {">": actual > expected, ">=": actual >= expected, "<": actual < expected, "<=": actual <= expected, "==": actual == expected, "!=": actual != expected}[operator][m[41m[m
[32m+[m[32m                return {">": value > expected, ">=": value >= expected, "<": value < expected, "<=": value <= expected, "==": value == expected, "!=": value != expected}[operator][m[41m[m
             if operator == "between":[m
[31m-                return expected[0] <= value <= expected[1][m
[32m+[m[32m                low, high = expected[0], expected[1][m[41m[m
[32m+[m[32m                if isinstance(low, (int, float)) or isinstance(high, (int, float)):[m[41m[m
[32m+[m[32m                    actual = pd.to_numeric(value, errors="coerce")[m[41m[m
[32m+[m[32m                    if pd.isna(actual):[m[41m[m
[32m+[m[32m                        return False[m[41m[m
[32m+[m[32m                    return low <= actual <= high[m[41m[m
[32m+[m[32m                return low <= value <= high[m[41m[m
             if operator == "in":[m
                 return value in expected[m
             if operator == "not_in":[m
[36m@@ -382,13 +411,27 @@[m [mdef _call_openai_quality_summary(metrics):[m
         f"Dataset metrics: {json.dumps(metrics, ensure_ascii=False, default=str)}"[m
     )[m
 [m
[31m-    response = client.responses.create([m
[31m-        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),[m
[31m-        input=prompt,[m
[31m-        temperature=0.2,[m
[31m-    )[m
[32m+[m[32m    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")[m[41m[m
[32m+[m[32m    content = None[m[41m[m
[32m+[m[32m    try:[m[41m[m
[32m+[m[32m        if hasattr(client, "responses"):[m[41m[m
[32m+[m[32m            response = client.responses.create([m[41m[m
[32m+[m[32m                model=model_name,[m[41m[m
[32m+[m[32m                input=prompt,[m[41m[m
[32m+[m[32m                temperature=0.2,[m[41m[m
[32m+[m[32m            )[m[41m[m
[32m+[m[32m            content = getattr(response, "output_text", None)[m[41m[m
[32m+[m[32m    except Exception:[m[41m[m
[32m+[m[32m        content = None[m[41m[m
[32m+[m[41m[m
[32m+[m[32m    if not content:[m[41m[m
[32m+[m[32m        response = client.chat.completions.create([m[41m[m
[32m+[m[32m            model=model_name,[m[41m[m
[32m+[m[32m            messages=[{"role": "user", "content": prompt}],[m[41m[m
[32m+[m[32m            temperature=0.2,[m[41m[m
[32m+[m[32m        )[m[41m[m
[32m+[m[32m        content = response.choices[0].message.content[m[41m[m
 [m
[31m-    content = getattr(response, "output_text", None)[m
     if not content:[m
         raise RuntimeError("OpenAI response did not include output text")[m
 [m
[36m@@ -508,6 +551,7 @@[m [mdef answer_monitoring_question([m
             "quality": ("quality", "score", "health", "good", "bad"),[m
             "anomaly": ("anomaly", "outlier", "abnormal"),[m
             "drift": ("drift", "shift", "distribution"),[m
[32m+[m[32m            "duplicate": ("duplicate", "duplicates", "dup", "repeated", "repeat", "copy"),[m[41m[m
             "summary": ("summary", "overall", "status", "monitor"),[m
         },[m
         "es": {[m
[36m@@ -515,6 +559,7 @@[m [mdef answer_monitoring_question([m
             "quality": ("calidad", "puntuación", "puntuacion", "puntaje", "salud"),[m
             "anomaly": ("anomalía", "anomalia", "atípico", "atipico", "anómalo", "anomalo"),[m
             "drift": ("deriva", "cambio", "distribución", "distribucion"),[m
[32m+[m[32m            "duplicate": ("duplicado", "duplicados", "repetido", "repetidos"),[m[41m[m
             "summary": ("resumen", "general", "estado", "supervisión", "supervision"),[m
         },[m
         "fr": {[m
[36m@@ -522,6 +567,7 @@[m [mdef answer_monitoring_question([m
             "quality": ("qualité", "qualite", "score", "santé", "sante"),[m
             "anomaly": ("anomalie", "anomalies", "aberrant", "anormal"),[m
             "drift": ("dérive", "derive", "changement", "distribution"),[m
[32m+[m[32m            "duplicate": ("doublon", "doublons", "dupliqué", "dupliqués", "répété"),[m[41m[m
             "summary": ("résumé", "resume", "global", "état", "etat", "surveillance"),[m
         },[m
         "hi": {[m
[36m@@ -529,6 +575,7 @@[m [mdef answer_monitoring_question([m
             "quality": ("गुणवत्ता", "स्कोर", "अंक", "स्वास्थ्य"),[m
             "anomaly": ("विसंगति", "विसंगतियां", "असामान्य", "आउटलायर"),[m
             "drift": ("ड्रिफ्ट", "बदलाव", "वितरण"),[m
[32m+[m[32m            "duplicate": ("दोहराव", "डुप्लिकेट", "नकल", "समान"),[m[41m[m
             "summary": ("सारांश", "स्थिति", "निगरानी", "कुल"),[m
         },[m
     }[m
[36m@@ -544,8 +591,10 @@[m [mdef answer_monitoring_question([m
             "drift_yes": "Yes—drift was detected. The overall drift score is {score:.1f} ({severity} severity).",[m
             "drift_no": "No material drift was detected. The overall drift score is {score:.1f} ({severity} severity).",[m
             "drift_unavailable": "No drift report is available for the current monitoring context.",[m
[32m+[m[32m            "duplicate_yes": "I found {total} duplicate rows ({percent:.1f}% of total rows) in the dataset.",[m[41m[m
[32m+[m[32m            "duplicate_no": "No duplicate rows were found in the current dataset.",[m[41m[m
             "summary": "Overall quality is {quality:.1f}/100, drift is {drift:.1f}, and there are {anomalies} anomalies in the current monitoring snapshot.",[m
[31m-            "help": "I can answer questions about quality scores, missing values, anomalies, drift, and monitoring status.",[m
[32m+[m[32m            "help": "I can answer questions about quality scores, missing values, duplicates, anomalies, drift, and monitoring status.",[m[41m[m
         },[m
         "es": {[m
             "missing_values": "Hay {total} valores faltantes en {count} columnas.",[m
[36m@@ -558,8 +607,10 @@[m [mdef answer_monitoring_question([m
             "drift_yes": "Sí, se detectó deriva. La puntuación general es {score:.1f} (severidad {severity}).",[m
             "drift_no": "No se detectó deriva significativa. La puntuación es {score:.1f} (severidad {severity}).",[m
             "drift_unavailable": "No hay un informe de deriva disponible en el contexto actual.",[m
[32m+[m[32m            "duplicate_yes": "Se encontraron {total} filas duplicadas ({percent:.1f}% del total) en el conjunto de datos.",[m[41m[m
[32m+[m[32m            "duplicate_no": "No se encontraron filas duplicadas en el conjunto de datos actual.",[m[41m[m
             "summary": "La calidad general es {quality:.1f}/100, la deriva es {drift:.1f} y hay {anomalies} anomalías en la supervisión actual.",[m
[31m-            "help": "Puedo responder sobre calidad, valores faltantes, anomalías, deriva y estado de supervisión.",[m
[32m+[m[32m            "help": "Puedo responder sobre calidad, valores faltantes, duplicados, anomalías, deriva y estado de supervisión.",[m[41m[m
         },[m
         "fr": {[m
             "missing_values": "Il y a {total} valeurs manquantes dans {count} colonnes.",[m
[36m@@ -572,8 +623,10 @@[m [mdef answer_monitoring_question([m
             "drift_yes": "Oui, une dérive a été détectée. Le score global est de {score:.1f} (gravité {severity}).",[m
             "drift_no": "Aucune dérive significative n’a été détectée. Le score est de {score:.1f} (gravité {severity}).",[m
             "drift_unavailable": "Aucun rapport de dérive n’est disponible dans le contexte actuel.",[m
[32m+[m[32m            "duplicate_yes": "{total} lignes en double ont été détectées ({percent:.1f}% des lignes) dans le jeu de données.",[m[41m[m
[32m+[m[32m            "duplicate_no": "Aucun doublon n'a été détecté dans le jeu de données actuel.",[m[41m[m
             "summary": "La qualité globale est de {quality:.1f}/100, la dérive est de {drift:.1f} et il y a {anomalies} anomalies dans la surveillance actuelle.",[m
[31m-            "help": "Je peux répondre sur la qualité, les valeurs manquantes, les anomalies, la dérive et l’état de surveillance.",[m
[32m+[m[32m            "help": "Je peux répondre sur la qualité, les valeurs manquantes, les doublons, les anomalies, la dérive et l’état de surveillance.",[m[41m[m
         },[m
         "hi": {[m
             "missing_values": "{count} कॉलम में {total} मान अनुपलब्ध हैं।",[m
[36m@@ -586,8 +639,10 @@[m [mdef answer_monitoring_question([m
             "drift_yes": "हां, डेटा ड्रिफ्ट मिला। कुल स्कोर {score:.1f} है (गंभीरता {severity})।",[m
             "drift_no": "कोई महत्वपूर्ण डेटा ड्रिफ्ट नहीं मिला। स्कोर {score:.1f} है (गंभीरता {severity})।",[m
             "drift_unavailable": "वर्तमान निगरानी संदर्भ में ड्रिफ्ट रिपोर्ट उपलब्ध नहीं है।",[m
[32m+[m[32m            "duplicate_yes": "डेटासेट में {total} डुप्लिकेट पंक्तियां ({percent:.1f}%) पाई गईं।",[m[41m[m
[32m+[m[32m            "duplicate_no": "वर्तमान डेटासेट में कोई डुप्लिकेट पंक्ति नहीं मिली।",[m[41m[m
             "summary": "कुल गुणवत्ता {quality:.1f}/100 है, ड्रिफ्ट {drift:.1f} है और वर्तमान निगरानी में {anomalies} विसंगतियां हैं।",[m
[31m-            "help": "मैं गुणवत्ता स्कोर, अनुपलब्ध मान, विसंगतियों, ड्रिफ्ट और निगरानी स्थिति के बारे में उत्तर दे सकता हूं।",[m
[32m+[m[32m            "help": "मैं गुणवत्ता स्कोर, अनुपलब्ध मान, डुप्लिकेट, विसंगतियों, ड्रिफ्ट और निगरानी स्थिति के बारे में उत्तर दे सकता हूं।",[m[41m[m
         },[m
     }[language][m
     intent_phrases = phrases[language][m
[36m@@ -642,6 +697,31 @@[m [mdef answer_monitoring_question([m
             },[m
         }[m
 [m
[32m+[m[32m    if has_intent("duplicate"):[m[41m[m
[32m+[m[32m        exact_duplicates = 0[m[41m[m
[32m+[m[32m        total_rows = 0[m[41m[m
[32m+[m[32m        if isinstance(quality_report, dict):[m[41m[m
[32m+[m[32m            exact_duplicates = int(quality_report.get("duplicates", 0))[m[41m[m
[32m+[m[32m            total_rows = int(quality_report.get("rows", 0))[m[41m[m
[32m+[m[32m        elif df is not None:[m[41m[m
[32m+[m[32m            exact_duplicates = int(df.duplicated().sum())[m[41m[m
[32m+[m[32m            total_rows = len(df)[m[41m[m
[32m+[m[32m        pct = (exact_duplicates / max(total_rows, 1)) * 100[m[41m[m
[32m+[m[32m        answer = ([m[41m[m
[32m+[m[32m            localized["duplicate_yes"].format(total=exact_duplicates, percent=pct)[m[41m[m
[32m+[m[32m            if exact_duplicates > 0[m[41m[m
[32m+[m[32m            else localized["duplicate_no"][m[41m[m
[32m+[m[32m        )[m[41m[m
[32m+[m[32m        return {[m[41m[m
[32m+[m[32m            "category": "quality",[m[41m[m
[32m+[m[32m            "answer": answer,[m[41m[m
[32m+[m[32m            "confidence": "high",[m[41m[m
[32m+[m[32m            "context": {[m[41m[m
[32m+[m[32m                "exact_duplicates": exact_duplicates,[m[41m[m
[32m+[m[32m                "duplicate_percentage": round(pct, 2),[m[41m[m
[32m+[m[32m            },[m[41m[m
[32m+[m[32m        }[m[41m[m
[32m+[m[41m[m
     if has_intent("quality"):[m
         score = quality_score.get("overall_score", 0) if quality_score else 0[m
         if score is not None:[m
[36m@@ -793,11 +873,21 @@[m [mdef detect_outliers_iqr(df):[m
     num = df.select_dtypes(include=np.number)[m
 [m
     for col in num.columns:[m
[31m-        q1, q3 = num[col].quantile([0.25, 0.75])[m
[32m+[m[32m        series = num[col].dropna()[m[41m[m
[32m+[m[32m        if series.empty:[m[41m[m
[32m+[m[32m            result[col] = {"count": 0, "lower_bound": None, "upper_bound": None, "percentage": 0.0}[m[41m[m
[32m+[m[32m            continue[m[41m[m
[32m+[m[32m        q1, q3 = series.quantile([0.25, 0.75])[m[41m[m
         iqr = q3 - q1[m
[31m-        out = num[(num[col] < q1 - 1.5 * iqr) |[m
[31m-                  (num[col] > q3 + 1.5 * iqr)][m
[31m-        result[col] = {"count": len(out)}[m
[32m+[m[32m        lower = q1 - 1.5 * iqr[m[41m[m
[32m+[m[32m        upper = q3 + 1.5 * iqr[m[41m[m
[32m+[m[32m        out = series[(series < lower) | (series > upper)][m[41m[m
[32m+[m[32m        result[col] = {[m[41m[m
[32m+[m[32m            "count": len(out),[m[41m[m
[32m+[m[32m            "lower_bound": round(float(lower), 4),[m[41m[m
[32m+[m[32m            "upper_bound": round(float(upper), 4),[m[41m[m
[32m+[m[32m            "percentage": round(float(len(out) / max(len(series), 1) * 100), 2),[m[41m[m
[32m+[m[32m        }[m[41m[m
     return result[m
 [m
 def plot_outliers(df, col):[m
[36m@@ -805,18 +895,25 @@[m [mdef plot_outliers(df, col):[m
     return apply_modern_theme(fig)[m
 [m
 # ANOMALIES[m
[31m-def detect_anomalies_isolation_forest(df):[m
[32m+[m[32mdef detect_anomalies_isolation_forest(df, contamination=0.05):[m[41m[m
     num = df.select_dtypes(include=np.number).dropna()[m
     if num.shape[1] == 0 or len(num) < 10:[m
         return None[m
 [m
[31m-    model = IsolationForest(contamination=0.05, random_state=42)[m
[32m+[m[32m    try:[m[41m[m
[32m+[m[32m        contam = float(contamination)[m[41m[m
[32m+[m[32m        if not 0.001 <= contam <= 0.5:[m[41m[m
[32m+[m[32m            contam = 0.05[m[41m[m
[32m+[m[32m    except (TypeError, ValueError):[m[41m[m
[32m+[m[32m        contam = 0.05[m[41m[m
[32m+[m[41m[m
[32m+[m[32m    model = IsolationForest(contamination=contam, random_state=42)[m[41m[m
     preds = model.fit_predict(num)[m
     idx = num.index[preds == -1][m
 [m
     return {[m
         "total_anomalies": len(idx),[m
[31m-        "anomaly_percentage": len(idx) / len(df) * 100,[m
[32m+[m[32m        "anomaly_percentage": len(idx) / max(len(df), 1) * 100,[m[41m[m
         "indices": idx.tolist()[m
     }[m
 [m
[36m@@ -863,14 +960,42 @@[m [mdef plot_memory_usage(df):[m
 # RECOMMENDATIONS[m
 def generate_recommendations(df, report):[m
     rec = [][m
[31m-    if report["total_nulls"] > 0:[m
[31m-        rec.append({"type":"Missing Values",[m
[31m-                    "issue":"Null values detected",[m
[31m-                    "solution":"Use fillna/dropna"})[m
[31m-    if report["duplicates"] > 0:[m
[31m-        rec.append({"type":"Duplicates",[m
[31m-                    "issue":"Duplicate rows found",[m
[31m-                    "solution":"Use drop_duplicates"})[m
[32m+[m[32m    if report.get("total_nulls", 0) > 0:[m[41m[m
[32m+[m[32m        rec.append({"type": "Missing Values",[m[41m[m
[32m+[m[32m                    "issue": "Null values detected",[m[41m[m
[32m+[m[32m                    "solution": "Use fillna/dropna"})[m[41m[m
[32m+[m[32m    if report.get("duplicates", 0) > 0:[m[41m[m
[32m+[m[32m        rec.append({"type": "Duplicates",[m[41m[m
[32m+[m[32m                    "issue": "Duplicate rows found",[m[41m[m
[32m+[m[32m                    "solution": "Use drop_duplicates"})[m[41m[m
[32m+[m[32m    if report.get("all_null_columns"):[m[41m[m
[32m+[m[32m        rec.append({"type": "Empty Columns",[m[41m[m
[32m+[m[32m                    "issue": f"{len(report['all_null_columns'])} column(s) are completely empty",[m[41m[m
[32m+[m[32m                    "solution": "Drop empty columns using df.drop(columns=...)"})[m[41m[m
[32m+[m[32m    dup_details = report.get("duplicate_detection", {})[m[41m[m
[32m+[m[32m    if dup_details.get("near_duplicate_pairs", 0) > 0:[m[41m[m
[32m+[m[32m        rec.append({"type": "Near Duplicates",[m[41m[m
[32m+[m[32m                    "issue": f"{dup_details['near_duplicate_pairs']} near-duplicate pair(s) detected",[m[41m[m
[32m+[m[32m                    "solution": "Review highly similar rows for potential data entry repetition"})[m[41m[m
[32m+[m[32m    business_rules = report.get("business_rule_validation", {})[m[41m[m
[32m+[m[32m    if business_rules.get("violations", 0) > 0:[m[41m[m
[32m+[m[32m        rec.append({"type": "Business Rules",[m[41m[m
[32m+[m[32m                    "issue": f"{business_rules['violations']} business rule violation(s) identified",[m[41m[m
[32m+[m[32m                    "solution": "Filter or correct rows violating column business rules"})[m[41m[m
[32m+[m[32m    freshness = report.get("data_freshness", {})[m[41m[m
[32m+[m[32m    if freshness.get("is_fresh") is False:[m[41m[m
[32m+[m[32m        rec.append({"type": "Data Freshness",[m[41m[m
[32m+[m[32m                    "issue": f"Data exceeds maximum allowed age ({freshness.get('age_hours')} hrs vs max {freshness.get('max_age_hours')} hrs)",[m[41m[m
[32m+[m[32m                    "solution": "Trigger upstream pipeline refresh to ensure dataset currency"})[m[41m[m
[32m+[m[32m    schema = report.get("schema_validation", {})[m[41m[m
[32m+[m[32m    if schema.get("missing_columns"):[m[41m[m
[32m+[m[32m        rec.append({"type": "Schema Validation",[m[41m[m
[32m+[m[32m                    "issue": f"Missing expected column(s): {', '.join(schema['missing_columns'])}",[m[41m[m
[32m+[m[32m                    "solution": "Ensure data source pipeline supplies all expected schema columns"})[m[41m[m
[32m+[m[32m    if schema.get("type_issues"):[m[41m[m
[32m+[m[32m        rec.append({"type": "Data Type Mismatch",[m[41m[m
[32m+[m[32m                    "issue": f"{len(schema['type_issues'])} column(s) do not match expected dtypes",[m[41m[m
[32m+[m[32m                    "solution": "Cast affected columns to the expected data types"})[m[41m[m
     return rec[m
 [m
 [m
