import math
import json
import os
import re
import warnings

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except ImportError:  # pragma: no cover
    ARIMA = None
    ConvergenceWarning = Warning

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

SUPPORTED_FORECAST_METHODS = {"auto", "arima", "linear", "prophet", "lstm"}

# 🎨 MODERN GRAPH THEME 
def apply_modern_theme(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )
    return fig

# DATA QUALITY
def check_data_quality(
    df,
    expected_columns=None,
    expected_dtypes=None,
    timestamp_column=None,
    max_age_hours=None,
    similarity_threshold=0.8,
    rules=None,
):
    exact_duplicates = int(df.duplicated().sum())

    duplicate_details = {"exact_duplicates": exact_duplicates, "near_duplicate_pairs": 0, "similarity_threshold": similarity_threshold}
    if df.shape[0] >= 2 and similarity_threshold is not None:
        near_duplicate_count = 0
        for i in range(len(df)):
            left = df.iloc[i].fillna("").astype(str)
            for j in range(i + 1, len(df)):
                right = df.iloc[j].fillna("").astype(str)
                scores = []
                for left_value, right_value in zip(left, right):
                    left_text = str(left_value).strip().lower()
                    right_text = str(right_value).strip().lower()
                    if left_text == right_text:
                        scores.append(1.0)
                    elif left_text and right_text:
                        from difflib import SequenceMatcher
                        scores.append(SequenceMatcher(None, left_text, right_text).ratio())
                    else:
                        scores.append(0.0)
                if scores and (sum(scores) / len(scores)) >= similarity_threshold:
                    near_duplicate_count += 1
        duplicate_details["near_duplicate_pairs"] = near_duplicate_count

    expected_columns = list(expected_columns or [])
    expected_dtypes = expected_dtypes or {}
    actual_columns = [str(column) for column in df.columns]
    missing_columns = [str(column) for column in expected_columns if str(column) not in actual_columns]
    extra_columns = [column for column in actual_columns if column not in {str(item) for item in expected_columns}]
    type_issues = []
    for column_name, expected_type in expected_dtypes.items():
        if column_name in df.columns:
            actual_type = str(df[column_name].dtype)
            expected_type = str(expected_type)
            aliases = {"string": "object", "str": "object", "integer": "int64", "float": "float64"}
            if aliases.get(expected_type.lower(), expected_type) != actual_type:
                type_issues.append({"column": column_name, "expected": expected_type, "actual": actual_type})

    renamed_columns = []
    if missing_columns and extra_columns:
        from difflib import SequenceMatcher

        for expected_column in missing_columns:
            candidates = []
            expected_type = expected_dtypes.get(expected_column)
            expected_type = aliases.get(str(expected_type).lower(), str(expected_type)) if expected_type else None
            for actual_column in extra_columns:
                actual_type = str(df[actual_column].dtype)
                if expected_type and expected_type != actual_type:
                    continue
                confidence = round(SequenceMatcher(None, expected_column.lower(), actual_column.lower()).ratio(), 2)
                if confidence >= 0.6:
                    candidates.append((confidence, actual_column))
            if candidates:
                confidence, actual_column = max(candidates)
                renamed_columns.append({"expected": expected_column, "actual": actual_column, "confidence": confidence})

    schema_report = {
        "expected_columns": expected_columns or None,
        "expected_dtypes": expected_dtypes or None,
        "actual_columns": actual_columns,
        "actual_dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "renamed_columns": renamed_columns,
        "type_issues": type_issues,
        "status": "ok",
    }
    if missing_columns or extra_columns or renamed_columns or type_issues:
        schema_report["status"] = "warning"

    freshness_report = {"timestamp_column": timestamp_column, "max_age_hours": max_age_hours, "latest_timestamp": None, "age_hours": None, "is_fresh": True}
    if timestamp_column is not None and timestamp_column in df.columns and max_age_hours is not None:
        try:
            timestamps = pd.to_datetime(df[timestamp_column], errors="coerce").dropna()
            if not timestamps.empty:
                latest_timestamp = timestamps.max()
                freshness_report["latest_timestamp"] = latest_timestamp.isoformat()
                freshness_report["age_hours"] = round(float((pd.Timestamp.now() - latest_timestamp).total_seconds() / 3600), 2)
                freshness_report["is_fresh"] = freshness_report["age_hours"] <= float(max_age_hours)
        except Exception:
            freshness_report["is_fresh"] = False

    normalized_rules = []
    if isinstance(rules, dict):
        for column_name, rule in rules.items():
            if isinstance(rule, str):
                normalized_rules.append({"id": f"{column_name}:{rule}", "column": column_name, "expression": rule})
            elif isinstance(rule, dict):
                normalized_rules.append({"id": rule.get("id", column_name), "column": column_name, **rule})
    elif isinstance(rules, list):
        normalized_rules = [rule for rule in rules if isinstance(rule, dict)]
    elif rules is not None:
        normalized_rules = []

    def evaluate_rule(value, rule):
        if pd.isna(value):
            return bool(rule.get("allow_null", False)) if rule.get("operator") != "is_null" else True

        operator = rule.get("operator")
        expected = rule.get("value")
        if rule.get("expression") == ">= 0":
            operator, expected = ">=", 0
        elif rule.get("expression") == "contains @":
            operator, expected = "contains", "@"

        try:
            if operator in {">", ">=", "<", "<=", "==", "!="}:
                actual = pd.to_numeric(value, errors="coerce") if isinstance(expected, (int, float)) else value
                return {">": actual > expected, ">=": actual >= expected, "<": actual < expected, "<=": actual <= expected, "==": actual == expected, "!=": actual != expected}[operator]
            if operator == "between":
                return expected[0] <= value <= expected[1]
            if operator == "in":
                return value in expected
            if operator == "not_in":
                return value not in expected
            text_value = str(value)
            if operator == "contains":
                return str(expected) in text_value
            if operator == "starts_with":
                return text_value.startswith(str(expected))
            if operator == "ends_with":
                return text_value.endswith(str(expected))
            if operator == "regex":
                return re.search(str(expected), text_value) is not None
            if operator == "is_not_null":
                return True
            if operator == "is_null":
                return False
        except (TypeError, ValueError, IndexError, re.error):
            return False
        return None

    violations = 0
    rule_details = []
    for position, rule in enumerate(normalized_rules):
        rule_id = str(rule.get("id", f"rule_{position + 1}"))
        column_name = rule.get("column")
        display_rule = {key: value for key, value in rule.items() if key != "id"}
        if column_name not in df.columns:
            rule_details.append({"id": rule_id, "rule": display_rule, "violations": len(df), "failed_rows": list(df.index), "status": "missing_column"})
            violations += len(df)
            continue

        failed_rows = []
        unsupported = False
        for row_index, value in df[column_name].items():
            result = evaluate_rule(value, rule)
            if result is None:
                unsupported = True
                break
            if not result:
                failed_rows.append(row_index)
        status = "unsupported_rule" if unsupported else ("ok" if not failed_rows else "violated")
        violations += len(failed_rows)
        rule_details.append({
            "id": rule_id,
            "rule": display_rule,
            "severity": rule.get("severity", "error"),
            "violations": len(failed_rows),
            "failed_rows": failed_rows,
            "pass_rate": round((len(df) - len(failed_rows)) / max(len(df), 1) * 100, 2),
            "status": status,
        })

    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "null_per_column": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicates": exact_duplicates,
        "all_null_columns": df.columns[df.isnull().all()].tolist(),
        "schema_validation": schema_report,
        "duplicate_detection": duplicate_details,
        "data_freshness": freshness_report,
        "business_rule_validation": {"rules": rule_details, "violations": violations, "status": "ok" if violations == 0 else "warning"},
    }
    return report

def calculate_data_quality_score(df):
    total_cells = df.size
    nulls = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    score = 100
    if total_cells:
        score -= (nulls / total_cells) * 60
        score -= (duplicates / len(df)) * 40

    return {"overall_score": round(float(max(score, 0)), 2)}


def _call_openai_quality_summary(metrics):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI API key is not configured")

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a data quality expert. Analyze the dataset metrics below and return only valid JSON "
        "with keys: summary, risk_level, key_findings, recommended_actions. "
        f"Dataset metrics: {json.dumps(metrics, ensure_ascii=False, default=str)}"
    )

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=prompt,
        temperature=0.2,
    )

    content = getattr(response, "output_text", None)
    if not content:
        raise RuntimeError("OpenAI response did not include output text")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = json.loads(content.strip("```json\n").strip("```"))

    if not isinstance(parsed, dict):
        raise ValueError("OpenAI returned an unexpected format")

    return parsed


def generate_ai_quality_summary(df, report=None, quality_score=None, use_llm=False):
    if report is None:
        report = check_data_quality(df)
    if quality_score is None:
        quality_score = calculate_data_quality_score(df)

    total_rows = max(len(df), 1)
    total_cells = max(df.size, 1)
    null_rate = (report.get("total_nulls", 0) / total_cells) * 100
    duplicate_rate = (report.get("duplicates", 0) / total_rows) * 100
    overall_score = float(quality_score.get("overall_score", 0.0))

    metrics = {
        "overall_score": round(overall_score, 2),
        "total_rows": int(len(df)),
        "total_columns": int(df.shape[1]),
        "total_nulls": int(report.get("total_nulls", 0)),
        "duplicates": int(report.get("duplicates", 0)),
        "null_rate_percent": round(float(null_rate), 2),
        "duplicate_rate_percent": round(float(duplicate_rate), 2),
        "all_null_columns": report.get("all_null_columns", []),
    }

    if use_llm:
        try:
            ai_result = _call_openai_quality_summary(metrics)
            required_fields = {"summary", "risk_level", "key_findings", "recommended_actions"}
            if required_fields.issubset(ai_result.keys()):
                return ai_result
        except Exception as exc:
            warnings.warn(
                f"OpenAI quality summary failed; falling back to built-in summary. Details: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    if overall_score >= 85:
        risk_level = "low"
    elif overall_score >= 60:
        risk_level = "medium"
    elif overall_score >= 30:
        risk_level = "high"
    else:
        risk_level = "critical"

    findings = []
    if report.get("total_nulls", 0) > 0:
        findings.append(f"{report['total_nulls']} missing values were found across {len(report.get('null_per_column', {}))} columns.")
    if report.get("duplicates", 0) > 0:
        findings.append(f"{report['duplicates']} duplicate rows were detected, which can distort downstream analytics.")
    if report.get("all_null_columns"):
        findings.append("Some columns are entirely empty and may need removal or imputation.")
    if not findings:
        findings.append("The dataset looks structurally clean with no obvious missing-value or duplicate anomalies.")

    recommended_actions = []
    if report.get("total_nulls", 0) > 0:
        recommended_actions.append("Impute missing values or drop rows/columns that are not business critical.")
    if report.get("duplicates", 0) > 0:
        recommended_actions.append("Remove repeated records with drop_duplicates before model training or reporting.")
    if report.get("all_null_columns"):
        recommended_actions.append("Review empty columns and remove them if they do not provide usable signal.")
    if null_rate < 5 and duplicate_rate < 2 and overall_score >= 85:
        recommended_actions.append("Keep the current quality baseline and monitor for drift as new data arrives.")
    else:
        recommended_actions.append("Run a validation pass on schema, range checks, and business rules before production use.")

    summary = (
        f"The dataset quality score is {overall_score:.1f}/100 with a {risk_level} risk profile. "
        f"Missing values account for {null_rate:.1f}% of cells and duplicates account for {duplicate_rate:.1f}% of rows. "
        "AI-assisted review recommends targeted cleanup before using the dataset in reporting or ML workflows."
    )

    return {
        "summary": summary,
        "risk_level": risk_level,
        "key_findings": findings,
        "recommended_actions": recommended_actions,
    }

# QUALITY VISUALS
def plot_null_distribution(df):
    nulls = df.isnull().sum()
    fig = px.bar(x=nulls.index, y=nulls.values,
                 color=nulls.values,
                 title="📉 Missing Values per Column")
    return apply_modern_theme(fig)

def plot_null_heatmap(df):
    nulls = df.isnull().sum().to_frame(name="Nulls")
    fig = px.imshow(nulls.T, text_auto=True,
                    title="🔥 Missing Values Heatmap")
    return apply_modern_theme(fig, 350)

def plot_duplicate_analysis(df):
    dup = df.duplicated().sum()
    fig = px.pie(values=[dup, len(df)-dup],
                 names=["Duplicates", "Unique"],
                 hole=0.5,
                 title="🧬 Duplicate Analysis")
    return apply_modern_theme(fig, 350)

# STATISTICS
def get_statistical_summary(df):
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return None
    return num.describe().to_dict()

def plot_statistical_summary(df):
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return None
    fig = px.box(num, title="📊 Statistical Summary")
    return apply_modern_theme(fig)

# CORRELATION
def plot_correlation_heatmap(df):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return None
    corr = num.corr()
    fig = px.imshow(corr, text_auto=True,
                    color_continuous_scale="tealrose",
                    title="🔥 Correlation Heatmap")
    return apply_modern_theme(fig, 500)

def analyze_column_relationships(df):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return {}

    corr = num.corr()
    high = {}
    for c1 in corr.columns:
        for c2 in corr.columns:
            if c1 != c2 and abs(corr.loc[c1, c2]) > 0.75:
                high[f"{c1} - {c2}"] = corr.loc[c1, c2]
    return high

# OUTLIERS
def detect_outliers_iqr(df):
    result = {}
    num = df.select_dtypes(include=np.number)

    for col in num.columns:
        q1, q3 = num[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        out = num[(num[col] < q1 - 1.5 * iqr) |
                  (num[col] > q3 + 1.5 * iqr)]
        result[col] = {"count": len(out)}
    return result

def plot_outliers(df, col):
    fig = px.box(df, y=col, title=f"🚨 Outliers in {col}")
    return apply_modern_theme(fig)

# ANOMALIES
def detect_anomalies_isolation_forest(df):
    num = df.select_dtypes(include=np.number).dropna()
    if num.shape[1] == 0 or len(num) < 10:
        return None

    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(num)
    idx = num.index[preds == -1]

    return {
        "total_anomalies": len(idx),
        "anomaly_percentage": len(idx) / len(df) * 100,
        "indices": idx.tolist()
    }

def plot_anomalies(df):
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) < 2:
        return None

    an = detect_anomalies_isolation_forest(df)
    dfp = df.copy()
    dfp["Anomaly"] = "Normal"
    if an:
        dfp.loc[an["indices"], "Anomaly"] = "Anomaly"

    fig = px.scatter(dfp, x=num_cols[0], y=num_cols[1],
                     color="Anomaly",
                     title="🧠 Anomaly Detection")
    return apply_modern_theme(fig)

# CARDINALITY
def analyze_cardinality(df):
    res = {}
    for c in df.columns:
        res[c] = df[c].nunique()
    return res

def plot_cardinality(df):
    uniques = df.nunique()
    fig = px.bar(x=uniques.index, y=uniques.values,
                 title="📦 Cardinality")
    return apply_modern_theme(fig)

# MEMORY
def analyze_memory_usage(df):
    mem = df.memory_usage(deep=True)
    return {"total_memory_mb": mem.sum()/1024**2}

def plot_memory_usage(df):
    mem = df.memory_usage(deep=True)/1024**2
    fig = px.bar(x=mem.index, y=mem.values,
                 title="💾 Memory Usage (MB)")
    return apply_modern_theme(fig)

# RECOMMENDATIONS
def generate_recommendations(df, report):
    rec = []
    if report["total_nulls"] > 0:
        rec.append({"type":"Missing Values",
                    "issue":"Null values detected",
                    "solution":"Use fillna/dropna"})
    if report["duplicates"] > 0:
        rec.append({"type":"Duplicates",
                    "issue":"Duplicate rows found",
                    "solution":"Use drop_duplicates"})
    return rec


def classify_drift_severity(score):
    score = float(score)
    if score < 25:
        return "low"
    if score < 50:
        return "medium"
    if score < 80:
        return "high"
    return "critical"


def _coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def _compute_ks_statistic(baseline, current):
    baseline_values = np.asarray(_coerce_numeric(baseline), dtype=float)
    current_values = np.asarray(_coerce_numeric(current), dtype=float)

    if baseline_values.size == 0 or current_values.size == 0:
        return 0.0

    baseline_sorted = np.sort(baseline_values)
    current_sorted = np.sort(current_values)
    values = np.unique(np.concatenate([baseline_sorted, current_sorted]))

    if values.size == 0:
        return 0.0

    baseline_cdf = np.searchsorted(baseline_sorted, values, side="right") / baseline_sorted.size
    current_cdf = np.searchsorted(current_sorted, values, side="right") / current_sorted.size
    return float(np.max(np.abs(baseline_cdf - current_cdf)))


def _compute_psi(baseline_values, current_values):
    baseline_series = pd.Series(baseline_values).dropna()
    current_series = pd.Series(current_values).dropna()

    if baseline_series.empty or current_series.empty:
        return 0.0

    # Numeric PSI uses quantile bins from the baseline distribution.
    if pd.api.types.is_numeric_dtype(baseline_series):
        baseline_numeric = pd.to_numeric(baseline_series, errors="coerce").dropna()
        current_numeric = pd.to_numeric(current_series, errors="coerce").dropna()
        if baseline_numeric.empty or current_numeric.empty:
            return 0.0

        quantiles = np.linspace(0, 1, 11)
        bins = np.quantile(baseline_numeric, quantiles)
        bins = np.unique(np.concatenate(([float("-inf")], bins, [float("inf")])) )
        baseline_hist, _ = np.histogram(baseline_numeric, bins=bins)
        current_hist, _ = np.histogram(current_numeric, bins=bins)
        baseline_pct = baseline_hist / max(baseline_hist.sum(), 1)
        current_pct = current_hist / max(current_hist.sum(), 1)
        epsilon = 1e-6
        return float(sum(
            (curr - base) * math.log((curr + epsilon) / (base + epsilon))
            for base, curr in zip(baseline_pct, current_pct)
        ))

    baseline_counts = baseline_series.astype(str).value_counts(normalize=True)
    current_counts = current_series.astype(str).value_counts(normalize=True)
    categories = sorted(set(baseline_counts.index) | set(current_counts.index))
    baseline_pct = [float(baseline_counts.get(cat, 0.0)) for cat in categories]
    current_pct = [float(current_counts.get(cat, 0.0)) for cat in categories]
    epsilon = 1e-6
    return float(sum(
        (curr - base) * math.log((curr + epsilon) / (base + epsilon))
        for base, curr in zip(baseline_pct, current_pct)
        if base > 0 or curr > 0
    ))


def _compare_numeric_feature(column_name, baseline_df, current_df):
    baseline_series = _coerce_numeric(baseline_df[column_name])
    current_series = _coerce_numeric(current_df[column_name])

    ks_statistic = _compute_ks_statistic(baseline_series, current_series)
    psi_value = _compute_psi(baseline_series, current_series)
    drift_score = min(100.0, max(0.0, (ks_statistic * 70.0) + (min(abs(psi_value), 1.5) / 1.5) * 30.0))

    return {
        "column": column_name,
        "data_type": "numeric",
        "method": "KS Test + PSI",
        "ks_statistic": round(float(ks_statistic), 4),
        "psi": round(float(psi_value), 4),
        "drift_score": round(float(drift_score), 2),
        "severity": classify_drift_severity(drift_score),
        "baseline_summary": {
            "mean": round(float(baseline_series.mean()), 4) if not baseline_series.empty else None,
            "std": round(float(baseline_series.std(ddof=0)), 4) if baseline_series.size > 1 else None,
            "min": round(float(baseline_series.min()), 4) if not baseline_series.empty else None,
            "max": round(float(baseline_series.max()), 4) if not baseline_series.empty else None,
        },
        "current_summary": {
            "mean": round(float(current_series.mean()), 4) if not current_series.empty else None,
            "std": round(float(current_series.std(ddof=0)), 4) if current_series.size > 1 else None,
            "min": round(float(current_series.min()), 4) if not current_series.empty else None,
            "max": round(float(current_series.max()), 4) if not current_series.empty else None,
        },
    }


def _compare_categorical_feature(column_name, baseline_df, current_df):
    baseline_series = baseline_df[column_name].fillna("missing").astype(str)
    current_series = current_df[column_name].fillna("missing").astype(str)

    psi_value = _compute_psi(baseline_series, current_series)
    baseline_pct = baseline_series.value_counts(normalize=True)
    current_pct = current_series.value_counts(normalize=True)
    category_shift = float((current_pct.reindex(baseline_pct.index, fill_value=0.0) - baseline_pct).abs().sum())
    drift_score = min(100.0, max(0.0, abs(psi_value) * 100.0 * 0.8 + category_shift * 100.0 * 0.2))

    return {
        "column": column_name,
        "data_type": "categorical",
        "method": "PSI",
        "psi": round(float(psi_value), 4),
        "drift_score": round(float(drift_score), 2),
        "severity": classify_drift_severity(drift_score),
        "baseline_distribution": baseline_pct.head(10).to_dict(),
        "current_distribution": current_pct.head(10).to_dict(),
        "category_shift": round(float(category_shift), 4),
    }


def analyze_dataset_drift(baseline_df, current_df):
    baseline_df = baseline_df.copy()
    current_df = current_df.copy()

    if baseline_df.empty or current_df.empty:
        raise ValueError("Both the baseline and current datasets must contain at least one row.")

    common_columns = [col for col in baseline_df.columns if col in current_df.columns]
    if not common_columns:
        raise ValueError("The baseline and current datasets do not share any columns for comparison.")

    feature_metrics = {}
    for column_name in common_columns:
        if pd.api.types.is_numeric_dtype(baseline_df[column_name]) or pd.api.types.is_numeric_dtype(current_df[column_name]):
            feature_metrics[column_name] = _compare_numeric_feature(column_name, baseline_df, current_df)
        else:
            feature_metrics[column_name] = _compare_categorical_feature(column_name, baseline_df, current_df)

    overall_score = round(
        float(sum(metric["drift_score"] for metric in feature_metrics.values()) / len(feature_metrics)),
        2,
    ) if feature_metrics else 0.0

    return {
        "baseline_rows": int(len(baseline_df)),
        "current_rows": int(len(current_df)),
        "baseline_columns": list(baseline_df.columns),
        "current_columns": list(current_df.columns),
        "shared_columns": common_columns,
        "overall_drift_score": overall_score,
        "overall_severity": classify_drift_severity(overall_score),
        "drift_detected": any(metric["drift_score"] >= 25 for metric in feature_metrics.values()),
        "feature_metrics": feature_metrics,
    }


def plot_drift_distribution(feature_name, baseline_df, current_df):
    column_reference = baseline_df[feature_name]
    column_current = current_df[feature_name]

    if pd.api.types.is_numeric_dtype(column_reference) and pd.api.types.is_numeric_dtype(column_current):
        baseline_hist = pd.Series(column_reference).dropna()
        current_hist = pd.Series(column_current).dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=baseline_hist, name="Baseline", opacity=0.7))
        fig.add_trace(go.Histogram(x=current_hist, name="Current", opacity=0.7))
        fig.update_layout(
            title=f"Distribution comparison for {feature_name}",
            barmode="overlay",
            template="plotly_dark",
        )
        return fig

    baseline_counts = pd.Series(column_reference.fillna("missing").astype(str)).value_counts(normalize=True)
    current_counts = pd.Series(column_current.fillna("missing").astype(str)).value_counts(normalize=True)
    combined = pd.concat([baseline_counts, current_counts], axis=1, keys=["Baseline", "Current"]).fillna(0)
    fig = px.bar(combined, barmode="group", title=f"Category drift for {feature_name}")
    return apply_modern_theme(fig)


def _prepare_time_series(df, date_column, value_column=None, frequency="W"):
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' was not found in the dataset.")

    dates = pd.to_datetime(df[date_column], errors="coerce")
    valid = df.loc[dates.notna()].copy()
    valid[date_column] = dates.loc[valid.index]
    if valid.empty:
        raise ValueError("The date column does not contain valid dates.")

    grouped = valid.set_index(date_column)
    if value_column is None:
        series = grouped.resample(frequency).size().astype(float)
    else:
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' was not found in the dataset.")
        values = pd.to_numeric(grouped[value_column], errors="coerce")
        series = values.resample(frequency).sum(min_count=1).fillna(0.0)

    return series.asfreq(frequency, fill_value=0.0).astype(float)


def analyze_trends(df, date_column, value_column=None):
    """Return weekly, monthly, and calendar-seasonality summaries."""
    weekly = _prepare_time_series(df, date_column, value_column, "W")
    monthly = _prepare_time_series(df, date_column, value_column, "ME")
    dates = pd.to_datetime(df[date_column], errors="coerce")
    valid = df.loc[dates.notna()].copy()
    valid[date_column] = dates.loc[valid.index]

    if value_column is None:
        valid["metric"] = 1.0
    else:
        valid["metric"] = pd.to_numeric(valid[value_column], errors="coerce").fillna(0.0)

    seasonal = valid.assign(
        month=valid[date_column].dt.month,
        weekday=valid[date_column].dt.day_name(),
    ).groupby(["month", "weekday"], sort=True)["metric"].agg(["count", "mean"]).reset_index()

    values = weekly.to_numpy(dtype=float)
    x_values = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x_values, values, 1)[0]) if len(values) >= 2 else 0.0
    baseline = abs(float(values[0])) if len(values) else 0.0
    if baseline == 0.0 and len(values):
        baseline = float(np.mean(np.abs(values)))
    growth_rate = ((float(values[-1]) - float(values[0])) / baseline * 100.0) if baseline else 0.0
    trend_direction = "stable" if abs(slope) < 1e-9 else ("increasing" if slope > 0 else "decreasing")
    moving_average = weekly.rolling(window=min(4, len(weekly)), min_periods=1).mean()

    return {
        "date_column": date_column,
        "value_column": value_column,
        "weekly": [{"date": index.isoformat(), "value": round(float(value), 4)} for index, value in weekly.items()],
        "monthly": [{"date": index.isoformat(), "value": round(float(value), 4)} for index, value in monthly.items()],
        "seasonal_patterns": seasonal.to_dict(orient="records"),
        "trend_direction": trend_direction,
        "trend_slope": round(slope, 6),
        "growth_rate_percent": round(float(growth_rate), 2),
        "volatility": round(float(weekly.std(ddof=0)), 6),
        "moving_average": [
            {"date": index.isoformat(), "value": round(float(value), 4)}
            for index, value in moving_average.items()
        ],
    }


def forecast_metric(df, date_column, value_column=None, periods=4, frequency="W", method="auto"):
    """Forecast future row volume or a numeric metric with optional advanced models."""
    if periods < 1 or periods > 365:
        raise ValueError("periods must be between 1 and 365.")
    series = _prepare_time_series(df, date_column, value_column, frequency)
    if len(series) < 3:
        raise ValueError("At least three time periods are required for forecasting.")

    requested_method = str(method).lower()
    if requested_method not in SUPPORTED_FORECAST_METHODS:
        raise ValueError("method must be one of: auto, arima, linear, prophet, lstm.")

    forecast_method = "linear"
    forecast_values = None
    fallback_reason = None

    if requested_method in {"auto", "prophet"}:
        try:
            from prophet import Prophet

            prophet_frame = pd.DataFrame({"ds": series.index, "y": series.to_numpy()})
            fitted = Prophet(weekly_seasonality=frequency == "W", daily_seasonality=False)
            fitted.fit(prophet_frame)
            future = fitted.make_future_dataframe(periods=periods, freq=frequency)
            prediction = fitted.predict(future)["yhat"].tail(periods).to_numpy(dtype=float)
            if np.all(np.isfinite(prediction)):
                forecast_values = prediction
                forecast_method = "Prophet"
        except ImportError as error:
            fallback_reason = f"Prophet is unavailable; used the available fallback model. Details: {error}"
        except Exception as error:
            fallback_reason = f"Prophet failed; used the available fallback model. Details: {error}"

    if requested_method == "lstm":
        fallback_reason = "LSTM requires an optional deep-learning runtime; used the linear fallback model."

    if forecast_values is None and requested_method in {"auto", "arima"} and ARIMA is not None and len(series) >= 8:
        candidate_orders = [(1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        if requested_method == "arima":
            candidate_orders = [(1, 1, 1)]

        for order in candidate_orders:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    fitted = ARIMA(series, order=order).fit()
                prediction = np.asarray(fitted.forecast(steps=periods), dtype=float)
                if np.all(np.isfinite(prediction)):
                    forecast_values = prediction
                    forecast_method = "ARIMA"
                    break
            except Exception:
                continue

        if requested_method == "arima" and forecast_values is None:
            raise RuntimeError("ARIMA forecasting failed for the supplied data.")

    if forecast_values is None:
        x_values = np.arange(len(series), dtype=float)
        slope, intercept = np.polyfit(x_values, series.to_numpy(), 1)
        forecast_values = intercept + slope * np.arange(len(series), len(series) + periods)

    future_index = pd.date_range(
        start=series.index[-1] + pd.tseries.frequencies.to_offset(frequency),
        periods=periods,
        freq=frequency,
    )
    return {
        "metric": value_column or "row_volume",
        "frequency": frequency,
        "requested_method": requested_method,
        "method": forecast_method,
        "fallback_reason": fallback_reason,
        "history": [{"date": index.isoformat(), "value": round(float(value), 4)} for index, value in series.items()],
        "forecast": [
            {"date": index.isoformat(), "value": round(float(max(value, 0.0)), 4)}
            for index, value in zip(future_index, forecast_values)
        ],
    }


def forecast_data_health(df, date_column, periods=4, frequency="W"):
    """Forecast volume, missing values, and quality score as separate time series."""
    date_values = pd.to_datetime(df[date_column], errors="coerce")
    working = df.loc[date_values.notna()].copy()
    working[date_column] = date_values.loc[working.index]
    working["missing_values"] = working.isna().sum(axis=1)
    working["quality_score"] = 100.0 - (working["missing_values"] / max(df.shape[1], 1)) * 60.0
    return {
        "volume": forecast_metric(working, date_column, None, periods, frequency),
        "missing_values": forecast_metric(working, date_column, "missing_values", periods, frequency),
        "quality_score": forecast_metric(working, date_column, "quality_score", periods, frequency),
    }


def train_and_explain_model(df, target_column, task="classification"):
    """Train a small baseline Random Forest and return feature importance plus SHAP explanations."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression.")

    working = df.dropna(subset=[target_column]).copy()
    if len(working) < 4:
        raise ValueError("At least four labeled rows are required for model explanations.")
    target = working.pop(target_column)
    for column_name in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[column_name]):
            working[column_name] = working[column_name].astype("int64") / 10**9
    features = pd.get_dummies(working, dummy_na=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    if features.shape[1] == 0:
        raise ValueError("At least one usable feature is required.")

    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)

    importances = sorted(
        [{"feature": name, "importance": round(float(value), 6)} for name, value in zip(features.columns, model.feature_importances_)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    result = {
        "task": task,
        "target": target_column,
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "feature_importance": importances,
        "shap_available": shap is not None,
        "shap_values": None,
        "explanation_summary": {
            "top_features": importances[:10],
            "model_type": type(model).__name__,
            "interpretation": "Higher feature importance indicates greater contribution to the fitted model predictions.",
        },
    }
    if shap is not None:
        try:
            explanation = shap.TreeExplainer(model)(features)
            values = explanation.values
            if values.ndim == 3:
                values = np.mean(np.abs(values), axis=2)
            else:
                values = np.abs(values)
            result["shap_values"] = [
                {"feature": name, "mean_abs_shap": round(float(value), 6)}
                for name, value in sorted(zip(features.columns, values.mean(axis=0)), key=lambda item: item[1], reverse=True)
            ]
            result["explanation_summary"]["top_shap_features"] = result["shap_values"][:10]
        except Exception:
            result["shap_available"] = False
    return result


"""`main.py` and app-level usage are intentionally minimal so the drift logic can be consumed by the API and notebook workflows without introducing a heavy UI dependency."""