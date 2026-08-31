import math
import json
import os

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

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
def check_data_quality(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "null_per_column": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "all_null_columns": df.columns[df.isnull().all()].tolist()
    }

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
        except Exception:
            pass

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
            if base > 0 and curr > 0
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
        if base > 0 and curr > 0
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


"""`main.py` and app-level usage are intentionally minimal so the drift logic can be consumed by the API and notebook workflows without introducing a heavy UI dependency."""