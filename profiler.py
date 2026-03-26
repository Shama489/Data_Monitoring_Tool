import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

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

    return {"overall_score": max(score, 0)}

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