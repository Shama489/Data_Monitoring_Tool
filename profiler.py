import math
import importlib
import hashlib
import json
import os
import random
import re
import warnings

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_recall_fscore_support,
    r2_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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
NEAR_DUPLICATE_EXACT_LIMIT = 500
NEAR_DUPLICATE_MAX_CANDIDATES = 1_000_000


def _near_duplicate_candidates(rows):
    row_count = len(rows)
    if row_count <= NEAR_DUPLICATE_EXACT_LIMIT:
        return {(left, right) for left in range(row_count) for right in range(left + 1, row_count)}, "exhaustive", False

    prime = 4_294_967_311
    random_generator = random.Random(42)
    coefficients = [
        (random_generator.randrange(1, prime), random_generator.randrange(0, prime))
        for _ in range(64)
    ]
    signatures = []
    for row in rows:
        text = "\x1f".join(row)
        tokens = re.findall(r"[a-z0-9]+", text)
        shingles = set(tokens)
        shingles.update(
            f"{left}\x1f{right}"
            for left, right in zip(tokens, tokens[1:])
        )
        if len(tokens) < 4:
            shingles.update(text[index:index + 2] for index in range(max(len(text) - 1, 0)))
            shingles.update(text[index:index + 3] for index in range(max(len(text) - 2, 0)))
        if not shingles:
            shingles.add(text)
        hashed_shingles = [
            int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=4).digest(), "big")
            for shingle in shingles
        ]
        if not hashed_shingles:
            hashed_shingles = [0]
        signatures.append([
            min((coefficient * value + offset) % prime for value in hashed_shingles)
            for coefficient, offset in coefficients
        ])

    candidates = set()
    truncated = False
    for band in range(32):
        buckets = {}
        start = band * 2
        for row_index, signature in enumerate(signatures):
            buckets.setdefault(tuple(signature[start:start + 2]), []).append(row_index)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for bucket_position, left in enumerate(bucket[:-1]):
                for right in bucket[bucket_position + 1:]:
                    candidates.add((left, right))
                    if len(candidates) >= NEAR_DUPLICATE_MAX_CANDIDATES:
                        truncated = True
                        break
                if truncated:
                    break
            if truncated:
                break
        if truncated:
            break

    return candidates, "minhash_lsh", truncated

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

    duplicate_details = {
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": 0,
        "similarity_threshold": similarity_threshold,
        "candidate_generation": "disabled",
        "candidate_generation_approximate": False,
        "candidate_pairs_checked": 0,
        "candidate_pairs_total": 0,
        "candidate_generation_truncated": False,
    }
    if df.shape[0] >= 2 and similarity_threshold is not None:
        from difflib import SequenceMatcher

        row_counts = {}
        for row in df.itertuples(index=False, name=None):
            normalized_row = tuple(
                "" if pd.isna(value) else str(value).strip().lower()
                for value in row
            )
            row_counts[normalized_row] = row_counts.get(normalized_row, 0) + 1

        unique_rows = list(row_counts)
        candidates, candidate_generation, truncated = _near_duplicate_candidates(unique_rows)
        near_duplicate_count = sum(count * (count - 1) // 2 for count in row_counts.values())
        for left_index, right_index in candidates:
            left = unique_rows[left_index]
            right = unique_rows[right_index]
            scores = [
                1.0 if left_text == right_text else (
                    SequenceMatcher(None, left_text, right_text).ratio()
                    if left_text and right_text else 0.0
                )
                for left_text, right_text in zip(left, right)
            ]
            if scores and (sum(scores) / len(scores)) >= similarity_threshold:
                near_duplicate_count += row_counts[left] * row_counts[right]

        duplicate_details.update({
            "candidate_generation": candidate_generation,
            "candidate_generation_approximate": candidate_generation == "minhash_lsh",
            "candidate_pairs_checked": len(candidates),
            "candidate_pairs_total": math.comb(len(unique_rows), 2),
            "candidate_generation_truncated": truncated,
        })
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
                now = pd.Timestamp.now(tz=latest_timestamp.tz)
                freshness_report["age_hours"] = round(float((now - latest_timestamp).total_seconds() / 3600), 2)
                freshness_report["is_fresh"] = freshness_report["age_hours"] <= float(max_age_hours)
            else:
                freshness_report["is_fresh"] = False
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


def answer_monitoring_question(
    question: str,
    df=None,
    quality_report=None,
    drift_report=None,
    anomaly_report=None,
    monitoring_results=None,
    language="en",
):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    normalized_question = question.strip().lower()
    language = str(language).lower()
    supported_languages = {"en", "es", "fr", "hi"}
    if language not in supported_languages:
        raise ValueError(f"language must be one of: {', '.join(sorted(supported_languages))}")

    phrases = {
        "en": {
            "missing": ("missing", "null", "blank", "na"),
            "quality": ("quality", "score", "health", "good", "bad"),
            "anomaly": ("anomaly", "outlier", "abnormal"),
            "drift": ("drift", "shift", "distribution"),
            "summary": ("summary", "overall", "status", "monitor"),
        },
        "es": {
            "missing": ("faltante", "faltantes", "nulo", "nulos", "vacío", "vacio"),
            "quality": ("calidad", "puntuación", "puntuacion", "puntaje", "salud"),
            "anomaly": ("anomalía", "anomalia", "atípico", "atipico", "anómalo", "anomalo"),
            "drift": ("deriva", "cambio", "distribución", "distribucion"),
            "summary": ("resumen", "general", "estado", "supervisión", "supervision"),
        },
        "fr": {
            "missing": ("manquant", "manquants", "manquante", "manquantes", "nul", "nulle", "vide"),
            "quality": ("qualité", "qualite", "score", "santé", "sante"),
            "anomaly": ("anomalie", "anomalies", "aberrant", "anormal"),
            "drift": ("dérive", "derive", "changement", "distribution"),
            "summary": ("résumé", "resume", "global", "état", "etat", "surveillance"),
        },
        "hi": {
            "missing": ("लापता", "गुम", "खाली", "शून्य"),
            "quality": ("गुणवत्ता", "स्कोर", "अंक", "स्वास्थ्य"),
            "anomaly": ("विसंगति", "विसंगतियां", "असामान्य", "आउटलायर"),
            "drift": ("ड्रिफ्ट", "बदलाव", "वितरण"),
            "summary": ("सारांश", "स्थिति", "निगरानी", "कुल"),
        },
    }
    localized = {
        "en": {
            "missing_values": "There are {total} missing values across {count} columns.",
            "no_missing": "There are no missing values in the current dataset.",
            "impacted": " Impacted columns include: {columns}.",
            "score": "The overall data quality score is {score:.1f}/100 ({profile} quality profile).",
            "score_unavailable": "The data quality score is not available from the provided monitoring context.",
            "anomalies": "I found {total} anomalies, which is {percent:.1f}% of the rows in the dataset.",
            "anomalies_unavailable": "No anomaly summary was provided for the current dataset.",
            "drift_yes": "Yes—drift was detected. The overall drift score is {score:.1f} ({severity} severity).",
            "drift_no": "No material drift was detected. The overall drift score is {score:.1f} ({severity} severity).",
            "drift_unavailable": "No drift report is available for the current monitoring context.",
            "summary": "Overall quality is {quality:.1f}/100, drift is {drift:.1f}, and there are {anomalies} anomalies in the current monitoring snapshot.",
            "help": "I can answer questions about quality scores, missing values, anomalies, drift, and monitoring status.",
        },
        "es": {
            "missing_values": "Hay {total} valores faltantes en {count} columnas.",
            "no_missing": "No hay valores faltantes en el conjunto de datos actual.",
            "impacted": " Columnas afectadas: {columns}.",
            "score": "La puntuación general de calidad de los datos es {score:.1f}/100 (calidad {profile}).",
            "score_unavailable": "La puntuación de calidad no está disponible en el contexto de supervisión proporcionado.",
            "anomalies": "Se encontraron {total} anomalías, que representan el {percent:.1f}% de las filas.",
            "anomalies_unavailable": "No se proporcionó un informe de anomalías para el conjunto de datos actual.",
            "drift_yes": "Sí, se detectó deriva. La puntuación general es {score:.1f} (severidad {severity}).",
            "drift_no": "No se detectó deriva significativa. La puntuación es {score:.1f} (severidad {severity}).",
            "drift_unavailable": "No hay un informe de deriva disponible en el contexto actual.",
            "summary": "La calidad general es {quality:.1f}/100, la deriva es {drift:.1f} y hay {anomalies} anomalías en la supervisión actual.",
            "help": "Puedo responder sobre calidad, valores faltantes, anomalías, deriva y estado de supervisión.",
        },
        "fr": {
            "missing_values": "Il y a {total} valeurs manquantes dans {count} colonnes.",
            "no_missing": "Il n’y a pas de valeurs manquantes dans le jeu de données actuel.",
            "impacted": " Colonnes concernées : {columns}.",
            "score": "Le score global de qualité des données est de {score:.1f}/100 (qualité {profile}).",
            "score_unavailable": "Le score de qualité n’est pas disponible dans le contexte de surveillance fourni.",
            "anomalies": "{total} anomalies ont été détectées, soit {percent:.1f}% des lignes.",
            "anomalies_unavailable": "Aucun rapport d’anomalies n’a été fourni pour le jeu de données actuel.",
            "drift_yes": "Oui, une dérive a été détectée. Le score global est de {score:.1f} (gravité {severity}).",
            "drift_no": "Aucune dérive significative n’a été détectée. Le score est de {score:.1f} (gravité {severity}).",
            "drift_unavailable": "Aucun rapport de dérive n’est disponible dans le contexte actuel.",
            "summary": "La qualité globale est de {quality:.1f}/100, la dérive est de {drift:.1f} et il y a {anomalies} anomalies dans la surveillance actuelle.",
            "help": "Je peux répondre sur la qualité, les valeurs manquantes, les anomalies, la dérive et l’état de surveillance.",
        },
        "hi": {
            "missing_values": "{count} कॉलम में {total} मान अनुपलब्ध हैं।",
            "no_missing": "वर्तमान डेटासेट में कोई अनुपलब्ध मान नहीं है।",
            "impacted": " प्रभावित कॉलम: {columns}।",
            "score": "डेटा गुणवत्ता का कुल स्कोर {score:.1f}/100 है (गुणवत्ता {profile})।",
            "score_unavailable": "दिए गए निगरानी संदर्भ में डेटा गुणवत्ता स्कोर उपलब्ध नहीं है।",
            "anomalies": "{total} विसंगतियां मिलीं, जो डेटासेट की {percent:.1f}% पंक्तियां हैं।",
            "anomalies_unavailable": "वर्तमान डेटासेट के लिए विसंगति रिपोर्ट उपलब्ध नहीं है।",
            "drift_yes": "हां, डेटा ड्रिफ्ट मिला। कुल स्कोर {score:.1f} है (गंभीरता {severity})।",
            "drift_no": "कोई महत्वपूर्ण डेटा ड्रिफ्ट नहीं मिला। स्कोर {score:.1f} है (गंभीरता {severity})।",
            "drift_unavailable": "वर्तमान निगरानी संदर्भ में ड्रिफ्ट रिपोर्ट उपलब्ध नहीं है।",
            "summary": "कुल गुणवत्ता {quality:.1f}/100 है, ड्रिफ्ट {drift:.1f} है और वर्तमान निगरानी में {anomalies} विसंगतियां हैं।",
            "help": "मैं गुणवत्ता स्कोर, अनुपलब्ध मान, विसंगतियों, ड्रिफ्ट और निगरानी स्थिति के बारे में उत्तर दे सकता हूं।",
        },
    }[language]
    intent_phrases = phrases[language]

    def has_intent(intent):
        return any(token in normalized_question for token in intent_phrases[intent])

    if monitoring_results is None:
        monitoring_results = {}

    if quality_report is None and df is not None:
        quality_report = check_data_quality(df)
    if drift_report is None and isinstance(monitoring_results, dict):
        drift_report = monitoring_results.get("drift")
    if anomaly_report is None and isinstance(monitoring_results, dict):
        anomaly_report = monitoring_results.get("anomalies")
    if anomaly_report is None and df is not None:
        anomaly_report = detect_anomalies_isolation_forest(df)

    quality_score = None
    if df is not None:
        quality_score = calculate_data_quality_score(df)

    missing_total = 0
    missing_columns = []
    if isinstance(quality_report, dict):
        missing_total = int(quality_report.get("total_nulls", 0))
        missing_columns = [
            column for column, value in quality_report.get("null_per_column", {}).items()
            if value and int(value) > 0
        ]

    if has_intent("missing"):
        answer = (
            localized["missing_values"].format(
                total=missing_total, count=len(missing_columns)
            )
            if missing_columns
            else localized["no_missing"]
        )
        if missing_columns:
            answer += localized["impacted"].format(
                columns=", ".join(missing_columns[:5])
            )
        return {
            "category": "quality",
            "answer": answer,
            "confidence": "high",
            "context": {
                "total_missing_values": missing_total,
                "columns_with_missing_values": missing_columns,
            },
        }

    if has_intent("quality"):
        score = quality_score.get("overall_score", 0) if quality_score else 0
        if score is not None:
            profile = (
                ("low" if score < 60 else "moderate" if score < 85 else "strong")
                if language == "en"
                else ("baja" if score < 60 else "moderada" if score < 85 else "alta")
                if language == "es"
                else ("faible" if score < 60 else "modérée" if score < 85 else "élevée")
                if language == "fr"
                else ("कम" if score < 60 else "मध्यम" if score < 85 else "अच्छी")
            )
            answer = localized["score"].format(score=float(score), profile=profile)
        else:
            answer = localized["score_unavailable"]
        return {
            "category": "quality",
            "answer": answer,
            "confidence": "high",
            "context": {"quality_score": score},
        }

    if has_intent("anomaly"):
        if isinstance(anomaly_report, dict) and anomaly_report.get("total_anomalies") is not None:
            total = int(anomaly_report.get("total_anomalies", 0))
            pct = float(anomaly_report.get("anomaly_percentage", 0.0))
            answer = localized["anomalies"].format(total=total, percent=pct)
        else:
            answer = localized["anomalies_unavailable"]
        return {
            "category": "anomaly",
            "answer": answer,
            "confidence": "medium",
            "context": {"anomaly_report": anomaly_report},
        }

    if has_intent("drift"):
        if isinstance(drift_report, dict):
            score = float(drift_report.get("overall_drift_score", 0.0))
            severity = drift_report.get("overall_severity", "unknown")
            if drift_report.get("drift_detected"):
                answer = localized["drift_yes"].format(score=score, severity=severity)
            else:
                answer = localized["drift_no"].format(score=score, severity=severity)
        else:
            answer = localized["drift_unavailable"]
        return {
            "category": "drift",
            "answer": answer,
            "confidence": "high",
            "context": {"drift_report": drift_report},
        }

    if has_intent("summary"):
        quality_score_value = quality_score.get("overall_score", 0) if quality_score else 0
        drift_score = float(drift_report.get("overall_drift_score", 0.0)) if isinstance(drift_report, dict) else 0.0
        anomaly_total = int(anomaly_report.get("total_anomalies", 0)) if isinstance(anomaly_report, dict) else 0
        answer = localized["summary"].format(
            quality=float(quality_score_value),
            drift=drift_score,
            anomalies=anomaly_total,
        )
        return {
            "category": "monitoring",
            "answer": answer,
            "confidence": "medium",
            "context": {
                "quality_score": quality_score_value,
                "drift_score": drift_score,
                "anomaly_total": anomaly_total,
            },
        }

    answer = localized["help"]
    return {
        "category": "general",
        "answer": answer,
        "confidence": "low",
        "context": {
            "quality_report": quality_report,
            "drift_report": drift_report,
            "anomaly_report": anomaly_report,
        },
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


def _forecast_lstm(series, periods):
    try:
        tf = importlib.import_module("tensorflow")
    except ImportError as error:
        raise ImportError("LSTM forecasting requires TensorFlow; install project requirements.") from error

    if len(series) < 10:
        raise ValueError("LSTM forecasting requires at least 10 time periods.")

    values = series.to_numpy(dtype=np.float32)
    center = float(np.mean(values))
    scale = float(np.std(values)) or 1.0
    normalized = (values - center) / scale
    lookback = min(12, max(3, len(normalized) // 4))
    train_features = np.asarray([
        normalized[index - lookback:index]
        for index in range(lookback, len(normalized))
    ], dtype=np.float32).reshape(-1, lookback, 1)
    train_targets = normalized[lookback:]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        train_features,
        train_targets,
        epochs=50,
        batch_size=min(16, len(train_features)),
        verbose=0,
        shuffle=False,
    )

    window = normalized[-lookback:].copy()
    predictions = []
    for _ in range(periods):
        prediction = float(model.predict(window.reshape(1, lookback, 1), verbose=0)[0, 0])
        predictions.append(prediction * scale + center)
        window = np.append(window[1:], prediction)
    return np.asarray(predictions, dtype=float)


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
        try:
            forecast_values = _forecast_lstm(series, periods)
            forecast_method = "LSTM"
        except ImportError as error:
            fallback_reason = f"{error} Used the linear fallback model."
        except Exception as error:
            fallback_reason = f"LSTM failed; used the linear fallback model. Details: {error}"

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


def _model_candidates(task: str):
    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "k_nearest_neighbors": KNeighborsClassifier(n_neighbors=5),
        }
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=42),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "k_nearest_neighbors": KNeighborsRegressor(n_neighbors=5),
    }


def _feature_importance_for_model(model, feature_names):
    if hasattr(model, "feature_importances_"):
        raw_scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_scores = np.abs(model.coef_)
        if raw_scores.ndim > 1:
            raw_scores = np.mean(np.abs(raw_scores), axis=0)
    else:
        return []

    if np.isscalar(raw_scores):
        raw_scores = np.full(len(feature_names), float(raw_scores))
    scores = [
        {"feature": name, "importance": round(float(value), 6)}
        for name, value in zip(feature_names, raw_scores)
    ]
    return sorted(scores, key=lambda item: item["importance"], reverse=True)


def _normalize_metric_name(task: str, metric: str) -> str:
    metric_name = str(metric).lower().replace(" ", "_")
    if task == "classification":
        aliases = {
            "accuracy": "accuracy",
            "acc": "accuracy",
            "precision": "precision_weighted",
            "precision_weighted": "precision_weighted",
            "weighted_precision": "precision_weighted",
            "recall": "recall_weighted",
            "recall_weighted": "recall_weighted",
            "weighted_recall": "recall_weighted",
            "f1": "f1_weighted",
            "f1_weighted": "f1_weighted",
            "weighted_f1": "f1_weighted",
            "f1_macro": "f1_macro",
            "macro_f1": "f1_macro",
        }
    elif task == "regression":
        aliases = {
            "r2": "r2",
            "r2_score": "r2",
            "mse": "mse",
            "mean_squared_error": "mse",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "mae": "mae",
            "mean_absolute_error": "mae",
            "msle": "msle",
            "mean_squared_log_error": "msle",
        }
    else:
        raise ValueError("task must be classification or regression.")

    if metric_name not in aliases:
        raise ValueError(f"Unsupported {task} metric: {metric}")
    return aliases[metric_name]


def _metric_value(y_true, y_pred, task: str, metric: str):
    metric_name = _normalize_metric_name(task, metric)
    if task == "classification":
        if metric_name == "accuracy":
            return accuracy_score(y_true, y_pred)
        if metric_name in {"precision_weighted", "recall_weighted"}:
            scores = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            return scores[0 if metric_name == "precision_weighted" else 1]
        if metric_name == "f1_weighted":
            return f1_score(y_true, y_pred, average="weighted", zero_division=0)
        if metric_name == "f1_macro":
            return f1_score(y_true, y_pred, average="macro", zero_division=0)

    if metric_name == "r2":
        return r2_score(y_true, y_pred)
    if metric_name == "mse":
        return mean_squared_error(y_true, y_pred)
    if metric_name == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    if metric_name == "mae":
        return mean_absolute_error(y_true, y_pred)
    if metric_name == "msle":
        return mean_squared_log_error(y_true, y_pred)


def _metric_direction(task: str, metric: str) -> str:
    metric_name = _normalize_metric_name(task, metric)
    if metric_name in {"mae", "mse", "rmse", "msle"}:
        return "lower"
    return "higher"


def compare_models(df, target_column, task="classification", metric=None, test_size=0.2, model_names=None):
    """Compare a shortlist of candidate models and return the best performer."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression.")

    try:
        test_size = float(test_size)
    except (TypeError, ValueError) as error:
        raise ValueError("test_size must be a number between 0.1 and 0.5.") from error
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5.")

    working = df.dropna(subset=[target_column]).copy()
    if len(working) < 5:
        raise ValueError("At least five labeled rows are required for model comparison.")
    target = working.pop(target_column)
    for column_name in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[column_name]):
            working[column_name] = working[column_name].astype("int64") / 10**9
    features = pd.get_dummies(working, dummy_na=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    if features.shape[1] == 0:
        raise ValueError("At least one usable feature is required for model comparison.")

    stratify = None
    if task == "classification":
        class_counts = target.value_counts()
        test_rows = math.ceil(len(target) * test_size)
        train_rows = len(target) - test_rows
        if class_counts.min() >= 2 and test_rows >= len(class_counts) and train_rows >= len(class_counts):
            stratify = target
    try:
        train_features, test_features, train_target, test_target = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as error:
        raise ValueError(f"Could not create a holdout split: {error}") from error

    default_metric = "accuracy" if task == "classification" else "r2"
    chosen_metric = default_metric if metric is None else str(metric)
    _normalize_metric_name(task, chosen_metric)
    candidate_names = _model_candidates(task)
    if model_names:
        requested = [str(name) for name in model_names]
        unsupported = [name for name in requested if name not in candidate_names]
        if unsupported:
            raise ValueError(f"Unsupported model names for {task}: {unsupported}")
        requested = list(dict.fromkeys(requested))
        candidate_names = {key: candidate_names[key] for key in requested}

    model_results = []
    model_failures = []
    for name, model in candidate_names.items():
        try:
            fitted = model.fit(train_features, train_target)
            predictions = fitted.predict(test_features)
            metric_value = _metric_value(test_target, predictions, task, chosen_metric)
            model_results.append({
                "model": name,
                "score": float(metric_value),
                "metrics": {
                    "accuracy": accuracy_score(test_target, predictions) if task == "classification" else None,
                    "precision_weighted": precision_recall_fscore_support(test_target, predictions, average="weighted", zero_division=0)[0] if task == "classification" else None,
                    "recall_weighted": precision_recall_fscore_support(test_target, predictions, average="weighted", zero_division=0)[1] if task == "classification" else None,
                    "f1_weighted": f1_score(test_target, predictions, average="weighted", zero_division=0) if task == "classification" else None,
                    "mae": mean_absolute_error(test_target, predictions) if task == "regression" else None,
                    "rmse": float(np.sqrt(mean_squared_error(test_target, predictions))) if task == "regression" else None,
                    "r2": r2_score(test_target, predictions) if task == "regression" else None,
                    chosen_metric: float(metric_value),
                },
                "feature_importance": _feature_importance_for_model(fitted, features.columns.tolist())[:10],
            })
        except Exception as error:
            model_failures.append({"model": name, "error": str(error)})
            continue

    if not model_results:
        failure_details = "; ".join(
            f"{failure['model']}: {failure['error']}" for failure in model_failures
        )
        raise ValueError(
            f"No candidate model could be trained for the {task} task. "
            f"Model failures: {failure_details}"
        )

    direction = _metric_direction(task, chosen_metric)
    best_entry = max(model_results, key=lambda item: item["score"]) if direction == "higher" else min(model_results, key=lambda item: item["score"])
    ranking = sorted(
        model_results,
        key=lambda item: item["score"],
        reverse=(direction == "higher"),
    )

    return {
        "task": task,
        "target": target_column,
        "metric": chosen_metric,
        "direction": direction,
        "model_failures": model_failures,
        "selected_model": best_entry["model"],
        "best_score": round(float(best_entry["score"]), 6),
        "models": [
            {
                "model": item["model"],
                "score": round(float(item["score"]), 6),
                "metrics": item["metrics"],
                "feature_importance": item["feature_importance"],
            }
            for item in ranking
        ],
        "ranked_models": [item["model"] for item in ranking],
        "best_model_summary": {
            "model": best_entry["model"],
            "score": round(float(best_entry["score"]), 6),
            "feature_importance": best_entry["feature_importance"],
        },
    }


def select_best_model(df, target_column, task="classification", metric=None, test_size=0.2, model_names=None):
    """Shortcut wrapper for AutoML model selection and ranking."""
    return compare_models(df, target_column, task=task, metric=metric, test_size=test_size, model_names=model_names)


def automl_model_selection(df, target_column, task="classification", metric=None, test_size=0.2, model_names=None):
    """Compatibility alias for model-selection workflows."""
    return compare_models(df, target_column, task=task, metric=metric, test_size=test_size, model_names=model_names)


def _advanced_feature_matrix(df, feature_columns=None, exclude_columns=()):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("A non-empty dataframe is required.")

    if feature_columns is None:
        selected_columns = [
            column for column in df.columns if column not in set(exclude_columns)
        ]
    else:
        if not isinstance(feature_columns, (list, tuple)) or not feature_columns:
            raise ValueError("feature_columns must be a non-empty list when provided.")
        missing_columns = [column for column in feature_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Feature columns were not found: {missing_columns}")
        selected_columns = list(dict.fromkeys(feature_columns))

    if not selected_columns:
        raise ValueError("At least one feature column is required.")

    working = df.loc[:, selected_columns].copy()
    for column in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[column]):
            parsed_dates = pd.to_datetime(working[column], errors="coerce")
            working[column] = parsed_dates.astype("int64").where(
                parsed_dates.notna(), np.nan
            ) / 10**9
        elif not pd.api.types.is_numeric_dtype(working[column]):
            working[column] = working[column].astype("string").fillna("__missing__")

    features = pd.get_dummies(working, dummy_na=True, dtype=float)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    varying_columns = features.columns[features.nunique(dropna=False) > 1]
    features = features.loc[:, varying_columns].astype(float)
    if features.shape[1] == 0:
        raise ValueError("At least one non-constant usable feature is required.")
    return features


def _json_safe_value(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def analyze_clusters(df, n_clusters=3, feature_columns=None):
    """Cluster rows and report assignments, cluster sizes, and separation quality."""
    if isinstance(n_clusters, bool):
        raise ValueError("n_clusters must be an integer of at least 2.")
    try:
        n_clusters = int(n_clusters)
    except (TypeError, ValueError) as error:
        raise ValueError("n_clusters must be an integer of at least 2.") from error
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if len(df) <= n_clusters:
        raise ValueError("n_clusters must be smaller than the number of rows.")

    features = _advanced_feature_matrix(df, feature_columns=feature_columns)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(scaled_features)
    cluster_counts = pd.Series(labels).value_counts().sort_index()

    score = None
    if 1 < len(cluster_counts) < len(labels):
        score = float(silhouette_score(
            scaled_features,
            labels,
            sample_size=min(2000, len(labels)),
            random_state=42,
        ))

    centers = scaler.inverse_transform(model.cluster_centers_)
    cluster_summaries = []
    for cluster_id, count in cluster_counts.items():
        center = centers[cluster_id]
        representative_features = sorted(
            zip(features.columns, center),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:5]
        cluster_summaries.append({
            "cluster": int(cluster_id),
            "rows": int(count),
            "center": {
                str(name): round(float(value), 6)
                for name, value in zip(features.columns, center)
            },
            "representative_features": [
                {"feature": str(name), "value": round(float(value), 6)}
                for name, value in representative_features
            ],
        })

    assignments = [
        {"row_index": str(index), "cluster": int(label)}
        for index, label in zip(df.index[:1000], labels[:1000])
    ]
    return {
        "analysis": "clustering",
        "rows": int(len(df)),
        "feature_count": int(features.shape[1]),
        "features": [str(column) for column in features.columns],
        "requested_clusters": n_clusters,
        "actual_clusters": int(len(cluster_counts)),
        "silhouette_score": round(score, 6) if score is not None else None,
        "clusters": cluster_summaries,
        "assignments": assignments,
        "assignments_truncated": len(df) > len(assignments),
    }


def detect_fraud_patterns(df, feature_columns=None, contamination=0.05):
    """Flag anomalous transactions for fraud review; detections are not fraud verdicts."""
    if isinstance(contamination, bool):
        raise ValueError("contamination must be between 0.001 and 0.5.")
    try:
        contamination = float(contamination)
    except (TypeError, ValueError) as error:
        raise ValueError("contamination must be between 0.001 and 0.5.") from error
    if not math.isfinite(contamination) or not 0.001 <= contamination <= 0.5:
        raise ValueError("contamination must be between 0.001 and 0.5.")

    features = _advanced_feature_matrix(df, feature_columns=feature_columns)
    if len(features) < 2:
        raise ValueError("At least two rows are required for fraud-pattern detection.")
    scaled_features = StandardScaler().fit_transform(features)
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
    ).fit(scaled_features)
    predictions = model.predict(scaled_features)
    risk_scores = -model.decision_function(scaled_features)

    suspicious = []
    for position in np.flatnonzero(predictions == -1):
        row = {
            "row_index": str(df.index[position]),
            "risk_score": round(float(risk_scores[position]), 6),
            "top_signals": [
                {"feature": str(features.columns[feature_index]), "deviation": round(float(scaled_features[position, feature_index]), 4)}
                for feature_index in np.argsort(np.abs(scaled_features[position]))[::-1][:5]
            ],
        }
        suspicious.append(row)

    return {
        "analysis": "fraud_detection",
        "method": "isolation_forest",
        "rows": int(len(df)),
        "feature_count": int(features.shape[1]),
        "contamination": contamination,
        "total_flagged": len(suspicious),
        "flagged_rows": suspicious[:1000],
        "flagged_rows_truncated": len(suspicious) > 1000,
        "interpretation": "Flags are unusual patterns for investigation, not confirmed fraud.",
    }


def analyze_predictive_maintenance(
    df,
    target_column,
    task="classification",
    metric=None,
    test_size=0.2,
    model_names=None,
):
    """Select and evaluate models for failure classification or remaining-life prediction."""
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression.")
    comparison = compare_models(
        df,
        target_column,
        task=task,
        metric=metric,
        test_size=test_size,
        model_names=model_names,
    )
    comparison["analysis"] = "predictive_maintenance"
    comparison["target_interpretation"] = (
        "Failure/event classification" if task == "classification"
        else "Remaining useful life or another numeric maintenance outcome"
    )
    comparison["interpretation"] = (
        "Model rankings estimate predictive performance on a held-out sample; "
        "validate against time-based splits before operational use."
    )
    return comparison


def recommend_items(
    df,
    user_column,
    item_column,
    rating_column=None,
    user_id=None,
    top_n=10,
):
    """Recommend unseen items using item similarity with a popularity fallback."""
    required_columns = [user_column, item_column]
    if rating_column:
        required_columns.append(rating_column)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Recommendation columns were not found: {missing_columns}")
    if isinstance(top_n, bool):
        raise ValueError("top_n must be an integer between 1 and 100.")
    try:
        top_n = int(top_n)
    except (TypeError, ValueError) as error:
        raise ValueError("top_n must be an integer between 1 and 100.") from error
    if not 1 <= top_n <= 100:
        raise ValueError("top_n must be between 1 and 100.")

    selected = df[required_columns].dropna(subset=[user_column, item_column]).copy()
    if selected.empty:
        raise ValueError("At least one user-item interaction is required.")
    if rating_column:
        selected["_rating"] = pd.to_numeric(selected[rating_column], errors="coerce")
        selected = selected.dropna(subset=["_rating"])
        if selected.empty:
            raise ValueError("The rating column must contain numeric ratings.")
    else:
        selected["_rating"] = 1.0

    users = pd.Index(pd.unique(selected[user_column]))
    items = pd.Index(pd.unique(selected[item_column]))
    user_lookup = {value: index for index, value in enumerate(users)}
    item_lookup = {value: index for index, value in enumerate(items)}
    selected["_user_code"] = selected[user_column].map(user_lookup)
    selected["_item_code"] = selected[item_column].map(item_lookup)
    interactions = selected.groupby(
        ["_user_code", "_item_code"], sort=False
    )["_rating"].mean()
    item_popularity = selected.groupby("_item_code", sort=False).size().reindex(
        range(len(items)), fill_value=0
    ).to_numpy(dtype=float)
    popular_codes = np.argsort(-item_popularity, kind="stable")

    selected_user_code = None
    if user_id is not None:
        try:
            selected_user_code = user_lookup.get(user_id)
        except TypeError:
            selected_user_code = None
        if selected_user_code is None:
            fallback_reason = "User was not found in interactions; returning popular items."

    seen_codes = set()
    scoring = None
    recommendation_method = "popularity"
    fallback_reason = None
    if selected_user_code is not None:
        seen_codes = set(
            selected.loc[
                selected["_user_code"] == selected_user_code, "_item_code"
            ].astype(int).tolist()
        )
        if len(items) <= 2000 and len(users) * len(items) <= 4_000_000:
            interaction_matrix = interactions.unstack(fill_value=0).reindex(
                index=range(len(users)),
                columns=range(len(items)),
                fill_value=0,
            )
            user_ratings = interaction_matrix.iloc[selected_user_code].to_numpy(dtype=float)
            similarity = cosine_similarity(interaction_matrix.to_numpy(dtype=float).T)
            weights = (user_ratings != 0).astype(float)
            denominator = np.abs(similarity) @ weights
            scoring = np.divide(
                similarity @ user_ratings,
                denominator,
                out=np.zeros(len(items), dtype=float),
                where=denominator > 0,
            )
            scoring[list(seen_codes)] = -np.inf
            recommendation_method = "item_similarity"
        else:
            fallback_reason = "Interaction matrix exceeded the item-similarity size limit."

    if scoring is not None and np.any(np.isfinite(scoring) & (scoring != 0)):
        candidate_codes = sorted(
            (code for code in range(len(items)) if code not in seen_codes),
            key=lambda code: (-float(scoring[code]), -item_popularity[code], code),
        )
        candidate_reason = "similarity to items already used by this user"
    else:
        candidate_codes = [
            int(code) for code in popular_codes if int(code) not in seen_codes
        ]
        if selected_user_code is not None and not fallback_reason:
            fallback_reason = "No usable item-similarity scores were available."
        if fallback_reason:
            recommendation_method = "popularity_fallback"
        candidate_reason = "overall interaction popularity"

    recommendations = []
    for code in candidate_codes[:top_n]:
        score = (
            float(scoring[code])
            if scoring is not None and np.isfinite(scoring[code])
            else float(item_popularity[code])
        )
        recommendations.append({
            "item": _json_safe_value(items[code]),
            "score": round(score, 6),
            "interaction_count": int(item_popularity[code]),
            "reason": candidate_reason,
        })

    return {
        "analysis": "recommendation",
        "user_id": _json_safe_value(user_id),
        "method": recommendation_method,
        "users": int(len(users)),
        "items": int(len(items)),
        "interactions": int(len(selected)),
        "recommendations": recommendations,
        "fallback_reason": fallback_reason,
    }


def train_and_explain_model(df, target_column, task="classification", test_size=0.2):
    """Train a Random Forest, evaluate a holdout set, and return global/local explanations."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression.")
    try:
        test_size = float(test_size)
    except (TypeError, ValueError) as error:
        raise ValueError("test_size must be a number between 0.1 and 0.5.") from error
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5.")

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

    stratify = None
    if task == "classification":
        class_counts = target.value_counts()
        test_rows = math.ceil(len(target) * test_size)
        train_rows = len(target) - test_rows
        if class_counts.min() >= 2 and test_rows >= len(class_counts) and train_rows >= len(class_counts):
            stratify = target

    try:
        train_features, test_features, train_target, test_target = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as error:
        raise ValueError(f"Could not create a holdout split: {error}") from error

    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)

    importances = sorted(
        [{"feature": name, "importance": round(float(value), 6)} for name, value in zip(features.columns, model.feature_importances_)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    if task == "classification":
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_target, predictions, average="weighted", zero_division=0
        )
        labels = list(model.classes_)
        holdout_metrics = {
            "accuracy": round(float(accuracy_score(test_target, predictions)), 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
            "confusion_matrix": confusion_matrix(test_target, predictions, labels=labels).tolist(),
            "class_labels": [str(label) for label in labels],
        }
    else:
        holdout_metrics = {
            "mae": round(float(mean_absolute_error(test_target, predictions)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(test_target, predictions))), 4),
            "r2": round(float(r2_score(test_target, predictions)), 4) if len(test_target) > 1 else None,
        }

    prediction_rows = []
    for row_index, actual, predicted in zip(test_target.index, test_target.tolist(), predictions.tolist()):
        prediction_rows.append({
            "row_index": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
            "actual": actual.item() if isinstance(actual, np.generic) else actual,
            "predicted": predicted.item() if isinstance(predicted, np.generic) else predicted,
        })

    result = {
        "task": task,
        "target": target_column,
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "test_size": test_size,
        "holdout": {
            "rows": int(len(test_features)),
            "metrics": holdout_metrics,
            "predictions": prediction_rows[:200],
        },
        "feature_importance": importances,
        "shap_available": shap is not None,
        "shap_values": None,
        "local_explanations": [],
        "explanation_summary": {
            "top_features": importances[:10],
            "model_type": type(model).__name__,
            "interpretation": "Global feature importance describes model reliance; holdout metrics estimate performance on unseen rows.",
        },
    }
    if shap is not None:
        try:
            explanation_rows = test_features.head(100)
            explanation = shap.TreeExplainer(model)(explanation_rows)
            values = explanation.values
            if values.ndim == 3:
                global_values = np.mean(np.abs(values), axis=(0, 2))
            else:
                global_values = np.mean(np.abs(values), axis=0)
            result["shap_values"] = [
                {"feature": name, "mean_abs_shap": round(float(value), 6)}
                for name, value in sorted(zip(features.columns, global_values), key=lambda item: item[1], reverse=True)
            ]
            result["explanation_summary"]["top_shap_features"] = result["shap_values"][:10]
            for position, row_index in enumerate(explanation_rows.index):
                if values.ndim == 3:
                    predicted_class = predictions[position]
                    class_position = list(model.classes_).index(predicted_class)
                    local_values = values[position, :, class_position]
                else:
                    local_values = values[position]
                top_local = sorted(
                    zip(features.columns, local_values),
                    key=lambda item: abs(float(item[1])),
                    reverse=True,
                )[:10]
                actual = test_target.loc[row_index]
                predicted = predictions[position]
                result["local_explanations"].append({
                    "row_index": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
                    "actual": actual.item() if isinstance(actual, np.generic) else actual,
                    "predicted": predicted.item() if isinstance(predicted, np.generic) else predicted,
                    "features": [
                        {"feature": name, "shap_value": round(float(value), 6)}
                        for name, value in top_local
                    ],
                })
        except Exception:
            result["shap_available"] = False
            result["shap_values"] = None
            result["local_explanations"] = []
    return result


"""`main.py` and app-level usage are intentionally minimal so the drift logic can be consumed by the API and notebook workflows without introducing a heavy UI dependency."""