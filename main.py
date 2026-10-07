import asyncio
import json
import io
import logging
import math
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request

from agents.monitoring_agent import (
    MonitoringStageError,
    build_tool_registry,
    run_monitoring,
)
from profiler import (
    analyze_dataset_drift,
    analyze_trends,
    answer_monitoring_question,
    calculate_data_quality_score,
    check_data_quality,
    forecast_data_health,
    forecast_metric,
    generate_ai_quality_summary,
    train_and_explain_model,
)
from notifications import (
    ALERT_RULES,
    NotificationError,
    SUPPORTED_CHANNELS,
    SUPPORTED_OPERATORS,
    evaluate_alert_rules,
    get_persisted_alert_rules,
    get_alert_history,
    persist_alert_rules,
    send_notifications,
)
from data_sources import DataSourceError, load_source, source_capabilities, summarize_source
from monitoring_store import (
    delete_configuration,
    delete_dataset,
    create_dataset_version,
    get_dataset_lineage,
    get_audit_records,
    get_configurations,
    get_dataset,
    get_monitoring_result,
    list_datasets,
    list_dataset_versions,
    list_monitoring_results,
    persist_monitoring_run,
    record_lineage_event,
    save_dataset,
    set_configuration,
    set_configurations,
    create_monitoring_schedule,
    delete_monitoring_schedule,
    list_monitoring_schedules,
    set_monitoring_schedule_enabled,
)
from tools.data_tools import load_dataset
from auth_security import (
    AuthenticationError,
    authenticate,
    authentication_middleware,
    create_account,
    current_user,
    current_user_id,
    current_user_scope,
    issue_token,
    login_attempt_key,
    RequestBodyLimitMiddleware,
)
from monitoring_store import (
    clear_login_failures,
    list_users,
    login_retry_after,
    record_login_failure,
    update_user,
)
from scheduled_monitoring import scheduler_loop

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = asyncio.create_task(scheduler_loop(), name="scheduled-monitoring")
    app.state.scheduler_task = worker
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
        del app.state.scheduler_task


app = FastAPI(title="Data Monitoring Tool", lifespan=lifespan)
app.middleware("http")(authentication_middleware)
app.add_middleware(RequestBodyLimitMiddleware)
logger = logging.getLogger(__name__)
MAX_DATASET_ROWS = 100_000
MAX_DATASET_COLUMNS = 1_000


def _bad_request(message: str) -> None:
    raise HTTPException(status_code=400, detail=message)


def _quality_options(payload: dict[str, Any]) -> dict[str, Any]:
    rules = payload.get("rules")
    if rules is not None and not isinstance(rules, (dict, list)):
        _bad_request("rules must be an object or list of rule objects")
    if isinstance(rules, list) and not all(isinstance(rule, dict) for rule in rules):
        _bad_request("rules list entries must be objects")

    expected_columns = payload.get("expected_columns")
    if expected_columns is not None and not isinstance(expected_columns, list):
        _bad_request("expected_columns must be a list")

    expected_dtypes = payload.get("expected_dtypes")
    if expected_dtypes is not None and not isinstance(expected_dtypes, dict):
        _bad_request("expected_dtypes must be an object")

    return {
        "expected_columns": expected_columns,
        "expected_dtypes": expected_dtypes,
        "timestamp_column": payload.get("timestamp_column"),
        "max_age_hours": payload.get("max_age_hours"),
        "similarity_threshold": payload.get("similarity_threshold", 0.8),
        "rules": rules,
    }


def _dataset_name(payload: dict[str, Any], default: str = "monitoring_dataset") -> str:
    name = payload.get("dataset_name", payload.get("name"))
    if isinstance(name, str) and name.strip():
        return name.strip()
    source = payload.get("source")
    if isinstance(source, dict):
        path = source.get("path") or source.get("location")
        if path:
            return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or default
    return default


def _dataset_source(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    if not isinstance(source, dict):
        return {"type": "inline"}
    details = {
        key: source[key]
        for key in ("type", "file_type", "name", "key", "collection")
        if source.get(key) is not None
    }
    path = source.get("path") or source.get("location")
    if path:
        details["name"] = str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return details


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _validate_dataset_dimensions(frame: pd.DataFrame) -> None:
    if len(frame.index) > MAX_DATASET_ROWS:
        _bad_request(f"Dataset must not exceed {MAX_DATASET_ROWS:,} rows")
    if len(frame.columns) > MAX_DATASET_COLUMNS:
        _bad_request(f"Dataset must not exceed {MAX_DATASET_COLUMNS:,} columns")


def _persist_endpoint_result(
    payload: dict[str, Any],
    frame: pd.DataFrame,
    check_type: str,
    result: dict[str, Any],
    baseline: pd.DataFrame | None = None,
) -> dict[str, Any]:
    specs = [{
        "key": "current",
        "name": _dataset_name(payload),
        "data": _frame_records(frame),
        "source": _dataset_source(payload),
    }]
    if baseline is not None:
        specs.append({
            "key": "baseline",
            "name": f"{_dataset_name(payload)}_baseline",
            "data": _frame_records(baseline),
            "source": {"type": "inline"},
        })
    dataset_ids, result_id = persist_monitoring_run(
        specs, check_type, result, owner_id=current_user_id()
    )
    return {
        "dataset_id": dataset_ids["current"],
        "result_id": result_id,
        **({"baseline_dataset_id": dataset_ids["baseline"]} if "baseline" in dataset_ids else {}),
    }


def _use_llm_quality_summary(payload: dict[str, Any]) -> bool:
    use_llm = payload.get("use_llm", False)
    if not isinstance(use_llm, bool):
        _bad_request("use_llm must be a boolean")
    return use_llm


def _notification_channels(payload: dict[str, Any]) -> list[Any]:
    for key in ("notifications", "alert_channels", "channels"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return value
        if isinstance(value, (str, dict)):
            return [value]
    return []


def _should_send_notifications(payload: dict[str, Any]) -> bool:
    return bool(payload.get("notify", payload.get("send_notifications", False))) or bool(_notification_channels(payload))


def _trigger_monitoring_notifications(
    payload: dict[str, Any],
    event_type: str,
    message: str,
    subject: str,
    severity: str = "warning",
) -> dict[str, Any]:
    channels = _notification_channels(payload)
    if not channels:
        return {"status": "skipped", "reason": "no notification channels configured"}

    alert = {
        "message": message,
        "subject": subject,
        "severity": severity,
        "event_type": event_type,
        "channels": channels,
    }
    try:
        return send_notifications(alert)
    except NotificationError as error:
        return {"status": "failed", "error": str(error)}


def _send_rule_based_notifications(payload: dict[str, Any], event_type: str, metrics: dict[str, Any], subject: str) -> dict[str, Any]:
    alert_rules = payload.get("alert_rules")
    if alert_rules is None:
        alert_rules = get_persisted_alert_rules()
    if not isinstance(alert_rules, list) or not alert_rules:
        return {"status": "skipped", "reason": "no alert rules configured"}

    rule_result = evaluate_alert_rules(metrics, alert_rules)
    if not rule_result["triggered"]:
        return {"status": "skipped", "reason": "no alert rules matched"}

    channels = []
    for match in rule_result["matches"]:
        channel = match.get("channel")
        recipient = match.get("recipient", "")
        if channel:
            channels.append({"channel": channel, "recipient": recipient})

    if not channels:
        return {"status": "skipped", "reason": "matched rules had no delivery channels"}

    summary = "; ".join(f"{m['metric']}={m['actual']}" for m in rule_result["matches"])
    severity_order = {"critical": 5, "high": 4, "medium": 3, "warning": 2, "low": 1}
    severity = max(
        (metrics.get("risk_level", "warning"), metrics.get("severity", "warning")),
        key=lambda value: severity_order.get(str(value).lower(), 0),
    )
    try:
        return send_notifications({
            "message": f"{event_type.replace('_', ' ').title()} alert: {summary}",
            "subject": subject,
            "severity": severity,
            "event_type": event_type,
            "channels": channels,
        })
    except NotificationError as error:
        return {"status": "failed", "error": str(error)}


@app.get("/")
def home():
    return {"message": "Data Monitoring Tool Running"}


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "data-monitoring-tool"}


@app.post("/api/auth/login")
def login_endpoint(payload: dict[str, Any], request: Request):
    username = payload.get("username")
    password = payload.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        _bad_request("username and password are required")
    if len(username) > 128 or len(password) > 1024:
        _bad_request("username or password exceeds the allowed length")
    try:
        client_key = login_attempt_key(
            request.client.host if request.client is not None else "unknown"
        )
        retry_after = login_retry_after(client_key)
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail="Too many failed sign-in attempts. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )
        user = authenticate(username.strip().lower(), password)
        if user is None:
            retry_after = record_login_failure(client_key)
            if retry_after:
                raise HTTPException(
                    status_code=429,
                    detail="Too many failed sign-in attempts. Try again later.",
                    headers={"Retry-After": str(retry_after)},
                )
            raise HTTPException(status_code=401, detail="Invalid username or password")
        clear_login_failures(client_key)
        token = issue_token(user)
    except AuthenticationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
        },
    }


@app.get("/api/auth/me")
def current_user_endpoint():
    user = current_user()
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
    }


@app.get("/api/auth/users")
def list_users_endpoint():
    return {"users": list_users()}


@app.post("/api/auth/users", status_code=201)
def create_user_endpoint(payload: dict[str, Any]):
    username, password, role = (
        payload.get("username"),
        payload.get("password"),
        payload.get("role"),
    )
    if not all(isinstance(value, str) for value in (username, password, role)):
        _bad_request("username, password, and role are required")
    try:
        user = create_account(username, password, role)
    except ValueError as error:
        _bad_request(str(error))
    return user


@app.patch("/api/auth/users/{user_id}")
def update_user_endpoint(user_id: str, payload: dict[str, Any]):
    role = payload.get("role")
    active = payload.get("is_active")
    if role not in {"admin", "analyst", "viewer"}:
        _bad_request("role must be admin, analyst, or viewer")
    if not isinstance(active, bool):
        _bad_request("is_active must be a boolean")
    if user_id == current_user_id() and (not active or role != "admin"):
        _bad_request("Administrators cannot deactivate or demote their own account")
    user = update_user(user_id, role, active)
    if user is None:
        raise HTTPException(status_code=404, detail="User was not found")
    return user


@app.get("/api/sources/capabilities")
def source_capabilities_endpoint():
    return source_capabilities()


@app.get("/api/monitoring/tools")
def monitoring_tools_endpoint():
    return {"tools": build_tool_registry().describe()}


@app.post("/api/assistant/ask")
def assistant_ask_endpoint(payload: dict[str, Any]):
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        _bad_request("question must be a non-empty string")

    df_value = payload.get("data") or payload.get("dataset") or payload.get("records")
    df = None
    if df_value is not None:
        if isinstance(df_value, list):
            df = pd.DataFrame(df_value)
        elif isinstance(df_value, pd.DataFrame):
            df = df_value
        elif isinstance(df_value, dict):
            df = pd.DataFrame([df_value])
        else:
            _bad_request("data, dataset, or records must be a list, dataframe, or object")

    quality_report = payload.get("quality_report")
    drift_report = payload.get("drift_report")
    anomaly_report = payload.get("anomaly_report")

    if quality_report is not None and not isinstance(quality_report, dict):
        _bad_request("quality_report must be an object")
    if drift_report is not None and not isinstance(drift_report, dict):
        _bad_request("drift_report must be an object")
    if anomaly_report is not None and not isinstance(anomaly_report, dict):
        _bad_request("anomaly_report must be an object")

    response = answer_monitoring_question(
        question,
        df=df,
        quality_report=quality_report,
        drift_report=drift_report,
        anomaly_report=anomaly_report,
    )
    return {"question": question, **response}


@app.post("/api/monitoring/analyze")
def monitoring_analyze_endpoint(payload: dict[str, Any]):
    try:
        return run_monitoring(
            payload,
            quality_options=_quality_options(payload),
            use_llm=_use_llm_quality_summary(payload),
            owner_id=current_user_scope(),
        )
    except MonitoringStageError as error:
        dataset_id = payload.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id.strip():
            dataset = get_dataset(dataset_id, owner_id=current_user_scope())
            if dataset is not None:
                record_lineage_event(
                    dataset_id,
                    "monitoring_run",
                    "monitoring",
                    "failed",
                    input_dataset_ids=[dataset_id],
                    details={
                        "checks": payload.get("checks", ["quality"]),
                        "failed_stage": error.stage,
                    },
                    error=f"{error.stage}: {type(error.cause).__name__}"[:500],
                    owner_id=current_user_scope(),
                )
        if isinstance(error.cause, (DataSourceError, TypeError, ValueError)):
            _bad_request(str(error.cause))
        logger.exception("Monitoring pipeline failed at stage %s", error.stage)
        raise HTTPException(
            status_code=500,
            detail=f"Monitoring pipeline stage '{error.stage}' failed",
        ) from error.cause
    except (DataSourceError, TypeError, ValueError) as error:
        dataset_id = payload.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id.strip():
            dataset = get_dataset(dataset_id, owner_id=current_user_scope())
            if dataset is not None:
                record_lineage_event(
                    dataset_id,
                    "monitoring_run",
                    "monitoring",
                    "failed",
                    input_dataset_ids=[dataset_id],
                    details={
                        "checks": payload.get("checks", ["quality"]),
                        "failed_stage": "validation",
                    },
                    error=f"{type(error).__name__}: {error}"[:500],
                    owner_id=current_user_scope(),
                )
        _bad_request(str(error))


@app.post("/api/sources/analyze")
def analyze_sources_endpoint(payload: dict[str, Any]):
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        _bad_request("sources must be a non-empty list")

    reports = []
    failures = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            failures.append({"index": index, "error": "Each source must be an object"})
            continue
        try:
            frame = load_source(source)
            if frame.empty:
                raise DataSourceError("Source dataset must not be empty")
            report = summarize_source(source, frame)
            report.update(
                _persist_endpoint_result(
                    {"source": source},
                    frame,
                    "source_summary",
                    report,
                )
            )
            reports.append(report)
        except (DataSourceError, TypeError, ValueError) as error:
            failures.append({"index": index, "type": source.get("type"), "error": str(error)})

    if not reports and failures:
        _bad_request(failures[0]["error"])
    return {"sources": reports, "failed": failures, "success": not failures}


@app.post("/api/notifications/send")
def send_notifications_endpoint(payload: dict[str, Any]):
    """Send one monitoring alert to email, SMS, WhatsApp, Slack, or Teams."""
    try:
        return send_notifications(payload)
    except NotificationError as error:
        _bad_request(str(error))


@app.get("/api/alerts/rules")
def list_alert_rules_endpoint():
    rules = get_persisted_alert_rules()
    ALERT_RULES[:] = rules
    return {"rules": rules}


@app.post("/api/alerts/rules")
def upsert_alert_rules_endpoint(payload: dict[str, Any]):
    if "rules" in payload:
        incoming = payload["rules"]
    else:
        incoming = [payload]

    if not isinstance(incoming, list):
        _bad_request("rules must be a list")

    valid_rules = []
    for rule in incoming:
        if not isinstance(rule, dict):
            _bad_request("each rule must be an object")
        metric = str(rule.get("metric", "")).strip()
        if not metric:
            _bad_request("rule metric is required")
        operator = str(rule.get("operator", "==")).strip().lower()
        if operator not in SUPPORTED_OPERATORS:
            _bad_request(f"unsupported alert rule operator: {operator}")
        if "value" not in rule:
            _bad_request("rule value is required")
        valid_rules.append({
            "metric": metric,
            "operator": operator,
            "value": rule.get("value"),
            "channel": rule.get("channel"),
            "recipient": rule.get("recipient", ""),
            "subject": rule.get("subject", "Data monitoring alert"),
            "message": rule.get("message", f"{metric} matched alert rule"),
        })

    persist_alert_rules(valid_rules)
    ALERT_RULES[:] = valid_rules
    return {"rules": ALERT_RULES, "count": len(ALERT_RULES)}


@app.delete("/api/alerts/rules")
def clear_alert_rules_endpoint():
    persist_alert_rules([])
    ALERT_RULES.clear()
    return {"rules": ALERT_RULES}


@app.get("/api/alerts/history")
def list_alert_history_endpoint(
    limit: int = Query(default=50, ge=1, le=500),
    status: str | None = None,
    event_type: str | None = None,
):
    return {"history": get_alert_history(limit=limit, status=status, event_type=event_type)}


@app.get("/api/datasets")
def list_datasets_endpoint(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return {"datasets": list_datasets(limit=limit, offset=offset, owner_id=current_user_scope())}


@app.post("/api/datasets", status_code=201)
def create_dataset_endpoint(payload: dict[str, Any]):
    try:
        frame = load_dataset(payload, owner_id=current_user_scope())
    except (DataSourceError, TypeError, ValueError) as error:
        _bad_request(str(error))
    _validate_dataset_dimensions(frame)
    name = _dataset_name(payload, default="")
    if not name:
        _bad_request("dataset name is required")
    dataset_id = save_dataset(
        name,
        _frame_records(frame),
        _dataset_source(payload),
        owner_id=current_user_id(),
    )
    return {"id": dataset_id, "name": name, "version_number": 1}


@app.post("/api/datasets/upload", status_code=201)
async def upload_dataset_file_endpoint(request: Request, name: str = Query(min_length=1, max_length=255)):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/csv":
        _bad_request("Uploaded datasets must use the text/csv content type")
    safe_name = Path(name.replace("\\", "/")).name
    if not safe_name or not safe_name.lower().endswith(".csv"):
        _bad_request("uploaded file name must end with .csv")
    content = await request.body()
    try:
        header = pd.read_csv(io.BytesIO(content), nrows=0)
        if len(header.columns) > MAX_DATASET_COLUMNS:
            _bad_request(f"Dataset must not exceed {MAX_DATASET_COLUMNS:,} columns")
        frame = pd.read_csv(io.BytesIO(content), nrows=MAX_DATASET_ROWS + 1)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        _bad_request(f"Invalid CSV content: {error}")
    if frame.empty:
        _bad_request("CSV dataset must not be empty")
    _validate_dataset_dimensions(frame)
    dataset_id = save_dataset(
        safe_name,
        _frame_records(frame),
        {"type": "upload", "file_type": "csv", "name": safe_name},
        owner_id=current_user_id(),
    )
    return {"id": dataset_id, "name": safe_name}


@app.post("/api/datasets/compare")
def compare_datasets_endpoint(payload: dict[str, Any]):
    baseline_id = payload.get("baseline_dataset_id")
    current_id = payload.get("current_dataset_id")
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        _bad_request("baseline_dataset_id is required")
    if not isinstance(current_id, str) or not current_id.strip():
        _bad_request("current_dataset_id is required")

    baseline = get_dataset(baseline_id, owner_id=current_user_scope())
    current = get_dataset(current_id, owner_id=current_user_scope())
    if baseline is None or current is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")
    if baseline.get("owner_id") != current.get("owner_id"):
        _bad_request("Datasets must have the same owner to be compared")

    baseline_frame = pd.DataFrame(baseline["data"])
    current_frame = pd.DataFrame(current["data"])
    baseline_columns = [str(column) for column in baseline_frame.columns]
    current_columns = [str(column) for column in current_frame.columns]
    shared_columns = [column for column in baseline_columns if column in current_columns]
    added_columns = [column for column in current_columns if column not in baseline_columns]
    removed_columns = [column for column in baseline_columns if column not in current_columns]
    type_changes = [
        {
            "column": column,
            "baseline": str(baseline_frame[column].dtype),
            "current": str(current_frame[column].dtype),
        }
        for column in shared_columns
        if str(baseline_frame[column].dtype) != str(current_frame[column].dtype)
    ]

    def quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
        missing_by_column = {
            str(column): int(count)
            for column, count in frame.isna().sum().items()
        }
        return {
            "rows": int(len(frame.index)),
            "columns": int(len(frame.columns)),
            "missing_values": {
                "total": sum(missing_by_column.values()),
                "by_column": missing_by_column,
            },
            "duplicate_rows": int(frame.duplicated().sum()),
            "quality_score": calculate_data_quality_score(frame)["overall_score"],
        }

    drift = None
    if shared_columns and not baseline_frame.empty and not current_frame.empty:
        drift = analyze_dataset_drift(
            baseline_frame[shared_columns],
            current_frame[shared_columns],
        )
    elif not shared_columns:
        drift = {"available": False, "reason": "No shared columns to compare"}
    else:
        drift = {"available": False, "reason": "Both datasets need at least one row"}

    comparison = {
        "baseline": {
            "id": baseline["id"],
            "name": baseline["name"],
            "version_number": baseline["version_number"],
        },
        "current": {
            "id": current["id"],
            "name": current["name"],
            "version_number": current["version_number"],
        },
        "schema": {
            "baseline_columns": baseline_columns,
            "current_columns": current_columns,
            "shared_columns": shared_columns,
            "added_columns": added_columns,
            "removed_columns": removed_columns,
            "type_changes": type_changes,
        },
        "quality": {
            "baseline": quality_summary(baseline_frame),
            "current": quality_summary(current_frame),
        },
        "distribution_drift": drift,
    }
    _, result_id = persist_monitoring_run(
        [
            {
                "key": "current",
                "id": current_id,
                "name": current["name"],
                "data": current["data"],
            },
            {
                "key": "baseline",
                "id": baseline_id,
                "name": baseline["name"],
                "data": baseline["data"],
            },
        ],
        "dataset_comparison",
        comparison,
        owner_id=current_user_scope(),
    )
    return {
        **comparison,
        "dataset_id": current_id,
        "baseline_dataset_id": baseline_id,
        "result_id": result_id,
    }


@app.post("/api/datasets/{dataset_id}/versions", status_code=201)
def create_dataset_version_endpoint(dataset_id: str, payload: dict[str, Any]):
    if "data" not in payload:
        _bad_request("data is required to create a dataset version")
    parent = get_dataset(dataset_id, owner_id=current_user_scope())
    if parent is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")
    try:
        frame = pd.DataFrame(payload["data"])
    except (TypeError, ValueError):
        _bad_request("Dataset payload must be list-of-records or column-oriented JSON")
    if frame.empty:
        _bad_request("Dataset must not be empty")
    _validate_dataset_dimensions(frame)

    change_summary = payload.get("change_summary")
    if change_summary is not None and (
        not isinstance(change_summary, str) or len(change_summary) > 1000
    ):
        _bad_request("change_summary must be a string of at most 1,000 characters")
    name = payload.get("name", parent["name"])
    if not isinstance(name, str) or not name.strip() or len(name) > 255:
        _bad_request("name must contain 1 to 255 characters")
    version_id = create_dataset_version(
        dataset_id,
        name.strip(),
        _frame_records(frame),
        change_summary,
        owner_id=current_user_scope(),
    )
    if version_id is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")
    version = get_dataset(version_id, owner_id=current_user_scope())
    return {
        "id": version_id,
        "name": version["name"],
        "parent_dataset_id": dataset_id,
        "version_number": version["version_number"],
        "change_summary": version["change_summary"],
    }


@app.get("/api/datasets/{dataset_id}/versions")
def list_dataset_versions_endpoint(dataset_id: str):
    versions = list_dataset_versions(dataset_id, owner_id=current_user_scope())
    if versions is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")
    return {"versions": versions}


@app.get("/api/datasets/{dataset_id}/lineage")
def get_dataset_lineage_endpoint(dataset_id: str):
    lineage = get_dataset_lineage(dataset_id, owner_id=current_user_scope())
    if lineage is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")
    return lineage


@app.get("/api/datasets/{dataset_id}")
def get_dataset_endpoint(dataset_id: str):
    dataset = get_dataset(dataset_id, owner_id=current_user_scope())
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")
    return dataset


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset_endpoint(dataset_id: str):
    if not delete_dataset(dataset_id, owner_id=current_user_scope()):
        raise HTTPException(status_code=404, detail="Dataset was not found")
    return {"deleted": True, "id": dataset_id}


@app.get("/api/monitoring/results")
def list_monitoring_results_endpoint(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    dataset_id: str | None = None,
):
    return {
        "results": list_monitoring_results(
            limit=limit,
            offset=offset,
            dataset_id=dataset_id,
            owner_id=current_user_scope(),
        )
    }


@app.get("/api/monitoring/results/{result_id}")
def get_monitoring_result_endpoint(result_id: str):
    result = get_monitoring_result(result_id, owner_id=current_user_scope())
    if result is None:
        raise HTTPException(status_code=404, detail="Monitoring result was not found")
    return result


@app.get("/api/schedules")
def list_monitoring_schedules_endpoint():
    return {
        "schedules": list_monitoring_schedules(owner_id=current_user_scope())
    }


@app.post("/api/schedules", status_code=201)
def create_monitoring_schedule_endpoint(payload: dict[str, Any]):
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        _bad_request("dataset_id is required")
    dataset = get_dataset(dataset_id, owner_id=current_user_scope())
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")

    try:
        interval_seconds = int(payload.get("interval_seconds", 3600))
    except (TypeError, ValueError):
        _bad_request("interval_seconds must be an integer")
    if not 60 <= interval_seconds <= 31 * 24 * 60 * 60:
        _bad_request("interval_seconds must be between 60 and 2,678,400")

    checks = payload.get("checks", ["quality"])
    allowed_checks = {"quality", "drift", "anomalies", "forecast", "explain"}
    if (
        not isinstance(checks, list)
        or not checks
        or not all(isinstance(check, str) and check in allowed_checks for check in checks)
    ):
        _bad_request(
            "checks must be a non-empty list containing quality, drift, anomalies, forecast, and explain"
        )
    checks = list(dict.fromkeys(checks))

    baseline_id = payload.get("baseline_dataset_id")
    if "drift" in checks:
        if not isinstance(baseline_id, str) or not baseline_id.strip():
            _bad_request("baseline_dataset_id is required when drift is selected")
        baseline_dataset = get_dataset(baseline_id, owner_id=current_user_scope())
        if baseline_dataset is None:
            raise HTTPException(status_code=404, detail="Baseline dataset was not found")
        if baseline_dataset.get("owner_id") != dataset.get("owner_id"):
            _bad_request("Current and baseline datasets must have the same owner")
    elif baseline_id is not None:
        _bad_request("baseline_dataset_id can only be set when drift is selected")

    columns = {str(column) for column in pd.DataFrame(dataset["data"]).columns}
    timestamp_column = payload.get("timestamp_column")
    max_age_hours = payload.get("max_age_hours")
    if (timestamp_column is None) != (max_age_hours is None):
        _bad_request("timestamp_column and max_age_hours must be provided together")
    if timestamp_column is not None:
        if not isinstance(timestamp_column, str) or timestamp_column not in columns:
            _bad_request("timestamp_column must name a column in the selected dataset")
        try:
            max_age_hours = float(max_age_hours)
        except (TypeError, ValueError):
            _bad_request("max_age_hours must be a positive number")
        if not math.isfinite(max_age_hours) or max_age_hours <= 0:
            _bad_request("max_age_hours must be a positive number")
        if "quality" not in checks:
            checks.append("quality")

    quality_options = _quality_options(payload)
    quality_options["timestamp_column"] = timestamp_column
    quality_options["max_age_hours"] = max_age_hours

    schedule: dict[str, Any] = {
        "checks": checks,
        "quality_options": quality_options,
        "run_owner_id": dataset.get("owner_id"),
    }
    if baseline_id is not None:
        schedule["baseline_dataset_id"] = baseline_id
    for key in ("date_column", "value_column", "metric", "target_column"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, str) or value not in columns:
                _bad_request(f"{key} must name a column in the selected dataset")
            schedule[key] = value
    if "forecast" in checks and not schedule.get("date_column"):
        _bad_request("date_column is required when forecast is selected")
    if "explain" in checks and not schedule.get("target_column"):
        _bad_request("target_column is required when explain is selected")
    try:
        periods = int(payload.get("periods", 4))
        test_size = float(payload.get("test_size", 0.2))
    except (TypeError, ValueError):
        _bad_request("periods and test_size must be valid numbers")
    frequency = payload.get("frequency", "W")
    method = payload.get("method", "auto")
    task = payload.get("task", "classification")
    if not 1 <= periods <= 365:
        _bad_request("periods must be between 1 and 365")
    if not isinstance(frequency, str) or not frequency.strip():
        _bad_request("frequency must be a non-empty string")
    try:
        pd.tseries.frequencies.to_offset(frequency)
    except ValueError:
        _bad_request("frequency is not a supported pandas offset")
    if method not in {"auto", "arima", "linear", "prophet", "lstm"}:
        _bad_request("method must be auto, arima, linear, prophet, or lstm")
    if task not in {"classification", "regression"}:
        _bad_request("task must be classification or regression")
    if not math.isfinite(test_size) or not 0.1 <= test_size <= 0.5:
        _bad_request("test_size must be between 0.1 and 0.5")
    schedule.update({
        "periods": periods,
        "frequency": frequency,
        "method": method,
        "task": task,
        "test_size": test_size,
    })

    channels = _notification_channels(payload)
    if not isinstance(channels, list):
        _bad_request("alert_channels must be a list of notification channels")
    validated_channels = []
    for channel in channels:
        if isinstance(channel, str):
            channel_name = channel.strip().lower()
            if channel_name not in SUPPORTED_CHANNELS:
                _bad_request(f"unsupported alert channel: {channel_name}")
            validated_channels.append(channel_name)
        elif isinstance(channel, dict):
            channel_name = str(channel.get("channel", "")).strip().lower()
            recipient = channel.get("recipient", "")
            if channel_name not in SUPPORTED_CHANNELS or not isinstance(recipient, str):
                _bad_request("alert_channels entries must contain a supported channel and string recipient")
            validated_channels.append({"channel": channel_name, "recipient": recipient})
        else:
            _bad_request("alert_channels entries must be channel names or objects")
    schedule["alert_channels"] = validated_channels

    saved = create_monitoring_schedule(
        current_user_id(),
        dataset_id,
        schedule,
        interval_seconds,
    )
    return saved


@app.patch("/api/schedules/{schedule_id}")
def update_monitoring_schedule_endpoint(schedule_id: str, payload: dict[str, Any]):
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        _bad_request("enabled must be a boolean")
    if not set_monitoring_schedule_enabled(
        schedule_id, enabled, owner_id=current_user_scope()
    ):
        raise HTTPException(status_code=404, detail="Monitoring schedule was not found")
    return {"id": schedule_id, "enabled": enabled}


@app.delete("/api/schedules/{schedule_id}")
def delete_monitoring_schedule_endpoint(schedule_id: str):
    if not delete_monitoring_schedule(
        schedule_id, owner_id=current_user_scope()
    ):
        raise HTTPException(status_code=404, detail="Monitoring schedule was not found")
    return {"deleted": True, "id": schedule_id}


@app.get("/api/configurations")
def get_configurations_endpoint():
    return {"configurations": get_configurations()}


@app.put("/api/configurations")
def update_configurations_endpoint(payload: dict[str, Any]):
    configurations = payload.get("configurations")
    if configurations is not None:
        if not isinstance(configurations, dict):
            _bad_request("configurations must be an object")
        if not all(isinstance(key, str) and key.strip() for key in configurations):
            _bad_request("configuration keys must be non-empty strings")
        set_configurations({
            key.strip(): value for key, value in configurations.items()
        })
    elif isinstance(payload.get("key"), str) and payload["key"].strip() and "value" in payload:
        set_configuration(payload["key"].strip(), payload["value"])
    else:
        _bad_request("provide a key/value pair or a configurations object")
    return {"configurations": get_configurations()}


@app.delete("/api/configurations/{key}")
def delete_configuration_endpoint(key: str):
    if not delete_configuration(key):
        raise HTTPException(status_code=404, detail="Configuration was not found")
    return {"deleted": True, "key": key}


@app.get("/api/audit")
def get_audit_records_endpoint(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    return {"records": get_audit_records(limit=limit, offset=offset)}


@app.post("/api/data-quality/analyze")
def analyze_data_quality_endpoint(payload: dict[str, Any]):
    dataset = payload.get("data") or payload.get("dataset") or payload.get("records")
    if dataset is None:
        _bad_request("Dataset payload is required.")

    try:
        df = pd.DataFrame(dataset)
    except Exception:
        _bad_request("Dataset payload must be list-of-records or column-oriented JSON.")

    if df.empty:
        _bad_request("Dataset must not be empty.")

    report = check_data_quality(df, **_quality_options(payload))
    quality_score = calculate_data_quality_score(df)
    ai_summary = generate_ai_quality_summary(
        df, report, quality_score, use_llm=_use_llm_quality_summary(payload)
    )
    notification_result = None
    if _should_send_notifications(payload):
        risk_level = ai_summary.get("risk_level", "unknown")
        summary_text = ai_summary.get("summary", "Data quality review completed.")
        notification_result = _trigger_monitoring_notifications(
            payload,
            "data_quality",
            f"Data quality alert: score={quality_score.get('overall_score', 0)}; risk={risk_level}; details={summary_text}",
            "Data Monitoring Alert",
            severity=risk_level,
        )
    rule_result = _send_rule_based_notifications(
        payload,
        "data_quality",
        {
            "quality_score": quality_score.get("overall_score", 0),
            "risk_level": ai_summary.get("risk_level", "unknown"),
            "severity": ai_summary.get("risk_level", "unknown"),
            "total_nulls": report.get("total_nulls", 0),
            "duplicates": report.get("duplicates", 0),
        },
        "Data Monitoring Alert",
    )
    if rule_result.get("status") != "skipped":
        notification_result = rule_result
    response = {
        "report": report,
        "quality_score": quality_score,
        "ai_summary": ai_summary,
        "notifications": notification_result or {"status": "skipped", "reason": "notifications not requested"},
        "message": "Data quality analysis completed successfully.",
    }
    response.update(_persist_endpoint_result(payload, df, "data_quality", response))
    return response


@app.post("/api/data-quality/analyze-csv")
def analyze_data_quality_csv_endpoint(payload: dict[str, Any]):
    csv_content = payload.get("csv") or payload.get("dataset_csv")
    if csv_content is None:
        _bad_request("CSV payload is required.")

    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_content))
    except Exception:
        _bad_request("Invalid CSV content.")

    if df.empty:
        _bad_request("CSV dataset must not be empty.")

    report = check_data_quality(df, **_quality_options(payload))
    quality_score = calculate_data_quality_score(df)
    ai_summary = generate_ai_quality_summary(
        df, report, quality_score, use_llm=_use_llm_quality_summary(payload)
    )
    notification_result = None
    if _should_send_notifications(payload):
        risk_level = ai_summary.get("risk_level", "unknown")
        summary_text = ai_summary.get("summary", "Data quality review completed.")
        notification_result = _trigger_monitoring_notifications(
            payload,
            "data_quality_csv",
            f"CSV data quality alert: score={quality_score.get('overall_score', 0)}; risk={risk_level}; details={summary_text}",
            "CSV Data Monitoring Alert",
            severity=risk_level,
        )
    rule_result = _send_rule_based_notifications(
        payload,
        "data_quality_csv",
        {
            "quality_score": quality_score.get("overall_score", 0),
            "risk_level": ai_summary.get("risk_level", "unknown"),
            "severity": ai_summary.get("risk_level", "unknown"),
            "total_nulls": report.get("total_nulls", 0),
            "duplicates": report.get("duplicates", 0),
        },
        "CSV Data Monitoring Alert",
    )
    if rule_result.get("status") != "skipped":
        notification_result = rule_result
    response = {
        "report": report,
        "quality_score": quality_score,
        "ai_summary": ai_summary,
        "notifications": notification_result or {"status": "skipped", "reason": "notifications not requested"},
    }
    response.update(_persist_endpoint_result(payload, df, "data_quality_csv", response))
    return response


@app.post("/api/drift/analyze")
def analyze_drift_endpoint(payload: dict[str, Any]):
    baseline_data = payload.get("baseline") or payload.get("baseline_dataset")
    current_data = payload.get("current") or payload.get("current_dataset")

    if baseline_data is None or current_data is None:
        _bad_request("Both baseline and current datasets are required.")

    try:
        baseline_df = pd.DataFrame(baseline_data)
        current_df = pd.DataFrame(current_data)
    except Exception:
        _bad_request("Dataset payloads must be list-of-records or column-oriented JSON.")

    if baseline_df.empty or current_df.empty:
        _bad_request("Baseline and current datasets must not be empty.")

    report = analyze_dataset_drift(baseline_df, current_df)
    notification_result = None
    if _should_send_notifications(payload):
        drift_score = report.get("overall_drift_score", 0)
        message = (
            f"Drift alert: overall drift score={drift_score}; "
            f"severity={report.get('overall_severity', 'low')}; "
            f"detected={report.get('drift_detected', False)}"
        )
        notification_result = _trigger_monitoring_notifications(
            payload,
            "data_drift",
            message,
            "Drift Detection Alert",
            severity=report.get("overall_severity", "low"),
        )
    rule_result = _send_rule_based_notifications(
        payload,
        "data_drift",
        {
            "drift_score": report.get("overall_drift_score", 0),
            "severity": report.get("overall_severity", "low"),
            "drift_detected": report.get("drift_detected", False),
        },
        "Drift Detection Alert",
    )
    if rule_result.get("status") != "skipped":
        notification_result = rule_result
    response = {
        "report": report,
        "notifications": notification_result or {"status": "skipped", "reason": "notifications not requested"},
        "message": "Drift analysis completed successfully.",
    }
    response.update(
        _persist_endpoint_result(
            payload,
            current_df,
            "data_drift",
            response,
            baseline=baseline_df,
        )
    )
    return response


@app.post("/api/drift/analyze-csv")
def analyze_drift_csv_endpoint(payload: dict[str, Any]):
    baseline_csv = payload.get("baseline_csv")
    current_csv = payload.get("current_csv")

    if baseline_csv is None or current_csv is None:
        _bad_request("CSV payloads are required for both datasets.")

    try:
        baseline_df = pd.read_csv(pd.io.common.StringIO(baseline_csv))
        current_df = pd.read_csv(pd.io.common.StringIO(current_csv))
    except Exception:
        _bad_request("Invalid CSV content for one or both datasets.")

    if baseline_df.empty or current_df.empty:
        _bad_request("Baseline and current CSV datasets must not be empty.")

    try:
        report = analyze_dataset_drift(baseline_df, current_df)
    except (TypeError, ValueError) as error:
        _bad_request(str(error))

    notification_result = None
    if _should_send_notifications(payload):
        drift_score = report.get("overall_drift_score", 0)
        message = (
            f"CSV drift alert: overall drift score={drift_score}; "
            f"severity={report.get('overall_severity', 'low')}; "
            f"detected={report.get('drift_detected', False)}"
        )
        notification_result = _trigger_monitoring_notifications(
            payload,
            "data_drift_csv",
            message,
            "CSV Drift Detection Alert",
            severity=report.get("overall_severity", "low"),
        )
    rule_result = _send_rule_based_notifications(
        payload,
        "data_drift_csv",
        {
            "drift_score": report.get("overall_drift_score", 0),
            "severity": report.get("overall_severity", "low"),
            "drift_detected": report.get("drift_detected", False),
        },
        "CSV Drift Detection Alert",
    )
    if rule_result.get("status") != "skipped":
        notification_result = rule_result
    response = {
        "report": report,
        "notifications": notification_result or {"status": "skipped", "reason": "notifications not requested"},
    }
    response.update(
        _persist_endpoint_result(
            payload,
            current_df,
            "data_drift_csv",
            response,
            baseline=baseline_df,
        )
    )
    return response


@app.post("/api/analytics/trends")
def analyze_trends_endpoint(payload: dict[str, Any]):
    dataset = payload.get("data") or payload.get("dataset") or payload.get("records")
    date_column = payload.get("date_column")
    if dataset is None or not date_column:
        _bad_request("Dataset and date_column are required.")
    try:
        frame = pd.DataFrame(dataset)
        report = analyze_trends(frame, date_column, payload.get("value_column"))
    except (TypeError, ValueError) as error:
        _bad_request(str(error))
    response = {"report": report}
    response.update(_persist_endpoint_result(payload, frame, "trends", response))
    return response


@app.post("/api/analytics/forecast")
def forecast_endpoint(payload: dict[str, Any]):
    dataset = payload.get("data") or payload.get("dataset") or payload.get("records")
    date_column = payload.get("date_column")
    if dataset is None or not date_column:
        _bad_request("Dataset and date_column are required.")
    try:
        df = pd.DataFrame(dataset)
        periods = int(payload.get("periods", 4))
        if periods < 1 or periods > 365:
            _bad_request("periods must be between 1 and 365.")

        method = str(payload.get("method", "auto")).lower()
        allowed_methods = {"auto", "arima", "linear", "prophet", "lstm"}
        if method not in allowed_methods:
            _bad_request("method must be one of: auto, arima, linear, prophet, lstm.")

        frequency = str(payload.get("frequency", "W"))
        if not frequency:
            _bad_request("frequency is required.")

        if payload.get("metric") == "data_health":
            report = forecast_data_health(df, date_column, periods, frequency)
        else:
            report = forecast_metric(df, date_column, payload.get("value_column"), periods, frequency, method)
    except (TypeError, ValueError) as error:
        _bad_request(str(error))
    response = {"report": report}
    response.update(_persist_endpoint_result(payload, df, "forecast", response))
    return response


@app.post("/api/analytics/explain")
def explain_model_endpoint(payload: dict[str, Any]):
    dataset = payload.get("data") or payload.get("dataset") or payload.get("records")
    target_column = payload.get("target_column")
    if dataset is None or not target_column:
        _bad_request("Dataset and target_column are required.")
    try:
        frame = pd.DataFrame(dataset)
        report = train_and_explain_model(
            frame,
            target_column,
            payload.get("task", "classification"),
            payload.get("test_size", 0.2),
        )
    except (TypeError, ValueError) as error:
        _bad_request(str(error))
    response = {"report": report}
    response.update(_persist_endpoint_result(payload, frame, "explanation", response))
    return response
