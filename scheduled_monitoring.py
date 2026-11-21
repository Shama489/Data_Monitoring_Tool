"""Persistent recurring monitoring job runner."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from agents.monitoring_agent import run_monitoring
from data_sources import (
    DataSourceError,
    filter_incremental_records,
    load_source,
)
from monitoring_store import (
    claim_due_monitoring_schedules,
    create_monitoring_schedule_dataset_version,
    create_monitoring_report,
    finish_monitoring_schedule,
    get_dataset,
    record_lineage_event,
    record_pipeline_run,
)
from reporting import generate_monitoring_reports

logger = logging.getLogger(__name__)


def _execute_schedule(item: dict[str, Any]) -> None:
    config: dict[str, Any] = item["schedule"]
    dataset_id = item["dataset_id"]
    started_at = time.time()
    run_id = f"{item['id']}:{uuid.uuid4().hex}"
    records_processed = None
    record_pipeline_run(
        run_id,
        "scheduled_monitoring",
        item["id"],
        "running",
        item["owner_id"],
        started_at=started_at,
        details={"schedule_id": item["id"]},
    )
    source = config.get("source")
    if isinstance(source, dict):
        try:
            incremental = config.get("incremental")
            source_for_load = source
            fetch_parameter = (
                incremental.get("fetch_parameter")
                if isinstance(incremental, dict)
                else None
            )
            if fetch_parameter is not None:
                source_type = str(source.get("type", "")).lower().strip()
                if source_type not in {
                    "sql", "postgresql", "mysql", "http_json", "rest_json",
                }:
                    raise DataSourceError(
                        "fetch_parameter is supported only for SQL and HTTP JSON sources"
                    )
                params = source.get("params", {})
                if not isinstance(params, dict):
                    raise DataSourceError(
                        "Source params must be an object when fetch_parameter is configured"
                    )
                source_for_load = {
                    **source,
                    "params": {
                        **params,
                        fetch_parameter: config.get("checkpoint_value"),
                    },
                }
            frame = load_source(source_for_load)
            if frame.empty and not fetch_parameter:
                raise DataSourceError("Source dataset must not be empty")
            checkpoint_value = None
            if incremental:
                checkpoint = config.get("checkpoint_value")
                if checkpoint is None:
                    raise DataSourceError("Incremental source checkpoint is unavailable")
                if frame.empty:
                    new_records = frame
                    checkpoint_value = checkpoint
                else:
                    new_records, checkpoint_value = filter_incremental_records(
                        frame,
                        incremental,
                        checkpoint,
                    )
                records_processed = len(new_records.index)
                latest_dataset_id = config.get(
                    "latest_dataset_id", item["dataset_id"]
                )
                if new_records.empty:
                    dataset_id = latest_dataset_id
                else:
                    latest_dataset = get_dataset(
                        latest_dataset_id,
                        owner_id=item["owner_id"],
                    )
                    if latest_dataset is None:
                        raise DataSourceError(
                            "Latest incremental dataset version was not found"
                        )
                    combined = pd.concat(
                        [
                            pd.DataFrame(latest_dataset["data"]),
                            new_records,
                        ],
                        ignore_index=True,
                    ).drop_duplicates(
                        subset=[incremental["key_column"]],
                        keep="last",
                    )
                    frame = combined
            else:
                records_processed = len(frame.index)
            source_metadata = {
                key: source[key]
                for key in (
                    "type",
                    "file_type",
                    "name",
                    "bucket",
                    "key",
                    "database",
                    "collection",
                    "file_id",
                )
                if source.get(key) is not None
            }
            if not incremental or records_processed:
                dataset_id = create_monitoring_schedule_dataset_version(
                    item["id"],
                    config.get("dataset_name", "monitoring_dataset"),
                    json.loads(frame.to_json(orient="records", date_format="iso")),
                    source_metadata,
                    "Scheduled source refresh",
                    checkpoint_value=checkpoint_value,
                )
        except Exception as error:
            logger.error(
                "Scheduled source refresh failed for %s (%s)",
                item["id"],
                type(error).__name__,
            )
            safe_error = f"{type(error).__name__}: source refresh failed"
            finish_monitoring_schedule(
                item["id"],
                succeeded=False,
                error=safe_error,
            )
            record_pipeline_run(
                run_id,
                "scheduled_monitoring",
                item["id"],
                "failed",
                item["owner_id"],
                finished_at=time.time(),
                details={
                    "schedule_id": item["id"],
                    "stage": "source_refresh",
                    "error": safe_error,
                },
            )
            try:
                record_lineage_event(
                    item["dataset_id"],
                    "source_refresh",
                    "scheduled_ingestion",
                    "failed",
                    input_dataset_ids=[item["dataset_id"]],
                    details={"schedule_id": item["id"]},
                    error=safe_error[:500],
                    owner_id=config.get("run_owner_id", item["owner_id"]),
                )
            except ValueError:
                logger.warning(
                    "Could not record lineage for failed source schedule %s",
                    item["id"],
                )
            return

    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "checks": config["checks"],
        "schedule_id": item["id"],
        "scheduled": True,
        "notify": bool(config.get("alert_channels")),
        "channels": config.get("alert_channels", []),
    }
    for key in (
        "baseline_dataset_id",
        "date_column",
        "value_column",
        "periods",
        "frequency",
        "method",
        "metric",
        "target_column",
        "task",
        "test_size",
    ):
        if key in config:
            payload[key] = config[key]

    try:
        result = run_monitoring(
            payload,
            quality_options=config.get("quality_options", {}),
            use_llm=False,
            owner_id=config.get("run_owner_id", item["owner_id"]),
        )
        report_formats = config.get("report_formats", [])
        if report_formats:
            artifacts = generate_monitoring_reports(
                result,
                config.get("dataset_name", item["dataset_id"]),
                report_formats,
            )
            create_monitoring_report(
                item["owner_id"],
                result["result_id"],
                artifacts,
                schedule_id=item["id"],
                result_owner_scope=config.get("run_owner_id", item["owner_id"]),
            )
    except Exception as error:
        logger.exception("Scheduled monitoring run failed for %s", item["id"])
        finish_monitoring_schedule(
            item["id"],
            succeeded=False,
            error=f"{type(error).__name__}: {error}",
        )
        record_pipeline_run(
            run_id,
            "scheduled_monitoring",
            item["id"],
            "failed",
            item["owner_id"],
            finished_at=time.time(),
            records_processed=records_processed,
            details={
                "schedule_id": item["id"],
                "stage": "monitoring",
                "error": f"{type(error).__name__}: {error}"[:500],
            },
        )
        try:
            record_lineage_event(
                dataset_id,
                "monitoring_run",
                "scheduled_monitoring",
                "failed",
                input_dataset_ids=[dataset_id],
                details={"schedule_id": item["id"]},
                error=f"{type(error).__name__}: {error}"[:500],
                owner_id=config.get("run_owner_id", item["owner_id"]),
            )
        except ValueError:
            logger.warning(
                "Could not record lineage for failed schedule %s because its dataset is unavailable",
                item["id"],
            )
    else:
        finish_monitoring_schedule(item["id"], succeeded=True)
        record_pipeline_run(
            run_id,
            "scheduled_monitoring",
            item["id"],
            "succeeded",
            item["owner_id"],
            finished_at=time.time(),
            records_processed=records_processed,
            details={"schedule_id": item["id"]},
        )


def run_due_schedules(now: float | None = None, limit: int = 20) -> int:
    """Claim and execute due work with a bounded worker pool."""
    schedules = claim_due_monitoring_schedules(now=now, limit=limit)
    if not schedules:
        return 0
    try:
        worker_count = max(1, min(int(os.getenv("MONITORING_WORKER_CONCURRENCY", "2")), 16))
    except ValueError as error:
        raise ValueError("MONITORING_WORKER_CONCURRENCY must be an integer.") from error
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="monitoring-job") as pool:
        list(pool.map(_execute_schedule, schedules))
    return len(schedules)


async def scheduler_loop(poll_interval_seconds: int = 15) -> None:
    """Poll durable schedule storage and execute due work off the event loop."""
    while True:
        try:
            await asyncio.to_thread(run_due_schedules)
        except Exception:
            logger.exception("Scheduled monitoring poll failed")
        await asyncio.sleep(poll_interval_seconds)
