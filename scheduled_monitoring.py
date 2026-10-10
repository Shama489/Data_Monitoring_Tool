"""Persistent recurring monitoring job runner."""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents.monitoring_agent import run_monitoring
from monitoring_store import (
    claim_due_monitoring_schedules,
    create_monitoring_report,
    finish_monitoring_schedule,
    record_lineage_event,
)
from reporting import generate_monitoring_reports

logger = logging.getLogger(__name__)


def _execute_schedule(item: dict[str, Any]) -> None:
    config: dict[str, Any] = item["schedule"]
    payload: dict[str, Any] = {
        "dataset_id": item["dataset_id"],
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
        try:
            record_lineage_event(
                item["dataset_id"],
                "monitoring_run",
                "scheduled_monitoring",
                "failed",
                input_dataset_ids=[item["dataset_id"]],
                details={"schedule_id": item["id"]},
                error=f"{type(error).__name__}: {error}"[:500],
            )
        except ValueError:
            logger.warning(
                "Could not record lineage for failed schedule %s because its dataset is unavailable",
                item["id"],
            )
    else:
        finish_monitoring_schedule(item["id"], succeeded=True)


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
