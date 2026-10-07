"""Persistent recurring monitoring job runner."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.monitoring_agent import run_monitoring
from monitoring_store import (
    claim_due_monitoring_schedules,
    finish_monitoring_schedule,
    record_lineage_event,
)

logger = logging.getLogger(__name__)


def run_due_schedules(now: float | None = None, limit: int = 20) -> int:
    """Claim and execute schedules that are due; return the number claimed."""
    schedules = claim_due_monitoring_schedules(now=now, limit=limit)
    for item in schedules:
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
            run_monitoring(
                payload,
                quality_options=config.get("quality_options", {}),
                use_llm=False,
                owner_id=config.get("run_owner_id", item["owner_id"]),
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
    return len(schedules)


async def scheduler_loop(poll_interval_seconds: int = 15) -> None:
    """Poll durable schedule storage and execute due work off the event loop."""
    while True:
        try:
            await asyncio.to_thread(run_due_schedules)
        except Exception:
            logger.exception("Scheduled monitoring poll failed")
        await asyncio.sleep(poll_interval_seconds)
