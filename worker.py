"""Run the durable monitoring scheduler as a separate worker process."""

from __future__ import annotations

import asyncio
import logging
import os

from scheduled_monitoring import scheduler_loop


async def main() -> None:
    interval = int(os.getenv("MONITORING_POLL_INTERVAL_SECONDS", "15"))
    if interval < 1:
        raise ValueError("MONITORING_POLL_INTERVAL_SECONDS must be at least 1.")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    await scheduler_loop(poll_interval_seconds=interval)


if __name__ == "__main__":
    asyncio.run(main())
