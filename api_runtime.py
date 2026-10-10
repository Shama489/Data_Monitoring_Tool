"""Small process-local telemetry and bounded cache for non-sensitive metadata."""

from __future__ import annotations

import threading
import time
from collections import Counter
from threading import RLock
from typing import Any

from cachetools import TTLCache, cached

_metrics_lock = threading.Lock()
_request_counts: Counter[str] = Counter()
_request_errors = 0
_request_duration_seconds = 0.0
_request_duration_max_seconds = 0.0
_started_at = time.time()


def record_request(method: str, route: str, status_code: int, duration: float) -> None:
    global _request_errors, _request_duration_seconds, _request_duration_max_seconds
    key = f"{method} {route} {status_code}"
    with _metrics_lock:
        _request_counts[key] += 1
        _request_duration_seconds += duration
        _request_duration_max_seconds = max(_request_duration_max_seconds, duration)
        if status_code >= 500:
            _request_errors += 1


def metrics_snapshot() -> dict[str, Any]:
    with _metrics_lock:
        counts = dict(sorted(_request_counts.items()))
        errors = _request_errors
        duration_total = _request_duration_seconds
        duration_max = _request_duration_max_seconds
    return {
        "uptime_seconds": round(time.time() - _started_at, 3),
        "requests_total": sum(counts.values()),
        "requests_by_method_route_status": counts,
        "server_errors_total": errors,
        "request_duration_seconds_total": round(duration_total, 6),
        "request_duration_seconds_max": round(duration_max, 6),
    }


@cached(cache=TTLCache(maxsize=1, ttl=60), lock=RLock())
def cached_source_capabilities() -> dict[str, Any]:
    from data_sources import source_capabilities

    return source_capabilities()


@cached(cache=TTLCache(maxsize=1, ttl=60), lock=RLock())
def cached_monitoring_tools() -> dict[str, Any]:
    from agents.monitoring_agent import build_tool_registry

    return {"tools": build_tool_registry().describe()}
