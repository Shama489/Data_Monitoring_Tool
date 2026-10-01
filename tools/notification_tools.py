from typing import Any

from notifications import NotificationError, send_notifications


def send_monitoring_notification(
    message: str,
    channels: list[Any],
    severity: str = "warning",
) -> dict[str, Any]:
    if not channels:
        return {"status": "skipped", "reason": "no notification channels configured"}
    try:
        return send_notifications({
            "message": message,
            "subject": "Data Monitoring Incident",
            "severity": severity,
            "event_type": "monitoring_workflow",
            "channels": channels,
        })
    except NotificationError as error:
        return {"status": "failed", "error": str(error)}