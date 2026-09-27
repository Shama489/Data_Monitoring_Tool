"""Notification delivery for monitoring alerts.

All credentials and endpoints are read from environment variables. Providers are
kept dependency-free so the API can run without optional SDKs.
"""

import base64
import json
import os
import smtplib
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage
from typing import Any

SUPPORTED_CHANNELS = {"email", "sms", "whatsapp", "slack", "teams"}
SUPPORTED_OPERATORS = {">", ">=", "<", "<=", "==", "!=", "contains", "in", "not_in"}


class NotificationError(RuntimeError):
    """Raised when a notification cannot be delivered."""


def _compare_values(actual: Any, operator: str, expected: Any) -> bool:
    if actual is None:
        return False

    operator = str(operator).lower().strip()
    if operator in {">", ">=", "<", "<=", "==", "!="}:
        try:
            comparable_actual = float(actual)
            comparable_expected = float(expected)
        except (TypeError, ValueError):
            if operator == "==":
                return str(actual) == str(expected)
            if operator == "!=":
                return str(actual) != str(expected)
            return False

        return {
            ">": comparable_actual > comparable_expected,
            ">=": comparable_actual >= comparable_expected,
            "<": comparable_actual < comparable_expected,
            "<=": comparable_actual <= comparable_expected,
            "==": comparable_actual == comparable_expected,
            "!=": comparable_actual != comparable_expected,
        }[operator]

    if operator == "contains":
        return str(expected).lower() in str(actual).lower()
    if operator == "in":
        if isinstance(expected, (list, tuple, set)):
            return actual in expected
        return str(actual) in str(expected)
    if operator == "not_in":
        if isinstance(expected, (list, tuple, set)):
            return actual not in expected
        return str(actual) not in str(expected)
    return False


def evaluate_alert_rules(metrics: dict[str, Any], rules: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Return matched rules for monitoring metrics with threshold comparisons."""
    if not isinstance(rules, list):
        return {"triggered": False, "matches": [], "total_checked": 0}

    matches: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        metric = str(rule.get("metric", "")).strip()
        if not metric:
            continue
        operator = str(rule.get("operator", "")).strip()
        if not operator:
            operator = "=="
        if operator not in SUPPORTED_OPERATORS and operator not in {">", ">=", "<", "<=", "==", "!="}:
            continue

        actual_value = metrics.get(metric)
        if actual_value is None and metric not in metrics:
            continue

        matched = _compare_values(actual_value, operator, rule.get("value"))
        if matched:
            matches.append(
                {
                    "metric": metric,
                    "actual": actual_value,
                    "operator": operator,
                    "expected": rule.get("value"),
                    "channel": rule.get("channel"),
                    "recipient": rule.get("recipient", ""),
                    "subject": rule.get("subject", "Data monitoring alert"),
                    "message": rule.get("message", f"{metric} matched alert rule"),
                }
            )

    return {"triggered": bool(matches), "matches": matches, "total_checked": len(matches)}


def _required_env(*names: str) -> dict[str, str]:
    values = {name: os.getenv(name, "") for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise NotificationError(f"Missing notification configuration: {', '.join(missing)}")
    return values


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            return {"status_code": response.status, "response": body[:500]}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise NotificationError(f"Notification provider returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise NotificationError(f"Notification provider request failed: {error.reason}") from error


def send_email(recipient: str, subject: str, message: str) -> dict[str, Any]:
    config = _required_env("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "ALERT_FROM_EMAIL")
    email = EmailMessage()
    email["From"] = config["ALERT_FROM_EMAIL"]
    email["To"] = recipient
    email["Subject"] = subject
    email.set_content(message)
    with smtplib.SMTP(config["SMTP_HOST"], int(config["SMTP_PORT"]), timeout=15) as server:
        server.starttls()
        server.login(config["SMTP_USERNAME"], config["SMTP_PASSWORD"])
        server.send_message(email)
    return {"channel": "email", "recipient": recipient, "status": "sent"}


def send_twilio(recipient: str, message: str, channel: str) -> dict[str, Any]:
    config = _required_env("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")
    from_env = "TWILIO_WHATSAPP_FROM" if channel == "whatsapp" else "TWILIO_SMS_FROM"
    config.update(_required_env(from_env))
    sender = config[from_env]
    destination = recipient if channel == "sms" else f"whatsapp:{recipient}"
    if channel == "whatsapp" and not sender.startswith("whatsapp:"):
        sender = f"whatsapp:{sender}"
    form = urllib.parse.urlencode({"From": sender, "To": destination, "Body": message}).encode("utf-8")
    token = base64.b64encode(f"{config['TWILIO_ACCOUNT_SID']}:{config['TWILIO_AUTH_TOKEN']}".encode()).decode()
    request = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{config['TWILIO_ACCOUNT_SID']}/Messages.json",
        data=form,
        headers={"Authorization": f"Basic {token}", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return {"channel": channel, "recipient": recipient, "status": "sent", "status_code": response.status}
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise NotificationError(f"Twilio {channel} request failed: {error}") from error


def send_webhook(recipient: str, message: str, channel: str) -> dict[str, Any]:
    env_name = "SLACK_WEBHOOK_URL" if channel == "slack" else "TEAMS_WEBHOOK_URL"
    url = recipient or os.getenv(env_name, "")
    if not url:
        raise NotificationError(f"Provide recipient or configure {env_name}")
    result = _post_json(url, {"text": message})
    return {"channel": channel, "status": "sent", **result}


def send_notification(channel: str, recipient: str, message: str, subject: str = "Data monitoring alert", dry_run: bool = False) -> dict[str, Any]:
    channel = channel.lower().strip()
    if channel not in SUPPORTED_CHANNELS:
        raise NotificationError(f"Unsupported channel: {channel}")
    if not message.strip():
        raise NotificationError("message must not be empty")
    if dry_run:
        return {"channel": channel, "recipient": recipient, "status": "dry_run"}
    if channel == "email":
        return send_email(recipient, subject, message)
    if channel in {"sms", "whatsapp"}:
        return send_twilio(recipient, message, channel)
    return send_webhook(recipient, message, channel)


def send_notifications(alert: dict[str, Any]) -> dict[str, Any]:
    channels = alert.get("channels", [])
    if not isinstance(channels, list) or not channels:
        raise NotificationError("channels must be a non-empty list")
    message = str(alert.get("message", ""))
    subject = str(alert.get("subject", "Data monitoring alert"))
    dry_run = bool(alert.get("dry_run", False))
    results = []
    failures = []
    for item in channels:
        if isinstance(item, str):
            channel, recipient = item, ""
        elif isinstance(item, dict):
            channel, recipient = str(item.get("channel", "")), str(item.get("recipient", ""))
        else:
            failures.append({"channel": "unknown", "error": "Each channel must be a string or object"})
            continue
        try:
            results.append(send_notification(channel, recipient, message, subject, dry_run))
        except (NotificationError, ValueError, TypeError) as error:
            failures.append({"channel": channel, "error": str(error)})
    return {"sent": results, "failed": failures, "success": not failures}
