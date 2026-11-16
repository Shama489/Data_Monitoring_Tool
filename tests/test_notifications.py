import base64
import json
import smtplib
from urllib.parse import parse_qs

import pytest
from fastapi.testclient import TestClient

import notifications
from conftest import admin_headers
from main import app
from notifications import (
    ALERT_HISTORY,
    ALERT_RULES,
    evaluate_alert_rules,
    get_alert_history,
    reset_alert_state,
    send_notifications,
)


def test_alert_history_and_cooldown_take_effect():
    reset_alert_state()

    first = send_notifications({
        "message": "Drift detected",
        "dry_run": True,
        "channels": [{"channel": "email", "recipient": "alerts@example.com"}],
    })
    second = send_notifications({
        "message": "Drift detected",
        "dry_run": True,
        "channels": [{"channel": "email", "recipient": "alerts@example.com"}],
    })

    assert first["success"] is True
    assert second["success"] is False
    assert second["status"] == "cooldown"
    assert len(ALERT_HISTORY) == 1


def test_alert_history_is_persisted_to_sqlite():
    reset_alert_state()
    send_notifications({
        "message": "Persisted alert",
        "subject": "Database test",
        "event_type": "data_quality",
        "dry_run": True,
        "channels": [{"channel": "email", "recipient": "alerts@example.com"}],
    })

    history = get_alert_history(limit=10)
    assert len(history) >= 1
    assert any(item["message"] == "Persisted alert" for item in history)


def test_alert_rules_trigger_when_thresholds_are_met():
    result = evaluate_alert_rules(
        {"quality_score": 62, "drift_score": 85, "severity": "high"},
        [
            {"metric": "quality_score", "operator": "<", "value": 70, "channel": "email", "recipient": "alerts@example.com"},
            {"metric": "drift_score", "operator": ">=", "value": 80, "channel": "slack"},
            {"metric": "severity", "operator": "==", "value": "high", "channel": "sms", "recipient": "+15550000000"},
        ],
    )

    assert result["triggered"] is True
    assert len(result["matches"]) == 3
    assert result["matches"][0]["metric"] == "quality_score"


def test_active_rules_can_be_stored_and_retrieved():
    reset_alert_state()
    ALERT_RULES.extend([
        {"metric": "quality_score", "operator": "<", "value": 70, "channel": "email", "recipient": "alerts@example.com"}
    ])

    assert len(ALERT_RULES) == 1
    assert ALERT_RULES[0]["metric"] == "quality_score"


def test_notifications_support_dry_run_for_all_channels():
    result = send_notifications(
        {
            "message": "Drift detected",
            "dry_run": True,
            "channels": [
                {"channel": "email", "recipient": "alerts@example.com"},
                {"channel": "sms", "recipient": "+15550000000"},
                {"channel": "whatsapp", "recipient": "+15550000000"},
                {"channel": "slack"},
                {"channel": "teams"},
            ],
        }
    )

    assert result["success"] is True
    assert len(result["sent"]) == 5
    assert not result["failed"]
    assert all(item["status"] == "dry_run" for item in result["sent"])


def test_notifications_report_invalid_channel_without_stopping_other_channels():
    result = send_notifications(
        {
            "message": "Quality alert",
            "dry_run": True,
            "channels": ["email", "pagerduty"],
        }
    )

    assert result["success"] is False
    assert result["sent"][0]["channel"] == "email"
    assert result["failed"][0]["channel"] == "pagerduty"


def test_notifications_catches_unexpected_provider_errors(monkeypatch):
    def raise_runtime_error(_channel, _recipient, _message, _subject, _dry_run):
        raise ValueError("provider timeout")

    monkeypatch.setattr("notifications.send_notification", raise_runtime_error)

    result = send_notifications({"message": "Alert", "channels": ["email"]})

    assert result["success"] is False
    assert result["sent"] == []
    assert result["failed"][0]["channel"] == "email"
    assert "provider timeout" in result["failed"][0]["error"]


def test_email_provider_delivers_through_configured_smtp(monkeypatch):
    seen = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            seen.update(host=host, port=port, timeout=timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def starttls(self):
            seen["starttls"] = True

        def login(self, username, password):
            seen["login"] = (username, password)

        def send_message(self, email):
            seen["email"] = email

    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "monitor")
    monkeypatch.setenv("SMTP_PASSWORD", "test-password")
    monkeypatch.setenv("ALERT_FROM_EMAIL", "monitor@example.test")
    monkeypatch.setattr(notifications.smtplib, "SMTP", FakeSMTP)

    result = notifications.send_email(
        "ops@example.test", "Test subject", "Test message"
    )

    assert result == {
        "channel": "email",
        "recipient": "ops@example.test",
        "status": "sent",
    }
    assert seen["host"] == "smtp.example.test"
    assert seen["port"] == 587
    assert seen["starttls"] is True
    assert seen["login"] == ("monitor", "test-password")
    assert seen["email"]["Subject"] == "Test subject"
    assert seen["email"].get_content().strip() == "Test message"


@pytest.mark.parametrize(
    ("channel", "sender_env", "sender", "destination"),
    [
        ("sms", "TWILIO_SMS_FROM", "+15551112222", "+15553334444"),
        (
            "whatsapp",
            "TWILIO_WHATSAPP_FROM",
            "whatsapp:+15551112222",
            "whatsapp:+15553334444",
        ),
    ],
)
def test_twilio_providers_submit_authorized_messages(
    monkeypatch, channel, sender_env, sender, destination
):
    seen = {}

    class FakeResponse:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv(sender_env, sender)
    monkeypatch.setattr("notifications.urllib.request.urlopen", fake_urlopen)

    result = notifications.send_twilio(
        "+15553334444", "Provider test", channel
    )

    request = seen["request"]
    assert result["channel"] == channel
    assert result["status"] == "sent"
    assert request.full_url.endswith("/Accounts/AC123/Messages.json")
    assert request.get_header("Authorization") == "Basic " + base64.b64encode(
        b"AC123:test-token"
    ).decode()
    form = parse_qs(request.data.decode())
    assert form == {
        "From": [sender],
        "To": [destination],
        "Body": ["Provider test"],
    }


@pytest.mark.parametrize(
    ("channel", "env_name", "webhook"),
    [
        ("slack", "SLACK_WEBHOOK_URL", "https://hooks.slack.test/services/abc"),
        ("teams", "TEAMS_WEBHOOK_URL", "https://teams.example.test/webhook/abc"),
    ],
)
def test_webhook_providers_post_messages(monkeypatch, channel, env_name, webhook):
    seen = {}

    class FakeResponse:
        status = 200

        def read(self):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv(env_name, webhook)
    monkeypatch.setattr("notifications.urllib.request.urlopen", fake_urlopen)

    result = notifications.send_notification(
        channel, "", "Provider test", dry_run=False
    )

    assert result["channel"] == channel
    assert result["status"] == "sent"
    assert seen["request"].full_url == webhook
    assert json.loads(seen["request"].data) == {"text": "Provider test"}
    assert seen["timeout"] == 15


def test_smtp_errors_are_reported_as_delivery_failures(monkeypatch):
    class FailingSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            raise smtplib.SMTPConnectError(421, "connection unavailable")

        def __exit__(self, *_args):
            return False

    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "monitor")
    monkeypatch.setenv("SMTP_PASSWORD", "test-password")
    monkeypatch.setenv("ALERT_FROM_EMAIL", "monitor@example.test")
    monkeypatch.setattr(notifications.smtplib, "SMTP", FailingSMTP)
    reset_alert_state()

    result = send_notifications({
        "message": "SMTP failure test",
        "channels": [{"channel": "email", "recipient": "ops@example.test"}],
    })

    assert result["success"] is False
    assert result["failed"][0]["channel"] == "email"
    assert "Email delivery failed" in result["failed"][0]["error"]


def test_notification_api_sends_and_records_a_provider_test(monkeypatch):
    reset_alert_state()
    delivered = []
    monkeypatch.setattr(
        notifications,
        "send_email",
        lambda recipient, subject, message: (
            delivered.append((recipient, subject, message))
            or {"channel": "email", "recipient": recipient, "status": "sent"}
        ),
    )
    client = TestClient(app, headers=admin_headers())

    response = client.post(
        "/api/notifications/send",
        json={
            "subject": "Data monitoring provider test",
            "message": "End-to-end test",
            "channels": [{"channel": "email", "recipient": "ops@example.test"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert delivered == [
        (
            "ops@example.test",
            "Data monitoring provider test",
            "End-to-end test",
        )
    ]
    assert any(
        item["message"] == "End-to-end test" for item in get_alert_history()
    )
