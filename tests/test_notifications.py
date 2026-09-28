from notifications import (
    ALERT_HISTORY,
    ALERT_RULES,
    evaluate_alert_rules,
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
