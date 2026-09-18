from notifications import send_notifications


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
