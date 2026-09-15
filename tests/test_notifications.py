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