# Copyright (c) 2026 Yash Bhatt. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from models import (
    AcknowledgeAlert,
    AlertInfo,
    LogLevel,
    QueryMetrics,
    ServiceHealth,
    Severity,
    SREAction,
    SREObservation,
)


class TestEnums:
    def test_severity_levels(self) -> None:
        assert len(Severity) == 5

    def test_service_health_states(self) -> None:
        assert len(ServiceHealth) == 3

    def test_log_levels(self) -> None:
        assert len(LogLevel) == 5

class TestSREAction:
    def test_action_serialization(self) -> None:
        adapter = TypeAdapter(SREAction)
        action = QueryMetrics(service="Payment-API")
        data = adapter.dump_python(action)
        assert data["tool"] == "query_metrics"
        assert data["service"] == "Payment-API"

    def test_action_json_roundtrip(self) -> None:
        adapter = TypeAdapter(SREAction)
        json_str = '{"tool": "acknowledge_alert", "alert_id": "INC-001"}'
        restored = adapter.validate_json(json_str)
        assert isinstance(restored, AcknowledgeAlert)
        assert restored.alert_id == "INC-001"

class TestSREObservation:
    def test_minimal_observation(self) -> None:
        obs = SREObservation(message="Hello")
        assert obs.message == "Hello"
        assert obs.alerts == []
        assert obs.reward == 0.0
        assert obs.done is False

class TestDomainObjects:
    def test_alert_info_frozen(self) -> None:
        alert = AlertInfo(
            alert_id="A-1",
            service="X",
            severity=Severity.CRITICAL,
            description="Test",
        )
        with pytest.raises(Exception):
            alert.acknowledged = True

    def test_json_schema_generation(self) -> None:
        schema = SREObservation.model_json_schema()
        assert "properties" in schema
        assert "message" in schema["properties"]
