# Copyright (c) 2026 OpSentrix Contributors. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import pytest

from models import (
    AcknowledgeAlert,
    FetchLogs,
    QueryMetrics,
    RestartPod,
    Severity,
    SREObservation,
    TaskDifficulty,
)
from server.environment import TASK_REGISTRY, OpSentrixEnvironment


@pytest.fixture
def env() -> OpSentrixEnvironment:
    return OpSentrixEnvironment()

class TestTaskRegistry:
    def test_all_tasks_registered(self) -> None:
        assert "latency_triage" in TASK_REGISTRY
        assert "root_cause_analysis" in TASK_REGISTRY
        assert "self_healing_remediation" in TASK_REGISTRY

    def test_task_difficulties(self) -> None:
        assert TASK_REGISTRY["latency_triage"].difficulty == TaskDifficulty.EASY
        assert TASK_REGISTRY["root_cause_analysis"].difficulty == TaskDifficulty.MEDIUM
        assert TASK_REGISTRY["self_healing_remediation"].difficulty == TaskDifficulty.HARD

class TestReset:
    def test_reset_returns_observation(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="latency_triage")
        assert isinstance(obs, SREObservation)
        assert not obs.done
        assert len(obs.alerts) == 1
        assert obs.alerts[0].alert_id == "INC-001"
        assert obs.alerts[0].severity == Severity.CRITICAL

    def test_reset_invalid_task_raises(self, env: OpSentrixEnvironment) -> None:
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent_task")

class TestTask1LatencyTriage:
    def test_acknowledge_alert_success(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="latency_triage")
        obs = env.step(AcknowledgeAlert(alert_id="INC-001"))
        assert obs.reward > 0
        assert obs.done is True
        assert obs.success is True

    def test_acknowledge_wrong_alert_id(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="latency_triage")
        obs = env.step(AcknowledgeAlert(alert_id="INC-999"))
        assert obs.reward == 0.0  # clamped to [0.0, 1.0]; step penalty yields 0.0 floor
        assert "not found" in obs.message.lower()

class TestTask3EndToEndRemediation:
    def test_restart_wrong_pod_terminates(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="self_healing_remediation")
        env.step(QueryMetrics(service="Payment-API"))
        obs = env.step(RestartPod(service="Payment-API", pod_id="wrong-pod"))
        assert obs.done is True
        assert obs.success is False
        assert obs.reward == 0.0  # wrong pod resets total_reward to 0.0
        assert env.state.total_reward == 0.0

    def test_restart_without_metrics_blocked(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="self_healing_remediation")
        obs = env.step(RestartPod(service="Payment-API", pod_id="pay-api-pod-3a2b1c"))
        assert "query_metrics" in obs.message.lower()
        assert obs.done is False

class TestFetchLogs:
    def test_fetch_logs_with_level_filter(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        obs = env.step(FetchLogs(service="Payment-API", level="critical"))
        assert len(obs.logs) >= 1
        for entry in obs.logs:
            assert entry.level.value == "critical"
