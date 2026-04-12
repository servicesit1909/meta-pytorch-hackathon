# Copyright (c) 2026 Yash Bhatt. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import pytest

from models import (
    AcknowledgeAlert,
    FetchLogs,
    QueryMetrics,
    RestartPod,
    RollbackConfig,
    Severity,
    SREObservation,
    SubmitPostmortem,
    TaskDifficulty,
    VerifyHealth,
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
        assert "cascade_diagnosis" in TASK_REGISTRY
        assert "incident_postmortem" in TASK_REGISTRY

    def test_task_difficulties(self) -> None:
        assert TASK_REGISTRY["latency_triage"].difficulty == TaskDifficulty.EASY
        assert TASK_REGISTRY["root_cause_analysis"].difficulty == TaskDifficulty.MEDIUM
        assert TASK_REGISTRY["self_healing_remediation"].difficulty == TaskDifficulty.HARD
        assert TASK_REGISTRY["cascade_diagnosis"].difficulty == TaskDifficulty.HARD
        assert TASK_REGISTRY["incident_postmortem"].difficulty == TaskDifficulty.EXPERT

    def test_five_tasks_total(self) -> None:
        assert len(TASK_REGISTRY) == 5


class TestReset:
    def test_reset_returns_observation(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="latency_triage")
        assert isinstance(obs, SREObservation)
        assert not obs.done
        assert len(obs.alerts) == 2  # critical + distractor warning
        critical = [a for a in obs.alerts if a.severity == Severity.CRITICAL]
        assert len(critical) == 1
        assert critical[0].alert_id.startswith("INC-")

    def test_reset_invalid_task_raises(self, env: OpSentrixEnvironment) -> None:
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent_task")

    def test_reset_clears_state(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="root_cause_analysis")
        env.step(QueryMetrics(service="Payment-API"))
        obs2 = env.reset(task_id="root_cause_analysis")
        assert obs2.done is False
        assert env.state.step_count == 0
        assert env.state.total_reward == 0.0


class TestTask1LatencyTriage:
    def test_acknowledge_without_metrics_blocked(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="latency_triage")
        crit = [a for a in obs.alerts if a.severity == Severity.CRITICAL][0]
        obs = env.step(AcknowledgeAlert(alert_id=crit.alert_id))
        assert obs.done is False
        assert "query_metrics" in obs.message.lower()

    def test_full_triage_success(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="latency_triage")
        crit = [a for a in obs.alerts if a.severity == Severity.CRITICAL][0]
        env.step(QueryMetrics(service="API-Gateway"))
        obs = env.step(AcknowledgeAlert(alert_id=crit.alert_id))
        assert obs.done is True
        assert obs.success is True

    def test_acknowledge_wrong_alert_id(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="latency_triage")
        obs = env.step(AcknowledgeAlert(alert_id="INC-999"))
        assert "not found" in obs.message.lower()

    def test_acknowledge_warning_not_success(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="latency_triage")
        warn = [a for a in obs.alerts if a.severity == Severity.WARNING][0]
        env.step(QueryMetrics(service="API-Gateway"))
        obs = env.step(AcknowledgeAlert(alert_id=warn.alert_id))
        assert obs.done is False
        assert obs.success is False
        assert "non-critical" in obs.message.lower()


class TestTask2RootCauseAnalysis:
    def test_rca_success_path(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        env.step(QueryMetrics(service="Payment-API"))
        obs = env.step(FetchLogs(service="Payment-API"))
        assert obs.done is True
        assert obs.success is True

    def test_rca_partial_milestones(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        obs = env.step(QueryMetrics(service="Payment-API"))
        assert obs.done is False  # service_identified alone is not enough

    def test_rca_wrong_service_no_milestone(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        obs = env.step(QueryMetrics(service="Auth-Service"))
        assert obs.done is False


class TestTask3EndToEndRemediation:
    def test_restart_wrong_pod_terminates(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="self_healing_remediation")
        env.step(QueryMetrics(service="Payment-API"))
        obs = env.step(RestartPod(service="Payment-API", pod_id="wrong-pod"))
        assert obs.done is True
        assert obs.success is False
        assert env.state.total_reward == 0.0  # total reward reset to 0

    def test_premature_remediation_blocked(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="self_healing_remediation")
        obs = env.step(RestartPod(service="Payment-API", pod_id="any"))
        assert "premature" in obs.message.lower() or "investigate" in obs.message.lower()
        assert obs.done is False

    def test_full_remediation_success(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="self_healing_remediation")
        alert = obs.alerts[0]
        env.step(AcknowledgeAlert(alert_id=alert.alert_id))
        obs = env.step(QueryMetrics(service="Payment-API"))
        pod_id = [s for s in obs.services if s.service == "Payment-API"][0].pod_id
        env.step(FetchLogs(service="Payment-API"))
        env.step(RestartPod(service="Payment-API", pod_id=pod_id))
        env.step(RollbackConfig(service="Payment-API"))
        obs = env.step(VerifyHealth(service="Payment-API"))
        assert obs.done is True
        assert obs.success is True


class TestTask4CascadeDiagnosis:
    def test_cascade_multiple_alerts(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="cascade_diagnosis")
        assert len(obs.alerts) == 3
        assert len(obs.services) == 5

    def test_cascade_success_path(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="cascade_diagnosis")
        env.step(QueryMetrics(service="Payment-API"))
        obs = env.step(FetchLogs(service="Payment-API"))
        assert obs.done is True
        assert obs.success is True

    def test_cascade_wrong_root_no_success(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="cascade_diagnosis")
        env.step(QueryMetrics(service="API-Gateway"))
        obs = env.step(FetchLogs(service="API-Gateway"))
        assert obs.done is False  # API-Gateway logs don't have OOM


class TestTask5IncidentPostmortem:
    def test_postmortem_full_success(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="incident_postmortem")
        alert = obs.alerts[0]
        env.step(AcknowledgeAlert(alert_id=alert.alert_id))
        obs = env.step(QueryMetrics(service="Payment-API"))
        pod_id = [s for s in obs.services if s.service == "Payment-API"][0].pod_id
        env.step(FetchLogs(service="Payment-API"))
        env.step(RestartPod(service="Payment-API", pod_id=pod_id))
        env.step(RollbackConfig(service="Payment-API"))
        obs = env.step(VerifyHealth(service="Payment-API"))
        assert obs.done is False  # need postmortem still
        obs = env.step(SubmitPostmortem(
            root_cause="Out of Memory in Payment-API JVM heap",
            affected_services=["Payment-API"],
            remediation_steps=["Restarted pod", "Rolled back config"],
        ))
        assert obs.done is True
        assert obs.success is True

    def test_postmortem_before_health_rejected(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="incident_postmortem")
        alert = obs.alerts[0]
        env.step(AcknowledgeAlert(alert_id=alert.alert_id))
        env.step(QueryMetrics(service="Payment-API"))
        obs = env.step(SubmitPostmortem(
            root_cause="OOM", affected_services=["Payment-API"],
            remediation_steps=["restart"],
        ))
        assert obs.done is False
        assert "remediation" in obs.message.lower()

    def test_postmortem_wrong_root_cause_rejected(self, env: OpSentrixEnvironment) -> None:
        obs = env.reset(task_id="incident_postmortem")
        alert = obs.alerts[0]
        env.step(AcknowledgeAlert(alert_id=alert.alert_id))
        obs = env.step(QueryMetrics(service="Payment-API"))
        pod_id = [s for s in obs.services if s.service == "Payment-API"][0].pod_id
        env.step(FetchLogs(service="Payment-API"))
        env.step(RestartPod(service="Payment-API", pod_id=pod_id))
        env.step(RollbackConfig(service="Payment-API"))
        env.step(VerifyHealth(service="Payment-API"))
        obs = env.step(SubmitPostmortem(
            root_cause="Network partition",
            affected_services=["Payment-API"],
            remediation_steps=["restart"],
        ))
        assert obs.done is False
        assert "root_cause" in obs.message.lower()


class TestRewardCalculations:
    def test_step_penalty_is_negative(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        obs = env.step(QueryMetrics(service="Auth-Service"))
        assert obs.reward < 0  # no milestone earned -> step penalty visible

    def test_milestone_reward_positive(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        obs = env.step(QueryMetrics(service="Payment-API"))
        assert obs.reward > 0  # service_identified milestone

    def test_reward_in_valid_range(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        obs = env.step(QueryMetrics(service="Payment-API"))
        assert -0.1 <= obs.reward <= 1.0
        obs = env.step(FetchLogs(service="Payment-API"))
        assert -0.1 <= obs.reward <= 1.0

    def test_total_reward_capped_at_one(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        env.step(QueryMetrics(service="Payment-API"))
        env.step(FetchLogs(service="Payment-API"))
        assert 0.0 <= env.state.total_reward <= 1.0


class TestFetchLogs:
    def test_fetch_logs_with_level_filter(self, env: OpSentrixEnvironment) -> None:
        env.reset(task_id="root_cause_analysis")
        obs = env.step(FetchLogs(service="Payment-API", level="critical"))
        assert len(obs.logs) >= 1
        for entry in obs.logs:
            assert entry.level.value == "critical"


class TestDeterminism:
    def test_seeded_reset_is_deterministic(self) -> None:
        env1 = OpSentrixEnvironment()
        env2 = OpSentrixEnvironment()
        obs1 = env1.reset(task_id="root_cause_analysis", seed=42)
        obs2 = env2.reset(task_id="root_cause_analysis", seed=42)
        assert obs1.alerts[0].alert_id == obs2.alerts[0].alert_id
        assert obs1.services[0].pod_id == obs2.services[0].pod_id
