# ==========================================================================
#  OpSentrix SRE Harness -- Simulation Engine
#  Author: Yash Bhatt  |  License: Apache-2.0
# ==========================================================================

"""
OpSentrix SRE Harness -- Core Simulation Engine.

Implements a stochastic Kubernetes-style microservices cluster. Every
reset() call generates a structurally identical but numerically distinct
episode: pod IDs, metric magnitudes, uptime strings, alert variants, and
log details are freshly sampled from a per-episode RNG -- guaranteeing
varied reward trajectories across runs even without an explicit seed.

OpenEnv / Gymnasium contract
----------------------------
    reset(seed, episode_id, task_id) -> SREObservation
    step(SREAction)                  -> SREObservation
    state                            -> SREState   (property)

PBRS reward model: F(s,a,s') = gamma * Phi(s') - Phi(s)
Potential Phi(s) accumulates per milestone (see _MILESTONE_WEIGHTS).
A step penalty is applied when no new milestone is earned.
"""

from __future__ import annotations

import hashlib
import logging
import random
import string
from abc import ABC
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment as _OpenEnvBase  # type: ignore
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False
    _OpenEnvBase = None  # type: ignore

try:
    from ..models import (
        AcknowledgeAlert, AlertInfo, FetchLogs, LogEntry, LogLevel,
        MetricData, PodPhase, QueryMetrics, RestartPod, RollbackConfig,
        Severity, ServiceHealth, ServiceStatus, SREObservation, SREState,
        SubmitPostmortem, TaskDifficulty, TaskManifest, VerifyHealth,
    )
except ImportError:
    from models import (  # type: ignore
        AcknowledgeAlert, AlertInfo, FetchLogs, LogEntry, LogLevel,
        MetricData, PodPhase, QueryMetrics, RestartPod, RollbackConfig,
        Severity, ServiceHealth, ServiceStatus, SREObservation, SREState,
        SubmitPostmortem, TaskDifficulty, TaskManifest, VerifyHealth,
    )

_log = logging.getLogger("opsentrix.environment")
UTC = timezone.utc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA: float = 0.99
STEP_PENALTY: float = 0.02

_MILESTONE_WEIGHTS: dict[str, float] = {
    "alert_acknowledged":     0.15,
    "service_identified":     0.10,
    "oom_signature_found":    0.20,
    "pod_restarted":          0.25,
    "config_rolled_back":     0.10,
    "health_confirmed":       0.30,
    "cascade_root_identified": 0.30,
    "postmortem_submitted":   0.20,
}

# Service dependency graph -- visible to agent in cascade scenarios
_SERVICE_DEPS: dict[str, list[str]] = {
    "API-Gateway": ["Payment-API", "Auth-Service", "Order-Service"],
    "Order-Service": ["Payment-API", "Cache-Service"],
    "Payment-API": [],
    "Auth-Service": [],
    "Cache-Service": [],
}

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, TaskManifest] = {
    "latency_triage": TaskManifest(
        id="latency_triage",
        name="Latency Triage",
        difficulty=TaskDifficulty.EASY,
        max_steps=5,
        description=(
            "Multiple alerts are firing. Query metrics to identify the critical service, "
            "then acknowledge the correct alert to transition the incident to ACK state."
        ),
    ),
    "root_cause_analysis": TaskManifest(
        id="root_cause_analysis",
        name="Root Cause Analysis",
        difficulty=TaskDifficulty.MEDIUM,
        max_steps=10,
        description=(
            "The Payments service is returning 5xx errors at elevated rate. "
            "Query metrics and fetch logs to identify the root cause: "
            "an Out-of-Memory condition in the transaction processor."
        ),
    ),
    "self_healing_remediation": TaskManifest(
        id="self_healing_remediation",
        name="Self-Healing Remediation",
        difficulty=TaskDifficulty.HARD,
        max_steps=15,
        description=(
            "Restart the correct pod, roll back the bad deployment, and "
            "confirm system health via verify_health. Full incident lifecycle required."
        ),
    ),
    "cascade_diagnosis": TaskManifest(
        id="cascade_diagnosis",
        name="Cascade Failure Diagnosis",
        difficulty=TaskDifficulty.HARD,
        max_steps=12,
        description=(
            "Multiple services are degraded simultaneously. Trace the dependency chain "
            "to identify the root-cause service (Payment-API OOM) that is cascading failures "
            "to Order-Service and API-Gateway. Confirm by fetching logs with OOM signature."
        ),
    ),
    "incident_postmortem": TaskManifest(
        id="incident_postmortem",
        name="Incident Postmortem",
        difficulty=TaskDifficulty.EXPERT,
        max_steps=18,
        description=(
            "Full incident lifecycle with structured postmortem. Investigate, remediate "
            "(restart + rollback), verify health, then submit a postmortem report identifying "
            "root cause, affected services, and remediation steps taken."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Randomisation helpers -- all values vary per episode
# ---------------------------------------------------------------------------

def _pod_id(prefix: str, rng: random.Random) -> str:
    """Generate a hex-suffix pod identifier, e.g. 'pay-8c3f2a'."""
    suffix = "".join(rng.choices(string.hexdigits[:16], k=6))
    return f"{prefix}-{suffix}"


def _semver(rng: random.Random) -> str:
    return f"{rng.randint(2,6)}.{rng.randint(0,9)}.{rng.randint(0,12)}"


def _uptime(rng: random.Random) -> str:
    d = rng.randint(0, 21)
    h = rng.randint(0, 23)
    return f"{d}d {h}h" if d else f"{h}h {rng.randint(1,59)}m"


def _noisy(base: float, spread: float, rng: random.Random) -> float:
    """Add uniform noise of +/- spread*base to base."""
    return round(base * (1.0 + rng.uniform(-spread, spread)), 2)


def _ts(rng: random.Random) -> str:
    """ISO-8601 timestamp up to 1 h in the past."""
    t = datetime.now(UTC).timestamp() - rng.randint(0, 3600)
    return datetime.fromtimestamp(t, tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# PBRS helpers
# ---------------------------------------------------------------------------

def _phi(ms: set[str]) -> float:
    return sum(_MILESTONE_WEIGHTS.get(m, 0.0) for m in ms)


def _pbrs(old: set[str], new: set[str]) -> float:
    return GAMMA * _phi(new) - _phi(old)


# ---------------------------------------------------------------------------
# Base-class shim
# ---------------------------------------------------------------------------

if _HAS_OPENENV and _OpenEnvBase is not None:
    class _Base(_OpenEnvBase):  # type: ignore
        pass
else:
    class _Base(ABC):  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            pass


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OpSentrixEnvironment(_Base):  # type: ignore
    """
    Stochastic SRE incident-response environment.

    Pod identifiers, metric values, uptime strings, alert variants, and
    log content are re-sampled each episode so reward trajectories differ
    across runs -- preventing hard-coded agent strategies from trivially
    succeeding.
    """

    def __init__(self, default_task_id: str = "latency_triage") -> None:
        super().__init__()
        self._default_task_id = default_task_id
        self._episode_id: str = ""
        self._task: TaskManifest | None = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._success: bool = False
        self._actions_taken: list[str] = []
        self._seed: int | None = None
        self._rng: random.Random = random.Random()
        self._alerts: dict[str, AlertInfo] = {}
        self._metrics: dict[str, list[MetricData]] = {}
        self._logs: dict[str, list[LogEntry]] = {}
        self._services: dict[str, ServiceStatus] = {}
        self._milestones: set[str] = set()
        self._metrics_queried: bool = False
        self._restarting_pods: set[str] = set()
        _log.info("OpSentrixEnvironment ready (default=%s)", default_task_id)

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **_: Any,
    ) -> SREObservation:
        resolved = task_id or self._default_task_id
        if resolved not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id={resolved!r}. Valid: {sorted(TASK_REGISTRY)}"
            )
        self._task = TASK_REGISTRY[resolved]
        self._seed = seed
        # Fresh RNG every episode -- randomness even without explicit seed
        self._rng = random.Random(seed) if seed is not None else random.Random()
        if seed is not None:
            h = hashlib.sha256(f"opsentrix-{seed}".encode()).hexdigest()[:10]
            self._episode_id = episode_id or f"ep-{h}"
        else:
            self._episode_id = episode_id or str(uuid4())

        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._success = False
        self._actions_taken = []
        self._milestones = set()
        self._metrics_queried = False
        self._restarting_pods = set()
        self._alerts.clear()
        self._metrics.clear()
        self._logs.clear()
        self._services.clear()

        {
            "latency_triage":           self._build_latency_triage,
            "root_cause_analysis":      self._build_root_cause_analysis,
            "self_healing_remediation": self._build_self_healing_remediation,
            "cascade_diagnosis":        self._build_cascade_diagnosis,
            "incident_postmortem":      self._build_incident_postmortem,
        }[resolved]()

        _log.info("Episode %s | task=%s | seed=%s", self._episode_id, resolved, seed)

        tools_line = (
            "Tools: acknowledge_alert . query_metrics . fetch_logs"
            " . restart_pod . rollback_config . verify_health"
        )
        if resolved == "incident_postmortem":
            tools_line += " . submit_postmortem"
        dep_info = ""
        if resolved in ("cascade_diagnosis", "incident_postmortem"):
            dep_lines = ["\nService Dependency Graph:"]
            for svc, deps in _SERVICE_DEPS.items():
                dep_lines.append(f"  {svc} -> {', '.join(deps) if deps else '(leaf service)'}")
            dep_info = "\n".join(dep_lines)

        return SREObservation(
            message=(
                f"Incident Console -- {self._episode_id}\n"
                f"{'-' * 52}\n"
                f"Task       : {self._task.name}\n"
                f"Difficulty : {self._task.difficulty.value.upper()}\n"
                f"Budget     : {self._task.max_steps} steps\n"
                f"{'-' * 52}\n"
                f"{self._task.description}\n\n"
                f"{tools_line}{dep_info}"
            ),
            alerts=list(self._alerts.values()),
            services=list(self._services.values()),
            reward=0.0, done=False, success=False,
        )

    # ------------------------------------------------------------------ step

    def step(self, action: Any, **_: Any) -> SREObservation:
        if self._done:
            return SREObservation(
                message="Episode ended. Call reset() to begin a new one.",
                alerts=list(self._alerts.values()),
                services=list(self._services.values()),
                reward=0.0, done=True, success=self._success,
            )
        if self._task is None:
            return SREObservation(
                message="No active episode. Call reset() first.",
                reward=0.0, done=True, success=False,
            )

        self._step_count += 1
        self._actions_taken.append(action.tool)
        _log.debug("Step %d -- %s", self._step_count, action.tool)

        old_ms = set(self._milestones)
        obs = self._dispatch(action)

        raw = _pbrs(old_ms, self._milestones)
        if not self._success and self._milestones == old_ms:
            raw = -STEP_PENALTY

        sr = max(-0.1, min(1.0, raw))
        if self._success:
            # Efficiency bonus: fewer steps -> higher final burst
            efficiency = 1.0 - (self._step_count / self._task.max_steps) * 0.3
            sr = max(0.0, (1.0 - self._total_reward) * efficiency)
        elif self._total_reward + sr > 1.0:
            sr = max(0.0, 1.0 - self._total_reward)

        self._total_reward = round(max(0.0, min(1.0, self._total_reward + sr)), 4)

        if self._step_count >= self._task.max_steps and not self._done:
            self._done = True
            return SREObservation(
                message=obs.message + f"\n\nStep budget exhausted ({self._task.max_steps}). Episode closed.",
                alerts=obs.alerts, metrics=obs.metrics, logs=obs.logs, services=obs.services,
                reward=round(sr, 4), done=True, success=self._success,
            )

        return SREObservation(
            message=obs.message, alerts=obs.alerts, metrics=obs.metrics,
            logs=obs.logs, services=obs.services,
            reward=round(sr, 4), done=obs.done, success=obs.success,
        )

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> SREState:
        return SREState(
            episode_id=self._episode_id,
            task_id=self._task.id if self._task else "",
            task_name=self._task.name if self._task else "",
            difficulty=self._task.difficulty if self._task else TaskDifficulty.EASY,
            step_count=self._step_count,
            max_steps=self._task.max_steps if self._task else 0,
            total_reward=self._total_reward,
            done=self._done,
            success=self._success,
            actions_taken=list(self._actions_taken),
        )

    # ====================================================================
    # Stochastic scenario builders
    # ====================================================================

    def _build_latency_triage(self) -> None:
        ts = _ts(self._rng)
        pod = _pod_id("gw", self._rng)
        cache_pod = _pod_id("cache", self._rng)
        # Randomise alert ID and description so each run looks distinct
        variants = [
            (f"INC-{self._rng.randint(1,49):03d}",
             f"API-Gateway P99 latency at {_noisy(5200, 0.12, self._rng):.0f} ms -- "
             f"SLO breach active for {self._rng.randint(3,15)} min. PagerDuty P1."),
            (f"INC-{self._rng.randint(50,99):03d}",
             f"Upstream timeout cascade on API-Gateway. Tail latency "
             f"{_noisy(5800, 0.10, self._rng):.0f} ms. Opsgenie escalation triggered."),
            (f"INC-{self._rng.randint(100,149):03d}",
             f"API-Gateway circuit breakers opening -- {self._rng.randint(20,75)}% "
             "of requests timing out. Downstream error propagation detected."),
        ]
        aid, adesc = self._rng.choice(variants)
        self._alerts[aid] = AlertInfo(
            alert_id=aid, service="API-Gateway", severity=Severity.CRITICAL,
            description=adesc, acknowledged=False, status="FIRING", created_at=ts,
        )
        # Distractor: low-severity warning on Cache-Service (should NOT be acknowledged)
        distractor_aid = f"INC-{self._rng.randint(800, 899):03d}"
        self._alerts[distractor_aid] = AlertInfo(
            alert_id=distractor_aid, service="Cache-Service", severity=Severity.WARNING,
            description=(
                f"Cache-Service hit-ratio dropped to {_noisy(72.0, 0.10, self._rng):.0f}% -- "
                "below optimal 85% threshold. Non-critical."
            ),
            acknowledged=False, status="FIRING", created_at=ts,
        )
        self._services["API-Gateway"] = ServiceStatus(
            service="API-Gateway", status=ServiceHealth.DEGRADED,
            pod_id=pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=self._rng.randint(2, 4),
        )
        self._services["Cache-Service"] = ServiceStatus(
            service="Cache-Service", status=ServiceHealth.HEALTHY,
            pod_id=cache_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=2,
        )
        self._metrics["API-Gateway"] = [
            MetricData(service="API-Gateway", metric_name="latency_p99",
                       value=_noisy(5200, 0.12, self._rng), unit="ms", timestamp=ts),
            MetricData(service="API-Gateway", metric_name="request_rate",
                       value=_noisy(2400, 0.18, self._rng), unit="req/s", timestamp=ts),
            MetricData(service="API-Gateway", metric_name="error_rate",
                       value=_noisy(13.5, 0.25, self._rng), unit="%", timestamp=ts),
        ]
        # Red herring metrics on Cache-Service
        self._metrics["Cache-Service"] = [
            MetricData(service="Cache-Service", metric_name="hit_ratio",
                       value=_noisy(72.0, 0.10, self._rng), unit="%", timestamp=ts),
            MetricData(service="Cache-Service", metric_name="memory_usage",
                       value=_noisy(45.0, 0.15, self._rng), unit="%", timestamp=ts),
        ]
        self._logs["API-Gateway"] = [
            LogEntry(service="API-Gateway", timestamp=ts, level=LogLevel.WARNING,
                     message=f"Upstream dependency latency {_noisy(4800, 0.1, self._rng):.0f} ms -- cascade forming",
                     pod_id=pod, namespace="production"),
            LogEntry(service="API-Gateway", timestamp=ts, level=LogLevel.ERROR,
                     message="Circuit breaker OPEN -- rejecting traffic to Payments cluster",
                     pod_id=pod, namespace="production"),
        ]

    def _build_root_cause_analysis(self) -> None:
        ts = _ts(self._rng)
        pay_pod = _pod_id("pay", self._rng)
        auth_pod = _pod_id("auth", self._rng)
        ord_pod = _pod_id("ord", self._rng)
        mem_pct = _noisy(94.0, 0.04, self._rng)
        err_pct = _noisy(86.0, 0.06, self._rng)
        aid = f"INC-{self._rng.randint(200, 399):03d}"
        heap_mb = self._rng.randint(3700, 4096)
        txn = f"txn-{self._rng.randint(10000, 99999)}"

        self._alerts[aid] = AlertInfo(
            alert_id=aid, service="Payment-API", severity=Severity.CRITICAL,
            description=(
                f"Payment-API HTTP 500 rate at {err_pct:.1f}% (threshold 80%). "
                f"Heap utilisation {mem_pct:.1f}%. Customer transactions impacted. Opsgenie P1."
            ),
            acknowledged=False, status="FIRING", created_at=ts,
        )
        self._services["Payment-API"] = ServiceStatus(
            service="Payment-API", status=ServiceHealth.DEGRADED,
            pod_id=pay_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=self._rng.randint(2, 3),
        )
        self._services["Auth-Service"] = ServiceStatus(
            service="Auth-Service", status=ServiceHealth.HEALTHY,
            pod_id=auth_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=2,
        )
        self._services["Order-Service"] = ServiceStatus(
            service="Order-Service", status=ServiceHealth.HEALTHY,
            pod_id=ord_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=3,
        )
        self._metrics["Payment-API"] = [
            MetricData(service="Payment-API", metric_name="error_rate",
                       value=err_pct, unit="%", timestamp=ts),
            MetricData(service="Payment-API", metric_name="memory_usage",
                       value=mem_pct, unit="%", timestamp=ts),
            MetricData(service="Payment-API", metric_name="cpu_throttle",
                       value=_noisy(35.0, 0.30, self._rng), unit="%", timestamp=ts),
            MetricData(service="Payment-API", metric_name="latency_p99",
                       value=_noisy(8100.0, 0.10, self._rng), unit="ms", timestamp=ts),
        ]
        self._metrics["Auth-Service"] = [
            MetricData(service="Auth-Service", metric_name="memory_usage",
                       value=_noisy(40.0, 0.20, self._rng), unit="%", timestamp=ts),
            MetricData(service="Auth-Service", metric_name="error_rate",
                       value=_noisy(0.2, 0.50, self._rng), unit="%", timestamp=ts),
        ]
        self._logs["Payment-API"] = [
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.CRITICAL,
                     message=f"OutOfMemoryError -- JVM heap exhausted ({heap_mb} MB / 4096 MB); transaction processor halting",
                     pod_id=pay_pod, namespace="production"),
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.ERROR,
                     message=f"HTTP 500 on POST /v2/payments/{txn} -- heap allocation failure",
                     pod_id=pay_pod, namespace="production"),
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.ERROR,
                     message="GC full-pause exceeding 8 s -- allocation rate outpaces collection",
                     pod_id=pay_pod, namespace="production"),
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.WARNING,
                     message="JDBC connection pool saturated -- queuing incoming requests",
                     pod_id=pay_pod, namespace="production"),
        ]
        self._logs["Auth-Service"] = [
            LogEntry(service="Auth-Service", timestamp=ts, level=LogLevel.INFO,
                     message="Token validation nominal -- avg 1.8 ms", pod_id=auth_pod, namespace="production"),
        ]

    def _build_self_healing_remediation(self) -> None:
        ts = _ts(self._rng)
        pay_pod = _pod_id("pay", self._rng)
        auth_pod = _pod_id("auth", self._rng)
        ord_pod = _pod_id("ord", self._rng)
        pay_ver = _semver(self._rng)
        mem_pct = _noisy(97.0, 0.02, self._rng)
        aid = f"INC-{self._rng.randint(500, 799):03d}"
        restarts = self._rng.randint(3, 9)

        self._alerts[aid] = AlertInfo(
            alert_id=aid, service="Payment-API", severity=Severity.CRITICAL,
            description=(
                f"Payment-API pod {pay_pod} approaching OOM kill -- "
                f"memory at {mem_pct:.1f}%, restarts={restarts}. "
                f"Release {pay_ver} suspected. PagerDuty P1 escalated."
            ),
            acknowledged=False, status="FIRING", created_at=ts,
        )
        self._services["Payment-API"] = ServiceStatus(
            service="Payment-API", status=ServiceHealth.DEGRADED,
            pod_id=pay_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=pay_ver,
            namespace="production", replicas=self._rng.randint(2, 3),
        )
        self._services["Auth-Service"] = ServiceStatus(
            service="Auth-Service", status=ServiceHealth.HEALTHY,
            pod_id=auth_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=2,
        )
        self._services["Order-Service"] = ServiceStatus(
            service="Order-Service", status=ServiceHealth.HEALTHY,
            pod_id=ord_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=3,
        )
        self._metrics["Payment-API"] = [
            MetricData(service="Payment-API", metric_name="memory_usage",
                       value=mem_pct, unit="%", timestamp=ts),
            MetricData(service="Payment-API", metric_name="error_rate",
                       value=_noisy(62.0, 0.10, self._rng), unit="%", timestamp=ts),
            MetricData(service="Payment-API", metric_name="latency_p99",
                       value=_noisy(9400.0, 0.08, self._rng), unit="ms", timestamp=ts),
            MetricData(service="Payment-API", metric_name="pod_oom_restarts",
                       value=float(restarts), unit="count", timestamp=ts),
        ]
        self._logs["Payment-API"] = [
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.CRITICAL,
                     message=f"OOMKill -- pod {pay_pod} exceeded container memory limit; kubelet scheduling restart",
                     pod_id=pay_pod, namespace="production"),
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.ERROR,
                     message=f"Memory leak in release {pay_ver} transaction buffer -- heap +80 MB/min, no reclaim",
                     pod_id=pay_pod, namespace="production"),
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.ERROR,
                     message="OutOfMemoryError: unable to allocate new transaction context object",
                     pod_id=pay_pod, namespace="production"),
        ]

    def _build_cascade_diagnosis(self) -> None:
        ts = _ts(self._rng)
        pay_pod = _pod_id("pay", self._rng)
        gw_pod = _pod_id("gw", self._rng)
        ord_pod = _pod_id("ord", self._rng)
        auth_pod = _pod_id("auth", self._rng)
        cache_pod = _pod_id("cache", self._rng)
        mem_pct = _noisy(96.0, 0.03, self._rng)
        heap_mb = self._rng.randint(3800, 4096)

        # Root cause: Payment-API is DOWN with OOM
        pay_aid = f"INC-{self._rng.randint(900, 949):03d}"
        self._alerts[pay_aid] = AlertInfo(
            alert_id=pay_aid, service="Payment-API", severity=Severity.CRITICAL,
            description=f"Payment-API returning 5xx at {_noisy(91.0, 0.05, self._rng):.0f}% -- cascading downstream.",
            acknowledged=False, status="FIRING", created_at=ts,
        )
        # Cascade victims
        gw_aid = f"INC-{self._rng.randint(950, 979):03d}"
        self._alerts[gw_aid] = AlertInfo(
            alert_id=gw_aid, service="API-Gateway", severity=Severity.HIGH,
            description=f"API-Gateway error rate elevated to {_noisy(45.0, 0.15, self._rng):.0f}% -- upstream dependency failure.",
            acknowledged=False, status="FIRING", created_at=ts,
        )
        ord_aid = f"INC-{self._rng.randint(980, 999):03d}"
        self._alerts[ord_aid] = AlertInfo(
            alert_id=ord_aid, service="Order-Service", severity=Severity.HIGH,
            description=f"Order-Service latency at {_noisy(6200, 0.10, self._rng):.0f} ms -- payment timeouts.",
            acknowledged=False, status="FIRING", created_at=ts,
        )

        self._services["Payment-API"] = ServiceStatus(
            service="Payment-API", status=ServiceHealth.DOWN,
            pod_id=pay_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=self._rng.randint(2, 3),
        )
        self._services["API-Gateway"] = ServiceStatus(
            service="API-Gateway", status=ServiceHealth.DEGRADED,
            pod_id=gw_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=self._rng.randint(2, 4),
        )
        self._services["Order-Service"] = ServiceStatus(
            service="Order-Service", status=ServiceHealth.DEGRADED,
            pod_id=ord_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=3,
        )
        self._services["Auth-Service"] = ServiceStatus(
            service="Auth-Service", status=ServiceHealth.HEALTHY,
            pod_id=auth_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=2,
        )
        self._services["Cache-Service"] = ServiceStatus(
            service="Cache-Service", status=ServiceHealth.HEALTHY,
            pod_id=cache_pod, pod_phase=PodPhase.RUNNING,
            uptime=_uptime(self._rng), version=_semver(self._rng),
            namespace="production", replicas=2,
        )

        # Metrics -- root cause in Payment-API; red herrings elsewhere
        self._metrics["Payment-API"] = [
            MetricData(service="Payment-API", metric_name="memory_usage",
                       value=mem_pct, unit="%", timestamp=ts),
            MetricData(service="Payment-API", metric_name="error_rate",
                       value=_noisy(91.0, 0.05, self._rng), unit="%", timestamp=ts),
            MetricData(service="Payment-API", metric_name="latency_p99",
                       value=_noisy(12000.0, 0.08, self._rng), unit="ms", timestamp=ts),
        ]
        self._metrics["API-Gateway"] = [
            MetricData(service="API-Gateway", metric_name="error_rate",
                       value=_noisy(45.0, 0.15, self._rng), unit="%", timestamp=ts),
            MetricData(service="API-Gateway", metric_name="latency_p99",
                       value=_noisy(7500, 0.10, self._rng), unit="ms", timestamp=ts),
        ]
        self._metrics["Order-Service"] = [
            MetricData(service="Order-Service", metric_name="error_rate",
                       value=_noisy(38.0, 0.12, self._rng), unit="%", timestamp=ts),
            MetricData(service="Order-Service", metric_name="latency_p99",
                       value=_noisy(6200, 0.10, self._rng), unit="ms", timestamp=ts),
        ]
        # Red herring: Auth-Service CPU looks high but is fine
        self._metrics["Auth-Service"] = [
            MetricData(service="Auth-Service", metric_name="cpu_usage",
                       value=_noisy(78.0, 0.10, self._rng), unit="%", timestamp=ts),
            MetricData(service="Auth-Service", metric_name="error_rate",
                       value=_noisy(0.3, 0.50, self._rng), unit="%", timestamp=ts),
        ]

        # Logs -- only Payment-API has OOM evidence
        self._logs["Payment-API"] = [
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.CRITICAL,
                     message=f"OutOfMemoryError -- JVM heap exhausted ({heap_mb} MB / 4096 MB)",
                     pod_id=pay_pod, namespace="production"),
            LogEntry(service="Payment-API", timestamp=ts, level=LogLevel.ERROR,
                     message="OOMKill triggered by kubelet -- container memory limit exceeded",
                     pod_id=pay_pod, namespace="production"),
        ]
        self._logs["API-Gateway"] = [
            LogEntry(service="API-Gateway", timestamp=ts, level=LogLevel.ERROR,
                     message="Upstream Payment-API returning HTTP 503 -- circuit breaker OPEN",
                     pod_id=gw_pod, namespace="production"),
        ]
        self._logs["Order-Service"] = [
            LogEntry(service="Order-Service", timestamp=ts, level=LogLevel.ERROR,
                     message="Payment-API call timeout after 10s -- order placement failed",
                     pod_id=ord_pod, namespace="production"),
        ]

    def _build_incident_postmortem(self) -> None:
        # Same physical scenario as self_healing_remediation + postmortem requirement
        self._build_self_healing_remediation()

    # ====================================================================
    # Tool dispatch
    # ====================================================================

    def _dispatch(self, action: Any) -> SREObservation:
        # Escalated penalty for premature remediation (no investigation yet)
        if (not self._milestones
                and action.tool in ("restart_pod", "rollback_config", "verify_health")):
            return SREObservation(
                message="Premature remediation: investigate first (query_metrics / fetch_logs) before remediating.",
                reward=0.0, done=False, success=False,
            )
        return {
            "acknowledge_alert": self._tool_acknowledge,
            "query_metrics":     self._tool_query_metrics,
            "fetch_logs":        self._tool_fetch_logs,
            "restart_pod":       self._tool_restart_pod,
            "rollback_config":   self._tool_rollback_config,
            "verify_health":     self._tool_verify_health,
            "submit_postmortem": self._tool_submit_postmortem,
        }.get(action.tool, lambda a: self._err(f"Unknown tool: {a.tool!r}"))(action)

    def _tool_acknowledge(self, action: AcknowledgeAlert) -> SREObservation:
        aid = action.alert_id
        if aid not in self._alerts:
            return self._err(f"Alert {aid!r} not found. Active: {list(self._alerts)}")
        if self._alerts[aid].acknowledged:
            return SREObservation(message=f"Alert {aid} already acknowledged.",
                                  alerts=list(self._alerts.values()), reward=0.0, done=False, success=False)
        # Latency triage: require metrics investigation before acknowledge
        if self._task and self._task.id == "latency_triage" and not self._metrics_queried:
            return SREObservation(
                message="Triage protocol: query_metrics first to identify the degraded service before acknowledging.",
                alerts=list(self._alerts.values()), reward=0.0, done=False, success=False,
            )
        old = self._alerts[aid]
        self._alerts[aid] = AlertInfo(
            alert_id=old.alert_id, service=old.service, severity=old.severity,
            description=old.description, acknowledged=True, status="ACK", created_at=old.created_at,
        )
        self._milestones.add("alert_acknowledged")
        if self._task and self._task.id == "latency_triage":
            # Only succeed if agent acknowledged the CRITICAL alert, not the warning distractor
            if old.severity == Severity.CRITICAL:
                self._done = self._success = True
                return SREObservation(message=f"Alert {aid} acknowledged -- incident closed. Latency recovering.",
                                      alerts=list(self._alerts.values()), reward=0.0, done=True, success=True)
            else:
                return SREObservation(
                    message=f"Alert {aid} acknowledged, but this was a non-critical warning. "
                            "The critical latency alert is still active -- acknowledge the correct one.",
                    alerts=list(self._alerts.values()), reward=0.0, done=False, success=False,
                )
        return SREObservation(
            message=f"Alert {aid} acknowledged (ACK). Continue investigating root cause.",
            alerts=list(self._alerts.values()), reward=0.0, done=False, success=False,
        )

    def _tool_query_metrics(self, action: QueryMetrics) -> SREObservation:
        svc = action.service
        if svc not in self._services:
            return self._err(f"Service {svc!r} not registered. Known: {list(self._services)}")
        metrics = self._metrics.get(svc, [])
        if action.metric_name:
            metrics = [m for m in metrics if m.metric_name == action.metric_name]
        self._metrics_queried = True
        if self._task and "service_identified" not in self._milestones:
            # Latency triage: identify degraded API-Gateway
            if self._task.id == "latency_triage" and svc == "API-Gateway":
                self._milestones.add("service_identified")
            # RCA / Remediation: identify Payment-API
            elif self._task.id in ("root_cause_analysis", "self_healing_remediation") and svc == "Payment-API":
                self._milestones.add("service_identified")
            # Cascade diagnosis: identify root cause Payment-API
            elif self._task.id == "cascade_diagnosis" and svc == "Payment-API":
                self._milestones.add("service_identified")
        pod = self._services[svc].pod_id
        lines = [f"Prometheus -- {svc} (pod: {pod})", ""]
        for m in metrics:
            lines.append(f"  {m.metric_name:<28s} {m.value:>10.2f} {m.unit}")
        return SREObservation(
            message="\n".join(lines) or f"No metrics for {svc!r}.",
            metrics=metrics, services=list(self._services.values()),
            reward=0.0, done=False, success=False,
        )

    def _tool_fetch_logs(self, action: FetchLogs) -> SREObservation:
        svc = action.service
        if svc not in self._services:
            return self._err(f"Service {svc!r} not registered. Known: {list(self._services)}")
        logs = self._logs.get(svc, [])
        if action.level:
            try:
                lvl = LogLevel(action.level.lower())
                logs = [e for e in logs if e.level == lvl]
            except ValueError:
                pass
        logs = logs[: action.limit]
        oom_kw = {"oom", "outofmemory", "oomkill", "heap", "gc"}
        if any(any(k in e.message.lower() for k in oom_kw) for e in logs) \
                and "oom_signature_found" not in self._milestones:
            self._milestones.add("oom_signature_found")
        # Success checks for diagnosis tasks
        done = success = False
        if (self._task and "oom_signature_found" in self._milestones
                and "service_identified" in self._milestones):
            if self._task.id == "root_cause_analysis":
                self._done = done = True
                self._success = success = True
            elif self._task.id == "cascade_diagnosis":
                self._milestones.add("cascade_root_identified")
                self._done = done = True
                self._success = success = True
        lines = [f"Log aggregator -- {svc} ({len(logs)} entries)"]
        for e in logs:
            lines.append(f"  [{e.level.value.upper():8s}] pod={e.pod_id}  {e.message}")
        msg = "\n".join(lines) if logs else "No entries matched filter."
        if success:
            msg += "\n\nRoot cause confirmed: Out-of-Memory condition identified. Incident diagnosed."
        return SREObservation(
            message=msg, logs=logs, reward=0.0, done=done, success=success,
        )

    def _tool_restart_pod(self, action: RestartPod) -> SREObservation:
        if not self._metrics_queried:
            return SREObservation(
                message="Restart blocked: call query_metrics first to identify the target pod.",
                reward=0.0, done=False, success=False,
            )
        svc = action.service
        if svc not in self._services:
            return self._err(f"Service {svc!r} not found. Available: {list(self._services)}")
        canonical = self._services[svc].pod_id
        if action.pod_id in self._restarting_pods:
            return SREObservation(message=f"Pod {action.pod_id} already restarting.",
                                  services=list(self._services.values()), reward=0.0, done=False, success=False)
        if action.pod_id != canonical:
            _log.warning("Wrong pod restart: expected=%s got=%s", canonical, action.pod_id)
            self._done = True
            self._success = False
            self._total_reward = 0.0
            return SREObservation(
                message=(
                    f"CRITICAL: Wrong pod targeted.\n"
                    f"Requested: {action.pod_id!r} -- Correct: {canonical!r} for {svc}.\n"
                    f"Episode terminated. Reward reset to 0."
                ),
                services=list(self._services.values()), reward=0.0, done=True, success=False,
            )
        self._restarting_pods.add(action.pod_id)
        old = self._services[svc]
        self._services[svc] = ServiceStatus(
            service=svc, status=ServiceHealth.HEALTHY,
            pod_id=canonical, pod_phase=PodPhase.RUNNING,
            uptime="0h 0m", version=old.version,
            namespace=old.namespace, replicas=old.replicas,
        )
        self._milestones.add("pod_restarted")
        return SREObservation(
            message=f"Pod {canonical} ({svc}) restarted -- Phase: Running | Status: Healthy.",
            services=list(self._services.values()), reward=0.0, done=False, success=False,
        )

    def _tool_rollback_config(self, action: RollbackConfig) -> SREObservation:
        svc = action.service
        if svc not in self._services:
            return self._err(f"Service {svc!r} not found. Available: {list(self._services)}")
        old = self._services[svc]
        new_ver = action.target_revision or _decrement_version(old.version)
        self._services[svc] = ServiceStatus(
            service=svc, status=old.status, pod_id=old.pod_id, pod_phase=old.pod_phase,
            uptime=old.uptime, version=new_ver, namespace=old.namespace, replicas=old.replicas,
        )
        self._milestones.add("config_rolled_back")
        return SREObservation(
            message=f"Rollback complete for {svc}: {old.version} -> {new_ver}.",
            services=list(self._services.values()), reward=0.0, done=False, success=False,
        )

    def _tool_verify_health(self, action: VerifyHealth) -> SREObservation:
        if action.service and action.service not in self._services:
            return self._err(f"Service {action.service!r} not found.")
        targets = [self._services[action.service]] if action.service else list(self._services.values())
        _ICON = {ServiceHealth.HEALTHY: "*", ServiceHealth.DEGRADED: "o", ServiceHealth.DOWN: "o"}
        lines = ["Health probe results:", ""]
        for s in targets:
            lines.append(
                f"  {_ICON.get(s.status,'?')} {s.service:<22s} {s.status.value.upper():<10s}"
                f"  pod={s.pod_id}  phase={s.pod_phase.value}  v{s.version}"
            )
        done = success = False
        if (self._task and self._task.id in ("self_healing_remediation", "incident_postmortem")
                and "pod_restarted" in self._milestones):
            t = self._services.get("Payment-API")
            if t and t.status == ServiceHealth.HEALTHY:
                self._milestones.add("health_confirmed")
                if self._task.id == "self_healing_remediation":
                    self._done = done = True
                    self._success = success = True
                # incident_postmortem: health confirmed but still need postmortem
        return SREObservation(message="\n".join(lines), services=targets,
                               reward=0.0, done=done, success=success)

    def _tool_submit_postmortem(self, action: SubmitPostmortem) -> SREObservation:
        if self._task is None or self._task.id != "incident_postmortem":
            return self._err("submit_postmortem is only available in the incident_postmortem task.")
        if "health_confirmed" not in self._milestones:
            return SREObservation(
                message="Postmortem rejected: complete remediation and verify health before submitting.",
                reward=0.0, done=False, success=False,
            )
        # Validate postmortem content
        rc = action.root_cause.lower()
        oom_match = any(k in rc for k in ("oom", "out of memory", "outofmemory", "memory", "heap"))
        svc_match = "payment" in " ".join(s.lower() for s in action.affected_services)
        has_steps = len(action.remediation_steps) >= 1

        if not oom_match:
            return SREObservation(
                message="Postmortem rejected: root_cause must identify the Out-of-Memory condition.",
                reward=0.0, done=False, success=False,
            )
        if not svc_match:
            return SREObservation(
                message="Postmortem rejected: affected_services must include Payment-API.",
                reward=0.0, done=False, success=False,
            )
        if not has_steps:
            return SREObservation(
                message="Postmortem rejected: remediation_steps must list at least one action taken.",
                reward=0.0, done=False, success=False,
            )
        self._milestones.add("postmortem_submitted")
        self._done = True
        self._success = True
        return SREObservation(
            message="Postmortem accepted. Root cause, affected services, and remediation documented. Incident closed.",
            reward=0.0, done=True, success=True,
        )

    def _err(self, msg: str) -> SREObservation:
        _log.warning("Tool error: %s", msg)
        return SREObservation(message=f"Error: {msg}", reward=0.0, done=False, success=False)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _decrement_version(v: str) -> str:
    parts = v.split(".")
    if len(parts) != 3:
        return "0.0.0"
    major, minor = int(parts[0]), int(parts[1])
    if minor > 0:
        return f"{major}.{minor - 1}.0"
    if major > 0:
        return f"{major - 1}.9.0"
    return "0.0.0"
