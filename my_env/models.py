# OpSentrix SRE Harness -- Pydantic v2 Domain Models
# Author: Yash B.  |  License: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field
from openenv.core.env_server.types import Action, Observation


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    FATAL = "fatal"

class ServiceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"

class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class PodPhase(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"

class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class AlertInfo(Action): # Base for pydantic models
    alert_id: str
    service: str
    severity: Severity
    description: str
    acknowledged: bool = False
    status: str = "FIRING"
    created_at: str | None = None
    model_config = ConfigDict(frozen=True)

class MetricData(Action):
    service: str
    metric_name: str
    value: float
    unit: str
    timestamp: str | None = None

class LogEntry(Action):
    service: str
    timestamp: str | None = None
    level: LogLevel
    message: str
    pod_id: str
    namespace: str = "default"

class ServiceStatus(Action):
    service: str
    status: ServiceHealth
    pod_id: str
    pod_phase: PodPhase
    uptime: str
    version: str = "1.0.0"
    namespace: str = "default"
    replicas: int = 1

class TaskManifest(Action):
    id: str
    name: str
    difficulty: TaskDifficulty
    max_steps: int
    description: str

# -- Discriminated Union Actions ---------------------------------------------

class AcknowledgeAlert(Action):
    tool: Literal["acknowledge_alert"] = "acknowledge_alert"
    alert_id: str

class QueryMetrics(Action):
    tool: Literal["query_metrics"] = "query_metrics"
    service: str
    metric_name: str | None = None

class FetchLogs(Action):
    tool: Literal["fetch_logs"] = "fetch_logs"
    service: str
    level: str | None = None
    limit: int = 50

class RestartPod(Action):
    tool: Literal["restart_pod"] = "restart_pod"
    service: str
    pod_id: str

class RollbackConfig(Action):
    tool: Literal["rollback_config"] = "rollback_config"
    service: str
    target_revision: str | None = None

class VerifyHealth(Action):
    tool: Literal["verify_health"] = "verify_health"
    service: str | None = None

# OpenEnv SDK requires the top-level Action class to be named MyAction or similar
# based on what app.py expects. In our case, app.py uses MyAction.
MyAction = Annotated[
    AcknowledgeAlert | QueryMetrics | FetchLogs | RestartPod | RollbackConfig | VerifyHealth,
    Field(discriminator="tool"),
]

# -- Observation & State -----------------------------------------------------

class MyObservation(Observation):
    message: str
    alerts: list[AlertInfo] = Field(default_factory=list)
    metrics: list[MetricData] = Field(default_factory=list)
    logs: list[LogEntry] = Field(default_factory=list)
    services: list[ServiceStatus] = Field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    success: bool = False
