# OpSentrix SRE Harness -- Pydantic v2 Domain Models
# Author: Yash Bhatt  |  License: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    EXPERT = "expert"

class AlertInfo(BaseModel):
    alert_id: str
    service: str
    severity: Severity
    description: str
    acknowledged: bool = False
    status: str = "FIRING"
    created_at: str | None = None
    model_config = ConfigDict(frozen=True)

class MetricData(BaseModel):
    service: str
    metric_name: str
    value: float
    unit: str
    timestamp: str | None = None

class LogEntry(BaseModel):
    service: str
    timestamp: str | None = None
    level: LogLevel
    message: str
    pod_id: str
    namespace: str = "default"

class ServiceStatus(BaseModel):
    service: str
    status: ServiceHealth
    pod_id: str
    pod_phase: PodPhase
    uptime: str
    version: str = "1.0.0"
    namespace: str = "default"
    replicas: int = 1

class TaskManifest(BaseModel):
    id: str
    name: str
    difficulty: TaskDifficulty
    max_steps: int
    description: str

class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None
    episode_id: str | None = None

# -- Discriminated Union Actions ---------------------------------------------

class AcknowledgeAlert(BaseModel):
    tool: Literal["acknowledge_alert"] = "acknowledge_alert"
    alert_id: str

class QueryMetrics(BaseModel):
    tool: Literal["query_metrics"] = "query_metrics"
    service: str
    metric_name: str | None = None

class FetchLogs(BaseModel):
    tool: Literal["fetch_logs"] = "fetch_logs"
    service: str
    level: str | None = None
    limit: int = 50

class RestartPod(BaseModel):
    tool: Literal["restart_pod"] = "restart_pod"
    service: str
    pod_id: str

class RollbackConfig(BaseModel):
    tool: Literal["rollback_config"] = "rollback_config"
    service: str
    target_revision: str | None = None

class VerifyHealth(BaseModel):
    tool: Literal["verify_health"] = "verify_health"
    service: str | None = None

class SubmitPostmortem(BaseModel):
    tool: Literal["submit_postmortem"] = "submit_postmortem"
    root_cause: str
    affected_services: list[str]
    remediation_steps: list[str]

SREAction = Annotated[
    AcknowledgeAlert | QueryMetrics | FetchLogs | RestartPod | RollbackConfig | VerifyHealth | SubmitPostmortem,
    Field(discriminator="tool"),
]

class SREActionWrapper(BaseModel):
    """Wrapper that makes the discriminated union compatible with create_app()."""
    tool: str
    alert_id: str | None = None
    service: str | None = None
    metric_name: str | None = None
    level: str | None = None
    limit: int = 50
    pod_id: str | None = None
    target_revision: str | None = None
    root_cause: str | None = None
    affected_services: list[str] | None = None
    remediation_steps: list[str] | None = None

    def to_typed_action(self) -> AcknowledgeAlert | QueryMetrics | FetchLogs | RestartPod | RollbackConfig | VerifyHealth | SubmitPostmortem:
        data = {k: v for k, v in self.model_dump().items() if v is not None}
        from pydantic import TypeAdapter
        adapter = TypeAdapter(SREAction)
        return adapter.validate_python(data)


class StepRequest(BaseModel):
    action: SREAction

# -- Observation & State -----------------------------------------------------

class SREObservation(BaseModel):
    message: str
    alerts: list[AlertInfo] = Field(default_factory=list)
    metrics: list[MetricData] = Field(default_factory=list)
    logs: list[LogEntry] = Field(default_factory=list)
    services: list[ServiceStatus] = Field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    success: bool = False

class SREState(BaseModel):
    episode_id: str
    task_id: str = ""
    task_name: str = ""
    difficulty: TaskDifficulty = TaskDifficulty.EASY
    step_count: int = 0
    max_steps: int = 0
    total_reward: float = 0.0
    done: bool = False
    success: bool = False
    actions_taken: list[str] = Field(default_factory=list)
