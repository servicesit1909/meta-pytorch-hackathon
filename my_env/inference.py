# ==========================================================================
#  OpSentrix SRE Harness -- Production Inference Agent
#  Author: Yash B.  |  License: Apache-2.0
# ==========================================================================

"""
OpSentrix SRE Harness -- Production Inference Agent.

Emits OpenEnv-compliant structured logs to stdout:

    [START] task=<n> env=<env> model=<model>
    [STEP]  step=<n> action=<tool> reward=<0.00> done=<true|false> error=<null|msg>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

SOLID / DRY principles throughout -- see module docstring of each class.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, Sequence

import httpx
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Env-var resolution -- single source of truth (DRY)
# ---------------------------------------------------------------------------

def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise ValueError(
            f"Required environment variable '{name}' is not set. "
            f"Run:  export {name}='<value>'"
        )
    return val


def _opt(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


HF_TOKEN: Final[str]     = _require("HF_TOKEN")
API_BASE_URL: Final[str] = _opt("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: Final[str]   = _opt("MODEL_NAME",  "gpt-4.1-mini")
SERVER_URL: Final[str]   = _opt("OPSENTRIX_SERVER_URL", "http://localhost:7860")
MAX_STEPS: Final[int]    = int(_opt("MAX_STEPS_PER_EPISODE", "20"))
MAX_RUNTIME: Final[int]  = int(_opt("MAX_RUNTIME_SECONDS",  "1140"))
ENV_NAME: Final[str]     = "opsentrix-sre"
TASKS: Final[list[str]]  = [
    "latency_triage",
    "root_cause_analysis",
    "self_healing_remediation",
]

logging.basicConfig(
    level=_opt("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    stream=sys.stderr,
)
_log = logging.getLogger("opsentrix.inference")

# Graceful shutdown flag
_shutdown = False
def _on_signal(sig: int, _: Any) -> None:
    global _shutdown
    _shutdown = True
for _s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_s, _on_signal)

# ---------------------------------------------------------------------------
# Tool schema -- DRY single definition
# ---------------------------------------------------------------------------

_TOOLS: Final[list[dict[str, Any]]] = [
    {"type": "function", "function": {
        "name": "acknowledge_alert",
        "description": (
            "Acknowledge an active alert in the incident manager. "
            "Always the FIRST action -- never skip this step."
        ),
        "parameters": {"type": "object",
                       "properties": {"alert_id": {"type": "string",
                           "description": "Exact alert ID from the observation (e.g. 'INC-042')."}},
                       "required": ["alert_id"]}}},
    {"type": "function", "function": {
        "name": "query_metrics",
        "description": (
            "Pull Prometheus metrics for a service. "
            "REQUIRED before restart_pod -- the pod_id is only in the metrics response."
        ),
        "parameters": {"type": "object",
                       "properties": {
                           "service": {"type": "string", "description": "Exact service name from the observation."},
                           "metric_name": {"type": "string", "description": "Optional metric filter."}},
                       "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "fetch_logs",
        "description": "Retrieve pod logs. Scan for OOM / heap / GC anomalies.",
        "parameters": {"type": "object",
                       "properties": {
                           "service": {"type": "string"},
                           "level": {"type": "string", "enum": ["debug","info","warning","error","critical"]},
                           "limit": {"type": "integer", "default": 50}},
                       "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "restart_pod",
        "description": (
            "Force-recreate a Kubernetes pod. "
            "CRITICAL: pod_id must be copied EXACTLY from query_metrics output. "
            "Wrong pod_id ends the episode with reward=0."
        ),
        "parameters": {"type": "object",
                       "properties": {
                           "service": {"type": "string"},
                           "pod_id": {"type": "string",
                               "description": "Verbatim pod_id returned by query_metrics. Never guess."}},
                       "required": ["service", "pod_id"]}}},
    {"type": "function", "function": {
        "name": "rollback_config",
        "description": "Roll back a deployment to a prior revision when the current version is the root cause.",
        "parameters": {"type": "object",
                       "properties": {
                           "service": {"type": "string"},
                           "target_revision": {"type": "string", "description": "Leave blank for previous revision."}},
                       "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "verify_health",
        "description": "Run readiness probes. Always call this after every restart or rollback.",
        "parameters": {"type": "object",
                       "properties": {"service": {"type": "string",
                           "description": "Omit to probe all services."}},
                       "required": []}}},
]

# ---------------------------------------------------------------------------
# Advanced chain-of-thought system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: Final[str] = """\
You are ARIA (Automated Remediation Intelligence Agent), an autonomous \
SRE operating within the OpSentrix incident-response simulator.

## Reasoning Protocol
For every turn, internally apply this mental checklist before issuing a tool call:

  OBSERVE   -> What alert, metrics, logs, and service states are visible?
  HYPOTHESIZE -> What failure mode best fits the evidence? (OOM? Cascade? Bad deploy?)
  PRIORITIZE  -> Which information gap, closed, would most reduce diagnostic uncertainty?
  ACT       -> Issue exactly ONE tool call to close that gap or execute the next remediation step.
  VALIDATE  -> After receiving results, confirm or revise the hypothesis.

## Hard Constraints
1. Issue EXACTLY ONE tool call per turn -- never combine or skip.
2. pod_id is discovered ONLY via query_metrics -- never assumed or fabricated.
3. Providing a wrong pod_id to restart_pod terminates the episode with total reward = 0.
4. Always call acknowledge_alert before any investigative or remediation action.
5. Always call verify_health after restart_pod or rollback_config.

## Incident Playbook
  Step 1 -- TRIAGE    : acknowledge_alert  (exact alert_id from observation)
  Step 2 -- TELEMETRY : query_metrics      (degraded service -> note pod_id and metric values)
  Step 3 -- CAUSATION : fetch_logs         (degraded service -> look for OOM / heap / GC / exception)
  Step 4 -- REMEDIATE : restart_pod        (use pod_id from Step 2, verbatim)
                        rollback_config    (if log shows bad release version)
  Step 5 -- CONFIRM   : verify_health      (confirms recovery and closes the episode)

## Diagnostic Heuristics
- memory_usage > 90% + OOM in logs       -> restart_pod, then rollback_config
- error_rate > 50% + normal memory       -> fetch_logs for application exception traces
- latency spike, healthy memory          -> likely upstream dependency; check circuit-breaker logs
- Multiple services degraded             -> query_metrics each individually; do not batch

## Response Format
Respond with a single tool call only. No prose, no commentary outside the function call.
"""

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    message: str
    reward: float
    done: bool
    success: bool
    metrics: list[dict[str, Any]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    services: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        return cls(
            message=str(d.get("message", "")),
            reward=float(d.get("reward", 0.0)),
            done=bool(d.get("done", False)),
            success=bool(d.get("success", False)),
            metrics=d.get("metrics", []),
            logs=d.get("logs", []),
            services=d.get("services", []),
            alerts=d.get("alerts", []),
        )

# ---------------------------------------------------------------------------
# ISP -- narrow protocols
# ---------------------------------------------------------------------------

class EnvPort(Protocol):
    async def reset(self, task_id: str) -> Observation: ...
    async def step(self, action: dict[str, Any]) -> Observation: ...
    async def health(self) -> dict[str, Any]: ...
    async def close(self) -> None: ...

# ---------------------------------------------------------------------------
# HTTP client (LSP -- satisfies EnvPort)
# ---------------------------------------------------------------------------

class OpSentrixClient:
    def __init__(self, base_url: str = SERVER_URL, timeout: float = 30.0) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url, timeout=timeout,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

    async def reset(self, task_id: str) -> Observation:
        r = await self._http.post("/reset", json={"task_id": task_id})
        r.raise_for_status()
        return Observation.from_dict(r.json())

    async def step(self, action: dict[str, Any]) -> Observation:
        r = await self._http.post("/step", json={"action": action})
        r.raise_for_status()
        return Observation.from_dict(r.json())

    async def health(self) -> dict[str, Any]:
        r = await self._http.get("/health")
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._http.aclose()

# ---------------------------------------------------------------------------
# Observation formatter (SRP)
# ---------------------------------------------------------------------------

class ObservationFormatter:
    def format(self, obs: Observation, step: int) -> str:
        parts = [f"### Observation @ step {step}", obs.message]
        if obs.alerts:
            parts.append("\n**Active Alerts**")
            for a in obs.alerts:
                parts.append(
                    f"  ID={a.get('alert_id')}  severity={a.get('severity')}  "
                    f"status={a.get('status')}  service={a.get('service')}"
                )
        if obs.metrics:
            parts.append("\n**Metrics** (use pod_id from service listing below for restart_pod)")
            for m in obs.metrics:
                parts.append(
                    f"  {str(m.get('service')) + '/' + str(m.get('metric_name')):<40s}"
                    f"  {m.get('value'):>10.2f} {m.get('unit')}"
                )
        if obs.logs:
            parts.append("\n**Logs** (scan for: OOM, OutOfMemory, OOMKill, heap, GC)")
            for lo in obs.logs:
                parts.append(
                    f"  [{str(lo.get('level','?')).upper():8s}] "
                    f"pod={lo.get('pod_id')}  {lo.get('message')}"
                )
        if obs.services:
            parts.append("\n**Kubernetes Services** (pod_id column = exact value for restart_pod)")
            for s in obs.services:
                parts.append(
                    f"  {str(s.get('service')):<22s}  {str(s.get('status','?')).upper():<10s}"
                    f"  pod_id={s.get('pod_id')}  phase={s.get('pod_phase')}"
                )
        parts.append(f"\nreward={obs.reward:.2f}  done={obs.done}  success={obs.success}")
        parts.append("\nApply OBSERVE->HYPOTHESIZE->PRIORITIZE->ACT. Issue exactly one tool call:")
        return "\n".join(parts)

# ---------------------------------------------------------------------------
# JSON fallback parser (SRP)
# ---------------------------------------------------------------------------

class TextActionParser:
    _FALLBACK: Final[dict[str, Any]] = {"tool": "verify_health"}

    def parse(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                l for l in cleaned.splitlines() if not l.strip().startswith("```")
            ).strip()
        s, e = cleaned.find("{"), cleaned.rfind("}") + 1
        if s == -1 or e <= 0:
            _log.warning("No JSON in model output: %.120s", text)
            return dict(self._FALLBACK)
        try:
            data: dict[str, Any] = json.loads(cleaned[s:e])
        except json.JSONDecodeError as exc:
            _log.warning("JSON parse error %s: %.120s", exc, text)
            return dict(self._FALLBACK)
        if "tool_name" in data and "tool" not in data:
            data["tool"] = data.pop("tool_name")
        if isinstance(data.get("parameters"), dict):
            data.update(data.pop("parameters"))
        return data

# ---------------------------------------------------------------------------
# LLM Agent -- ARIA (SRP + DIP)
# ---------------------------------------------------------------------------

class SREAgent:
    """
    ARIA -- Automated Remediation Intelligence Agent.
    
    Dual-mode: OpenAI function-calling -> JSON-in-text fallback.
    History is bounded to prevent OOM on long episodes.
    Strictly maintains OpenAI's tool-call / tool-result message ordering.
    """

    _HISTORY_CAP: Final[int] = 40

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        formatter: ObservationFormatter,
        parser: TextActionParser,
    ) -> None:
        self._client = client
        self._model = model
        self._fmt = formatter
        self._parser = parser
        self._history: list[dict[str, Any]] = []
        self._pending_tc_id: str | None = None
        self._step = 0

    def reset(self) -> None:
        self._history = [{"role": "system", "content": _SYSTEM_PROMPT}]
        self._pending_tc_id = None
        self._step = 0

    async def decide(self, obs: Observation) -> dict[str, Any]:
        self._step += 1
        content = self._fmt.format(obs, self._step)

        if self._pending_tc_id:
            self._history.append(
                {"role": "tool", "tool_call_id": self._pending_tc_id, "content": content}
            )
            self._pending_tc_id = None
        else:
            self._history.append({"role": "user", "content": content})

        # Enforce memory bound
        if len(self._history) > self._HISTORY_CAP:
            self._history = [self._history[0]] + self._history[-(self._HISTORY_CAP - 1):]

        return await self._infer()

    async def _infer(self) -> dict[str, Any]:
        # Mode 1: function-calling
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=self._history,  # type: ignore[arg-type]
                tools=_TOOLS,  # type: ignore[arg-type]
                tool_choice="auto",
                temperature=0.15,
                max_tokens=512,
            )
            msg = resp.choices[0].message
            self._history.append(msg.model_dump())  # type: ignore[arg-type]
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                self._pending_tc_id = tc.id
                try:
                    params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    params = {}
                return {"tool": tc.function.name, **params}
            if msg.content:
                return self._parser.parse(msg.content)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Function-call mode failed: %s", exc)

        # Mode 2: text fallback
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=self._history,  # type: ignore[arg-type]
                temperature=0.15,
                max_tokens=512,
            )
            content = resp.choices[0].message.content or ""
            self._history.append({"role": "assistant", "content": content})
            return self._parser.parse(content)
        except Exception as exc:  # noqa: BLE001
            _log.error("Text-mode fallback failed: %s", exc)
            return {"tool": "verify_health"}

# ---------------------------------------------------------------------------
# Structured logger -- owns all stdout (SRP)
# ---------------------------------------------------------------------------

class StructuredLogger:
    @staticmethod
    def _b(v: bool) -> str:
        return "true" if v else "false"

    def start(self, task: str) -> None:
        print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    def step(self, *, n: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
        err = f'"{error[:80]}"' if error else "null"
        print(f"[STEP] step={n} action={action} reward={reward:.2f} done={self._b(done)} error={err}", flush=True)

    def end(self, *, success: bool, steps: int, rewards: Sequence[float]) -> None:
        rw = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={self._b(success)} steps={steps} rewards={rw}", flush=True)

# ---------------------------------------------------------------------------
# Episode runner (SRP)
# ---------------------------------------------------------------------------

class EpisodeRunner:
    """Runs one task episode; [END] is guaranteed via finally."""

    def __init__(self, env: EnvPort, agent: SREAgent, logger: StructuredLogger) -> None:
        self._env = env
        self._agent = agent
        self._log = logger

    async def run(self, task_id: str, wall_start: float) -> bool:
        self._log.start(task_id)
        self._agent.reset()
        step = 0
        rewards: list[float] = []
        success = False

        try:
            obs = await self._env.reset(task_id)
            while not obs.done:
                if _shutdown or time.monotonic() - wall_start > MAX_RUNTIME:
                    self._log.step(n=step + 1, action="timeout", reward=0.0, done=True, error="runtime_limit")
                    rewards.append(0.0)
                    break
                if step >= MAX_STEPS:
                    self._log.step(n=step + 1, action="step_budget_exhausted", reward=0.0, done=True, error="max_steps")
                    rewards.append(0.0)
                    break
                action = await self._agent.decide(obs)
                step += 1
                tool = action.get("tool", "unknown")
                try:
                    obs = await self._env.step(action)
                except Exception as exc:  # noqa: BLE001
                    self._log.step(n=step, action=tool, reward=0.0, done=True, error=str(exc)[:80])
                    rewards.append(0.0)
                    break
                rewards.append(obs.reward)
                self._log.step(n=step, action=tool, reward=obs.reward, done=obs.done)
            success = obs.success
        except Exception as exc:  # noqa: BLE001
            _log.exception("Unhandled error in episode '%s': %s", task_id, exc)
        finally:
            self._log.end(success=success, steps=step, rewards=rewards)

        return success

# ---------------------------------------------------------------------------
# Evaluator (SRP)
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, env: OpSentrixClient, runner: EpisodeRunner, tasks: Sequence[str] = TASKS) -> None:
        self._env = env
        self._runner = runner
        self._tasks = tasks

    async def run(self) -> None:
        _log.info("=" * 68)
        _log.info("OpSentrix Evaluation | model=%s | server=%s", MODEL_NAME, SERVER_URL)
        _log.info("=" * 68)
        try:
            h = await self._env.health()
            _log.info("Server health: %s", h)
        except Exception as exc:
            raise RuntimeError(f"Server unreachable at {SERVER_URL}: {exc}") from exc

        wall = time.monotonic()
        results = []
        for tid in self._tasks:
            _log.info("-- Running task: %s", tid)
            ok = await self._runner.run(tid, wall)
            results.append(ok)

        passed = sum(results)
        _log.info("%d / %d tasks passed", passed, len(results))

# ---------------------------------------------------------------------------
# Composition root + entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    env_client = OpSentrixClient(base_url=SERVER_URL)
    try:
        agent = SREAgent(
            client=AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL),
            model=MODEL_NAME,
            formatter=ObservationFormatter(),
            parser=TextActionParser(),
        )
        runner = EpisodeRunner(env=env_client, agent=agent, logger=StructuredLogger())
        await Evaluator(env=env_client, runner=runner).run()
    finally:
        await env_client.close()


def main() -> None:
    try:
        asyncio.run(_main())
    except RuntimeError as exc:
        _log.critical("Fatal: %s", exc)
        log = StructuredLogger()
        for tid in TASKS:
            log.start(tid)
            log.end(success=False, steps=0, rewards=[])
        sys.exit(1)


if __name__ == "__main__":
    main()
