# ==========================================================================
#  OpSentrix SRE Harness -- FastAPI Application
#  Author: Yash Bhatt
#  License: Apache-2.0
# ==========================================================================

"""
OpSentrix SRE Harness -- FastAPI Application.

Uses OpenEnv's create_app() for full spec compliance including the
Gradio web interface at /web when ENABLE_WEB_INTERFACE=true.

Custom endpoints (/tasks, /tools) are added on top of the base app.

Usage::

    # Development
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import TypeAdapter

try:
    from ..models import (
        SREAction,
        SREActionWrapper,
        SREObservation,
        SREState,
    )
    from .environment import OpSentrixEnvironment, TASK_REGISTRY
except ImportError:
    from models import (  # type: ignore[no-redef]
        SREAction,
        SREActionWrapper,
        SREObservation,
        SREState,
    )
    from server.environment import OpSentrixEnvironment, TASK_REGISTRY  # type: ignore[import,no-redef]

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("opsentrix.server")

_action_adapter = TypeAdapter(SREAction)

# ---------------------------------------------------------------------------
# Build the app via OpenEnv's create_app() for full spec + Gradio /web
# ---------------------------------------------------------------------------

def _build_app():
    """Build app using openenv create_app() with Gradio web interface support."""
    from openenv.core.env_server.http_server import create_app

    application = create_app(
        env=OpSentrixEnvironment,
        action_cls=SREActionWrapper,
        observation_cls=SREObservation,
        env_name="opsentrix-sre",
        max_concurrent_envs=1,
    )

    # -- Custom Endpoints (on top of OpenEnv base) -------------------------

    @application.get(
        "/tasks",
        summary="List available tasks",
        description="Return the registry of available task scenarios.",
        tags=["Environment Info"],
    )
    async def list_tasks() -> dict[str, Any]:
        return {
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "difficulty": t.difficulty.value,
                    "max_steps": t.max_steps,
                    "description": t.description,
                }
                for t in TASK_REGISTRY.values()
            ]
        }

    @application.get(
        "/tools",
        summary="List available tools",
        description="Return the action space -- all typed tools with their schemas.",
        tags=["Environment Info"],
    )
    async def list_tools() -> dict[str, Any]:
        tool_schemas = {
            "acknowledge_alert": {
                "description": "Acknowledge an active PagerDuty/Opsgenie alert.",
                "parameters": {"alert_id": {"type": "string", "required": True}},
            },
            "query_metrics": {
                "description": "Query Prometheus/Grafana metrics for a service.",
                "parameters": {
                    "service": {"type": "string", "required": True},
                    "metric_name": {"type": "string", "required": False},
                },
            },
            "fetch_logs": {
                "description": "Fetch log entries from the log aggregator.",
                "parameters": {
                    "service": {"type": "string", "required": True},
                    "level": {"type": "string", "required": False},
                    "limit": {"type": "integer", "required": False, "default": 50},
                },
            },
            "restart_pod": {
                "description": "Restart a K8s pod (requires prior query_metrics).",
                "parameters": {
                    "service": {"type": "string", "required": True},
                    "pod_id": {"type": "string", "required": True},
                },
            },
            "rollback_config": {
                "description": "Roll back a K8s deployment to a previous revision.",
                "parameters": {
                    "service": {"type": "string", "required": True},
                    "target_revision": {"type": "string", "required": False},
                },
            },
            "verify_health": {
                "description": "Run health/readiness probes on services.",
                "parameters": {
                    "service": {"type": "string", "required": False},
                },
            },
            "submit_postmortem": {
                "description": "Submit a structured incident postmortem (incident_postmortem task only).",
                "parameters": {
                    "root_cause": {"type": "string", "required": True},
                    "affected_services": {"type": "array", "items": {"type": "string"}, "required": True},
                    "remediation_steps": {"type": "array", "items": {"type": "string"}, "required": True},
                },
            },
        }
        return {"tools": tool_schemas}

    return application


def _build_standalone_app():
    """Fallback: build standalone FastAPI app without openenv dependency."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    application = FastAPI(
        title="OpSentrix SRE Harness",
        version="1.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    _env: OpSentrixEnvironment | None = None

    def _get_env() -> OpSentrixEnvironment:
        nonlocal _env
        if _env is None:
            _env = OpSentrixEnvironment()
        return _env

    @application.exception_handler(ValueError)
    async def _value_error_handler(_req: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    from models import ResetRequest  # type: ignore[import]

    @application.post("/reset", tags=["Environment Control"])
    async def reset_endpoint(request: ResetRequest) -> SREObservation:
        return _get_env().reset(seed=request.seed, episode_id=request.episode_id, task_id=request.task_id)

    @application.post("/step", tags=["Environment Control"])
    async def step_endpoint(request: Request) -> SREObservation:
        body = await request.json()
        action_data = body.get("action", body)
        action = _action_adapter.validate_python(action_data)
        return _get_env().step(action)

    @application.get("/state", tags=["State Management"])
    async def state_endpoint() -> SREState:
        return _get_env().state

    @application.get("/health", tags=["Health"])
    async def health_check() -> dict[str, Any]:
        return {"status": "ok", "service": "opsentrix-sre-harness", "version": "1.1.0"}

    @application.get("/ready", tags=["Health"])
    async def readiness_check() -> dict[str, Any]:
        return {"status": "ready", "service": "opsentrix-sre-harness", "version": "1.1.0"}

    @application.get("/tasks", tags=["Environment Info"])
    async def list_tasks() -> dict[str, Any]:
        return {"tasks": [{"id": t.id, "name": t.name, "difficulty": t.difficulty.value, "max_steps": t.max_steps, "description": t.description} for t in TASK_REGISTRY.values()]}

    @application.get("/tools", tags=["Environment Info"])
    async def list_tools() -> dict[str, Any]:
        return {"tools": {}}

    return application


# ---------------------------------------------------------------------------
# Build app -- prefer openenv create_app, fallback to standalone
# ---------------------------------------------------------------------------

try:
    app = _build_app()
    logger.info("OpSentrix app built via openenv create_app() with web interface support")
except Exception as exc:
    logger.warning("openenv create_app() failed (%s), using standalone app", exc)
    app = _build_standalone_app()


# ---------------------------------------------------------------------------
# Direct execution entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the OpSentrix server directly via ``python -m server.app``."""
    import uvicorn

    host = os.getenv("OPSENTRIX_HOST", "0.0.0.0")
    port = int(os.getenv("OPSENTRIX_PORT", "7860"))
    workers = int(os.getenv("OPSENTRIX_WORKERS", "1"))

    logger.info("Starting OpSentrix server on %s:%d (workers=%d)", host, port, workers)

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
