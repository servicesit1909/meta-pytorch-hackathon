# ==========================================================================
#  OpSentrix SRE Harness -- FastAPI Application
#  Author: Yash Bhatt
#  License: Apache-2.0
# ==========================================================================

"""
OpSentrix SRE Harness -- FastAPI Application.

This module wires together the :class:`OpSentrixEnvironment` and exposes it
via HTTP endpoints compatible with the OpenEnv protocol:

    POST /reset   -- Start a new episode (optionally specify task_id, seed).
    POST /step    -- Submit an SRE action and receive an SREObservation.
    GET  /state   -- Retrieve current episode metadata.
    GET  /health  -- Kubernetes-style liveness probe.
    GET  /ready   -- Kubernetes-style readiness probe.

Usage::

    # Development
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter

# ---------------------------------------------------------------------------
# Local imports -- dual-import pattern
# ---------------------------------------------------------------------------
try:
    from ..models import (
        AcknowledgeAlert,
        FetchLogs,
        QueryMetrics,
        ResetRequest,
        RestartPod,
        RollbackConfig,
        SREAction,
        SREObservation,
        SREState,
        StepRequest,
        VerifyHealth,
    )
    from .environment import OpSentrixEnvironment
except ImportError:
    from models import (  # type: ignore[no-redef]
        ResetRequest,
        SREAction,
        SREObservation,
        SREState,
    )
    from server.environment import OpSentrixEnvironment  # type: ignore[import,no-redef]

# Load variables from .env file if available
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("opsentrix.server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_APP_TITLE = "OpSentrix SRE Harness"
_APP_VERSION = "1.0.0"
_APP_DESCRIPTION = (
    "Enterprise-grade OpenEnv environment for IT Incident Management & SRE. "
    "Simulates a Kubernetes microservices cluster with mock Prometheus/Grafana "
    "and PagerDuty/Opsgenie for agent training."
)

# TypeAdapter for discriminated union deserialization
_action_adapter = TypeAdapter(SREAction)

# ---------------------------------------------------------------------------
# Singleton Environment
# ---------------------------------------------------------------------------

_env: OpSentrixEnvironment | None = None


def _get_env() -> OpSentrixEnvironment:
    """Lazily initialise and return the singleton environment instance."""
    global _env
    if _env is None:
        default_task = os.getenv("OPSENTRIX_DEFAULT_TASK", "latency_triage")
        _env = OpSentrixEnvironment(default_task_id=default_task)
        logger.info("OpSentrixEnvironment instantiated (default_task=%s)", default_task)
    return _env


# ---------------------------------------------------------------------------
# Note: We use our own standalone FastAPI app rather than openenv's create_app()
# because SREAction is a Pydantic discriminated union (Annotated[Union[...]])
# which is incompatible with create_app()'s Type[Action] requirement.
# Our standalone app implements the full OpenEnv HTTP protocol:
#   POST /reset, POST /step, GET /state, GET /health, GET /ready.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Startup/shutdown lifecycle hooks."""
    logger.info("OpSentrix SRE Harness starting -- version %s", _APP_VERSION)
    _get_env()
    yield
    logger.info("OpSentrix SRE Harness shutting down")


# ---------------------------------------------------------------------------
# Build the FastAPI application
# ---------------------------------------------------------------------------


def _build_standalone_app() -> FastAPI:
    """Build the standalone FastAPI application with all endpoints."""
    application = FastAPI(
        title=_APP_TITLE,
        version=_APP_VERSION,
        description=_APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    # -- Middleware ------------------------------------------------------

    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def _add_headers(request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    # -- Exception handlers ---------------------------------------------

    @application.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.exception_handler(Exception)
    async def _generic_error_handler(_request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check server logs."},
        )

    # -- Core Endpoints -------------------------------------------------

    @application.post(
        "/reset",
        response_model=SREObservation,
        summary="Reset the environment",
        description="Start a new episode. Specify task_id and optional seed.",
        tags=["Environment"],
    )
    async def reset_endpoint(request: ResetRequest) -> SREObservation:
        env = _get_env()
        return env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_id=request.task_id,
        )

    @application.post(
        "/step",
        response_model=SREObservation,
        summary="Execute one agent action",
        description="Submit a typed SRE tool call and receive the resulting observation.",
        tags=["Environment"],
    )
    async def step_endpoint(request: Request) -> SREObservation:
        """
        Accept raw JSON and deserialize via the discriminated union adapter.

        This allows clients to POST the action directly (without a wrapper)
        or wrapped in ``{"action": ...}``.
        """
        env = _get_env()
        body = await request.json()

        # Support both wrapped and unwrapped formats
        if "action" in body:
            action_data = body["action"]
        else:
            action_data = body

        action = _action_adapter.validate_python(action_data)
        return env.step(action)

    @application.get(
        "/state",
        response_model=SREState,
        summary="Get episode state",
        description="Retrieve metadata about the current episode.",
        tags=["Environment"],
    )
    async def state_endpoint() -> SREState:
        env = _get_env()
        return env.state

    # -- Operational Endpoints ------------------------------------------

    @application.get(
        "/health",
        summary="Liveness probe",
        description="Returns 200 if the server process is alive.",
        tags=["Operations"],
    )
    async def health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "opsentrix-sre-harness",
            "version": _APP_VERSION,
        }

    @application.get(
        "/ready",
        summary="Readiness probe",
        description="Returns 200 if the environment is initialised and ready.",
        tags=["Operations"],
    )
    async def readiness_check() -> dict[str, Any]:
        env = _get_env()
        return {
            "status": "ready",
            "service": "opsentrix-sre-harness",
            "version": _APP_VERSION,
            "environment_initialized": env is not None,
        }

    @application.get(
        "/tasks",
        summary="List available tasks",
        description="Return the registry of available task scenarios.",
        tags=["Environment"],
    )
    async def list_tasks() -> dict[str, Any]:
        from server.environment import TASK_REGISTRY

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
        tags=["Environment"],
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


# ---------------------------------------------------------------------------
# Exported app instance
# ---------------------------------------------------------------------------

app: FastAPI = _build_standalone_app()


# ---------------------------------------------------------------------------
# Direct execution entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the OpSentrix server directly via ``python -m server.app``."""
    import uvicorn

    host = os.getenv("OPSENTRIX_HOST", "0.0.0.0")
    port = int(os.getenv("OPSENTRIX_PORT", "7860"))
    workers = int(os.getenv("OPSENTRIX_WORKERS", "1"))
    reload = os.getenv("OPSENTRIX_RELOAD", "false").lower() == "true"

    logger.info("Starting OpSentrix server on %s:%d (workers=%d)", host, port, workers)

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
