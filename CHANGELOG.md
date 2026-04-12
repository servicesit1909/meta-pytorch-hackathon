# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] -- 2026-04-08

### Changed

- **inference.py**: Fixed `[STEP]` error format -- bare strings instead of quoted
  (matches OpenEnv spec exactly: `error=<msg|null>`)
- **inference.py**: Added `MAX_STEPS_PER_EPISODE` env var (default: 20) to
  prevent runaway agent loops under resource constraints
- **inference.py**: Added memory-bounded conversation history (`_MAX_HISTORY_LENGTH=40`)
  to stay within 8 GB RAM limit
- **inference.py**: Added graceful shutdown signal handling (SIGINT/SIGTERM)
- **inference.py**: Added sentinel `[STEP]` before `[END]` on fatal server errors
  so every `[START]` has at least one `[STEP]`
- **inference.py**: Sanitised error messages (collapsed whitespace/newlines)
- **requirements.txt**: Removed `openenv-core` from hard dependencies (moved to
  optional `[openenv]` extra in `pyproject.toml`)
- **Dockerfile**: Uses `requirements.txt` for deps (DRY); copies files to `/app/`
  root (inference.py at root per spec); added OCI labels for HF Spaces
- **README.md**: Added pre-submission checklist, resource constraints section,
  correct output format documentation
- **.env.example**: Renamed env vars to `OPSENTRIX_*` prefix for consistency
- **server/__init__.py**: Fixed naming consistency across modules

### Fixed

- `openenv-core>=0.1.0` was a hard dependency that could fail pip install if
  the package isn't published on PyPI -- now gracefully optional

## [1.0.0] -- 2026-04-06

### Added

- **Core Environment**: `OpSentrixEnvironment` with full Gymnasium/OpenEnv protocol
  support (`reset()`, `step()`, `state`)
- **3 Task Scenarios**:
  - Task 1 -- Latency Triage (Easy, max 3 steps)
  - Task 2 -- Root Cause Analysis (Medium, max 10 steps)
  - Task 3 -- Self-Healing Remediation (Hard, max 15 steps)
- **6 SRE Tools**: `acknowledge_alert`, `query_metrics`, `fetch_logs`,
  `restart_pod`, `rollback_config`, `verify_health`
- **PBRS Reward Shaping**: Potential-Based Reward Shaping with milestone tracking
  and step penalties, bounded rewards
- **Dependency Gates**: `restart_pod` requires prior `query_metrics` call
- **Hard Constraints**: Wrong pod_id on restart terminates episode with reward=0.0
- **FastAPI Server**: Standalone app with CORS, security headers, operational
  endpoints (/health, /ready, /tasks, /tools)
- **Pydantic v2 Models**: Fully typed data contracts with JSON schema generation
- **Baseline Inference Agent**: OpenAI-compatible LLM agent with function-calling
  and JSON-in-text fallback modes
- **Docker Support**: Multi-stage build with uv, non-root user, health checks
- **Test Suite**: Unit and integration tests with pytest + coverage
- **OpenEnv Manifest**: `openenv.yaml` with task registry and hardware requirements
- **Configuration**: 12-factor app via environment variables with `.env.example`
