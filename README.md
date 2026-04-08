---
title: OpSentrix SRE Harness
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# 🛠️ OpSentrix SRE Harness

**OpenEnv Environment for IT Incident Management & SRE**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-ff6f00.svg)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> Train AI agents to triage, diagnose, and remediate cloud infrastructure incidents inside a deterministic, Kubernetes-style microservices simulation with mocked Prometheus/Grafana and PagerDuty/Opsgenie.

---

## 📖 Overview

**OpSentrix SRE Harness** is a production-ready simulation environment for training RL agents in L1/L2 SRE operations. It exposes the standard OpenEnv contract (`reset`, `step`, `state`) over HTTP and includes a fully compliant `inference.py` baseline agent.

### Architecture Highlights

| Concern | Design choice |
|---|---|
| SOLID | SRP/OCP/LSP/ISP/DIP throughout `inference.py` |
| DRY | Single-source env-var resolution, tool schema, system prompt |
| Type safety | Full `typing` annotations; Pydantic v2 server models |
| Observability | Structured stdout (`[START]`/`[STEP]`/`[END]`); diagnostics to stderr |
| Resilience | Dual-mode LLM (function-calling → text fallback); HTTP retries |
| Security | `HF_TOKEN` required at startup; non-root Docker user |
| Resource safety | Memory-bounded history; MAX_STEPS_PER_EPISODE cap; signal handling |

---

## 🎯 Tasks

| ID | Name | Difficulty | Max Steps |
|---|---|---|---|
| `latency_triage` | Latency Triage | 🟢 Easy | 3 |
| `root_cause_analysis` | Root Cause Analysis | 🟡 Medium | 10 |
| `self_healing_remediation` | Self-Healing Remediation | 🔴 Hard | 15 |

---

## 📁 Repository Structure

```
opsentrix-sre/
├── inference.py          ← Main entry point (submission requirement)
├── demo.py               ← Self-contained local demo runner
├── requirements.txt      ← Runtime dependencies
├── pyproject.toml        ← Build & dev tooling config
├── Dockerfile            ← Multi-stage production build
├── README.md
├── CHANGELOG.md
├── LICENSE
├── openenv.yaml          ← OpenEnv task manifest
├── models.py             ← Pydantic v2 domain models
├── __init__.py            ← Package init
├── .env.example          ← Environment variable template
├── .gitignore
├── .dockerignore
├── server/
│   ├── __init__.py
│   ├── app.py            ← FastAPI application
│   └── environment.py    ← Core simulation engine
└── tests/
    ├── __init__.py
    ├── test_environment.py
    ├── test_models.py
    └── test_server.py
```

---

## 🚀 Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Install

```bash
# Clone the repository
git clone <your-repo-url>
cd opsentrix-sre

# Install dependencies (uv — fast)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ Yes | — | LLM API authentication token |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM endpoint base URL |
| `MODEL_NAME` | No | `gpt-4.1-mini` | Model identifier |
| `OPSENTRIX_SERVER_URL` | No | `http://localhost:7860` | Environment server URL |
| `MAX_RETRIES` | No | `3` | HTTP retry count |
| `REQUEST_TIMEOUT` | No | `60` | Per-request timeout (seconds) |
| `MAX_STEPS_PER_EPISODE` | No | `20` | Safety cap on agent loop iterations |
| `LOG_LEVEL` | No | `INFO` | Stderr log level |

```bash
export HF_TOKEN="sk-..."
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
```

---

## ▶️ Run

### Option 1 — All-in-one demo (recommended)

Starts the embedded server and runs the agent automatically:

```bash
python demo.py
```

### Option 2 — Manual (server + agent separately)

```bash
# Terminal 1 — start the environment server
python -m server.app

# Terminal 2 — run the agent
python inference.py
```

### Option 3 — Docker

```bash
docker build -t opsentrix-sre .
docker run -p 7860:7860 \
  --memory=8g --cpus=2 \
  -e HF_TOKEN="sk-..." \
  opsentrix-sre
```

---

## 📤 Output Format

`inference.py` emits exactly three line types to **stdout**:

```
[START] task=latency_triage env=opsentrix-sre model=gpt-4.1-mini
[STEP] step=1 action=acknowledge_alert reward=0.15 done=false error=null
[END] success=true steps=1 rewards=0.15
```

**Format rules:**
- Rewards are formatted to **2 decimal places**
- Booleans are **lowercase** (`true` / `false`)
- `[END]` is **always** printed, even on errors
- Error field is `null` when no error, or a bare message string

All diagnostic logs go to **stderr** and are safe to redirect or suppress.

---

## ⚡ Resource Constraints

This solution is designed to run within:

| Resource | Limit |
|---|---|
| vCPU | 2 |
| RAM | 8 GB |
| GPU | Not required |

Memory-bounded conversation history and step caps ensure the agent stays within limits.

---

## 🧪 Testing

```bash
# Install dev extras
pip install -e ".[dev]"

# Run full test suite with coverage
pytest

# Run a specific test class
pytest tests/test_environment.py::TestTask1LatencyTriage -v
```

---

## 🤖 Agent Design (`inference.py`)

The baseline agent follows SOLID and DRY principles:

```
┌─────────────────────────────────────────────────────────────────┐
│  main()  →  _build_evaluator()  [Composition Root / DIP]        │
│                                                                  │
│  Evaluator                                                       │
│    └── EpisodeRunner (one episode per task)                      │
│          ├── OpSentrixHttpClient  (EnvironmentClient protocol)   │
│          ├── SREAgent             (ActionStrategy protocol)      │
│          │     ├── ObservationFormatter  (SRP)                   │
│          │     └── JsonActionParser      (SRP)                   │
│          └── StructuredLogger     (SRP — stdout only)            │
└─────────────────────────────────────────────────────────────────┘
```

**Dual-mode inference:** The agent first tries OpenAI function-calling (structured); falls back to JSON-in-text parsing for non-native endpoints.

---

## ✅ Pre-Submission Checklist

Before submitting, verify every item:

- [ ] `inference.py` is in the project root
- [ ] `API_BASE_URL` has a default (`https://api.openai.com/v1`)
- [ ] `MODEL_NAME` has a default (`gpt-4.1-mini`)
- [ ] `HF_TOKEN` is required and validated (raises `ValueError` if missing)
- [ ] All LLM calls use the `openai` Python client — no Anthropic, LangChain, or `requests.post`
- [ ] Output format is exactly `[START]`, `[STEP]`, `[END]`
- [ ] Rewards are 2-decimal formatted (`0.15`, not `0.1500`)
- [ ] Booleans are lowercase (`true` / `false`)
- [ ] `[END]` is always printed, even on errors
- [ ] HF Space is in Running state
- [ ] Repo includes: `inference.py`, `requirements.txt`, `README.md`, `demo.py`
- [ ] Code stays within 2 vCPU / 8 GB RAM

---

## 🔗 Links

- **Hugging Face Space:** *(deploy and add URL before submission)*
- **GitHub Repository:** *(add URL)*
- **API Docs:** `http://localhost:7860/docs` (when server is running)

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).