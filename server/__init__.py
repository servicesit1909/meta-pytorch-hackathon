# ==========================================================================
#  OpSentrix SRE Harness -- Server Package
#  Author: Yash Bhatt  |  License: Apache-2.0
# ==========================================================================

"""
OpSentrix Server Package.

Contains the core simulation engine (environment.py) and the FastAPI
HTTP layer (app.py) that together implement the OpenEnv contract:
  POST /reset  -- initialise a new incident-response episode
  POST /step   -- submit one SRE tool call
  GET  /state  -- read episode metadata
  GET  /health -- Kubernetes-compatible liveness probe
"""
