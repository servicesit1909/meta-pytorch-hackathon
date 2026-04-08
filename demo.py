#!/usr/bin/env python3
# OpSentrix SRE Harness — Local Demo Runner
# Author: Yash B.  |  License: Apache-2.0

"""
OpSentrix SRE Harness — Local Demo Runner.

This script spins up the OpSentrix FastAPI server in a background thread,
waits for it to become healthy, then executes ``inference.py``'s evaluation
loop — all from a single terminal command.

Usage::

    export HF_TOKEN="sk-..."                     # Required
    export API_BASE_URL="https://api.openai.com/v1"  # Optional (has default)
    export MODEL_NAME="gpt-4.1-mini"             # Optional (has default)
    python demo.py

The script prints the [START] / [STEP] / [END] structured lines to stdout
and server diagnostics to stderr.

Environment Variables
---------------------
Same as ``inference.py`` plus:

DEMO_SERVER_PORT : int
    Port for the embedded server. Default: ``7860``
DEMO_STARTUP_TIMEOUT : int
    Seconds to wait for the server to become ready. Default: ``30``
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_PORT: int = int(os.environ.get("DEMO_SERVER_PORT", "7860"))
SERVER_URL: str = f"http://localhost:{SERVER_PORT}"
STARTUP_TIMEOUT: int = int(os.environ.get("DEMO_STARTUP_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Embedded server
# ---------------------------------------------------------------------------

def _start_server() -> None:
    """Import and run the FastAPI app with uvicorn (blocking call)."""
    import uvicorn

    try:
        from server.app import app  # package execution
    except ImportError:
        # Fallback: add cwd to sys.path and retry
        sys.path.insert(0, os.path.dirname(__file__))
        from server.app import app  # type: ignore[no-redef]

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="warning",  # keep demo output clean
    )


def _wait_for_server(timeout: int = STARTUP_TIMEOUT) -> bool:
    """Poll the /health endpoint until the server is ready or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68, flush=True)
    print("  OpSentrix SRE Harness — Demo Runner", flush=True)
    print("=" * 68, flush=True)

    # Validate required env-vars early so the error is obvious
    if not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN is not set.\n"
            "Export it before running:  export HF_TOKEN='sk-...'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Point inference.py at our embedded server
    os.environ.setdefault("OPSENTRIX_SERVER_URL", SERVER_URL)
    os.environ.setdefault("API_BASE_URL", "https://api.openai.com/v1")
    os.environ.setdefault("MODEL_NAME", "gpt-4.1-mini")

    # Start server in a daemon thread
    print(f"Starting OpSentrix server on port {SERVER_PORT} …", flush=True)
    thread = threading.Thread(target=_start_server, daemon=True)
    thread.start()

    if not _wait_for_server():
        print(
            f"ERROR: Server did not become healthy within {STARTUP_TIMEOUT}s.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Server ready at {SERVER_URL}\n", flush=True)

    # Run the evaluation loop
    from inference import main as run_inference

    run_inference()


if __name__ == "__main__":
    main()
