# Copyright (c) 2026 Yash Bhatt. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _get_obs(resp_json: dict) -> dict:
    """Extract observation from response (handles both openenv wrapped and flat formats)."""
    if "observation" in resp_json:
        return resp_json["observation"]
    return resp_json


class TestOperationalEndpoints:
    def test_health_check(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "healthy")

    def test_list_tasks(self, client: TestClient) -> None:
        resp = client.get("/tasks")
        assert resp.status_code == 200
        tasks = resp.json()["tasks"]
        assert len(tasks) == 5
        task_ids = {t["id"] for t in tasks}
        assert "latency_triage" in task_ids
        assert "cascade_diagnosis" in task_ids
        assert "incident_postmortem" in task_ids


class TestResetEndpoint:
    def test_reset_default_task(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"task_id": "latency_triage"})
        assert resp.status_code == 200
        obs = _get_obs(resp.json())
        assert len(obs.get("alerts", [])) > 0

    def test_reset_specific_task(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"task_id": "self_healing_remediation"})
        assert resp.status_code == 200
        obs = _get_obs(resp.json())
        assert "Restart the correct pod" in obs.get("message", "")

    def test_reset_cascade_task(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"task_id": "cascade_diagnosis"})
        assert resp.status_code == 200
        obs = _get_obs(resp.json())
        assert len(obs.get("alerts", [])) == 3
        assert len(obs.get("services", [])) == 5


class TestStepEndpoint:
    def test_step_acknowledge(self, client: TestClient) -> None:
        reset_resp = client.post("/reset", json={"task_id": "latency_triage"})
        obs = _get_obs(reset_resp.json())
        alerts = obs["alerts"]
        crit = [a for a in alerts if a["severity"] == "critical"][0]
        # Must query metrics first for latency_triage
        client.post("/step", json={"action": {
            "tool": "query_metrics", "service": "API-Gateway"
        }})
        resp = client.post("/step", json={"action": {
            "tool": "acknowledge_alert", "alert_id": crit["alert_id"]
        }})
        assert resp.status_code == 200
        step_data = resp.json()
        step_obs = _get_obs(step_data)
        assert step_data.get("done", step_obs.get("done")) is True


class TestStateEndpoint:
    def test_state_accessible(self, client: TestClient) -> None:
        client.post("/reset", json={"task_id": "latency_triage"})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "step_count" in data or "episode_id" in data


class TestOpenAPISchema:
    def test_docs_accessible(self, client: TestClient) -> None:
        resp = client.get("/docs")
        assert resp.status_code == 200
