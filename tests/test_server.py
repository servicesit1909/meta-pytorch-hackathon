# Copyright (c) 2026 Yash Bhatt. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import _build_standalone_app


@pytest.fixture
def client() -> TestClient:
    app = _build_standalone_app()
    return TestClient(app)

class TestOperationalEndpoints:
    def test_health_check(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

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
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is False
        assert len(data["alerts"]) > 0

    def test_reset_specific_task(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"task_id": "self_healing_remediation"})
        assert resp.status_code == 200
        assert "Restart the correct pod" in resp.json()["message"]

    def test_reset_cascade_task(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"task_id": "cascade_diagnosis"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["alerts"]) == 3
        assert len(data["services"]) == 5

class TestStepEndpoint:
    def test_step_acknowledge(self, client: TestClient) -> None:
        reset_resp = client.post("/reset", json={"task_id": "latency_triage"})
        alerts = reset_resp.json()["alerts"]
        crit = [a for a in alerts if a["severity"] == "critical"][0]
        # Must query metrics first for latency_triage
        client.post("/step", json={"action": {
            "tool": "query_metrics", "service": "API-Gateway"
        }})
        resp = client.post("/step", json={"action": {
            "tool": "acknowledge_alert", "alert_id": crit["alert_id"]
        }})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["success"] is True

class TestStateEndpoint:
    def test_state_after_step(self, client: TestClient) -> None:
        reset_resp = client.post("/reset", json={"task_id": "latency_triage"})
        crit = [a for a in reset_resp.json()["alerts"] if a["severity"] == "critical"][0]
        client.post("/step", json={"action": {
            "tool": "query_metrics", "service": "API-Gateway"
        }})
        client.post("/step", json={"action": {
            "tool": "acknowledge_alert", "alert_id": crit["alert_id"]
        }})
        resp = client.get("/state")
        data = resp.json()
        assert data["step_count"] == 2
        assert data["done"] is True

class TestOpenAPISchema:
    def test_openapi_json_accessible(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "/reset" in schema["paths"]
        assert "/step" in schema["paths"]
