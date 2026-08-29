"""HTTP surface: the contract the dashboard depends on."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from labguard.config import reset_settings_cache


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LABGUARD_MODE", "demo")
    monkeypatch.setenv("SIMULATED_QUEUE_LATENCY", "0.0")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    reset_settings_cache()

    from labguard.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    reset_settings_cache()


def test_health_reports_the_wired_infrastructure(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["infrastructure"]["mode"] == "demo"
    assert body["infrastructure"]["job_bus"] == "InProcessJobBus"


def test_config_exposes_the_whole_action_registry(client):
    body = client.get("/api/config").json()
    names = {a["name"] for a in body["actions"]}

    assert {
        "run_seed_comparison",
        "recalculate_metrics",
        "evaluate_classwise",
        "check_data_overlap",
        "compare_configurations",
        "inspect_training_curve",
        "apply_early_stopping",
        "retry_transient_failure",
        "reduce_batch_size",
        "adjust_learning_rate_within_bounds",
        "resume_from_checkpoint",
        "generate_reliability_report",
    } <= names
    for action in body["actions"]:
        assert action["parameters"]["type"] == "object", "every action must publish a parameter schema"
        assert action["min_autonomy"] in {"observe_only", "safe_repair", "managed_autonomy"}
    assert body["demo_scenario"]["text"]


def test_submitting_the_demo_claim_reaches_the_approval_gate(client):
    created = client.post("/api/claims", json={"text": "demo", "use_demo_scenario": True}).json()
    claim_id = created["id"]

    snapshot = client.get(f"/api/claims/{claim_id}").json()
    assert snapshot["claim"]["state"] == "awaiting_approval"
    assert snapshot["subclaims"] and snapshot["loopholes"]
    assert snapshot["report_available"] is False

    plan = snapshot["plans"][-1]
    assert plan["requires_approval"] is True
    assert all(j["state"] == "awaiting_approval" for j in snapshot["jobs"])


def test_full_run_through_the_api_produces_a_downloadable_report(client):
    created = client.post(
        "/api/claims", json={"text": "demo", "use_demo_scenario": True, "autonomy_mode": "managed_autonomy"}
    ).json()
    claim_id = created["id"]

    for _ in range(30):
        snapshot = client.get(f"/api/claims/{claim_id}").json()
        state = snapshot["claim"]["state"]
        if state == "awaiting_approval":
            plan = next(p for p in snapshot["plans"] if p["status"] == "awaiting_approval")
            response = client.post(
                f"/api/claims/{claim_id}/plans/{plan['id']}/decision",
                json={"approved": True, "decided_by": "pytest"},
            )
            assert response.status_code == 200
        if state in {"verdict", "halted_budget", "halted_loop", "halted_approval"}:
            break

    snapshot = client.get(f"/api/claims/{claim_id}").json()
    assert snapshot["claim"]["state"] == "verdict"
    assert snapshot["verdict"]["status"] in {"not_sufficiently_supported", "fragile", "refuted"}
    assert snapshot["report_available"] is True
    assert len(snapshot["score"]["dimensions"]) == 7

    report = client.get(f"/api/claims/{claim_id}/report")
    assert report.status_code == 200
    assert "LabGuard AI reliability report" in report.text
    assert "attachment" in report.headers["content-disposition"]


def test_rejecting_the_plan_halts_and_runs_nothing(client):
    created = client.post("/api/claims", json={"text": "demo", "use_demo_scenario": True}).json()
    claim_id = created["id"]
    snapshot = client.get(f"/api/claims/{claim_id}").json()
    plan = snapshot["plans"][-1]

    client.post(
        f"/api/claims/{claim_id}/plans/{plan['id']}/decision",
        json={"approved": False, "decided_by": "pytest"},
    )
    snapshot = client.get(f"/api/claims/{claim_id}").json()
    assert snapshot["claim"]["state"] == "halted_approval"
    assert all(j["state"] == "rejected" for j in snapshot["jobs"])


def test_a_claim_without_a_comparison_is_rejected(client):
    response = client.post("/api/claims", json={"text": "Our model is better than theirs."})
    assert response.status_code == 422
    assert "two model configurations" in response.text


def test_unknown_claim_is_a_404(client):
    assert client.get("/api/claims/claim_missing").status_code == 404
    assert client.get("/api/claims/claim_missing/report").status_code == 404


def test_a_custom_claim_is_analysed_too(client):
    payload = {
        "text": "Our re-ranker beats the BM25 baseline on recall@10.",
        "autonomy_mode": "safe_repair",
        "context": {
            "dataset": {"name": "custom", "n_samples": 2000, "n_features": 16, "positive_rate": 0.1},
            "models": [
                {"name": "BM25", "family": "linear", "epochs": 20, "is_baseline": True},
                {"name": "Re-ranker", "family": "mlp", "hidden_units": 16, "epochs": 60},
            ],
            "existing_results": [{"model_name": "Re-ranker", "metric": "accuracy", "value": 0.9, "seed": 1}],
        },
    }
    created = client.post("/api/claims", json=payload).json()
    snapshot = client.get(f"/api/claims/{created['id']}").json()

    assert snapshot["claim"]["text"] == payload["text"]
    assert snapshot["subclaims"]
    kinds = {h["kind"] for h in snapshot["loopholes"]}
    assert "seed_sensitivity" in kinds and "unfair_baseline" in kinds


def test_pubsub_push_rejects_a_bad_secret(client, monkeypatch):
    from labguard.api import app as app_module

    monkeypatch.setattr(app_module.get_services().settings, "worker_shared_secret", "s3cret")
    response = client.post(
        "/internal/pubsub/push", json={"message": {"data": ""}}, headers={"Authorization": "Bearer wrong"}
    )
    assert response.status_code == 401


def test_logging_falls_back_to_structured_stdout_without_a_project(tmp_path, monkeypatch, capsys):
    """`ENABLE_CLOUD_LOGGING` must do something real, or not be offered."""
    import json
    import logging

    from labguard.config import Settings
    from labguard.infra.telemetry import configure_logging

    sink = configure_logging(Settings(ENABLE_CLOUD_LOGGING=True, GOOGLE_CLOUD_PROJECT=""))
    assert sink == "structured-stdout"

    logging.getLogger("labguard.test").warning("hello %s", "world")
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["severity"] == "WARNING"
    assert payload["message"] == "hello world"
    assert payload["logger"] == "labguard.test"
