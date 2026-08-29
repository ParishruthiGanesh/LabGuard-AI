# API reference

Base URL is the LabGuard API service. Interactive docs are served at `/docs`
(OpenAPI at `/openapi.json`).

The dashboard never calls the API directly: Next.js rewrites `/api/backend/*`
to the backend, so browser requests stay same-origin.

---

## `GET /health`

Liveness, plus which adapters are actually wired up. Used by the dashboard's
status strip, so what you see in the header is what is really running.

```json
{
  "status": "ok",
  "infrastructure": {
    "mode": "demo",
    "state_store": "InMemoryStateStore",
    "job_bus": "InProcessJobBus",
    "artifact_store": "LocalArtifactStore",
    "reasoning": "deterministic",
    "project": "(none)"
  }
}
```

In a cloud deployment those read `FirestoreStateStore`, `PubSubJobBus`,
`GcsArtifactStore` and `gemini-adk:<model id>`.

---

## `GET /api/config`

Everything the client needs to render the launcher: the full action registry
with each action's JSON-Schema parameters, cost, retry limit and minimum
autonomy level, plus the bundled demo scenario.

```json
{
  "infrastructure": { "...": "..." },
  "autonomy_modes": ["observe_only", "safe_repair", "managed_autonomy"],
  "actions": [
    {
      "name": "run_seed_comparison",
      "category": "experiment",
      "summary": "Retrain both arms across several seeds under an equal budget and compare paired deltas.",
      "base_cost_units": 1.6,
      "max_retries": 2,
      "min_autonomy": "managed_autonomy",
      "addresses": ["seed_sensitivity", "statistical_uncertainty", "unfair_baseline", "cherry_picked_checkpoint"],
      "parameters": { "type": "object", "properties": { "...": "..." } }
    }
  ],
  "demo_scenario": { "text": "...", "context": { "...": "..." } }
}
```

---

## `POST /api/claims`

Submit a claim for verification. Returns `201` with the created claim;
analysis begins immediately in the background.

```jsonc
{
  "text": "Our re-ranker beats the BM25 baseline on recall@10.",
  "autonomy_mode": "managed_autonomy",     // default: managed_autonomy
  "budget": {                               // omit to use the default policy
    "total_units": 40.0,
    "consumed_units": 0.0,
    "approval_threshold_units": 6.0
  },
  "context": {
    "dataset": { "name": "custom", "n_samples": 4000, "n_features": 24, "positive_rate": 0.08 },
    "models": [
      { "name": "BM25", "family": "linear", "epochs": 25, "is_baseline": true },
      { "name": "Re-ranker", "family": "mlp", "hidden_units": 24, "epochs": 90 }
    ],
    "existing_results": [
      { "model_name": "Re-ranker", "metric": "accuracy", "value": 0.912, "seed": 11,
        "checkpoint_selected_on": "test", "epochs_trained": 90 }
    ]
  },
  "use_demo_scenario": false                // true ignores `context` and uses the bundled scenario
}
```

`use_demo_scenario: true` is the only case where `text` and `context` may be
omitted. Otherwise `text` must be at least 8 characters and `context.models`
must contain at least two configurations to compare — a claim with nothing to
compare cannot be verified, and is rejected with `422` rather than accepted
and quietly stalled.

---

## `GET /api/claims`

All claims, newest first.

## `GET /api/claims/{claim_id}`

The complete snapshot: claim, subclaims, loopholes, alternative explanations,
plans, jobs (with live training curves and health events), evidence, ledger,
verdict and reliability score.

This is deliberately one document rather than a dozen endpoints. A poll is a
single request, and the client can never render a mix of two different states.
`revision` increments whenever anything changes, so a client can skip
re-rendering cheaply.

Poll it while the claim is active; stop on a terminal state (`verdict`,
`halted_approval`, `halted_budget`, `halted_loop`). Returns `404` for an
unknown claim.

---

## `POST /api/claims/{claim_id}/plans/{plan_id}/decision`

Approve or reject a round that needs a human decision.

```json
{ "approved": true, "decided_by": "researcher" }
```

Approving queues every held job. Rejecting moves them all to `rejected` and
halts the claim. Under `observe_only`, approving does **not** execute anything:
the claim halts with an explanation that the autonomy level must be raised
first. Returns the updated claim, or `404` if the claim or plan is unknown.

---

## `POST /api/claims/{claim_id}/jobs/{job_id}/decision`

Approve or decline a single run that RunMedic held for a decision — a repair
that needs more autonomy than the claim is running at.

```json
{ "approved": true, "decided_by": "researcher" }
```

Returns `409` if the job is not in `awaiting_approval`.

---

## `GET /api/claims/{claim_id}/report`

The reliability report as Markdown, with `Content-Disposition: attachment`.
Available once a verdict exists; `404` before that.

## `GET /api/claims/{claim_id}/artifacts/{path}`

A stored artifact — configuration diffs, per-seed results, recorded curves.
Paths come from `job.artifact_uris` in the snapshot.

---

## `POST /internal/pubsub/push`

Pub/Sub push delivery for the worker service. Not for clients.

Expects a standard push envelope with base64 `message.data`. When
`WORKER_SHARED_SECRET` is set, requires `Authorization: Bearer <secret>` and
answers `401` otherwise.

It always answers `204` once a message is accepted, including for an
undecodable payload. A non-2xx would make Pub/Sub redeliver, and whether a
failed job should be retried is RunMedic's decision — based on the failure
signature and the retry limit — not the transport's.

---

## Errors

| Status | Meaning |
| --- | --- |
| `422` | The request body failed validation. `detail` lists each problem. |
| `404` | Unknown claim, plan, job, report or artifact. |
| `409` | The object is not in a state where the request makes sense. |
| `401` | Bad or missing worker credentials on the push endpoint. |

Action parameters are validated twice: once when the plan is built, and again
in the worker immediately before execution. A job whose parameters no longer
validate is failed with an audit record rather than run.
