"""Seed a running LabGuard deployment with the demo scenario.

Submits the bundled claim through the HTTP API, approves the plan when the
platform asks, and waits for the verdict. Works against a local backend or a
deployed Cloud Run service.

    python scripts/seed_demo.py
    python scripts/seed_demo.py --api https://labguard-api-xxxx.run.app
    python scripts/seed_demo.py --autonomy safe_repair --no-approve
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TERMINAL = {"verdict", "halted_budget", "halted_loop", "halted_approval"}


def call(api: str, path: str, payload: dict | None = None) -> dict:
    url = f"{api.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{url} -> {exc.code}: {exc.read().decode()[:400]}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach {url}: {exc.reason}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8080", help="LabGuard API base URL")
    parser.add_argument("--autonomy", default="managed_autonomy",
                        choices=["observe_only", "safe_repair", "managed_autonomy"])
    parser.add_argument("--budget", type=float, default=40.0)
    parser.add_argument("--approve", action="store_true", default=True)
    parser.add_argument("--no-approve", dest="approve", action="store_false",
                        help="stop at the approval gate instead of approving")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    health = call(args.api, "/health")
    print(f"connected: {json.dumps(health['infrastructure'])}")

    claim = call(
        args.api,
        "/api/claims",
        {
            "use_demo_scenario": True,
            "autonomy_mode": args.autonomy,
            "budget": {"total_units": args.budget, "consumed_units": 0.0, "approval_threshold_units": 6.0},
        },
    )
    claim_id = claim["id"]
    print(f"claim {claim_id}: {claim['text']}")

    deadline = time.time() + args.timeout
    last_state = ""
    while time.time() < deadline:
        snapshot = call(args.api, f"/api/claims/{claim_id}")
        state = snapshot["claim"]["state"]
        if state != last_state:
            print(f"  state -> {state}")
            last_state = state

        if state == "awaiting_approval":
            pending = next((p for p in snapshot["plans"] if p["status"] == "awaiting_approval"), None)
            if pending is None:
                break
            print(f"  approval requested: {pending['summary']}")
            if not args.approve:
                print("  stopping at the approval gate (--no-approve)")
                return 0
            call(
                args.api,
                f"/api/claims/{claim_id}/plans/{pending['id']}/decision",
                {"approved": True, "decided_by": "seed_demo.py"},
            )
            print("  approved")

        if state in TERMINAL:
            verdict = snapshot.get("verdict")
            score = snapshot.get("score")
            print()
            if verdict:
                print(f"verdict: {verdict['status']}")
                print(f"{verdict['headline']}")
            if score:
                print()
                for dimension in score["dimensions"]:
                    print(f"  {dimension['dimension']:<26} {dimension['score']:>3}   {dimension['calculation']}")
            print()
            print(f"report: {args.api.rstrip('/')}/api/claims/{claim_id}/report")
            return 0
        time.sleep(1.5)

    print(f"timed out after {args.timeout}s in state '{last_state}'", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
