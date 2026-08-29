"""Run the bundled demo scenario end to end, headless.

Exercises the exact same orchestrator, worker, action registry and scoring the
API serves, so it doubles as a smoke test for the whole workflow.

    python scripts/run_demo.py [--autonomy managed_autonomy] [--json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from labguard.config import Settings  # noqa: E402
from labguard.experiments.scenario import demo_claim  # noqa: E402
from labguard.models.enums import AutonomyMode, ClaimState, JobState  # noqa: E402
from labguard.services import Services  # noqa: E402


async def drive(autonomy: AutonomyMode, auto_approve: bool, quiet: bool) -> dict:
    settings = Settings(
        LABGUARD_MODE="demo",
        SIMULATED_QUEUE_LATENCY=float(os.getenv("SIMULATED_QUEUE_LATENCY", "0.05")),
        ARTIFACT_DIR=os.getenv("ARTIFACT_DIR", "./artifacts"),
    )
    services = Services(settings)
    await services.start()
    say = (lambda *a: None) if quiet else print

    claim = await services.orchestrator.create_claim(demo_claim(autonomy))
    say(f"claim {claim.id}: {claim.text}")
    say(f"reasoning backend: {services.runtime.name}")

    for _ in range(40):
        await services.orchestrator.advance(claim.id)
        current = await services.store.get_claim(claim.id)
        if current is None:
            break
        if current.state == ClaimState.AWAITING_APPROVAL:
            plan = await services.orchestrator.pending_plan(claim.id)
            if plan is None:
                # A repair is waiting on a decision rather than a whole round.
                held = [j for j in await services.store.list_jobs(claim.id) if j.state == JobState.AWAITING_APPROVAL]
                if not held:
                    break
                say(f"\nrepair approval requested for: {', '.join(j.action_type for j in held)}")
                if not auto_approve:
                    say("stopping: pass --approve to continue past the approval gate")
                    break
                for job in held:
                    await services.orchestrator.approve_repair(current, job)
                current.state = ClaimState.EXECUTING
                await services.store.save_claim(current)
                say("approved\n")
                await services.bus.drain(timeout=180)
                continue
            say(f"\napproval requested: {plan.summary}")
            for item in plan.items:
                flag = " [needs approval]" if item.requires_approval else ""
                say(f"  - {item.action_type:<26} {item.estimated_cost_units:>5.2f}u  gain {item.expected_information_gain}{flag}")
            if not auto_approve:
                say("\nstopping: pass --approve to continue past the approval gate")
                break
            await services.orchestrator.decide_plan(claim.id, plan.id, True, "demo-script")
            say("approved\n")
        if current.state == ClaimState.EXECUTING:
            await services.bus.drain(timeout=180)
            continue
        if current.state.is_terminal:
            break
        await asyncio.sleep(0.05)

    await services.bus.drain(timeout=180)
    await services.orchestrator.advance(claim.id)
    await services.bus.drain(timeout=180)

    snapshot = await _summarise(services, claim.id)
    await services.close()
    return snapshot


async def _summarise(services: Services, claim_id: str) -> dict:
    claim = await services.store.get_claim(claim_id)
    jobs = await services.store.list_jobs(claim_id)
    verdict = await services.store.get_verdict(claim_id)
    subclaims = await services.store.list_subclaims(claim_id)
    ledger = await services.store.list_ledger(claim_id)
    evidence = await services.store.list_evidence(claim_id)
    return {
        "claim_id": claim_id,
        "state": claim.state.value if claim else "missing",
        "planning_rounds": (claim.planning_round + 1) if claim else 0,
        "budget_consumed": claim.budget.consumed_units if claim else 0,
        "halt_reason": claim.halt_reason if claim else "",
        "jobs": [
            {
                "action": j.action_type,
                "config": j.params.get("config_name"),
                "state": j.state.value,
                "attempts": j.attempts,
                "health": j.health.status.value,
                "anomalies": [e.anomaly.value for e in j.health.events],
                "recoveries": [r for r in j.recovery_actions if r.startswith("recovery:")],
            }
            for j in jobs
        ],
        "subclaims": {s.statement[:60]: s.status.value for s in subclaims},
        "evidence_count": len(evidence),
        "ledger_entries": len(ledger),
        "verdict": verdict.status.value if verdict else None,
        "headline": verdict.headline if verdict else None,
        "scores": {d.dimension.value: d.score for d in verdict.score.dimensions} if verdict and verdict.score else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LabGuard demo scenario end to end.")
    parser.add_argument("--autonomy", default="managed_autonomy", choices=[m.value for m in AutonomyMode])
    parser.add_argument("--approve", action="store_true", default=True, help="auto-approve the plan (default)")
    parser.add_argument("--no-approve", dest="approve", action="store_false")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON only")
    args = parser.parse_args()

    result = asyncio.run(drive(AutonomyMode(args.autonomy), args.approve, args.json))
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    if result["halt_reason"]:
        print(f"\nhalted in state '{result['state']}': {result['halt_reason']}")

    print("\n=== jobs ===")
    for job in result["jobs"]:
        target = f" [{job['config']}]" if job["config"] else ""
        print(
            f"  {job['action']:<26}{target:<32} {job['state']:<13} attempts={job['attempts']} "
            f"health={job['health']:<10} {','.join(job['anomalies']) or '-'} {','.join(job['recoveries'])}"
        )
    print("\n=== subclaims ===")
    for statement, status in result["subclaims"].items():
        print(f"  {status:<14} {statement}")
    print("\n=== reliability ===")
    for dimension, score in result["scores"].items():
        print(f"  {dimension:<26} {score:>3}")
    if result["verdict"]:
        print(f"\nverdict: {result['verdict']}")
        print(f"{result['headline']}")
    else:
        print(f"\nno verdict: the claim stopped in state '{result['state']}'.")
    print(
        f"\nrounds={result['planning_rounds']} budget={result['budget_consumed']} "
        f"evidence={result['evidence_count']} ledger={result['ledger_entries']}"
    )
    # Halting on policy is a correct outcome, not a failure of the run.
    return 0 if (result["verdict"] or result["halt_reason"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
