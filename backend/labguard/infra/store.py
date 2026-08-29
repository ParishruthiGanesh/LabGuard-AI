"""Persistence port.

`StateStore` is the only way agents touch persistent state, so the Firestore
and in-memory implementations are fully interchangeable.  Document shapes are
identical in both: `claims/{claim_id}` with subcollections for subclaims,
loopholes, plans, jobs, evidence, events and the verdict.
"""

from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from ..models.domain import (
    AlternativeExplanation,
    Claim,
    Evidence,
    ExperimentPlan,
    Job,
    LedgerEntry,
    Loophole,
    Subclaim,
    Verdict,
    utcnow,
)

SUBCLAIMS = "subclaims"
LOOPHOLES = "loopholes"
ALTERNATIVES = "alternatives"
PLANS = "plans"
JOBS = "jobs"
EVIDENCE = "evidence"
LEDGER = "ledger"


class StateStore(ABC):
    """Shared, persistent agent state."""

    @abstractmethod
    async def save_claim(self, claim: Claim) -> None: ...

    @abstractmethod
    async def get_claim(self, claim_id: str) -> Claim | None: ...

    @abstractmethod
    async def list_claims(self) -> list[Claim]: ...

    @abstractmethod
    async def put(self, claim_id: str, collection: str, doc_id: str, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    async def list(self, claim_id: str, collection: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def get(self, claim_id: str, collection: str, doc_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def next_sequence(self, claim_id: str) -> int: ...

    # -- typed convenience wrappers --------------------------------------

    async def save_subclaims(self, items: Iterable[Subclaim]) -> None:
        for item in items:
            await self.put(item.claim_id, SUBCLAIMS, item.id, item.model_dump(mode="json"))

    async def list_subclaims(self, claim_id: str) -> list[Subclaim]:
        rows = await self.list(claim_id, SUBCLAIMS)
        return [Subclaim.model_validate(r) for r in rows]

    async def save_loopholes(self, items: Iterable[Loophole]) -> None:
        for item in items:
            await self.put(item.claim_id, LOOPHOLES, item.id, item.model_dump(mode="json"))

    async def list_loopholes(self, claim_id: str) -> list[Loophole]:
        rows = await self.list(claim_id, LOOPHOLES)
        return [Loophole.model_validate(r) for r in rows]

    async def save_alternatives(self, items: Iterable[AlternativeExplanation]) -> None:
        for item in items:
            await self.put(item.claim_id, ALTERNATIVES, item.id, item.model_dump(mode="json"))

    async def list_alternatives(self, claim_id: str) -> list[AlternativeExplanation]:
        rows = await self.list(claim_id, ALTERNATIVES)
        return [AlternativeExplanation.model_validate(r) for r in rows]

    async def save_plan(self, plan: ExperimentPlan) -> None:
        await self.put(plan.claim_id, PLANS, plan.id, plan.model_dump(mode="json"))

    async def get_plan(self, claim_id: str, plan_id: str) -> ExperimentPlan | None:
        row = await self.get(claim_id, PLANS, plan_id)
        return ExperimentPlan.model_validate(row) if row else None

    async def list_plans(self, claim_id: str) -> list[ExperimentPlan]:
        rows = await self.list(claim_id, PLANS)
        plans = [ExperimentPlan.model_validate(r) for r in rows]
        return sorted(plans, key=lambda p: p.round_index)

    async def save_job(self, job: Job) -> None:
        job.updated_at = utcnow()
        await self.put(job.claim_id, JOBS, job.id, job.model_dump(mode="json"))

    async def get_job(self, claim_id: str, job_id: str) -> Job | None:
        row = await self.get(claim_id, JOBS, job_id)
        return Job.model_validate(row) if row else None

    async def list_jobs(self, claim_id: str) -> list[Job]:
        rows = await self.list(claim_id, JOBS)
        jobs = [Job.model_validate(r) for r in rows]
        return sorted(jobs, key=lambda j: j.created_at)

    async def save_evidence(self, items: Iterable[Evidence]) -> None:
        for item in items:
            await self.put(item.claim_id, EVIDENCE, item.id, item.model_dump(mode="json"))

    async def list_evidence(self, claim_id: str) -> list[Evidence]:
        rows = await self.list(claim_id, EVIDENCE)
        items = [Evidence.model_validate(r) for r in rows]
        return sorted(items, key=lambda e: e.created_at)

    async def append_ledger(self, entry: LedgerEntry) -> LedgerEntry:
        entry.sequence = await self.next_sequence(entry.claim_id)
        await self.put(entry.claim_id, LEDGER, entry.id, entry.model_dump(mode="json"))
        return entry

    async def list_ledger(self, claim_id: str) -> list[LedgerEntry]:
        rows = await self.list(claim_id, LEDGER)
        entries = [LedgerEntry.model_validate(r) for r in rows]
        return sorted(entries, key=lambda e: e.sequence)

    async def save_verdict(self, verdict: Verdict) -> None:
        await self.put(verdict.claim_id, "verdict", "current", verdict.model_dump(mode="json"))

    async def get_verdict(self, claim_id: str) -> Verdict | None:
        row = await self.get(claim_id, "verdict", "current")
        return Verdict.model_validate(row) if row else None


class InMemoryStateStore(StateStore):
    """Demo-mode store. Same document shapes, no cloud dependency."""

    def __init__(self) -> None:
        self._claims: dict[str, dict[str, Any]] = {}
        self._collections: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        self._sequences: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def save_claim(self, claim: Claim) -> None:
        claim.updated_at = utcnow()
        async with self._lock:
            self._claims[claim.id] = claim.model_dump(mode="json")

    async def get_claim(self, claim_id: str) -> Claim | None:
        async with self._lock:
            row = self._claims.get(claim_id)
        return Claim.model_validate(copy.deepcopy(row)) if row else None

    async def list_claims(self) -> list[Claim]:
        async with self._lock:
            rows = list(self._claims.values())
        claims = [Claim.model_validate(copy.deepcopy(r)) for r in rows]
        return sorted(claims, key=lambda c: c.created_at, reverse=True)

    async def put(self, claim_id: str, collection: str, doc_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            bucket = self._collections.setdefault((claim_id, collection), {})
            bucket[doc_id] = copy.deepcopy(payload)

    async def list(self, claim_id: str, collection: str) -> list[dict[str, Any]]:
        async with self._lock:
            bucket = self._collections.get((claim_id, collection), {})
            return [copy.deepcopy(v) for v in bucket.values()]

    async def get(self, claim_id: str, collection: str, doc_id: str) -> dict[str, Any] | None:
        async with self._lock:
            row = self._collections.get((claim_id, collection), {}).get(doc_id)
            return copy.deepcopy(row) if row else None

    async def next_sequence(self, claim_id: str) -> int:
        async with self._lock:
            nxt = self._sequences.get(claim_id, 0) + 1
            self._sequences[claim_id] = nxt
            return nxt


class FirestoreStateStore(StateStore):
    """Cloud-mode store backed by Firestore in Native mode.

    Firestore's Python client is synchronous, so every call is pushed to a
    worker thread to keep the FastAPI event loop free.
    """

    def __init__(self, project: str, root: str, database: str = "(default)") -> None:
        from google.cloud import firestore  # imported lazily: cloud-only dep

        self._client = firestore.Client(project=project, database=database)
        self._root = root
        self._transaction_module = firestore

    def _claim_ref(self, claim_id: str):
        return self._client.collection(self._root).document(claim_id)

    async def save_claim(self, claim: Claim) -> None:
        claim.updated_at = utcnow()
        payload = claim.model_dump(mode="json")
        await asyncio.to_thread(self._claim_ref(claim.id).set, payload)

    async def get_claim(self, claim_id: str) -> Claim | None:
        snap = await asyncio.to_thread(self._claim_ref(claim_id).get)
        if not snap.exists:
            return None
        return Claim.model_validate(snap.to_dict())

    async def list_claims(self) -> list[Claim]:
        def _fetch() -> list[dict[str, Any]]:
            return [d.to_dict() for d in self._client.collection(self._root).stream()]

        rows = await asyncio.to_thread(_fetch)
        claims = [Claim.model_validate(r) for r in rows]
        return sorted(claims, key=lambda c: c.created_at, reverse=True)

    async def put(self, claim_id: str, collection: str, doc_id: str, payload: dict[str, Any]) -> None:
        ref = self._claim_ref(claim_id).collection(collection).document(doc_id)
        await asyncio.to_thread(ref.set, payload)

    async def list(self, claim_id: str, collection: str) -> list[dict[str, Any]]:
        def _fetch() -> list[dict[str, Any]]:
            ref = self._claim_ref(claim_id).collection(collection)
            return [d.to_dict() for d in ref.stream()]

        return await asyncio.to_thread(_fetch)

    async def get(self, claim_id: str, collection: str, doc_id: str) -> dict[str, Any] | None:
        ref = self._claim_ref(claim_id).collection(collection).document(doc_id)
        snap = await asyncio.to_thread(ref.get)
        return snap.to_dict() if snap.exists else None

    async def next_sequence(self, claim_id: str) -> int:
        """Atomically increment the per-claim ledger counter."""
        firestore = self._transaction_module
        counter_ref = self._claim_ref(claim_id).collection("counters").document("ledger")

        def _increment() -> int:
            transaction = self._client.transaction()

            @firestore.transactional
            def _apply(tx) -> int:
                snap = counter_ref.get(transaction=tx)
                current = (snap.to_dict() or {}).get("value", 0) if snap.exists else 0
                nxt = int(current) + 1
                tx.set(counter_ref, {"value": nxt})
                return nxt

            return _apply(transaction)

        return await asyncio.to_thread(_increment)
