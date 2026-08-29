"""Dependency wiring.

Builds the adapter set for the configured deployment mode and hands the same
orchestrator/worker pair to every entry point.
"""

from __future__ import annotations

import logging

from .agents.base import AgentRuntime
from .agents.runtime_deterministic import DeterministicRuntime
from .config import Settings, get_settings
from .infra.artifacts import ArtifactStore, GcsArtifactStore, LocalArtifactStore
from .infra.bus import InProcessJobBus, JobBus, PubSubJobBus
from .infra.store import FirestoreStateStore, InMemoryStateStore, StateStore
from .orchestrator import Orchestrator
from .worker import ExperimentWorker

log = logging.getLogger("labguard.services")


def build_store(settings: Settings) -> StateStore:
    if settings.is_cloud and settings.google_cloud_project:
        try:
            return FirestoreStateStore(
                settings.google_cloud_project, settings.firestore_root, settings.firestore_database
            )
        except Exception as exc:  # pragma: no cover - depends on cloud creds
            log.error("Firestore unavailable (%s); falling back to the in-memory store", exc)
    return InMemoryStateStore()


def build_bus(settings: Settings) -> JobBus:
    if settings.is_cloud and settings.google_cloud_project:
        try:
            return PubSubJobBus(settings.google_cloud_project, settings.pubsub_jobs_topic, settings.pubsub_events_topic)
        except Exception as exc:  # pragma: no cover - depends on cloud creds
            log.error("Pub/Sub unavailable (%s); falling back to the in-process bus", exc)
    return InProcessJobBus(latency_seconds=settings.simulated_queue_latency)


def build_artifacts(settings: Settings) -> ArtifactStore:
    if settings.is_cloud and settings.gcs_bucket:
        try:
            return GcsArtifactStore(settings.gcs_bucket)
        except Exception as exc:  # pragma: no cover - depends on cloud creds
            log.error("Cloud Storage unavailable (%s); writing artifacts locally", exc)
    return LocalArtifactStore(settings.artifact_dir)


def build_runtime(settings: Settings) -> AgentRuntime:
    """Gemini through ADK when credentials exist, the rule engine otherwise."""
    if settings.gemini_enabled:
        try:
            from .agents.gemini_runtime import AdkAgentRuntime

            return AdkAgentRuntime(settings)
        except Exception as exc:  # pragma: no cover - depends on credentials
            log.error("Gemini/ADK runtime unavailable (%s); using the deterministic runtime", exc)
    return DeterministicRuntime()


class Services:
    """Everything the API and the worker share for one process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = build_store(self.settings)
        self.bus = build_bus(self.settings)
        self.artifacts = build_artifacts(self.settings)
        self.runtime = build_runtime(self.settings)
        self.orchestrator = Orchestrator(self.store, self.bus, self.artifacts, self.runtime, self.settings)
        self.worker = ExperimentWorker(self.store, self.artifacts, self.orchestrator)

    async def start(self) -> None:
        """Attach the worker to the bus (a no-op for push transports)."""
        await self.bus.start(self.worker.handle_message)

    async def close(self) -> None:
        await self.bus.close()

    @property
    def infrastructure(self) -> dict[str, str]:
        """What is actually wired up, for the dashboard's status strip."""
        return {
            "mode": "cloud" if self.settings.is_cloud else "demo",
            "state_store": type(self.store).__name__,
            "job_bus": type(self.bus).__name__,
            "artifact_store": type(self.artifacts).__name__,
            "reasoning": self.runtime.name,
            "project": self.settings.google_cloud_project or "(none)",
        }
