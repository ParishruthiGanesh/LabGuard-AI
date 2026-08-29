"""Runtime configuration.

LabGuard runs in one of two topologies driven entirely by environment
variables.  `demo` uses in-process adapters so the whole claim-to-verdict
workflow runs with no cloud project; `cloud` swaps in Firestore, Pub/Sub,
Cloud Storage and Gemini.  Nothing above the adapter layer knows which is
active.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -- topology ---------------------------------------------------------
    #: "demo" (in-process, deterministic) or "cloud" (Firestore/Pub/Sub/GCS).
    deployment_mode: str = Field(default="demo", alias="LABGUARD_MODE")
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_region: str = Field(default="us-central1", alias="GOOGLE_CLOUD_REGION")

    # -- Gemini -----------------------------------------------------------
    gemini_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    #: Primary model id. The client falls back through `gemini_model_fallbacks`
    #: if the deployment does not expose this exact id.
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")
    gemini_model_fallbacks: str = Field(
        default="gemini-flash-latest,gemini-2.5-flash",
        alias="GEMINI_MODEL_FALLBACKS",
    )
    use_vertex_ai: bool = Field(default=False, alias="GOOGLE_GENAI_USE_VERTEXAI")

    # -- infrastructure ---------------------------------------------------
    firestore_database: str = Field(default="(default)", alias="FIRESTORE_DATABASE")
    firestore_root: str = Field(default="labguard_claims", alias="FIRESTORE_ROOT")
    pubsub_jobs_topic: str = Field(default="labguard-jobs", alias="PUBSUB_JOBS_TOPIC")
    pubsub_events_topic: str = Field(default="labguard-events", alias="PUBSUB_EVENTS_TOPIC")
    worker_push_url: str = Field(default="", alias="WORKER_PUSH_URL")
    gcs_bucket: str = Field(default="", alias="GCS_BUCKET")
    artifact_dir: str = Field(default="./artifacts", alias="ARTIFACT_DIR")
    enable_cloud_logging: bool = Field(default=False, alias="ENABLE_CLOUD_LOGGING")

    # -- behaviour --------------------------------------------------------
    #: Seconds of simulated queue latency so demo runs show real transitions.
    simulated_queue_latency: float = Field(default=0.6, alias="SIMULATED_QUEUE_LATENCY")
    simulated_epoch_delay: float = Field(default=0.045, alias="SIMULATED_EPOCH_DELAY")
    max_planning_rounds: int = Field(default=4, alias="MAX_PLANNING_ROUNDS")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    #: Shared secret required by the worker push endpoint in cloud mode.
    worker_shared_secret: str = Field(default="", alias="WORKER_SHARED_SECRET")

    @property
    def is_cloud(self) -> bool:
        return self.deployment_mode.strip().lower() == "cloud"

    @property
    def gemini_enabled(self) -> bool:
        """Gemini is used when a key is present, or on Vertex with a project."""
        if self.use_vertex_ai:
            return bool(self.google_cloud_project)
        return bool(self.gemini_api_key)

    @property
    def model_candidates(self) -> list[str]:
        seen: list[str] = []
        for name in [self.gemini_model, *self.gemini_model_fallbacks.split(",")]:
            cleaned = name.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that mutate the environment."""
    get_settings.cache_clear()
