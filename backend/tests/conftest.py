from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.config import Settings
from labguard.services import Services


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        LABGUARD_MODE="demo",
        SIMULATED_QUEUE_LATENCY=0.0,
        SIMULATED_EPOCH_DELAY=0.0,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
    )


# `loop_scope="module"` matters: modules that pin their tests to a module-scoped
# event loop would otherwise get a Services built on a different loop, leaving
# the job bus consumer task on a loop that never runs.
@pytest_asyncio.fixture(loop_scope="module")
async def services(settings) -> Services:
    svc = Services(settings)
    await svc.start()
    try:
        yield svc
    finally:
        await svc.close()
