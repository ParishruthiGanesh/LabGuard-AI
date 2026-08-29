"""Asynchronous job bus.

The orchestrator never executes an experiment inline: it publishes a job
message and returns.  In cloud mode that message goes to Pub/Sub, which push-
delivers it to the worker Cloud Run service.  In demo mode an asyncio queue
delivers the identical payload to the identical handler, after a simulated
delay, so the dashboard shows the same queued -> running transitions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger("labguard.bus")

JobHandler = Callable[[dict[str, Any]], Awaitable[None]]


class JobBus(ABC):
    @abstractmethod
    async def publish_job(self, message: dict[str, Any]) -> str: ...

    @abstractmethod
    async def publish_event(self, message: dict[str, Any]) -> str: ...

    async def start(self, handler: JobHandler) -> None:
        """Attach the worker handler. No-op for push-based transports."""

    async def close(self) -> None:
        """Release transport resources."""


class InProcessJobBus(JobBus):
    """Demo transport: an asyncio queue with a simulated delivery delay.

    Delivery is genuinely asynchronous — `publish_job` returns immediately and
    the job is executed later by a background consumer task — so the state
    machine exercises exactly the same paths as the Pub/Sub transport.
    """

    def __init__(self, latency_seconds: float = 0.6) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._latency = latency_seconds
        self._handler: JobHandler | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._inflight: set[asyncio.Task[None]] = set()
        #: Messages taken off the queue but not yet finished. Counted
        #: separately from `_inflight` so `drain` cannot observe the gap
        #: between dequeue and task creation.
        self._pending = 0
        self.published_events: list[dict[str, Any]] = []

    async def publish_job(self, message: dict[str, Any]) -> str:
        await self._queue.put(message)
        return f"local-{id(message)}"

    async def publish_event(self, message: dict[str, Any]) -> str:
        self.published_events.append(message)
        log.info("event %s", json.dumps(message, default=str)[:400])
        return f"local-event-{len(self.published_events)}"

    async def start(self, handler: JobHandler) -> None:
        self._handler = handler
        if self._consumer is None or self._consumer.done():
            self._consumer = asyncio.create_task(self._consume(), name="labguard-bus")

    async def _consume(self) -> None:
        while True:
            message = await self._queue.get()
            self._pending += 1
            if self._latency:
                await asyncio.sleep(self._latency)
            task = asyncio.create_task(self._dispatch(message))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _dispatch(self, message: dict[str, Any]) -> None:
        if self._handler is None:
            log.warning("dropping job, no handler attached: %s", message)
            self._pending -= 1
            return
        try:
            await self._handler(message)
        except Exception:  # pragma: no cover - defensive; worker logs detail
            log.exception("job handler failed for %s", message.get("job_id"))
        finally:
            self._pending -= 1

    async def drain(self, timeout: float = 60.0) -> None:
        """Wait until the queue is empty and all dispatched jobs finished."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if self._queue.empty() and not self._inflight and self._pending == 0:
                return
            await asyncio.sleep(0.05)
        raise TimeoutError("job bus did not drain in time")

    async def close(self) -> None:
        if self._consumer:
            self._consumer.cancel()
        for task in list(self._inflight):
            task.cancel()


class PubSubJobBus(JobBus):
    """Cloud transport: Pub/Sub topics, push-delivered to the worker service."""

    def __init__(self, project: str, jobs_topic: str, events_topic: str) -> None:
        from google.cloud import pubsub_v1  # imported lazily: cloud-only dep

        self._publisher = pubsub_v1.PublisherClient()
        self._jobs_topic = self._publisher.topic_path(project, jobs_topic)
        self._events_topic = self._publisher.topic_path(project, events_topic)

    async def _publish(self, topic: str, message: dict[str, Any]) -> str:
        data = json.dumps(message, default=str).encode("utf-8")
        future = self._publisher.publish(topic, data)
        return await asyncio.to_thread(future.result, 30)

    async def publish_job(self, message: dict[str, Any]) -> str:
        return await self._publish(self._jobs_topic, message)

    async def publish_event(self, message: dict[str, Any]) -> str:
        return await self._publish(self._events_topic, message)
