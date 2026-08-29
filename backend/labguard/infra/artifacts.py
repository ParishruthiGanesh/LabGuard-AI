"""Artifact storage for reports, run logs, curves and configurations."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactStore(ABC):
    @abstractmethod
    async def write_json(self, path: str, payload: Any) -> str: ...

    @abstractmethod
    async def write_text(self, path: str, text: str, content_type: str = "text/plain") -> str: ...

    @abstractmethod
    async def read_text(self, path: str) -> str | None: ...


class LocalArtifactStore(ArtifactStore):
    """Writes under `ARTIFACT_DIR`; URIs are `file://` paths."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        target = (self._root / path).resolve()
        root = self._root.resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"artifact path escapes root: {path}")
        return target

    async def write_json(self, path: str, payload: Any) -> str:
        return await self.write_text(path, json.dumps(payload, indent=2, default=str), "application/json")

    async def write_text(self, path: str, text: str, content_type: str = "text/plain") -> str:
        target = self._resolve(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

        await asyncio.to_thread(_write)
        return f"file://{target}"

    async def read_text(self, path: str) -> str | None:
        target = self._resolve(path)
        if not target.exists():
            return None
        return await asyncio.to_thread(target.read_text, "utf-8")


class GcsArtifactStore(ArtifactStore):
    """Cloud Storage-backed artifacts; URIs are `gs://bucket/path`."""

    def __init__(self, bucket: str) -> None:
        from google.cloud import storage  # imported lazily: cloud-only dep

        self._client = storage.Client()
        self._bucket_name = bucket
        self._bucket = self._client.bucket(bucket)

    async def write_json(self, path: str, payload: Any) -> str:
        return await self.write_text(path, json.dumps(payload, indent=2, default=str), "application/json")

    async def write_text(self, path: str, text: str, content_type: str = "text/plain") -> str:
        blob = self._bucket.blob(path)
        await asyncio.to_thread(blob.upload_from_string, text, content_type)
        return f"gs://{self._bucket_name}/{path}"

    async def read_text(self, path: str) -> str | None:
        blob = self._bucket.blob(path)
        exists = await asyncio.to_thread(blob.exists)
        if not exists:
            return None
        return await asyncio.to_thread(blob.download_as_text)
