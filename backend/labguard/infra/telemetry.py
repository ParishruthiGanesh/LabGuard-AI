"""Logging setup.

On Cloud Run, structured JSON on stdout is picked up by Cloud Logging with the
right severity and trace, so that is the default. `ENABLE_CLOUD_LOGGING`
additionally attaches the Cloud Logging handler, which is what you want when
running somewhere that is not Cloud Run but still reports into the project.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

from ..config import Settings

#: Python level names to the severities Cloud Logging understands.
_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class StructuredFormatter(logging.Formatter):
    """Emit one JSON object per line, in the shape Cloud Logging expects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "severity": _SEVERITY.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Cloud Logging correlates entries to a request through this field.
        trace = getattr(record, "trace", None) or os.getenv("X_CLOUD_TRACE_CONTEXT")
        if trace:
            payload["logging.googleapis.com/trace"] = trace
        for key, value in getattr(record, "labguard", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings) -> str:
    """Set up logging for this process. Returns which sink was attached."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO)

    if settings.enable_cloud_logging and settings.google_cloud_project:
        try:
            import google.cloud.logging  # imported lazily: cloud-only dep

            client = google.cloud.logging.Client(project=settings.google_cloud_project)
            client.setup_logging(log_level=logging.INFO)
            logging.getLogger("labguard").info("Cloud Logging attached")
            return "cloud-logging"
        except Exception as exc:  # pragma: no cover - depends on cloud creds
            logging.getLogger("labguard").warning(
                "Cloud Logging unavailable (%s); falling back to structured stdout", exc
            )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)
    # Uvicorn installs its own handlers; let them propagate to ours instead.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    return "structured-stdout"
