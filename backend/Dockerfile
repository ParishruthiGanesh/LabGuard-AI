# LabGuard AI — API and experiment worker.
# The same image serves both Cloud Run services; the role is chosen by
# environment (the worker service receives Pub/Sub pushes at
# /internal/pubsub/push, the API service serves the dashboard).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so application edits do not invalidate the layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY labguard ./labguard

# Cloud Run sets PORT; default to 8080 for local `docker run`.
ENV PORT=8080 \
    LABGUARD_MODE=cloud \
    ARTIFACT_DIR=/tmp/artifacts

RUN useradd --create-home --uid 1001 labguard && chown -R labguard /app
USER labguard

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8080') + '/health').read()"

CMD exec uvicorn labguard.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1
