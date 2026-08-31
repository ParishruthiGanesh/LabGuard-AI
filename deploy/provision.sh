#!/usr/bin/env bash
# Provision the Google Cloud resources LabGuard AI needs, then deploy.
#
#   ./deploy/provision.sh my-project us-central1
#
# Safe to re-run: every step is idempotent.
set -euo pipefail

PROJECT="${1:-${GOOGLE_CLOUD_PROJECT:-}}"
REGION="${2:-${GOOGLE_CLOUD_REGION:-us-central1}}"
REPO="labguard"
BUCKET="${PROJECT}-labguard-artifacts"
JOBS_TOPIC="labguard-jobs"
EVENTS_TOPIC="labguard-events"

if [[ -z "${PROJECT}" ]]; then
  echo "usage: $0 <project-id> [region]" >&2
  exit 2
fi

echo "==> Project ${PROJECT}, region ${REGION}"
gcloud config set project "${PROJECT}" >/dev/null

echo "==> Enabling APIs"
gcloud services enable \
  run.googleapis.com \
  pubsub.googleapis.com \
  firestore.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  logging.googleapis.com

echo "==> Artifact Registry"
gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker --location="${REGION}" \
    --description="LabGuard AI container images"

echo "==> Firestore (Native mode)"
gcloud firestore databases describe --database='(default)' >/dev/null 2>&1 || \
  gcloud firestore databases create --location="${REGION}" --type=firestore-native

echo "==> Cloud Storage bucket for artifacts and reports"
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" --uniform-bucket-level-access

echo "==> Pub/Sub topics"
for topic in "${JOBS_TOPIC}" "${EVENTS_TOPIC}"; do
  gcloud pubsub topics describe "${topic}" >/dev/null 2>&1 || gcloud pubsub topics create "${topic}"
done

echo "==> Service accounts"
for sa in labguard-api labguard-worker labguard-pubsub; do
  gcloud iam service-accounts describe "${sa}@${PROJECT}.iam.gserviceaccount.com" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "${sa}" --display-name="LabGuard ${sa}"
done

echo "==> IAM: least privilege for each role"
# The API reads and writes claim state, publishes jobs, and calls Gemini.
for role in roles/datastore.user roles/pubsub.publisher roles/storage.objectAdmin roles/aiplatform.user roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:labguard-api@${PROJECT}.iam.gserviceaccount.com" \
    --role="${role}" --condition=None >/dev/null
done
# The worker runs experiments: same state and storage access, no need to be public.
for role in roles/datastore.user roles/pubsub.publisher roles/storage.objectAdmin roles/aiplatform.user roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:labguard-worker@${PROJECT}.iam.gserviceaccount.com" \
    --role="${role}" --condition=None >/dev/null
done

echo "==> IAM: let Cloud Build deploy Cloud Run and act as the runtime accounts"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
# Depending on when the project was created, builds run as either the legacy
# Cloud Build account or the default compute account. Grant both; the one that
# does not exist is skipped.
BUILD_SAS=(
  "${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
  "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
)
for build_sa in "${BUILD_SAS[@]}"; do
  gcloud iam service-accounts describe "${build_sa}" >/dev/null 2>&1 || continue
  for role in roles/run.admin roles/artifactregistry.writer roles/logging.logWriter; do
    gcloud projects add-iam-policy-binding "${PROJECT}" \
      --member="serviceAccount:${build_sa}" --role="${role}" --condition=None >/dev/null
  done
  # `gcloud run deploy --service-account=X` needs actAs on X.
  for runtime_sa in labguard-api labguard-worker; do
    gcloud iam service-accounts add-iam-policy-binding \
      "${runtime_sa}@${PROJECT}.iam.gserviceaccount.com" \
      --member="serviceAccount:${build_sa}" \
      --role=roles/iam.serviceAccountUser >/dev/null
  done
done

echo "==> Building and deploying (Cloud Build)"
TAG="$(git rev-parse --short HEAD 2>/dev/null || echo manual)"
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions="_REGION=${REGION},_REPO=${REPO},_BUCKET=${BUCKET},_TAG=${TAG}"

WORKER_URL="$(gcloud run services describe labguard-worker --region="${REGION}" --format='value(status.url)')"
API_URL="$(gcloud run services describe labguard-api --region="${REGION}" --format='value(status.url)')"
DASH_URL="$(gcloud run services describe labguard-dashboard --region="${REGION}" --format='value(status.url)')"

echo "==> Pointing the dashboard at the API"
# Cloud Build also sets this, but doing it here from a known-good value means a
# failed shell expansion inside the build cannot leave the dashboard proxying
# to an empty URL (which 404s every API call).
gcloud run services update labguard-dashboard \
  --region="${REGION}" \
  --update-env-vars="LABGUARD_API_URL=${API_URL}" >/dev/null

echo "==> Pub/Sub push subscription -> worker"
# The push service account needs permission to invoke the private worker.
gcloud run services add-iam-policy-binding labguard-worker \
  --region="${REGION}" \
  --member="serviceAccount:labguard-pubsub@${PROJECT}.iam.gserviceaccount.com" \
  --role=roles/run.invoker >/dev/null

gcloud pubsub subscriptions describe labguard-jobs-push >/dev/null 2>&1 || \
  gcloud pubsub subscriptions create labguard-jobs-push \
    --topic="${JOBS_TOPIC}" \
    --push-endpoint="${WORKER_URL}/internal/pubsub/push" \
    --push-auth-service-account="labguard-pubsub@${PROJECT}.iam.gserviceaccount.com" \
    --ack-deadline=600 \
    --message-retention-duration=1h

cat <<EOF

Deployed.

  Dashboard : ${DASH_URL}
  API       : ${API_URL}
  Worker    : ${WORKER_URL}  (private; invoked by Pub/Sub push only)
  Artifacts : gs://${BUCKET}

Check it is healthy:
  curl -s "${API_URL}/health" | python3 -m json.tool
EOF
