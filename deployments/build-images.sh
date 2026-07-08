#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PROJECT_ID="first-orc-chien"
REPO="asia-southeast1-docker.pkg.dev/first-orc-chien/swift-json-vlm-container-finetuned"

echo "====> Building inference-bill (model: electric-bill)"
gcloud builds submit --config=cloudbuild-bake.yaml --project="$PROJECT_ID" \
  --substitutions=_MODEL_PATH=gs://electric-bill-dataset-gcs/output/merged/v0-20260623-161620,_IMAGE_TAG="${REPO}/inference-bill:latest" \
  .

echo "====> Building inference-identity (model: identity-doc)"
gcloud builds submit --config=cloudbuild-bake.yaml --project="$PROJECT_ID" \
  --substitutions=_MODEL_PATH=gs://documents-dataset-gcs/merged/v0-20260626-142348,_IMAGE_TAG="${REPO}/inference-identity:latest" \
  .

echo "====> Done. Images:"
echo "  ${REPO}/inference-bill:latest"
echo "  ${REPO}/inference-identity:latest"
