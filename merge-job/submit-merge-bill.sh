#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

gcloud builds submit \
  --config=cloudbuild-merge.yaml \
  --project=first-orc-chien \
  --substitutions=_CHECKPOINT_GCS_PATH="gs://electric-bill-dataset-gcs/output/model/v0-20260623-161620/checkpoint-350",_MERGED_OUTPUT_GCS_PATH="gs://electric-bill-dataset-gcs/output/merged/v0-20260623-161620" \
  .
