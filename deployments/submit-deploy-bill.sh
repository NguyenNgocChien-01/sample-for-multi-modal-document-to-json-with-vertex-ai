#!/bin/bash
# Wrapper deploy model electric bill — chỉ cần: bash submit-deploy-bill.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SERVICE_NAME="vllm-serve-bill"
ENV_FILE="deploy.env.electric-bill"
IMAGE_URI="asia-southeast1-docker.pkg.dev/first-orc-chien/swift-json-vlm-container-finetuned/inference-bill:latest"

YAML_FILE="env.${SERVICE_NAME}.yaml"
grep -E '^[^#]+=' "$ENV_FILE" | sed -E 's/^([^=]+)=(.*)$/\1: "\2"/' > "$YAML_FILE"

gcloud beta run deploy "$SERVICE_NAME" \
  --image "$IMAGE_URI" \
  --region asia-southeast1 \
  --execution-environment gen2 \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --no-cpu-throttling --cpu 8 --memory 32Gi \
  --min-instances 0 --max-instances 1 --concurrency 4 --timeout 600 \
  --startup-probe "httpGet.path=/health,initialDelaySeconds=30,failureThreshold=60,periodSeconds=10" \
  --env-vars-file "$YAML_FILE" \
  --no-allow-unauthenticated \
  --project first-orc-chien

rm -f "$YAML_FILE"

echo "====> Deployed. Service URL:"
gcloud run services describe "$SERVICE_NAME" --region asia-southeast1 --project first-orc-chien --format 'value(status.url)'
