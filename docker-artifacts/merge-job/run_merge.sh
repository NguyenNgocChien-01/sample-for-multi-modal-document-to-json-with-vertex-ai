#!/bin/bash
set -euo pipefail
pip install -q -U "transformers>=4.49" "peft>=0.14" accelerate safetensors
mkdir -p /workspace/checkpoint
gsutil -m cp -r "gs://electric-bill-dataset-gcs/output/model/v0-20260623-161620/checkpoint-350/*" /workspace/checkpoint/
gsutil cp "gs://electric-bill-dataset-gcs/output/model/v0-20260623-161620/args.json" /workspace/checkpoint/
echo "====> Free memory before merge:"
free -h
rm -rf /workspace/merged-model
echo "====> Starting merge with peft..."
python3 -u /workspace/merge.py
echo "====> Contents:"
ls -la /workspace/merged-model/
gsutil -m cp -r /workspace/merged-model/* gs://electric-bill-dataset-gcs/output/merged/v0-20260623-161620/
echo "====> Upload complete."
