#!/bin/bash
set -euo pipefail

# ---- Parse arguments ----
ADAPTER_GCS_PATH=""
OUTPUT_GCS_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter_path)
      ADAPTER_GCS_PATH="$2"
      shift 2
      ;;
    --output_gcs_path)
      OUTPUT_GCS_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$ADAPTER_GCS_PATH" || -z "$OUTPUT_GCS_PATH" ]]; then
  echo "ERROR: --adapter_path and --output_gcs_path are required."
  echo "Usage: bash run_merge.sh --adapter_path gs://... --output_gcs_path gs://..."
  exit 1
fi

echo "====> Adapter source: ${ADAPTER_GCS_PATH}"
echo "====> Output target:  ${OUTPUT_GCS_PATH}"

# Resolve paths relative to THIS script's location instead of a hardcoded
# /workspace/... path. On Windows Git Bash, /workspace gets silently
# rewritten to the Git install directory (e.g. D:\Git\workspace), which is
# never where this project actually lives.
#
# Use `pwd -W` (Git Bash builtin) to get a Windows-style path (D:/...).
# This matters because Git Bash auto-converts POSIX-style argv paths to
# Windows paths when calling a native .exe, but does NOT do this for
# environment variables -- so ADAPTER_DIR/OUTPUT_DIR must already be in
# Windows form or python.exe will fail to find them.
if SCRIPT_DIR_W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W 2>/dev/null)"; then
  SCRIPT_DIR="$SCRIPT_DIR_W"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
WORKSPACE_DIR="${SCRIPT_DIR}/workspace"
MERGE_PY="${SCRIPT_DIR}/merge.py"

pip install -q -U "transformers==4.49.0" "peft>=0.14" accelerate safetensors

rm -rf "${WORKSPACE_DIR}/checkpoint"
mkdir -p "${WORKSPACE_DIR}/checkpoint"

# Copy the checkpoint dir contents (adapter_config.json, adapter_model.safetensors, etc.)
gsutil -m cp -r "${ADAPTER_GCS_PATH%/}/*" "${WORKSPACE_DIR}/checkpoint/"

# If the adapter_path pointed at the parent training-run dir (not the
# checkpoint-N dir itself), the actual adapter files end up nested one level
# deeper, e.g. checkpoint/checkpoint-300/adapter_config.json. Flatten that
# up so downstream code finds adapter_config.json directly under checkpoint/.
if [[ ! -f "${WORKSPACE_DIR}/checkpoint/adapter_config.json" ]]; then
  NESTED_CKPT=$(find "${WORKSPACE_DIR}/checkpoint" -maxdepth 1 -type d -name "checkpoint-*" | head -n 1)
  if [[ -n "$NESTED_CKPT" && -f "${NESTED_CKPT}/adapter_config.json" ]]; then
    echo "====> Detected nested checkpoint dir, flattening: ${NESTED_CKPT}"
    mv "${NESTED_CKPT}"/* "${WORKSPACE_DIR}/checkpoint/"
  fi
fi

echo "====> Checkpoint contents downloaded:"
ls -la "${WORKSPACE_DIR}/checkpoint/"

if [[ ! -f "${WORKSPACE_DIR}/checkpoint/adapter_config.json" ]]; then
  echo "ERROR: adapter_config.json not found in ${WORKSPACE_DIR}/checkpoint/ after download/flatten."
  echo "Check --adapter_path points at the checkpoint dir (or its direct parent)."
  exit 1
fi

echo "====> Free memory before merge:"
free -h || echo "(free not available, skipping)"

rm -rf "${WORKSPACE_DIR}/merged-model"
echo "====> Starting merge with peft..."

# Windows Git Bash: 'python'/'python3' on PATH may resolve to the Microsoft
# Store alias stub instead of a real interpreter, even when a conda env is
# active. Prefer the conda env's actual python.exe if CONDA_PREFIX is set.
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/python.exe" ]]; then
  PY_BIN="${CONDA_PREFIX}/python.exe"
elif [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/bin/python" ]]; then
  PY_BIN="${CONDA_PREFIX}/bin/python"
elif command -v python3 &>/dev/null && python3 -c "" &>/dev/null; then
  PY_BIN=python3
elif command -v python &>/dev/null && python -c "" &>/dev/null; then
  PY_BIN=python
else
  echo "ERROR: could not find a working Python interpreter."
  echo "CONDA_PREFIX=${CONDA_PREFIX:-<not set>}"
  echo "Try: conda activate ocr   (or run 'which python' to diagnose)"
  exit 1
fi

if [[ ! -f "$MERGE_PY" ]]; then
  echo "ERROR: merge.py not found at ${MERGE_PY}"
  echo "Expected it next to run_merge.sh. If it lives elsewhere, edit MERGE_PY above."
  exit 1
fi

echo "====> Using Python: ${PY_BIN}"
echo "====> Using merge.py: ${MERGE_PY}"
export ADAPTER_DIR="${WORKSPACE_DIR}/checkpoint"
export OUTPUT_DIR="${WORKSPACE_DIR}/merged-model"
echo "====> ADAPTER_DIR=${ADAPTER_DIR}"
echo "====> OUTPUT_DIR=${OUTPUT_DIR}"
"$PY_BIN" -u "$MERGE_PY"

echo "====> Contents:"
ls -la "${WORKSPACE_DIR}/merged-model/"

gsutil -m cp -r "${WORKSPACE_DIR}/merged-model"/* "${OUTPUT_GCS_PATH%/}/"
echo "====> Upload complete to ${OUTPUT_GCS_PATH}"