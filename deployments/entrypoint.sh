#!/bin/bash
set -x
PREFIX="SM_VLLM_"
ARG_PREFIX="--"

VERTEX_PORT="${PORT:-8080}"
echo "Starting server on port $VERTEX_PORT"

ARGS=()
while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

MODEL_DIR=/opt/ml/model

# Model đã bake sẵn vào image lúc build — không tải GCS nữa.
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: No baked model found at $MODEL_DIR. Rebuild image with --build-arg MODEL_GCS_PATH=..."
    exit 1
fi
echo "====> Using baked model at: $MODEL_DIR"
ls -lh "$MODEL_DIR"

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "====> Detected $NUM_GPUS GPUs available"
else
    NUM_GPUS=1
    echo "====> No GPUs detected, defaulting to 1"
fi

GPU_MEMORY_UTILIZATION=0.95

echo "====> Starting vllm serve with baked model: $MODEL_DIR"
exec vllm serve \
  "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --no-skip-tokenizer-init \
  --port "$VERTEX_PORT" \
  --host 0.0.0.0 \
  --tokenizer-mode auto \
  --trust-remote-code \
  --load-format auto \
  --kv-cache-dtype auto \
  --seed 0 \
  --tensor-parallel-size "$NUM_GPUS" \
  --swap-space 4 \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-logprobs 20 \
  --disable-log-stats \
  --disable-log-requests \
  --guided-decoding-backend xgrammar \
  "${ARGS[@]}"
