#!/bin/bash
set -x
# This script downloads a pre-merged model from GCS and serves it with vLLM
# on Vertex AI. No LoRA merging needed — model is already merged at deploy time.

PREFIX="SM_VLLM_"
ARG_PREFIX="--"

# Initialize an array for storing the arguments
# port 8080 required by Vertex AI
echo "Starting server on port $PORT"
ARGS=(--port $PORT)

# Loop through all environment variables
while IFS='=' read -r key value; do
    # Remove the prefix from the key, convert to lowercase, and replace underscores with dashes
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    # Add the argument name and value to the ARGS array
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

# Construct the environment variable name
ENV_VAR_NAME="ADAPTER_URI"

# Get the value from the environment variable — points to pre-merged model on GCS
MODEL_URI="${!ENV_VAR_NAME}"

# Check if the environment variable exists and is not empty
if [ -z "$MODEL_URI" ]; then
    echo "Error: Environment variable ${ENV_VAR_NAME} is not set or empty"
    exit 1
fi

# Download the pre-merged model from GCS
MODEL_DIR=/opt/ml/model
mkdir -p $MODEL_DIR

export USE_HF_TRANSFER=1

echo "====> Downloading pre-merged model from GCS: $MODEL_URI"
gsutil -m cp -r "${MODEL_URI}/*" "$MODEL_DIR/"
if [ $? -ne 0 ]; then
    echo "====> ERROR: gsutil cp failed. Check Service Account permissions on bucket."
    exit 1
fi
echo "====> Download complete. Model dir contents:"
ls -lh "$MODEL_DIR"

# Detect number of available GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "====> Detected $NUM_GPUS GPUs available"
else
    NUM_GPUS=1
    echo "====> No GPUs detected, defaulting to 1"
fi

GPU_MEMORY_UTILIZATION=0.95

echo "====> Starting vllm serve with merged model: $MODEL_DIR"
# Serve the pre-merged model directly with vLLM — no LoRA merging needed
exec vllm serve \
  "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --no-skip-tokenizer-init \
  --port 8080 \
  --host 0.0.0.0 \
  --tokenizer-mode auto \
  --trust-remote-code \
  --load-format auto \
  --kv-cache-dtype auto \
  --seed 0 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size $NUM_GPUS \
  --swap-space 4 \
  --cpu-offload-gb 0 \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-logprobs 20 \
  --disable-log-stats \
  --max-seq-len-to-capture 8192 \
  --guided-decoding-backend xgrammar \
  --disable-async-output-proc \
  --scheduling-policy fcfs \
  --disable-log-requests \
  "${ARGS[@]}"