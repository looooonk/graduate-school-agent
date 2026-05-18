#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B-FP8}"
HOST="${HOST:-0.0.0.0}"
START_PORT="${START_PORT:-8001}"
VLLM_ARGS="${VLLM_ARGS:---trust-remote-code}"
LOG_DIR="${LOG_DIR:-logs/vllm}"

mkdir -p "$LOG_DIR"

for gpu in 0 1 2 3; do
  port=$((START_PORT + gpu))
  echo "starting ${MODEL_ID} on GPU ${gpu}, port ${port}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    vllm serve "$MODEL_ID" \
      --host "$HOST" \
      --port "$port" \
      $VLLM_ARGS \
      >"${LOG_DIR}/vllm-${gpu}.log" 2>&1 &
done

wait

