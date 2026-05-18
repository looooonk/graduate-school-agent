#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B-FP8}"
START_PORT="${START_PORT:-8001}"

for gpu in 0 1 2 3; do
  port=$((START_PORT + gpu))
  curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data "{
      \"model\": \"${MODEL_ID}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Return OK.\"}],
      \"max_tokens\": 8,
      \"temperature\": 0
    }" >/dev/null
  echo "port ${port}: ok"
done

