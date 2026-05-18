#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B-FP8}"
START_PORT="${START_PORT:-8001}"
MODEL_COUNT="${MODEL_COUNT:-1}"

if ! [[ "$MODEL_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODEL_COUNT must be an integer >= 1" >&2
  exit 1
fi

for ((gpu = 0; gpu < MODEL_COUNT; gpu++)); do
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
