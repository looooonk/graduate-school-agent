#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/load-config-env.sh"

if ! [[ "$DEPLOY_MODEL_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "retrieval.local_model_count must be an integer >= 1" >&2
  exit 1
fi

read -r -a endpoints <<<"$DEPLOY_VLLM_ENDPOINTS"
if [[ "${#endpoints[@]}" -ne "$DEPLOY_MODEL_COUNT" ]]; then
  echo "retrieval.local_base_urls must provide one endpoint per local model" >&2
  exit 1
fi

for ((idx = 0; idx < DEPLOY_MODEL_COUNT; idx++)); do
  endpoint="${endpoints[$idx]%/}"
  curl -fsS "${endpoint}/chat/completions" \
    -H "Content-Type: application/json" \
    --data "{
      \"model\": \"${DEPLOY_MODEL_ID}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Return OK.\"}],
      \"max_tokens\": 8,
      \"temperature\": 0
    }" >/dev/null
  echo "${endpoint}: ok"
done
