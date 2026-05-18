#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/load-config-env.sh"

if ! [[ "$VAST_MODEL_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "retrieval.local_model_count must be an integer >= 1" >&2
  exit 1
fi

read -r -a ports <<<"$VAST_VLLM_PORTS"
read -r -a vllm_args <<<"$VAST_VLLM_ARGS"
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  vllm_args+=(--api-key "$VLLM_API_KEY")
fi
if [[ "${#ports[@]}" -ne "$VAST_MODEL_COUNT" ]]; then
  echo "retrieval.local_base_urls must provide one port per local model" >&2
  exit 1
fi

runner=()
if ! command -v vllm >/dev/null 2>&1; then
  micromamba_bin="$(command -v micromamba || true)"
  if [[ -z "$micromamba_bin" && -x "${HOME}/.local/bin/micromamba" ]]; then
    micromamba_bin="${HOME}/.local/bin/micromamba"
  fi
  if [[ -z "$micromamba_bin" ]]; then
    echo "vllm is not on PATH and micromamba is unavailable; run setup-node.sh first" >&2
    exit 1
  fi
  runner=("$micromamba_bin" run -n "$VAST_MICROMAMBA_ENV")
fi

mkdir -p "$VAST_VLLM_LOG_DIR"

for ((gpu = 0; gpu < VAST_MODEL_COUNT; gpu++)); do
  port="${ports[$gpu]}"
  echo "starting ${VAST_MODEL_ID} on GPU ${gpu}, port ${port}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "${runner[@]}" vllm serve "$VAST_MODEL_ID" \
      --host "$VAST_VLLM_HOST" \
      --port "$port" \
      "${vllm_args[@]}" \
      >"${VAST_VLLM_LOG_DIR}/vllm-${gpu}.log" 2>&1 &
done

wait
