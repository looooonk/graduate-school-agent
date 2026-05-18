#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/load-config-env.sh"

if ! [[ "$DEPLOY_MODEL_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "retrieval.local_model_count must be an integer >= 1" >&2
  exit 1
fi

read -r -a ports <<<"$DEPLOY_VLLM_PORTS"
read -r -a endpoints <<<"$DEPLOY_VLLM_ENDPOINTS"
read -r -a vllm_args <<<"$DEPLOY_VLLM_ARGS"
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  vllm_args+=(--api-key "$VLLM_API_KEY")
fi
if [[ "${#ports[@]}" -ne "$DEPLOY_MODEL_COUNT" ]]; then
  echo "retrieval.local_base_urls must provide one port per local model" >&2
  exit 1
fi
if [[ "${#endpoints[@]}" -ne "$DEPLOY_MODEL_COUNT" ]]; then
  echo "retrieval.local_base_urls must provide one endpoint per local model" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for vLLM readiness checks; run setup-node.sh first" >&2
  exit 1
fi

startup_timeout="${DEPLOY_VLLM_STARTUP_TIMEOUT:-600}"
if ! [[ "$startup_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "DEPLOY_VLLM_STARTUP_TIMEOUT must be an integer >= 1" >&2
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
  runner=("$micromamba_bin" run -n "$DEPLOY_MICROMAMBA_ENV")
fi

mkdir -p "$DEPLOY_VLLM_LOG_DIR"

pids=()
log_files=()
ready=()

tail_log() {
  local log_file="$1"
  if [[ -s "$log_file" ]]; then
    echo "recent log output from ${log_file}:" >&2
    tail -n 40 "$log_file" >&2
  else
    echo "log file is empty: ${log_file}" >&2
  fi
}

stop_servers() {
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

handle_stop() {
  echo "stopping vLLM servers" >&2
  stop_servers
  exit 130
}

endpoint_ready() {
  local endpoint="${1%/}"
  local health_url="${endpoint%/v1}/health"
  local models_url="${endpoint}/models"
  local curl_args=(-fsS --max-time 2)

  if curl "${curl_args[@]}" "$health_url" >/dev/null 2>&1; then
    return 0
  fi

  if [[ -n "${VLLM_API_KEY:-}" ]]; then
    curl_args+=(-H "Authorization: Bearer ${VLLM_API_KEY}")
  fi
  curl "${curl_args[@]}" "$models_url" >/dev/null 2>&1
}

process_exit_status() {
  local pid="$1"
  if wait "$pid"; then
    return 0
  else
    return "$?"
  fi
}

trap handle_stop INT TERM

for ((gpu = 0; gpu < DEPLOY_MODEL_COUNT; gpu++)); do
  port="${ports[$gpu]}"
  log_file="${DEPLOY_VLLM_LOG_DIR}/vllm-${gpu}.log"
  echo "starting ${DEPLOY_MODEL_ID} on GPU ${gpu}, port ${port}; log: ${log_file}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "${runner[@]}" vllm serve "$DEPLOY_MODEL_ID" \
      --host "$DEPLOY_VLLM_HOST" \
      --port "$port" \
      "${vllm_args[@]}" \
      >"$log_file" 2>&1 &
  pids[$gpu]=$!
  log_files[$gpu]="$log_file"
  ready[$gpu]=0
done

deadline=$((SECONDS + startup_timeout))
remaining="$DEPLOY_MODEL_COUNT"
echo "waiting up to ${startup_timeout}s for vLLM endpoints to become ready"

while ((remaining > 0)); do
  for ((gpu = 0; gpu < DEPLOY_MODEL_COUNT; gpu++)); do
    if [[ "${ready[$gpu]}" == "1" ]]; then
      continue
    fi

    pid="${pids[$gpu]}"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      if process_exit_status "$pid"; then
        status=1
      else
        status=$?
      fi
      echo "vLLM failed on GPU ${gpu} before readiness; exit status ${status}" >&2
      tail_log "${log_files[$gpu]}"
      stop_servers
      exit "$status"
    fi

    if endpoint_ready "${endpoints[$gpu]}"; then
      ready[$gpu]=1
      remaining=$((remaining - 1))
      echo "ready ${DEPLOY_MODEL_ID} on GPU ${gpu}: ${endpoints[$gpu]}"
    fi
  done

  if ((remaining == 0)); then
    break
  fi
  if ((SECONDS >= deadline)); then
    echo "timed out waiting for ${remaining} vLLM endpoint(s) to become ready" >&2
    for ((gpu = 0; gpu < DEPLOY_MODEL_COUNT; gpu++)); do
      if [[ "${ready[$gpu]}" != "1" ]]; then
        echo "not ready on GPU ${gpu}: ${endpoints[$gpu]}" >&2
        tail_log "${log_files[$gpu]}"
      fi
    done
    stop_servers
    exit 1
  fi
  sleep 2
done

echo "all ${DEPLOY_MODEL_COUNT} vLLM server(s) are ready"

while true; do
  for ((gpu = 0; gpu < DEPLOY_MODEL_COUNT; gpu++)); do
    pid="${pids[$gpu]}"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      if process_exit_status "$pid"; then
        status=0
      else
        status=$?
      fi
      echo "vLLM exited on GPU ${gpu}; exit status ${status}" >&2
      tail_log "${log_files[$gpu]}"
      stop_servers
      exit "$status"
    fi
  done
  sleep 5
done
