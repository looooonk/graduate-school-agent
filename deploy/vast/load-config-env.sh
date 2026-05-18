#!/usr/bin/env bash
set -euo pipefail

vast_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vast_repo_root="$(cd "${vast_script_dir}/../.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-${vast_repo_root}/config.yaml}"
VAST_CONFIG_PYTHON="${VAST_CONFIG_PYTHON:-python3}"

eval "$("${VAST_CONFIG_PYTHON}" "${vast_script_dir}/config-env.py" "${CONFIG_PATH}")"
