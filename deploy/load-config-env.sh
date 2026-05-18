#!/usr/bin/env bash
set -euo pipefail

deploy_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
deploy_repo_root="$(cd "${deploy_script_dir}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-${deploy_repo_root}/config.yaml}"
DEPLOY_CONFIG_PYTHON="${DEPLOY_CONFIG_PYTHON:-python3}"

eval "$("${DEPLOY_CONFIG_PYTHON}" "${deploy_script_dir}/config-env.py" "${CONFIG_PATH}")"
