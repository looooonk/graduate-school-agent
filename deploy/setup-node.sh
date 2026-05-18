#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
. "${script_dir}/load-config-env.sh"

if command -v apt-get >/dev/null 2>&1 && [[ -n "$DEPLOY_SYSTEM_PACKAGES" ]]; then
  sudo_cmd=()
  if [[ "$(id -u)" -ne 0 ]]; then
    sudo_cmd=(sudo)
  fi
  "${sudo_cmd[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${sudo_cmd[@]}" apt-get install -y $DEPLOY_SYSTEM_PACKAGES
fi

micromamba_bin="$(command -v micromamba || true)"
if [[ -z "$micromamba_bin" && -x "${HOME}/.local/bin/micromamba" ]]; then
  micromamba_bin="${HOME}/.local/bin/micromamba"
fi

if [[ -z "$micromamba_bin" ]]; then
  install_dir="${HOME}/.local/bin"
  mkdir -p "$install_dir"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$install_dir" --strip-components=1 bin/micromamba
  micromamba_bin="${install_dir}/micromamba"
fi

if ! "$micromamba_bin" env list | awk '{print $1}' | grep -qx "$DEPLOY_MICROMAMBA_ENV"; then
  "$micromamba_bin" create -y -n "$DEPLOY_MICROMAMBA_ENV" "python=${DEPLOY_PYTHON_VERSION}" pip
fi

"$micromamba_bin" run -n "$DEPLOY_MICROMAMBA_ENV" python -m pip install --upgrade pip
"$micromamba_bin" run -n "$DEPLOY_MICROMAMBA_ENV" python -m pip install -e "$repo_root"

if [[ -n "$DEPLOY_PIP_PACKAGES" ]]; then
  "$micromamba_bin" run -n "$DEPLOY_MICROMAMBA_ENV" python -m pip install $DEPLOY_PIP_PACKAGES
fi

mkdir -p "$repo_root/input" "$repo_root/output" "$repo_root/logs"
"$micromamba_bin" run -n "$DEPLOY_MICROMAMBA_ENV" grad-agent --help >/dev/null
"$micromamba_bin" run -n "$DEPLOY_MICROMAMBA_ENV" python -c "import vllm" >/dev/null

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "warning: nvidia-smi was not found; verify the node exposes GPUs before starting vLLM" >&2
fi

cat <<EOF
Setup complete.

Start vLLM:
  deploy/start-vllm.sh

Check endpoints:
  deploy/healthcheck.sh

Run the agent after setting ANTHROPIC_API_KEY and BRAVE_API_KEY:
  ${micromamba_bin} run -n ${DEPLOY_MICROMAMBA_ENV} grad-agent --schools input/schools.json --cv input/cv.md
EOF
