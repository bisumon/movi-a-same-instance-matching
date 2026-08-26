#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_dir="${MOVI_DE_VENV:-${repository_dir}/.venv-movi-de}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_dir}"
fi
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install -r "${repository_dir}/requirements-lock.txt"
exec "${venv_dir}/bin/python" "${repository_dir}/run_movi_de_experiment.py" "$@"
