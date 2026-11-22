#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}/src:${repo_dir}/eval/TB-harness/Harness/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "$repo_dir"
exec conda run --no-capture-output -n "${CONDA_ENV:-terminal-bench-2-1}" \
  python -m lh_harness.eval_inspect_tb2_original "$@"
