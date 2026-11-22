#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
exec conda run --no-capture-output -n "${CONDA_ENV:-terminal-bench-2-1}" python \
  experiments/skill_evolution/scripts/run_paired_eval.py --track skill_enhanced "$@"
