#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/src:$repo_dir/eval/TB-harness/Harness/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec conda run --no-capture-output -n "${CONDA_ENV:-terminal-bench-2-1}" \
  python experiments/skill_evolution/scripts/train_skills.py "$@"
