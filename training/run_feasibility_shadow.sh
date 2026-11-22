#!/usr/bin/env bash
# Run the original main-package MEA with a read-only, non-enforcing observer.
set -euo pipefail

reflector_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${reflector_repo}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

if [[ -n "${LH_HARNESS_PYTHON:-}" ]]; then
  exec "$LH_HARNESS_PYTHON" -u -m lh_harness.feasibility.shadow_run "$@"
elif [[ -n "${CONDA_ENV:-}" ]]; then
  exec conda run --no-capture-output -n "$CONDA_ENV" python -u -m lh_harness.feasibility.shadow_run "$@"
elif [[ -x "$reflector_repo/.venv/bin/python" ]]; then
  exec "$reflector_repo/.venv/bin/python" -u -m lh_harness.feasibility.shadow_run "$@"
else
  exec python -u -m lh_harness.feasibility.shadow_run "$@"
fi
