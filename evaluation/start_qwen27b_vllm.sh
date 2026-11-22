#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
inference_python="${LLH_VLLM_PYTHON:-${repo_dir}/../../.cache/a-evolve-qwen-vllm/bin/python}"
if [[ ! -x "$inference_python" ]]; then
  echo "Set LLH_VLLM_PYTHON to the Python executable in the installed vLLM environment." >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$inference_python" -m lh_harness.serve_local_qwen "$@"
