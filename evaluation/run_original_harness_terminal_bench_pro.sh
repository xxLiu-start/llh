#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env="${CONDA_ENV:-terminal-bench-2-1}"
output_root="${ORIGINAL_HARNESS_EVAL_OUTPUT:-${repo_dir}/output/origin}"

export PYTHONPATH="${repo_dir}/src:${repo_dir}/eval/TB-harness/Harness/src${PYTHONPATH:+:${PYTHONPATH}}"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

proxy_bypass="${NO_PROXY:-}"
if [[ -n "${no_proxy:-}" ]]; then
  proxy_bypass="${proxy_bypass:+${proxy_bypass},}${no_proxy}"
fi
proxy_bypass="${proxy_bypass:+${proxy_bypass},}127.0.0.1,localhost"
export NO_PROXY="$proxy_bypass" no_proxy="$proxy_bypass"

cd "$repo_dir"
if [[ -z "${ANTHROPIC_API_KEY:-}" || -z "${ANTHROPIC_BASE_URL:-}" ]]; then
  echo "ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL are required" >&2
  exit 2
fi

mkdir -p "$output_root/jobs/console_logs" "$output_root/generated_configs" "$output_root/results/tasks"
log_file="$(mktemp "$output_root/jobs/console_logs/eval-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
printf 'Terminal output log: %s\n' "$log_file" | tee "$log_file"

conda run --no-capture-output -n "$conda_env" \
  python -m lh_harness.eval_terminal_bench_pro_original \
  --output "$output_root" "$@" 2>&1 | tee -a "$log_file"
