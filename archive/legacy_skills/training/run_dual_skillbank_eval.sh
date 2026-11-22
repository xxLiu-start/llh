#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env="${CONDA_ENV:-terminal-bench-2-1}"
export PYTHONPATH="${repo_dir}/src:${repo_dir}/eval/TB-harness/Harness/src${PYTHONPATH:+:${PYTHONPATH}}"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
proxy_bypass="${NO_PROXY:-}"
if [[ -n "${no_proxy:-}" ]]; then
  proxy_bypass="${proxy_bypass:+${proxy_bypass},}${no_proxy}"
fi
proxy_bypass="${proxy_bypass:+${proxy_bypass},}127.0.0.1,localhost"
export NO_PROXY="$proxy_bypass" no_proxy="$proxy_bypass"
cd "$repo_dir"

output_root="output/dual-skillbank-eval"
args=("$@")
for ((index=0; index<${#args[@]}; index++)); do
  case "${args[index]}" in
    --output)
      if ((index + 1 < ${#args[@]})); then
        output_root="${args[index + 1]}"
      fi
      ;;
    --output=*) output_root="${args[index]#--output=}" ;;
  esac
done
mkdir -p "$output_root/console_logs"
output_root="$(cd "$output_root" && pwd -P)"
log_file="$(mktemp "$output_root/console_logs/eval-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
printf 'Terminal output log: %s\n' "$log_file" | tee "$log_file"
conda run --no-capture-output -n "$conda_env" python -m lh_harness.eval_terminal_bench_skillbanks "$@" 2>&1 | tee -a "$log_file"
