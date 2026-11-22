#!/usr/bin/env bash
set -euo pipefail
reflector_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${reflector_repo}/src:${reflector_repo}/eval/TB-harness/Harness/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export NO_PROXY="${NO_PROXY:-}${NO_PROXY:+,}${no_proxy:-}${no_proxy:+,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
cd "$reflector_repo"
reflector_output="output/terminal-bench-pro-reflector-shadow"
reflector_args=("$@")
for ((reflector_index=0; reflector_index<${#reflector_args[@]}; reflector_index++)); do
  case "${reflector_args[reflector_index]}" in
    --output) reflector_output="${reflector_args[reflector_index+1]:?Missing --output value}" ;;
    --output=*) reflector_output="${reflector_args[reflector_index]#--output=}" ;;
  esac
done
mkdir -p "$reflector_output/console_logs"
reflector_output="$(cd "$reflector_output" && pwd -P)"
reflector_log="$(mktemp "$reflector_output/console_logs/shadow-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
printf 'Terminal output log: %s\n' "$reflector_log" | tee "$reflector_log"
if [[ -n "${LH_HARNESS_PYTHON:-}" ]]; then
  "$LH_HARNESS_PYTHON" -u -m lh_harness.feasibility.terminal_bench_shadow "$@" 2>&1 | tee -a "$reflector_log"
else
  conda run --no-capture-output -n "${CONDA_ENV:-terminal-bench-2-1}" \
    python -u -m lh_harness.feasibility.terminal_bench_shadow "$@" 2>&1 | tee -a "$reflector_log"
fi
