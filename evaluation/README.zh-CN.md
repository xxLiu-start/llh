# 原版 LongHorizon-Harness：Terminal-Bench Pro 评估

该评估使用 `eval/TB-harness/Harness/src/harbor_agent.py` 中的原版角色编排，不加载 SkillBank、Skill Router 或 Skill 学习。默认使用 `deepseek-v4.1-flash`，依次评估 `train_dataset/terminal-bench-pro` 的 30 道题，每题运行一次。

```bash
read -rsp 'API key: ' ANTHROPIC_API_KEY; echo
export ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL='https://movoapi.top/v1'
sg docker -c './evaluation/run_original_harness_terminal_bench_pro.sh'
```

单题测试：

```bash
sg docker -c './evaluation/run_original_harness_terminal_bench_pro.sh --task advanced-poker-hand-classifier --force'
```

从指定题目继续：

```bash
sg docker -c './evaluation/run_original_harness_terminal_bench_pro.sh --start-from analyze-arm-shellcode-network-connections'
```

已有且完整的单题结果默认复用；错误或缺失 reward 的记录会自动重试。`--force` 会重跑完整记录。新的当前记录写入前，旧记录会归档到 `results/history/`，旧 Harbor 日志也会保留。

结果位于 `output/origin/results/`：

- `tasks/<任务名>.json`：该题 reward、正确与否、输入/输出/cache/总 Token、Harness 状态和日志路径。
- `per_task.csv`、`per_task.jsonl`：所有已记录题目的表格和逐行 JSON。
- `summary.json`：正确数、准确率和总 Token。

`correct` 依据 Terminal-Bench verifier 的 `reward >= 1.0`，不依据 Harness 自己声称是否完成。`total_tokens` 等于该题 Harbor job 聚合的 `input_tokens + output_tokens`，因此包含自动重试产生的实际消耗；最终一次尝试的计数另存为 `final_attempt_*`。Harbor 的输入 Token 已包含 cache Token，所以 cache 不再重复加入总量。
