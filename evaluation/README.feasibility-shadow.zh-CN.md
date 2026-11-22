# Terminal-Bench 原版 LHH + 只读 Reflector

任务 ID 通过 `evaluation/run_feasibility_shadow_terminal_bench_pro.sh` 运行；
`training/run_feasibility_shadow.sh` 的 `--task` 则接收工作区任务全文。

```bash
cd /data0/liuxule/project/LHH/LongHorizon-Harness
CONDA_ENV=terminal-bench-2-1 \
./evaluation/run_feasibility_shadow_terminal_bench_pro.sh \
  --output output/feasibility-shadow-10 \
  --model deepseek-v4.1-flash \
  --skill-bank output/feasibility-shadow-10/empty-skillbank \
  --task advanced-poker-hand-classifier \
  --task analyze-arm-shellcode-network-connections \
  --task bash-ddos-traffic-analyzer \
  --task capture-iot-firmware-boot-message \
  --task configure-apache-analytics-virtualhost \
  --task detect-c-feature-flags \
  --task detect-corrupted-blockchain-transaction \
  --task fix-color-lib-hash-and-pickle \
  --task fix-docker-python-dependency-conflicts \
  --task flood-escape-path-planner
```

沿用原评估的 API 环境变量 `ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`，以及已有的
Claude Code 离线缓存和 uv 支持；不把密钥写进命令或日志。缺少匹配的离线二进制时，
可按原评估方式指定 `LH_CLAUDE_OFFLINE_BINARY`。

新输出目录下的 `empty-skillbank` 默认不存在，按空 Bank 处理。运行不会创建或更新
Skill，Reflector 始终依据当前证据判断；已有适用 Active Skill 时才作为补充。

脚本使用原任务路径、Docker 环境、原 Harbor Agent 的 setup/run、原角色工厂、
原 orchestrator、原角色 Prompt、工具、独立 Auditor 及原评分器。新增 Harbor 子类
只包装 Manager 的 AgentAdapter：先接收其原始结果、读取已有历史审计，再运行只读
Reflector，最后返回**同一个**结果对象。REVISE/BLOCK/UNKNOWN 均不改变原始派发。
角色 token 计数仍来自原适配器，Reflector 开销单独记录。

原 `eval/TB-harness/Harness/src`、原评估驱动和 YAML 模板均未修改；本脚本仅在新生成的
运行配置中选择 shadow 包装类。没有新数据集、任务专属 Skill 或题名匹配策略。

输出目录：

```text
output/feasibility-shadow-10/
  console_logs/shadow-*.log           # 实时终端输出
  generated_configs/                 # 每题 Harbor 配置，直接引用现有任务
  jobs/<task>/                       # 原始 Harbor 任务执行记录
  reflector_logs/<task>/<run_id>/
    per_round.csv                    # 每轮 Manager Contract 判断
    round_001.json                   # 原契约、执行前快照、判断、后续真实审计链接
    summary.json                     # 次数、错误和 Reflector token 开销
  results/per_task.csv               # 原 reward/正确性和角色 token，加上独立观察器字段
  results/summary.json
```

有效的每次 Manager 输出均会被观察，包括完成/阻塞输出；契约按原解析器继承。
Manager 自身失败且没有有效输出时会注明未评估。Reflector 无法访问未来执行结果、
测试答案或实时执行工具。其 UNKNOWN 或内部错误不会修改 Manager 返回结果。

在命令后追加 `--dry-run` 只检查任务和生成配置，不执行模型、Docker 或评分。
Reflector 默认输入上限现为 48,000 字符，可用 `--reflector-max-input-chars` 调整。
输入仅去除 Manager Plan 中与独立契约/状态字段完全重复的文本，不截断原任务、
当前契约或历史审计。模型返回的 JSON/字段/证据校验失败时最多修复一次，两次成本
均单独记录；模型超时/API 错误不会触发格式修复。具体原因直接出现在终端事件的
`uncertainty`、`error_stage`、`validation_error`，以及每轮 JSON 的 `trace.model_attempts`。
修复未通过时仍返回 UNKNOWN，不把缺少证据的结果强行转为 ACCEPT。

历史日志中只有 `runtime_error=ValueError` 时，无法从它恢复当时的具体失败字段；
不能把它解释为模型确实判断任务不可行或单纯缺少 Skill。输入超限的 UNKNOWN
则可由 `llm_calls=0` 和 `input_budget_exceeded` 区分。新代码仅对新启动的 Harbor
进程生效，正在运行的任务不会热更新；已经记录的判定不被覆盖或伪造。

任务按命令中的顺序串行运行。再次运行会创建新的 job/config 并通过原结果存档函数
保留上一轮记录。原评分字段从原结果解析函数获取，未成功收集的开销保持未知；
Harbor 重试时观察器开销标记为部分统计，不声称已得到完整总成本。

验证包括：实际原 orchestrator 的前后运行对照（Prompt、返回对象、执行、审计和完成
状态一致）、Harbor 子类对角色工厂的包装、原配置差异检查和 10 题 dry-run。
这些是受控测试，未宣称真实模型任务通过。
