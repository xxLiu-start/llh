# 三角色 Skill Evolution（当前入口）

当前实现与命令请阅读 [Skill 系统](../../docs/SKILL_SYSTEM.md)、[实验协议](../../docs/SKILL_EXPERIMENT_PROTOCOL.md)、[Baseline 保护](../../docs/BASELINE_PRESERVATION.md)。

八种训练组合、训练/使用独立开关、原生 JSON Schema、候选/版本/回滚、BM25 与 Token 预算、离线学习和冻结测试已在新模块中实现。
原始与增强轨道共用已验证的本地模型接入和离线评分依赖。新增实验在 `results/baseline`、`results/skill_enhanced`、`results/comparison` 中分别保存。

以下是早期本地模型接入及烟雾测试说明；其中旧 JSON bank 和任务后训练描述仅作为历史记录，当前 `.local.toml` 已迁移到独立 `library_dir` 与显式离线学习，原配置已备份。

---

# Original LLH 与 Skill-enhanced LLH 的本地模型对照

两条轨道共用现有 `AgentAdapter`、角色模型选择、全局 `base_url` 和 MEA 执行内核。
`manager.py`、`role_prompts.py`、`auditor_agent.py`、数据集和 Benchmark Verifier 均不由本实验层改写。
主目录原有 Feasibility Reflector 保留；本目录提供独立、显式启用的三角色 Skill 对照包装层，不导入归档的历史 SkillBank。
baseline 禁用 Skill；其传给 Adapter 的 Prompt 与原流程逐字相同。

## 已有服务与 Model ID

2026-11-20 检查 `http://127.0.0.1:8010/v1/models` 时，服务公布的 ID 为
`/data0/liuxule/project/models/Qwen3.8-27B`，框架为 vLLM 0.23.0，`max_model_len=1000000`。
用户提供的别名 `qwen3-27b` 返回 404。五份配置使用经用户确认的完整 ID，不推断别名。
`base_url` 应填写 API 根路径 `/v1`，不能填写 `/v1/chat/completions`。

## 直接使用原有 Endpoint 配置

不需要任何 Skill 配置即可连接本地 Responses 服务：

```toml
[run]
agent = "codex"
model = "/data0/liuxule/project/models/Qwen3.8-27B"
base_url = "http://127.0.0.1:8010/v1"
env = "local"
max_rounds = 25
```

```bash
lh-harness doctor --check-model-endpoint --config path/to/direct.local.toml
lh-harness run --config path/to/direct.local.toml --task @task.txt --no-dashboard
```

使用 Python >=3.10，先安装当前主目录（例如 `python -m pip install -e .`）。
源码运行也可设置 `PYTHONPATH="$PWD/src"` 后使用 `python -m lh_harness`。
原配置位置 `.lh-harness/config.toml` 和 CLI 参数优先级继续有效。
配置指定本地服务或 `[local_model]` 启用时，启动前会对所有有效角色绑定去重检查。
缺失 Model ID、不可访问 Endpoint、协议不符、工具参数异常、流提前结束均阻止启动。
不跟随 HTTP 重定向，不选择其他模型，不修改原配置，不回退云端。

认证由环境变量提供。Codex 读取 `OPENAI_API_KEY` / `CODEX_API_KEY`；Claude Code 读取
`ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`；DeepSeek 使用 `DEEPSEEK_API_KEY`；
OpenCode 使用其 Provider 的环境配置或 `OPENCODE_API_KEY`。
不需要认证的服务可以不设置 Key；不要将真实 Key 放入 TOML 或日志。
`--api-key` 为现有兼容参数；实验建议只用环境变量。

## 相同推理参数与双轨运行

五份配置的模型、Backend、全局 Endpoint、Context、采样、Reasoning、Tool Calling、并发均相同。
仅 `[skills]` 区域不同：`baseline`、`manager_only`、`executor_only`、`auditor_only`、`all_roles`。
角色覆盖沿用 `[run.roles.manager]`、`executor`、`auditor` 以及 GUI/CLI 子角色配置；
不增加原 Baseline 的角色级 Base URL。

Codex 本身没有通用 Temperature / Top-p CLI 参数。本实验使用显式的独立推理策略代理，
两条轨道都通过已有 `base_url` 指向它，不改写 Agent Backend。
代理监听 8011，唯一上游是用户的 8010 服务，保持原协议和逐块 SSE 输出，强制相同采样、
8192 输出上限、`enable_thinking=false`、非并行 Tool Calling 和单请求并发。
它不提供 Chat Completions 到 Responses 的隐式转换。协议不匹配必须另行配置明确的适配服务。
`doctor` 会核对代理实际生效策略与实验 TOML，配置不一致即拒绝启动。

```bash
# 终端一：baseline 和全部增强变体共用同一个代理
python -m lh_harness.local_gateway \
  --config experiments/skill_evolution/configs/baseline.local.toml --port 8011

# 终端二：检查；再独立运行任意一个轨道
lh-harness doctor --check-model-endpoint \
  --config experiments/skill_evolution/configs/baseline.local.toml
lh-harness run --config experiments/skill_evolution/configs/executor_only.local.toml \
  --task @task.txt --no-dashboard

# 两组顺序运行；禁止除 Skills 外的任何配置差异
python -m lh_harness.local_pair --task @task.txt --output /tmp/llh-pair \
  --baseline experiments/skill_evolution/configs/baseline.local.toml \
  --variant experiments/skill_evolution/configs/all_roles.local.toml
```

`local_pair` 在上游 Endpoint 的共享文件锁下顺序运行，启动前、两轨之间、结束后重查配置。
各轨使用独立 workspace、run、日志和 SkillBank。需要任务初始文件时应先准备相同内容的工作目录；
本启动器默认创建空 workspace，不能替代带有容器环境和任务资产的 Harbor Benchmark 启动器。
同一 GPU 上其他用户/进程不受本锁管理，正式实验需自行保留资源。

正式运行前必须填写真实权重路径、权重摘要、量化、实际 GPU、服务版本等元数据。
模板不伪造这些信息，缺失时严格双轨启动器拒绝正式比较。
`--smoke` 只跳过硬件/权重元数据完整性门槛，仍检查协议和配置一致性，并明确标为
`smoke_only=true`、`fair_comparison_verified=false`，不能作为公平 Benchmark 结论。
固定一轮的烟雾测试可能以 `max_rounds_exhausted`、任务退出码 1 结束，即使 Auditor 已确认产物正确；
启动器保留这些原始结果。其连通性检查要求两轨都实际执行 Manager、Executor、Auditor，
不会把任务预算耗尽改写成任务完成。
Manifest 的 null 表示未知；操作者填写的值不冒充服务独立证明。
`weights_sha256` 建议使用完整权重文件 SHA256 清单的确定性摘要；记录摘要算法和原清单。

每个 run 的 `experiment_manifest.json` 保存有效角色、Backend、Model ID、Endpoint、推理策略、
真实服务探测、未核实字段、配置指纹及评估源码 SHA256。`comparison_manifest.json` 保存两轨结果。
Prompt、Verifier 与 MEA 代码摘要使后续检查能发现评估路线变化；不将烟雾测试的完成状态当作 Benchmark 得分。

## Token 与 Skill 范围

`token_usage.json` 分开记录 Manager / Executor / Auditor Input 和 Output Tokens，以及
Skill Injection、Skill Proposer、Skill Curator；final response 和格式修复单独列出。
Codex `turn.completed`、Claude terminal `result` 和 OpenCode `step_finish` 的真实 Usage 统一解析。
不重复统计流中的回显；缺少任一调用的计数时整组标记 `unavailable`，另列已报告的小计。
未发生的 Skill 调用是 `not_applicable`，不会伪装成已测得的 0 Tokens。
注入 Tokens 若提供同模型 `tokenizer_path`，用本地 `tokenizers` 离线计算并标记 `estimated`；
否则为 `unavailable`。它是角色 Input 的子集，不再加入总成本一次。
同一对照使用相同 tokenizer 与计数口径；没有字符数换算 Tokens 的默认估计。

Skill 层只追加有界建议文本；训练只对 `train_roles` 生效，使用同一角色的 Backend/模型及
final-response 权限，通过 Proposer 和 Curator 两次调用生成带逐字证据的建议。
SkillBank 按变体隔离；它不是历史归档 Skill 系统，也不为任务提供完成判定权。
训练失败单独记录，不改写已完成的 MEA/Verifier 结果。
每次训练保存原始输出和轨迹；JSON 格式或字段不正确时，最多通过同一 Adapter、同一模型修复一次，
修复调用也计入对应 Proposer/Curator Usage。Executor/Auditor 训练采用各自 CLI 子角色的有效模型绑定，
避免 GUI/CLI 覆盖配置存在时意外采用 Backend 默认模型。
Codex Manager/Auditor 使用 read-only sandbox，Executor 沿用执行权限；Claude 原权限隔离和审计快照保留。
OpenCode 的 Manager/Auditor 另经独立快照包装层检查工作区；发现写入或快照失败即拒绝该 episode 的成功状态。
该包装层属于事后检测，不声称提供 OS 沙箱，也不自动恢复被修改的用户文件；真实 OpenCode CLI 尚未在本机验收。

## Backend / 服务兼容性

| Backend | Endpoint / Model | API 与工具 | 流与 Usage | 本机验收范围 |
|---|---|---|---|---|
| CodexAdapter | 自定义 `model_providers.lh_harness.base_url`；`--model` 原样传递 | Responses；函数/CLI 工具由 Codex 执行 | Responses SSE → Codex JSONL；`turn.completed.usage` | 已实测本机 vLLM；见 validation |
| ClaudeCodeAdapter | `ANTHROPIC_BASE_URL` 去除尾部 `/v1`；`--model` | Anthropic Messages，`tool_use` / `tool_result` | Messages SSE → `stream-json`；terminal `result.usage` | 本机 Messages 工具/SSE/Usage/错误处理已实测；未安装 Claude CLI，真实 CLI 验收未验证 |
| OpenCodeAdapter | `OPENCODE_CONFIG` Provider `options.baseURL`；必须是 `provider/model` | 由 Provider 决定；默认探测 Chat Completions，可显式指定协议 | Provider 流 → `--format json`；`step_finish.part.tokens` | 本机 Chat 工具/SSE/Usage/错误处理已实测；未安装 OpenCode CLI；须先在原配置中注册本地 Provider 和模型 |
| DeepSeekHarnessAdapter | `DEEPSEEK_BASE_URL`；独立 DSH_HOME；模型通过 patch YAML 传给 `deepseek-official` | DeepSeek/OpenAI Chat Completions；实际兼容性取决于 DSH 版本和服务 | 当前 runner 收集完整 stdout 后输出 `dsh.result`；常无可解析 Usage | 本机 Chat 工具/SSE/Usage/错误处理已实测；未安装 DSH CLI，真实 CLI 验收未验证；runner 实时流仍不满足要求 |

vLLM、SGLang、Ollama、LM Studio 仅在对应版本/启动参数实现所选 Backend 必要协议、
工具与 SSE 时兼容。仅支持 Chat Completions 的服务不能直接作为 Codex Responses Endpoint。
不支持的组合会明确失败；所有已兼容组合对 Original 与 Skill-enhanced 同时可用。
GUI 角色调用本地模型和实际桌面能力是两个验收项；本目录的文件任务不证明视觉或桌面工具可用。
已知的 DSH 缓冲输出限制会在本地实验预检中明确拒绝；API 探测通过不能绕过该启动门槛。
需要独立、经真实 CLI 验证的流式适配层后才能放行；本次没有重写 DSH Backend 或伪造 Token 流。

安装的 Codex 对未收录的本地 Model ID 会提示 `fallback model metadata`。这是能力目录元数据，
不等于换模型；实际请求仍受明确的 Provider Endpoint 与代理 Model ID 白名单限制。
实验配置的 Context Length 另通过其原生 `model_context_window` 参数显式传入。
可在顺序实验结束后运行全部角色的真实 Adapter 检查：

```bash
python experiments/skill_evolution/validate_roles.py \
  --config experiments/skill_evolution/configs/baseline.local.toml \
  --output /tmp/llh-local-role-check
```

官方协议/配置参考：
[Codex config schema](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/config.schema.json)、
[vLLM OpenAI server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)、
[Claude LLM gateway](https://code.claude.com/docs/en/llm-gateway)、
[OpenCode providers](https://opencode.ai/docs/providers/)、
[DeepSeek Harness CLI reference](https://github.com/deepseek-ai/deepseek-harness/blob/master/apps/cli/reference/README.md)。
