# SkillBank 训练与测试

## 数据划分

| 用途 | 数据集 | 配置 |
|---|---|---|
| 训练与候选重跑 | `train_dataset/terminal-bench-pro` | `terminal_bench_pro_skill_learning.yaml` |
| 测试 | 原有 Terminal-Bench 2.1：`eval/TB-harness/datasets/terminal-bench-2-1/tasks` | `terminal_bench21_skill_test.yaml` |

测试任务仍使用原评测目录。新的测试配置用当前 `src/lh_harness/` 主实现加载训练后 SkillBank，并设置 `learning_enabled: false`，因此测试任务不会更新 SkillBank。

训练使用中文版 Manager/Executor/Auditor、Router 和 Skill 学习 Prompt（`prompt_language: zh`），生成中文 Skill（`content_language: zh`）。TB2.1 评估保留英文角色与 Router Prompt（`prompt_language: en`）、关闭学习；英文 Router 仍可选择和注入中文 Skill。数据集任务原文保持原样。当前工作区缺少 TB2.1 原测试任务，无法在本地确认任务说明语言。
Router 只读取精简的 `routing_summary`、版本和适用范围等选择信息；选中后才加载当前角色的四字段指导。
本方法的训练与测试配置均使用 `deepseek-v4.1-flash`。Claude Code 仍向本地代理发送 Anthropic Messages；代理把文本与工具消息转换为 `/v1/chat/completions` 请求，再把 OpenAI 的结构化函数调用转回 Anthropic `tool_use`。这使用了接口检查中能够返回结构化工具调用的协议。训练驱动与测试脚本会在启动 Harbor 前规范化 Base URL 的结尾 `/v1`，代理把密钥作为 `Authorization: Bearer` 转发。

## 运行

使用原 Terminal-Bench 2.1 环境（`eval/TB-harness/environment-tbench21.yml`），准备 Docker、Harbor 和模型 API 环境变量：

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=...
python training/check_model_endpoint.py
./training/run_terminal_bench_pro.sh --dry-run
./training/run_terminal_bench_pro.sh --task maze-shortest-path-solver
./training/run_terminal_bench_pro.sh
```

从指定题目继续时，先用 `--dry-run` 核对范围，再运行：

```bash
./training/run_terminal_bench_pro.sh --start-from analyze-arm-shellcode-network-connections --dry-run
./training/run_terminal_bench_pro.sh --start-from analyze-arm-shellcode-network-connections
```

如果任务容器无法访问 `downloads.claude.ai`，可将匹配版本的可执行文件放在 `output/<数据集名>/jobs/claude_code_cache/<版本号>/claude`，或把其绝对路径设为 `LH_CLAUDE_OFFLINE_BINARY`。训练和原版 LHH 评估入口都会检查其他本地输出目录及 `output/his/` 下的同版本缓存，核对 `claude --version` 后挂载，避免 Harbor 联网安装。原版评估找不到缓存时会在第一题启动前报错。

任务按目录名字典序逐题运行。基线轨迹先提取经验，再由 Evolution Planner 分别规划新建、修订与 Router 更新；每个数组可以有多项。Builder 必须使用精确 Skill Schema，结构错误时只根据草稿和计划进行一次有限校正，证据或安全检查失败则拒绝。Harbor 与 Harness 的完成判断冲突时隔离该次候选。其余合格候选从试跑的 SkillBank 快照转入训练库的 `skillbank_temporary.json`，以 provisional 参与后续训练。来源题目组合重跑只提供组合观察，不能自动把每个候选升级为 validated；升级需要单项独立验证。已验证旧版仍保存在主库，修订候选可以回退。`--task` 可重复指定，`--force` 重新运行已有决定的题；每次重跑使用独立 Harbor 作业并保留旧日志。

修订 Skill 时，Builder 必须保留旧版仍有用的通用功能，并把单题路径、数值和示例泛化。修订内容变化后会增加一次独立模型审查，逐字段核对新旧版本。Evolution Planner 将逐项审查结果保存在 `request_outcomes`，审查失败只拒绝该项；旧版批量流程将结果保存在 `skill_learning.revision_preservation_reviews`，失败时拒绝整批候选。内容审查不替代独立验证。

默认训练配置最多执行 25 轮；Executor 每次回合最多 200 次模型交互、最长 4200 秒，Manager 和 Auditor 各有 200 次交互上限。模型单次输出上限保持 65536 token。角色历史和已验证上下文分别限制在 30000、20000 字符。每个基线或候选试跑在完整回合之间检查 150 万 token 软上限；达到后记录 `token_budget_exhausted`，仍运行失败诊断。它不是精确的计费上限：单回合可能超出阈值，诊断和 Builder 也会额外调用模型。需要调整限额时，可复制 `terminal_bench_pro_skill_learning.yaml`，修改后用 `--template <副本路径>` 运行。`skill_token_budget` 只限制注入的 Skill 内容，`skill_reward_reference_tokens` 只用于奖励计算。超时回合的已完成请求用量会从代理记录补计；正在超时瞬间处理的请求仍可能无法精确计量。候选独立验证会再次运行同一题，因此总费用可能高于单次基线。

训练运行的 Manager 只接收原任务、当前状态、任务契约、每轮简短审计结论及近期 Harness 反馈；Executor 和 Auditor 只接收本轮子任务及明确引用的审计证据。任务结束后先从每个已记录轮次的有界摘录提取正向与负向经验；第二阶段依据这些经验、SkillBank 和实际路由记录进行匹配，并接收所有轮次的简短索引及最多 6 轮关键摘录。Builder 最多接收 8 轮（优先包含诊断引用的轮次）。经验结果或提取错误保存在 skill_learning 中。完整角色报告、模型输入输出与原始轨迹仍存放在运行日志和只读溯源记录中；原始工具轨迹不会整批注入学习提示词。有轨迹的任务会增加一次经验提取模型调用。Claude Code 在单个角色回合内为连续工具调用维护的对话历史仍由其自身处理。

每个 Skill 有一段简短的 `routing_summary`，概括可复用能力、适用时机和关键排除条件。Router 只收到这段简述及 ID、版本、范围、状态等选择元数据；确认选中后，程序才从 SkillBank 读取该 Skill 对当前角色的 `input_use`、`output_spec`、`guidance`、`avoid` 并注入角色 Prompt。旧 SkillBank 条目会在内存中生成有长度上限的简述；仅精确匹配旧版内置 Router Prompt 时自动切换到新版，用户自定义 Router Prompt 保留。

Evolution Planner 的候选及逐项处理结果写在本次运行的 `skill_candidate.json` 和溯源记录中。旧版候选重跑流程还会在 `output/<数据集名>/skill_tmp/<题目>/<run_id>/candidate.json` 另存可读副本；`validation.json` 保存整组候选共用的一次重跑结果与 reward 差值。这些副本仅供过程溯源，训练流程不会读取；即使保存失败，也不改变训练结果。批量候选重跑所需配置放在 `generated_configs/`。

Harbor 训练和测试使用无人值守策略：Manager 应先从任务说明、元数据、环境文件和审计证据界定输入并继续执行；无法可靠解决的任务定义问题应明确阻塞。若训练中的中文 Manager 仍输出 `下一步: 请示用户` 且还有轮次，Harness 会给一次带反馈的重新决策机会；再次请求人工输入或轮次已尽时记为 `unattended_needs_human_input` 并保存问题，不会捏造答复。普通交互运行仍保留请示用户能力。未成功但有实际 Manager、Executor 或 Auditor 轨迹的运行会尝试失败诊断；没有可分析轨迹则记录 `skip_reason`。`skill_learning` 报告保留真实动作、诊断理由、原始解析失败输出和 `error_stage`，不会把默认状态冒充模型的 `NO_CHANGE`。

重跑单题使用 `./training/run_terminal_bench_pro.sh --task advanced-poker-hand-classifier --force`（需要 Docker 权限时在外层使用 `sg docker -c '...'`）。不加 `--force` 且已有非失败决策时，通常返回 `execution: reused_prior_decision`；旧版证据引用校验错误会自动重试。`--force` 会重跑整个任务，再重新诊断及按需验证候选；`--validation-replay` 可显式开启默认的候选重跑验证。切换开关时会新建任务运行，而不会复用另一模式的旧决策；重新打开验证不会自动验证之前留下的原候选。当前没有只对旧轨迹重新诊断的命令。每次真实运行都有独立 `run_id`、Harbor 作业、配置、临时 SkillBank 快照和 `decisions/attempts/<题目>/<run_id>.json`。最新结果仍写 `decisions/<题目>.json`，覆盖前会把旧版备份到 `decisions/history/<题目>/`；Provider 等基础设施失败写入当次 attempt，不覆盖最新决策。

通过 `./training/run_terminal_bench_pro.sh` 启动时，标准输出和错误输出同步保存到 `output/<数据集名>/console_logs/train-*.log`，终端会打印路径。直接执行 Python 模块不会自动创建该日志。

Terminal-Bench Pro 题目自带的验证脚本会在每次判分时从 GitHub 下载 `uv 0.7.13`，网络故障曾使真实测试无法启动并被记为 0 分。原版 LHH 评估、Skill 训练和双 SkillBank 评估现在只在使用这段安装脚本的题目中，为 Harbor 验证阶段挂载主机已有的同版本 `uv`，并跳过重复下载；题目的 `tests/test.sh`、测试用例、Dockerfile 与 Agent 阶段的 PATH 保持原样。运行主机需要有可执行的 `uv 0.7.13`；若不在 PATH 中，设置 `LH_VERIFIER_UV_BINARY=/绝对路径/uv`。版本不符或文件缺失时会在启动试跑前明确报错。旧 0 分记录不会被自动修改，需用 `--force` 重跑相关题目。

## 只读溯源记录

每道题使用一个追加式 JSONL 文件：`output/<数据集名>/provenance/<题目>.jsonl`；任务开始时会打印具体路径。重跑会追加到同一文件，每条记录保留 `run_id` 以区分运行。训练程序不会读取它，也不会用它做 Skill 选择或候选验收。Docker 预检记录保存在 `output/<数据集名>/docker_preflight/`，Harbor 任务本身继续保存在 `output/<数据集名>/jobs/`。

记录覆盖每个角色及 Skill 学习调用的完整模型输入和输出，用同一 `trace_id` 关联客户端与上游模型调用。桥接模式分别保存收到的 Anthropic 请求、转成的 OpenAI 请求、OpenAI 响应和转回的 Anthropic 响应。JSON 与 SSE 保存结构化内容和原始文本；不记录认证请求头。每个 Harbor 回合目录也单独保留 `model_io.jsonl`。

Claude Code 可能在 `assistant` 和 `result` 两个事件中重复提供同一最终答复。适配器现在只用最终 `result` 作为 `assistant_visible_output`，缺失时才取最后一条 assistant 文本；原始事件日志仍保留两个协议事件供检查。

同一文件还记录当前与历史 Skill 版本、当前与历史 Router Prompt 版本、生成及验收后的候选、带 SHA-256 版本号且与本次运行语言一致的静态 Prompt、基线与候选报告、试跑失败原因，以及最终决策。训练前、每次试跑后及提交后的 SkillBank 快照可用于跨运行比较。文件采用仅所有者可读写的权限，因为完整 Prompt 和响应可能包含任务数据。功能上线前已经结束的运行无法补录精确的模型请求和响应。

每条实际模型输入还生成 `runtime_prompt_version` 记录，保存当次系统消息、对话消息和工具定义，并用 `trace_id` 与 `model_io` 请求/响应关联；`previous_version_id` 串起同一阶段、角色和回合内的 Prompt 修改。Skill 快照和候选也带内容哈希，修订候选指向原 Skill 的哈希。溯源写入失败会记录警告，不参与训练决策。

训练完成并确认 SkillBank 已生成后，使用原测试任务目录运行：

```bash
./training/run_terminal_bench21_test.sh
```

若要明确使用正式和临时两个 SkillBank、只评估而不训练，运行：

```bash
sg docker -c './training/run_dual_skillbank_eval.sh'
```

该命令默认评估原 TB2.1 测试集，每题运行一次，读取 `output/terminal-bench-pro/state/skillbank.json` 和同目录下的 `skillbank_temporary.json`。启动时两份文件被复制到 `output/dual-skillbank-eval/state/`，后续续跑继续使用同一快照；不会写回训练 SkillBank，也不会生成或修订 Skill。逐题正确性、Reward、Token 和实际选中的 Skill 写到 `output/dual-skillbank-eval/results/per_task.csv`，终端日志在 `console_logs/`。需要重新使用训练后更新的 SkillBank 时，请指定新的 `--output` 目录。当前工作区缺少 TB2.1 测试任务，需先按上文准备数据集。

也可指定其他数据集与题目，例如 `./training/run_dual_skillbank_eval.sh --dataset train_dataset/terminal-bench-pro --output output/pro-dual-skillbank-eval --task advanced-poker-hand-classifier`。对训练集进行评估属于同数据评估，不代表对 TB2.1 测试集的泛化效果。

测试输出放在 `output/terminal-bench-2-1/`，其中 `jobs/` 保存 Harbor 日志；测试从 `output/terminal-bench-pro/state/skillbank.json` 加载已训练的 SkillBank，不更新它。测试配置保持原实验的 `n_attempts: 3` 和 `n_concurrent_trials: 16`。当前工作区未放入原 Terminal-Bench 2.1 测试 tasks，运行测试前仍需按 `eval/TB-harness/README.zh-CN.md` 准备到上述路径。这里没有自动执行模型训练或测试任务；它们需要 Docker、Harbor、模型凭据及对应的运行时间。

`python training/check_model_endpoint.py` 会用同一 Bearer 密钥分别请求 `/v1/chat/completions` 和 `/v1/messages`，只打印 HTTP 状态码。请求代理的 `request_proxy_requests.jsonl` 还会记录 `forwarded_auth` 事件，只有 `bearer`、`x_api_key` 布尔值，不记录密钥。
如果两个接口均报 `ConnectionRefusedError`，运行 `python training/check_model_endpoint.py --direct` 绕过已配置的 HTTP(S) 代理，区分本机代理故障与 API 返回的错误。
运行 `python training/check_model_endpoint.py --direct --tools` 可检查两种协议是否遵循强制结构化工具调用。Claude Code 无法执行纯文本模拟的工具调用；训练驱动会将这种情况报为接口协议错误，不再记录为普通的“没有候选”。桥接器目前支持文本消息和客户端工具调用；不支持的内容块会明确报错。
DeepSeek 思考模式下，桥接器会在单个角色回合的代理进程内暂存 `reasoning_content`，并在后续工具请求中回传；代理审计日志不写入这段内容。
