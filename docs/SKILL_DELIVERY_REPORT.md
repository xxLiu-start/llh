# Skill Evolution 开发与实测交付报告

## 当前结论

核心实现与双轨兼容性 pilot 已完成，完整研究验收尚未完成。没有宣称成功率提高或 Token 降低。
本次沿用用户先前指定的 89 题 a-evolve TB2.0 Inspect 缓存；不是 TB2.1 全套评测。

原始与增强使用同一原生 Claude Code Backend、同一 Qwen 服务及实际权重、同一任务镜像、
相同采样、角色权限、25 MEA 轮数上限和原始 900 秒任务预算，顺序执行。
原始冻结实现来自官方 `d53df0aa11e8506dfc9518ef63620c62baee04d3` Worktree。
用户主目录 main 和已有修改/删除均保留，未 reset、清理或切换。

## 功能与配置

新增 `src/lh_harness/skills/` 包含 schema、library、router、experience、trajectory_compactor、
attribution、proposer、curator、learning、integration、client、statistics、prompts、cli。
支持八种独立训练组合及独立 usage 选择。未启用角色不运行 Proposer/Curator、不更新其 Skill。
根因不属于选中角色时 TARGET_ROLE_NOT_ENABLED，运行故障和证据不足时 DEFER。
正常 MEA 仅检索和注入，不调用离线学习模型；关闭 Skill 时不构造 Router。

Proposer 有三套角色约束与成功/失败分支；服务端原生 JSON Schema 与本地闭合对象校验共同执行。
证据从已记录引用和逐字摘录中选择。Topic/General Curator 支持角色内合并和保守更新，
保留历史与回滚。单任务不能自动晋升；至少两个独立任务、包含原任务指纹去重。
BM25 Router 按角色/GUI-CLI/trigger/Top-K/完整片段 Token 预算选择，无额外 LLM 路由。
Auditor 权限、任务状态、契约和原始评分器保持原样。

配置、启动器、配对和比较脚本位于 `experiments/skill_evolution/`。六份设计文档位于 docs。
文件清单与 SHA256：`experiments/skill_evolution/manifests/delivery.json`。

## 原始保护

1260 个冻结评估文件与 5 个根包内核文件的摘要保持一致。
开发前主目录冻结 TB 的三处既有兼容修改没有被覆盖；配对进程显式加载官方原版模块。
额外的 Harbor extra_env setter、离线安装与事件格式兼容位于独立实验适配层，两轨一致。
原版 run 直接继承，MEA、Prompt、路由、Verifier、奖励及成功标准未改。
两轨第一次 Manager Prompt 逐字相同，SHA256 已记录。

## 测试

| 测试范围 | 通过 | 跳过 |
|---|---:|---:|
| 原版官方 Worktree | 466 | 2 |
| 当前主目录完整测试 | 718 | 2 |

新增 64 项测试覆盖八组合、角色/模态隔离、预算、真实日志映射、缺失与错误证据、
候选/版本/回滚/冻结、Curator 动作、JSON Schema 客户端、双轨参数与源码保护、GUI overlay
和超时缺失 Usage。原 Dashboard HTTP 测试需要为 localhost 设置 NO_PROXY；未修改原测试标准。

## 真实本地角色验证

静态测试库分别给 Manager、Executor、Auditor 注入有界 Skill，实际调用同一个本地模型并执行
Read/Bash/图像 Read。所有角色通过、每次完整注入不超过 400 Token，Auditor 工作区无修改。
该库使用明确标识的 synthetic fixture，**不是研究训练数据，不参与下面的 Benchmark**。
记录：`results/skill_enhanced/static-role-check-6a33b8c8/validation.json`。

## 历史学习结果

学习只读取固定 train 集中的 code-from-image 与 cancel-async-tasks，未读取 held-out 任务结果。
代码图像任务原评分 1；取消任务原评分 0。后者暴露 Auditor complete 与真实 cleanup 失败断言不一致，
被按负向证据处理，没有伪装成正向成功。全部模型调用和格式/验证失败的开销均保留。
最终保留 Executor candidate：`cancel-queued-async-tasks`，一个独立任务支持，故没有晋升 Active。
Manager/Auditor 未获得正式 Active Skill，未人为复制经验或扩大支持数。
冻结库 version=1，SHA256=`a543665a100af5a013e0ed5401f2598c273435cf2d9ea3d167d73ddf19f97b73`。

## Held-out 配对结果

任务 `log-summary-date-ranges`，每轨一次：

| 指标 | Original LLH | Skill-enhanced LLH |
|---|---:|---:|
| 原始 Verifier reward | 1.0 | 1.0 |
| 任务成功 | 1/1 | 1/1 |
| 已开始 MEA 轮数 | 3 | 3 |
| Agent 时长（秒） | 661.858 | 900.008 |
| 完整总 Token | 856905 | unavailable |
| 已报告 Token 小计 | 856905 | 933049 |
| 实际 Skill 注入次数 | 0 | 0 |

增强轨道触及原有 900 秒 Agent 预算，异常为 AgentTimeoutError；任务产物仍通过原始 Verifier，
所以权威 reward 保持 1。没有加时、改分或把过程审计当评分。
最后一次审计 Usage 不可用；完整总 Token 为 null，不把已知小计冒充完整消耗。
已修复后续运行的 in-flight/cancelled 记账，取消测试验证其 missing_calls=1。

Manager/Executor/Auditor 的 Input/Output、审计率、契约率、工具/重复动作及原始异常均在 comparison.json。
未完成的审计计入审计率分母；缺失工具/Token 数据保持 null。
单次 pilot 无有效重复方差结论，且 Active=0、实际注入=0，因此只是**冻结候选库兼容性 pilot**。
不能据此归因任何性能变化，也不能宣称已验证 Skill 泛化或效率收益。

## 学习成本

所有 6 次调试/学习尝试合计 24 次模型调用，真实总 Usage **518844 Token**，包含失败调用。
价格未知，API Cost 为 null。本地 GPU/电力成本不冒充零。
训练轨迹的原始 Agent 运行成本另存 baseline/training；上述数字只统计离线学习调用。
原版本次 Tokens per Successful Task=856905.0；增强完整计数缺失，
单次与包含学习摊销的指标都保持 null。没有忽略学习成本来制造节省。

## 交付与未完成研究

代码、配置、两轨启动器和比较工具可独立使用。增强 Worktree 使用独立冻结库副本，
依赖缓存与官方只读原版 Worktree 明确共享；用户主目录不受覆盖。
WeaveBench 和 OSWorld-V2 的实验 overlay 已生成，复制源码摘要及禁用行为已测试，
尚未运行真实 GUI Benchmark；生成适配层不等于 GUI 成功率实测。

后续完整研究还需要：收集两个以上独立任务支持同一可迁移方法、验证并冻结 Active 库；
执行 manager/executor/auditor/all-role 真实消融与多次配对；完整 TB2.1/GUI 基准和跨任务迁移实测。
只有这些实验才能判断成功率、协作错误、重复操作与 Token 消耗是否改善。

## 修改文件清单

- `.gitignore`
- `docs/BASELINE_PRESERVATION.md`
- `docs/SKILL_ARCHITECTURE.md`
- `docs/SKILL_EXPERIMENT_PROTOCOL.md`
- `docs/SKILL_LEARNING.md`
- `docs/SKILL_PROMPTS.md`
- `docs/SKILL_SYSTEM.md`
- `experiments/skill_evolution/adapters/harbor_skill_agent.py`
- `experiments/skill_evolution/configs/all_roles.local.toml`
- `experiments/skill_evolution/configs/all_roles.toml`
- `experiments/skill_evolution/configs/auditor_only.local.toml`
- `experiments/skill_evolution/configs/auditor_only.toml`
- `experiments/skill_evolution/configs/baseline.local.toml`
- `experiments/skill_evolution/configs/baseline.toml`
- `experiments/skill_evolution/configs/disabled_compatibility.toml`
- `experiments/skill_evolution/configs/executor_auditor.local.toml`
- `experiments/skill_evolution/configs/executor_auditor.toml`
- `experiments/skill_evolution/configs/executor_only.local.toml`
- `experiments/skill_evolution/configs/executor_only.toml`
- `experiments/skill_evolution/configs/manager_auditor.local.toml`
- `experiments/skill_evolution/configs/manager_auditor.toml`
- `experiments/skill_evolution/configs/manager_executor.local.toml`
- `experiments/skill_evolution/configs/manager_executor.toml`
- `experiments/skill_evolution/configs/manager_only.local.toml`
- `experiments/skill_evolution/configs/manager_only.toml`
- `experiments/skill_evolution/launchers/run_baseline.sh`
- `experiments/skill_evolution/launchers/run_skill_enhanced.sh`
- `experiments/skill_evolution/scripts/compare_results.py`
- `experiments/skill_evolution/scripts/prepare_gui_adapter.py`
- `experiments/skill_evolution/scripts/run_paired_eval.py`
- `experiments/skill_evolution/scripts/validate_live_skill_roles.py`
- `experiments/skill_evolution/scripts/validate_manifest.py`
- `src/lh_harness/cli.py`
- `src/lh_harness/local_experiment.py`
- `src/lh_harness/skill_experiment.py`
- `src/lh_harness/skills/__init__.py`
- `src/lh_harness/skills/attribution.py`
- `src/lh_harness/skills/cli.py`
- `src/lh_harness/skills/client.py`
- `src/lh_harness/skills/curator.py`
- `src/lh_harness/skills/experience.py`
- `src/lh_harness/skills/integration.py`
- `src/lh_harness/skills/learning.py`
- `src/lh_harness/skills/library.py`
- `src/lh_harness/skills/prompts.py`
- `src/lh_harness/skills/proposer.py`
- `src/lh_harness/skills/router.py`
- `src/lh_harness/skills/schema.py`
- `src/lh_harness/skills/statistics.py`
- `src/lh_harness/skills/trajectory_compactor.py`
- `tests/test_role_skill_evolution.py`
- `tests/test_skill_experiment_protocol.py`
- `tests/test_skill_learning_client.py`
