# 经验学习与训练隔离

训练、验证、测试分割 JSON 必须包含 dataset、train、validation、test。三个集合必须互斥。
第一次模型调用之前校验整个学习批次；test/validation 任务不能进入离线学习。
本次 pilot 的固定分割记录在 `experiments/skill_evolution/manifests/pilot-splits.json`。
所选本地数据为 89 题 a-evolve TB2.0 Inspect 缓存，不冒充 Terminal-Bench 2.1。

## 经验读取

根包 LLH 日志位于 run 的 `role_orchestration/rounds/round_NNN/`：
manager_input/plan、executor_prompt/output、auditor_input/report，以及各角色 raw_trajectory 和 metadata。
轮次汇总由 `role_orchestration/rounds.jsonl`、events.jsonl、report.json 给出。

冻结 TB 日志位于 Harbor trial 的
`agent/cua_harness_claudecode_harbor/cua_harness/role_orchestration/`。
对应别名是 orchestrator、task、verifier；Actor 的原始 Claude 流也保存在 episodes 子目录。
`verifier/ctrf.json` 与 `verifier/test-stdout.txt` 是任务结束后的原始评分证据。

结构化 bundle 包含原任务指纹、计划/状态/契约、角色工具事件、错误、重复动作、审计、
实际评分、Usage、证据引用和源文件 SHA256。完整原始轨迹不会送入 Proposer。
审计与失败断言独立保留；普通大段输出和重复字段在发给离线模型前压缩。
缺少轨迹的 Usage 为 unavailable/null，不当成真实 0。

## 证据门禁

正向经验要求原始审计解析得到 complete、clean、aligned，且正文支持全部原任务要求。
存在实际 Benchmark 0 分时，即使 Auditor 报 complete，也不能生成正向 Skill。
评分器确实执行并给出失败断言时，可以学习审计与评分器不一致的负向经验；不能凭 0 分猜测角色。
blocked、unknown、suspect、violation 或缺少可信审计的经验不能直接成为正向 Skill。
实际连接、认证、环境安装和 Verifier 故障继续排除，不能当作角色能力缺陷。
`AgentTimeoutError` 单独标记为任务执行预算耗尽，不再因 exception 非空就排除整个任务。
成功经验与负面经验分开准入。负面经验要求角色自身已记录的行为，并由 clean 的独立审计
（aligned 或 needs_revision）或真正执行的 Benchmark 失败断言提供独立佐证。0 分、超时本身
均不是角色责任证据。即使外部评分为 1，超时任务也不能生成正向能力 Skill，原始分数仍保留。

`learning_evidence.py` 是独立学习适配层，仅对“阻断约束：无（非矛盾补充说明）”等明确无阻断
的格式做学习侧归一化，记录转换规则并保留原文、源 SHA256 和原审计解析结果；存在实际阻断
或矛盾说明时不归一化。学习中的负面证据资格变化不改变原始审计控制结果、Verifier 或评分。

当前策略为 `evidence-aware-role-skill-benefit-v5`。成功或失败任务均分别检查 Manager、Executor、Auditor，
一次共享归因调用返回三个独立的角色判断，primary_role 仅保留全局根因描述，不作为独占学习门槛。
角色评审只判断证据资格，不提前以“常规操作”为由跳过 Proposer。是否有非显然、可迁移的方法由 Proposer 判断，最终由 Curator 接纳。
每个角色的经验必须引用自己的真实行为和合格的独立佐证。正向经验继续要求 clean/aligned 独立审计；
负面经验也可以引用 clean/needs_revision 审计或实际 Benchmark 失败。Auditor 不得只引用自身报告。
Proposal 的 JSON Schema 固定自身行为与独立佐证的引用槽位；正向 Auditor 仍引用合格审计和
角色外证据。NONE/DEFER 允许空证据，NEW/ENHANCE 必须通过本地独立证据校验。
Curator 的 ACCEPT 只表示内容审核通过。单任务来源先保存为 Candidate；至少两个不同任务指纹的独立来源支持同一方法后，才晋升为 Active Skill。重复轮次、重试和同一任务的不同运行不增加独立支持数。
未经 Curator 审核的 Proposal 仍留在运行目录，不能由 Router 注入；REJECT/DEFER 不激活。
置信度不足 0.8 时 DEFER，正向经验还需 requirements_covered=true。
未启用训练的角色仍记录检查结果，但其可学习经验标记 TARGET_ROLE_NOT_ENABLED，不调用其 Proposer/Curator，
不修改其 Skill 文件。不会把 Executor 的错误改标给 Manager 来绕过角色选择。

每批报告包含 `role_decisions`、`role_summary`、学习策略版本及源码 SHA256。
`library/learning_policy_history.json` 记录策略切换。已完成批次的旧报告不覆盖，后续学习进程加载新策略。
冻结库禁止追加策略记录。`.jsonl` 和冻结 Harbor 使用的 `.txt` JSONL 轨迹都读取。
未结束轮次中的已保存计划、输出和工具轨迹标记为 partial，并保留单独的文件 SHA256；
输入 Prompt 不算已经执行的行为。可读到原数据集 instruction.md 时将其作为任务要求来源。
partial 记录不能用作正向完成证据，也不能把缺失输出解释为从未尝试执行。

仅有 Manager 初始计划和失败评分、没有 Executor 行为或独立审计时，可以进入诊断 Review，
但 Manager 仍强制 DEFER，原因为 `MISSING_EXECUTION_PROGRESS_FOR_MANAGER_CAUSALITY`。
这避免把 Executor 未返回、推理缓慢等不确定性错误归给 Manager。实际基础设施故障始终排除。
其他门禁原因保存在 `learning_eligibility.reason_codes`，不再只给笼统的“无可信轮次”。

## Proposal 与 Curation

学习分两个阶段：每题的 Review/Proposer 只读取该题的压缩轨迹、审计证据和 Batch 开始时的旧库，
不读取其他题的 Proposal。每题保存独立的 `proposals.json`，无经验时仍记录 NONE/DEFER。
这些未审核经验存放在运行目录，尚未进入 Library 的 candidates，更不会被 Router 使用。

Batch 结束后才启动 Curator：先校验所有 Proposal 的 SHA256、训练角色、分割、学习策略、
来源引文和旧库摘要，拒绝被修改、重复任务或过期的暂存产物。Topic Curator 同时参考同角色的
全部 Batch Proposal，逐条审核；后续条目可与前面接纳的同模式经验合并，随后 General Curator
检查角色内跨主题重复。库只在这一阶段更新，下一个 Batch 才可使用新 Active Skill。

三角色有独立规则和成功/失败分支。输出由服务端 JSON Schema 约束，再本地验证：
证据编号必须存在，证据引文必须逐字匹配，不能包含当前任务答案、敏感信息或权限覆盖建议。
每个 Proposer 动作均需非空 `reason`。Adherence Analyzer 先按注入记录中的 key/version 读取当时的
Skill 历史版本，并通过每次 injection 对应的 `claude_stream_path` 将 Skill 与精确 Episode 轨迹关联，
不把同一角色的其他 Episode 当作该 Skill 的执行证据。注入记录本身不构成遵循证据，任务成功也不等于
Skill 被遵循。缺少历史版本、轨迹路径或无法匹配 Episode 时标记 unavailable/insufficient，不猜测。
本地绝对路径只用于训练侧解析，不传入模型可见的 bundle。
遵循度结论保存于 `skill_adherence`，并提供精确引用。无有效非显然经验时 NONE，应解释缺乏新颖性或与已有库
重合的部分；证据不足时 DEFER，必须说明缺少哪类证据。结果保存于
`role_decisions.<role>.proposal_diagnostics`，不能用空白 NONE 隐藏过滤原因。

失败经验可以总结已观察错误的预防或诊断方法，不要求该失败任务已经成功修复。
负面 Proposal 的 `proposal_context.repair_validation` 标记为 `not_validated`；Curator 不能仅因
没有成功修复过程而拒绝，但必须拒绝伪造修复成功、缺乏因果支持或违反原 MEA 协议的建议。
证据摘录优先选择具体失败、预算与契约事实，过滤纯标题和控制头，避免只引用“未完成：”。

Topic Curator 支持 ACCEPT/MERGE/UPDATE/REJECT/DEFER。General Curator 只在本轮确有该角色更新后，
检查角色内跨主题重复。MERGE/UPDATE 取保守并集，保留旧 procedure、verification、pitfalls 和证据；
超出 200 词则拒绝该更新，不悄悄删掉旧能力。支持数真实记录，单任务来源不得被宣称已获得跨任务验证。
Curator ACCEPT 只完成内容审核；Candidate 至少有两个不同任务指纹支持后才会成为 Active。

参考 Evo-Harness 的 [Topic Curator 与写入路径](https://github.com/A-EVO-Lab/a-evolve/blob/3c7c7b8c62a3f07da1d4407aa03f0ca079954bcf/evo_harness/terminal_bench.py#L1158)：
上游 Evo-Harness 的 Topic Curator 可接受单任务 Proposal。本项目在保留单任务 Candidate 的同时采用更严格的晋升门槛：ACCEPT 不直接激活，必须有两个独立任务指纹支持后才能成为 Active。
它的 [General Curator Prompt](https://github.com/A-EVO-Lab/a-evolve/blob/3c7c7b8c62a3f07da1d4407aa03f0ca079954bcf/evo_harness/terminal_bench.py#L495)
才要求从两个以上不同任务中归纳共同模式。这里不把跨任务归纳条件应用到普通角色 Skill 的接纳上。
配置默认 `min_independent_tasks=2`；显式设置为 1 仅用于复现旧实验，不作为当前推荐策略。

## 冻结与后续更新

```bash
lh-harness skills freeze --library experiments/skill_evolution/libraries/all_roles \
  --split-manifest experiments/skill_evolution/manifests/pilot-splits.json
```

冻结记录 version、SHA256、时间和测试 ID。每次角色调用前核对摘要，整个配对运行结束再核对。
正式测试不更新内容或 library 统计。要继续进化，应复制成新的训练库，在独立训练集上更新，
再创建下一个冻结版本。测试结果只能报告，不作为同一主实验的反向学习材料。

## Token 与成本

服务端 Usage 优先；角色执行与离线 adherence/attribution/proposer/curator/general_curator 分开统计。Adherence 分析成本计入学习成本，不计入 Benchmark 执行成本。
同模型 tokenizer 的 Skill 片段计数是 estimated，已经包含在角色输入中，不重复加入总 Token。
`learning_cost.json` 包含失败调用的已返回 Usage。所有尝试的学习成本均应汇总，不能只统计最后一次成功。
未提供价格时 API Cost 为 null；本地推理不等于硬件成本为零。
每次任务成本与包含离线学习摊销的成本分别报告，并同时报告成功率和 Tokens per Successful Task。
收集训练轨迹的 Agent 运行成本应另列，说明是否纳入研究预算。

## 逐任务在线学习

`bash train_skills.sh manager --task all` 默认启动独立 Online Skill Learning Experiment。
`--task code-from-image cancel-async-tasks` 可按给定顺序选择子集；不读取旧 run-index。
默认 `--batch-size 5`，每题复用官方冻结 MEA 和原始 Verifier。每题评分保存后，调用
`skills learn --phase online --stage propose`，只反思该题，不写库。5 题均反思完成后，调用
`skills learn --phase online --stage curate --proposal-input ... --flush`，仅消费本批暂存 Proposal，
不重新读取轨迹，不重跑 Review/Proposer。末批不足 5 题也审核。
审核更新结束后才启动下一批。所有批次结束后自动冻结最终库，重新评估选中的全部任务；
复测使用新环境和独立日志，`phase=test`、`training.enabled=false`，不再学习。
同一轮使用新建且共享的库，后续任务通过原有确定性 Router 使用此前接纳且达到支持门槛的 Active Skill。
每批学习不保证每题产生 Skill；归因和审计证据规则保持不变，单任务经验经审核后先进入 Candidate。

在线任务清单保存为实验独立的 `online-split.json`，不重写正式分割。
`manifest.json` 和每批学习报告明确标记 `online_skill_learning`，成本仍单列；
Manifest 的每题 `reflection` 记录 Proposal 路径、文件摘要、角色决策、成本和异常；`batches` 记录
批次审核结果与更新前后库摘要。逐题 adherence/attribution/proposer 和批次 curator/general_curator 成本
分别记录并各累加一次，尚未审核的已完成反思也计入总成本。
最终训练集复测独立保存在 `evaluation/`，标记 `training_set_recheck`、`held_out_evaluation=false`，不能用作未见任务泛化结果。
`library/online_provenance.json` 记录实际观察的任务，即便该题没有接纳任何 Skill。
正式冻结库评估会拒绝与这些已观察任务重叠的 held-out 测试。

学习失败保留原始分数并继续下一批；模型、依赖、任务身份或评分产物异常则中止，
禁止回退云端或修改原始评估标准。`--dry-run` 不启动 Benchmark 或模型请求。
原来的历史轨迹学习通过 `bash train_skills.sh manager --offline` 保留。

## 原目录续跑

`bash train_skills.sh manager --resume <原运行目录> --from-task 3` 从原清单第 3 题（包含该题）重新执行，
保留前两题结果、角色选择、模型配置、原 Batch 大小和当前 Skill 库。剩余任务从续跑点重新分批，
最终复测仍覆盖原评估清单，不只复测续跑后半段。角色、数据集、权重或服务参数变化则拒绝续跑。

续跑前的根配置、Manifest、汇总与 Library 快照保存在原目录的 `resumes/<id>/`；新任务与 Proposal
写入原题目目录下的 `attempt-<id>/`，批次编号接着递增，旧评分与轨迹不覆盖。原两任务激活配置
在保存旧配置后升级为当前的双独立任务晋升策略。已冻结或已有最终复测的目录禁止继续训练。

当前成功率仅按各题最新选定结果统计，被替代尝试放入 `superseded_tasks`；运行、反思和审核成本
包含所有尝试且各计一次。缺失 Usage 仍为 unavailable/null。已观察任务集合取并集，续跑不能
抹去此前已观察题目或将其重新标记为未见测试题。`--dry-run` 不写文件，模型健康检查失败也不改旧运行。
