# Role-conditioned Skill Evolution

本系统学习可复用操作经验，不更新 LLM 权重。入口在 `src/lh_harness/skills/`，默认关闭。
保留两条评估轨道：Original LLH 与 Skill-enhanced LLH。八种角色组合是增强轨道的消融配置，
不是八个不同的 Benchmark 或评分器。

正常 MEA 只做确定性检索、附加建议和使用记录；默认不限制 Skill 注入数量或注入 Token 总量。Proposer、Curator 和跨角色归因
只在任务评分后的显式学习阶段调用。逐题 Review/Proposer 不写库；Batch 结束后才运行 Curator。
学习失败写独立报告，不改变任务完成状态或 Benchmark 奖励。

## 开始使用

终端便捷入口现在默认在线：`bash train_skills.sh executor --task all`。
也可用 `--task code-from-image cancel-async-tasks` 选择任务：按 5 题一批执行和原始评分，
每题独立反思提出 Proposal，批次末统一审核更新后运行下一批，最后冻结最终库复测全部训练任务。
`--dry-run` 只预演；`--offline` 才读取旧历史轨迹。
在线学习单独标记，不与下面的冻结库测试主实验混用，详见根目录 `start.md`。

在项目根目录，使用已有 Python 环境：

```bash
export PYTHONPATH="$PWD/src"
conda run --no-capture-output -n terminal-bench-2-1 python -m lh_harness skills --help
conda run --no-capture-output -n terminal-bench-2-1 python -m lh_harness skills init \
  --library experiments/skill_evolution/libraries/all_roles
```

安装本项目后可直接使用 `lh-harness skills ...`。独立命令：

```bash
lh-harness skills list --library <library>
lh-harness skills inspect executor/cli/<name> --library <library>
lh-harness skills stats --library <library>
lh-harness skills rollback executor/cli/<name> --version 1 --library <library>
```

`skills learn --roles` 接受 `manager`、`executor`、`auditor`、任意逗号组合和 `none`。
`--run TASK_ID=RUN_DIRECTORY` 可以重复；也可以提供 `--run-index` JSON 映射。
每个目录必须仅包含一个任务的 `rounds.jsonl`，不能把整个多任务 job 当作一个经验。

```bash
lh-harness skills learn --config experiments/skill_evolution/configs/all_roles.toml \
  --roles manager,executor,auditor \
  --split-manifest experiments/skill_evolution/manifests/pilot-splits.json \
  --run-index experiments/skill_evolution/manifests/training-runs.json \
  --output experiments/skill_evolution/results/skill_enhanced/new-learning-batch
```

默认等待 5 个独立任务后统一审核。最后不足一批时可用 `--flush`；单任务经验经 Curator 接纳后
也可成为 Active Skill，不要求两个任务支持。当前所有候选、活跃项、历史、索引和统计存放于同一个 library 下。

## 配置与权限

```toml
[skills]
enabled = true
library_dir = "experiments/skill_evolution/libraries/executor_only"
tokenizer_path = "/path/to/the/same/model/tokenizer.json"
phase = "train"
max_per_role = "unlimited"
token_budget_per_role = "unlimited"

[skills.training]
enabled = true
train_roles = ["executor"]
batch_size = 5
min_independent_tasks = 1
learn_from_success = true
learn_from_failure = true

[skills.usage]
enabled = true
use_roles = ["manager", "executor", "auditor"]

[skills.router]
method = "bm25"
max_skills_per_role = "unlimited"
```

`phase=train` 表示收集训练经验、允许显式离线更新；不表示 MEA 会自动调用学习模型。
`phase=test` 要求 library 已冻结。`phase=online` 的离线学习输出标记为
`online_skill_learning`，不能作为冻结库主实验。验证阶段禁止更新。

默认 `min_independent_tasks=1`：Curator 接纳即激活，不再对所有 Skill 强制要求两个独立任务。
显式设置为 2 仍可复现旧实验策略。支持数和来源仍真实记录；单任务支持不能被表述为多任务验证。

`"unlimited"` 表示无上限；省略上述三个限制字段时也默认无上限。Router 按 BM25 排序，
注入当前角色、GUI/CLI 类型和 Trigger 都匹配的全部 Active Skill，不再截取前两条或卡在 400 Token。
这只取消 Skill 注入上限，模型上下文容量、输出参数和任务时间预算仍沿用原配置。

为复现旧实验，仍支持显式填写非负整数；0 表示不注入。角色 Token 预算也支持分别配置，
未指定的角色默认无上限。若同时配置两个数量限制，使用其中较小的有限值。
无上限模式不依赖 Tokenizer 才能注入；缺少同模型 Tokenizer 时注入 Token 数为 null/unavailable。
显式有限 Token 预算仍要求同模型 Tokenizer，否则不注入。离线片段计数标记 estimated；
服务端角色 Usage 单独记录为 reported/unavailable。已有结果目录中的历史配置不修改。

Skill 仅作为方法建议。原任务、Task Contract、Task State、Auditor 报告和权限不被替换。
Auditor Skill 不允许写入程序；包装层保留原 Adapter 的工具限制和工作区保护，不能增加工具。
格式修复和最终回复不注入角色 Skill。

参见 [架构](SKILL_ARCHITECTURE.md)、[学习](SKILL_LEARNING.md)、[实验协议](SKILL_EXPERIMENT_PROTOCOL.md)
和 [Baseline 保护](BASELINE_PRESERVATION.md)。
