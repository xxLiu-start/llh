# 原始评估与 Batch Skill 训练及最终复测

先进入主目录，下面每条命令可直接复制执行：

```bash
cd /data0/liuxule/project/LHH/LongHorizon-Harness
```

本地模型：`/data0/liuxule/project/models/Qwen3.8-27B`。
当前数据集：a-evolve 缓存的 **TB2.0 / Inspect → Harbor，共 89 题**，不是 TB2.1。
Skill 训练更新经验库，不更新模型权重。

Skill 使用默认不限制条数或注入 Token 总量：注入当前角色、GUI/CLI 类型和触发条件匹配的全部
Active Skill，按 BM25 排序。配置模板的 `max_per_role`、`token_budget_per_role` 和
`skills.router.max_skills_per_role` 均为 `"unlimited"`，下方训练命令直接生效。
模型上下文容量和输出参数仍沿用原配置；已有运行保存的配置不变。

## 1. 原始 LLH 评估

全部 89 题，不训练、不使用 Skill：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --serving-manifest evaluation/validation/qwen27b-vision-serving.json
```

只评估一道题：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --task log-summary-date-ranges --serving-manifest evaluation/validation/qwen27b-vision-serving.json
```

多题可重复添加 `--task 题目ID`。结果保存在 `output/original-a-evolve-tb20-qwen27b/`，每次创建新 job。

## 2. 在线训练：每 5 题统一学习，全部训练后再复测

`train_skills.sh` 现在默认在线运行，**不读取已有训练日志索引**：

**每题执行并评分 → 该题独立反思、提出 Proposal → 满 5 题统一 Curator 审核、归纳去重和更新库 → 下一个 Batch → 冻结最终库 → 复测。**

默认 `--batch-size 5`，最后不足 5 题也统一审核。每道题评分后，先进行三角色证据检查和所选角色的
Proposer，总结该题自己的经验并保存待审核 Proposal，不调用 Curator、不写 Skill 库。
Batch 结束后，Curator 同时参考本批同角色的全部 Proposal，审核、跨任务去重合并，再更新库。
Batch 内各题使用同一份旧库，后续 Batch 可以使用此前已接纳的 Active Skill。
反思与审核均在任务评分完成后运行，不占用题目的原始 MEA 时间预算。
训练完成后自动冻结最终库，在新环境中重跑本轮选中的全部任务，
复测期间关闭 Training、只使用最终 Skill，不覆盖训练阶段的评分。

全部 89 题分为 18 个 batch（末批 4 题），随后复测 89 题，共执行 178 次任务。

### 2.1 选择一个任务

只训练 Manager，运行 `code-from-image`：

```bash
bash train_skills.sh manager --task code-from-image
```

只训练 Executor，运行 `cancel-async-tasks`：

```bash
bash train_skills.sh executor --task cancel-async-tasks
```

只训练 Auditor，运行 `code-from-image`：

```bash
bash train_skills.sh auditor --task code-from-image
```

### 2.2 选择多个任务

按命令给定的顺序执行。下面 3 题作为不足 5 题的最后一批统一学习，然后复测全部 3 题：

```bash
bash train_skills.sh manager,executor --task code-from-image cancel-async-tasks largest-eigenval
```

也支持重复参数或逗号分隔：

```bash
bash train_skills.sh all --task code-from-image --task cancel-async-tasks
```

```bash
bash train_skills.sh all --task code-from-image,cancel-async-tasks
```

### 2.3 全部 89 题：八种角色组合

按需选择一条命令执行；多个组合顺序运行，避免争用同一个模型服务。

```bash
# 仅 Manager
bash train_skills.sh manager --task all

# 仅 Executor
bash train_skills.sh executor --task all

# 仅 Auditor
bash train_skills.sh auditor --task all

# Manager + Executor
bash train_skills.sh manager,executor --task all

# Manager + Auditor
bash train_skills.sh manager,auditor --task all

# Executor + Auditor
bash train_skills.sh executor,auditor --task all

# 三个角色
bash train_skills.sh all --task all

# 原始控制：执行全部任务，但关闭 Skill 使用和训练
bash train_skills.sh none --task all
```

省略 `--task` 也表示全部 89 题。例如 `bash train_skills.sh manager` 现在会实际运行全部任务。
选择训练角色不影响原有三角色 MEA 的执行：选 Manager-only 时 Executor、Auditor 仍正常工作。

## 3. 常用选项

显式指定 batch 为 5，并在最后自动复测全部选择的任务：

```bash
bash train_skills.sh all --batch-size 5 --task all
```

只运行所选顺序中的前 3 题：

```bash
bash train_skills.sh all --task all --limit 3
```

只检查任务选择和配置，不启动模型请求、不启动 Docker、不写结果文件：

```bash
bash train_skills.sh manager --task code-from-image cancel-async-tasks --dry-run
```

列出默认的全部任务顺序，也可以使用预演：

```bash
bash train_skills.sh all --task all --dry-run
```

指定容易记住的新输出目录：

```bash
bash train_skills.sh executor --task code-from-image cancel-async-tasks --output output/online-executor-01
```

输出目录必须不存在，避免覆盖旧结果。默认自动创建带组合名、时间和随机编号的新目录：
`experiments/skill_evolution/results/skill_enhanced/online-<组合>-<时间>-<编号>/`。
`none` 控制组的默认输出位于对应 `results/baseline/`。

指定宿主机和 Docker 都能访问的本地地址：

```bash
bash train_skills.sh all --task code-from-image --base-url http://172.17.0.1:8010/v1
```

地址优先级为 `--base-url`、`LLH_LOCAL_BASE_URL`、组合配置中的地址。
当前服务不要求 API Key；如自行开启鉴权，使用 `--auth-scheme bearer --api-key-env 环境变量名`。
请勿把真实密钥写入命令、配置或日志。不会自动切换模型或云端服务。

默认在最终复测前自动冻结库，不需要另外运行冻结命令。只训练、不进行最终复测时：

```bash
bash train_skills.sh executor --batch-size 5 --task code-from-image cancel-async-tasks --skip-final-eval
```

### 动态选题：不固定题目或数量

每次运行都重新确定任务清单，batch 数量随选择变化；默认最终评估跟随本次训练选题。
推荐修改 [任务清单文件](evaluation/skill-training-tasks.txt)，然后一直使用同一条命令：

```bash
bash train_skills.sh manager --batch-size 5 --task-file evaluation/skill-training-tasks.txt
```

清单目前默认是一行 `all`，表示全部 89 题。想运行部分任务，就将 `all` 替换为想运行的 ID，
每行一个，例如：

```text
code-from-image
cancel-async-tasks
largest-eigenval
```

增删或调整文件行序，就会改变本次任务、分批和默认最终评估，不需要改启动脚本。
`all` 必须单独使用，不能和具体 ID 混写。每次选题的文件摘要和实际任务顺序都会保存。

也可以从任意一个已有 Harbor job 目录，自动读取其中的任务 ID：

```bash
bash train_skills.sh manager --batch-size 5 --tasks-from-run output/original-a-evolve-tb20-qwen27b/original-tb20-qwen27b-15287b9a58
```

将目录换成其他 job 就会自动选择另一些任务，数量不固定为 22。
这里仅读取题目 ID 和顺序，所有任务都会重新执行，不使用旧轨迹进行训练。
同一题的多个历史尝试只选择一次。
`--task`、`--task-file`、`--tasks-from-run` 三选一。

如需最终评估另一组题，可独立指定 `--eval-task`；省略时始终评估本次训练所选全部任务：

```bash
bash train_skills.sh manager --batch-size 5 --task-file evaluation/skill-training-tasks.txt --eval-task all
```

以上命令的 `manager` 可换成其他单角色、逗号组合或 `all`。
任意命令末尾加 `--dry-run` 可先查看实际选中的任务、分批数量与最终评估范围，不执行任务。

## 原目录续跑

从原运行第 3 题开始（包含第 3 题，重新执行），保留前两题评分与现有 Skill 库：

```bash
bash train_skills.sh manager --resume experiments/skill_evolution/results/skill_enhanced/online-manager_only-20261122T091035Z-5486973f --from-task 3
```

该运行原 Batch 大小为 3，原任务共 21 题；从 `kv-store-grpc` 起续跑剩余 19 题，按原 Batch 大小
重新分组。最终仍冻结库并复测原来的全部 21 题。`--from-task` 也接受原任务 ID。

所有输出都在同一个原目录中：

- `resumes/<续跑ID>/` 保存续跑前的 Manifest、配置、汇总与 Skill 库快照。
- `tasks/003-kv-store-grpc/attempt-<续跑ID>/` 等目录追加新的任务日志，旧日志保留。
- `learning/proposals/003-kv-store-grpc/attempt-<续跑ID>/` 追加新一轮逐题 Proposal。
- `learning/batch-002/` 起追加批次审核；`library/` 继续更新原库，保留 Skill 历史版本。
- 根目录 `manifest.json`、`summary.json` 更新当前进度；被重跑的旧尝试保存在历史记录中，成本继续计入。
- `evaluation/` 保存全部训练完成后的最终复测结果。

在命令末尾添加 `--dry-run` 可只检查续跑计划，不写文件、不调用模型。启动前仍完整检查模型、
离线依赖、模型权重和推理服务参数；检查失败时不改写原目录。已冻结或已有最终复测的目录禁止继续训练。

## 4. 在线运行的行为与结果

- 每题先独立反思并暂存 Proposal，每个 Batch 结束后才统一审核与更新，不保证每题或每批都接纳 Skill。
- 成功和失败均分别检查三个角色；未启用训练的角色只记录检查结果，其可用经验标记 `TARGET_ROLE_NOT_ENABLED`，不更新其库。
- `AgentTimeoutError` 会检查已完成轮次及标为 partial 的已保存行为，结合独立审计或真实评分失败审查负面经验；超时本身不代表某角色有错，实际基础设施故障继续排除。
- 学习策略为 `evidence-aware-negative-learning-v4`：支持从 clean/needs_revision 审计中学习契约错误；只在独立学习层兼容“无（说明）”格式，原始评分不变。NONE/DEFER 必须记录具体原因。
- 单任务经验通过 Batch Curator 审核即可成为 Active Skill，不再等待第二个独立任务；未经审核的 Proposal、REJECT 和 DEFER 不会注入。
- 训练阶段的新 Skill 仅影响后续 batch。最后使用最终库复测所有训练任务；这是训练集复测，不是未见任务泛化评估。
- 每题先保存原始评分，再反思；Batch 内禁止修改库。反思或审核失败独立记录，继续处理其他任务和 Proposal；模型或依赖故障则停止，不在训练未完成时启动复测。
- 任务顺序固定并记录在 Manifest，不并行执行；原始 Prompt、角色权限、只读 Auditor、时间预算和评分器保持原样。

结果目录包括：

```text
config.toml                 # 本轮角色与在线配置
online-split.json           # 本实验专用的学习任务清单，不覆盖正式 train/validation/test 分割
manifest.json               # 任务顺序、模型/环境、各题评分、学习决策、库更新前后摘要
summary.json                # 逐题反思、批次审核和最终复测分别统计；各阶段成本只计算一次
library/                    # 本轮共享的 Skill 库及在线观察任务记录
tasks/001-<task>/          # 每题独立的原始 Harbor 日志和评分结果
learning/proposals/001-<task>/ # proposals.json、逐题反思报告、原始模型输出、learning_cost.json
learning/batch-001/         # 批次审核 learning_report.json、原始 Curator 输出、learning_cost.json
evaluation/config.toml      # phase=test、training.enabled=false
evaluation/manifest.json    # 最终库摘要、相同任务清单、新一轮独立评分
evaluation/summary.json     # 最终训练集复测的成功率、角色 Token 等
evaluation/tasks/001-<task>/ # 独立复测轨迹；不再学习
```

训练结果标记为 `online_skill_learning`、`phase=online`；最终复测标记为
`training_set_recheck`、`learning_enabled=false`。两者均为 `held_out_evaluation=false`。
它与冻结库主实验分开保存；在线用过的题目不能再被称为该库的未见测试题，
即便该题最终没有产生 Skill。冻结库配对启动器会检查在线观察记录，拒绝这种数据泄漏。
原有正式分割清单与 Original 评估路线不会因此修改。

## 5. 模型与依赖检查（按需）

逐题汇总文件为每轮运行目录内的 `tasks/task_summary.json`，包含所有选中题目的执行/反思 Token、
成功状态、耗时、三角色使用的 Skill、提出的 Skill 内容、审核结果与拒绝/延迟原因。
共享 Batch 审核成本在 `batches` 中单独记录一次；注入 Token 标为 estimated 且不重复加入总量，
缺失 Usage 为 null/unavailable。旧 NONE 未记录原因时如实说明，待审核 Proposal 的 accepted 为 null。

新启动与续跑会自动在进度更新时刷新。对已经运行的旧进程，可另开终端持续刷新（只读取日志，不调用模型）：

```bash
conda run --no-capture-output -n terminal-bench-2-1 python experiments/skill_evolution/scripts/report_task_details.py --run-dir experiments/skill_evolution/results/skill_enhanced/online-manager_only-20261122T091035Z-5486973f --watch
```

去掉 `--watch` 可手动生成一次。当前这轮已启动后台汇总器，每 15 秒刷新；不会覆盖任务原始日志或评分。

在线启动器会检查模型协议、工具调用、SSE、Usage、所需图像能力、Docker 和离线评分依赖。
预演 `--dry-run` 只检查本地配置，不代表实际服务已经通过健康检查。

8010 已有服务时无需再次启动。服务尚未启动时，在另一终端执行：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash evaluation/start_qwen27b_vllm.sh --enable-vision --max-images 32 --gpu-memory-utilization 0.85 --max-model-len 1000000 --max-num-seqs 8 --max-num-batched-tokens 8192
```

检查离线评分依赖；如未准备好，先运行准备命令：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --check-verifier-runtime
```

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --prepare-verifier-runtime
```

检查 Docker 内的模型、多图、工具与流式输出，不运行题目：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --check-model-docker
```

## 6. 保留的离线训练入口

只有显式添加 `--offline` 才读取已有训练日志，沿用正式训练集分割：

```bash
bash train_skills.sh manager --offline
```

```bash
bash train_skills.sh executor --offline --run-index experiments/skill_evolution/manifests/training-runs.json
```

离线模式不支持 `--task`。冻结库主实验的配置、启动和比较命令见
[实验协议](docs/SKILL_EXPERIMENT_PROTOCOL.md)，其他说明见 [Skill 系统](docs/SKILL_SYSTEM.md)
和 [学习机制](docs/SKILL_LEARNING.md)。

