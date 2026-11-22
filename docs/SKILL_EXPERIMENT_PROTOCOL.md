# 双轨实验协议

两条轨道使用相同任务版本、ID、初始 Docker 镜像、模型权重、Model ID、Endpoint、Backend、
采样、Reasoning、输出上限、上下文、工具权限、角色时间预算、MEA 轮数、并发、重试、Verifier 和奖励。
唯一机制差异是 Skill。测试前冻结库；主实验与在线学习分开。

## 在线实验入口

`bash train_skills.sh all --task all` 依次运行当前数据集的全部 89 题，每 5 题原始评分后统一学习一次，末批不足 5 题也处理；训练完成后冻结最终库复测全部 89 题。
角色参数接受任意组合，`--task` 可选择一个或多个 ID，顺序保存在 Manifest；
`none` 为相同官方 MEA 的无 Skill 控制，不调用学习器。
一个 batch 完成后才更新同一轮 Skill 库，新增能力用于后续 batch；最终复测关闭训练，只使用同一份最终冻结库。

在线实验写入独立 `online-*` 目录，标记 `held_out_evaluation=false`，其序列内表现不能
作为冻结库主实验的未见任务泛化结果。全量 89 题在线学习意味着这些题都已经被观察。
库即使冻结也保留在线观察清单；下面的正式配对入口检查该清单与 test 的交集，
即便没有生成 Active Skill 也不能绕过隔离。仅对从未观察过、且与训练证据互斥的任务
才允许后续独立冻结测试。原始 `eval/`、任务预算、Verifier 和正式分割清单保持不变。
简明的全部角色组合、选题、预演和离线命令见项目根目录 `start.md`。

## 配置

`experiments/skill_evolution/configs/` 提供 baseline、manager_only、executor_only、auditor_only、
manager_executor、manager_auditor、executor_auditor、all_roles 共八组。
disabled_compatibility 只做代码兼容性检查，不代替 Original Baseline。

不带 `.local` 的 TOML 用于冻结 TB 实验，指定原生 Claude Code/Messages 和当前 8010 服务。
`.local.toml` 保留现有根包 CLI 的 Codex/Responses 策略代理设置；八组策略完全一致，
只有 `[skills]` 不同。启动代理沿用实验 README 的 `lh_harness.local_gateway`。
这些根包连通性运行不能当作 Harbor Benchmark 评分。

## 本地模型

当前真实 ID：`/data0/liuxule/project/models/Qwen3.8-27B`；Docker 访问
`http://172.17.0.1:8010/v1`。原始与增强都使用同一服务，顺序运行。
vLLM 服务需支持所选原生 API、工具、SSE、Usage；图像任务还需开启视觉编码器。
API Key 仅放环境变量。当前无鉴权，客户端 local 只是占位符。
协议不兼容、模型缺失或接口不可用时拒绝运行，不切到其他模型或云端。

服务与离线安装预检沿用已验证的 `evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh`。
评分依赖、原始镜像、上游测试资源和 Claude CLI 必须提前准备；正式配对过程中不下载这些依赖。

## Terminal Benchmark 配对

先完成独立训练，固定并冻结库。以下会运行两个真实 Harbor job，原始代码来自受保护的官方 Worktree：

```bash
cd /data0/liuxule/project/LHH/LongHorizon-Harness
conda run --no-capture-output -n terminal-bench-2-1 python \
  experiments/skill_evolution/scripts/run_paired_eval.py \
  --variant-config experiments/skill_evolution/configs/all_roles.toml \
  --split-manifest experiments/skill_evolution/manifests/pilot-splits.json \
  --task log-summary-date-ranges \
  --output experiments/skill_evolution/results/comparison/NEW_PAIR
```

独立运行两轨时使用 `launchers/run_baseline.sh` 或 `run_skill_enhanced.sh`，其余参数相同。
Original 不依赖 Skill Library；增强 test 要求库已冻结。输出目录必须全新，不覆盖旧数据。
`--repetitions N` 为两个轨道设置相同尝试数。保留默认角色和任务预算，不缩短任一方来制造 Token 改善。

Manifest 记录官方 Commit、所有实际加载的冻结源码摘要、数据集摘要、每个权重文件 SHA256、
真实服务探测、GPU/服务快照、角色采样和原始配置、库版本与摘要。配对前比较剔除 Skill 的配置，
两轨之间检查模型/服务、源码和库；接口或依赖故障中止并标记无效，保留原始分数与日志。

```bash
conda run --no-capture-output -n terminal-bench-2-1 python \
  experiments/skill_evolution/scripts/compare_results.py \
  experiments/skill_evolution/results/comparison/NEW_PAIR/manifest.json \
  --output experiments/skill_evolution/results/comparison/NEW_PAIR/comparison.json \
  --training-cost <combined-learning-cost.json>
```

只能比较相同任务 ID 和同次数的权威奖励。未返回的 Usage/成本/指标保留 null。
一题一次的 pilot 不是全套成功率结论，不能证明泛化增益；多次独立运行需报告均值、方差和任务覆盖。
没有注入 Active Skill 时，明确记录为冻结空库/候选库兼容性 pilot，不能称作 Skill 性能提升。

## WeaveBench 与 OSWorld-V2

```bash
python experiments/skill_evolution/scripts/prepare_gui_adapter.py weave \
  --output experiments/skill_evolution/compat/weave-NEW
python experiments/skill_evolution/scripts/prepare_gui_adapter.py osworld \
  --output experiments/skill_evolution/compat/osworld-NEW
```

Original 的 CUA_HARNESS_SOURCE_DIR 指向官方 Worktree 中对应 cua-harness。
增强指向生成的 overlay，并设置 LLH_SKILL_SETTINGS 为冻结 test 配置的 JSON。
其余使用各自原 README 的同一启动命令、任务版本、VM、网站、环境、模型和评分器。
仅生成 overlay 或 mock 测试不能作为 GUI 实测。资产、KVM/VM 或认证缺失时应报告未运行。
