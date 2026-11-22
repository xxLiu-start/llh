# GitHub 源码副本与运行准备

GitHub 仓库保存当前源码、实验适配层、配置模板、训练入口、任务 ID 列表、测试及文档。
原始 LLH 的上游归属和许可证保留。上传基于目标仓库现有 `main` 新增提交，不导入本地
Git 历史中的大量实验输出，不改变开发目录的分支和已有未提交修改。

以下本地内容不随源码上传：

- `output/`、`his/`、实验 `results/`、运行日志和验证输出。
- 下载的数据集、模型权重、Python/Node 环境、依赖缓存和离线二进制。
- 已训练的 Skill 库、Worktree、源码保护备份和当前机器的运行状态 Manifest。
- `.env`、真实 API Key 和其他本地凭据。

上游评测代码自带的任务定义、示例和测试 fixture 保留；另外下载的 TB 数据集不上传。
源码中的模型路径、GPU 编号、Endpoint 和 Conda 环境名称是当前机器的配置，换机器后
应显式调整。不要将配置模板视为目标机器已经具备对应模型和依赖。

## 在新克隆中准备原始评估副本

配对评估和在线训练需要独立的官方 Original Worktree。首次使用时，在仓库根目录执行：

```bash
git fetch https://github.com/AMAP-ML/LongHorizon-Harness.git d53df0aa11e8506dfc9518ef63620c62baee04d3
mkdir -p experiments/skill_evolution/worktrees
git worktree add --detach experiments/skill_evolution/worktrees/original d53df0aa11e8506dfc9518ef63620c62baee04d3
python experiments/skill_evolution/scripts/validate_manifest.py
```

已经存在 Original Worktree 时先检查其状态，不覆盖、不重置。
下载数据、安装 Backend CLI、准备评分依赖和检查 Docker 内模型连接的步骤见
[本地评估准备说明](../evaluation/README.a-evolve-tb20.zh-CN.md)。

`evaluation/validation/qwen27b-vision-serving.json` 属于真实服务运行证据，本次未上传；
运行前须按实际服务重新记录，不能编造框架版本、权重、量化、上下文或硬件信息。
缺失数据、缓存或服务时，预检应阻止 Benchmark 启动，不能回退云端。

准备完成后，原始评估、角色组合训练、Batch 更新及续跑命令见 [start.md](../start.md)。
上传源码不代表新机器已通过完整 Benchmark，也不包含已有训练结果。
