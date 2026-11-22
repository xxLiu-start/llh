# 原始 LLH 保护与恢复

开发目录为 `/data0/liuxule/project/LHH/LongHorizon-Harness`。开始时 main 指向
`4bf0e746488f00d50fac7be4d0886d42db9e774f`，存在 17 个已修改文件、23371 个删除记录和未跟踪内容。
这些删除和修改不是本次 Skill 开发造成，不能 reset、清理、切换或覆盖。

备份位于 `experiments/skill_evolution/protection/20261121T100333Z/`：
Git 状态、Commit、分支、二进制 diff、实际工作源码归档和 SHA256；旧本地实验配置另行备份。
独立增强分支为 `skill-evolution/role-conditioned-20261121`，对应 nested enhanced Worktree。
当前主目录分支未切换；新增实现也同步至增强 Worktree 以独立保存。

干净的官方原版在 `experiments/skill_evolution/worktrees/original`，固定 Commit：
`d53df0aa11e8506dfc9518ef63620c62baee04d3`。该工作树不写入 Skill 或修改。
Original CLI 可用其原有 Base URL 连接本地模型，保留原始配置能力。

## 保护清单

`experiments/skill_evolution/manifests/baseline-protection.json` 保存三个冻结 Benchmark 的
1260 个评估文件及根包 Manager/角色 Prompt/Auditor/types 等五个内核文件摘要。
不包含运行产物、下载环境和动态数据集适配输出。

```bash
python experiments/skill_evolution/scripts/validate_manifest.py
```

任何已保护文件修改或缺失都会拒绝实验。本次不编辑 `eval/` 下原始评估代码，
不更改成功标准、任务要求、Task Contract/State、路由、审计权限或结果奖励。
三角色 Prompt Builder 的增强包装位于 skills/integration.py；原 Builder 源文件不动，
无 Skill Context 时返回逐字相同的 Prompt。

## 既有差异与兼容层

开始开发前，主目录冻结 TB 中 harbor_agent.py、harbor_backends/claude_code.py 和
assets/claudecode_request_proxy.py 已与官方 d53 存在历史差异。此次保留它们，不倒退覆盖用户修改。
新的配对进程将官方 Worktree 的冻结模块置于 PYTHONPATH，而非把当前副本冒充官方原版。

新版 Harbor 把 extra_env 变为只读属性，官方旧类仍在构造器赋值。实验适配层
PristineOriginalAgent 只补兼容 setter。离线 CLI/依赖安装、CLI 事件 role 显式归一化等
已有本地服务传输兼容层共用于两轨，写入 Manifest；不改变 MEA 或 Verifier。
PristineOriginalAgent.run 与官方 CuaHarnessClaudeCodeAgent.run 是同一个继承方法。
Skill-enhanced 只在 Adapter 边界追加建议，同样直接继承这个原版 run。

隐藏上游测试资源仅在 Harbor 解题结束上传测试时传入，不在 Agent 阶段挂载。
共享依赖缓存只读；不把任务解答或隐藏断言注入角色 Prompt。

## 恢复

运行官方 Worktree 即得到不含本次改造的 Original；不需要撤销用户主目录修改。
回滚某个 Skill 用 `skills rollback`，冻结库先另建训练副本，不能原地解冻主实验。
需要取回开发前某文件时，从 working-code.tar.gz 提取到单独临时目录进行比较，再选择性恢复。
不要直接把整个备份覆盖当前工作区，也不要执行 git reset --hard。
