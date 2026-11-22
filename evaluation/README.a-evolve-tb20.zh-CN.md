# 原始 LLH + 本地 Qwen：a-evolve 的 89 道题

按用户选择，暂时使用 a-evolve 已缓存的这套题，记录为 **TB2.0 / Inspect → Harbor 适配版**。
不将它标记为 TB2.1；先前的 TB2.1 配置保留，后续可单独切换数据集。

源目录：`/data0/liuxule/project/.cache/a-evolve-tb2/challenges`。
生成目录：`eval/TB-harness/datasets/a-evolve-tb20-harbor`。
配置：`evaluation/original_harness_a_evolve_tb20_qwen27b.local.yaml`。

## 运行

先在一个终端启动模型服务，并保持该终端运行。用户已为本次实验分配 GPU `4,5,6,7`：

```bash
cd /data0/liuxule/project/LHH/LongHorizon-Harness
CUDA_VISIBLE_DEVICES=4,5,6,7 bash evaluation/start_qwen27b_vllm.sh \
  --enable-vision --max-images 32 \
  --gpu-memory-utilization 0.85 --max-model-len 1000000 \
  --max-num-seqs 8 --max-num-batched-tokens 8192
```

此入口复用 a-evolve 的 vLLM 启动配方：四卡 Tensor Parallel、百万 Token 上下文、
原有 YaRN 和工具解析器、关闭 thinking。默认监听 `0.0.0.0:8010`，让宿主机和 Harbor 容器
访问同一个服务；上述命令显式启用视觉编码器，每次请求最多 32 张图片，不启用 API Key 鉴权。
已有服务占用 8010 时，应先确认并停止该服务再启动，
不要同时加载第二份模型。可添加 `--dry-run` 只查看命令。

等待启动完成后，在另一个终端先检查宿主机接口：

```bash
cd /data0/liuxule/project/LHH/LongHorizon-Harness
curl --noproxy '*' -fsS http://127.0.0.1:8010/v1/models
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh \
  --base-url http://127.0.0.1:8010/v1 --check-model-only
```

该检查不启动 Benchmark，也不验证 Docker 连通性。实际评估使用容器可访问的宿主机网关，
不能将容器的 Endpoint 配成 `127.0.0.1`。确认后再执行：

```bash
cd /data0/liuxule/project/LHH/LongHorizon-Harness
export LLH_LOCAL_BASE_URL="http://$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'):8010"
export LLH_LOCAL_AUTH_SCHEME=none

# 检查评估实际使用的 Endpoint，不启动 Benchmark。
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --check-model-only

# 已完成的准备步骤；再次执行会校验并复用，不覆盖源数据。
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --prepare-only

# 一次性准备评分依赖；下载发生在准备阶段，不调用模型。
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --prepare-verifier-runtime

# 核对 89 题的依赖缓存与原脚本摘要，成功应显示 status=ready。
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --check-verifier-runtime

# 从 Docker 内验证真实工具调用、SSE、Usage、图片和错误处理，不运行题目。
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --check-model-docker

# 查看 Harbor 最终配置，不启动模型或容器。
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --print-config

# 先试跑 1 题。
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --limit 1

# 正式跑全部 89 题，每题 1 次、单任务并发。
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh
```

脚本使用已有 Conda 环境 `terminal-bench-2-1`；该环境名不代表本次数据集版本。
可用 `CONDA_ENV=其他环境名` 覆盖。输出位于 `output/original-a-evolve-tb20-qwen27b/`，
每次生成独立 job ID。按题指定或限制数量也可使用：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --task cancel-async-tasks
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --limit 3
```

## 本地模型与认证

三个角色均使用已确认的 Model ID：`/data0/liuxule/project/models/Qwen3.8-27B`。
采用原始 `harbor_agent:CuaHarnessClaudeCodeAgent` 和原版 MEA / Prompt / Auditor 路线，
不加载 a-evolve Solver / Evolution，也不启用 LLH Skill。

Endpoint 优先级为 `--base-url`、`LLH_LOCAL_BASE_URL`、兼容的 `ANTHROPIC_BASE_URL`、Docker bridge 宿主机网关的 8010 端口。
从旧 `ANTHROPIC_BASE_URL` 隐式继承公网地址会明确拒绝；若确实是部署在公网的自有模型，请显式用前两个选项指定。
这是任务容器可访问的宿主机地址。vLLM 需监听 `0.0.0.0:8010` 或可从 Docker 访问的宿主机地址。
若环境里遗留了旧的云端地址或 localhost 地址，请先清除，或显式设置正确的地址：

```bash
unset ANTHROPIC_BASE_URL
# 或 export LLH_LOCAL_BASE_URL="http://你的宿主机IP:8010"
```

这个 Backend 使用本机已经验证过的原生 Anthropic Messages API；Base URL 不填写 `/v1/messages`，
尾部 `/v1` 会按现有机制规范化。不使用 Chat → Messages 隐式转换。
本次服务没有开启鉴权，启动器默认使用 `none`，忽略继承的 Key；Claude Code 客户端使用 `local`
占位值，原有代理不向服务发送鉴权 Header。vLLM 不会自动生成 API Key，`local` 不是服务密钥。
若环境中曾设置 `LLH_LOCAL_AUTH_SCHEME`，请显式设为 `none` 或清除。
启动脚本也会从模型服务子进程中移除继承的 `VLLM_API_KEY`，保证该启动配置不启用鉴权。
如果以后明确开启了服务端认证，可选择 `--auth-scheme bearer` 或 `--auth-scheme x-api-key`，
通过 `--api-key-env MY_LOCAL_KEY` 指定凭据环境变量；未指定时按 `LLH_LOCAL_API_KEY`、
`ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN`、`OPENAI_API_KEY`、`VLLM_API_KEY` 顺序读取。
真实 Key 不写入 YAML、命令参数或 Manifest。代理仅按显式配置处理鉴权 Header，不转换 Messages 协议。
启动前会检查同一 Endpoint 的 Model ID、上下文长度、工具调用、SSE、Usage 和无效模型错误处理；
全量评估或显式选择 `code-from-image` / `video-processing` 时，预检还会实际发送图片，
纯文本服务会在启动 Benchmark 前报错，检查结果中的 `image_probe` 必须为 `verified`。
失败时不启动 Harbor，不更换模型，不回退云端。

正式评估启动前也会自动执行 Docker 内接口检查。运行中每 10 秒核对同一 Endpoint
仍提供指定模型；连续 3 次失败，或角色代理日志出现模型连接/API 错误时，停止剩余题目，
在 `.verification-status.json` 标记 `benchmark_valid=false`、`model_score=null`。
保留 Harbor 原始结果和日志，不把连接故障当成普通模型答错，也不更换模型或自动重试推理。

### 系统依赖和外部测试资源

`--prepare-verifier-runtime` 还按每题原始镜像准备系统 deb 包、procps，以及评分使用的
MobileSAM 权重、HTML sanitizer 测试压缩包、OCaml / pyknotid 上游测试仓库快照。
原始镜像未缓存时，只在准备阶段明确拉取。记录镜像 ID、包文件 SHA256、仓库 Commit 和
资源 SHA256；对于上游原本指向 master 的 URL，后续实验固定复用这次记录的字节。
旧 Debian 安全包若已从主站移至官方归档站，只在准备阶段从 `archive.debian.org` 补取，
必须与原 apt 已验证索引中的版本、大小和 SHA256 完全一致；不更换包版本，也不绕过摘要检查。
这套资源仅用于评分依赖，不包含 Agent 解答；任务描述、评分脚本、断言和奖励规则不变。
上游测试资源不挂载到 Agent 阶段；只有 Harbor 在解题结束后上传原始测试时才传入相应资源，
避免提前向模型暴露隐藏的测试用例。原始 Agent 的 `run` 方法保持继承不变。

评分安装阶段不更新远程 apt 索引、不下载 uv/Python/Python 包；已安装系统包直接核验，
缺少的包由 apt 从仅含本地文件的临时包源安装，并交给原生 dpkg 完成安装。
不配置远程包源；Agent 启动只安装 procps 及其依赖，不提前安装评分阶段才需要的包。
已知外部评分资源从只读缓存提供。
缓存缺失、变更或镜像 ID 不匹配会在评估前拒绝启动，评分中出现缓存问题则中止批次。
Claude Code 二进制缓存缺失时也会拒绝启动，不再临时在线安装。

可独立逐题在断网容器里验证依赖安装和 pytest 启动，不调用模型、不执行题目断言或计分：

```bash
PYTHONPATH=src conda run --no-capture-output -n terminal-bench-2-1 \
  python evaluation/validate_verifier_bootstrap.py
```

该检查还会从缓存实际克隆上游测试仓库、读取测试压缩包和权重，验证四项外部资源的
Commit / SHA256。它不包含解题正确率评估。

任务执行阶段由模型选择的任意新依赖仍可能需要外网；本地任务 HTTP/SSH 服务与功能测试的
网络行为保持原样。断网验证覆盖已声明的评分安装链路，并不能保证外部站点或本地服务今后永不故障。

### 服务未启动或 HTTP 401

连接被拒绝、超时或没有服务响应时，先执行上面的启动命令，等待服务就绪；
不要通过填写真实 API Key 来处理服务未启动的问题。先明确本地地址，再仅运行检查：

```bash
export LLH_LOCAL_BASE_URL="http://$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'):8010"
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --check-model-only
```

检查会显示 `Model endpoint` 和 `Authentication`，不会启动 Benchmark。
当前应显示 `Authentication: none`。如果无鉴权服务仍返回 401，应确认请求实际到达的 Endpoint
及其网关是否与预期一致，不能单凭 401 认定缺少 Key 或协议不兼容；
启动器不会尝试其他地址、模型或云端服务。

### 安装 Claude Code 时下载超时

如果任务结果为 `NetworkConnectionError`，且日志包含 `Failed to connect to downloads.claude.ai`，
说明容器尚未安装好执行客户端，任务还没有进入解题或评分阶段。此类结果不能作为模型能力评分。
先停止正在重复安装失败的旧任务，再使用本机缓存的同一版本 Claude Code。

启动器现在优先查找 `.cache/claude-code/2.1.176/claude`，然后查找已有实验缓存，
校验实际版本，并以只读方式挂载到任务容器。独立缓存不在 `output/` 内，清理实验输出不会删除它；
也可显式指定文件：

```bash
export LH_CLAUDE_OFFLINE_BINARY=/data0/liuxule/project/LHH/LongHorizon-Harness/.cache/claude-code/2.1.176/claude
```

显式文件缺失或版本不符会直接报错，不重新下载其他版本。
如果当前 shell 的 `LH_CLAUDE_OFFLINE_BINARY` 仍指向已清理的 `output/his/...`，
执行上面的 export 更新路径，或 `unset LH_CLAUDE_OFFLINE_BINARY` 让启动器自动发现独立缓存。
适配层复用 `lh_harness.original_harness_eval:OriginalHarnessHarborAgent`，只在 setup 阶段
补齐 `procps` 并调用原有安装流程；原有安装器识别到正确版本后跳过 Claude Code 下载。
原始 `run()`、角色 Prompt、MEA 和评分流程不变。安装 `procps` 或任务自身依赖仍可能需要软件源网络，
此选项不表示整个 Benchmark 都不需要外网。

先只验证第一道失败题的环境与客户端安装，不调用解题流程或 Verifier：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh \
  --task portfolio-optimization --install-only \
  --jobs-dir output/original-a-evolve-tb20-qwen27b-setup-check
```

检查成功后，去掉 `--install-only` 单题试跑，最后再启动全部 89 题：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh --task portfolio-optimization
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh
```

新任务使用独立 job ID，保留旧的异常记录。每次任务另存实际运行配置和安装记录，
包括客户端版本、缓存路径、SHA256 及安装适配范围；不修改原始配置和数据集。

### Verifier 下载 uv 失败、uvx 不存在或评分缓存未就绪

这套题的评分依赖按 39 组准备，放在 `.cache/verifier-runtime-095/`，独立于 `output/`。
启动器默认要求所选题目的缓存齐备、版本和 SHA256 校验通过；缺少缓存时直接拒绝启动模型任务。
第一次运行上面的 `--prepare-verifier-runtime`，以后可直接使用原评估命令。
准备进度逐组保存；中断后再次执行会复用通过校验的缓存。全量未完成时，已经准备好的单题
可以用 `--task 题名 --check-verifier-runtime` 单独检查。

准备阶段默认直连软件源，避免继承失效的 SSH 代理隧道。确实需要代理时，可显式设置
`LLH_VERIFIER_PREPARE_USE_PROXY=1` 使用已有代理环境变量；模型 Endpoint 配置不受影响。

评分时只读挂载缓存，提供准确的 uv/uvx 0.9.5，并在每个评分容器内创建私有可写副本供 uv 使用。
uv/Python 包的安装使用离线缓存，缓存缺失不回退到在线下载。原脚本中的版本约束和 pytest
命令保持不变；任务镜像已有匹配 Python 时继续使用其解释器，缺少时使用预备的匹配版本。
实际解释器记录在 `verifier/bootstrap.json` 中。Agent 不使用评分阶段的 PATH 或环境设置，
无法修改共享依赖缓存；测试和答案不放入依赖缓存。

若评分基础设施仍异常，启动器终止本批次并写入 `<job>.verification-status.json`，
将该批次标为 `benchmark_valid=false`、`model_score=null`。Harbor 原始分数和日志保持原样；
真正的测试断言失败仍按原标准得到 0 分。

可在完全断网的独立容器中检查原始测试是否正常运行：

```bash
PYTHONPATH=src conda run --no-capture-output -n terminal-bench-2-1 \
  python evaluation/validate_verifier_offline.py
```

该命令使用未解题骨架，预期测试失败并得到真实 0 分；这是依赖/评分验证，不是模型成绩。
`--oracle` 可使用数据集提供的答案做另一组独立验证，同样不计入模型成绩。
任务自身的 apt/git/外部资产或应用联网测试保持原样；本适配层消除的是 uv、解释器及所声明
Python 评分依赖的临时下载，不改写任务的联网功能。

### Claude Code 请求返回 reasoning effort high / HTTP 500

如果日志为 `Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.`，
表示请求已经到达本地模型，但 Claude Code 的自适应 thinking 参数与 Qwen 模板不兼容。
本地服务启动时关闭 thinking，不足以保证所有客户端请求都采用相同策略。

当前 Qwen 配置通过原始 LLH 已有的 `llm_call_kwargs.extra_body` 显式设置：

```yaml
thinking:
  type: disabled
chat_template_kwargs:
  enable_thinking: false
```

原有请求代理将这份配置用于 Manager、Executor、Auditor 的每次请求；不修改角色 Prompt、
工具定义或评分流程，也不转换 API 协议。启动前的 Messages 预检采用相同的 thinking 设置，
实际参数写入 Manifest。旧进程已经读取的配置不会随文件修改而更新，需要停止后启动新 job。

可单独运行真实角色客户端检查。它创建独立容器，依次让三个原有角色通过 Read/Bash 读取随机内容，
验证工具调用、流式日志、Usage 和 Auditor 工作区未被修改；不执行或评分 Benchmark 题目：

```bash
PYTHONPATH="$PWD/src:$PWD/eval/TB-harness/Harness/src${PYTHONPATH:+:$PYTHONPATH}" \
  conda run --no-capture-output -n terminal-bench-2-1 \
  python -m lh_harness.original_agent_probe --base-url "$LLH_LOCAL_BASE_URL" --with-image
```

检查产物位于 `output/original-qwen-role-check/<独立ID>/role_probe.json`。成功后先单题重跑，
再开始 89 题评估；旧的参数错误记录保留，不作为模型能力结果。

### 调用成功但反复审计、最终 AgentTimeoutError

Claude Code 的原生流式事件可能用外层 `type: assistant` 标识角色，而不在 `message` 内重复
提供 `role`。原始 LLH 的审计解析器按 `message.role` 提取报告；如果字段缺失，即使模型已经
输出正确的三行控制头，也可能把整段 JSONL 当作正文，触发无效报告、格式修复和重复轮次。

独立 `claude_stream_compat` 层现在仅根据明确的外层 `type: assistant` 补齐缺失的
`message.role: assistant`；不根据文本猜测角色、不改变控制头和审计结论，不把工具输出当报告。
原始执行方法和 Verifier 保持不变，完整原始轨迹仍保存在各 episode 的 `agent.log` 与
`claude_stream.jsonl`。新的真实角色检查还会将 Auditor 结果交给原始 Verifier 解析，
验证其状态、完整性与契约审计结果，覆盖此前仅验证工具调用留下的缺口。

### 图像读取请求返回 At most 0 image(s)

`--language-model-only` 服务不能接收 Claude Code Read 工具产生的图像内容，图像/视频题可能
因此反复返回 HTTP 500。不能靠延长超时解决，也不能删除图像后将其标作正常评估。

现已提供显式图像服务选项，保留同一模型、四卡和其余推理配方：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash evaluation/start_qwen27b_vllm.sh \
  --enable-vision --max-images 32 \
  --gpu-memory-utilization 0.85 --max-model-len 1000000 \
  --max-num-seqs 8 --max-num-batched-tokens 8192
```

先停止评估和现有 8010 服务，再运行此命令；不要在同一个实验 job 内混用服务模式。
此选项启用同一权重的视觉编码器，设置每次请求最多 32 张图片；它不启用直接视频输入 API。
必须在服务实际就绪后验证图像请求、工具调用与 Usage，再用新 job 单题验证。
Original 与 Skill 对照应连接同一个服务，并记录相同的模态限制。
此次实际服务的版本、GPU、模态限制和采样参数快照保存在
`evaluation/validation/qwen27b-vision-serving.json`。可在新运行时显式附加：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh \
  --serving-manifest evaluation/validation/qwen27b-vision-serving.json
```

启动器会核对快照的模型、Endpoint、Context Length、服务版本和采样配置，不一致则拒绝启动。
快照是已观察到的运行配置；更换服务、GPU 或模态设置后应重新采集，不能沿用旧快照冒充实测。

### 视频任务返回 At most 4 image(s)

`video-processing` 的图片会随工具历史累积，单次读取少量图片仍可能使后续请求超过
原来的 4 图上限。2026-11-22 起启动器的视觉模式默认上限为 32 张；上述服务命令已同步。
Original 和 Skill-enhanced 的本地模型预检共用同一实现，图像任务和全量运行会实际发送
32 张小图片，检查工具调用、SSE、真实 Usage 和无效模型拒绝，不能再用单图成功替代多图验收。
尚未调整的 4 图服务会在启动 Benchmark 前失败；旧的 4 图 serving 快照也会被拒绝。

修复记录：`evaluation/validation/qwen27b-images32-host-check.json` 和
`evaluation/validation/qwen27b-images32-docker-check.log`。
图片不会被丢弃、裁剪历史或改写来绕过限制。32 张仍是有限上限；若新任务需要更多图片，
应重新评估服务资源、更新两轨共同配置并重新验证。不要在同一个 job 内混用服务配置。

模型参数为输出上限 8192、Temperature 0.6、Top-p 0.95、Top-k 20、最多 25 个 MEA 轮次。
Reasoning 在服务启动和原始 LLH 角色请求中均显式关闭，与现有本地模型实验配置保持一致。

## 适配保证与边界

- 原始说明从 `variants.default.prompt` 原样写入 `instruction.md`，不添加解题提示。
- `files` 中的 `/tests/...` 按原映射复制；测试和评分脚本逐字节保持一致。
  它们不进入镜像构建，也不在 Agent 阶段上传，仍由 Harbor Verifier 在评分时上传。
- 使用原 Compose 的镜像、平台、启动命令、init、CPU 和内存设置，仅将服务名从 `default` 映射成 Harbor 的 `main`。
  容器 WORKDIR 继承原镜像，原始 LLH 自动发现工作目录。
- 原有 Agent / Verifier 超时从每题 metadata 映射到 `task.toml`，配置不覆盖这些超时。
  25 轮是轮数上限，各题仍受自身总时间预算约束。
- 原始 `tests/test.sh` 和 reward 文件评分规则不变；不重写依赖安装器或评分逻辑。
  因而运行时仍需原测试脚本所需的软件包/网络条件；准备成功不等于正式评测成功。
- 生成目录包含 `dataset_manifest.json`：记录版本标签、全部题名、文件映射、源和目标 SHA256、资源与超时。
  每次启动重新校验；源数据或生成的测试文件发生变化即拒绝复用，不静默覆盖。
- 开始正式评估时，在输出目录保存单独的 `<job-id>.manifest.json`，记录模型探测、采样参数、
  数据集摘要、原始 LLH 源码摘要和实际选择的任务。未核实的模型运行元数据不编造。

只做数据准备时，不需要 Docker、模型服务或 API Key。切换数据缓存或生成目录可传入：

```bash
bash evaluation/run_original_harness_a_evolve_tb20_qwen27b.sh \
  --source /其他位置/challenges \
  --dataset-output /独立位置/a-evolve-tb20-harbor \
  --prepare-only
```

以上是独立的数据/实验适配层，没有修改 a-evolve 的缓存，也没有修改原始 LLH 的执行内核和 Verifier。
