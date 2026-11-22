"""Sequential task batches, followed by frozen-library training-set recheck.

Frozen Original MEA and graders are loaded through the experiment adapter.
No previous run index is read and no held-out evaluation split is overwritten.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Shared launch helpers prepare their own source search path. Keep that
# preparation local to import: only Harbor subprocesses should select the
# official frozen backend through their explicit PYTHONPATH below.
_entry_path = list(sys.path)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_paired_eval as shared
    from train_skills import GROUPS, ROLES
finally:
    sys.path[:] = _entry_path
from lh_harness.local_models import EndpointClient, base_url, is_local
from lh_harness.skills.experience import load_split
from lh_harness.skills.library import SkillLibrary, atomic_json
from lh_harness.skills.schema import Settings
from lh_harness.skills.statistics import summarize
from lh_harness.token_usage import aggregate, from_trajectory, unavailable


def select_tasks(blocks, names, option):
    requested = [name for block in blocks for item in block for name in item.split(",")]
    if "all" in requested:
        if requested != ["all"]:
            raise ValueError(f"{option} all 不能与其他任务 ID 混用")
        return list(names)
    if len(set(requested)) != len(requested) or not requested or set(requested) - set(names):
        raise ValueError(f"{option} 任务 ID 为空、重复或不在数据集中：{sorted(set(requested) - set(names))}")
    return requested


def tasks_from_run(directory, names):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"结果目录不存在：{root}")
    rows = []
    for trial in root.iterdir():
        result = trial / "result.json"
        if not trial.is_dir() or (not result.exists() and "__" not in trial.name):
            continue
        data = json.loads(result.read_text()) if result.exists() else {}
        if not isinstance(data, dict):
            raise ValueError(f"任务结果不是 JSON 对象：{result}")
        prefix = trial.name.rsplit("__", 1)[0]
        task = data.get("task_name") or prefix
        if not isinstance(task, str) or task not in names:
            raise ValueError(f"结果目录含有不在当前数据集中的任务：{task}")
        if "__" in trial.name and prefix != task:
            raise ValueError(f"任务目录名与 result.json 的任务 ID 不一致：{trial}")
        stamp = data.get("started_at")
        if stamp:
            if not isinstance(stamp, str):
                raise ValueError(f"任务开始时间格式不合法：{result}")
            timestamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            order = timestamp.replace(tzinfo=timezone.utc).timestamp() if timestamp.tzinfo is None else timestamp.timestamp()
        else:
            order = trial.stat().st_mtime
        rows.append((order, trial.name, task, shared.native.digest(result) if result.exists() else None))
    if not rows:
        raise ValueError("结果目录中没有可识别的任务；请指定单个 Harbor job 目录")
    rows.sort()
    tasks = list(dict.fromkeys(row[2] for row in rows))
    snapshot = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
    return tasks, {"kind": "harbor_run_directory", "path": str(root), "sha256": hashlib.sha256(snapshot).hexdigest(),
                   "unique_tasks": len(tasks), "scope": "task IDs only; all selected tasks will be executed afresh"}


def plan(args):
    selected_roles = list(ROLES) if args.roles == "all" else ([] if args.roles == "none" else args.roles.split(","))
    if len(selected_roles) != len(set(selected_roles)) or any(role not in ROLES for role in selected_roles):
        raise ValueError("角色只能是 manager、executor、auditor（逗号连接），或 all / none")
    roles = tuple(role for role in ROLES if role in selected_roles)
    if getattr(args, "resume", None):
        return resume_plan(args, roles)
    if getattr(args, "from_task", None) is not None:
        raise ValueError("--from-task 必须与 --resume 一起使用")
    batch_size = getattr(args, "batch_size", None)
    batch_size = 5 if batch_size is None else batch_size
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("--batch-size 必须为正整数")
    if args.freeze and not roles:
        raise ValueError("none 不创建 Skill 库，不需要 --freeze")
    group = GROUPS[roles]
    dataset = ROOT / shared.native.DEFAULT_OUTPUT
    inventory = json.loads((dataset / shared.native.MANIFEST).read_text())
    names = [row["name"] for row in inventory["tasks"]]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names):
        raise ValueError("数据集任务 ID 不合法或重复")
    task_file = getattr(args, "task_file", None)
    result_directory = getattr(args, "tasks_from_run", None)
    if sum(bool(value) for value in (args.task, task_file, result_directory)) > 1:
        raise ValueError("--task、--task-file、--tasks-from-run 请选择一种")
    task_source = None
    if result_directory:
        tasks, task_source = tasks_from_run(result_directory, names)
    elif task_file:
        path = Path(task_file).expanduser().resolve()
        source_bytes = path.read_bytes()
        contents = source_bytes.decode("utf-8-sig")
        blocks = [[item for line in contents.splitlines() for item in line.split("#", 1)[0].split()]]
        tasks = select_tasks(blocks, names, "--task-file")
        task_source = {"path": str(path), "sha256": hashlib.sha256(source_bytes).hexdigest()}
    else:
        tasks = select_tasks(args.task or [["all"]], names, "--task")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit 必须为正整数")
        tasks = tasks[:args.limit]
    eval_requested = getattr(args, "eval_task", None)
    evaluation_tasks = select_tasks(eval_requested, names, "--eval-task") if eval_requested is not None else list(tasks)
    source = ROOT / "experiments/skill_evolution/configs" / f"{group}.toml"
    text = source.read_text()
    raw = tomllib.loads(text)
    endpoint = base_url(args.base_url or os.environ.get("LLH_LOCAL_BASE_URL") or raw["run"]["base_url"])
    if not is_local(endpoint):
        raise ValueError("在线入口仅连接显式本地模型，不切换云端服务")
    # The same address must be reachable from task containers.
    shared.native.model_url({"LLH_LOCAL_BASE_URL": endpoint})
    if args.api_key_env and (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env)
                             or not os.environ.get(args.api_key_env)):
        raise ValueError("--api-key-env 必须是已设置的环境变量名称，不能是密钥值")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    track = "skill_enhanced" if roles else "baseline"
    output = (args.output or ROOT / "experiments/skill_evolution/results" / track /
              f"online-{group}-{stamp}-{uuid.uuid4().hex[:8]}").expanduser().resolve()
    if output.exists():
        raise ValueError(f"输出目录已存在，不覆盖旧结果：{output}")
    for field, value in (("library_dir", str(output / "library")), ("base_url", endpoint), ("phase", "online")):
        text, count = re.subn(rf"^{field}\s*=.*$", lambda _: field + " = " + json.dumps(value), text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"配置中的 {field} 不是唯一字段，拒绝修改")
    text, count = re.subn(r"^batch_size\s*=.*$", f"batch_size = {batch_size}", text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("配置中缺少唯一的训练 batch_size")
    config = tomllib.loads(text)
    if config["run"]["model"] != raw["run"]["model"] or config.get("local_model") != raw.get("local_model"):
        raise ValueError("在线配置意外修改了模型或推理参数")
    formal = load_split(ROOT / "experiments/skill_evolution/manifests/pilot-splits.json")
    # This separate online split makes observed tasks learning tasks only for
    # this experiment, while leaving the formal held-out split intact.
    split = {"dataset": formal["dataset"], "train": list(tasks), "validation": [], "test": []}
    settings = Settings.from_dict(config["skills"])
    return {"roles": roles, "group": group, "track": track, "tasks": tasks, "dataset": dataset, "batch_size": batch_size,
            "final_eval": not getattr(args, "skip_final_eval", False), "evaluation_tasks": evaluation_tasks,
            "task_selection_source": task_source,
            "benchmark": inventory.get("dataset_label"), "dataset_task_count": len(names),
            "output": output, "endpoint": endpoint, "config_text": text, "config": config,
            "settings": settings, "online_split": split, "formal_split": formal,
            "inventory_sha256": shared.native.digest(dataset / shared.native.MANIFEST)}


def resume_plan(args, roles):
    """Read-only continuation plan; inherit task order and inference settings."""
    output = Path(args.resume).expanduser().resolve()
    if args.output and Path(args.output).expanduser().resolve() != output:
        raise ValueError("--resume 与 --output 必须指向同一个原目录")
    if any(getattr(args, field, None) is not None for field in ('task', 'task_file', 'tasks_from_run', 'limit', 'eval_task')):
        raise ValueError("续跑沿用原任务清单，不能同时重新选题")
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest.get('experiment') != 'online_skill_learning' or manifest.get('status') not in {
        'interrupted', 'failed', 'infrastructure_failure', 'completed_with_learning_errors', 'completed'}:
        raise ValueError("原运行未停止或不是在线训练目录，拒绝续跑")
    if set(manifest.get('train_roles', [])) != set(roles):
        raise ValueError("续跑训练角色必须与原运行一致")
    if (output/'library/frozen.json').exists() or (output/'evaluation').exists():
        raise ValueError("原库已冻结或已有最终复测，不能继续修改该训练库")
    batch_size = manifest['batch_size']
    if getattr(args, 'batch_size', None) not in (None, batch_size):
        raise ValueError("续跑必须沿用原 Batch 大小")
    tasks = manifest['task_order']
    if len(set(tasks)) != len(tasks) or not tasks:
        raise ValueError("原任务顺序不合法")
    start = getattr(args, 'from_task', None)
    if start is None:
        raise ValueError("请用 --from-task 指定从第几题（从 1 开始，含该题）或任务 ID 续跑")
    start = int(start) if str(start).isdigit() else tasks.index(start)+1 if start in tasks else 0
    if not 1 <= start <= len(tasks):
        raise ValueError("--from-task 不在原任务清单范围内")
    dataset = ROOT/shared.native.DEFAULT_OUTPUT
    inventory_path = dataset/shared.native.MANIFEST
    if shared.native.digest(inventory_path) != manifest['dataset_sha256']:
        raise ValueError("数据集版本与原运行不同，拒绝续跑")
    names = {row['name'] for row in json.loads(inventory_path.read_text())['tasks']}
    if set(tasks) - names:
        raise ValueError("原任务不在当前数据集中")
    previous = {row['index']: row for row in manifest['tasks']}
    for index in range(1, start):
        row = previous.get(index, {})
        if row.get('status') != 'completed' or row.get('task_id') != tasks[index-1] or not row.get('trial'):
            raise ValueError("续跑起点之前存在未完成任务，不能跳过")
        trial = Path(row['trial'])
        result = json.loads((trial/'result.json').read_text())
        if (shared.native.digest(trial/'result.json') != row.get('raw_result_sha256') or
            ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward') != row.get('reward')):
            raise ValueError("保留任务的原始评分摘要不一致")
    text = (output/'config.toml').read_text()
    raw = tomllib.loads(text)
    endpoint = base_url(raw['run']['base_url'])
    if not is_local(endpoint) or (args.base_url and base_url(args.base_url) != endpoint):
        raise ValueError("续跑必须使用原本地模型 Endpoint")
    if args.api_key_env and (not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.api_key_env) or not os.environ.get(args.api_key_env)):
        raise ValueError("--api-key-env 必须是已设置的环境变量名称，不能是密钥值")
    if Path(raw['skills']['library_dir']).expanduser().resolve() != output/'library':
        raise ValueError("原 Skill 库不在该运行目录内，拒绝隐式切换")
    # The only configuration migration already requested by the user is the
    # removal of the old two-task activation gate. Save the old file on apply.
    text, count = re.subn(r'^min_independent_tasks\s*=.*$', 'min_independent_tasks = 1', text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("原激活策略配置不是唯一字段")
    config = tomllib.loads(text)
    settings = Settings.from_dict(config['skills'])
    if settings.phase != 'online' or not settings.training['enabled'] or set(settings.training['train_roles']) != set(roles):
        raise ValueError("续跑只支持原有在线训练配置")
    library = SkillLibrary(output/'library')
    library.index()
    formal = load_split(ROOT/'experiments/skill_evolution/manifests/pilot-splits.json')
    if formal['sha256'] != manifest['formal_split_sha256']:
        raise ValueError("正式数据分割已变化，拒绝续跑")
    split = json.loads((output/'online-split.json').read_text())
    if split != manifest['online_split'] or split['train'] != tasks:
        raise ValueError("原在线学习清单与 Manifest 不一致")
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return {'roles': roles, 'group': manifest['group'], 'track': manifest['track'], 'tasks': tasks,
            'dataset': dataset, 'batch_size': batch_size, 'final_eval': manifest['final_evaluation_enabled'] and not args.skip_final_eval,
            'evaluation_tasks': manifest['evaluation_task_order'], 'task_selection_source': manifest.get('task_selection_source'),
            'benchmark': manifest['benchmark'], 'dataset_task_count': manifest['dataset_task_count'],
            'output': output, 'endpoint': endpoint, 'config_text': text, 'config': config, 'settings': settings,
            'online_split': split, 'formal_split': formal, 'inventory_sha256': manifest['dataset_sha256'],
            'resume': True, 'start_index': start, 'resume_id': stamp+'-'+uuid.uuid4().hex[:8],
            'saved_manifest': manifest, 'library_before_resume': library.digest(),
            'resume_file_hashes': {name: shared.native.digest(output/name) for name in ('manifest.json','summary.json','config.toml','online-split.json')}}


def validate_resume_infrastructure(plan, observed):
    before = plan['saved_manifest'].get('infrastructure') or {}
    for key in ('original_commit','original_sources','weights_files_sha256'):
        if not before.get(key) or before[key] != observed.get(key):
            raise ValueError(f"续跑的代码或模型权重与原运行不一致：{key}")
    for key in ('serving_version','quantization','dtype','gpu_hardware','context_length','max_output_tokens',
                'temperature','top_p','top_k','reasoning','tool_calling','max_num_seqs','benchmark_concurrency'):
        if before['serving']['observed'].get(key) != observed['serving']['observed'].get(key):
            raise ValueError(f"续跑推理服务配置与原运行不一致：{key}")


def prepare_resume(plan, infrastructure):
    """Snapshot metadata and library before appending new attempts in place."""
    output = plan['output']
    for name, digest in plan['resume_file_hashes'].items():
        if shared.native.digest(output/name) != digest:
            raise ValueError("续跑计划生成后原运行文件发生变化")
    library = SkillLibrary(output/'library')
    if library.digest() != plan['library_before_resume']:
        raise ValueError("续跑计划生成后 Skill 库发生变化")
    archive = output/'resumes'/plan['resume_id']
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('manifest.json','summary.json','config.toml','online-split.json'):
        shutil.copy2(output/name, archive/name)
    shutil.copytree(library.root, archive/'library-before')
    manifest = copy.deepcopy(plan['saved_manifest'])
    superseded = [row for row in manifest['tasks'] if row['index'] >= plan['start_index']]
    for row in superseded:
        row['superseded_by_resume'] = plan['resume_id']
        if not row.get('runtime_usage'):
            trials = list((output/'tasks'/f"{row['index']:03d}-{row['task_id']}").glob(f"*/{row['task_id']}__*/result.json"))
            if len(trials) == 1:
                trial = trials[0].parent
                usage = trial/'agent/skill_evolution/skill_usage.json'
                row.update(trial=str(trial), runtime_usage=runtime_usage(trial),
                           skill_usage=json.loads(usage.read_text()) if usage.exists() else None)
    manifest.setdefault('superseded_tasks', []).extend(superseded)
    manifest['tasks'] = [row for row in manifest['tasks'] if row['index'] < plan['start_index']]
    for batch in manifest.get('batches', []):
        if batch.get('status') == 'running':
            batch.update(status='interrupted', returncode=130)
    event = {'resume_id': plan['resume_id'], 'start_index': plan['start_index'],
             'start_task': plan['tasks'][plan['start_index']-1], 'archive': str(archive),
             'retained_library_sha256': library.digest(), 'retained_completed_tasks': len(manifest['tasks']),
             'superseded_task_attempts': len(superseded), 'infrastructure': infrastructure,
             'batching': 'remaining tasks regrouped from resume point using original batch size',
             'activation_min_independent_tasks': 1}
    manifest.setdefault('resumptions', []).append(event)
    manifest.update(status='preparing', infrastructure=infrastructure, final_evaluation_enabled=plan['final_eval'])
    (output/'config.toml').write_text(plan['config_text'])
    atomic_json(archive/'resume-plan.json', event)
    return manifest


def library_state(library):
    return {"sha256": library.digest(), "revision": library.index()["revision"],
            "active_skills": len(library.list()), "candidate_skills": sum(s.status == "candidate" for s in library.list(candidates=True))}


def provenance(plan, observed, status):
    return {"experiment": "online_skill_learning", "phase": "online", "dataset": plan["online_split"]["dataset"],
            "observed_task_ids": list(observed), "task_order": list(plan["tasks"]), "status": status,
            "formal_split_sha256": plan["formal_split"]["sha256"], "held_out_evaluation": False}


def save_attempted_provenance(plan, manifest):
    library = plan["output"] / "library"
    if plan["roles"] and (library / "index.json").exists() and not (library / "frozen.json").exists():
        prior = library/'online_provenance.json'
        observed = json.loads(prior.read_text()).get('observed_task_ids', []) if prior.exists() else []
        observed = list(dict.fromkeys([*observed, *[row['task_id'] for row in manifest.get('superseded_tasks', [])],
                                      *[row['task_id'] for row in manifest['tasks']]]))
        atomic_json(library / "online_provenance.json",
                    provenance(plan, observed, manifest["status"]))


def score_hashes(trial):
    files = [trial / "result.json", *sorted((trial / "verifier").rglob("*"))]
    return {str(path.relative_to(trial)): shared.native.digest(path) for path in files if path.is_file()}


def runtime_usage(trial):
    report = trial / "agent/skill_evolution/token_usage.json"
    if report.is_file():
        return json.loads(report.read_text())
    # Original control has no SkillSession; read each native episode exactly
    # once, including unavailable usage for interrupted episode directories.
    records = []
    for role, native_role in (("manager", "orchestrator"), ("executor", "cli_task"), ("auditor", "cli_verifier")):
        for episode in sorted(trial.glob(f"agent/**/episodes/{native_role}/ep*")):
            stream = episode / "claude_stream.jsonl"
            if not stream.exists():
                stream = episode / "agent.log"
            records.append({"role": role, "usage": from_trajectory(stream.read_text(errors="replace")) if stream.exists() else unavailable()})
    value = summarize(records)
    if not records:
        value["runtime"] = unavailable()
        value["categories"] = {role: unavailable() for role in ROLES}
    return value


class HarborRuntime:
    def __init__(self, plan, args):
        self.plan, self.args = plan, args
        self.stage = "training"

    def prepare(self):
        p, args = self.plan, self.args
        print("校验原始代码、数据集和离线评分依赖……", flush=True)
        source_record = shared.protected_sources(ROOT)
        shared.native.prepare_dataset(shared.native.DEFAULT_SOURCE, p["dataset"], expected_tasks=89)
        config = shared.native.resolved_config(shared.native.DEFAULT_CONFIG, p["dataset"])
        run, kwargs = p["config"]["run"], config["agents"][0]["kwargs"]
        if run.get("agent") != "claude_code" or run.get("model") != config["agents"][0]["model_name"] or run.get("roles"):
            raise ValueError("在线 TB 路线必须沿用原始全局 Backend / Model，不能隐式换模型")
        if run.get("max_rounds", kwargs["max_rounds"]) != kwargs["max_rounds"]:
            raise ValueError("在线 MEA 轮数与原始配置不一致")
        for key, actual in (("temperature", kwargs["temperature"]), ("top_p", kwargs["top_p"]), ("max_output_tokens", kwargs["max_tokens"])):
            if p["config"]["local_model"].get(key) != actual:
                raise ValueError(f"在线模型推理参数与原始配置不一致：{key}")
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LLH_LOCAL_BASE_URL=p["endpoint"])
        self.env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments/skill_evolution/adapters"),
            str(shared.ORIGINAL / "eval/TB-harness/Harness/src")])
        self.env["ANTHROPIC_BASE_URL"] = shared.native.model_url({"LLH_LOCAL_BASE_URL": p["endpoint"]})
        self.env["NO_PROXY"] = self.env["no_proxy"] = ",".join(filter(None, [self.env.get("NO_PROXY"), "127.0.0.1", "localhost", shared.native.urlsplit(p["endpoint"]).hostname]))
        self.key, _ = shared.native.configure_auth(self.env, config, args.auth_scheme, args.api_key_env)
        installation = shared.native.configure_installation(config, self.env)
        required_tasks = list(dict.fromkeys(p["tasks"] + (p["evaluation_tasks"] if p["final_eval"] else [])))
        runtime = shared.verifier_runtime.configure(config, self.env, p["dataset"], required_tasks)
        print("检查本地模型、多图、工具协议和 Docker 连通性……", flush=True)
        profile = shared.native.endpoint_profile(config, require_image=bool(set(required_tasks) & {"code-from-image", "video-processing"}))
        endpoint = shared.check_endpoint(p["endpoint"], "claude_code", run["model"], profile, api_key=self.key, auth_scheme=args.auth_scheme)
        if p["config"]["local_model"]["context_length"] != endpoint["context_length"]:
            raise ValueError("模型实际 Context Length 与实验配置不一致")
        image = tomllib.loads((p["dataset"] / p["tasks"][0] / "task.toml").read_text())["environment"]["docker_image"]
        docker = shared.check_docker(p["endpoint"], run["model"], profile, image, shared.verifier_runtime.DEFAULT_ROOT,
                                     api_key=self.key, auth_scheme=args.auth_scheme)
        serving = shared.native.serving_evidence(args.serving_manifest, endpoint, kwargs)
        print("记录本地模型权重 SHA256……", flush=True)
        weights = {path.name: shared.native.digest(path) for path in Path(run["model"]).glob("*.safetensors")}
        if not weights:
            raise ValueError("无法读取实际模型权重，拒绝编造模型版本")
        self.client = EndpointClient(p["endpoint"], "claude_code", api_key=self.key, auth_scheme=args.auth_scheme)
        self.expected_context = endpoint["context_length"]
        config["agents"][0]["import_path"] = "harbor_skill_agent:SkillEnhancedAgent" if p["roles"] else "harbor_skill_agent:PristineOriginalAgent"
        if p["roles"]:
            config["agents"][0]["kwargs"]["skills_config"] = asdict(p["settings"])
        self.config = config
        original_sources = {str(path.relative_to(shared.ORIGINAL)): shared.native.digest(path)
                            for path in (shared.ORIGINAL / "eval/TB-harness/Harness/src").rglob("*.py")}
        return {"original_commit": source_record["original_commit"], "source_protection": source_record, "original_sources": original_sources,
                "endpoint": endpoint, "docker_endpoint_check": docker, "serving": serving,
                "weights_files_sha256": weights, "verifier_runtime": runtime, "installation": installation}

    def run_task(self, task, index, folder):
        shared.protected_sources(ROOT)
        inventory = self.client.json("/models")
        loaded = next((model for model in inventory.get("data", []) if model.get("id") == self.plan["config"]["run"]["model"]), None)
        if not loaded or loaded.get("max_model_len") != self.expected_context:
            raise ValueError("模型服务不可用、Model ID 或上下文已变化；不启动下一题，不回退")
        folder.mkdir(parents=True, exist_ok=False)
        cfg = folder / "config.yaml"
        shared.native.write_runtime_config(copy.deepcopy(self.config), cfg)
        job_name = ("recheck-" if self.stage == "training_set_recheck" else "online-") + uuid.uuid4().hex[:8]
        job = folder / job_name
        command = [str(Path(sys.executable).with_name("harbor")), "run", "-c", str(cfg), "--path", str(self.plan["dataset"]),
                   "--jobs-dir", str(folder), "--job-name", job_name, "--include-task-name", task,
                   "--allow-agent-host", shared.native.urlsplit(self.plan["endpoint"]).hostname]
        monitor = shared.Monitor(self.plan["endpoint"], self.plan["config"]["run"]["model"], api_key=self.key, auth_scheme=self.args.auth_scheme)
        code = shared.verifier_runtime.run_harbor(command, self.env, ROOT, job, model_monitor=monitor)
        results = list(job.glob("*/result.json"))
        return {"returncode": code, "job": str(job), "trial": str(results[0].parent) if len(results) == 1 else None,
                "config_sha256": shared.native.digest(cfg)}

    def _learn(self, rows, output, stage, inputs=()):
        index = output / "run-index.json"
        if stage == "propose":
            atomic_json(index, {row["task_id"]: row["trial"] for row in rows})
        command = [sys.executable, "-m", "lh_harness", "skills", "learn", "--config", str(self.plan["output"] / "config.toml"),
                   "--roles", ",".join(self.plan["roles"]), "--phase", "online", "--split-manifest", str(self.plan["output"] / "online-split.json"),
                   "--stage", stage, "--output", str(output), "--auth-scheme", self.args.auth_scheme, "--flush"]
        if stage == "propose":
            command += ["--run-index", str(index)]
        else:
            for path in inputs:
                command += ["--proposal-input", str(path)]
        if self.args.api_key_env:
            command += ["--api-key-env", self.args.api_key_env]
        code = subprocess.run(command, cwd=ROOT, env=self.env, check=False).returncode
        report, cost = output / "learning_report.json", output / "learning_cost.json"
        value = {"attempted": True, "returncode": code, "stage": stage, "report_path": str(report),
                "report": json.loads(report.read_text()) if report.exists() else None,
                "cost": json.loads(cost.read_text()) if cost.exists() else None}
        proposal = output / "proposals.json"
        if stage == "propose" and proposal.exists():
            value.update(proposal_path=str(proposal), proposal_sha256=shared.native.digest(proposal))
        return value

    def reflect_task(self, task, trial, output):
        return self._learn([{"task_id": task, "trial": str(trial)}], output, "propose")

    def learn_batch(self, rows, output):
        inputs, missing = [], []
        for row in rows:
            reflection = row.get("reflection", {})
            path = Path(reflection["proposal_path"]) if reflection.get("proposal_path") else None
            if path and path.is_file():
                if shared.native.digest(path) != reflection["proposal_sha256"]:
                    raise ValueError("逐题 Proposal 暂存产物发生变化")
                inputs.append(path)
            else:
                missing.append(row["task_id"])
        if inputs:
            result = self._learn(rows, output, "curate", inputs)
        else:
            result = {"attempted": True, "stage": "curate", "returncode": 1,
                      "cost": summarize([]), "report": {"status": "no_proposal_artifacts"}}
        if missing:
            result.update(returncode=result["returncode"] or 1, missing_proposal_tasks=missing)
        return result

    def learn_task(self, task, trial, output):
        # Retained as a task-local entry: curation is exclusively batch-level.
        return self.reflect_task(task, trial, output)

    def evaluation_runtime(self, settings):
        runtime = copy.copy(self)
        runtime.config = copy.deepcopy(self.config)
        runtime.stage = "training_set_recheck"
        if self.plan["roles"]:
            runtime.config["agents"][0]["kwargs"]["skills_config"] = asdict(settings)
        return runtime


def combined(values):
    return aggregate([value for value in values if value.get("status") != "not_applicable"])


def write_summary(plan, manifest):
    rows = manifest["tasks"]
    completed = [row for row in rows if "reward" in row]
    superseded = manifest.get('superseded_tasks', [])
    attempts = [*completed, *superseded]
    runtimes = [(row.get("runtime_usage") or {}).get("runtime", unavailable()) for row in attempts]
    batch_entries = manifest.get("batches", [row.get("learning", {}) for row in completed])
    reflections = [row["reflection"] for row in attempts if row.get("reflection", {}).get("attempted")]
    learning_entries = [*reflections, *batch_entries]
    attempted = [entry for entry in learning_entries if entry.get("attempted")]
    costs = [(entry.get("cost") or {}).get("total", unavailable()) for entry in attempted]
    runtime, learning = combined(runtimes), combined(costs)
    categories = {role: combined([(row.get("runtime_usage") or {}).get("categories", {}).get(role, unavailable()) for row in attempts]) for role in ROLES}
    learning_categories = {stage: combined([(entry.get("cost") or {}).get("categories", {}).get(stage, unavailable())
        for entry in attempted]) for stage in ("attribution", "proposer", "curator", "general_curator")}
    injection_missing = bool(plan["roles"]) and any(row.get("skill_usage") is None for row in attempts)
    injections = [item for row in attempts for item in (row.get("skill_usage") or {}).get("injections", [])]
    injection_missing = injection_missing or any(item.get("skills") and item.get("tokens") is None for item in injections)
    total = combined([runtime, learning])
    successful = sum(row["reward"] > 0 for row in completed)
    report = {"experiment": manifest.get("experiment", "online_skill_learning"), "held_out_evaluation": False, "status": manifest["status"],
              "batch_size": plan.get("batch_size"), "started_batches": len(batch_entries),
              "completed_batches": sum(entry.get("status") in {"completed", "completed_with_errors"} for entry in batch_entries),
              "task_reflections": len(reflections),
              "reflection_usage": combined([(entry.get("cost") or {}).get("total", unavailable()) for entry in reflections]),
              "curation_usage": combined([(entry.get("cost") or {}).get("total", unavailable()) for entry in batch_entries if entry.get("attempted")]),
              "task_order": plan["tasks"], "completed_tasks": len(completed), "selected_tasks": len(plan["tasks"]),
              "superseded_task_attempts": len(superseded), "resume_count": len(manifest.get('resumptions', [])),
              "runtime_cost_scope": "all attempts, including superseded attempts; success rate uses latest selected results",
              "raw_reward_mean_so_far": sum(row["reward"] for row in completed) / len(completed) if completed else None,
              "successful_tasks": successful, "agent_errors": sum(bool(row.get("agent_exception")) for row in completed),
              "raw_scores_modified": False, "role_usage": categories, "runtime_usage": runtime, "learning_usage": learning,
              "learning_stage_usage": learning_categories,
              "skill_injection_tokens": None if injection_missing else sum(item.get("tokens") or 0 for item in injections),
              "skill_injection_token_status": "unavailable" if injection_missing else "estimated" if any(item.get("skills") for item in injections) else "not_applicable",
              "skill_injection_accounting": "subset of role input, never added again",
              "runtime_plus_learning_usage": total,
              "runtime_tokens_per_successful_task": runtime["total_tokens"] / successful if successful and runtime.get("total_tokens") is not None else None,
              "runtime_plus_learning_tokens_per_successful_task": total["total_tokens"] / successful if successful and total.get("total_tokens") is not None else None,
              "execution_duration_seconds": sum(row.get("execution_duration_seconds", 0) for row in [*rows,*superseded]),
              "learning_duration_seconds": sum(entry.get("learning_duration_seconds", 0) for entry in learning_entries),
              "learning_errors": sum(bool(entry.get("returncode")) for entry in learning_entries),
              "api_cost": None, "cost_note": "No price invented; learning_usage includes all reported attribution/proposer/curator calls."}
    final = manifest.get("final_evaluation")
    if final:
        report["final_evaluation"] = {"experiment": final["experiment"], "held_out_evaluation": False,
                                      "status": final["status"], "output": final["output"], "summary": final.get("summary")}
        report["workflow_total_usage"] = combined([total, (final.get("summary") or {}).get("runtime_usage", unavailable())])
        successful_final = (final.get("summary") or {}).get("successful_tasks", 0)
        workflow_tokens = report["workflow_total_usage"].get("total_tokens")
        report["workflow_tokens_per_successful_final_task"] = workflow_tokens / successful_final if workflow_tokens is not None and successful_final else None
    atomic_json(plan["output"] / "manifest.json", manifest)
    atomic_json(plan["output"] / "summary.json", report)
    try:
        from lh_harness.skills.task_report import write_task_report
        write_task_report(plan['output'],manifest)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        # Reporting cannot change MEA outcomes or interrupt normal learning.
        print(f'逐题汇总暂未刷新：{exc}',file=sys.stderr,flush=True)


def run_sequence(plan, manifest, runtime):
    library = SkillLibrary(plan["output"] / "library") if plan["roles"] else None
    pending = []
    if not plan.get('resume'):
        manifest["batches"] = []
    if library:
        library.create()
        if not plan.get('resume'):
            atomic_json(library.root / "online_provenance.json", provenance(plan, [], "running"))
    manifest["status"] = "running"
    for index, task in enumerate(plan["tasks"], 1):
        if index < plan.get('start_index', 1):
            continue
        row = {"task_id": task, "index": index, "status": "running", "library_before": library_state(library) if library else None}
        manifest["tasks"].append(row)
        write_summary(plan, manifest)
        print(f"[{index}/{len(plan['tasks'])}] 执行任务：{task}", flush=True)
        started = time.monotonic()
        task_folder = plan["output"] / "tasks" / f"{index:03d}-{task}"
        if plan.get('resume'):
            task_folder = task_folder/('attempt-'+plan['resume_id'])
            row['resume_id'] = plan['resume_id']
        outcome = runtime.run_task(task, index, task_folder)
        row.update(outcome, execution_duration_seconds=time.monotonic() - started)
        if outcome["returncode"] != 0 or not outcome.get("trial"):
            row["status"] = "interrupted" if outcome["returncode"] == 130 else "infrastructure_failure"
            manifest["status"] = row["status"]
            save_attempted_provenance(plan, manifest)
            write_summary(plan, manifest)
            return outcome["returncode"] or 2
        trial = Path(outcome["trial"])
        result = json.loads((trial / "result.json").read_text())
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if type(reward) not in (int, float):
            row["status"], manifest["status"] = "missing_verifier_reward", "infrastructure_failure"
            save_attempted_provenance(plan, manifest)
            write_summary(plan, manifest)
            return 2
        identity = result.get("task_name") or trial.name.rsplit("__", 1)[0]
        if identity != task:
            raise ValueError("返回的评分任务 ID 与选择任务不一致")
        skill_usage = trial / "agent/skill_evolution/skill_usage.json"
        row.update(reward=reward, agent_exception=(result.get("exception_info") or {}).get("exception_type"), runtime_usage=runtime_usage(trial),
                   skill_usage=json.loads(skill_usage.read_text()) if skill_usage.exists() else None)
        row["raw_result_sha256"] = shared.native.digest(trial / "result.json")
        if library and library_state(library) != row["library_before"]:
            raise ValueError("Skill 库在任务执行期间发生变化；只允许 batch 之间学习")
        row.update(status="graded", learning={"attempted": False, "status": "pending_batch"},
                   library_after=library_state(library) if library else None)
        pending.append(row)
        write_summary(plan, manifest)
        if library:
            row["reflection"] = {"attempted": True, "stage": "propose", "status": "running"}
            write_summary(plan, manifest)
            hashes = score_hashes(trial)
            print(f"[{index}/{len(plan['tasks'])}] 逐题反思与 Proposal：{task}；不更新 Skill 库", flush=True)
            started = time.monotonic()
            try:
                proposal_folder = plan["output"] / "learning" / "proposals" / f"{index:03d}-{task}"
                if plan.get('resume'):
                    proposal_folder = proposal_folder/('attempt-'+plan['resume_id'])
                row["reflection"].update(runtime.reflect_task(task, trial, proposal_folder))
            except Exception as exc:
                row["reflection"].update(returncode=1, error=type(exc).__name__, detail=str(exc))
            row["reflection"].update(learning_duration_seconds=time.monotonic() - started,
                status="completed_with_errors" if row["reflection"].get("returncode") else "completed")
            if score_hashes(trial) != hashes:
                raise ValueError("逐题反思修改了原始评分产物，停止运行并保留证据")
            if library_state(library) != row["library_before"]:
                raise ValueError("逐题 Proposal 阶段修改了 Skill 库；只允许 Batch Curator 更新")
            write_summary(plan, manifest)
        if len(pending) < plan["batch_size"] and index < len(plan["tasks"]):
            continue
        batch_index = max((batch['batch_id'] for batch in manifest['batches']), default=0) + 1
        batch = {"batch_id": batch_index, "task_ids": [item["task_id"] for item in pending],
                 "size": len(pending), "attempted": bool(library), "status": "running",
                 "library_before": library_state(library) if library else None}
        manifest["batches"].append(batch)
        hashes = {item["trial"]: score_hashes(Path(item["trial"])) for item in pending}
        if library:
            print(f"[Batch {batch_index}] {len(pending)} 题已独立反思，统一审核、合并并更新 Skill：{', '.join(batch['task_ids'])}", flush=True)
            write_summary(plan, manifest)
            learning_started = time.monotonic()
            try:
                batch.update(runtime.learn_batch(pending, plan["output"] / "learning" / f"batch-{batch_index:03d}"))
            except Exception as exc:
                batch.update(returncode=1, error=type(exc).__name__, detail=str(exc))
            batch["learning_duration_seconds"] = time.monotonic() - learning_started
            save_attempted_provenance(plan, manifest)
            batch["library_after"] = library_state(library)
            row["library_after"] = batch["library_after"]
        else:
            batch.update(returncode=0, decision="TRAINING_DISABLED", library_after=None)
        for item in pending:
            if score_hashes(Path(item["trial"])) != hashes[item["trial"]]:
                raise ValueError("学习阶段修改了原始评分产物，停止运行并保留证据")
            item.update(status="completed", learning={"attempted": bool(library), "batch_id": batch_index,
                        "returncode": batch["returncode"], "report_path": batch.get("report_path")})
        batch["status"] = "completed_with_errors" if batch["returncode"] else "completed"
        pending = []
        write_summary(plan, manifest)
    errors = any(batch["returncode"] for batch in manifest["batches"]) or any(
        row.get("reflection", {}).get("returncode") for row in manifest["tasks"])
    manifest["status"] = "completed_with_learning_errors" if errors else "completed"
    save_attempted_provenance(plan, manifest)
    write_summary(plan, manifest)
    return 1 if errors else 0


def _run_final_evaluation(plan, manifest, runtime):
    """Fresh environments and fixed final library; default to selected training IDs."""
    manifest["training_status"] = manifest["status"]
    folder = plan["output"] / "evaluation"
    folder.mkdir(parents=True, exist_ok=False)
    text = re.sub(r'^phase\s*=.*$', 'phase = "test"', plan["config_text"], count=1, flags=re.MULTILINE)
    text, count = re.subn(r'(^\[skills\.training\]\n)enabled\s*=.*$',
                          r'\1enabled = false', text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("无法明确关闭最终复测中的 Skill Training")
    settings = Settings.from_dict(tomllib.loads(text)["skills"])
    if settings.phase != "test" or settings.training["enabled"]:
        raise ValueError("最终复测必须关闭学习并使用冻结库")
    (folder / "config.toml").write_text(text)
    library = SkillLibrary(plan["output"] / "library") if plan["roles"] else None
    frozen = (library.verify_frozen() if (library.root / "frozen.json").exists() else library.freeze()) if library else None
    evaluation_tasks = plan["evaluation_tasks"]
    overlap = [task for task in evaluation_tasks if task in set(plan["tasks"])]
    experiment = "training_set_recheck" if len(overlap) == len(evaluation_tasks) else "post_training_evaluation"
    final = {"experiment": experiment, "held_out_evaluation": False, "learning_enabled": False,
             "training_task_overlap": overlap, "task_order": list(evaluation_tasks), "output": str(folder), "frozen_library": frozen,
             "tasks": [], "batches": [], "status": "running"}
    manifest.update(status="evaluating", final_evaluation=final)
    final_plan = {**plan, "output": folder, "tasks": evaluation_tasks}
    evaluation_runtime = runtime.evaluation_runtime(settings)
    training_scores = {row["trial"]: score_hashes(Path(row["trial"])) for row in manifest["tasks"]}
    write_summary(plan, manifest)
    write_summary(final_plan, final)
    for index, task in enumerate(evaluation_tasks, 1):
        if library:
            library.verify_frozen()
        row = {"task_id": task, "index": index, "status": "running", "learning": {"attempted": False, "returncode": 0}}
        final["tasks"].append(row)
        write_summary(final_plan, final)
        write_summary(plan, manifest)
        print(f"[最终评估 {index}/{len(evaluation_tasks)}] {task}；使用固定最终库，不学习", flush=True)
        started = time.monotonic()
        outcome = evaluation_runtime.run_task(task, index, folder / "tasks" / f"{index:03d}-{task}")
        row.update(outcome, execution_duration_seconds=time.monotonic() - started)
        if library:
            library.verify_frozen()
        if outcome["returncode"] != 0 or not outcome.get("trial"):
            row["status"] = final["status"] = "interrupted" if outcome["returncode"] == 130 else "infrastructure_failure"
            manifest["status"] = "final_evaluation_failed"
            write_summary(final_plan, final)
            write_summary(plan, manifest)
            return outcome["returncode"] or 2
        trial = Path(outcome["trial"])
        result = json.loads((trial / "result.json").read_text())
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if type(reward) not in (int, float) or (result.get("task_name") or trial.name.rsplit("__", 1)[0]) != task:
            raise ValueError("最终复测缺少对应任务的原始评分")
        usage = trial / "agent/skill_evolution/skill_usage.json"
        row.update(status="completed", reward=reward, agent_exception=(result.get("exception_info") or {}).get("exception_type"),
                   runtime_usage=runtime_usage(trial), skill_usage=json.loads(usage.read_text()) if usage.exists() else None,
                   raw_result_sha256=shared.native.digest(trial / "result.json"))
        write_summary(final_plan, final)
        final["summary"] = json.loads((folder / "summary.json").read_text())
        write_summary(plan, manifest)
    for trial, expected in training_scores.items():
        if score_hashes(Path(trial)) != expected:
            raise ValueError("最终复测修改了训练阶段的原始评分")
    if library:
        library.verify_frozen()
    final["status"] = "completed"
    write_summary(final_plan, final)
    final["summary"] = json.loads((folder / "summary.json").read_text())
    manifest["status"] = manifest["training_status"]
    write_summary(plan, manifest)
    return 0


def run_final_evaluation(plan, manifest, runtime):
    try:
        return _run_final_evaluation(plan, manifest, runtime)
    except (Exception, KeyboardInterrupt) as exc:
        final = manifest.get("final_evaluation")
        if final:
            final["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            for row in final["tasks"]:
                if row["status"] == "running":
                    row["status"] = final["status"]
            write_summary({**plan, "output": Path(final["output"]), "tasks": final["task_order"]}, final)
            final["summary"] = json.loads((Path(final["output"]) / "summary.json").read_text())
            manifest["status"] = "final_evaluation_failed"
            write_summary(plan, manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="每题独立提取 Proposal，每 5 题统一审核更新 Skill，最后冻结库复测。默认全部 89 题。")
    parser.add_argument("roles", help="manager / executor / auditor / manager,executor / all / none")
    parser.add_argument("--task", action="append", nargs="+", metavar="TASK_ID", help="all（默认），或一个/多个任务 ID；支持重复 --task 或逗号分隔")
    parser.add_argument("--tasks-from-run", type=Path, help="从单个 Harbor 结果目录动态提取任务 ID；重新执行这些题，不用旧轨迹训练")
    parser.add_argument("--task-file", type=Path, help="可编辑的任务清单，每行一个 ID；支持空行、# 注释及 all，与 --task 互斥")
    parser.add_argument("--eval-task", action="append", nargs="+", metavar="TASK_ID", help="可选：独立选择最终评估任务；默认跟随本次训练选题，支持 all")
    parser.add_argument("--limit", type=int, help="只运行所选顺序中的前 N 题")
    parser.add_argument("--batch-size", type=int, help="默认 5；续跑时继承原 Batch 大小，不能隐式改变")
    parser.add_argument("--skip-final-eval", action="store_true", help="仅训练，显式跳过默认的训练集最终复测")
    parser.add_argument("--output", type=Path, help="独立的新输出目录")
    parser.add_argument("--resume", type=Path, help="在已停止的原运行目录续跑，保留旧结果与现有 Skill 库")
    parser.add_argument("--from-task", help="续跑起点：从 1 开始的题号或任务 ID，包含该题并重新执行")
    parser.add_argument("--base-url", help="宿主机与 Docker 均可访问的同一本地 Endpoint")
    parser.add_argument("--auth-scheme", choices=("none", "bearer", "x-api-key"), default=os.environ.get("LLH_LOCAL_AUTH_SCHEME", "none"))
    parser.add_argument("--api-key-env", help="API Key 环境变量名称，不是密钥值")
    parser.add_argument("--serving-manifest", type=Path, default=ROOT / "evaluation/validation/qwen27b-vision-serving.json")
    parser.add_argument("--freeze", action="store_true", help="兼容选项：默认最终复测前自动冻结；跳过复测时可单独冻结存档")
    parser.add_argument("--dry-run", action="store_true", help="只检查任务选择与配置，不写文件、不调用模型、不启动 Docker")
    args = parser.parse_args(argv)
    manifest = None
    p = None
    try:
        p = plan(args)
        print(f"模式：在线 Skill Learning；组合：{p['group']}；角色：{','.join(p['roles']) or 'none（原始控制）'}", flush=True)
        print(f"模型：{p['config']['run']['model']}；Endpoint：{p['endpoint']}", flush=True)
        print(f"任务数：{len(p['tasks'])}；任务顺序：{', '.join(p['tasks'])}", flush=True)
        print(f"Batch 大小：{p['batch_size']}；共 {(len(p['tasks']) + p['batch_size'] - 1) // p['batch_size']} 批；最后不足一批也学习", flush=True)
        final_action = "冻结最终库" if p["roles"] else "固定原始控制配置"
        scope = "训练集复测" if set(p["evaluation_tasks"]) <= set(p["tasks"]) else "自定义最终评估"
        print(f"首轮结束后：{final_action + '并评估 ' + str(len(p['evaluation_tasks'])) + ' 题（' + scope + '）' if p['final_eval'] else '按参数跳过最终复测'}", flush=True)
        if p["evaluation_tasks"] != p["tasks"]:
            print(f"最终评估任务：{', '.join(p['evaluation_tasks'])}", flush=True)
        print(f"输出目录：{p['output']}", flush=True)
        if p.get('resume'):
            print(f"原目录续跑：从第 {p['start_index']} 题 {p['tasks'][p['start_index']-1]} 开始；"
                  f"剩余 {len(p['tasks'])-p['start_index']+1} 题；保留前面的评分和当前 Skill 库", flush=True)
            print("新尝试追加在原目录内；剩余任务按原 Batch 大小重新分批；最终复测仍覆盖原评估清单。", flush=True)
        if args.dry_run:
            behavior = f"每题独立提取 Proposal，每 {p['batch_size']} 题统一审核更新" if p["roles"] else "原始控制，不更新 Skill"
            print(f"检查通过：{'沿用原运行状态' if p.get('resume') else '不会读取旧训练索引'}；{behavior}。未写文件、未调用模型。", flush=True)
            return 0
        lock_id = hashlib.sha256(p["endpoint"].encode()).hexdigest()[:20]
        with (ROOT / ".cache" / f"skill-pair-{lock_id}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if p.get('resume'):
                runtime = HarborRuntime(p, args)
                infrastructure = runtime.prepare()
                validate_resume_infrastructure(p, infrastructure)
                manifest = prepare_resume(p, infrastructure)
                write_summary(p, manifest)
                code = run_sequence(p, manifest, runtime)
                if code in (0, 1) and manifest['status'] in {'completed','completed_with_learning_errors'} and p['final_eval']:
                    code = run_final_evaluation(p, manifest, runtime) or code
                elif code == 0 and args.freeze and p['roles']:
                    manifest['frozen_library'] = SkillLibrary(p['output']/'library').freeze()
                    write_summary(p, manifest)
                print(f"原目录续跑结束，退出码 {code}；结果：{p['output']}", flush=True)
                return code
            p["output"].mkdir(parents=True, exist_ok=False)
            (p["output"] / "config.toml").write_text(p["config_text"])
            atomic_json(p["output"] / "online-split.json", p["online_split"])
            manifest = {"schema_version": 1, "experiment": "online_skill_learning", "track": p["track"], "group": p["group"],
                        "benchmark": p["benchmark"], "dataset_task_count": p["dataset_task_count"],
                        "train_roles": list(p["roles"]), "phase": "online", "held_out_evaluation": False,
                        "online_training_enabled": bool(p["roles"]),
                        "batch_size": p["batch_size"], "final_evaluation_enabled": p["final_eval"],
                        "normal_mea_outcome_modified": False, "task_order": p["tasks"], "dataset_sha256": p["inventory_sha256"],
                        "evaluation_task_order": p["evaluation_tasks"], "task_selection_source": p["task_selection_source"],
                        "formal_split_sha256": p["formal_split"]["sha256"], "online_split": p["online_split"],
                        "formal_test_overlap": sorted(set(p["tasks"]) & set(p["formal_split"]["test"])),
                        "schedule": "each task: grade -> reflect/propose without library writes; batch boundary: curate all task proposals -> update library; freeze final library -> recheck without learning",
                        "tasks": [], "batches": [], "status": "preparing"}
            write_summary(p, manifest)
            runtime = HarborRuntime(p, args)
            manifest["infrastructure"] = runtime.prepare()
            code = run_sequence(p, manifest, runtime)
            if code in (0, 1) and manifest["status"] in {"completed", "completed_with_learning_errors"} and p["final_eval"]:
                evaluation_code = run_final_evaluation(p, manifest, runtime)
                code = evaluation_code or code
            elif code == 0 and args.freeze and p["roles"]:
                manifest["frozen_library"] = SkillLibrary(p["output"] / "library").freeze()
                write_summary(p, manifest)
            print(f"在线运行结束，退出码 {code}；结果：{p['output']}", flush=True)
            return code
    except KeyboardInterrupt:
        if manifest is not None:
            manifest["status"] = "interrupted"
            save_attempted_provenance(p, manifest)
            write_summary(p, manifest)
        return 130
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        if manifest is not None:
            manifest.update(status="failed", error=type(exc).__name__, detail=str(exc))
            save_attempted_provenance(p, manifest)
            write_summary(p, manifest)
        print(f"在线运行拒绝启动或已停止：{exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
