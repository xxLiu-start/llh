"""Default to running tasks and learning online; retain explicit --offline."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lh_harness.local_models import base_url, is_local
from lh_harness.skills.experience import load_split

ROLES = ("manager", "executor", "auditor")
GROUPS = {
    (): "baseline",
    ("manager",): "manager_only",
    ("executor",): "executor_only",
    ("auditor",): "auditor_only",
    ("manager", "executor"): "manager_executor",
    ("manager", "auditor"): "manager_auditor",
    ("executor", "auditor"): "executor_auditor",
    ROLES: "all_roles",
}


def prepare(args):
    selected = list(ROLES) if args.roles == "all" else ([] if args.roles == "none" else args.roles.split(","))
    if len(set(selected)) != len(selected) or any(role not in ROLES for role in selected):
        raise ValueError("角色只能是 manager、executor、auditor（逗号连接），或 all / none")
    roles = tuple(role for role in ROLES if role in selected)
    group = GROUPS[roles]
    if args.freeze and not roles:
        raise ValueError("none 不创建 Skill Library，不需要 --freeze")
    split_path = args.split_manifest.expanduser().resolve()
    split = load_split(split_path)
    template = ROOT / "experiments/skill_evolution/configs" / (group + ".toml")
    text = template.read_text()
    source = tomllib.loads(text)
    endpoint = base_url(args.base_url or os.environ.get("LLH_LOCAL_BASE_URL") or source["run"]["base_url"])
    if not is_local(endpoint):
        raise ValueError("此入口仅连接本地模型，请提供本地 Endpoint；不会切换云端服务")
    if args.api_key_env and (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env)
                             or not os.environ.get(args.api_key_env)):
        raise ValueError("--api-key-env 必须是已设置的环境变量名称，不能是密钥值")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "experiments/skill_evolution/results/skill_enhanced" /
              f"training-{group}-{stamp}-{uuid.uuid4().hex[:8]}").expanduser().resolve()
    if output.exists():
        raise ValueError(f"输出目录已存在，请选择新目录，避免覆盖历史结果：{output}")
    runs, omitted = {}, []
    if roles:
        index_path = args.run_index.expanduser().resolve()
        index = json.loads(index_path.read_text())
        if not isinstance(index, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in index.items()):
            raise ValueError("训练索引必须是 task_id 到单任务日志目录的 JSON 对象")
        for task, location in index.items():
            if task not in split["train"]:
                raise ValueError(f"训练索引包含非 train 任务，拒绝运行：{task}")
            run = Path(location).expanduser()
            if not run.is_absolute():
                run = index_path.parent / run
            run = run.resolve()
            if not run.is_dir():
                omitted.append({"task_id": task, "path": str(run), "reason": "directory_missing"})
                continue
            if len(list(run.rglob("rounds.jsonl"))) != 1:
                raise ValueError(f"需要恰好包含一个 rounds.jsonl 的单任务目录：{task}: {run}")
            runs[task] = str(run)
        if not runs:
            raise ValueError("没有现存训练日志，请补充训练集历史任务并更新 --run-index；不会启动 Benchmark")
    for field, value in (("library_dir", str(output / "library")), ("base_url", endpoint)):
        text, count = re.subn(rf"^{field}\s*=.*$", lambda _: field + " = " + json.dumps(value),
                              text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"配置中的 {field} 不是唯一字段，拒绝自动修改：{template}")
    prepared = tomllib.loads(text)
    assert prepared["run"]["model"] == source["run"]["model"]
    assert prepared.get("local_model") == source.get("local_model")
    command = [sys.executable, "-m", "lh_harness", "skills", "learn", "--config", str(output / "config.toml"),
               "--roles", ",".join(roles) or "none", "--split-manifest", str(output / "split-manifest.json"),
               "--run-index", str(output / "training-runs.json"), "--output", str(output / "learning"),
               "--auth-scheme", args.auth_scheme, "--flush"]
    if args.api_key_env:
        command += ["--api-key-env", args.api_key_env]
    record = {"group": group, "train_roles": list(roles), "model_id": prepared["run"]["model"],
              "backend": prepared["run"]["agent"], "base_url": endpoint, "output": str(output),
              "library": str(output / "library"), "training_tasks": sorted(runs), "omitted_runs": omitted,
              "split_sha256": split["sha256"], "auth_scheme": args.auth_scheme, "api_key_env": args.api_key_env,
              "freeze_after_learning": args.freeze, "dry_run": args.dry_run, "command": command}
    return output, text, runs, split_path, record


def offline_main(argv=None):
    parser = argparse.ArgumentParser(description="从已有轨迹训练指定角色的 Skill；不运行 Benchmark，不修改模型权重。")
    parser.add_argument("roles", help="manager / executor / auditor / manager,executor / all / none")
    parser.add_argument("--output", type=Path, help="新输出目录；默认自动生成独立目录")
    parser.add_argument("--run-index", type=Path, default=ROOT / "experiments/skill_evolution/manifests/training-runs.json")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "experiments/skill_evolution/manifests/pilot-splits.json")
    parser.add_argument("--base-url", help="本地 API 根地址；默认沿用 LLH_LOCAL_BASE_URL 或组合配置")
    parser.add_argument("--auth-scheme", choices=("none", "bearer", "x-api-key"), default=os.environ.get("LLH_LOCAL_AUTH_SCHEME", "none"))
    parser.add_argument("--api-key-env", help="API Key 环境变量名称；请勿传入密钥值")
    parser.add_argument("--freeze", action="store_true", help="学习成功返回后冻结本轮库，供独立测试使用")
    parser.add_argument("--dry-run", action="store_true", help="检查配置与日志路径，不写文件、不调用模型")
    args = parser.parse_args(argv)
    try:
        output, text, runs, split_path, record = prepare(args)
        print(f"训练组合：{record['group']}；角色：{','.join(record['train_roles']) or 'none'}", flush=True)
        print(f"模型：{record['model_id']}；Endpoint：{record['base_url']}", flush=True)
        print(f"训练任务：{', '.join(record['training_tasks']) or '无（训练关闭）'}", flush=True)
        for omitted in record["omitted_runs"]:
            print(f"历史目录不存在，未纳入训练：{omitted['task_id']}: {omitted['path']}", flush=True)
        print(f"输出目录：{output}", flush=True)
        if args.dry_run:
            print("仅检查通过，未写文件、未调用模型。", flush=True)
            print(shlex.join(record["command"]), flush=True)
            return 0
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.toml").write_text(text)
        (output / "training-runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        (output / "split-manifest.json").write_text(split_path.read_text())
        (output / "session.json").write_text(json.dumps(record, indent=2) + "\n")
        if record["train_roles"]:
            print("提示：单任务经验通过 Curator 审核即可成为 Active Skill；无有效经验或证据不足仍可返回 NONE/DEFER。", flush=True)
        status = subprocess.run(record["command"], cwd=ROOT, check=False).returncode
        if status == 0 and args.freeze:
            status = subprocess.run([sys.executable, "-m", "lh_harness", "skills", "freeze", "--library",
                                     str(output / "library"), "--split-manifest", str(output / "split-manifest.json")],
                                    cwd=ROOT, check=False).returncode
        print(f"运行结束，退出码 {status}；结果保存在：{output}", flush=True)
        return status
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"训练未启动或未完成：{exc}", file=sys.stderr, flush=True)
        return 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--offline" in argv:
        argv.remove("--offline")
        return offline_main(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from online_skill_training import main as online_main
    return online_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
