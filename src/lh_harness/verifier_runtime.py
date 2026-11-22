"""Prepare and verify pinned scoring dependencies before any model work starts.

Original test scripts remain byte-identical. Only their known bootstrap tools
use prepared caches. Runtime cache misses are infrastructure errors, never
online fallbacks or ordinary model scores.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / ".cache/verifier-runtime-095"
SUPPORT = ROOT / "training/verifier_runtime_support"
UV_VERSION = "0.9.5"
UV_HASHES = {"uv": "d3dc3ca8e29337dd602a9a4df9e6edb15ebc52aed91461debeedb96939dc4ce0",
             "uvx": "421b1bb58429180a04ade98da066b5a5b6763e8ed74edef97359701c4cb42db8"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def profile_id(argv: list[str]) -> str:
    return hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()[:16]


def dependency_plan(dataset: Path, names: list[str]) -> dict:
    profiles, tasks = {}, {}
    for name in sorted(names):
        script = dataset / name / "tests/test.sh"
        raw = script.read_text()
        variables = {}
        for line in raw.splitlines():
            match = re.fullmatch(r'\s*COMMIT_HASH=["\x27]?([0-9a-f]{40})["\x27]?\s*', line)
            if match:
                variables["COMMIT_HASH"] = match[1]
        task_profiles = []
        for line in raw.replace("\\\n", " ").splitlines():
            if not re.match(r"^\s*(uvx\s|pip\s+install\s|uv\s+pip\s+install\s)", line):
                continue
            for key, value in variables.items():
                line = line.replace("${" + key + "}", value)
            argv = shlex.split(line, comments=True)
            if argv[0] == "uvx":
                if "pytest" not in argv:
                    raise ValueError(f"{name}: unsupported verifier tool command; prepare it explicitly")
                argv = argv[:argv.index("pytest") + 1]
                minor = argv[argv.index("-p") + 1] if "-p" in argv else "3.13"
            elif argv[:3] == ["uv", "pip", "install"]:
                minor = "3.12"
            else:
                minor = "system"
            if any("$" in item or "`" in item for item in argv):
                raise ValueError(f"{name}: unresolved variable in verifier requirements")
            key = profile_id(argv)
            profiles.setdefault(key, {"argv": argv, "python": minor, "tasks": []})["tasks"].append(name)
            task_profiles.append(key)
        tasks[name] = {"script_sha256": digest(script), "profiles": task_profiles,
                       "uv_installer": "https://astral.sh/uv/0.9.5/install.sh" in raw}
    return {"profiles": profiles, "tasks": tasks}


def cache_hashes(root: Path) -> dict[str, str]:
    result, inodes = {}, {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            result[str(p.relative_to(root))] = "symlink:" + os.readlink(p)
        elif p.is_file():
            s = p.stat()
            inode = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            if inode not in inodes:
                inodes[inode] = digest(p)
            result[str(p.relative_to(root))] = inodes[inode]
    return result


def bootstrap(root: Path) -> None:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    for name, expected in UV_HASHES.items():
        target = root / "bin" / name
        if not target.exists():
            shutil.copy2(ROOT / ".cache/uv/0.9.5" / name, target)
        if digest(target) != expected:
            raise ValueError(f"pinned {name} cache checksum mismatch: {target}")
    source = ROOT.parents[1] / ".cache/a-evolve-tb-runtime/python"
    target = root / "python"
    if not target.exists():
        shutil.copytree(source, target, symlinks=True)


def preparation_env(root: Path, cache: Path) -> dict:
    env = dict(os.environ)
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                 "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "VLLM_API_KEY", "LLH_LOCAL_API_KEY"):
        env.pop(name, None)
    env.pop("UV_OFFLINE", None)
    # Host proxy settings may point at a short-lived SSH tunnel. Preparation
    # uses the directly verified package hosts unless explicitly requested.
    if env.get("LLH_VERIFIER_PREPARE_USE_PROXY") != "1":
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(name, None)
    env.update(UV_CACHE_DIR=str(cache), UV_PYTHON_INSTALL_DIR=str(root / "python"),
               UV_PYTHON_PREFERENCE="only-managed", UV_PYTHON_DOWNLOADS="never",
               UV_HTTP_RETRIES="5", UV_HTTP_TIMEOUT="120", UV_TOOL_DIR=str(cache / "tools"),
               CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    return env


def run_logged(command: list[str], env: dict, log: Path, *, timeout: int = 1800) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        p = subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if p.returncode:
        raise ValueError(f"dependency preparation failed (exit {p.returncode}); inspect {log}")


def prepare_profile(root: Path, key: str, profile: dict) -> dict:
    directory = root / "profiles" / key
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "profile.json"
    cache = directory / "cache"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if (record.get("argv") == profile["argv"] and record.get("cache_files") == cache_hashes(cache)
                and record.get("wheel_files") == cache_hashes(directory / "wheels")):
            print(f"Verifier dependencies cached: {key} ({','.join(profile['tasks'][:2])})", flush=True)
            return record
        raise ValueError(f"verifier dependency cache was modified: {directory}")
    env = preparation_env(root, cache)
    argv = profile["argv"]
    log = directory / "prepare.log"
    print(f"Preparing verifier dependencies: {key} ({','.join(profile['tasks'][:2])})", flush=True)
    if argv[0] == "uvx":
        run_logged([str(root / "bin/uvx"), "-q", *argv[1:], "--version"], env, log)
    elif argv[0] == "uv":
        venv = directory / "prepare-venv"
        run_logged([str(root / "bin/uv"), "venv", "-q", "-p", profile["python"], str(venv)], env, log)
        run_logged([str(root / "bin/uv"), "pip", "install", "-q", "--python", str(venv / "bin/python"), *argv[3:]], env, log)
    else:
        python = root / "python/cpython-3.13.9-linux-x86_64-gnu/bin/python3.13"
        requirements = [arg for arg in argv[2:] if not arg.startswith("-")]
        wheels = directory / "wheels"
        for minor in ("3.11", "3.12", "3.13"):
            run_logged([str(python), "-m", "pip", "download", "--quiet", "--only-binary=:all:",
                        "--python-version", minor, "--implementation", "cp", "--abi", "cp" + minor.replace(".", ""),
                        "--abi", "abi3", "--abi", "none", "--platform", "manylinux2014_x86_64",
                        "--platform", "manylinux_2_28_x86_64", "--dest", str(wheels), *requirements], env, log)
    record = {**profile, "prepared_at": time.time(), "cache_files": cache_hashes(cache),
              "wheel_files": cache_hashes(directory / "wheels")}
    atomic_json(record_path, record)
    return record


def prepare(dataset: Path, names: list[str], root: Path = DEFAULT_ROOT, *, workers: int = 2) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = dependency_plan(dataset, names)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        bootstrap(root)
        minors = sorted({p["python"] for p in plan["profiles"].values()} - {"system"})
        for minor in minors:
            env = preparation_env(root, root / "bootstrap-cache")
            env["UV_PYTHON_DOWNLOADS"] = "automatic"
            run_logged([str(root / "bin/uv"), "python", "install", minor], env, root / "python-prepare.log")
        records, errors = {}, {}
        existing = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {"tasks": {}, "profiles": {}}
        existing["tasks"].update(plan["tasks"])
        existing.update(schema_version=1, uv_version=UV_VERSION, uv_hashes=UV_HASHES,
                        python_files=cache_hashes(root / "python"), support_files=cache_hashes(SUPPORT))
        def publish_progress():
            existing["profiles"].update(records)
            for key in errors:
                existing["profiles"].pop(key, None)
            existing.update(errors=dict(errors), preparation_complete=False)
            atomic_json(root / "manifest.json", existing)
        publish_progress()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(prepare_profile, root, key, profile): key for key, profile in plan["profiles"].items()}
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    records[key] = future.result()
                except Exception as exc:
                    errors[key] = str(exc)
                    print(f"Verifier preparation unavailable: {key}: {exc}", flush=True)
                publish_progress()
        existing.update(preparation_complete=not errors, errors=errors)
        atomic_json(root / "manifest.json", existing)
    if errors:
        raise ValueError(f"{len(errors)} verifier dependency profile(s) unavailable; benchmark will not start; see {root / 'manifest.json'}")
    from . import verifier_system
    verifier_system.prepare(dataset, names, root)
    return check(dataset, names, root)


def check(dataset: Path, names: list[str], root: Path = DEFAULT_ROOT) -> dict:
    root = root.resolve()
    path = root / "manifest.json"
    if not path.exists():
        raise ValueError("Verifier cache is not prepared. Run this launcher with --prepare-verifier-runtime first; benchmark not started")
    manifest = json.loads(path.read_text())
    plan = dependency_plan(dataset, names)
    if manifest.get("uv_hashes") != UV_HASHES:
        raise ValueError("verifier cache has an unexpected uv release")
    for name, expected in UV_HASHES.items():
        if not os.access(root / "bin" / name, os.X_OK) or digest(root / "bin" / name) != expected:
            raise ValueError(f"verifier {name} cache is corrupt")
    for name in ("curl", "uv", "uvx", "pip", "apt-get", "git", "asset_tools/git", "asset_tools/apt-get"):
        if not os.access(SUPPORT / name, os.X_OK):
            raise ValueError(f"verifier bootstrap helper is not executable: {SUPPORT / name}")
    if cache_hashes(root / "python") != manifest["python_files"]:
        raise ValueError("verifier Python cache was modified")
    if cache_hashes(SUPPORT) != manifest.get("support_files"):
        raise ValueError("verifier bootstrap helpers changed; refresh the prepared manifest before benchmarking")
    for name, task in plan["tasks"].items():
        if manifest["tasks"].get(name) != task:
            raise ValueError(f"verifier cache does not match the unchanged script for {name}; prepare again")
    missing = [key for key in plan["profiles"] if key not in manifest["profiles"]]
    if missing:
        tasks = sorted({name for key in missing for name in plan["profiles"][key]["tasks"]})
        raise ValueError(f"Verifier dependencies incomplete: {len(missing)}/{len(plan['profiles'])} profiles unavailable "
                         f"({', '.join(tasks)}). Continue --prepare-verifier-runtime; benchmark not started")
    for key in plan["profiles"]:
        profile = manifest["profiles"].get(key)
        directory = root / "profiles" / key
        if profile is None or profile.get("cache_files") != cache_hashes(directory / "cache") or profile.get("wheel_files") != cache_hashes(directory / "wheels"):
            raise ValueError(f"verifier dependency profile {key} is missing/corrupt; prepare before benchmarking")
    from . import verifier_system
    system = verifier_system.check(dataset, names, root)
    return {"status": "ready", "uv_version": UV_VERSION, "root": str(root), "manifest_sha256": digest(path),
            "tasks": len(names), "profiles": len(plan["profiles"]), "uv_python_downloads_during_grading": False,
            "system_dependencies": system,
            "other_test_networking": "unchanged"}


def configure(config: dict, env: dict, dataset: Path, names: list[str], root: Path = DEFAULT_ROOT) -> dict:
    record = check(dataset, names, root)
    compose = ROOT / "training/docker_verifier_runtime.yaml"
    config.setdefault("environment", {}).setdefault("extra_docker_compose", []).append(str(compose))
    values = {"LLH_VERIFIER_RUNTIME": str(root.resolve()), "LLH_VERIFIER_SUPPORT": str(SUPPORT),
              "BASH_ENV": str(SUPPORT / "activate")}
    config.setdefault("verifier", {}).setdefault("env", {}).update(values)
    config["agents"][0].setdefault("kwargs", {})["verifier_system_root"] = str(root.resolve())
    env.update(LLH_VERIFIER_RUNTIME=str(root.resolve()), LLH_VERIFIER_SUPPORT=str(SUPPORT))
    return record


def infrastructure_failure(trial: Path) -> str | None:
    sentinel = trial / "verifier/infra_error.json"
    if sentinel.exists():
        return "offline verifier bootstrap failed; inspect verifier/infra_error.json"
    log = trial / "verifier/test-stdout.txt"
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    before_tests = text.split("test session starts", 1)[0]
    if ("uvx: command not found" in before_tests or "failed to download https://github.com/astral-sh/uv/" in before_tests
            or re.search(r"curl: \(\d+\)", before_tests)
            or ("/root/.local/bin/env: No such file" in before_tests and "curl:" in before_tests)):
        return "uv bootstrap/network failure; authoritative tests did not start"
    return None


def audit_job(job: Path) -> dict:
    failures = {}
    for trial in job.iterdir() if job.exists() else []:
        if trial.is_dir():
            reason = infrastructure_failure(trial)
            if reason:
                failures[trial.name] = reason
    return {"benchmark_valid": not failures, "infrastructure_failures": failures,
            "model_score": None if failures else "use_original_harbor_result",
            "raw_scores_modified": False}


def run_harbor(command: list[str], env: dict, cwd: Path, job: Path, *, model_monitor=None) -> int:
    process = subprocess.Popen(command, env=env, cwd=cwd)
    stopped_for_infrastructure = False
    model_failure = None
    try:
        while process.poll() is None:
            report = audit_job(job)
            model_failure = model_monitor.check(job) if model_monitor else None
            if not report["benchmark_valid"] or model_failure:
                print("Evaluation invalid: dependency/model infrastructure failure; stopping remaining trials. Raw results preserved.", file=sys.stderr, flush=True)
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                stopped_for_infrastructure = True
                break
            time.sleep(0.5)
        code = process.wait()
    except KeyboardInterrupt:
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        process.wait()
        return 130
    report = audit_job(job)
    model_failure = model_failure or (model_monitor.check(job) if model_monitor else None)
    report["model_connection_failure"] = model_failure
    if model_failure:
        report.update(benchmark_valid=False, model_score=None)
    report["stopped_for_infrastructure"] = stopped_for_infrastructure
    atomic_json(job.with_suffix(".verification-status.json"), report)
    return 2 if not report["benchmark_valid"] else code
