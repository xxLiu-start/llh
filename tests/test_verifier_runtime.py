from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest

from lh_harness import verifier_runtime as runtime


def task(root, name="one"):
    path = root / name / "tests/test.sh"
    path.parent.mkdir(parents=True)
    path.write_text("#!/bin/bash\ncurl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n"
                    "uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n")
    return path


def prepared(tmp_path, monkeypatch):
    from lh_harness import verifier_system
    monkeypatch.setattr(verifier_system, "check", lambda *args: {"status": "ready"})
    dataset = tmp_path / "dataset"
    task(dataset)
    root = tmp_path / "cache"
    (root / "bin").mkdir(parents=True)
    uv = root / "bin/uv"
    uv.write_text("pinned-test-only")
    uv.chmod(0o755)
    monkeypatch.setattr(runtime, "UV_HASHES", {"uv": runtime.digest(uv)})
    (root / "python").mkdir()
    (root / "python/version").write_text("3.13-test")
    plan = runtime.dependency_plan(dataset, ["one"])
    profiles = {}
    for key, spec in plan["profiles"].items():
        cache = root / "profiles" / key / "cache"
        cache.mkdir(parents=True)
        (cache / "package").write_text("exact dependency")
        profiles[key] = {**spec, "cache_files": runtime.cache_hashes(cache), "wheel_files": {}}
    runtime.atomic_json(root / "manifest.json", {**plan, "profiles": profiles, "uv_hashes": runtime.UV_HASHES,
                                               "python_files": runtime.cache_hashes(root / "python"),
                                               "support_files": runtime.cache_hashes(runtime.SUPPORT)})
    return dataset, root


def test_plan_keeps_original_commands_pins_and_script_bytes(tmp_path):
    path = task(tmp_path)
    before = path.read_bytes()
    plan = runtime.dependency_plan(tmp_path, ["one"])
    profile = next(iter(plan["profiles"].values()))
    assert profile["argv"] == ["uvx", "-p", "3.13", "-w", "pytest==8.4.1", "-w", "pytest-json-ctrf==0.3.5", "pytest"]
    assert plan["tasks"]["one"]["script_sha256"] == runtime.digest(path)
    assert path.read_bytes() == before


def test_missing_cache_refuses_before_benchmark(tmp_path):
    dataset = tmp_path / "dataset"
    task(dataset)
    with pytest.raises(ValueError, match="prepare-verifier-runtime"):
        runtime.check(dataset, ["one"], tmp_path / "missing")


def test_preparation_uses_direct_network_without_changing_parent_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://unavailable-proxy.invalid:1234")
    monkeypatch.setenv("http_proxy", "http://unavailable-proxy.invalid:1234")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-do-not-forward")
    monkeypatch.delenv("LLH_VERIFIER_PREPARE_USE_PROXY", raising=False)
    env = runtime.preparation_env(tmp_path, tmp_path / "cache")
    assert "HTTPS_PROXY" not in env and "http_proxy" not in env and "OPENAI_API_KEY" not in env
    assert __import__("os").environ["HTTPS_PROXY"] == "http://unavailable-proxy.invalid:1234"
    monkeypatch.setenv("LLH_VERIFIER_PREPARE_USE_PROXY", "1")
    assert runtime.preparation_env(tmp_path, tmp_path / "cache")["HTTPS_PROXY"] == "http://unavailable-proxy.invalid:1234"


def test_ready_task_can_run_while_an_unrelated_profile_is_still_preparing(tmp_path, monkeypatch):
    dataset, root = prepared(tmp_path, monkeypatch)
    second = task(dataset, "two")
    second.write_text(second.read_text().replace("pytest==8.4.1", "pytest==8.4.2"))
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tasks"].update(runtime.dependency_plan(dataset, ["two"])["tasks"])
    manifest["preparation_complete"] = False
    runtime.atomic_json(path, manifest)
    assert runtime.check(dataset, ["one"], root)["status"] == "ready"
    with pytest.raises(ValueError, match="dependencies incomplete: 1/2"):
        runtime.check(dataset, ["one", "two"], root)


def test_offline_bootstrap_preserves_task_python_and_all_other_arguments(monkeypatch):
    spec = importlib.util.spec_from_file_location("verifier_dispatch_test", runtime.SUPPORT / "dispatch.py")
    dispatcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatcher)
    monkeypatch.setattr(dispatcher.shutil, "which", lambda name: "/usr/local/bin/python3.13")
    monkeypatch.setattr(dispatcher.subprocess, "check_output", lambda *a, **kw: "3.13\n")
    args = ["-p", "3.13", "-w", "pytest==8.4.1", "pytest", "/tests/test_outputs.py"]
    result, image_python = dispatcher.task_interpreter_args(args, Path("/cache"))
    assert image_python and result == ["-p", "/usr/local/bin/python3.13", *args[2:]]
    assert args[1] == "3.13"
    monkeypatch.setattr(dispatcher.subprocess, "check_output", lambda *a, **kw: "3.12\n")
    assert dispatcher.task_interpreter_args(args, Path("/cache")) == (args, False)


def test_cache_integrity_and_verifier_only_configuration(tmp_path, monkeypatch):
    dataset, root = prepared(tmp_path, monkeypatch)
    config = {"agents": [{"env": {"EXISTING": "unchanged"}}]}
    env = {}
    record = runtime.configure(config, env, dataset, ["one"], root)
    assert record["uv_python_downloads_during_grading"] is False
    assert config["agents"] == [{"env": {"EXISTING": "unchanged"}, "kwargs": {"verifier_system_root": str(root.resolve())}}]
    assert "BASH_ENV" in config["verifier"]["env"] and "PATH" not in config["verifier"]["env"]
    path = next((root / "profiles").glob("*/cache/package"))
    path.write_text("corrupt cache")
    with pytest.raises(ValueError, match="missing/corrupt"):
        runtime.check(dataset, ["one"], root)


def test_changed_test_script_cannot_reuse_prepared_cache(tmp_path, monkeypatch):
    dataset, root = prepared(tmp_path, monkeypatch)
    path = dataset / "one/tests/test.sh"
    path.write_text(path.read_text().replace("pytest==8.4.1", "pytest==8.4.2"))
    with pytest.raises(ValueError, match="does not match"):
        runtime.check(dataset, ["one"], root)


def test_infrastructure_zero_is_distinct_from_real_test_failure(tmp_path):
    trial = tmp_path / "job/trial"
    (trial / "verifier").mkdir(parents=True)
    log = trial / "verifier/test-stdout.txt"
    log.write_text("curl: (18) truncated download\nuvx: command not found\n")
    report = runtime.audit_job(trial.parent)
    assert report["benchmark_valid"] is False and report["model_score"] is None
    assert report["raw_scores_modified"] is False
    log.write_text("================ test session starts ================\nFAILED test_answer - AssertionError\n1 failed\n")
    assert runtime.audit_job(trial.parent)["benchmark_valid"] is True


def test_guard_stops_on_bootstrap_error_and_preserves_raw_results(tmp_path):
    job = tmp_path / "job"
    script = ("import json,time;from pathlib import Path;"
              f"p=Path({str(job / 'trial/verifier')!r});p.mkdir(parents=True);"
              "(p/'infra_error.json').write_text('{}');"
              "(p.parent/'result.json').write_text('{\"reward\":0}');time.sleep(10)")
    code = runtime.run_harbor([sys.executable, "-c", script], {}, tmp_path, job)
    assert code == 2
    assert json.loads((job / "trial/result.json").read_text()) == {"reward": 0}
    status = json.loads(job.with_suffix(".verification-status.json").read_text())
    assert status["stopped_for_infrastructure"] is True and status["model_score"] is None
