from __future__ import annotations

import importlib.util
import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("skill_training_launcher", ROOT / "experiments/skill_evolution/scripts/train_skills.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "experiments/skill_evolution/configs", root / "experiments/skill_evolution/configs")
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"dataset": "fixture", "train": ["train-one", "train-missing"], "validation": [], "test": ["test-one"]}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "rounds.jsonl").write_text("")
    index = tmp_path / "runs.json"
    index.write_text(json.dumps({"train-one": str(run), "train-missing": str(tmp_path / "missing")}))
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.delenv("LLH_LOCAL_BASE_URL", raising=False)
    monkeypatch.delenv("LLH_LOCAL_AUTH_SCHEME", raising=False)
    return tmp_path, root, split, index


def arguments(setup, roles="executor", **extra):
    tmp, root, split, index = setup
    return SimpleNamespace(roles=roles, output=tmp / "new-output", run_index=index, split_manifest=split,
                           base_url=None, api_key_env=None, auth_scheme="none", freeze=False, dry_run=True, **extra)


@pytest.mark.parametrize("roles,group", [
    ("none", "baseline"), ("manager", "manager_only"), ("executor", "executor_only"),
    ("auditor", "auditor_only"), ("manager,executor", "manager_executor"),
    ("manager,auditor", "manager_auditor"), ("executor,auditor", "executor_auditor"), ("all", "all_roles"),
])
def test_all_role_combinations_preserve_model_and_existing_libraries(setup, roles, group):
    args = arguments(setup, roles)
    old = setup[1] / "experiments/skill_evolution/libraries/all_roles/index.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"frozen":true}')
    output, text, runs, split, record = launcher.prepare(args)
    prepared = tomllib.loads(text)
    original = tomllib.loads((setup[1] / f"experiments/skill_evolution/configs/{group}.toml").read_text())
    assert record["group"] == group
    assert prepared["run"] == original["run"]
    assert prepared["local_model"] == original["local_model"]
    assert prepared["skills"]["training"]["train_roles"] == record["train_roles"]
    assert prepared["skills"]["library_dir"] == str(output / "library")
    assert old.read_text() == '{"frozen":true}'
    assert not output.exists()
    assert sorted(runs) == ([] if roles == "none" else ["train-one"])
    assert "--flush" in record["command"]


def test_dry_run_never_launches_learning_or_writes_files(setup, monkeypatch):
    tmp, root, split, index = setup
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("dry-run must not invoke a model or CLI"))
    assert launcher.offline_main(["executor", "--split-manifest", str(split), "--run-index", str(index),
                          "--output", str(tmp / "dry"), "--dry-run"]) == 0
    assert not (tmp / "dry").exists()


def test_test_task_is_rejected_before_any_output_even_if_path_missing(setup):
    args = arguments(setup)
    args.run_index.write_text(json.dumps({"test-one": "/missing"}))
    with pytest.raises(ValueError, match="非 train"):
        launcher.prepare(args)
    assert not args.output.exists()


def test_existing_output_is_not_overwritten(setup):
    args = arguments(setup)
    args.output.mkdir()
    marker = args.output / "keep"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="输出目录已存在"):
        launcher.prepare(args)
    assert marker.read_text() == "keep"


def test_endpoint_environment_overrides_only_endpoint(setup, monkeypatch):
    monkeypatch.setenv("LLH_LOCAL_BASE_URL", "http://127.0.0.1:8010")
    _, text, _, _, record = launcher.prepare(arguments(setup))
    assert record["base_url"] == "http://127.0.0.1:8010/v1"
    assert tomllib.loads(text)["run"]["model"] == "/data0/liuxule/project/models/Qwen3.8-27B"


@pytest.mark.parametrize("value", ["executor,executor", "unknown", "manager,,executor"])
def test_invalid_roles_do_not_start_training(setup, value):
    with pytest.raises(ValueError, match="角色只能"):
        launcher.prepare(arguments(setup, value))


def test_freeze_runs_only_after_success_and_snapshots_are_self_contained(setup, monkeypatch):
    tmp, root, split, index = setup
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0))
    output = tmp / "train"
    assert launcher.offline_main(["executor", "--split-manifest", str(split), "--run-index", str(index),
                          "--output", str(output), "--freeze"]) == 0
    assert calls[0][4] == "learn" and calls[1][4] == "freeze"
    assert json.loads((output / "training-runs.json").read_text()).keys() == {"train-one"}
    assert (output / "split-manifest.json").read_bytes() == split.read_bytes()
    assert tomllib.loads((output / "config.toml").read_text())["skills"]["library_dir"] == str(output / "library")


def test_learning_failure_does_not_freeze_library(setup, monkeypatch):
    tmp, root, split, index = setup
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=1))
    assert launcher.offline_main(["executor", "--split-manifest", str(split), "--run-index", str(index),
                          "--output", str(tmp / "failed"), "--freeze"]) == 1
    assert len(calls) == 1
