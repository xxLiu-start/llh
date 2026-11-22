from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("harbor")
yaml = pytest.importorskip("yaml")
from harbor.models.task.task import Task
from lh_harness.inspect_tb2_dataset import prepare_dataset, tree_hashes, verify_dataset
from lh_harness import eval_inspect_tb2_original as runner


def fixture_task(root: Path, name="one") -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests/test.sh").write_bytes(b"#!/bin/bash\nprintf 1 > /logs/verifier/reward.txt\n")
    (task / "tests/original_helper.py").write_text("# unchanged verifier\n")
    (task / "solution/solve.sh").write_text("#!/bin/bash\necho fixture\n")
    definition = {"name": name, "sandbox": {"type": "docker", "config": "compose.yaml"},
                  "files": {"/tests/test.sh": "tests/test.sh", "/tests/helper.py": "tests/original_helper.py"},
                  "variants": {"default": {"prompt": "精确保留题目说明。\n\n"},
                               "solution": {"prompt": "Run bash /solution/solve.sh", "files": {"/solution/solve.sh": "solution/solve.sh"}}},
                  "metadata": {"agent_timeout_sec": 750, "verifier_timeout_sec": 360, "difficulty": "hard"}}
    service = {"image": "example/fixture:original", "platform": "linux/amd64",
               "command": ["sh", "-c", "sleep infinity"], "init": True, "mem_limit": "4g", "cpus": 2}
    (task / "eval.yaml").write_text(yaml.safe_dump(definition, allow_unicode=True))
    (task / "compose.yaml").write_text(yaml.safe_dump({"services": {"default": service}}))
    return task


def test_conversion_preserves_verifier_prompt_image_resources_and_timeouts(tmp_path):
    source = tmp_path / "source"
    task = fixture_task(source)
    before = tree_hashes(source)
    output = tmp_path / "harbor"
    manifest = prepare_dataset(source, output, expected_tasks=1)
    converted = output / task.name
    assert tree_hashes(source) == before
    assert (converted / "instruction.md").read_text() == "精确保留题目说明。\n\n"
    assert (converted / "tests/test.sh").read_bytes() == (task / "tests/test.sh").read_bytes()
    assert (converted / "tests/helper.py").read_bytes() == (task / "tests/original_helper.py").read_bytes()
    assert (converted / "solution/solve.sh").read_bytes() == (task / "solution/solve.sh").read_bytes()
    original_service = yaml.safe_load((task / "compose.yaml").read_text())["services"]["default"]
    assert yaml.safe_load((converted / "environment/docker-compose.yaml").read_text())["services"]["main"] == original_service
    assert (converted / "environment/Dockerfile").read_text() == "FROM example/fixture:original\n"
    loaded = Task(converted)
    assert loaded.config.environment.memory_mb == 4096
    assert loaded.config.environment.cpus == 2
    assert loaded.config.agent.timeout_sec == 750 and loaded.config.verifier.timeout_sec == 360
    assert loaded.config.environment.workdir is None
    assert manifest["benchmark_version"] == "2.0"
    assert manifest["instructions_and_verifiers_modified"] is False


def test_idempotent_reuse_verifies_all_files_and_refuses_source_changes(tmp_path):
    source = tmp_path / "source"
    task = fixture_task(source)
    output = tmp_path / "harbor"
    first = prepare_dataset(source, output, expected_tasks=1)
    stamp = (output / "dataset_manifest.json").stat().st_mtime_ns
    assert prepare_dataset(source, output, expected_tasks=1) == first
    assert (output / "dataset_manifest.json").stat().st_mtime_ns == stamp
    (task / "tests/test.sh").write_text("changed source verifier")
    with pytest.raises(ValueError, match="source definitions changed"):
        prepare_dataset(source, output, expected_tasks=1)
    assert (output / "one/tests/test.sh").read_bytes() != (task / "tests/test.sh").read_bytes()


def test_modified_adapted_tests_are_not_silently_used(tmp_path):
    source = tmp_path / "source"
    fixture_task(source)
    output = tmp_path / "harbor"
    prepare_dataset(source, output, expected_tasks=1)
    (output / "one/tests/test.sh").write_text("echo fabricated success")
    with pytest.raises(ValueError, match="adapted dataset was modified"):
        verify_dataset(source, output)


@pytest.mark.parametrize("destination,relative_source", [("/app/leaked_test.py", "tests/original_helper.py"),
                                                        ("/tests/../../app/leak.py", "tests/original_helper.py"),
                                                        ("/tests/leak.py", "../../outside.py")])
def test_unsupported_or_escaping_mappings_fail_atomically(tmp_path, destination, relative_source):
    source = tmp_path / "source"
    task = fixture_task(source)
    (tmp_path / "outside.py").write_text("must not be copied")
    definition = yaml.safe_load((task / "eval.yaml").read_text())
    definition["files"] = {destination: relative_source}
    (task / "eval.yaml").write_text(yaml.safe_dump(definition))
    output = tmp_path / "harbor"
    with pytest.raises(ValueError):
        prepare_dataset(source, output, expected_tasks=1)
    assert not output.exists()


def test_wrong_task_count_cannot_launch_partial_dataset(tmp_path):
    source = tmp_path / "source"
    fixture_task(source)
    with pytest.raises(ValueError, match="expected 89 tasks, found 1"):
        prepare_dataset(source, tmp_path / "harbor", expected_tasks=89)


def test_original_configuration_uses_same_local_model_without_timeout_override(tmp_path):
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "harbor")
    agent = config["agents"][0]
    assert agent["import_path"] == "harbor_agent:CuaHarnessClaudeCodeAgent"
    assert agent["model_name"] == agent["kwargs"]["harness_model"] == agent["kwargs"]["verifier_model"]
    assert "override_timeout_sec" not in agent and "override_timeout_sec" not in config.get("verifier", {})
    assert "workspace_path" not in agent["kwargs"]
    assert agent["kwargs"]["max_rounds"] == 25


def test_unavailable_model_refuses_harbor_start(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runner, "prepare_dataset", lambda *a, **k: {"tasks": [{"name": "one"}]})
    monkeypatch.setattr(runner, "model_url", lambda env: "http://172.17.0.1:8010")
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/fake/harbor")
    def unavailable(*args, **kwargs):
        raise ValueError("local endpoint unavailable; no fallback")
    monkeypatch.setattr(runner, "check_endpoint", unavailable)
    def forbidden(*args, **kwargs):
        pytest.fail("Harbor must not start when the local model is unavailable")
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    jobs = tmp_path / "jobs"
    assert runner.main(["--jobs-dir", str(jobs), "--dataset-output", str(tmp_path / "harbor")]) == 2
    assert not jobs.exists()
    assert "no fallback" in capsys.readouterr().err
