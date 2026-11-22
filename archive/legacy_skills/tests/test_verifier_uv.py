from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from lh_harness.eval_terminal_bench_skillbanks import _write_eval_config
from lh_harness.eval_terminal_bench_pro_original import _write_config as _write_original_config
from lh_harness.eval_terminal_bench_pro_original import main as original_eval_main
from lh_harness.train_terminal_bench_pro import _write_trial_config
from lh_harness.verifier_uv import (COMPOSE_NAME, INSTALLER_URL, add_verifier_uv,
                                    scoped_verifier_uv_host_env, verifier_uv_host_env)


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    tests = dataset / "task" / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text(
        f"#!/bin/bash\ncurl -LsSf {INSTALLER_URL} | sh\n"
        "source $HOME/.local/bin/env\nuv run pytest /tests/test_outputs.py\n"
    )
    return dataset


def _fake_uv(tmp_path: Path, version: str = "0.7.13") -> Path:
    binary = tmp_path / "uv"
    binary.write_text(f'#!/bin/sh\necho "uv {version}"\n')
    binary.chmod(0o755)
    return binary


def test_all_tb_pro_drivers_supply_uv_only_for_matching_verifier(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    original_script = (dataset / "task" / "tests" / "test.sh").read_bytes()
    template = {"environment": {"type": "docker"}, "agents": [{"kwargs": {}}]}
    training_path = tmp_path / "training.yaml"
    _write_trial_config(template, task="task", phase="baseline", dataset=dataset,
                        bank=tmp_path / "bank.json", jobs_dir=tmp_path / "jobs",
                        config_path=training_path)
    training = yaml.safe_load(training_path.read_text())
    evaluation, _ = _write_eval_config(
        template, task="task", dataset=dataset, output_root=tmp_path / "eval",
        config_path=tmp_path / "eval.yaml", run_id="run12345", frozen_bank=tmp_path / "bank.json")
    original, _ = _write_original_config(
        template, task="task", dataset=dataset, output_root=tmp_path / "original",
        config_path=tmp_path / "original.yaml", run_id="run12345")
    for config in (training, evaluation, original):
        assert COMPOSE_NAME in [Path(path).name for path in config["environment"]["extra_docker_compose"]]
        assert config["verifier"]["env"]["HOME"] == "/root"
        assert config["verifier"]["env"]["PATH"].startswith("/opt/lhh-verifier-uv:")
        assert "PATH" not in config["agents"][0].get("env", {})
    assert (dataset / "task" / "tests" / "test.sh").read_bytes() == original_script
    plain = {"environment": {"type": "docker"}}
    add_verifier_uv(plain, dataset=dataset, task="other")
    assert plain == {"environment": {"type": "docker"}}


def test_original_eval_passes_uv_mounts_to_harbor(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset(tmp_path)
    (dataset / "task" / "task.toml").write_text("")
    (dataset / "task" / "instruction.md").write_text("")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({
        "environment": {"type": "docker"}, "agents": [{"kwargs": {}}],
        "verifier": {}, "datasets": [],
    }))
    binary = _fake_uv(tmp_path)
    monkeypatch.setenv("LH_VERIFIER_UV_BINARY", str(binary))
    monkeypatch.delenv("LH_VERIFIER_UV_SUPPORT_DIR", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    claude = tmp_path / "claude"
    claude.write_text("#!/bin/sh\necho '2.1.176 (Claude Code)'\n")
    claude.chmod(0o755)
    monkeypatch.setenv("LH_CLAUDE_OFFLINE_BINARY", str(claude))
    observed = []

    def fake_run(config_path, job_dir, use_build_proxy):
        del job_dir, use_build_proxy
        config = yaml.safe_load(config_path.read_text())
        observed.append((config, os.environ.get("LH_VERIFIER_UV_SUPPORT_DIR")))
        return 0, {"task_name": "task", "verifier_result": {"rewards": {"reward": 1.0}}}, config_path, None

    monkeypatch.setattr("lh_harness.eval_terminal_bench_pro_original._run_harbor", fake_run)
    assert original_eval_main(["--dataset", str(dataset), "--template", str(template),
                               "--output", str(tmp_path / "output")]) == 0
    assert len(observed) == 1
    assert COMPOSE_NAME in [Path(path).name for path in observed[0][0]["environment"]["extra_docker_compose"]]
    assert observed[0][1] == str(Path(__file__).resolve().parents[1] / "training" / "verifier_uv_support")
    assert "LH_VERIFIER_UV_SUPPORT_DIR" not in os.environ


def test_host_uv_is_pinned_and_scoped(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset(tmp_path)
    config = {"environment": {"type": "docker"}}
    add_verifier_uv(config, dataset=dataset, task="task")
    binary = _fake_uv(tmp_path)
    monkeypatch.setenv("LH_VERIFIER_UV_BINARY", str(binary))
    monkeypatch.delenv("LH_VERIFIER_UV_SUPPORT_DIR", raising=False)
    resolved = verifier_uv_host_env(config)
    assert resolved["LH_VERIFIER_UV_BINARY"] == str(binary)
    assert Path(resolved["LH_VERIFIER_UV_SUPPORT_DIR"], "curl").is_file()
    with scoped_verifier_uv_host_env(config):
        assert os.environ["LH_VERIFIER_UV_SUPPORT_DIR"] == resolved["LH_VERIFIER_UV_SUPPORT_DIR"]
    assert "LH_VERIFIER_UV_SUPPORT_DIR" not in os.environ
    monkeypatch.setenv("LH_VERIFIER_UV_BINARY", str(_fake_uv(tmp_path, "0.9.0")))
    with pytest.raises(RuntimeError, match="requires uv 0.7.13"):
        verifier_uv_host_env(config)


def test_offline_installer_shim_only_handles_pinned_url() -> None:
    wrapper = Path(__file__).resolve().parents[1] / "training" / "verifier_uv_support" / "curl"
    response = subprocess.run([str(wrapper), "-LsSf", INSTALLER_URL], check=True,
                              capture_output=True, text=True)
    assert "/opt/lhh-verifier-uv/uv --version" in response.stdout
    assert "uv 0.7.13" in response.stdout
