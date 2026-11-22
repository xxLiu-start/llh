"""Installation fixes must preserve the original evaluation and exact CLI release."""
from __future__ import annotations

import copy
import shutil
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
from harbor_agent import CuaHarnessClaudeCodeAgent
from lh_harness.original_harness_eval import OriginalHarnessHarborAgent
from lh_harness import eval_inspect_tb2_original as runner
from lh_harness.claude_cache import find_cached_claude_binary


def cached_binary(tmp_path):
    path = tmp_path / "claude"
    path.write_bytes(b"test-only cached executable")
    path.chmod(0o755)
    return path


def test_cached_installation_preserves_original_execution_and_configuration(tmp_path, monkeypatch):
    binary = cached_binary(tmp_path)
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    before = copy.deepcopy(config)
    monkeypatch.setattr(runner, "find_cached_claude_binary", lambda *a: binary)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="2.1.176 (Claude Code)"))
    env = {}
    record = runner.configure_installation(config, env)
    assert record["version"] == "2.1.176" and record["mode"] == "host_cache"
    assert record["sha256"] == runner.digest(binary)
    assert env == {"LH_CLAUDE_OFFLINE_BINARY": str(binary)}
    assert OriginalHarnessHarborAgent.run is CuaHarnessClaudeCodeAgent.run
    config["agents"][0]["import_path"] = before["agents"][0]["import_path"]
    assert config["agents"][0]["kwargs"].pop("offline_claude_binary") is True
    compose = config["environment"].pop("extra_docker_compose")
    assert compose == [str(runner.ROOT / "training/docker_claude_offline.yaml")]
    assert config == before


def test_wrong_explicit_release_cannot_silently_download_another_version(tmp_path, monkeypatch):
    binary = cached_binary(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="2.1.175 (Claude Code)"))
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="required release 2.1.176"):
        runner.configure_installation(config, {}, binary)
    assert config == before


def test_missing_explicit_binary_does_not_fall_back_to_online_install(tmp_path):
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    with pytest.raises(ValueError, match="missing or not executable"):
        runner.configure_installation(config, {"LH_CLAUDE_OFFLINE_BINARY": str(tmp_path / "missing")})


def test_no_local_cache_refuses_runtime_network_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "find_cached_claude_binary", lambda *a: None)
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="no runtime network installation"):
        runner.configure_installation(config, {})
    assert config == before


def test_persistent_cache_is_preferred_and_survives_output_cleanup(tmp_path, monkeypatch):
    cache = tmp_path / ".cache/claude-code"
    stable = cache / "2.1.176/claude"
    stable.parent.mkdir(parents=True)
    stable.write_text("test cached binary")
    stable.chmod(0o755)
    output = tmp_path / "output/original"
    legacy = output / "jobs/claude_code_cache/2.1.176/claude"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("test old cache")
    legacy.chmod(0o755)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="2.1.176 (Claude Code)"))
    assert find_cached_claude_binary(output, "2.1.176", cache) == stable
    shutil.rmtree(output.parent)
    assert find_cached_claude_binary(output, "2.1.176", cache) == stable


def test_stale_explicit_environment_reports_how_to_use_persistent_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DEFAULT_CLAUDE_CACHE", tmp_path / ".cache/claude-code")
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    with pytest.raises(ValueError, match="unset LH_CLAUDE_OFFLINE_BINARY"):
        runner.configure_installation(config, {"LH_CLAUDE_OFFLINE_BINARY": str(tmp_path / "deleted/output/claude")})
