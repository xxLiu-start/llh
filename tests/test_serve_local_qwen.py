from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from lh_harness import serve_local_qwen as launcher


def recipe(tmp_path, monkeypatch):
    root = tmp_path / "a-evolve"
    (root / "agent_evolve/llm").mkdir(parents=True)
    (root / "agent_evolve/llm/local.py").write_text("# fixture only\n")
    calls = []
    native = SimpleNamespace(LOCAL_HOST="127.0.0.1")
    def main():
        calls.append({"host": native.LOCAL_HOST, "port": os.environ.get("AEVOLVE_LOCAL_PORT"),
                      "has_server_auth": bool(os.environ.get("VLLM_API_KEY")),
                      "argv": list(sys.argv), "gpus": os.environ.get("CUDA_VISIBLE_DEVICES")})
    native.main = main
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: native)
    return root, native, calls


def test_real_start_requires_explicit_gpu_allocation(monkeypatch, capsys):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: pytest.fail("do not import/load model before allocation"))
    with pytest.raises(SystemExit) as error:
        launcher.main([])
    assert error.value.code == 2
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err


def test_launch_reuses_recipe_with_selected_gpus_docker_binding_and_no_auth(tmp_path, monkeypatch):
    root, native, calls = recipe(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("VLLM_API_KEY", "inherited-test-secret")
    monkeypatch.setenv("AEVOLVE_LOCAL_PORT", "9999")
    launcher.main(["--a-evolve-root", str(root), "--dry-run", "--max-model-len", "1000000"])
    assert calls == [{"host": "0.0.0.0", "port": "8010", "has_server_auth": False,
                      "argv": ["agent_evolve.llm.local", "--max-model-len", "1000000", "--tensor-parallel-size", "4", "--dry-run"],
                      "gpus": "4,5,6,7"}]
    assert native.LOCAL_HOST == "127.0.0.1"
    assert os.environ["AEVOLVE_LOCAL_PORT"] == "9999"
    assert os.environ["VLLM_API_KEY"] == "inherited-test-secret"


def test_explicit_tensor_parallel_size_is_preserved(tmp_path, monkeypatch):
    root, native, calls = recipe(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    launcher.main(["--a-evolve-root", str(root), "--dry-run", "--tensor-parallel-size", "2", "--host", "127.0.0.1"])
    assert calls[0]["argv"].count("--tensor-parallel-size") == 1
    assert calls[0]["host"] == "127.0.0.1"


def test_dry_run_can_inspect_recipe_without_allocating_any_gpu(tmp_path, monkeypatch):
    root, native, calls = recipe(tmp_path, monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    launcher.main(["--a-evolve-root", str(root), "--dry-run", "--tensor-parallel-size", "4"])
    assert calls[0]["gpus"] is None
    assert "--dry-run" in calls[0]["argv"]
def test_vision_command_changes_only_explicit_modality_configuration():
    from lh_harness.serve_local_qwen import vision_command
    command = ["python", "-m", "vllm.entrypoints.cli.main", "serve", "same-model", "--language-model-only",
               "--tensor-parallel-size", "4", "--max-model-len", "1000000"]
    enabled = vision_command(command, 4)
    assert enabled[:-2] == [item for item in command if item != "--language-model-only"]
    assert enabled[-2:] == ["--limit-mm-per-prompt", '{"image":4,"video":0}']
    assert "--language-model-only" in command


