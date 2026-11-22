"""Keep Qwen thinking policy identical across real original LLH role adapters."""
from __future__ import annotations

import importlib.util
import json

import pytest

pytest.importorskip("harbor")
from harbor_agent import CuaHarnessClaudeCodeAgent
from harbor_backends.constants import CLAUDE_CODE_PROXY_HOST
from lh_harness import eval_inspect_tb2_original as runner


def test_all_original_roles_disable_thinking_without_rewriting_prompts_or_tools(tmp_path):
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    declaration = config["agents"][0]
    agent = CuaHarnessClaudeCodeAgent(logs_dir=tmp_path, model_name=declaration["model_name"], **declaration["kwargs"])
    agent._installed_claude_code_proxy = True
    roles = agent._make_role_agents(environment=None, env_adapter=None, output_dir=tmp_path,
                                    model=declaration["model_name"], verifier_model=declaration["model_name"],
                                    api_key="local", base_url="http://172.17.0.1:8010", workspace_path="/app",
                                    event_logger=None)
    spec = importlib.util.spec_from_file_location("original_proxy_reasoning_test", CLAUDE_CODE_PROXY_HOST)
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    incoming = {"model": declaration["model_name"], "stream": True, "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"}, "messages": [{"role": "user", "content": "original prompt"}],
                "tools": [{"name": "Read", "input_schema": {"type": "object"}}]}
    assert set(roles) == {"orchestrator", "cli_task", "cli_verifier"}
    for role in roles.values():
        env = role._command_env("/tmp/check-role")
        overrides = proxy.parse_overrides(env["CUA_HARNESS_CLAUDECODE_REQUEST_OVERRIDES"])
        rewritten, changed, _ = proxy.rewrite_body(json.dumps(incoming).encode(), overrides=overrides)
        payload = json.loads(rewritten)
        assert changed and payload["thinking"] == {"type": "disabled"}
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        for key in ("messages", "tools", "model", "stream", "output_config"):
            assert payload[key] == incoming[key]
    assert "Bash" in roles["orchestrator"].disallowed_tools
    assert "Write" in roles["cli_verifier"].disallowed_tools
    assert "Write" not in roles["cli_task"].disallowed_tools
    assert runner.endpoint_profile(config)["reasoning"] == {"enable_thinking": False}


def test_serving_snapshot_must_match_active_model_and_excludes_unknown_fields(tmp_path):
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    kwargs = config["agents"][0]["kwargs"]
    check = {"model_id": "test-model", "base_url": "http://127.0.0.1:8010/v1", "context_length": 1000000,
             "serving_framework": "vllm", "serving_version": "test-version"}
    observed = {**check, "temperature": kwargs["temperature"], "top_p": kwargs["top_p"], "top_k": kwargs["top_k"],
                "max_output_tokens": kwargs["max_tokens"], "api_key": "test-only-do-not-copy"}
    path = tmp_path / "serving.json"
    path.write_text(json.dumps(observed))
    evidence = runner.serving_evidence(path, check, kwargs)
    assert "api_key" not in evidence["observed"] and "test-only-do-not-copy" not in json.dumps(evidence)
    observed["model_id"] = "other-model"
    path.write_text(json.dumps(observed))
    with pytest.raises(ValueError, match="model_id differs from the active endpoint"):
        runner.serving_evidence(path, check, kwargs)


def test_original_and_paired_preflight_require_multi_image_capacity(tmp_path):
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    assert runner.endpoint_profile(config, require_image=True)["image_probe_count"] == 32
    assert "image_probe_count" not in runner.endpoint_profile(config)


def test_old_four_image_snapshot_cannot_validate_new_multi_image_run(tmp_path):
    config = runner.resolved_config(runner.DEFAULT_CONFIG, tmp_path / "dataset")
    kwargs = config["agents"][0]["kwargs"]
    check = {"model_id": "test-model", "base_url": "http://127.0.0.1:8010/v1", "context_length": 1000000,
             "serving_framework": "vllm", "serving_version": "test-version", "image_probe": "verified", "image_probe_count": 32}
    observed = {**check, "temperature": kwargs["temperature"], "top_p": kwargs["top_p"], "top_k": kwargs["top_k"],
                "max_output_tokens": kwargs["max_tokens"], "modalities": {"image": 4, "video": 0}}
    path = tmp_path / "serving.json"
    path.write_text(json.dumps(observed))
    with pytest.raises(ValueError, match="serving manifest image limit"):
        runner.serving_evidence(path, check, kwargs)
    observed["modalities"]["image"] = 32
    path.write_text(json.dumps(observed))
    assert runner.serving_evidence(path, check, kwargs)["observed"]["modalities"]["image"] == 32
