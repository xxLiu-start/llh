from __future__ import annotations

import json
import importlib.util
import contextlib
import os
import socket
import socketserver
import threading
from io import BytesIO
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import yaml

from lh_harness.harbor_url import claude_code_upstream_base_url
from lh_harness import docker_build_proxy as build_proxy_module
from lh_harness.skill_learning import SkillBank, stage_skill_candidate
from lh_harness.training_provenance import ProvenanceWriter
from lh_harness.train_terminal_bench_pro import _find_cached_claude_binary, _learning_reason, _require_valid_trial_report, _run_trial, _transfer_evolution_candidates, _trial_token_usage, _write_trial_config, discover_tasks, main, train_one


def test_evolution_trial_candidate_reaches_persistent_training_bank(tmp_path: Path, monkeypatch) -> None:
    candidate = {**_candidate(), "routing_summary": "Use for missing handoff evidence",
                 "triggers": ["Evidence handoff is required"],
                 "exclusions": ["No handoff"], "validation": {}}
    bank_path = tmp_path / "bank.json"

    def fake_run_trial(config_path, jobs_dir):
        config = yaml.safe_load(config_path.read_text())
        trial_bank = SkillBank.load(Path(config["agents"][0]["kwargs"]["skill_bank_path"]))
        stage_skill_candidate(trial_bank, candidate)
        output = jobs_dir / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        (routes / "skill_candidate.json").write_text(json.dumps({
            "kind": "evolution", "candidate": [candidate], "router_candidate": None,
            "assessment": {"create_skills": [], "revise_skills": [], "update_router": []},
            "request_outcomes": [{"branch": "create_skills", "status": "provisional"}],
        }))
        (routes / "skill_routes.json").write_text("[]")
        return {"task": "Do the task", "status": "incomplete", "completion_satisfied": False,
                "rounds": [{"round_index": 1, "plan_text": "Attempted task"}],
                "skill_learning": {"action": "EVOLVE", "task_tokens": 100}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {"prompt_language": "en"}}]},
        dataset=tmp_path / "dataset", bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert result["reason"] == "provisional_staged"
    assert result["candidate_ids"] == [candidate["skill_id"]]
    assert result["transfer_outcomes"][0]["status"] == "provisional"
    bank = SkillBank.load(bank_path)
    assert bank.skills[0]["status"] == "provisional"
    assert bank.eligible("manager", "en", benchmark_id="terminal_bench_2_1",
                         interaction_mode="cli")[0]["skill_id"] == candidate["skill_id"]


def test_evolution_transfer_keeps_valid_items_when_sibling_is_missing(tmp_path: Path) -> None:
    good = {**_candidate(), "routing_summary": "Use for evidence handoff", "validation": {}}
    missing = {**_candidate(), "skill_id": "missing_sibling",
               "routing_summary": "Use for missing sibling", "validation": {}}
    source_path, bank_path = tmp_path / "trial.json", tmp_path / "bank.json"
    stage_skill_candidate(SkillBank(source_path), good)
    target, outcomes, router_transferred = _transfer_evolution_candidates(
        proposal={"candidate": [good, missing], "router_candidate": None},
        trial_bank_path=source_path, bank_path=bank_path, language="en",
    )
    assert [item["status"] for item in outcomes] == ["provisional", "transfer_failed"]
    assert router_transferred is False
    assert [item["skill_id"] for item in SkillBank.load(bank_path).skills] == [good["skill_id"]]


def test_evolution_router_dependency_follows_transferred_skill(tmp_path: Path) -> None:
    skill = {**_candidate(), "routing_summary": "Use for evidence handoff", "validation": {}}
    source_path, bank_path = tmp_path / "trial.json", tmp_path / "bank.json"
    source = SkillBank(source_path)
    stage_skill_candidate(source, skill, save=False)
    source.router_candidates.append({
        "language": "en", "prompt": source.router_prompts["en"] + "\nUse the new handoff Skill.",
        "status": "provisional", "dependencies": [skill["skill_id"]],
        "evidence_refs": ["round_001"], "validation": {},
    })
    source.save()
    proposal = {"candidate": [skill], "router_candidate": source.router_candidates[0]["prompt"]}
    target, outcomes, transferred = _transfer_evolution_candidates(
        proposal=proposal, trial_bank_path=source_path, bank_path=bank_path, language="en")
    assert transferred is True
    assert [item["status"] for item in outcomes] == ["provisional", "provisional"]
    assert SkillBank.load(bank_path).router_candidates[0]["dependencies"] == [skill["skill_id"]]

    blocked_path = tmp_path / "blocked.json"
    _, blocked_outcomes, blocked = _transfer_evolution_candidates(
        proposal={"candidate": [], "router_candidate": proposal["router_candidate"]},
        trial_bank_path=source_path, bank_path=blocked_path, language="en")
    assert blocked is False
    assert blocked_outcomes[0]["status"] == "transfer_failed"


def test_evolution_no_candidate_reason_is_explicit() -> None:
    assert _learning_reason({"action": "EVOLVE", "request_outcomes": [
        {"branch": "create_skills", "status": "rejected", "reason": "missing problem_mechanism"},
    ]}) == "evolution_candidates_rejected"
    assert _learning_reason({"action": "EVOLVE", "request_outcomes": []}) == "evolution_no_update"


def _candidate() -> dict:
    view = {"input_use": "Read audited evidence", "output_spec": "Report proof", "guidance": "Check state", "avoid": "Do not guess"}
    return {
        "skill_id": "evidence_handoff", "category_id": "evidence_handoff", "version": 1,
        "version_parent": None, "status": "candidate", "content_language": "en",
        "benchmark_scope": "terminal_bench_2_1", "interaction_modes": ["cli"],
        "problem_mechanism": "Missing handoff evidence", "subcases": [],
        "triggers": [], "exclusions": [], "roles": {role: view for role in ("manager", "executor", "auditor")},
        "compatible_with": [], "conflicts_with": [], "provenance": {"evidence_refs": ["round_001"]},
        "validation": {"baseline_success": None, "candidate_success": None},
    }


def _candidate_zh() -> dict:
    candidate = _candidate()
    candidate["content_language"] = "zh"
    candidate["problem_mechanism"] = "角色交接缺少证据"
    candidate["roles"] = {role: {
        "input_use": "核对上游证据", "output_spec": "交付状态和证明",
        "guidance": "独立验证关键步骤", "avoid": "不要猜测输入",
    } for role in ("manager", "executor", "auditor")}
    return candidate


def test_trial_token_usage_keeps_task_and_provider_counts(tmp_path: Path) -> None:
    output = tmp_path / "trial"
    output.mkdir()
    (output / "training_report.json").write_text(json.dumps({
        "report": {}, "usage_totals": {"n_input_tokens": 90, "n_output_tokens": 30, "n_cache_tokens": 20},
        "usage_partial": False,
    }))
    usage = _trial_token_usage({"skill_learning": {"task_tokens": 100}}, output)
    assert usage["total_tokens"] == 100 and usage["source"] == "skill_session"
    assert usage["usage_totals"]["n_output_tokens"] == 30
    assert usage["usage_partial"] is False
    fallback = _trial_token_usage({"skill_learning": {"task_tokens": None}}, output)
    assert fallback["total_tokens"] == 120 and fallback["source"] == "harbor_provider_totals"
    assert fallback["task_tokens"] is None and fallback["comparison_missing"] is True
    (output / "training_report.json").unlink()
    missing = _trial_token_usage({}, output)
    assert missing["missing"] is True and missing["total_tokens"] is None


def test_training_config_uses_pro_and_test_config_keeps_original_dataset() -> None:
    root = Path(__file__).resolve().parents[1]
    train = yaml.safe_load((root / "training/terminal_bench_pro_skill_learning.yaml").read_text())
    test = yaml.safe_load((root / "training/terminal_bench21_skill_test.yaml").read_text())
    assert train["datasets"][0]["path"] == "train_dataset/terminal-bench-pro"
    assert train["jobs_dir"] == "output/terminal-bench-pro/jobs"
    assert train["agents"][0]["kwargs"]["prompt_language"] == "zh"
    assert train["agents"][0]["env"]["LH_HARNESS_MODEL_IO_TRACE"] == "1"
    for config in (train, test):
        agent = config["agents"][0]
        assert agent["model_name"] == "deepseek-v4.1-flash"
        assert agent["kwargs"]["harness_model"] == "deepseek-v4.1-flash"
        assert agent["kwargs"]["verifier_model"] == "deepseek-v4.1-flash"
        assert agent["env"]["LH_HARNESS_PROXY_BEARER_AUTH"] == "1"
        assert agent["env"]["LH_HARNESS_PROXY_OPENAI_BRIDGE"] == "1"
    assert train["n_concurrent_trials"] == 1
    assert test["datasets"][0]["path"] == "eval/TB-harness/datasets/terminal-bench-2-1/tasks"
    assert test["jobs_dir"] == "output/terminal-bench-2-1/jobs"
    assert test["agents"][0]["kwargs"]["prompt_language"] == "en"
    assert test["agents"][0]["kwargs"]["learning_enabled"] is False
    assert test["n_attempts"] == 3


def test_claude_code_upstream_does_not_duplicate_v1() -> None:
    assert claude_code_upstream_base_url("https://api.example.test/v1") == "https://api.example.test"
    assert claude_code_upstream_base_url("https://api.example.test/v1/v1") == "https://api.example.test"
    assert claude_code_upstream_base_url("https://api.example.test/anthropic/v1/") == "https://api.example.test/anthropic"
    assert claude_code_upstream_base_url("https://api.example.test/anthropic") == "https://api.example.test/anthropic"


def _proxy_module():
    root = Path(__file__).resolve().parents[1]
    proxy_path = root / "eval/TB-harness/Harness/src/assets/claudecode_request_proxy.py"
    spec = importlib.util.spec_from_file_location("claudecode_request_proxy", proxy_path)
    assert spec is not None and spec.loader is not None
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    return proxy


def test_gateway_proxy_sends_bearer_auth_without_x_api_key(monkeypatch) -> None:
    proxy = _proxy_module()
    handler = object.__new__(proxy.RequestProxyHandler)
    headers = Message()
    headers["x-api-key"] = "test-secret"
    headers["anthropic-version"] = "2023-06-01"
    handler.headers = headers
    handler.config = proxy.ProxyConfig(
        upstream="https://example.test", overrides={}, dashscope_wait_timeout=None,
        log_file=None, requests_log_file=None,
    )

    monkeypatch.setenv("LH_HARNESS_PROXY_BEARER_AUTH", "1")
    forwarded = handler._forward_headers(b"{}")
    assert forwarded["Authorization"] == "Bearer test-secret"
    assert all(key.lower() != "x-api-key" for key in forwarded)
    assert forwarded["anthropic-version"] == "2023-06-01"

    monkeypatch.delenv("LH_HARNESS_PROXY_BEARER_AUTH")
    forwarded = handler._forward_headers(b"{}")
    assert forwarded["x-api-key"] == "test-secret"
    assert all(key.lower() != "authorization" for key in forwarded)


def test_proxy_records_exact_client_model_input_separately(monkeypatch, tmp_path: Path) -> None:
    proxy = _proxy_module()
    handler = object.__new__(proxy.RequestProxyHandler)
    trace_file = tmp_path / "model_io.jsonl"
    handler.config = proxy.ProxyConfig(
        upstream="https://example.test", overrides={}, dashscope_wait_timeout=None,
        log_file=tmp_path / "proxy.log", requests_log_file=None, model_io_file=trace_file,
    )
    handler.command = "POST"
    handler.path = "/v1/messages?beta=true"
    handler.headers = Message()
    handler.headers["x-api-key"] = "test-secret"
    payload = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "read /app"}]}
    handler._read_body = lambda: json.dumps(payload).encode()
    bridged = []
    handler._forward_openai_bridge = lambda body, headers, *, trace_id: bridged.append((body, headers, trace_id))
    monkeypatch.setenv("LH_HARNESS_PROXY_BEARER_AUTH", "1")
    monkeypatch.setenv("LH_HARNESS_PROXY_OPENAI_BRIDGE", "1")
    handler._forward()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert records[0]["event"] == "client_request"
    assert records[0]["payload"]["value"] == payload
    assert bridged[0][1]["Authorization"] == "Bearer test-secret"
    assert bridged[0][2] == records[0]["trace_id"]
    assert "test-secret" not in trace_file.read_text()


def test_openai_bridge_preserves_tool_call_loop() -> None:
    proxy = _proxy_module()
    request = {
        "model": "deepseek-v4.1-flash", "max_tokens": 128, "stream": True,
        "system": [{"type": "text", "text": "Use tools.", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "system", "content": {"type": "text", "text": "Only use /app."}},
            {"role": "user", "content": "Read a file"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_previous", "name": "Bash", "input": {"command": "pwd"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_previous", "content": "/app"}]},
        ],
        "tools": [{"name": "Bash", "description": "Run shell", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}],
    }
    translated = proxy.anthropic_to_openai(request)
    assert translated["stream"] is True
    assert translated["messages"][0] == {"role": "system", "content": "Use tools."}
    assert translated["messages"][1] == {"role": "system", "content": "Only use /app."}
    assert translated["messages"][3]["tool_calls"][0]["id"] == "toolu_previous"
    assert translated["messages"][4] == {"role": "tool", "tool_call_id": "toolu_previous", "content": "/app"}
    assert translated["tools"][0]["function"]["name"] == "Bash"
    tool_history = dict(request)
    tool_history["messages"] = [{"role": "tool", "tool_call_id": "toolu_previous", "content": "ok"}]
    assert proxy.anthropic_to_openai(tool_history)["messages"][-1] == {
        "role": "tool", "tool_call_id": "toolu_previous", "content": "ok",
    }

    completion = {
        "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "reasoning_content": "checked the next step", "tool_calls": [
            {"type": "function", "id": "call_1", "function": {"name": "Bash", "arguments": '{"command":"ls"}'}}
        ]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12},
    }
    reasoning = {}
    message = proxy.openai_to_anthropic(completion, request["model"], reasoning_by_tool_id=reasoning)
    assert message["stop_reason"] == "tool_use"
    assert message["content"][0]["input"] == {"command": "ls"}
    assert message["content"][0]["id"].startswith("toolu_")
    assert reasoning[message["content"][0]["id"]] == "checked the next step"
    next_request = dict(request)
    next_request["messages"] = [
        {"role": "user", "content": "Read a file"},
        {"role": "assistant", "content": message["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": message["content"][0]["id"], "content": "ok"}]},
    ]
    continued = proxy.anthropic_to_openai(next_request, reasoning_by_tool_id=reasoning)
    assert continued["messages"][2]["reasoning_content"] == "checked the next step"
    events = [json.loads(line[6:]) for line in proxy.anthropic_sse(message).decode().splitlines() if line.startswith("data: ")]
    assert events[0]["type"] == "message_start"
    assert any(event.get("content_block", {}).get("type") == "tool_use" for event in events)
    assert events[-2]["delta"]["stop_reason"] == "tool_use"
    assert events[-1]["type"] == "message_stop"


def test_openai_bridge_forwards_to_chat_endpoint_and_returns_tool_sse(monkeypatch, tmp_path: Path) -> None:
    proxy = _proxy_module()
    handler = object.__new__(proxy.RequestProxyHandler)
    handler.config = proxy.ProxyConfig(
        upstream="https://example.test", overrides={}, dashscope_wait_timeout=None,
        log_file=None, requests_log_file=None, model_io_file=tmp_path / "model_io.jsonl",
    )
    handler.path = "/v1/messages?beta=true"
    handler.wfile = BytesIO()
    response_status = []
    response_headers = {}
    handler.send_response = lambda status: response_status.append(status)
    handler.send_header = lambda name, value: response_headers.__setitem__(name.lower(), value)
    handler.end_headers = lambda: None
    upstream_calls = []

    class FakeResponse:
        status = 200

        def read(self):
            chunks = [
                {"choices": [{"index": 0, "delta": {"reasoning_content": "check files ", "tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command":'}}
                ]}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {"reasoning_content": "first", "tool_calls": [
                    {"index": 0, "function": {"arguments": '"pwd"}'}}
                ]}, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
            ]
            return ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()

    class FakeConnection:
        def request(self, method, target, body, headers):
            upstream_calls.append((method, target, json.loads(body), headers))

        def getresponse(self):
            return FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr(proxy, "make_connection", lambda parsed: FakeConnection())
    body = json.dumps({
        "model": "deepseek-v4.1-flash", "stream": True, "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Run pwd"},
        ],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    }).encode()
    handler._forward_openai_bridge(body, {"Authorization": "Bearer test-secret"})

    assert response_status == [200]
    assert response_headers["content-type"] == "text/event-stream"
    assert upstream_calls[0][0:2] == ("POST", "/v1/chat/completions")
    assert upstream_calls[0][2]["stream"] is True
    assert upstream_calls[0][2]["messages"][0] == {"role": "system", "content": "Use tools."}
    assert upstream_calls[0][3]["Authorization"] == "Bearer test-secret"
    assert b'"type":"tool_use"' in handler.wfile.getvalue()
    assert list(handler.config.reasoning_by_tool_id.values()) == ["check files first"]
    model_events = [json.loads(line) for line in (tmp_path / "model_io.jsonl").read_text().splitlines()]
    assert [item["event"] for item in model_events] == ["model_input", "model_output", "client_response"]
    assert model_events[0]["payload"]["messages"][0]["role"] == "system"
    assert model_events[1]["payload"]["format"] == "sse"
    assert model_events[2]["payload"]["format"] == "sse"
    assert "test-secret" not in (tmp_path / "model_io.jsonl").read_text()


def test_openai_stream_text_and_multiple_tool_calls() -> None:
    proxy = _proxy_module()
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "Hello", "tool_calls": [
            {"index": 1, "id": "call_2", "type": "function", "function": {"name": "Bash", "arguments": '{"command":'}},
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command":'}},
        ]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": " world", "tool_calls": [
            {"index": 0, "function": {"arguments": '"pwd"}'}},
            {"index": 1, "function": {"arguments": '"ls"}'}},
        ]}, "finish_reason": "tool_calls"}]},
    ]
    raw = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
    completion = proxy.openai_stream_to_completion(raw)
    assert completion["choices"][0]["message"]["content"] == "Hello world"
    message = proxy.openai_to_anthropic(completion, "deepseek-v4.1-flash")
    assert [block["type"] for block in message["content"]] == ["text", "tool_use", "tool_use"]
    assert [block["input"]["command"] for block in message["content"][1:]] == ["pwd", "ls"]


def test_openai_stream_rejects_incomplete_or_error_events() -> None:
    proxy = _proxy_module()
    chunk = {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
    with pytest.raises(ValueError, match="incomplete OpenAI stream"):
        proxy.openai_stream_to_completion(("data: " + json.dumps(chunk) + "\n\n").encode())
    with pytest.raises(ValueError, match="OpenAI stream error"):
        proxy.openai_stream_to_completion(b'data: {"error":{"message":"upstream failed"}}\n\n')


def test_dry_run_uses_dataset_name_for_output_root(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "custom-tasks"
    task_dir = dataset / "sample-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('version = "1.0"')
    (task_dir / "instruction.md").write_text("Do the task")
    assert main(["--dataset", str(dataset), "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["output_root"] == str(Path.cwd() / "output" / "custom-tasks")


def test_start_from_selects_that_task_and_later_tasks(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "custom-tasks"
    for name in ("alpha", "beta", "gamma"):
        task_dir = dataset / name
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text('version = "1.0"')
        (task_dir / "instruction.md").write_text("Do the task")

    assert main(["--dataset", str(dataset), "--start-from", "beta", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["tasks"] == ["beta", "gamma"]
    assert main(["--dataset", str(dataset), "--start-from", "beta", "--limit", "1", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["tasks"] == ["beta"]
    with pytest.raises(SystemExit, match="2"):
        main(["--dataset", str(dataset), "--start-from", "missing", "--dry-run"])


def test_repeated_harbor_trials_receive_distinct_job_names(tmp_path: Path) -> None:
    template = {"agents": [{"kwargs": {}}]}
    configs = [tmp_path / f"trial-{index}.yaml" for index in range(2)]
    for path in configs:
        _write_trial_config(
            template, task="sample-task", phase="baseline", dataset=tmp_path / "dataset",
            bank=tmp_path / "bank.json", jobs_dir=tmp_path / "jobs", config_path=path,
        )
    names = [yaml.safe_load(path.read_text())["job_name"] for path in configs]
    assert names[0] != names[1]
    assert all(name.startswith("skill-sample-task-baseline-") for name in names)
    assert all(yaml.safe_load(path.read_text())["agents"][0]["env"]["LH_HARNESS_MODEL_IO_TRACE"] == "1" for path in configs)


def test_github_dependent_task_uses_build_proxy_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LH_CLAUDE_OFFLINE_BINARY", raising=False)
    template = {"environment": {"type": "docker"}, "agents": [{"kwargs": {}}]}
    configs = []
    for task in ("jq-github-contributor-report", "sample-task"):
        path = tmp_path / f"{task}.yaml"
        _write_trial_config(
            template, task=task, phase="baseline", dataset=tmp_path / "dataset",
            bank=tmp_path / "bank.json", jobs_dir=tmp_path / "jobs", config_path=path,
        )
        configs.append(yaml.safe_load(path.read_text()))
    proxy_paths = configs[0]["environment"]["extra_docker_compose"]
    assert len(proxy_paths) == 1
    assert Path(proxy_paths[0]).name == "docker_build_proxy.yaml"
    assert Path(proxy_paths[0]).is_file()
    assert "extra_docker_compose" not in configs[1]["environment"]


def test_training_config_mounts_cached_claude_code(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "claude"
    binary.write_text('#!/bin/sh\nprintf "2.1.176 (Claude Code)\\n"\n')
    binary.chmod(0o755)
    monkeypatch.setenv("LH_CLAUDE_OFFLINE_BINARY", str(binary))
    config_path = tmp_path / "trial.yaml"
    _write_trial_config(
        {"environment": {"type": "docker"}, "agents": [{"kwargs": {}}]},
        task="sample-task", phase="baseline", dataset=tmp_path / "dataset",
        bank=tmp_path / "bank.json", jobs_dir=tmp_path / "jobs", config_path=config_path,
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["agents"][0]["kwargs"]["offline_claude_binary"] is True
    assert [Path(path).name for path in config["environment"]["extra_docker_compose"]] == [
        "docker_claude_offline.yaml"
    ]


def test_main_detects_cached_claude_code_without_online_install(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "train_dataset" / "terminal-bench-pro"
    task = dataset / "sample-task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"')
    (task / "instruction.md").write_text("Do the task")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"agents": [{"kwargs": {}}]}))
    binary = tmp_path / "output" / dataset.name / "jobs" / "claude_code_cache" / "2.1.176" / "claude"
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\nprintf "2.1.176 (Claude Code)\\n"\n')
    binary.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("LH_CLAUDE_OFFLINE_BINARY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_VERSION", raising=False)

    def fake_train_one(name, **kwargs):
        assert name == "sample-task"
        assert os.environ["LH_CLAUDE_OFFLINE_BINARY"] == str(binary)
        return {"task": name}

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro.train_one", fake_train_one)
    assert main(["--dataset", str(dataset), "--template", str(template), "--task", "sample-task"]) == 0
    assert (tmp_path / "output" / dataset.name / "state" / "skillbank.json").is_file()


def test_cached_claude_lookup_reuses_matching_release_from_another_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    current = output / "terminal-bench-pro"
    other = output / "terminal-bench-pro_test5" / "jobs" / "claude_code_cache" / "2.1.176" / "claude"
    other.parent.mkdir(parents=True)
    other.write_text('#!/bin/sh\nprintf "2.1.176 (Claude Code)\\n"\n')
    other.chmod(0o755)
    assert _find_cached_claude_binary(current, "2.1.176") == other.resolve()
    assert _find_cached_claude_binary(current, "2.1.175") is None
    other.write_text('#!/bin/sh\nprintf "2.1.175 (Claude Code)\\n"\n')
    assert _find_cached_claude_binary(current, "2.1.176") is None


def test_main_forwards_validation_replay_switch(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    task = dataset / "sample-task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"')
    (task / "instruction.md").write_text("Do the task")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"agents": [{"kwargs": {}}]}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.test/v1")
    modes = []

    def fake_train_one(name, **kwargs):
        modes.append(kwargs["validate_replay"])
        return {"task": name}

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro.train_one", fake_train_one)
    args = ["--dataset", str(dataset), "--template", str(template), "--task", "sample-task"]
    assert main([*args, "--no-validation-replay"]) == 0
    assert main(args) == 0
    assert modes == [False, True]


def test_docker_build_proxy_relays_bytes_and_stops(monkeypatch) -> None:
    class EchoHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.sendall(self.request.recv(4))

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), EchoHandler) as upstream:
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{upstream.server_address[1]}")
        monkeypatch.setenv("LH_DOCKER_BUILD_PROXY_PORT", "0")
        monkeypatch.setattr(build_proxy_module, "_docker_bridge_gateway", lambda: "127.0.0.1")
        try:
            with build_proxy_module.docker_build_proxy() as relay_url:
                relay = urlsplit(relay_url)
                with socket.create_connection((relay.hostname, relay.port), timeout=2) as client:
                    client.sendall(b"ping")
                    assert client.recv(4) == b"ping"
        finally:
            upstream.shutdown()
            upstream_thread.join(timeout=2)


@pytest.mark.parametrize("candidate_complete", [True, False])
@pytest.mark.parametrize("block_skill_tmp", [True, False])
def test_training_driver_replays_same_task_and_commits_only_after_success(
    tmp_path: Path, monkeypatch, candidate_complete: bool, block_skill_tmp: bool,
) -> None:
    dataset = tmp_path / "terminal-bench-pro"
    task_dir = dataset / "sample-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('version = "1.0"')
    (task_dir / "instruction.md").write_text("Do the task")
    assert discover_tasks(dataset) == ["sample-task"]
    template = {"agents": [{"kwargs": {"skill_reward_reference_tokens": 1000}}]}
    bank_path = tmp_path / "bank.json"
    if block_skill_tmp:
        (tmp_path / "skill_tmp").write_text("archive unavailable")
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        config = yaml.safe_load(config_path.read_text())
        calls.append(config)
        output = jobs_dir / "trial" / "skill_harness"
        routes_dir = output / "lh_harness" / "role_orchestration"
        routes_dir.mkdir(parents=True)
        report = {"task": "Do the task", "rounds_run": 1, "rounds": [], "abort_reason": "max_rounds_exhausted", "skill_learning": {"task_tokens": 100}}
        if len(calls) == 1:
            report.update({"status": "incomplete", "completion_satisfied": False, "latest_auditor_report": ""})
            (routes_dir / "skill_candidate.json").write_text(json.dumps({"kind": "skill", "candidate": _candidate()}))
            (routes_dir / "skill_routes.json").write_text("[]")
        else:
            report.update({
                "status": "complete" if candidate_complete else "incomplete",
                "completion_satisfied": candidate_complete, "abort_reason": "",
                "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned" if candidate_complete else "",
                "skill_learning": {"task_tokens": 150},
            })
            (routes_dir / "skill_routes.json").write_text(json.dumps([{"selected_skills": [{"skill_id": "evidence_handoff", "version": 1}], "injected_context": "Skill: evidence_handoff@1"}]))
        return report, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    result = train_one(
        "sample-task", template=template, dataset=dataset, bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert result["accepted"] is candidate_complete
    assert [call["datasets"][0]["task_names"] for call in calls] == [["sample-task"], ["sample-task"]]
    assert calls[0]["agents"][0]["kwargs"]["skill_bank_path"] == calls[1]["agents"][0]["kwargs"]["skill_bank_path"]
    assert all(call["agents"][0]["kwargs"]["prompt_language"] == "en" for call in calls)
    assert calls[1]["agents"][0]["kwargs"]["candidate_path"].endswith("skill_candidate.json")
    assert json.loads(bank_path.read_text())["skills"] == []
    assert SkillBank.load(bank_path).skills[0]["status"] == "provisional"
    if block_skill_tmp:
        assert "skill_tmp_path" not in result
    else:
        temporary_skill = tmp_path / "skill_tmp" / "sample-task" / result["run_id"]
        assert result["skill_tmp_path"] == str(temporary_skill / "candidate.json")
        assert json.loads((temporary_skill / "candidate.json").read_text()) == _candidate()
        archived_validation = json.loads((temporary_skill / "validation.json").read_text())
        assert archived_validation["validation"] == result["validation"]
        assert archived_validation["bank_committed"] is False
        assert (temporary_skill / "candidate.json").stat().st_mode & 0o777 == 0o600
    provenance_path = tmp_path / "provenance" / "sample-task.jsonl"
    assert provenance_path.is_file()
    provenance_events = [json.loads(line)["event"] for line in provenance_path.read_text().splitlines()]
    assert "candidate_version" in provenance_events
    assert "decision" in provenance_events
    assert provenance_events.count("bank_snapshot") == 5
    assert provenance_events.count("candidate_version") == 2


@pytest.mark.parametrize("baseline_correct,candidate_correct,accepted", [
    (False, True, True), (False, False, False),
])
def test_training_validation_uses_harbor_score_when_harness_disagrees(
    tmp_path: Path, monkeypatch, baseline_correct: bool, candidate_correct: bool, accepted: bool,
) -> None:
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        is_candidate = len(calls) == 2
        output = jobs_dir / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        (jobs_dir / "result.json").write_text(json.dumps({
            "verifier_result": {"rewards": {"reward": float(
                candidate_correct if is_candidate else baseline_correct
            )}}, "exception_info": None,
        }))
        if is_candidate:
            (routes / "skill_routes.json").write_text(json.dumps([{
                "selected_skills": [{"skill_id": "evidence_handoff", "version": 1}],
                "injected_context": "Skill: evidence_handoff@1",
            }]))
            return {"status": "incomplete", "completion_satisfied": False,
                    "rounds": [], "skill_learning": {"task_tokens": 150}}, output
        (routes / "skill_candidate.json").write_text(json.dumps({
            "kind": "skill", "candidate": _candidate(),
        }))
        return {"status": "incomplete", "completion_satisfied": False,
                "rounds": [], "skill_learning": {"task_tokens": 100}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    bank_path = tmp_path / "bank.json"
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {}}]}, dataset=tmp_path / "dataset",
        bank_path=bank_path, generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert result["accepted"] is accepted
    assert result["validation"]["score_source"] == "harbor_verifier"
    assert result["validation"]["candidate_success"] is candidate_correct
    assert result["baseline_learning_outcome_disagreement"] is False
    assert result["candidate_learning_outcome_disagreement"] is candidate_correct
    assert SkillBank.load(bank_path).skills[0]["success_count"] == int(candidate_correct)


def test_harbor_harness_disagreement_quarantines_pre_verifier_candidate(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        output = jobs_dir / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        (routes / "skill_candidate.json").write_text(json.dumps({
            "kind": "skill", "candidate": _candidate(),
        }))
        (jobs_dir / "result.json").write_text(json.dumps({
            "verifier_result": {"rewards": {"reward": 1.0}}, "exception_info": None,
        }))
        return {"status": "incomplete", "completion_satisfied": False,
                "rounds": [], "skill_learning": {"action": "CREATE_NEW_SKILL", "task_tokens": 100}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    bank_path = tmp_path / "bank.json"
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {}}]}, dataset=tmp_path / "dataset",
        bank_path=bank_path, generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert len(calls) == 1
    assert result["reason"] == "learning_outcome_mismatch"
    assert Path(result["quarantined_candidate_path"]).is_file()
    assert result["baseline_learning_outcome_disagreement"] is True
    assert SkillBank.load(bank_path).skills == []


def test_chinese_training_replay_commits_chinese_skill(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        config = yaml.safe_load(config_path.read_text())
        calls.append(config)
        output = jobs_dir / config["job_name"] / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        if len(calls) == 1:
            (routes / "skill_candidate.json").write_text(json.dumps({"kind": "skill", "candidate": _candidate_zh()}))
            return {"task": "完成任务", "status": "incomplete", "completion_satisfied": False,
                    "rounds": [], "skill_learning": {"task_tokens": 100}}, output
        (routes / "skill_routes.json").write_text(json.dumps([{
            "selected_skills": [{"skill_id": "evidence_handoff", "version": 1}],
            "injected_context": "Skill: evidence_handoff@1",
        }]))
        return {"task": "完成任务", "status": "complete", "completion_satisfied": True,
                "latest_auditor_report": "状态: complete\n完整性: clean\n契约审计: aligned",
                "rounds": [], "skill_learning": {"task_tokens": 150}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    bank_path = tmp_path / "bank.json"
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {"prompt_language": "zh"}}]},
        dataset=tmp_path / "dataset", bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert result["accepted"]
    assert all(call["agents"][0]["kwargs"]["prompt_language"] == "zh" for call in calls)
    assert SkillBank.load(bank_path).skills[0]["content_language"] == "zh"
    provenance_path = tmp_path / "provenance" / "sample-task.jsonl"
    records = [json.loads(line) for line in provenance_path.read_text().splitlines()]
    assert any(item.get("prompt_name") == "skill_builder" and item.get("language") == "zh" for item in records)


@pytest.mark.parametrize("kind", ["new", "revision", "router"])
def test_disabled_validation_replay_keeps_candidates_unverified(tmp_path: Path, monkeypatch, kind: str) -> None:
    bank_path = tmp_path / "bank.json"
    old = {**_candidate(), "status": "validated", "validation": {},
           "selection_count": 4, "success_count": 2}
    if kind != "new":
        SkillBank(bank_path, skills=[old]).save()
    candidate = ({**old, "version": 2, "version_parent": 1, "status": "candidate"}
                 if kind == "revision" else _candidate())
    action = {"new": "CREATE_NEW_SKILL", "revision": "REVISE_SKILL",
              "router": "UPDATE_ROUTER_PROMPT"}[kind]
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        output = jobs_dir / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        proposal = {"kind": "router" if kind == "router" else "skill",
                    "candidate": "Select relevant Skills using routing evidence." if kind == "router" else candidate,
                    "assessment": {"action": action, "target_skill_id": old["skill_id"],
                                   "evidence_refs": ["round_001"]}}
        (routes / "skill_candidate.json").write_text(json.dumps(proposal))
        (routes / "skill_routes.json").write_text(json.dumps([{
            "selected_skills": [{"skill_id": old["skill_id"], "version": 1}] if kind == "revision" else [],
            "injected_context": "Old Skill injected" if kind == "revision" else "",
        }]))
        return {"task": "Do the task", "status": "incomplete", "completion_satisfied": False,
                "rounds": [{"round_index": 1, "plan_text": "Attempted task"}],
                "skill_learning": {"action": action, "task_tokens": 100}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {"prompt_language": "en"}}]},
        dataset=tmp_path / "dataset", bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
        validate_replay=False,
    )
    assert len(calls) == 1
    assert result["replay_validation_enabled"] is False
    assert result["reason"] == "validation_replay_disabled"
    assert result["validation"] == {"status": "skipped", "accepted": False,
                                     "reason": "validation_replay_disabled"}
    assert result["accepted"] is False
    assert not list((tmp_path / "generated").glob("*candidate.yaml"))
    bank = SkillBank.load(bank_path)
    if kind == "router":
        assert bank.router_prompts["en"] != "Select relevant Skills using routing evidence."
        assert [(item["skill_id"], item["status"]) for item in bank.skills] == [(old["skill_id"], "validated")]
    else:
        assert bank.skills[0]["status"] == "provisional"
        assert bank.skills[0]["validation"] == result["validation"]
        assert [item["version"] for item in json.loads(bank_path.read_text())["skills"]] == ([1] if kind == "revision" else [])
        if kind == "revision":
            assert bank.shadowed_skills[0]["version"] == 1
            assert (bank.shadowed_skills[0]["selection_count"], bank.shadowed_skills[0]["success_count"]) == (5, 2)
            assert (bank.skills[0]["selection_count"], bank.skills[0]["success_count"]) == (0, 0)
        archived = tmp_path / "skill_tmp" / "sample-task" / result["run_id"] / "validation.json"
        assert json.loads(archived.read_text())["validation"] == result["validation"]


def test_new_skill_replays_task_before_validation(tmp_path: Path, monkeypatch) -> None:
    candidate = _candidate()
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        output = jobs_dir / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        if len(calls) == 1:
            (routes / "skill_candidate.json").write_text(json.dumps({
                "kind": "skill", "candidate": candidate,
                "assessment": {"action": "CREATE_NEW_SKILL", "evidence_refs": ["round_001"]},
            }))
            return {
                "task": "Do the task", "status": "incomplete", "rounds_run": 1,
                "rounds": [{"round_index": 1, "plan_text": "Tried to complete the task"}],
                "skill_learning": {"action": "CREATE_NEW_SKILL", "task_tokens": 100},
            }, output
        (routes / "skill_routes.json").write_text(json.dumps([{
            "selected_skills": [{"skill_id": candidate["skill_id"], "version": 1}],
            "injected_context": "Skill: evidence_handoff@1",
        }]))
        return {
            "task": "Do the task", "status": "complete", "completion_satisfied": True,
            "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
            "rounds": [{"round_index": 1, "plan_text": "Completed task"}],
            "skill_learning": {"task_tokens": 150},
        }, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    bank_path = tmp_path / "bank.json"
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {"prompt_language": "en"}}]},
        dataset=tmp_path / "dataset", bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert len(calls) == 2
    assert result["accepted"] is True
    assert result["validation"]["reward_delta"] > 0
    assert result["baseline_referenced_skill_names"] == []
    assert result["candidate_referenced_skill_names"] == ["evidence_handoff"]
    assert result["referenced_skill_names"] == ["evidence_handoff"]
    decision = json.loads((tmp_path / "decisions" / "sample-task.json").read_text())
    assert decision["baseline_referenced_skill_names"] == []
    assert decision["candidate_referenced_skill_names"] == ["evidence_handoff"]
    assert decision["referenced_skill_names"] == ["evidence_handoff"]
    attempt = json.loads(next((tmp_path / "decisions" / "attempts" / "sample-task").glob("*.json")).read_text())
    assert attempt["referenced_skill_names"] == ["evidence_handoff"]
    bank = SkillBank.load(bank_path)
    assert bank.skills[0]["status"] == "provisional"
    assert bank.skills[0]["selection_count"] == 1
    assert bank.skills[0]["success_count"] == 1
    assert bank.temporary_path.exists()
    assert bank.eligible("manager", "en", benchmark_id="terminal_bench_2_1", interaction_mode="cli")[0]["skill_id"] == candidate["skill_id"]
    assert json.loads((tmp_path / "skill_tmp" / "sample-task" / result["run_id"] / "validation.json").read_text())["bank_committed"] is False


def test_training_cli_completes_baseline_replay_and_bank_commit(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "sample-dataset"
    task_dir = dataset / "sample-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('version = "1.0"')
    (task_dir / "instruction.md").write_text("Do the task")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"agents": [{"kwargs": {"prompt_language": "en"}}]}))
    candidate = _candidate()
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        config = yaml.safe_load(config_path.read_text())
        calls.append(config)
        output = jobs_dir / config["job_name"] / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        if len(calls) == 1:
            (routes / "skill_candidate.json").write_text(json.dumps({
                "kind": "skill", "candidate": candidate,
                "assessment": {"action": "CREATE_NEW_SKILL", "evidence_refs": ["round_001"]},
            }))
            return {"task": "Do the task", "status": "incomplete",
                    "rounds": [{"round_index": 1, "plan_text": "Attempted task"}],
                    "skill_learning": {"action": "CREATE_NEW_SKILL", "task_tokens": 100}}, output
        replayed = json.loads(Path(config["agents"][0]["kwargs"]["candidate_path"]).read_text())
        assert replayed["candidate"]["skill_id"] == candidate["skill_id"]
        (routes / "skill_routes.json").write_text(json.dumps([{
            "selected_skills": [{"skill_id": candidate["skill_id"], "version": 1}],
            "injected_context": "Skill: evidence_handoff@1",
        }]))
        return {"task": "Do the task", "status": "complete", "completion_satisfied": True,
                "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
                "rounds": [{"round_index": 1, "plan_text": "Completed task"}],
                "skill_learning": {"task_tokens": 150}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://example.invalid")
    monkeypatch.chdir(tmp_path)
    assert main(["--dataset", str(dataset), "--template", str(template), "--task", "sample-task"]) == 0
    assert len(calls) == 2
    state = tmp_path / "output" / dataset.name / "state" / "skillbank.json"
    assert SkillBank.load(state).skills[0]["status"] == "provisional"
    decision = json.loads((tmp_path / "output" / dataset.name / "decisions" / "sample-task.json").read_text())
    assert decision["accepted"] is True
    assert decision["validation"]["reward_delta"] > 0


def test_new_skill_stays_temporary_when_reward_does_not_improve(tmp_path: Path, monkeypatch) -> None:
    candidate = _candidate()
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        output = jobs_dir / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        if len(calls) == 1:
            (routes / "skill_candidate.json").write_text(json.dumps({
                "kind": "skill", "candidate": candidate,
                "assessment": {"action": "CREATE_NEW_SKILL", "evidence_refs": ["round_001"]},
            }))
        else:
            (routes / "skill_routes.json").write_text(json.dumps([{
                "selected_skills": [{"skill_id": candidate["skill_id"], "version": 1}],
                "injected_context": "Skill: evidence_handoff@1",
            }]))
        return {
            "task": "Do the task", "status": "complete", "completion_satisfied": True,
            "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
            "rounds": [{"round_index": 1, "plan_text": "Completed task"}],
            "skill_learning": {"action": "CREATE_NEW_SKILL", "task_tokens": 100 if len(calls) == 1 else 150},
        }, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    bank_path = tmp_path / "bank.json"
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {"prompt_language": "en"}}]},
        dataset=tmp_path / "dataset", bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert len(calls) == 2
    assert result["validation"]["reward_delta"] < 0
    assert result["accepted"] is False
    assert json.loads(bank_path.read_text())["skills"] == []
    assert SkillBank.load(bank_path).skills[0]["status"] == "provisional"


@pytest.mark.parametrize("block_skill_tmp", [False, True])
@pytest.mark.parametrize("all_selected", [False, True])
def test_multiple_new_skills_replay_together(tmp_path: Path, monkeypatch, block_skill_tmp: bool, all_selected: bool) -> None:
    first = _candidate()
    second = {**_candidate(), "skill_id": "second_handoff", "category_id": "second_handoff",
              "problem_mechanism": "A second reusable evidence gap"}
    calls = []
    if block_skill_tmp:
        (tmp_path / "skill_tmp").write_text("archive unavailable")

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        output = jobs_dir / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        if len(calls) == 1:
            (routes / "skill_candidate.json").write_text(json.dumps({
                "kind": "skill", "candidate": [first, second],
                "assessment": {"action": "CREATE_NEW_SKILL", "evidence_refs": ["round_001"]},
            }))
            return {
                "task": "Do the task", "status": "incomplete", "rounds_run": 1,
                "rounds": [{"round_index": 1, "plan_text": "Observed two evidence gaps"}],
                "skill_learning": {"action": "CREATE_NEW_SKILL", "task_tokens": 100},
            }, output
        selected = [first, second] if all_selected else [first]
        (routes / "skill_routes.json").write_text(json.dumps([{
            "selected_skills": [{"skill_id": item["skill_id"], "version": 1}],
            "injected_context": f"Skill: {item['skill_id']}@1",
        } for item in selected]))
        completed = True
        return {
            "task": "Do the task", "status": "complete" if completed else "incomplete",
            "completion_satisfied": completed,
            "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned" if completed else "",
            "rounds": [{"round_index": 1, "plan_text": "Retried task"}],
            "skill_learning": {"task_tokens": 150},
        }, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    bank_path = tmp_path / "bank.json"
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {"prompt_language": "en"}}]},
        dataset=tmp_path / "dataset", bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert len(calls) == 2
    assert result["accepted"] is all_selected
    assert result["baseline_token_usage"]["total_tokens"] == 100
    assert result["candidate_token_usage"]["total_tokens"] == 150
    decision = json.loads((tmp_path / "decisions" / "sample-task.json").read_text())
    assert decision["baseline_token_usage"] == result["baseline_token_usage"]
    assert decision["candidate_token_usage"] == result["candidate_token_usage"]
    proposal_path = Path(yaml.safe_load(calls[1].read_text())["agents"][0]["kwargs"]["candidate_path"])
    assert proposal_path.is_file()
    assert [item["skill_id"] for item in json.loads(proposal_path.read_text())["candidate"]] == [first["skill_id"], second["skill_id"]]
    assert ("skill_tmp_path" in result) is not block_skill_tmp
    assert set(result["candidate_ids"]) == {"evidence_handoff", "second_handoff"}
    assert result["validation"]["accepted"] is all_selected
    assert result["validation"]["selected_candidates"] == {first["skill_id"]: True, second["skill_id"]: all_selected}
    assert {item["skill_id"] for item in json.loads(bank_path.read_text())["skills"]} == set()
    temporary_path = SkillBank.load(bank_path).temporary_path
    temporary_skills = json.loads(temporary_path.read_text())["skills"] if temporary_path.exists() else []
    assert {item["skill_id"] for item in temporary_skills} == {first["skill_id"], second["skill_id"]}
    if not all_selected:
        assert next(item for item in temporary_skills if item["skill_id"] == first["skill_id"])["selection_count"] == 1


@pytest.mark.parametrize("bad_ref", ["round_999", "fake_round_001", "round_0010"])
def test_new_skill_without_matching_execution_evidence_is_not_committed(tmp_path: Path, monkeypatch, bad_ref: str) -> None:
    candidate = _candidate()
    candidate["provenance"] = {"evidence_refs": [bad_ref]}

    def fake_run_trial(config_path, jobs_dir):
        output = jobs_dir / "trial" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        (routes / "skill_candidate.json").write_text(json.dumps({
            "kind": "skill", "candidate": candidate,
            "assessment": {"action": "CREATE_NEW_SKILL", "evidence_refs": ["round_001"]},
        }))
        return {"task": "Do the task", "status": "incomplete",
                "rounds": [{"round_index": 1, "plan_text": "Tried to complete the task"}],
                "skill_learning": {"action": "CREATE_NEW_SKILL"}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    bank_path = tmp_path / "bank.json"
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {"prompt_language": "en"}}]},
        dataset=tmp_path / "dataset", bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert result["accepted"] is False
    assert result["reason"] == "new_skill_missing_execution_evidence"
    assert not bank_path.exists()


@pytest.mark.parametrize("candidate_complete", [True, False])
@pytest.mark.parametrize("old_status", ["validated", "provisional"])
def test_revised_skill_still_requires_successful_replay(tmp_path: Path, monkeypatch, candidate_complete: bool, old_status: str) -> None:
    old = {**_candidate(), "status": old_status, "validation": {},
           "selection_count": 5, "success_count": 2}
    bank_path = tmp_path / "bank.json"
    SkillBank(bank_path, skills=[old]).save()
    revision = {**_candidate(), "version": 2, "version_parent": 1}
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        output = jobs_dir / f"trial-{len(calls)}" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        if len(calls) == 1:
            (routes / "skill_candidate.json").write_text(json.dumps({
                "kind": "skill", "candidate": revision,
                "assessment": {"action": "REVISE_SKILL", "target_skill_id": old["skill_id"]},
                "validation": "pending_replay",
            }))
            return {"task": "Do the task", "status": "incomplete", "completion_satisfied": False,
                    "rounds": [{"round_index": 1, "plan_text": "Attempted task"}],
                    "skill_learning": {"action": "REVISE_SKILL", "task_tokens": 100}}, output
        assert [item["version"] for item in json.loads(bank_path.read_text())["skills"]] == ([1] if old_status == "validated" else [])
        staged_bank = SkillBank.load(bank_path)
        assert staged_bank.skills[0]["version"] == 2
        assert staged_bank.shadowed_skills[0]["version"] == 1
        (routes / "skill_routes.json").write_text(json.dumps([{
            "selected_skills": [{"skill_id": old["skill_id"], "version": 2}],
            "injected_context": "Skill: evidence_handoff@2",
        }]))
        return {"task": "Do the task", "status": "complete" if candidate_complete else "incomplete",
                "completion_satisfied": candidate_complete,
                "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned" if candidate_complete else "",
                "rounds": [{"round_index": 1, "plan_text": "Attempted task"}],
                "skill_learning": {"task_tokens": 150}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {"prompt_language": "en"}}]},
        dataset=tmp_path / "dataset", bank_path=bank_path,
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert len(calls) == 2
    assert result["accepted"] is candidate_complete
    assert result["validation"]["accepted"] is candidate_complete
    bank = SkillBank.load(bank_path)
    assert bank.skills[0]["version"] == 2
    assert bank.skills[0]["update_count"] == 1
    assert (bank.skills[0]["selection_count"], bank.skills[0]["success_count"]) == (
        1, int(candidate_complete))
    assert bank.skills[0]["status"] == "provisional"
    assert [item["version"] for item in json.loads(bank_path.read_text())["skills"]] == ([1] if old_status == "validated" else [])
    assert bank.shadowed_skills[0]["version"] == 1
    assert (bank.shadowed_skills[0]["selection_count"], bank.shadowed_skills[0]["success_count"]) == (5, 2)
    assert len(bank.history) == 0


@pytest.mark.parametrize("select_both", [True, False])
def test_training_driver_replays_mixed_new_and_revised_skills_once(
    tmp_path: Path, monkeypatch, select_both: bool,
) -> None:
    old = {**_candidate(), "status": "validated"}
    revised = {**_candidate(), "version": 2, "version_parent": 1}
    new = {**_candidate(), "skill_id": "missing_capability"}
    bank_path = tmp_path / "bank.json"
    SkillBank(bank_path, skills=[old]).save()
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        output = jobs_dir / f"trial-{len(calls)}" / "skill_harness"
        routes = output / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        if len(calls) == 1:
            (routes / "skill_candidate.json").write_text(json.dumps({
                "kind": "skill", "candidate": [new, revised],
                "assessment": {"action": "CREATE_NEW_SKILL", "evidence_refs": ["round_001"],
                               "revision_targets": [{"target_skill_id": old["skill_id"],
                                                     "failure_mechanism": "Bad handoff",
                                                     "evidence_refs": ["round_001"]}]},
                "validation": "pending_replay",
            }))
            (routes / "skill_routes.json").write_text(json.dumps([{
                "selected_skills": [{"skill_id": old["skill_id"], "version": 1}],
                "injected_context": "Old Skill injected",
            }]))
            return {"task": "Do the task", "status": "incomplete", "completion_satisfied": False,
                    "rounds": [{"round_index": 1, "plan_text": "Attempted task"}],
                    "skill_learning": {"action": "CREATE_NEW_SKILL", "task_tokens": 200}}, output
        selected = [{"skill_id": new["skill_id"], "version": 1}]
        if select_both:
            selected.append({"skill_id": old["skill_id"], "version": 2})
        (routes / "skill_routes.json").write_text(json.dumps([{
            "selected_skills": selected, "injected_context": "Candidate Skills injected",
        }]))
        return {"task": "Do the task", "status": "complete", "completion_satisfied": True,
                "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
                "rounds": [{"round_index": 1, "plan_text": "Completed task"}],
                "skill_learning": {"task_tokens": 150}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    result = train_one("sample-task", template={"agents": [{"kwargs": {"prompt_language": "en"}}]},
                       dataset=tmp_path / "dataset", bank_path=bank_path,
                       generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
                       decisions_dir=tmp_path / "decisions", reference_tokens=1000)
    assert len(calls) == 2
    assert result["accepted"] is select_both
    assert set(result["candidate_ids"]) == {old["skill_id"], new["skill_id"]}
    bank = SkillBank.load(bank_path)
    assert {item["skill_id"] for item in bank.skills} == {old["skill_id"], new["skill_id"]}
    assert all(item["status"] == "provisional" for item in bank.skills)


def test_harbor_exception_surfaces_even_with_zero_exit_and_stale_report(tmp_path: Path, monkeypatch) -> None:
    jobs_dir = tmp_path / "jobs"
    old_trial = jobs_dir / "old"
    old_trial.mkdir(parents=True)
    (old_trial / "training_report.json").write_text(json.dumps({"report": {"status": "complete"}}))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.example.test/v1/v1")
    config_path = tmp_path / "trial.yaml"
    config_path.write_text("job_name: new\n")

    def fake_run(*args, **kwargs):
        assert kwargs["env"]["ANTHROPIC_BASE_URL"] == "https://api.example.test"
        new_trial = jobs_dir / "new"
        new_trial.mkdir()
        (new_trial / "exception.txt").write_text("Docker build failed: proxyconnect 127.0.0.1:17897")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="proxyconnect 127.0.0.1:17897"):
        _run_trial(config_path, jobs_dir)


def test_harbor_trial_receives_temporary_build_proxy(tmp_path: Path, monkeypatch) -> None:
    jobs_dir = tmp_path / "jobs"
    config_path = tmp_path / "trial.yaml"
    config_path.write_text(yaml.safe_dump({
        "job_name": "new",
        "environment": {"extra_docker_compose": ["/tmp/docker_build_proxy.yaml"]},
    }))

    @contextlib.contextmanager
    def fake_proxy():
        yield "http://172.17.0.1:17897"

    monkeypatch.setattr(build_proxy_module, "docker_build_proxy", fake_proxy)

    def fake_run(*args, **kwargs):
        assert kwargs["env"]["LH_DOCKER_BUILD_PROXY"] == "http://172.17.0.1:17897"
        report_dir = jobs_dir / "new" / "trial"
        report_dir.mkdir(parents=True)
        (report_dir / "training_report.json").write_text(json.dumps({"report": {"status": "complete"}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro.subprocess.run", fake_run)
    report, _ = _run_trial(config_path, jobs_dir)
    assert report["status"] == "complete"


def test_provider_failure_is_retried_and_not_recorded_as_no_candidate(tmp_path: Path, monkeypatch) -> None:
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    decision_path = decisions / "sample-task.json"
    decision_path.write_text(json.dumps({"baseline_status": "failed", "reason": "no candidate proposed"}))
    calls = []

    def fake_run(config_path, jobs_dir):
        calls.append(config_path)
        return {
            "status": "failed", "abort_reason": "provider_provider_error",
            "failure_reason": "selected model is unavailable",
        }, jobs_dir / "trial" / "skill_harness"

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run)
    with pytest.raises(RuntimeError, match="selected model is unavailable"):
        train_one(
            "sample-task", template={"agents": [{"kwargs": {}}]},
            dataset=tmp_path / "dataset", bank_path=tmp_path / "bank.json",
            generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
            decisions_dir=decisions, reference_tokens=1000,
        )
    assert len(calls) == 1
    assert json.loads(decision_path.read_text())["reason"] == "no candidate proposed"
    attempts = list((decisions / "attempts" / "sample-task").glob("*.json"))
    assert len(attempts) == 1
    failed_attempt = json.loads(attempts[0].read_text())
    assert failed_attempt["reason"] == "infrastructure_error"
    assert failed_attempt["phase"] == "baseline"
    traces = [tmp_path / "provenance" / "sample-task.jsonl"]
    assert traces[0].is_file()
    assert any(
        json.loads(line).get("event") == "trial_failure"
        for line in traces[0].read_text().splitlines()
    )


def test_provider_failure_includes_proxy_response(tmp_path: Path) -> None:
    output_dir = tmp_path / "skill_harness"
    proxy_log = output_dir / "episodes" / "orchestrator" / "ep001" / "request_proxy_requests.jsonl"
    proxy_log.parent.mkdir(parents=True)
    proxy_log.write_text(json.dumps({
        "event": "upstream_http_error", "status": 404,
        "body_head": "Invalid URL (POST /v1/v1/messages)",
    }) + "\n")
    with pytest.raises(RuntimeError, match=r"Provider HTTP 404: Invalid URL \(POST /v1/v1/messages\)"):
        _require_valid_trial_report({"status": "failed", "failure_reason": "model error"}, output_dir, "baseline")


def test_plain_text_tool_calls_are_not_recorded_as_no_candidate(tmp_path: Path) -> None:
    output_dir = tmp_path / "skill_harness"
    stream = output_dir / "episodes" / "cli_task" / "ep001" / "claude_stream.jsonl"
    stream.parent.mkdir(parents=True)
    stream.write_text(json.dumps({
        "type": "assistant", "message": {"content": [{"type": "text", "text": "<｜｜DSML｜｜ calls>"}]},
    }) + "\n")
    report = {
        "status": "incomplete",
        "rounds": [{"executor_output": '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="bash">'}],
    }
    with pytest.raises(RuntimeError, match="plain text rather than Anthropic tool_use"):
        _require_valid_trial_report(report, output_dir, "baseline")


def test_view_only_provenance_keeps_skill_prompt_history_and_model_io(tmp_path: Path) -> None:
    path = tmp_path / "jobs" / "provenance" / "sample-task.jsonl"
    writer = ProvenanceWriter(path, task="sample-task", run_id="run-1")
    previous = _candidate()
    current = {**_candidate(), "version": 2, "version_parent": 1, "status": "validated"}
    bank = SkillBank(
        tmp_path / "state" / "skillbank.json",
        skills=[current], router_prompts={"en": "current router prompt"},
        history=[{"kind": "skill", "skill": previous}, {"kind": "router", "language": "en", "prompt": "previous router prompt"}],
    )
    writer.static_prompts()
    writer.bank_snapshot("before_baseline", bank)
    proposal_path = tmp_path / "proposal.json"
    writer.candidate(
        "proposed", {"kind": "skill", "candidate": current, "assessment": {"action": "REVISE_SKILL"}},
        source_path=proposal_path, previous_skill=previous,
    )
    writer.candidate(
        "proposed", {"kind": "router", "candidate": "new router prompt"},
        source_path=proposal_path, previous_router_prompt="current router prompt",
    )
    config_path = tmp_path / "trial.yaml"
    config_path.write_text("job_name: job-1\n")
    model_io_path = tmp_path / "jobs" / "job-1" / "trial" / "episodes" / "manager" / "ep001" / "model_io.jsonl"
    model_io_path.parent.mkdir(parents=True)
    model_io_path.write_text("\n".join(json.dumps(item) for item in (
        {"event": "model_input", "trace_id": "call-1", "protocol": "openai_chat_completions",
         "payload": {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "first prompt"}]}},
        {"event": "model_output", "trace_id": "call-1", "payload": {"choices": [{"message": {"content": "response"}}]}},
        {"event": "model_input", "trace_id": "call-2", "protocol": "openai_chat_completions",
         "payload": {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "revised prompt"}]}},
    )) + "\n")
    writer.ingest_model_io(config_path, tmp_path / "jobs", phase="baseline")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(item["event"] == "skill_version" and item["source"] == "bank_history" and item["version"] == 1 for item in records)
    assert any(item["event"] == "skill_version" and item["source"] == "active_bank" and item["version"] == 2 for item in records)
    assert any(item["event"] == "prompt_version" and item["source"] == "bank_history" and item["content"] == "previous router prompt" for item in records)
    assert any(item["event"] == "prompt_version" and item["source"] == "code_template" for item in records)
    assert any(item["event"] == "prompt_version" and item.get("prompt_name") == "skill_experience_extraction"
               for item in records)
    assert any(item["event"] == "prompt_version" and item.get("prompt_name") == "skill_revision_preservation_review"
               for item in records)
    assert any(item["event"] == "candidate_version" and item["kind"] == "skill" for item in records)
    old_skill = next(item for item in records if item["event"] == "skill_version" and item["source"] == "bank_history")
    revised_skill = next(item for item in records if item["event"] == "candidate_version" and item["kind"] == "skill")
    assert revised_skill["previous_version_id"] == old_skill["version_id"]
    assert any(item["event"] == "prompt_version" and item["source"] == "router_candidate" and item["previous_version_id"] for item in records)
    assert any(item["event"] == "model_io" and item["record"]["payload"]["model"] == "deepseek-v4.1-flash" for item in records)
    prompts = [item for item in records if item["event"] == "runtime_prompt_version"]
    assert len(prompts) == 2
    assert prompts[0]["content"]["messages"][0]["content"] == "first prompt"
    assert prompts[1]["previous_version_id"] == prompts[0]["version_id"]
    assert prompts[1]["trace_id"] == "call-2"
    assert (prompts[1]["role"], prompts[1]["episode"]) == ("manager", "ep001")
    assert path.stat().st_mode & 0o777 == 0o600
    assert not bank.path.exists()


def test_view_only_trace_write_failure_does_not_stop_training(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not_a_directory"
    blocked_parent.write_text("keep")
    writer = ProvenanceWriter(blocked_parent / "trace.jsonl", task="task", run_id="run")
    writer.append("run_start")
    writer.append("unserializable", value={"set"})
    assert blocked_parent.read_text() == "keep"

    proxy = _proxy_module()
    config = proxy.ProxyConfig(
        upstream="https://example.test", overrides={}, dashscope_wait_timeout=None,
        log_file=None, requests_log_file=None, model_io_file=blocked_parent / "model_io.jsonl",
    )
    config.record_model_io({"event": "model_input", "payload": {"model": "deepseek-v4.1-flash"}})
    assert blocked_parent.read_text() == "keep"

    writer_ok = ProvenanceWriter(tmp_path / "valid" / "trace.jsonl", task="task", run_id="run")
    writer_ok.append("unserializable", value={"set"})
    config_ok = proxy.ProxyConfig(
        upstream="https://example.test", overrides={}, dashscope_wait_timeout=None,
        log_file=None, requests_log_file=None, model_io_file=tmp_path / "valid" / "model_io.jsonl",
    )
    config_ok.record_model_io({"event": "model_input", "payload": {"set"}})


def test_training_reason_distinguishes_diagnosis_from_default_no_change() -> None:
    assert _learning_reason({}) == "learning_report_missing"
    assert _learning_reason({"action": "NO_CHANGE", "diagnosis_attempted": True}) == "diagnosis_no_change"
    assert _learning_reason({"action": "NO_CHANGE"}) == "legacy_unverified_no_change"
    assert _learning_reason({"action": "INSUFFICIENT_EVIDENCE", "diagnosis_attempted": True}) == "diagnosis_insufficient_evidence"
    assert _learning_reason({"skip_reason": "no_analyzable_execution_trajectory"}) == "diagnosis_skipped"
    assert _learning_reason({"action": "CREATE_NEW_SKILL", "learning_error": "invalid Skill"}) == "skill_learning_error"


def test_force_rerun_preserves_old_decision_and_uses_new_attempt(tmp_path: Path, monkeypatch) -> None:
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    old_decision = {"task": "sample-task", "baseline_status": "incomplete", "reason": "no candidate proposed"}
    decision_path = decisions / "sample-task.json"
    decision_path.write_text(json.dumps(old_decision))
    calls = []

    def fake_run(config_path, jobs_dir):
        config = yaml.safe_load(config_path.read_text())
        calls.append((config_path, config["job_name"]))
        output = jobs_dir / config["job_name"] / "trial" / "skill_harness"
        output.mkdir(parents=True)
        return {
            "status": "incomplete", "abort_reason": "needs_human_input",
            "rounds": [{"round_index": 1, "plan_text": "Next: ask"}],
            "skill_learning": {"diagnosis_attempted": True, "action": "NO_CHANGE",
                               "assessment_reason": "Task definition is ambiguous"},
        }, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run)
    args = dict(
        task="sample-task", template={"agents": [{"kwargs": {}}]},
        dataset=tmp_path / "dataset", bank_path=tmp_path / "bank.json",
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=decisions, reference_tokens=1000,
    )
    cached = train_one(**args)
    assert cached["execution"] == "reused_prior_decision"
    assert not calls
    first = train_one(**args, force=True)
    second = train_one(**args, force=True)
    assert len(calls) == 2 and calls[0][1] != calls[1][1]
    assert calls[0][0] != calls[1][0]
    assert first["run_id"] != second["run_id"]
    provenance_path = tmp_path / "provenance" / "sample-task.jsonl"
    records = [json.loads(line) for line in provenance_path.read_text().splitlines()]
    assert {record["run_id"] for record in records} == {first["run_id"], second["run_id"]}
    assert [record["run_id"] for record in records if record["event"] == "run_start"] == [first["run_id"], second["run_id"]]
    assert [record["run_id"] for record in records if record["event"] == "decision"] == [first["run_id"], second["run_id"]]
    assert list((tmp_path / "provenance").glob("*.jsonl")) == [provenance_path]
    assert first["reason"] == second["reason"] == "diagnosis_no_change"
    assert first["skill_learning"]["assessment_reason"] == "Task definition is ambiguous"
    assert len(list((decisions / "attempts" / "sample-task").glob("*.json"))) == 2
    archives = [json.loads(path.read_text()) for path in (decisions / "history" / "sample-task").glob("*.json")]
    assert old_decision in archives
    assert json.loads(decision_path.read_text())["run_id"] == second["run_id"]


def test_switching_replay_mode_does_not_reuse_old_decision(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run_trial(config_path, jobs_dir):
        calls.append(config_path)
        output = jobs_dir / f"trial-{len(calls)}" / "skill_harness"
        output.mkdir(parents=True)
        return {"task": "task", "status": "incomplete", "rounds": [
            {"round_index": 1, "plan_text": "Attempted task"}],
            "skill_learning": {"diagnosis_attempted": True, "action": "NO_CHANGE",
                               "assessment_reason": "No supported update"}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run_trial)
    args = dict(task="sample-task", template={"agents": [{"kwargs": {}}]},
                dataset=tmp_path / "dataset", bank_path=tmp_path / "bank.json",
                generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
                decisions_dir=tmp_path / "decisions", reference_tokens=1000)
    first = train_one(**args, validate_replay=False)
    cached = train_one(**args, validate_replay=False)
    switched = train_one(**args, validate_replay=True)
    assert len(calls) == 2
    assert cached["execution"] == "reused_prior_decision"
    assert first["run_id"] != switched["run_id"]
    assert switched["replay_validation_enabled"] is True
    assert len(list((tmp_path / "decisions" / "history" / "sample-task").glob("*.json"))) == 1


def test_old_evidence_validation_failure_is_retried_without_force(tmp_path: Path, monkeypatch) -> None:
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    decision_path = decisions / "sample-task.json"
    decision_path.write_text(json.dumps({
        "task": "sample-task", "baseline_status": "incomplete", "reason": "skill_learning_error",
        "skill_learning": {"error_stage": "evidence_validation",
                           "learning_error": "update cites a round absent from the recorded trajectory"},
    }))
    calls = []

    def fake_run(config_path, jobs_dir):
        calls.append(config_path)
        config = yaml.safe_load(config_path.read_text())
        output = jobs_dir / config["job_name"] / "trial" / "skill_harness"
        output.mkdir(parents=True)
        return {"status": "incomplete", "rounds": [{"round_index": 1, "plan_text": "Attempted task"}],
                "skill_learning": {"action": "NO_CHANGE", "diagnosis_attempted": True}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run)
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {}}]},
        dataset=tmp_path / "dataset", bank_path=tmp_path / "bank.json",
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=decisions, reference_tokens=1000,
    )
    assert len(calls) == 1
    assert result["execution"] == "fresh_run"
    assert result["reason"] == "diagnosis_no_change"


def test_invalid_candidate_artifact_is_explained_without_replay(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(config_path, jobs_dir):
        calls.append(config_path)
        config = yaml.safe_load(config_path.read_text())
        output = jobs_dir / config["job_name"] / "trial" / "skill_harness"
        candidate = output / "lh_harness" / "role_orchestration" / "skill_candidate.json"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("{invalid json")
        return {"status": "incomplete", "rounds": [{"plan_text": "Next: cli"}],
                "skill_learning": {"diagnosis_attempted": True, "action": "CREATE_NEW_SKILL"}}, output

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro._run_trial", fake_run)
    result = train_one(
        "sample-task", template={"agents": [{"kwargs": {}}]},
        dataset=tmp_path / "dataset", bank_path=tmp_path / "bank.json",
        generated_dir=tmp_path / "generated", jobs_root=tmp_path / "jobs",
        decisions_dir=tmp_path / "decisions", reference_tokens=1000,
    )
    assert result["reason"] == "candidate_artifact_invalid"
    assert result["error_stage"] == "candidate_artifact"
    assert len(calls) == 1
