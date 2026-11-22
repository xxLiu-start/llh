from __future__ import annotations

import asyncio
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lh_harness.config import ProjectConfigError, load_run_defaults
from lh_harness.local_gateway import policy_payload
from lh_harness.local_models import EndpointError, base_url, check_endpoint, stream_events
from lh_harness.local_pair import check_parity, require_evidence
from lh_harness.skill_experiment import ExperimentSession
from lh_harness.token_usage import aggregate, from_trajectory
from lh_harness.types import EpisodeBudget, EpisodeResult


@pytest.fixture
def endpoint():
    state = {"usage": True, "tool": True, "terminal": True, "requests": [], "redirect": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if state["redirect"]:
                self.send_response(302)
                self.send_header("Location", "https://cloud.invalid/v1/models")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "local", "max_model_len": 32768, "owned_by": "vllm"}], "version": "test-only"}).encode())

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(value)
            if value["model"] != "local":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [{"type": "response.created"}]
            if state["tool"]:
                events += [{"type": "response.output_item.done", "item": {
                    "type": "function_call", "name": "endpoint_probe", "arguments": '{"value":"OK"}'}}]
            if state["terminal"]:
                events += [{"type": "response.completed", "response": {"usage": {"input_tokens": 9, "output_tokens": 4} if state["usage"] else None}}]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_active_probe_checks_tools_stream_and_real_usage(endpoint):
    url, state = endpoint
    result = check_endpoint(url, "codex", "local")
    assert result["usage"]["status"] == "reported"
    assert result["usage"]["total_tokens"] == 13
    assert result["invalid_model_rejected"]
    assert state["requests"][0]["tools"][0]["type"] == "function"


@pytest.mark.parametrize("field", ["tool", "terminal"])
def test_incomplete_protocol_is_not_text_generation_success(endpoint, field):
    url, state = endpoint
    state[field] = False
    with pytest.raises(EndpointError, match="stream/tool-call"):
        check_endpoint(url, "codex", "local")


def test_missing_usage_is_unavailable(endpoint):
    url, state = endpoint
    state["usage"] = False
    result = check_endpoint(url, "codex", "local")
    assert result["usage"]["status"] == "unavailable"
    assert result["usage"]["input_tokens"] is None


def test_mismatched_model_is_not_substituted(endpoint):
    url, state = endpoint
    with pytest.raises(EndpointError, match="no substitution"):
        check_endpoint(url, "codex", "cloud-default")
    assert state["requests"] == []


def test_redirect_to_cloud_is_rejected(endpoint):
    url, state = endpoint
    state["redirect"] = True
    with pytest.raises(EndpointError, match="HTTP 302"):
        check_endpoint(url, "codex", "local")


def test_unavailable_endpoint_stops_before_benchmark(monkeypatch, tmp_path):
    import lh_harness.cli as cli
    def fail(*args, **kwargs):
        raise EndpointError("endpoint unavailable; no fallback")
    monkeypatch.setattr("lh_harness.local_experiment.check_endpoint", fail)
    config = tmp_path / "local.toml"
    config.write_text('[run]\nagent="codex"\nmodel="local"\nbase_url="http://127.0.0.1:1/v1"\n')
    root = tmp_path / "runs"
    assert cli.main(["run", "--config", str(config), "--task", "never execute", "--runs-root", str(root)]) == 2
    assert not root.exists()


def test_protocol_mismatch_and_full_route_are_explicit_errors():
    with pytest.raises(EndpointError, match="requires responses"):
        check_endpoint("http://127.0.0.1:1/v1", "codex", "local", {"protocol": "chat_completions"})
    with pytest.raises(EndpointError, match="API root"):
        base_url("http://127.0.0.1:1/v1/chat/completions")
    with pytest.raises(EndpointError, match="provider/model"):
        check_endpoint("http://127.0.0.1:1/v1", "opencode", "local")


def test_sse_parser_does_not_accept_truncated_or_malformed_frames():
    with pytest.raises(EndpointError, match="incomplete"):
        list(stream_events(io.BytesIO(b'data: {}\n')))
    with pytest.raises(EndpointError, match="malformed"):
        list(stream_events(io.BytesIO(b'data: not-json\n\n')))


def test_usage_no_echo_double_count_and_no_fake_zero():
    raw = '\n'.join(json.dumps(e) for e in [
        {"type": "assistant", "message": {"id": "one", "usage": {"input_tokens": 10, "output_tokens": 2}}},
        {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 4}}])
    value = from_trajectory(raw)
    assert value["input_tokens"] == 14 and value["total_tokens"] == 16
    missing = from_trajectory('{"type":"turn.completed"}')
    assert missing["status"] == "unavailable" and missing["input_tokens"] is None
    assert aggregate([value, missing])["total_tokens"] is None


def test_opencode_step_usage_is_normalized():
    raw = json.dumps({"type": "step_finish", "part": {"tokens": {"input": 14, "output": 7}}})
    assert from_trajectory(raw)["total_tokens"] == 21


def test_proxy_enforces_identical_sampling_without_model_switch():
    profile = {"max_output_tokens": 8192, "temperature": 0.6, "top_p": 0.95,
               "reasoning": {"enable_thinking": False}, "tool_calling": {"parallel_tool_calls": False}}
    value = policy_payload("/responses", {"model": "local", "max_output_tokens": 9}, profile, {"local"})
    assert value["model"] == "local" and value["max_output_tokens"] == 8192
    assert value["temperature"] == 0.6 and value["chat_template_kwargs"]["enable_thinking"] is False
    with pytest.raises(EndpointError, match="explicitly permitted"):
        policy_payload("/responses", {"model": "another-model"}, profile, {"local"})


def test_all_five_configs_share_model_backend_endpoint_and_inference():
    root = Path(__file__).parents[1] / "experiments/skill_evolution/configs"
    baseline = load_run_defaults(root / "baseline.local.toml")
    for name, role in (("manager_only", "manager"), ("executor_only", "executor"), ("auditor_only", "auditor"), ("all_roles", None)):
        variant = load_run_defaults(root / f"{name}.local.toml")
        check_parity(baseline, variant)
        if role:
            assert variant["skills_config"]["training"]["train_roles"] == [role]
    wrong = {**variant, "model": "different"}
    with pytest.raises(EndpointError, match="outside Skills"):
        check_parity(baseline, wrong)
    with pytest.raises(EndpointError, match="metadata is incomplete"):
        require_evidence({"manager": {}})


def test_secrets_cannot_be_saved_in_config(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[run]\napi_key="do-not-save-this"\n')
    with pytest.raises(ProjectConfigError, match="unknown"):
        load_run_defaults(path)


def test_baseline_prompt_is_byte_identical_and_skill_roles_are_gated(tmp_path):
    class Adapter:
        def __init__(self): self.prompts = []
        async def run_episode(self, prompt, *args, **kwargs):
            self.prompts.append(prompt)
            return EpisodeResult(status="done", metadata={"usage": {"status": "reported", "input_tokens": 4, "output_tokens": 2, "total_tokens": 6}})
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps([{"role": "executor", "status": "curated", "guidance": "artifact verify with exact read"}]))
    baseline = ExperimentSession({"enabled": False}, str(tmp_path / "baseline"))
    original = Adapter()
    asyncio.run(baseline.wrap("manager", original).run_episode("original prompt\n", None, EpisodeBudget()))
    assert original.prompts == ["original prompt\n"]
    settings = {"enabled": True, "bank": str(bank), "usage": {"enabled": True, "use_roles": ["executor"]}}
    enhanced = ExperimentSession(settings, str(tmp_path / "enhanced"))
    agent = Adapter()
    asyncio.run(enhanced.wrap("cli_executor", agent).run_episode("verify artifact", None, EpisodeBudget()))
    asyncio.run(enhanced.wrap("gui_auditor", agent).run_episode("verify artifact", None, EpisodeBudget()))
    assert "Advisory Skills" in agent.prompts[0] and agent.prompts[1] == "verify artifact"
    summary = json.loads((tmp_path / "enhanced/token_usage.json").read_text())["categories"]
    assert summary["skill_injection"]["status"] == "unavailable"
    assert summary["executor"]["input_tokens"] == 4


def test_training_uses_only_selected_role_and_accounts_both_stages(tmp_path):
    evidence = "The independent byte check confirmed exact file contents."
    class Trainer:
        def __init__(self): self.calls = 0
        async def run_episode(self, prompt, *args, **kwargs):
            self.calls += 1
            text = {"guidance": "Verify persisted bytes in a separate read-only check.", "evidence_quote": evidence} if self.calls == 1 else {"accept": True}
            return EpisodeResult(status="done", metadata={"assistant_visible_output": json.dumps(text),
                "usage": {"status": "reported", "input_tokens": 10, "output_tokens": 3, "total_tokens": 13}})
    session = ExperimentSession({"enabled": True, "bank": str(tmp_path / "bank.json"),
        "training": {"enabled": True, "train_roles": ["executor"]}}, str(tmp_path / "logs"))
    session.records.append({"group": "executor", "status": "done", "output": evidence,
                            "usage": {"status": "unavailable"}})
    trainer = Trainer()
    report = {"status": "complete", "completion_satisfied": True}
    asyncio.run(session.finish(report, {"executor": trainer}, None, EpisodeBudget()))
    assert report == {"status": "complete", "completion_satisfied": True}
    assert trainer.calls == 2 and session.bank[0]["role"] == "executor"
    usage = json.loads((tmp_path / "logs/token_usage.json").read_text())["categories"]
    assert usage["skill_proposer"]["total_tokens"] == 13
    assert usage["skill_curator"]["total_tokens"] == 13
    assert usage["executor"]["status"] == "unavailable"


def test_skill_injection_offline_token_counts_remain_estimated(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "verify": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps([{"role": "executor", "status": "curated", "guidance": "verify persisted file"}]))
    session = ExperimentSession({"enabled": True, "bank": str(bank),
        "usage": {"enabled": True, "use_roles": ["executor"]}}, str(tmp_path / "logs"), str(tmp_path))
    class Adapter:
        async def run_episode(self, *args, **kwargs): return EpisodeResult(status="done")
    asyncio.run(session.wrap("cli_executor", Adapter()).run_episode("verify file", None, EpisodeBudget()))
    value = json.loads((tmp_path / "logs/token_usage.json").read_text())["categories"]["skill_injection"]
    assert value["status"] == "estimated" and value["tokens"] > 0
    assert value["included_in_role_input_tokens"] is True


@pytest.mark.parametrize("role", ["manager", "gui_auditor", "cli_auditor", "auditor_format_repair"])
def test_codex_local_read_only_roles_keep_the_sandbox(role, tmp_path):
    from lh_harness.cli import _build_agent
    adapter = _build_agent("codex", role=role, model="local", base_url="http://127.0.0.1:8010/v1", api_key=None,
                           workspace_path=str(tmp_path), prompt_dir=str(tmp_path / "prompts"), context_length=32768)
    assert "--sandbox read-only" in adapter.command_template
    assert "model_context_window=32768" in adapter.command_template
    assert "--dangerously-bypass-approvals-and-sandbox" not in adapter.command_template


def test_skill_json_repair_reuses_adapter_and_accounts_failed_format(tmp_path):
    class Adapter:
        def __init__(self): self.calls = 0
        async def run_episode(self, *args, **kwargs):
            self.calls += 1
            text = 'Here is a proposal.' if self.calls == 1 else '{"guidance":"Read persisted bytes independently.","evidence_quote":"verified evidence"}'
            return EpisodeResult(status="done", metadata={"assistant_visible_output": text,
                "usage": {"status": "reported", "input_tokens": 7, "output_tokens": 2, "total_tokens": 9}})
    session = ExperimentSession({"enabled": False}, str(tmp_path))
    adapter = Adapter()
    result = asyncio.run(session._call("skill_proposer", adapter, "Return a proposal", None, EpisodeBudget()))
    assert adapter.calls == 2 and result["evidence_quote"] == "verified evidence"
    counts = json.loads((tmp_path / "token_usage.json").read_text())["categories"]["skill_proposer"]
    assert counts["total_tokens"] == 18 and counts["calls"] == 2
    assert (tmp_path / "skill_proposer_000.output.json").exists()
    assert json.loads((tmp_path / "skill_proposer_001.input.json").read_text())["format_repair"] is True


@pytest.mark.parametrize("variable,header", [("ANTHROPIC_AUTH_TOKEN", "Authorization"), ("ANTHROPIC_API_KEY", "X-api-key")])
def test_anthropic_credential_environment_preserves_authentication_protocol(monkeypatch, variable, header):
    from lh_harness.local_models import EndpointClient
    from types import SimpleNamespace
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(variable, "test-only-token")
    requests = []
    client = EndpointClient("http://127.0.0.1:8010/v1", "claude_code")
    client.opener = SimpleNamespace(open=lambda request, **kwargs: requests.append(request))
    client.request("/messages", {})
    expected = "Bearer test-only-token" if header == "Authorization" else "test-only-token"
    assert requests[0].get_header(header) == expected


def test_non_native_read_only_guard_rejects_successful_auditor_mutation(tmp_path):
    from lh_harness.adapters.read_only import ReadOnlyGuardAdapter
    original = tmp_path / "original.txt"
    original.write_text("original bytes")
    class Adapter:
        workspace_path = str(tmp_path)
        async def run_episode(self, *args, **kwargs):
            original.write_text("unexpected auditor change")
            return EpisodeResult(status="done", metadata={"assistant_visible_output": "Status: complete"})
    result = asyncio.run(ReadOnlyGuardAdapter(Adapter()).run_episode("audit", None, EpisodeBudget()))
    assert result.status == "error"
    assert result.metadata["verifier_workspace_mutation_detected"] is True
    assert result.metadata["verifier_workspace_restore_on_mutation"] is False


def test_known_buffered_backend_is_rejected_before_local_benchmark():
    import argparse
    from lh_harness.local_experiment import preflight
    args = argparse.Namespace(agent="deepseek_harness", model="local", base_url="http://127.0.0.1:1/v1",
                              local_model={"enabled": True}, reasoning_effort=None)
    with pytest.raises(EndpointError, match="real-time streaming"):
        preflight(args)
