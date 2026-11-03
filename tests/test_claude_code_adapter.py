from __future__ import annotations

import ast
import json
import shlex
from pathlib import Path

import pytest

from lh_harness.adapters.claude_code import ClaudeCodeAdapter
from lh_harness.adapters.claude_permissions import ClaudeRole


ROLES: tuple[ClaudeRole, ...] = (
    "manager",
    "cli_executor",
    "gui_executor",
    "cli_auditor",
    "gui_auditor",
    "auditor_format_repair",
    "final_response",
)


def _tokens(adapter: ClaudeCodeAdapter) -> list[str]:
    return shlex.split(adapter.command_template.replace("< {prompt_path}", ""))


def _adapter(role: ClaudeRole, **overrides: object) -> ClaudeCodeAdapter:
    kwargs: dict[str, object] = {
        "model": "claude-sonnet-5",
        "role": role,
        "workspace_path": "/tmp/ws",
        "prompt_dir": "/tmp/prompts",
    }
    kwargs.update(overrides)
    return ClaudeCodeAdapter(**kwargs)  # type: ignore[arg-type]


def _extract_claude_visible_output(stream: str) -> str:
    # The Harbor backend imports Harbor at module import time. Compile its two
    # dependency-free stream helpers so this regression runs without Docker or
    # the Harbor package in the host test environment.
    source = Path(__file__).resolve().parents[1] / "eval/TB-harness/Harness/src/harbor_backends/claude_code.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
        "_extract_claude_visible_output", "_visible_text_from_content",
    }]
    namespace = {"json": json}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_extract_claude_visible_output"](stream)


@pytest.mark.parametrize("role", ROLES)
def test_every_role_ignores_the_operator_mcp_registries(role: ClaudeRole) -> None:
    """No role may inherit user- or project-scope MCP servers.

    The CLI loads `~/.claude.json` and a workspace `.mcp.json` unless it is told
    not to, so an operator's unrelated servers otherwise reach every episode.
    """
    assert "--strict-mcp-config" in _tokens(_adapter(role))


def test_a_configured_mcp_config_is_still_the_only_source(tmp_path: Path) -> None:
    """An explicit plugin config is loaded, and remains the sole source."""
    config = tmp_path / "computer-use.mcp.json"
    config.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

    tokens = _tokens(_adapter("cli_executor", mcp_config=str(config)))

    assert tokens[tokens.index("--mcp-config") + 1] == str(config.resolve())
    assert "--strict-mcp-config" in tokens


def test_the_environment_mcp_config_is_scoped_the_same_way(
    monkeypatch, tmp_path: Path
) -> None:
    """`LH_HARNESS_CLAUDECODE_MCP_CONFIG` widens the allow-list, not the scope."""
    config = tmp_path / "env.mcp.json"
    config.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    monkeypatch.setenv("LH_HARNESS_CLAUDECODE_MCP_CONFIG", str(config))

    tokens = _tokens(_adapter("cli_auditor"))

    assert tokens[tokens.index("--mcp-config") + 1] == str(config.resolve())
    assert "--strict-mcp-config" in tokens


def test_claude_stream_result_does_not_duplicate_final_json() -> None:
    answer = '{"action":"CREATE_NEW_SKILL","candidate_spec":{"category_id":"example"}}'
    stream = "\n".join(json.dumps(event) for event in (
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Checking evidence first."}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]}},
        {"type": "result", "result": answer},
    ))
    visible = _extract_claude_visible_output(stream)
    assert visible == answer
    assert json.loads(visible)["action"] == "CREATE_NEW_SKILL"


def test_claude_stream_uses_last_assistant_text_without_result() -> None:
    stream = "\n".join(json.dumps(event) for event in (
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Working."}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": '{"action":"NO_CHANGE"}'}]}},
    ))
    assert _extract_claude_visible_output(stream) == '{"action":"NO_CHANGE"}'


def test_timeout_usage_recovers_completed_requests_from_proxy_trace(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "eval/TB-harness/Harness/src/harbor_backends/claude_code.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_usage_metadata_from_model_io")
    namespace = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    recover = namespace["_usage_metadata_from_model_io"]
    trace = tmp_path / "model_io.jsonl"
    records = [
        {"event": "model_output", "status": 200, "payload": {"format": "sse", "events": [
            {"data": {"choices": [], "usage": None}},
            {"data": {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 7, "cache_read_input_tokens": 20}}},
        ]}},
        {"event": "model_output", "status": 200, "payload": {"format": "json", "value": {
            "usage": {"prompt_tokens": 200, "completion_tokens": 9},
        }}},
        {"event": "model_output", "status": 403, "payload": {"format": "json", "value": {
            "usage": {"prompt_tokens": 99999, "completion_tokens": 99999},
        }}},
    ]
    trace.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    assert recover(trace) == {"n_input_tokens": 300, "n_output_tokens": 16, "n_cache_tokens": 20}
    assert recover(tmp_path / "missing.jsonl") is None
