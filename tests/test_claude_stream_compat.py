from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lh_harness.claude_stream_compat import ClaudeStreamCompatibilityAdapter, normalize_assistant_roles


def test_only_explicit_assistant_events_receive_missing_role():
    report = "状态: complete\n完整性: clean\n契约审计: aligned"
    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": report}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": "unchanged"}},
        {"type": "user", "message": {"content": [{"type": "text", "text": report}]}},
        {"type": "result", "result": report},
    ]
    raw = "diagnostic warning\n" + "\n".join(json.dumps(e) for e in events)
    fixed, count = normalize_assistant_roles(raw)
    assert count == 1
    assert fixed.startswith("diagnostic warning\n")
    parsed = [json.loads(line) for line in fixed.splitlines()[1:]]
    assert parsed[0]["message"].pop("role") == "assistant"
    assert parsed == events
    assert normalize_assistant_roles(fixed) == (fixed, 0)


def test_native_cli_report_is_consumed_by_unchanged_original_verifier():
    pytest.importorskip("harbor")
    from harness_types import EpisodeResult
    from verifier_agent import verify_report_from_episode_result
    text = "状态: complete\n完整性: clean\n契约审计: aligned\n审计事实: checked."
    raw = "CLI diagnostic\n" + json.dumps({"type": "assistant", "message": {
        "content": [{"type": "text", "text": text}]}}, ensure_ascii=False)
    before = verify_report_from_episode_result(EpisodeResult(status="done", actions_log=raw), 1)
    assert before.status == "blocked" and before.contract_audit_status == "unknown"
    fixed, _ = normalize_assistant_roles(raw)
    after = verify_report_from_episode_result(EpisodeResult(status="done", actions_log=fixed), 1)
    assert (after.status, after.integrity_status, after.contract_audit_status) == ("complete", "clean", "aligned")
    assert after.report_text.startswith(text.splitlines()[0])


def test_wrapper_preserves_status_usage_and_execution_inputs():
    original_result = SimpleNamespace(status="error", actions_log='{"type":"assistant","message":{"content":[]}}',
                                     metadata={"agent_log_path": "/raw/agent.log", "usage": {"input_tokens": 12}})
    calls = []
    class Original:
        usage_totals = {"input_tokens": 12}
        async def run_episode(self, *args, **kwargs):
            calls.append((args, kwargs))
            return original_result
    original = Original()
    wrapped = ClaudeStreamCompatibilityAdapter(original)
    result = asyncio.run(wrapped.run_episode("same prompt", "same environment", budget="same budget"))
    assert calls == [(("same prompt", "same environment"), {"budget": "same budget"})]
    assert result.status == "error" and result.metadata["usage"] == {"input_tokens": 12}
    assert wrapped.usage_totals is original.usage_totals
    assert result.metadata["stream_role_compatibility"]["raw_trajectory_path"] == "/raw/agent.log"
