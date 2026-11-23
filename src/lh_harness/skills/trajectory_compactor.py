from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from ..agent_logs import parse_trajectory


def redact(text):
    text = re.sub(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?bearer\s+)\S+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|password|secret|authorization|access_token)\s*[=:]\s*)[^\s,}\"]+", r"\1[REDACTED]", text)
    return re.sub(r"\bsk-[A-Za-z0-9_-]{16,}\b", "[REDACTED]", text)


def compact(path, *, max_events=64):
    """Streaming, deterministic extraction; errors are retained separately.

    Large tool output and images are never sent to the offline learner. Raw
    evidence remains on disk, referenced by file digest and line number.
    """
    tools, results, errors, actions, tails = [], [], [], Counter(), []
    malformed = 0

    def visible_result(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(text for item in value if (text := visible_result(item)))
        if isinstance(value, dict):
            if value.get("type") == "image":
                return "[image omitted]"
            text = value.get("text")
            if isinstance(text, str):
                return text
        return ""

    seen_tool_ids = set()
    with path.open(errors="replace") as source:
        for index, line in enumerate(source, 1):
            try:
                json.loads(line)
            except ValueError:
                malformed += 1
                if re.search(r"error|timeout|failed|exception", line, re.I):
                    errors.append({"line": index, "detail": redact(line[:1000])})
                continue
            # Reuse the harness's provider-neutral parser so Claude, Codex
            # (including MCP GUI/file-change calls), OpenCode and DeepSeek
            # episodes all produce the same bounded evidence shape.
            for step in parse_trajectory(line):
                kind = step.get("kind")
                if kind == "tool_use":
                    tool_id = str(step.get("id") or "")
                    if tool_id and tool_id in seen_tool_ids:
                        continue
                    if tool_id:
                        seen_tool_ids.add(tool_id)
                    name = str(step.get("name") or "tool")
                    command = redact(json.dumps(step.get("input", {}), ensure_ascii=False)[:1200])
                    signature = name + ":" + command
                    actions[signature] += 1
                    tools.append({"line": index, "tool": name, "input": command})
                elif kind == "tool_result":
                    detail = redact(str(step.get("text") or "")[:800])
                    if not detail and step.get("has_image"):
                        detail = "[image omitted]"
                    if detail:
                        results.append({"line": index, "tool_use_id": str(step.get("tool_use_id", "")),
                                        "detail": detail})
                    if step.get("is_error") or step.get("status") in {"error", "timeout", "failed"}:
                        errors.append({"line": index, "detail": detail or "tool reported an error"})
                elif kind == "result":
                    detail = redact(str(step.get("text") or "")[:1000])
                    tails.append({"line": index, "type": "result", "detail": detail})
                    if step.get("is_error"):
                        errors.append({"line": index, "detail": detail or "agent reported an error"})
    # Error excerpts are not dropped by ordinary event sampling. Their full
    # source digest permits checking causes outside the compressed envelope.
    return {"tool_calls": tools[:max_events // 2] + tools[max(max_events // 2, len(tools) - max_events // 2):],
            "tool_results": results[:max_events // 2] + results[max(max_events // 2, len(results) - max_events // 2):],
            "tool_call_count": len(tools), "errors": errors, "final_events": tails[-4:],
            "repeated_actions": [{"action": k, "count": v} for k, v in sorted(actions.items()) if v > 1],
            "malformed_lines": malformed, "raw_trajectory_passed_to_model": False}
