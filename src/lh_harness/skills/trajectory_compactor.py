from __future__ import annotations

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

    seen_tool_ids = set()
    with path.open(errors="replace") as source:
        for index, line in enumerate(source, 1):
            try:
                event = json.loads(line)
            except ValueError:
                malformed += 1
                if re.search(r"error|timeout|failed|exception", line, re.I):
                    errors.append({"line": index, "detail": redact(line[:1000])})
                continue

            def collect_errors(value):
                if isinstance(value, dict):
                    if value.get("is_error") or value.get("error") or value.get("status") in {"error", "timeout", "failed"}:
                        errors.append({"line": index, "detail": redact(json.dumps(value, ensure_ascii=False)[:1500])})
                    for key, item in value.items():
                        if key not in {"data", "image", "image_url", "screenshot", "base64"}:
                            collect_errors(item)
                elif isinstance(value, list):
                    for item in value:
                        collect_errors(item)

            collect_errors(event)
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
                elif kind == "result":
                    detail = redact(str(step.get("text") or "")[:1000])
                    tails.append({"line": index, "type": "result", "detail": detail})
    # Error excerpts are not dropped by ordinary event sampling. Their full
    # source digest permits checking causes outside the compressed envelope.
    return {"tool_calls": tools[:max_events // 2] + tools[max(max_events // 2, len(tools) - max_events // 2):],
            "tool_results": results[:max_events // 2] + results[max(max_events // 2, len(results) - max_events // 2):],
            "tool_call_count": len(tools), "errors": errors, "final_events": tails[-4:],
            "repeated_actions": [{"action": k, "count": v} for k, v in sorted(actions.items()) if v > 1],
            "malformed_lines": malformed, "raw_trajectory_passed_to_model": False}
