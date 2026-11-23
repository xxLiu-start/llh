from __future__ import annotations

import hashlib
import json
import re
from collections import Counter


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
            return "\\n".join(text for item in value if (text := visible_result(item)))
        if isinstance(value, dict):
            if value.get("type") == "image":
                return "[image omitted]"
            text = value.get("text")
            if isinstance(text, str):
                return text
        return ""

    with path.open(errors="replace") as source:
        for index, line in enumerate(source, 1):
            try:
                event = json.loads(line)
            except ValueError:
                malformed += 1
                if re.search(r"error|timeout|failed|exception", line, re.I):
                    errors.append({"line": index, "detail": redact(line[:1000])})
                continue
            def walk(value):
                if isinstance(value, dict):
                    kind = value.get("type", "")
                    if kind in {"tool_use", "function_call", "command_execution"} or "tool_calls" in value:
                        command = value.get("input", value.get("arguments", value.get("command", "")))
                        command = redact(json.dumps(command, ensure_ascii=False)[:1200])
                        name = value.get("name", kind)
                        signature = str(name) + ":" + command
                        actions[signature] += 1
                        tools.append({"line": index, "tool": name, "input": command})
                    if kind in {"tool_result", "function_call_output"} or (kind == "command_execution" and "aggregated_output" in value):
                        result_value = value.get("content", value.get("output", value.get("aggregated_output", value.get("result", ""))))
                        detail = redact(visible_result(result_value)[:800])
                        if detail:
                            results.append({"line": index, "tool_use_id": str(value.get("tool_use_id", value.get("id", ""))),
                                            "detail": detail})
                    if value.get("is_error") or value.get("error") or value.get("status") in {"error", "timeout", "failed"}:
                        errors.append({"line": index, "detail": redact(json.dumps(value, ensure_ascii=False)[:1500])})
                    for key, item in value.items():
                        if key not in {"data", "image", "image_url", "screenshot", "base64"}:
                            walk(item)
                elif isinstance(value, list):
                    for item in value:
                        walk(item)
            walk(event)
            if event.get("type") in {"result", "turn.completed", "error"}:
                tails.append({"line": index, "type": event["type"], "detail": redact(str(event.get("result", event.get("error", "")))[:1000])})
    # Error excerpts are not dropped by ordinary event sampling. Their full
    # source digest permits checking causes outside the compressed envelope.
    return {"tool_calls": tools[:max_events // 2] + tools[max(max_events // 2, len(tools) - max_events // 2):],
            "tool_results": results[:max_events // 2] + results[max(max_events // 2, len(results) - max_events // 2):],
            "tool_call_count": len(tools), "errors": errors, "final_events": tails[-4:],
            "repeated_actions": [{"action": k, "count": v} for k, v in sorted(actions.items()) if v > 1],
            "malformed_lines": malformed, "raw_trajectory_passed_to_model": False}
