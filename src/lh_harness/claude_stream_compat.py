"""Normalize explicit CLI event roles for the unchanged original LLH parser."""
from __future__ import annotations

import json


def normalize_assistant_roles(raw: str) -> tuple[str, int]:
    """Use the event's explicit type; never infer roles from report contents."""
    added = 0
    lines = []
    for line in raw.splitlines(keepends=True):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            lines.append(line)
            continue
        if (isinstance(event, dict) and event.get("type") == "assistant"
                and isinstance(event.get("message"), dict) and "role" not in event["message"]):
            event["message"]["role"] = "assistant"
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + ending
            added += 1
        lines.append(line)
    return "".join(lines), added


class ClaudeStreamCompatibilityAdapter:
    """Delegate original execution/permissions and only normalize returned JSONL."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def run_episode(self, *args, **kwargs):
        result = await self.original.run_episode(*args, **kwargs)
        normalized, added = normalize_assistant_roles(result.actions_log)
        if added:
            result.actions_log = normalized
            result.metadata = {**result.metadata, "stream_role_compatibility": {
                "added_assistant_roles": added,
                "raw_trajectory_path": result.metadata.get("agent_log_path"),
                "scope": "explicit assistant event type mapped to missing message.role; report text unchanged",
            }}
        return result
