"""Usage accounting shared by the original and experimental execution paths.

Absent counters stay absent. No character-to-token conversion is used here.
"""
from __future__ import annotations

import json


def unavailable() -> dict:
    return {"status": "unavailable", "input_tokens": None, "output_tokens": None,
            "total_tokens": None, "cache_tokens": None}


def normalize(value: object) -> dict:
    if not isinstance(value, dict):
        return unavailable()
    inp = value.get("input_tokens", value.get("prompt_tokens", value.get("input")))
    out = value.get("output_tokens", value.get("completion_tokens", value.get("output")))
    if not all(type(n) is int and n >= 0 for n in (inp, out)):
        return unavailable()
    # Anthropic reports uncached input separately; OpenAI includes cached input.
    cache = value.get("cache_read_input_tokens", value.get("cached_input_tokens"))
    if "cache_read_input_tokens" in value or "cache_creation_input_tokens" in value:
        inp += sum(value.get(k, 0) for k in ("cache_read_input_tokens", "cache_creation_input_tokens")
                   if type(value.get(k, 0)) is int)
    return {"status": "reported", "input_tokens": inp, "output_tokens": out,
            "total_tokens": inp + out, "cache_tokens": cache if type(cache) is int else None}


def aggregate(values: list[dict]) -> dict:
    if not values:
        return {"status": "not_applicable", "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "cache_tokens": 0, "calls": 0, "missing_calls": 0}
    known = [v for v in values if v.get("status") == "reported"]
    missing = len(values) - len(known)
    return {"status": "unavailable" if missing else "reported", "calls": len(values),
            "missing_calls": missing,
            **{key: sum(v[key] for v in known) if not missing else None
               for key in ("input_tokens", "output_tokens", "total_tokens")},
            "reported_subtotal": {key: sum(v[key] for v in known)
                                  for key in ("input_tokens", "output_tokens", "total_tokens")}}


def from_trajectory(raw: str, metadata: dict | None = None) -> dict:
    """Prefer terminal CLI totals; never sum them with their streamed echoes."""
    terminals, turns, steps, messages = [], [], [], {}
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind in {"result", "dsh.result"} and isinstance(event.get("usage"), dict):
            terminals.append(normalize(event["usage"]))
        elif kind == "turn.completed":
            turns.append(normalize(event.get("usage")))
        elif kind == "step_finish":
            steps.append(normalize((event.get("part") or {}).get("tokens")))
        elif kind == "assistant" and isinstance(event.get("message"), dict):
            msg = event["message"]
            if msg.get("id"):
                messages[msg["id"]] = normalize(msg.get("usage"))
    if terminals:
        return terminals[-1]
    values = turns or steps or list(messages.values())
    if values:
        return aggregate(values)
    return normalize((metadata or {}).get("usage"))
