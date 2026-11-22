"""Lossless deduplication of the Reflector's model input, not task state."""
from dataclasses import replace
import json

from .bank import runtime_view


def prepare_context(snapshot, selected, limit):
    def packed(value, skills):
        return json.dumps({"snapshot": value.to_dict(), "skills": [runtime_view(s) for s in skills]}, ensure_ascii=False)
    raw = packed(snapshot, selected)
    plan = snapshot.manager_plan
    removed = []
    # These full texts remain in their dedicated snapshot fields. Replace only
    # exact duplicate occurrences; do not summarize/drop acceptance constraints.
    for field in ("contract", "manager_state"):
        text = getattr(snapshot, field)
        if len(text) >= 80 and text in plan:
            plan = plan.replace(text, f"[Full unchanged text is in snapshot.{field}]")
            removed.append(field)
    view = replace(snapshot, manager_plan=plan)
    content = packed(view, selected)
    ids = [s["skill_id"] for s in selected]
    trace = {"raw_input_chars": len(raw), "input_chars": len(content),
             "input_char_limit": limit, "deduplicated_fields": removed,
             "used_skill_guidance": bool(selected)}
    if len(content) > limit and selected:
        trace.update(omitted_skills=ids, skill_omission_reason="input_budget_exceeded", used_skill_guidance=False)
        ids = []
        content = packed(view, [])
        trace["input_chars"] = len(content)
    return view, content, ids, trace
