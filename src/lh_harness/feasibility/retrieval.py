"""Small lexical retrieval with applicability/counterexample checks. No LLM router."""
from __future__ import annotations

import re
from typing import Any

from .schema import Snapshot

STOP = set("the a an of in on to for and or is are be should required check verify task subtask contract available current when before after whether validate validation missing information dependencies prerequisite feasibility readiness".split())


def terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z][a-z0-9_+-]{2,}", text.lower())) - STOP
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(chunk[i:i+2] for i in range(len(chunk)-1))
    return words


def retrieve(skills: list[dict[str, Any]], snapshot: Snapshot, top_k: int):
    # Prefer the assigned contract; the broad original task alone must not route unrelated advice.
    query = terms(snapshot.contract + " " + snapshot.manager_plan)
    selected, trace = [], []
    for skill in skills:
        sid = skill["skill_id"]
        if skill["status"] != "active":
            trace.append({"skill_id": sid, "reason": "not_active", "score": 0})
            continue
        applicability = terms(" ".join(skill["applicability_conditions"]))
        matched = applicability & query
        required = min(2, len(applicability))
        score = len(matched) / max(1, len(applicability))
        if not matched or len(matched) < required:
            trace.append({"skill_id": sid, "reason": "low_applicability", "score": score})
            continue
        # Counterexample evidence must be an observation or Auditor fact, not a claim in the plan.
        observed = " ".join(str(r.get("report", "")) for r in snapshot.verified_auditor_reports)
        observed += " ".join(str(o.get("content", "")) for o in snapshot.environment_observations)
        if any(c.strip().lower() in observed.lower() for c in skill["counterexamples"]):
            trace.append({"skill_id": sid, "reason": "counterexample_observed", "score": score})
            continue
        selected.append((score, sid, skill))
    selected.sort(key=lambda x: (-x[0], x[1]))
    result = []
    mechanisms = set()
    for score, sid, skill in selected:
        # Preserve complementary checks. Dedup only identical checks AND corrections.
        mechanism = (tuple(sorted(skill["preconditions_to_check"])), tuple(sorted(skill["suggested_corrections"])))
        conflict = any(s["skill_id"] in skill.get("potential_conflicts", []) or sid in s.get("potential_conflicts", []) for s in result)
        reason = "conflict" if conflict else "duplicate" if mechanism in mechanisms else "top_k_limit" if len(result) >= top_k else "selected"
        trace.append({"skill_id": sid, "score": score, "reason": reason, "matched": sorted(query & terms(" ".join(skill["applicability_conditions"])))})
        if reason == "selected":
            result.append(skill)
            mechanisms.add(mechanism)
    return result, trace
