"""Small, deterministic role-aware retrieval. No model or environment access."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_STOP = set("the a an and or for to in of with from this that task current output input state evidence verify check use role skill required result file when only not no is are on as by at be has have should must".split())
_STOP.update("read write end start set get open close header name path format generated verification audit auditor manager executor guidance contract control example first new old same all each every one two before after files directory script run code error failure success data status true false value values need using used given return returns create ensure instruction instructions do does can will may it its your you we our any both which these those than then also such into over most more less full final expected specify following based want step steps function functions implementation optional include including contain contains available existing keep report information text string line lines without within well print valid invalid produce changes change solution required requirements config site independent deterministic numeric runtime".split())
_STOP.update("tool artifact dual reproduction isolated matrix side effect rerun importability provenance blocker triage readonly baseline integrity object highest id app complete csv json sha256".split())
_ZH_STOP = {"任务", "当前", "结果", "输出", "输入", "检查", "验证", "证据", "使用", "需要", "进行", "文件", "状态", "要求", "原始", "是否", "相关", "执行", "可以", "不得", "不要", "必须", "仅在", "适用", "情况"}


def terms(text: str) -> set[str]:
    words = {w for w in re.findall(r"[a-z][a-z0-9+#.-]{1,}", text.lower()) if w not in _STOP}
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(run[i:i+2] for i in range(len(run)-1) if run[i:i+2] not in _ZH_STOP)
    return words


def _query(brief: dict[str, Any]) -> str:
    # The brief is already constructed by the Harness. Never search the bank
    # against a full role system prompt or a serialized history of all Skills.
    return "\n".join(str(brief.get(key) or "")[:6000] for key in
                     ("task_goal", "subtask", "current_subtask", "current_state", "constraints", "evidence", "recent_actions"))


def select(cards: list[dict[str, Any]], skills: dict[str, dict[str, Any]], role: str,
           brief: dict[str, Any], *, top_k: int = 2, targets: set[str] | None = None) -> dict[str, Any]:
    if not 1 <= top_k <= 2:
        raise ValueError("Skill Top-K must be 1 or 2")
    query = _query(brief)
    query_terms = terms(query)
    view = "manager" if role == "manager" else "auditor" if role.endswith("auditor") else "executor"
    texts = {c["skill_id"]: " ".join([c.get("routing_summary", ""), c.get("role_applicability", ""),
        skills[c["skill_id"]].get("problem_mechanism", ""),
        *skills[c["skill_id"]].get("triggers", []),
        skills[c["skill_id"]]["roles"][view]["guidance"], c["skill_id"].replace("_", " ")]) for c in cards}
    docs = {sid:terms(text) for sid,text in texts.items()}
    # Case/digit-bearing technical names (CMake, SHA256, HTTP, Apache2) are
    # concrete lexical anchors. A lone generic word such as "header" or
    # "failure" is insufficient even when it happens to be rare in the bank.
    technical = {w.lower() for w in re.findall(r"\b[A-Za-z][A-Za-z0-9+#.-]+\b", query)
                 if sum(ch.isupper() for ch in w) >= 2 or any(ch.isdigit() for ch in w)} - _STOP
    frequency = Counter(t for doc in docs.values() for t in doc)
    ranked, reasons = [], []
    for card in cards:
        sid = card["skill_id"]
        skill = skills[sid]
        metadata = skill.get("routing_metadata") or {}
        negative = metadata.get("negative_keywords", [])
        if metadata.get("roles") and view not in metadata["roles"]:
            reasons.append({"skill_id": sid, "score": 0.0, "reason": "role_excluded"}); continue
        if any(word.casefold() in query.casefold() for word in negative if word):
            reasons.append({"skill_id": sid, "score": 0.0, "reason": "negative_trigger"}); continue
        overlap = query_terms & docs[sid]
        positive = [word for word in metadata.get("positive_keywords", []) if word.casefold() in query.casefold()]
        # Require either a concrete English token or multiple Chinese bigrams;
        # generic handoff/evidence prose cannot make every Skill relevant.
        specific = {t for t in overlap if re.search(r"[a-z]", t) or frequency[t] <= max(1, len(cards)//3)}
        domain = terms(sid.replace("_", " "))
        short_identifier = {t for t in specific if re.fullmatch(r"[a-z]{2,3}", t) and frequency[t] == 1}
        if not positive and (len(domain) >= 2 and len(domain & specific) < 2 or
                not (specific & technical or short_identifier or len(specific) >= 2)):
            reasons.append({"skill_id": sid, "score": 0.0, "reason": "low_relevance", "matched_terms": sorted(overlap)}); continue
        score = sum(math.log(1 + len(cards)/(1+frequency[t])) for t in specific) + 3*len(positive) + 2*len(specific & technical)
        # Only paired independent evidence influences utility; task co-success
        # and selection counts are not evidence that the Skill helped.
        independent = skill.get("independent_validation_results") or []
        benefit = sum(bool(r.get("candidate_selected")) and not r.get("hard_constraint_regression") and
                      (r.get("candidate_success") and not r.get("baseline_success") or r.get("cost_improved") is True)
                      for r in independent)
        regression = sum(bool(r.get("hard_constraint_regression")) or
                         bool(r.get("baseline_success") and not r.get("candidate_success")) for r in independent)
        score *= max(0.1, 1 + min(benefit, 3)*0.1 - min(regression, 3)*0.3)
        row = {"skill_id": sid, "version": card["version"], "score": round(score, 6), "matched_terms": sorted(specific),
               "reason": "relevant", "validation_target": sid in (targets or set())}
        ranked.append(row)
    ranked.sort(key=lambda x: (-int(x["validation_target"]), -x["score"], x["skill_id"]))
    chosen, goals = [], set()
    for row in ranked:
        skill = skills[row["skill_id"]]
        # Same mechanism AND same role execution guidance is redundant. Similar
        # text or a shared category alone does not erase complementary Skills.
        goal = (skill["problem_mechanism"].strip().casefold(), skill["roles"][view]["guidance"].strip().casefold())
        if goal in goals:
            row["reason"] = "duplicate_capability"
        elif len(chosen) >= top_k:
            row["reason"] = "below_top_k"
        else:
            goals.add(goal); chosen.append({"skill_id": row["skill_id"], "version": row["version"]})
        reasons.append(row)
    return {"selected_skills": chosen, "scores": reasons, "selection_note": "deterministic_role_retrieval",
            "uncovered_need": "" if chosen else "no sufficiently relevant Skill", "router_llm_calls": 0}
