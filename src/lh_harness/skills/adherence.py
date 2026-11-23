from __future__ import annotations

from .experience import checked_evidence
from .prompts import ADHERENCE_PROMPT
from .schema import ROLES, object_schema, validate_object

STATUSES = ("followed", "partially_followed", "not_followed", "not_applicable", "insufficient_evidence")
MAX_SKILLS_PER_TASK = 4
MAX_SOURCES_PER_SKILL = 8
MAX_QUOTES_PER_SOURCE = 6
MAX_EVIDENCE_OPTIONS = 48


def analyze_injected_skills(bundle, library, model, *, roles=ROLES):
    """Analyze injected Skill adherence from preserved traces; never changes benchmark outcomes."""
    ledgers = bundle.get("skill_usage") or []
    if isinstance(ledgers, dict):
        ledgers = [ledgers]
    injected, malformed = {}, []
    selected_roles = set(roles) & set(ROLES)
    for ledger in ledgers:
        if not isinstance(ledger, dict):
            continue
        for call in ledger.get("injections", []):
            if not isinstance(call, dict) or call.get("role") not in selected_roles:
                continue
            keys, versions = call.get("skills") or [], call.get("versions") or []
            if len(keys) != len(versions):
                malformed.append({"role": call.get("role"), "reason": "skill/version count mismatch"})
            for key, version in zip(keys, versions):
                if not isinstance(key, str) or type(version) is not int or version < 1:
                    malformed.append({"role": call.get("role"), "reason": "invalid skill identity"})
                    continue
                item = injected.setdefault((key, version), {"role": call["role"], "count": 0})
                item["count"] += 1
    if not injected:
        return {"status": "not_applicable", "assessments": [], "injected_skill_count": 0,
                "malformed_injections": malformed, "model_calls": 0}
    loaded, unavailable = [], []
    for (key, version), record in injected.items():
        try:
            skill = library.get_version(key, version)
        except (KeyError, ValueError, OSError) as exc:
            unavailable.append({"key": key, "version": version, "reason": type(exc).__name__})
            continue
        if skill.role != record["role"]:
            unavailable.append({"key": key, "version": version, "reason": "role mismatch"})
        else:
            loaded.append((skill, record))
    loaded.sort(key=lambda pair: (-pair[1]["count"], pair[0].key, pair[0].version))
    deferred = [{"key": s.key, "version": s.version, "reason": "per-task analysis budget"} for s, _ in loaded[MAX_SKILLS_PER_TASK:]]
    loaded = loaded[:MAX_SKILLS_PER_TASK]
    evidence = bundle.get("evidence", {})
    inputs, schemas = [], []
    for skill, record in loaded:
        own_refs = [ref for ref, src in evidence.items() if src.get("role") == skill.role and ref.startswith(("trajectory:", "partial:"))]
        allowed = {skill.role, "auditor", "benchmark"}
        if skill.role == "auditor":
            allowed.update({"manager", "executor"})
        options = []
        ordered_refs = list(own_refs)
        ordered_refs.extend(ref for ref, src in evidence.items()
                            if ref not in own_refs and src.get("role") in allowed and
                            ref.startswith(("trajectory:", "partial:", "rounds.jsonl:", "benchmark-verifier")))
        for ref in ordered_refs[:MAX_SOURCES_PER_SKILL]:
            src = evidence[ref]
            if src.get("role") not in allowed:
                continue
            quotes = []
            for line in src.get("text", "").splitlines():
                quote = line.strip()
                if len(quote) >= 8 and quote not in quotes:
                    quotes.append(quote[:500])
                if len(quotes) >= MAX_QUOTES_PER_SOURCE:
                    break
            for quote in quotes:
                options.append(object_schema({"reference": {"type": "string", "const": ref},
                                              "quote": {"type": "string", "enum": [quote]}}))
                if len(options) >= MAX_EVIDENCE_OPTIONS:
                    break
            if len(options) >= MAX_EVIDENCE_OPTIONS:
                break
        if not own_refs or not options:
            unavailable.append({"key": skill.key, "version": skill.version, "reason": "missing role trajectory evidence"})
            continue
        inputs.append({"key": skill.key, "version": str(skill.version), "role": skill.role, "modality": skill.modality,
            "injection_count": record["count"], "description": skill.description, "trigger": skill.trigger,
            "preconditions": skill.preconditions,
            "procedure": [{"id": f"step-{i+1}", "instruction": step} for i, step in enumerate(skill.procedure)],
            "verification": skill.verification, "pitfalls": skill.pitfalls, "generalization": skill.generalization,
            "trajectory_evidence": [{"reference": ref, "text": evidence[ref]["text"][:6000]} for ref in own_refs[:6]],
            "outcome": bundle.get("outcome"), "agent_timeout": bundle.get("runtime", {}).get("agent_timeout", False)})
        ids = [f"step-{i+1}" for i in range(len(skill.procedure))]
        steps = {"type": "array", "maxItems": len(ids), "items": {"type": "string", "enum": ids}}
        empty = {"type": "array", "maxItems": 0, "items": {"type": "string"}}
        ev = {"anyOf": [empty, {"type": "array", "minItems": 1, "maxItems": 6, "items": {"anyOf": options}}]}
        schemas.append(object_schema({"key": {"type": "string", "const": skill.key},
            "version": {"type": "string", "const": str(skill.version)},
            "status": {"type": "string", "enum": list(STATUSES)}, "followed_steps": steps,
            "skipped_steps": steps, "not_applicable_steps": steps, "unassessed_steps": steps,
            "evidence": ev, "reason": {"type": "string", "maxLength": 600}, "confidence": {"type": "number"}}))
    if not inputs:
        return {"status": "insufficient_evidence", "assessments": [], "injected_skill_count": len(injected),
                "unavailable_skills": unavailable, "malformed_injections": malformed, "deferred_skills": deferred, "model_calls": 0}
    schema = object_schema({"assessments": {"type": "array", "minItems": len(schemas), "maxItems": len(schemas), "items": {"anyOf": schemas}}})
    result = validate_object(model("adherence", "shared", ADHERENCE_PROMPT, {"skills": inputs}, schema), schema)
    expected = {(x["key"], x["version"]) for x in inputs}
    seen, output = set(), []
    for item in result["assessments"]:
        ident = (item["key"], item["version"])
        if ident not in expected or ident in seen:
            raise ValueError("unknown or duplicate adherence Skill identity")
        seen.add(ident)
        skill, record = next((s, r) for s, r in loaded if s.key == item["key"] and str(s.version) == item["version"])
        ids = [f"step-{i+1}" for i in range(len(skill.procedure))]
        groups = {k: list(item[k]) for k in ("followed_steps", "skipped_steps", "not_applicable_steps", "unassessed_steps")}
        flat = [v for values in groups.values() for v in values]
        if len(flat) != len(set(flat)) or set(flat) - set(ids):
            raise ValueError("overlapping or unknown adherence step IDs")
        groups["unassessed_steps"].extend(x for x in ids if x not in flat)
        # Missing steps in an interrupted/partial-only trace are unknown, not
        # evidence that the agent skipped them.
        complete_own_trace = any(ref.startswith("trajectory:") for ref in own_refs)
        partial_execution = bool(bundle.get("runtime", {}).get("agent_timeout")) or not complete_own_trace
        if partial_execution:
            groups["unassessed_steps"].extend(groups["skipped_steps"])
            groups["skipped_steps"] = []
        refs, quotes = [x["reference"] for x in item["evidence"]], [x["quote"] for x in item["evidence"]]
        checked = checked_evidence(bundle, refs, quotes) if refs else []
        own = any(bundle["evidence"][e.reference]["role"] == skill.role and
                   e.reference.startswith(("trajectory:", "partial:")) for e in checked)
        status = item["status"]
        if item["confidence"] < .65 or not own:
            status = "insufficient_evidence"
        elif status == "not_applicable":
            status = "not_applicable" if set(groups["not_applicable_steps"]) == set(ids) else "insufficient_evidence"
        elif status == "followed":
            if groups["followed_steps"] and not groups["skipped_steps"] and not groups["unassessed_steps"]:
                status = "followed"
            elif groups["followed_steps"]:
                status = "partially_followed"
            elif groups["skipped_steps"] and not groups["unassessed_steps"]:
                status = "not_followed"
            else:
                status = "insufficient_evidence"
        elif status == "not_followed":
            status = "not_followed" if groups["skipped_steps"] and not groups["followed_steps"] and not groups["unassessed_steps"] else "insufficient_evidence"
        elif status == "partially_followed":
            status = "partially_followed" if groups["followed_steps"] and (groups["skipped_steps"] or groups["unassessed_steps"]) else "insufficient_evidence"
        def labels(name):
            return [skill.procedure[int(x.split("-")[1])-1] for x in groups[name]]
        output.append({"key": skill.key, "version": skill.version, "role": skill.role, "modality": skill.modality,
            "injection_count": record["count"], "status": status, "followed_steps": labels("followed_steps"),
            "skipped_steps": labels("skipped_steps"), "not_applicable_steps": labels("not_applicable_steps"),
            "unassessed_steps": labels("unassessed_steps"),
            "evidence": [{"reference": e.reference, "quote": e.quote} for e in checked],
            "reason": item["reason"], "confidence": item["confidence"]})
    if seen != expected:
        raise ValueError("adherence response omitted injected Skills")
    return {"status": "completed", "assessments": output, "injected_skill_count": len(injected),
            "analyzed_skill_count": len(output), "unavailable_skills": unavailable,
            "malformed_injections": malformed, "deferred_skills": deferred, "model_calls": 1}
