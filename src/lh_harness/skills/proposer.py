from __future__ import annotations
import copy
import re
from .schema import Skill, ROLES, MODALITIES, object_schema, validate_object
from .experience import checked_evidence
from .prompts import PROPOSER_PROMPTS, SUCCESS_PROMPT, FAILURE_PROMPT, TIMEOUT_PROMPT
from .attribution import role_evidence_valid, role_reference_slots

STR = {"type": "string"}
LIST = {"type": "array", "items": STR}
SCHEMA = object_schema({"action": {"type": "string", "enum": ["NEW", "ENHANCE", "NONE", "DEFER"]},
    "reason": {"type":"string","maxLength":600},
    "target_role": {"type": "string", "enum": list(ROLES)}, "modality": {"type": "string", "enum": list(MODALITIES)},
    "name": STR, "description": STR, "trigger": LIST, "preconditions": LIST, "procedure": LIST,
    "verification": LIST, "pitfalls": LIST, "generalization": STR, "topic": STR,
    "evidence": {"type": "array", "items": object_schema({"reference": STR, "quote": STR})}})


def propose(bundle, attribution, role, library, model, *, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else {}
    assessment = attribution.get("role_assessments", {}).get(role)
    if assessment is not None:
        if assessment["decision"] != "LEARN":
            return None, assessment["decision"]
        positive = assessment["signal"] == "positive"
    else:
        if attribution["primary_role"] != role or attribution.get("decision") != "ATTRIBUTED":
            return None, "DEFER"
        positive = bundle["outcome"] == "success"
    prompt = PROPOSER_PROMPTS[role] + (SUCCESS_PROMPT if positive else FAILURE_PROMPT)
    if bundle["runtime"].get("agent_timeout"):
        prompt += "\n" + TIMEOUT_PROMPT
    existing = [s.to_dict() for s in library.list(role, candidates=True)]
    schema = copy.deepcopy(SCHEMA)
    schema["properties"]["target_role"] = {"type": "string", "const": role}
    schema["properties"]["modality"] = {"type":"string","enum":["shared"] if role=="manager" else ["shared",bundle.get("modality","cli")]}
    # SCHEMA deliberately reuses the immutable STR template. Replace this
    # field, rather than mutating an aliased dict and restricting every string.
    options = []
    for reference, source in bundle["evidence"].items():
        lines = [line.strip() for line in source["text"].splitlines() if len(line.strip()) >= 8]
        lines = [line for line in lines if not line.rstrip('*# ').endswith((':','：'))
                 and not re.fullmatch(r'(?:状态|完整性|契约审计|status|integrity|contract audit)\s*[:：].*',line,re.I)]
        focus = r'budget|deadline|timeout|needs_revision|耗尽|超时|预算|修订|未完成|尚未|缺失|未满足' if not positive else r'verif|independ|verified|独立|验证|审计'
        ranked = sorted(enumerate(lines), key=lambda row:(-len(re.findall(focus,row[1],re.I)),row[0]))
        excerpts = list(dict.fromkeys(line[:500] for _,line in ranked[:8]))
        if excerpts:
            options.append(object_schema({"reference": {"type":"string","const":reference},
                                          "quote": {"type":"string","enum":excerpts}}))
    schema["properties"]["evidence"] = {"type":"array","maxItems":3,"items":{"anyOf":options}}
    if assessment is not None:
        first_refs, second_refs = role_reference_slots(bundle, role)
        first = [option for option in options if option["properties"]["reference"]["const"] in first_refs]
        second = [option for option in options if option["properties"]["reference"]["const"] in second_refs]
        if not first or not second:
            diagnostics.update(reason='No own-role and independent-evidence quote slots are available', validation_reason='MISSING_EVIDENCE_QUOTE_SLOTS')
            return None, "DEFER"
        # Prefix slots keep native constrained decoding from returning three
        # self-references while omitting the required independent evidence.
        # Empty evidence remains available for NONE/DEFER; NEW/ENHANCE still
        # passes the grounded role/audit checks below before any file is saved.
        grounded_array = {"type": "array", "minItems": 2, "maxItems": 3,
                          "prefixItems": [{"anyOf": first}, {"anyOf": second}], "items": {"anyOf": options}}
        schema["properties"]["evidence"] = {"anyOf": [
            {"type": "array", "maxItems": 0, "items": {"anyOf": options}}, grounded_array]}
        prompt += ("\nFor NEW/ENHANCE, evidence[0] must cite your own role, evidence[1] qualified independent evidence. "
                   "For positive Auditor lessons, evidence[0] is qualified audit and evidence[1] is outside-auditor evidence. "
                   "NONE/DEFER must use an empty evidence array.")
        if not positive:
            prompt += '\nNegative lessons may cite clean needs_revision audits or actual benchmark failures as independent corroboration. Cite the observed mistake, not an untested repair as if it succeeded.'
    # Keep the wire schema portable: some constrained decoders mishandle a
    # regex with an empty alternative. Kebab-case is checked locally by Skill.
    schema["properties"]["name"] = {"type":"string","maxLength":80}
    for field in ("description", "generalization", "topic"):
        schema["properties"][field] = {"type":"string","maxLength":160}
    for field in ("trigger", "preconditions", "procedure", "verification", "pitfalls"):
        schema["properties"][field] = {"type":"array","maxItems":3 if field=="procedure" else 2,
                                       "items":{"type":"string","maxLength":110}}
    value = validate_object(model("proposer", role, prompt, {"bundle": bundle, "attribution": attribution, "existing": existing}, schema), schema)
    if value["target_role"] != role:
        raise ValueError("proposal targets an unselected or causally unsupported role")
    action = value.pop("action")
    reason = value.pop('reason').strip()
    if not reason:
        raise ValueError('every proposal decision, including NONE/DEFER, requires a reason')
    diagnostics.update(reason=reason, signal='positive' if positive else 'negative',
                       repair_validation='not_applicable' if positive else 'not_validated')
    if action in {"NONE", "DEFER"}:
        return None, action
    evidence = value.pop("evidence")
    grounded = checked_evidence(bundle, [e["reference"] for e in evidence], [e["quote"] for e in evidence])
    if positive and not any(bundle["evidence"][e.reference]["role"] == "auditor" for e in grounded):
        raise ValueError("proposal lacks independent auditor evidence")
    if assessment is not None and not role_evidence_valid(bundle, role, [e.reference for e in grounded]):
        raise ValueError("proposal lacks qualified independent evidence for its own role")
    skill = Skill(role=value.pop("target_role"), evidence=grounded, **value)
    if bundle["task_id"].lower() in skill.text().lower():
        raise ValueError("proposal memorizes its source task ID")
    return skill, action
