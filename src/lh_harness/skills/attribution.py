from __future__ import annotations
import copy
from .schema import ROLES, object_schema, validate_object
from .experience import checked_evidence
from .prompts import ATTRIBUTION_PROMPT, ROLE_REVIEW_PROMPT

SCHEMA = object_schema({"primary_role": {"type": "string", "enum": [*ROLES, "runtime", "DEFER"]},
    "contributing_roles": {"type": "array", "items": {"type": "string", "enum": list(ROLES)}},
    "root_cause": {"type": "string"}, "evidence_references": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "number"}, "requirements_covered": {"type": "boolean"}})


def attribute(bundle, model):
    if bundle["runtime"]["infrastructure_failure"]:
        return {"primary_role": "runtime", "decision": "HARNESS_RUNTIME_FAILURE"}
    if bundle["outcome"] == "defer":
        return {"primary_role": "DEFER", "decision": "DEFER"}
    schema = copy.deepcopy(SCHEMA)
    schema["properties"]["evidence_references"]["items"]["enum"] = list(bundle["evidence"])
    value = validate_object(model("attribution", "shared", ATTRIBUTION_PROMPT +
        "\nEvidence references must be EXACT dictionary keys, without appended descriptions or quotes.",
        {"bundle": bundle}, schema), schema)
    evidence = checked_evidence(bundle, value["evidence_references"])
    cited_roles = {bundle["evidence"][e.reference]["role"] for e in evidence}
    primary = value["primary_role"]
    if (value["confidence"] < .8 or "auditor" not in cited_roles or len(cited_roles) < 2 or primary not in ROLES
            or primary not in cited_roles or (bundle["outcome"] == "success" and not value["requirements_covered"])):
        return {**value, "decision": "DEFER"}
    return {**value, "decision": "ATTRIBUTED"}


ROLE_SCHEMA = object_schema({
    "decision": {"type": "string", "enum": ["LEARN", "NONE", "DEFER"]},
    "signal": {"type": "string", "enum": ["positive", "negative", "none"]},
    "rationale": {"type": "string", "maxLength": 1000},
    "evidence_references": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "requirements_covered": {"type": "boolean"},
})
REVIEW_SCHEMA = object_schema({**copy.deepcopy(SCHEMA["properties"]),
    "role_assessments": object_schema({role: copy.deepcopy(ROLE_SCHEMA) for role in ROLES})})


def missing_causal_progress(bundle, role):
    # A reasonable first plan followed by a never-returning Executor is not
    # proof of faulty Manager scheduling. Keep diagnosis possible, but do not
    # turn plan-only timeouts into Manager Skills.
    if (role == 'manager' and bundle.get('outcome') == 'failure' and bundle.get('partial_rounds')
            and not bundle.get('negative_auditor_evidence')
            and not any(v['role']=='executor' for v in bundle['evidence'].values())):
        return 'MISSING_EXECUTION_PROGRESS_FOR_MANAGER_CAUSALITY'
    return None


def role_evidence_valid(bundle, role, references):
    evidence = checked_evidence(bundle, references)
    sources = {bundle["evidence"][item.reference]["role"] for item in evidence}
    negative = bundle.get('outcome') == 'failure'
    if not negative and any(bundle['evidence'][ref].get('completeness') == 'partial' for ref in references):
        return False
    eligible = bundle.get('negative_auditor_evidence', bundle.get('qualified_auditor_evidence')) if negative else bundle.get('qualified_auditor_evidence')
    audit = any(bundle["evidence"][ref]["role"] == "auditor" and (eligible is None or ref in eligible)
                for ref in references)
    external = sources & {"manager", "executor", "benchmark"} if role == "auditor" else sources - {role, "runtime"}
    independent = set(bundle.get('negative_independent_evidence',[])) & set(references) if negative else set()
    if independent:
        independent = {ref for ref in independent if bundle['evidence'][ref]['role'] != role
                       and bundle['evidence'][ref]['role'] in {'auditor','benchmark'}}
    return bool(role in sources and ((audit and external) or independent))


def role_reference_slots(bundle, role):
    negative = bundle.get('outcome') == 'failure'
    sources = {key:value for key,value in bundle['evidence'].items()
               if negative or value.get('completeness') != 'partial'}
    eligible = bundle.get('negative_auditor_evidence', bundle.get('qualified_auditor_evidence')) if negative else bundle.get('qualified_auditor_evidence')
    audits = [key for key, value in sources.items() if value["role"] == "auditor" and (eligible is None or key in eligible)]
    if negative and 'negative_independent_evidence' in bundle:
        own = [key for key,value in sources.items() if value['role']==role]
        independent = [key for key in bundle['negative_independent_evidence'] if key in sources
                       and sources[key]['role'] != role and sources[key]['role'] in {'auditor','benchmark'}]
        if role == 'auditor' and audits:
            independent += [key for key,value in sources.items() if value['role'] in {'manager','executor','benchmark'}]
        return own, list(dict.fromkeys(independent))
    if role == "auditor":
        return audits, [key for key, value in sources.items() if value["role"] in {"manager", "executor", "benchmark"}]
    return [key for key, value in sources.items() if value["role"] == role], audits


def review_roles(bundle, model):
    """One shared model call reviews all roles; only selected roles propose."""
    reason = "HARNESS_RUNTIME_FAILURE" if bundle["runtime"]["infrastructure_failure"] else "DEFER" if bundle["outcome"] == "defer" else None
    if reason:
        codes = bundle.get('learning_eligibility',{}).get('reason_codes',[])
        return {"primary_role": "runtime" if reason == "HARNESS_RUNTIME_FAILURE" else "DEFER",
                "decision": reason, "role_assessments": {role: {"decision": "DEFER", "signal": "none",
                    "rationale": "Actual infrastructure failure" if reason == "HARNESS_RUNTIME_FAILURE" else 'Evidence gate: '+(', '.join(codes) or 'NO_INDEPENDENT_ROLE_EVIDENCE'),
                    "confidence": 0, "evidence_references": [], "requirements_covered": False,
                    "validation_reason": reason, "evidence_gate_reasons":codes} for role in ROLES}}
    schema = copy.deepcopy(REVIEW_SCHEMA)
    schema["properties"]["evidence_references"]["items"]["enum"] = list(bundle["evidence"])
    local_schema = copy.deepcopy(schema)
    for role in ROLES:
        local_schema["properties"]["role_assessments"]["properties"][role]["properties"]["evidence_references"]["items"]["enum"] = list(bundle["evidence"])
        properties = schema["properties"]["role_assessments"]["properties"][role]["properties"]
        item = {"type": "string", "enum": list(bundle["evidence"])}
        first, second = role_reference_slots(bundle, role)
        empty = {"type": "array", "maxItems": 0, "items": item}
        if first and second and not missing_causal_progress(bundle, role):
            properties["evidence_references"] = {"anyOf": [empty,
                {"type": "array", "minItems": 2, "maxItems": 6,
                 "prefixItems": [{"type": "string", "enum": first}, {"type": "string", "enum": second}], "items": item}]}
        else:
            properties["evidence_references"] = empty
            properties["decision"] = {"type": "string", "enum": ["NONE", "DEFER"]}
    value = validate_object(model("attribution", "shared", ROLE_REVIEW_PROMPT, {"bundle": bundle}, schema), local_schema)
    checked_evidence(bundle, value["evidence_references"])
    for role, assessment in value["role_assessments"].items():
        checked_evidence(bundle, assessment["evidence_references"])
        if missing_causal_progress(bundle, role):
            assessment.update(decision='DEFER', signal='none', validation_reason=missing_causal_progress(bundle, role))
            continue
        if assessment["decision"] != "LEARN":
            continue
        invalid = None
        if assessment["confidence"] < .8 or not role_evidence_valid(bundle, role, assessment["evidence_references"]):
            invalid = "INSUFFICIENT_INDEPENDENT_ROLE_EVIDENCE"
        if missing_causal_progress(bundle, role):
            invalid = missing_causal_progress(bundle, role)
        if assessment["signal"] == "none":
            invalid = "NO_LEARNING_SIGNAL"
        if assessment["signal"] == "positive" and (bundle["outcome"] != "success" or bundle["runtime"].get("agent_timeout")
                                                    or not assessment["requirements_covered"] or not value["requirements_covered"]):
            invalid = "POSITIVE_LESSON_REQUIRES_COMPLETE_SUCCESS"
        if invalid:
            assessment.update(decision="DEFER", validation_reason=invalid)
    return {**value, "decision": "REVIEWED"}
