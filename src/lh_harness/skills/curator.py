from __future__ import annotations
from .schema import Skill, object_schema, validate_object
from .prompts import CURATOR_PROMPT, GENERAL_CURATOR_PROMPT

STR = {"type": "string"}
SCHEMA = object_schema({"action": {"type": "string", "enum": ["ACCEPT", "MERGE", "UPDATE", "REJECT", "DEFER"]}, "target": STR, "reason": STR})
GENERAL_SCHEMA = object_schema({"action": {"type": "string", "enum": ["MERGE", "NONE", "DEFER"]}, "source": STR, "target": STR, "reason": STR})


def combine(prior, proposal):
    if (prior.role, prior.modality) != (proposal.role, proposal.modality):
        raise ValueError("cross-role or cross-modality merge is forbidden")
    value = prior.to_dict()
    for field in ("trigger", "preconditions", "procedure", "verification", "pitfalls"):
        value[field] = list(dict.fromkeys([*getattr(prior, field), *getattr(proposal, field)]))
    evidence = {tuple(vars(e).values()): e for e in [*prior.evidence, *proposal.evidence]}
    value["evidence"] = list(evidence.values())
    return Skill.from_dict(value)


def curate(proposal, library, model, *, min_support=1, batch_proposals=None, proposal_context=None):
    existing = [s for s in library.list(proposal.role, candidates=True) if s.status != "rejected"]
    decision = validate_object(model("curator", proposal.role, CURATOR_PROMPT,
        {"proposal": proposal.to_dict(), "existing": [s.to_dict() for s in existing],
         "batch_proposals": batch_proposals or [], "proposal_context":proposal_context or {}}, SCHEMA), SCHEMA)
    action = decision["action"]
    if action in {"REJECT", "DEFER"}:
        return {**decision, "changed": False}
    prior = next((s for s in existing if s.key == proposal.key or
        (s.modality == proposal.modality and s.procedure == proposal.procedure and s.verification == proposal.verification)), None)
    if action in {"MERGE", "UPDATE"}:
        prior = library.get(decision["target"])
        proposal = combine(prior, proposal)
    elif prior:
        proposal = combine(prior, proposal)
    proposal.status = "active" if proposal.support >= min_support else "candidate"
    stored = library.put(proposal, min_support=min_support)
    return {**decision, "changed": True, "key": stored.key, "status": stored.status, "support": stored.support}


def general_curate(role, library, model, *, min_support=1):
    skills = [s for s in library.list(role, candidates=True) if s.status != "rejected"]
    if len(skills) < 2:
        return {"action": "NONE", "reason": "fewer than two skill patterns"}
    decision = validate_object(model("general_curator", role, GENERAL_CURATOR_PROMPT,
        {"role": role, "skills": [s.to_dict() for s in skills]}, GENERAL_SCHEMA), GENERAL_SCHEMA)
    if decision["action"] != "MERGE":
        return decision
    source, target = library.get(decision["source"]), library.get(decision["target"])
    if source.key == target.key or source.role != role or target.role != role:
        raise ValueError("general curator cannot merge across roles or into itself")
    merged = combine(target, source)
    merged.status = "active" if merged.support >= min_support else "candidate"
    stored = library.put(merged, min_support=min_support)
    # Preserve the source in history and mark its current entry rejected. It
    # remains recoverable, and no active duplicate consumes router budget.
    source.status = "rejected"
    library.put(source, min_support=min_support)
    return {**decision, "version": stored.version}
