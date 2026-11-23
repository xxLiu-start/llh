from __future__ import annotations

import hashlib
import json
import copy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .attribution import review_roles, role_evidence_valid
from .experience import checked_evidence
from .adherence import analyze_injected_skills
from .proposer import propose
from .curator import curate, general_curate
from .library import atomic_json
from .schema import ROLES, Skill, sha

LEARNING_POLICY_VERSION = "evidence-aware-role-skill-benefit-v5"
LEGACY_POLICY_SHA256 = {'40635d1ef195716907fc7aa31020e4f3a940b2f0c98a7f93e7353c6d713f9c5f'}


def policy_record():
    root = Path(__file__).parent
    names = ("experience.py", "learning_evidence.py", "attribution.py", "adherence.py", "learning.py", "proposer.py", "prompts.py", "curator.py", "schema.py", "client.py", "library.py", "integration.py", "trajectory_compactor.py", "statistics.py")
    return {"version": LEARNING_POLICY_VERSION,
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
            "reviewed_roles": list(ROLES), "agent_timeout": "negative-only review of recorded actions corroborated by clean audits or actual verifier failures; unfinished rounds explicitly partial",
            "schedule": "independent task proposals, then batch-wide curation; no proposal-stage library writes",
            "activation": "Curator acceptance keeps skills candidate until two independent dataset/task identities support the technique",
            "infrastructure_failures": "excluded", "positive_lessons": "complete independently audited success only"}


def _roles(settings, library):
    roles = set(settings.training["train_roles"]) if settings.enabled and settings.training["enabled"] else set()
    if not roles:
        return roles
    if settings.phase not in {"train", "online"}:
        raise ValueError("learning requires train or explicitly labelled online phase")
    if (library.root / "frozen.json").exists():
        raise ValueError("cannot learn into a frozen evaluation library")
    return roles


def _bundles(bundles, splits):
    for bundle in bundles:
        if bundle["dataset"] != splits["dataset"] or bundle["task_id"] not in splits["train"] or bundle["split"] != "train":
            raise ValueError("test leakage: all learning evidence must belong to the declared training set")
    return {(bundle["dataset"], bundle["task_id"]): bundle for bundle in bundles}


def _row_decision(row):
    values = [detail["decision"] for detail in row["role_decisions"].values()]
    if any(isinstance(value, dict) and value.get("changed") for value in values):
        return "UPDATED"
    review = row.get("attribution", {})
    if review and review["decision"] != "REVIEWED":
        return review["decision"]
    for decision in ("PROPOSED", "SOURCE_OUTCOME_DISABLED", "TARGET_ROLE_NOT_ENABLED", "DEFER"):
        if decision in values:
            return decision
    return "NONE"


def prepare_proposals(bundles, settings, library, model, splits, *, output=None):
    """Reflect on each task independently; never invoke Curator or write the library."""
    roles = _roles(settings, library)
    if not roles:
        return {"status": "disabled", "roles": [], "model_calls": 0, "decisions": []}
    unique = _bundles(bundles, splits)
    before = library.digest()
    decisions, errors, proposals = [], [], []
    counts = {role: Counter() for role in ROLES}
    adherence_summary = Counter()
    for bundle in unique.values():
        row = {"task_id": bundle["task_id"], "outcome": bundle["outcome"],
               "agent_timeout": bundle["runtime"].get("agent_timeout", False), "role_decisions": {}}
        if not settings.training.get("learn_from_" + bundle["outcome"], True):
            row.update(decision="SOURCE_OUTCOME_DISABLED", role_decisions={role: {
                "training_enabled": role in roles, "decision": "SOURCE_OUTCOME_DISABLED"} for role in ROLES})
            decisions.append(row)
            continue
        try:
            adherence = analyze_injected_skills(bundle, library, model, roles=roles)
        except Exception as exc:
            adherence = {"status": "error", "assessments": [], "error_type": type(exc).__name__}
            errors.append({"task_id": bundle["task_id"], "stage": "adherence", "type": type(exc).__name__, "detail": str(exc)})
        bundle["skill_adherence"] = adherence
        row["skill_adherence"] = adherence
        adherence_summary[adherence.get("status", "unknown")] += 1
        try:
            review = review_roles(bundle, model)
            row["attribution"] = review
        except Exception as exc:
            errors.append({"task_id": bundle["task_id"], "stage": "role_review", "type": type(exc).__name__, "detail": str(exc)})
            row.update(decision="DEFER", role_decisions={role: {"training_enabled": role in roles,
                "decision": "DEFER", "reason": "ROLE_REVIEW_ERROR"} for role in ROLES})
            decisions.append(row)
            continue
        for role in ROLES:
            assessment = review["role_assessments"][role]
            detail = {"training_enabled": role in roles, "assessment": assessment}
            counts[role]["reviewed"] += 1
            try:
                if review["decision"] != "REVIEWED":
                    detail["decision"] = review["decision"]
                elif assessment["decision"] != "LEARN":
                    detail["decision"] = assessment["decision"]
                elif role not in roles:
                    detail["decision"] = "TARGET_ROLE_NOT_ENABLED"
                else:
                    counts[role]["proposer_calls"] += 1
                    diagnostics = {"skill_adherence": bundle.get("skill_adherence", {})}
                    proposal, action = propose(bundle, review, role, library, model, diagnostics=diagnostics)
                    detail['proposal_diagnostics'] = diagnostics
                    detail["decision"] = "PROPOSED" if proposal else action
                    if proposal:
                        proposal_id = sha([bundle["dataset"], bundle["task_id"], role])
                        detail["proposal_id"] = proposal_id
                        proposals.append({"proposal_id": proposal_id, "task_id": bundle["task_id"], "role": role,
                                          "action": action, "skill": proposal.to_dict(), "proposal_context":diagnostics})
            except Exception as exc:
                errors.append({"task_id": bundle["task_id"], "role": role, "type": type(exc).__name__, "detail": str(exc)})
                detail.update(decision="DEFER", error=type(exc).__name__, reason=str(exc))
            row["role_decisions"][role] = detail
            decision = detail["decision"]
            counts[role]["decision:" + (decision.get("action", "UNKNOWN") if isinstance(decision, dict) else decision)] += 1
        row["decision"] = _row_decision(row)
        decisions.append(row)
    if library.digest() != before:
        raise ValueError("proposal stage changed the Skill library")
    report = {"schema_version": 1, "stage": "propose", "status": "completed_with_errors" if errors else "completed",
              "experiment": "online_skill_learning" if settings.phase == "online" else "offline_skill_learning",
              "learning_policy": policy_record(), "roles": sorted(roles), "reviewed_roles": list(ROLES),
              "training_settings": settings.training, "phase": settings.phase, "library_path": str(library.root),
              "role_summary": {role: dict(counts[role]) for role in ROLES},
              "skill_adherence_summary": dict(adherence_summary), "independent_tasks": len(unique),
              "split_sha256": splits["sha256"], "library_sha256": before, "decisions": decisions,
              "bundles": list(unique.values()), "proposals": proposals, "errors": errors,
              "library_updated": False, "normal_mea_outcome_modified": False}
    report["artifact_sha256"] = sha(report)
    if output:
        atomic_json(Path(output) / "proposals.json", report)
        # Keep compressed source bundles in the immutable proposal artifact,
        # rather than duplicating every trajectory in the online manifest.
        atomic_json(Path(output) / "learning_report.json", {key: value for key, value in report.items() if key != "bundles"})
    return report


def curate_proposals(prepared, settings, library, model, splits, *, output=None, flush=False):
    """Validate all task artifacts before reviewing the entire batch of proposals."""
    roles = _roles(settings, library)
    if not roles:
        return {"status": "disabled", "roles": [], "model_calls": 0, "decisions": []}
    before = library.digest()
    unique, proposals, decisions, errors = {}, [], [], []
    counts = {role: Counter() for role in ROLES}
    for document in prepared:
        payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
        if document.get("artifact_sha256") != sha(payload) or document.get("stage") != "propose":
            raise ValueError("proposal artifact checksum or stage mismatch")
        if (document["roles"] != sorted(roles) or document["training_settings"] != settings.training or
            document["phase"] != settings.phase or document["split_sha256"] != splits["sha256"] or
            (document["learning_policy"] != policy_record() and sha(document['learning_policy']) not in LEGACY_POLICY_SHA256)):
            raise ValueError("proposal artifact policy, roles or training split mismatch")
        if document["library_sha256"] != before or document["library_path"] != str(library.root):
            raise ValueError("Skill library changed since task proposals were extracted")
        bundles = _bundles(document["bundles"], splits)
        if unique.keys() & bundles.keys():
            raise ValueError("duplicate task proposal artifacts in batch")
        unique.update(bundles)
        for item in document["proposals"]:
            skill = Skill.from_dict(item["skill"])
            bundle = next((b for b in bundles.values() if b["task_id"] == item["task_id"]), None)
            if bundle is None or skill.role not in roles or item["role"] != skill.role or item["action"] not in {"NEW", "ENHANCE"}:
                raise ValueError("proposal has an invalid source task or training role")
            if item["proposal_id"] != sha([bundle["dataset"], bundle["task_id"], skill.role]):
                raise ValueError("proposal identity mismatch")
            grounded = checked_evidence(bundle, [e.reference for e in skill.evidence], [e.quote for e in skill.evidence])
            if [vars(e) for e in grounded] != [vars(e) for e in skill.evidence] or not role_evidence_valid(bundle, skill.role, [e.reference for e in grounded]):
                raise ValueError("proposal lost its grounded independent role evidence")
            proposals.append(item)
        decisions.extend(copy.deepcopy(document["decisions"]))
        errors.extend(copy.deepcopy(document["errors"]))
        for role in ROLES:
            counts[role].update(document["role_summary"][role])
    if len(unique) < settings.training["batch_size"] and not flush:
        return {"status": "queued", "independent_tasks": len(unique), "required": settings.training["batch_size"], "decisions": []}
    library.create()
    policy = policy_record()
    history_path = library.root / "learning_policy_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if not history or history[-1]["policy"] != policy:
        history.append({"applied_at": datetime.now(timezone.utc).isoformat(), "policy": policy})
        atomic_json(history_path, history)
    changed_roles = set()
    for item in proposals:
        role = item["role"]
        row = next(row for row in decisions if row["task_id"] == item["task_id"])
        detail = row["role_decisions"][role]
        try:
            counts[role]["curator_calls"] += 1
            detail["decision"] = curate(Skill.from_dict(item["skill"]), library, model,
                min_support=settings.training["min_independent_tasks"],
                batch_proposals=[p for p in proposals if p["role"] == role], proposal_context=item.get('proposal_context'))
            if detail["decision"].get("changed"):
                changed_roles.add(role)
                counts[role][detail["decision"]["status"] + "_updates"] += 1
        except Exception as exc:
            errors.append({"task_id": item["task_id"], "role": role, "stage": "curator", "type": type(exc).__name__, "detail": str(exc)})
            detail.update(decision="DEFER", error=type(exc).__name__, reason=str(exc))
        counts[role]["decision:PROPOSED"] -= 1
        decision = detail["decision"]
        counts[role]["decision:" + (decision["action"] if isinstance(decision, dict) else decision)] += 1
        row["decision"] = _row_decision(row)
    for role in sorted(changed_roles):
        try:
            decisions.append({"role": role, "general_curation": general_curate(role, library, model,
                min_support=settings.training["min_independent_tasks"])})
        except Exception as exc:
            errors.append({"role": role, "type": type(exc).__name__, "detail": str(exc)})
    report = {"status": "completed_with_errors" if errors else "completed",
              "experiment": "online_skill_learning" if settings.phase == "online" else "offline_skill_learning",
              "stage": "curate", "learning_policy": policy, "roles": sorted(roles), "reviewed_roles": list(ROLES),
              "role_summary": {role: dict(counts[role]) for role in ROLES}, "independent_tasks": len(unique),
              "proposal_count": len(proposals), "proposal_artifacts": [doc["artifact_sha256"] for doc in prepared],
              "proposal_policies":list({sha(doc['learning_policy']):doc['learning_policy']['version'] for doc in prepared}.items()),
              "library_before_sha256": before,
              "split_sha256": splits["sha256"], "library_sha256": library.digest(), "decisions": decisions,
              "errors": errors, "normal_mea_outcome_modified": False}
    if output:
        atomic_json(output / "learning_report.json", report)
    return report


def learn(bundles, settings, library, model, splits, *, output=None, flush=False):
    """Offline compatibility entry: finish all independent proposals before curation."""
    roles = _roles(settings, library)
    if not roles:
        return {"status": "disabled", "roles": [], "model_calls": 0, "decisions": []}
    unique = _bundles(bundles, splits)
    if len(unique) < settings.training["batch_size"] and not flush:
        return {"status": "queued", "independent_tasks": len(unique), "required": settings.training["batch_size"], "decisions": []}
    library.create()
    prepared = prepare_proposals(list(unique.values()), settings, library, model, splits, output=output)
    return curate_proposals([prepared], settings, library, model, splits, output=output, flush=flush)
