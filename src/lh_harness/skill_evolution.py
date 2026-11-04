"""Evidence and capability contracts for progressive Skill evolution."""
from __future__ import annotations

import re
from typing import Any


BRANCHES = ("create_skills", "revise_skills", "update_router")
CAPABILITY_FIELDS = (
    "problem_mechanism", "routing_summary", "subcases", "triggers", "exclusions",
    "compatible_with", "conflicts_with", "interaction_modes",
)
ROLE_FIELDS = ("input_use", "output_spec", "guidance", "avoid")


def capability_inventory(skill: dict[str, Any]) -> dict[str, Any]:
    """Capture every applicability and role contract before asking a Builder to revise."""
    result = {key: skill.get(key) for key in CAPABILITY_FIELDS}
    for role in ("manager", "executor", "auditor"):
        for field in ROLE_FIELDS:
            result[f"roles.{role}.{field}"] = skill["roles"][role][field]
    return result


def preservation_map(old: dict[str, Any], new: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Conservative static gate; semantic review is an additional gate.

    Old strings must survive verbatim when a field changes. A Builder may add
    text or relocate an old contract to another field, but cannot silently
    replace an execution, verification, recovery, trigger, or exclusion rule.
    """
    before, after = capability_inventory(old), capability_inventory(new)
    repairs = request.get("fix_defects", [])
    repair_items = [item for item in repairs if isinstance(item, dict)
                    and isinstance(item.get("field"), str)
                    and isinstance(item.get("old_text"), str) and item["old_text"].strip()
                    and isinstance(item.get("reason"), str) and item["reason"].strip()
                    and isinstance(item.get("replacement_text"), str) and item["replacement_text"].strip()
                    and isinstance(item.get("evidence_refs"), list) and item["evidence_refs"]]

    def evidenced_repair(field: str, clause: str) -> bool:
        return any(item["field"] == field and item["old_text"].strip() == clause
                   and item["replacement_text"].strip() in str(after[field])
                   and set(item["evidence_refs"]).issubset(set(request.get("evidence_refs", [])))
                   for item in repair_items)

    removed = []
    changed = []
    for key, value in before.items():
        if value == after[key]:
            continue
        changed.append(key)
        if isinstance(value, list):
            if any(item not in after[key] and not evidenced_repair(key, str(item)) for item in value):
                removed.append(key)
        else:
            if key.startswith("roles."):
                role = key.split(".")[1]
                retained_text = "\n".join(str(after[f"roles.{role}.{field}"]) for field in ROLE_FIELDS)
            else:
                retained_text = "\n".join(str(item) for field, item in after.items()
                                          if not field.startswith("roles."))
            clauses = [part.strip() for part in re.split(r"[;；。\n]+", str(value)) if part.strip()]
            if any(clause not in retained_text and not evidenced_repair(key, clause)
                   for clause in clauses):
                removed.append(key)
    if removed:
        raise ValueError("revision removed old capability contracts: " + ", ".join(removed))
    if not request.get("add_capabilities") and not request.get("fix_defects"):
        raise ValueError("revision lacks an added capability or evidenced defect")
    return {
        "skill_id": old["skill_id"], "old_version": old["version"],
        "new_version": new["version"],
        "preserved_capabilities": list(before),
        "added_capabilities": request.get("add_capabilities", []),
        "repaired_defects": request.get("fix_defects", []),
        "changed_fields": changed,
        "text_compression_only": [],
        "preservation_evidence": request.get("evidence_refs", []),
    }


def validate_plan_shape(plan: dict[str, Any]) -> None:
    for branch in BRANCHES:
        if not isinstance(plan.get(branch), list):
            raise ValueError(f"Evolution Plan {branch} must be an array")
    if not isinstance(plan.get("no_update_reasons", []), list):
        raise ValueError("no_update_reasons must be an array")
