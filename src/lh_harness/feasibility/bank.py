"""Versioned feasibility skills. Active-only retrieval; locked atomic writes."""
from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

LIST_FIELDS = ("applicability_conditions", "preconditions_to_check", "failure_patterns",
               "required_evidence", "counterexamples", "suggested_corrections", "source_trajectory_ids")


def validate_skill(skill: dict[str, Any]) -> None:
    if not isinstance(skill, dict):
        raise ValueError("skill must be an object")
    for key in ("skill_id", "name", "description"):
        if not isinstance(skill.get(key), str) or not skill[key].strip():
            raise ValueError(f"missing {key}")
    if not re.fullmatch(r"FS_[A-Za-z0-9_-]{1,80}", skill["skill_id"]):
        raise ValueError("invalid skill_id")
    for key in LIST_FIELDS:
        value = skill.get(key)
        limit = 10000 if key == "source_trajectory_ids" else 30
        if not isinstance(value, list) or not value or len(value) > limit or not all(
            isinstance(v, str) and 0 < len(v.strip()) <= 1000 for v in value):
            raise ValueError(f"invalid {key}")
    if type(skill.get("version")) is not int or skill["version"] < 1:
        raise ValueError("invalid version")
    if skill.get("status") not in {"candidate", "active", "deprecated"}:
        raise ValueError("invalid status")
    if type(skill.get("support_count")) is not int or skill["support_count"] < 0:
        raise ValueError("invalid support_count")
    if type(skill.get("confidence")) not in {int, float} or not 0 <= skill["confidence"] <= 1:
        raise ValueError("invalid confidence")
    if skill["status"] == "deprecated" and not skill.get("deprecation_reason"):
        raise ValueError("deprecated skill needs explanation")
    # Do not promote a task answer disguised as a reusable precondition.
    text = " ".join([skill["description"], *[v for k in LIST_FIELDS[:-1] for v in skill[k]]])
    if re.search(r"(?:/tests/|/data0/|task[_ -]?id\s*(?:==|equals|is)|benchmark question|题号)", text, re.I):
        raise ValueError("task-specific skill content")
    for source in skill["source_trajectory_ids"]:
        if len(source) >= 8 and source.lower() in text.lower():
            raise ValueError("source task leaked into skill rule")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".feasibility-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class SkillBank:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    @contextlib.contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load(self, *, candidates: bool = False) -> list[dict[str, Any]]:
        # Skill files are canonical. A torn/stale index never hides a saved skill.
        records = []
        for directory in (["skills", "candidates"] if candidates else ["skills"]):
            for path in sorted((self.root / directory).glob("FS_*.json")):
                skill = json.loads(path.read_text())
                validate_skill(skill)
                if skill["status"] == "active" or candidates and skill["status"] == "candidate":
                    records.append(skill)
        return records

    def _index(self):
        files = sorted([*(self.root/"skills").glob("FS_*.json"), *(self.root/"candidates").glob("FS_*.json")])
        atomic_json(self.root/"index.json", {"schema_version": 1, "skills": [
            {"path": str(p.relative_to(self.root)), "skill_id": s["skill_id"], "version": s["version"], "status": s["status"]}
            for p in files for s in [json.loads(p.read_text())]]})

    def recover_index(self):
        with self.lock():
            self._index()

    def put(self, skill: dict[str, Any], *, reason: str, validation: dict[str, Any] | None = None):
        validate_skill(skill)
        with self.lock():
            self._put_locked(skill, reason=reason, validation=validation)

    def _put_locked(self, skill, *, reason, validation=None):
        validate_skill(skill)
        sid = skill["skill_id"]
        old_paths = [p for d in ("skills", "candidates") if (p := self.root/d/f"{sid}.json").exists()]
        old = [json.loads(p.read_text()) for p in old_paths]
        if old and skill["version"] <= max(v["version"] for v in old):
            raise ValueError("version must increase; existing skill retained")
        if skill["status"] == "active" and not (validation or {}).get("promote"):
            raise ValueError("active skill requires held-out validation")
        for v in old:
            atomic_json(self.root/"history"/sid/f"v{v['version']}.json", v)
        target = "candidates" if skill["status"] == "candidate" else "skills"
        atomic_json(self.root/target/f"{sid}.json", skill)
        # A provisional revision must not replace its still-valid active parent.
        if skill["status"] != "candidate":
            for p in old_paths:
                if p != self.root/target/f"{sid}.json":
                    p.unlink()
        self._index()
        event = {"skill_id": sid, "version": skill["version"], "reason": reason,
                 "validation": validation, "status": skill["status"]}
        with (self.root/"history.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False)+"\n")


def fingerprint(skill: dict[str, Any]) -> str:
    # Equivalence is a structured mechanism+correction match, never word similarity alone.
    parts = [skill[k] for k in ("preconditions_to_check", "failure_patterns", "suggested_corrections", "counterexamples")]
    normalized = [[re.sub(r"\s+", " ", s.strip().lower()) for s in sorted(p)] for p in parts]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def runtime_view(skill: dict[str, Any]) -> dict[str, Any]:
    # Evidence provenance/history can grow indefinitely; it is not runtime guidance.
    return {k: skill[k] for k in ("skill_id", "version", "name", "description",
        "applicability_conditions", "preconditions_to_check", "failure_patterns",
        "required_evidence", "counterexamples", "suggested_corrections")}


def stage(bank: SkillBank, proposal: dict[str, Any]) -> tuple[dict[str, Any], str]:
    with bank.lock():
        return _stage_locked(bank, proposal)


def _stage_locked(bank: SkillBank, proposal: dict[str, Any]) -> tuple[dict[str, Any], str]:
    skill = copy.deepcopy(proposal)
    validate_skill(skill)
    for old in bank.load(candidates=True):
        if fingerprint(old) == fingerprint(skill):
            if old["status"] == "active":
                return old, "existing_active_retained"
            if set(skill["source_trajectory_ids"]) <= set(old["source_trajectory_ids"]):
                return old, "existing_candidate_retained"
            old["source_trajectory_ids"] = sorted(set(old["source_trajectory_ids"]+skill["source_trajectory_ids"]))
            old["source_task_ids"] = sorted(set(old.get("source_task_ids", [])+skill.get("source_task_ids", [])))
            old["source_task_fingerprints"] = sorted(set(old.get("source_task_fingerprints", [])+skill.get("source_task_fingerprints", [])))
            old["support_count"] = len(old["source_trajectory_ids"])
            old["version"] += 1
            bank._put_locked(old, reason="merge_equivalent_candidate")
            return old, "merged"
        same_conditions = set(old["applicability_conditions"]) == set(skill["applicability_conditions"])
        if same_conditions and set(old["suggested_corrections"]) != set(skill["suggested_corrections"]):
            # Conservative conflict flag: keep both candidates, require explicit resolution.
            skill.setdefault("potential_conflicts", []).append(old["skill_id"])
    bank._put_locked(skill, reason="offline_training_candidate")
    return skill, "staged"


def merge_semantic_candidate(bank: SkillBank, proposal: dict, equivalence: dict) -> dict | None:
    """Model-assisted equivalence with complete check/correction alignment.

Missing mappings, added checks, different applicability, or contradictions retain
separate candidates. Active abilities are never replaced by a model merge.
"""
    if not isinstance(equivalence, dict) or equivalence.get("equivalent") is not True or equivalence.get("contradictions") != []:
        return None
    with bank.lock():
        old = next((s for s in bank.load(candidates=True) if s["skill_id"] == equivalence.get("skill_id") and s["status"] == "candidate"), None)
        if old is None:
            return None
        for field in ("applicability_conditions", "preconditions_to_check", "suggested_corrections", "failure_patterns"):
            mapping = equivalence.get("alignment", {}).get(field)
            if not isinstance(mapping, list) or not all(isinstance(m, dict) for m in mapping):
                return None
            if {m.get("new") for m in mapping} != set(proposal[field]) or {m.get("existing") for m in mapping} != set(old[field]):
                return None
        # Keep the old checks and strategies, enrich only evidence/counterexamples/provenance.
        old["counterexamples"] = sorted(set(old["counterexamples"]+proposal["counterexamples"]))
        old["required_evidence"] = sorted(set(old["required_evidence"]+proposal["required_evidence"]))
        old["source_trajectory_ids"] = sorted(set(old["source_trajectory_ids"]+proposal["source_trajectory_ids"]))
        old["source_task_ids"] = sorted(set(old.get("source_task_ids", [])+proposal.get("source_task_ids", [])))
        old["source_task_fingerprints"] = sorted(set(old.get("source_task_fingerprints", [])+proposal.get("source_task_fingerprints", [])))
        old["support_count"] = len(old["source_trajectory_ids"])
        old["version"] += 1
        old["semantic_merge_evidence"] = {"skill_id": equivalence["skill_id"], "equivalent": True,
            "contradictions": [], "alignment": {field: [{"new": m["new"], "existing": m["existing"]}
                for m in equivalence["alignment"][field]] for field in
                ("applicability_conditions", "preconditions_to_check", "suggested_corrections", "failure_patterns")}}
        bank._put_locked(old, reason="structured_semantic_equivalence_merge")
        return old
