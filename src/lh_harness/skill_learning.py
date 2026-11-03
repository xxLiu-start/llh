"""Prompt-driven, category-level Skill learning for the role-managed harness.

The bank is opt-in. Generated Skills enter a temporary bank as provisional;
validated Skills move to the regular bank after an independent replay.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .adapters.base import AgentAdapter
from .environment.base import Environment
from .types import EpisodeBudget, EpisodeResult
from .agent_logs import visible_output
from .auditor_agent import parse_audit_report
from .supervisor.control_bus import _atomic_bytes_write, _ensure_dir_nofollow
from .skill_prompt_texts import (
    ROUTER_PROMPT_EN,
    EXPERIENCE_EXTRACTION_PROMPT_EN,
    SUCCESS_SUMMARIZER_PROMPT_EN,
    FAILURE_DIAGNOSIS_PROMPT_EN,
    ROUTER_REPAIR_PROMPT_EN,
    SKILL_BUILDER_PROMPT_EN,
    REVISION_PRESERVATION_REVIEW_PROMPT_EN,
)
from .skill_prompt_texts_zh import (
    ROUTER_PROMPT_ZH, SUCCESS_SUMMARIZER_PROMPT_ZH,
    EXPERIENCE_EXTRACTION_PROMPT_ZH,
    FAILURE_DIAGNOSIS_PROMPT_ZH, ROUTER_REPAIR_PROMPT_ZH,
    SKILL_BUILDER_PROMPT_ZH,
    REVISION_PRESERVATION_REVIEW_PROMPT_ZH,
)

ROLES = {"manager", "gui_executor", "cli_executor", "gui_auditor", "cli_auditor"}
VIEWS = {"manager": "manager", "gui_executor": "executor", "cli_executor": "executor", "gui_auditor": "auditor", "cli_auditor": "auditor"}
FIELDS = ("input_use", "output_spec", "guidance", "avoid")
ROUTING_SUMMARY_MAX_CHARS = 320
BENCHMARK_SCOPES = {"shared", "generic", "weavebench", "osworld_v2", "terminal_bench_2_1"}
INTERACTION_MODES = {"gui", "cli", "hybrid"}
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
NON_ENGLISH_SCRIPT_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
HAN_RE = re.compile(r"[\u3400-\u9fff]")
logger = logging.getLogger(__name__)

ROUTER_PROMPTS = {"en": ROUTER_PROMPT_EN, "zh": ROUTER_PROMPT_ZH}
# The former built-in prompts requested input_use/output_spec in Router cards.
# Upgrade only these exact defaults; learned or hand-edited prompts are kept.
OLD_DEFAULT_ROUTER_PROMPT_HASHES = {
    "en": "13feb075e832c1918b186b26c1bd11d4442556777a75e0db546224473040aa0f",
    "zh": "cf9a604594d33c1936cb874ae61353f8e36a679222a16671346f2d316a6979e6",
}
SUMMARY_PROMPTS = {"en": SUCCESS_SUMMARIZER_PROMPT_EN, "zh": SUCCESS_SUMMARIZER_PROMPT_ZH}
EXTRACTION_PROMPTS = {"en": EXPERIENCE_EXTRACTION_PROMPT_EN, "zh": EXPERIENCE_EXTRACTION_PROMPT_ZH}
DIAGNOSIS_PROMPTS = {"en": FAILURE_DIAGNOSIS_PROMPT_EN, "zh": FAILURE_DIAGNOSIS_PROMPT_ZH}
BUILDER_PROMPTS = {"en": SKILL_BUILDER_PROMPT_EN, "zh": SKILL_BUILDER_PROMPT_ZH}
REPAIR_PROMPTS = {"en": ROUTER_REPAIR_PROMPT_EN, "zh": ROUTER_REPAIR_PROMPT_ZH}
REVISION_REVIEW_PROMPTS = {
    "en": REVISION_PRESERVATION_REVIEW_PROMPT_EN,
    "zh": REVISION_PRESERVATION_REVIEW_PROMPT_ZH,
}


class ModelOutputError(ValueError):
    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def parse_json_result(result: EpisodeResult) -> dict[str, Any]:
    if result.status != "done":
        raise ValueError(f"model episode status: {result.status}")
    raw = str(result.metadata.get("assistant_visible_output") or visible_output(result.actions_log)).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Some provider bridges repeat the same complete assistant message.
        # Accept only identical JSON objects: differing replies are ambiguous,
        # and arbitrary trailing text must still fail closed.
        decoder = json.JSONDecoder()
        try:
            value, end = decoder.raw_decode(raw)
            remainder = raw[end:].strip()
            duplicate, duplicate_end = decoder.raw_decode(remainder)
        except (json.JSONDecodeError, TypeError):
            raise exc
        if remainder[duplicate_end:].strip() or not isinstance(value, dict) or duplicate != value:
            raise exc
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


async def model_json(agent: AgentAdapter, env: Environment, budget: EpisodeBudget, system: str, payload: dict[str, Any]) -> tuple[dict[str, Any], EpisodeResult]:
    prompt = system + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    result = await agent.run_episode(prompt, env, budget)
    try:
        return parse_json_result(result), result
    except (ValueError, TypeError) as exc:
        raw = str(result.metadata.get("assistant_visible_output") or visible_output(result.actions_log))
        raise ModelOutputError(str(exc), raw) from exc


def validate_skill(skill: dict[str, Any], *, language: str, candidate: bool = False) -> None:
    if language not in {"en", "zh"}:
        raise ValueError("Skill language must be en or zh")
    if not isinstance(skill, dict) or not ID_RE.fullmatch(str(skill.get("skill_id", ""))):
        raise ValueError("invalid skill_id")
    if not ID_RE.fullmatch(str(skill.get("category_id", ""))):
        raise ValueError("invalid category_id")
    if type(skill.get("version")) is not int or skill["version"] < 1:
        raise ValueError("invalid Skill version")
    if skill.get("content_language") != language:
        raise ValueError("Skill language mismatch")
    if skill.get("benchmark_scope") not in BENCHMARK_SCOPES:
        raise ValueError("invalid Skill benchmark scope")
    if (not isinstance(skill.get("interaction_modes"), list)
            or not skill["interaction_modes"]
            or any(mode not in INTERACTION_MODES for mode in skill["interaction_modes"])):
        raise ValueError("invalid Skill interaction modes")
    if skill.get("status") not in ({"candidate"} if candidate else {"provisional", "validated", "verified"}):
        raise ValueError("invalid Skill status")
    for name in ("selection_count", "success_count", "update_count"):
        if name in skill and (type(skill[name]) is not int or skill[name] < 0):
            raise ValueError(f"invalid {name}")
    if skill.get("success_count", 0) > skill.get("selection_count", 0):
        raise ValueError("Skill success_count exceeds selection_count")
    for name in ("problem_mechanism",):
        if not isinstance(skill.get(name), str) or not skill[name].strip():
            raise ValueError(f"missing {name}")
        if language == "en" and NON_ENGLISH_SCRIPT_RE.search(skill[name]):
            raise ValueError(f"{name} must be in English")
        if language == "zh" and not HAN_RE.search(skill[name]):
            raise ValueError(f"{name} must be in Chinese")
    # Older SkillBank entries predate routing_summary. They remain readable and
    # receive a bounded description when loaded or saved.
    if "routing_summary" in skill:
        summary = skill["routing_summary"]
        if not isinstance(summary, str) or not summary.strip() or len(summary) > ROUTING_SUMMARY_MAX_CHARS:
            raise ValueError("invalid routing_summary")
        if language == "en" and NON_ENGLISH_SCRIPT_RE.search(summary):
            raise ValueError("routing_summary must be in English")
        if language == "zh" and not HAN_RE.search(summary):
            raise ValueError("routing_summary must be in Chinese")
    for name in ("subcases", "triggers", "exclusions", "compatible_with", "conflicts_with"):
        if not isinstance(skill.get(name), list) or any(not isinstance(v, str) for v in skill[name]):
            raise ValueError(f"invalid {name}")
        if language == "en" and any(NON_ENGLISH_SCRIPT_RE.search(v) for v in skill[name]):
            raise ValueError(f"{name} must be in English")
    roles = skill.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"manager", "executor", "auditor"}:
        raise ValueError("Skill needs all three role views")
    for view in roles.values():
        if not isinstance(view, dict) or any(not isinstance(view.get(f), str) or not view[f].strip() for f in FIELDS):
            raise ValueError("incomplete role view")
        if language == "en" and any(NON_ENGLISH_SCRIPT_RE.search(view[field_name]) for field_name in FIELDS):
            raise ValueError("Skill role guidance must be in English")
        if language == "zh" and any(not HAN_RE.search(view[field_name]) for field_name in FIELDS):
            raise ValueError("Skill role guidance must be in Chinese")
    if not isinstance(skill.get("provenance"), dict) or not isinstance(skill.get("validation"), dict):
        raise ValueError("Skill needs provenance and validation records")
    if candidate and not _blank_validation(skill["validation"]):
        raise ValueError("candidate cannot self-validate")


def revision_changed_fields(old: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Identify guidance and applicability changes requiring semantic review."""
    metadata = ("problem_mechanism", "routing_summary", "subcases", "triggers",
                "exclusions", "compatible_with", "conflicts_with", "interaction_modes")
    changed = [key for key in metadata if old.get(key) != candidate.get(key)]
    for role in ("manager", "executor", "auditor"):
        for field_name in FIELDS:
            if old["roles"][role][field_name] != candidate["roles"][role][field_name]:
                changed.append(f"roles.{role}.{field_name}")
    return changed


def validate_revision_preservation_review(review: dict[str, Any], changed_fields: list[str]) -> None:
    """Fail closed unless every changed field has a reasoned preservation check."""
    checks = review.get("checks")
    if (type(review.get("preserves_reusable")) is not bool
            or type(review.get("task_specific_generalized")) is not bool
            or not isinstance(checks, list)
            or not isinstance(review.get("reason"), str) or not review["reason"].strip()):
        raise ValueError("invalid revision preservation review")
    names = [item.get("field") for item in checks if isinstance(item, dict)]
    if (len(checks) != len(changed_fields) or len(names) != len(checks)
            or any(not isinstance(name, str) for name in names)
            or set(names) != set(changed_fields)):
        raise ValueError("revision preservation review omitted a changed field")
    if (not review["preserves_reusable"] or not review["task_specific_generalized"]
            or any(type(item.get("preserved")) is not bool or not item["preserved"]
                   or type(item.get("generalized")) is not bool or not item["generalized"]
                   or not isinstance(item.get("reason"), str) or not item["reason"].strip()
                   for item in checks)):
        raise ValueError("revision weakens reusable guidance or retains task-specific instructions")


def _blank_validation(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_blank_validation(item) for item in value.values())
    if isinstance(value, list):
        return not value
    return value is None


def _fallback_routing_summary(skill: dict[str, Any], language: str) -> str:
    """Give legacy Skills a small selection card without sending role guidance."""
    def short(value: object, limit: int) -> str:
        text = str(value).strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    parts = [short(skill.get("problem_mechanism", ""), 145)]
    for field_name, limit, label in (
        ("triggers", 85, "适用" if language == "zh" else "Use when"),
        ("exclusions", 60, "排除" if language == "zh" else "Exclude"),
    ):
        values = skill.get(field_name)
        if isinstance(values, list) and values:
            parts.append(f"{label}: {short('; '.join(str(v) for v in values), limit)}")
    return " ".join(part for part in parts if part)[:ROUTING_SUMMARY_MAX_CHARS]


def candidate_from_success_spec(
    spec: dict[str, Any], *, benchmark_id: str, interaction_mode: str,
    task_id: str, run_id: str, evidence_refs: list[str], language: str = "en",
) -> dict[str, Any]:
    """Wrap the summarizer's complete role guidance without another model call."""
    category = spec.get("category_id")
    mechanism = spec.get("problem_mechanism")
    if not isinstance(category, str) or not ID_RE.fullmatch(category):
        raise ValueError("invalid success Skill category")
    if not isinstance(mechanism, str) or not mechanism.strip():
        raise ValueError("missing success Skill mechanism")
    digest = hashlib.sha256((category + "\n" + mechanism).encode("utf-8")).hexdigest()[:8]
    candidate = {
        "skill_id": f"{category[:70]}_{digest}", "category_id": category,
        "version": 1, "version_parent": None, "status": "candidate",
        "content_language": language, "benchmark_scope": benchmark_id,
        "interaction_modes": [interaction_mode],
        "problem_mechanism": mechanism,
        "routing_summary": spec.get("routing_summary") or _fallback_routing_summary(spec, language),
        "subcases": spec.get("subcases", []),
        "triggers": spec.get("triggers", []),
        "exclusions": spec.get("exclusions", []),
        "roles": spec.get("roles"),
        "compatible_with": [], "conflicts_with": [],
        "provenance": {"task_id": task_id, "run_id": run_id, "evidence_refs": evidence_refs},
        "validation": {"baseline_success": None, "candidate_success": None, "reward_delta": None, "hard_constraint_regression": None},
        "selection_count": 0, "success_count": 0, "update_count": 0,
    }
    validate_skill(candidate, language=language, candidate=True)
    return candidate


def compact_skill_cards(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("skill_id", "category_id", "version", "status", "benchmark_scope", "interaction_modes", "routing_summary", "problem_mechanism", "triggers", "exclusions")
    return [{key: skill[key] for key in fields if key in skill} for skill in skills]


def compact_trajectory(report: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    def bounded(value: Any, limit: int) -> Any:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        return value if len(rendered) <= limit else {"truncated": True, "preview": rendered[:limit]}

    rounds = [
        {
            "round_id": f"round_{item.get('round_index', 0):03d}",
            "route": item.get("next_step"),
            "manager_plan": str(item.get("plan_text") or "")[:1600],
            "executor_output": str(item.get("executor_output") or "")[:2400],
            "auditor_report": str(item.get("auditor_report") or "")[:2400],
            "harness_feedback": str(item.get("harness_feedback") or "")[:1200],
        }
        for item in report.get("rounds", [])
        if isinstance(item, dict)
    ]
    routes = []
    for record in records:
        router_input = record.get("router_input") or {}
        if not isinstance(router_input, dict):
            router_input = {}
        eligible = router_input.get("eligible_skill_cards") or []
        route = {
            "round": record.get("round"), "role": record.get("role"),
            "eligible_skill_cards": [
                {key: card.get(key) for key in ("skill_id", "version", "category_id", "routing_summary", "problem_mechanism") if key in card}
                for card in eligible if isinstance(card, dict)
            ],
            "selected_skills": record.get("selected_skills") or [],
            "router_output": bounded(record.get("router_output"), 2400),
            "router_skipped": record.get("router_skipped"),
            "router_error": str(record.get("router_error") or "")[:600],
            "injected_context": str(record.get("injected_context") or "")[:1200],
        }
        # The original task and per-round evidence are already in the
        # diagnosis input. Repeating the full routing brief for every role
        # inflated one 15-round prompt to more than 500,000 characters.
        if eligible:
            brief = router_input.get("routing_brief") or {}
            route["routing_brief"] = {
                "task_status": brief.get("task_status"),
                "current_state": str(brief.get("current_state") or "")[:800],
                "recent_actions": str(brief.get("recent_actions") or "")[:800],
                "evidence": str(brief.get("evidence") or "")[:800],
            }
        routes.append(route)
    # The original task is supplied separately to the diagnosis model.
    final = {
        key: bounded(value, 4000) if isinstance(value, (str, list, dict)) else value
        for key, value in report.items() if key not in ("task", "rounds", "final_response")
    }
    return {"final_report": final, "rounds": rounds, "router_inputs_and_outputs": routes}


def trajectory_round_index(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A navigable index of every round, without replaying its role outputs."""
    return [{
        "round_id": item["round_id"],
        "route": item.get("route"),
        "audit_header": " | ".join(str(item.get("auditor_report") or "").splitlines()[:3])[:300],
        "feedback": str(item.get("harness_feedback") or "")[:160],
    } for item in rounds]


def select_trajectory_evidence(
    rounds: list[dict[str, Any]], *, evidence_refs: list[str] | None = None, limit: int = 6,
) -> list[dict[str, Any]]:
    """Send representative and cited excerpts; retain all rounds in local logs."""
    if len(rounds) <= limit:
        return rounds
    by_id = {item["round_id"]: index for index, item in enumerate(rounds)}
    chosen: set[int] = set()
    for ref in evidence_refs or []:
        if ref in by_id and len(chosen) < limit:
            chosen.add(by_id[ref])
    for index in (0, len(rounds) - 1, len(rounds) - 2):
        if len(chosen) < limit:
            chosen.add(index)
    abnormal = [index for index, item in enumerate(rounds) if any(
        marker in line.lower()
        for line in str(item.get("auditor_report") or "").splitlines()[:3]
        for marker in ("violation", "needs_revision", "invalid", "blocked")
    ) or item.get("harness_feedback")]
    for index in (abnormal[0], abnormal[-1]) if abnormal else ():
        if len(chosen) < limit:
            chosen.add(index)
    for index in range(len(rounds) - 1, -1, -1):
        if len(chosen) >= limit:
            break
        chosen.add(index)
    return [rounds[index] for index in sorted(chosen)[:limit]]


def resolve_trajectory_round_ref(ref: Any, available_round_ids: set[str]) -> str | None:
    """Resolve a round ID or a path within that round to a recorded round."""
    if not isinstance(ref, str):
        return None
    match = re.fullmatch(r"round_(\d{1,6})(?:[/#:].*)?", ref.strip())
    if match is None:
        return None
    round_id = f"round_{int(match.group(1)):03d}"
    return round_id if round_id in available_round_ids else None


def valid_trajectory_evidence_ref(ref: Any, available_round_ids: set[str]) -> bool:
    """Accept recorded rounds and the two global evidence sections in the diagnosis input."""
    if resolve_trajectory_round_ref(ref, available_round_ids) is not None:
        return True
    return isinstance(ref, str) and re.fullmatch(
        r"(?:final_report|router_inputs_and_outputs)(?:[/#:].*)?", ref.strip()
    ) is not None


def validate_extracted_experiences(experiences: dict[str, Any], round_ids: set[str]) -> None:
    """Reject fabricated or unusable experience references before evolution."""
    for group, strategy_field in (("positive_experiences", "strategy"),
                                  ("negative_experiences", "avoidance_strategy")):
        items = experiences.get(group)
        if not isinstance(items, list):
            raise ValueError(f"{group} must be an array")
        for item in items:
            if not isinstance(item, dict) or any(
                not isinstance(item.get(field), str) or not item[field].strip()
                for field in ("mechanism", strategy_field)
            ):
                raise ValueError(f"invalid {group} entry")
            refs = item.get("evidence_refs")
            if (not isinstance(refs, list) or not refs
                    or any(resolve_trajectory_round_ref(ref, round_ids) is None for ref in refs)
                    or type(item.get("reusable")) is not bool
                    or item.get("confidence") not in {"high", "medium", "low"}):
                raise ValueError(f"{group} entry lacks recorded round evidence")


def normalize_extracted_experiences(experiences: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize one observed provider alias without weakening evidence checks.

    Some models use the positive experience field ``strategy`` for negative
    experiences as well. Preserve the raw response at the caller and record
    every normalization so the original model behavior remains inspectable.
    """
    changes: list[dict[str, str]] = []
    negatives = experiences.get("negative_experiences")
    if not isinstance(negatives, list):
        return changes
    for index, item in enumerate(negatives):
        if not isinstance(item, dict) or "strategy" not in item:
            continue
        alias = item["strategy"]
        canonical = item.get("avoidance_strategy")
        if canonical is None and isinstance(alias, str) and alias.strip():
            item["avoidance_strategy"] = alias
            item.pop("strategy")
            changes.append({"field": f"negative_experiences[{index}].strategy",
                            "normalized_to": "avoidance_strategy"})
        elif canonical == alias:
            item.pop("strategy")
            changes.append({"field": f"negative_experiences[{index}].strategy",
                            "normalized_to": "avoidance_strategy"})
        elif canonical != alias:
            raise ValueError("conflicting negative_experiences strategy fields")
    return changes


def split_assessment_artifact_refs(
    assessment: dict[str, Any], report: dict[str, Any], round_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Keep observed artifact paths distinct from recorded-round citations.

    An artifact path alone cannot establish the causal round. The existing
    round-reference requirement still applies to new Skill proposals.
    """
    refs = assessment.get("evidence_refs")
    if not isinstance(refs, list):
        return assessment, []
    # Do not search the learning output itself: the model's own assertion is
    # not independent evidence that an artifact existed.
    observed_text = json.dumps({
        "rounds": report.get("rounds"),
        "latest_auditor_report": report.get("latest_auditor_report"),
    }, ensure_ascii=False)
    has_round_citation = any(resolve_trajectory_round_ref(ref, round_ids) for ref in refs)
    kept: list[Any] = []
    artifacts: list[str] = []
    unverified: list[str] = []
    for ref in refs:
        if valid_trajectory_evidence_ref(ref, round_ids):
            kept.append(ref)
        elif isinstance(ref, str) and ref.startswith("/") and has_round_citation:
            (artifacts if ref in observed_text else unverified).append(ref)
        else:
            kept.append(ref)
    if not artifacts and not unverified:
        return assessment, []
    prior = assessment.get("artifact_refs")
    normalized = {**assessment, "evidence_refs": kept, "artifact_refs": [
        *(prior if isinstance(prior, list) else []), *artifacts,
    ]}
    if unverified:
        normalized["unverified_artifact_refs"] = unverified
    return normalized, artifacts


@dataclass
class SkillBank:
    path: Path
    skills: list[dict[str, Any]] = field(default_factory=list)
    router_prompts: dict[str, str] = field(default_factory=lambda: dict(ROUTER_PROMPTS))
    history: list[dict[str, Any]] = field(default_factory=list)
    shadowed_skills: list[dict[str, Any]] = field(default_factory=list)

    @property
    def temporary_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_temporary" + self.path.suffix)

    @classmethod
    def load(cls, path: str | Path) -> "SkillBank":
        path = Path(path)
        temporary_path = path.with_name(path.stem + "_temporary" + path.suffix)
        if not path.exists() and not temporary_path.exists():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"skills": []}
        if not isinstance(data, dict) or not isinstance(data.get("skills"), list):
            raise ValueError("invalid SkillBank")
        temporary = json.loads(temporary_path.read_text(encoding="utf-8")) if temporary_path.exists() else {"skills": []}
        if not isinstance(temporary, dict) or not isinstance(temporary.get("skills"), list):
            raise ValueError("invalid temporary SkillBank")
        stored_prompts = data.get("router_prompts", {})
        if not isinstance(stored_prompts, dict) or set(stored_prompts) - {"en", "zh"}:
            raise ValueError("SkillBank Router Prompt language must be en or zh")
        migrated_prompts = dict(stored_prompts)
        for language, prompt in stored_prompts.items():
            if (isinstance(prompt, str)
                    and hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    == OLD_DEFAULT_ROUTER_PROMPT_HASHES.get(language)):
                migrated_prompts[language] = ROUTER_PROMPTS[language]
        regular_versions = {
            skill.get("skill_id"): skill.get("version") for skill in data["skills"]
            if isinstance(skill, dict) and skill.get("status") in {"validated", "verified"}
            and isinstance(skill.get("skill_id"), str) and type(skill.get("version")) is int
        }
        temporary_skills = []
        for skill in temporary["skills"]:
            validate_skill(skill, language=skill.get("content_language", "") if isinstance(skill, dict) else "")
            if skill["status"] != "provisional":
                raise ValueError("temporary SkillBank contains validated Skill")
            if regular_versions.get(skill["skill_id"], 0) >= skill["version"]:
                continue  # Interrupted promotion left an older temporary copy.
            temporary_skills.append(skill)
        shadowed = temporary.get("shadowed_skills", [])
        if not isinstance(shadowed, list):
            raise ValueError("invalid temporary SkillBank shadowed Skills")
        for skill in shadowed:
            validate_skill(skill, language=skill.get("content_language", "") if isinstance(skill, dict) else "")
        active_temporary_ids = {skill["skill_id"] for skill in temporary_skills}
        shadowed = [skill for skill in shadowed if skill["skill_id"] in active_temporary_ids]
        bank = cls(path, [*data["skills"], *temporary_skills], {**ROUTER_PROMPTS, **migrated_prompts}, data.get("history", []), shadowed)
        if not isinstance(bank.history, list):
            raise ValueError("invalid SkillBank history")
        for entry in bank.history:
            if not isinstance(entry, dict):
                raise ValueError("invalid SkillBank history entry")
            if entry.get("kind") == "router":
                entry_language = entry.get("language")
                if entry_language not in {"en", "zh"} or not isinstance(entry.get("prompt"), str):
                    raise ValueError("invalid SkillBank Router Prompt history")
                if entry_language == "en" and NON_ENGLISH_SCRIPT_RE.search(entry["prompt"]):
                    raise ValueError("English Router Prompt history contains non-English script")
                if entry_language == "zh" and not HAN_RE.search(entry["prompt"]):
                    raise ValueError("Chinese Router Prompt history lacks Chinese text")
            elif entry.get("kind") == "skill":
                historical_skill = entry.get("skill")
                validate_skill(historical_skill, language=historical_skill.get("content_language", "") if isinstance(historical_skill, dict) else "")
                historical_skill.setdefault("update_count", historical_skill["version"] - 1)
            else:
                raise ValueError("invalid SkillBank history entry")
        if any(not isinstance(value, str) or not value.strip() for value in bank.router_prompts.values()):
            raise ValueError("invalid Router Prompt")
        if NON_ENGLISH_SCRIPT_RE.search(bank.router_prompts["en"]) or not HAN_RE.search(bank.router_prompts["zh"]):
            raise ValueError("Router Prompt language mismatch")
        ids: set[str] = set()
        if any(not isinstance(skill, dict) or not isinstance(skill.get("skill_id"), str) for skill in data["skills"]):
            raise ValueError("invalid SkillBank Skill")
        regular = {skill["skill_id"]: skill for skill in data["skills"]}
        for skill in bank.skills:
            validate_skill(skill, language=skill.get("content_language", ""))
            skill.setdefault("routing_summary", _fallback_routing_summary(skill, skill["content_language"]))
            skill.setdefault("selection_count", 0)
            skill.setdefault("success_count", 0)
            skill.setdefault("update_count", skill["version"] - 1)
            if skill["skill_id"] in ids:
                old = regular.get(skill["skill_id"])
                if (skill["status"] != "provisional" or old is None
                        or old["status"] not in {"validated", "verified"}
                        or skill["version"] != old["version"] + 1
                        or skill.get("version_parent") != old["version"]):
                    raise ValueError("duplicate Skill ID")
                if not any(s["skill_id"] == old["skill_id"] and s["version"] == old["version"] for s in bank.shadowed_skills):
                    bank.shadowed_skills.append(old)
                continue
            ids.add(skill["skill_id"])
        if bank.shadowed_skills:
            shadowed_ids = {skill["skill_id"] for skill in bank.shadowed_skills}
            bank.skills = [skill for skill in bank.skills if skill["skill_id"] not in shadowed_ids or skill["status"] == "provisional"]
        for skill in bank.shadowed_skills:
            skill.setdefault("selection_count", 0)
            skill.setdefault("success_count", 0)
            skill.setdefault("update_count", skill["version"] - 1)
        return bank

    def save(self) -> None:
        self.router_prompts = {**ROUTER_PROMPTS, **self.router_prompts}
        if set(self.router_prompts) != {"en", "zh"} or NON_ENGLISH_SCRIPT_RE.search(self.router_prompts["en"]) or not HAN_RE.search(self.router_prompts["zh"]):
            raise ValueError("SkillBank Router Prompt language mismatch")
        for skill in self.skills:
            skill.setdefault("routing_summary", _fallback_routing_summary(skill, skill.get("content_language", "")))
            skill.setdefault("selection_count", 0)
            skill.setdefault("success_count", 0)
            skill.setdefault("update_count", skill["version"] - 1)
            validate_skill(skill, language=skill.get("content_language", ""))
        for skill in self.shadowed_skills:
            validate_skill(skill, language=skill.get("content_language", ""))
            skill.setdefault("selection_count", 0)
            skill.setdefault("success_count", 0)
            skill.setdefault("update_count", skill["version"] - 1)
        for entry in self.history:
            if isinstance(entry, dict) and entry.get("kind") == "skill" and isinstance(entry.get("skill"), dict):
                historical_skill = entry["skill"]
                historical_skill.setdefault("update_count", historical_skill["version"] - 1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = [s for s in self.skills if s["status"] == "provisional"]
        if temporary or self.shadowed_skills:
            temporary_tmp = self.temporary_path.with_suffix(self.path.suffix + ".tmp")
            temporary_tmp.write_text(json.dumps({"schema_version": 1, "skills": temporary, "shadowed_skills": self.shadowed_skills}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary_tmp.replace(self.temporary_path)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"schema_version": 1, "skills": [s for s in self.skills if s["status"] in {"validated", "verified"}], "router_prompts": self.router_prompts, "history": self.history}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        if not temporary and not self.shadowed_skills and self.temporary_path.exists():
            self.temporary_path.unlink()

    def eligible(
        self, role: str, language: str, candidate: dict[str, Any] | list[dict[str, Any]] | None = None,
        *, benchmark_id: str = "generic", interaction_mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        if language not in {"en", "zh"}:
            raise ValueError("Skill routing language must be en or zh")
        if role not in ROLES:
            raise ValueError("unknown role")
        result = []
        role_mode = "gui" if role.startswith("gui_") else "cli" if role.startswith("cli_") else interaction_mode
        candidates = candidate if isinstance(candidate, list) else [candidate] if candidate else []
        candidate_ids = {skill["skill_id"] for skill in candidates}
        active = [s for s in self.skills if s["skill_id"] not in candidate_ids]
        for skill in [*active, *candidates]:
            if skill["status"] not in ({"provisional", "validated", "verified", "candidate"} if candidate else {"provisional", "validated", "verified"}):
                continue
            if skill["benchmark_scope"] not in ("shared", benchmark_id):
                continue
            if not (
                role == "manager" and interaction_mode == "hybrid"
                or role_mode in skill["interaction_modes"]
                or interaction_mode in skill["interaction_modes"]
            ):
                continue
            # The Router receives only a concise selection description. The
            # chosen role view is loaded from the bank after ID/version checks.
            role_view = skill["roles"][VIEWS[role]]
            result.append({key: skill[key] for key in (
                "skill_id", "category_id", "version", "status", "content_language",
                "compatible_with", "conflicts_with", "benchmark_scope", "interaction_modes",
            )} | {
                "routing_summary": skill.get("routing_summary") or _fallback_routing_summary(skill, skill["content_language"]),
                "role_token_estimate": context_token_estimate(render_skill_context([{
                    "skill_id": skill["skill_id"], "version": skill["version"],
                    "role_view": role_view,
                }], language)),
            })
        return result


def commit_validated_candidate(
    bank: SkillBank,
    kind: str,
    candidate: dict[str, Any] | str,
    *,
    language: str,
    validation: dict[str, Any],
    save: bool = True,
) -> None:
    """Commit one independently verified object; retain the previous version."""
    if language not in {"en", "zh"}:
        raise ValueError("Skill learning language must be en or zh")
    delta = validation.get("reward_delta")
    if (not validation.get("accepted") or not validation.get("effective")
            or not validation.get("candidate_success")
            or validation.get("hard_constraint_regression")
            or not isinstance(delta, (int, float)) or delta <= 0):
        raise ValueError("candidate has not passed validation")
    if kind == "router":
        if not isinstance(candidate, str) or not candidate.strip() or (language == "en" and NON_ENGLISH_SCRIPT_RE.search(candidate)) or (language == "zh" and not HAN_RE.search(candidate)):
            raise ValueError("invalid Router candidate")
        bank.history.append({"kind": "router", "language": language, "prompt": bank.router_prompts[language]})
        bank.router_prompts[language] = candidate
    elif kind == "skill":
        validate_skill(candidate, language=candidate.get("content_language", "") if isinstance(candidate, dict) else "", candidate=True)
        active = next((s for s in bank.skills if s["skill_id"] == candidate["skill_id"]), None)
        previous_versions = [s for s in bank.shadowed_skills if s["skill_id"] == candidate["skill_id"]]
        old = max(previous_versions, key=lambda s: s["version"]) if previous_versions else active
        if old is None:
            if candidate["version"] != 1 or candidate.get("version_parent") is not None:
                raise ValueError("invalid new Skill version")
        elif (old["status"] == "provisional" and candidate["version"] == old["version"]
              and candidate.get("version_parent") == old.get("version_parent")
              and candidate["category_id"] == old["category_id"]
              and candidate["benchmark_scope"] == old["benchmark_scope"]):
            pass
        elif (candidate["version"] != old["version"] + 1
              or candidate.get("version_parent") != old["version"]
              or candidate["category_id"] != old["category_id"]
              or candidate["benchmark_scope"] != old["benchmark_scope"]):
            raise ValueError("invalid Skill revision lineage")
        if old is None:
            candidate["update_count"] = 0
        elif old["status"] == "provisional" and candidate["version"] == old["version"]:
            # Finalizing a staged revision must not count the same update twice.
            candidate["update_count"] = old.get("update_count", old["version"] - 1)
        else:
            candidate["update_count"] = old.get("update_count", old["version"] - 1) + 1
        # Historical versions keep their own observed counts. The revised
        # version starts a separate count; staging and replay may already have
        # recorded selections for that exact version.
        for previous in previous_versions:
            bank.history.append({"kind": "skill", "skill": copy.deepcopy(previous)})
        if old is not None and not previous_versions and candidate["version"] > old["version"]:
            bank.history.append({"kind": "skill", "skill": copy.deepcopy(old)})
        new_version_counts = {
            name: active.get(name, 0) if active is not None and active["version"] == candidate["version"] else 0
            for name in ("selection_count", "success_count")
        }
        if old is not None:
            bank.skills = [s for s in bank.skills if s["skill_id"] != candidate["skill_id"]]
            bank.shadowed_skills = [s for s in bank.shadowed_skills if s["skill_id"] != candidate["skill_id"]]
        candidate.update(new_version_counts)
        candidate["status"] = "validated"
        candidate["validation"] = validation
        bank.skills.append(candidate)
    else:
        raise ValueError("unknown candidate kind")
    if save:
        bank.save()


def stage_skill_candidate(bank: SkillBank, candidate: dict[str, Any], *, save: bool = True) -> dict[str, Any]:
    """Store generated guidance in the temporary bank until replay validates it."""
    validate_skill(candidate, language=candidate.get("content_language", "") if isinstance(candidate, dict) else "", candidate=True)
    old = next((s for s in bank.skills if s["skill_id"] == candidate["skill_id"]), None)
    if old is None:
        if candidate["version"] != 1 or candidate.get("version_parent") is not None:
            raise ValueError("invalid new Skill version")
    elif (candidate["version"] != old["version"] + 1
          or candidate.get("version_parent") != old["version"]
          or candidate["category_id"] != old["category_id"]):
        raise ValueError("invalid Skill revision lineage")
    staged = copy.deepcopy(candidate)
    staged["status"] = "provisional"
    staged["validation"] = {"replay": "pending"}
    staged["selection_count"] = 0
    staged["success_count"] = 0
    staged["update_count"] = old.get("update_count", old["version"] - 1) + 1 if old is not None else 0
    if old is not None:
        bank.shadowed_skills.append(old)
        bank.skills = [s for s in bank.skills if s["skill_id"] != old["skill_id"]]
    bank.skills.append(staged)
    if save:
        bank.save()
    return staged


def record_selected_skill_outcomes(bank: SkillBank, routes: list[dict[str, Any]], *, success: bool) -> None:
    """Count each selected Skill version once per task, regardless of route count."""
    selected = {
        (item.get("skill_id"), item.get("version"))
        for route in routes if isinstance(route, dict)
        for item in (route.get("selected_skills") if isinstance(route.get("selected_skills"), list) else [])
        if isinstance(item, dict) and isinstance(item.get("skill_id"), str) and type(item.get("version")) is int
    }
    changed = False
    for skill in bank.skills:
        if (skill["skill_id"], skill["version"]) in selected:
            skill["selection_count"] = skill.get("selection_count", 0) + 1
            skill["success_count"] = skill.get("success_count", 0) + int(success)
            changed = True
    if changed:
        bank.save()


def render_skill_context(selected: list[dict[str, Any]], language: str) -> str:
    if language not in {"en", "zh"}:
        raise ValueError("Skill injection language must be en or zh")
    if not selected:
        return ""
    lines = (
        ["[角色专用 Skill 指导 — 辅助]", "Skill 只补充角色指令，不覆盖任务契约、角色权限、完成守卫或输出协议。"]
        if language == "zh" else
        ["[ROLE-SPECIFIC SKILL GUIDANCE — AUXILIARY]", "These Skills supplement role instructions. They do not override the task contract, role permissions, completion guards or output protocol."]
    )
    labels = ("输入利用", "输出规格", "指导", "避免") if language == "zh" else ("Input use", "Output specification", "Guidance", "Avoid")
    seen: set[tuple[str, str]] = set()
    for skill in selected:
        lines.append(f"Skill: {skill['skill_id']}@{skill['version']}")
        for label, field_name in zip(labels, FIELDS):
            value = skill["role_view"][field_name].strip()
            if (field_name, value) not in seen:
                lines.extend((f"{label}:", value))
                seen.add((field_name, value))
    lines.append("[角色专用 Skill 指导结束]" if language == "zh" else "[END ROLE-SPECIFIC SKILL GUIDANCE]")
    return "\n\n".join(lines)


def validate_selection(output: dict[str, Any], eligible: list[dict[str, Any]], *, budget: int, target: str | None = None) -> list[dict[str, Any]]:
    choices = output.get("selected_skills")
    if not isinstance(choices, list):
        raise ValueError("selected_skills must be an array")
    by_id = {s["skill_id"]: s for s in eligible}
    selected = []
    ids: set[str] = set()
    for choice in choices:
        if not isinstance(choice, dict) or choice.get("skill_id") not in by_id:
            raise ValueError("unknown Skill ID")
        skill = by_id[choice["skill_id"]]
        if choice.get("version") != skill["version"] or skill["skill_id"] in ids:
            raise ValueError("Skill version mismatch or duplicate")
        ids.add(skill["skill_id"])
        selected.append(skill)
    if target is not None and target not in ids:
        raise ValueError("candidate was not selected")
    for skill in selected:
        if ids.intersection(skill["conflicts_with"]):
            raise ValueError("conflicting Skills")
    return selected


def supported_router_error(records: list[dict[str, Any]], assessment: dict[str, Any]) -> bool:
    """Confirm that a proposed Router error has a matching recorded selection."""
    error_type = assessment.get("router_error_type")
    target = assessment.get("target_skill_id")
    return any(
        record.get("router_output") and (
            error_type == "omission" and target
            and any(card.get("skill_id") == target for card in (record.get("router_input") or {}).get("eligible_skill_cards", []))
            and not any(item.get("skill_id") == target for item in (record.get("selected_skills") or []))
            or error_type == "wrong_skill" and bool(record.get("selected_skills"))
            or error_type == "conflict" and len(record.get("selected_skills") or []) >= 2
        ) for record in records
    )


def context_token_estimate(text: str) -> int:
    """Conservative admission estimate, never used for task reward."""
    return max((len(text) + 3) // 4, (len(text.encode("utf-8")) + 2) // 3)


def validate_single_skill_budget(skill: dict[str, Any], *, language: str, token_budget: int) -> None:
    """A candidate must fit by itself before Router can ever inject it."""
    for role, role_view in skill["roles"].items():
        context = render_skill_context([{
            "skill_id": skill["skill_id"], "version": skill["version"], "role_view": role_view,
        }], language)
        if context_token_estimate(context) > token_budget:
            raise ValueError(f"Skill {role} view exceeds the per-role token budget")


def token_usage(result: EpisodeResult) -> int | None:
    """Only provider reported usage counts; no character-based substitute."""
    sources = [result.metadata, result.metadata.get("usage")]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("total_tokens", "totalTokens"):
            value = source.get(key)
            if type(value) is int and value >= 0:
                return value
        inp = source.get("input_tokens", source.get("inputTokens"))
        out = source.get("output_tokens", source.get("outputTokens"))
        if type(inp) is int and type(out) is int and inp >= 0 and out >= 0:
            return inp + out
        # Harbor's Claude Code bridge already includes cached input in
        # n_input_tokens, so do not add n_cache_tokens again.
        inp, out = source.get("n_input_tokens"), source.get("n_output_tokens")
        if type(inp) is int and type(out) is int and inp >= 0 and out >= 0:
            return inp + out
    # CLI adapters preserve provider JSONL in actions_log. Parse only known
    # provider usage envelopes; never infer tokens from text length.
    codex_turns: list[int] = []
    opencode_steps: list[int] = []
    opencode_ids: set[str] = set()
    for line in result.actions_log.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("type") == "turn.completed":
            usage = record.get("usage", {})
            if isinstance(usage, dict) and type(usage.get("input_tokens")) is int and type(usage.get("output_tokens")) is int:
                codex_turns.append(usage["input_tokens"] + usage["output_tokens"])
        if record.get("type") == "result" and isinstance(record.get("usage"), dict):
            usage = record["usage"]
            if type(usage.get("input_tokens")) is int and type(usage.get("output_tokens")) is int:
                return usage["input_tokens"] + usage["output_tokens"]
        if record.get("type") == "step_finish" and isinstance(record.get("part"), dict):
            part = record["part"]
            step_id = part.get("id")
            tokens = part.get("tokens")
            if isinstance(tokens, dict) and isinstance(step_id, str) and step_id not in opencode_ids:
                total = tokens.get("total")
                inp, out = tokens.get("input"), tokens.get("output")
                if type(total) is int and total >= 0:
                    opencode_steps.append(total)
                    opencode_ids.add(step_id)
                elif type(inp) is int and type(out) is int and inp >= 0 and out >= 0:
                    opencode_steps.append(inp + out)
                    opencode_ids.add(step_id)
    if codex_turns:
        return sum(codex_turns)
    if opencode_steps:
        return sum(opencode_steps)
    return None


def task_success(report: dict[str, Any], language: str = "en") -> bool:
    if not (report.get("completion_satisfied") and report.get("status") == "complete"):
        return False
    text = report.get("latest_auditor_report")
    if not isinstance(text, str) or not text.strip():
        return False
    audit = parse_audit_report(text, int(report.get("rounds_run") or 0), language=language)
    return audit.status == "complete" and audit.integrity_status == "clean" and audit.contract_audit_status == "aligned"


def task_reward(report: dict[str, Any], tokens: int | None, *, reference_tokens: int, beta: float = 0.1, language: str = "en", success_override: bool | None = None) -> float | None:
    if type(tokens) is not int or tokens < 0 or reference_tokens <= 0:
        return None
    success = task_success(report, language) if success_override is None else success_override
    return float(success) - beta * min(1.0, tokens / reference_tokens)


def accept_candidate(baseline: dict[str, Any], candidate: dict[str, Any], *, baseline_tokens: int | None, candidate_tokens: int | None, reference_tokens: int, effective: bool, hard_constraint_regression: bool, language: str = "en", baseline_success_override: bool | None = None, candidate_success_override: bool | None = None) -> tuple[bool, float | None]:
    base = task_reward(baseline, baseline_tokens, reference_tokens=reference_tokens, language=language, success_override=baseline_success_override)
    cand = task_reward(candidate, candidate_tokens, reference_tokens=reference_tokens, language=language, success_override=candidate_success_override)
    delta = cand - base if base is not None and cand is not None else None
    candidate_success = (task_success(candidate, language) if candidate_success_override is None
                         else candidate_success_override)
    accepted = bool(effective and candidate_success and not hard_constraint_regression and delta is not None and delta > 0)
    return accepted, delta

Replay = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class SkillSession:
    bank: SkillBank
    language: str
    agent: AgentAdapter
    env: Environment
    budget: EpisodeBudget
    log_dir: Path
    token_budget: int = 2000
    candidate: dict[str, Any] | list[dict[str, Any]] | None = None
    router_prompt_override: str | None = None
    benchmark_id: str = "generic"
    interaction_mode: str = "hybrid"
    dataset_id: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)
    token_counts: list[int | None] = field(default_factory=list)
    bank_snapshot: list[dict[str, Any]] = field(default_factory=list, init=False)
    outcome_recorded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.language not in {"en", "zh"}:
            raise ValueError("Skill session language must be en or zh")
        if type(self.token_budget) is not int or self.token_budget < 1:
            raise ValueError("Skill token budget must be a positive integer")
        if self.benchmark_id not in BENCHMARK_SCOPES - {"shared"} or self.interaction_mode not in INTERACTION_MODES:
            raise ValueError("invalid Skill routing context")
        self.bank_snapshot = copy.deepcopy(self.bank.skills)

    async def route(self, role: str, base_prompt: str, round_index: int, routing_brief: dict[str, Any] | None = None) -> str:
        brief = routing_brief or {"task_goal": base_prompt[:2000], "current_state": "", "recent_actions": [], "evidence": [], "task_status": "running"}
        eligible = self.bank.eligible(
            role, self.language, self.candidate,
            benchmark_id=self.benchmark_id, interaction_mode=self.interaction_mode,
        )
        if not eligible:
            self.records.append({"round": round_index, "role": role, "role_base_prompt": base_prompt, "router_input": {"benchmark_id": self.benchmark_id, "dataset_id": self.dataset_id, "interaction_mode": self.interaction_mode, "routing_brief": brief, "eligible_skill_cards": []}, "router_skipped": "no_eligible_skill_cards", "selected_skills": [], "injected_context": ""})
            self._save_records()
            return ""
        candidates = self.candidate if isinstance(self.candidate, list) else [self.candidate] if self.candidate else []
        target_ids = [skill["skill_id"] for skill in candidates]
        target = target_ids[0] if len(target_ids) == 1 else None
        payload = {
            "mode": "candidate_validation" if target_ids else "production",
            "prompt_language": self.language,
            "role": role,
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.dataset_id,
            "interaction_mode": self.interaction_mode,
            "routing_brief": brief,
            "eligible_skill_cards": eligible,
            "validation_target_skill_id": target,
            "validation_target_skill_ids": target_ids,
            "token_budget": self.token_budget,
            "validation_history": [
                {"skill_id": s["skill_id"], "status": s["status"]} for s in eligible
            ],
        }
        record: dict[str, Any] = {"round": round_index, "role": role, "role_base_prompt": base_prompt, "router_input": payload, "selected_skills": [], "injected_context": ""}
        try:
            card_contract = (
                "Router cards contain only routing_summary and metadata. Select by that description; "
                "role guidance is loaded after selection. In candidate validation, each applicable "
                "validation target must be selected during the task for the batch to pass."
                if self.language == "en" else
                "Router 候选卡片只包含 routing_summary 和元数据；根据简述选择，选中后才加载角色指导。"
                "候选验证时，本次任务中每个适用的候选 Skill 都必须至少被选中一次，整组才能通过。"
            )
            router_prompt = (self.router_prompt_override or self.bank.router_prompts[self.language]) + "\n\n" + card_contract + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            result = await self.agent.run_episode(router_prompt, self.env, self.budget)
            self.token_counts.append(token_usage(result))
            record["router_raw_output"] = str(result.metadata.get("assistant_visible_output") or visible_output(result.actions_log))
            output = parse_json_result(result)
            record["router_output"] = output
            selected_cards = validate_selection(output, eligible, budget=self.token_budget, target=target)
            active = {skill["skill_id"]: skill for skill in [*self.bank.skills, *candidates]}
            selected = [
                {"skill_id": card["skill_id"], "version": card["version"], "role_view": active[card["skill_id"]]["roles"][VIEWS[role]]}
                for card in selected_cards
            ]
            context = render_skill_context(selected, self.language)
            if context_token_estimate(context) > self.token_budget:
                raise ValueError("Skill token budget exceeded")
            record["selected_skills"] = [{"skill_id": s["skill_id"], "version": s["version"]} for s in selected]
            record["injected_context"] = context
            record["injected_context_token_estimate"] = context_token_estimate(context)
            return context
        except Exception as exc:
            record["router_error"] = str(exc)
            # During validation a failed injection makes the replay ineffective.
            return ""
        finally:
            self.records.append(record)
            self._save_records()

    def record_episode(self, result: EpisodeResult) -> None:
        self.token_counts.append(token_usage(result))

    def record_task_outcome(self, report: dict[str, Any]) -> None:
        if self.outcome_recorded:
            return
        record_selected_skill_outcomes(self.bank, self.records, success=task_success(report, self.language))
        self.outcome_recorded = True

    def record_injected_prompt(self, role: str, round_index: int, prompt: str) -> None:
        for record in reversed(self.records):
            if record["role"] == role and record["round"] == round_index:
                record["actual_role_prompt"] = prompt
                self._save_records()
                return

    def total_tokens(self) -> int | None:
        return sum(self.token_counts) if all(v is not None for v in self.token_counts) else None

    def _save_records(self) -> None:
        try:
            _ensure_dir_nofollow(self.log_dir)
            _atomic_bytes_write(self.log_dir / "skill_routes.json", (json.dumps(self.records, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        except OSError:
            logger.warning("Could not save Skill routing records", exc_info=True)

    def _save_candidate(self, kind: str, candidate: Any, assessment: dict[str, Any], validation: Any) -> None:
        _ensure_dir_nofollow(self.log_dir)
        payload = {"kind": kind, "candidate": candidate, "assessment": assessment, "validation": validation}
        _atomic_bytes_write(self.log_dir / "skill_candidate.json", (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    async def finish(self, report: dict[str, Any], *, replay: Replay | None = None, reference_tokens: int = 100_000) -> dict[str, Any]:
        """Learn candidate Skills; commit each only after a full-task replay.

        replay receives {kind, candidate, mode}. It must restore the original
        workspace/environment and return the complete run report, token count,
        effective flag and hard_constraint_regression flag.
        """
        self._save_records()
        self.record_task_outcome(report)
        status: dict[str, Any] = {
            "task_tokens": self.total_tokens(), "token_usage_missing": self.total_tokens() is None,
            "baseline_token_usage": {"total_tokens": self.total_tokens(), "missing": self.total_tokens() is None},
            "action": None, "validation": "not_requested", "learning_phase": "eligibility",
            "diagnosis_attempted": False, "experience_extraction_attempted": False,
        }
        success = task_success(report, self.language)
        rounds = report.get("rounds")
        has_trajectory = isinstance(rounds, list) and any(
            isinstance(item, dict) and any(
                isinstance(item.get(key), str) and item[key].strip()
                for key in ("plan_text", "executor_output", "auditor_report")
            ) for item in rounds
        )
        if not success and not has_trajectory:
            status["learning_phase"] = "skipped"
            status["skip_reason"] = "no_analyzable_execution_trajectory"
            return status
        task_id = hashlib.sha256((self.dataset_id + "\n" + str(report.get("task", ""))).encode("utf-8")).hexdigest()
        if success:
            injected = {item["skill_id"] for record in self.records if record.get("injected_context") for item in record["selected_skills"]}
            changed = False
            for skill in self.bank.skills:
                if skill["skill_id"] not in injected:
                    continue
                if skill.get("provenance", {}).get("task_id") == task_id:
                    continue
                # These are observational task outcomes. Without a comparable
                # cached baseline they do not establish causal benefit.
                evidence = skill.setdefault("validation", {}).setdefault("observational_task_ids", [])
                if task_id not in evidence:
                    evidence.append(task_id)
                    changed = True
            if changed:
                self.bank.save()
        trajectory = compact_trajectory(report, self.records)
        round_index = trajectory_round_index(trajectory["rounds"])
        key_evidence = select_trajectory_evidence(trajectory["rounds"])
        relevant_skills = [
            skill for skill in self.bank_snapshot
            if skill["benchmark_scope"] in ("shared", self.benchmark_id)
            and (self.interaction_mode == "hybrid" or self.interaction_mode in skill["interaction_modes"])
        ]
        skill_index = compact_skill_cards(relevant_skills)
        injected_guidance = []
        seen_injections = set()
        for record in self.records:
            context = record.get("injected_context")
            if not isinstance(context, str) or not context.strip():
                continue
            key = (record.get("role"), context)
            if key not in seen_injections:
                seen_injections.add(key)
                injected_guidance.append({"role": key[0],
                                          "selected_skills": record.get("selected_skills", []),
                                          "injected_context": context})
        experiences: dict[str, Any] = {"positive_experiences": [], "negative_experiences": []}
        if has_trajectory:
            # Cover every recorded round without repeating full tool logs or
            # large role transcripts in the evolution decision prompt.
            extraction_input = {
                "original_task": str(report.get("task") or "")[:3000],
                "final_report": trajectory["final_report"],
                "task_success": success,
                "all_rounds": [{
                    "round_id": item["round_id"], "route": item.get("route"),
                    "manager_plan": str(item.get("manager_plan") or "")[:1000],
                    "executor_output": str(item.get("executor_output") or "")[:1200],
                    "auditor_report": str(item.get("auditor_report") or "")[:1200],
                    "harness_feedback": str(item.get("harness_feedback") or "")[:400],
                } for item in trajectory["rounds"]],
                "routing_record": [{
                    "round": route.get("round"), "role": route.get("role"),
                    "eligible_skill_ids": [card.get("skill_id") for card in route.get("eligible_skill_cards", [])],
                    "selected_skills": route.get("selected_skills"),
                    "router_output": route.get("router_output"),
                } for route in trajectory["router_inputs_and_outputs"]],
            }
            status["learning_phase"] = "experience_extraction"
            status["experience_extraction_attempted"] = True
            try:
                experiences, extraction_result = await model_json(
                    self.agent, self.env, self.budget, EXTRACTION_PROMPTS[self.language], extraction_input,
                )
                status["experience_extraction_token_usage"] = token_usage(extraction_result)
            except Exception as exc:
                status["error_stage"] = "experience_extraction"
                status["learning_error"] = str(exc)
                if isinstance(exc, ModelOutputError):
                    status["raw_model_output"] = exc.raw_output
                return status
            try:
                raw_experiences = copy.deepcopy(experiences)
                normalizations = normalize_extracted_experiences(experiences)
                if normalizations:
                    status["extracted_experiences_raw"] = raw_experiences
                    status["experience_normalizations"] = normalizations
                status["extracted_experiences"] = experiences
                validate_extracted_experiences(
                    experiences, {item["round_id"] for item in trajectory["rounds"]},
                )
            except ValueError as exc:
                status.setdefault("extracted_experiences", experiences)
                status["error_stage"] = "experience_validation"
                status["learning_error"] = str(exc)
                return status
        else:
            status["experience_extraction_skip_reason"] = "no_recorded_rounds"
            status["extracted_experiences"] = experiences
        reliable_experience = any(
            item["reusable"] and item["confidence"] in {"high", "medium"}
            for group in ("positive_experiences", "negative_experiences")
            for item in experiences[group]
        )
        status["reliable_experience_found"] = reliable_experience
        assessment_input = (
            {
                "benchmark_id_and_interaction_mode": {"benchmark_id": self.benchmark_id, "interaction_mode": self.interaction_mode},
                "trajectory_batch": [{
                    "task_id": task_id, "task": report.get("task"),
                    "final_report": trajectory["final_report"],
                    "compact_trajectory_summary": round_index,
                    "key_evidence_excerpts": key_evidence,
                    "selected_skill_ids_versions_and_injections": trajectory["router_inputs_and_outputs"],
                }],
                "relevant_skill_cards": skill_index,
                "extracted_experiences": experiences,
                "actual_skill_injections": [
                    {key: record.get(key) for key in ("round", "role", "selected_skills")}
                    for record in self.records if record.get("injected_context")
                ],
                "injected_skill_guidance": injected_guidance,
                "language": self.language,
            }
            if success else {
                "original_task": report.get("task"),
                "final_report": trajectory["final_report"],
                "trajectory_index": round_index,
                "relevant_trajectory_evidence": key_evidence,
                "router_inputs_and_outputs": trajectory["router_inputs_and_outputs"],
                "actual_skill_injections": [
                    {key: record.get(key) for key in ("round", "role", "selected_skills")}
                    for record in self.records if record.get("injected_context")
                ],
                "injected_skill_guidance": injected_guidance,
                "skillbank_index_at_run_start": skill_index,
                "extracted_experiences": experiences,
                "runtime_and_environment_errors": {
                    "abort_reason": report.get("abort_reason"),
                    "failure_reason": report.get("failure_reason"),
                    "error": report.get("error"),
                },
            }
        )
        prompt = SUMMARY_PROMPTS[self.language] if success else DIAGNOSIS_PROMPTS[self.language]
        status["learning_phase"] = "success_summary" if success else "failure_diagnosis"
        status["diagnosis_attempted"] = not success
        try:
            assessment, _ = await model_json(self.agent, self.env, self.budget, prompt, assessment_input)
        except Exception as exc:
            status["error_stage"] = status["learning_phase"]
            status["learning_error"] = str(exc)
            if isinstance(exc, ModelOutputError):
                status["raw_model_output"] = exc.raw_output
            return status
        status["assessment_raw"] = copy.deepcopy(assessment)
        assessment, artifact_refs = split_assessment_artifact_refs(
            assessment, report, {item["round_id"] for item in trajectory["rounds"]},
        )
        if artifact_refs:
            status["assessment_artifact_refs"] = artifact_refs
        if assessment.get("unverified_artifact_refs"):
            status["unverified_artifact_refs"] = assessment.pop("unverified_artifact_refs")
        status["assessment"] = assessment
        status["assessment_reason"] = next((
            assessment[key] for key in ("failure_mechanism", "novelty_reason", "reason")
            if isinstance(assessment.get(key), str) and assessment[key].strip()
        ), None)
        action = assessment.get("action")
        allowed = ({"NO_NEW_SKILL", "CREATE_NEW_SKILL", "REVISE_SKILL",
                    "UPDATE_ROUTER_PROMPT", "INSUFFICIENT_EVIDENCE"} if success else
                   {"UPDATE_ROUTER_PROMPT", "CREATE_NEW_SKILL", "REVISE_SKILL",
                    "NO_CHANGE", "INSUFFICIENT_EVIDENCE"})
        if action not in allowed:
            status["error_stage"] = "assessment_validation"
            status["learning_error"] = "invalid assessment action"
            return status
        status["model_action"] = action
        no_eligible_skill = bool(self.records) and all(
            isinstance(record.get("router_input"), dict)
            and not record["router_input"].get("eligible_skill_cards")
            for record in self.records
        )
        no_skill_injected = bool(self.records) and all(not record.get("injected_context") for record in self.records)
        infrastructure_failure = report.get("status") == "failed" and bool(report.get("error") or report.get("failure_reason"))
        router_issue = any(record.get("router_error") for record in self.records) or supported_router_error(self.records, assessment)
        router_uncovered_need = any(
            isinstance(record.get("router_output"), dict)
            and isinstance(record["router_output"].get("uncovered_need"), str)
            and bool(record["router_output"]["uncovered_need"].strip())
            for record in self.records
        )
        # An empty Router selection with available cards is not by itself a
        # missing capability: it may be a Router omission. Require an actual
        # uncovered-need signal or an explicit evidence-backed diagnosis.
        coverage_gap = no_eligible_skill or router_uncovered_need or assessment.get("coverage_gap") is True
        status["coverage_gap_detected"] = coverage_gap
        status["no_eligible_skill"] = no_eligible_skill
        status["no_skill_injected"] = no_skill_injected
        status["router_uncovered_need"] = router_uncovered_need
        if (has_trajectory and reliable_experience and not infrastructure_failure and coverage_gap and not router_issue
                and action in {"NO_NEW_SKILL", "NO_CHANGE", "INSUFFICIENT_EVIDENCE", "REVISE_SKILL", "UPDATE_ROUTER_PROMPT"}):
            # The model's original decision remains in status["assessment"].
            # A documented coverage gap must reach the Builder even when the
            # assessment incorrectly chooses a no-update action.
            shown_round_ids = {item["round_id"] for item in key_evidence}
            refs = [ref for ref in (assessment.get("evidence_refs") or [])
                    if isinstance(ref, str) and ref in shown_round_ids]
            if not refs:
                refs = [item["round_id"] for item in key_evidence]
            revisions = assessment.get("revision_targets", [])
            if action == "REVISE_SKILL" and any(
                record.get("injected_context") and any(
                    item.get("skill_id") == assessment.get("target_skill_id")
                    for item in record.get("selected_skills", [])) for record in self.records
            ):
                revisions = [*(revisions if isinstance(revisions, list) else []), {
                    "target_skill_id": assessment.get("target_skill_id"),
                    "failure_mechanism": assessment.get("failure_mechanism"),
                    "evidence_refs": assessment.get("evidence_refs"),
                }]
            elif action == "REVISE_SKILL":
                status["revision_skipped_reason"] = "diagnosed Skill was not injected"
            assessment = {
                **assessment, "action": "CREATE_NEW_SKILL",
                "evidence_refs": refs[:1],
                "revision_targets": revisions,
                "candidate_spec": None,
                "policy_override_reason": (
                    "No eligible Skill was available for this task; create a reusable Skill from the cited evidence."
                    if no_eligible_skill else
                    "A Skill coverage gap was recorded; create a complementary Skill from the cited evidence."
                ),
            }
            status["policy_override"] = {
                "from_action": action, "to_action": "CREATE_NEW_SKILL",
                "reason": "no_eligible_skill" if no_eligible_skill else
                    "router_uncovered_need" if router_uncovered_need else "diagnosed_coverage_gap",
            }
            action = "CREATE_NEW_SKILL"
        status["action"] = action
        status["learning_phase"] = "assessment_complete"
        if has_trajectory and action in {"CREATE_NEW_SKILL", "REVISE_SKILL"} and not reliable_experience:
            status["policy_override"] = {
                "from_action": action, "to_action": "INSUFFICIENT_EVIDENCE",
                "reason": "no_reliable_extracted_experience",
            }
            status["action"] = "INSUFFICIENT_EVIDENCE"
            status["grounding_skip_reason"] = "no_reliable_extracted_experience_for_skill_update"
            return status
        if action not in {"UPDATE_ROUTER_PROMPT", "CREATE_NEW_SKILL", "REVISE_SKILL"}:
            if not success and not status["assessment_reason"]:
                status["error_stage"] = "assessment_validation"
                status["learning_error"] = "diagnosis did not explain the no-update action"
            return status
        if not isinstance(assessment.get("evidence_refs"), list) or not assessment["evidence_refs"]:
            status["error_stage"] = "evidence_validation"
            status["learning_error"] = "update needs concrete evidence references"
            return status
        available_round_ids = {item["round_id"] for item in trajectory["rounds"]}
        revision_requests = assessment.get("revision_targets", []) if action == "CREATE_NEW_SKILL" else []
        if not isinstance(revision_requests, list):
            status.update(error_stage="assessment_validation", learning_error="revision_targets must be a list")
            return status
        revision_targets = {}
        for request in revision_requests:
            if not isinstance(request, dict) or not isinstance(request.get("target_skill_id"), str):
                status.update(error_stage="assessment_validation", learning_error="invalid revision target")
                return status
            target_id = request["target_skill_id"]
            old = next((s for s in self.bank.skills if s["skill_id"] == target_id), None)
            injected = any(record.get("injected_context") and any(
                item.get("skill_id") == target_id and item.get("version") == old["version"]
                for item in record.get("selected_skills", [])) for record in self.records) if old else False
            refs = request.get("evidence_refs")
            if (target_id in revision_targets or old is None or not injected
                    or not isinstance(refs, list) or not refs
                    or any(not valid_trajectory_evidence_ref(ref, available_round_ids) for ref in refs)
                    or not isinstance(request.get("failure_mechanism"), str)
                    or not request["failure_mechanism"].strip()):
                status.update(error_stage="evidence_validation",
                              learning_error=f"revision target {target_id} lacks injected guidance or concrete evidence")
                return status
            revision_targets[target_id] = old
        status["revision_target_ids"] = list(revision_targets)
        resolved_refs = [resolve_trajectory_round_ref(ref, available_round_ids) for ref in assessment["evidence_refs"]]
        invalid_refs = [
            ref for ref in assessment["evidence_refs"]
            if not valid_trajectory_evidence_ref(ref, available_round_ids)
        ]
        if has_trajectory and invalid_refs:
            status["error_stage"] = "evidence_validation"
            status["learning_error"] = "update cites evidence absent from the recorded trajectory"
            status["invalid_evidence_refs"] = invalid_refs
            return status
        cited_round_ids = list(dict.fromkeys(ref for ref in resolved_refs if ref is not None))
        for request in revision_requests:
            cited_round_ids.extend(resolve_trajectory_round_ref(ref, available_round_ids)
                                    for ref in request["evidence_refs"])
        cited_round_ids = list(dict.fromkeys(ref for ref in cited_round_ids if ref is not None))
        if has_trajectory and action == "CREATE_NEW_SKILL" and not cited_round_ids:
            status["error_stage"] = "evidence_validation"
            status["learning_error"] = "new Skill needs a recorded round reference"
            return status
        if len(cited_round_ids) > 8:
            status["evidence_refs_omitted_from_builder"] = cited_round_ids[8:]
            cited_round_ids = cited_round_ids[:8]
            assessment = {
                **assessment,
                "evidence_refs": [
                    ref for ref, resolved in zip(assessment["evidence_refs"], resolved_refs)
                    if resolved is None or resolved in cited_round_ids
                ],
            }
        status["builder_evidence_round_ids"] = cited_round_ids
        cited_evidence = select_trajectory_evidence(
            trajectory["rounds"], evidence_refs=cited_round_ids,
            limit=8,
        )
        success_specs = assessment.get("candidate_specs", assessment.get("candidate_spec"))
        if (success and action == "CREATE_NEW_SKILL" and not isinstance(success_specs, (dict, list))
                and not coverage_gap):
            status["error_stage"] = "assessment_validation"
            status["learning_error"] = "successful-task candidate needs a specification"
            return status
        target_old = next((s for s in self.bank.skills if s["skill_id"] == assessment.get("target_skill_id")), None)
        actually_injected = any(
            record.get("injected_context") and target_old is not None and any(
                item.get("skill_id") == target_old["skill_id"] and item.get("version") == target_old["version"]
                for item in record.get("selected_skills", []))
            for record in self.records
        )
        if action == "REVISE_SKILL" and (target_old is None or not actually_injected):
            status["error_stage"] = "evidence_validation"
            status["learning_error"] = "revision target was not injected"
            return status
        if action == "UPDATE_ROUTER_PROMPT":
            if not supported_router_error(self.records, assessment):
                status["error_stage"] = "evidence_validation"
                status["learning_error"] = "no supported Router selection error to repair"
                return status
            system = REPAIR_PROMPTS[self.language]
            payload = {
                "old_router_prompt": self.bank.router_prompts[self.language],
                "diagnosis": assessment,
                "router_input_and_output": trajectory["router_inputs_and_outputs"],
                "eligible_skill_cards": [route["eligible_skill_cards"] for route in trajectory["router_inputs_and_outputs"]],
                "skillbank_index_at_run_start": skill_index,
                "language": self.language,
            }
        else:
            system = BUILDER_PROMPTS[self.language]
            payload = {
                "update_action": action, "diagnosis_or_success_summary": assessment,
                "extracted_experiences": experiences,
                "trajectory_evidence": {
                    "final_report": trajectory["final_report"], "round_index": round_index,
                    "key_evidence_excerpts": cited_evidence,
                    "router_inputs_and_outputs": trajectory["router_inputs_and_outputs"],
                },
                "existing_target_skill": target_old,
                "existing_target_skills": list(revision_targets.values()),
                "revision_requests": revision_requests,
                "benchmark_id_and_interaction_mode": {"benchmark_id": self.benchmark_id, "interaction_mode": self.interaction_mode},
                "skillbank_snapshot": skill_index, "content_language": self.language,
                "benchmark_scope": self.benchmark_id,
                "interaction_modes": [self.interaction_mode],
                "skill_token_budget": self.token_budget,
            }
        try:
            if success and action == "CREATE_NEW_SKILL" and not revision_targets and isinstance(success_specs, (dict, list)):
                specs = success_specs if isinstance(success_specs, list) else [success_specs]
                if not specs:
                    raise ValueError("new Skill needs at least one specification")
                generated = [candidate_from_success_spec(
                    spec, benchmark_id=self.benchmark_id,
                    interaction_mode=self.interaction_mode,
                    task_id=task_id, run_id=str(self.log_dir),
                    evidence_refs=assessment["evidence_refs"], language=self.language,
                ) for spec in specs]
                proposal = {"action": action, "skills": generated}
            else:
                proposal, _ = await model_json(self.agent, self.env, self.budget, system, payload)
            status["builder_output"] = proposal
            if (action == "CREATE_NEW_SKILL" and revision_targets
                    and proposal.get("action") == "REVISE_SKILL"):
                # A legacy Builder may follow the revision request and omit
                # the independent new Skill. Give it one bounded correction
                # before rejecting the batch; never stage the partial result.
                retry_payload = {**payload, "correction_request": {
                    "error": "A new Skill and all requested revisions are required in one skills array.",
                    "previous_action": proposal.get("action"),
                    "previous_skill_ids": [item.get("skill_id") for item in
                                           (proposal.get("skills") or [proposal.get("skill")])
                                           if isinstance(item, dict)],
                    "required_revision_ids": list(revision_targets),
                    "required_action": "CREATE_NEW_SKILL",
                }}
                proposal, _ = await model_json(self.agent, self.env, self.budget, system, retry_payload)
                status["builder_retry_output"] = proposal
            if proposal.get("action") != action:
                raise ValueError("proposal action mismatch")
            if action == "UPDATE_ROUTER_PROMPT":
                candidate = proposal.get("candidate_router_prompt")
                if (not isinstance(candidate, str) or not candidate.strip()
                        or (self.language == "en" and NON_ENGLISH_SCRIPT_RE.search(candidate))
                        or (self.language == "zh" and not HAN_RE.search(candidate))
                        or candidate == self.bank.router_prompts[self.language]):
                    raise ValueError("invalid Router revision")
                kind = "router"
            else:
                raw_candidates = proposal.get("skills", proposal.get("skill"))
                candidates = raw_candidates if isinstance(raw_candidates, list) else [raw_candidates]
                if not candidates or action == "REVISE_SKILL" and len(candidates) != 1:
                    raise ValueError("invalid Skill candidate count")
                normalized = []
                generated_revision_ids = set()
                generated_new_count = 0
                for candidate in candidates:
                    revision_old = revision_targets.get(candidate.get("skill_id")) if isinstance(candidate, dict) else None
                    if action == "CREATE_NEW_SKILL" and isinstance(candidate, dict) and revision_old is None:
                        # These fields describe the observed run, not model-authored guidance.
                        candidate = {
                            **candidate,
                            "version": 1, "version_parent": None, "status": "candidate",
                            "content_language": self.language,
                            "benchmark_scope": self.benchmark_id,
                            "interaction_modes": [self.interaction_mode],
                            "provenance": {
                                "task_id": task_id, "run_id": str(self.log_dir),
                                "evidence_refs": assessment["evidence_refs"],
                            },
                            "validation": {},
                            "update_count": 0,
                        }
                        generated_new_count += 1
                    elif isinstance(candidate, dict) and (action == "REVISE_SKILL" or revision_old is not None):
                        old_for_revision = target_old if action == "REVISE_SKILL" else revision_old
                        revision_evidence = (assessment["evidence_refs"] if action == "REVISE_SKILL" else next(
                            request["evidence_refs"] for request in revision_requests
                            if request["target_skill_id"] == candidate.get("skill_id")))
                        candidate = {
                            **candidate,
                            "update_count": old_for_revision.get("update_count", old_for_revision["version"] - 1) + 1,
                            "provenance": {**(candidate.get("provenance") or {}),
                                           "task_id": task_id, "run_id": str(self.log_dir),
                                           "evidence_refs": revision_evidence},
                        }
                        generated_revision_ids.add(candidate.get("skill_id"))
                    if isinstance(candidate, dict):
                        candidate = {
                            **candidate,
                            "routing_summary": candidate.get("routing_summary")
                            or _fallback_routing_summary(candidate, self.language),
                        }
                    validate_skill(candidate, language=self.language, candidate=True)
                    if action == "CREATE_NEW_SKILL" and revision_old is None and any(s["skill_id"] == candidate["skill_id"] for s in self.bank.skills):
                        raise ValueError("new Skill ID already exists")
                    old_for_revision = target_old if action == "REVISE_SKILL" else revision_old
                    if old_for_revision is not None and (candidate["skill_id"] != old_for_revision["skill_id"] or candidate["category_id"] != old_for_revision["category_id"] or candidate["version"] != old_for_revision["version"] + 1 or candidate.get("version_parent") != old_for_revision["version"]):
                        raise ValueError("invalid Skill version lineage")
                    if old_for_revision is not None and candidate["benchmark_scope"] != old_for_revision["benchmark_scope"]:
                        raise ValueError("Skill revision cannot change benchmark scope")
                    validate_single_skill_budget(candidate, language=self.language, token_budget=self.token_budget)
                    normalized.append(candidate)
                if len({skill["skill_id"] for skill in normalized}) != len(normalized):
                    raise ValueError("duplicate generated Skill ID")
                if action == "CREATE_NEW_SKILL" and (generated_new_count < 1 or generated_revision_ids != set(revision_targets)):
                    raise ValueError("Builder omitted a required new Skill or requested revision")
                candidate = normalized if action == "CREATE_NEW_SKILL" and len(normalized) > 1 else normalized[0]
                kind = "skill"
        except Exception as exc:
            status["error_stage"] = "candidate_builder" if "proposal" not in locals() else "candidate_validation"
            status["learning_error"] = str(exc)
            if isinstance(exc, ModelOutputError):
                status["raw_model_output"] = exc.raw_output
            return status
        if kind == "skill":
            preservation_reviews: list[dict[str, Any]] = []
            for item in normalized:
                old = target_old if action == "REVISE_SKILL" else revision_targets.get(item["skill_id"])
                if old is None:
                    continue
                changed = revision_changed_fields(old, item)
                if not changed:
                    preservation_reviews.append({"skill_id": item["skill_id"],
                                                 "review_skipped_reason": "guidance_unchanged"})
                    continue
                try:
                    status["learning_phase"] = "revision_preservation_review"
                    review, review_result = await model_json(
                        self.agent, self.env, self.budget, REVISION_REVIEW_PROMPTS[self.language], {
                            "old_skill": old, "revised_skill": item,
                            "changed_fields": changed,
                            "diagnosis": assessment,
                            "key_evidence_excerpts": cited_evidence,
                        },
                    )
                    preservation_reviews.append({
                        "skill_id": item["skill_id"], "changed_fields": changed,
                        "review": review, "token_usage": token_usage(review_result),
                    })
                    validate_revision_preservation_review(review, changed)
                except Exception as exc:
                    status["revision_preservation_reviews"] = preservation_reviews
                    status["error_stage"] = "revision_preservation_review"
                    status["learning_error"] = str(exc)
                    if isinstance(exc, ModelOutputError):
                        status["raw_model_output"] = exc.raw_output
                    return status
            if preservation_reviews:
                status["revision_preservation_reviews"] = preservation_reviews
        status["candidate"] = candidate
        if isinstance(candidate, list):
            status["candidates"] = candidate
        status["validation"] = "pending_replay"
        status["learning_phase"] = "candidate_ready"
        self._save_candidate(kind, candidate, assessment, status["validation"])
        if kind == "skill":
            try:
                for item in candidate if isinstance(candidate, list) else [candidate]:
                    stage_skill_candidate(self.bank, item, save=False)
                self.bank.save()
            except Exception as exc:
                status["error_stage"] = "temporary_skillbank"
                status["learning_error"] = str(exc)
                return status
        if replay is None:
            return status
        items = candidate if isinstance(candidate, list) else [candidate]
        staged_skills = copy.deepcopy(self.bank.skills)
        staged_history = copy.deepcopy(self.bank.history)
        staged_shadowed = copy.deepcopy(self.bank.shadowed_skills)
        staged_router_prompts = copy.deepcopy(self.bank.router_prompts)
        status["replay_token_usage"] = {"total_tokens": None, "missing": True, "reason": "replay_not_completed"}
        try:
            status["learning_phase"] = "candidate_replay"
            replay_result = await replay({"kind": kind, "candidate": candidate,
                                          "mode": "candidate_validation" if kind == "skill" else "production"})
            if not isinstance(replay_result, dict) or type(replay_result.get("effective")) is not bool or type(replay_result.get("hard_constraint_regression")) is not bool:
                raise ValueError("replay must report effective and hard_constraint_regression as booleans")
            candidate_report = replay_result["report"]
            if not isinstance(candidate_report, dict):
                raise ValueError("replay must return a complete report")
            replay_tokens = replay_result.get("task_tokens")
            status["replay_token_usage"] = {
                "total_tokens": replay_tokens if type(replay_tokens) is int and replay_tokens >= 0 else None,
                "missing": type(replay_tokens) is not int or replay_tokens < 0,
            }
            effective = replay_result["effective"]
            routes = replay_result.get("skill_routes", [])
            selected = {}
            if kind == "skill":
                selected = {item["skill_id"]: any(
                    item["skill_id"] == choice.get("skill_id") and item["version"] == choice.get("version")
                    for route in routes if route.get("injected_context")
                    for choice in route.get("selected_skills", [])
                ) for item in items}
                effective = effective and all(selected.values())
                record_selected_skill_outcomes(self.bank, routes, success=task_success(candidate_report, self.language))
            accepted, delta = accept_candidate(
                report, candidate_report, baseline_tokens=status["task_tokens"],
                candidate_tokens=replay_result.get("task_tokens"), reference_tokens=reference_tokens,
                effective=effective, hard_constraint_regression=replay_result["hard_constraint_regression"], language=self.language,
            )
            validation = {
                "accepted": accepted, "reward_delta": delta, "effective": effective,
                "selected_candidates": selected,
                "baseline_success": task_success(report, self.language),
                "candidate_success": task_success(candidate_report, self.language),
                "baseline_tokens": status["task_tokens"], "candidate_tokens": replay_result.get("task_tokens"),
                "hard_constraint_regression": replay_result["hard_constraint_regression"],
            }
            for item in items:
                if accepted:
                    commit_validated_candidate(self.bank, kind, item, language=self.language,
                                               validation=validation, save=False)
                elif kind == "skill":
                    staged = next(s for s in self.bank.skills if s["skill_id"] == item["skill_id"])
                    staged["validation"] = validation
            self.bank.save()
        except Exception as exc:
            status["error_stage"] = "candidate_replay"
            validation = {"accepted": False, "error": str(exc)}
            self.bank.skills = staged_skills
            self.bank.history = staged_history
            self.bank.shadowed_skills = staged_shadowed
            self.bank.router_prompts = staged_router_prompts
            if kind == "skill":
                for item in items:
                    staged = next((s for s in self.bank.skills if s["skill_id"] == item["skill_id"]), None)
                    if staged is not None:
                        staged["validation"] = validation
                self.bank.save()
        validations = [validation for _ in items]
        status["learning_phase"] = "validation_complete"
        status["validation"] = validations if isinstance(candidate, list) else validations[0]
        if isinstance(candidate, list):
            status["validations"] = [dict(skill_id=item["skill_id"], **validation) for item, validation in zip(candidate, validations)]
        self._save_candidate(kind, candidate, assessment, status["validation"])
        return status
