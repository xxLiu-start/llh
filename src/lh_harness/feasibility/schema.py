from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Decision = Literal["ACCEPT", "REVISE", "BLOCK", "UNKNOWN"]
DIMENSIONS = ["Goal Alignment", "Preconditions", "Dependency Readiness",
              "Tool Executability", "Scope and Granularity", "Verifiability",
              "Safety and Boundary Constraints"]


@dataclass(frozen=True)
class ReflectorConfig:
    enabled: bool = False
    mode: str = "disabled"
    skill_bank_path: str = "~/.lh-harness/feasibility-skills"
    top_k: int = 5
    max_revision_rounds: int = 1
    unknown_policy: str = "pass_through"
    runtime_error_policy: str = "fail_open"
    revision_exhausted_policy: str = "pass_through"
    timeout_seconds: int = 60
    max_input_chars: int = 48000
    max_validation_repairs: int = 1
    enable_skill_learning: bool = False
    # Accepted for old configuration files only; evidence assessment is unconditional
    # when enabled, even if an old file explicitly set this obsolete flag to false.
    assess_without_skills: bool = True
    skill_validation: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "min_support": 3, "check_false_rejection": True,
        "check_cross_task_transfer": True,
    })

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.enable_skill_learning) is not bool:
            raise ValueError("reflector enabled flags must be booleans")
        if type(self.assess_without_skills) is not bool:
            raise ValueError("assess_without_skills must be boolean")
        if self.mode not in {"disabled", "shadow", "gate"}:
            raise ValueError("invalid reflector mode")
        for name, lo, hi in [("top_k", 1, 5), ("max_revision_rounds", 0, 5),
                             ("timeout_seconds", 1, 600), ("max_input_chars", 2000, 100000),
                             ("max_validation_repairs", 0, 1)]:
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"invalid reflector {name}")
        if not isinstance(self.skill_bank_path, str) or not self.skill_bank_path.strip():
            raise ValueError("reflector skill_bank_path must be nonempty")
        if self.unknown_policy not in {"pass_through", "block"}:
            raise ValueError("invalid unknown_policy")
        if self.runtime_error_policy not in {"fail_open", "block"}:
            raise ValueError("invalid runtime_error_policy")
        if self.revision_exhausted_policy not in {"pass_through", "block"}:
            raise ValueError("invalid revision_exhausted_policy")
        if self.enable_skill_learning:
            raise ValueError("Feasibility learning is offline only; use the offline pipeline")
        allowed = {"enabled", "min_support", "check_false_rejection", "check_cross_task_transfer"}
        if not isinstance(self.skill_validation, dict) or set(self.skill_validation) - allowed:
            raise ValueError("invalid skill_validation configuration")
        for key, value in self.skill_validation.items():
            if key == "min_support":
                if type(value) is not int or value < 3:
                    raise ValueError("min_support must be >= 3")
            elif type(value) is not bool:
                raise ValueError(f"skill_validation.{key} must be boolean")

    @property
    def active(self) -> bool:
        return self.enabled and self.mode != "disabled"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ReflectorConfig:
        if data is not None and not isinstance(data, dict):
            raise ValueError("feasibility_reflector must be an object")
        return cls(**(data or {}))


@dataclass(frozen=True)
class Snapshot:
    original_task: str
    contract: str
    manager_plan: str
    manager_state: str
    verified_auditor_reports: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    tools_complete: bool = False
    environment_observations: tuple[dict[str, Any], ...] = ()
    state_reference: str = ""
    route: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def evidence_text(self) -> str:
        # Manager assertions are evidence about a plan, never proof of environment facts.
        def strings(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, dict):
                return [s for v in value.values() for s in strings(v)]
            if isinstance(value, (tuple, list)):
                return [s for v in value for s in strings(v)]
            return []
        return "\n".join(strings(self.to_dict()))

    def verified_evidence_text(self) -> str:
        return "\n".join(str(r.get("report", "")) for r in self.verified_auditor_reports) + "\n" + \
            "\n".join(str(o.get("content", "")) for o in self.environment_observations)


@dataclass
class Assessment:
    decision: Decision
    feasibility: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    retrieved_skills: list[str] = field(default_factory=list)
    confidence: float = 0.0
    uncertainty: str = ""
    check_evidence: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def parse(cls, data: dict[str, Any], snapshot: Snapshot, skill_ids: list[str]) -> Assessment:
        if not isinstance(data, dict) or data.get("decision") not in {"ACCEPT", "REVISE", "BLOCK", "UNKNOWN"}:
            raise ValueError("invalid reflector decision")
        decision = data["decision"]
        if data.get("feasibility") not in {"feasible", "infeasible", "uncertain"}:
            raise ValueError("invalid feasibility")
        expected = {"ACCEPT": "feasible", "REVISE": "infeasible", "BLOCK": "infeasible", "UNKNOWN": "uncertain"}
        if expected[decision] != data["feasibility"]:
            raise ValueError("contradictory decision and feasibility")
        confidence = data.get("confidence", 0)
        if type(confidence) not in {int, float} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid confidence")
        issues = data.get("issues", [])
        checks = data.get("passed_checks", [])
        if not isinstance(issues, list) or len(issues) > 12:
            raise ValueError("invalid issues")
        if not isinstance(checks, list) or any(c not in DIMENSIONS for c in checks):
            raise ValueError("invalid passed_checks")
        if decision in {"BLOCK", "REVISE"} and not issues:
            raise ValueError("rejection needs evidence-supported issues")
        if decision == "ACCEPT" and (issues or not checks):
            raise ValueError("acceptance needs passed checks and no unresolved issues")
        evidence_text = snapshot.evidence_text()
        check_evidence = data.get("check_evidence", {})
        if not isinstance(check_evidence, dict):
            raise ValueError("invalid check_evidence")
        if checks:
            for check in checks:
                quotes = check_evidence.get(check)
                if not isinstance(quotes, list) or not quotes or not all(
                    isinstance(q, str) and len(q.strip()) >= 5 and quoted_in(q, evidence_text) for q in quotes):
                    raise ValueError(f"check_evidence[{check}] needs verbatim observable evidence")
        check_evidence = {k: check_evidence[k] for k in checks if k in check_evidence}
        for index, issue in enumerate(issues):
            if not isinstance(issue, dict) or not all(isinstance(issue.get(k), str) and issue[k].strip()
                for k in ("type", "severity", "description", "affected_requirement", "suggested_correction")):
                raise ValueError("invalid issue fields")
            allowed = {"type", "severity", "description", "evidence", "affected_requirement", "suggested_correction"}
            if set(issue) - allowed or any(len(issue[k]) > 1000 for k in allowed - {"evidence"}):
                raise ValueError("issue explanations must be concise structured fields")
            if issue["severity"] not in {"low", "medium", "high"}:
                raise ValueError("invalid severity")
            evidence = issue.get("evidence")
            if not isinstance(evidence, list) or not 0 < len(evidence) <= 8 or not all(
                isinstance(e, str) and 5 <= len(e.strip()) <= 1000 and quoted_in(e, evidence_text) for e in evidence):
                raise ValueError(f"issues[{index}].evidence must quote observable input, not invent facts")
            if issue["type"] in {"MISSING_PRECONDITION", "DEPENDENCY_NOT_READY", "TOOL_UNAVAILABLE"}:
                verified = snapshot.verified_evidence_text()
                inventory_proof = issue["type"] == "TOOL_UNAVAILABLE" and snapshot.tools_complete
                if not inventory_proof and not any(quoted_in(e, verified) for e in evidence):
                    raise ValueError(f"issues[{index}] missing resource claim needs an observation/audit, not Manager assertion")
        if decision == "BLOCK" and not any(i["severity"] == "high" for i in issues):
            raise ValueError("block needs a high-severity supported issue")
        uncertainty = data.get("uncertainty", "")
        if not isinstance(uncertainty, str) or len(uncertainty) > 1000:
            raise ValueError("invalid uncertainty")
        return cls(decision, data["feasibility"], issues, checks, list(skill_ids), float(confidence), uncertainty, check_evidence)


def quoted_in(quote: str, text: str) -> bool:
    # Ignore line wrapping/whitespace only; never accept paraphrases, missing
    # negation, substituted identifiers or changed numeric constraints.
    return quote in text or " ".join(quote.split()) in " ".join(text.split())
