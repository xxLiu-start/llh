from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

ROLES = ("manager", "executor", "auditor")
MODALITIES = ("shared", "cli", "gui")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{1,79}", value):
        raise ValueError("skill name must be a short kebab-case identifier")
    return value


@dataclass(frozen=True)
class Evidence:
    task_id: str
    dataset: str
    run_id: str
    reference: str
    quote: str
    source_sha256: str
    task_fingerprint: str = ""

    def __post_init__(self):
        if not all(isinstance(v, str) and v for key, v in asdict(self).items() if key != "task_fingerprint"):
            raise ValueError("evidence needs task, dataset, run, reference, quote and source digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("invalid evidence SHA256")

    @property
    def independent_task(self):
        return self.dataset + ":" + (self.task_fingerprint or self.task_id)


@dataclass
class Skill:
    name: str
    role: str
    modality: str
    description: str
    trigger: list[str]
    preconditions: list[str]
    procedure: list[str]
    verification: list[str]
    pitfalls: list[str]
    generalization: str
    evidence: list[Evidence] = field(default_factory=list)
    status: str = "candidate"
    version: int = 1
    updated_at: str = ""
    topic: str = "general"

    def __post_init__(self):
        identifier(self.name)
        if self.role not in ROLES or self.modality not in MODALITIES or self.status not in {"candidate", "active", "rejected"}:
            raise ValueError("invalid skill role, modality or status")
        if self.role == "manager" and self.modality != "shared":
            raise ValueError("manager skills use shared modality")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("skill version must be a positive integer")
        for key in ("trigger", "preconditions", "procedure", "verification", "pitfalls"):
            value = getattr(self, key)
            if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
                raise ValueError(f"{key} must be an array of nonempty strings")
        if not self.procedure or not self.verification or not self.trigger or not self.description or not self.generalization:
            raise ValueError("skill requires trigger, procedure, verification and generalization")
        if len(self.text().split()) > 200 or len(self.text()) > 4000:
            raise ValueError("skill exceeds 200 words / 4000 characters")
        if not self.evidence:
            raise ValueError("skill requires grounded independent evidence")
        content = self.text().lower()
        if re.search(r"(?:ignore|override|bypass|disable) (?:the |all |original )?(?:instructions|audit|verifier|permissions|contract|safety)|fabricate (?:evidence|results)", content):
            raise ValueError("skill attempts to override task or audit authority")
        if self.role == "auditor" and any(re.match(r"\s*(?:write|edit|delete|modify|patch|create|install)\b", x, re.I) for x in self.procedure):
            raise ValueError("auditor skills must remain read-only")
        if re.search(r"(?:sk-[A-Za-z0-9]{16,}|(?:api[_-]?key|password|secret)\s*[=:]\s*\S{8,}|https?://\S+@)", content):
            raise ValueError("sensitive information is forbidden in skills")
        if re.search(r"/(?:data\d*|home|app|workspace)/[A-Za-z0-9_.-]+|https?://[^<\s]+", content):
            raise ValueError("replace task-specific paths and URLs with transferable placeholders")
        if re.search(r"\btest_[a-zA-Z0-9_]+\b", content):
            raise ValueError("benchmark-specific test names must not appear in reusable skill content")

    def text(self):
        return "\n".join([self.description, *[f"{key}: " + "; ".join(getattr(self, key))
                        for key in ("trigger", "preconditions", "procedure", "verification", "pitfalls")],
                          "generalization: " + self.generalization])

    @property
    def key(self):
        return f"{self.role}/{self.modality}/{self.name}"

    @property
    def support(self):
        return len({e.independent_task for e in self.evidence})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["evidence"] = [Evidence(**x) if isinstance(x, dict) else x for x in value.get("evidence", [])]
        return cls(**value)


@dataclass
class Settings:
    enabled: bool = False
    library_dir: str = "~/.lh-harness/skills"
    max_per_role: int | str | None = None
    token_budget_per_role: int | str | dict | None = None
    training: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    router: dict = field(default_factory=dict)
    tokenizer_path: str | None = None
    phase: str = "test"

    def __post_init__(self):
        for name in ("enabled",):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name, role_key in (("training", "train_roles"), ("usage", "use_roles")):
            table = dict(getattr(self, name))
            roles = table.get(role_key, [] if name == "training" else list(ROLES))
            if not isinstance(roles, list) or any(r not in ROLES for r in roles):
                raise ValueError(f"invalid {name} role selection")
            if type(table.get("enabled", False)) is not bool:
                raise ValueError(f"{name}.enabled must be boolean")
            table.update(enabled=table.get("enabled", False), **{role_key: sorted(set(roles))})
            setattr(self, name, table)
        self.training = {"batch_size": 5, "min_independent_tasks": 1, "learn_from_success": True,
                         "learn_from_failure": True, **self.training}
        if not self.enabled and (self.training["enabled"] or self.usage["enabled"]):
            raise ValueError("disabled skills conflict with enabled training or usage")
        for key in ("learn_from_success", "learn_from_failure"):
            if type(self.training[key]) is not bool:
                raise ValueError(f"training.{key} must be boolean")
        for key in ("batch_size", "min_independent_tasks"):
            if type(self.training[key]) is not int or self.training[key] < 1:
                raise ValueError(f"invalid training.{key}")
        if self.phase not in {"train", "validation", "test", "online"}:
            raise ValueError("invalid skill experiment phase")
        if self.phase in {"test", "validation"} and self.training["enabled"]:
            # Evaluation may share a training-enabled experiment template, but
            # it never performs learning. Explicit learn commands require train.
            pass
        if self.router.get("method", "bm25") != "bm25":
            raise ValueError("only deterministic BM25 routing is supported")
        self.router = dict(self.router)
        if "max_skills_per_role" in self.router:
            self.router["max_skills_per_role"] = self.limit(self.router["max_skills_per_role"], "router.max_skills_per_role")
        self.max_per_role = self.limit(self.max_per_role, "max_per_role")
        if isinstance(self.token_budget_per_role, dict):
            self.token_budget_per_role = {role: self.limit(value, "role token budget")
                                          for role, value in self.token_budget_per_role.items()}
        else:
            self.token_budget_per_role = self.limit(self.token_budget_per_role, "role token budget")

    @staticmethod
    def limit(value, name):
        # TOML has no null literal; "unlimited" makes the setting explicit.
        # Numeric limits remain supported for replaying previous experiments.
        if value is None or value == "unlimited":
            return None
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer or 'unlimited'")
        return value

    def budget(self, role):
        value = self.token_budget_per_role
        return value.get(role) if isinstance(value, dict) else value

    @classmethod
    def from_dict(cls, value):
        value = dict(value or {})
        if "bank" in value:
            raise ValueError("legacy experimental JSON banks must be explicitly migrated to skills.library_dir")
        from dataclasses import fields
        if set(value) - {f.name for f in fields(cls)}:
            raise ValueError("unknown skills configuration keys")
        return cls(**value)


def object_schema(fields: dict, required=None):
    return {"type": "object", "properties": fields, "required": required or list(fields), "additionalProperties": False}


def validate_object(value, schema):
    """Validate the closed JSON shapes used by the offline model boundary."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                return validate_object(value, option)
            except ValueError:
                pass
        raise ValueError("JSON value does not match a grounded evidence option")
    if schema.get("type") == "object":
        if not isinstance(value, dict) or set(value) - set(schema["properties"]) or set(schema["required"]) - set(value):
            raise ValueError("unexpected or missing JSON fields")
        for key, item in value.items():
            validate_object(item, schema["properties"][key])
    elif schema.get("type") == "array":
        if not isinstance(value, list):
            raise ValueError("expected JSON array")
        prefix = schema.get("prefixItems", [])
        for index, item in enumerate(value):
            validate_object(item, prefix[index] if index < len(prefix) else schema["items"])
    elif schema.get("type") == "string" and not isinstance(value, str):
        raise ValueError("expected JSON string")
    elif schema.get("type") == "boolean" and type(value) is not bool:
        raise ValueError("expected JSON boolean")
    elif schema.get("type") == "number" and (type(value) not in (int, float) or not 0 <= value <= 1):
        raise ValueError("confidence must be between zero and one")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("JSON enum value is invalid")
    if "const" in schema and value != schema["const"]:
        raise ValueError("JSON constant is invalid")
    if isinstance(value, str) and (len(value) > schema.get("maxLength", len(value)) or
            ("pattern" in schema and not re.search(schema["pattern"], value))):
        raise ValueError("JSON string violates length or pattern constraint")
    if isinstance(value, list) and len(value) > schema.get("maxItems", len(value)):
        raise ValueError("JSON array exceeds item limit")
    if isinstance(value, list) and len(value) < schema.get("minItems", 0):
        raise ValueError("JSON array has too few items")
    return value
