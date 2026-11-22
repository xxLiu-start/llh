"""Opt-in dual-track experiment adapter. The MEA execution kernel is untouched.

Role guidance is advisory. Training uses the same role's model with the
final-response permission policy, never another provider.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .supervisor.control_bus import _atomic_bytes_write
from .token_usage import aggregate, from_trajectory, unavailable

ROLES = {"manager", "executor", "auditor"}


def validate_skills(value: dict) -> dict:
    if isinstance(value, dict) and "bank" not in value:
        from dataclasses import asdict
        from .skills.schema import Settings
        return asdict(Settings.from_dict(value))
    if not isinstance(value, dict) or set(value) - {"enabled", "bank", "training", "usage"}:
        raise ValueError("invalid [skills] configuration")
    result = {"enabled": False, "bank": ".lh-harness/experimental-skills.json", **value}
    if type(result["enabled"]) is not bool or not isinstance(result["bank"], str):
        raise ValueError("skills.enabled must be boolean and skills.bank must be a path")
    for name, role_key in (("training", "train_roles"), ("usage", "use_roles")):
        table = result.setdefault(name, {})
        if not isinstance(table, dict) or set(table) - {"enabled", role_key}:
            raise ValueError(f"invalid skills.{name}")
        enabled = table.get("enabled", False)
        roles = table.get(role_key, sorted(ROLES))
        if type(enabled) is not bool or not isinstance(roles, list) or not all(r in ROLES for r in roles):
            raise ValueError(f"invalid skills.{name} role selection")
        result[name] = {"enabled": enabled, role_key: sorted(set(roles))}
    if not result["enabled"] and any(result[n]["enabled"] for n in ("training", "usage")):
        raise ValueError("skills.enabled=false conflicts with enabled training/usage")
    return result


def role_group(role: str) -> str:
    return "executor" if role.endswith("executor") else "auditor" if role.endswith("auditor") else role


def save(path: Path, value):
    _atomic_bytes_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


class ExperimentSession:
    def __init__(self, settings: dict, log_dir: str, tokenizer_path: str | None = None):
        # This public class is retained for explicit legacy-bank callers. New
        # configurations are routed to skills.integration.SkillSession instead.
        self.settings = validate_skills({"bank": ".lh-harness/experimental-skills.json", **settings})
        self.root = Path(log_dir)
        self.records: list[dict] = []
        self.injections: list[dict] = []
        self.tokenizer = None
        self.tokenizer_error = None
        if tokenizer_path:
            try:
                from tokenizers import Tokenizer
                path = Path(tokenizer_path)
                self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json" if path.is_dir() else path))
            except Exception as exc:
                self.tokenizer_error = type(exc).__name__
        self.tokenizer_path = tokenizer_path
        self.bank_path = Path(self.settings["bank"]).expanduser()
        self.bank = json.loads(self.bank_path.read_text()) if self.settings["enabled"] and self.bank_path.exists() else []
        if not isinstance(self.bank, list):
            raise ValueError("experimental Skill bank must be an array")

    def wrap(self, role, adapter):
        session = self
        group = role_group(role)

        class TrackedAdapter:
            async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
                context = ""
                settings = session.settings
                if settings["enabled"] and settings["usage"]["enabled"] and group in settings["usage"]["use_roles"]:
                    # Bounded, deterministic selection: no extra router model.
                    words = set(prompt.lower().split())
                    candidates = [s for s in session.bank if s.get("role") == group and s.get("status") == "curated"]
                    ranked = sorted(candidates, key=lambda s: -len(words & set(s.get("guidance", "").lower().split())))
                    guidance = [s["guidance"] for s in ranked[:2] if words & set(s.get("guidance", "").lower().split())]
                    if guidance:
                        context = "\n\nAdvisory Skills (original instructions and auditor authority take precedence):\n" + "\n".join(guidance)[:5000]
                count = len(session.tokenizer.encode(context, add_special_tokens=False).ids) if context and session.tokenizer else None
                session.injections.append({"role": role, "characters": len(context),
                    "tokens": count if context else 0,
                    "status": ("estimated" if count is not None else "unavailable") if context else "not_applicable",
                    "reason": "Offline same-model tokenizer; segment boundary estimate" if count is not None else "No matching offline tokenizer supplied" if context else None})
                result = await adapter.run_episode(prompt + context, env, budget, live_trajectory_path=live_trajectory_path)
                usage = result.metadata.get("usage") or from_trajectory(result.actions_log, result.metadata)
                session.records.append({"role": role, "group": group, "usage": usage,
                    "status": result.status, "output": str(result.metadata.get("assistant_visible_output") or result.actions_log)[-12000:]})
                session.write_usage()
                return result

        return TrackedAdapter()

    def write_usage(self):
        data = {role: aggregate([r["usage"] for r in self.records if r["group"] == role]) for role in sorted(ROLES)}
        data.update({role: aggregate([r["usage"] for r in self.records if r["group"] == role])
                     for role in ("skill_proposer", "skill_curator", "final_response", "auditor_format_repair")})
        injected = [v for v in self.injections if v["characters"]]
        missing = any(v["tokens"] is None for v in injected)
        data["skill_injection"] = {"status": ("unavailable" if missing else "estimated") if injected else "not_applicable",
            "tokens": None if missing else sum(v["tokens"] for v in injected), "characters": sum(v["characters"] for v in injected),
            "tokenizer_path": self.tokenizer_path,
            "tokenizer_error": self.tokenizer_error,
            "included_in_role_input_tokens": True, "accounting": "reported_only; no character estimates"}
        save(self.root / "token_usage.json", {"schema_version": 1, "policy": "reported_only", "categories": data})

    async def finish(self, report, trainers, env, budget):
        training_error = None
        before = len(self.bank)
        try:
            if not self.settings["enabled"] or not self.settings["training"]["enabled"]:
                return
            for role in self.settings["training"]["train_roles"]:
                evidence = [r["output"] for r in self.records if r["group"] == role and r["status"] == "done"][-3:]
                if not evidence:
                    continue
                adapter = trainers[role]
                prompt = ("Propose one reusable advisory Skill for role " + role + ". Return only JSON with a guidance string and an evidence_quote string copied verbatim from the supplied evidence. "
                          "Preserve the original task, permissions, verifier, and completion criteria. Do not use tools.\n" + json.dumps(evidence, ensure_ascii=False))
                proposal = await self._call("skill_proposer", adapter, prompt, env, budget)
                if not isinstance(proposal.get("guidance"), str) or not 8 <= len(proposal["guidance"]) <= 2500:
                    raise ValueError("invalid proposed Skill guidance")
                quote = proposal.get("evidence_quote")
                if not isinstance(quote, str) or len(quote) < 8 or not any(quote in item for item in evidence):
                    raise ValueError("Skill proposal is not grounded in role evidence")
                review = await self._call("skill_curator", adapter,
                    "Review this advisory Skill and its role evidence. Reject changes to task goals, verifier rules, permissions, or completion authority. Return only JSON {\"accept\": true/false}. Do not use tools.\n" + json.dumps({"proposal": proposal, "evidence": evidence}, ensure_ascii=False), env, budget)
                if review.get("accept") is True:
                    item = {"role": role, "guidance": proposal["guidance"], "evidence_quote": quote, "status": "curated",
                            "id": hashlib.sha256((role + proposal["guidance"]).encode()).hexdigest()}
                    if not any(s.get("id") == item["id"] for s in self.bank):
                        self.bank.append(item)
                        save(self.bank_path, self.bank)
        except Exception as exc:
            training_error = str(exc)
            # Training never supplies completion authority to the baseline.
            save(self.root / "skill_training_error.json", {"error": type(exc).__name__, "detail": str(exc)})
        finally:
            self.write_usage()
            save(self.root / "skill_training_summary.json", {
                "enabled": self.settings["enabled"] and self.settings["training"]["enabled"],
                "train_roles": self.settings["training"]["train_roles"],
                "accepted_skills": len(self.bank) - before,
                "proposer_calls": sum(r["group"] == "skill_proposer" for r in self.records),
                "curator_calls": sum(r["group"] == "skill_curator" for r in self.records),
                "error": training_error, "task_outcome_authority": False,
            })

    async def _call(self, category, adapter, prompt, env, budget):
        from .agent_logs import assistant_texts, visible_output
        text, problem = "", ""
        for attempt in range(2):
            index = len([r for r in self.records if r["group"] == category])
            path = self.root / f"{category}_{index:03d}"
            request = prompt if not attempt else (
                "Your previous response did not satisfy the required JSON shape: " + problem +
                ". Return one JSON object only, with no preamble, no Markdown, and no wrapper key. "
                "For a Skill proposal the top-level keys are guidance and evidence_quote; for a review the key is accept (boolean). "
                "Do not inspect the workspace or call tools.\nOriginal request:\n" + prompt +
                "\nPrevious response:\n" + text[-5000:])
            save(path.with_suffix(".input.json"), {"prompt": request, "format_repair": bool(attempt)})
            result = await adapter.run_episode(request, env, budget, live_trajectory_path=str(path.with_suffix(".jsonl")))
            texts = assistant_texts(result.actions_log)
            text = str(texts[-1] if texts else result.metadata.get("assistant_visible_output") or visible_output(result.actions_log)).strip()
            record = {"role": category, "group": category, "status": result.status, "format_repair": bool(attempt),
                      "usage": result.metadata.get("usage") or from_trajectory(result.actions_log, result.metadata)}
            self.records.append(record)
            save(path.with_suffix(".output.json"), {**record, "output": text, "error": result.error})
            self.write_usage()
            if result.status != "done":
                raise ValueError(f"{category} failed; no alternate provider attempted")
            candidate = text
            if candidate.startswith("```") and "\n" in candidate:
                candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0]
            try:
                value = json.loads(candidate)
                if not isinstance(value, dict):
                    raise ValueError("Skill model returned a non-object")
                if category == "skill_proposer" and not all(isinstance(value.get(k), str) for k in ("guidance", "evidence_quote")):
                    raise ValueError("proposal needs top-level guidance and evidence_quote strings")
                if category == "skill_curator" and type(value.get("accept")) is not bool:
                    raise ValueError("review needs a top-level accept boolean")
                return value
            except ValueError as exc:
                problem = str(exc)
        raise ValueError(f"{category} JSON format repair failed: {problem}")
