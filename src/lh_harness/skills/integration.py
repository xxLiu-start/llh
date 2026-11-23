from __future__ import annotations

import json
import time
from pathlib import Path
from .schema import Settings
from .library import SkillLibrary, atomic_json
from .router import SkillRouter
from .statistics import summarize
from ..token_usage import from_trajectory, normalize

ALIASES = {"orchestrator": ("manager", "shared"), "manager": ("manager", "shared"),
           "cli_task": ("executor", "cli"), "task": ("executor", "cli"), "cli_executor": ("executor", "cli"),
           "gui_task": ("executor", "gui"), "gui_executor": ("executor", "gui"),
           "cli_verifier": ("auditor", "cli"), "verifier": ("auditor", "cli"), "cli_auditor": ("auditor", "cli"),
           "gui_verifier": ("auditor", "gui"), "gui_auditor": ("auditor", "gui")}

REPAIR_PREFIXES = ("你的上一份 verifier 报告缺少有效的前三行控制头。",
                   "Your previous auditor report lacks a valid three-line control header.",
                   "上一份 auditor 报告缺少有效的前三行控制头。")


class SkillSession:
    def __init__(self, settings, log_dir, *, counter=None):
        self.settings = settings if isinstance(settings, Settings) else Settings.from_dict(settings)
        self.root, self.records, self.injections = Path(log_dir), [], []
        self.library = SkillLibrary(self.settings.library_dir)
        self.frozen = None
        if self.settings.enabled and self.settings.phase == "test":
            self.frozen = self.library.verify_frozen()
        self.router = SkillRouter(self.library, self.settings, counter=counter) if self.settings.enabled and self.settings.usage["enabled"] else None

    def wrap(self, role, adapter):
        session = self
        group, modality = ALIASES.get(role, ("auxiliary", "shared"))
        class Adapter:
            def __getattr__(self, name):
                return getattr(adapter, name)

            async def run_episode(self, prompt, *args, **kwargs):
                if session.frozen:
                    session.library.verify_frozen()
                effective_group = "auxiliary" if prompt.startswith(REPAIR_PREFIXES) else group
                skills, text, count = session.router.select(effective_group, modality, prompt) if session.router and effective_group != "auxiliary" else ([], "", None)
                # Persist the in-flight call before awaiting. A task timeout
                # must leave unavailable usage, never disappear from totals.
                started=time.monotonic()
                row={"role":effective_group,"native_role":role,"modality":modality,
                     "usage":normalize(None),"duration_ms":0,"status":"running"}
                session.records.append(row)
                injection = {"role": effective_group, "native_role": role, "skills": [s.key for s in skills],
                    "versions": [s.version for s in skills], "tokens": count,
                    "token_status": ("estimated" if count is not None else "unavailable") if text else "not_applicable",
                    "method": "same-model offline tokenizer; injected segment including wrapper", "included_in_role_input": True,
                    "no_match_or_tokenizer_unavailable": not bool(text)}
                # Local adapters stream their raw JSONL to live_trajectory_path;
                # Harbor-backed adapters may instead expose claude_stream_path
                # in result metadata. Persist the path before awaiting so a
                # killed/timed-out worker still leaves a recoverable partial trace.
                stream_path = kwargs.get("live_trajectory_path")
                if isinstance(stream_path, (str, Path)) and str(stream_path):
                    injection["trajectory_path"] = str(Path(stream_path).expanduser().resolve(strict=False))
                    injection["trajectory_status"] = "running"
                session.injections.append(injection)
                session.write()
                try:
                    result = await adapter.run_episode(prompt + text, *args, **kwargs)
                    metadata = getattr(result, "metadata", {})
                    if isinstance(metadata, dict):
                        metadata_path = metadata.get("claude_stream_path")
                        if isinstance(metadata_path, str) and metadata_path:
                            injection["trajectory_path"] = str(Path(metadata_path).expanduser().resolve(strict=False))
                    if injection.get("trajectory_path"):
                        injection["trajectory_status"] = getattr(result, "status", "unknown")
                    row.update(usage=from_trajectory(getattr(result, "actions_log", ""), metadata),
                               duration_ms=getattr(result,"duration_ms",0),status=result.status)
                    return result
                except BaseException as exc:
                    injection["trajectory_status"] = "cancelled" if type(exc).__name__ == "CancelledError" else "error"
                    row.update(status="cancelled" if type(exc).__name__=="CancelledError" else "error",
                               duration_ms=int((time.monotonic()-started)*1000))
                    raise
                finally:
                    session.write()
        return Adapter()

    def write(self):
        self.root.mkdir(parents=True, exist_ok=True)
        missing = any(x["skills"] and x["tokens"] is None for x in self.injections)
        atomic_json(self.root / "skill_usage.json", {"schema_version": 1, "library": self.frozen,
            "injections": self.injections, "training_calls_in_mea": 0,
            "skill_injection_tokens": None if missing else sum(x["tokens"] or 0 for x in self.injections),
            "skill_injection_token_status": "unavailable" if missing else "estimated" if any(x["tokens"] is not None for x in self.injections) else "not_applicable"})
        atomic_json(self.root / "token_usage.json", summarize(self.records))


def build_role_manager_prompt(*, skill_context="", **kwargs):
    from ..role_prompts import build_role_manager_prompt as original
    return original(**kwargs) + skill_context


def build_role_executor_prompt(*, skill_context="", **kwargs):
    from ..role_prompts import build_role_executor_prompt as original
    return original(**kwargs) + skill_context


def build_role_auditor_prompt(*, skill_context="", **kwargs):
    from ..role_prompts import build_role_auditor_prompt as original
    return original(**kwargs) + skill_context
