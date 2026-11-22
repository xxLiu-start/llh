"""Append-only, view-only provenance for Terminal-Bench Skill training.

Nothing in the training or evaluation decision path reads these records.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .role_prompts import (
    LH_HARNESS_CLI_AUDITOR_INSTRUCTIONS,
    LH_HARNESS_CLI_EXECUTOR_INSTRUCTIONS,
    LH_HARNESS_GUI_AUDITOR_INSTRUCTIONS,
    LH_HARNESS_GUI_EXECUTOR_INSTRUCTIONS,
    LH_HARNESS_MANAGER_INSTRUCTIONS,
)
from .prompt_texts import (
    MANAGER_INSTRUCTIONS, CLI_EXECUTOR_INSTRUCTIONS, GUI_EXECUTOR_INSTRUCTIONS,
    CLI_AUDITOR_INSTRUCTIONS, GUI_AUDITOR_INSTRUCTIONS,
)
from .skill_learning import SkillBank
from .skill_prompt_texts import (
    EXPERIENCE_EXTRACTION_PROMPT_EN,
    FAILURE_DIAGNOSIS_PROMPT_EN,
    ROUTER_PROMPT_EN,
    ROUTER_REPAIR_PROMPT_EN,
    SKILL_BUILDER_PROMPT_EN,
    REVISION_PRESERVATION_REVIEW_PROMPT_EN,
    SUCCESS_SUMMARIZER_PROMPT_EN,
)
from .skill_prompt_texts_zh import (
    EXPERIENCE_EXTRACTION_PROMPT_ZH,
    ROUTER_PROMPT_ZH, SUCCESS_SUMMARIZER_PROMPT_ZH,
    FAILURE_DIAGNOSIS_PROMPT_ZH, ROUTER_REPAIR_PROMPT_ZH,
    SKILL_BUILDER_PROMPT_ZH,
    REVISION_PRESERVATION_REVIEW_PROMPT_ZH,
)

logger = logging.getLogger(__name__)

STATIC_PROMPTS = {
    "role_manager": LH_HARNESS_MANAGER_INSTRUCTIONS,
    "role_cli_executor": LH_HARNESS_CLI_EXECUTOR_INSTRUCTIONS,
    "role_gui_executor": LH_HARNESS_GUI_EXECUTOR_INSTRUCTIONS,
    "role_cli_auditor": LH_HARNESS_CLI_AUDITOR_INSTRUCTIONS,
    "role_gui_auditor": LH_HARNESS_GUI_AUDITOR_INSTRUCTIONS,
    "skill_router_default": ROUTER_PROMPT_EN,
    "skill_experience_extraction": EXPERIENCE_EXTRACTION_PROMPT_EN,
    "skill_success_summarizer": SUCCESS_SUMMARIZER_PROMPT_EN,
    "skill_failure_diagnosis": FAILURE_DIAGNOSIS_PROMPT_EN,
    "skill_router_repair": ROUTER_REPAIR_PROMPT_EN,
    "skill_builder": SKILL_BUILDER_PROMPT_EN,
    "skill_revision_preservation_review": REVISION_PRESERVATION_REVIEW_PROMPT_EN,
}
STATIC_PROMPTS_ZH = {
    "role_manager": MANAGER_INSTRUCTIONS["zh"],
    "role_cli_executor": CLI_EXECUTOR_INSTRUCTIONS["zh"],
    "role_gui_executor": GUI_EXECUTOR_INSTRUCTIONS["zh"],
    "role_cli_auditor": CLI_AUDITOR_INSTRUCTIONS["zh"],
    "role_gui_auditor": GUI_AUDITOR_INSTRUCTIONS["zh"],
    "skill_router_default": ROUTER_PROMPT_ZH,
    "skill_experience_extraction": EXPERIENCE_EXTRACTION_PROMPT_ZH,
    "skill_success_summarizer": SUCCESS_SUMMARIZER_PROMPT_ZH,
    "skill_failure_diagnosis": FAILURE_DIAGNOSIS_PROMPT_ZH,
    "skill_router_repair": ROUTER_REPAIR_PROMPT_ZH,
    "skill_builder": SKILL_BUILDER_PROMPT_ZH,
    "skill_revision_preservation_review": REVISION_PRESERVATION_REVIEW_PROMPT_ZH,
}


def _version(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_version(value: Any) -> str:
    return _version(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


class ProvenanceWriter:
    def __init__(self, path: Path, *, task: str, run_id: str, prompt_language: str = "en") -> None:
        self.path = path
        self.task = task
        self.run_id = run_id
        if prompt_language not in {"en", "zh"}:
            raise ValueError("prompt_language must be en or zh")
        self.prompt_language = prompt_language
        self._previous_runtime_prompts: dict[tuple[str, str, str], str] = {}

    def append(self, event: str, **fields: Any) -> None:
        record = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": self.task,
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("Could not write view-only provenance %s: %s", self.path, exc)

    def static_prompts(self) -> None:
        prompts = STATIC_PROMPTS_ZH if self.prompt_language == "zh" else STATIC_PROMPTS
        for name, content in prompts.items():
            self.append(
                "prompt_version", stage="run_start", source="code_template",
                prompt_name=name, language=self.prompt_language, version_id=_version(content), content=content,
            )

    def bank_snapshot(self, stage: str, bank: SkillBank) -> None:
        self.append("bank_snapshot", stage=stage, bank_path=str(bank.path),
                    active_skill_count=len(bank.skills), history_count=len(bank.history))
        for skill in bank.skills:
            self.append(
                "skill_version", stage=stage, source="active_bank",
                skill_id=skill.get("skill_id"), version=skill.get("version"),
                version_parent=skill.get("version_parent"), version_id=_content_version(skill),
                content=skill,
            )
        for index, item in enumerate(bank.history):
            if item.get("kind") == "skill" and isinstance(item.get("skill"), dict):
                skill = item["skill"]
                self.append(
                    "skill_version", stage=stage, source="bank_history", history_index=index,
                    skill_id=skill.get("skill_id"), version=skill.get("version"), content=skill,
                    version_parent=skill.get("version_parent"), version_id=_content_version(skill),
                )
        for language, content in bank.router_prompts.items():
            self.append(
                "prompt_version", stage=stage, source="active_router_prompt",
                prompt_name="skill_router", language=language,
                version_id=_version(content), content=content,
            )
        for index, item in enumerate(bank.history):
            if item.get("kind") == "router" and isinstance(item.get("prompt"), str):
                content = item["prompt"]
                self.append(
                    "prompt_version", stage=stage, source="bank_history", history_index=index,
                    prompt_name="skill_router", language=item.get("language"),
                    version_id=_version(content), content=content,
                )

    def bank_snapshot_file(self, stage: str, path: Path) -> None:
        """Archive a trial's disposable bank without affecting training decisions."""
        try:
            if path.is_file():
                self.bank_snapshot(stage, SkillBank.load(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Could not archive view-only SkillBank snapshot %s: %s", path, exc)

    def candidate(
        self, stage: str, proposal: dict[str, Any], *, source_path: Path,
        previous_router_prompt: str | None = None,
        previous_skill: dict[str, Any] | None = None,
    ) -> None:
        kind = proposal.get("kind")
        content = proposal.get("candidate")
        self.append(
            "candidate_version", stage=stage, kind=kind,
            source_path=str(source_path),
            skill_id=content.get("skill_id") if isinstance(content, dict) else None,
            version=content.get("version") if isinstance(content, dict) else None,
            version_parent=content.get("version_parent") if isinstance(content, dict) else None,
            version_id=_version(content) if isinstance(content, str) else _content_version(content),
            previous_version_id=(
                _content_version(previous_skill) if kind == "skill" and previous_skill is not None
                else _version(previous_router_prompt) if kind == "router" and previous_router_prompt else None
            ),
            content=content, assessment=proposal.get("assessment"),
            validation=proposal.get("validation"),
        )
        if kind == "router" and isinstance(content, str):
            self.append(
                "prompt_version", stage=stage, source="router_candidate",
                prompt_name="skill_router", language=self.prompt_language, version_id=_version(content),
                previous_version_id=_version(previous_router_prompt) if previous_router_prompt else None,
                content=content,
            )

    def ingest_model_io(self, config_path: Path, jobs_dir: Path, *, phase: str) -> None:
        """Copy this trial's per-episode model records into the task run file."""
        if not config_path.is_file():
            return
        try:
            import yaml
            job_name = yaml.safe_load(config_path.read_text(encoding="utf-8"))["job_name"]
            job_dir = jobs_dir / job_name
            for model_io_path in sorted(job_dir.rglob("model_io.jsonl")):
                with model_io_path.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            self.append("model_io_parse_error", phase=phase, source_path=str(model_io_path), line_number=line_number)
                            continue
                        self.append(
                            "model_io", phase=phase, source_path=str(model_io_path),
                            line_number=line_number, record=record,
                        )
                        if isinstance(record, dict) and record.get("event") == "model_input":
                            self._runtime_prompt(phase, model_io_path, line_number, record)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Could not collect view-only model I/O for %s: %s", self.task, exc)

    def _runtime_prompt(
        self, phase: str, source_path: Path, line_number: int, record: dict[str, Any],
    ) -> None:
        payload = record.get("payload")
        if isinstance(payload, dict) and payload.get("format") == "json":
            payload = payload.get("value")
        if not isinstance(payload, dict):
            return
        prompt = {key: payload[key] for key in ("system", "messages", "tools") if key in payload}
        if not prompt:
            return
        parts = source_path.parts
        try:
            index = parts.index("episodes")
            role, episode = parts[index + 1:index + 3]
        except (ValueError, IndexError):
            role, episode = "unknown", "unknown"
        version_id = _content_version(prompt)
        key = (phase, role, episode)
        self.append(
            "runtime_prompt_version", phase=phase, role=role, episode=episode,
            protocol=record.get("protocol"), model=payload.get("model"),
            trace_id=record.get("trace_id"), version_id=version_id,
            previous_version_id=self._previous_runtime_prompts.get(key),
            model_io_source_path=str(source_path), model_io_line_number=line_number,
            content=prompt,
        )
        self._previous_runtime_prompts[key] = version_id
