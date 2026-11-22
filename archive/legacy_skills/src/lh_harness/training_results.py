"""Readable per-task Skill-training results derived from immutable run artifacts."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


FIELDS = (
    "task", "correct", "reward", "status", "input_tokens", "output_tokens",
    "cache_tokens", "total_tokens", "rounds_run", "harness_status",
    "harness_completion_satisfied", "result_path", "report_path",
    "token_usage_missing", "token_usage_partial", "run_id", "record_type",
    "accepted", "reason", "error_stage", "error", "learning_action",
    "baseline_correct", "baseline_reward", "baseline_harbor_status",
    "baseline_harness_status", "baseline_input_tokens", "baseline_output_tokens",
    "baseline_cache_tokens", "baseline_total_tokens", "baseline_usage_partial",
    "baseline_result_path", "baseline_report_path",
    "candidate_correct", "candidate_reward", "candidate_harbor_status",
    "candidate_harness_status", "candidate_input_tokens", "candidate_output_tokens",
    "candidate_cache_tokens", "candidate_total_tokens", "candidate_usage_partial",
    "candidate_result_path", "candidate_report_path",
    "baseline_referenced_skill_names", "candidate_referenced_skill_names",
    "referenced_skill_names",
    "skill_learning_tokens", "skill_learning_llm_calls", "skill_validation_tokens",
    "skill_validation_status", "total_tokens_including_skill_overhead",
)


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _latest_record(decisions_dir: Path, task: str) -> tuple[dict[str, Any] | None, str]:
    decision_path = decisions_dir / f"{task}.json"
    decision = _read_json(decision_path)
    attempts = list((decisions_dir / "attempts" / task).glob("*.json"))
    if not attempts:
        return decision, "decision" if decision else "not_run"
    latest_path = max(attempts, key=lambda path: (path.stat().st_mtime_ns, path.name))
    attempt = _read_json(latest_path)
    if attempt is None:
        return decision, "decision" if decision else "not_run"
    if decision and (
        decision.get("run_id") == attempt.get("run_id")
        or decision_path.stat().st_mtime_ns >= latest_path.stat().st_mtime_ns
    ):
        return decision, "decision"
    return attempt, "attempt"


def _phase(record: dict[str, Any], name: str) -> dict[str, Any]:
    report_name = record.get(f"{name}_report_path")
    report_path = Path(report_name) if isinstance(report_name, str) and report_name else None
    trial_path = report_path.parents[3] / "result.json" if report_path else None
    trial = _read_json(trial_path)
    harness = _read_json(report_path) or record.get(f"{name}_report") or {}
    training_path = report_path.parent.parent / "training_report.json" if report_path else None
    training = _read_json(training_path) or {}
    stored_usage = record.get(f"{name}_token_usage") or {}
    usage = training.get("usage_totals") or stored_usage.get("usage_totals") or {}
    agent = (trial or {}).get("agent_result") or {}

    def count(key: str) -> int | None:
        value = usage.get(key)
        if type(value) is not int:
            value = agent.get(key)
        return value if type(value) is int and value >= 0 else None

    inp, out, cache = (count(key) for key in ("n_input_tokens", "n_output_tokens", "n_cache_tokens"))
    total = inp + out if inp is not None and out is not None else None
    if total is None:
        value = stored_usage.get("total_tokens")
        total = value if type(value) is int and value >= 0 else None
    verifier = (trial or {}).get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    reward = rewards.get("reward")
    reward = float(reward) if type(reward) in (int, float) else None
    exception = (trial or {}).get("exception_info")
    status = "errored" if exception else "completed" if reward is not None else "missing_reward" if trial else None
    correct = reward >= 1.0 and not exception if reward is not None else False if exception else None
    partial = training.get("usage_partial") is True or stored_usage.get("usage_partial") is True
    return {
        "correct": correct, "reward": reward, "harbor_status": status,
        "harness_status": harness.get("status") if isinstance(harness, dict) else None,
        "completion_satisfied": harness.get("completion_satisfied") if isinstance(harness, dict) else None,
        "rounds_run": harness.get("rounds_run") if isinstance(harness, dict) else None,
        "input_tokens": inp, "output_tokens": out, "cache_tokens": cache,
        "total_tokens": total, "usage_partial": partial,
        "result_path": str(trial_path) if trial_path and trial else None,
        "report_path": str(report_path) if report_path else None,
    }


def _sum_known(values: list[int | None]) -> int | None:
    known = [value for value in values if type(value) is int]
    return sum(known) if known else None


def _row(task: str, record: dict[str, Any] | None, record_type: str) -> dict[str, Any]:
    if record is None:
        return {"task": task, "status": "not_run", "record_type": "not_run"}
    baseline, candidate = _phase(record, "baseline"), _phase(record, "candidate")
    selected = candidate if record.get("accepted") is True and candidate["harbor_status"] else baseline
    failed_attempt = record_type == "attempt" and bool(record.get("error") or record.get("reason") == "infrastructure_error")
    attempted = [baseline]
    if record.get("candidate_report_path") or record.get("candidate_token_usage") or record.get("phase") == "candidate":
        attempted.append(candidate)
    missing = any(item["total_tokens"] is None for item in attempted)
    partial = missing or any(item["usage_partial"] for item in attempted)
    row = {
        "task": task,
        "correct": None if failed_attempt else selected["correct"],
        "reward": None if failed_attempt else selected["reward"],
        "status": "infrastructure_error" if failed_attempt else selected["harbor_status"] or "report_missing",
        "input_tokens": _sum_known([item["input_tokens"] for item in attempted]),
        "output_tokens": _sum_known([item["output_tokens"] for item in attempted]),
        "cache_tokens": _sum_known([item["cache_tokens"] for item in attempted]),
        "total_tokens": _sum_known([item["total_tokens"] for item in attempted]),
        "rounds_run": selected["rounds_run"],
        "harness_status": selected["harness_status"],
        "harness_completion_satisfied": selected["completion_satisfied"],
        "result_path": selected["result_path"], "report_path": selected["report_path"],
        "token_usage_missing": missing, "token_usage_partial": partial,
        "run_id": record.get("run_id"), "record_type": record_type,
        "accepted": record.get("accepted"), "reason": record.get("reason"),
        "error_stage": record.get("error_stage"), "error": record.get("error"),
        "learning_action": (record.get("skill_learning") or {}).get("action"),
    }
    learning = record.get("post_verifier_learning") or {}
    validation = record.get("independent_skill_validation") or {}
    learning_tokens = learning.get("learning_tokens")
    validation_tokens = (validation.get("contrast_token_usage") or {}).get("total_tokens")
    row.update(skill_learning_tokens=learning_tokens,
               skill_learning_llm_calls=learning.get("learning_llm_calls"),
               skill_validation_tokens=validation_tokens,
               skill_validation_status=validation.get("status"))
    known = row["total_tokens"]
    row["total_tokens_including_skill_overhead"] = (
        known + (learning_tokens or 0) + (validation_tokens or 0)
        if type(known) is int and (not learning or type(learning_tokens) is int)
        and (not validation or not validation.get("additional_trials") or type(validation_tokens) is int) else None)
    for name, phase in (("baseline", baseline), ("candidate", candidate)):
        for key in ("correct", "reward", "harbor_status", "harness_status", "input_tokens",
                    "output_tokens", "cache_tokens", "total_tokens", "usage_partial", "result_path", "report_path"):
            row[f"{name}_{key}"] = phase[key] if phase["report_path"] else None
        names = record.get(f"{name}_referenced_skill_names") or []
        row[f"{name}_referenced_skill_names"] = "; ".join(names) if isinstance(names, list) else ""
    names = record.get("referenced_skill_names") or []
    row["referenced_skill_names"] = "; ".join(names) if isinstance(names, list) else ""
    return row


def write_training_per_task_csv(output_root: Path, tasks: list[str]) -> Path:
    """Refresh one row per dataset task from current decisions and attempts."""
    output_root = Path(output_root)
    decisions_dir = output_root / "decisions"
    target = output_root / "results" / "per_task.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=target.parent,
                                         prefix=".per_task-", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for task in tasks:
                record, record_type = _latest_record(decisions_dir, task)
                writer.writerow(_row(task, record, record_type))
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return target
