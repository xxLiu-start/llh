"""Sequential SkillBank training with paired fresh Harbor trials.

Each baseline/candidate comparison uses the same Terminal-Bench Pro task. Harbor
creates a fresh Docker environment for each trial; the driver updates only the
SkillBank on the host after comparing complete task reports.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import contextlib
import json
import logging
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .auditor_agent import parse_audit_report
from .claude_cache import find_cached_claude_binary as _find_cached_claude_binary
from .harbor_url import claude_code_upstream_base_url
from .skill_learning import SkillBank, accept_candidate, commit_validated_candidate, record_selected_skill_outcomes, resolve_trajectory_round_ref, stage_skill_candidate, task_success, validate_skill
from .training_provenance import ProvenanceWriter
from .training_results import write_training_per_task_csv
from .verifier_uv import add_verifier_uv, verifier_uv_host_env


logger = logging.getLogger(__name__)


def discover_tasks(root: Path) -> list[str]:
    if not root.is_dir():
        raise ValueError(f"training dataset does not exist: {root}")
    return sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and (path / "task.toml").is_file() and (path / "instruction.md").is_file()
    )


def _write_trial_config(
    template: dict[str, Any], *, task: str, phase: str,
    dataset: Path, bank: Path, jobs_dir: Path, config_path: Path,
    candidate_path: Path | None = None,
) -> None:
    import yaml

    config = copy.deepcopy(template)
    # Harbor reuses an existing job directory when job_name is repeated.
    # Give every baseline/candidate replay a fresh job so --force really runs
    # a new trial while keeping prior logs intact.
    config["job_name"] = f"skill-{task}-{phase}-{uuid.uuid4().hex[:8]}"
    config["jobs_dir"] = str(jobs_dir)
    config["n_attempts"] = 1
    config["n_concurrent_trials"] = 1
    config["datasets"] = [{"path": str(dataset), "task_names": [task]}]
    offline_claude = os.getenv("LH_CLAUDE_OFFLINE_BINARY")
    if offline_claude:
        offline_binary = Path(offline_claude).expanduser().resolve()
        if not offline_binary.is_file() or not os.access(offline_binary, os.X_OK):
            raise ValueError(f"offline Claude Code binary is missing or not executable: {offline_binary}")
        offline_compose = Path(__file__).resolve().parents[2] / "training" / "docker_claude_offline.yaml"
        config.setdefault("environment", {}).setdefault("extra_docker_compose", []).append(str(offline_compose))
    if task == "jq-github-contributor-report":
        # This task downloads yq from GitHub during its Docker build. The
        # server reaches GitHub through its loopback proxy, which Docker build
        # containers cannot reach directly. A temporary bridge relay makes it
        # available without changing the task source or runtime networking.
        proxy_compose = Path(__file__).resolve().parents[2] / "training" / "docker_build_proxy.yaml"
        config.setdefault("environment", {}).setdefault("extra_docker_compose", []).append(str(proxy_compose))
    add_verifier_uv(config, dataset=dataset, task=task)
    config["agents"][0].setdefault("env", {})["LH_HARNESS_MODEL_IO_TRACE"] = "1"
    kwargs = config["agents"][0]["kwargs"]
    language = kwargs.get("prompt_language", "en")
    if language not in {"en", "zh"}:
        raise ValueError("Skill training prompt language must be en or zh")
    kwargs["prompt_language"] = language
    kwargs["skill_bank_path"] = str(bank)
    kwargs["learning_enabled"] = True
    kwargs["defer_skill_learning"] = True
    if offline_claude:
        kwargs["offline_claude_binary"] = True
    if candidate_path is not None:
        kwargs["candidate_path"] = str(candidate_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run_trial(config_path: Path, jobs_dir: Path) -> tuple[dict[str, Any], Path]:
    import yaml

    trial_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    job_name = trial_config["job_name"]
    job_dir = jobs_dir / job_name
    if job_dir.exists():
        raise RuntimeError(f"fresh Harbor job already exists: {job_dir}")
    # Harbor applies AgentConfig.env as a scoped environment that overrides
    # the role adapter's per-exec ANTHROPIC_BASE_URL. Normalize the value
    # before Harbor expands ${ANTHROPIC_BASE_URL} into that scoped environment.
    harbor_env = verifier_uv_host_env(trial_config)
    if harbor_env.get("ANTHROPIC_BASE_URL"):
        harbor_env["ANTHROPIC_BASE_URL"] = claude_code_upstream_base_url(harbor_env["ANTHROPIC_BASE_URL"])
    proxy_overrides = trial_config.get("environment", {}).get("extra_docker_compose", [])
    use_build_proxy = any(Path(path).name == "docker_build_proxy.yaml" for path in proxy_overrides)
    from lh_harness.docker_build_proxy import docker_build_proxy

    with docker_build_proxy() if use_build_proxy else contextlib.nullcontext(None) as build_proxy:
        if build_proxy:
            harbor_env["LH_DOCKER_BUILD_PROXY"] = build_proxy
        completed = subprocess.run(["harbor", "run", "-c", str(config_path)], check=False, env=harbor_env)
    reports = sorted(job_dir.rglob("training_report.json"), key=lambda path: path.stat().st_mtime_ns)
    if completed.returncode != 0 or not reports:
        exceptions = sorted(job_dir.rglob("exception.txt"), key=lambda path: path.stat().st_mtime_ns)
        detail = ""
        if exceptions:
            exception_path = exceptions[-1]
            detail = f"\nHarbor exception: {exception_path}\n{exception_path.read_text(encoding='utf-8', errors='replace')[-3000:]}"
        raise RuntimeError(
            f"Harbor trial failed (exit {completed.returncode}; fresh report: {bool(reports)}): {config_path}{detail}"
        )
    path = reports[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("report"), dict):
        raise ValueError(f"invalid training report: {path}")
    return payload["report"], path.parent


def _learning_reason(learning: dict[str, Any]) -> str:
    if not learning:
        return "learning_report_missing"
    if learning.get("learning_error"):
        return "skill_learning_error"
    if learning.get("skip_reason"):
        return "diagnosis_skipped"
    if learning.get("diagnosis_attempted") is not True and learning.get("action") == "NO_CHANGE":
        return "legacy_unverified_no_change"
    if learning.get("diagnosis_attempted") is False and not learning.get("action"):
        return "diagnosis_status_unknown"
    if learning.get("action") == "EVOLVE":
        requests = learning.get("request_outcomes") or []
        if any(isinstance(item, dict) and item.get("status") == "provisional" for item in requests):
            return "evolution_candidate_artifact_missing"
        return "evolution_candidates_rejected" if requests else "evolution_no_update"
    return {
        "NO_CHANGE": "diagnosis_no_change",
        "INSUFFICIENT_EVIDENCE": "diagnosis_insufficient_evidence",
        "NO_NEW_SKILL": "summary_no_new_skill",
    }.get(learning.get("action"), "candidate_not_proposed")


def _write_attempt(path: Path, result: dict[str, Any]) -> None:
    _ensure_skill_reference_fields(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _finish_decision(
    decisions_dir: Path, decision_path: Path, attempt_path: Path,
    run_id: str, result: dict[str, Any], provenance: ProvenanceWriter,
) -> dict[str, Any]:
    _ensure_skill_reference_fields(result)
    decisions_dir.mkdir(parents=True, exist_ok=True)
    _write_attempt(attempt_path, result)
    if decision_path.exists():
        history_path = decisions_dir / "history" / result["task"] / f"{run_id}.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(history_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(decision_path.read_bytes())
    temporary_decision = decisions_dir / f".{result['task']}-{run_id}.tmp"
    fd = os.open(temporary_decision, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary_decision, decision_path)
    provenance.append("decision", decision_path=str(decision_path), result=result)
    return result


def _write_skill_tmp(path: Path, value: dict[str, Any]) -> bool:
    """Save a view-only Skill artifact; never let archive I/O affect training."""
    try:
        payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        return True
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        logger.warning("Could not save view-only temporary Skill %s: %s", path, exc)
        return False


def _require_valid_trial_report(report: dict[str, Any], output_dir: Path, phase: str) -> None:
    # A gateway may return a successful HTTP response while serializing tool
    # calls into plain text. Claude Code cannot execute those calls, so this is
    # a provider/tool-protocol failure rather than a valid no-candidate trial.
    pseudo_tool_calls = any(
        isinstance(round_.get("executor_output"), str)
        and "<｜｜DSML｜｜ calls>" in round_["executor_output"]
        and "invoke name=" in round_["executor_output"]
        for round_ in report.get("rounds", []) if isinstance(round_, dict)
    )
    real_tool_calls = False
    if pseudo_tool_calls:
        for stream_path in (output_dir / "episodes" / "cli_task").glob("ep*/claude_stream.jsonl"):
            for line in stream_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                message = record.get("message") if isinstance(record, dict) else None
                if isinstance(message, dict) and any(
                    isinstance(block, dict) and block.get("type") == "tool_use"
                    for block in message.get("content", [])
                ):
                    real_tool_calls = True
                    break
            if real_tool_calls:
                break
    if pseudo_tool_calls and not real_tool_calls:
        raise RuntimeError(
            f"{phase} trial could not execute tools: the model returned DSML tool-call markup "
            f"as plain text rather than Anthropic tool_use blocks. "
            f"Check structured tool calling with `python training/check_model_endpoint.py --direct --tools`. "
            f"Report: {output_dir / 'lh_harness' / 'report.json'}"
        )
    if report.get("status") == "failed":
        reason = report.get("failure_reason") or report.get("abort_reason") or "unknown provider/runtime error"
        proxy_detail = ""
        proxy_logs = sorted(
            output_dir.rglob("request_proxy_requests.jsonl"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for proxy_log in proxy_logs:
            for line in reversed(proxy_log.read_text(encoding="utf-8", errors="replace").splitlines()):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("event") in {"upstream_http_error", "upstream_error"}:
                    detail = str(event.get("body_head") or event.get("error") or "").replace("\n", " ")[:500]
                    proxy_detail = (
                        f" Provider HTTP {event.get('status', 'error')}: {detail}."
                        f" Proxy log: {proxy_log}"
                    )
                    break
            if proxy_detail:
                break
        raise RuntimeError(
            f"{phase} trial failed before Skill learning: {reason}.{proxy_detail} "
            f"Report: {output_dir / 'lh_harness' / 'report.json'}"
        )


def _routes(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "lh_harness" / "role_orchestration" / "skill_routes.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def _harbor_outcome(output_dir: Path) -> dict[str, Any]:
    """Read the completed trial's independent verifier result, if available."""
    result_path = output_dir.parent.parent / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"correct": None, "reward": None, "result_path": str(result_path)}
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    if type(reward) not in (int, float) or not math.isfinite(reward) or result.get("exception_info"):
        return {"correct": None, "reward": None, "result_path": str(result_path)}
    return {"correct": reward >= 1.0, "reward": float(reward),
            "result_path": str(result_path)}


def _referenced_skill_names(routes: list[dict[str, Any]]) -> list[str]:
    """Return stable Skill names actually selected by the Router in a trial."""
    names = {
        item["skill_id"].strip()
        for route in routes if isinstance(route, dict)
        for item in route.get("selected_skills", []) if isinstance(item, dict)
        if isinstance(item.get("skill_id"), str) and item["skill_id"].strip()
    }
    return sorted(names)


def _ensure_skill_reference_fields(result: dict[str, Any]) -> None:
    """Keep per-phase and aggregate Skill references in every task record."""
    baseline = result.get("baseline_referenced_skill_names")
    candidate = result.get("candidate_referenced_skill_names")
    baseline_names = sorted({item for item in baseline or [] if isinstance(item, str) and item})
    candidate_names = sorted({item for item in candidate or [] if isinstance(item, str) and item})
    result["baseline_referenced_skill_names"] = baseline_names
    result["candidate_referenced_skill_names"] = candidate_names
    result["referenced_skill_names"] = sorted(set(baseline_names) | set(candidate_names))


def _trial_token_usage(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Record provider usage for every completed trial without inventing counts."""
    learning = report.get("skill_learning") or {}
    session_total = learning.get("task_tokens") if isinstance(learning, dict) else None
    record: dict[str, Any] = {
        "task_tokens": session_total if type(session_total) is int and session_total >= 0 else None,
        "total_tokens": session_total if type(session_total) is int and session_total >= 0 else None,
        "source": "skill_session" if type(session_total) is int and session_total >= 0 else None,
        "usage_totals": None, "usage_partial": None,
    }
    path = output_dir / "training_report.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            totals = payload.get("usage_totals") if isinstance(payload, dict) else None
            if isinstance(totals, dict):
                record["usage_totals"] = totals
                record["usage_partial"] = payload.get("usage_partial")
                inp, out = totals.get("n_input_tokens"), totals.get("n_output_tokens")
                if record["total_tokens"] is None and type(inp) is int and type(out) is int and inp >= 0 and out >= 0:
                    record["total_tokens"] = inp + out
                    record["source"] = "harbor_provider_totals"
        except (OSError, ValueError):
            pass
    record["missing"] = record["total_tokens"] is None
    record["comparison_missing"] = record["task_tokens"] is None
    return record


def _effective(kind: str, candidate: Any, baseline_dir: Path, candidate_dir: Path) -> bool:
    candidate_routes = _routes(candidate_dir)
    if kind == "skill":
        return any(
            route.get("injected_context") and any(
                item.get("skill_id") == candidate["skill_id"] and item.get("version") == candidate["version"]
                for item in route.get("selected_skills", [])
            ) for route in candidate_routes
        )
    def signature(routes: list[dict[str, Any]]) -> list[Any]:
        return [(r.get("round"), r.get("role"), r.get("selected_skills")) for r in routes]
    return bool(candidate_routes and signature(candidate_routes) != signature(_routes(baseline_dir)))


def _hard_regression(report: dict[str, Any], language: str = "en") -> bool:
    for item in report.get("rounds", []):
        audit = item.get("auditor_status", {})
        if isinstance(audit, dict) and any(audit.get(key) for key in (
            "verifier_workspace_mutation_detected", "auditor_workspace_mutation_detected", "contract_violation",
        )):
            return True
        text = item.get("auditor_report")
        if isinstance(text, str) and text.strip():
            parsed = parse_audit_report(text, int(item.get("round_index") or 0), language=language)
            if parsed.integrity_status == "violation" or parsed.contract_audit_status == "invalid":
                return True
    return False


def _train_skill_candidates(
    *, candidates: list[dict[str, Any]], proposal: dict[str, Any], proposal_path: Path,
    baseline: dict[str, Any], base_output: Path, frozen_bank: SkillBank, bank_path: Path,
    task: str, template: dict[str, Any], dataset: Path, generated_dir: Path, jobs_root: Path,
    decisions_dir: Path, decision_path: Path, attempt_path: Path, run_id: str,
    skill_tmp_dir: Path, reference_tokens: int, training_language: str,
    provenance: ProvenanceWriter, result: dict[str, Any], validate_replay: bool,
) -> dict[str, Any]:
    result["candidate_kind"] = "skill"
    result["candidate_ids"] = [item["skill_id"] for item in candidates]
    if len(candidates) == 1:
        result["candidate_id"] = candidates[0]["skill_id"]
    temporary_skill_path = skill_tmp_dir / "candidate.json"
    if _write_skill_tmp(temporary_skill_path, candidates if len(candidates) > 1 else candidates[0]):
        result["skill_tmp_path"] = str(temporary_skill_path)
    round_ids = {
        f"round_{round_['round_index']:03d}"
        for round_ in (baseline.get("rounds") or [])
        if isinstance(round_, dict) and isinstance(round_.get("round_index"), int)
    }
    has_trajectory = any(
        isinstance(round_, dict) and any(
            isinstance(round_.get(key), str) and round_[key].strip()
            for key in ("plan_text", "executor_output", "auditor_report")
        ) for round_ in (baseline.get("rounds") or [])
    )
    action = (proposal.get("assessment") or {}).get("action")
    requested_revisions = {
        request.get("target_skill_id") for request in
        (proposal.get("assessment") or {}).get("revision_targets", [])
        if isinstance(request, dict)
    }
    if action == "CREATE_NEW_SKILL":
        if not has_trajectory or any(not any(
            resolve_trajectory_round_ref(ref, round_ids) is not None
            for ref in item.get("provenance", {}).get("evidence_refs", [])
        ) for item in candidates if item["skill_id"] not in requested_revisions):
            result.update(reason="new_skill_missing_execution_evidence", error_stage="candidate_eligibility")
            return _finish_decision(decisions_dir, decision_path, attempt_path, run_id, result, provenance)
    bank = SkillBank.load(bank_path)
    try:
        for item in candidates:
            previous = next((skill for skill in bank.skills if skill["skill_id"] == item["skill_id"]), None)
            is_revision = action == "REVISE_SKILL" or item["skill_id"] in requested_revisions
            if is_revision and previous is None:
                raise ValueError("revision target is absent from SkillBank")
            if not is_revision and previous is not None:
                raise ValueError("new Skill ID already exists in SkillBank")
            provenance.candidate("proposed", {**proposal, "candidate": item}, source_path=proposal_path, previous_skill=previous)
            stage_skill_candidate(bank, item, save=False)
        bank.save()
    except Exception as exc:
        result.update(reason="skillbank_stage_error", error_stage="skillbank_stage", error=str(exc))
        _write_attempt(attempt_path, result)
        provenance.append("stage_failure", error_type=type(exc).__name__, error=str(exc))
        raise
    provenance.bank_snapshot("before_candidate_replay" if validate_replay else "candidate_staged_without_replay", bank)
    if not validate_replay:
        validation = {"status": "skipped", "accepted": False,
                      "reason": "validation_replay_disabled"}
        outcomes = []
        for item in candidates:
            staged = next(skill for skill in bank.skills if skill["skill_id"] == item["skill_id"])
            staged["validation"] = validation
            outcomes.append({"skill_id": item["skill_id"], "version": item["version"],
                             "selected": None, "bank_committed": False})
            provenance.candidate("validation_skipped", {**proposal, "candidate": item,
                                                        "validation": validation}, source_path=proposal_path)
        bank.save()
        result.update(reason="validation_replay_disabled", validation=validation, candidates=outcomes)
        _write_skill_tmp(skill_tmp_dir / "validation.json", {
            "validation": validation, "candidates": outcomes, "bank_committed": False,
            "candidate_report_path": None,
        })
        return _finish_decision(decisions_dir, decision_path, attempt_path, run_id, result, provenance)
    phase = "candidate"
    candidate_jobs = jobs_root / task / phase
    candidate_config = generated_dir / f"{task}-{run_id}-{phase}.yaml"
    frozen_bank.save()
    _write_trial_config(template, task=task, phase=phase, dataset=dataset, bank=frozen_bank.path,
                        jobs_dir=candidate_jobs, config_path=candidate_config, candidate_path=proposal_path)
    try:
        candidate_report, candidate_output = _run_trial(candidate_config, candidate_jobs)
    except Exception as exc:
        provenance.append("trial_failure", phase=phase, error_type=type(exc).__name__, error=str(exc))
        _write_attempt(attempt_path, {**result, "phase": phase, "reason": "infrastructure_error",
                                      "error": str(exc), "error_type": type(exc).__name__,
                                      "candidate_token_usage": {"task_tokens": None, "total_tokens": None, "missing": True, "reason": "trial_report_unavailable"}})
        raise
    finally:
        provenance.ingest_model_io(candidate_config, candidate_jobs, phase=phase)
        provenance.bank_snapshot_file(f"after_{phase}_trial", frozen_bank.path)
    result["candidate_token_usage"] = _trial_token_usage(candidate_report, candidate_output)
    provenance.append("trial_report", phase=phase,
                      report_path=str(candidate_output / "lh_harness" / "report.json"), report=candidate_report,
                      token_usage=result["candidate_token_usage"])
    try:
        _require_valid_trial_report(candidate_report, candidate_output, phase)
    except Exception as exc:
        provenance.append("trial_failure", phase=phase, error_type=type(exc).__name__, error=str(exc))
        _write_attempt(attempt_path, {**result, "phase": phase, "reason": "infrastructure_error",
                                      "error": str(exc), "error_type": type(exc).__name__, "candidate_report": candidate_report})
        raise
    routes = _routes(candidate_output)
    result["candidate_referenced_skill_names"] = _referenced_skill_names(routes)
    selected = {item["skill_id"]: _effective("skill", item, base_output, candidate_output) for item in candidates}
    effective = all(selected.values())
    candidate_outcome = _harbor_outcome(candidate_output)
    result["candidate_harbor_outcome"] = candidate_outcome
    candidate_harness_success = task_success(candidate_report, training_language)
    candidate_correct = candidate_outcome["correct"]
    result["candidate_learning_outcome_disagreement"] = (
        candidate_correct != candidate_harness_success if candidate_correct is not None else None
    )
    record_selected_skill_outcomes(bank, routes, success=(
        candidate_correct if candidate_correct is not None else candidate_harness_success
    ))
    baseline_tokens = result["baseline_token_usage"]["task_tokens"]
    candidate_tokens = result["candidate_token_usage"]["task_tokens"]
    hard_regression = _hard_regression(candidate_report, training_language)
    baseline_correct = result["baseline_harbor_outcome"]["correct"]
    verifier_scores_available = baseline_correct is not None and candidate_correct is not None
    accepted, delta = accept_candidate(
        baseline, candidate_report, baseline_tokens=baseline_tokens, candidate_tokens=candidate_tokens,
        reference_tokens=reference_tokens, effective=effective,
        hard_constraint_regression=hard_regression, language=training_language,
        baseline_success_override=baseline_correct if verifier_scores_available else None,
        candidate_success_override=candidate_correct if verifier_scores_available else None,
    )
    validation = {
        "accepted": accepted, "reward_delta": delta, "effective": effective,
        "scope": "source_task_combination", "independent_causal_credit": False,
        "selected_candidates": selected,
        "score_source": "harbor_verifier" if verifier_scores_available else "harness_fallback",
        "baseline_success": baseline_correct if verifier_scores_available else task_success(baseline, training_language),
        "candidate_success": candidate_correct if verifier_scores_available else candidate_harness_success,
        "baseline_harness_success": task_success(baseline, training_language),
        "candidate_harness_success": candidate_harness_success,
        "baseline_tokens": baseline_tokens, "candidate_tokens": candidate_tokens,
        "hard_constraint_regression": hard_regression,
    }
    for item in candidates:
        staged = next(skill for skill in bank.skills if skill["skill_id"] == item["skill_id"])
        staged["validation"] = validation
        if hard_regression:
            staged["lifecycle"] = "quarantined"
    bank.save()
    for item in candidates:
        provenance.candidate("assessed", {**proposal, "candidate": item, "validation": validation},
                             source_path=proposal_path)
    provenance.bank_snapshot("after_candidate_observation", bank)
    outcomes = [{"skill_id": item["skill_id"], "version": item["version"],
                 "selected": selected[item["skill_id"]], "bank_committed": False}
                for item in candidates]
    result.update(accepted=accepted, validation=validation, candidates=outcomes,
                  candidate_status=candidate_report.get("status"),
                  candidate_report_path=str(candidate_output / "lh_harness" / "report.json"))
    result["reason"] = "provisional_pending_independent_validation" if accepted else "candidate batch failed full-task acceptance checks"
    _write_skill_tmp(skill_tmp_dir / "validation.json", {
        "validation": validation, "candidates": outcomes, "bank_committed": False,
        "candidate_report_path": result["candidate_report_path"],
    })
    return _finish_decision(decisions_dir, decision_path, attempt_path, run_id, result, provenance)


def _transfer_evolution_candidates(
    *, proposal: dict[str, Any], trial_bank_path: Path, bank_path: Path,
    language: str,
) -> tuple[SkillBank, list[dict[str, Any]], bool]:
    """Transfer only independently admissible trial candidates to the training bank.

    The baseline runs on a disposable SkillBank snapshot. Its provisional
    entries must be copied to the persistent bank after Harbor outcome checks;
    copying the entire snapshot would overwrite newer observational counts.
    """
    source = SkillBank.load(trial_bank_path)
    target = SkillBank.load(bank_path)
    outcomes: list[dict[str, Any]] = []
    transferred_ids: set[str] = set()
    guidance_fields = (
        "skill_id", "category_id", "version", "version_parent", "content_language",
        "benchmark_scope", "interaction_modes", "problem_mechanism", "routing_summary",
        "subcases", "triggers", "exclusions", "roles", "compatible_with",
        "conflicts_with", "provenance", "capability_mapping",
        "routing_metadata",
    )
    for candidate in proposal["candidate"]:
        outcome = {"kind": "skill", "skill_id": candidate.get("skill_id") if isinstance(candidate, dict) else None,
                   "version": candidate.get("version") if isinstance(candidate, dict) else None,
                   "status": "transfer_failed"}
        outcomes.append(outcome)
        try:
            if not isinstance(candidate, dict):
                raise ValueError("evolution candidate must be an object")
            validate_skill(candidate, language=language, candidate=True)
            staged_source = next((skill for skill in source.skills
                                  if skill["skill_id"] == candidate["skill_id"]
                                  and skill["version"] == candidate["version"]
                                  and skill["status"] == "provisional"), None)
            if staged_source is None or any(staged_source.get(key) != candidate.get(key)
                                             for key in guidance_fields):
                raise ValueError("candidate does not match the provisional trial snapshot")
            stage_skill_candidate(target, candidate, save=False)
            transferred_ids.add(candidate["skill_id"])
            outcome["status"] = "provisional"
        except (TypeError, ValueError) as exc:
            outcome["reason"] = str(exc)
    router_transferred = False
    for update in proposal.get("routing_updates", []):
        entry = {"kind": "routing_metadata", "skill_id": update.get("skill_id"), "status": "transfer_failed"}
        outcomes.append(entry)
        try:
            src = next(s for s in source.skills if s["skill_id"] == update["skill_id"] and s["version"] == update["version"])
            dst = next(s for s in target.skills if s["skill_id"] == update["skill_id"] and s["version"] == update["version"])
            if src.get("routing_metadata") != update.get("metadata"):
                raise ValueError("routing update does not match trial snapshot")
            target.history.append({"kind": "routing_metadata", "skill_id": dst["skill_id"],
                                   "old": copy.deepcopy(dst.get("routing_metadata") or {}),
                                   "new": copy.deepcopy(update["metadata"]), "evidence_refs": update.get("evidence_refs", [])})
            dst["routing_metadata"] = copy.deepcopy(update["metadata"])
            entry["status"] = "provisional"
            router_transferred = True
        except (KeyError, ValueError, StopIteration) as exc:
            entry["reason"] = str(exc)
    router_prompt = proposal.get("router_candidate")
    if router_prompt is not None:
        outcome = {"kind": "router", "status": "transfer_failed"}
        outcomes.append(outcome)
        try:
            source_router = next((entry for entry in source.router_candidates
                                  if entry.get("language") == language
                                  and entry.get("prompt") == router_prompt
                                  and entry.get("status") == "provisional"), None)
            if source_router is None:
                raise ValueError("Router candidate is absent from the trial snapshot")
            if any(dep not in transferred_ids for dep in source_router.get("dependencies", [])):
                raise ValueError("Router candidate depends on a Skill that was not transferred")
            if any(entry.get("language") == language and entry.get("prompt") == router_prompt
                   for entry in target.router_candidates):
                raise ValueError("Router candidate is already present")
            target.router_candidates.append(copy.deepcopy(source_router))
            router_transferred = True
            outcome["status"] = "provisional"
        except (TypeError, ValueError) as exc:
            outcome["reason"] = str(exc)
    if transferred_ids or router_transferred:
        target.save()
    return target, outcomes, router_transferred


def train_one(
    task: str, *, template: dict[str, Any], dataset: Path, bank_path: Path,
    generated_dir: Path, jobs_root: Path, decisions_dir: Path, reference_tokens: int,
    force: bool = False, validate_replay: bool = True,
) -> dict[str, Any]:
    training_language = template["agents"][0]["kwargs"].get("prompt_language", "en")
    decision_path = decisions_dir / f"{task}.json"
    if decision_path.exists() and not force:
        previous = json.loads(decision_path.read_text(encoding="utf-8"))
        # Older runs recorded provider failures as ordinary no-candidate
        # decisions. Also retry the specific evidence-reference failure fixed
        # in SkillSession so an old decision cannot hide the repaired learner.
        old_evidence_ref_failure = (
            previous.get("reason") == "skill_learning_error"
            and isinstance(previous.get("skill_learning"), dict)
            and previous["skill_learning"].get("learning_error")
                == "update cites a round absent from the recorded trajectory"
        )
        same_replay_mode = previous.get("replay_validation_enabled", True) == validate_replay
        if (previous.get("baseline_status") != "failed" and not old_evidence_ref_failure and same_replay_mode
                and previous.get("skill_method_version") == 2):
            return {**previous, "execution": "reused_prior_decision", "decision_path": str(decision_path)}
    # Both complete runs must see the same persistent bank. The baseline may
    # accumulate positive evidence, so keep it on a disposable snapshot.
    frozen_bank = SkillBank.load(bank_path)
    run_id = uuid.uuid4().hex
    attempt_path = decisions_dir / "attempts" / task / f"{run_id}.json"
    skill_tmp_dir = decisions_dir.parent / "skill_tmp" / task / run_id
    print(f"Fresh training run: {task} ({run_id}); attempt record: {attempt_path}", file=sys.stderr, flush=True)
    provenance = ProvenanceWriter(
        decisions_dir.parent / "provenance" / f"{task}.jsonl", task=task, run_id=run_id,
        prompt_language=training_language,
    )
    print(f"View-only provenance log: {provenance.path}", file=sys.stderr, flush=True)
    provenance.append("run_start", dataset=str(dataset), bank_path=str(bank_path),
                      prompt_language=training_language, replay_validation_enabled=validate_replay)
    provenance.static_prompts()
    provenance.bank_snapshot("before_baseline", frozen_bank)
    trial_bank_path = jobs_root / task / "bank_snapshots" / f"{run_id}.json"
    frozen_bank.path = trial_bank_path
    frozen_bank.save()
    base_jobs = jobs_root / task / "baseline"
    base_config = generated_dir / f"{task}-{run_id}-baseline.yaml"
    _write_trial_config(template, task=task, phase="baseline", dataset=dataset, bank=trial_bank_path, jobs_dir=base_jobs, config_path=base_config)
    try:
        baseline, base_output = _run_trial(base_config, base_jobs)
    except Exception as exc:
        provenance.append("trial_failure", phase="baseline", error_type=type(exc).__name__, error=str(exc))
        _write_attempt(attempt_path, {"task": task, "run_id": run_id, "execution": "fresh_run", "phase": "baseline", "reason": "infrastructure_error", "error": str(exc), "error_type": type(exc).__name__, "baseline_token_usage": {"task_tokens": None, "total_tokens": None, "missing": True, "reason": "trial_report_unavailable"}})
        raise
    finally:
        provenance.ingest_model_io(base_config, base_jobs, phase="baseline")
        provenance.bank_snapshot_file("after_baseline_trial", trial_bank_path)
    baseline_token_usage = _trial_token_usage(baseline, base_output)
    provenance.append("trial_report", phase="baseline", report_path=str(base_output / "lh_harness" / "report.json"), report=baseline, token_usage=baseline_token_usage)
    try:
        _require_valid_trial_report(baseline, base_output, "baseline")
    except Exception as exc:
        provenance.append("trial_failure", phase="baseline", error_type=type(exc).__name__, error=str(exc))
        _write_attempt(attempt_path, {"task": task, "run_id": run_id, "execution": "fresh_run", "phase": "baseline", "reason": "infrastructure_error", "error": str(exc), "error_type": type(exc).__name__, "baseline_report": baseline, "baseline_token_usage": baseline_token_usage})
        raise
    baseline_routes = _routes(base_output)
    baseline_outcome = _harbor_outcome(base_output)
    baseline_harness_success = task_success(baseline, training_language)
    baseline_correct = baseline_outcome["correct"]
    skill_validation = None
    if (baseline.get("skill_learning") or {}).get("deferred"):
        from .skill_validation import validate_reused_skill
        skill_validation = validate_reused_skill(
            task=task, template=template, dataset=dataset, frozen=frozen_bank,
            bank_path=bank_path, baseline=baseline, baseline_output=base_output,
            generated_dir=generated_dir, jobs_root=jobs_root, run_id=run_id,
            enabled=validate_replay)
        provenance.append("independent_skill_validation", result=skill_validation)
    post_learning = None
    if (baseline.get("skill_learning") or {}).get("deferred"):
        from .skill_feedback import SkillFeedbackAgent, learn_from_verifier
        feedback_dir = base_output / "lh_harness" / "role_orchestration"
        stdout = base_output.parent.parent / "verifier" / "test-stdout.txt"
        feedback = {**baseline_outcome, "harness_success": baseline_harness_success,
                    "verifier_excerpt": stdout.read_text(errors="replace")[-6000:] if stdout.exists() else ""}
        try:
            post_learning = asyncio.run(learn_from_verifier(
                report=baseline, routes=baseline_routes, outcome=feedback,
                bank_path=trial_bank_path, log_dir=feedback_dir, language=training_language,
                agent=SkillFeedbackAgent(template, feedback_dir / "post_verifier_episodes"),
                token_budget=int(template["agents"][0]["kwargs"].get("skill_token_budget", 2000))))
        except Exception as exc:
            post_learning = {"mode": "post_verifier", "learning_phase": "error",
                             "learning_error": str(exc), "error_stage": "post_verifier_learning"}
        # Update only the in-memory learning record. Preserve original role
        # report, completion decision, Harbor reward and provider task usage.
        baseline = {**baseline, "skill_learning": post_learning}
        provenance.append("post_verifier_learning", learning=post_learning, external_outcome=feedback)
    record_selected_skill_outcomes(
        SkillBank.load(bank_path), baseline_routes,
        success=baseline_correct if baseline_correct is not None else baseline_harness_success,
    )
    proposal_path = base_output / "lh_harness" / "role_orchestration" / "skill_candidate.json"
    learning = baseline.get("skill_learning") if isinstance(baseline.get("skill_learning"), dict) else {}
    result: dict[str, Any] = {
        "skill_method_version": 2,
        "task": task, "run_id": run_id, "execution": "fresh_run",
        "training_dataset": str(dataset), "baseline_status": baseline.get("status"),
        "baseline_report_path": str(base_output / "lh_harness" / "report.json"),
        "baseline_token_usage": baseline_token_usage,
        "baseline_harbor_outcome": baseline_outcome,
        "baseline_harness_success": baseline_harness_success,
        "baseline_learning_outcome_disagreement": (
            baseline_correct != baseline_harness_success if baseline_correct is not None else None
        ),
        "baseline_referenced_skill_names": _referenced_skill_names(baseline_routes),
        "skill_learning": learning, "accepted": False,
        "replay_validation_enabled": validate_replay,
        "post_verifier_learning": post_learning,
        "independent_skill_validation": skill_validation,
    }
    if result["baseline_learning_outcome_disagreement"] and not (post_learning and post_learning.get("mode") == "post_verifier"):
        # Learning happened inside the agent before Harbor's independent
        # verifier ran. Its success/failure prompt therefore used the wrong
        # label. Keep the proposal for inspection, but never route or promote
        # a candidate learned from that mismatched label.
        result.update(reason="learning_outcome_mismatch", error_stage="post_verifier_outcome",
                      learning_outcome_note=(
                          "Harbor verifier and Harness completion disagree; the candidate was "
                          "generated before the verifier result and was not staged."
                      ))
        if proposal_path.is_file():
            result["quarantined_candidate_path"] = str(proposal_path)
        provenance.append("learning_outcome_mismatch", harbor_outcome=baseline_outcome,
                          harness_success=baseline_harness_success,
                          candidate_path=str(proposal_path) if proposal_path.is_file() else None)
        return _finish_decision(decisions_dir, decision_path, attempt_path, run_id, result, provenance)
    proposal = None
    if proposal_path.is_file():
        try:
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            if not isinstance(proposal, dict) or proposal.get("kind") not in {"skill", "router", "evolution"}:
                raise ValueError("invalid candidate artifact")
            if proposal["kind"] == "evolution":
                if not isinstance(proposal.get("candidate"), list):
                    raise ValueError("invalid evolution Skill candidates")
                staged_bank, transfers, router_transferred = _transfer_evolution_candidates(
                    proposal=proposal, trial_bank_path=trial_bank_path,
                    bank_path=bank_path, language=training_language,
                )
                transferred_ids = {entry["skill_id"] for entry in transfers
                                   if entry["kind"] == "skill" and entry["status"] == "provisional"}
                result.update(reason="provisional_staged" if transferred_ids or router_transferred
                              else "evolution_transfer_failed", candidate_kind="evolution",
                              candidate_ids=sorted(transferred_ids), transfer_outcomes=transfers,
                              request_outcomes=proposal.get("request_outcomes", []),
                              validation={"status": "pending_independent_validation",
                                          "independent_causal_credit": False})
                for item in proposal["candidate"]:
                    if not isinstance(item, dict) or item.get("skill_id") not in transferred_ids:
                        continue
                    previous = next((skill for skill in staged_bank.shadowed_skills
                                     if skill["skill_id"] == item["skill_id"]), None)
                    provenance.candidate("provisional_staged", {**proposal, "candidate": item},
                                         source_path=proposal_path, previous_skill=previous)
                if router_transferred and proposal.get("router_candidate"):
                    provenance.candidate("router_provisional_staged", {
                        **proposal, "candidate": proposal["router_candidate"]},
                        source_path=proposal_path,
                        previous_router_prompt=staged_bank.router_prompts[training_language])
                if transferred_ids or router_transferred:
                    provenance.bank_snapshot("evolution_provisional_staged", staged_bank)
                return _finish_decision(decisions_dir, decision_path, attempt_path, run_id, result, provenance)
            candidate_value = proposal.get("candidate")
            if (proposal["kind"] == "skill" and not isinstance(candidate_value, (dict, list))) or (
                proposal["kind"] == "router" and not isinstance(candidate_value, str)
            ):
                raise ValueError("invalid candidate content")
            if proposal["kind"] == "skill":
                candidates = candidate_value if isinstance(candidate_value, list) else [candidate_value]
                assessment = proposal.get("assessment") or {}
                action = assessment.get("action")
                legacy_single = action is None and len(candidates) == 1 and isinstance(candidate_value, dict)
                if (not candidates or action not in {"CREATE_NEW_SKILL", "REVISE_SKILL"}
                        and not legacy_single
                        or not legacy_single and learning.get("action") != action
                        or action == "REVISE_SKILL" and len(candidates) != 1):
                    raise ValueError("invalid Skill candidate batch")
                for item in candidates:
                    validate_skill(item, language=training_language, candidate=True)
                if len({item["skill_id"] for item in candidates}) != len(candidates):
                    raise ValueError("duplicate Skill IDs in candidate batch")
                revision_requests = assessment.get("revision_targets", []) if action == "CREATE_NEW_SKILL" else []
                if not isinstance(revision_requests, list) or any(
                    not isinstance(request, dict) or not isinstance(request.get("target_skill_id"), str)
                    for request in revision_requests
                ):
                    raise ValueError("invalid revision requests")
                requested_ids = {request["target_skill_id"] for request in revision_requests}
                if len(requested_ids) != len(revision_requests):
                    raise ValueError("duplicate revision requests")
                current_ids = {item["skill_id"] for item in frozen_bank.skills}
                candidate_ids = {item["skill_id"] for item in candidates}
                if (action == "CREATE_NEW_SKILL" and
                        (candidate_ids & current_ids != requested_ids or not candidate_ids - current_ids)):
                    raise ValueError("candidate batch does not match diagnosed new and revised Skills")
                if action == "REVISE_SKILL" and candidate_ids != {assessment.get("target_skill_id")}:
                    raise ValueError("revision candidate does not match diagnosis")
                injected_ids = {
                    item.get("skill_id") for route in baseline_routes if route.get("injected_context")
                    for item in route.get("selected_skills", []) if isinstance(item, dict)
                }
                if not requested_ids.issubset(injected_ids):
                    raise ValueError("revision target was not injected in baseline")
            elif not candidate_value.strip():
                raise ValueError("empty Router candidate")
        except (OSError, ValueError, TypeError) as exc:
            proposal = None
            result.update(reason="candidate_artifact_invalid", error_stage="candidate_artifact", error=str(exc))
            provenance.append("candidate_artifact_error", source_path=str(proposal_path), error=str(exc))
    if proposal is None:
        if "reason" not in result:
            result["reason"] = _learning_reason(learning)
    else:
        if proposal["kind"] == "skill":
            return _train_skill_candidates(
                candidates=proposal["candidate"] if isinstance(proposal["candidate"], list) else [proposal["candidate"]],
                proposal=proposal, proposal_path=proposal_path, baseline=baseline, base_output=base_output,
                frozen_bank=frozen_bank, bank_path=bank_path, task=task, template=template, dataset=dataset,
                generated_dir=generated_dir, jobs_root=jobs_root, decisions_dir=decisions_dir,
                decision_path=decision_path, attempt_path=attempt_path, run_id=run_id,
                skill_tmp_dir=skill_tmp_dir, reference_tokens=reference_tokens,
                training_language=training_language, provenance=provenance, result=result,
                validate_replay=validate_replay,
            )
        kind, candidate = "router", proposal["candidate"]
        provenance.candidate(
            "proposed", proposal, source_path=proposal_path,
            previous_router_prompt=frozen_bank.router_prompts.get(training_language),
        )
        result["candidate_kind"] = "router"
        result["candidate_id"] = "router_prompt"
        if not validate_replay:
            validation = {"status": "skipped", "accepted": False,
                          "reason": "validation_replay_disabled"}
            result.update(reason="validation_replay_disabled", validation=validation)
            provenance.candidate("validation_skipped", {**proposal, "validation": validation},
                                 source_path=proposal_path,
                                 previous_router_prompt=frozen_bank.router_prompts.get(training_language))
            return _finish_decision(decisions_dir, decision_path, attempt_path, run_id, result, provenance)
        candidate_jobs = jobs_root / task / "candidate"
        candidate_config = generated_dir / f"{task}-{run_id}-candidate.yaml"
        frozen_bank.save()
        _write_trial_config(template, task=task, phase="candidate", dataset=dataset, bank=trial_bank_path, jobs_dir=candidate_jobs, config_path=candidate_config, candidate_path=proposal_path)
        try:
            candidate_report, candidate_output = _run_trial(candidate_config, candidate_jobs)
        except Exception as exc:
            provenance.append("trial_failure", phase="candidate", error_type=type(exc).__name__, error=str(exc))
            _write_attempt(attempt_path, {**result, "phase": "candidate", "reason": "infrastructure_error", "error": str(exc), "error_type": type(exc).__name__, "candidate_token_usage": {"task_tokens": None, "total_tokens": None, "missing": True, "reason": "trial_report_unavailable"}})
            raise
        finally:
            provenance.ingest_model_io(candidate_config, candidate_jobs, phase="candidate")
            provenance.bank_snapshot_file("after_candidate_trial", trial_bank_path)
        result["candidate_token_usage"] = _trial_token_usage(candidate_report, candidate_output)
        provenance.append("trial_report", phase="candidate", report_path=str(candidate_output / "lh_harness" / "report.json"), report=candidate_report, token_usage=result["candidate_token_usage"])
        try:
            _require_valid_trial_report(candidate_report, candidate_output, "candidate")
        except Exception as exc:
            provenance.append("trial_failure", phase="candidate", error_type=type(exc).__name__, error=str(exc))
            _write_attempt(attempt_path, {**result, "phase": "candidate", "reason": "infrastructure_error", "error": str(exc), "error_type": type(exc).__name__, "candidate_report": candidate_report})
            raise
        candidate_routes = _routes(candidate_output)
        result["candidate_referenced_skill_names"] = _referenced_skill_names(candidate_routes)
        candidate_outcome = _harbor_outcome(candidate_output)
        result["candidate_harbor_outcome"] = candidate_outcome
        candidate_harness_success = task_success(candidate_report, training_language)
        candidate_correct = candidate_outcome["correct"]
        result["candidate_learning_outcome_disagreement"] = (
            candidate_correct != candidate_harness_success if candidate_correct is not None else None
        )
        record_selected_skill_outcomes(
            SkillBank.load(bank_path), candidate_routes,
            success=candidate_correct if candidate_correct is not None else candidate_harness_success,
        )
        baseline_tokens = result["baseline_token_usage"]["task_tokens"]
        candidate_tokens = result["candidate_token_usage"]["task_tokens"]
        effective = _effective(kind, candidate, base_output, candidate_output)
        hard_regression = _hard_regression(candidate_report, training_language)
        baseline_correct = result["baseline_harbor_outcome"]["correct"]
        verifier_scores_available = baseline_correct is not None and candidate_correct is not None
        accepted, delta = accept_candidate(
            baseline, candidate_report,
            baseline_tokens=baseline_tokens, candidate_tokens=candidate_tokens,
            reference_tokens=reference_tokens, effective=effective,
            hard_constraint_regression=hard_regression,
            language=training_language,
            baseline_success_override=baseline_correct if verifier_scores_available else None,
            candidate_success_override=candidate_correct if verifier_scores_available else None,
        )
        validation = {
            "accepted": accepted, "reward_delta": delta, "effective": effective,
            "scope": "source_task_combination", "independent_causal_credit": False,
            "score_source": "harbor_verifier" if verifier_scores_available else "harness_fallback",
            "baseline_success": baseline_correct if verifier_scores_available else task_success(baseline, training_language),
            "candidate_success": candidate_correct if verifier_scores_available else candidate_harness_success,
            "baseline_harness_success": task_success(baseline, training_language),
            "candidate_harness_success": candidate_harness_success,
            "baseline_tokens": baseline_tokens, "candidate_tokens": candidate_tokens,
            "hard_constraint_regression": hard_regression,
        }
        result["validation"] = validation
        result["candidate_status"] = candidate_report.get("status")
        result["candidate_report_path"] = str(candidate_output / "lh_harness" / "report.json")
        provenance.candidate(
            "assessed", {**proposal, "validation": validation}, source_path=proposal_path,
            previous_router_prompt=frozen_bank.router_prompts.get(training_language),
        )
        bank = SkillBank.load(bank_path)
        bank.router_candidates.append({
            "language": training_language, "prompt": candidate,
            "status": "quarantined" if hard_regression else "provisional",
            "validation": validation, "evidence_refs": (proposal.get("assessment") or {}).get("evidence_refs", []),
            "dependencies": [],
        })
        bank.save()
        provenance.bank_snapshot("after_router_observation", bank)
        result["reason"] = "provisional_pending_independent_validation" if accepted else "candidate failed full-task acceptance checks"
    return _finish_decision(decisions_dir, decision_path, attempt_path, run_id, result, provenance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a SkillBank on local Terminal-Bench Pro tasks using fresh Harbor trials")
    parser.add_argument("--dataset", type=Path, default=Path("train_dataset/terminal-bench-pro"))
    parser.add_argument("--template", type=Path, default=Path("training/terminal_bench_pro_skill_learning.yaml"))
    parser.add_argument("--bank", type=Path, help="SkillBank path (default: output/<dataset-name>/state/skillbank.json)")
    parser.add_argument("--task", action="append", help="Train only this task name; repeatable")
    parser.add_argument("--start-from", metavar="TASK", help="Start at this task name and continue through later tasks in sorted order")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run tasks that already have decisions")
    parser.add_argument("--validation-replay", action=argparse.BooleanOptionalAction, default=True,
                        help="Replay candidate tasks for validation (default: enabled)")
    args = parser.parse_args(argv)
    root = Path.cwd()
    dataset = args.dataset.resolve()
    output_root = root / "output" / dataset.name
    all_names = discover_tasks(dataset)
    names = list(all_names)
    all_names = set(names)
    if args.start_from:
        if args.start_from not in names:
            parser.error(f"unknown --start-from task: {args.start_from}")
        names = names[names.index(args.start_from):]
    if args.task:
        unknown = set(args.task) - all_names
        if unknown:
            parser.error(f"unknown training tasks: {', '.join(sorted(unknown))}")
        excluded = set(args.task) - set(names)
        if excluded:
            parser.error(f"--task names are not at or after --start-from: {', '.join(sorted(excluded))}")
        names = [name for name in names if name in set(args.task)]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        names = names[:args.limit]
    if args.dry_run:
        print(json.dumps({"dataset": str(dataset), "output_root": str(output_root), "start_from": args.start_from,
                          "tasks": names, "replay_validation_enabled": args.validation_replay,
                          "test_dataset": "eval/TB-harness/datasets/terminal-bench-2-1/tasks"}, ensure_ascii=False, indent=2))
        return 0
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("ANTHROPIC_BASE_URL"):
        parser.error("ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL are required")
    if not os.getenv("LH_CLAUDE_OFFLINE_BINARY"):
        claude_version = os.getenv("CLAUDE_CODE_VERSION", "2.1.176")
        cached_binary = _find_cached_claude_binary(output_root, claude_version)
        if cached_binary is not None:
            os.environ["LH_CLAUDE_OFFLINE_BINARY"] = str(cached_binary)
            print(f"Offline Claude Code: {cached_binary}", file=sys.stderr, flush=True)
    import yaml
    template = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        parser.error("invalid training YAML")
    bank_path = args.bank.resolve() if args.bank else output_root / "state" / "skillbank.json"
    if not bank_path.exists():
        SkillBank.load(bank_path).save()
    write_training_per_task_csv(output_root, all_names)
    for name in names:
        try:
            result = train_one(
                name, template=template, dataset=dataset, bank_path=bank_path,
                generated_dir=output_root / "generated_configs",
                jobs_root=output_root / "jobs", decisions_dir=output_root / "decisions",
                reference_tokens=int(template["agents"][0]["kwargs"].get("skill_reward_reference_tokens", 100_000)),
                force=args.force, validate_replay=args.validation_replay,
            )
        finally:
            write_training_per_task_csv(output_root, all_names)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
