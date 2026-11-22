"""Bounded paired validation of a reused Skill on a different training task."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .skill_learning import SkillBank, task_success


def _provider_cost(report: dict[str, Any], output: Path) -> int | None:
    """Full inference usage, including the final reply; never a text estimate."""
    from .train_terminal_bench_pro import _trial_token_usage
    observed = _trial_token_usage(report, output)
    if observed.get("usage_partial") is True:
        return None
    totals = observed.get("usage_totals") or {}
    if not totals:
        try:
            totals = json.loads((output.parent.parent/'result.json').read_text()).get('agent_result') or {}
        except (OSError, ValueError):
            return None
    inp, out = totals.get('n_input_tokens'), totals.get('n_output_tokens')
    return inp+out if type(inp) is int and type(out) is int and inp >= 0 and out >= 0 else None


def validate_reused_skill(*, task: str, template: dict[str, Any], dataset: Path,
        frozen: SkillBank, bank_path: Path, baseline: dict[str, Any], baseline_output: Path,
        generated_dir: Path, jobs_root: Path, run_id: str, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled", "additional_trials": 0}
    from .train_terminal_bench_pro import (_write_trial_config, _run_trial, _routes,
        _harbor_outcome, _trial_token_usage, _hard_regression, _require_valid_trial_report)
    language = template["agents"][0]["kwargs"].get("prompt_language", "en")
    task_id = hashlib.sha256(("terminal_bench_pro\n"+str(baseline.get("task", ""))).encode()).hexdigest()
    actual = {(s["skill_id"], s["version"]) for r in _routes(baseline_output) if r.get("injected_context")
              for s in r.get("selected_skills", [])}
    pending = [s for s in frozen.skills if s["status"] == "provisional" and
               s.get("lifecycle") != "quarantined" and s.get("provenance", {}).get("task_id") != task_id and
               task_id not in s.get("provenance", {}).get("source_task_ids", []) and
               not any(r["task_id"] == task_id for r in s.get("independent_validation_results", [])) and
               ((s["skill_id"], s["version"]) in actual or any(
                   (old["skill_id"], old["version"]) in actual for old in frozen.shadowed_skills
                   if old["skill_id"] == s["skill_id"]))]
    if not pending:
        return {"status": "no_independent_applicable_candidate", "additional_trials": 0}
    outcome = _harbor_outcome(baseline_output)
    if type(outcome.get("correct")) is not bool:
        return {"status": "external_outcome_unavailable", "additional_trials": 0}
    # One bounded contrast per task. The original run serves as one arm; all
    # other Skills, prompts, model and tool budgets stay identical.
    skill = min(pending, key=lambda s: (len(s.get("independent_validation_results", [])), s["skill_id"]))
    sid, version = skill["skill_id"], skill["version"]
    original_is_treatment = (sid, version) in actual
    contrast = copy.deepcopy(frozen)
    contrast.path = jobs_root/task/"bank_snapshots"/f"{run_id}-contrast.json"
    proposal_path = None
    if original_is_treatment:
        contrast.skills = [s for s in contrast.skills if s["skill_id"] != sid]
        previous = [s for s in contrast.shadowed_skills if s["skill_id"] == sid]
        if previous:
            contrast.skills.append(max(previous, key=lambda s: s["version"]))
        contrast.shadowed_skills = [s for s in contrast.shadowed_skills if s["skill_id"] != sid]
    else:
        proposal_path = generated_dir/f"{task}-{run_id}-validation-target.json"
        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        proposal_path.write_text(json.dumps({"kind": "skill", "candidate": skill}, ensure_ascii=False)+"\n")
    contrast.save()
    config = generated_dir/f"{task}-{run_id}-skill-contrast.yaml"
    jobs = jobs_root/task/"skill-validation"
    _write_trial_config(template, task=task, phase="skill-validation", dataset=dataset,
                       bank=contrast.path, jobs_dir=jobs, config_path=config, candidate_path=proposal_path)
    result = {"status": "pending", "skill_id": sid, "version": version, "task_id": task_id,
              "additional_trials": 1, "contrast_config": str(config)}
    try:
        replay, replay_output = _run_trial(config, jobs)
        _require_valid_trial_report(replay, replay_output, "skill-validation")
        replay_outcome = _harbor_outcome(replay_output)
        if type(replay_outcome.get("correct")) is not bool:
            return {**result, "status": "external_outcome_unavailable"}
        candidate, candidate_out = (baseline, baseline_output) if original_is_treatment else (replay, replay_output)
        control, control_out = (replay, replay_output) if original_is_treatment else (baseline, baseline_output)
        cand_score, control_score = _harbor_outcome(candidate_out), _harbor_outcome(control_out)
        selected = any(r.get("injected_context") and any(s["skill_id"] == sid and s["version"] == version
            for s in r.get("selected_skills", [])) for r in _routes(candidate_out))
        # A failed ablation (the target remains injected) is not a contrast.
        control_selected = any(any(s["skill_id"] == sid and s["version"] == version for s in r.get("selected_skills", []))
                               for r in _routes(control_out) if r.get("injected_context"))
        if not selected or control_selected:
            return {**result, "status": "ineffective_contrast"}
        record = {"task_id": task_id, "candidate_selected": True,
            "baseline_success": control_score["correct"],
            "candidate_success": cand_score["correct"] and task_success(candidate, language),
            "candidate_harness_success": task_success(candidate, language),
            "hard_constraint_regression": _hard_regression(candidate, language) and not _hard_regression(control, language),
            "baseline_tokens": _provider_cost(control, control_out),
            "candidate_tokens": _provider_cost(candidate, candidate_out),
            "baseline_result_path": control_score["result_path"], "candidate_result_path": cand_score["result_path"],
            "evidence_kind": "paired_fresh_trial", "controlled_component": sid,
            "config_fingerprint": hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()}
        bank = SkillBank.load(bank_path)
        bank.record_independent_validation(sid, version, record)
        try:
            bank.promote_skill(sid, version)
            result["promoted"] = True
        except ValueError as exc:
            result.update(promoted=False, promotion_reason=str(exc))
        result.update(status="recorded", evidence=record,
                      contrast_token_usage={**_trial_token_usage(replay, replay_output),
                                            "total_tokens": _provider_cost(replay, replay_output),
                                            "source": "provider_totals"})
    except Exception as exc:
        # A valid task result survives an infrastructure failure in the
        # optional calibration arm. No evidence or promotion is fabricated.
        result.update(status="validation_error", error_type=type(exc).__name__, error=str(exc))
    return result
