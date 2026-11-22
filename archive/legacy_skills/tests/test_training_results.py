from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
import yaml

from lh_harness.train_terminal_bench_pro import main
from lh_harness.training_results import write_training_per_task_csv


def _phase(root: Path, task: str, phase: str, *, reward: float, inp: int,
           out: int, partial: bool = False) -> Path:
    trial = root / "jobs" / task / phase / "job" / "trial"
    report = trial / "agent" / "skill_harness" / "lh_harness" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "complete", "completion_satisfied": True, "rounds_run": 2}))
    (report.parent.parent / "training_report.json").write_text(json.dumps({
        "usage_totals": {"n_input_tokens": inp, "n_output_tokens": out, "n_cache_tokens": 3},
        "usage_partial": partial,
    }))
    (trial / "result.json").write_text(json.dumps({
        "verifier_result": {"rewards": {"reward": reward}}, "exception_info": None,
    }))
    return report


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_training_csv_uses_harbor_reward_and_counts_both_trials(tmp_path: Path) -> None:
    base = _phase(tmp_path, "one", "baseline", reward=0.0, inp=100, out=20)
    replay = _phase(tmp_path, "one", "candidate", reward=1.0, inp=200, out=50, partial=True)
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    (decisions / "one.json").write_text(json.dumps({
        "task": "one", "run_id": "accepted", "accepted": True,
        "baseline_report_path": str(base), "candidate_report_path": str(replay),
        "baseline_referenced_skill_names": ["existing_skill"],
        "candidate_referenced_skill_names": ["new_skill"],
        "referenced_skill_names": ["existing_skill", "new_skill"],
        "skill_learning": {"action": "CREATE_NEW_SKILL"},
    }))
    rows = _rows(write_training_per_task_csv(tmp_path, ["one", "two"]))
    assert len(rows) == 2
    first = rows[0]
    assert first["correct"] == "True" and first["reward"] == "1.0"
    assert first["baseline_correct"] == "False" and first["candidate_correct"] == "True"
    assert first["input_tokens"] == "300" and first["output_tokens"] == "70"
    assert first["total_tokens"] == "370" and first["token_usage_partial"] == "True"
    assert first["baseline_referenced_skill_names"] == "existing_skill"
    assert first["candidate_referenced_skill_names"] == "new_skill"
    assert rows[1]["status"] == "not_run"


def test_training_csv_shows_latest_failed_force_attempt_not_old_decision(tmp_path: Path) -> None:
    report = _phase(tmp_path, "one", "baseline", reward=1.0, inp=10, out=2)
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    previous = decisions / "one.json"
    previous.write_text(json.dumps({
        "task": "one", "run_id": "old", "accepted": False,
        "baseline_report_path": str(report),
    }))
    attempts = decisions / "attempts" / "one"
    attempts.mkdir(parents=True)
    failed = attempts / "new.json"
    failed.write_text(json.dumps({
        "task": "one", "run_id": "new", "phase": "baseline",
        "reason": "infrastructure_error", "error": "Docker build failed",
    }))
    os.utime(failed, ns=(previous.stat().st_mtime_ns + 1_000_000_000,) * 2)
    row = _rows(write_training_per_task_csv(tmp_path, ["one"]))[0]
    assert row["run_id"] == "new" and row["record_type"] == "attempt"
    assert row["status"] == "infrastructure_error" and row["correct"] == ""
    assert row["error"] == "Docker build failed"


def test_training_main_refreshes_csv_after_failed_task(tmp_path: Path, monkeypatch) -> None:
    task_dir = tmp_path / "train_dataset" / "terminal-bench-pro" / "one"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("")
    (task_dir / "instruction.md").write_text("")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"agents": [{"kwargs": {}}]}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("LH_CLAUDE_OFFLINE_BINARY", "already-set")

    def fake_train_one(task, **kwargs):
        path = kwargs["decisions_dir"] / "attempts" / task / "new.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "task": task, "run_id": "new", "phase": "baseline",
            "reason": "infrastructure_error", "error": "Docker build failed",
        }))
        raise RuntimeError("Docker build failed")

    monkeypatch.setattr("lh_harness.train_terminal_bench_pro.train_one", fake_train_one)
    with pytest.raises(RuntimeError, match="Docker build failed"):
        main(["--template", str(template)])
    csv_path = tmp_path / "output" / "terminal-bench-pro" / "results" / "per_task.csv"
    row = _rows(csv_path)[0]
    assert row["task"] == "one" and row["status"] == "infrastructure_error"
    assert row["run_id"] == "new"
