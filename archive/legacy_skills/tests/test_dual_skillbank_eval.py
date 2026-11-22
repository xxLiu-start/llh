from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from lh_harness.eval_terminal_bench_skillbanks import _freeze_bank, _write_eval_config, main
from lh_harness.eval_terminal_bench_pro_original import result_record
from lh_harness.skill_learning import SkillBank


def _skill(skill_id: str, status: str) -> dict:
    view = {"input_use": "Read audited evidence", "output_spec": "Report status and proof",
            "guidance": "Check the artifact", "avoid": "Do not guess"}
    return {
        "skill_id": skill_id, "category_id": "evidence_handoff", "version": 1,
        "version_parent": None, "status": status, "content_language": "en",
        "benchmark_scope": "terminal_bench_2_1", "interaction_modes": ["cli"],
        "problem_mechanism": "Missing audited evidence", "subcases": [],
        "triggers": [], "exclusions": [],
        "roles": {role: dict(view) for role in ("manager", "executor", "auditor")},
        "compatible_with": [], "conflicts_with": [], "provenance": {}, "validation": {},
    }


def _source_bank(tmp_path: Path) -> Path:
    path = tmp_path / "source" / "skillbank.json"
    SkillBank(path, skills=[_skill("regular", "validated"), _skill("temporary", "provisional")]).save()
    return path


def _dataset(tmp_path: Path, *tasks: str) -> Path:
    dataset = tmp_path / "dataset"
    for name in tasks:
        task = dataset / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text('version = "1.0"')
        (task / "instruction.md").write_text("Do the task")
    return dataset


def test_frozen_evaluation_reads_both_banks_and_does_not_follow_source_changes(tmp_path: Path) -> None:
    source = _source_bank(tmp_path)
    dataset = _dataset(tmp_path, "task-a")
    output = tmp_path / "evaluation"
    frozen = _freeze_bank(source, output, dataset)
    assert {skill["skill_id"] for skill in SkillBank.load(frozen).skills} == {"regular", "temporary"}
    first = frozen.read_bytes(), frozen.with_name("skillbank_temporary.json").read_bytes()
    SkillBank(source, skills=[_skill("changed", "validated")]).save()
    assert _freeze_bank(source, output, dataset) == frozen
    assert first == (frozen.read_bytes(), frozen.with_name("skillbank_temporary.json").read_bytes())
    with pytest.raises(ValueError, match="different dataset"):
        _freeze_bank(source, output, _dataset(tmp_path / "other", "task-b"))


def test_generated_evaluation_config_disables_learning_and_uses_frozen_bank(tmp_path: Path) -> None:
    source = _source_bank(tmp_path)
    dataset = _dataset(tmp_path, "task-a")
    output = tmp_path / "evaluation"
    frozen = _freeze_bank(source, output, dataset)
    template = {"agents": [{"kwargs": {"learning_enabled": True, "candidate_path": "old.json"}}]}
    config_path = output / "generated_configs" / "task-a--run.yaml"
    config, job_dir = _write_eval_config(
        template, task="task-a", dataset=dataset, output_root=output,
        config_path=config_path, run_id="run12345", frozen_bank=frozen,
    )
    assert config["agents"][0]["import_path"] == "lh_harness.harbor_training:SkillLearningHarborAgent"
    assert config["agents"][0]["kwargs"]["learning_enabled"] is False
    assert config["agents"][0]["kwargs"]["skill_bank_path"] == str(frozen)
    assert "candidate_path" not in config["agents"][0]["kwargs"]
    assert config["datasets"] == [{"path": str(dataset), "task_names": ["task-a"]}]
    assert config["n_attempts"] == config["n_concurrent_trials"] == 1
    assert yaml.safe_load(config_path.read_text()) == config
    assert job_dir == output / "jobs" / "task-a" / config["job_name"]


def test_eval_records_selected_skills_and_resume_skips_completed_task(tmp_path: Path, monkeypatch) -> None:
    source = _source_bank(tmp_path)
    dataset = _dataset(tmp_path, "task-a", "task-b")
    output = tmp_path / "evaluation"
    template_path = tmp_path / "template.yaml"
    template_path.write_text(yaml.safe_dump({"agents": [{"kwargs": {}}]}))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.test/v1")
    calls = []

    def fake_harbor(config_path, job_dir, use_build_proxy):
        calls.append(config_path)
        assert use_build_proxy is False
        trial = job_dir / "trial"
        routes = trial / "agent" / "skill_harness" / "lh_harness" / "role_orchestration"
        routes.mkdir(parents=True)
        (routes / "skill_routes.json").write_text(json.dumps([{
            "selected_skills": [{"skill_id": "regular", "version": 1},
                                {"skill_id": "temporary", "version": 1}],
        }]))
        trial_path = trial / "result.json"
        return 0, {"task_name": "task-a", "verifier_result": {"rewards": {"reward": 1.0}},
                   "agent_result": {"n_input_tokens": 100, "n_output_tokens": 25}}, trial_path, None

    monkeypatch.setattr("lh_harness.eval_terminal_bench_skillbanks._run_harbor", fake_harbor)
    args = ["--dataset", str(dataset), "--bank", str(source), "--output", str(output),
            "--template", str(template_path), "--task", "task-a"]
    before = source.read_bytes(), source.with_name("skillbank_temporary.json").read_bytes()
    assert main(args) == 0
    assert main(args) == 0
    assert len(calls) == 1
    assert before == (source.read_bytes(), source.with_name("skillbank_temporary.json").read_bytes())
    record = json.loads((output / "results" / "tasks" / "task-a.json").read_text())
    assert record["correct"] is True and record["total_tokens"] == 125
    assert record["referenced_skill_names"] == ["regular", "temporary"]
    assert record["learning_enabled"] is False
    with (output / "results" / "per_task.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["referenced_skill_names_text"] == "regular; temporary"
    assert json.loads((output / "results" / "summary.json").read_text())["method"] == "dual-skillbank-eval-no-learning"


def test_skill_episode_stream_recovers_tokens_when_harbor_totals_are_missing(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    stream = trial / "agent" / "skill_harness" / "episodes" / "cli_task" / "ep001" / "claude_stream.jsonl"
    stream.parent.mkdir(parents=True)
    stream.write_text(json.dumps({"type": "result", "usage": {
        "input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 3,
    }}) + "\n")
    record = result_record(
        "task-a", "run-1", {"task_name": "task-a", "verifier_result": {"rewards": {"reward": 0.0}}},
        trial / "result.json", config_path=tmp_path / "config.yaml", harbor_exit_code=0,
    )
    assert (record["input_tokens"], record["output_tokens"], record["cache_tokens"]) == (12, 5, 3)
    assert record["total_tokens"] == 17
    assert record["token_usage_source"] == "episode_streams"
