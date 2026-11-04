from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

from lh_harness.eval_terminal_bench_pro_original import (
    _run_harbor,
    _write_task_record,
    _write_config,
    main,
    result_record,
    write_summary,
)


def test_result_record_uses_verifier_reward_and_provider_tokens(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    result = {
        "task_name": "sample-task",
        "verifier_result": {"rewards": {"reward": 1.0}},
        "agent_result": {
            "n_input_tokens": 120,
            "n_output_tokens": 30,
            "n_cache_tokens": 40,
            "metadata": {"status": "complete", "completion_satisfied": True, "rounds_run": 2},
        },
        "exception_info": None,
    }
    record = result_record(
        "sample-task", "run", result, path,
        config_path=tmp_path / "config.yaml", harbor_exit_code=0,
        job_result={"stats": {"n_input_tokens": 200, "n_output_tokens": 50,
                              "n_cache_tokens": 70, "n_retries": 1}},
    )
    assert record["correct"] is True
    assert record["reward"] == 1.0
    assert record["total_tokens"] == 250
    assert record["cache_tokens"] == 70
    assert record["final_attempt_total_tokens"] == 150
    assert record["harbor_retries"] == 1
    assert record["rounds_run"] == 2


def test_harness_completion_does_not_replace_benchmark_reward(tmp_path: Path) -> None:
    result = {
        "task_name": "sample-task",
        "verifier_result": {"rewards": {"reward": 0.0}},
        "agent_result": {
            "n_input_tokens": 10, "n_output_tokens": 5, "n_cache_tokens": 0,
            "metadata": {"status": "complete", "completion_satisfied": True},
        },
        "exception_info": None,
    }
    record = result_record(
        "sample-task", "run", result, tmp_path / "result.json",
        config_path=tmp_path / "config.yaml", harbor_exit_code=0,
    )
    assert record["correct"] is False
    assert record["harness_completion_satisfied"] is True


def test_timeout_recovers_completed_episode_tokens_as_partial(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    episode = trial / "agent/cua_harness_claudecode_harbor/episodes/cli_task/ep001"
    episode.mkdir(parents=True)
    (episode / "claude_stream.jsonl").write_text(
        json.dumps({"type": "result", "usage": {
            "input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 5,
        }}) + "\n",
        encoding="utf-8",
    )
    result = {
        "task_name": "sample-task", "verifier_result": {"rewards": {"reward": 0.0}},
        "agent_result": None, "exception_info": {"exception_type": "AgentTimeoutError"},
    }
    record = result_record(
        "sample-task", "run", result, trial / "result.json",
        config_path=tmp_path / "config.yaml", harbor_exit_code=0,
        job_result={"stats": {"n_input_tokens": None, "n_output_tokens": None}},
    )
    assert record["status"] == "errored"
    assert record["correct"] is False
    assert record["total_tokens"] == 150
    assert record["cache_tokens"] == 5
    assert record["token_usage_source"] == "episode_streams"
    assert record["token_usage_partial"] is True
    assert record["final_attempt_total_tokens"] is None


def test_generated_config_uses_original_harness_without_skills(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LH_CLAUDE_OFFLINE_BINARY", raising=False)
    template = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "evaluation/original_harness_terminal_bench_pro.yaml").read_text()
    )
    config_path = tmp_path / "sample-task--run.yaml"
    config, _ = _write_config(
        template, task="sample-task", dataset=tmp_path / "dataset",
        output_root=tmp_path / "output", config_path=config_path, run_id="1234567890",
    )
    agent = config["agents"][0]
    assert agent["import_path"] == "lh_harness.original_harness_eval:OriginalHarnessHarborAgent"
    assert "skill_bank_path" not in agent["kwargs"]
    assert "learning_enabled" not in agent["kwargs"]
    assert config["n_attempts"] == 1
    assert config["datasets"][0]["task_names"] == ["sample-task"]


def test_summary_records_correctness_and_tokens(tmp_path: Path) -> None:
    task_dir = tmp_path / "results" / "tasks"
    task_dir.mkdir(parents=True)
    (task_dir / "one.json").write_text(json.dumps({
        "task": "one", "status": "completed", "correct": True, "total_tokens": 10,
    }))
    (task_dir / "two.json").write_text(json.dumps({
        "task": "two", "status": "completed", "correct": False, "total_tokens": 20,
    }))
    summary = write_summary(tmp_path, ["one", "two", "three"])
    assert summary["expected_tasks"] == 3
    assert summary["recorded_tasks"] == 2
    assert summary["correct_tasks"] == 1
    assert summary["incorrect_tasks"] == 1
    assert summary["accuracy_over_completed"] == 0.5
    assert summary["total_tokens"] == 30
    assert (tmp_path / "results" / "per_task.csv").is_file()


def test_replaced_task_record_is_archived(tmp_path: Path) -> None:
    path = tmp_path / "results" / "tasks" / "one.json"
    _write_task_record(path, {"task": "one", "run_id": "old", "status": "errored"})
    _write_task_record(path, {"task": "one", "run_id": "new", "status": "completed"})
    assert json.loads(path.read_text())["run_id"] == "new"
    archived = tmp_path / "results" / "history" / "one" / "old.json"
    assert json.loads(archived.read_text())["run_id"] == "old"


def test_template_is_reloaded_for_each_task(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    for task in ("one", "two"):
        task_dir = dataset / task
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text("")
        (task_dir / "instruction.md").write_text("")
    template = tmp_path / "template.yaml"

    def write_template(timeout: int) -> None:
        template.write_text(yaml.safe_dump({
            "agents": [{"override_timeout_sec": timeout, "kwargs": {}}],
            "verifier": {}, "datasets": [],
        }))

    write_template(7200)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    cached = tmp_path / "claude"
    cached.write_text("#!/bin/sh\necho '2.1.176 (Claude Code)'\n")
    cached.chmod(0o755)
    monkeypatch.setenv("LH_CLAUDE_OFFLINE_BINARY", str(cached))
    configs = []

    def fake_run(config_path, job_dir, use_build_proxy):
        del job_dir, use_build_proxy
        configs.append(yaml.safe_load(config_path.read_text()))
        if len(configs) == 1:
            write_template(18000)
        return 0, {"task_name": configs[-1]["datasets"][0]["task_names"][0],
                   "verifier_result": {"rewards": {"reward": 0.0}}}, config_path, None

    monkeypatch.setattr("lh_harness.eval_terminal_bench_pro_original._run_harbor", fake_run)
    assert main(["--dataset", str(dataset), "--template", str(template),
                 "--output", str(tmp_path / "output")]) == 0
    assert [config["agents"][0]["override_timeout_sec"] for config in configs] == [7200, 18000]


def test_original_eval_reuses_archived_claude_cache(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    task = dataset / "sample-task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text("")
    (task / "instruction.md").write_text("")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({
        "environment": {"type": "docker"}, "agents": [{"kwargs": {}}],
        "verifier": {}, "datasets": [],
    }))
    output = tmp_path / "output"
    cached = output / "his" / "previous-run" / "jobs" / "claude_code_cache" / "2.1.176" / "claude"
    cached.parent.mkdir(parents=True)
    cached.write_text("#!/bin/sh\necho '2.1.176 (Claude Code)'\n")
    cached.chmod(0o755)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.delenv("LH_CLAUDE_OFFLINE_BINARY", raising=False)
    observed = []

    def fake_run(config_path, job_dir, use_build_proxy):
        del job_dir, use_build_proxy
        observed.append(yaml.safe_load(config_path.read_text()))
        return 0, {"task_name": "sample-task", "verifier_result": {"rewards": {"reward": 0.0}}}, config_path, None

    monkeypatch.setattr("lh_harness.eval_terminal_bench_pro_original._run_harbor", fake_run)
    assert main(["--dataset", str(dataset), "--template", str(template),
                 "--output", str(output / "original")]) == 0
    assert observed[0]["agents"][0]["kwargs"]["offline_claude_binary"] is True
    assert any(Path(path).name == "docker_claude_offline.yaml"
               for path in observed[0]["environment"]["extra_docker_compose"])
    assert os.environ["LH_CLAUDE_OFFLINE_BINARY"] == str(cached.resolve())


def test_harbor_child_can_import_original_harness_agent(tmp_path: Path, monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))
    observed = {}

    def fake_run(args, *, check, env):
        observed["args"] = args
        observed["pythonpath"] = env["PYTHONPATH"].split(os.pathsep)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("lh_harness.eval_terminal_bench_pro_original.subprocess.run", fake_run)
    assert _run_harbor(tmp_path / "task--run.yaml", tmp_path / "job", False)[0] == 0
    assert observed["args"] == ["harbor", "run", "-c", str(tmp_path / "task--run.yaml")]
    assert str(repo / "src") in observed["pythonpath"]
    assert str(repo / "eval/TB-harness/Harness/src") in observed["pythonpath"]


def test_original_wrapper_logs_to_selected_output(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_conda = bin_dir / "conda"
    fake_conda.write_text("#!/bin/sh\nprintf 'fake conda ran\\n'\n")
    fake_conda.chmod(0o755)
    output = tmp_path / "custom-output"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               ANTHROPIC_API_KEY="test", ANTHROPIC_BASE_URL="https://example.invalid")
    completed = subprocess.run(
        ["bash", str(repo / "evaluation/run_original_harness_terminal_bench_pro.sh"),
         "--output", str(output), "--dry-run"],
        env=env, check=True, capture_output=True, text=True,
    )
    logs = list((output / "console_logs").glob("eval-*.log"))
    assert len(logs) == 1
    assert "fake conda ran" in completed.stdout
    assert "fake conda ran" in logs[0].read_text()
