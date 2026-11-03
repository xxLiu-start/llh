"""Evaluate Terminal-Bench Pro with the original LongHorizon-Harness."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .docker_build_proxy import docker_build_proxy
from .harbor_url import claude_code_upstream_base_url


def discover_tasks(root: Path) -> list[str]:
    if not root.is_dir():
        raise ValueError(f"dataset does not exist: {root}")
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "task.toml").is_file() and (path / "instruction.md").is_file()
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_task_record(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_run = previous.get("run_id") if isinstance(previous, dict) else None
        if isinstance(previous_run, str) and previous_run:
            history = path.parents[1] / "history" / path.stem / f"{previous_run}.json"
            if not history.exists():
                _atomic_json(history, previous)
    _atomic_json(path, value)


def _write_config(
    template: dict[str, Any], *, task: str, dataset: Path, output_root: Path,
    config_path: Path, run_id: str,
) -> tuple[dict[str, Any], Path]:
    config = copy.deepcopy(template)
    job_name = f"original-harness-{task}-{run_id[:8]}"
    jobs_dir = output_root / "jobs" / task
    config["job_name"] = job_name
    config["jobs_dir"] = str(jobs_dir)
    config["n_attempts"] = 1
    config["n_concurrent_trials"] = 1
    config["datasets"] = [{"path": str(dataset), "task_names": [task]}]
    config["agents"][0]["import_path"] = "lh_harness.original_harness_eval:OriginalHarnessHarborAgent"

    offline_binary = os.getenv("LH_CLAUDE_OFFLINE_BINARY")
    if offline_binary:
        binary = Path(offline_binary).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"offline Claude Code binary is missing or not executable: {binary}")
        compose = Path(__file__).resolve().parents[2] / "training" / "docker_claude_offline.yaml"
        config.setdefault("environment", {}).setdefault("extra_docker_compose", []).append(str(compose))
        config["agents"][0].setdefault("kwargs", {})["offline_claude_binary"] = True

    if task == "jq-github-contributor-report":
        compose = Path(__file__).resolve().parents[2] / "training" / "docker_build_proxy.yaml"
        config.setdefault("environment", {}).setdefault("extra_docker_compose", []).append(str(compose))

    config_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config, jobs_dir / job_name


def _trial_result(job_dir: Path, task: str) -> tuple[dict[str, Any] | None, Path | None]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    if job_dir.is_dir():
        for path in job_dir.rglob("result.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(value, dict) and value.get("task_name") == task:
                candidates.append((path, value))
    if not candidates:
        return None, None
    path, value = max(candidates, key=lambda item: item[0].stat().st_mtime_ns)
    return value, path


def _job_result(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / "result.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _episode_stream_usage(result_path: Path | None) -> dict[str, int] | None:
    """Recover completed role-episode usage when Harbor loses agent totals."""
    if result_path is None:
        return None
    agent_dir = result_path.parent / "agent"
    episode_dirs = (
        agent_dir / "cua_harness_claudecode_harbor" / "episodes",
        agent_dir / "skill_harness" / "episodes",
    )
    totals = {"n_input_tokens": 0, "n_output_tokens": 0, "n_cache_tokens": 0}
    found = False
    for path in (path for episodes in episode_dirs for path in episodes.glob("*/*/claude_stream.jsonl")):
        terminal = None
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("type") == "result":
                        terminal = event
        except OSError:
            continue
        usage = terminal.get("usage") if isinstance(terminal, dict) else None
        if not isinstance(usage, dict):
            continue
        inp, out = usage.get("input_tokens"), usage.get("output_tokens")
        if type(inp) is not int or type(out) is not int:
            continue
        totals["n_input_tokens"] += inp
        totals["n_output_tokens"] += out
        cache = usage.get("cache_read_input_tokens")
        if type(cache) is int:
            totals["n_cache_tokens"] += cache
        found = True
    return totals if found else None


def result_record(
    task: str, run_id: str, result: dict[str, Any] | None, result_path: Path | None,
    *, config_path: Path, harbor_exit_code: int, job_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if result is None:
        stats = job_result.get("stats") if isinstance(job_result, dict) and isinstance(job_result.get("stats"), dict) else {}
        input_tokens = stats.get("n_input_tokens") if type(stats.get("n_input_tokens")) is int else None
        output_tokens = stats.get("n_output_tokens") if type(stats.get("n_output_tokens")) is int else None
        cache_tokens = stats.get("n_cache_tokens") if type(stats.get("n_cache_tokens")) is int else None
        total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
        return {
            "task": task, "run_id": run_id, "correct": False, "reward": None,
            "status": "infrastructure_error", "harbor_exit_code": harbor_exit_code,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "cache_tokens": cache_tokens,
            "total_tokens": total_tokens, "token_usage_missing": total_tokens is None,
            "harbor_retries": stats.get("n_retries") if type(stats.get("n_retries")) is int else None,
            "config_path": str(config_path), "result_path": None,
            "error": "Harbor produced no task-level result.json",
        }

    verifier = result.get("verifier_result") if isinstance(result.get("verifier_result"), dict) else {}
    rewards = verifier.get("rewards") if isinstance(verifier.get("rewards"), dict) else {}
    reward = rewards.get("reward")
    reward_number = float(reward) if type(reward) in (int, float) else None
    agent = result.get("agent_result") if isinstance(result.get("agent_result"), dict) else {}
    final_input = agent.get("n_input_tokens") if type(agent.get("n_input_tokens")) is int else None
    final_output = agent.get("n_output_tokens") if type(agent.get("n_output_tokens")) is int else None
    final_cache = agent.get("n_cache_tokens") if type(agent.get("n_cache_tokens")) is int else None
    stats = job_result.get("stats") if isinstance(job_result, dict) and isinstance(job_result.get("stats"), dict) else {}
    recovered = None
    if (final_input is None or final_output is None) and (
        type(stats.get("n_input_tokens")) is not int or type(stats.get("n_output_tokens")) is not int
    ):
        recovered = _episode_stream_usage(result_path)
    input_tokens = stats.get("n_input_tokens") if type(stats.get("n_input_tokens")) is int else final_input
    output_tokens = stats.get("n_output_tokens") if type(stats.get("n_output_tokens")) is int else final_output
    cache_tokens = stats.get("n_cache_tokens") if type(stats.get("n_cache_tokens")) is int else final_cache
    if recovered:
        input_tokens = input_tokens if input_tokens is not None else recovered["n_input_tokens"]
        output_tokens = output_tokens if output_tokens is not None else recovered["n_output_tokens"]
        cache_tokens = cache_tokens if cache_tokens is not None else recovered["n_cache_tokens"]
    total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    metadata = agent.get("metadata") if isinstance(agent.get("metadata"), dict) else {}
    exception = result.get("exception_info")
    status = "errored" if exception else "completed" if reward_number is not None else "missing_reward"
    return {
        "task": task,
        "run_id": run_id,
        "correct": bool(reward_number is not None and reward_number >= 1.0 and not exception),
        "reward": reward_number,
        "all_rewards": rewards,
        "status": status,
        "harbor_exit_code": harbor_exit_code,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_tokens": cache_tokens,
        "total_tokens": total_tokens,
        "final_attempt_input_tokens": final_input,
        "final_attempt_output_tokens": final_output,
        "final_attempt_cache_tokens": final_cache,
        "final_attempt_total_tokens": (
            final_input + final_output if final_input is not None and final_output is not None else None
        ),
        "harbor_retries": stats.get("n_retries") if type(stats.get("n_retries")) is int else None,
        "token_usage_missing": total_tokens is None,
        "token_usage_source": "episode_streams" if recovered else "harbor",
        "token_usage_partial": bool(recovered),
        "cost_usd": agent.get("cost_usd"),
        "harness_status": metadata.get("status"),
        "harness_completion_satisfied": metadata.get("completion_satisfied"),
        "rounds_run": metadata.get("rounds_run"),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "config_path": str(config_path),
        "result_path": str(result_path) if result_path else None,
        "report_path": metadata.get("report_path"),
        "exception_info": exception,
    }


def write_summary(
    output_root: Path, task_names: list[str], *,
    dataset_label: str = "terminal-bench-pro",
    method: str = "original-long-horizon-harness-no-skills",
    extra_fields: list[str] | None = None,
) -> dict[str, Any]:
    records = []
    for task in task_names:
        path = output_root / "results" / "tasks" / f"{task}.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                records.append(value)
    completed = [item for item in records if item.get("status") == "completed"]
    known_tokens = [item["total_tokens"] for item in records if type(item.get("total_tokens")) is int]
    summary = {
        "dataset": dataset_label,
        "method": method,
        "expected_tasks": len(task_names),
        "recorded_tasks": len(records),
        "completed_tasks": len(completed),
        "correct_tasks": sum(item.get("correct") is True for item in records),
        "incorrect_tasks": sum(item.get("status") == "completed" and item.get("correct") is False for item in records),
        "accuracy_over_recorded": (
            sum(item.get("correct") is True for item in records) / len(records) if records else None
        ),
        "accuracy_over_completed": (
            sum(item.get("correct") is True for item in completed) / len(completed) if completed else None
        ),
        "total_tokens": sum(known_tokens),
        "tasks_with_token_usage": len(known_tokens),
        "tasks_missing_token_usage": len(records) - len(known_tokens),
        "tasks_with_partial_token_usage": sum(item.get("token_usage_partial") is True for item in records),
    }
    results_dir = output_root / "results"
    _atomic_json(results_dir / "summary.json", summary)
    jsonl = results_dir / "per_task.jsonl"
    jsonl.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    csv_path = results_dir / "per_task.csv"
    fields = [
        "task", "correct", "reward", "status", "input_tokens", "output_tokens",
        "cache_tokens", "total_tokens", "rounds_run", "harness_status",
        "harness_completion_satisfied", "token_usage_partial", "token_usage_source",
        "result_path", "report_path",
    ]
    fields.extend(extra_fields or [])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return summary


def _run_harbor(
    config_path: Path, job_dir: Path, use_build_proxy: bool,
) -> tuple[int, dict[str, Any] | None, Path | None, dict[str, Any] | None]:
    harbor_env = os.environ.copy()
    if harbor_env.get("ANTHROPIC_BASE_URL"):
        harbor_env["ANTHROPIC_BASE_URL"] = claude_code_upstream_base_url(harbor_env["ANTHROPIC_BASE_URL"])
    with docker_build_proxy() if use_build_proxy else contextlib.nullcontext(None) as build_proxy:
        if build_proxy:
            harbor_env["LH_DOCKER_BUILD_PROXY"] = build_proxy
        completed = subprocess.run(["harbor", "run", "-c", str(config_path)], check=False, env=harbor_env)
    task = config_path.name.split("--", 1)[0]
    result, path = _trial_result(job_dir, task)
    return completed.returncode, result, path, _job_result(job_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Terminal-Bench Pro with the original LongHorizon-Harness")
    parser.add_argument("--dataset", type=Path, default=Path("train_dataset/terminal-bench-pro"))
    parser.add_argument("--template", type=Path, default=Path("evaluation/original_harness_terminal_bench_pro.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/origin"))
    parser.add_argument("--task", action="append", help="Evaluate only this task; repeatable")
    parser.add_argument("--start-from", metavar="TASK")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Run tasks again instead of reusing their records")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    template_path = args.template.resolve()
    output_root = args.output.resolve()
    all_tasks = discover_tasks(dataset)
    selected = list(all_tasks)
    if args.start_from:
        if args.start_from not in selected:
            parser.error(f"unknown --start-from task: {args.start_from}")
        selected = selected[selected.index(args.start_from):]
    if args.task:
        unknown = set(args.task) - set(all_tasks)
        if unknown:
            parser.error(f"unknown tasks: {', '.join(sorted(unknown))}")
        selected = [task for task in selected if task in set(args.task)]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        selected = selected[:args.limit]
    if args.dry_run:
        print(json.dumps({"dataset": str(dataset), "output": str(output_root), "tasks": selected}, ensure_ascii=False, indent=2))
        return 0
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("ANTHROPIC_BASE_URL"):
        parser.error("ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL are required")

    if not os.getenv("LH_CLAUDE_OFFLINE_BINARY"):
        version = os.getenv("CLAUDE_CODE_VERSION", "2.1.176")
        cached = Path("output/terminal-bench-pro/jobs/claude_code_cache") / version / "claude"
        if cached.is_file() and os.access(cached, os.X_OK):
            os.environ["LH_CLAUDE_OFFLINE_BINARY"] = str(cached.resolve())
            print(f"Offline Claude Code: {cached}", file=sys.stderr, flush=True)

    failures = 0
    for task in selected:
        record_path = output_root / "results" / "tasks" / f"{task}.json"
        if record_path.is_file() and not args.force:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("status") == "completed":
                print(json.dumps({**record, "execution": "reused_prior_record"}, ensure_ascii=False), flush=True)
                continue
            print(f"Retrying non-completed prior record: {task}", file=sys.stderr, flush=True)
        run_id = uuid.uuid4().hex
        config_path = output_root / "generated_configs" / f"{task}--{run_id}.yaml"
        # Long evaluations can span template edits; read it for each new task.
        import yaml

        template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        if not isinstance(template, dict):
            parser.error("invalid YAML template")
        config, job_dir = _write_config(
            template, task=task, dataset=dataset, output_root=output_root,
            config_path=config_path, run_id=run_id,
        )
        timeout = config["agents"][0].get("override_timeout_sec")
        print(
            f"Fresh original-Harness evaluation: {task} ({run_id}); agent timeout: {timeout}s",
            file=sys.stderr, flush=True,
        )
        try:
            exit_code, trial, trial_path, job_summary = _run_harbor(
                config_path, job_dir,
                any(Path(path).name == "docker_build_proxy.yaml" for path in config.get("environment", {}).get("extra_docker_compose", [])),
            )
            record = result_record(
                task, run_id, trial, trial_path, config_path=config_path,
                harbor_exit_code=exit_code, job_result=job_summary,
            )
        except Exception as exc:
            record = result_record(task, run_id, None, None, config_path=config_path, harbor_exit_code=-1)
            record["error"] = str(exc)
            record["error_type"] = type(exc).__name__
        _write_task_record(record_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if record["status"] != "completed":
            failures += 1
        write_summary(output_root, all_tasks)
    summary = write_summary(output_root, all_tasks)
    print(json.dumps({"summary": summary, "summary_path": str(output_root / "results" / "summary.json")}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
