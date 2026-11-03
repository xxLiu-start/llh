"""Evaluate tasks with a frozen regular plus temporary SkillBank, without learning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from .eval_terminal_bench_pro_original import (
    _atomic_json, _run_harbor, _write_config as _write_original_config,
    _write_task_record, discover_tasks, result_record, write_summary,
)
from .skill_learning import SkillBank
from .train_terminal_bench_pro import _find_cached_claude_binary


def _freeze_bank(source: Path, output_root: Path, dataset: Path) -> Path:
    """Snapshot both files once; resumed evaluations keep the same Skill versions."""
    source = source.resolve()
    temporary = source.with_name(source.stem + "_temporary" + source.suffix)
    target = output_root / "state" / "skillbank.json"
    target_temporary = target.with_name("skillbank_temporary.json")
    manifest = output_root / "state" / "bank_snapshot.json"
    if target.resolve() == source:
        raise ValueError("evaluation output must differ from the source SkillBank directory")
    if manifest.exists():
        details = json.loads(manifest.read_text(encoding="utf-8"))
        if details.get("dataset") != str(dataset) or details.get("source_bank") != str(source):
            raise ValueError("evaluation output belongs to a different dataset or SkillBank; choose a new --output")
        for path, key in ((target, "regular_sha256"), (target_temporary, "temporary_sha256")):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != details.get(key):
                raise ValueError(f"evaluation SkillBank snapshot changed or is incomplete: {path}")
        SkillBank.load(target)
        return target
    if target.exists() or target_temporary.exists():
        raise ValueError("evaluation bank snapshot exists without a manifest; choose a new --output")
    if not source.is_file() or not temporary.is_file():
        raise ValueError(f"both SkillBank files are required: {source}, {temporary}")
    # Validate the source before copying. Evaluation must not repair or save it.
    SkillBank.load(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    regular_bytes, temporary_bytes = source.read_bytes(), temporary.read_bytes()
    for path, data in ((target, regular_bytes), (target_temporary, temporary_bytes)):
        scratch = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        scratch.write_bytes(data)
        os.replace(scratch, path)
    SkillBank.load(target)
    _atomic_json(manifest, {
        "dataset": str(dataset), "source_bank": str(source),
        "source_temporary_bank": str(temporary),
        "regular_sha256": hashlib.sha256(regular_bytes).hexdigest(),
        "temporary_sha256": hashlib.sha256(temporary_bytes).hexdigest(),
    })
    return target


def _write_eval_config(
    template: dict[str, Any], *, task: str, dataset: Path, output_root: Path,
    config_path: Path, run_id: str, frozen_bank: Path,
) -> tuple[dict[str, Any], Path]:
    import yaml

    config, _ = _write_original_config(
        template, task=task, dataset=dataset, output_root=output_root,
        config_path=config_path, run_id=run_id,
    )
    config["job_name"] = f"dual-skillbank-eval-{task}-{run_id[:8]}"
    agent = config["agents"][0]
    agent["import_path"] = "lh_harness.harbor_training:SkillLearningHarborAgent"
    kwargs = agent.setdefault("kwargs", {})
    kwargs["skill_bank_path"] = str(frozen_bank)
    kwargs["learning_enabled"] = False
    kwargs.pop("candidate_path", None)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config, output_root / "jobs" / task / config["job_name"]


def _selected_skill_names(result_path: Path | None) -> list[str]:
    if result_path is None:
        return []
    routes_path = result_path.parent / "agent" / "skill_harness" / "lh_harness" / "role_orchestration" / "skill_routes.json"
    try:
        routes = json.loads(routes_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(routes, list):
        return []
    return sorted({
        choice["skill_id"] for route in routes if isinstance(route, dict)
        for choice in route.get("selected_skills", []) if isinstance(choice, dict)
        if isinstance(choice.get("skill_id"), str) and choice["skill_id"]
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate with frozen regular and temporary SkillBanks; no training")
    parser.add_argument("--dataset", type=Path, default=Path("eval/TB-harness/datasets/terminal-bench-2-1/tasks"))
    parser.add_argument("--template", type=Path, default=Path("training/terminal_bench21_skill_test.yaml"))
    parser.add_argument("--bank", type=Path, default=Path("output/terminal-bench-pro/state/skillbank.json"))
    parser.add_argument("--output", type=Path, default=Path("output/dual-skillbank-eval"))
    parser.add_argument("--task", action="append", help="Evaluate only this task; repeatable")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Rerun completed tasks, keeping the frozen SkillBank")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    dataset, output_root = args.dataset.resolve(), args.output.resolve()
    if not dataset.is_dir():
        parser.error(f"evaluation dataset does not exist: {dataset}")
    tasks = discover_tasks(dataset)
    selected = [task for task in tasks if not args.task or task in set(args.task)]
    unknown = set(args.task or []) - set(tasks)
    if unknown:
        parser.error(f"unknown tasks: {', '.join(sorted(unknown))}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        selected = selected[:args.limit]
    if args.dry_run:
        print(json.dumps({"dataset": str(dataset), "output": str(output_root),
                          "source_bank": str(args.bank.resolve()), "tasks": selected,
                          "learning_enabled": False}, ensure_ascii=False, indent=2))
        return 0
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("ANTHROPIC_BASE_URL"):
        parser.error("ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL are required")
    frozen_bank = _freeze_bank(args.bank, output_root, dataset)
    if not os.getenv("LH_CLAUDE_OFFLINE_BINARY"):
        cached = _find_cached_claude_binary(output_root, os.getenv("CLAUDE_CODE_VERSION", "2.1.176"))
        if cached is not None:
            os.environ["LH_CLAUDE_OFFLINE_BINARY"] = str(cached)
            print(f"Offline Claude Code: {cached}", file=sys.stderr, flush=True)
    failures = 0
    for task in selected:
        record_path = output_root / "results" / "tasks" / f"{task}.json"
        if record_path.is_file() and not args.force:
            previous = json.loads(record_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and previous.get("status") == "completed":
                print(json.dumps({**previous, "execution": "reused_prior_record"}, ensure_ascii=False), flush=True)
                continue
        import yaml

        template = yaml.safe_load(args.template.read_text(encoding="utf-8"))
        if not isinstance(template, dict):
            parser.error("invalid YAML template")
        run_id = uuid.uuid4().hex
        config_path = output_root / "generated_configs" / f"{task}--{run_id}.yaml"
        config, job_dir = _write_eval_config(
            template, task=task, dataset=dataset, output_root=output_root,
            config_path=config_path, run_id=run_id, frozen_bank=frozen_bank,
        )
        print(f"Fresh dual-SkillBank evaluation: {task} ({run_id})", file=sys.stderr, flush=True)
        try:
            exit_code, trial, trial_path, job_summary = _run_harbor(
                config_path, job_dir, any(Path(path).name == "docker_build_proxy.yaml"
                                          for path in config.get("environment", {}).get("extra_docker_compose", [])),
            )
            record = result_record(task, run_id, trial, trial_path, config_path=config_path,
                                   harbor_exit_code=exit_code, job_result=job_summary)
        except Exception as exc:
            record = result_record(task, run_id, None, None, config_path=config_path, harbor_exit_code=-1)
            record.update(error=str(exc), error_type=type(exc).__name__)
            trial_path = None
        names = _selected_skill_names(trial_path)
        record.update(referenced_skill_names=names, referenced_skill_names_text="; ".join(names),
                      skillbank_snapshot=str(frozen_bank), learning_enabled=False)
        _write_task_record(record_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        failures += record["status"] != "completed"
        write_summary(output_root, tasks, dataset_label=dataset.parent.name if dataset.name == "tasks" else dataset.name,
                      method="dual-skillbank-eval-no-learning",
                      extra_fields=["referenced_skill_names_text", "skillbank_snapshot"])
    summary = write_summary(output_root, tasks,
                            dataset_label=dataset.parent.name if dataset.name == "tasks" else dataset.name,
                            method="dual-skillbank-eval-no-learning",
                            extra_fields=["referenced_skill_names_text", "skillbank_snapshot"])
    print(json.dumps({"summary": summary, "summary_path": str(output_root / "results" / "summary.json")},
                     ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
