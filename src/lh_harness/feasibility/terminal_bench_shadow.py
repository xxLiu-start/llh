"""Run real Terminal-Bench tasks with the original evaluator and a shadow observer."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import uuid

from .. import eval_terminal_bench_pro_original as original
from ..claude_cache import find_cached_claude_binary
from ..verifier_uv import scoped_verifier_uv_host_env
from .bank import atomic_json
from .schema import ReflectorConfig


def write_config(template, *, task, dataset, output, run_id, bank, model=None, max_input_chars=48000):
    path = output/"generated_configs"/f"{task}--{run_id}.yaml"
    config, job = original._write_config(template, task=task, dataset=dataset,
        output_root=output, config_path=path, run_id=run_id)
    agent = config["agents"][0]
    agent["import_path"] = "lh_harness.feasibility.harbor_shadow:ShadowReflectorHarborAgent"
    settings = {"enabled": True, "mode": "shadow", "skill_bank_path": str(bank),
        "max_revision_rounds": 0, "unknown_policy": "pass_through", "runtime_error_policy": "fail_open",
        "enable_skill_learning": False, "max_input_chars": max_input_chars}
    ReflectorConfig.from_dict(settings)
    agent.setdefault("kwargs", {})["feasibility_reflector"] = settings
    if model:
        agent["model_name"] = model
        agent["kwargs"].update(harness_model=model, verifier_model=model)
    import yaml
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return config, path, job


def collect_shadow(trial_path, output, task, run_id):
    if trial_path is None:
        return {"reflector_status": "no_trial_result"}
    found = list((trial_path.parent/"agent").rglob("reflector_shadow.json"))
    if not found:
        return {"reflector_status": "no_shadow_report"}
    report_path = max(found, key=lambda p: p.stat().st_mtime_ns)
    summary = json.loads(report_path.read_text())
    target = output/"reflector_logs"/task/run_id
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target/"summary.json", summary)
    rows = []
    for item in summary["rounds"]:
        source = report_path.parent/"rounds"/f"round_{item['round']:03d}"/"feasibility.json"
        entry = json.loads(source.read_text()) if source.is_file() else {"result": item["result"], "trace": item["trace"]}
        atomic_json(target/f"round_{item['round']:03d}.json", entry)
        result = item.get("result") or {}
        rows.append({"round": item["round"], "next_step": item["next_step"],
            "decision": result.get("decision"), "skill_ids": json.dumps(result.get("retrieved_skills", [])),
            "uncertainty": result.get("uncertainty", ""), "enforced": False})
    with (target/"per_round.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "next_step", "decision", "skill_ids", "uncertainty", "enforced"])
        writer.writeheader(); writer.writerows(rows)
    overhead = summary.get("overhead") or {}
    inp, out = overhead.get("input_tokens"), overhead.get("output_tokens")
    errors = len(summary.get("observation_errors", []))+sum(bool(r["trace"].get("runtime_error")) for r in summary["rounds"])
    return {"reflector_status": "recorded_with_errors" if errors else "recorded", "reflector_report": str(target/"summary.json"),
        "reflector_error_rounds": errors,
        "reflector_rounds": summary["assessed_rounds"], "reflector_llm_calls": overhead.get("llm_calls"),
        "reflector_tokens": inp+out if type(inp) is int and type(out) is int else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("train_dataset/terminal-bench-pro"))
    parser.add_argument("--template", type=Path, default=Path("evaluation/original_harness_terminal_bench_pro.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/terminal-bench-pro-reflector-shadow"))
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--model")
    parser.add_argument("--skill-bank", type=Path, default=Path("~/.lh-harness/feasibility-skills"))
    parser.add_argument("--reflector-max-input-chars", type=int, default=48000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 2000 <= args.reflector_max_input_chars <= 100000:
        parser.error("--reflector-max-input-chars must be between 2000 and 100000")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    tasks = list(dict.fromkeys(args.task))
    unknown = set(tasks)-set(original.discover_tasks(dataset))
    if unknown:
        parser.error("unknown tasks: "+", ".join(sorted(unknown)))
    if not args.dry_run:
        if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("ANTHROPIC_BASE_URL"):
            parser.error("ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL are required")
        if not os.getenv("LH_CLAUDE_OFFLINE_BINARY"):
            cached = find_cached_claude_binary(output, os.getenv("CLAUDE_CODE_VERSION", "2.1.176"))
            if cached is None:
                parser.error("Set LH_CLAUDE_OFFLINE_BINARY to the matching cached Claude Code executable")
            os.environ["LH_CLAUDE_OFFLINE_BINARY"] = str(cached)
    import yaml
    template = yaml.safe_load(args.template.read_text())
    if not isinstance(template, dict):
        parser.error("invalid original evaluation template")
    failures = 0
    plans = []
    extras = ["reflector_status", "reflector_rounds", "reflector_error_rounds", "reflector_llm_calls", "reflector_tokens", "reflector_token_usage_partial", "total_tokens_with_reflector", "reflector_report"]
    for task in tasks:
        run_id = uuid.uuid4().hex
        config, path, job = write_config(template, task=task, dataset=dataset, output=output,
            run_id=run_id, bank=args.skill_bank.expanduser().resolve(), model=args.model,
            max_input_chars=args.reflector_max_input_chars)
        if args.dry_run:
            plans.append({"task": task, "config_path": str(path), "mode": "shadow"})
            continue
        print(json.dumps({"event": "task_start", "task": task, "mode": "shadow", "run_id": run_id}), flush=True)
        try:
            with scoped_verifier_uv_host_env(config):
                code, trial, trial_path, job_result = original._run_harbor(path, job,
                    any(Path(p).name == "docker_build_proxy.yaml" for p in config.get("environment", {}).get("extra_docker_compose", [])))
            record = original.result_record(task, run_id, trial, trial_path,
                config_path=path, harbor_exit_code=code, job_result=job_result)
        except Exception as exc:
            record = original.result_record(task, run_id, None, None, config_path=path, harbor_exit_code=-1)
            record["error_type"] = type(exc).__name__
            trial_path = None
        try:
            record.update(collect_shadow(trial_path, output, task, run_id))
        except Exception as exc:
            record.update(reflector_status="collection_error", reflector_error_type=type(exc).__name__)
        base, added = record.get("total_tokens"), record.get("reflector_tokens")
        record["reflector_token_usage_partial"] = bool(record.get("harbor_retries"))
        record["total_tokens_with_reflector"] = base+added if type(base) is int and type(added) is int and not record["reflector_token_usage_partial"] else None
        original._write_task_record(output/"results"/"tasks"/f"{task}.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        failures += record["status"] != "completed"
        original.write_summary(output, tasks, method="original-lhh-with-shadow-feasibility-observer", extra_fields=extras)
    if args.dry_run:
        print(json.dumps({"mode": "shadow", "dataset": str(dataset), "output": str(output), "tasks": plans}, ensure_ascii=False, indent=2))
    else:
        summary = original.write_summary(output, tasks, method="original-lhh-with-shadow-feasibility-observer", extra_fields=extras)
        print(json.dumps({"summary": summary, "summary_path": str(output/"results/summary.json")}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
