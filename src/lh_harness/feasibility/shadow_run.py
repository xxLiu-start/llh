"""Launch the original CLI in shadow mode and retain per-round contract judgments."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from .bank import atomic_json
from .schema import ReflectorConfig


def collect(output: Path, exit_code: int, interrupted_signal: int | None) -> dict:
    role = output/"runtime"/"main"/"lh_harness"/"role_orchestration"
    rows = []
    for directory in sorted((role/"rounds").glob("round_*")):
        if not directory.is_dir():
            continue
        def read(name):
            path = directory/name
            return path.read_text(encoding="utf-8") if path.is_file() else ""
        record_path = directory/"round.json"
        # The primary local ledger holds round metadata; remote mirrors are optional.
        record = {}
        ledger = role/"rounds.jsonl"
        if ledger.is_file():
            for line in ledger.read_text().splitlines():
                try:
                    value = json.loads(line)
                    if f"round_{value.get('round_index', 0):03d}" == directory.name:
                        record = value
                except (ValueError, TypeError):
                    continue
        if not record and record_path.is_file():
            record = json.loads(record_path.read_text())
        sidecar = directory/"feasibility.json"
        assessment = json.loads(sidecar.read_text()) if sidecar.is_file() else None
        rows.append({"round": directory.name, "next_step": record.get("next_step"),
            "contract": read("task_contract.txt"), "manager_plan_reference": str(directory/"manager_plan.txt"),
            "assessed": assessment is not None,
            "assessment_status": "recorded" if assessment else "no_dispatch_assessment",
            "decision": assessment["result"]["decision"] if assessment else None,
            "dispatch": assessment["dispatch"] if assessment else None,
            "issues": assessment["result"].get("issues", []) if assessment else [],
            "retrieved_skills": assessment["result"].get("retrieved_skills", []) if assessment else [],
            "uncertainty": assessment["result"].get("uncertainty", "") if assessment else "",
            "trace": assessment.get("trace", {}) if assessment else {},
            "assessment_reference": str(sidecar) if assessment else None,
            "auditor_report_reference": str(directory/"auditor_report.txt") if (directory/"auditor_report.txt").is_file() else None})
    with (output/"manager_contracts.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    fields = ["round", "next_step", "assessed", "decision", "dispatch", "retrieved_skills", "uncertainty", "assessment_reference"]
    with (output/"manager_contracts.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], list) else row[k] for k in fields})
    report_path = output/"runtime"/"main"/"lh_harness"/"report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    assessed = [r for r in rows if r["assessed"]]
    counts = {d: sum(r["decision"] == d for r in assessed) for d in ("ACCEPT", "REVISE", "BLOCK", "UNKNOWN")}
    def total(key):
        values = [r["trace"].get(key) for r in assessed]
        return sum(values) if all(type(v) is int for v in values) else None
    summary = {"mode": "shadow", "harness_exit_code": exit_code,
        "interrupted_signal": interrupted_signal, "task_status": report.get("status"),
        "completion_satisfied": report.get("completion_satisfied"),
        "manager_rounds": len(rows), "assessed_rounds": len(assessed), "decisions": counts,
        "reflector_overhead": {k: total(k) for k in ("llm_calls", "input_tokens", "output_tokens", "runtime_ms")},
        "reflector_error_rounds": sum(bool(r["trace"].get("runtime_error")) for r in assessed),
        "skill_guided_rounds": sum(bool(r["retrieved_skills"]) for r in assessed),
        "task_report": str(report_path),
        "note": "Shadow verdicts never control dispatch. Non-execution/unfinished rounds have no assessment; missing usage remains unknown."}
    atomic_json(output/"shadow_summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Other arguments are passed to lh-harness run unchanged, e.g. --task TEXT --agent claude_code --model MODEL --workspace DIR.")
    parser.add_argument("--output", type=Path, help="Fresh output directory (must not already exist)")
    parser.add_argument("--skill-bank", type=Path, default=Path("~/.lh-harness/feasibility-skills"), help="Feasibility bank, read only during the run")
    parser.add_argument("--task-file", type=Path, help="Use an existing task text file as original --task @path")
    parser.add_argument("--reflector-max-input-chars", type=int, default=48000)
    parser.add_argument("--dry-run", action="store_true", help="Write/validate shadow config and show paths without launching a task")
    args, original_args = parser.parse_known_args(argv)
    if not 2000 <= args.reflector_max_input_chars <= 100000:
        parser.error("--reflector-max-input-chars must be between 2000 and 100000")
    if original_args[:1] == ["--"]:
        original_args = original_args[1:]
    if any(a.split("=", 1)[0] == "--skip-without-skills" for a in original_args):
        parser.error("--skip-without-skills was removed: an enabled Reflector always assesses current evidence")
    reserved = {"--feasibility-reflector-config", "--runs-root", "--run-id", "--log-dir", "--harness-dir", "--resume", "--supervised"}
    if any(a.split("=", 1)[0] in reserved for a in original_args):
        parser.error("shadow wrapper owns the Reflector configuration and run paths; use --output")
    if args.task_file:
        if any(a.split("=", 1)[0] == "--task" for a in original_args):
            parser.error("use --task or --task-file, not both")
        task_file = args.task_file.expanduser().resolve()
        if not task_file.is_file():
            parser.error("--task-file must exist")
        original_args += ["--task", "@"+str(task_file)]
    if not args.dry_run and not any(a.split("=", 1)[0] == "--task" for a in original_args):
        parser.error("provide original task text with --task or --task-file")
    repo = Path(__file__).resolve().parents[3]
    output = (args.output or repo/"output"/f"feasibility-shadow-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}").expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output already exists; choose a fresh directory to preserve prior records")
    config = {"enabled": True, "mode": "shadow", "skill_bank_path": str(args.skill_bank.expanduser().resolve()),
        "top_k": 5, "max_revision_rounds": 0, "unknown_policy": "pass_through",
        "runtime_error_policy": "fail_open", "revision_exhausted_policy": "pass_through",
        "enable_skill_learning": False, "max_input_chars": args.reflector_max_input_chars}
    ReflectorConfig.from_dict(config)
    config_path = output/"reflector_config.json"
    atomic_json(config_path, config)
    print(f"Output directory: {output}", flush=True)
    print("Reflector mode: shadow; evidence-based assessment with optional Skill guidance", flush=True)
    if args.dry_run:
        print(f"Validated configuration: {config_path}", flush=True)
        return 0
    console = output/"console_logs"/"run.log"
    console.parent.mkdir()
    command = [sys.executable, "-u", "-m", "lh_harness", "run", "--no-dashboard", *original_args,
        "--feasibility-reflector-config", str(config_path), "--runs-root", str(output/"runtime"), "--run-id", "main"]
    # No shell interpolation and no command/credential dump in run metadata.
    interrupted = None
    with console.open("w", encoding="utf-8") as log:
        def emit(line):
            sys.stdout.write(line); sys.stdout.flush()
            log.write(line); log.flush()
        emit(f"Terminal output log: {console}\n")
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", start_new_session=True, bufsize=1)
        def forward(signum, _frame):
            nonlocal interrupted
            interrupted = signum
            emit(f"[shadow-run] forwarding {signal.Signals(signum).name}; preserving partial records\n")
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        previous = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            assert child.stdout is not None
            for line in child.stdout:
                emit(line)
            exit_code = child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        try:
            summary = collect(output, exit_code, interrupted)
            emit(f"[shadow-run] manager_rounds={summary['manager_rounds']} assessed={summary['assessed_rounds']} decisions={json.dumps(summary['decisions'])}\n")
            emit(f"[shadow-run] contracts: {output/'manager_contracts.jsonl'}\n")
        except Exception as exc:
            # Observation summarization must not replace the original task's exit status.
            emit(f"[shadow-run] summary_error={type(exc).__name__}; original logs retained\n")
        return 128+interrupted if interrupted else exit_code if exit_code >= 0 else 128-exit_code


if __name__ == "__main__":
    raise SystemExit(main())
