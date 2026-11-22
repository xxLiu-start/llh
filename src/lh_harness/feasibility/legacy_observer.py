"""Read-only observer around the repository's legacy Manager AgentAdapter.

The original orchestrator still receives the identical EpisodeResult object and
controls every dispatch, audit, state update and completion decision itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..types import ManagedRound
from .bank import atomic_json
from .integration import snapshot_for_contract


class LegacyContractObserver:
    def __init__(self, original, *, task: str, role_dir: Path, reflector):
        self.original, self.task, self.role_dir, self.reflector = original, task, role_dir, reflector
        self.round_index = 0
        self.records = []
        self.observation_errors = []

    def __getattr__(self, name):
        # Preserve original usage accounting/backend properties. Observer cost
        # is reported separately and never substituted for original role cost.
        return getattr(self.original, name)

    def previous_rounds(self):
        ledger = self.role_dir/"rounds.jsonl"
        result = []
        if ledger.is_file():
            for line in ledger.read_text().splitlines():
                row = json.loads(line)
                if row["round_index"] >= self.round_index:
                    continue
                result.append(ManagedRound(round_index=row["round_index"], next_step=row["next_step"],
                    plan_text=row.get("plan_text", ""), task_state=row.get("task_state", ""),
                    task_contract=row.get("task_contract", ""), auditor_report=row.get("verifier_report", ""),
                    auditor_status=row.get("verifier_status", {})))
        return result

    async def run_episode(self, prompt, env, budget, *args, **kwargs):
        result = await self.original.run_episode(prompt, env, budget, *args, **kwargs)
        self.round_index += 1
        try:
            from orchestrator import _visible_output
            from role_prompts import (extract_role_orchestrator_plan_text, extract_role_task_contract,
                                      extract_role_task_state, parse_role_orchestrator_next_step)
            previous = self.previous_rounds()
            plan = extract_role_orchestrator_plan_text(_visible_output(result)).strip()
            contract = extract_role_task_contract(plan, fallback=previous[-1].task_contract if previous else "")
            state = extract_role_task_state(plan, fallback=previous[-1].task_state if previous else "")
            route = parse_role_orchestrator_next_step(plan)
            snapshot = snapshot_for_contract(task=self.task, contract=contract, plan=plan,
                manager_state=state, rounds=previous, route=route,
                tools=({"route": "cli", "description": "Original CLI role backend; native tool catalog/permissions not exposed"},))
            if result.status == "done" and plan:
                print(json.dumps({"event": "reflector_shadow_start", "round": self.round_index}), flush=True)
                assessment, trace = await self.reflector.assess(snapshot)
                report = assessment.to_dict()
                status = ("input_budget_exceeded" if report.get("uncertainty") == "input_budget_exceeded"
                          else "runtime_error" if trace.get("runtime_error") else "assessed")
            else:
                report, trace, status = None, {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}, "manager_runtime_error"
            entry = {"mode": "shadow", "round": self.round_index, "next_step": route,
                "snapshot": snapshot.to_dict(), "original_contract": contract, "final_contract": contract,
                "result": report, "assessment_status": status, "trace": trace,
                "dispatch": "original_orchestrator", "enforced": False}
            path = self.role_dir/"rounds"/f"round_{self.round_index:03d}"/"feasibility.json"
            atomic_json(path, entry)
            self.records.append((path, entry))
            print(json.dumps({"event": "reflector_shadow", "round": self.round_index,
                "decision": report["decision"] if report else None, "next_step": route,
                "skill_ids": report["retrieved_skills"] if report else [], "enforced": False,
                "uncertainty": report.get("uncertainty", "") if report else "",
                "assessment_status": status,
                "error_stage": trace.get("error_stage"), "validation_error": trace.get("validation_error"),
                "runtime_error": trace.get("runtime_error"),
                "input_chars": trace.get("input_chars"), "input_char_limit": trace.get("input_char_limit"),
                "llm_calls": trace.get("llm_calls"),
                "log": str(path)}, ensure_ascii=False), flush=True)
        except Exception as exc:
            # Observing must never replace/alter/raise over an original role result.
            error = {"event": "reflector_shadow_error", "round": self.round_index,
                "error_type": type(exc).__name__, "enforced": False}
            self.observation_errors.append(error)
            print(json.dumps(error), flush=True)
        return result

    def finalize(self):
        rows = []
        for path, entry in self.records:
            audit = path.parent/"verifier_report.txt"
            if audit.is_file():
                entry["subsequent_auditor"] = {"report": audit.read_text(), "reference": str(audit)}
            # Link the actual original dispatch/outcome, not the observer's verdict.
            ledger = self.role_dir/"rounds.jsonl"
            if ledger.is_file():
                for line in ledger.read_text().splitlines():
                    original = json.loads(line)
                    if original.get("round_index") == entry["round"]:
                        entry["original_round"] = original
                        break
            atomic_json(path, entry)
            rows.append({"round": entry["round"], "next_step": entry["next_step"],
                "result": entry["result"], "trace": entry["trace"], "reference": str(path)})
        def total(key):
            counts = [r["trace"].get(key) for r in rows]
            return sum(counts) if all(type(c) is int for c in counts) else None
        summary = {"mode": "shadow", "enforced": False, "manager_rounds_observed": self.round_index,
            "assessed_rounds": sum(r["result"] is not None for r in rows), "rounds": rows,
            "observation_errors": self.observation_errors,
            "overhead": {k: total(k) for k in ("llm_calls", "input_tokens", "output_tokens", "runtime_ms")}}
        atomic_json(self.role_dir/"reflector_shadow.json", summary)
        return summary
