from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from .schema import sha
from .trajectory_compactor import compact, redact
from .learning_evidence import learning_audit, partial_round_evidence, eligibility
from ..token_usage import from_trajectory, aggregate, unavailable


def load_split(path):
    data = json.loads(Path(path).read_text())
    if set(data) != {"dataset", "train", "validation", "test"} or not data["dataset"]:
        raise ValueError("split manifest requires dataset, train, validation and test")
    sets = []
    for key in ("train", "validation", "test"):
        values = data[key]
        if not isinstance(values, list) or not all(isinstance(x, str) and x for x in values) or len(set(values)) != len(values):
            raise ValueError("split task IDs must be unique strings")
        sets.append(set(values))
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train/validation/test overlap")
    return {**data, "sha256": sha(data)}


def extract_run(root, *, task_id, dataset, split="train"):
    from ..auditor_agent import parse_audit_report
    root = Path(root).resolve()
    ledgers = [root] if root.name == "rounds.jsonl" else list(root.rglob("rounds.jsonl"))
    if root.name == "rounds.jsonl":
        root = root.parent
    if len(ledgers) > 1 or (not ledgers and not (root / "result.json").exists()):
        raise ValueError("select one individual run/trial containing exactly one rounds.jsonl")
    ledger = ledgers[0] if ledgers else None
    ledger_root = ledger.parent if ledger else root
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger else []
    refs, roles, audit, trajectories, usage = {}, {r: {} for r in ("manager", "executor", "auditor")}, None, {}, {}
    audits, qualified_audits, negative_audits, normalizations = [], [], [], []
    completed = {int(row.get("round_index") or pos) for pos, row in enumerate(rows, 1)}
    aliases = {"manager": ("manager", "orchestrator"), "executor": ("executor", "task"), "auditor": ("auditor", "verifier")}
    for role, names in aliases.items():
        counters, summaries, seen = [], [], set()
        for name in names:
            # Frozen Harbor writes JSONL inside .txt; root LLH uses .jsonl.
            for path in sorted(ledger_root.rglob(name + "_raw_trajectory.*")) if ledger else []:
                if path.suffix not in {".jsonl", ".txt"}:
                    continue
                match = next((re.fullmatch(r"round_(\d+)", parent.name) for parent in path.parents
                              if re.fullmatch(r"round_(\d+)", parent.name)), None)
                if match and int(match.group(1)) not in completed:
                    continue
                key = (path.parent, name)
                if key in seen:
                    continue
                preferred = path.with_suffix(".jsonl")
                path = preferred if preferred.exists() else path
                seen.add(key)
                summary = compact(path)
                source_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                relative_source = str(path.relative_to(root))
                evidence_reference = f"trajectory:{role}:{len(summaries)}"
                summary["evidence_reference"] = evidence_reference
                summary["source_sha256"] = source_digest
                summaries.append({"reference": relative_source, **summary})
                observed = [f"line {x['line']} {x.get('tool', x.get('tool_use_id', x.get('type', 'event')))}: {x.get('input', x.get('detail', ''))}"
                            for x in [*summary["tool_calls"], *summary.get("tool_results", []), *summary["errors"], *summary["final_events"]]]
                if observed:
                    refs[evidence_reference] = {"role": role, "text": redact("\n".join(observed)),
                        "source_sha256": source_digest, "source_path": relative_source, "kind": "trajectory"}
                counters.append(from_trajectory(path.read_text(errors="replace")))
        trajectories[role], usage[role] = summaries, aggregate(counters) if counters else unavailable()
    ledger_digest = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger else None
    for position, row in enumerate(rows, 1):
        round_id = row.get("round_index") or position
        mapping = {"manager": {"plan": row.get("plan_text", ""), "task_state": row.get("task_state", ""), "task_contract": row.get("task_contract", "")},
                   "executor": {"output": row.get("executor_output", row.get("task_output", ""))},
                   "auditor": {"report": row.get("auditor_report", row.get("verifier_report", ""))}}
        for role, value in mapping.items():
            for key, text in value.items():
                if isinstance(text, str) and text.strip():
                    reference = f"rounds.jsonl:round={round_id}:{role}:{key}"
                    refs[reference] = {"role": role, "text": redact(text), "source_sha256": ledger_digest}
                    roles[role][key] = redact(text)
        text = mapping["auditor"]["report"]
        if text:
            audit, learning_parsed, adapted = learning_audit(text, int(round_id))
            audits.append({"round_index": int(round_id), **audit,
                           "learning_interpretation": {k:v for k,v in learning_parsed.items() if k != 'report_text'}})
            if adapted:
                normalizations.append({'reference':f'rounds.jsonl:round={round_id}:auditor:report', 'rules':adapted,
                                       'evaluation_unchanged':True})
            if (audit["status"] in {"complete", "incomplete"} and audit["integrity_status"] == "clean"
                    and audit["contract_audit_status"] == "aligned" and audit.get("report_text", "").strip()):
                qualified_audits.append(f"rounds.jsonl:round={round_id}:auditor:report")
            if (learning_parsed['status'] in {'complete','incomplete'} and learning_parsed['integrity_status']=='clean'
                    and learning_parsed['contract_audit_status'] in {'aligned','needs_revision'}):
                negative_audits.append(f'rounds.jsonl:round={round_id}:auditor:report')
    partial_refs, partial_rounds, partial_audits, partial_trajectories = partial_round_evidence(root, ledger_root, completed)
    refs.update(partial_refs)
    for item in partial_audits:
        parsed = item['parsed']
        if parsed['status'] in {'complete','incomplete'} and parsed['integrity_status']=='clean' and parsed['contract_audit_status'] in {'aligned','needs_revision'}:
            negative_audits.append(item['reference'])
        if item['normalizations']:
            normalizations.append({'reference':item['reference'],'rules':item['normalizations'],'evaluation_unchanged':True})
    result, terminal_result = {}, {}
    for path in [root / "result.json", ledger_root / "report.json"]:
        if path.exists():
            loaded = json.loads(path.read_text())
            if loaded.get("task_name") and loaded["task_name"] != task_id:
                raise ValueError("task ID does not match the preserved benchmark trial")
            if path == root / "result.json":
                terminal_result = loaded
            result.update(loaded)
    original_task = result.get("task") or result.get("task_name")
    task_config = root/'config.json'
    declared = json.loads(task_config.read_text()).get('task',{}) if task_config.exists() else {}
    task_identifier = terminal_result.get('task_id')
    task_path = declared.get('path') or (task_identifier.get('path') if isinstance(task_identifier,dict) else None)
    if task_path:
        instruction = Path(task_path)/'instruction.md'
        if instruction.is_file() and instruction.parent.name == task_id:
            original_task = instruction.read_text(errors='replace')
            refs['original-task-instruction'] = {'role':'task','text':redact(original_task),
                'source_sha256':hashlib.sha256(instruction.read_bytes()).hexdigest(), 'source_path':str(instruction.resolve())}
    if not isinstance(original_task, str) or not original_task.strip():
        raise ValueError("run lacks an original task identity; cannot establish independent support")
    exception = terminal_result.get("exception_info") or result.get("exception_info")
    exception_type = (exception or {}).get("exception_type", "") if isinstance(exception, dict) else ""
    agent_timeout = exception_type.rsplit(".", 1)[-1] == "AgentTimeoutError"
    infrastructure_reasons = []
    if exception and not agent_timeout:
        infrastructure_reasons.append("terminal_exception:" + (exception_type or "unknown"))
    for pending in partial_rounds:
        for detail in pending['roles'].values():
            if detail.get('status') == 'error':
                infrastructure_reasons.append('partial_backend_episode:error')
    # A tool's connection error can be a task-level mistake (e.g. contacting
    # a service before starting it). Only typed harness failures and native
    # model-proxy events below establish infrastructure failure.
    for row in rows:
        for key in ("manager_status", "executor_status", "auditor_status", "task_status", "verifier_status"):
            value = row.get(key) or {}
            status = value.get("status") if isinstance(value, dict) else value
            if status in {"cancelled", "error"} or (status == "timeout" and not agent_timeout):
                infrastructure_reasons.append("backend_episode:" + str(status))
    # Proxy events independently establish actual service failures. Task budget
    # exhaustion alone is not evidence that the model endpoint was unavailable.
    for path in root.glob("agent/**/request_proxy_requests.jsonl"):
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") in {"upstream_error", "upstream_http_error"}:
                infrastructure_reasons.append("model_proxy:" + event["event"])
                break
    validity = root.parent.with_suffix(".verification-status.json")
    if validity.exists() and not json.loads(validity.read_text()).get("benchmark_valid", False):
        infrastructure_reasons.append("benchmark_infrastructure_invalid")
    if (root / "verifier/infra_error.json").exists():
        infrastructure_reasons.append("verifier_infrastructure_failure")
    runtime_failure = bool(infrastructure_reasons)
    root_result = terminal_result or result
    reward = ((root_result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    result_path = root / "result.json"
    if result_path.exists():
        result_digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
        refs["benchmark-result"] = {"role": "benchmark", "source_sha256": result_digest,
            "text": json.dumps({"task_id": task_id, "verifier_result": root_result.get("verifier_result"),
                                "exception_type": exception_type or None}, ensure_ascii=False)}
        if agent_timeout:
            refs["task-termination"] = {"role": "runtime", "source_sha256": result_digest,
                "text": redact(json.dumps({"exception_type": exception_type,
                    "message": exception.get("exception_message"), "agent_execution": root_result.get("agent_execution")}, ensure_ascii=False))}
    ctrf = root / "verifier/ctrf.json"
    grading_summary = json.loads(ctrf.read_text()).get("results", {}).get("summary", {}) if ctrf.exists() else {}
    grading_log = root / "verifier/test-stdout.txt"
    if grading_log.exists():
        text = grading_log.read_text(errors="replace").split("test session starts", 1)[-1]
        refs["benchmark-verifier"] = {"role": "benchmark", "text": redact(text[-18000:]),
            "source_sha256": hashlib.sha256(grading_log.read_bytes()).hexdigest()}
    audit = audit or {"status": "unknown", "integrity_status": "unknown", "contract_audit_status": "unknown", "report_text": ""}
    positive = (audit["status"] == "complete" and audit["integrity_status"] == "clean" and audit["contract_audit_status"] == "aligned"
                and bool(audit.get("report_text", "").strip()) and not runtime_failure and not agent_timeout and reward != 0)
    disagreement = reward == 0 and grading_summary.get("failed", 0) > 0
    eligible = eligibility(positive=positive, negative_audits=negative_audits, refs=refs,
        failed_tests=grading_summary.get('failed',0)>0, runtime_failure=runtime_failure,
        agent_timeout=agent_timeout, partial_rounds=partial_rounds)
    negative = eligible['negative'] and (agent_timeout or reward == 0 or audit['status']=='incomplete'
                                       or audit['contract_audit_status']=='needs_revision')
    eligible['negative'] = bool(negative)
    eligible['review_allowed'] = bool(positive or negative)
    skill_usage = [json.loads(path.read_text()) for path in root.rglob("skill_usage.json")]
    for ledger_index, usage_ledger in enumerate(skill_usage):
        if not isinstance(usage_ledger, dict):
            continue
        injections = usage_ledger.get("injections", [])
        if not isinstance(injections, list):
            continue
        for call_index, call in enumerate(injections):
            if not isinstance(call, dict):
                continue
            # Only this extractor may create a trajectory reference. Never
            # trust a reference copied into the raw usage ledger or expose
            # absolute local paths to the learning model.
            call.pop("trajectory_reference", None)
            call.pop("trajectory_completeness", None)
            trajectory_status = call.pop("trajectory_status", None)
            raw_path = call.pop("trajectory_path", None)
            role = call.get("role")
            if role not in aliases or not isinstance(raw_path, str) or not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                if candidate.is_symlink():
                    continue
                trajectory_path = candidate.resolve(strict=True)
                relative_source = str(trajectory_path.relative_to(root))
            except (OSError, ValueError):
                continue
            if not trajectory_path.is_file():
                continue
            summary = compact(trajectory_path)
            observed = [f"line {item['line']} {item.get('tool', item.get('tool_use_id', item.get('type', 'event')))}: {item.get('input', item.get('detail', ''))}"
                        for item in [*summary["tool_calls"], *summary.get("tool_results", []), *summary["errors"], *summary["final_events"]]]
            if not observed:
                continue
            source_digest = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
            partial = trajectory_status != "done"
            prefix = "partial" if partial else "trajectory"
            reference = f"{prefix}:{role}:injection:{ledger_index}:{call_index}"
            refs[reference] = {"role": role, "text": redact("\n".join(observed)),
                "source_sha256": source_digest, "source_path": relative_source, "kind": "trajectory",
                "completeness": "partial" if partial else "complete"}
            call["trajectory_reference"] = reference
            call["trajectory_completeness"] = "partial" if partial else "complete"
            trajectories.setdefault(role, []).append({"reference": relative_source,
                "evidence_reference": reference, "completeness": "partial" if partial else "complete", **summary})
    return {"run_id": root.name, "task_id": task_id, "dataset": dataset, "split": split,
            "modality": "gui" if any(row.get("next_step") == "gui" for row in rows) or any(v.get('modality')=='gui' for v in partial_refs.values()) else "cli",
            "task": redact(original_task), "task_fingerprint": sha(original_task),
            "outcome": "success" if positive else "failure" if negative else "defer",
            "manager": roles["manager"], "executor": roles["executor"], "auditor": {**roles["auditor"], **audit},
            "runtime": {"role_tokens": usage, "infrastructure_failure": runtime_failure,
                        "infrastructure_reasons": sorted(set(infrastructure_reasons)), "agent_timeout": agent_timeout,
                        "exception_type": exception_type or None, "completed_rounds": len(rows),
                        "termination_kind": "infrastructure_failure" if runtime_failure else "task_budget_exhausted" if agent_timeout else "normal"},
            "qualified_auditor_evidence": qualified_audits, "audit_rounds": [{k:v for k,v in item.items() if k != "report_text"} for item in audits],
            "negative_auditor_evidence":negative_audits,
            "negative_independent_evidence":eligible['negative_independent_evidence'],
            "learning_eligibility":eligible, "learning_audit_normalizations":normalizations,
            "partial_rounds":partial_rounds, "partial_trajectories":partial_trajectories,
            "partial_evidence_policy":"observed actions only; never proof of completed execution or a verified repair",
            "trajectories": trajectories, "benchmark_reward": reward, "evidence": refs, "source": str(root),
            "benchmark_verifier_summary": grading_summary, "audit_grader_disagreement": disagreement,
            "skill_usage": skill_usage, "skill_usage_is_completion_evidence": False}


def checked_evidence(bundle, references, quotes=None):
    from .schema import Evidence
    result = []
    for index, reference in enumerate(references):
        if reference not in bundle["evidence"]:
            raise ValueError("model cited unknown evidence")
        source = bundle["evidence"][reference]
        quote = quotes[index] if quotes else source["text"][:500]
        if not isinstance(quote, str) or len(quote.strip()) < 8 or quote not in source["text"]:
            raise ValueError("evidence quote is not grounded in the original run")
        result.append(Evidence(bundle["task_id"], bundle["dataset"], bundle["run_id"], reference, quote, source["source_sha256"], bundle.get("task_fingerprint", "")))
    return result
