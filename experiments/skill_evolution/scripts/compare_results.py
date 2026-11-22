"""Paired outcomes from unchanged Harbor rewards, with missing metrics as null."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from lh_harness.skills.library import atomic_json
from lh_harness.skills.trajectory_compactor import compact
from lh_harness.token_usage import from_trajectory,aggregate,unavailable


def trial_record(path):
    value = json.loads(path.read_text())
    if "task_name" not in value:
        return None
    agent = value.get("agent_result") or {}
    metadata = agent.get("metadata") or {}
    inputs, outputs = agent.get("n_input_tokens"), agent.get("n_output_tokens")
    aliases={'orchestrator':'manager','manager':'manager','cli_task':'executor','gui_task':'executor',
             'cli_verifier':'auditor','gui_verifier':'auditor'}
    episode_usage=[]
    for prompt in path.parent.rglob('prompt.md'):
        role=aliases.get(prompt.parent.parent.name)
        if role:
            raw=prompt.parent/'claude_stream.jsonl'
            fallback=prompt.parent/'agent.log'
            source=raw if raw.exists() else fallback if fallback.exists() else None
            usage=from_trajectory(source.read_text(errors='replace')) if source else unavailable()
            episode_usage.append({'role':role,'usage':usage})
    role_usage={role:aggregate([r['usage'] for r in episode_usage if r['role']==role]) for role in ('manager','executor','auditor')}
    measured=aggregate([r['usage'] for r in episode_usage])
    if (inputs is None or outputs is None) and episode_usage:
        inputs,outputs=measured.get('input_tokens'),measured.get('output_tokens')
    reward = ((value.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    total = inputs + outputs if type(inputs) is int and type(outputs) is int else None
    ledgers=list(path.parent.rglob('rounds.jsonl'))
    rounds=[json.loads(line) for line in ledgers[0].read_text().splitlines() if line.strip()] if len(ledgers)==1 else []
    from lh_harness.auditor_agent import parse_audit_report
    audits=[parse_audit_report(r.get('auditor_report') or r.get('verifier_report'),int(r.get('round_index',0)))
            for r in rounds if r.get('auditor_report') or r.get('verifier_report')]
    raw=list(path.parent.rglob('*_raw_trajectory.jsonl'))
    trajectories=[compact(p) for p in raw]
    calls=sum(t['tool_call_count'] for t in trajectories) if raw else None
    known_tool_calls=calls
    if measured.get('status')=='unavailable':calls=None
    repetitions=sum(x['count']-1 for t in trajectories for x in t['repeated_actions']) if raw else None
    rework=sum(a.status=='incomplete' for a in audits[:-1])
    audit_attempts=max(len(audits),sum(r['role']=='auditor' for r in episode_usage))
    injection_files=list(path.parent.rglob('skill_usage.json'))
    injection_records=[json.loads(p.read_text()) for p in injection_files]
    injection_count=sum(len(x['skills']) for record in injection_records for x in record['injections'])
    injection_missing=any(record.get('skill_injection_tokens') is None or
        any(x.get('skills') and x.get('tokens') is None for x in record['injections']) for record in injection_records)
    injection_tokens=None if injection_missing else sum(record['skill_injection_tokens'] for record in injection_records)
    return {"task_id": value["task_name"], "success": reward == 1.0, "reward": reward,
            "exception": value.get("exception_info"), "input_tokens": inputs, "output_tokens": outputs,
            "role_tokens":role_usage,"usage_coverage":measured.get('status'),
            "reported_usage_subtotal":measured.get('reported_subtotal'),
            "total_tokens": total, "rounds": metadata.get("rounds_run",sum(r['role']=='manager' for r in episode_usage) or None),
            "audit_complete": (metadata.get("report") or {}).get("completion_satisfied"),
            "api_cost": None, "tool_calls": calls,
            "reported_tool_call_subtotal":known_tool_calls,
            "repeated_action_rate": repetitions/calls if calls else None,
            "audit_complete_rate":sum(a.status=='complete' for a in audits)/audit_attempts if audit_attempts else None,
            "contract_alignment_rate":sum(a.contract_audit_status=='aligned' for a in audits)/audit_attempts if audit_attempts else None,
            "rework_rate":rework/len(audits) if audits else None,
            "rework_definition":"incomplete audited rounds preceding another audit / audited rounds",
            "execution_seconds":metadata.get('elapsed_seconds') if metadata.get('elapsed_seconds') is not None else execution_seconds(value),
            "skill_injection_count":injection_count,"skill_injection_tokens_estimated":injection_tokens,
            "skill_injection_token_status":"unavailable" if injection_missing else "estimated" if injection_count else "not_applicable",
            "result": str(path)}


def execution_seconds(value):
    from datetime import datetime
    timing=value.get('agent_execution') or {}
    if not timing.get('started_at') or not timing.get('finished_at'):return None
    return (datetime.fromisoformat(timing['finished_at'].replace('Z','+00:00'))-
            datetime.fromisoformat(timing['started_at'].replace('Z','+00:00'))).total_seconds()


def combine_independent(baseline_path, enhanced_path):
    left=json.loads(Path(baseline_path).read_text());right=json.loads(Path(enhanced_path).read_text())
    for key in ('original_commit','original_sources','dataset_sha256','weights_files_sha256','weights_identity_sha256','task_ids','repetitions'):
        if left.get(key)!=right.get(key):raise ValueError(f'independent experiments differ: {key}')
    for key in ('model_id','base_url','protocol','context_length','max_output_tokens','temperature','top_p','reasoning','tool_calling','concurrency'):
        if left['endpoint'].get(key)!=right['endpoint'].get(key):raise ValueError(f'independent inference settings differ: {key}')
    for key in ('serving_version','quantization','dtype','gpu_hardware','context_length','max_output_tokens','temperature','top_p','top_k','reasoning','tool_calling','max_num_seqs','benchmark_concurrency'):
        if left['serving']['observed'].get(key)!=right['serving']['observed'].get(key):raise ValueError(f'independent serving settings differ: {key}')
    if [x['track'] for x in left['tracks']]!=['baseline'] or [x['track'] for x in right['tracks']]!=['skill_enhanced']:
        raise ValueError('provide one independently completed original and one enhanced manifest')
    import yaml
    from validate_manifest import parity
    a=yaml.safe_load((Path(left['tracks'][0]['job']).parent/'config.yaml').read_text())
    b=yaml.safe_load((Path(right['tracks'][0]['job']).parent/'config.yaml').read_text())
    parity(a,b)
    return {**right,'tracks':left['tracks']+right['tracks'],'fair_comparison_verified':True,
            'independent_run_manifests':[str(baseline_path),str(enhanced_path)]}


def compare(manifest_path, *, training_cost=None, independent_variant=None):
    manifest = combine_independent(manifest_path,independent_variant) if independent_variant else json.loads(Path(manifest_path).read_text())
    if not manifest.get("fair_comparison_verified") or {t["track"] for t in manifest["tracks"]} != {"baseline", "skill_enhanced"}:
        raise ValueError("comparison requires one valid original and one valid enhanced track")
    tracks = {}
    for track in manifest["tracks"]:
        job = Path(track["job"])
        validity = json.loads(job.with_suffix(".verification-status.json").read_text())
        if not validity["benchmark_valid"]:
            raise ValueError("infrastructure-invalid jobs cannot enter the performance comparison")
        rows = [trial_record(p) for p in job.glob("*/result.json")]
        rows = [r for r in rows if r]
        if {r["task_id"] for r in rows} != set(manifest["task_ids"]):
            raise ValueError("missing or mismatched paired task IDs")
        grouped = {task: [r for r in rows if r["task_id"] == task] for task in manifest["task_ids"]}
        if len({len(v) for v in grouped.values()}) > 1:
            raise ValueError("unequal repetitions across paired tasks")
        successes = sum(r["success"] for r in rows)
        tokens = sum(r["total_tokens"] for r in rows) if all(r["total_tokens"] is not None for r in rows) else None
        rounds = [r["rounds"] for r in rows]
        def mean_field(name):
            vals=[r[name] for r in rows]
            return statistics.mean(vals) if vals and all(v is not None for v in vals) else None
        tracks[track["track"]] = {"tasks": len(rows), "successes": successes, "success_rate": successes / len(rows),
            "per_trial_success_variance": statistics.pvariance([int(r["success"]) for r in rows]) if len(rows)>1 else None,
            "total_tokens": tokens, "tokens_per_successful_task": tokens / successes if tokens is not None and successes else None,
            "amortized_tokens_per_successful_task": (tokens + training_cost) / successes
                if track["track"] == "skill_enhanced" and tokens is not None and training_cost is not None and successes else None,
            "mean_mea_rounds": statistics.mean(rounds) if rounds and all(x is not None for x in rounds) else None,
            "audit_complete_rate":mean_field('audit_complete_rate'),"contract_alignment_rate":mean_field('contract_alignment_rate'),
            "mean_tool_calls":mean_field('tool_calls'),"repeated_action_rate":mean_field('repeated_action_rate'),
            "rework_rate":mean_field('rework_rate'),"mean_execution_seconds":mean_field('execution_seconds'),
            "skill_injection_count":sum(r['skill_injection_count'] for r in rows),
            "rows": rows}
    return {"scope": "held-out paired pilot; full_suite=false; one run cannot establish a general performance gain",
            "active_skill_performance_test":tracks['skill_enhanced']['skill_injection_count']>0,
            "manifest": str(manifest_path), "tracks": tracks,
            "offline_training_tokens": training_cost,
            "success_rate_delta": tracks["skill_enhanced"]["success_rate"] - tracks["baseline"]["success_rate"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-cost", type=Path)
    p.add_argument("--independent-variant", type=Path, help="Validate and compare two independently launched track manifests")
    args = p.parse_args()
    cost = json.loads(args.training_cost.read_text())["total"].get("total_tokens") if args.training_cost else None
    report = compare(args.manifest, training_cost=cost,independent_variant=args.independent_variant)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
