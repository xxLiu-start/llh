"""Read-only per-task reporting from preserved outcomes, usage and learning logs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json

from .library import atomic_json
from .schema import ROLES
from ..token_usage import aggregate, unavailable

REPORT_NAME = 'task_summary.json'
KIND = 'per_task_skill_training_summary'
_CACHE = {}


def read_json(path):
    path = Path(path)
    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        prior = _CACHE.get(str(path))
        if prior and prior[0] == signature:
            return prior[1]
        value = json.loads(path.read_text())
        if len(_CACHE) > 4096:
            _CACHE.clear()
        _CACHE[str(path)] = (signature, value)
        return value
    except (OSError, ValueError):
        return None


def obj(value):
    return value if isinstance(value, dict) else {}


def combined(values):
    values=[value for value in values if value.get('status') != 'not_applicable']
    result=aggregate(values)
    if values:
        result['reported_subtotal']={key:sum((value.get(key) or 0) if value.get('status')=='reported'
            else obj(value.get('reported_subtotal')).get(key,0) for value in values)
            for key in ('input_tokens','output_tokens','total_tokens')}
    return result


def seconds(timing):
    try:
        return (datetime.fromisoformat(timing['finished_at'].replace('Z','+00:00')) -
                datetime.fromisoformat(timing['started_at'].replace('Z','+00:00'))).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def locate_trial(root, row, manifest):
    if row.get('trial'):
        return Path(row['trial'])
    folder = root/'tasks'/f"{row['index']:03d}-{row['task_id']}"
    resumes = manifest.get('resumptions', [])
    resume_id = row.get('resume_id')
    if row.get('superseded_by_resume'):
        marker=next((i for i,event in enumerate(resumes) if event['resume_id']==row['superseded_by_resume']),0)
        resumes=resumes[:marker]
    if not resume_id and resumes and row['index'] >= resumes[-1]['start_index']:
        resume_id = resumes[-1]['resume_id']
    if resume_id:
        folder = folder/('attempt-'+resume_id)
    if not folder.is_dir():
        return None
    candidates = [p for p in folder.glob(f"*/{row['task_id']}__*") if p.is_dir()]
    return max(candidates, key=lambda p:p.stat().st_mtime_ns) if candidates else None


def used_skills(ledger, role, *, available=True):
    calls = [row for row in ledger.get('injections', []) if row.get('role') == role]
    counts, versions = Counter(), {}
    for row in calls:
        for index, key in enumerate(row.get('skills', [])):
            counts[key] += 1
            if index < len(row.get('versions', [])):
                versions.setdefault(key, set()).add(row['versions'][index])
    injected = [row for row in calls if row.get('skills')]
    missing = any(row.get('tokens') is None for row in injected)
    return {'status':'unavailable' if not available else 'recorded' if calls else 'not_applicable',
            'skills':[{'key':key,'injection_count':count,'versions':sorted(versions.get(key, []))}
                      for key,count in sorted(counts.items())],
            'role_calls':len(calls), 'injection_tokens':None if missing or not available else sum(row.get('tokens') or 0 for row in injected),
            'injection_token_status':'unavailable' if missing or not available else 'estimated' if injected else 'not_applicable',
            'injection_tokens_are_subset_of_role_input':True}


def reflection_data(row):
    stage = obj(row.get('reflection'))
    proposal_path = stage.get('proposal_path')
    proposal = obj(read_json(proposal_path)) if proposal_path else {}
    report_path = stage.get('report_path')
    report = (obj(read_json(report_path)) if report_path else {}) or obj(stage.get('report')) or proposal
    costs = obj(stage.get('cost'))
    if report_path:
        costs = obj(read_json(Path(report_path).parent/'learning_cost.json')) or costs
    return stage, proposal, report, costs


def decision_reason(detail, decision, training_enabled):
    diagnostics = obj(detail.get('proposal_diagnostics'))
    if isinstance(decision, dict):
        return decision.get('reason'), 'recorded'
    if detail.get('reason'):
        return detail['reason'], 'recorded'
    if diagnostics.get('reason'):
        return diagnostics['reason'], 'recorded'
    if decision == 'TARGET_ROLE_NOT_ENABLED' or training_enabled is False:
        return '该角色未启用训练，没有运行其 Proposer/Curator。', 'configuration'
    if decision == 'NONE':
        return 'Proposer 返回 NONE；此旧版报告没有记录具体原因，不能判断是重复还是缺少可迁移经验。', 'unavailable'
    if decision in {'DEFER','HARNESS_RUNTIME_FAILURE'}:
        assessment = obj(detail.get('assessment'))
        return assessment.get('rationale') or assessment.get('validation_reason') or '记录为 DEFER，但原日志未提供具体原因。', 'recorded' if assessment else 'unavailable'
    if decision == 'SOURCE_OUTCOME_DISABLED':
        return '配置未启用从该类任务结果学习。', 'configuration'
    return None, 'pending'


def learning_role(role, row, proposal, reflection, curated, enabled, library):
    first = next((r for r in reflection.get('decisions', []) if r.get('task_id')==row['task_id']), {})
    original = obj(first.get('role_decisions', {}).get(role))
    final = next((r for r in curated.get('decisions', []) if r.get('task_id')==row['task_id']), {})
    reviewed = bool(final.get('role_decisions', {}).get(role))
    detail = obj(final.get('role_decisions', {}).get(role)) or original
    if 'training_enabled' in detail:
        enabled = detail['training_enabled']
    decision = detail.get('decision')
    reason, reason_status = decision_reason(detail, decision, enabled)
    proposals = []
    for item in proposal.get('proposals', []):
        if item.get('role') != role or item.get('task_id') != row['task_id']:
            continue
        skill = obj(item.get('skill'))
        review = decision if isinstance(decision, dict) else None
        accepted = (review.get('action') in {'ACCEPT','MERGE','UPDATE'} and review.get('changed') is not False) if review else False if reviewed else None
        target = review.get('key') or review.get('target') if review else None
        current = library.get(target, {})
        proposals.append({'proposal_id':item.get('proposal_id'),'action':item.get('action'),
            'name':skill.get('name'),'skill':skill,'proposal_reason':obj(item.get('proposal_context')).get('reason') or obj(original.get('proposal_diagnostics')).get('reason'),
            'curator_decision':review.get('action') if review else decision if reviewed else 'PENDING_BATCH',
            'accepted':accepted,'reason':reason if reviewed else '等待 Batch Curator 审核。',
            'reason_status':reason_status if reviewed else 'pending','target_skill':target,
            'status_at_review':review.get('status') if review else None,
            'current_library_status':current.get('status'),'current_library_version':current.get('version')})
    if not proposals and detail.get('error'):
        report_path=obj(row.get('reflection')).get('report_path')
        if report_path:
            for path in sorted(Path(report_path).parent.glob(f'*/*-proposer-{role}.json')):
                raw=obj(obj(read_json(path)).get('output'))
                if raw.get('action') in {'NEW','ENHANCE'}:
                    proposals.append({'proposal_id':None,'action':raw['action'],'name':raw.get('name'),
                        'skill':raw,'validated':False,'curator_decision':'NOT_SUBMITTED_VALIDATION_FAILED',
                        'accepted':False,'reason':reason,'reason_status':reason_status,'raw_proposer_record':str(path)})
    return {'training_enabled':enabled, 'assessment':detail.get('assessment') or original.get('assessment'),
            'decision':decision if decision is not None else 'TRAINING_DISABLED' if enabled is False else 'NOT_STARTED', 'reason':reason,
            'reason_status':reason_status,'diagnostics':detail.get('proposal_diagnostics') or original.get('proposal_diagnostics'),
            'proposed_skills':proposals,'source_decision_exists':bool(detail)}


def task_record(root, row, manifest, batches, library, train_roles):
    trial = locate_trial(root,row,manifest)
    result = obj(read_json(trial/'result.json')) if trial else {}
    runtime = obj(read_json(trial/'agent/skill_evolution/token_usage.json')) if trial else {}
    runtime = runtime or obj(row.get('runtime_usage'))
    ledger = obj(read_json(trial/'agent/skill_evolution/skill_usage.json')) if trial else {}
    ledger = ledger or obj(row.get('skill_usage'))
    stage, proposal, reflection, costs = reflection_data(row)
    batch_id = obj(row.get('learning')).get('batch_id')
    batch = batches.get(batch_id,{})
    curated = batch.get('report',{})
    if not curated and obj(row.get('learning')).get('report_path'):
        curated = obj(read_json(row['learning']['report_path']))
    reward = row.get('reward')
    if reward is None:
        reward = obj(obj(result.get('verifier_result')).get('rewards')).get('reward')
    scored = type(reward) in (int,float)
    started = row.get('status') != 'not_started'
    execution = obj(runtime.get('runtime')) or (unavailable() if started else aggregate([]))
    reflection_usage = obj(costs.get('total')) or (unavailable() if stage.get('attempted') else aggregate([]))
    roles = {}
    for role in ROLES:
        roles[role] = {'tokens':obj(runtime.get('categories',{}).get(role)) or (unavailable() if started else aggregate([])),
            'skill_usage':used_skills(ledger,role,available=bool(ledger) or not started),
            'skill_learning':learning_role(role,row,proposal,reflection,curated,
                role in train_roles if train_roles is not None else None,library)}
    normalized_roles = [roles[r]['skill_usage'] for r in ROLES]
    injection_missing = any(u['injection_token_status']=='unavailable' for u in normalized_roles)
    bundle = next((b for b in proposal.get('bundles',[]) if b.get('task_id')==row['task_id']),{})
    return {'index':row['index'],'task_id':row['task_id'],'status':row.get('status','unknown'),
        'success':reward==1.0 if scored else None,'success_basis':'原始 Benchmark reward == 1；与 Agent 是否超时单独记录',
        'reward':reward if scored else None,'agent_exception':row.get('agent_exception') or obj(result.get('exception_info')).get('exception_type'),
        'trial':str(trial) if trial else None,'resume_id':row.get('resume_id'),
        'duration_seconds':{'task_wall':row.get('execution_duration_seconds') if row.get('execution_duration_seconds') is not None else seconds(result),
            'agent_execution':seconds(obj(result.get('agent_execution'))),
            'reflection':stage.get('learning_duration_seconds'),'shared_curation_batch_id':batch_id,
            'task_wall_scope':'任务环境准备、Agent 执行、Verifier 和清理；不含逐题反思及共享批次审核'},
        'tokens':{'execution':execution,'reflection':reflection_usage,'execution_plus_reflection':combined([execution,reflection_usage]),
            'reflection_by_stage':costs.get('categories',{}),'shared_curation_batch_id':batch_id,
            'injection_tokens_estimated':None if injection_missing else sum(u['injection_tokens'] or 0 for u in normalized_roles),
            'injection_status':'unavailable' if injection_missing else 'estimated' if any(u['skills'] for u in normalized_roles) else 'not_applicable',
            'injection_is_already_in_execution_input':True,
            'shared_curation_accounting':'共享 Batch 审核成本见 batches，仅记录一次，不虚构每题分摊'},
        'roles':roles,'learning_evidence_gate':bundle.get('learning_eligibility'),
        'historical_audit_status':{key:bundle.get('auditor',{}).get(key) for key in ('status','integrity_status','contract_audit_status')},
        'sources':{'proposal':stage.get('proposal_path'),'reflection_report':stage.get('report_path'),
                   'curation_report':obj(row.get('learning')).get('report_path')}}


def build_task_report(root, manifest=None):
    root=Path(root).expanduser().resolve()
    manifest=obj(manifest) or obj(read_json(root/'manifest.json'))
    if not manifest:
        raise ValueError('运行目录缺少可读取的 manifest.json')
    library={}
    for key,entry in obj(obj(read_json(root/'library/index.json')).get('skills')).items():
        library[key]=obj(read_json(root/'library'/entry['path']))
    batches={}
    for entry in manifest.get('batches',[]):
        path=entry.get('report_path')
        report=obj(read_json(path)) if path else obj(entry.get('report'))
        cost=obj(entry.get('cost'))
        if path:
            cost=obj(read_json(Path(path).parent/'learning_cost.json')) or cost
        batches[entry['batch_id']]={'batch_id':entry['batch_id'],'task_ids':entry.get('task_ids',[]),
            'status':entry.get('status'),'tokens':obj(cost.get('total')) or (unavailable() if entry.get('attempted') else aggregate([])),
            'tokens_by_stage':cost.get('categories',{}),'duration_seconds':entry.get('learning_duration_seconds'),
            'shared':True,'report_path':path,'report':report}
    task_order=manifest.get('task_order') or [row['task_id'] for row in manifest.get('tasks',[])]
    known={row['task_id']:row for row in manifest.get('tasks',[])}
    train_roles=[] if manifest.get('learning_enabled') is False or manifest.get('online_training_enabled') is False else manifest.get('train_roles')
    tasks=[task_record(root,known.get(task,{'task_id':task,'index':i,'status':'not_started'}),manifest,batches,library,train_roles)
           for i,task in enumerate(task_order,1)]
    return {'schema_version':1,'report_kind':KIND,'generated_at':datetime.now(timezone.utc).isoformat(),
        'run_directory':str(root),'run_status':manifest.get('status'),'train_roles':train_roles,'task_order':task_order,
        'summary':{'selected_tasks':len(tasks),'scored_tasks':sum(t['success'] is not None for t in tasks),
                   'successful_tasks':sum(t['success'] is True for t in tasks),
                   'failed_tasks':sum(t['success'] is False for t in tasks),
                   'not_started_tasks':sum(t['status']=='not_started' for t in tasks)},
        'accounting':{'actual_tokens':'status=reported 使用 Backend Usage；缺失为 null/unavailable，保留 reported_subtotal',
            'estimated_tokens':'Skill 注入片段由同模型 Tokenizer 估算，已包含于执行输入，不重复计入总量',
            'shared_costs':'Batch Curator/General Curator 的真实成本仅在 batches 中记一次，不假定按题均分',
            'pending':'尚未评分时 success=null；尚未审核时 accepted=null；旧 NONE 未记录原因时如实标记 unavailable'},
        'tasks':tasks,'batches':[{k:v for k,v in b.items() if k!='report'} for b in batches.values()],
        'superseded_attempts':[task_record(root,row,manifest,batches,library,train_roles) for row in manifest.get('superseded_tasks',[])],
        'final_evaluation':manifest.get('final_evaluation') and {key:manifest['final_evaluation'].get(key) for key in ('status','experiment','output','summary')},
        'normal_mea_outcome_modified':False}


def write_task_report(root, manifest=None):
    root=Path(root).expanduser().resolve()
    cache=Path(__file__).resolve().parents[3]/'.cache';cache.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(str(root).encode()).hexdigest()[:20]
    with (cache/('task-report-write-'+identity+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        output=root/'tasks'/REPORT_NAME
        if output.exists():
            previous=obj(read_json(output))
            if previous.get('report_kind')!=KIND or previous.get('run_directory')!=str(root):
                raise ValueError('拒绝覆盖非本工具生成的文件：'+str(output))
        atomic_json(output,build_task_report(root,manifest))
        return output
