from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from lh_harness.skills.library import atomic_json
from lh_harness.skills.task_report import build_task_report, write_task_report


def usage(inp=11,out=3):
    return {'status':'reported','input_tokens':inp,'output_tokens':out,'total_tokens':inp+out}


@pytest.fixture
def run(tmp_path):
    trial=tmp_path/'tasks/001-a/job/a__trial'
    atomic_json(trial/'result.json',{'task_name':'a','started_at':'2026-11-22T01:00:00Z','finished_at':'2026-11-22T01:02:00Z',
        'agent_execution':{'started_at':'2026-11-22T01:00:10Z','finished_at':'2026-11-22T01:01:50Z'},
        'verifier_result':{'rewards':{'reward':1}},'exception_info':None})
    atomic_json(trial/'agent/skill_evolution/token_usage.json',{'runtime':usage(),'categories':{'manager':usage(),'executor':usage(0,0),'auditor':usage(0,0)}})
    atomic_json(trial/'agent/skill_evolution/skill_usage.json',{'injections':[
        {'role':'manager','native_role':'orchestrator','skills':['manager/shared/method'],'versions':[1],'tokens':3},
        {'role':'manager','native_role':'orchestrator','skills':['manager/shared/method'],'versions':[2],'tokens':3}]})
    proposals=tmp_path/'learning/proposals/001-a'
    skill={'name':'method','role':'manager','modality':'shared','description':'Reusable planning method','procedure':['Inspect remaining work']}
    detail={'training_enabled':True,'decision':'PROPOSED','assessment':{'decision':'LEARN','signal':'positive'},
            'proposal_diagnostics':{'reason':'Useful new method'}}
    role_decisions={'manager':detail,'executor':{'training_enabled':False,'decision':'TARGET_ROLE_NOT_ENABLED'},
                    'auditor':{'training_enabled':False,'decision':'TARGET_ROLE_NOT_ENABLED'}}
    reflection={'decisions':[{'task_id':'a','role_decisions':role_decisions}]}
    atomic_json(proposals/'learning_report.json',reflection)
    adherence={'status':'completed','assessments':[{'key':'manager/shared/method','version':1,'status':'followed'}]}
    atomic_json(proposals/'proposals.json',{**reflection,'bundles':[{'task_id':'a','skill_adherence':adherence}],
        'proposals':[{'task_id':'a','role':'manager','proposal_id':'p1',
        'action':'NEW','skill':skill,'proposal_context':{'reason':'Useful new method'}}]})
    atomic_json(proposals/'learning_cost.json',{'total':usage(5,2),'categories':{'proposer':usage(5,2)}})
    batch=tmp_path/'learning/batch-001'
    curated=copy.deepcopy(reflection)
    curated['decisions'][0]['role_decisions']['manager']['decision']={'action':'ACCEPT','changed':True,
        'key':'manager/shared/method','status':'candidate','reason':'Independent evidence supports the method'}
    atomic_json(batch/'learning_report.json',curated)
    atomic_json(batch/'learning_cost.json',{'total':usage(30,4),'categories':{'curator':usage(30,4)}})
    atomic_json(tmp_path/'library/index.json',{'skills':{'manager/shared/method':{'path':'manager/shared/method.json'}}})
    atomic_json(tmp_path/'library/manager/shared/method.json',{**skill,'status':'active','version':2})
    manifest={'status':'running','task_order':['a','b'],'train_roles':['manager'],'tasks':[{'task_id':'a','index':1,
        'status':'completed','trial':str(trial),'reward':1,'execution_duration_seconds':123.5,
        'reflection':{'attempted':True,'status':'completed','proposal_path':str(proposals/'proposals.json'),
                      'report_path':str(proposals/'learning_report.json'),'learning_duration_seconds':2.5},
        'learning':{'batch_id':1,'report_path':str(batch/'learning_report.json')}}],
        'batches':[{'batch_id':1,'task_ids':['a'],'status':'completed','attempted':True,
                    'report_path':str(batch/'learning_report.json'),'learning_duration_seconds':3.5}]}
    atomic_json(tmp_path/'manifest.json',manifest)
    return tmp_path,manifest


def test_per_task_tokens_success_times_used_skills_and_accepted_proposal(run):
    root,m=run;r=build_task_report(root)['tasks'][0]
    assert r['success'] is True and r['reward']==1
    assert r['tokens']['execution']['total_tokens']==14
    assert r['tokens']['reflection']['total_tokens']==7
    assert r['tokens']['execution_plus_reflection']['total_tokens']==21
    assert r['tokens']['injection_tokens_estimated']==6
    assert r['duration_seconds']['task_wall']==123.5 and r['duration_seconds']['agent_execution']==100
    assert r['duration_seconds']['reflection']==2.5
    assert r['skill_adherence']=={'status':'completed','assessments':[{'key':'manager/shared/method','version':1,'status':'followed'}]}
    manager=r['roles']['manager'];u=manager['skill_usage']
    assert u['skills']==[{'key':'manager/shared/method','injection_count':2,'versions':[1,2]}]
    p=manager['skill_learning']['proposed_skills'][0]
    assert p['accepted'] is True and p['curator_decision']=='ACCEPT'
    assert p['status_at_review']=='candidate' and p['current_library_status']=='active'
    assert p['skill']['description']=='Reusable planning method'
    assert '未启用训练' in r['roles']['executor']['skill_learning']['reason']
    batch=build_task_report(root)['batches'][0]
    assert batch['tokens']['total_tokens']==34 and batch['shared'] is True


@pytest.mark.parametrize('action',['REJECT','DEFER'])
def test_rejected_or_deferred_proposal_preserves_curator_reason(run,action):
    root,m=run;p=Path(m['tasks'][0]['learning']['report_path']);v=json.loads(p.read_text())
    v['decisions'][0]['role_decisions']['manager']['decision']={'action':action,'changed':False,'reason':'The cited evidence does not support this proposed repair'}
    atomic_json(p,v)
    proposal=build_task_report(root)['tasks'][0]['roles']['manager']['skill_learning']['proposed_skills'][0]
    assert proposal['accepted'] is False and proposal['curator_decision']==action
    assert proposal['reason']=='The cited evidence does not support this proposed repair'


def test_pending_new_attempt_never_uses_old_batch_decision(run):
    root,m=run;m['tasks'][0]['learning']={'status':'pending_batch','attempted':False}
    r=build_task_report(root,m)['tasks'][0]
    p=r['roles']['manager']['skill_learning']['proposed_skills'][0]
    assert p['accepted'] is None and p['curator_decision']=='PENDING_BATCH'


def test_curator_error_is_rejected_not_pending_and_error_reason_takes_precedence(run):
    root,m=run;p=Path(m['tasks'][0]['learning']['report_path']);v=json.loads(p.read_text())
    d=v['decisions'][0]['role_decisions']['manager'];d.update(decision='DEFER',reason='Curator endpoint failed')
    atomic_json(p,v)
    proposal=build_task_report(root)['tasks'][0]['roles']['manager']['skill_learning']['proposed_skills'][0]
    assert proposal['accepted'] is False and proposal['reason']=='Curator endpoint failed'


def test_unknown_usage_preserves_known_subtotal_without_inventing_total(run):
    root,m=run;trial=Path(m['tasks'][0]['trial'])
    atomic_json(trial/'agent/skill_evolution/token_usage.json',{'runtime':{'status':'unavailable','input_tokens':None,
        'output_tokens':None,'total_tokens':None,'reported_subtotal':{'input_tokens':11,'output_tokens':3,'total_tokens':14}}})
    t=build_task_report(root)['tasks'][0]['tokens']
    assert t['execution_plus_reflection']['total_tokens'] is None
    assert t['execution_plus_reflection']['reported_subtotal']['total_tokens']==21


def test_missing_skill_usage_is_unavailable_not_zero_injections(run):
    root,m=run;(Path(m['tasks'][0]['trial'])/'agent/skill_evolution/skill_usage.json').unlink()
    r=build_task_report(root)['tasks'][0]
    assert r['roles']['manager']['skill_usage']['status']=='unavailable'
    assert r['tokens']['injection_tokens_estimated'] is None


def test_not_started_tasks_are_present_without_false_failure(run):
    root,m=run;r=build_task_report(root)['tasks'][1]
    assert r['task_id']=='b' and r['status']=='not_started' and r['success'] is None
    assert r['tokens']['execution']['status']=='not_applicable'
    assert r['duration_seconds']['task_wall'] is None


def test_none_does_not_fabricate_legacy_reason(run):
    root,m=run;row=m['tasks'][0];row['learning']={}
    path=Path(row['reflection']['report_path']);d=json.loads(path.read_text())
    detail=d['decisions'][0]['role_decisions']['manager'];detail.pop('proposal_diagnostics');detail['decision']='NONE'
    atomic_json(path,d);atomic_json(Path(row['reflection']['proposal_path']),{**d,'proposals':[]})
    r=build_task_report(root,m)['tasks'][0]['roles']['manager']['skill_learning']
    assert r['reason_status']=='unavailable' and '旧版报告' in r['reason']


def test_locally_invalid_proposal_is_reported_with_failure_reason(run):
    root,m=run;row=m['tasks'][0];row['learning']={}
    path=Path(row['reflection']['report_path']);d=json.loads(path.read_text())
    detail=d['decisions'][0]['role_decisions']['manager']
    detail.update(decision='DEFER',error='ValueError',reason='Skill violates read-only constraints')
    atomic_json(path,d);atomic_json(Path(row['reflection']['proposal_path']),{**d,'proposals':[]})
    atomic_json(path.parent/'manager/0002-proposer-manager.json',{'output':{
        'action':'NEW','name':'invalid-method','description':'Invalid proposal body'}})
    role=build_task_report(root,m)['tasks'][0]['roles']['manager']['skill_learning']
    proposal=role['proposed_skills'][0]
    assert proposal['name']=='invalid-method' and proposal['accepted'] is False
    assert proposal['curator_decision']=='NOT_SUBMITTED_VALIDATION_FAILED'
    assert proposal['reason']=='Skill violates read-only constraints'


def test_final_evaluation_explicitly_disables_learning(run):
    root,m=run;m['learning_enabled']=False;m.pop('train_roles');m['tasks'][0].pop('reflection');m['tasks'][0].pop('learning')
    report=build_task_report(root,m)
    assert report['train_roles']==[]
    assert report['tasks'][0]['roles']['manager']['skill_learning']['decision']=='TRAINING_DISABLED'


def test_report_only_writes_its_own_file_and_protects_user_files(run):
    root,m=run;before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    output=write_task_report(root)
    assert output==root/'tasks/task_summary.json'
    assert all(Path(p).read_bytes()==data for p,data in before.items())
    output.write_text('user-owned notes')
    with pytest.raises(ValueError,match='拒绝覆盖'):write_task_report(root)
    assert output.read_text()=='user-owned notes'


def test_watcher_one_shot_and_terminal_exit_without_model_calls(run):
    root,m=run
    script=Path(__file__).resolve().parents[1]/'experiments/skill_evolution/scripts/report_task_details.py'
    spec=importlib.util.spec_from_file_location('task_report_cli_test',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.main(['--run-dir',str(root)])==0
    m['status']='completed';atomic_json(root/'manifest.json',m)
    assert module.main(['--run-dir',str(root),'--watch','--settle-seconds','0'])==0
