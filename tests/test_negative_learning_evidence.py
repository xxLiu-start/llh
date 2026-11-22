from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from lh_harness.skills.experience import extract_run
from lh_harness.skills.learning_evidence import learning_audit
from lh_harness.skills.attribution import review_roles, role_evidence_valid, role_reference_slots
from lh_harness.skills.learning import prepare_proposals, curate_proposals
from lh_harness.skills.library import SkillLibrary, atomic_json
from lh_harness.skills.schema import ROLES, sha

spec=importlib.util.spec_from_file_location('negative_fixtures',Path(__file__).with_name('test_role_learning_v2.py'))
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def benchmark(root):
    atomic_json(root/'verifier/ctrf.json',{'results':{'summary':{'tests':1,'failed':1,'passed':0}}})
    (root/'verifier/test-stdout.txt').write_text('AssertionError: the required artifact is missing after execution')


def partial_trial(tmp_path, *, verifier=True, executed=True):
    root=fixtures.trial(tmp_path,completed=False)
    folder=root/'agent/cua_harness/role_orchestration/rounds/round_001'
    folder.mkdir(parents=True)
    (folder/'orchestrator_plan.txt').write_text('Repeat the previously verified inspection, leaving the required artifact unimplemented despite the short budget.')
    atomic_json(folder/'orchestrator_metadata.json',{'status':'done','duration_ms':120})
    (folder/'task_contract.txt').write_text('The required artifact must exist before final verification.')
    if executed:
        (folder/'task_output.txt').write_text('Repeated the inspection as assigned; the required artifact is still missing.')
        atomic_json(folder/'task_metadata.json',{'status':'done','duration_ms':300})
    if verifier:benchmark(root)
    return root


@pytest.mark.parametrize('none_clause',['无（本轮适用的约束均 verified）','none (all applicable constraints verified)'])
def test_parenthetical_none_is_normalized_only_for_learning(tmp_path, none_clause):
    header='Status: complete\nIntegrity: clean\nContract audit: aligned\nAudit facts: independent inspection verified the recorded behavior.\n'
    text=header+'Blocking constraints: '+none_clause+'\n'
    original,adapted,changes=learning_audit(text,1)
    assert original['contract_audit_status']=='unknown'
    assert adapted['contract_audit_status']=='aligned' and changes
    root=fixtures.trial(tmp_path,audit=text)
    before=hashlib.sha256((root/'role_orchestration/rounds.jsonl').read_bytes()).hexdigest()
    b=extract_run(root,task_id='task-a',dataset='fixture-v1')
    assert b['outcome']=='failure' and b['negative_auditor_evidence']
    assert b['qualified_auditor_evidence']==[]
    assert b['auditor']['contract_audit_status']=='unknown'
    assert b['learning_audit_normalizations'][0]['evaluation_unchanged']
    assert hashlib.sha256((root/'role_orchestration/rounds.jsonl').read_bytes()).hexdigest()==before


@pytest.mark.parametrize('clause',['none (but a required artifact is still missing)','无（但仍有阻断项）','missing artifact'])
def test_actual_or_ambiguous_blockers_are_not_normalized(clause):
    text='Status: complete\nIntegrity: clean\nContract audit: aligned\nBlocking constraints: '+clause+'\n'
    original,adapted,changes=learning_audit(text,1)
    assert not changes and original==adapted and adapted['contract_audit_status']=='unknown'


def test_needs_revision_is_negative_evidence_not_positive_success(tmp_path):
    audit=fixtures.AUDIT.replace('Contract audit: aligned','Contract audit: needs_revision')
    root=fixtures.trial(tmp_path,audit=audit)
    b=extract_run(root,task_id='task-a',dataset='fixture-v1')
    assert b['outcome']=='failure' and b['qualified_auditor_evidence']==[]
    assert b['negative_auditor_evidence']==['rounds.jsonl:round=1:auditor:report']
    own=next(ref for ref,v in b['evidence'].items() if v['role']=='manager')
    assert role_evidence_valid(b,'manager',[own,b['negative_auditor_evidence'][0]])
    positive=copy.deepcopy(b);positive['outcome']='success'
    assert not role_evidence_valid(positive,'manager',[own,b['negative_auditor_evidence'][0]])


def test_interrupted_round_keeps_saved_plan_and_benchmark_failure(tmp_path):
    root=partial_trial(tmp_path)
    b=extract_run(root,task_id='task-a',dataset='fixture-v1')
    assert b['runtime']['completed_rounds']==0 and b['outcome']=='failure'
    assert b['partial_rounds'][0]['complete'] is False
    assert b['partial_rounds'][0]['roles']['manager']['status']=='done'
    own,other=role_reference_slots(b,'manager')
    assert own and other==['benchmark-verifier']
    assert role_evidence_valid(b,'manager',[own[0],other[0]])
    assert not role_evidence_valid(b,'executor',[own[0],other[0]])
    assert b['evidence'][own[0]]['source_sha256']==hashlib.sha256((root/'agent/cua_harness/role_orchestration/rounds/round_001/orchestrator_plan.txt').read_bytes()).hexdigest()


def test_original_dataset_instruction_is_preserved_for_partial_attribution(tmp_path):
    root=partial_trial(tmp_path)
    task=tmp_path/'dataset/task-a';task.mkdir(parents=True)
    (task/'instruction.md').write_text('Produce the required artifact and verify its exact contents.')
    atomic_json(root/'config.json',{'task':{'path':str(task)}})
    b=extract_run(root,task_id='task-a',dataset='fixture-v1')
    assert b['task']=='Produce the required artifact and verify its exact contents.'
    assert b['evidence']['original-task-instruction']['role']=='task'


def test_native_partial_tool_actions_are_used_but_prompt_is_not_behavior(tmp_path):
    root=partial_trial(tmp_path,executed=False)
    episode=root/'agent/cua_harness/episodes/cli_task/ep001';episode.mkdir(parents=True)
    (episode/'prompt.md').write_text('Please create the artifact; this is an instruction, not execution evidence.')
    (episode/'claude_stream.jsonl').write_text(json.dumps({'type':'assistant','content':[{'type':'tool_use','name':'Bash','input':{'command':'inspect the previously verified state'}}]})+'\n')
    b=extract_run(root,task_id='task-a',dataset='fixture-v1')
    own,_=role_reference_slots(b,'executor')
    assert own and any('native:cli_task' in ref for ref in own)
    assert all('Please create the artifact' not in b['evidence'][ref]['text'] for ref in own)
    assert b['runtime']['completed_rounds']==0


def test_partial_record_alone_never_establishes_causal_role_failure(tmp_path):
    root=partial_trial(tmp_path,verifier=False)
    b=extract_run(root,task_id='task-a',dataset='fixture-v1')
    assert b['partial_rounds'] and b['outcome']=='defer'
    assert 'NO_INDEPENDENT_NEGATIVE_EVIDENCE' in b['learning_eligibility']['reason_codes']
    def no_model(*args):pytest.fail('no independent evidence: do not call the model')
    r=review_roles(b,no_model)
    assert r['role_assessments']['manager']['evidence_gate_reasons']


def test_partial_behavior_can_never_be_positive_evidence(tmp_path):
    b=extract_run(partial_trial(tmp_path),task_id='task-a',dataset='fixture-v1')
    own,_=role_reference_slots(b,'manager')
    positive=copy.deepcopy(b);positive['outcome']='success'
    positive['qualified_auditor_evidence']=['fake-audit']
    positive['evidence']['fake-audit']={'role':'auditor','text':'Independent complete audit of the prior completed round','source_sha256':'a'*64}
    assert not role_evidence_valid(positive,'manager',[own[0],'fake-audit'])
    assert own[0] not in role_reference_slots(positive,'manager')[0]


def test_infrastructure_failure_stays_excluded_with_partial_records_and_grading(tmp_path):
    root=partial_trial(tmp_path)
    result=json.loads((root/'result.json').read_text());result['exception_info']['exception_type']='EndpointUnavailableError'
    atomic_json(root/'result.json',result)
    b=extract_run(root,task_id='task-a',dataset='fixture-v1')
    assert b['runtime']['infrastructure_failure'] and b['outcome']=='defer'
    assert not b['learning_eligibility']['negative']


class Model:
    def __init__(self,action='NEW',reason='Observed plan leaves the required artifact missing; retain a diagnostic checkpoint.',confidence=.95):
        self.calls=[];self.action=action;self.reason=reason;self.confidence=confidence;self.context=None
    def __call__(self,stage,role,prompt,data,schema):
        self.calls.append((stage,role))
        if stage=='attribution':
            b=data['bundle'];own,other=role_reference_slots(b,'manager')
            refs=[own[0],other[0]]
            assessments={r:{'decision':'LEARN' if r=='manager' else 'DEFER','signal':'negative' if r=='manager' else 'none',
                'rationale':'Observed manager plan and benchmark artifact failure support this diagnostic lesson' if r=='manager' else 'Own behavior evidence is absent',
                'evidence_references':refs if r=='manager' else [],'confidence':self.confidence if r=='manager' else 0,'requirements_covered':False} for r in ROLES}
            return {'primary_role':'manager','contributing_roles':[],'root_cause':'Observed plan does not schedule the required artifact',
                    'evidence_references':refs,'confidence':self.confidence,'requirements_covered':False,'role_assessments':assessments}
        if stage=='proposer':
            assert 'reason' in schema['required']
            b=data['bundle'];own,other=role_reference_slots(b,'manager')
            s=fixtures.skill('manager').to_dict()
            for key in ('role','status','version','updated_at','evidence'):s.pop(key)
            return {**s,'action':self.action,'reason':self.reason,'target_role':'manager',
                    'evidence':[{'reference':ref,'quote':b['evidence'][ref]['text']} for ref in (own[0],other[0])] if self.action in {'NEW','ENHANCE'} else []}
        if stage=='curator':
            self.context=data['proposal_context']
            return {'action':'ACCEPT','target':'','reason':'Verified failure observation supports a prevention lesson, without claiming a proven repair'}
        return {'action':'NONE','source':'','target':'','reason':'No duplicate'}


def test_partial_negative_lesson_is_independently_proposed_then_curated(tmp_path):
    b=extract_run(partial_trial(tmp_path),task_id='task-a',dataset='fixture-v1')
    cfg=fixtures.settings(['manager']);lib=SkillLibrary(tmp_path/'library');lib.create();model=Model()
    before=lib.digest()
    doc=prepare_proposals([b],cfg,lib,model,fixtures.split('task-a'))
    assert doc['errors']==[] and len(doc['proposals'])==1 and lib.digest()==before
    assert all(stage!='curator' for stage,_ in model.calls)
    report=curate_proposals([doc],cfg,lib,model,fixtures.split('task-a'),flush=True)
    assert report['errors']==[] and len(lib.list('manager'))==1
    assert model.context['signal']=='negative' and model.context['repair_validation']=='not_validated'


@pytest.mark.parametrize('action',['NONE','DEFER'])
def test_none_and_defer_record_concrete_proposer_reason(tmp_path,action):
    b=extract_run(partial_trial(tmp_path),task_id='task-a',dataset='fixture-v1')
    model=Model(action=action,reason='Existing skill already covers this exact diagnostic; no novel procedure is supported.')
    doc=prepare_proposals([b],fixtures.settings(['manager']),SkillLibrary(tmp_path/'lib'),model,fixtures.split('task-a'))
    detail=doc['decisions'][0]['role_decisions']['manager']
    assert not doc['errors'] and detail['decision']==action
    assert detail['proposal_diagnostics']['reason']==model.reason and not doc['proposals']


def test_empty_proposer_reason_is_not_silently_accepted(tmp_path):
    b=extract_run(partial_trial(tmp_path),task_id='task-a',dataset='fixture-v1')
    doc=prepare_proposals([b],fixtures.settings(['manager']),SkillLibrary(tmp_path/'lib'),Model(action='NONE',reason=''),fixtures.split('task-a'))
    assert doc['errors'] and 'requires a reason' in doc['errors'][0]['detail']


def test_low_confidence_partial_causality_defers_instead_of_forcing_skill(tmp_path):
    b=extract_run(partial_trial(tmp_path),task_id='task-a',dataset='fixture-v1')
    model=Model(confidence=.3);r=review_roles(b,model)
    assert r['role_assessments']['manager']['decision']=='DEFER'


def test_plan_only_timeout_does_not_blame_manager_for_unobserved_executor(tmp_path):
    b=extract_run(partial_trial(tmp_path,executed=False),task_id='task-a',dataset='fixture-v1')
    assert b['partial_rounds'] and b['outcome']=='failure'
    r=review_roles(b,Model())
    a=r['role_assessments']['manager']
    assert a['decision']=='DEFER' and a['validation_reason']=='MISSING_EXECUTION_PROGRESS_FOR_MANAGER_CAUSALITY'


def test_known_legacy_batch_artifacts_are_read_without_rewriting(tmp_path,monkeypatch):
    import lh_harness.skills.learning as learning
    b=fixtures.bundle();cfg=fixtures.settings(['manager']);lib=SkillLibrary(tmp_path/'lib');lib.create()
    model=fixtures.EveryRoleModel(selected=['manager'])
    doc=prepare_proposals([b],cfg,lib,model,fixtures.split('task-a'))
    legacy=copy.deepcopy(doc['learning_policy']);legacy['version']='known-legacy-policy'
    doc['learning_policy']=legacy;doc['artifact_sha256']=sha({k:v for k,v in doc.items() if k!='artifact_sha256'})
    monkeypatch.setattr(learning,'LEGACY_POLICY_SHA256',{sha(legacy)})
    before=copy.deepcopy(doc)
    report=curate_proposals([doc],cfg,lib,model,fixtures.split('task-a'),flush=True)
    assert report['errors']==[] and doc==before
    assert report['proposal_policies']==[(sha(legacy),'known-legacy-policy')]
