from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from lh_harness.skill_feedback import SkillFeedbackAgent, learn_from_verifier
from lh_harness.skill_learning import SkillBank, SkillSession, compact_trajectory, stage_skill_candidate
from lh_harness.skill_validation import validate_reused_skill
from lh_harness.types import EpisodeBudget, EpisodeResult
from tests.test_skill_evolution import BuilderAgent, make_skill, plan


class NoModel:
    async def run_episode(self, *args, **kwargs):
        raise AssertionError("deterministic routing must not call a model")


def routed_skill(sid="cmake", *, status="provisional"):
    s = make_skill(sid, status=status)
    s["routing_summary"] = "Use for CMake compiler flags and executable build provenance"
    for role in s["roles"]:
        s["roles"][role]["guidance"] = f"{role} only action for {sid}"
    return s


def session(tmp_path, skills):
    bank = SkillBank(tmp_path/'bank.json', skills=skills)
    bank.save()
    return SkillSession(bank, "en", NoModel(), None, EpisodeBudget(), tmp_path/'routes')


def test_default_router_has_no_llm_role_isolation_and_low_relevance(tmp_path):
    s = session(tmp_path, [routed_skill()])
    text = asyncio.run(s.route("manager", "Original prompt", 1, {"task_goal": "Build using CMake compiler flags"}))
    assert "manager only action" in text
    assert "executor only action" not in text and "auditor only action" not in text
    assert s.records[0]["router_output"]["router_llm_calls"] == 0
    assert s.records[0]["router_output"]["scores"][0]["score"] > 0
    assert asyncio.run(s.route("manager", "Original prompt", 2, {"task_goal": "Classify a poker hand"})) == ""
    assert SkillBank.load(s.bank.path).skills[0]["status"] == "provisional"


def test_default_router_limits_two_keeps_complements_and_deduplicates(tmp_path):
    a, b, duplicate, c = [routed_skill(x) for x in ("cmake_a", "cmake_b", "cmake_d", "cmake_c")]
    duplicate["problem_mechanism"] = a["problem_mechanism"]
    duplicate["roles"] = copy.deepcopy(a["roles"])
    s = session(tmp_path, [a,b,duplicate,c])
    asyncio.run(s.route("manager", "base", 1, {"task_goal": "CMake compiler build"}))
    selected = s.records[-1]["selected_skills"]
    assert len(selected) == 2
    assert {x['skill_id'] for x in selected} == {'cmake_a','cmake_b'}
    assert any(x['reason'] == 'duplicate_capability' for x in s.records[-1]['router_output']['scores'])


def test_negative_trigger_and_role_metadata_filter(tmp_path):
    s = routed_skill()
    s['routing_metadata'] = {'negative_keywords':['readonly investigation'], 'roles':['executor']}
    run = session(tmp_path, [s])
    assert asyncio.run(run.route('manager','base',1,{'task_goal':'CMake compiler'})) == ''
    assert asyncio.run(run.route('cli_executor','base',1,{'task_goal':'CMake readonly investigation'})) == ''
    assert 'Skill: cmake@1' in asyncio.run(run.route('cli_executor','base',2,{'task_goal':'CMake compiler'}))


def test_active_revision_defaults_to_old_and_explicit_validation_selects_new(tmp_path):
    old = routed_skill(status='validated')
    run = session(tmp_path,[old])
    revision = {**copy.deepcopy(old), 'version':2,'version_parent':1,'status':'candidate'}
    revision['roles']['manager']['guidance'] = 'new manager only action'
    stage_skill_candidate(run.bank,revision)
    loaded=SkillBank.load(run.bank.path)
    assert loaded.eligible('manager','en')[0]['version'] == 1
    run.bank=loaded
    assert 'Skill: cmake@1' in asyncio.run(run.route('manager','base',1,{'task_goal':'CMake compiler'}))
    run.candidate=revision
    text=asyncio.run(run.route('manager','base',2,{'task_goal':'CMake compiler'}))
    assert 'Skill: cmake@2' in text and 'new manager only action' in text
    assert loaded.shadowed_skills[0]['version']==1


def test_provisional_revision_keeps_version_history_and_reloads(tmp_path):
    old=make_skill('old',status='provisional')
    bank=SkillBank(tmp_path/'bank.json',skills=[old]);bank.save()
    agent=BuilderAgent()
    run=SkillSession(bank,'en',agent,None,EpisodeBudget(),tmp_path/'run')
    run.records=[{'round':1,'role':'manager','selected_skills':[{'skill_id':'old','version':1}],
                  'injected_context':'old guidance','router_input':{'eligible_skill_cards':[]}}]
    report={'task':'work','status':'incomplete','rounds':[{'round_index':1,'plan_text':'plan','executor_output':'result'}]}
    result=asyncio.run(run._finish_evolution(plan(0,1),report=report,trajectory=compact_trajectory(report,run.records),
        skill_index=[],status={},task_id='new_task',replay=None,reliable_experience=True))
    assert result['request_outcomes'][0]['status']=='provisional'
    reloaded=SkillBank.load(bank.path)
    assert reloaded.skills[0]['version']==2 and reloaded.shadowed_skills[0]['version']==1


def test_rule_router_patch_changes_metadata_only_and_survives_transfer(tmp_path):
    from lh_harness.train_terminal_bench_pro import _transfer_evolution_candidates
    old=make_skill('old',status='provisional');bank=SkillBank(tmp_path/'trial.json',skills=[old]);bank.save()
    target=SkillBank(tmp_path/'persistent.json',skills=[copy.deepcopy(old)]);target.save()
    before=copy.deepcopy(old['roles'])
    run=SkillSession(bank,'en',NoModel(),None,EpisodeBudget(),tmp_path/'run')
    run.records=[{'round':1,'role':'manager','selected_skills':[], 'injected_context':'',
        'router_input':{'eligible_skill_cards':[{'skill_id':'old'}]},'router_output':{'selected_skills':[]}}]
    request=plan(0,0,1);request['update_router'][0]['routing_metadata']={'positive_keywords':['compiler'],'roles':['manager']}
    report={'task':'work','rounds':[{'round_index':1,'plan_text':'compiler problem'}]}
    result=asyncio.run(run._finish_evolution(request,report=report,trajectory=compact_trajectory(report,run.records),
        skill_index=[],status={},task_id='t',replay=None,reliable_experience=True))
    assert result['request_outcomes'][0]['status']=='provisional'
    assert result['router_candidate'] is None
    assert bank.skills[0]['roles']==before
    proposal=json.loads((run.log_dir/'skill_candidate.json').read_text())
    loaded,outcomes,changed=_transfer_evolution_candidates(proposal=proposal,trial_bank_path=bank.path,
                                                       bank_path=target.path,language='en')
    assert changed and outcomes[0]['status']=='provisional'
    persisted=SkillBank.load(target.path)
    assert persisted.skills[0]['routing_metadata']['positive_keywords']==['compiler']
    assert persisted.skills[0]['roles']==before


class FeedbackAgent(BuilderAgent):
    def __init__(self, failure_type='SKILL_GAP'):
        super().__init__();self.failure_type=failure_type;self.payloads=[]
    async def run_episode(self,prompt,env,budget):
        payload=json.loads(prompt.split('INPUT JSON:\n',1)[1]);self.payloads.append(payload)
        if 'Extract reusable experience' in prompt:
            value={'positive_experiences':[{'mechanism':'Reusable handoff gap','strategy':'Pass required evidence',
                    'evidence_refs':['round_001'],'reusable':True,'confidence':'high'}], 'negative_experiences':[]}
        elif 'Plan evidence-grounded Skill evolution' in prompt:
            value={**plan(1), 'failure_attribution':{'failure_type':self.failure_type,
                    'evidence':['round_001','external verifier assertion'],'root_cause':'Observed missing handoff'}}
        else:
            result=await super().run_episode(prompt,env,budget)
            result.metadata['usage']={'input_tokens':10,'output_tokens':5}
            return result
        return EpisodeResult(status='done',metadata={'assistant_visible_output':json.dumps(value),
                                                     'usage':{'input_tokens':10,'output_tokens':5}})


def source_report():
    return {'task':'implement reusable pipeline','status':'complete','completion_satisfied':True,'rounds_run':1,
            'latest_auditor_report':'Status: complete\nIntegrity: clean\nContract audit: aligned',
            'rounds':[{'round_index':1,'plan_text':'Pass evidence','executor_output':'some result',
                       'auditor_report':'Status: complete\nIntegrity: clean\nContract audit: aligned'}]}


def test_external_failure_overrides_internal_success_and_persists_new_skill(tmp_path):
    agent=FeedbackAgent();report=source_report();before=copy.deepcopy(report)
    result=asyncio.run(learn_from_verifier(report=report,routes=[],outcome={'correct':False,'reward':0,
        'result_path':'verifier-result.json'},bank_path=tmp_path/'bank.json',log_dir=tmp_path/'feedback',language='en',agent=agent))
    assert report==before
    assert agent.payloads[0]['task_success'] is False
    assert agent.payloads[1]['external_outcome']['correct'] is False
    assert result['diagnosis_attempted'] is True and result['learning_tokens']==45
    assert result['learning_llm_calls']==3
    assert SkillBank.load(tmp_path/'bank.json').skills[0]['status']=='provisional'
    assert (tmp_path/'feedback/feedback.json').exists()


@pytest.mark.parametrize('failure_type',['ENVIRONMENT_ERROR','TASK_DEFINITION_ERROR','UNKNOWN'])
def test_environment_and_task_definition_failures_do_not_mutate_bank(tmp_path,failure_type):
    path=tmp_path/'bank.json';SkillBank(path).save();before=path.read_bytes()
    agent=FeedbackAgent(failure_type)
    result=asyncio.run(learn_from_verifier(report=source_report(),routes=[],outcome={'correct':False},
        bank_path=path,log_dir=tmp_path/'feedback',language='en',agent=agent))
    assert result['action']=='NO_CHANGE' and result['learning_llm_calls']==2
    assert path.read_bytes()==before and not path.with_name('bank_temporary.json').exists()


def test_missing_verifier_does_not_call_model(tmp_path):
    result=asyncio.run(learn_from_verifier(report=source_report(),routes=[],outcome={'correct':None},
        bank_path=tmp_path/'bank.json',log_dir=tmp_path/'feedback',language='en',agent=NoModel()))
    assert result['skip_reason']=='external_outcome_unavailable'


def test_cost_only_independent_evidence_can_promote_but_same_task_cannot(tmp_path):
    s=routed_skill();s['provenance']={'task_id':'source'}
    run=session(tmp_path,[s]);bank=run.bank
    evidence={'candidate_selected':True,'baseline_success':True,'candidate_success':True,
              'hard_constraint_regression':False,'baseline_tokens':1000,'candidate_tokens':700}
    with pytest.raises(ValueError,match='source task'):
        bank.record_independent_validation('cmake',1,{'task_id':'source',**evidence})
    for _ in range(2):bank.record_independent_validation('cmake',1,{'task_id':'a',**evidence})
    with pytest.raises(ValueError,match='insufficient'):bank.promote_skill('cmake',1)
    bank.record_independent_validation('cmake',1,{'task_id':'b',**evidence});bank.promote_skill('cmake',1)
    assert SkillBank.load(bank.path).skills[0]['status']=='validated'


def trial_files(tmp_path,label,report,*,selected=True):
    out=tmp_path/label/'trial/agent/skill_harness';route=out/'lh_harness/role_orchestration';route.mkdir(parents=True)
    (route/'skill_routes.json').write_text(json.dumps([{'injected_context':'cmake advice' if selected else '',
        'selected_skills':[{'skill_id':'cmake','version':1}] if selected else []}]))
    (out.parent.parent/'result.json').write_text(json.dumps({'verifier_result':{'rewards':{'reward':1}},
        'agent_result':{'n_input_tokens':(report.get('skill_learning') or {}).get('task_tokens',0),'n_output_tokens':0}}))
    return out


def test_real_driver_connection_records_contrast_and_promotes_on_second_task(tmp_path,monkeypatch):
    from lh_harness import train_terminal_bench_pro as train
    path=tmp_path/'bank.json';s=routed_skill();s['provenance']={'task_id':'origin'}
    SkillBank(path,skills=[s]).save();calls=[]
    def replay(config,jobs):
        cfg=yaml.safe_load(config.read_text());calls.append(cfg)
        contrast=SkillBank.load(cfg['agents'][0]['kwargs']['skill_bank_path'])
        assert all(x['skill_id']!='cmake' for x in contrast.skills)
        report={**source_report(),'skill_learning':{'task_tokens':1000}}
        return report,trial_files(tmp_path,f'control_{len(calls)}',report,selected=False)
    monkeypatch.setattr(train,'_run_trial',replay)
    for idx in range(2):
        bank=SkillBank.load(path);report={**source_report(),'task':f'new independent task {idx}',
                                        'skill_learning':{'task_tokens':700}}
        out=trial_files(tmp_path,f'candidate_{idx}',report)
        result=validate_reused_skill(task=f'task_{idx}',template={'agents':[{'kwargs':{'prompt_language':'en'}}]},
            dataset=tmp_path/'dataset',frozen=bank,bank_path=path,baseline=report,baseline_output=out,
            generated_dir=tmp_path/'configs',jobs_root=tmp_path/'jobs',run_id=str(idx),enabled=True)
        assert result['status']=='recorded' and result['additional_trials']==1
        assert result['promoted'] is (idx==1)
    assert len(calls)==2
    assert SkillBank.load(path).skills[0]['status']=='validated'


def test_feedback_transport_normalizes_v1_has_no_tools_and_no_secret_trace(tmp_path,monkeypatch):
    monkeypatch.setenv('ANTHROPIC_BASE_URL','https://example.test/v1/v1')
    monkeypatch.setenv('ANTHROPIC_API_KEY','do-not-log-this-secret')
    monkeypatch.setenv('LH_SKILL_FEEDBACK_PROTOCOL','openai')
    requests=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps({'choices':[{'message':{'content':'{"ok":true}'}}],
                                         'usage':{'prompt_tokens':10,'completion_tokens':2}}).encode()
    def urlopen(req,timeout):requests.append(req);return Response()
    monkeypatch.setattr('lh_harness.skill_feedback.urllib.request.urlopen',urlopen)
    agent=SkillFeedbackAgent({'agents':[{'model_name':'model','kwargs':{}}]},tmp_path/'trace')
    result=agent._request('Return JSON',10)
    assert requests[0].full_url=='https://example.test/v1/chat/completions'
    assert 'tools' not in json.loads(requests[0].data)
    assert result.metadata['usage']=={'input_tokens':10,'output_tokens':2}
    assert 'do-not-log-this-secret' not in (tmp_path/'trace/episode_001.json').read_text()


def test_training_driver_learns_after_real_outcome_without_rewriting_role_report(tmp_path,monkeypatch):
    from lh_harness import train_terminal_bench_pro as train
    agent=FeedbackAgent();role_bytes=[];role_paths=[]
    def execute(config,jobs):
        cfg=yaml.safe_load(config.read_text())
        assert cfg['agents'][0]['kwargs']['defer_skill_learning'] is True
        report={**source_report(),'skill_learning':{'deferred':True,'mode':'awaiting_external_verifier','task_tokens':100}}
        out=trial_files(tmp_path,'baseline',report,selected=False)
        (out.parent.parent/'result.json').write_text(json.dumps({'verifier_result':{'rewards':{'reward':0}}}))
        role=out/'lh_harness/report.json';role.write_text(json.dumps(report));role_paths.append(role);role_bytes.append(role.read_bytes())
        return report,out
    monkeypatch.setattr(train,'_run_trial',execute)
    monkeypatch.setattr('lh_harness.skill_feedback.SkillFeedbackAgent',lambda *args:agent)
    path=tmp_path/'bank.json'
    result=train.train_one('example',template={'agents':[{'kwargs':{'prompt_language':'en'}}]},
        dataset=tmp_path/'dataset',bank_path=path,generated_dir=tmp_path/'configs',
        jobs_root=tmp_path/'jobs',decisions_dir=tmp_path/'decisions',reference_tokens=1000,validate_replay=False)
    assert result['baseline_harbor_outcome']['correct'] is False
    assert result['baseline_harness_success'] is True
    assert result['post_verifier_learning']['external_correct'] is False
    assert result['reason']=='provisional_staged'
    assert role_paths[0].read_bytes()==role_bytes[0]
    assert SkillBank.load(path).skills[0]['status']=='provisional'


def test_old_version_survives_regression_and_becomes_fallback(tmp_path):
    run=session(tmp_path,[routed_skill()]);old=copy.deepcopy(run.bank.skills[0])
    revision={**copy.deepcopy(old),'version':2,'version_parent':1,'status':'candidate'}
    stage_skill_candidate(run.bank,revision)
    run.bank.record_independent_validation('cmake',2,{'task_id':'other','candidate_selected':True,
        'baseline_success':True,'candidate_success':False,'hard_constraint_regression':True})
    loaded=SkillBank.load(run.bank.path)
    assert loaded.skills[0]['lifecycle']=='quarantined'
    assert loaded.eligible('manager','en')[0]['version']==1
    assert loaded.shadowed_skills[0]['roles']==old['roles']


def test_generic_header_does_not_route_audit_control_to_source_header_task(tmp_path):
    s=make_skill('audit_control_header_blocker_triage',status='provisional')
    s['routing_summary']='Use when an auditor control header contradicts its audit body'
    run=session(tmp_path,[s])
    assert asyncio.run(run.route('manager','base',1,{'task_goal':'Build a C++ executable with a header file'}))==''


def test_new_training_costs_are_separate_from_original_trial_tokens():
    from lh_harness.training_results import _row
    record={'post_verifier_learning':{'learning_tokens':45,'learning_llm_calls':3},
            'independent_skill_validation':{'status':'recorded','additional_trials':1,
                 'contrast_token_usage':{'total_tokens':100}}}
    row=_row('example',record,'decision')
    assert row['skill_learning_tokens']==45 and row['skill_validation_tokens']==100
    assert row['total_tokens_including_skill_overhead'] is None  # task usage is absent, not zero


def test_no_reusable_evidence_skips_planner_and_builder(tmp_path):
    class Empty:
        async def run_episode(self,prompt,env,budget):
            assert 'Extract reusable experience' in prompt
            return EpisodeResult(status='done',metadata={'assistant_visible_output':json.dumps({
                'positive_experiences':[],'negative_experiences':[]}), 'usage':{'input_tokens':5,'output_tokens':2}})
    result=asyncio.run(learn_from_verifier(report=source_report(),routes=[],outcome={'correct':True},
        bank_path=tmp_path/'bank.json',log_dir=tmp_path/'feedback',language='en',agent=Empty()))
    assert result['learning_llm_calls']==1 and result['learning_tokens']==7
    assert result['learning_phase']=='no_reusable_evidence'


@pytest.mark.parametrize('kind',['ROUTE_ERROR','TRANSFER_ERROR','COMPLIANCE_ERROR'])
def test_attribution_cannot_create_a_new_core_strategy_for_role_or_routing_error(tmp_path,kind):
    run=session(tmp_path,[]);report=source_report()
    result=asyncio.run(run._finish_evolution(plan(1),report=report,trajectory=compact_trajectory(report,[]),
        skill_index=[],status={'failure_attribution':{'failure_type':kind}},task_id='t',replay=None,reliable_experience=True))
    assert result['request_outcomes'][0]['status']=='rejected'
    assert not SkillBank.load(run.bank.path).skills


def test_source_task_is_not_replayed_as_independent_validation(tmp_path,monkeypatch):
    from lh_harness import train_terminal_bench_pro as train
    report={**source_report(),'skill_learning':{'task_tokens':100}}
    s=routed_skill();s['provenance']={'task_id':hashlib.sha256(('terminal_bench_pro\n'+report['task']).encode()).hexdigest()}
    bank=SkillBank(tmp_path/'bank.json',skills=[s]);bank.save()
    out=trial_files(tmp_path,'source',report)
    monkeypatch.setattr(train,'_run_trial',lambda *args:pytest.fail('source task must not be replayed'))
    result=validate_reused_skill(task='source',template={'agents':[{'kwargs':{}}]},dataset=tmp_path,
        frozen=bank,bank_path=bank.path,baseline=report,baseline_output=out,generated_dir=tmp_path/'configs',
        jobs_root=tmp_path/'jobs',run_id='id',enabled=True)
    assert result['additional_trials']==0


def test_failed_learning_episode_usage_is_unknown_not_free(tmp_path):
    class Broken:
        async def run_episode(self,*args):raise OSError('model unavailable')
    result=asyncio.run(learn_from_verifier(report=source_report(),routes=[],outcome={'correct':False},
        bank_path=tmp_path/'bank.json',log_dir=tmp_path/'feedback',language='en',agent=Broken()))
    assert result['learning_tokens'] is None and result['learning_llm_calls']==1
    assert result['error_stage']=='experience_extraction'


def test_conflicting_multi_target_route_patch_is_atomic(tmp_path):
    a,b=make_skill('old',status='provisional'),make_skill('second',status='provisional')
    b['routing_metadata']={'negative_keywords':['compiler']}
    run=session(tmp_path,[a,b]);before=copy.deepcopy(run.bank.skills)
    run.records=[{'round':1,'role':'manager','selected_skills':[], 'injected_context':'',
        'router_input':{'eligible_skill_cards':[{'skill_id':'old'},{'skill_id':'second'}]},
        'router_output':{'selected_skills':[]}}]
    request=plan(0,0,1);request['update_router'][0].update(affected_skill_ids=['old','second'],
        routing_metadata={'positive_keywords':['compiler']})
    report=source_report()
    result=asyncio.run(run._finish_evolution(request,report=report,trajectory=compact_trajectory(report,run.records),
        skill_index=[],status={},task_id='t',replay=None,reliable_experience=True))
    assert result['request_outcomes'][0]['status']=='rejected'
    assert run.bank.skills==before
