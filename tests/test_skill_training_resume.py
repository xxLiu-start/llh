from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('resume_fixtures', Path(__file__).with_name('test_online_skill_training.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
online = fixtures.online


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    context = fixtures.fixture.__wrapped__(tmp_path, monkeypatch)
    dataset = context[0]/online.shared.native.DEFAULT_OUTPUT
    online.atomic_json(dataset/online.shared.native.MANIFEST, {'tasks':[{'name':name} for name in 'abcdef']})
    p = online.plan(fixtures.args(context, batch_size=3))
    p['output'].mkdir()
    (p['output']/'config.toml').write_text(p['config_text'])
    online.atomic_json(p['output']/'online-split.json', p['online_split'])
    infrastructure = {'original_commit':'fixture', 'original_sources':{'frozen.py':'digest'},
                      'weights_files_sha256':{'weights':'digest'}, 'serving':{'observed':{'serving_version':'fixture'}}}
    manifest = {'experiment':'online_skill_learning','status':'preparing','group':p['group'],'track':p['track'],
                'phase':'online','train_roles':list(p['roles']),'tasks':[],'task_order':p['tasks'],
                'batch_size':p['batch_size'],'dataset_sha256':p['inventory_sha256'],
                'formal_split_sha256':p['formal_split']['sha256'],'online_split':p['online_split'],
                'final_evaluation_enabled':True,'evaluation_task_order':p['evaluation_tasks'],
                'benchmark':p['benchmark'],'dataset_task_count':p['dataset_task_count'],'infrastructure':infrastructure}
    assert online.run_sequence(p, manifest, fixtures.SimulatedRuntime(p)) == 0
    manifest['status'] = 'interrupted'
    online.write_summary(p, manifest)
    return context, p, manifest, infrastructure


def resume_args(stopped, **changes):
    context, p, _, _ = stopped
    return fixtures.args(context, resume=p['output'], from_task='3', output=None, batch_size=None, **changes)


def hashes(folder):
    return {str(p.relative_to(folder)):online.shared.native.digest(p) for p in folder.rglob('*') if p.is_file()}


def test_resume_plan_inherits_full_task_order_model_and_batch_size_without_writes(stopped):
    _, p, manifest, _ = stopped
    before = hashes(p['output'])
    q = online.plan(resume_args(stopped))
    assert q['resume'] and q['start_index']==3 and q['tasks']==list('abcdef')
    assert q['batch_size']==3 and q['output']==p['output'] and q['evaluation_tasks']==p['evaluation_tasks']
    assert q['config']['run']==p['config']['run'] and q['config']['local_model']==p['config']['local_model']
    assert q['library_before_resume']==online.SkillLibrary(p['output']/'library').digest()
    assert hashes(p['output'])==before


def test_resume_appends_attempts_preserves_prior_files_and_reuses_existing_skills(stopped):
    _, p, original, infrastructure = stopped
    preserved = {key:value for key,value in hashes(p['output']).items()
                 if key.startswith(('tasks/','learning/')) and key != 'tasks/task_summary.json'}
    q = online.plan(resume_args(stopped))
    online.validate_resume_infrastructure(q, infrastructure)
    resumed = online.prepare_resume(q, infrastructure)
    archive = q['output']/'resumes'/q['resume_id']
    assert json.loads((archive/'manifest.json').read_text())==original
    assert resumed['tasks']==original['tasks'][:2] and len(resumed['superseded_tasks'])==4
    runtime = fixtures.SimulatedRuntime(q)
    assert online.run_sequence(q, resumed, runtime)==0
    assert [task for stage,task in runtime.events if stage=='execute']==list('cdef')
    assert runtime.used[0]==('c',['cancel-pending-work'])
    assert [row['task_id'] for row in resumed['tasks']]==list('abcdef')
    assert resumed['tasks'][:2]==original['tasks'][:2]
    assert resumed['tasks'][2]['trial']!=original['tasks'][2]['trial']
    assert '/attempt-'+q['resume_id']+'/' in resumed['tasks'][2]['trial']
    assert runtime.batch_calls==[list('cde'), ['f']]
    assert [batch['batch_id'] for batch in resumed['batches']]==[1,2,3,4]
    after = hashes(p['output'])
    assert all(after[name]==digest for name,digest in preserved.items())
    observed = json.loads((p['output']/'library/online_provenance.json').read_text())['observed_task_ids']
    assert set(observed)==set('abcdef')


def test_resume_dry_run_uses_original_directory_without_starting_or_writing(stopped, monkeypatch):
    _, p, _, _ = stopped
    before = hashes(p['output'])
    monkeypatch.setattr(online.HarborRuntime,'prepare',lambda *a:pytest.fail('dry-run must not call endpoint'))
    assert online.main(['executor','--resume',str(p['output']),'--from-task','3','--dry-run'])==0
    assert hashes(p['output'])==before


@pytest.mark.parametrize('problem',['roles','tasks','batch','endpoint','dataset','prefix_result','running','frozen','start'])
def test_resume_rejects_incompatible_or_incomplete_state_before_writes(stopped, problem):
    _, p, manifest, _ = stopped
    args = resume_args(stopped)
    if problem=='roles':args.roles='manager'
    elif problem=='tasks':args.task=[['c']]
    elif problem=='batch':args.batch_size=5
    elif problem=='endpoint':args.base_url='http://127.0.0.1:9999/v1'
    elif problem=='dataset':online.atomic_json(p['dataset']/online.shared.native.MANIFEST, {'tasks':[]})
    elif problem=='prefix_result':Path(manifest['tasks'][0]['trial'],'result.json').write_text('{}')
    elif problem=='running':
        manifest['status']='running';online.atomic_json(p['output']/'manifest.json',manifest)
    elif problem=='frozen':online.SkillLibrary(p['output']/'library').freeze()
    elif problem=='start':args.from_task='99'
    before = hashes(p['output'])
    with pytest.raises(ValueError):online.plan(args)
    assert hashes(p['output'])==before


def test_resume_preflight_failure_preserves_original_state(stopped, monkeypatch):
    _, p, _, _ = stopped
    before = hashes(p['output'])
    def fail(*args):raise ValueError('endpoint unavailable')
    monkeypatch.setattr(online.HarborRuntime,'prepare',fail)
    assert online.main(['executor','--resume',str(p['output']),'--from-task','3'])==2
    assert hashes(p['output'])==before


def test_resume_refuses_changed_model_weights(stopped):
    _, _, _, infrastructure = stopped
    q = online.plan(resume_args(stopped))
    actual = copy.deepcopy(infrastructure)
    actual['weights_files_sha256']['weights']='changed'
    with pytest.raises(ValueError,match='权重'):online.validate_resume_infrastructure(q,actual)


def test_resume_accounts_for_superseded_attempt_cost_once(stopped):
    _, p, _, infrastructure = stopped
    q = online.plan(resume_args(stopped))
    manifest = online.prepare_resume(q,infrastructure)
    usage = {'status':'reported','input_tokens':7,'output_tokens':3,'total_tokens':10}
    for row in [*manifest['tasks'],*manifest['superseded_tasks']]:
        row['runtime_usage']={'runtime':usage,'categories':{role:usage for role in online.ROLES}}
    online.write_summary(q,manifest)
    summary=json.loads((p['output']/'summary.json').read_text())
    assert summary['completed_tasks']==2 and summary['superseded_task_attempts']==4
    assert summary['runtime_usage']['total_tokens']==60


def test_resume_final_recheck_includes_retained_tasks_and_never_learns(stopped):
    _, p, original, infrastructure = stopped
    q = online.plan(resume_args(stopped))
    manifest = online.prepare_resume(q,infrastructure)
    runtime = fixtures.SimulatedRuntime(q)
    assert online.run_sequence(q,manifest,runtime)==0
    batches = len(runtime.batch_calls)
    assert online.run_final_evaluation(q,manifest,runtime)==0
    assert [task for stage,task in runtime.events if stage=='evaluate']==list('abcdef')
    assert len(runtime.batch_calls)==batches
    assert manifest['tasks'][:2]==original['tasks'][:2]
    assert manifest['final_evaluation']['status']=='completed'
    assert len(manifest['final_evaluation']['tasks'])==6
    assert (p['output']/'evaluation/config.toml').exists()
    assert online.SkillLibrary(p['output']/'library').verify_frozen()
