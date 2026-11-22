from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    path=ROOT/'experiments/skill_evolution/scripts'/name
    spec=importlib.util.spec_from_file_location('test_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_all_generated_role_configs_share_model_endpoint_and_budget():
    import tomllib
    paths=[ROOT/'experiments/skill_evolution/configs'/f'{name}.toml' for name in
           ('baseline','manager_only','executor_only','auditor_only','manager_executor','manager_auditor','executor_auditor','all_roles')]
    data=[tomllib.loads(p.read_text()) for p in paths]
    expected={k:v for k,v in data[0].items() if k!='skills'}
    assert all({k:v for k,v in d.items() if k!='skills'}==expected for d in data)
    local=[tomllib.loads(p.with_name(p.stem+'.local.toml').read_text()) for p in paths]
    local_expected={k:v for k,v in local[0].items() if k!='skills'}
    assert all({k:v for k,v in d.items() if k!='skills'}==local_expected for d in local)
    assert len({tuple(d['skills']['training']['train_roles']) for d in data})==8


def test_all_experiment_templates_have_unlimited_skill_injection():
    import tomllib
    from lh_harness.skills.schema import Settings, ROLES
    paths=list((ROOT/'experiments/skill_evolution/configs').glob('*.toml'))
    assert len(paths) == 17
    for path in paths:
        settings=Settings.from_dict(tomllib.loads(path.read_text())['skills'])
        assert settings.max_per_role is None, path
        assert settings.router['max_skills_per_role'] is None, path
        assert all(settings.budget(role) is None for role in ROLES), path
        assert settings.training['min_independent_tasks'] == 1, path


def test_model_and_verifier_parity_changes_are_rejected():
    import copy
    module=script('validate_manifest.py')
    left={'agents':[{'import_path':'original','model_name':'same','kwargs':{'max_rounds':25}}], 'verifier':{'timeout':900}}
    right=copy.deepcopy(left);right['agents'][0]['import_path']='enhanced';right['agents'][0]['kwargs']['skills_config']={'enabled':True}
    assert module.parity(left,right)
    right['verifier']['timeout']=1200
    with pytest.raises(ValueError,match='outside skills'):module.parity(left,right)


def test_source_protection_detects_actual_byte_changes(tmp_path):
    import hashlib
    module=script('validate_manifest.py')
    p=tmp_path/'eval/frozen.py';p.parent.mkdir();p.write_text('original')
    out=tmp_path/'experiments/skill_evolution/manifests';out.mkdir(parents=True)
    (out/'baseline-protection.json').write_text(json.dumps({'original_commit':'commit','protected_files':{'eval/frozen.py':hashlib.sha256(p.read_bytes()).hexdigest()}}))
    assert module.protected_sources(tmp_path)['protected_files']==1
    p.write_text('changed')
    with pytest.raises(ValueError,match='changed'):module.protected_sources(tmp_path)


@pytest.mark.parametrize('benchmark',['weave','osworld'])
def test_gui_overlay_preserves_original_orchestrator_and_disabled_behavior(tmp_path,monkeypatch,benchmark):
    import asyncio,sys,types,hashlib
    module=script('prepare_gui_adapter.py')
    source=tmp_path/'experiments/skill_evolution/worktrees/original'/module.SOURCES[benchmark]/'src/cua_harness'
    source.mkdir(parents=True)
    original=b'async def run(*args, **kwargs):\n    return {"original": kwargs["task"]}\n'
    (source/'orchestrator.py').write_bytes(original);(source/'__init__.py').write_text('')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    output=tmp_path/'overlay';record=module.prepare(benchmark,output)
    assert (output/'src/cua_harness/_original_orchestrator.py').read_bytes()==original
    assert record['original_orchestrator_sha256']==hashlib.sha256(original).hexdigest()
    package_name='fixture_gui_'+benchmark
    package=types.ModuleType(package_name);package.__path__=[str(output/'src/cua_harness')]
    monkeypatch.setitem(sys.modules,package_name,package)
    spec=importlib.util.spec_from_file_location(package_name+'.orchestrator',output/'src/cua_harness/orchestrator.py')
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    config=tmp_path/'disabled.json';config.write_text('{"enabled":false}')
    monkeypatch.setenv('LLH_SKILL_SETTINGS',str(config))
    assert asyncio.run(wrapper.run(task='same task'))=={'original':'same task'}


def test_partial_episode_usage_and_unfinished_audit_are_not_counted_as_complete(tmp_path):
    module=script('compare_results.py')
    first=tmp_path/'agent/episodes/cli_verifier/ep001';second=tmp_path/'agent/episodes/cli_verifier/ep002'
    first.mkdir(parents=True);second.mkdir(parents=True)
    (first/'prompt.md').write_text('first');(second/'prompt.md').write_text('interrupted')
    (first/'claude_stream.jsonl').write_text(json.dumps({'type':'result','usage':{'input_tokens':10,'output_tokens':4}})+'\n')
    ledger=tmp_path/'agent/role_orchestration/rounds.jsonl';ledger.parent.mkdir()
    ledger.write_text(json.dumps({'round_index':1,'verifier_report':'Status: complete\nIntegrity: clean\nContract audit: aligned\nAudit facts: independently inspected output.\nBlocking constraints: none\n'})+'\n')
    result=tmp_path/'result.json';result.write_text(json.dumps({'task_name':'fixture','verifier_result':{'rewards':{'reward':1}}}))
    row=module.trial_record(result)
    assert row['reward']==1 and row['total_tokens'] is None
    assert row['reported_usage_subtotal']['total_tokens']==14
    assert row['audit_complete_rate']==.5 and row['contract_alignment_rate']==.5
    assert row['role_tokens']['auditor']['missing_calls']==1


@pytest.mark.parametrize('tokens,status', [(None,'unavailable'), (5000,'estimated')])
def test_comparison_handles_unlimited_injection_without_tokenizer(tmp_path,tokens,status):
    module=script('compare_results.py')
    result=tmp_path/'result.json'
    result.write_text(json.dumps({'task_name':'fixture','verifier_result':{'rewards':{'reward':1}}}))
    (tmp_path/'skill_usage.json').write_text(json.dumps({'skill_injection_tokens':tokens,
        'injections':[{'skills':['manager/shared/relevant-method'],'tokens':tokens}]}))
    row=module.trial_record(result)
    assert row['skill_injection_count']==1
    assert row['skill_injection_tokens_estimated']==tokens
    assert row['skill_injection_token_status']==status
