from __future__ import annotations

import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from lh_harness.skills.learning import prepare_proposals, curate_proposals, learn
from lh_harness.skills.library import SkillLibrary
from lh_harness.skills.schema import ROLES, sha

spec = importlib.util.spec_from_file_location('stage_fixtures', Path(__file__).with_name('test_role_learning_v2.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class Model(fixtures.EveryRoleModel):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def __call__(self, stage, role, prompt, data, schema):
        self.inputs.append((stage, role, copy.deepcopy(data)))
        return super().__call__(stage, role, prompt, data, schema)


COMBINATIONS = [list(roles) for n in range(4) for roles in itertools.combinations(ROLES, n)]


@pytest.mark.parametrize('roles', COMBINATIONS)
def test_tasks_propose_independently_then_batch_curates_all_selected_roles(tmp_path, roles):
    lib = SkillLibrary(tmp_path/'library')
    lib.create()
    before = lib.digest()
    model, cfg, splits = Model(), fixtures.settings(roles), fixtures.split('task-a', 'task-b')
    docs = [prepare_proposals([fixtures.bundle(task)], cfg, lib, model, splits, output=tmp_path/task)
            for task in ('task-a', 'task-b')]
    assert lib.digest() == before and not lib.list(candidates=True)
    assert all(stage in {'attribution', 'proposer'} for stage, _ in model.calls)
    if not roles:
        assert model.calls == [] and all(doc['status'] == 'disabled' for doc in docs)
        return
    assert all(len(doc['proposals']) == len(roles) for doc in docs)
    assert all(json.loads((tmp_path/task/'proposals.json').read_text()) == doc for task, doc in zip(('task-a','task-b'), docs))
    assert all('bundles' not in json.loads((tmp_path/task/'learning_report.json').read_text()) for task in ('task-a','task-b'))
    assert all(not data['existing'] for stage, _, data in model.inputs if stage == 'proposer')
    assert [data['bundle']['task_id'] for stage, _, data in model.inputs if stage == 'attribution'] == ['task-a','task-b']
    proposal_calls = len(model.calls)
    report = curate_proposals(docs, cfg, lib, model, splits, output=tmp_path/'batch', flush=True)
    assert report['errors'] == [] and report['proposal_count'] == 2*len(roles)
    assert all(stage in {'curator', 'general_curator'} for stage, _ in model.calls[proposal_calls:])
    assert {s.role for s in lib.list()} == set(roles)
    assert all(s.support == 2 for s in lib.list())
    for stage, role, data in model.inputs:
        if stage == 'curator':
            assert {item['task_id'] for item in data['batch_proposals']} == {'task-a','task-b'}
            assert all(item['role'] == role for item in data['batch_proposals'])


def test_offline_learning_finishes_every_proposer_before_any_curator(tmp_path):
    lib, model = SkillLibrary(tmp_path/'library'), Model()
    report = learn([fixtures.bundle('task-a'), fixtures.bundle('task-b')], fixtures.settings(ROLES), lib,
                   model, fixtures.split('task-a','task-b'), output=tmp_path/'batch', flush=True)
    assert report['errors'] == []
    stages = [stage for stage, _ in model.calls]
    assert max(i for i, stage in enumerate(stages) if stage == 'proposer') < min(i for i, stage in enumerate(stages) if stage == 'curator')


@pytest.mark.parametrize('problem', ['checksum', 'duplicate', 'stale_library', 'roles', 'split', 'policy', 'quote'])
def test_invalid_or_stale_task_artifacts_fail_before_batch_model_calls_or_writes(tmp_path, problem):
    lib, model = SkillLibrary(tmp_path/'library'), Model()
    lib.create()
    cfg, splits = fixtures.settings(ROLES), fixtures.split('task-a')
    doc = prepare_proposals([fixtures.bundle()], cfg, lib, model, splits)
    if problem == 'checksum':
        doc['proposals'][0]['skill']['description'] = 'Changed after reflection'
    elif problem == 'stale_library':
        lib.put(fixtures.skill())
    elif problem == 'roles':
        doc['roles'] = ['executor']
    elif problem == 'split':
        doc['split_sha256'] = '0'*64
    elif problem == 'policy':
        doc['learning_policy']['version'] = 'old'
    elif problem == 'quote':
        doc['proposals'][0]['skill']['evidence'][0]['quote'] = 'Invented evidence absent from the task'
    if problem not in {'checksum', 'stale_library', 'duplicate'}:
        doc['artifact_sha256'] = sha({k:v for k,v in doc.items() if k != 'artifact_sha256'})
    before, calls = lib.digest(), len(model.calls)
    with pytest.raises(ValueError):
        curate_proposals([doc, doc] if problem == 'duplicate' else [doc], cfg, lib, model, splits, flush=True)
    assert len(model.calls) == calls and lib.digest() == before


def test_task_reflection_ignores_batch_gate_but_partial_curation_requires_flush(tmp_path):
    lib, model = SkillLibrary(tmp_path/'library'), Model()
    lib.create()
    cfg, splits = fixtures.settings(['manager']), fixtures.split('task-a')
    cfg.training['batch_size'] = 5
    doc = prepare_proposals([fixtures.bundle()], cfg, lib, model, splits)
    assert len(doc['proposals']) == 1
    before, calls = lib.digest(), len(model.calls)
    queued = curate_proposals([doc], cfg, lib, model, splits)
    assert queued['status'] == 'queued' and lib.digest() == before and len(model.calls) == calls
    report = curate_proposals([doc], cfg, lib, model, splits, flush=True)
    assert not report['errors'] and lib.list() == []
    assert len(lib.list(candidates=True)) == 1
    assert lib.list(candidates=True)[0].support == 1


@pytest.mark.parametrize('role', ROLES)
def test_single_task_proposal_remains_candidate_after_batch_curator_acceptance(tmp_path, role):
    from lh_harness.skills.router import SkillRouter
    lib, model = SkillLibrary(tmp_path/'library'), Model()
    lib.create()
    cfg, splits = fixtures.settings([role]), fixtures.split('task-a')
    cfg.training['batch_size'] = 5
    doc = prepare_proposals([fixtures.bundle()], cfg, lib, model, splits)
    assert len(doc['proposals']) == 1 and not lib.list(candidates=True)
    assert SkillRouter(lib, cfg).select(role, 'cli', 'python test')[0] == []
    report = curate_proposals([doc], cfg, lib, model, splits, flush=True)
    assert report['errors'] == [] and lib.list(role) == []
    candidate=lib.list(role,candidates=True)[0]
    assert candidate.support == 1 and candidate.status == 'candidate'
    selected, text, count = SkillRouter(lib, cfg).select(role, 'cli', 'python test')
    assert selected == [] and text == '' and count is None


def test_one_task_role_proposer_error_preserves_other_task_proposals(tmp_path):
    lib, model = SkillLibrary(tmp_path/'library'), Model()
    lib.create()
    cfg, splits = fixtures.settings(ROLES), fixtures.split('task-a','task-b')
    def flaky(stage, role, prompt, data, schema):
        if stage == 'proposer' and role == 'executor' and data['bundle']['task_id'] == 'task-a':
            raise ValueError('one failed reflection')
        return model(stage, role, prompt, data, schema)
    docs = [prepare_proposals([fixtures.bundle(task)], cfg, lib, flaky, splits) for task in ('task-a','task-b')]
    report = curate_proposals(docs, cfg, lib, model, splits, flush=True)
    assert report['status'] == 'completed_with_errors' and len(report['errors']) == 1
    assert lib.list() == []
    assert {s.role for s in lib.list(candidates=True)} == set(ROLES)
    assert lib.list('executor',candidates=True)[0].support == 1


def test_cli_runs_task_reflection_then_batch_curation_without_rereading_runs(tmp_path, monkeypatch):
    from lh_harness import cli
    import lh_harness.skills.experience as experience
    import lh_harness.skills.client as client
    lib, model = SkillLibrary(tmp_path/'library'), Model()
    lib.create()
    before = lib.digest()
    config = tmp_path/'config.toml'
    config.write_text('[run]\nagent="claude_code"\nmodel="fixture-model"\nbase_url="http://127.0.0.1:8010/v1"\n'
                      f'[skills]\nenabled=true\nphase="online"\nlibrary_dir="{lib.root}"\n'
                      '[skills.training]\nenabled=true\ntrain_roles=["manager"]\nbatch_size=5\n')
    split = tmp_path/'split.json'
    split.write_text(json.dumps({k:v for k,v in fixtures.split('task-a','task-b').items() if k != 'sha256'}))
    class Client:
        def __init__(self, **kwargs):
            self.records = []
        def __call__(self, *args):
            return model(*args)
    monkeypatch.setattr(client, 'JsonClient', Client)
    monkeypatch.setattr(experience, 'extract_run', lambda path, task_id, **kw: fixtures.bundle(task_id))
    common = ['skills','learn','--config',str(config),'--roles','manager','--phase','online','--split-manifest',str(split)]
    for task in ('task-a','task-b'):
        assert cli.main([*common,'--stage','propose','--run',task+'=/fixture','--output',str(tmp_path/task)]) == 0
        assert lib.digest() == before
    calls = len(model.calls)
    monkeypatch.setattr(experience, 'extract_run', lambda *a, **kw: pytest.fail('curation must consume saved proposals'))
    assert cli.main([*common,'--stage','curate','--proposal-input',str(tmp_path/'task-a/proposals.json'),
                     '--proposal-input',str(tmp_path/'task-b/proposals.json'),'--flush','--output',str(tmp_path/'batch')]) == 0
    assert all(stage in {'curator','general_curator'} for stage, _ in model.calls[calls:])
    assert len(lib.list('manager')) == 1 and lib.list('manager')[0].support == 2
