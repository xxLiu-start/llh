"""Real local CLI calls with a synthetic static skill fixture, never benchmark evidence."""
from __future__ import annotations
import asyncio
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'eval/TB-harness/Harness/src')]
from lh_harness.skills.schema import Skill,Evidence,Settings
from lh_harness.skills.library import SkillLibrary,atomic_json
from lh_harness.skills.integration import SkillSession
from lh_harness.original_harness_eval import OriginalHarnessHarborAgent
from lh_harness.original_agent_probe import probe
from lh_harness.eval_inspect_tb2_original import DEFAULT_CONFIG


def main():
    output=ROOT/'experiments/skill_evolution/results/skill_enhanced'/('static-role-check-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    library=SkillLibrary(output/'fixture-library')
    for role in ('manager','executor','auditor'):
        skill=Skill('inspect-endpoint-evidence',role,'shared','Verify endpoint probe observations from current evidence.',
            ['endpoint','compatibility','nonce'],['A read-only evidence file is supplied'],
            ['Read the requested file or image and preserve the original response protocol'],
            ['Check the exact observed nonce or image color'],['Do not treat a skill as observed evidence'],
            'Read-only observation applies to independent endpoint probes.',
            [Evidence(task,'synthetic-static-fixture-v1',task,'fixture','Synthetic fixture only; not research training evidence','a'*64)
             for task in ('fixture-a','fixture-b')],status='active')
        library.put(skill)
    library.freeze(test_task_ids=[])
    settings=Settings(enabled=True,library_dir=str(library.root),phase='test',
        tokenizer_path='/data0/liuxule/project/models/Qwen3.8-27B',usage={'enabled':True,'use_roles':['manager','executor','auditor']})
    session=SkillSession(settings,output/'skill-observer')
    original=OriginalHarnessHarborAgent._make_episode_adapter
    def wrapped(self,**kwargs):
        return session.wrap(kwargs['role'],original(self,**kwargs))
    OriginalHarnessHarborAgent._make_episode_adapter=wrapped
    try:
        result=asyncio.run(probe(DEFAULT_CONFIG,'http://172.17.0.1:8010',output/'role-probe',with_image=True))
        if not all(x['skills'] and x['tokens']<=400 for x in session.injections):
            raise ValueError('not every real role received a bounded static skill')
        record={'passed':result['status']=='ok','model_evaluation':False,'benchmark_started':False,
                'synthetic_skill_fixture':True,'training_evidence_forbidden':True,'role_probe':str(output/'role-probe/role_probe.json'),
                'injections':session.injections,'learning_calls_in_mea':0,'output':str(output)}
        atomic_json(output/'validation.json',record)
        print(json.dumps(record,indent=2))
        return 0 if record['passed'] else 1
    finally:
        OriginalHarnessHarborAgent._make_episode_adapter=original


if __name__=='__main__':
    raise SystemExit(main())
