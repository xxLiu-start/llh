from __future__ import annotations
import json
import pytest
from lh_harness.skills.client import JsonClient
from lh_harness.skills.schema import object_schema


@pytest.mark.parametrize('backend,protocol,route',[('claude_code','messages','/messages'),('codex','responses','/responses'),('opencode','chat_completions','/chat/completions'),('deepseek_harness','chat_completions','/chat/completions')])
def test_offline_client_uses_native_json_schema_and_actual_usage_without_tools(tmp_path,monkeypatch,backend,protocol,route):
    client=JsonClient(backend=backend,model='explicit-local-model',base_url='http://127.0.0.1:8010/v1',output=tmp_path)
    def request(path,body):
        assert path==route and body['model']=='explicit-local-model' and not body.get('tools')
        if protocol=='messages':
            assert body['output_config']['format']['schema']['additionalProperties'] is False
            return {'content':[{'type':'text','text':'{"decision":"NONE"}'}],'usage':{'input_tokens':20,'output_tokens':6}}
        if protocol=='responses':
            assert body['text']['format']['type']=='json_schema'
            return {'output':[{'content':[{'type':'output_text','text':'{"decision":"NONE"}'}]}],'usage':{'input_tokens':20,'output_tokens':6}}
        assert body['response_format']['type']=='json_schema'
        return {'choices':[{'message':{'content':'{"decision":"NONE"}'}}],'usage':{'prompt_tokens':20,'completion_tokens':6}}
    monkeypatch.setattr(client.client,'json',request)
    assert client('proposer','executor','extract a technique',{},object_schema({'decision':{'type':'string','enum':['NONE']}}))=={'decision':'NONE'}
    assert client.records[0]['usage']['total_tokens']==26
    assert json.loads((tmp_path/'learning_cost.json').read_text())['offline_learning']['total_tokens']==26


def test_schema_failure_keeps_raw_output_and_returned_usage(tmp_path,monkeypatch):
    client=JsonClient(backend='claude_code',model='local',base_url='http://127.0.0.1:8010',output=tmp_path)
    monkeypatch.setattr(client.client,'json',lambda *a:{'content':[{'type':'text','text':'{"unapproved":"field"}'}],'usage':{'input_tokens':20,'output_tokens':6}})
    with pytest.raises(ValueError):client('curator','auditor','review',{},object_schema({'decision':{'type':'string'}}))
    assert client.records[0]['usage']['total_tokens']==26 and client.records[0]['raw_output']


def test_ordinary_mea_does_not_construct_new_offline_trainers():
    # Integration has no imported proposer/curator/client call; explicit CLI
    # learning owns all model construction and costs.
    from pathlib import Path
    import lh_harness.skills.integration as module
    text=Path(module.__file__).read_text()
    assert 'JsonClient' not in text and 'from .learning' not in text
