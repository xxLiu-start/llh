from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from lh_harness.feasibility import FeasibilityReflector, ReflectorConfig
from lh_harness.feasibility.legacy_observer import LegacyContractObserver
from lh_harness.feasibility.terminal_bench_shadow import main, write_config
from tests.test_feasibility import JsonAgent, MemoryEnvironment, verdict


PLAN = "当前任务状态:\n待验证\n\n任务契约:\nfixture service package\n\n下一步: CLI任务\n\n任务:\nInspect fixture service package"
DONE = "当前任务状态:\n已完成，见 round_001\n\n任务契约:\nfixture service package\n\n下一步: 完成"
AUDIT = "状态: complete\n完整性: clean\n契约审计: aligned\n\n审计事实: 原始任务全部要求已完成。"


class NativeAgent:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []
        self.usage_totals = {"n_input_tokens": 10, "n_output_tokens": 5, "n_cache_tokens": 0}
        self.results = []
    async def run_episode(self, prompt, env, budget):
        from harness_types import EpisodeResult
        self.prompts.append(prompt)
        result = EpisodeResult(status="done", actions_log=self.replies.pop(0))
        self.results.append(result)
        return result


@pytest.mark.asyncio
async def test_shadow_preserves_the_actual_original_orchestrator(tmp_path):
    import orchestrator
    from harness_types import HarnessConfig
    baseline_roles = [NativeAgent([PLAN, DONE]), NativeAgent(["UNVERIFIED_EXECUTOR_SELF_REPORT"]), NativeAgent([AUDIT])]
    shadow_roles = [NativeAgent([PLAN, DONE]), NativeAgent(["UNVERIFIED_EXECUTOR_SELF_REPORT"]), NativeAgent([AUDIT])]
    baseline = await orchestrator.run(task="fixture service package", env=MemoryEnvironment(),
        config=HarnessConfig(log_dir=str(tmp_path/"baseline"), harness_dir="/fixture", max_total_episodes=2),
        agent=baseline_roles[1], orchestrator_agent=baseline_roles[0], cli_task_agent=baseline_roles[1], cli_verifier_agent=baseline_roles[2])
    block = verdict("BLOCK", quote="fixture service package")
    accept = verdict("ACCEPT"); accept["check_evidence"] = {"Goal Alignment": ["fixture service package"]}
    judge = JsonAgent([block, accept])
    observer = LegacyContractObserver(shadow_roles[0], task="fixture service package",
        role_dir=tmp_path/"shadow"/"role_orchestration",
        reflector=FeasibilityReflector(ReflectorConfig(enabled=True, mode="shadow"), judge, skills=[]))
    observed_results = []
    native_run = observer.run_episode
    async def capture(*args, **kwargs):
        result = await native_run(*args, **kwargs)
        observed_results.append(result)
        return result
    observer.run_episode = capture
    shadow = await orchestrator.run(task="fixture service package", env=MemoryEnvironment(),
        config=HarnessConfig(log_dir=str(tmp_path/"shadow"), harness_dir="/fixture", max_total_episodes=2),
        agent=shadow_roles[1], orchestrator_agent=observer, cli_task_agent=shadow_roles[1], cli_verifier_agent=shadow_roles[2])
    for b, s in zip(baseline_roles, shadow_roles):
        assert b.prompts == s.prompts
    assert baseline["rounds"] == shadow["rounds"]
    assert baseline["completion_satisfied"] == shadow["completion_satisfied"] is True
    assert all(a is b for a, b in zip(observed_results, shadow_roles[0].results))
    assert observer.usage_totals is shadow_roles[0].usage_totals
    summary = observer.finalize()
    assert summary["assessed_rounds"] == 2 and summary["overhead"]["llm_calls"] == 2
    first = json.loads((observer.role_dir/"rounds/round_001/feasibility.json").read_text())
    assert first["result"]["decision"] == "BLOCK" and first["enforced"] is False
    assert first["original_round"]["task_output"] == "UNVERIFIED_EXECUTOR_SELF_REPORT"
    assert first["subsequent_auditor"]["report"]
    assert "UNVERIFIED_EXECUTOR_SELF_REPORT" not in judge.prompts[0]+judge.prompts[1]
    assert AUDIT in json.loads((observer.role_dir/"rounds/round_002/feasibility.json").read_text())["snapshot"]["verified_auditor_reports"][0]["report"]


def test_harbor_config_wraps_original_adapter_without_changing_evaluation(tmp_path):
    from lh_harness import eval_terminal_bench_pro_original as original
    template = yaml.safe_load(Path("evaluation/original_harness_terminal_bench_pro.yaml").read_text())
    before = copy.deepcopy(template)
    original_config, _ = original._write_config(template, task="sample", dataset=tmp_path/"data",
        output_root=tmp_path/"out", config_path=tmp_path/"original.yaml", run_id="same")
    config, path, job = write_config(template, task="sample", dataset=tmp_path/"data",
        output=tmp_path/"out", run_id="same", bank=tmp_path/"bank")
    native = copy.deepcopy(original_config)
    wrapped = copy.deepcopy(config)
    observer = wrapped["agents"][0]
    assert observer["import_path"] == "lh_harness.feasibility.harbor_shadow:ShadowReflectorHarborAgent"
    assert observer["kwargs"].pop("feasibility_reflector")["mode"] == "shadow"
    observer["import_path"] = native["agents"][0]["import_path"]
    assert wrapped == native and template == before
    assert config["datasets"][0]["task_names"] == ["sample"]
    assert yaml.safe_load(path.read_text()) == config


def test_harbor_subclass_uses_the_original_run_and_role_factory(monkeypatch, tmp_path):
    pytest.importorskip("harbor")
    from lh_harness.original_harness_eval import OriginalHarnessHarborAgent
    from lh_harness.feasibility.harbor_shadow import ShadowReflectorHarborAgent
    native = {"orchestrator": NativeAgent([PLAN]), "cli_task": object(), "cli_verifier": object()}
    monkeypatch.setattr(OriginalHarnessHarborAgent, "_make_role_agents", lambda self, **kw: dict(native))
    agent = ShadowReflectorHarborAgent(logs_dir=tmp_path, feasibility_reflector={"enabled": True, "mode": "shadow"})
    agent._shadow_task = "fixture service package"
    result = agent._make_role_agents(output_dir=tmp_path, model="fixture", api_key="fixture-key", base_url="https://example.invalid")
    assert result["cli_task"] is native["cli_task"] and result["cli_verifier"] is native["cli_verifier"]
    assert result["orchestrator"].original is native["orchestrator"]


def test_dry_run_validates_task_ids_without_new_dataset_or_model_call(tmp_path, monkeypatch, capsys):
    data = tmp_path/"dataset"
    for task in ("one", "two"):
        directory = data/task; directory.mkdir(parents=True)
        (directory/"task.toml").write_text("")
        (directory/"instruction.md").write_text("actual original task")
    from lh_harness import eval_terminal_bench_pro_original as original
    monkeypatch.setattr(original, "_run_harbor", lambda *args: pytest.fail("dry run must not execute"))
    assert main(["--dataset", str(data), "--output", str(tmp_path/"out"), "--dry-run", "--task", "two", "--task", "one"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [r["task"] for r in result["tasks"]] == ["two", "one"]
    for row in result["tasks"]:
        config = yaml.safe_load(Path(row["config_path"]).read_text())
        assert config["datasets"] == [{"path": str(data), "task_names": [row["task"]]}]
