from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from lh_harness.feasibility import Assessment, FeasibilityReflector, ReflectorConfig, Snapshot
from lh_harness.feasibility.bank import SkillBank, stage
from lh_harness.feasibility.integration import disposition, snapshot_for_contract
from lh_harness.feasibility.model import NoToolEnvironment
from lh_harness.feasibility.offline import generate, load_examples, manifest_entries, validate
from lh_harness.feasibility.retrieval import retrieve
from lh_harness.manager import run
from lh_harness.types import EpisodeResult, HarnessConfig, ManagedRound


def skill(sid="FS_dependencies", status="candidate"):
    return {"skill_id": sid, "name": "Runtime package readiness",
        "description": "Check package availability before launching a service.",
        "applicability_conditions": ["launch service package"],
        "preconditions_to_check": ["Required runtime packages are available."],
        "failure_patterns": ["A startup contract assumes absent runtime packages."],
        "required_evidence": ["Observed package status"],
        "counterexamples": ["All runtime packages are already installed."],
        "suggested_corrections": ["Inspect missing packages before startup; reuse installed packages."],
        "source_trajectory_ids": ["trajectory_origin"], "source_task_ids": ["origin"],
        "support_count": 1, "confidence": .9, "status": status, "version": 1}


def snapshot():
    return Snapshot("Launch the service", "launch service with runtime package", "launch service package",
                    "Manager assumes dependencies ready", environment_observations=({"content": "runtime package missing"},))


def verdict(decision="REVISE", quote="runtime package missing"):
    return {"decision": decision, "feasibility": "infeasible" if decision in {"REVISE", "BLOCK"} else "feasible" if decision == "ACCEPT" else "uncertain",
        "issues": [{"type": "MISSING_PRECONDITION" if quote == "runtime package missing" else "UNVERIFIABLE", "severity": "high", "description": "Startup contract has an evidenced defect",
          "evidence": [quote], "affected_requirement": "service startup", "suggested_correction": "Inspect runtime readiness"}]
          if decision in {"REVISE", "BLOCK"} else [],
        "passed_checks": ["Goal Alignment"] if decision == "ACCEPT" else [], "confidence": .9,
        "check_evidence": {"Goal Alignment": ["launch service package"]} if decision == "ACCEPT" else {}}


class JsonAgent:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
    async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
        assert isinstance(env, NoToolEnvironment)
        self.prompts.append(prompt)
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return EpisodeResult(status="done", metadata={"assistant_visible_output": json.dumps(value),
            "usage": {"input_tokens": 12, "output_tokens": 8}})


@pytest.mark.asyncio
async def test_empty_bank_assesses_evidence_without_creating_bank(tmp_path):
    agent = JsonAgent([verdict("ACCEPT")])
    config = ReflectorConfig(enabled=True, mode="gate", skill_bank_path=str(tmp_path/"missing"))
    result, trace = await FeasibilityReflector(config, agent).assess(snapshot())
    assert result.decision == "ACCEPT" and trace["llm_calls"] == 1
    assert result.retrieved_skills == [] and trace["used_skill_guidance"] is False
    assert agent.prompts and not (tmp_path/"missing").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["ACCEPT", "REVISE", "BLOCK", "UNKNOWN"])
async def test_structured_decisions_grounded_and_metered(decision):
    agent = JsonAgent([verdict(decision)])
    result, trace = await FeasibilityReflector(ReflectorConfig(enabled=True, mode="shadow"), agent,
        skills=[skill(status="active")]).assess(snapshot())
    assert result.decision == decision
    assert result.retrieved_skills == ["FS_dependencies"]
    assert trace["input_tokens"] == 12 and trace["llm_calls"] == 1
    assert "future_executor" not in agent.prompts[0]
    assert "source_trajectory_ids" not in agent.prompts[0]


@pytest.mark.asyncio
async def test_fabricated_evidence_or_model_error_fail_open():
    for reply in [verdict(quote="nonexistent observation"), ValueError("api_key=SECRET"), {"decision": "wrong"}]:
        config = ReflectorConfig(enabled=True, mode="gate")
        result, trace = await FeasibilityReflector(config, JsonAgent([reply]), skills=[skill(status="active")]).assess(snapshot())
        assert result.decision == "UNKNOWN"
        assert disposition(config, result, trace, 0) == "dispatch"
        assert "SECRET" not in json.dumps(trace)


@pytest.mark.asyncio
async def test_read_only_capability_is_enforced():
    env = NoToolEnvironment()
    for method in [env.exec, env.upload, env.download, env.screenshot]:
        with pytest.raises(PermissionError):
            await method("ignored")


def test_routing_candidates_irrelevance_counterexamples_complements_and_conflict():
    relevant = skill(status="active")
    unrelated = {**skill("FS_geometry", "active"), "applicability_conditions": ["polytope vertex enumeration"]}
    result, trace = retrieve([skill(), relevant, unrelated], snapshot(), 5)
    assert [s["skill_id"] for s in result] == ["FS_dependencies"]
    assert any(t["reason"] == "not_active" for t in trace)
    observed = replace(snapshot(), environment_observations=({"content": relevant["counterexamples"][0]},))
    assert not retrieve([relevant], observed, 5)[0]
    complement = {**skill("FS_permissions", "active"), "preconditions_to_check": ["Startup directory is writable."],
                  "suggested_corrections": ["Inspect directory permission status."]}
    assert len(retrieve([relevant, complement], snapshot(), 5)[0]) == 2
    complement["potential_conflicts"] = [relevant["skill_id"]]
    assert len(retrieve([relevant, complement], snapshot(), 5)[0]) == 1


def test_snapshot_pre_execution_no_future_output_and_no_claim_as_audit():
    round = ManagedRound(1, "cli", "old plan", executor_output="private_future_like_trace",
                         auditor_report="Only this audit is known", task_state="unverified assertion")
    result = snapshot_for_contract(task="original", contract="opaque original contract", plan="manager plan",
        manager_state="unverified assertion", rounds=[round], route="gui")
    assert result.contract == "opaque original contract"
    assert "private_future_like_trace" not in json.dumps(result.to_dict())
    assert "unverified assertion" not in json.dumps(result.verified_auditor_reports)


def test_policies_and_revision_limit():
    config = ReflectorConfig(enabled=True, mode="gate")
    revise = Assessment("REVISE", "infeasible")
    assert disposition(config, revise, {}, 0) == "revise"
    assert disposition(config, revise, {}, 1) == "dispatch"
    assert disposition(replace(config, revision_exhausted_policy="block"), revise, {}, 1) == "block"
    assert disposition(replace(config, mode="shadow"), Assessment("BLOCK", "infeasible"), {}, 0) == "dispatch"
    assert disposition(config, Assessment("UNKNOWN", "uncertain"), {}, 0) == "dispatch"
    assert disposition(replace(config, unknown_policy="block"), Assessment("UNKNOWN", "uncertain"), {}, 0) == "block"


def test_bank_persistence_duplicates_recovery_and_history(tmp_path):
    bank = SkillBank(tmp_path)
    first, action = stage(bank, skill())
    assert action == "staged" and not bank.load()
    _, action = stage(bank, skill())
    assert action == "existing_candidate_retained"
    another = skill("FS_different_id")
    another["source_trajectory_ids"] = ["trajectory_second"]
    another["source_task_ids"] = ["second"]
    merged, action = stage(bank, another)
    assert action == "merged" and merged["version"] == 2
    assert (tmp_path/"history/FS_dependencies/v1.json").exists()
    (tmp_path/"index.json").write_text("corrupt")
    assert len(SkillBank(tmp_path).load(candidates=True)) == 1
    bank.recover_index()
    assert json.loads((tmp_path/"index.json").read_text())["skills"]
    with pytest.raises(ValueError):
        bank.put({**merged, "status": "active", "version": 3}, reason="unsafe")
    assert not bank.load()


def test_original_role_prompt_output_byte_for_byte():
    # Frozen pre-removal builders: empty Skill slots produced whitespace only.
    import types
    from lh_harness import role_prompts as current
    original = types.ModuleType("lh_harness._old_role_prompts")
    original.__package__ = "lh_harness"
    source = Path(__file__).parents[1]/"archive/legacy_skills/role_prompts_before.py.txt"
    exec(compile(source.read_text(), str(source), "exec"), original.__dict__)
    for language in ("en", "zh"):
        manager = dict(task="original", rounds=[], round_index=1, language=language)
        executor = dict(task="original", plan_text="plan", next_step="cli", workspace_path="/workspace", language=language)
        auditor = {**executor, "executor_output": "observed output"}
        for name, kwargs in [("build_role_manager_prompt", manager), ("build_role_executor_prompt", executor), ("build_role_auditor_prompt", auditor)]:
            assert getattr(original, name)(**kwargs) == getattr(current, name)(**kwargs)


PLAN = "Next: cli\n\nCurrent Task State:\nworking\n\nTask Contract:\nlaunch service package\n\nPlan:\nlaunch service package"
DONE = "Next: done\n\nCurrent Task State:\ncompleted"
AUDIT = "Status: complete\nIntegrity: clean\nContract audit: aligned\n\nSummary:\nArtifact independently verified"


class RoleAgent:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []
    async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
        self.prompts.append(prompt)
        return EpisodeResult(status="done", actions_log=self.replies.pop(0) if self.replies else "closing reply")


class MemoryEnvironment:
    """Only original harness trace I/O; no model tools in this test fixture."""
    async def exec(self, *args, **kwargs):
        from lh_harness.types import ExecResult
        return ExecResult("", "", 0, 0)
    async def upload(self, *args): pass
    async def download(self, *args): pass
    async def screenshot(self): return b""


class StubReflector:
    def __init__(self, decisions): self.decisions, self.snapshots = list(decisions), []
    async def assess(self, snap):
        self.snapshots.append(snap)
        value = self.decisions.pop(0)
        if isinstance(value, Exception): raise value
        return Assessment(value, "infeasible" if value in {"BLOCK", "REVISE"} else "uncertain" if value == "UNKNOWN" else "feasible"), {"llm_calls": 0}


async def run_roles(tmp_path, settings=None, reflector=None, plans=None):
    manager = RoleAgent(plans or [PLAN, DONE, "closing reply"])
    executor, auditor = RoleAgent(["artifact created"]), RoleAgent([AUDIT])
    result = await run(task="Launch service", env=MemoryEnvironment(), config=HarnessConfig(
        log_dir=str(tmp_path/"logs"), harness_dir="/trace", workspace_path="/workspace",
        max_total_episodes=3, feasibility_reflector=settings),
        manager_agent=manager, cli_executor_agent=executor, gui_executor_agent=executor,
        cli_auditor_agent=auditor, gui_auditor_agent=auditor, feasibility_reflector=reflector)
    return result, manager, executor, auditor


@pytest.mark.asyncio
async def test_disabled_shadow_accept_unknown_and_runtime_preserve_original_inputs(tmp_path):
    baseline, bm, be, ba = await run_roles(tmp_path/"baseline")
    assert baseline["completion_satisfied"]
    for name, settings, decisions in [
        ("disabled", {"enabled": False, "mode": "gate"}, []),
        ("shadow", {"enabled": True, "mode": "shadow"}, ["BLOCK"]),
        ("accept", {"enabled": True, "mode": "gate"}, ["ACCEPT"]),
        ("unknown", {"enabled": True, "mode": "gate"}, ["UNKNOWN"]),
        ("failure", {"enabled": True, "mode": "gate"}, [RuntimeError("network")]),
    ]:
        reflector = StubReflector(decisions)
        result, manager, executor, auditor = await run_roles(tmp_path/name, settings, reflector)
        assert result["completion_satisfied"]
        assert manager.prompts == bm.prompts
        assert executor.prompts == be.prompts
        assert auditor.prompts == ba.prompts
        assert result["rounds"] == baseline["rounds"]
        if name == "disabled": assert not reflector.snapshots
        else:
            sidecar = json.loads((tmp_path/name/"logs/role_orchestration/rounds/round_001/feasibility.json").read_text())
            assert sidecar["dispatch"] == "dispatch" and sidecar["subsequent_auditor"]["report"] == AUDIT


@pytest.mark.asyncio
async def test_revise_same_manager_bounded_and_block_never_dispatches(tmp_path):
    settings = {"enabled": True, "mode": "gate"}
    result, manager, executor, auditor = await run_roles(tmp_path/"revision", settings,
        StubReflector(["REVISE", "REVISE"]), [PLAN, PLAN, DONE, "closing reply"])
    assert result["completion_satisfied"] and len(executor.prompts) == 1
    assert len(auditor.prompts) == 1 and "Feasibility feedback" in manager.prompts[1]
    assert result["rounds"][0]["auditor_report"] == ""
    result, _, executor, auditor = await run_roles(tmp_path/"block", settings, StubReflector(["BLOCK"]))
    assert not result["completion_satisfied"] and not executor.prompts and not auditor.prompts
    assert result["abort_reason"] == "feasibility_blocked"


def make_manifest(tmp_path, rows):
    manifest = []
    for i, row in enumerate(rows):
        report = {"task": f"Launch service variant {i}", "rounds": [asdict(ManagedRound(1, "cli", PLAN,
            executor_output="runtime package missing", auditor_report="Runtime startup unavailable",
            task_contract="launch service package", task_state="working"))]}
        file = tmp_path/f"report{i}.json"
        file.write_text(json.dumps(report))
        manifest.append({"trajectory_id": f"trajectory_{i}", "task_id": f"task_{i}", "path": str(file), **row})
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_no_test_leakage_and_task_split_guard(tmp_path):
    path = make_manifest(tmp_path, [{"split": "train"}, {"split": "test"}])
    rows = json.loads(path.read_text()); Path(rows[1]["path"]).unlink()
    assert len(load_examples(path, "train")) == 1
    with pytest.raises(ValueError): load_examples(path, "test")
    rows[1]["task_id"] = rows[0]["task_id"]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="overlap"): manifest_entries(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["PLANNING_DEFECT", "DEPENDENCY_DEFECT", "TOOL_DEFECT", "EXECUTOR_FAILURE", "ENVIRONMENT_FAILURE", "AUDITOR_FAILURE", "UNCERTAIN"])
async def test_learning_only_supported_planning_failures(tmp_path, category):
    manifest = make_manifest(tmp_path, [{"split": "train"}])
    proposal = {"failure_type": category, "confidence": .9, "evidence": ["runtime package missing"],
                "root_cause": "A contract prerequisite is unsupported", "skill": skill()}
    proposal["skill"]["private_chain_of_thought"] = "DO_NOT_PERSIST_PRIVATE_REASONING"
    bank = SkillBank(tmp_path/"bank")
    result = await generate(manifest, bank, JsonAgent([proposal]))
    count = len(bank.load(candidates=True))
    assert count == int(category in {"PLANNING_DEFECT", "DEPENDENCY_DEFECT", "TOOL_DEFECT"})
    assert not bank.load()
    assert "DO_NOT_PERSIST_PRIVATE_REASONING" not in json.dumps(bank.load(candidates=True))
    assert result["records"][0]["attribution"]["failure_type"] == category


@pytest.mark.asyncio
async def test_no_promotion_without_independent_validation_and_downstream_evidence(tmp_path):
    manifest = make_manifest(tmp_path, [{"split": "validation", "labels": {"1": {
        "feasibility_defect": True, "evidence": ["runtime package missing"]}}} for _ in range(3)])
    bank = SkillBank(tmp_path/"bank"); stage(bank, skill())
    # Model can cite only pre-execution plan, not future failure stored in validation labels.
    agent = JsonAgent([verdict(quote="launch service package") for _ in range(3)])
    result = await validate(manifest, bank, agent)
    report = result["candidates"][0]
    assert report["recall"] == 1 and report["independent_support"] == 3
    assert not report["promote"] and "downstream_improvement_unverified" in report["reasons"]
    assert all('"labels"' not in p and '"executor_output"' not in p for p in agent.prompts)
    assert not bank.load()


@pytest.mark.asyncio
@pytest.mark.parametrize("regression", [False, True])
async def test_held_out_promotion_requires_false_rejection_and_paired_actual_artifacts(tmp_path, regression):
    labels = [{"split": "validation", "labels": {"1": {
        "feasibility_defect": i < 3, "evidence": ["runtime package missing"]}}} for i in range(4)]
    manifest = make_manifest(tmp_path, labels)
    entries = json.loads(manifest.read_text())
    for i, entry in enumerate(entries):
        files = []
        for condition in ("baseline", "reflector"):
            path = tmp_path/f"paired_{i}_{condition}.json"
            path.write_text(json.dumps({"task_id": entry["task_id"], "split": "validation", "condition": condition,
                "skill_id": "FS_dependencies", "success": (True if condition == "baseline" else False) if regression else condition == "reflector",
                "controls": {"model": "test-model", "tools": "fixture-tools", "budget": "same", "environment": "fixture"}}))
            files.append(str(path))
        entry["validation_artifact_paths"] = files
        entry["labels"]["1"]["paired_downstream"] = {"evidence_paths": files}
    manifest.write_text(json.dumps(entries))
    bank = SkillBank(tmp_path/"bank"); stage(bank, skill())
    agent = JsonAgent([*[verdict(quote="launch service package") for _ in range(3)], verdict("ACCEPT")])
    result = await validate(manifest, bank, agent)
    report = result["candidates"][0]
    assert report["precision"] == 1 and report["false_rejection_rate"] == 0
    assert report["promote"] is (not regression)
    assert bool(SkillBank(bank.root).load()) is (not regression)
    if not regression:
        assert bank.load()[0]["version"] == 2
        assert (bank.root/"history/FS_dependencies/v1.json").exists()
        selected, _ = retrieve(bank.load(), snapshot(), 5)
        assert selected[0]["skill_id"] == "FS_dependencies"
    else:
        assert "downstream_regression" in report["reasons"]


def test_semantic_merge_requires_full_alignment_preserves_complement(tmp_path):
    from lh_harness.feasibility.bank import merge_semantic_candidate
    bank = SkillBank(tmp_path/"bank"); stage(bank, skill())
    new = skill("FS_paraphrase")
    new["source_trajectory_ids"] = ["trajectory_other"]
    new["preconditions_to_check"] = ["Runtime dependencies have been observed present."]
    mapping = {k: [{"new": n, "existing": o} for n, o in zip(new[k], skill()[k])]
        for k in ("applicability_conditions", "preconditions_to_check", "suggested_corrections", "failure_patterns")}
    equivalent = {"skill_id": "FS_dependencies", "equivalent": True, "contradictions": [], "alignment": mapping}
    assert merge_semantic_candidate(bank, new, {**equivalent, "alignment": {}}) is None
    merged = merge_semantic_candidate(bank, new, equivalent)
    assert merged["preconditions_to_check"] == skill()["preconditions_to_check"]
    assert len(merged["source_trajectory_ids"]) == 2
    complementary = {**new, "preconditions_to_check": new["preconditions_to_check"]+["Verify directory write permissions."]}
    assert merge_semantic_candidate(bank, complementary, equivalent) is None


def test_failed_atomic_update_retains_previous_version(tmp_path, monkeypatch):
    import lh_harness.feasibility.bank as module
    bank = SkillBank(tmp_path/"bank"); stage(bank, skill())
    original = (bank.root/"candidates/FS_dependencies.json").read_bytes()
    replace_file = module.os.replace
    def fail(src, dst):
        if Path(dst).parent.name == "candidates": raise OSError("disk error fixture")
        replace_file(src, dst)
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError): bank.put({**skill(), "version": 2}, reason="fixture revision")
    assert (bank.root/"candidates/FS_dependencies.json").read_bytes() == original


def test_project_configuration_and_removed_legacy_flags(tmp_path):
    from lh_harness.config import load_run_defaults
    file = tmp_path/"config.toml"
    file.write_text('[run.feasibility_reflector]\nenabled = true\nmode = "shadow"\n')
    defaults = load_run_defaults(file)
    assert defaults["feasibility_reflector"]["mode"] == "shadow"
    assert not any(k.startswith("skill_") for k in HarnessConfig.__dataclass_fields__)
    assert "skill_learning" not in Path(__file__).parents[1].joinpath("src/lh_harness/manager.py").read_text()


@pytest.mark.asyncio
async def test_json_transport_no_tools_and_no_credentials_in_trace(monkeypatch):
    from lh_harness.feasibility.model import JsonModelAgent
    import lh_harness.feasibility.model as module
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, *args): return json.dumps({"choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}).encode()
    def urlopen(request, **kwargs):
        requests.append(request)
        return Response()
    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    agent = JsonModelAgent(model="fixture", base_url="https://example.invalid/v1/v1", api_key="private-key")
    result = await agent.run_episode("snapshot", NoToolEnvironment(), __import__('lh_harness.types',fromlist=['EpisodeBudget']).EpisodeBudget(1))
    assert requests[0].full_url == "https://example.invalid/v1/chat/completions"
    assert "tools" not in json.loads(requests[0].data)
    assert "private-key" not in json.dumps(asdict(result))


@pytest.mark.asyncio
async def test_positive_experience_adds_counterexample_only(tmp_path):
    manifest = make_manifest(tmp_path, [{"split": "train"}])
    entry = json.loads(manifest.read_text())[0]
    report_path = Path(entry["path"])
    report = json.loads(report_path.read_text())
    report["rounds"][0]["auditor_report"] = AUDIT
    report_path.write_text(json.dumps(report))
    bank = SkillBank(tmp_path/"bank"); stage(bank, skill())
    old = copy.deepcopy(bank.load(candidates=True)[0])
    agent = JsonAgent([{"failure_type": "POSITIVE_EXPERIENCE", "confidence": .9,
        "root_cause": "Existing package availability makes installation unnecessary",
        "evidence": ["launch service package"], "positive_updates": [{"skill_id": old["skill_id"],
        "counterexample": "A trusted readiness report already confirms the needed runtime packages."}]}])
    await generate(manifest, bank, agent)
    new = bank.load(candidates=True)[0]
    assert len(new["counterexamples"]) == len(old["counterexamples"])+1
    assert new["preconditions_to_check"] == old["preconditions_to_check"]
    assert not bank.load()


@pytest.mark.asyncio
async def test_manager_claim_cannot_prove_missing_environment_resource():
    no_observation = replace(snapshot(), environment_observations=())
    fabricated = verdict(quote="launch service package")
    fabricated["issues"][0]["type"] = "MISSING_PRECONDITION"
    result, trace = await FeasibilityReflector(ReflectorConfig(enabled=True, mode="gate"),
        JsonAgent([fabricated]), skills=[skill(status="active")]).assess(no_observation)
    assert result.decision == "UNKNOWN" and trace["runtime_error"] == "ValueError"
    from lh_harness.feasibility.offline import observable_records
    records = '\n'.join([json.dumps({"type": "assistant", "content": [
        {"type": "thinking", "thinking": "PRIVATE_THOUGHT"},
        {"type": "tool_use", "name": "inspect", "input": {"path": "observed"}}]})])
    sanitized = observable_records(records)
    assert "PRIVATE_THOUGHT" not in sanitized and "inspect" in sanitized
