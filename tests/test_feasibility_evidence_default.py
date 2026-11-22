from __future__ import annotations

import pytest

from lh_harness.feasibility import FeasibilityReflector, ReflectorConfig
from lh_harness.feasibility.integration import disposition
from tests.test_feasibility import JsonAgent, skill, snapshot, verdict, make_manifest


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["shadow", "gate"])
@pytest.mark.parametrize("decision", ["ACCEPT", "REVISE", "BLOCK", "UNKNOWN"])
@pytest.mark.parametrize("bank_kind", ["empty", "irrelevant", "candidate_only"])
async def test_no_skill_is_not_an_automatic_unknown(mode, decision, bank_kind):
    skills = []
    if bank_kind == "irrelevant":
        skills = [{**skill(status="active"), "applicability_conditions": ["polytope vertex enumeration"]}]
    elif bank_kind == "candidate_only":
        skills = [skill()]
    model = JsonAgent([verdict(decision)])
    config = ReflectorConfig(enabled=True, mode=mode)
    result, trace = await FeasibilityReflector(config, model, skills=skills).assess(snapshot())
    assert result.decision == decision and len(model.prompts) == trace["llm_calls"] == 1
    assert result.retrieved_skills == [] and trace["used_skill_guidance"] is False
    if mode == "shadow":
        assert disposition(config, result, trace, 0) == "dispatch"
    else:
        assert disposition(config, result, trace, 0) == {"ACCEPT": "dispatch", "REVISE": "revise", "BLOCK": "block", "UNKNOWN": "dispatch"}[decision]


@pytest.mark.asyncio
async def test_legacy_false_flag_cannot_disable_evidence_assessment():
    config = ReflectorConfig.from_dict({"enabled": True, "mode": "gate", "assess_without_skills": False})
    model = JsonAgent([verdict("REVISE")])
    result, _ = await FeasibilityReflector(config, model, skills=[]).assess(snapshot())
    assert result.decision == "REVISE" and len(model.prompts) == 1


@pytest.mark.asyncio
async def test_broken_bank_does_not_disable_evidence_judgment(tmp_path):
    directory = tmp_path/"bank/skills"; directory.mkdir(parents=True)
    (directory/"FS_bad.json").write_text("not valid JSON")
    config = ReflectorConfig(enabled=True, mode="gate", skill_bank_path=str(directory.parent))
    model = JsonAgent([verdict("REVISE")])
    result, trace = await FeasibilityReflector(config, model).assess(snapshot())
    assert result.decision == "REVISE" and trace["skill_retrieval_error"] == "JSONDecodeError"
    assert "runtime_error" not in trace and disposition(config, result, trace, 0) == "revise"


@pytest.mark.asyncio
async def test_large_optional_skill_falls_back_to_evidence_within_budget():
    selected = {**skill(status="active"), "description": "Long optional historical advice "*300}
    model = JsonAgent([verdict("ACCEPT")])
    config = ReflectorConfig(enabled=True, mode="gate", max_input_chars=2000)
    result, trace = await FeasibilityReflector(config, model, skills=[selected]).assess(snapshot())
    assert result.decision == "ACCEPT" and result.retrieved_skills == []
    assert trace["used_skill_guidance"] is False and trace["omitted_skills"] == [selected["skill_id"]]
    assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_disabled_stays_no_call_and_bad_evidence_or_model_failure_stays_unknown():
    model = JsonAgent([])
    result, trace = await FeasibilityReflector(ReflectorConfig(), model, skills=[]).assess(snapshot())
    assert not model.prompts and trace["llm_calls"] == 0 and result.uncertainty == "disabled"
    for reply in [verdict("UNKNOWN"), RuntimeError("unavailable"), verdict(quote="invented resource observation")]:
        config = ReflectorConfig(enabled=True, mode="gate")
        result, trace = await FeasibilityReflector(config, JsonAgent([reply]), skills=[]).assess(snapshot())
        assert result.decision == "UNKNOWN" and disposition(config, result, trace, 0) == "dispatch"


@pytest.mark.asyncio
async def test_evidence_only_success_not_credited_to_unselected_candidate(tmp_path):
    from lh_harness.feasibility.bank import SkillBank, stage
    from lh_harness.feasibility.offline import validate
    manifest = make_manifest(tmp_path, [{"split": "validation", "labels": {"1": {
        "feasibility_defect": True, "evidence": ["runtime package missing"]}}} for _ in range(3)])
    bank = SkillBank(tmp_path/"bank")
    stage(bank, {**skill(), "applicability_conditions": ["polytope vertex enumeration"]})
    model = JsonAgent([verdict(quote="launch service package") for _ in range(3)])
    result = await validate(manifest, bank, model)
    report = result["candidates"][0]
    assert len(model.prompts) == 3 and report["independent_support"] == 0
    assert report["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0} and not report["promote"]
    assert all(r["candidate_used"] is False for r in report["records"])
