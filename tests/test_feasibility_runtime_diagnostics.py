import json
from dataclasses import replace

import pytest

from lh_harness.feasibility import FeasibilityReflector, ReflectorConfig
from lh_harness.feasibility.context import prepare_context
from lh_harness.feasibility.integration import disposition
from lh_harness.feasibility.model import parse_json
from tests.test_feasibility import JsonAgent, snapshot, verdict


@pytest.mark.asyncio
async def test_invalid_evidence_has_field_diagnostic_and_one_bounded_repair():
    bad = verdict("ACCEPT")
    bad["check_evidence"] = {"Goal Alignment": ["A paraphrase not in the input"]}
    model = JsonAgent([bad, verdict("ACCEPT")])
    result, trace = await FeasibilityReflector(ReflectorConfig(enabled=True, mode="gate"), model, skills=[]).assess(snapshot())
    assert result.decision == "ACCEPT" and trace["validation_repaired"] is True
    assert trace["llm_calls"] == 2 and trace["input_tokens"] == 24 and trace["output_tokens"] == 16
    assert "check_evidence[Goal Alignment]" in trace["model_attempts"][0]["validation_error"]
    assert trace["model_attempts"][0]["error_stage"] == "assessment_validation"
    assert "runtime_error" not in trace and "error_stage" not in trace
    assert "ONE FORMAT/EVIDENCE REPAIR ATTEMPT" in model.prompts[1]


@pytest.mark.asyncio
async def test_repaired_block_uses_normal_gate_policy_not_error_fallback():
    model = JsonAgent([{"decision": "WRONG"}, verdict("BLOCK")])
    config = ReflectorConfig(enabled=True, mode="gate")
    result, trace = await FeasibilityReflector(config, model, skills=[]).assess(snapshot())
    assert result.decision == "BLOCK" and trace["validation_repaired"]
    assert disposition(config, result, trace, 0) == "block"


@pytest.mark.asyncio
async def test_two_bad_responses_stay_unknown_with_cost_and_diagnostic():
    invalid = verdict(quote="An invented resource absence")
    model = JsonAgent([invalid, invalid])
    config = ReflectorConfig(enabled=True, mode="gate")
    result, trace = await FeasibilityReflector(config, model, skills=[]).assess(snapshot())
    assert result.decision == "UNKNOWN" and result.uncertainty == "assessment_validation_failed"
    assert trace["llm_calls"] == len(model.prompts) == 2
    assert trace["runtime_error"] == "ValueError" and trace["error_stage"] == "assessment_validation"
    assert "issues[0].evidence" in trace["validation_error"]
    assert disposition(config, result, trace, 0) == "dispatch"


@pytest.mark.asyncio
async def test_zero_repair_budget_and_provider_exception_are_not_leaked():
    config = ReflectorConfig(enabled=True, mode="shadow", max_validation_repairs=0)
    model = JsonAgent([{"decision": "WRONG"}])
    result, trace = await FeasibilityReflector(config, model, skills=[]).assess(snapshot())
    assert result.decision == "UNKNOWN" and trace["llm_calls"] == 1
    assert trace["validation_error"] == "invalid reflector decision"
    result, trace = await FeasibilityReflector(config, JsonAgent([ValueError("KEY=DO_NOT_LEAK")]), skills=[]).assess(snapshot())
    assert result.decision == "UNKNOWN" and trace["error_stage"] == "model_call"
    assert "DO_NOT_LEAK" not in json.dumps(trace)


def test_context_dedup_keeps_full_contract_user_task_and_all_previous_audits():
    contract = "MUST PRESERVE EXACT FILE AND NUMERIC CONSTRAINTS. "*150
    state = "Unverified Manager state; trust cited audits only. "*50
    source = replace(snapshot(), original_task="Original task exact requirements. "*300,
        contract=contract, manager_state=state,
        manager_plan="Task Contract:\n"+contract+"\nTask State:\n"+state+"\nSubtask: launch service package",
        verified_auditor_reports=({"round": 1, "report": "Prior actual audit facts. "*200},))
    view, packed, ids, trace = prepare_context(source, [], ReflectorConfig().max_input_chars)
    assert view.original_task == source.original_task and view.contract == source.contract
    assert view.manager_state == source.manager_state and view.verified_auditor_reports == source.verified_auditor_reports
    assert "Subtask: launch service package" in view.manager_plan
    assert "snapshot.contract" in view.manager_plan and "snapshot.manager_state" in view.manager_plan
    assert trace["input_chars"] < trace["raw_input_chars"]
    assert json.loads(packed)["snapshot"]["contract"] == contract


@pytest.mark.asyncio
async def test_snapshot_too_large_reports_limit_and_does_not_truncate_constraints():
    source = replace(snapshot(), original_task="Exact acceptance constraint. "*200)
    model = JsonAgent([])
    result, trace = await FeasibilityReflector(ReflectorConfig(enabled=True, mode="shadow", max_input_chars=2000), model, skills=[]).assess(source)
    assert result.uncertainty == "input_budget_exceeded" and not model.prompts
    assert trace["error_stage"] == "input_budget" and trace["input_chars"] > trace["input_char_limit"] == 2000


def test_json_fences_whitespace_and_non_paraphrased_evidence():
    from lh_harness.feasibility.schema import Assessment
    data = verdict("ACCEPT")
    data["check_evidence"] = {"Goal Alignment": ["launch  service\npackage"]}
    result = Assessment.parse(data, snapshot(), [])
    assert result.decision == "ACCEPT"
    assert parse_json('```\n{"decision":"UNKNOWN"}\n```')["decision"] == "UNKNOWN"
    assert parse_json('```JSON\n{"decision":"UNKNOWN"}\n```')["decision"] == "UNKNOWN"
