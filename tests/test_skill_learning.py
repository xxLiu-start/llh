from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from lh_harness.skill_learning import (
    SkillBank, SkillSession, accept_candidate, compact_trajectory, render_skill_context,
    commit_validated_candidate, record_selected_skill_outcomes, stage_skill_candidate,
    select_trajectory_evidence, trajectory_round_index,
    parse_json_result, split_assessment_artifact_refs, task_success, token_usage,
    validate_revision_preservation_review, validate_selection, validate_skill,
)
from lh_harness.skill_prompt_texts import ROUTER_PROMPT_EN
from lh_harness.types import EpisodeBudget, EpisodeResult, HarnessConfig


def skill(skill_id: str, *, conflict: list[str] | None = None) -> dict:
    view = {"input_use": "Check upstream evidence", "output_spec": "Report status and proof", "guidance": "Verify one step", "avoid": "Do not guess"}
    return {
        "skill_id": skill_id, "category_id": "evidence_handoff", "version": 1,
        "version_parent": None, "status": "validated", "content_language": "en",
        "benchmark_scope": "shared", "interaction_modes": ["gui", "cli", "hybrid"],
        "problem_mechanism": "Missing evidence at handoff", "subcases": [],
        "triggers": [], "exclusions": [], "roles": {r: view for r in ("manager", "executor", "auditor")},
        "compatible_with": [], "conflicts_with": conflict or [], "provenance": {}, "validation": {},
    }


def chinese_skill(skill_id: str) -> dict:
    result = skill(skill_id)
    view = {"input_use": "核对上游证据", "output_spec": "交付状态和证明", "guidance": "独立验证关键步骤", "avoid": "不要猜测输入"}
    result.update(content_language="zh", problem_mechanism="角色交接缺少证据")
    result["roles"] = {role: dict(view) for role in ("manager", "executor", "auditor")}
    return result


class FakeAgent:
    def __init__(self, replies: list[dict], extraction_reply: object | None = None):
        self.replies = replies
        self.prompts: list[str] = []
        self.extraction_prompts: list[str] = []
        self.extraction_reply = extraction_reply

    async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
        if "Extract reusable experience" in prompt or "提取可复用经验" in prompt:
            self.extraction_prompts.append(prompt)
            payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
            round_id = payload["all_rounds"][0]["round_id"]
            reply = self.extraction_reply if self.extraction_reply is not None else {
                "positive_experiences": [{"mechanism": "Observed reusable step", "strategy": "Use audited evidence",
                                          "evidence_refs": [round_id], "reusable": True, "confidence": "medium"}],
                "negative_experiences": [],
            }
            if isinstance(reply, Exception):
                raise reply
            raw = reply if isinstance(reply, str) else json.dumps(reply)
            return EpisodeResult(status="done", metadata={"assistant_visible_output": raw,
                                                          "usage": {"input_tokens": 20, "output_tokens": 10}})
        self.prompts.append(prompt)
        return EpisodeResult(status="done", metadata={"assistant_visible_output": json.dumps(self.replies.pop(0)), "usage": {"input_tokens": 20, "output_tokens": 10}})


@pytest.mark.parametrize("setting", ["skill_token_budget", "skill_reward_reference_tokens"])
def test_skill_budgets_must_be_positive(setting: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HarnessConfig(**{setting: 0})
    if setting == "skill_token_budget":
        with pytest.raises(ValueError, match="positive integer"):
            SkillSession(SkillBank(tmp_path / "bank.json"), "en", FakeAgent([]), None,
                         EpisodeBudget(), tmp_path / "run", token_budget=0)


def test_identical_repeated_json_output_is_accepted() -> None:
    proposal = {"action": "CREATE_NEW_SKILL", "candidate_spec": {"category_id": "example"}}
    raw = json.dumps(proposal) + "\n\n" + json.dumps(proposal)
    assert parse_json_result(EpisodeResult(status="done", metadata={"assistant_visible_output": raw})) == proposal


def test_diagnosis_trajectory_keeps_routing_evidence_without_repeating_full_briefs() -> None:
    report = {"task": "original task", "status": "incomplete", "rounds": [
        {"round_index": index, "plan_text": "plan " + "P" * 4000, "executor_output": "result " + "E" * 4000}
        for index in range(1, 4)
    ]}
    records = [
        {"round": index, "role": "manager", "router_input": {
            "eligible_skill_cards": [{"skill_id": "existing", "version": 2, "category_id": "example"}],
            "routing_brief": {"task_goal": "T" * 20000, "current_state": "S" * 20000},
        }, "router_output": {"selected_skills": []}, "selected_skills": [], "injected_context": ""}
        for index in range(1, 4)
    ]
    compact = compact_trajectory(report, records)
    assert [item["round_id"] for item in compact["rounds"]] == ["round_001", "round_002", "round_003"]
    assert compact["router_inputs_and_outputs"][0]["eligible_skill_cards"][0]["skill_id"] == "existing"
    assert compact["router_inputs_and_outputs"][0]["router_output"] == {"selected_skills": []}
    assert "task" not in compact["final_report"]
    assert len(json.dumps(compact)) < 25000


def test_learning_prompt_uses_round_index_and_selected_evidence(tmp_path: Path) -> None:
    rounds = [
        {"round_index": index, "plan_text": f"UNIQUE_PLAN_{index:02d}",
         "executor_output": f"work {index}", "auditor_report": "Status: incomplete\nIntegrity: clean\nContract audit: needs_revision"}
        for index in range(1, 13)
    ]
    compact = compact_trajectory({"task": "original task", "status": "incomplete", "rounds": rounds}, [])
    assert len(trajectory_round_index(compact["rounds"])) == 12
    selected = select_trajectory_evidence(compact["rounds"])
    assert len(selected) == 6 and selected[0]["round_id"] == "round_001"
    assert "round_004" in {item["round_id"] for item in select_trajectory_evidence(
        compact["rounds"], evidence_refs=["round_004"], limit=8,
    )}
    cited = [f"round_{index:03d}" for index in range(2, 10)]
    assert {item["round_id"] for item in select_trajectory_evidence(
        compact["rounds"], evidence_refs=cited, limit=8,
    )} == set(cited)

    agent = FakeAgent([{"action": "NO_CHANGE", "failure_mechanism": "Task-specific failure"}])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    result = asyncio.run(session.finish({"task": "original task", "status": "incomplete", "rounds": rounds}))
    assert result["action"] == "NO_CHANGE"
    payload = json.loads(agent.prompts[0].split("INPUT JSON:\n", 1)[1])
    assert len(payload["trajectory_index"]) == 12
    assert len(payload["relevant_trajectory_evidence"]) == 6
    assert "UNIQUE_PLAN_04" not in agent.prompts[0]


def test_extraction_covers_every_round_and_passes_both_experience_types_to_evolution(tmp_path: Path) -> None:
    extracted = {
        "positive_experiences": [{"mechanism": "Checkpointing preserves progress",
                                  "strategy": "Save verified intermediate state",
                                  "evidence_refs": ["round_001"], "reusable": True, "confidence": "high"}],
        "negative_experiences": [{"mechanism": "Unverified input scope caused a wrong assumption",
                                  "avoidance_strategy": "Audit input scope before implementation",
                                  "evidence_refs": ["round_003"], "reusable": True, "confidence": "medium"}],
    }
    agent = FakeAgent([{"action": "NO_CHANGE", "failure_mechanism": "No Skill update is justified"}],
                      extraction_reply=extracted)
    report = {"task": "task", "status": "incomplete", "rounds": [
        {"round_index": index, "plan_text": f"PLAN_{index}",
         "executor_output": f"EXECUTION_{index}", "auditor_report": f"AUDIT_{index}"}
        for index in range(1, 4)
    ]}
    result = asyncio.run(SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent,
                                       None, EpisodeBudget(), tmp_path / "run").finish(report))
    extraction_input = json.loads(agent.extraction_prompts[0].split("INPUT JSON:\n", 1)[1])
    decision_input = json.loads(agent.prompts[0].split("INPUT JSON:\n", 1)[1])
    assert [item["round_id"] for item in extraction_input["all_rounds"]] == [
        "round_001", "round_002", "round_003"]
    assert "EXECUTION_2" in agent.extraction_prompts[0]
    assert decision_input["extracted_experiences"] == extracted
    assert result["extracted_experiences"] == extracted
    assert result["experience_extraction_attempted"] is True
    assert result["diagnosis_attempted"] is True


def test_negative_experience_strategy_alias_reaches_diagnosis(tmp_path: Path) -> None:
    extracted = {"positive_experiences": [], "negative_experiences": [{
        "mechanism": "Observed parser failure", "strategy": "Check documented cases first",
        "evidence_refs": ["round_001"], "reusable": True, "confidence": "high",
    }]}
    agent = FakeAgent([{"action": "NO_CHANGE", "failure_mechanism": "Existing guidance is sufficient"}],
                      extraction_reply=extracted)
    result = asyncio.run(SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent,
                                       None, EpisodeBudget(), tmp_path / "run").finish({
        "task": "parser", "status": "incomplete",
        "rounds": [{"round_index": 1, "executor_output": "Parser failed documented case"}],
    }))
    assert result["action"] == "NO_CHANGE"
    assert result["diagnosis_attempted"] is True
    assert result["extracted_experiences"]["negative_experiences"][0]["avoidance_strategy"] == "Check documented cases first"
    assert "strategy" not in result["extracted_experiences"]["negative_experiences"][0]
    assert result["extracted_experiences_raw"] == extracted
    assert result["experience_normalizations"][0]["normalized_to"] == "avoidance_strategy"


def test_artifact_refs_require_literal_report_evidence_and_keep_round_refs() -> None:
    report = {"rounds": [{"round_index": 1, "executor_output": "Wrote /tmp/result.txt"}]}
    assessment = {"evidence_refs": ["round_001/executor_output", "/tmp/result.txt", "/tmp/imagined.txt"]}
    normalized, artifacts = split_assessment_artifact_refs(assessment, report, {"round_001"})
    assert normalized["evidence_refs"] == ["round_001/executor_output"]
    assert normalized["artifact_refs"] == artifacts == ["/tmp/result.txt"]
    assert normalized["unverified_artifact_refs"] == ["/tmp/imagined.txt"]
    assert assessment["evidence_refs"] == ["round_001/executor_output", "/tmp/result.txt", "/tmp/imagined.txt"]


def test_learning_output_cannot_self_verify_an_artifact_path() -> None:
    assessment = {"evidence_refs": ["round_001", "/tmp/imagined.txt"]}
    report = {"rounds": [{"round_index": 1, "plan_text": "Tried the task"}],
              "skill_learning": {"assessment": assessment}}
    normalized, observed = split_assessment_artifact_refs(assessment, report, {"round_001"})
    assert observed == []
    assert normalized["unverified_artifact_refs"] == ["/tmp/imagined.txt"]


def test_observed_artifact_path_does_not_block_skill_builder(tmp_path: Path) -> None:
    candidate = {**skill("new_capability"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "Missing parser guidance",
         "evidence_refs": ["round_001/executor_output", "/tmp/result.txt"]},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    result = asyncio.run(SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent,
                                       None, EpisodeBudget(), tmp_path / "run").finish({
        "task": "parser", "status": "incomplete",
        "rounds": [{"round_index": 1, "executor_output": "Wrote /tmp/result.txt"}],
    }))
    assert result["validation"] == "pending_replay"
    assert result["assessment_artifact_refs"] == ["/tmp/result.txt"]
    assert result["assessment_raw"]["evidence_refs"] == ["round_001/executor_output", "/tmp/result.txt"]
    assert result["assessment"]["evidence_refs"] == ["round_001/executor_output"]


@pytest.mark.parametrize("extraction_reply, expected_stage", [
    (OSError("experience provider unavailable"), "experience_extraction"),
    ({"positive_experiences": [{"mechanism": "Fabricated", "strategy": "Guess",
                                "evidence_refs": ["round_099"], "reusable": True,
                                "confidence": "high"}], "negative_experiences": []}, "experience_validation"),
])
def test_invalid_extraction_stops_before_evolution(tmp_path: Path, extraction_reply, expected_stage: str) -> None:
    agent = FakeAgent([{"action": "CREATE_NEW_SKILL"}], extraction_reply=extraction_reply)
    report = {"task": "task", "status": "incomplete", "rounds": [
        {"round_index": 1, "plan_text": "Actual execution"}]}
    result = asyncio.run(SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent,
                                       None, EpisodeBudget(), tmp_path / "run").finish(report))
    assert result["error_stage"] == expected_stage
    assert result["diagnosis_attempted"] is False
    assert result["action"] is None
    assert not agent.prompts


def test_no_reliable_experience_blocks_ungrounded_skill_creation(tmp_path: Path) -> None:
    agent = FakeAgent([{"action": "CREATE_NEW_SKILL", "failure_mechanism": "Claimed gap",
                        "evidence_refs": ["round_001"]}],
                      extraction_reply={"positive_experiences": [], "negative_experiences": []})
    report = {"task": "task", "status": "incomplete", "rounds": [
        {"round_index": 1, "plan_text": "Attempted task"}]}
    result = asyncio.run(SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent,
                                       None, EpisodeBudget(), tmp_path / "run").finish(report))
    assert result["model_action"] == "CREATE_NEW_SKILL"
    assert result["action"] == "INSUFFICIENT_EVIDENCE"
    assert result["policy_override"]["reason"] == "no_reliable_extracted_experience"
    assert not (tmp_path / "run" / "skill_candidate.json").exists()
    assert len(agent.prompts) == 1


def test_success_experiences_can_propose_new_and_revised_skills_together(tmp_path: Path) -> None:
    old = skill("existing")
    new = {**skill("missing_capability"), "status": "candidate", "validation": {}}
    revised = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    experiences = {
        "positive_experiences": [{"mechanism": "Reusable new strategy", "strategy": "Preserve verified progress",
                                  "evidence_refs": ["round_001"], "reusable": True, "confidence": "high"}],
        "negative_experiences": [{"mechanism": "Old handoff omitted a check",
                                  "avoidance_strategy": "Include audit evidence in handoff",
                                  "evidence_refs": ["round_001"], "reusable": True, "confidence": "high"}],
    }
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "novelty_reason": "New strategy and defective old guidance",
         "evidence_refs": ["round_001"], "candidate_specs": [{"category_id": "missing_capability"}],
         "revision_targets": [{"target_skill_id": "existing", "failure_mechanism": "Missing audit check",
                               "evidence_refs": ["round_001"]}]},
        {"action": "CREATE_NEW_SKILL", "skills": [new, revised]},
    ], extraction_reply=experiences)
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager",
                            "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "selected_skills": [{"skill_id": "existing", "version": 1}],
                            "injected_context": "Skill: existing@1"})
    report = {"task": "task", "status": "complete", "completion_satisfied": True,
              "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
              "rounds_run": 1, "rounds": [{"round_index": 1, "plan_text": "Completed task"}]}
    result = asyncio.run(session.finish(report))
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["revision_target_ids"] == ["existing"]
    assert {item["skill_id"] for item in result["candidates"]} == {"existing", "missing_capability"}
    assert len(agent.prompts) == 2
    assert json.loads(agent.prompts[0].split("INPUT JSON:\n", 1)[1])["extracted_experiences"] == experiences


def test_success_experience_can_propose_supported_router_repair(tmp_path: Path) -> None:
    old = skill("existing")
    experiences = {"positive_experiences": [], "negative_experiences": [{
        "mechanism": "Router omitted a matching Skill", "avoidance_strategy": "Select available matching guidance",
        "evidence_refs": ["round_001"], "reusable": True, "confidence": "high",
    }]}
    agent = FakeAgent([
        {"action": "UPDATE_ROUTER_PROMPT", "success_mechanism": "Task succeeded after recovery",
         "evidence_refs": ["round_001"], "router_error_type": "omission", "target_skill_id": "existing"},
        {"action": "UPDATE_ROUTER_PROMPT", "candidate_router_prompt": "Select matching Skills using routing evidence."},
    ], extraction_reply=experiences)
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager",
                            "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "router_output": {"selected_skills": []}, "selected_skills": [],
                            "injected_context": ""})
    report = {"task": "task", "status": "complete", "completion_satisfied": True,
              "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
              "rounds_run": 1, "rounds": [{"round_index": 1, "plan_text": "Recovered"}]}
    result = asyncio.run(session.finish(report))
    assert result["action"] == "UPDATE_ROUTER_PROMPT"
    assert result["validation"] == "pending_replay"
    assert bank.router_prompts["en"] != result["candidate"]


def test_skill_update_hydrates_a_round_shown_only_in_index(tmp_path: Path) -> None:
    rounds = [
        {"round_index": index, "plan_text": f"plan {index}", "executor_output": f"work {index}"}
        for index in range(1, 13)
    ]
    candidate = {**skill("new_capability"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "failure seen only in index",
         "evidence_refs": ["round_004/auditor_report.txt"]},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    result = asyncio.run(session.finish({"task": "original task", "status": "incomplete", "rounds": rounds}))
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["builder_evidence_round_ids"] == ["round_004"]
    assert result["validation"] == "pending_replay"
    assert len(agent.prompts) == 2
    assert "plan 4" not in agent.prompts[0]
    assert '"round_id": "round_004"' in agent.prompts[1]
    assert '"manager_plan": "plan 4"' in agent.prompts[1]


def test_skill_update_accepts_report_and_routing_evidence_with_recorded_rounds(tmp_path: Path) -> None:
    candidate = {**skill("new_capability"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
    refs = [
        "round_001:executor_output: attempted task",
        "final_report: abort_reason=token_budget_exhausted",
        "router_inputs_and_outputs: no_eligible_skill_cards",
    ]
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "No applicable Skill", "evidence_refs": refs},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({
        "round": 1, "role": "manager", "router_input": {"eligible_skill_cards": []},
        "router_skipped": "no_eligible_skill_cards", "selected_skills": [], "injected_context": "",
    })
    report = {"task": "original task", "status": "incomplete", "abort_reason": "token_budget_exhausted",
              "rounds": [{"round_index": 1, "executor_output": "attempted task"}]}
    result = asyncio.run(session.finish(report))
    assert result["builder_evidence_round_ids"] == ["round_001"]
    assert result["validation"] == "pending_replay"
    assert len(agent.prompts) == 2
    assert json.loads(agent.prompts[1].split("INPUT JSON:\n", 1)[1])["diagnosis_or_success_summary"]["evidence_refs"] == refs
    assert result["candidate"]["provenance"]["evidence_refs"] == refs
    assert result["candidate"]["provenance"]["task_id"]
    assert json.loads((tmp_path / "run" / "skill_candidate.json").read_text())["candidate"]["provenance"]["evidence_refs"] == refs


def test_skill_update_keeps_report_evidence_when_round_citations_are_capped(tmp_path: Path) -> None:
    candidate = {**skill("new_capability"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
    refs = [f"round_{index:03d}" for index in range(1, 10)] + ["final_report:status=incomplete"]
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "Repeated failure", "evidence_refs": refs},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    rounds = [{"round_index": index, "plan_text": f"Attempt {index}"} for index in range(1, 10)]
    result = asyncio.run(session.finish({"task": "original task", "status": "incomplete", "rounds": rounds}))
    assert result["validation"] == "pending_replay"
    assert result["evidence_refs_omitted_from_builder"] == ["round_009"]
    builder_refs = json.loads(agent.prompts[1].split("INPUT JSON:\n", 1)[1])["diagnosis_or_success_summary"]["evidence_refs"]
    assert builder_refs == refs[:8] + ["final_report:status=incomplete"]


def test_new_skill_uses_recorded_run_metadata_and_stays_eligible(tmp_path: Path) -> None:
    candidate = skill("new_capability")
    candidate.update(version=9, version_parent=8, status="validated", benchmark_scope="shared",
                     interaction_modes=["gui"])
    candidate.pop("provenance")
    candidate.pop("validation")
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "Missing reusable guidance",
         "evidence_refs": ["round_001"]},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(
        SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run",
        benchmark_id="terminal_bench_2_1", interaction_mode="cli",
    )
    result = asyncio.run(session.finish({
        "task": "original task", "status": "incomplete",
        "rounds": [{"round_index": 1, "executor_output": "Attempted task"}],
    }))
    assert result["validation"] == "pending_replay"
    new_skill = result["candidate"]
    assert (new_skill["version"], new_skill["version_parent"], new_skill["status"]) == (1, None, "candidate")
    assert new_skill["benchmark_scope"] == "terminal_bench_2_1"
    assert new_skill["interaction_modes"] == ["cli"]
    assert new_skill["provenance"]["evidence_refs"] == ["round_001"]
    assert new_skill["validation"] == {}
    assert session.bank.eligible("cli_executor", "en", new_skill,
                                 benchmark_id="terminal_bench_2_1", interaction_mode="cli")
    assert json.loads(agent.prompts[1].split("INPUT JSON:\n", 1)[1])["skill_token_budget"] == 2000


def test_oversized_skill_is_rejected_before_it_can_be_committed(tmp_path: Path) -> None:
    candidate = {**skill("oversized"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
    candidate["roles"] = {role: {**view, "guidance": "A" * 5000}
                          for role, view in candidate["roles"].items()}
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "Missing guidance",
         "evidence_refs": ["round_001"]},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(),
                           tmp_path / "run", token_budget=200)
    result = asyncio.run(session.finish({
        "task": "original task", "status": "incomplete",
        "rounds": [{"round_index": 1, "executor_output": "Attempted task"}],
    }))
    assert result["error_stage"] == "candidate_validation"
    assert "per-role token budget" in result["learning_error"]
    assert not (tmp_path / "run" / "skill_candidate.json").exists()


def test_skill_update_still_rejects_a_nonexistent_round(tmp_path: Path) -> None:
    agent = FakeAgent([{
        "action": "CREATE_NEW_SKILL", "failure_mechanism": "unsupported claim",
        "evidence_refs": ["round_999"],
    }])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    result = asyncio.run(session.finish({"task": "original task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Attempted task"}]}))
    assert result["error_stage"] == "evidence_validation"
    assert result["invalid_evidence_refs"] == ["round_999"]
    assert len(agent.prompts) == 1


def test_skill_update_rejects_unknown_evidence_section(tmp_path: Path) -> None:
    agent = FakeAgent([{
        "action": "CREATE_NEW_SKILL", "failure_mechanism": "unsupported claim",
        "evidence_refs": ["round_001", "unrecorded_summary: claim"],
    }])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    result = asyncio.run(session.finish({"task": "original task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Attempted task"}]}))
    assert result["error_stage"] == "evidence_validation"
    assert result["invalid_evidence_refs"] == ["unrecorded_summary: claim"]
    assert len(agent.prompts) == 1


def test_new_skill_requires_a_recorded_round_even_with_report_evidence(tmp_path: Path) -> None:
    agent = FakeAgent([{
        "action": "CREATE_NEW_SKILL", "failure_mechanism": "unsupported claim",
        "evidence_refs": ["final_report: status=incomplete"],
    }])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    result = asyncio.run(session.finish({"task": "original task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Attempted task"}]}))
    assert result["error_stage"] == "evidence_validation"
    assert result["learning_error"] == "new Skill needs a recorded round reference"
    assert len(agent.prompts) == 1


@pytest.mark.parametrize("suffix", ['{"action": "NO_NEW_SKILL"}', "unexpected trailing text"])
def test_conflicting_or_extra_json_output_is_rejected(suffix: str) -> None:
    raw = '{"action": "CREATE_NEW_SKILL"}\n' + suffix
    with pytest.raises(ValueError, match="Extra data"):
        parse_json_result(EpisodeResult(status="done", metadata={"assistant_visible_output": raw}))


def test_router_selects_multiple_and_logs_exact_inputs(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path / "bank.json", skills=[skill("first"), skill("second")])
    bank.save()
    agent = FakeAgent([{"active_issue_atoms": [], "selected_skills": [{"skill_id": "first", "version": 1}, {"skill_id": "second", "version": 1}], "rejected_candidates": [], "uncovered_issue_ids": [], "selection_rationale": "complementary"}])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    context = asyncio.run(session.route("manager", "Original manager prompt", 1))
    assert "Skill: first@1" in context and "Skill: second@1" in context
    assert "Output specification" in context
    assert session.records[0]["role_base_prompt"] == "Original manager prompt"
    assert "role_base_prompt" not in session.records[0]["router_input"]
    assert session.records[0]["injected_context"] == context
    assert "role_view" not in session.records[0]["router_input"]["eligible_skill_cards"][0]
    card = session.records[0]["router_input"]["eligible_skill_cards"][0]
    assert "Missing evidence at handoff" in card["routing_summary"]
    assert isinstance(card["role_token_estimate"], int) and card["role_token_estimate"] > 0
    assert "input_use" not in card and "output_spec" not in card
    assert "guidance" not in card and "problem_mechanism" not in card
    assert "Check upstream evidence" not in agent.prompts[0]
    assert "Report status and proof" not in agent.prompts[0]
    assert "routing_summary" in ROUTER_PROMPT_EN
    assert "output_handoff" not in ROUTER_PROMPT_EN
    assert "Eligible Skill content may be English or Chinese" in ROUTER_PROMPT_EN
    assert (tmp_path / "run" / "skill_routes.json").exists()


def test_router_can_validate_a_batch_of_new_and_revised_skills(tmp_path: Path) -> None:
    old = skill("revised")
    revised = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    added = {**skill("added"), "status": "candidate", "validation": {}}
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    agent = FakeAgent([{"selected_skills": [
        {"skill_id": "revised", "version": 2}, {"skill_id": "added", "version": 1},
    ]}])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run",
                           candidate=[revised, added])
    context = asyncio.run(session.route("manager", "Manager task", 1))
    assert "Skill: revised@2" in context and "Skill: added@1" in context
    assert "Skill: revised@1" not in context
    assert session.records[0]["router_input"]["validation_target_skill_ids"] == ["revised", "added"]


def test_router_receives_only_summary_then_injects_selected_role_view(tmp_path: Path) -> None:
    selected = skill("selected")
    selected["routing_summary"] = "Use for audited evidence handoffs; exclude tasks without a handoff."
    selected["roles"] = {
        "manager": {**selected["roles"]["manager"], "guidance": "MANAGER_PRIVATE_GUIDANCE"},
        "executor": {**selected["roles"]["executor"], "guidance": "EXECUTOR_PRIVATE_GUIDANCE"},
        "auditor": {**selected["roles"]["auditor"], "guidance": "AUDITOR_PRIVATE_GUIDANCE"},
    }
    unselected = skill("unselected")
    unselected["roles"] = {role: {**view, "guidance": "UNSELECTED_PRIVATE_GUIDANCE"}
                           for role, view in unselected["roles"].items()}
    bank = SkillBank(tmp_path / "bank.json", skills=[selected, unselected])
    agent = FakeAgent([{"selected_skills": [{"skill_id": "selected", "version": 1}]}])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")

    context = asyncio.run(session.route("manager", "Manager task", 1))
    assert "Use for audited evidence handoffs" in agent.prompts[0]
    assert "PRIVATE_GUIDANCE" not in agent.prompts[0]
    assert "MANAGER_PRIVATE_GUIDANCE" in context
    assert "EXECUTOR_PRIVATE_GUIDANCE" not in context
    assert "AUDITOR_PRIVATE_GUIDANCE" not in context
    assert "UNSELECTED_PRIVATE_GUIDANCE" not in context
    compact = compact_trajectory({"status": "incomplete", "rounds": []}, session.records)
    assert compact["router_inputs_and_outputs"][0]["eligible_skill_cards"][0]["routing_summary"] == selected["routing_summary"]


def test_legacy_skillbank_gets_bounded_routing_summary(tmp_path: Path) -> None:
    old = skill("legacy")
    old["problem_mechanism"] = "Missing evidence " * 100
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"skills": [old], "router_prompts": {}}), encoding="utf-8")

    bank = SkillBank.load(path)
    card = bank.eligible("manager", "en")[0]
    assert 0 < len(card["routing_summary"]) <= 320
    assert "input_use" not in card and "output_spec" not in card
    bank.save()
    assert json.loads(path.read_text(encoding="utf-8"))["skills"][0]["routing_summary"] == card["routing_summary"]


def test_only_known_old_default_router_prompt_is_migrated(tmp_path: Path, monkeypatch) -> None:
    from lh_harness import skill_learning

    old_default = "Old English Router prompt with input_use/output_spec"
    custom = "请使用我定制的路由规则。"
    monkeypatch.setattr(skill_learning, "OLD_DEFAULT_ROUTER_PROMPT_HASHES", {
        "en": hashlib.sha256(old_default.encode("utf-8")).hexdigest(),
    })
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"skills": [], "router_prompts": {
        "en": old_default, "zh": custom,
    }}, ensure_ascii=False), encoding="utf-8")

    bank = SkillBank.load(path)
    assert bank.router_prompts["en"] == ROUTER_PROMPT_EN
    assert bank.router_prompts["zh"] == custom
    assert json.loads(path.read_text(encoding="utf-8"))["router_prompts"]["en"] == old_default


def test_invalid_router_output_falls_back_without_injection(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path / "bank.json", skills=[skill("first")])
    agent = FakeAgent([{"selected_skills": [{"skill_id": "unknown", "version": 1}]}])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    assert asyncio.run(session.route("manager", "base", 1)) == ""
    assert "unknown Skill ID" in session.records[0]["router_error"]


def test_conflicts_and_candidate_injection_are_hard_checked() -> None:
    eligible = [skill("first", conflict=["second"]), skill("second")]
    eligible = [{**s, "role_view": s["roles"]["manager"]} for s in eligible]
    with pytest.raises(ValueError, match="conflicting"):
        validate_selection({"selected_skills": [{"skill_id": "first", "version": 1}, {"skill_id": "second", "version": 1}]}, eligible, budget=2000)
    with pytest.raises(ValueError, match="candidate was not selected"):
        validate_selection({"selected_skills": []}, eligible, budget=2000, target="first")
    assert not render_skill_context([], "en")


def test_candidate_revision_replaces_old_version_in_router_view(tmp_path: Path) -> None:
    old = skill("first")
    candidate = {**skill("first"), "version": 2, "version_parent": 1, "status": "candidate"}
    views = SkillBank(tmp_path / "bank.json", skills=[old]).eligible("manager", "en", candidate)
    assert [(view["skill_id"], view["version"]) for view in views] == [("first", 2)]


def test_scope_and_modality_filter_cards_before_routing(tmp_path: Path) -> None:
    terminal_cli = {**skill("terminal_cli"), "benchmark_scope": "terminal_bench_2_1", "interaction_modes": ["cli"]}
    terminal_cli["roles"] = {**terminal_cli["roles"], "executor": {**terminal_cli["roles"]["executor"], "output_spec": "CLI proof"}}
    osworld_gui = {**skill("osworld_gui"), "benchmark_scope": "osworld_v2", "interaction_modes": ["gui"]}
    shared_gui = {**skill("shared_gui"), "interaction_modes": ["gui"]}
    bank = SkillBank(tmp_path / "bank.json", skills=[terminal_cli, osworld_gui, shared_gui])
    cli_cards = bank.eligible("cli_executor", "en", benchmark_id="terminal_bench_2_1", interaction_mode="cli")
    assert [card["skill_id"] for card in cli_cards] == ["terminal_cli"]
    assert all("role_view" not in card for card in cli_cards)
    assert "routing_summary" in cli_cards[0]
    assert "output_spec" not in cli_cards[0]
    assert "guidance" not in cli_cards[0]
    manager_cards = bank.eligible("manager", "en", benchmark_id="terminal_bench_2_1", interaction_mode="hybrid")
    assert [card["skill_id"] for card in manager_cards] == ["terminal_cli", "shared_gui"]
    assert "output_spec" not in manager_cards[0]


def test_success_observation_does_not_verify_skill(tmp_path: Path) -> None:
    existing = skill("first")
    bank = SkillBank(tmp_path / "bank.json", skills=[existing])
    agent = FakeAgent([{"action": "NO_NEW_SKILL", "candidate_spec": None, "novelty_reason": "The existing Skill covers the mechanism"}])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run", dataset_id="training")
    session.records.append({"round": 1, "role": "manager", "selected_skills": [{"skill_id": "first", "version": 1}], "injected_context": "Skill: first@1"})
    report = {"task": "independent task", "status": "complete", "completion_satisfied": True, "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned", "rounds": []}
    result = asyncio.run(session.finish(report))
    assert result["action"] == "NO_NEW_SKILL"
    assert result["assessment_reason"] == "The existing Skill covers the mechanism"
    assert bank.skills[0]["status"] == "validated"
    assert len(bank.skills[0]["validation"]["observational_task_ids"]) == 1


def test_skill_workflow_supports_chinese_training_and_english_evaluation(tmp_path: Path) -> None:
    from lh_harness.role_prompts import (
        build_role_auditor_prompt, build_role_executor_prompt, build_role_manager_prompt,
    )
    from lh_harness.types import HarnessConfig

    validate_skill(skill("first"), language="en")
    with pytest.raises(ValueError, match="Chinese"):
        validate_skill({**skill("first"), "content_language": "zh"}, language="zh")
    translated = skill("first")
    translated["roles"]["manager"] = {**translated["roles"]["manager"], "guidance": "核验状态"}
    with pytest.raises(ValueError, match="English"):
        validate_skill(translated, language="en")
    assert HarnessConfig(skill_bank_path=str(tmp_path / "bank.json"), prompt_language="zh").prompt_language == "zh"
    assert SkillSession(SkillBank(tmp_path / "bank.json"), "zh", FakeAgent([]), None, EpisodeBudget(), tmp_path / "run").language == "zh"
    old_bank = tmp_path / "legacy_bank.json"
    old_bank.write_text(json.dumps({"skills": [], "router_prompts": {"en": "English router"}}))
    assert set(SkillBank.load(old_bank).router_prompts) == {"en", "zh"}
    for builder, kwargs in (
        (build_role_manager_prompt, {"task": "task", "rounds": [], "round_index": 1}),
        (build_role_executor_prompt, {"task": "task", "plan_text": "plan", "next_step": "cli"}),
        (build_role_auditor_prompt, {"task": "task", "plan_text": "plan", "executor_output": "done", "next_step": "cli"}),
    ):
        assert "Skill: first@1" in builder(**kwargs, language="zh", skill_context="Skill: first@1")


def test_chinese_skill_is_routed_during_english_evaluation(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path / "bank.json", skills=[chinese_skill("chinese_handoff")])
    bank.save()
    loaded = SkillBank.load(bank.path)
    agent = FakeAgent([{"selected_skills": [{"skill_id": "chinese_handoff", "version": 1}]}])
    session = SkillSession(loaded, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    context = asyncio.run(session.route("manager", "Original English manager prompt", 1))
    assert "核对上游证据" in context
    assert "交付状态和证明" not in agent.prompts[0]
    assert "角色交接缺少证据" in agent.prompts[0]
    assert session.records[0]["router_input"]["eligible_skill_cards"][0]["content_language"] == "zh"
    training_agent = FakeAgent([{"selected_skills": [{"skill_id": "chinese_handoff", "version": 1}]}])
    training_session = SkillSession(loaded, "zh", training_agent, None, EpisodeBudget(), tmp_path / "train_run")
    training_context = asyncio.run(training_session.route("manager", "中文任务管理 Prompt", 1))
    assert "角色专用 Skill 指导" in training_context
    assert "你是 LongHorizon-Harness 的 Skill Router" in training_agent.prompts[0]


def test_chinese_success_summary_builds_chinese_skill_candidate(tmp_path: Path) -> None:
    source = chinese_skill("new_mechanism")
    spec = {key: source[key] for key in ("category_id", "problem_mechanism", "subcases", "triggers", "exclusions", "roles")}
    spec["routing_summary"] = "交接证据不足时使用；无需交接的任务不适用。"
    agent = FakeAgent([{
        "action": "CREATE_NEW_SKILL", "candidate_spec": spec,
        "evidence_refs": ["round_001"], "novelty_reason": "可跨任务复用",
    }])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "zh", agent, None, EpisodeBudget(), tmp_path / "run")
    report = {
        "task": "完成任务", "status": "complete", "completion_satisfied": True,
        "latest_auditor_report": "状态: complete\n完整性: clean\n契约审计: aligned",
        "rounds": [], "rounds_run": 1,
    }
    result = asyncio.run(session.finish(report))
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["candidate"]["content_language"] == "zh"
    assert result["candidate"]["routing_summary"] == spec["routing_summary"]
    assert result["validation"] == "pending_replay"
    assert "总结已完成" in agent.prompts[0]
    assert json.loads((tmp_path / "run" / "skill_candidate.json").read_text())["candidate"]["content_language"] == "zh"


def test_chinese_failure_diagnosis_uses_chinese_builder(tmp_path: Path) -> None:
    candidate = chinese_skill("new_mechanism")
    candidate.update(status="candidate", benchmark_scope="generic", validation={},
                     routing_summary="交接证据缺失时使用；无交接需求时不适用。")
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "交接证据缺失", "evidence_refs": ["round_001"]},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "zh", agent, None, EpisodeBudget(), tmp_path / "run")
    report = {"task": "完成任务", "status": "incomplete", "rounds": [{"round_index": 1, "plan_text": "已尝试执行"}]}
    result = asyncio.run(session.finish(report))
    assert result["diagnosis_attempted"]
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["candidate"]["content_language"] == "zh"
    assert result["candidate"]["routing_summary"] == candidate["routing_summary"]
    assert result["validation"] == "pending_replay"
    assert "诊断一次失败" in agent.prompts[0]
    assert "通用 Skill Builder" in agent.prompts[1]
    assert session.bank.skills[0]["status"] == "provisional"
    assert session.bank.temporary_path.exists()


def test_success_without_eligible_skill_builds_new_skill_even_after_no_new_skill_reply(tmp_path: Path) -> None:
    candidate = {**skill("missing_capability"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
    agent = FakeAgent([
        {"action": "NO_NEW_SKILL", "novelty_reason": "Nothing proposed", "candidate_spec": None},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager", "router_input": {"eligible_skill_cards": []},
                            "selected_skills": [], "injected_context": ""})
    report = {"task": "finish task", "status": "complete", "completion_satisfied": True,
              "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
              "rounds": [{"round_index": 1, "plan_text": "Finished task", "auditor_report": "Status: complete"}],
              "rounds_run": 1}
    result = asyncio.run(session.finish(report))
    assert result["model_action"] == "NO_NEW_SKILL"
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["policy_override"]["reason"] == "no_eligible_skill"
    assert result["validation"] == "pending_replay"
    assert len(agent.prompts) == 2


def test_empty_selection_with_available_skill_does_not_imply_new_skill(tmp_path: Path) -> None:
    existing = skill("existing")
    agent = FakeAgent([
        {"action": "NO_CHANGE", "failure_mechanism": "Task still failed", "router_error_type": "none"},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json", skills=[existing]), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager", "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "router_output": {"selected_skills": [], "uncovered_need": ""},
                            "selected_skills": [], "injected_context": ""})
    result = asyncio.run(session.finish({"task": "task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Attempted task"}]}))
    assert result["action"] == "NO_CHANGE"
    assert result["no_skill_injected"] is True
    assert result["coverage_gap_detected"] is False
    assert not (tmp_path / "run" / "skill_candidate.json").exists()


def test_unsupported_router_claim_does_not_hide_missing_skill(tmp_path: Path) -> None:
    existing = skill("existing")
    candidate = {**skill("new_capability"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
    agent = FakeAgent([
        {"action": "UPDATE_ROUTER_PROMPT", "router_error_type": "omission",
         "target_skill_id": "not_in_cards", "failure_mechanism": "Claimed omission",
         "coverage_gap": True},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json", skills=[existing]), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager", "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "router_output": {"selected_skills": []}, "selected_skills": [], "injected_context": ""})
    result = asyncio.run(session.finish({"task": "task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Attempted task"}]}))
    assert result["model_action"] == "UPDATE_ROUTER_PROMPT"
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["policy_override"]["reason"] == "diagnosed_coverage_gap"


@pytest.mark.parametrize("gap_source", ["diagnosis", "router"])
def test_injected_skill_coverage_gap_builds_complementary_new_skill(tmp_path: Path, gap_source: str) -> None:
    old = skill("existing")
    candidate = {**skill("complementary_capability"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
    agent = FakeAgent([
        {"action": "NO_CHANGE", "failure_mechanism": "Existing Skill misses a necessary reusable step",
         "coverage_gap": gap_source == "diagnosis", "router_error_type": "none", "evidence_refs": ["round_001"]},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json", skills=[old]), "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager", "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "router_output": {"uncovered_need": "Missing second capability" if gap_source == "router" else ""},
                            "selected_skills": [{"skill_id": "existing", "version": 1}], "injected_context": "Skill: existing@1"})
    report = {"task": "finish task", "status": "incomplete",
              "rounds": [{"round_index": 1, "plan_text": "Attempted task", "auditor_report": "Status: incomplete"}]}
    result = asyncio.run(session.finish(report))
    assert result["model_action"] == "NO_CHANGE"
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["policy_override"]["reason"] == ("diagnosed_coverage_gap" if gap_source == "diagnosis" else "router_uncovered_need")
    assert result["candidate"]["skill_id"] == "complementary_capability"
    assert json.loads(session.bank.path.read_text())["skills"] == [old]
    assert SkillBank.load(session.bank.path).skills[-1]["status"] == "provisional"


def test_failure_can_create_and_revise_in_one_replay(tmp_path: Path) -> None:
    old = skill("existing")
    new = {**skill("missing_capability"), "status": "candidate", "validation": {}}
    revised = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "One missing capability and one defective handoff",
         "coverage_gap": True, "evidence_refs": ["round_001"], "revision_targets": [
             {"target_skill_id": "existing", "failure_mechanism": "Its handoff omits a required check",
              "evidence_refs": ["round_001:auditor_report"]}]},
        {"action": "CREATE_NEW_SKILL", "skills": [new, revised]},
    ])
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager",
                            "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "router_output": {"selected_skills": [{"skill_id": "existing", "version": 1}]},
                            "selected_skills": [{"skill_id": "existing", "version": 1}],
                            "injected_context": "Skill: existing@1\nOutput specification: omit check"})
    session.token_counts.append(200)
    baseline = {"task": "task", "status": "incomplete", "completion_satisfied": False,
                "rounds": [{"round_index": 1, "plan_text": "Attempted task",
                            "auditor_report": "Status: incomplete; handoff omitted check"}], "rounds_run": 1}
    successful = {**baseline, "status": "complete", "completion_satisfied": True,
                  "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned"}
    calls = []

    async def replay(request):
        calls.append(request)
        assert {item["skill_id"] for item in request["candidate"]} == {"existing", "missing_capability"}
        return {"report": successful, "task_tokens": 100, "effective": True,
                "hard_constraint_regression": False, "skill_routes": [
                    {"selected_skills": [{"skill_id": "existing", "version": 2},
                                         {"skill_id": "missing_capability", "version": 1}],
                     "injected_context": "both Skills injected"}]}

    result = asyncio.run(session.finish(baseline, replay=replay, reference_tokens=1000))
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["revision_target_ids"] == ["existing"]
    assert all(item["accepted"] is True for item in result["validation"])
    assert len(calls) == 1
    assert {item["skill_id"]: item["version"] for item in SkillBank.load(bank.path).skills} == {
        "existing": 2, "missing_capability": 1}
    assert all(item["status"] == "validated" for item in SkillBank.load(bank.path).skills)
    assert "injected_skill_guidance" in agent.prompts[0]
    assert json.loads(agent.prompts[1].split("INPUT JSON:\n", 1)[1])["extracted_experiences"] == result["extracted_experiences"]


def test_rejected_revision_blocks_mixed_new_skill_batch(tmp_path: Path) -> None:
    old = skill("existing")
    old["roles"] = {role: dict(view) for role, view in old["roles"].items()}
    old["roles"]["manager"]["guidance"] = "Verify audited evidence and preserve acceptance checks"
    revised = copy.deepcopy(old)
    revised.update(version=2, version_parent=1, status="candidate", validation={})
    revised["roles"]["manager"]["guidance"] = "Only inspect one fixed path"
    new = {**skill("missing_capability"), "status": "candidate", "validation": {}}
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "Missing capability and defective old guidance",
         "coverage_gap": True, "evidence_refs": ["round_001"], "revision_targets": [{
             "target_skill_id": "existing", "failure_mechanism": "Old handoff omitted a check",
             "evidence_refs": ["round_001"]}]},
        {"action": "CREATE_NEW_SKILL", "skills": [new, revised]},
        {"preserves_reusable": False, "task_specific_generalized": False,
         "checks": [{"field": "roles.manager.guidance", "preserved": False,
                     "generalized": False, "reason": "Acceptance check was removed"}],
         "reason": "Revision is too narrow"},
    ])
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    bank.save()
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager",
                            "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "selected_skills": [{"skill_id": "existing", "version": 1}],
                            "injected_context": "Skill: existing@1"})
    result = asyncio.run(session.finish({"task": "task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Attempted task"}]}))
    assert result["error_stage"] == "revision_preservation_review"
    assert result["revision_preservation_reviews"][0]["skill_id"] == "existing"
    assert not (tmp_path / "run" / "skill_candidate.json").exists()
    assert [item["skill_id"] for item in SkillBank.load(bank.path).skills] == ["existing"]


def test_coverage_override_preserves_a_diagnosed_revision(tmp_path: Path) -> None:
    old = skill("existing")
    new = {**skill("missing_capability"), "status": "candidate", "validation": {}}
    revised = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    agent = FakeAgent([
        {"action": "REVISE_SKILL", "target_skill_id": "existing", "coverage_gap": True,
         "failure_mechanism": "Injected guidance omits handoff check", "evidence_refs": ["round_001"]},
        {"action": "CREATE_NEW_SKILL", "skills": [new, revised]},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json", skills=[old]), "en", agent,
                           None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager",
                            "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "selected_skills": [{"skill_id": "existing", "version": 1}],
                            "injected_context": "Skill: existing@1"})
    result = asyncio.run(session.finish({"task": "task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Tried"}]}))
    assert result["model_action"] == "REVISE_SKILL"
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["revision_target_ids"] == ["existing"]
    assert {item["skill_id"] for item in result["candidates"]} == {"existing", "missing_capability"}


def test_builder_revision_only_response_gets_one_mixed_batch_retry(tmp_path: Path) -> None:
    old = skill("existing")
    new = {**skill("missing_capability"), "status": "candidate", "validation": {}}
    revised = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    agent = FakeAgent([
        {"action": "REVISE_SKILL", "target_skill_id": "existing", "coverage_gap": True,
         "failure_mechanism": "Old guidance is defective and another capability is absent",
         "evidence_refs": ["round_001"]},
        {"action": "REVISE_SKILL", "skill": revised},
        {"action": "CREATE_NEW_SKILL", "skills": [new, revised]},
    ])
    session = SkillSession(SkillBank(tmp_path / "bank.json", skills=[old]), "en", agent,
                           None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager",
                            "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "selected_skills": [{"skill_id": "existing", "version": 1}],
                            "injected_context": "Skill: existing@1"})
    result = asyncio.run(session.finish({"task": "task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Tried"}]}))
    assert result["action"] == "CREATE_NEW_SKILL"
    assert result["builder_output"]["action"] == "REVISE_SKILL"
    assert result["builder_retry_output"]["action"] == "CREATE_NEW_SKILL"
    assert len(agent.prompts) == 3
    assert {item["skill_id"] for item in result["candidates"]} == {"existing", "missing_capability"}


def test_combined_revision_rejects_skill_that_was_not_injected(tmp_path: Path) -> None:
    old = skill("existing")
    agent = FakeAgent([{"action": "CREATE_NEW_SKILL", "failure_mechanism": "Gap",
                        "evidence_refs": ["round_001"], "revision_targets": [
                            {"target_skill_id": "existing", "failure_mechanism": "Claimed defect",
                             "evidence_refs": ["round_001"]}]}])
    session = SkillSession(SkillBank(tmp_path / "bank.json", skills=[old]), "en", agent,
                           None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager",
                            "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
                            "selected_skills": [], "injected_context": ""})
    result = asyncio.run(session.finish({"task": "task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Tried"}]}))
    assert result["error_stage"] == "evidence_validation"
    assert "lacks injected guidance" in result["learning_error"]
    assert len(agent.prompts) == 1


def test_chinese_router_repair_remains_a_candidate(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path / "bank.json", skills=[chinese_skill("existing")])
    agent = FakeAgent([
        {"action": "UPDATE_ROUTER_PROMPT", "failure_mechanism": "Router 漏选适用 Skill",
         "evidence_refs": ["round_001"], "router_error_type": "omission", "target_skill_id": "existing"},
        {"action": "UPDATE_ROUTER_PROMPT", "candidate_router_prompt": "新的中文 Router Prompt：根据证据选择 Skill。"},
    ])
    session = SkillSession(bank, "zh", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({
        "round": 1, "role": "manager", "router_input": {"eligible_skill_cards": [{"skill_id": "existing"}]},
        "router_output": {"selected_skills": []}, "selected_skills": [], "injected_context": "",
    })
    report = {"task": "完成任务", "status": "incomplete", "rounds": [{"round_index": 1, "plan_text": "已尝试执行"}]}
    result = asyncio.run(session.finish(report))
    assert result["action"] == "UPDATE_ROUTER_PROMPT"
    assert result["validation"] == "pending_replay"
    assert result["candidate"].startswith("新的中文")
    assert "修订中文 Skill Router Prompt" in agent.prompts[1]
    assert '"language": "zh"' in agent.prompts[1]
    assert bank.router_prompts["zh"] != result["candidate"]


def test_reward_requires_auditor_guard_and_reported_tokens() -> None:
    base = {"status": "incomplete", "completion_satisfied": False, "latest_auditor_report": ""}
    good = {"status": "complete", "completion_satisfied": True, "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned\nAudit facts: verified"}
    bad = {**good, "latest_auditor_report": "Status: incomplete\nIntegrity: clean\nContract audit: aligned"}
    assert task_success(good)
    assert not task_success(bad)
    assert accept_candidate(base, good, baseline_tokens=100, candidate_tokens=200, reference_tokens=1000, effective=True, hard_constraint_regression=False)[0]
    assert not accept_candidate(base, bad, baseline_tokens=100, candidate_tokens=200, reference_tokens=1000, effective=True, hard_constraint_regression=False)[0]
    assert not accept_candidate(base, good, baseline_tokens=None, candidate_tokens=200, reference_tokens=1000, effective=True, hard_constraint_regression=False)[0]
    assert token_usage(EpisodeResult(status="done", actions_log='{"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":4}}')) == 7
    assert token_usage(EpisodeResult(status="done", actions_log='{"type":"step_finish","part":{"id":"one","tokens":{"total":9,"input":7,"output":2}}}')) == 9
    assert token_usage(EpisodeResult(status="done", metadata={"n_input_tokens": 12, "n_output_tokens": 3, "n_cache_tokens": 4})) == 15
    assert token_usage(EpisodeResult(status="done", actions_log="plain text")) is None


def test_learning_candidate_enters_temporary_bank_without_replay(tmp_path: Path) -> None:
    source = skill("new_mechanism")
    spec = {"category_id": "evidence_handoff", "problem_mechanism": source["problem_mechanism"], "subcases": [], "triggers": [], "exclusions": [], "roles": source["roles"]}
    bank = SkillBank(tmp_path / "bank.json")
    agent = FakeAgent([
        {"action": "CREATE_NEW_SKILL", "candidate_spec": spec, "evidence_refs": ["round_001"]},
    ])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.token_counts.append(100)
    report = {"task": "first task", "status": "complete", "completion_satisfied": True, "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned", "rounds": [], "rounds_run": 1}
    result = asyncio.run(session.finish(report))
    assert result["validation"] == "pending_replay"
    assert bank.skills[0]["status"] == "provisional"
    assert json.loads(bank.path.read_text())["skills"] == []
    assert json.loads(bank.temporary_path.read_text())["skills"][0]["skill_id"] == bank.skills[0]["skill_id"]
    assert (tmp_path / "run" / "skill_candidate.json").exists()
    assert len(agent.prompts) == 1
    assert bank.skills[0]["update_count"] == 0


def test_temporary_revision_routes_then_moves_to_validated_bank(tmp_path: Path) -> None:
    old = skill("revision_target")
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    bank.save()
    revision = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    stage_skill_candidate(bank, revision)
    loaded = SkillBank.load(bank.path)
    assert [card["version"] for card in loaded.eligible("manager", "en")] == [2]
    assert json.loads(bank.path.read_text())["skills"] == []
    temporary = json.loads(bank.temporary_path.read_text())
    assert temporary["skills"][0]["version"] == 2
    assert temporary["skills"][0]["update_count"] == 1
    assert temporary["shadowed_skills"][0]["version"] == 1
    assert temporary["shadowed_skills"][0]["update_count"] == 0
    commit_validated_candidate(loaded, "skill", revision, language="en", validation={
        "accepted": True, "effective": True, "candidate_success": True,
        "hard_constraint_regression": False, "reward_delta": 1,
    })
    assert not bank.temporary_path.exists()
    assert SkillBank.load(bank.path).skills[0]["version"] == 2
    assert SkillBank.load(bank.path).skills[0]["update_count"] == 1
    assert len(SkillBank.load(bank.path).history) == 1


def test_revision_counts_start_over_and_old_version_counts_are_archived(tmp_path: Path) -> None:
    old = {**skill("revision_target"), "selection_count": 7, "success_count": 3}
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    bank.save()
    revision = {**old, "version": 2, "version_parent": 1, "status": "candidate",
                "validation": {}, "selection_count": 77, "success_count": 66}
    stage_skill_candidate(bank, revision)
    staged = SkillBank.load(bank.path)
    assert (staged.skills[0]["selection_count"], staged.skills[0]["success_count"]) == (0, 0)
    assert (staged.shadowed_skills[0]["selection_count"], staged.shadowed_skills[0]["success_count"]) == (7, 3)

    # A replay counts once for the new version, even when several roles select it.
    routes = [{"selected_skills": [{"skill_id": "revision_target", "version": 2}]}] * 3
    record_selected_skill_outcomes(staged, routes, success=True)
    commit_validated_candidate(staged, "skill", revision, language="en", validation={
        "accepted": True, "effective": True, "candidate_success": True,
        "hard_constraint_regression": False, "reward_delta": 1,
    })
    loaded = SkillBank.load(bank.path)
    assert (loaded.skills[0]["selection_count"], loaded.skills[0]["success_count"]) == (1, 1)
    historical = [entry["skill"] for entry in loaded.history if entry["kind"] == "skill"]
    assert [(item["version"], item["selection_count"], item["success_count"]) for item in historical] == [(1, 7, 3)]

    second = {**loaded.skills[0], "version": 3, "version_parent": 2,
              "status": "candidate", "validation": {}}
    stage_skill_candidate(loaded, second)
    assert (loaded.skills[0]["selection_count"], loaded.skills[0]["success_count"]) == (0, 0)
    assert (loaded.shadowed_skills[0]["selection_count"], loaded.shadowed_skills[0]["success_count"]) == (1, 1)
    record_selected_skill_outcomes(loaded, routes=[{
        "selected_skills": [{"skill_id": "revision_target", "version": 3}],
    }], success=False)
    commit_validated_candidate(loaded, "skill", second, language="en", validation={
        "accepted": True, "effective": True, "candidate_success": True,
        "hard_constraint_regression": False, "reward_delta": 1,
    })
    reloaded = SkillBank.load(bank.path)
    assert (reloaded.skills[0]["selection_count"], reloaded.skills[0]["success_count"]) == (1, 0)
    assert [(entry["skill"]["version"], entry["skill"]["selection_count"],
             entry["skill"]["success_count"]) for entry in reloaded.history if entry["kind"] == "skill"] == [
                 (1, 7, 3), (2, 1, 1)]


def test_unvalidated_revision_keeps_old_version_counts_in_temporary_bank(tmp_path: Path) -> None:
    old = {**skill("revision_target"), "selection_count": 4, "success_count": 2}
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    bank.save()
    revision = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    stage_skill_candidate(bank, revision)
    record_selected_skill_outcomes(bank, [{"selected_skills": [{"skill_id": "revision_target", "version": 2}]}],
                                   success=False)
    reloaded = SkillBank.load(bank.path)
    assert (reloaded.skills[0]["selection_count"], reloaded.skills[0]["success_count"]) == (1, 0)
    assert (reloaded.shadowed_skills[0]["selection_count"], reloaded.shadowed_skills[0]["success_count"]) == (4, 2)


def test_update_count_increments_for_each_staged_revision(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path / "bank.json", skills=[skill("revision_target")])
    bank.save()
    first = {**skill("revision_target"), "version": 2, "version_parent": 1,
             "status": "candidate", "validation": {}, "update_count": 99}
    stage_skill_candidate(bank, first)
    assert bank.skills[0]["update_count"] == 1
    second = {**first, "version": 3, "version_parent": 2}
    stage_skill_candidate(bank, second)
    assert SkillBank.load(bank.path).skills[0]["update_count"] == 2
    assert [item["update_count"] for item in bank.shadowed_skills] == [0, 1]


def test_direct_revision_commit_counts_one_update(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path / "bank.json", skills=[{
        **skill("revision_target"), "selection_count": 5, "success_count": 2}])
    bank.save()
    revision = {**skill("revision_target"), "version": 2, "version_parent": 1,
                "status": "candidate", "validation": {}, "update_count": 99,
                "selection_count": 90, "success_count": 80}
    commit_validated_candidate(bank, "skill", revision, language="en", validation={
        "accepted": True, "effective": True, "candidate_success": True,
        "hard_constraint_regression": False, "reward_delta": 1,
    })
    assert SkillBank.load(bank.path).skills[0]["update_count"] == 1
    loaded = SkillBank.load(bank.path)
    assert (loaded.skills[0]["selection_count"], loaded.skills[0]["success_count"]) == (0, 0)
    assert (loaded.history[0]["skill"]["selection_count"], loaded.history[0]["skill"]["success_count"]) == (5, 2)


def test_direct_new_skill_commit_starts_at_zero(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path / "bank.json")
    candidate = {**skill("new_skill"), "status": "candidate", "validation": {}, "update_count": 99}
    commit_validated_candidate(bank, "skill", candidate, language="en", validation={
        "accepted": True, "effective": True, "candidate_success": True,
        "hard_constraint_regression": False, "reward_delta": 1,
    })
    assert SkillBank.load(bank.path).skills[0]["update_count"] == 0


def test_legacy_skill_update_count_is_backfilled_from_version(tmp_path: Path) -> None:
    path = tmp_path / "bank.json"
    old = {**skill("legacy"), "version": 3, "version_parent": 2}
    path.write_text(json.dumps({"schema_version": 1, "skills": [old]}))
    bank = SkillBank.load(path)
    assert bank.skills[0]["update_count"] == 2
    bank.save()
    assert json.loads(path.read_text())["skills"][0]["update_count"] == 2


@pytest.mark.parametrize("invalid", [-1, True, "1"])
def test_invalid_update_count_is_rejected(invalid) -> None:
    with pytest.raises(ValueError, match="invalid update_count"):
        validate_skill({**skill("invalid_count"), "update_count": invalid}, language="en")


def test_dual_bank_load_handles_interrupted_revision_move(tmp_path: Path) -> None:
    old = skill("revision_target")
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    bank.save()
    original_regular = bank.path.read_text()
    revision = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    stage_skill_candidate(bank, revision)
    bank.path.write_text(original_regular)
    recovered = SkillBank.load(bank.path)
    assert [card["version"] for card in recovered.eligible("manager", "en")] == [2]
    assert len(recovered.shadowed_skills) == 1
    recovered.save()
    assert json.loads(bank.path.read_text())["skills"] == []


def test_unvalidated_revision_keeps_previous_skill_in_temporary_bank(tmp_path: Path) -> None:
    old = skill("revision_target")
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    bank.save()
    revision = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    stage_skill_candidate(bank, revision)
    loaded = SkillBank.load(bank.path)
    assert [card["version"] for card in loaded.eligible("manager", "en")] == [2]
    assert json.loads(loaded.path.read_text())["skills"] == []
    assert loaded.shadowed_skills[0]["version"] == 1


def test_revision_of_provisional_skill_keeps_old_version_in_temporary_bank(tmp_path: Path) -> None:
    old = {**skill("revision_target"), "status": "provisional"}
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    bank.save()
    revision = {**old, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    stage_skill_candidate(bank, revision)
    loaded = SkillBank.load(bank.path)
    assert [card["version"] for card in loaded.eligible("manager", "en")] == [2]
    assert loaded.skills[0]["status"] == "provisional"
    assert loaded.shadowed_skills[0]["version"] == 1


def test_success_summary_generates_multiple_temporary_skills(tmp_path: Path) -> None:
    first = skill("first")
    second = skill("second")
    specs = [
        {"category_id": category, "problem_mechanism": mechanism,
         "roles": source["roles"], "subcases": [], "triggers": [], "exclusions": []}
        for category, mechanism, source in (
            ("first_category", "First reusable handoff gap", first),
            ("second_category", "Second reusable verification gap", second),
        )
    ]
    bank = SkillBank(tmp_path / "bank.json")
    agent = FakeAgent([{"action": "CREATE_NEW_SKILL", "candidate_specs": specs, "evidence_refs": ["round_001"]}])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    report = {"task": "first task", "status": "complete", "completion_satisfied": True,
              "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
              "rounds": [{"round_index": 1, "plan_text": "Completed two reusable steps"}]}
    result = asyncio.run(session.finish(report))
    assert len(result["candidates"]) == 2
    assert len(json.loads((tmp_path / "run" / "skill_candidate.json").read_text())["candidate"]) == 2
    assert len(json.loads(bank.temporary_path.read_text())["skills"]) == 2
    assert len(SkillBank.load(bank.path).eligible("manager", "en")) == 2


def test_multiple_skills_use_one_replay_and_all_must_be_injected(tmp_path: Path) -> None:
    specs = [{"category_id": name, "problem_mechanism": f"Reusable {name} evidence",
              "roles": skill(name)["roles"], "subcases": [], "triggers": [], "exclusions": []}
             for name in ("first", "second")]
    bank = SkillBank(tmp_path / "bank.json")
    agent = FakeAgent([{"action": "CREATE_NEW_SKILL", "candidate_specs": specs, "evidence_refs": ["round_001"]}])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.token_counts.append(200)
    report = {"task": "first task", "status": "complete", "completion_satisfied": True,
              "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned",
              "rounds": [{"round_index": 1, "plan_text": "Completed two reusable steps"}]}
    calls = []

    async def replay(request):
        calls.append(request)
        assert isinstance(request["candidate"], list) and len(request["candidate"]) == 2
        chosen = request["candidate"][0]
        return {"report": report, "task_tokens": 100, "effective": True,
                "hard_constraint_regression": False,
                "skill_routes": [{"selected_skills": [{"skill_id": chosen["skill_id"], "version": 1}],
                                  "injected_context": "first only"}]}

    result = asyncio.run(session.finish(report, replay=replay, reference_tokens=1000))
    assert len(calls) == 1
    assert result["validation"][0]["accepted"] is False
    assert result["baseline_token_usage"]["total_tokens"] == 200
    assert result["replay_token_usage"]["total_tokens"] == 100
    assert len(SkillBank.load(bank.path).skills) == 2
    assert all(item["status"] == "provisional" for item in SkillBank.load(bank.path).skills)


def test_skill_selection_and_success_count_once_per_task(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path / "bank.json", skills=[skill("first"), skill("second")])
    bank.save()
    routes = [
        {"selected_skills": [{"skill_id": "first", "version": 1}]},
        {"selected_skills": [{"skill_id": "first", "version": 1}]},
    ]
    record_selected_skill_outcomes(bank, routes, success=True)
    record_selected_skill_outcomes(bank, routes, success=False)
    loaded = SkillBank.load(bank.path)
    assert (loaded.skills[0]["selection_count"], loaded.skills[0]["success_count"]) == (2, 1)
    assert (loaded.skills[1]["selection_count"], loaded.skills[1]["success_count"]) == (0, 0)


def test_revised_skill_requires_replay_and_effective_reward_gain(tmp_path: Path) -> None:
    source = skill("evidence_handoff")
    revision = {**source, "version": 2, "version_parent": 1, "status": "candidate", "validation": {}}
    bank = SkillBank(tmp_path / "bank.json", skills=[source])
    agent = FakeAgent([
        {"action": "REVISE_SKILL", "target_skill_id": source["skill_id"],
         "failure_mechanism": "Existing guidance omitted a handoff", "evidence_refs": ["round_001"]},
        {"action": "REVISE_SKILL", "skill": revision},
    ])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager", "selected_skills": [
        {"skill_id": source["skill_id"], "version": 1}], "injected_context": "Skill: evidence_handoff@1"})
    session.token_counts.append(200)
    baseline = {"task": "first task", "status": "incomplete", "completion_satisfied": False,
                "rounds": [{"round_index": 1, "plan_text": "Attempted the task"}], "rounds_run": 1}
    successful = {**baseline, "status": "complete", "completion_satisfied": True,
                  "latest_auditor_report": "Status: complete\nIntegrity: clean\nContract audit: aligned"}

    async def replay(request):
        assert request["kind"] == "skill" and request["mode"] == "candidate_validation"
        candidate_id = request["candidate"]["skill_id"]
        return {"report": successful, "task_tokens": 100, "effective": True, "hard_constraint_regression": False,
                "skill_routes": [{"selected_skills": [{"skill_id": candidate_id, "version": 2}], "injected_context": f"Skill: {candidate_id}@2"}]}

    result = asyncio.run(session.finish(baseline, replay=replay, reference_tokens=1000))
    assert result["validation"]["accepted"]
    assert SkillBank.load(bank.path).skills[0]["status"] == "validated"
    assert SkillBank.load(bank.path).skills[0]["version"] == 2


def test_revised_skill_that_drops_reusable_guidance_is_not_staged(tmp_path: Path) -> None:
    old = skill("existing")
    old["roles"] = {role: dict(view) for role, view in old["roles"].items()}
    old["roles"]["manager"]["guidance"] = "Verify audited evidence and preserve the acceptance check"
    revised = copy.deepcopy(old)
    revised.update(version=2, version_parent=1, status="candidate", validation={})
    revised["roles"]["manager"]["guidance"] = "Read /tmp/one_task.json"
    agent = FakeAgent([
        {"action": "REVISE_SKILL", "target_skill_id": "existing",
         "failure_mechanism": "Missing additional parser check", "evidence_refs": ["round_001"]},
        {"action": "REVISE_SKILL", "skill": revised},
        {"preserves_reusable": False, "task_specific_generalized": False,
         "checks": [{"field": "roles.manager.guidance", "preserved": False,
                     "generalized": False, "reason": "Old acceptance check is missing"}],
         "reason": "The revision narrows useful guidance to one task"},
    ])
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager", "selected_skills": [
        {"skill_id": "existing", "version": 1}], "injected_context": "Skill: existing@1"})
    result = asyncio.run(session.finish({"task": "task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Attempted task"}]}))
    assert result["error_stage"] == "revision_preservation_review"
    assert result["revision_preservation_reviews"][0]["changed_fields"] == ["roles.manager.guidance"]
    assert len(agent.prompts) == 3
    assert not (tmp_path / "run" / "skill_candidate.json").exists()
    assert SkillBank.load(bank.path).skills == [old]


def test_task_specific_old_guidance_can_be_generalized_without_losing_function(tmp_path: Path) -> None:
    old = skill("existing")
    old["roles"] = {role: dict(view) for role, view in old["roles"].items()}
    old["roles"]["manager"]["guidance"] = "Verify audited evidence for /home/project/input.json"
    revised = copy.deepcopy(old)
    revised.update(version=2, version_parent=1, status="candidate", validation={})
    revised["roles"]["manager"]["guidance"] = "Verify audited evidence for each declared input artifact"
    agent = FakeAgent([
        {"action": "REVISE_SKILL", "target_skill_id": "existing",
         "failure_mechanism": "Old path is task-specific", "evidence_refs": ["round_001"]},
        {"action": "REVISE_SKILL", "skill": revised},
        {"preserves_reusable": True, "task_specific_generalized": True,
         "checks": [{"field": "roles.manager.guidance", "preserved": True,
                     "generalized": True,
                     "reason": "The same audit check now applies to each declared input artifact"}],
         "reason": "The useful check remains and the path is generalized"},
    ])
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    session = SkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    session.records.append({"round": 1, "role": "manager", "selected_skills": [
        {"skill_id": "existing", "version": 1}], "injected_context": "Skill: existing@1"})
    result = asyncio.run(session.finish({"task": "task", "status": "incomplete",
                                         "rounds": [{"round_index": 1, "plan_text": "Attempted task"}]}))
    assert result["validation"] == "pending_replay"
    assert result["revision_preservation_reviews"][0]["review"]["preserves_reusable"] is True
    assert SkillBank.load(bank.path).skills[0]["status"] == "provisional"
    assert SkillBank.load(bank.path).skills[0]["roles"]["manager"]["guidance"] == revised["roles"]["manager"]["guidance"]


def test_revision_review_must_cover_every_changed_field() -> None:
    with pytest.raises(ValueError, match="omitted a changed field"):
        validate_revision_preservation_review({
            "preserves_reusable": True, "task_specific_generalized": True,
            "checks": [{"field": "roles.manager.guidance", "preserved": True,
                        "generalized": True, "reason": "Retained the audit check"}],
            "reason": "Reviewed",
        }, ["roles.manager.guidance", "roles.executor.output_spec"])


def test_manager_run_routes_and_injects_skill(tmp_path: Path) -> None:
    from lh_harness.manager import run
    from lh_harness.types import ExecResult, HarnessConfig

    class Env:
        staging_dir = str(tmp_path / "staging")

        async def exec(self, command, timeout=30, tee_path=None):
            return ExecResult("", "", 0, 0)

        async def upload(self, local_path, remote_path):
            return None

        async def download(self, remote_path, local_path):
            return None

        async def screenshot(self):
            return b""

    class Agent:
        def __init__(self):
            self.prompts = []

        async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
            self.prompts.append(prompt)
            if "Skill Router for LongHorizon-Harness" in prompt:
                output = json.dumps({"selected_skills": [{"skill_id": "first", "version": 1}]})
            else:
                output = "Next: blocked\n\nReason: unavailable"
            return EpisodeResult(status="done", actions_log=output, metadata={"assistant_visible_output": output, "usage": {"input_tokens": 5, "output_tokens": 5}})

    bank = SkillBank(tmp_path / "bank.json", skills=[skill("first")])
    bank.save()
    agent = Agent()
    report = asyncio.run(run(
        task="test task", env=Env(), agent=agent,
        config=HarnessConfig(max_total_episodes=1, workspace_path=str(tmp_path), harness_dir=str(tmp_path / "harness"), log_dir=str(tmp_path / "logs"), skill_bank_path=str(bank.path)),
    ))
    assert report["status"] == "blocked"
    assert any("Skill: first@1" in prompt for prompt in agent.prompts)
    assert report["skill_learning"]["task_tokens"] == 20
    routes = json.loads((tmp_path / "logs" / "role_orchestration" / "skill_routes.json").read_text())
    assert routes[0]["selected_skills"] == [{"skill_id": "first", "version": 1}]
    counted = SkillBank.load(bank.path).skills[0]
    assert counted["selection_count"] == 1
    assert counted["success_count"] == 0
    before_evaluation = bank.path.read_bytes()
    asyncio.run(run(
        task="evaluation task", env=Env(), agent=Agent(),
        config=HarnessConfig(max_total_episodes=1, workspace_path=str(tmp_path),
                             harness_dir=str(tmp_path / "harness_eval"), log_dir=str(tmp_path / "logs_eval"),
                             skill_bank_path=str(bank.path), skill_update_enabled=False),
    ))
    assert bank.path.read_bytes() == before_evaluation


def test_task_token_soft_limit_stops_new_round_and_still_diagnoses(tmp_path: Path) -> None:
    from lh_harness.manager import run
    from lh_harness.types import ExecResult, HarnessConfig

    class Env:
        staging_dir = str(tmp_path / "staging")

        async def exec(self, command, timeout=30, tee_path=None):
            return ExecResult("", "", 0, 0)

        async def upload(self, local_path, remote_path):
            return None

        async def download(self, remote_path, local_path):
            return None

        async def screenshot(self):
            return b""

    class Agent:
        def __init__(self):
            candidate = {**skill("missing_capability"), "status": "candidate", "benchmark_scope": "generic", "validation": {}}
            self.replies = [
                "Next: cli\nCurrent Task State: working\nPlan: implement",
                "partial implementation",
                "Status: incomplete\nIntegrity: clean\nContract audit: needs_revision\nSummary: work remains",
                json.dumps({"action": "NO_CHANGE", "failure_mechanism": "No reusable failure mechanism"}),
                json.dumps({"action": "CREATE_NEW_SKILL", "skill": candidate}),
            ]

        async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
            if "Extract reusable experience" in prompt:
                output = json.dumps({
                    "positive_experiences": [{"mechanism": "Preserve partial work", "strategy": "Record progress",
                                              "evidence_refs": ["round_001"], "reusable": True,
                                              "confidence": "medium"}],
                    "negative_experiences": [],
                })
                return EpisodeResult(status="done", metadata={"assistant_visible_output": output})
            reply = self.replies.pop(0)
            return EpisodeResult(status="done", actions_log=reply, metadata={
                "assistant_visible_output": reply,
                "usage": {"input_tokens": 500000, "output_tokens": 0},
            })

    bank = SkillBank(tmp_path / "bank.json")
    agent = Agent()
    report = asyncio.run(run(
        task="finish the task", env=Env(), agent=agent,
        config=HarnessConfig(
            max_total_episodes=3, workspace_path=str(tmp_path),
            harness_dir=str(tmp_path / "harness"), log_dir=str(tmp_path / "logs"),
            skill_bank_path=str(bank.path), task_token_soft_limit=1000000,
        ),
    ))
    assert report["status"] == "incomplete"
    assert report["rounds_run"] == 1
    assert report["abort_reason"] == "token_budget_exhausted"
    assert report["skill_learning"]["diagnosis_attempted"] is True
    assert report["skill_learning"]["model_action"] == "NO_CHANGE"
    assert report["skill_learning"]["action"] == "CREATE_NEW_SKILL"
    assert report["skill_learning"]["policy_override"]["reason"] == "no_eligible_skill"
    assert report["skill_learning"]["validation"] == "pending_replay"
    assert report["skill_learning"]["task_tokens"] == 1500000
    assert agent.replies == []
