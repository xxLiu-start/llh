from __future__ import annotations

import asyncio
import copy
import json

import pytest

from lh_harness.skill_evolution import preservation_map
from lh_harness.skill_learning import (SkillBank, SkillSession, compact_trajectory,
                                       normalize_builder_skill_shape, task_reward,
                                       validate_skill)
from lh_harness.skill_prompt_texts import CREATE_SKILL_BUILDER_PROMPT_EN
from lh_harness.skill_prompt_texts_zh import CREATE_SKILL_BUILDER_PROMPT_ZH
from lh_harness.types import EpisodeBudget, EpisodeResult


class LegacySkillSession(SkillSession):
    """Exercise the retained LLM protocol explicitly; new defaults have separate tests."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("router_mode", "legacy_llm")
        super().__init__(*args, **kwargs)


def make_skill(skill_id: str, *, status: str = "validated") -> dict:
    view = {"applicability": "Use when this role needs an evidence handoff; skip otherwise",
            "input_use": "Read upstream status and evidence", "output_spec": "Pass result and evidence",
            "guidance": "Check input; execute; verify result; recover from last valid state",
            "avoid": "Never claim success without proof"}
    return {"skill_id": skill_id, "category_id": skill_id, "version": 1,
            "version_parent": None, "status": status, "content_language": "en",
            "benchmark_scope": "generic", "interaction_modes": ["hybrid"],
            "problem_mechanism": f"Missing {skill_id} evidence", "routing_summary": f"Use for {skill_id} handoff",
            "subcases": [], "triggers": ["Evidence is needed"], "exclusions": ["No handoff"],
            "roles": {role: copy.deepcopy(view) for role in ("manager", "executor", "auditor")},
            "compatible_with": [], "conflicts_with": [], "provenance": {}, "validation": {}}


class BuilderAgent:
    def __init__(self) -> None:
        self.prompts = []
        self.created = 0

    async def run_episode(self, prompt, env, budget):
        self.prompts.append(prompt)
        payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
        if "Build exactly one new" in prompt:
            self.created += 1
            candidate = make_skill(f"new_{self.created}", status="candidate")
            return self._reply({"skill": candidate})
        if "Incrementally revise" in prompt:
            candidate = copy.deepcopy(payload["old_skill"])
            candidate["version"] += 1
            candidate["version_parent"] = 1
            candidate["status"] = "candidate"
            candidate["roles"]["executor"]["guidance"] += "; recheck changed dependencies"
            return self._reply({"skill": candidate})
        if "Independently review one revised Skill" in prompt:
            return self._reply({"preserves_reusable": True, "task_specific_generalized": True,
                                "checks": [{"field": field, "preserved": True, "generalized": True,
                                            "reason": "Old guidance remains verbatim"}
                                           for field in payload["changed_fields"]], "reason": "Preserved"})
        if "Revise the supplied old LLM Router" in prompt:
            return self._reply({"candidate_router_prompt": payload["old_router_prompt"] + "\nUse cited handoff evidence."})
        raise AssertionError("unexpected model call")

    @staticmethod
    def _reply(value):
        return EpisodeResult(status="done", metadata={"assistant_visible_output": json.dumps(value)})


def plan(n_create=0, n_revise=0, n_router=0):
    return {
        "create_skills": [{"mechanism": f"new mechanism {i}", "general_strategy": "Verify and recover",
                           "evidence_refs": ["round_001"]} for i in range(n_create)],
        "revise_skills": [{"target_skill_id": "old", "add_capabilities": ["Recheck changed dependencies"],
                           "fix_defects": [], "evidence_refs": ["round_001"]} for _ in range(n_revise)],
        "update_router": [{"router_error_type": "omission", "affected_skill_ids": ["old"],
                           "repair_instruction": "Select old when eligible and applicable",
                           "evidence_refs": ["round_001"], "depends_on_skill_ids": []}
                          for _ in range(n_router)],
        "no_update_reasons": [],
    }


def setup_evolution(tmp_path):
    bank = SkillBank(tmp_path / "bank.json", skills=[make_skill("old")])
    bank.save()
    agent = BuilderAgent()
    session = LegacySkillSession(bank, "en", agent, None, EpisodeBudget(), tmp_path / "run")
    selected = {"skill_id": "old", "version": 1}
    session.records = [
        {"round": 1, "role": "manager", "router_input": {"eligible_skill_cards": [{"skill_id": "old"}]},
         "router_output": {"selected_skills": [selected]}, "selected_skills": [selected],
         "injected_context": "old guidance"},
        {"round": 1, "role": "cli_executor", "router_input": {"eligible_skill_cards": [{"skill_id": "old"}]},
         "router_output": {"selected_skills": []}, "selected_skills": [], "injected_context": ""},
    ]
    report = {"task": "do work", "status": "incomplete", "rounds": [
        {"round_index": 1, "plan_text": "planned", "executor_output": "partial progress"}]}
    return session, agent, report, compact_trajectory(report, session.records)


@pytest.mark.parametrize("counts", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                                     (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 2)])
def test_independent_evolution_branches(tmp_path, counts):
    session, agent, report, trajectory = setup_evolution(tmp_path)
    result = asyncio.run(session._finish_evolution(
        plan(*counts), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    assert len(result["candidates"]) == counts[0] + min(counts[1], 1)
    assert len(agent.prompts) == counts[0] + min(counts[1], 1) * 2 + int(counts[2] > 0)
    assert all(item["status"] in {"provisional", "rejected"} for item in result["request_outcomes"])
    assert SkillBank.load(session.bank.path).router_candidates == session.bank.router_candidates
    if counts == (0, 0, 0):
        assert result["learning_phase"] == "no_candidate"
        assert result["validation"] == "not_requested"


def test_invalid_item_and_dependency_do_not_discard_independent_skill(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)
    request = plan(1, 0, 1)
    request["create_skills"].append({"mechanism": "invalid", "general_strategy": "do", "evidence_refs": ["round_999"]})
    request["update_router"][0]["depends_on_skill_ids"] = ["missing"]
    result = asyncio.run(session._finish_evolution(
        request, report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    assert len(result["candidates"]) == 1
    assert [item["status"] for item in result["request_outcomes"]] == ["provisional", "rejected", "rejected"]
    assert result["router_candidate"] is None


def test_new_skill_prose_ids_are_normalized_before_staging(tmp_path):
    session, agent, report, trajectory = setup_evolution(tmp_path)

    class ProseIdAgent(BuilderAgent):
        async def run_episode(self, prompt, env, budget):
            if "Build exactly one new" in prompt:
                candidate = make_skill("placeholder", status="candidate")
                candidate["skill_id"] = "中文 Skill ID"
                candidate["category_id"] = "build-checks"
                return self._reply({"skill": candidate})
            return await super().run_episode(prompt, env, budget)

    session.agent = ProseIdAgent()
    result = asyncio.run(session._finish_evolution(
        plan(1), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    assert result["request_outcomes"][0]["status"] == "provisional"
    assert result["request_outcomes"][0]["normalized_ids"] == {
        "skill_id": "中文 Skill ID", "category_id": "build-checks"}
    candidate = result["candidates"][0]
    assert candidate["skill_id"].startswith("skill_")
    assert candidate["category_id"] == candidate["skill_id"]
    assert SkillBank.load(session.bank.path).skills[-1]["status"] == "provisional"


def test_schema_repair_uses_draft_without_repeating_trajectory(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)

    class AliasDraftAgent(BuilderAgent):
        def __init__(self):
            super().__init__()
            self.repair_payload = None

        async def run_episode(self, prompt, env, budget):
            if "The input contains your earlier draft" in prompt:
                self.repair_payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
                return self._reply({"skill": make_skill("new_repaired", status="candidate")})
            if "Build exactly one new" in prompt:
                return self._reply({"skill": {
                    "skill_id": "new_repaired", "category_id": "new_repaired",
                    "mechanism": "Reusable evidence handoff", "triggers": ["handoff"],
                    "exclusions": ["no handoff"],
                    "role_views": {"manager": {"subtask": "brief"}},
                }})
            return await super().run_episode(prompt, env, budget)

    agent = AliasDraftAgent()
    session.agent = agent
    result = asyncio.run(session._finish_evolution(
        plan(1), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    assert result["request_outcomes"][0]["status"] == "provisional"
    assert result["request_outcomes"][0]["schema_repair_error"] == "Skill needs all three role views"
    assert result["request_outcomes"][0]["schema_normalizations"] == {
        "problem_mechanism": "mechanism", "subcases": "default_empty_list",
        "compatible_with": "default_empty_list", "conflicts_with": "default_empty_list"}
    assert agent.repair_payload["draft_skill"]["mechanism"] == "Reusable evidence handoff"
    assert agent.repair_payload["normalized_draft_skill"]["problem_mechanism"] == "Reusable evidence handoff"
    assert "roles must contain exactly manager, executor and auditor" in agent.repair_payload["schema_errors"]
    assert agent.repair_payload["normalized_draft_skill"]["subcases"] == []
    assert "evidence_excerpts" not in agent.repair_payload
    assert result["candidates"][0]["problem_mechanism"]


def test_exact_builder_aliases_are_normalized_without_rewriting_role_content(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)

    class ExactAliasAgent(BuilderAgent):
        async def run_episode(self, prompt, env, budget):
            if "Build exactly one new" in prompt:
                draft = make_skill("alias_complete", status="candidate")
                draft["mechanism"] = draft.pop("problem_mechanism")
                draft["route_summary"] = draft.pop("routing_summary")
                draft["role_views"] = draft.pop("roles")
                draft["trigger"] = draft.pop("triggers")[0]
                draft["exclusion"] = draft.pop("exclusions")[0]
                return self._reply({"skill": draft})
            return await super().run_episode(prompt, env, budget)

    agent = ExactAliasAgent()
    session.agent = agent
    result = asyncio.run(session._finish_evolution(
        plan(1), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    outcome = result["request_outcomes"][0]
    assert outcome["status"] == "provisional"
    assert outcome["schema_normalizations"] == {
        "problem_mechanism": "mechanism", "routing_summary": "route_summary",
        "triggers": "trigger", "exclusions": "exclusion", "roles": "role_views",
    }
    assert "schema_repair_error" not in outcome
    assert result["candidates"][0]["roles"]["executor"]["guidance"] == make_skill("alias_complete")["roles"]["executor"]["guidance"]


def test_incomplete_role_alias_is_not_silently_promoted():
    shaped, aliases = normalize_builder_skill_shape({
        "mechanism": "Evidence handoff",
        "role_views": {"manager": {"subtask": "Select the next step"},
                       "executor": {}, "auditor": {}},
    })
    assert shaped["problem_mechanism"] == "Evidence handoff"
    assert "roles" not in shaped
    assert aliases == {"problem_mechanism": "mechanism", "subcases": "default_empty_list",
                       "compatible_with": "default_empty_list", "conflicts_with": "default_empty_list"}


def test_both_builders_spell_out_the_actual_candidate_schema():
    for language, prompt in (("en", CREATE_SKILL_BUILDER_PROMPT_EN),
                             ("zh", CREATE_SKILL_BUILDER_PROMPT_ZH)):
        for field in ('"problem_mechanism"', '"routing_summary"', '"roles"',
                      '"manager"', '"executor"', '"auditor"', '"input_use"',
                      '"output_spec"', '"guidance"', '"avoid"'):
            assert field in prompt
        example = json.JSONDecoder().raw_decode(
            prompt[prompt.index('{"skill":{"skill_id"'):])[0]["skill"]
        example.update(version=1, version_parent=None, status="candidate",
                       content_language=language, benchmark_scope="generic",
                       interaction_modes=["hybrid"], provenance={}, validation={})
        validate_skill(example, language=language, candidate=True)


def test_malformed_builder_json_is_retried_once_and_staged(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)

    class MalformedBuilder(BuilderAgent):
        def __init__(self):
            super().__init__()
            self.builder_calls = 0

        async def run_episode(self, prompt, env, budget):
            if "Build exactly one new" in prompt:
                self.prompts.append(prompt)
                self.builder_calls += 1
                if self.builder_calls == 1:
                    return EpisodeResult(status="done", metadata={"assistant_visible_output": '{"skill":'})
                return self._reply({"skill": make_skill("recovered", status="candidate")})
            return await super().run_episode(prompt, env, budget)

    agent = MalformedBuilder()
    session.agent = agent
    result = asyncio.run(session._finish_evolution(
        plan(1), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    assert agent.builder_calls == 2
    assert "JSON error:" in agent.prompts[1]
    assert result["request_outcomes"][0]["status"] == "provisional"
    assert SkillBank.load(session.bank.path).skills[-1]["skill_id"] == "recovered"


def test_repeated_malformed_builder_json_records_output(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)

    class BrokenBuilder(BuilderAgent):
        async def run_episode(self, prompt, env, budget):
            if "Build exactly one new" in prompt:
                self.prompts.append(prompt)
                return EpisodeResult(status="done", metadata={"assistant_visible_output": '{"skill":'})
            return await super().run_episode(prompt, env, budget)

    agent = BrokenBuilder()
    session.agent = agent
    result = asyncio.run(session._finish_evolution(
        plan(1), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    assert len(agent.prompts) == 2
    outcome = result["request_outcomes"][0]
    assert outcome["status"] == "rejected"
    assert "JSON format retry failed" in outcome["reason"]
    assert outcome["raw_model_output"] == '{"skill":'


def test_preservation_static_gate_and_stable_version_freeze(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)
    old = copy.deepcopy(session.bank.skills[0])
    bad = copy.deepcopy(old)
    bad["version"] = 2
    bad["roles"]["executor"]["guidance"] = "Be careful"
    with pytest.raises(ValueError, match="removed old capability"):
        preservation_map(old, bad, {"add_capabilities": ["new"], "evidence_refs": ["round_001"]})
    result = asyncio.run(session._finish_evolution(
        plan(0, 1, 0), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    assert result["candidates"][0]["capability_mapping"]["old_version"] == 1
    validated = tmp_path / "validated.json"
    exploratory = tmp_path / "exploratory.json"
    session.bank.export_frozen(validated)
    session.bank.export_frozen(exploratory, include_provisional=True)
    assert SkillBank.load(validated).skills[0]["version"] == 1
    assert SkillBank.load(exploratory).skills[0]["version"] == 2


def test_combination_replay_is_observation_and_failure_keeps_provisional(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)

    async def replay(_):
        return {"report": {"status": "failed"}, "task_tokens": None,
                "effective": False, "hard_constraint_regression": False}

    result = asyncio.run(session._finish_evolution(
        plan(2, 0, 0), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=replay, reliable_experience=True))
    assert result["validation"]["independent_causal_credit"] is False
    assert all(s["status"] == "provisional" for s in session.bank.skills if s["skill_id"].startswith("new_"))
    assert task_reward(report, None, reference_tokens=1000) is None
    assert task_reward(report, 2000, reference_tokens=1000) < task_reward(report, 1000, reference_tokens=1000)


def test_independent_validation_promotion_and_quarantine(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)
    result = asyncio.run(session._finish_evolution(
        plan(1, 1, 0), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    revision = next(s for s in result["candidates"] if s["skill_id"] == "old")
    with pytest.raises(ValueError, match="insufficient"):
        session.bank.promote_skill("old", 2)
    for task in ("a", "b"):
        session.bank.record_independent_validation("old", 2, {
            "task_id": task, "candidate_selected": True, "baseline_success": False,
            "candidate_success": True, "hard_constraint_regression": False})
    session.bank.promote_skill("old", 2)
    assert SkillBank.load(session.bank.path).skills[0]["version"] == 2
    new = next(s for s in result["candidates"] if s["skill_id"].startswith("new_"))
    session.bank.record_independent_validation(new["skill_id"], 1, {
        "task_id": "c", "candidate_selected": True, "baseline_success": True,
        "candidate_success": False, "hard_constraint_regression": True})
    assert new["skill_id"] not in {s["skill_id"] for s in session.bank.eligible("manager", "en")}


def test_full_finish_runs_extraction_planner_then_only_needed_builder(tmp_path):
    session, _, report, _ = setup_evolution(tmp_path)

    class PlannerAgent(BuilderAgent):
        async def run_episode(self, prompt, env, budget):
            if "Extract reusable experience" in prompt:
                self.prompts.append(prompt)
                return self._reply({"positive_experiences": [{"mechanism": "handoff gap",
                    "strategy": "Pass exact evidence", "evidence_refs": ["round_001"],
                    "reusable": True, "confidence": "high"}], "negative_experiences": []})
            if "Plan evidence-grounded Skill evolution" in prompt:
                self.prompts.append(prompt)
                return self._reply(plan(1, 0, 0))
            return await super().run_episode(prompt, env, budget)

    agent = PlannerAgent()
    session.agent = agent
    result = asyncio.run(session.finish(report))
    assert result["action"] == "EVOLVE"
    assert len(agent.prompts) == 3
    assert result["request_outcomes"][0]["status"] == "provisional"
    artifact = json.loads((session.log_dir / "skill_candidate.json").read_text())
    assert artifact["kind"] == "evolution"


def test_malformed_evolution_plan_is_retried(tmp_path):
    session, _, report, _ = setup_evolution(tmp_path)

    class PlannerRetryAgent(BuilderAgent):
        def __init__(self):
            super().__init__()
            self.plan_calls = 0

        async def run_episode(self, prompt, env, budget):
            if "Extract reusable experience" in prompt:
                return self._reply({"positive_experiences": [{"mechanism": "handoff gap",
                    "strategy": "Pass exact evidence", "evidence_refs": ["round_001"],
                    "reusable": True, "confidence": "high"}], "negative_experiences": []})
            if "Plan evidence-grounded Skill evolution" in prompt:
                self.plan_calls += 1
                if self.plan_calls == 1:
                    return EpisodeResult(status="done", metadata={"assistant_visible_output": '{"create_skills":'})
                return self._reply(plan(1))
            return await super().run_episode(prompt, env, budget)

    agent = PlannerRetryAgent()
    session.agent = agent
    result = asyncio.run(session.finish(report))
    assert agent.plan_calls == 2
    assert result["action"] == "EVOLVE"
    assert result["request_outcomes"][0]["status"] == "provisional"


def test_evidenced_defect_repair_only_removes_named_clause():
    old = make_skill("old")
    new = copy.deepcopy(old)
    new["version"] = 2
    new["roles"]["executor"]["guidance"] = "Check input; execute; verify result; recheck dependencies and recover from last revalidated state"
    mapping = preservation_map(old, new, {"add_capabilities": ["Recheck"],
        "evidence_refs": ["round_001"], "fix_defects": [{
            "field": "roles.executor.guidance", "old_text": "recover from last valid state",
            "replacement_text": "recover from last revalidated state",
            "reason": "Recovery point can be stale", "evidence_refs": ["round_001"]}]})
    assert mapping["changed_fields"] == ["roles.executor.guidance"]
    with pytest.raises(ValueError, match="removed old capability"):
        preservation_map(old, new, {"add_capabilities": ["Recheck"],
            "evidence_refs": ["round_001"], "fix_defects": []})


def test_multiple_distinct_revisions_and_router_dependency(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)
    session.bank.skills.append(make_skill("old2"))
    session.bank.save()
    session.records[0]["selected_skills"].append({"skill_id": "old2", "version": 1})
    request = plan(1, 1, 1)
    request["revise_skills"].append({"target_skill_id": "old2",
        "add_capabilities": ["Recheck dependencies"], "fix_defects": [],
        "evidence_refs": ["round_001"]})
    request["update_router"][0]["depends_on_skill_ids"] = ["new_1"]
    result = asyncio.run(session._finish_evolution(
        request, report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    assert {item["skill_id"] for item in result["candidates"]} == {"new_1", "old", "old2"}
    assert result["router_candidate"]
    frozen = tmp_path / "exploratory.json"
    session.bank.export_frozen(frozen, include_provisional=True)
    assert "Use cited handoff evidence." in SkillBank.load(frozen).router_prompts["en"]


def test_router_candidate_needs_independent_evidence_to_promote(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)
    result = asyncio.run(session._finish_evolution(
        plan(0, 0, 1), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    candidate = result["router_candidate"]
    with pytest.raises(ValueError, match="insufficient"):
        session.bank.promote_router("en", candidate)
    for task in ("a", "b"):
        session.bank.record_router_validation("en", candidate, {
            "task_id": task, "routing_improved": True, "hard_constraint_regression": False})
    session.bank.promote_router("en", candidate)
    assert SkillBank.load(session.bank.path).router_prompts["en"] == candidate
    session.bank.rollback_router("en")
    assert SkillBank.load(session.bank.path).router_prompts["en"] != candidate


def test_rollback_provisional_revision_restores_validated_skill(tmp_path):
    session, _, report, trajectory = setup_evolution(tmp_path)
    asyncio.run(session._finish_evolution(
        plan(0, 1, 0), report=report, trajectory=trajectory, skill_index=[],
        status={}, task_id="task", replay=None, reliable_experience=True))
    session.bank.rollback_skill("old")
    loaded = SkillBank.load(session.bank.path)
    assert loaded.skills[0]["version"] == 1
    assert loaded.skills[0]["status"] == "validated"
    assert any(entry.get("skill", {}).get("version") == 2 for entry in loaded.history)
