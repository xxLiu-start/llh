"""Unattended Manager and post-task Skill learning regressions."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lh_harness.manager import run
from lh_harness.skill_learning import SkillBank, SkillSession
from lh_harness.types import EpisodeBudget, EpisodeResult, ExecResult, HarnessConfig


class Environment:
    def __init__(self, root: Path) -> None:
        self.staging_dir = str(root / "staging")

    async def exec(self, command, timeout=30, tee_path=None):
        return ExecResult("", "", 0, 0)

    async def upload(self, local_path, remote_path):
        return None

    async def download(self, remote_path, local_path):
        return None

    async def screenshot(self):
        return b""


class RoleAgent:
    def __init__(self, manager_reply: str | list[str]) -> None:
        self.manager_replies = [manager_reply] if isinstance(manager_reply, str) else list(manager_reply)
        self.prompts: list[str] = []

    async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
        self.prompts.append(prompt)
        if "LongHorizon-Harness manager agent" in prompt or "LongHorizon-Harness 的任务管理器 agent" in prompt:
            reply = self.manager_replies.pop(0)
        elif "LongHorizon-Harness CLI executor" in prompt:
            reply = "Implemented the task from the task-defined input set."
        elif "LongHorizon-Harness CLI auditor" in prompt:
            reply = "Status: incomplete\nIntegrity: clean\nContract audit: aligned\nAudit facts: implementation needs final verification"
        else:
            reply = "Task remains incomplete."
        return EpisodeResult(status="done", actions_log=reply, metadata={"assistant_visible_output": reply})


def _run_manager(tmp_path: Path, reply: str | list[str], *, unattended: bool = True, rounds: int = 1, language: str = "en") -> tuple[dict, RoleAgent]:
    agent = RoleAgent(reply)
    report = asyncio.run(run(
        task="Build a poker classifier from the task-defined examples.",
        env=Environment(tmp_path), agent=agent,
        config=HarnessConfig(
            max_total_episodes=rounds, workspace_path=str(tmp_path),
            harness_dir=str(tmp_path / "harness"), log_dir=str(tmp_path / "logs"),
            unattended_benchmark=unattended, prompt_language=language,
        ),
    ))
    return report, agent


def test_unattended_manager_uses_evidence_and_routes_execution(tmp_path: Path) -> None:
    report, agent = _run_manager(
        tmp_path,
        "Current task state: input set identified from task statement\n"
        "Task contract: use examples named by task.toml; other JSON files need evidence before inclusion\n"
        "Dependency assessment: task metadata identifies the input files\n"
        "Next: cli\nTask: implement the classifier using those examples",
    )
    assert any("Unattended benchmark policy" in prompt and "Do not use `Next: ask`" in prompt for prompt in agent.prompts)
    assert any("LongHorizon-Harness CLI executor" in prompt for prompt in agent.prompts)
    assert report["abort_reason"] == "max_rounds_exhausted"
    assert "task.toml" in report["current_task_contract"]


def test_executor_turn_limit_preserves_work_and_replans(tmp_path: Path) -> None:
    """A local Claude Code turn cap must not abort the whole training trial."""
    outputs = iter([
        EpisodeResult(status="done", actions_log="Next: cli\nTask: implement classifier"),
        EpisodeResult(
            status="error",
            actions_log=json.dumps({
                "type": "result", "subtype": "error_max_turns",
                "num_turns": 13, "is_error": True,
            }),
            metadata={
                "steps_capped": True,
                "assistant_visible_output": "Created poker_classifier.py; verification still pending.",
            },
        ),
        EpisodeResult(status="done", actions_log="Next: blocked\nReason: test inputs are unresolved"),
        EpisodeResult(status="done", actions_log="Partial implementation remains in workspace."),
    ])

    class SequencedAgent:
        async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
            return next(outputs)

    report = asyncio.run(run(
        task="Build a poker classifier",
        env=Environment(tmp_path), agent=SequencedAgent(),
        config=HarnessConfig(
            max_total_episodes=2, workspace_path=str(tmp_path),
            harness_dir=str(tmp_path / "harness"), log_dir=str(tmp_path / "logs"),
            unattended_benchmark=True,
        ),
    ))

    assert report["rounds_run"] == 2
    assert report["status"] == "blocked"
    assert report["abort_reason"] == "manager_blocked"
    assert report["rounds"][0]["executor_status"]["steps_capped"] is True
    assert report["rounds"][0]["executor_output"] == (
        "Created poker_classifier.py; verification still pending."
    )
    assert "交互轮数上限" in report["rounds"][0]["harness_feedback"]


def test_unattended_unresolved_question_is_explicit_blocker(tmp_path: Path) -> None:
    report, agent = _run_manager(
        tmp_path,
        "Current task state: input specification unresolved\nTask contract: input scope unknown\n"
        "Dependency assessment: available task files do not identify authoritative input\n"
        "Next: ask\nQuestion: Which input set should be used?",
    )
    assert report["status"] == "blocked"
    assert report["abort_reason"] == "unattended_needs_human_input"
    assert report["block_reason"] == "Which input set should be used?"
    assert report["rounds"][0]["harness_feedback"].startswith("Unattended benchmark cannot answer")
    assert not any("LongHorizon-Harness CLI executor" in prompt for prompt in agent.prompts)


def test_unattended_ask_gets_one_grounded_replan_before_stopping(tmp_path: Path) -> None:
    ask = "Current task state: scope uncertain\nTask contract: inspect task metadata\nNext: ask\nQuestion: Which files?"
    execute = (
        "Current task state: metadata identifies the inputs\n"
        "Task contract: process files named by the task configuration\n"
        "Dependency assessment: source identified\nNext: cli\nTask: implement the classifier"
    )
    report, agent = _run_manager(tmp_path, [ask, execute], rounds=2)
    assert report["abort_reason"] == "max_rounds_exhausted"
    assert report["rounds"][0]["harness_feedback"].startswith("Unattended benchmark replan:")
    assert any("LongHorizon-Harness CLI executor" in prompt for prompt in agent.prompts)
    manager_prompts = [prompt for prompt in agent.prompts if "LongHorizon-Harness manager agent" in prompt]
    assert len(manager_prompts) == 2 and "Unattended benchmark replan:" in manager_prompts[1]


def test_repeated_unattended_ask_becomes_explicit_blocker(tmp_path: Path) -> None:
    ask = "Current task state: scope uncertain\nTask contract: input unknown\nNext: ask\nQuestion: Which files?"
    report, _ = _run_manager(tmp_path, [ask, ask], rounds=3)
    assert report["status"] == "blocked"
    assert report["abort_reason"] == "unattended_needs_human_input"
    assert report["rounds_run"] == 2


def test_chinese_unattended_manager_replans_then_blocks(tmp_path: Path) -> None:
    ask = "当前任务状态: 输入不明确\n任务契约: 需要核对来源\n下一步: 请示用户\n问题: 使用哪些输入？"
    report, agent = _run_manager(tmp_path, [ask, ask], rounds=3, language="zh")
    assert report["status"] == "blocked"
    assert report["abort_reason"] == "unattended_needs_human_input"
    assert report["rounds_run"] == 2
    assert report["rounds"][0]["harness_feedback"].startswith("无人值守 Benchmark 重新规划：")
    assert report["rounds"][1]["harness_feedback"].startswith("无人值守 Benchmark 无法回答")
    assert any("无人值守 Benchmark 策略" in prompt for prompt in agent.prompts)
    assert not any("用户澄清通道约束" in prompt for prompt in agent.prompts)


def test_unattended_manager_can_report_task_definition_blocker(tmp_path: Path) -> None:
    report, agent = _run_manager(
        tmp_path,
        "Current task state: available sources conflict\n"
        "Task contract: source of input remains unresolved\n"
        "Dependency assessment: task statement and environment do not resolve the conflict\n"
        "Next: blocked\nReason: task definition does not identify which conflicting input is authoritative",
    )
    assert report["status"] == "blocked" and report["abort_reason"] == "manager_blocked"
    assert "conflicting input" in report["last_plan"]
    assert "conflicting input" in report["block_reason"]
    assert not any("LongHorizon-Harness CLI executor" in prompt for prompt in agent.prompts)


def test_interactive_ask_route_is_preserved(tmp_path: Path) -> None:
    report, agent = _run_manager(
        tmp_path,
        "Current task state: waiting for user\nTask contract: user must select input\n"
        "Dependency assessment: user selection is missing\nNext: ask\nQuestion: Which input?",
        unattended=False,
    )
    assert report["abort_reason"] == "needs_human_input"
    assert any("User-clarification channel" in prompt for prompt in agent.prompts)


class LearningAgent:
    def __init__(self, replies: list[object], extraction_reply: object | None = None) -> None:
        self.replies = replies
        self.prompts: list[str] = []
        self.extraction_prompts: list[str] = []
        self.extraction_reply = extraction_reply

    async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
        if "Extract reusable experience" in prompt or "提取可复用经验" in prompt:
            self.extraction_prompts.append(prompt)
            round_id = json.loads(prompt.split("INPUT JSON:\n", 1)[1])["all_rounds"][0]["round_id"]
            reply = self.extraction_reply if self.extraction_reply is not None else {
                "positive_experiences": [{"mechanism": "Auditable handoff", "strategy": "Preserve evidence",
                                          "evidence_refs": [round_id], "reusable": True, "confidence": "medium"}],
                "negative_experiences": [],
            }
            if isinstance(reply, Exception):
                raise reply
            output = reply if isinstance(reply, str) else json.dumps(reply)
            return EpisodeResult(status="done", actions_log=output, metadata={"assistant_visible_output": output})
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        output = reply if isinstance(reply, str) else json.dumps(reply)
        return EpisodeResult(status="done", actions_log=output, metadata={"assistant_visible_output": output})


def _failed_report(abort_reason: str, *, trajectory: bool = True) -> dict:
    return {
        "task": "Classify poker hands", "status": "incomplete", "completion_satisfied": False,
        "abort_reason": abort_reason, "rounds_run": 3 if trajectory else 0,
        "rounds": [{"round_index": 3, "plan_text": "Next: ask; input scope unresolved",
                    "executor_output": "Read task data", "auditor_report": "Status: incomplete"}] if trajectory else [],
    }


def _session(tmp_path: Path, agent: LearningAgent, bank: SkillBank | None = None) -> SkillSession:
    return SkillSession(bank or SkillBank(tmp_path / "bank.json"), "en", agent, None, EpisodeBudget(), tmp_path / "run")


def _candidate_skill() -> dict:
    view = {"input_use": "Read audited evidence", "output_spec": "Report proof", "guidance": "Verify scope", "avoid": "Do not guess"}
    return {
        "skill_id": "input_scope", "category_id": "input_scope", "version": 1,
        "version_parent": None, "status": "candidate", "content_language": "en",
        "benchmark_scope": "generic", "interaction_modes": ["hybrid"],
        "problem_mechanism": "Input scope ambiguity", "subcases": [], "triggers": [],
        "exclusions": [], "roles": {role: view for role in ("manager", "executor", "auditor")},
        "compatible_with": [], "conflicts_with": [], "provenance": {"evidence_refs": ["round_003"]},
        "validation": {"baseline_success": None, "candidate_success": None},
    }


def test_failed_ask_trajectory_runs_diagnosis_and_records_no_change(tmp_path: Path) -> None:
    agent = LearningAgent([{"action": "NO_CHANGE", "failure_mechanism": "Task definition lacks an input contract", "evidence_refs": ["round_003"]}])
    result = asyncio.run(_session(tmp_path, agent).finish(_failed_report("needs_human_input")))
    assert result["diagnosis_attempted"] is True
    assert result["action"] == "NO_CHANGE"
    assert result["assessment_reason"] == "Task definition lacks an input contract"
    assert "failure diagnosis" in agent.prompts[0].lower() or "Diagnose a failed" in agent.prompts[0]
    assert not (tmp_path / "run" / "skill_candidate.json").exists()


def test_max_round_trajectory_runs_diagnosis_and_records_insufficient_evidence(tmp_path: Path) -> None:
    agent = LearningAgent([{"action": "INSUFFICIENT_EVIDENCE", "failure_mechanism": "Auditor evidence is inconclusive", "evidence_refs": ["round_003"]}])
    result = asyncio.run(_session(tmp_path, agent).finish(_failed_report("max_rounds_exhausted")))
    assert result["diagnosis_attempted"] and result["action"] == "INSUFFICIENT_EVIDENCE"
    assert result["assessment_reason"] == "Auditor evidence is inconclusive"
    assert not (tmp_path / "run" / "skill_candidate.json").exists()


def test_failed_diagnosis_can_generate_candidate_without_committing(tmp_path: Path) -> None:
    candidate = _candidate_skill()
    agent = LearningAgent([
        {"action": "CREATE_NEW_SKILL", "failure_mechanism": "Evidence handoff omitted input scope", "evidence_refs": ["round_003"]},
        {"action": "CREATE_NEW_SKILL", "skill": candidate},
    ])
    bank = SkillBank(tmp_path / "bank.json")
    result = asyncio.run(_session(tmp_path, agent, bank).finish(_failed_report("manager_blocked")))
    assert result["candidate"]["skill_id"] == "input_scope"
    assert result["validation"] == "pending_replay"
    assert (tmp_path / "run" / "skill_candidate.json").exists()
    assert bank.skills[0]["status"] == "provisional"
    assert bank.temporary_path.exists()


def test_no_trajectory_is_skipped_without_fake_action(tmp_path: Path) -> None:
    agent = LearningAgent([])
    result = asyncio.run(_session(tmp_path, agent).finish(_failed_report("needs_human_input", trajectory=False)))
    assert result["action"] is None and result["diagnosis_attempted"] is False
    assert result["skip_reason"] == "no_analyzable_execution_trajectory"
    assert not agent.prompts


def test_diagnosis_provider_failure_is_not_no_change(tmp_path: Path) -> None:
    agent = LearningAgent([OSError("gateway unavailable")])
    result = asyncio.run(_session(tmp_path, agent).finish(_failed_report("needs_human_input")))
    assert result["action"] is None and result["diagnosis_attempted"] is True
    assert result["error_stage"] == "failure_diagnosis"
    assert "gateway unavailable" in result["learning_error"]


def test_diagnosis_invalid_json_preserves_raw_output(tmp_path: Path) -> None:
    result = asyncio.run(_session(tmp_path, LearningAgent(["not a JSON diagnosis"])).finish(_failed_report("needs_human_input")))
    assert result["action"] is None
    assert result["error_stage"] == "failure_diagnosis"
    assert result["raw_model_output"] == "not a JSON diagnosis"


def test_builder_invalid_json_and_invalid_skill_are_recorded(tmp_path: Path) -> None:
    diagnosis = {"action": "CREATE_NEW_SKILL", "failure_mechanism": "Reusable handoff issue", "evidence_refs": ["round_003"]}
    for suffix, builder_reply, expected_stage in (
        ("json", "not JSON", "candidate_builder"),
        ("structure", {"action": "CREATE_NEW_SKILL", "skill": {"skill_id": "broken"}}, "candidate_validation"),
    ):
        run_dir = tmp_path / suffix
        agent = LearningAgent([diagnosis, builder_reply])
        result = asyncio.run(_session(run_dir, agent).finish(_failed_report("max_rounds_exhausted")))
        assert result["error_stage"] == expected_stage
        assert result["action"] == "CREATE_NEW_SKILL"
        assert result["learning_error"]
        if suffix == "json":
            assert result["raw_model_output"] == "not JSON"
        assert not (run_dir / "run" / "skill_candidate.json").exists()


def test_revision_requires_existing_injected_skill(tmp_path: Path) -> None:
    diagnosis = {"action": "REVISE_SKILL", "target_skill_id": "input_scope", "failure_mechanism": "Guidance was insufficient", "evidence_refs": ["round_003"]}
    for bank in (SkillBank(tmp_path / "none.json"), SkillBank(tmp_path / "existing.json", skills=[{**_candidate_skill(), "status": "validated", "validation": {}}])):
        agent = LearningAgent([diagnosis])
        result = asyncio.run(_session(tmp_path, agent, bank).finish(_failed_report("manager_blocked")))
        assert result["error_stage"] == "evidence_validation"
        assert result["learning_error"] == "revision target was not injected"
        assert len(agent.prompts) == 1


def test_router_update_requires_observed_routing_error(tmp_path: Path) -> None:
    agent = LearningAgent([{
        "action": "UPDATE_ROUTER_PROMPT", "failure_mechanism": "Router omission",
        "router_error_type": "omission", "target_skill_id": "missing",
        "evidence_refs": ["round_003"],
    }])
    result = asyncio.run(_session(tmp_path, agent).finish(_failed_report("manager_blocked")))
    assert result["error_stage"] == "evidence_validation"
    assert result["learning_error"] == "no supported Router selection error to repair"
    assert len(agent.prompts) == 1


def test_observed_router_omission_can_propose_revision(tmp_path: Path) -> None:
    existing = {**_candidate_skill(), "status": "validated", "validation": {}}
    bank = SkillBank(tmp_path / "bank.json", skills=[existing])
    agent = LearningAgent([
        {"action": "UPDATE_ROUTER_PROMPT", "failure_mechanism": "Router omitted the available Skill",
         "router_error_type": "omission", "target_skill_id": "input_scope", "evidence_refs": ["round_003"]},
        {"action": "UPDATE_ROUTER_PROMPT", "candidate_router_prompt": "Select input_scope when its trigger matches."},
    ])
    session = _session(tmp_path, agent, bank)
    session.records.append({
        "round": 3, "role": "manager", "router_output": {"selected_skills": []},
        "router_input": {"eligible_skill_cards": [{"skill_id": "input_scope"}]},
        "selected_skills": [], "injected_context": "",
    })
    result = asyncio.run(session.finish(_failed_report("max_rounds_exhausted")))
    assert result["action"] == "UPDATE_ROUTER_PROMPT"
    assert result["validation"] == "pending_replay"
    assert bank.router_prompts["en"] != result["candidate"]


def test_rejected_replay_does_not_commit_candidate(tmp_path: Path) -> None:
    old = {**_candidate_skill(), "status": "validated", "validation": {}}
    candidate = {**_candidate_skill(), "version": 2, "version_parent": 1}
    bank = SkillBank(tmp_path / "bank.json", skills=[old])
    agent = LearningAgent([
        {"action": "REVISE_SKILL", "target_skill_id": old["skill_id"],
         "failure_mechanism": "Existing guidance needs revision", "evidence_refs": ["round_003"]},
        {"action": "REVISE_SKILL", "skill": candidate},
    ])

    async def replay(request):
        return {"report": _failed_report("max_rounds_exhausted"), "task_tokens": 50,
                "effective": True, "hard_constraint_regression": False,
                "skill_routes": [{"selected_skills": [{"skill_id": candidate["skill_id"], "version": 2}], "injected_context": "Skill"}]}

    session = _session(tmp_path, agent, bank)
    session.records.append({"round": 3, "role": "manager", "selected_skills": [
        {"skill_id": old["skill_id"], "version": 1}], "injected_context": "Skill: input_scope@1"})
    result = asyncio.run(session.finish(_failed_report("manager_blocked"), replay=replay))
    assert result["validation"]["accepted"] is False
    assert bank.skills[0]["version"] == 2
    assert bank.skills[0]["status"] == "provisional"
    assert bank.shadowed_skills[0]["version"] == 1
