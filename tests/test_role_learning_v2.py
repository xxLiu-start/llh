from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from lh_harness.skills.attribution import review_roles
from lh_harness.skills.experience import extract_run
from lh_harness.skills.learning import LEARNING_POLICY_VERSION, learn
from lh_harness.skills.library import SkillLibrary
from lh_harness.skills.schema import ROLES
_spec = importlib.util.spec_from_file_location("role_learning_fixtures", Path(__file__).with_name("test_role_skill_evolution.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
Model, bundle, settings, skill = _fixtures.Model, _fixtures.bundle, _fixtures.settings, _fixtures.skill


AUDIT = "Status: incomplete\nIntegrity: clean\nContract audit: aligned\nAudit facts: repeated checks consumed the budget without producing the required artifact.\nBlocking constraints: required artifact missing.\n"


def trial(tmp_path, *, exception="AgentTimeoutError", audit=AUDIT, reward=0, completed=True):
    root = tmp_path / "trial"
    root.mkdir()
    orchestration = root / "role_orchestration"
    orchestration.mkdir()
    if completed:
        row = {"round_index": 1, "plan_text": "Repeat the already verified inspection before scheduling the remaining artifact.",
               "executor_output": "Inspected the same state twice; required artifact is still missing.", "auditor_report": audit}
        (orchestration / "rounds.jsonl").write_text(json.dumps(row) + "\n")
        round_dir = orchestration / "rounds/round_001"
        round_dir.mkdir(parents=True)
        (round_dir / "task_raw_trajectory.txt").write_text(json.dumps({"type": "result", "usage": {"input_tokens": 9, "output_tokens": 4}}) + "\n")
        pending = orchestration / "rounds/round_002"
        pending.mkdir()
        (pending / "task_raw_trajectory.txt").write_text(json.dumps({"type": "result", "usage": {"input_tokens": 999, "output_tokens": 999}}) + "\n")
    result = {"task_name": "task-a", "verifier_result": {"rewards": {"reward": reward}},
              "exception_info": {"exception_type": exception, "exception_message": "Agent execution timed out after 900.0 seconds"} if exception else None}
    (root / "result.json").write_text(json.dumps(result))
    return root


class EveryRoleModel(Model):
    """Synthetic model decisions only; never invokes an endpoint."""
    def __init__(self, primary="executor", selected=ROLES, signal=None):
        super().__init__(primary)
        self.selected = set(selected)
        self.signal = signal
        self.prompts = []

    def __call__(self, stage, role, prompt, data, schema):
        if stage == "proposer":
            self.prompts.append((role, prompt))
        value = super().__call__(stage, role, prompt, data, schema)
        if stage == "attribution":
            signal = self.signal or ("negative" if data["bundle"]["outcome"] == "failure" else "positive")
            for target, assessment in value["role_assessments"].items():
                assessment.update(decision="LEARN" if target in self.selected else "NONE",
                                  signal=signal if target in self.selected else "none")
        return value


def split(*tasks):
    return {"dataset": "fixture-v1", "train": list(tasks), "validation": [], "test": ["held-out"], "sha256": "f" * 64}


@pytest.mark.parametrize("reward", [0, 1])
def test_agent_timeout_is_negative_experience_not_infrastructure_or_success(tmp_path, reward):
    root = trial(tmp_path, reward=reward)
    before = hashlib.sha256((root / "result.json").read_bytes()).hexdigest()
    b = extract_run(root, task_id="task-a", dataset="fixture-v1")
    assert b["outcome"] == "failure"
    assert b["runtime"]["agent_timeout"] is True
    assert b["runtime"]["infrastructure_failure"] is False
    assert b["runtime"]["completed_rounds"] == 1
    assert b["runtime"]["role_tokens"]["executor"]["total_tokens"] == 13
    assert len(b["trajectories"]["executor"]) == 1
    assert "task-termination" in b["evidence"]
    assert hashlib.sha256((root / "result.json").read_bytes()).hexdigest() == before


def test_agent_timeout_without_completed_rounds_defers_each_role_instead_of_failing_batch(tmp_path):
    root = trial(tmp_path, completed=False)
    b = extract_run(root, task_id="task-a", dataset="fixture-v1")
    assert b["outcome"] == "defer" and not b["runtime"]["infrastructure_failure"]
    model = EveryRoleModel()
    report = learn([b], settings(ROLES), SkillLibrary(tmp_path / "library"), model, split("task-a"), flush=True)
    row = report["decisions"][0]
    assert set(row["role_decisions"]) == set(ROLES)
    assert all(item["decision"] == "DEFER" for item in row["role_decisions"].values())
    assert report["errors"] == [] and model.calls == []


@pytest.mark.parametrize("failure", ["backend", "proxy", "verifier"])
def test_actual_infrastructure_failure_still_excludes_all_roles(tmp_path, failure):
    root = trial(tmp_path, exception="EndpointUnavailableError" if failure == "backend" else "AgentTimeoutError")
    if failure == "proxy":
        log = root / "agent/request_proxy_requests.jsonl"
        log.parent.mkdir()
        log.write_text(json.dumps({"event": "upstream_http_error", "status": 500}) + "\n")
    if failure == "verifier":
        log = root / "verifier/infra_error.json"
        log.parent.mkdir()
        log.write_text("{}")
    b = extract_run(root, task_id="task-a", dataset="fixture-v1")
    assert b["runtime"]["infrastructure_failure"] is True
    model = EveryRoleModel()
    report = learn([b], settings(ROLES), SkillLibrary(tmp_path / "library"), model, split("task-a"), flush=True)
    assert model.calls == []
    assert all(item["decision"] == "HARNESS_RUNTIME_FAILURE" for item in report["decisions"][0]["role_decisions"].values())


def test_task_tool_connection_error_is_not_model_endpoint_failure(tmp_path):
    root = trial(tmp_path)
    path = root / "role_orchestration/rounds/round_001/task_raw_trajectory.txt"
    path.write_text(json.dumps({"type": "tool_result", "is_error": True, "content": "ConnectionRefusedError: task service not started"}) + "\n")
    b = extract_run(root, task_id="task-a", dataset="fixture-v1")
    assert b["outcome"] == "failure" and b["runtime"]["infrastructure_failure"] is False


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_each_role_can_learn_from_same_task_without_being_primary(tmp_path, outcome):
    model = EveryRoleModel(primary="executor")
    lib = SkillLibrary(tmp_path)
    report = learn([bundle(outcome=outcome), bundle("task-b", outcome=outcome)], settings(ROLES), lib, model,
                   split("task-a", "task-b"), flush=True)
    assert report["errors"] == []
    assert len(lib.list()) == 3
    for role in ROLES:
        assert report["role_summary"][role]["reviewed"] == 2
        assert report["role_summary"][role]["proposer_calls"] == 2
        assert any(s.role == role and s.support == 2 for s in lib.list())
    assert all(row["attribution"]["primary_role"] == "executor" for row in report["decisions"] if "task_id" in row)


COMBINATIONS = [list(roles) for n in range(4) for roles in itertools.combinations(ROLES, n)]


@pytest.mark.parametrize("selected", COMBINATIONS)
def test_all_roles_reviewed_but_unselected_role_files_never_change(tmp_path, selected):
    lib = SkillLibrary(tmp_path)
    lib.create()
    for role in ROLES:
        lib.put(skill(role, name="existing-" + role))
    def files(role):
        return {str(path.relative_to(tmp_path)): path.read_bytes() for base in [tmp_path / role, tmp_path / "candidates" / role, tmp_path / ".history" / role]
                for path in base.rglob("*") if path.is_file()}
    before = {role: files(role) for role in ROLES}
    model = EveryRoleModel()
    report = learn([bundle()], settings(selected), lib, model, split("task-a"), flush=True)
    for role in set(ROLES) - set(selected):
        assert before[role] == files(role)
    assert all(role in selected for stage, role in model.calls if stage in {"proposer", "curator", "general_curator"})
    if selected:
        assert report["reviewed_roles"] == list(ROLES)
        assert set(report["decisions"][0]["role_decisions"]) == set(ROLES)
    else:
        assert model.calls == []


def test_timeout_cannot_be_turned_into_positive_capability(tmp_path):
    b = bundle(outcome="failure")
    b["runtime"]["agent_timeout"] = True
    model = EveryRoleModel(signal="positive")
    report = learn([b], settings(ROLES), SkillLibrary(tmp_path), model, split("task-a"), flush=True)
    assert [stage for stage, role in model.calls] == ["attribution"]
    assert all(item["decision"] == "DEFER" for item in report["decisions"][0]["role_decisions"].values())


def test_negative_timeout_lessons_reach_all_selected_proposers(tmp_path):
    b = bundle(outcome="failure")
    b["runtime"].update(agent_timeout=True, completed_rounds=2)
    model = EveryRoleModel()
    lib = SkillLibrary(tmp_path)
    report = learn([b], settings(ROLES), lib, model, split("task-a"), flush=True)
    assert report["errors"] == []
    assert lib.list() == []
    assert len(lib.list(candidates=True)) == 3 and all(s.support == 1 for s in lib.list(candidates=True))
    assert all("exhausted its execution budget" in prompt for role, prompt in model.prompts)
    assert report["learning_policy"]["version"] == LEARNING_POLICY_VERSION
    history = json.loads((tmp_path / "learning_policy_history.json").read_text())
    assert len(history) == 1 and history[0]["policy"] == report["learning_policy"]


def test_auditor_requires_evidence_outside_its_own_reports():
    model = EveryRoleModel(selected=["auditor"])
    def incomplete(stage, role, prompt, data, schema):
        value = model(stage, role, prompt, data, schema)
        value["role_assessments"]["auditor"]["evidence_references"] = ["auditor-evidence"]
        return value
    result = review_roles(bundle(), incomplete)
    assert result["role_assessments"]["auditor"]["decision"] == "DEFER"


def test_success_extraction_retains_benchmark_evidence_for_auditor(tmp_path):
    audit = AUDIT.replace("incomplete", "complete").replace("required artifact missing", "none")
    b = extract_run(trial(tmp_path, exception=None, audit=audit, reward=1), task_id="task-a", dataset="fixture-v1")
    assert b["outcome"] == "success" and "benchmark-result" in b["evidence"]
    assert b["qualified_auditor_evidence"] == ["rounds.jsonl:round=1:auditor:report"]


@pytest.mark.parametrize("role", ROLES)
def test_proposal_schema_requires_role_and_independent_audit_sources(tmp_path, role):
    model = EveryRoleModel(selected=[role])
    def self_only(stage, target, prompt, data, schema):
        value = model(stage, target, prompt, data, schema)
        if stage == "proposer":
            # Multiple references to the same role cannot replace independent
            # audit evidence, or external evidence for the Auditor.
            reference = target + "-evidence"
            item = {"reference": reference, "quote": data["bundle"]["evidence"][reference]["text"]}
            value["evidence"] = [item, item]
        return value
    report = learn([bundle()], settings([role]), SkillLibrary(tmp_path), self_only, split("task-a"), flush=True)
    assert report["errors"]
    assert report["decisions"][0]["role_decisions"][role]["decision"] == "DEFER"
    assert SkillLibrary(tmp_path).list(candidates=True) == []
    assert not any(stage == "curator" for stage, _ in model.calls)


def test_none_proposal_still_accepts_empty_evidence_without_forcing_a_skill(tmp_path):
    model = EveryRoleModel(selected=["manager"])
    def no_lesson(stage, role, prompt, data, schema):
        value = model(stage, role, prompt, data, schema)
        if stage == "proposer":
            value.update(action="NONE", evidence=[])
        return value
    report = learn([bundle()], settings(["manager"]), SkillLibrary(tmp_path), no_lesson, split("task-a"), flush=True)
    assert report["errors"] == []
    assert report["decisions"][0]["role_decisions"]["manager"]["decision"] == "NONE"
    assert SkillLibrary(tmp_path).list(candidates=True) == []


def test_timeout_termination_is_not_independent_evidence_of_an_auditor_method():
    b = bundle(outcome="failure")
    b["runtime"]["agent_timeout"] = True
    b["evidence"]["termination"] = {"role": "runtime", "text": "Agent execution exceeded the total task budget", "source_sha256": "c" * 64}
    model = EveryRoleModel(selected=["auditor"])
    def self_and_timeout(stage, role, prompt, data, schema):
        value = model(stage, role, prompt, data, schema)
        value["role_assessments"]["auditor"]["evidence_references"] = ["auditor-evidence", "termination"]
        return value
    result = review_roles(b, self_and_timeout)
    assert result["role_assessments"]["auditor"]["decision"] == "DEFER"


def test_wire_review_references_cannot_substitute_an_unqualified_audit():
    from lh_harness.skills.schema import validate_object
    b = bundle()
    b["qualified_auditor_evidence"] = ["auditor-evidence"]
    b["evidence"]["bad-auditor"] = {"role": "auditor", "text": "Untrusted audit with unknown contract status", "source_sha256": "d" * 64}
    model = EveryRoleModel(selected=[])
    schemas = []
    def capture(stage, role, prompt, data, schema):
        schemas.append(schema)
        return model(stage, role, prompt, data, schema)
    review_roles(b, capture)
    schema = schemas[0]["properties"]["role_assessments"]["properties"]["manager"]["properties"]["evidence_references"]
    assert validate_object(["manager-evidence", "auditor-evidence"], schema)
    with pytest.raises(ValueError):
        validate_object(["manager-evidence", "bad-auditor"], schema)
