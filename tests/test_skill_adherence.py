from __future__ import annotations

from lh_harness.skills.adherence import analyze_injected_skills
from lh_harness.skills.library import SkillLibrary
from lh_harness.skills.schema import Evidence, Skill
from lh_harness.skills.statistics import summarize
from lh_harness.token_usage import normalize


def make_skill():
    return Skill(
        name="inspect-before-retry",
        role="executor",
        modality="cli",
        description="Inspect the current failure before retrying an operation.",
        trigger=["retry", "failure"],
        preconditions=["A prior operation failed"],
        procedure=["Inspect the latest error and current state", "Change the cause before retrying"],
        verification=["Confirm the retry changed the observed state"],
        pitfalls=["Do not repeat an unchanged failing command"],
        generalization="Transfer this diagnostic loop to other stateful workflows.",
        evidence=[Evidence("task-a", "fixture", "run-a", "trajectory:executor:0",
                           "line 1 shell: inspect current state before retry", "a" * 64, "task-fingerprint-a")],
    )


def make_bundle(skill, *, version=1, trace=True, partial=False, timeout=False):
    ref = "partial:round=2:executor:trajectory" if partial else "trajectory:executor:0"
    evidence = {}
    if trace:
        evidence[ref] = {
            "role": "executor",
            "text": "line 1 shell: inspect current state before retry\nline 2 shell: apply a targeted correction",
            "source_sha256": "b" * 64,
            "completeness": "partial" if partial else "complete",
        }
    return {
        "task_id": "task-a",
        "dataset": "fixture",
        "run_id": "run-a",
        "split": "train",
        "outcome": "failure" if timeout else "success",
        "runtime": {"agent_timeout": timeout},
        "evidence": evidence,
        "skill_usage": [{
            "injections": [{
                "role": "executor",
                "modality": "cli",
                "skills": [skill.key],
                "versions": [version],
            }]
        }],
    }, ref


class AdherenceModel:
    def __init__(self, *, status="followed", refs=None, quote=None):
        self.status = status
        self.refs = refs
        self.quote = quote
        self.calls = []

    def __call__(self, stage, role, prompt, data, schema):
        self.calls.append((stage, role, data, prompt, schema))
        assessments = []
        for skill in data["skills"]:
            ids = [item["id"] for item in skill["procedure"]]
            if self.status == "followed":
                groups = (ids, [], [], [])
            elif self.status == "not_followed":
                groups = ([], ids, [], [])
            elif self.status == "not_applicable":
                groups = ([], [], ids, [])
            else:
                groups = ([], [], [], ids)
            reference = self.refs or skill["trajectory_evidence"][0]["reference"]
            quote = self.quote or skill["trajectory_evidence"][0]["text"].splitlines()[0]
            evidence = [] if self.status == "insufficient_evidence" else [{"reference": reference, "quote": quote}]
            assessments.append({
                "key": skill["key"],
                "version": skill["version"],
                "status": self.status,
                "followed_steps": groups[0],
                "skipped_steps": groups[1],
                "not_applicable_steps": groups[2],
                "unassessed_steps": groups[3],
                "evidence": evidence,
                "reason": "The preserved trace supports this classification.",
                "confidence": 0.95 if self.status != "insufficient_evidence" else 0.2,
            })
        return {"assessments": assessments}


def test_injected_skill_is_classified_from_own_role_trajectory(tmp_path):
    library = SkillLibrary(tmp_path / "library")
    skill = library.put(make_skill())
    bundle, ref = make_bundle(skill)
    model = AdherenceModel()
    result = analyze_injected_skills(bundle, library, model, roles=["executor"])

    assert result["status"] == "completed"
    assert result["analyzed_skill_count"] == 1
    assessment = result["assessments"][0]
    assert assessment["status"] == "followed"
    assert assessment["followed_steps"] == skill.procedure
    assert assessment["evidence"][0]["reference"] == ref
    assert model.calls[0][0:2] == ("adherence", "shared")


def test_missing_role_trajectory_does_not_infer_adherence_from_injection(tmp_path):
    library = SkillLibrary(tmp_path / "library")
    skill = library.put(make_skill())
    bundle, _ = make_bundle(skill, trace=False)
    model = AdherenceModel()
    result = analyze_injected_skills(bundle, library, model, roles=["executor"])

    assert result["status"] == "insufficient_evidence"
    assert result["model_calls"] == 0
    assert model.calls == []
    assert result["unavailable_skills"][0]["reason"] == "missing role trajectory evidence"


def test_injected_historical_version_is_loaded_not_current_skill_version(tmp_path):
    library = SkillLibrary(tmp_path / "library")
    original = make_skill()
    library.put(original)
    changed = make_skill()
    changed.procedure = ["Use a different diagnostic step", "Verify a different result"]
    library.put(changed)
    bundle, _ = make_bundle(original, version=1)
    model = AdherenceModel()

    result = analyze_injected_skills(bundle, library, model, roles=["executor"])

    assert result["assessments"][0]["version"] == 1
    assert model.calls[0][2]["skills"][0]["procedure"][0]["instruction"] == original.procedure[0]
    assert library.get(original.key).version == 2


def test_partial_timeout_trace_is_never_reported_as_fully_followed(tmp_path):
    library = SkillLibrary(tmp_path / "library")
    skill = library.put(make_skill())
    bundle, _ = make_bundle(skill, partial=True, timeout=True)
    model = AdherenceModel(status="not_followed")
    result = analyze_injected_skills(bundle, library, model, roles=["executor"])

    assessment = result["assessments"][0]
    assert assessment["status"] == "insufficient_evidence"
    assert assessment["skipped_steps"] == []
    assert assessment["unassessed_steps"] == skill.procedure
    assert model.calls[0][2]["skills"][0]["agent_timeout"] is True
    assert "Partial traces cannot prove later steps were skipped" in model.calls[0][3]


def test_unknown_injected_version_is_reported_without_model_call(tmp_path):
    library = SkillLibrary(tmp_path / "library")
    skill = make_skill()
    bundle, _ = make_bundle(skill, version=9)
    model = AdherenceModel()
    result = analyze_injected_skills(bundle, library, model, roles=["executor"])

    assert result["status"] == "unavailable"
    assert result["model_calls"] == 0
    assert result["unavailable_skills"][0]["version"] == 9



def test_adherence_usage_is_included_in_offline_learning_cost():
    usage = normalize({"input_tokens": 11, "output_tokens": 7})
    report = summarize([{"role": "adherence", "usage": usage}])

    assert report["categories"]["adherence"]["total_tokens"] == 18
    assert report["offline_learning"]["total_tokens"] == 18
    assert report["runtime"]["total_tokens"] == 0
