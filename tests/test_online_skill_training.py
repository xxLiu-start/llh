from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/skill_evolution/scripts/online_skill_training.py"
spec = importlib.util.spec_from_file_location("online_training_test_module", SCRIPT)
online = importlib.util.module_from_spec(spec)
spec.loader.exec_module(online)

from lh_harness.skills.library import SkillLibrary, atomic_json
from lh_harness.skills.router import SkillRouter
from lh_harness.skills.schema import Evidence, Skill


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "experiments/skill_evolution/configs", root / "experiments/skill_evolution/configs")
    dataset = root / online.shared.native.DEFAULT_OUTPUT
    dataset.mkdir(parents=True)
    atomic_json(dataset / online.shared.native.MANIFEST, {"tasks": [{"name": task} for task in ["a", "b", "c"]]})
    split = root / "experiments/skill_evolution/manifests/pilot-splits.json"
    atomic_json(split, {"dataset": "fixture", "train": ["a"], "validation": ["b"], "test": ["c"]})
    (root / ".cache").mkdir()
    monkeypatch.setattr(online, "ROOT", root)
    monkeypatch.delenv("LLH_LOCAL_BASE_URL", raising=False)
    monkeypatch.delenv("LLH_LOCAL_AUTH_SCHEME", raising=False)
    return root, tmp_path, split


def args(fixture, roles="executor", tasks=None, **overrides):
    root, tmp, split = fixture
    values = dict(roles=roles, task=tasks, limit=None, output=tmp / "online", base_url=None,
                  api_key_env=None, auth_scheme="none", freeze=False, dry_run=True,
                  serving_manifest=root / "serving.json", batch_size=1, skip_final_eval=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("roles", ["none", "manager", "executor", "auditor", "manager,executor", "manager,auditor", "executor,auditor", "all"])
def test_all_role_combinations_use_online_phase_and_same_inference(fixture, roles):
    p = online.plan(args(fixture, roles))
    original = tomllib.loads((fixture[0] / f"experiments/skill_evolution/configs/{p['group']}.toml").read_text())
    assert p["tasks"] == ["a", "b", "c"]
    assert p["config"]["run"] == original["run"]
    assert p["config"]["local_model"] == original["local_model"]
    assert p["settings"].phase == "online"
    assert p["settings"].training["batch_size"] == 1
    assert p["settings"].training["min_independent_tasks"] == 1
    assert set(p["settings"].training["train_roles"]) == set(p["roles"])
    assert p["settings"].max_per_role is None
    assert p["settings"].router["max_skills_per_role"] is None
    assert all(p["settings"].budget(role) is None for role in online.ROLES)
    assert not p["output"].exists()


@pytest.mark.parametrize("tasks,expected", [(None, ["a", "b", "c"]), ([["all"]], ["a", "b", "c"]),
    ([["b", "a"]], ["b", "a"]), ([["c"], ["a"]], ["c", "a"]), ([["b,a"]], ["b", "a"])])
def test_task_selection_and_order_do_not_read_old_history(fixture, tasks, expected):
    before = fixture[2].read_bytes()
    p = online.plan(args(fixture, tasks=tasks))
    assert p["tasks"] == expected
    assert p["online_split"] == {"dataset": "fixture", "train": expected, "validation": [], "test": []}
    assert fixture[2].read_bytes() == before
    assert not (fixture[0] / "experiments/skill_evolution/manifests/training-runs.json").exists()


@pytest.mark.parametrize("tasks", [[["all", "a"]], [["missing"]], [["a", "a"]]])
def test_invalid_task_selection_is_rejected(fixture, tasks):
    with pytest.raises(ValueError):
        online.plan(args(fixture, tasks=tasks))


def test_limit_and_new_output_protection(fixture):
    assert online.plan(args(fixture, limit=1))["tasks"] == ["a"]
    with pytest.raises(ValueError, match="正整数"):
        online.plan(args(fixture, limit=0))
    fixture[1].joinpath("online").mkdir()
    with pytest.raises(ValueError, match="输出目录已存在"):
        online.plan(args(fixture))


class SimulatedRuntime:
    """All evidence and counters here are synthetic; no network or Docker."""
    def __init__(self, p, *, fail_task=None, fail_learning=None, fail_reflection=None, mutate_score=False, missing_reward=False):
        self.p, self.events, self.used = p, [], []
        self.fail_task, self.fail_learning = fail_task, fail_learning
        self.fail_reflection = fail_reflection
        self.mutate_score, self.missing_reward = mutate_score, missing_reward
        self.learned = []
        self.batch_calls = []
        self.is_evaluation = False

    def run_task(self, task, index, folder):
        self.events.append(("evaluate" if self.is_evaluation else "execute", task))
        if self.fail_task == task:
            return {"returncode": 2, "job": str(folder), "trial": None}
        if self.p["roles"]:
            router = SkillRouter(SkillLibrary(self.p["output"] / "library"), self.p["settings"], counter=lambda text: len(text.split()))
            selected, _, _ = router.select(self.p["roles"][0], "cli", "cancel pending work")
            self.used.append((task, [skill.name for skill in selected]))
        trial = folder / f"{task}__fixture"
        trial.mkdir(parents=True)
        atomic_json(trial / "result.json", {"task_name": task, "verifier_result": None if self.missing_reward else {"rewards": {"reward": 1.0}}})
        (trial / "verifier").mkdir()
        (trial / "verifier/test-stdout.txt").write_text("synthetic original test evidence")
        self.events.append(("graded", task))
        return {"returncode": 0, "job": str(folder), "trial": str(trial)}

    def reflect_task(self, task, trial, output):
        assert not self.is_evaluation
        assert (trial / 'result.json').exists()
        self.events.append(('reflect', task))
        if self.fail_reflection == task:
            raise ValueError('simulated proposer failure')
        return {'attempted': True, 'returncode': 0, 'stage': 'propose', 'cost': online.summarize([])}

    def learn_task(self, task, trial, output):
        assert not self.is_evaluation, "final evaluation must never learn"
        self.events.append(("learn", task))
        assert (trial / "result.json").exists()
        if self.fail_learning == task:
            raise ValueError("simulated learner error")
        if self.mutate_score:
            (trial / "result.json").write_text("{}")
        self.learned.append(task)
        role = self.p["roles"][0]
        evidence = [Evidence(t, "fixture", "run-" + t, "independent-audit", "Independent auditor verified cancellation", "a" * 64) for t in self.learned]
        skill = Skill("cancel-pending-work", role, "shared" if role == "manager" else "cli",
            "Cancel pending work before awaiting cleanup.", ["cancel"], ["Pending work exists."], ["Cancel queued work."],
            ["Verify pending work did not start."], [], "Applicable to bounded asynchronous task queues.", evidence=evidence, status="active")
        SkillLibrary(self.p["output"] / "library").put(skill)
        return {"attempted": True, "returncode": 0, "report": {"experiment": "online_skill_learning"}}

    def learn_batch(self, rows, output):
        self.batch_calls.append([row["task_id"] for row in rows])
        for row in rows:
            self.learn_task(row["task_id"], Path(row["trial"]), output)
        return {"attempted": True, "returncode": 0, "report": {"experiment": "online_skill_learning"}}

    def evaluation_runtime(self, settings):
        import copy
        assert settings.phase == "test" and settings.training["enabled"] is False
        runtime = copy.copy(self)
        runtime.is_evaluation = True
        return runtime


def execute(p, runtime):
    p["output"].mkdir()
    manifest = {"status": "preparing", "tasks": []}
    code = online.run_sequence(p, manifest, runtime)
    return code, manifest


def test_each_task_is_graded_then_learned_before_next_and_reuses_only_past_skills(fixture):
    p = online.plan(args(fixture))
    runtime = SimulatedRuntime(p)
    code, manifest = execute(p, runtime)
    assert code == 0
    assert runtime.events == [(stage, task) for task in ["a", "b", "c"] for stage in ["execute", "graded", "reflect", "learn"]]
    assert runtime.used == [("a", []), ("b", ["cancel-pending-work"]), ("c", ["cancel-pending-work"])]
    rows = manifest["tasks"]
    assert rows[0]["library_after"] == rows[1]["library_before"]
    assert rows[1]["library_after"] == rows[2]["library_before"]
    assert rows[0]["library_after"]["active_skills"] == 1
    assert rows[1]["library_after"]["active_skills"] == 1
    source = json.loads((p["output"] / "library/online_provenance.json").read_text())
    assert source["observed_task_ids"] == ["a", "b", "c"]
    assert source["held_out_evaluation"] is False
    for row in rows:
        assert online.shared.native.digest(Path(row["trial"]) / "result.json") == row["raw_result_sha256"]


def test_learning_failure_does_not_change_reward_or_prevent_next_task(fixture):
    p = online.plan(args(fixture))
    runtime = SimulatedRuntime(p, fail_learning="a")
    code, manifest = execute(p, runtime)
    assert code == 1 and manifest["status"] == "completed_with_learning_errors"
    assert all(row["reward"] == 1.0 for row in manifest["tasks"])
    assert ("execute", "c") in runtime.events


@pytest.mark.parametrize("failure", ["infrastructure", "missing_reward"])
def test_infrastructure_or_missing_grade_stops_without_learning_or_next_task(fixture, failure):
    p = online.plan(args(fixture))
    runtime = SimulatedRuntime(p, fail_task="a" if failure == "infrastructure" else None, missing_reward=failure == "missing_reward")
    code, manifest = execute(p, runtime)
    assert code == 2
    assert not any(event[0] == "learn" or event[1] == "b" for event in runtime.events)
    source = json.loads((p["output"] / "library/online_provenance.json").read_text())
    assert source["observed_task_ids"] == ["a"]


def test_learning_cannot_rewrite_original_grades(fixture):
    p = online.plan(args(fixture))
    runtime = SimulatedRuntime(p, mutate_score=True)
    p["output"].mkdir()
    with pytest.raises(ValueError, match="修改了原始评分"):
        online.run_sequence(p, {"tasks": []}, runtime)
    assert ("execute", "b") not in runtime.events


def test_disabled_training_runs_selected_tasks_without_any_learning(fixture):
    p = online.plan(args(fixture, roles="none", tasks=[["b"]]))
    runtime = SimulatedRuntime(p)
    code, manifest = execute(p, runtime)
    assert code == 0 and runtime.events == [("execute", "b"), ("graded", "b")]
    assert not (p["output"] / "library").exists()
    assert manifest["tasks"][0]["learning"]["attempted"] is False


def test_dry_run_does_not_prepare_model_or_write_files(fixture, monkeypatch):
    monkeypatch.setattr(online.HarborRuntime, "prepare", lambda *a: pytest.fail("no endpoint or Docker access"))
    assert online.main(["manager", "--task", "c", "a", "--dry-run", "--output", str(fixture[1] / "dry")]) == 0
    assert not (fixture[1] / "dry").exists()


def test_model_preflight_failure_prevents_any_benchmark(fixture, monkeypatch):
    def unavailable(*args):
        raise ValueError("model endpoint unavailable; no fallback")
    monkeypatch.setattr(online.HarborRuntime, "prepare", unavailable)
    monkeypatch.setattr(online.HarborRuntime, "run_task", lambda *a: pytest.fail("no benchmark after failed preflight"))
    output = fixture[1] / "unavailable"
    assert online.main(["manager", "--task", "a", "--output", str(output)]) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["tasks"] == []


def test_observed_online_test_tasks_cannot_be_relabelled_held_out_even_without_skills(fixture):
    library = SkillLibrary(fixture[1] / "library")
    library.create()
    atomic_json(library.root / "online_provenance.json", {"observed_task_ids": ["c"]})
    library.freeze()
    assert not library.list(candidates=True)
    with pytest.raises(ValueError, match="already observed"):
        online.shared.online_provenance(library, ["c"])
    assert online.shared.online_provenance(library, ["unseen"])["observed_task_ids"] == ["c"]


def test_default_dispatch_is_online_and_offline_requires_flag(monkeypatch):
    import train_skills
    calls = []
    monkeypatch.setitem(sys.modules, "online_skill_training", SimpleNamespace(main=lambda argv: calls.append(("online", argv)) or 0))
    monkeypatch.setattr(train_skills, "offline_main", lambda argv: calls.append(("offline", argv)) or 0)
    assert train_skills.main(["manager", "--task", "a"]) == 0
    assert train_skills.main(["manager", "--offline"]) == 0
    assert calls == [("online", ["manager", "--task", "a"]), ("offline", ["manager"])]


def test_learning_cli_keeps_explicit_online_label_in_reports(fixture, monkeypatch):
    from lh_harness import cli
    import lh_harness.skills.learning as learning
    import lh_harness.skills.experience as experience
    phases = []
    monkeypatch.setattr(experience, "extract_run", lambda *a, **k: {})
    def learner(bundles, settings, library, model, splits, **kwargs):
        phases.append(settings.phase)
        return {"status": "completed", "experiment": "online_skill_learning"}
    monkeypatch.setattr(learning, "learn", learner)
    output = fixture[1] / "cost"
    config = fixture[0] / "experiments/skill_evolution/configs/manager_only.toml"
    assert cli.main(["skills", "learn", "--config", str(config), "--phase", "online", "--roles", "manager",
                     "--run", "a=/unused", "--split-manifest", str(fixture[2]), "--output", str(output)]) == 0
    assert phases == ["online"]
    assert json.loads((output / "learning_cost.json").read_text())["experiment"] == "online_skill_learning"


@pytest.mark.parametrize("roles,entry", [("executor", "harbor_skill_agent:SkillEnhancedAgent"),
    ("none", "harbor_skill_agent:PristineOriginalAgent")])
def test_real_runtime_builds_original_budget_config_and_single_task_harbor_command(fixture, monkeypatch, roles, entry):
    import copy
    import yaml
    p = online.plan(args(fixture, roles=roles, tasks=[["a"]]))
    model = fixture[1] / "model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"synthetic weight fixture")
    p["config"]["run"]["model"] = str(model)
    config = yaml.safe_load((ROOT / "evaluation/original_harness_a_evolve_tb20_qwen27b.local.yaml").read_text())
    config["agents"][0]["model_name"] = str(model)
    for key in ("harness_model", "verifier_model"):
        config["agents"][0]["kwargs"][key] = str(model)
    (p["dataset"] / "a").mkdir()
    (p["dataset"] / "a/task.toml").write_text('[environment]\ndocker_image="fixture-image"\n')
    monkeypatch.setattr(online.shared, "protected_sources", lambda root: {"original_commit": "fixture-commit"})
    monkeypatch.setattr(online.shared.native, "prepare_dataset", lambda *a, **k: {})
    monkeypatch.setattr(online.shared.native, "resolved_config", lambda *a, **k: copy.deepcopy(config))
    monkeypatch.setattr(online.shared.native, "configure_auth", lambda *a: (None, "none"))
    monkeypatch.setattr(online.shared.native, "configure_installation", lambda *a: {})
    monkeypatch.setattr(online.shared.verifier_runtime, "configure", lambda *a: {})
    monkeypatch.setattr(online.shared, "check_endpoint", lambda *a, **k: {"context_length": 1000000})
    monkeypatch.setattr(online.shared, "check_docker", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(online.shared.native, "serving_evidence", lambda *a: {})
    runtime = online.HarborRuntime(p, args(fixture, roles=roles))
    runtime.prepare()
    assert runtime.config["agents"][0]["import_path"] == entry
    assert runtime.config["agents"][0]["kwargs"]["cli_task_timeout_sec"] == config["agents"][0]["kwargs"]["cli_task_timeout_sec"]
    assert "override_timeout_sec" not in runtime.config["agents"][0]
    if roles != "none":
        assert runtime.config["agents"][0]["kwargs"]["skills_config"]["phase"] == "online"

    else:
        assert "skills_config" not in runtime.config["agents"][0]["kwargs"]
    monkeypatch.setattr(runtime.client, "json", lambda *a: {"data": [{"id": str(model), "max_model_len": 1000000}]})
    commands = []
    def harbor(command, env, root, job, **kwargs):
        commands.append(command)
        atomic_json(job / "a__fixture/result.json", {"task_name": "a", "verifier_result": {"rewards": {"reward": 1}}})
        return 0
    monkeypatch.setattr(online.shared.verifier_runtime, "run_harbor", harbor)
    result = runtime.run_task("a", 1, fixture[1] / "task")
    assert result["returncode"] == 0
    assert commands[0][commands[0].index("--include-task-name") + 1] == "a"
    assert commands[0].count("--include-task-name") == 1
    if roles != "none":
        def learner(command, **kwargs):
            commands.append(command)
            assert command[command.index("--phase") + 1] == "online"
            assert command[command.index('--stage') + 1] == 'propose'
            assert "--run" not in command
            index = Path(command[command.index("--run-index") + 1])
            assert json.loads(index.read_text()) == {"a": result["trial"]}
            return SimpleNamespace(returncode=0)
        monkeypatch.setattr(online.subprocess, "run", learner)
        runtime.learn_task("a", Path(result["trial"]), fixture[1] / "learning")
        library = SkillLibrary(p["output"] / "library")
        library.create()
        library.freeze()
        from dataclasses import asdict
        evaluation_settings = online.Settings.from_dict({**asdict(p["settings"]), "phase": "test",
                                                       "training": {**p["settings"].training, "enabled": False}})
        evaluation = runtime.evaluation_runtime(evaluation_settings)
        assert evaluation.config["agents"][0]["kwargs"]["skills_config"]["phase"] == "test"
        assert evaluation.config["agents"][0]["kwargs"]["skills_config"]["training"]["enabled"] is False
        assert runtime.config["agents"][0]["kwargs"]["skills_config"]["phase"] == "online"


def test_real_batch_runtime_only_consumes_immutable_task_proposals(fixture, monkeypatch):
    p = online.plan(args(fixture, roles='manager', batch_size=5))
    runtime = online.HarborRuntime(p, args(fixture, roles='manager'))
    runtime.env = {}
    path = fixture[1]/'task-proposals.json'
    atomic_json(path, {'synthetic': 'proposal artifact'})
    row = {'task_id': 'a', 'trial': '/never-reread', 'reflection': {
        'proposal_path': str(path), 'proposal_sha256': online.shared.native.digest(path)}}
    commands = []
    def curation(command, **kwargs):
        commands.append(command)
        assert command[command.index('--stage')+1] == 'curate'
        assert '--run-index' not in command and '--run' not in command
        assert command[command.index('--proposal-input')+1] == str(path)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(online.subprocess, 'run', curation)
    assert runtime.learn_batch([row], fixture[1]/'batch')['returncode'] == 0
    path.write_text('{}')
    with pytest.raises(ValueError, match='暂存产物发生变化'):
        runtime.learn_batch([row], fixture[1]/'next-batch')
    assert len(commands) == 1


def many_tasks(fixture, count):
    names = [f"task-{index:02d}" for index in range(count)]
    dataset = fixture[0] / online.shared.native.DEFAULT_OUTPUT
    atomic_json(dataset / online.shared.native.MANIFEST, {"tasks": [{"name": task} for task in names]})
    return names


def test_result_directory_selection_changes_with_its_tasks_and_preserves_order(fixture):
    directory = fixture[1] / "old-job"
    atomic_json(directory / "b__second/result.json", {"task_name": "b", "started_at": "2026-11-22T00:02:00Z"})
    atomic_json(directory / "a__first/result.json", {"task_name": "a", "started_at": "2026-11-22T00:01:00Z"})
    atomic_json(directory / "a__rerun/result.json", {"task_name": "a", "started_at": "2026-11-22T00:03:00Z"})
    p = online.plan(args(fixture, tasks_from_run=directory))
    assert p["tasks"] == p["evaluation_tasks"] == ["a", "b"]
    first = p["task_selection_source"]["sha256"]
    atomic_json(directory / "c__added/result.json", {"task_name": "c", "started_at": "2026-11-22T00:04:00Z"})
    p = online.plan(args(fixture, tasks_from_run=directory))
    assert p["tasks"] == p["evaluation_tasks"] == ["a", "b", "c"]
    assert p["task_selection_source"]["sha256"] != first
    assert not p["output"].exists()


def test_result_directory_can_include_unfinished_trials_and_limit_selection(fixture):
    directory = fixture[1] / "old-job"
    (directory / "a__unfinished").mkdir(parents=True)
    (directory / "artifacts").mkdir()
    p = online.plan(args(fixture, tasks_from_run=directory, limit=1, eval_task=[["c"]]))
    assert p["tasks"] == ["a"] and p["evaluation_tasks"] == ["c"]


@pytest.mark.parametrize("problem", ["empty", "unknown", "mismatch"])
def test_invalid_result_directories_fail_before_execution(fixture, problem):
    directory = fixture[1] / "old-job"
    directory.mkdir()
    if problem == "unknown":
        atomic_json(directory / "unknown__trial/result.json", {"task_name": "unknown"})
    elif problem == "mismatch":
        atomic_json(directory / "a__trial/result.json", {"task_name": "b"})
    with pytest.raises(ValueError):
        online.plan(args(fixture, tasks_from_run=directory))


@pytest.mark.parametrize("selector", ["inline", "file"])
def test_result_directory_is_mutually_exclusive_with_other_task_sources(fixture, selector):
    selected = args(fixture, tasks=[["a"]] if selector == "inline" else None,
                    tasks_from_run=fixture[1] / "old-job",
                    task_file=fixture[1] / "tasks.txt" if selector == "file" else None)
    with pytest.raises(ValueError, match="请选择一种"):
        online.plan(selected)


def test_five_task_batches_hold_library_fixed_and_flush_final_partial(fixture):
    names = many_tasks(fixture, 22)
    p = online.plan(args(fixture, batch_size=5))
    runtime = SimulatedRuntime(p)
    code, manifest = execute(p, runtime)
    assert code == 0
    assert list(map(len, runtime.batch_calls)) == [5, 5, 5, 5, 2]
    assert [task for batch in runtime.batch_calls for task in batch] == names
    assert [name for name, skills in runtime.used[:5] if skills] == []
    assert all(skills == ["cancel-pending-work"] for _, skills in runtime.used[5:])
    first = [row["library_before"] for row in manifest["tasks"][:5]]
    assert all(state == first[0] for state in first)
    assert manifest["batches"][0]["library_after"] == manifest["tasks"][5]["library_before"]
    assert len(manifest["batches"]) == 5
    first_curation = next(i for i, event in enumerate(runtime.events) if event[0] == 'learn')
    assert runtime.events[:first_curation] == [(stage, task) for task in names[:5] for stage in ('execute', 'graded', 'reflect')]
    assert runtime.events[first_curation:first_curation+5] == [('learn', task) for task in names[:5]]


def test_task_reflection_failure_does_not_change_scores_or_skip_next_tasks(fixture):
    p = online.plan(args(fixture, batch_size=5))
    runtime = SimulatedRuntime(p, fail_reflection='a')
    code, manifest = execute(p, runtime)
    assert code == 1 and manifest['status'] == 'completed_with_learning_errors'
    assert [task for stage, task in runtime.events if stage == 'reflect'] == ['a', 'b', 'c']
    assert all(row['reward'] == 1 for row in manifest['tasks'])
    assert manifest['tasks'][0]['reflection']['returncode'] == 1


@pytest.mark.parametrize('mutation', ['library', 'score'])
def test_reflection_cannot_update_library_or_original_scores(fixture, mutation):
    p = online.plan(args(fixture, batch_size=5))
    runtime = SimulatedRuntime(p)
    original = runtime.reflect_task
    def mutate(task, trial, output):
        result = original(task, trial, output)
        if mutation == 'library':
            atomic_json(p['output']/'library/illegal-write.json', {})
        else:
            (trial/'result.json').write_text('{}')
        return result
    runtime.reflect_task = mutate
    p['output'].mkdir()
    with pytest.raises(ValueError, match='修改了'):
        online.run_sequence(p, {'tasks': []}, runtime)
    assert ('execute', 'b') not in runtime.events


@pytest.mark.parametrize("size", [0, -1, True])
def test_invalid_batch_sizes_are_rejected(fixture, size):
    with pytest.raises(ValueError, match="batch-size"):
        online.plan(args(fixture, batch_size=size))


def test_batch_cost_is_counted_once_not_per_task(fixture):
    p = online.plan(args(fixture, batch_size=5))
    runtime = SimulatedRuntime(p)
    original = runtime.learn_batch
    def charged(rows, output):
        result = original(rows, output)
        result["cost"] = {"total": {"status": "reported", "input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
        return result
    runtime.learn_batch = charged
    original_reflect = runtime.reflect_task
    def charged_reflect(*args):
        result = original_reflect(*args)
        result['cost'] = {'total': {'status': 'reported', 'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3},
                          'categories': {'proposer': {'status': 'reported', 'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3}}}
        return result
    runtime.reflect_task = charged_reflect
    execute(p, runtime)
    summary = json.loads((p["output"] / "summary.json").read_text())
    assert summary["learning_usage"]["total_tokens"] == 21
    assert summary['reflection_usage']['total_tokens'] == 9
    assert summary['curation_usage']['total_tokens'] == 12
    assert summary["completed_batches"] == 1


@pytest.mark.parametrize('tokens,status', [(None, 'unavailable'), (5000, 'estimated')])
def test_summary_preserves_unknown_unlimited_injection_tokens(fixture, tokens, status):
    p = online.plan(args(fixture))
    p['output'].mkdir()
    manifest = {'status': 'completed', 'tasks': [{'reward': 1, 'skill_usage': {
        'injections': [{'skills': ['executor/cli/inspect-failures'], 'tokens': tokens}]}}]}
    online.write_summary(p, manifest)
    summary = json.loads((p['output']/'summary.json').read_text())
    assert summary['skill_injection_tokens'] == tokens
    assert summary['skill_injection_token_status'] == status


@pytest.mark.parametrize("roles", ["executor", "none"])
def test_final_evaluation_rechecks_every_task_without_learning_or_overwriting_training(fixture, roles):
    p = online.plan(args(fixture, roles=roles, batch_size=5))
    runtime = SimulatedRuntime(p)
    code, manifest = execute(p, runtime)
    assert code == 0
    before = {row["trial"]: online.score_hashes(Path(row["trial"])) for row in manifest["tasks"]}
    batch_count = len(runtime.batch_calls)
    assert online.run_final_evaluation(p, manifest, runtime) == 0
    final = manifest["final_evaluation"]
    assert [row["task_id"] for row in final["tasks"]] == p["tasks"]
    assert final["experiment"] == "training_set_recheck" and final["held_out_evaluation"] is False
    assert final["learning_enabled"] is False
    assert all(row["learning"]["attempted"] is False for row in final["tasks"])
    assert len(runtime.batch_calls) == batch_count
    assert [task for stage, task in runtime.events if stage == "evaluate"] == p["tasks"]
    for trial, hashes in before.items():
        assert online.score_hashes(Path(trial)) == hashes
    settings = tomllib.loads((p["output"] / "evaluation/config.toml").read_text())["skills"]
    assert settings["phase"] == "test" and settings["training"]["enabled"] is False
    if p["roles"]:
        assert SkillLibrary(p["output"] / "library").verify_frozen() == final["frozen_library"]
    assert json.loads((p["output"] / "evaluation/summary.json").read_text())["learning_usage"]["total_tokens"] == 0


def test_final_evaluation_detects_frozen_library_mutation(fixture):
    p = online.plan(args(fixture, batch_size=5))
    runtime = SimulatedRuntime(p)
    _, manifest = execute(p, runtime)
    original = runtime.evaluation_runtime
    def invalid(settings):
        evaluation = original(settings)
        run = evaluation.run_task
        def mutate(*args):
            result = run(*args)
            (p["output"] / "library/stats.json").write_text("{}")
            return result
        evaluation.run_task = mutate
        return evaluation
    runtime.evaluation_runtime = invalid
    with pytest.raises(ValueError, match="frozen skill library was changed"):
        online.run_final_evaluation(p, manifest, runtime)


def test_final_evaluation_still_runs_after_learning_errors(fixture):
    p = online.plan(args(fixture, batch_size=5))
    runtime = SimulatedRuntime(p, fail_learning="a")
    code, manifest = execute(p, runtime)
    assert code == 1
    assert online.run_final_evaluation(p, manifest, runtime) == 0
    assert len(manifest["final_evaluation"]["tasks"]) == len(p["tasks"])
    assert manifest["status"] == "completed_with_learning_errors"


@pytest.mark.parametrize("skip", [False, True])
def test_default_command_uses_batch_five_and_final_evaluation(fixture, monkeypatch, skip):
    class Runtime(SimulatedRuntime):
        def __init__(self, p, args):
            super().__init__(p)
        def prepare(self):
            return {"fixture": True}
    monkeypatch.setattr(online, "HarborRuntime", Runtime)
    output = fixture[1] / "default-command"
    command = ["manager", "--task", "all", "--output", str(output)]
    if skip:
        command.append("--skip-final-eval")
    assert online.main(command) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["batch_size"] == 5
    assert len(manifest["batches"]) == 1 and manifest["batches"][0]["size"] == 3
    assert (output / "evaluation").exists() is not skip
    if not skip:
        assert manifest["final_evaluation"]["status"] == "completed"


def test_task_file_changes_training_and_default_evaluation_together(fixture):
    task_file = fixture[1] / "tasks.txt"
    task_file.write_text("# choose tasks\nb\na\n")
    p = online.plan(args(fixture, task_file=task_file))
    assert p["tasks"] == p["evaluation_tasks"] == ["b", "a"]
    first_hash = p["task_selection_source"]["sha256"]
    task_file.write_text("c\n")
    p = online.plan(args(fixture, task_file=task_file))
    assert p["tasks"] == p["evaluation_tasks"] == ["c"]
    assert p["task_selection_source"]["sha256"] != first_hash


@pytest.mark.parametrize("text", ["# empty\n", "a\na\n", "not-present\n"])
def test_invalid_task_files_fail_before_any_execution(fixture, text):
    task_file = fixture[1] / "tasks.txt"
    task_file.write_text(text)
    with pytest.raises(ValueError, match="task-file"):
        online.plan(args(fixture, task_file=task_file))


def test_task_file_and_inline_ids_are_mutually_exclusive(fixture):
    task_file = fixture[1] / "tasks.txt"
    task_file.write_text("a\n")
    with pytest.raises(ValueError, match="请选择一种"):
        online.plan(args(fixture, tasks=[["b"]], task_file=task_file))


def test_limit_also_limits_default_final_evaluation(fixture):
    p = online.plan(args(fixture, limit=2))
    assert p["tasks"] == p["evaluation_tasks"] == ["a", "b"]


def test_explicit_final_tasks_are_validated_and_executed_without_learning(fixture):
    p = online.plan(args(fixture, tasks=[["a", "b"]], eval_task=[["c"]], batch_size=5))
    runtime = SimulatedRuntime(p)
    _, manifest = execute(p, runtime)
    assert online.run_final_evaluation(p, manifest, runtime) == 0
    final = manifest["final_evaluation"]
    assert final["task_order"] == ["c"]
    assert final["training_task_overlap"] == []
    assert final["experiment"] == "post_training_evaluation"
    assert final["summary"]["selected_tasks"] == 1
    assert [task for stage, task in runtime.events if stage == "evaluate"] == ["c"]
    assert runtime.learned == ["a", "b"]


def test_invalid_final_task_is_rejected_before_training(fixture):
    with pytest.raises(ValueError, match="eval-task"):
        online.plan(args(fixture, eval_task=[["missing"]]))
