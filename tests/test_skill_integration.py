from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from lh_harness.skills.integration import SkillSession
from lh_harness.skills.library import SkillLibrary
from lh_harness.skills.schema import Evidence, Skill


def _active_skill():
    return Skill(
        name="inspect-before-retry",
        role="executor",
        modality="cli",
        description="Inspect the current failure before retrying.",
        trigger=["inspect", "retry"],
        preconditions=["A prior operation failed"],
        procedure=["Inspect the latest error before retrying"],
        verification=["Confirm the retry changes the observed state"],
        pitfalls=["Do not repeat an unchanged failing command"],
        generalization="Transfer the diagnostic loop to other command workflows.",
        evidence=[
            Evidence("task-a", "fixture", "run-a", "audit-a", "Independent audit evidence for task A", "a" * 64, "fingerprint-a"),
            Evidence("task-b", "fixture", "run-b", "audit-b", "Independent audit evidence for task B", "b" * 64, "fingerprint-b"),
        ],
        status="active",
    )


class NativeStreamingAdapter:
    def __init__(self):
        self.prompt = None

    async def run_episode(self, prompt, env, budget, *, live_trajectory_path=None):
        self.prompt = prompt
        path = Path(live_trajectory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "type": "assistant",
            "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash",
                         "input": {"command": "inspect the current failure"}}],
        }) + "\n", encoding="utf-8")
        return SimpleNamespace(status="done", actions_log=path.read_text(), metadata={}, duration_ms=10)


async def test_skill_session_persists_native_live_trajectory_link(tmp_path):
    library_root = tmp_path / "library"
    library = SkillLibrary(library_root)
    library.put(_active_skill())

    settings = {
        "enabled": True,
        "library_dir": str(library_root),
        "phase": "online",
        "training": {"enabled": False, "train_roles": []},
        "usage": {"enabled": True, "use_roles": ["executor"]},
    }
    session = SkillSession(settings, tmp_path / "run")
    adapter = NativeStreamingAdapter()
    wrapped = session.wrap("cli_executor", adapter)
    trajectory = tmp_path / "run" / "role_orchestration" / "round_001" / "executor_raw_trajectory.jsonl"

    await wrapped.run_episode("inspect the failure and retry", object(), object(),
                              live_trajectory_path=str(trajectory))

    ledger = json.loads((tmp_path / "run" / "skill_usage.json").read_text())
    injection = next(item for item in ledger["injections"] if item["skills"])
    assert injection["skills"] == [_active_skill().key]
    assert injection["trajectory_path"] == str(trajectory.resolve())
    assert injection["trajectory_status"] == "done"
