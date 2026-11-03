"""Offline Claude Code setup keeps Harbor's process-tool prerequisite."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor_agent import CuaHarnessClaudeCodeAgent  # noqa: E402
from lh_harness.harbor_training import SkillLearningHarborAgent  # noqa: E402


@pytest.mark.parametrize("has_procps", [True, False])
def test_offline_setup_installs_procps_only_when_missing(monkeypatch, has_procps: bool) -> None:
    calls: list[str] = []

    class Environment:
        async def exec(self, *, command: str, **kwargs):
            calls.append(command)
            return SimpleNamespace(return_code=0 if has_procps or "apt-get" in command else 1,
                                   stdout="", stderr="")

    async def parent_setup(self, environment) -> None:
        calls.append("parent_setup")

    monkeypatch.setattr(CuaHarnessClaudeCodeAgent, "setup", parent_setup)
    agent = object.__new__(SkillLearningHarborAgent)
    agent.offline_claude_binary = True
    asyncio.run(agent.setup(Environment()))

    assert calls[-1] == "parent_setup"
    assert any("command -v ps" in command for command in calls)
    assert any("apt-get install -y procps" in command for command in calls) is not has_procps
