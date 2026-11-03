"""Harbor adapter for evaluating the original LongHorizon-Harness.

This class only adds offline Claude Code installation support. Task execution,
prompts, role orchestration, and completion decisions remain those of the
original ``CuaHarnessClaudeCodeAgent`` from ``eval/TB-harness/Harness``.
"""
from __future__ import annotations

from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor_agent import CuaHarnessClaudeCodeAgent


class OriginalHarnessHarborAgent(CuaHarnessClaudeCodeAgent):
    """Original Harness agent with an optional host-cached Claude binary."""

    def __init__(self, *args: Any, offline_claude_binary: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.offline_claude_binary = bool(offline_claude_binary)

    async def setup(self, environment: BaseEnvironment) -> None:
        if self.offline_claude_binary:
            check = await environment.exec(
                command="command -v ps >/dev/null 2>&1 && command -v pgrep >/dev/null 2>&1",
                user="root",
                timeout_sec=30,
            )
            if check.return_code != 0:
                install = await environment.exec(
                    command="apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y procps",
                    user="root",
                    timeout_sec=600,
                )
                if install.return_code != 0:
                    raise RuntimeError(
                        f"could not install procps for offline Claude Code: {install.stdout} {install.stderr}"
                    )
        await super().setup(environment)
