"""Harbor adapter for evaluating the original LongHorizon-Harness.

This class adds offline installation and explicit CLI event-role normalization. Task execution,
prompts, role orchestration, and completion decisions remain those of the
original ``CuaHarnessClaudeCodeAgent`` from ``eval/TB-harness/Harness``.
"""
from __future__ import annotations

from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor_agent import CuaHarnessClaudeCodeAgent
from .claude_stream_compat import ClaudeStreamCompatibilityAdapter


class OriginalHarnessHarborAgent(CuaHarnessClaudeCodeAgent):
    """Original Harness agent with an optional host-cached Claude binary."""

    def __init__(self, *args: Any, offline_claude_binary: bool = False, verifier_system_root: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.offline_claude_binary = bool(offline_claude_binary)
        self.verifier_system_root = verifier_system_root

    def _make_episode_adapter(self, **kwargs):
        return ClaudeStreamCompatibilityAdapter(super()._make_episode_adapter(**kwargs))

    async def setup(self, environment: BaseEnvironment) -> None:
        task = None
        if self.verifier_system_root:
            import json
            import shlex
            from pathlib import Path
            systems = json.loads((Path(self.verifier_system_root) / "system/manifest.json").read_text())
            image = environment.task_env_config.docker_image
            task = next((name for name, spec in systems["images"].items() if spec["spec"]["image"] == image), None)
            if not task:
                raise RuntimeError("offline system profile does not match the task image; prepare before benchmarking")
            from .verifier_system import install_asset_upload
            install_asset_upload(environment, task, Path(self.verifier_system_root))
            identity = await environment.exec(command=f"printf '%s' {shlex.quote(task)} > /tmp/.llh-verifier-task", user="root", timeout_sec=30)
            if identity.return_code:
                raise RuntimeError("could not record the isolated verifier dependency profile")
        if self.offline_claude_binary:
            check = await environment.exec(
                command="command -v ps >/dev/null 2>&1 && command -v pgrep >/dev/null 2>&1 && (command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1)",
                user="root",
                timeout_sec=30,
            )
            if check.return_code != 0:
                if not self.verifier_system_root:
                    raise RuntimeError("procps/Python is missing; prepare the offline verifier runtime before agent setup")
                import shlex
                from .verifier_runtime import SUPPORT
                packages = systems["images"][task]["setup_packages"]
                command = (f"export LLH_VERIFIER_RUNTIME={shlex.quote(self.verifier_system_root)} "
                           f"LLH_VERIFIER_SUPPORT={shlex.quote(str(SUPPORT))} LLH_VERIFIER_TASK={shlex.quote(task)}; "
                           f"{shlex.quote(str(SUPPORT / 'apt-get'))} install -y {shlex.join(packages)}")
                install = await environment.exec(
                    command=command,
                    user="root",
                    timeout_sec=600,
                )
                if install.return_code != 0:
                    raise RuntimeError(
                        f"could not install procps for offline Claude Code: {install.stdout} {install.stderr}"
                    )
        await super().setup(environment)
