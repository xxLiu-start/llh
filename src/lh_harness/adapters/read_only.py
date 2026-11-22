"""Independent read-only guard for CLIs without a native role sandbox."""
from __future__ import annotations

from ..provider_errors import GUARD_REJECTION_MESSAGE
from ..types import EpisodeResult
from .claude_permissions import snapshot_workspace, workspace_snapshot_diff


class ReadOnlyGuardAdapter:
    def __init__(self, adapter, *, excluded_paths=()):
        self.adapter = adapter
        self.excluded_paths = tuple(excluded_paths)

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
        before = snapshot_workspace(self.workspace_path, self.excluded_paths)
        if before.errors:
            return EpisodeResult(status="error", error=GUARD_REJECTION_MESSAGE,
                metadata={"verifier_workspace_snapshot_errors": list(before.errors),
                          "verifier_workspace_guard": True, "model_call_started": False})
        result = await self.adapter.run_episode(prompt, env, budget, live_trajectory_path=live_trajectory_path)
        after = snapshot_workspace(self.workspace_path, self.excluded_paths)
        diff = workspace_snapshot_diff(before, after)
        # This is a detection guard, matching the existing auditor metadata
        # contract. Never claim to have restored modified user files.
        diff["verifier_workspace_restore_on_mutation"] = False
        result.metadata.update(diff)
        result.metadata["verifier_guard_exclude_paths"] = list(self.excluded_paths)
        if diff["verifier_workspace_mutation_detected"] or diff["verifier_workspace_snapshot_errors"]:
            if result.status == "done":
                result.status = "error"
            result.error = (result.error + "\n" if result.error else "") + GUARD_REJECTION_MESSAGE
        return result
