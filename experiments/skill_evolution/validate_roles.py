"""Real local Adapter smoke checks for every role (no desktop benchmark).

Run only after local_pair finishes, to avoid competing with the paired runs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from lh_harness.cli import _build_agent, _resolve_role_model, _resolve_role_option
from lh_harness.config import load_run_defaults
from lh_harness.environment.local import LocalEnvironment
from lh_harness.skill_experiment import save
from lh_harness.types import EpisodeBudget


async def validate(args):
    values = load_run_defaults(args.config)
    settings = argparse.Namespace(**values)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    workspace = root / "workspace"
    workspace.mkdir()
    marker = workspace / "probe.txt"
    marker.write_text("LLH_LOCAL_ROLE_OK\n")
    before = hashlib.sha256(marker.read_bytes()).hexdigest()
    env = LocalEnvironment(tmp_dir=str(root / "tmp"))
    records = []
    for role in ("manager", "gui_executor", "cli_executor", "gui_auditor", "cli_auditor"):
        backend = _resolve_role_option(settings, role, "agent")
        model = _resolve_role_model(settings, role)
        adapter = _build_agent(backend, role=role, model=model, api_key=None, base_url=settings.base_url,
            workspace_path=str(workspace), prompt_dir=str(root / "prompts"),
            hidden_paths=(str(root / "prompts"),), context_length=values.get("local_model", {}).get("context_length"))
        prompt = "Use a read-only shell command to read probe.txt. Reply only with its exact contents. Do not write any file."
        result = await adapter.run_episode(prompt, env, EpisodeBudget(120), live_trajectory_path=str(root / f"{role}.jsonl"))
        value = {"role": role, "backend": backend, "model_id": model, "base_url": settings.base_url,
                 "status": result.status, "usage": result.metadata.get("usage"),
                 "marker_returned": "LLH_LOCAL_ROLE_OK" in str(result.metadata.get("assistant_visible_output")),
                 "workspace_unchanged": sorted(p.name for p in workspace.iterdir()) == ["probe.txt"] and hashlib.sha256(marker.read_bytes()).hexdigest() == before,
                 "tool_execution_observed": '"command_execution"' in result.actions_log,
                 "command": result.metadata.get("command"), "error": result.error}
        records.append(value)
        save(root / "roles.json", {"scope": "local model and read-only shell; no visual desktop actions", "roles": records})
        print(json.dumps(value, ensure_ascii=False), flush=True)
        if result.status != "done" or not value["marker_returned"] or not value["workspace_unchanged"]:
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return asyncio.run(validate(args))


if __name__ == "__main__":
    raise SystemExit(main())
