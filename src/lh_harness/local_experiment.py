"""Local-model experiment plumbing outside the original evaluation kernel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .local_models import EndpointClient, EndpointError, check_endpoint, fingerprint, is_local
from .skill_experiment import ExperimentSession, save

ROLE_NAMES = ("manager", "gui_executor", "cli_executor", "gui_auditor", "cli_auditor", "auditor", "final_response")


def preflight(args) -> dict:
    from .cli import _resolve_role_model, _resolve_role_option, _resolve_role_reasoning_effort
    profile = getattr(args, "local_model", {}) or {}
    url = getattr(args, "base_url", None)
    if not (profile.get("enabled", bool(profile)) or is_local(url)):
        return {}
    if not url:
        raise EndpointError("local model mode requires an explicit base_url; cloud defaults are forbidden")
    if profile.get("upstream_base_url"):
        client = EndpointClient(url, args.agent)
        client.url = client.url.removesuffix("/v1")
        policy = client.json("/llh-policy")
        if policy.get("policy") != profile or policy.get("upstream_base_url") != profile["upstream_base_url"].rstrip("/"):
            raise EndpointError("inference policy proxy does not match this experiment configuration")
    checked, result = {}, {}
    for role in ROLE_NAMES:
        backend = _resolve_role_option(args, role, "agent")
        if backend == "deepseek_harness":
            raise EndpointError(
                "DeepSeek Harness headless runner buffers stdout and cannot satisfy required real-time streaming; "
                "its local Chat API may be compatible, but a separately validated streaming adapter is required before Benchmark startup; no fallback"
            )
        model = _resolve_role_model(args, role)
        if not model:
            raise EndpointError(f"{role}: explicit local Model ID required; backend defaults are forbidden")
        key = backend, model
        if key not in checked:
            checked[key] = check_endpoint(url, backend, model, profile, api_key=getattr(args, "api_key", None))
        result[role] = {**checked[key], "reasoning_effort": _resolve_role_reasoning_effort(args, role, backend)}
    return result


def write_manifest(args, log_dir: str, checks: dict) -> dict:
    roles = {role: {**value, "configuration_fingerprint": fingerprint(value)} for role, value in checks.items()}
    root = Path(__file__).parent
    protected = [root / name for name in ("manager.py", "role_prompts.py", "auditor_agent.py")]
    protected += list((root.parent.parent / "evaluation").glob("*.yaml"))
    manifest = {"schema_version": 1, "model_source": "local", "roles": roles,
                "skills": getattr(args, "skills_config", {"enabled": False}),
                "inference_policy": getattr(args, "local_model", {}), "token_accounting": "reported_only",
                "evaluation_sources": {str(p.relative_to(root.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in protected if p.is_file()},
                "metadata_note": "null means unknown; operator-declared metadata is not independent attestation",
                "resource_schedule": "use local_pair for sequential runs under a shared endpoint lock"}
    runtime_files = [root / name for name in ("cli.py", "local_models.py", "local_gateway.py", "local_experiment.py",
                     "local_pair.py", "skill_experiment.py", "token_usage.py", "config.py")]
    runtime_files += list((root / "adapters").glob("*.py"))
    manifest["runtime_sources"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in runtime_files}
    save(Path(log_dir) / "experiment_manifest.json", manifest)
    return manifest


async def execute(*, task, config, role_agents, args, env, trainers, **kwargs):
    from .manager import run
    settings = getattr(args, "skills_config", {}) or {}
    if "library_dir" in settings:
        from .skills.integration import SkillSession
        session = SkillSession(settings, config.log_dir)
        wrapped = {key: session.wrap(key.removesuffix("_agent"), value) for key, value in role_agents.items()}
        report = await run(task=task, config=config, env=env, **wrapped, **kwargs)
        session.write()
        # Learning is an explicit offline command; no proposer/curator runs
        # inside task execution, even if this role is enabled for training.
        return report
    session = ExperimentSession(getattr(args, "skills_config", {"enabled": False}) or {}, config.log_dir,
                                (getattr(args, "local_model", {}) or {}).get("tokenizer_path"))
    wrapped = {key: session.wrap(key.removesuffix("_agent"), value) for key, value in role_agents.items()}
    report = await run(task=task, config=config, env=env, **wrapped, **kwargs)
    # The result of MEA and Verifier is finalized before optional training.
    await session.finish(report, trainers, env, config.manager_budget)
    return report
