"""Probe original LLH role CLIs in a disposable task container, without scoring."""
from __future__ import annotations

import argparse
import base64
import asyncio
import json
import os
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .eval_inspect_tb2_original import DEFAULT_CONFIG, ROOT, configure_installation, resolved_config
from .inspect_tb2_dataset import DEFAULT_OUTPUT, digest
from .token_usage import from_trajectory


async def probe(config_path: Path, url: str, output: Path, *, with_image: bool = False) -> dict:
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths
    from harbor_backends.environment import HarborEnvironmentAdapter
    from harness_types import EpisodeBudget
    from .original_harness_eval import OriginalHarnessHarborAgent

    task_path = ROOT / DEFAULT_OUTPUT / "portfolio-optimization"
    task = Task(task_path)
    config = resolved_config(config_path, task_path.parent)
    env_vars = dict(os.environ)
    installation = configure_installation(config, env_vars)
    if installation["mode"] != "host_cache":
        raise ValueError("this probe requires a verified cached Claude Code binary")
    os.environ["LH_CLAUDE_OFFLINE_BINARY"] = env_vars["LH_CLAUDE_OFFLINE_BINARY"]
    from . import verifier_runtime
    runtime = verifier_runtime.configure(config, env_vars, task_path.parent, [task_path.name])
    for key in ("LLH_VERIFIER_RUNTIME", "LLH_VERIFIER_SUPPORT"):
        os.environ[key] = env_vars[key]
    paths = TrialPaths(trial_dir=output)
    paths.mkdir()
    environment = DockerEnvironment(environment_dir=task_path / "environment", environment_name="original-qwen-role-probe",
                                    session_id=f"qwen-role-probe-{uuid.uuid4().hex[:10]}", trial_paths=paths,
                                    task_env_config=task.config.environment,
                                    extra_docker_compose=config["environment"]["extra_docker_compose"])
    extra_env = {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "local",
                 "LH_HARNESS_PROXY_BEARER_AUTH": "0", "LH_HARNESS_PROXY_OPENAI_BRIDGE": "0",
                 "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    declaration = config["agents"][0]
    agent = OriginalHarnessHarborAgent(logs_dir=paths.agent_dir, model_name=declaration["model_name"],
                                       base_url=url, api_key="local", extra_env=extra_env, **declaration["kwargs"])
    record = {"status": "running", "checked_at": datetime.now(timezone.utc).isoformat(), "benchmark_started": False,
              "scope": "original role CLI/tool/SSE/usage probe; no benchmark task is solved or scored",
              "model_id": declaration["model_name"], "base_url": url, "installation": installation,
              "verifier_runtime": runtime,
              "sampling": agent._claude_sampling_config, "roles": {},
              "original_sources": {name: digest(ROOT / "eval/TB-harness/Harness/src" / name)
                                   for name in ("orchestrator.py", "role_prompts.py", "harbor_agent.py", "verifier_agent.py")}}
    try:
        await environment.start(force_build=False)
        with environment.scoped_exec_env(extra_env):
            await agent.setup(environment)
            isolated = await environment.exec(command=f"test ! -e {shlex.quote(str(verifier_runtime.DEFAULT_ROOT / 'system/assets'))}",
                                              user="root", timeout_sec=30)
            record["hidden_test_assets_absent_during_agent_phase"] = isolated.return_code == 0
            if isolated.return_code:
                raise RuntimeError("hidden upstream verifier assets must not be mounted during agent execution")
            nonce = uuid.uuid4().hex
            seeded = await environment.exec(command=f"printf '%s' '{nonce}' > /app/llh_endpoint_probe.txt", timeout_sec=30)
            if seeded.return_code:
                raise RuntimeError("could not seed the isolated probe workspace")
            bridge = HarborEnvironmentAdapter(environment, workspace_path="/app")
            roles = agent._make_role_agents(environment=environment, env_adapter=bridge, output_dir=paths.agent_dir,
                                            model=declaration["model_name"], verifier_model=declaration["model_name"],
                                            api_key="local", base_url=url, workspace_path="/app", event_logger=None)
            checks = list(roles.items())
            if with_image:
                from .local_models import probe_image_base64
                image = output / "endpoint_probe.png"
                image.write_bytes(base64.b64decode(probe_image_base64()))
                await environment.upload_file(image, "/app/llh_endpoint_probe.png")
                checks.append(("cli_task_image", roles["cli_task"]))
            for name, role in checks:
                tool = "Bash" if name == "cli_task" else "Read"
                expected = nonce
                prompt = (f"Endpoint compatibility check only. Use the {tool} tool to read /app/llh_endpoint_probe.txt "
                          "and reply with exactly its contents. Do not modify any file or solve any other task.")
                if name == "cli_verifier":
                    prompt = ("Endpoint compatibility check only. Use Read to read /app/llh_endpoint_probe.txt. "
                              "Audit only that the file can be read without modifications. Do not solve any other task. "
                              "After checking it, return a report beginning with these three exact lines:\n"
                              "状态: complete\n完整性: clean\n契约审计: aligned\n"
                              "Then include the exact file contents as your audit evidence.")
                if name == "cli_task_image":
                    prompt = ("Endpoint image compatibility check only. Use Read to open /app/llh_endpoint_probe.png "
                              "as an image and reply with its dominant color as a single English word. "
                              "Do not modify files or use any other tool.")
                    expected = "red"
                result = await role.run_episode(prompt, bridge, EpisodeBudget(max_turns=4, max_duration_seconds=90))
                episode = Path(result.metadata["episode_dir"])
                raw = (episode / "claude_stream.jsonl").read_text()
                calls = []
                for line in raw.splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    for block in (event.get("message") or {}).get("content", []):
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            calls.append(block.get("name"))
                requests = [json.loads(line) for line in (episode / "request_proxy_requests.jsonl").read_text().splitlines()]
                errors = [event for event in requests if event.get("event") in {"upstream_http_error", "upstream_error"}]
                forwarded = [event for event in requests if event.get("event") == "request"]
                policy_verified = bool(forwarded) and all(event["after"].get("thinking") == {"type": "disabled"} for event in forwarded)
                row = {"status": result.status, "exit_code": result.metadata.get("exit_code"), "tools": calls,
                       "nonce_verified": expected in result.metadata.get("assistant_visible_output", "").lower(),
                       "thinking_disabled": policy_verified, "http_errors": errors, "usage": from_trajectory(raw),
                       "read_only_mutation_detected": result.metadata.get("verifier_workspace_mutation_detected"),
                       "duration_ms": result.duration_ms}
                if name == "cli_verifier":
                    from verifier_agent import verify_report_from_episode_result
                    parsed = verify_report_from_episode_result(result, 1, require_contract_audit=True)
                    row["original_verifier_parser"] = {"status": parsed.status, "integrity": parsed.integrity_status,
                                                       "contract_audit": parsed.contract_audit_status}
                row["passed"] = (result.status == "done" and row["exit_code"] == 0 and row["nonce_verified"]
                                 and tool in calls and policy_verified and not errors
                                 and not row["read_only_mutation_detected"])
                if name == "cli_verifier":
                    row["passed"] &= row["original_verifier_parser"] == {
                        "status": "complete", "integrity": "clean", "contract_audit": "aligned"}
                record["roles"][name] = row
                print(json.dumps({"role": name, **row}), flush=True)
                if not row["passed"]:
                    raise RuntimeError(f"real CLI role probe failed: {name}; inspect {episode}")
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        raise
    finally:
        try:
            await environment.stop(delete=True)
        finally:
            (output / "role_probe.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-url", required=True, help="Docker-reachable unauthenticated local Messages endpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "output/original-qwen-role-check" / uuid.uuid4().hex[:10])
    parser.add_argument("--with-image", action="store_true", help="Also exercise the actual Claude Code Read image tool and visual inference.")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = asyncio.run(probe(args.config, args.base_url, args.output.resolve(), with_image=args.with_image))
    except Exception as exc:
        print(f"Role check failed: {exc}", flush=True)
        return 2
    print(f"Real original LLH role probes: {result['status']}; {args.output / 'role_probe.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
