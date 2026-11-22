"""Run the existing original LLH Harbor agent on a-evolve's 89 TB2.0 tasks."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .harbor_url import claude_code_upstream_base_url
from .claude_cache import find_cached_claude_binary
from .inspect_tb2_dataset import DEFAULT_OUTPUT, DEFAULT_SOURCE, MANIFEST, digest, prepare_dataset
from .local_models import EndpointAuthenticationError, EndpointUnavailableError, base_url, check_endpoint, is_local
from . import verifier_runtime
from .serve_local_qwen import DEFAULT_MAX_IMAGES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "evaluation/original_harness_a_evolve_tb20_qwen27b.local.yaml"
DEFAULT_CLAUDE_CACHE = ROOT / ".cache/claude-code"


def resolved_config(path: Path, dataset: Path) -> dict:
    import yaml
    config = yaml.safe_load(path.read_text())
    if len(config.get("agents", [])) != 1:
        raise ValueError("one original LLH agent is required")
    agent = config["agents"][0]
    if agent.get("import_path") != "harbor_agent:CuaHarnessClaudeCodeAgent" or agent.get("skills"):
        raise ValueError("this runner requires the original LLH entry with Skills disabled")
    kwargs = agent.get("kwargs", {})
    model = agent.get("model_name")
    if not model or any(kwargs.get(key) != model for key in ("harness_model", "verifier_model")):
        raise ValueError("all three original LLH roles must use the same explicit local Model ID")
    if "override_timeout_sec" in agent or "override_timeout_sec" in config.get("verifier", {}):
        raise ValueError("do not override the source dataset's agent/verifier timeouts")
    if config.get("n_concurrent_trials", 1) != 1:
        raise ValueError("local baseline evaluation requires sequential task execution")
    if agent.get("env", {}).get("LH_HARNESS_PROXY_OPENAI_BRIDGE") != "0":
        raise ValueError("this configuration requires the native local Messages API")
    config["datasets"] = [{"path": str(dataset)}]
    return config


def model_url(env: dict, *, allow_loopback: bool = False) -> str:
    url = env.get("LLH_LOCAL_BASE_URL") or env.get("ANTHROPIC_BASE_URL")
    inherited = not env.get("LLH_LOCAL_BASE_URL") and bool(env.get("ANTHROPIC_BASE_URL"))
    if not url:
        if allow_loopback:
            url = "http://127.0.0.1:8010"
        else:
            probe = subprocess.run(["docker", "network", "inspect", "bridge", "--format",
                                    "{{(index .IPAM.Config 0).Gateway}}"], capture_output=True, text=True, check=True, timeout=15)
            address = ipaddress.ip_address(probe.stdout.strip())
            url = f"http://{address}:8010" if address.version == 4 else f"http://[{address}]:8010"
    url = claude_code_upstream_base_url(base_url(url))
    host = urlsplit(url).hostname
    if host in {"localhost", "127.0.0.1", "::1"} and not allow_loopback:
        raise ValueError("ANTHROPIC_BASE_URL must be reachable from Docker; localhost refers to the task container")
    private_name = host and ("." not in host or host.endswith((".local", ".lan", ".internal")))
    if inherited and not is_local(url) and not private_name:
        raise ValueError(
            f"inherited ANTHROPIC_BASE_URL points to {url}; refusing to select a public endpoint implicitly for a local run. "
            "Unset it to use the Docker gateway, or explicitly set --base-url / LLH_LOCAL_BASE_URL"
        )
    return url


def local_credentials(env: dict, scheme: str, key_env: str | None = None) -> tuple[str | None, str]:
    if scheme == "none":
        return None, "none"
    if key_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
            raise ValueError("--api-key-env must name an environment variable, not contain a key")
        if not env.get(key_env):
            raise ValueError("the credential variable named by --api-key-env is unset or empty")
        return env[key_env], key_env
    for name in ("LLH_LOCAL_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "VLLM_API_KEY"):
        if env.get(name):
            return env[name], name
    return None, "none"


def configure_auth(env: dict, config: dict, scheme: str, key_env: str | None) -> tuple[str | None, str]:
    key, source = local_credentials(env, scheme, key_env)
    flag = "1" if scheme == "bearer" else "0"
    declared = config["agents"][0].get("env", {}).get("LH_HARNESS_PROXY_BEARER_AUTH")
    if declared not in {flag, "${LLH_LOCAL_BEARER_AUTH:-0}", "${LLH_LOCAL_BEARER_AUTH:-1}"}:
        raise ValueError("configuration must declare LH_HARNESS_PROXY_BEARER_AUTH=${LLH_LOCAL_BEARER_AUTH:-0} to match preflight/runtime authentication")
    env["LLH_LOCAL_BEARER_AUTH"] = flag
    # The original Harbor adapter reads this env var. Never put the value into
    # YAML, argv, diagnostics or the experiment manifest.
    env["ANTHROPIC_API_KEY"] = key or "local"
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return key, source


def configure_installation(config: dict, env: dict, binary: Path | None = None) -> dict:
    """Reuse the existing installation-only adapter; keep original run() intact."""
    version = env.get("CLAUDE_CODE_VERSION") or "2.1.176"
    explicit = binary or env.get("LH_CLAUDE_OFFLINE_BINARY")
    cached = Path(explicit).expanduser().resolve() if explicit else find_cached_claude_binary(
        ROOT / "output/original-a-evolve-tb20-qwen27b", version, DEFAULT_CLAUDE_CACHE)
    if cached is None:
        raise ValueError(f"offline Claude Code {version} is not cached; prepare {DEFAULT_CLAUDE_CACHE / version / 'claude'} "
                         "or set LH_CLAUDE_OFFLINE_BINARY before benchmarking; no runtime network installation")
    if not cached.is_file() or not os.access(cached, os.X_OK):
        raise ValueError(
            f"offline Claude Code binary is missing or not executable: {cached}. "
            f"Set LH_CLAUDE_OFFLINE_BINARY to {DEFAULT_CLAUDE_CACHE / version / 'claude'}, "
            "or unset LH_CLAUDE_OFFLINE_BINARY to discover the verified local cache. "
            "Experiment output directories may have been cleaned; no alternative version is selected"
        )
    probe = subprocess.run([str(cached), "--version"], capture_output=True, text=True, timeout=15, check=False)
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", probe.stdout)
    if probe.returncode != 0 or not match or match.group(1) != version:
        raise ValueError(f"offline Claude Code binary does not match the required release {version}: {cached}")
    env["LH_CLAUDE_OFFLINE_BINARY"] = str(cached)
    compose = ROOT / "training/docker_claude_offline.yaml"
    config.setdefault("environment", {}).setdefault("extra_docker_compose", []).append(str(compose))
    agent = config["agents"][0]
    agent["import_path"] = "lh_harness.original_harness_eval:OriginalHarnessHarborAgent"
    agent.setdefault("kwargs", {})["offline_claude_binary"] = True
    return {"mode": "host_cache", "version": version, "binary": str(cached), "sha256": digest(cached),
            "compose": str(compose), "execution_entry": "harbor_agent:CuaHarnessClaudeCodeAgent",
            "scope": "offline installation and explicit CLI event-role normalization; original run, prompts, and verification unchanged",
            "stream_compatibility": "assistant event type supplies missing message.role; raw logs preserved"}


def write_runtime_config(config: dict, path: Path) -> None:
    import yaml
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def endpoint_profile(config: dict, *, require_image: bool = False) -> dict:
    kwargs = config["agents"][0]["kwargs"]
    profile = {"max_output_tokens": kwargs["max_tokens"], "minimum_context_length": 32768,
               "temperature": kwargs["temperature"], "top_p": kwargs["top_p"]}
    extra = kwargs.get("llm_call_kwargs", {}).get("extra_body", {})
    template = extra.get("chat_template_kwargs", {})
    reasoning = {}
    if "enable_thinking" in template:
        reasoning["enable_thinking"] = template["enable_thinking"]
    if kwargs.get("effort"):
        reasoning["effort"] = kwargs["effort"]
    if reasoning:
        profile["reasoning"] = reasoning
    if require_image:
        profile["require_image"] = True
        # Tool history retains earlier images. A single-image probe cannot
        # detect a server limit that fails once video frames accumulate.
        profile["image_probe_count"] = DEFAULT_MAX_IMAGES
    return profile


def serving_evidence(path: Path | None, check: dict, kwargs: dict) -> dict | None:
    if path is None:
        return None
    observed = json.loads(path.read_text())
    if not isinstance(observed, dict):
        raise ValueError("serving manifest must be a JSON object")
    for key in ("model_id", "base_url", "context_length", "serving_framework", "serving_version"):
        if observed.get(key) != check.get(key):
            raise ValueError(f"serving manifest {key} differs from the active endpoint; benchmark not started")
    for key in ("temperature", "top_p", "top_k"):
        if observed.get(key) != kwargs.get(key):
            raise ValueError(f"serving manifest {key} differs from the role configuration")
    if observed.get("max_output_tokens") != kwargs.get("max_tokens"):
        raise ValueError("serving manifest max_output_tokens differs from the role configuration")
    if check.get("image_probe") == "verified":
        declared = observed.get("modalities", {}).get("image")
        if type(declared) is not int or declared < check.get("image_probe_count", 1):
            raise ValueError("serving manifest image limit is missing or below the verified multi-image probe; refresh the serving snapshot")
    fields = ("checked_at", "model_source", "model_id", "base_url", "serving_framework", "serving_version", "weights",
              "weights_sha256", "quantization", "dtype", "gpu_hardware", "context_length", "max_output_tokens",
              "max_output_tokens_scope", "temperature", "top_p", "top_k", "reasoning", "tool_calling", "modalities",
              "max_num_seqs", "benchmark_concurrency")
    return {"path": str(path.resolve()), "sha256": digest(path),
            "scope": "operator-observed serving snapshot; endpoint identity/context/version checked at launch",
            "observed": {key: observed.get(key) for key in fields}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-output", type=Path, default=ROOT / DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--jobs-dir", type=Path, default=ROOT / "output/original-a-evolve-tb20-qwen27b")
    parser.add_argument("--job-name")
    parser.add_argument("--serving-manifest", type=Path, help="Attach a verified local serving snapshot, including GPU and modality settings, to this run's manifest.")
    parser.add_argument("--task", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-verifier-runtime", action="store_true", help="Download/cache the selected tasks' exact scoring dependencies before any model work.")
    parser.add_argument("--check-verifier-runtime", action="store_true", help="Verify scoring caches without starting a benchmark or contacting the model.")
    parser.add_argument("--verifier-runtime", type=Path, default=verifier_runtime.DEFAULT_ROOT)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--check-model-only", action="store_true", help="Check the selected local endpoint/auth without launching Harbor.")
    parser.add_argument("--check-model-docker", action="store_true", help="Check tools/SSE/usage/images from a disposable Docker container without launching the benchmark.")
    parser.add_argument("--offline-claude-binary", type=Path, help="Mount the exact Claude Code release read-only instead of downloading it. Defaults to LH_CLAUDE_OFFLINE_BINARY or a verified existing cache.")
    parser.add_argument("--install-only", action="store_true", help="Validate task environment and agent setup without solving or scoring tasks.")
    parser.add_argument("--base-url", help="Explicit local endpoint; loopback is allowed for --check-model-only, but Benchmark containers need a host-reachable address.")
    parser.add_argument("--auth-scheme", choices=("bearer", "x-api-key", "none"),
                        default=os.getenv("LLH_LOCAL_AUTH_SCHEME", "none"), help="This local service has no authentication by default; select bearer only if server authentication was explicitly enabled.")
    parser.add_argument("--api-key-env", help="Name of the environment variable containing the local API key; never pass its value here.")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_dataset(args.source, args.dataset_output, expected_tasks=89)
        config = resolved_config(args.config, args.dataset_output.resolve())
        if args.check_model_only and args.check_model_docker:
            raise ValueError("choose --check-model-only or --check-model-docker")
        names = {task["name"] for task in manifest["tasks"]}
        if args.task and set(args.task) - names:
            raise ValueError(f"unknown tasks: {sorted(set(args.task) - names)}")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.prepare_only:
            print(json.dumps({"dataset": str(args.dataset_output.resolve()), "tasks": len(names),
                              "benchmark_version": "2.0", "source_tests_unchanged": True}, indent=2))
            return 0
        selected = sorted(set(args.task) if args.task else names)
        if args.prepare_verifier_runtime or args.check_verifier_runtime:
            verify = verifier_runtime.prepare if args.prepare_verifier_runtime else verifier_runtime.check
            record = verify(args.dataset_output.resolve(), selected, args.verifier_runtime)
            print(json.dumps({**record, "benchmark_started": False}, indent=2))
            return 0
        harbor = str(Path(sys.executable).with_name("harbor"))
        if not Path(harbor).is_file():
            harbor = shutil.which("harbor")
        if not harbor:
            raise ValueError("Harbor is unavailable; activate terminal-bench-2-1")
        job_name = args.job_name or f"original-tb20-qwen27b-{uuid.uuid4().hex[:10]}"
        command = [harbor, "run", "-c", str(args.config.resolve()), "--path", str(args.dataset_output.resolve()),
                   "--jobs-dir", str(args.jobs_dir.resolve()), "--job-name", job_name]
        for task in args.task or []:
            command.extend(["--include-task-name", task])
        if args.limit:
            command.extend(["--n-tasks", str(args.limit)])
        if args.install_only:
            command.append("--install-only")
        env = dict(os.environ)
        if args.base_url:
            env["LLH_LOCAL_BASE_URL"] = args.base_url
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "eval/TB-harness/Harness/src"), env.get("PYTHONPATH", "")])
        if args.print_config:
            installation = configure_installation(config, env, args.offline_claude_binary)
            verifier_runtime.configure(config, env, args.dataset_output.resolve(), selected, args.verifier_runtime)
            with tempfile.TemporaryDirectory(prefix="llh-original-config-") as temporary:
                runtime_config = Path(temporary) / "config.yaml"
                write_runtime_config(config, runtime_config)
                command[3] = str(runtime_config)
                return subprocess.run([*command, "--print-config"], env=env, cwd=ROOT, check=False).returncode
        url = model_url(env, allow_loopback=True) if args.check_model_only else model_url(env)
        key, credential_source = configure_auth(env, config, args.auth_scheme, args.api_key_env)
        endpoint_source = "--base-url" if args.base_url else "LLH_LOCAL_BASE_URL" if os.getenv("LLH_LOCAL_BASE_URL") else "ANTHROPIC_BASE_URL" if os.getenv("ANTHROPIC_BASE_URL") else "host loopback (model-only check)" if args.check_model_only else "Docker bridge gateway"
        print(f"Model endpoint: {url} (source: {endpoint_source})", flush=True)
        print(f"Model ID: {config['agents'][0]['model_name']}", flush=True)
        print(f"Authentication: {args.auth_scheme}; credential variable: {credential_source} (values hidden)", flush=True)
        kwargs = config["agents"][0]["kwargs"]
        # Probe the same host address that the task containers will use. It
        # must support native Messages tool calls/SSE/usage/error handling.
        try:
            needs_image = not args.task or bool(set(args.task) & {"code-from-image", "video-processing"})
            check = check_endpoint(url, "claude_code", config["agents"][0]["model_name"], endpoint_profile(config, require_image=needs_image),
                api_key=key, auth_scheme=args.auth_scheme)
        except EndpointAuthenticationError as exc:
            if args.auth_scheme == "none":
                raise ValueError(
                    f"{exc}. This run is configured for an unauthenticated local service; local is only a client placeholder, "
                    "not a generated server key. Check the selected endpoint/gateway. "
                    "Use --auth-scheme bearer only if server authentication was explicitly enabled"
                ) from None
            raise ValueError(
                f"{exc}. Credential source: {credential_source}. Check the endpoint printed above; "
                "if the local server requires authentication, set LLH_LOCAL_API_KEY or use --api-key-env NAME. "
                "The client placeholder local is not a valid key for an authenticated server"
            ) from None
        except EndpointUnavailableError as exc:
            raise ValueError(f"{exc}. The model service is not ready; start vLLM and wait for /v1/models before evaluation") from None
        if args.check_model_only:
            print(json.dumps({"status": "ok", "base_url": check["base_url"], "model_id": check["model_id"],
                              "auth_scheme": args.auth_scheme, "credential_source": credential_source,
                              "usage": check["usage"], "benchmark_started": False,
                              "image_probe": check.get("image_probe", "not_requested"),
                              "image_probe_count": check.get("image_probe_count", 0),
                              "validation_scope": "host model endpoint only; Docker reachability is not tested"}, indent=2))
            return 0
        env["ANTHROPIC_BASE_URL"] = url
        host = urlsplit(url).hostname
        env["NO_PROXY"] = env["no_proxy"] = ",".join(filter(None, [env.get("NO_PROXY"), env.get("no_proxy"), "127.0.0.1", "localhost", host]))
        command.extend(["--allow-agent-host", host])
        installation = configure_installation(config, env, args.offline_claude_binary)
        evidence = serving_evidence(args.serving_manifest, check, kwargs)
        runtime = verifier_runtime.configure(
            config, env, args.dataset_output.resolve(), selected, args.verifier_runtime)
        from .model_runtime import Monitor, check_docker
        import tomllib
        image = tomllib.loads((args.dataset_output / selected[0] / "task.toml").read_text())["environment"]["docker_image"]
        container_check = check_docker(url, config["agents"][0]["model_name"], endpoint_profile(config, require_image=needs_image),
                                       image, args.verifier_runtime, api_key=key, auth_scheme=args.auth_scheme)
        if args.check_model_docker:
            print(json.dumps({**container_check, "benchmark_started": False}, indent=2))
            return 0
        print(f"Claude Code installation: {installation['mode']} (version {installation['version']})", flush=True)
        args.jobs_dir.mkdir(parents=True, exist_ok=True)
        runtime_config = args.jobs_dir / f"{job_name}.config.yaml"
        write_runtime_config(config, runtime_config)
        command[3] = str(runtime_config)
        original = ROOT / "eval/TB-harness/Harness/src"
        record = {"benchmark": manifest["dataset_label"], "task_count": len(names),
                  "selected_tasks": args.task, "limit": args.limit, "job_name": job_name,
                  "agent_entry": config["agents"][0]["import_path"], "skills_enabled": False,
                  "agent_installation": installation, "install_only": args.install_only,
                  "serving_evidence": evidence,
                  "verifier_runtime": runtime,
                  "model_endpoint_check": check, "dataset_manifest_sha256": digest(args.dataset_output / MANIFEST),
                  "docker_model_endpoint_check": container_check,
                  "authentication": {"scheme": args.auth_scheme, "credential_source": credential_source},
                  "source_tests_unchanged": True, "source_timeouts_preserved": True,
                  "sampling": {key: kwargs.get(key) for key in ("max_tokens", "temperature", "top_p", "top_k", "llm_call_kwargs")},
                  "original_sources": {name: digest(original / name) for name in ("orchestrator.py", "role_prompts.py", "harbor_agent.py", "verifier_agent.py")}}
        record_path = args.jobs_dir / f"{job_name}.manifest.json"
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"Original LLH / a-evolve TB2.0 / {len(names)} available tasks; manifest: {record_path}", flush=True)
        monitor = Monitor(url, config["agents"][0]["model_name"], api_key=key, auth_scheme=args.auth_scheme)
        return verifier_runtime.run_harbor(command, env, ROOT, args.jobs_dir.resolve() / job_name, model_monitor=monitor)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Evaluation not started: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
