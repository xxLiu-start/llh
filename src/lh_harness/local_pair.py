"""Sequential, locked dual-track launcher with strict model parity checks."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .cli import _resolve_role_model, _resolve_role_option, _resolve_role_reasoning_effort
from .config import load_run_defaults
from .local_experiment import ROLE_NAMES, preflight
from .local_models import EndpointError, fingerprint
from .skill_experiment import save


def check_parity(baseline: dict, variant: dict):
    left = {k: v for k, v in baseline.items() if k != "skills_config"}
    right = {k: v for k, v in variant.items() if k != "skills_config"}
    if left != right:
        changed = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
        raise EndpointError(f"paired experiments differ outside Skills: {changed}")
    if baseline.get("skills_config", {}).get("enabled"):
        raise EndpointError("baseline must have Skills disabled")
    if not variant.get("skills_config", {}).get("enabled"):
        raise EndpointError("variant must have Skills enabled")


def checked_configuration(values: dict):
    namespace = argparse.Namespace(reasoning_effort=None, **{k: v for k, v in values.items() if k != "reasoning_effort"})
    namespace.reasoning_effort = values.get("reasoning_effort")
    checks = preflight(namespace)
    if not checks:
        raise EndpointError("paired local experiment requires explicit local model configuration")
    return checks


def require_evidence(checks: dict):
    for role, checked in checks.items():
        required = ("weights", "weights_sha256", "quantization", "gpu_hardware", "serving_version",
                    "context_length", "max_output_tokens", "temperature", "top_p", "reasoning", "tool_calling", "concurrency")
        missing = [k for k in required if checked.get(k) is None]
        if missing:
            raise EndpointError(f"{role}: actual run metadata is incomplete: {missing}; record evidence before claiming a fair comparison")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="experiments/skill_evolution/configs/baseline.local.toml")
    parser.add_argument("--variant", default="experiments/skill_evolution/configs/all_roles.local.toml")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true", help="Allow explicitly unverified hardware/weights metadata for connectivity smoke tests only.")
    parser.add_argument("--max-rounds", type=int)
    args = parser.parse_args(argv)
    try:
        left, right = load_run_defaults(args.baseline), load_run_defaults(args.variant)
        check_parity(left, right)
        if args.max_rounds is not None:
            left["max_rounds"] = right["max_rounds"] = args.max_rounds
        endpoint = left.get("local_model", {}).get("upstream_base_url", left.get("base_url", ""))
        lock_id = hashlib.sha256(endpoint.encode()).hexdigest()[:24]
        lock_path = Path(tempfile.gettempdir()) / f"lh-harness-model-{lock_id}.lock"
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checks = checked_configuration(left)
            if not args.smoke:
                require_evidence(checks)
            initial = {k: fingerprint(v) for k, v in checks.items()}
            record = {"model_source": "local", "schedule": "sequential", "smoke_only": args.smoke,
                      "fair_comparison_verified": False, "baseline": str(Path(args.baseline).resolve()),
                      "variant": str(Path(args.variant).resolve()), "initial_roles": checks, "runs": []}
            save(output / "comparison_manifest.json", record)
            source_checks = None
            for label, path, values in (("baseline", args.baseline, left), ("variant", args.variant, right)):
                current = checked_configuration(values)
                if {k: fingerprint(v) for k, v in current.items()} != initial:
                    raise EndpointError("model/service configuration changed between paired runs")
                cmd = [sys.executable, "-m", "lh_harness", "run", "--config", str(Path(path).resolve()),
                       "--task", args.task, "--runs-root", str(output / label / "runs"),
                       "--workspace", str(output / label / "workspace"), "--no-dashboard"]
                if args.max_rounds is not None:
                    cmd.extend(["--max-rounds", str(args.max_rounds)])
                with (output / f"{label}.log").open("w") as log:
                    completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                manifests = list((output / label).rglob("experiment_manifest.json"))
                if not manifests:
                    raise EndpointError(f"{label} did not start: inspect {label}.log; second track will not launch")
                manifest = json.loads(manifests[0].read_text())
                actual_sources = {key: manifest.get(key) for key in ("runtime_sources", "evaluation_sources")}
                if source_checks is not None and source_checks != actual_sources:
                    raise EndpointError("runtime/evaluation sources changed between the paired runs")
                source_checks = actual_sources
                usage_path = manifests[0].with_name("token_usage.json")
                categories = json.loads(usage_path.read_text())["categories"] if usage_path.is_file() else {}
                if args.smoke and any(categories.get(role, {}).get("calls", 0) < 1 for role in ("manager", "executor", "auditor")):
                    raise EndpointError(f"{label}: smoke did not exercise every MEA role; inspect the report")
                record["runs"].append({"track": label, "returncode": completed.returncode, "manifest": str(manifests[0])})
                save(output / "comparison_manifest.json", record)
            final = checked_configuration(left)
            if {k: fingerprint(v) for k, v in final.items()} != initial:
                raise EndpointError("model/service configuration changed during the paired experiment")
            record["fair_comparison_verified"] = not args.smoke
            record["connectivity_smoke_completed"] = args.smoke
            save(output / "comparison_manifest.json", record)
            print(json.dumps({"output": str(output), "runs": record["runs"], "smoke_only": args.smoke}, indent=2))
            return 0
    except (ValueError, OSError) as exc:
        print(f"Local pair refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
