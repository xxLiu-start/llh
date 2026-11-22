"""Exercise unchanged benchmark tests in a disconnected disposable container.

The optional supplied oracle is an installer test fixture, never a model score.
No tests or solutions are copied into agent images or dependency caches.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
import uuid
from pathlib import Path

from lh_harness.verifier_runtime import DEFAULT_ROOT, ROOT, SUPPORT, check, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="portfolio-optimization")
    parser.add_argument("--oracle", action="store_true", help="Use the provided oracle only as an independent verifier fixture.")
    parser.add_argument("--runtime", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    dataset = ROOT / "eval/TB-harness/datasets/a-evolve-tb20-harbor"
    task = dataset / args.task
    prepared = check(dataset, [args.task], args.runtime)
    config = tomllib.loads((task / "task.toml").read_text())["environment"]
    output = ROOT / "output/verifier-runtime-validation" / uuid.uuid4().hex[:10]
    logs = output / "verifier"
    logs.mkdir(parents=True)
    logs.chmod(0o777)
    apt = output / "apt.conf"
    apt.write_text('Acquire::Retries "0";\nAcquire::http::Timeout "1";\nAcquire::https::Timeout "1";\n')
    before = digest(task / "tests/test.sh")
    command = ["docker", "run", "--rm", "--pull", "never", "--network", "none",
               "--cpus", str(config.get("cpus", 1)), "--memory", str(config.get("memory_mb", 4096)) + "m"]
    mounts = [(args.runtime.resolve(), args.runtime.resolve(), True), (SUPPORT, SUPPORT, True),
              (SUPPORT / "env", Path("/root/.local/bin/env"), True), (task / "tests", Path("/tests"), True),
              (logs, Path("/logs/verifier"), False), (apt, Path("/tmp/llh-offline-apt.conf"), True)]
    if args.oracle:
        mounts.append((task / "solution", Path("/solution"), True))
    for source, target, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for key, value in {"LLH_VERIFIER_RUNTIME": str(args.runtime.resolve()), "LLH_VERIFIER_SUPPORT": str(SUPPORT),
                       "LLH_VERIFIER_TASK": args.task,
                       "BASH_ENV": str(SUPPORT / "activate"), "APT_CONFIG": "/tmp/llh-offline-apt.conf",
                       "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}.items():
        command += ["-e", f"{key}={value}"]
    shell = ("bash /solution/solve.sh && " if args.oracle else "") + "bash /tests/test.sh"
    command += ["--entrypoint", "/bin/bash", config["docker_image"], "-c", shell]
    with (output / "container.log").open("w") as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, timeout=900, check=False)
    ctrf = logs / "ctrf.json"
    summary = json.loads(ctrf.read_text()).get("results", {}).get("summary", {}) if ctrf.exists() else {}
    reward = (logs / "reward.txt").read_text().strip() if (logs / "reward.txt").exists() else None
    expected = "0" if summary.get("failed", 0) or summary.get("other", 0) else "1"
    stdout = (output / "container.log").read_text(errors="replace")
    collected = re.search(r"collected (\d+) items", stdout)
    report = {"model_evaluation": False, "fixture": "provided_oracle" if args.oracle else "unchanged_skeleton",
              "network_mode": "none", "uv_version": "0.9.5", "task": args.task,
              "source_test_sha256": before, "source_tests_unchanged": digest(task / "tests/test.sh") == before,
              "runtime_manifest_sha256": prepared["manifest_sha256"], "test_summary": summary,
              "expected_fixture_reward": expected, "actual_fixture_reward": reward,
              "pytest_collected_cases": int(collected[1]) if collected else None,
              "fixture_tests_passed": reward == "1",
              "infra_error": (logs / "infra_error.json").exists(), "exit_code": result.returncode}
    report["passed"] = (result.returncode == 0 and summary.get("tests", 0) > 0 and reward == expected
                        and not report["infra_error"] and report["source_tests_unchanged"])
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "output": str(output)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
