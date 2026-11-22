"""Check every task image's dependency setup with Docker --network none.

No agent, model, benchmark assertions, oracle or reward is executed. This checks
real apt/uv/pip installation and pytest startup in each original image.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import uuid
from pathlib import Path

from lh_harness import verifier_runtime as runtime
from lh_harness import verifier_system


def validate(name, dataset, root, output, specs, plan):
    directory = output / name
    logs = directory / "verifier"
    logs.mkdir(parents=True)
    logs.chmod(0o777)
    command = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--user", "root",
               "--cpus", "2", "--memory", "4g", "--entrypoint", "/bin/bash"]
    for source, target, readonly in [(root, root, True), (runtime.SUPPORT, runtime.SUPPORT, True),
                                    (runtime.SUPPORT / "env", Path("/root/.local/bin/env"), True),
                                    (dataset / name / "tests", Path("/tests"), True), (logs, Path("/logs/verifier"), False)]:
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for key, value in {"LLH_VERIFIER_RUNTIME": str(root), "LLH_VERIFIER_SUPPORT": str(runtime.SUPPORT),
                       "BASH_ENV": str(runtime.SUPPORT / "activate"), "LLH_VERIFIER_TASK": name,
                       "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}.items():
        command += ["-e", f"{key}={value}"]
    lines = ["set -e", "if ! command -v python3 && ! command -v python; then apt-get install -y python3; fi", "apt-get update",
             shlex.join(["apt-get", "install", "-y", *specs[name]["packages"]]),
             "command -v ps && command -v pgrep"]
    for key in plan["tasks"][name]["profiles"]:
        argv = plan["profiles"][key]["argv"]
        if argv[0] == "uvx":
            lines.append(shlex.join([*argv, "--version"]))
        elif argv[0] == "uv":
            lines += ["uv venv -p 3.12 /tmp/llh-bootstrap-venv", "source /tmp/llh-bootstrap-venv/bin/activate",
                      shlex.join(argv), "uv run pytest --version"]
        else:
            lines += [shlex.join(argv), "pytest --version"]
    # Exercise the actual external-fixture transport without invoking any
    # benchmark assertion. A populated cache alone is insufficient evidence.
    if name in verifier_system.ASSETS:
        system = json.loads((root / "system/manifest.json").read_text())
        asset = system["assets"][name]
        if asset.get("ref"):
            lines += [shlex.join(["git", "clone", "--depth", "1", "--branch", asset["ref"],
                                  asset["url"], "/tmp/llh-asset-probe"]),
                      'test "$(git -C /tmp/llh-asset-probe rev-parse HEAD)" = ' + shlex.quote(asset["commit"])]
        elif name == "sam-cell-seg":
            lines += [shlex.join(["curl", "-L", "-o", "mobile_sam.pt", asset["url"]]),
                      shlex.join(["sha256sum", "mobile_sam.pt"]) + " | grep -F " + shlex.quote(asset["sha256"])]
        else:
            code = ("from asset_transport import install;import urllib.request,hashlib;"
                    f"install({str(root)!r});urllib.request.urlretrieve({asset['url']!r},'/tmp/llh-asset.zip');"
                    f"assert hashlib.sha256(open('/tmp/llh-asset.zip','rb').read()).hexdigest()=={asset['sha256']!r}")
            lines.append(shlex.join([str(root / "python/cpython-3.13.9-linux-x86_64-gnu/bin/python3.13"), "-c", code]))
    command += [specs[name]["image"], "-c", "\n".join(lines)]
    try:
        with (directory / "container.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        passed = result.returncode == 0 and not (logs / "infra_error.json").exists()
        record = {"task": name, "passed": passed, "exit_code": result.returncode,
                  "external_asset_checked": name in verifier_system.ASSETS,
                  "network_mode": "none", "image": specs[name]["image"], "log": str(directory / "container.log")}
    except Exception as exc:
        record = {"task": name, "passed": False, "error": str(exc), "network_mode": "none"}
    runtime.atomic_json(directory / "validation.json", record)
    print(f"Offline dependency check: {name}: {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append")
    parser.add_argument("--runtime", type=Path, default=runtime.DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = runtime.ROOT / "eval/TB-harness/datasets/a-evolve-tb20-harbor"
    names = sorted(set(args.task) if args.task else (p.name for p in dataset.iterdir() if (p / "task.toml").exists()))
    prepared = runtime.check(dataset, names, args.runtime)
    specs, plan = verifier_system.plan(dataset, names), runtime.dependency_plan(dataset, names)
    output = args.output or runtime.ROOT / "output/verifier-bootstrap-validation" / uuid.uuid4().hex[:10]
    output = output.resolve()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(validate, name, dataset, args.runtime.resolve(), output, specs, plan) for name in names]
        results = [future.result() for future in futures]
    record = {"passed": all(r["passed"] for r in results), "model_evaluation": False, "network_mode": "none",
              "scope": "dependency installation and pytest startup only; no benchmark tests or scoring",
              "prepared": prepared, "tasks": results}
    runtime.atomic_json(output / "validation.json", record)
    print(json.dumps({"passed": record["passed"], "tasks": len(results), "output": str(output)}, indent=2))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
