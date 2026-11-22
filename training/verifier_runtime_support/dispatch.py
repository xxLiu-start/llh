"""Verifier bootstrap only. No test bodies, assertions, or rewards are changed."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def fail(message):
    print(f"LLH_VERIFIER_INFRA_ERROR: {message}", file=sys.stderr, flush=True)
    path = Path("/logs/verifier/infra_error.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"classification": "infrastructure_error", "message": message,
                                "model_score": None, "runtime_download_fallback": False}) + "\n")
    raise SystemExit(86)


def task_interpreter_args(args, root):
    """Keep a matching interpreter already supplied by the task image."""
    if "-p" not in args:
        return args, False
    index = args.index("-p") + 1
    minor = args[index]
    if minor not in {"3.11", "3.12", "3.13"}:
        return args, False
    candidate = shutil.which("python" + minor)
    if not candidate or candidate.startswith(str(root)):
        return args, False
    version = subprocess.check_output([candidate, "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                                      text=True, timeout=10).strip()
    if version != minor:
        return args, False
    resolved = list(args)
    resolved[index] = candidate
    return resolved, True


def system_tool(tool, args, root, manifest):
    system_path = root / "system/manifest.json"
    if not system_path.exists():
        fail("system packages/assets were not prepared; no online fallback")
    system = json.loads(system_path.read_text())
    if tool == "git":
        local_repository = None
        if args and args[0] == "clone":
            for name, asset in system["assets"].items():
                if asset.get("ref") and asset["url"] in args:
                    local_repository = str(root / "system/assets" / name / "repo.git")
                    args = [local_repository if x == asset["url"] else x for x in args]
                    # file:// honors shallow cloning without contacting GitHub.
                    args = ["file://" + x if x == str(root / "system/assets" / name / "repo.git") else x for x in args]
                    break
        actual = os.environ.get("LLH_VERIFIER_REAL_GIT") or "/usr/bin/git"
        if local_repository:
            # Git upload-pack intentionally drops clone's command-line config.
            # Copy the verified source under the current container user's UID;
            # no global Git trust exception or writable shared cache is needed.
            with tempfile.TemporaryDirectory(prefix="llh-verifier-git-") as temporary:
                private = Path(temporary) / "repo.git"
                shutil.copytree(local_repository, private, symlinks=True)
                local_args = ["file://" + str(private) if x == "file://" + local_repository else x for x in args]
                completed = subprocess.run([actual, *local_args], check=False)
            if completed.returncode:
                fail(f"offline verifier test asset clone failed (exit {completed.returncode}); no online fallback")
            return
        os.execv(actual, [actual, *args])
    if tool == "apt-get":
        script = Path("/tests/test.sh")
        identity = Path("/tmp/.llh-verifier-task")
        task = os.environ.get("LLH_VERIFIER_TASK") or (identity.read_text().strip() if identity.exists() else None)
        if not task and script.exists():
            sha = hashlib.sha256(script.read_bytes()).hexdigest()
            task = next((name for name, spec in manifest["tasks"].items() if spec["script_sha256"] == sha), None)
        record = system["images"].get(task)
        if not record:
            # Setup can precede /tests upload. Only procps is requested there;
            # the task identity is supplied explicitly by the environment.
            fail("cannot identify the prepared task system package profile")
        if script.exists() and manifest["tasks"].get(task, {}).get("script_sha256") != hashlib.sha256(script.read_bytes()).hexdigest():
            fail("task-specific system package profile does not match the original test script")
        if args in (["update"], ["update", "-qq"], ["-qq", "update"]):
            print("Using verified offline system package cache; apt index download unnecessary", flush=True)
            return
        if not args or args[0] != "install":
            fail("unprepared verifier apt-get operation; no network fallback")
        packages = [x for x in args[1:] if not x.startswith("-")]
        if not packages or not set(packages) <= set(record["spec"]["packages"]) | set(record.get("setup_packages", [])):
            fail("unprepared verifier system package request")
        missing = []
        for package in packages:
            p = subprocess.run(["dpkg-query", "-W", "-f=${db:Status-Abbrev}", package], capture_output=True, text=True)
            if p.returncode or not p.stdout.startswith("ii"):
                missing.append(package)
        if not missing:
            print("Required system packages already installed: " + ", ".join(packages), flush=True)
            return
        debs = sorted((root / "system/images" / task / "debs").glob("*.deb"))
        if not debs:
            fail("system packages missing from task image and offline cache")
        # Use an exclusively local apt repository. This preserves dependency
        # selection: setup requesting procps must not prematurely install other
        # packages needed only by scoring. No remote sources are configured.
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        with tempfile.TemporaryDirectory(prefix="llh-offline-apt-") as temporary:
            directory = Path(temporary)
            directory.chmod(0o755)
            repo = directory / "repo"
            repo.mkdir()
            entries = []
            for deb in debs:
                control = subprocess.check_output(["dpkg-deb", "--field", str(deb)], text=True)
                (repo / deb.name).symlink_to(deb)
                entries.append(control.rstrip() + f"\nFilename: ./{deb.name}\nSize: {deb.stat().st_size}\n"
                               + "SHA256: " + hashlib.sha256(deb.read_bytes()).hexdigest() + "\n\n")
            (repo / "Packages").write_text("".join(entries))
            sources = directory / "sources.list"
            sources.write_text(f"deb [trusted=yes] file:{repo} ./\n")
            for child in ("lists/partial", "archives/partial"):
                (directory / child).mkdir(parents=True)
            options = ["-o", f"Dir::Etc::sourcelist={sources}", "-o", "Dir::Etc::sourceparts=-",
                       "-o", f"Dir::State::lists={directory / 'lists'}", "-o", f"Dir::Cache::archives={directory / 'archives'}",
                       "-o", "Acquire::Languages=none", "-o", "Acquire::Retries=0"]
            for operation in (["update"], ["install", "-y", *missing]):
                completed = subprocess.run(["/usr/bin/apt-get", *options, *operation], check=False, env=env, stdin=subprocess.DEVNULL)
                if completed.returncode:
                    fail(f"offline system dependency install failed (exit {completed.returncode}); local sources only")
        return


def main():
    tool, *args = sys.argv[1:]
    root = Path(os.environ["LLH_VERIFIER_RUNTIME"])
    manifest = json.loads((root / "manifest.json").read_text())
    native = root / "bin" / tool
    if tool in {"apt-get", "git"}:
        return system_tool(tool, args, root, manifest)
    if tool == "curl":
        if args == ["-LsSf", "https://astral.sh/uv/0.9.5/install.sh"]:
            version = subprocess.check_output([str(root / "bin/uv"), "--version"], text=True).strip()
            if version != "uv 0.9.5":
                fail("pinned uv version does not match 0.9.5")
            command = shlex.quote(str(root / "bin/uv"))
            print(f'#!/bin/sh\ntest "$({command} --version)" = "uv 0.9.5" || exit 86\n'
                  'echo "Using verified offline uv/uvx 0.9.5" >&2')
            return
        system = json.loads((root / "system/manifest.json").read_text()) if (root / "system/manifest.json").exists() else {}
        for name, asset in system.get("assets", {}).items():
            if not asset.get("ref") and args == ["-L", "-o", "mobile_sam.pt", asset["url"]]:
                shutil.copyfile(root / "system/assets" / name / "payload", "mobile_sam.pt")
                print("Using verified offline verifier asset: " + name, flush=True)
                return
        actual = os.environ.get("LLH_VERIFIER_REAL_CURL") or "/usr/bin/curl"
        os.execv(actual, [actual, *args])
    prefix = [tool, *args]
    if tool == "uvx":
        if "pytest" not in args:
            fail("unprepared verifier uvx command; no online fallback")
        prefix = [tool, *args[:args.index("pytest") + 1]]
    key = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest()[:16]
    profile = manifest["profiles"].get(key)
    if tool == "pip":
        actual = os.environ.get("LLH_VERIFIER_REAL_PIP") or "/usr/local/bin/pip"
        if profile is not None:
            os.environ.update(PIP_NO_INDEX="1", PIP_FIND_LINKS=str(root / "profiles" / key / "wheels"),
                              PIP_DISABLE_PIP_VERSION_CHECK="1")
            completed = subprocess.run([actual, *args], check=False)
            if completed.returncode:
                fail(f"prepared pip bootstrap failed (exit {completed.returncode}); no online fallback")
            return
        os.execv(actual, [actual, *args])
    if tool == "uvx" and profile is None:
        fail(f"missing prepared verifier profile {key}; no online fallback")
    private = Path("/tmp/llh-verifier-cache") / (key if tool == "uvx" else "uv")
    if profile is not None:
        marker = private / f".{key}.ready"
        if not marker.exists():
            private.mkdir(parents=True, exist_ok=True)
            source = root / "profiles" / key / "cache"
            if not source.is_dir():
                fail(f"missing cache for verifier profile {key}")
            subprocess.run(["cp", "-a", "--no-preserve=ownership", "--reflink=auto", str(source) + "/.", str(private)], check=True)
            marker.touch()
    else:
        private.mkdir(parents=True, exist_ok=True)
    os.environ.update(UV_CACHE_DIR=str(private), UV_PYTHON_INSTALL_DIR=str(root / "python"),
                      UV_PYTHON_PREFERENCE="only-managed", UV_PYTHON_DOWNLOADS="never", UV_OFFLINE="1",
                      UV_TOOL_DIR=str(private / "tools"))
    # Preserve the interpreter the original task image already supplies. Cache
    # warming must not upgrade 3.13.7 to 3.13.9 and change timing-sensitive tests.
    args, image_python = task_interpreter_args(args, root)
    if image_python:
        os.environ["UV_PYTHON_PREFERENCE"] = "system"
    # Real uv/uvx performs the original command with the exact prepared pins.
    # A private copy permits uv lock files without making shared caches writable.
    completed = subprocess.run([str(native), *args], check=False)
    if completed.returncode:
        # Pytest's own nonzero status is a legitimate test outcome. Only mark
        # bootstrap errors when no CTRF test result was produced at all.
        ctrf = Path("/logs/verifier/ctrf.json")
        if not ctrf.exists() and not Path("/logs/verifier/bootstrap.json").exists():
            fail(f"verifier {tool} exited {completed.returncode} before producing test results")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        fail(str(exc))
