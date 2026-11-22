"""Prepare image-specific system packages and exact external verifier assets.

Only transport/bootstrap is adapted. Images, test scripts, assertions and scores
remain unchanged. Missing artifacts fail before the benchmark, with no download
fallback during scoring.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import tomllib
import urllib.request
from urllib.parse import unquote, urlsplit
from pathlib import Path

ASSETS = {
    "sam-cell-seg": ("https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt", None),
    "filter-js-from-html": ("https://github.com/davidwagner/html-sanitizer-testbed/archive/refs/heads/master.zip", None),
    "fix-ocaml-gc": ("https://github.com/sadiqj/ocaml/", "tag_purposefully_broken_sweeping_changes"),
    "build-cython-ext": ("https://github.com/SPOCKnots/pyknotid.git", "0.5.3"),
}


def fetch_archived_debs(directory: Path) -> list[dict]:
    """Recover unchanged Debian packages against apt's authenticated SHA256.

    Some old security indices remain published after their payloads move to
    archive.debian.org. This is explicit preparation, never runtime fallback.
    """
    from .verifier_runtime import atomic_json, digest
    records = []
    for paragraph in (directory / "package-index.txt").read_text().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in paragraph.splitlines()
                      if ": " in line and not line.startswith(" "))
        if "Filename" in fields:
            records.append(fields)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    recovered = []
    for line in (directory / "package-uris.txt").read_text().splitlines():
        if not line.startswith("'"):
            continue
        uri, filename, size, *_ = shlex.split(line)
        if Path(filename).name != filename:
            raise ValueError("invalid apt package cache filename")
        parsed = urlsplit(uri)
        if parsed.hostname != "deb.debian.org" or not parsed.path.startswith(("/debian/", "/debian-security/")):
            continue
        prefix, relative = unquote(parsed.path).lstrip("/").split("/", 1)
        metadata = next((r for r in records if r["Filename"] == relative and r.get("Size") == size), None)
        if not metadata or not re.fullmatch(r"[0-9a-f]{64}", metadata.get("SHA256", "")):
            raise ValueError(f"missing authenticated SHA256 for archived package: {filename}")
        target = directory / "debs" / filename
        if target.exists() and target.stat().st_size == int(size) and digest(target) == metadata["SHA256"]:
            continue
        url = f"https://archive.debian.org/{prefix}/{relative}"
        temporary = target.with_suffix(".partial")
        with opener.open(url, timeout=90) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size != int(size) or digest(temporary) != metadata["SHA256"]:
            temporary.unlink()
            raise ValueError(f"official archive package does not match apt metadata: {filename}")
        os.replace(temporary, target)
        recovered.append({"original_url": uri, "download_url": url, "sha256": metadata["SHA256"],
                          "version": metadata.get("Version"), "filename": filename})
    atomic_json(directory / "archive-transport.json", {"packages": recovered, "scope": "preparation only; same apt versions and SHA256"})
    return recovered


def plan(dataset: Path, names: list[str]) -> dict:
    result = {}
    for name in names:
        # Shell helpers can install packages from inside pytest too (for
        # example configure-git-webserver/tests/verify.sh). Read all original
        # test helpers, without relying on their invocation order.
        text = "\n".join(path.read_text().replace("\\\n", " ")
                         for path in sorted((dataset / name / "tests").rglob("*"))
                         if path.is_file() and path.suffix in {".sh", ".py"})
        packages = {"procps"}
        for match in re.finditer(r"apt-get\s+install\s+([^\n;&]+)", text):
            packages.update(x for x in shlex.split(match[1]) if re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", x))
        config = tomllib.loads((dataset / name / "task.toml").read_text())["environment"]
        result[name] = {"image": config["docker_image"], "packages": sorted(packages)}
    return result


def prepare_image(root: Path, name: str, spec: dict) -> dict:
    from .verifier_runtime import atomic_json, cache_hashes
    directory = root / "system/images" / name
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("debs") != cache_hashes(directory / "debs"):
            raise ValueError(f"system dependency cache changed for {name}")
        actual = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", spec["image"]], text=True).strip()
        if actual != record["image_id"]:
            raise ValueError(f"task image changed for {name}; use a fresh verifier runtime")
        if record.get("spec") != spec:
            # Explicit preparation may discover additional original test
            # helpers. Rebuild this profile against the same pinned image;
            # ordinary preflight still rejects the old profile meanwhile.
            return _download_image_packages(root, name, spec, directory)
        if "setup_packages" in record:
            return record
        native = subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "/bin/sh",
                                 spec["image"], "-c", "command -v python3 || command -v python"], capture_output=True, text=True, timeout=45)
        if native.returncode == 0:
            record.update(setup_packages=["procps"], image_python=native.stdout.strip())
            atomic_json(record_path, record)
            return record
    return _download_image_packages(root, name, spec, directory)


def _download_image_packages(root: Path, name: str, spec: dict, directory: Path) -> dict:
    from .verifier_runtime import atomic_json, cache_hashes
    record_path = directory / "record.json"
    probe = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", spec["image"]], capture_output=True, text=True)
    if probe.returncode:
        # Pulling is explicit preparation, never a hidden benchmark fallback.
        subprocess.run(["docker", "pull", spec["image"]], check=True, timeout=3600)
        probe = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", spec["image"]], capture_output=True, text=True, check=True)
    script = '''set -eu
trap 'chown -R "$LLH_PREPARE_UID:$LLH_PREPARE_GID" /cache; chmod -R a+rX /cache' EXIT
mkdir -p /cache/debs/partial
missing=""
native_python="$(command -v python3 || command -v python || true)"
printf '%s' "$native_python" > /cache/image-python.txt
if [ -z "$native_python" ]; then missing=" python3"; fi
for pkg in "$@"; do
  if ! dpkg-query -W -f='${db:Status-Abbrev}' "$pkg" 2>/dev/null | grep -q '^ii'; then missing="$missing $pkg"; fi
done
if [ -n "$missing" ]; then
  apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update
  apt-get --print-uris --download-only install -y $missing > /cache/package-uris.txt
  grep "^'" /cache/package-uris.txt | cut -d' ' -f2 | cut -d_ -f1 | xargs -r apt-cache show > /cache/package-index.txt
  apt-get -o Dir::Cache::archives=/cache/debs -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 --download-only install -y $missing
fi
dpkg-query -W -f='${Package} ${Version} ${db:Status-Abbrev}\\n' > /cache/image-packages.txt
chmod -R a+rX /cache
'''
    command = ["docker", "run", "--rm", "--pull", "never", "--user", "root", "--entrypoint", "/bin/bash",
               "-e", f"LLH_PREPARE_UID={os.getuid()}", "-e", f"LLH_PREPARE_GID={os.getgid()}",
               "--mount", f"type=bind,src={directory},dst=/cache", spec["image"], "-c", script, "prepare", *spec["packages"]]
    print(f"Preparing system dependencies: {name}", flush=True)
    with (directory / "prepare.log").open("w") as log:
        result = subprocess.run(command, check=False, timeout=1800, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode and "404" in (directory / "prepare.log").read_text() and (directory / "package-index.txt").exists():
        print(f"Preparing archived Debian packages with authenticated SHA256: {name}", flush=True)
        fetch_archived_debs(directory)
        with (directory / "prepare.log").open("a") as log:
            subprocess.run(command, check=True, timeout=1800, stdout=log, stderr=subprocess.STDOUT)
    elif result.returncode:
        raise ValueError(f"system dependency download failed for {name} (exit {result.returncode}); see {directory / 'prepare.log'}")
    record = {"spec": spec, "image_id": probe.stdout.strip(), "debs": cache_hashes(directory / "debs"),
              "setup_packages": ["procps"] + ([] if (directory / "image-python.txt").read_text() else ["python3"]),
              "image_python": (directory / "image-python.txt").read_text(),
              "installed_packages": (directory / "image-packages.txt").read_text()}
    atomic_json(record_path, record)
    return record


def prepare_asset(root: Path, name: str) -> dict:
    from .verifier_runtime import atomic_json, cache_hashes, digest, preparation_env
    url, ref = ASSETS[name]
    directory = root / "system/assets" / name
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        actual = cache_hashes(directory / "repo.git") if ref else digest(directory / "payload")
        if actual != record["sha256"]:
            raise ValueError(f"verifier asset cache changed: {name}")
        return record
    record = {"url": url, "ref": ref}
    if ref:
        repo = directory / "repo.git"
        env = preparation_env(root, root / "bootstrap-cache")
        subprocess.run(["git", "init", "--bare", str(repo)], check=True, env=env, stdout=subprocess.DEVNULL)
        command = ["git", "-c", "http.version=HTTP/1.1", "-C", str(repo), "fetch", "--depth", "1", url, ref]
        for attempt in range(4):
            try:
                subprocess.run(command, check=True, env=env, timeout=600)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == 3:
                    raise
                print(f"Retrying verifier asset transport: {name} ({attempt + 2}/4); no model calls", flush=True)
                time.sleep(2 ** attempt)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "FETCH_HEAD"], text=True).strip()
        for target in ("refs/heads/snapshot", "refs/tags/" + ref):
            subprocess.run(["git", "-C", str(repo), "update-ref", target, commit], check=True)
        subprocess.run(["git", "-C", str(repo), "symbolic-ref", "HEAD", "refs/heads/snapshot"], check=True)
        record.update(commit=commit, sha256=cache_hashes(repo))
    else:
        temporary = directory / "payload.partial"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if name == "sam-cell-seg":
            # uv already fetched this upstream repository for its pinned Git
            # dependency. Reuse weights only after comparing Git's blob hash
            # with the current upstream URL, never by assuming files match.
            candidates = list((root / "profiles").glob("*/cache/git-v0/checkouts/*/*/weights/mobile_sam.pt"))
            if candidates:
                metadata_url = "https://api.github.com/repos/ChaoningZhang/MobileSAM/contents/weights/mobile_sam.pt?ref=master"
                with opener.open(metadata_url, timeout=30) as response:
                    metadata = json.load(response)
                for candidate in candidates:
                    h = hashlib.sha1(f"blob {candidate.stat().st_size}\0".encode())
                    with candidate.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            h.update(chunk)
                    if h.hexdigest() == metadata.get("sha") and candidate.stat().st_size == metadata.get("size"):
                        shutil.copyfile(candidate, directory / "payload")
                        record.update(sha256=digest(directory / "payload"), upstream_git_blob=h.hexdigest(),
                                      metadata_url=metadata_url, reused_verified_git_cache=True)
                        atomic_json(record_path, record)
                        return record
        # GitHub archive/raw URLs are recorded verbatim; fetch byte-identical
        # content through their official download hosts when needed.
        download = url.replace("https://github.com/ChaoningZhang/MobileSAM/raw/", "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/")
        download = download.replace("https://github.com/davidwagner/html-sanitizer-testbed/archive/refs/heads/master.zip",
                                    "https://codeload.github.com/davidwagner/html-sanitizer-testbed/zip/refs/heads/master")
        with opener.open(download, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size < 1024:
            raise ValueError(f"unexpected verifier asset: {name}")
        os.replace(temporary, directory / "payload")
        record.update(download_url=download, sha256=digest(directory / "payload"))
    atomic_json(record_path, record)
    return record


def prepare(dataset: Path, names: list[str], root: Path) -> dict:
    (root / "system").mkdir(parents=True, exist_ok=True)
    with (root / "system/.prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare(dataset, names, root)


def _prepare(dataset: Path, names: list[str], root: Path) -> dict:
    from .verifier_runtime import atomic_json
    specs = plan(dataset, names)
    path = root / "system/manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"images": {}, "assets": {}}
    errors = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(prepare_image, root, name, spec): ("images", name) for name, spec in specs.items()}
        jobs.update({pool.submit(prepare_asset, root, name): ("assets", name) for name in names if name in ASSETS})
        for future in concurrent.futures.as_completed(jobs):
            kind, name = jobs[future]
            try:
                manifest[kind][name] = future.result()
            except Exception as exc:
                errors[name] = str(exc)
                manifest[kind].pop(name, None)
                print(f"System preparation unavailable: {name}: {exc}", flush=True)
            manifest["errors"] = dict(errors)
            atomic_json(path, manifest)
    if errors:
        raise ValueError(f"system/asset dependencies unavailable for {sorted(errors)}; benchmark not started")
    return check(dataset, names, root)


def check(dataset: Path, names: list[str], root: Path) -> dict:
    from .verifier_runtime import cache_hashes, digest
    path = root / "system/manifest.json"
    if not path.exists():
        raise ValueError("System dependency cache is not prepared; run --prepare-verifier-runtime before benchmarking")
    manifest = json.loads(path.read_text())
    specs = plan(dataset, names)
    actual = subprocess.run(["docker", "image", "inspect", *sorted({s["image"] for s in specs.values()})],
                            capture_output=True, text=True, check=True)
    image_ids = {tag: image["Id"] for image in json.loads(actual.stdout) for tag in image.get("RepoTags", [])}
    for name, spec in specs.items():
        record = manifest["images"].get(name)
        if not record or record["spec"] != spec or record["image_id"] != image_ids.get(spec["image"]):
            raise ValueError(f"system cache/image missing or changed: {name}; run --prepare-verifier-runtime")
        if "setup_packages" not in record:
            raise ValueError(f"system cache needs offline Python prerequisite preparation: {name}; run --prepare-verifier-runtime")
        if record["debs"] != cache_hashes(root / "system/images" / name / "debs"):
            raise ValueError(f"system package cache corrupt: {name}")
        if name in ASSETS:
            record = manifest["assets"].get(name)
            directory = root / "system/assets" / name
            hashes = cache_hashes(directory / "repo.git") if ASSETS[name][1] else digest(directory / "payload") if (directory / "payload").exists() else None
            if not record or record["sha256"] != hashes or record["url"] != ASSETS[name][0]:
                raise ValueError(f"verifier asset missing or corrupt: {name}")
    return {"status": "ready", "manifest_sha256": digest(path), "images": len(names),
            "external_assets": sum(name in ASSETS for name in names)}


def install_asset_upload(environment, task: str, root: Path) -> None:
    """Stage hidden upstream test resources only when Harbor uploads tests.

    The original agent's run method and Harbor's verifier are unchanged. These
    assets are deliberately absent from the agent-phase read-only mounts.
    """
    if task not in ASSETS:
        return
    from .verifier_runtime import digest
    expected = json.loads((root / "manifest.json").read_text())["tasks"][task]["script_sha256"]
    upload = environment.upload_dir

    async def upload_with_verifier_assets(source_dir, target_dir):
        result = await upload(source_dir=source_dir, target_dir=target_dir)
        if str(target_dir).rstrip("/") == "/tests":
            if digest(Path(source_dir) / "test.sh") != expected:
                raise ValueError("verifier asset upload does not match the prepared original tests")
            source = root / "system/assets" / task
            await upload(source_dir=source, target_dir=str(source))
        return result

    environment.upload_dir = upload_with_verifier_assets
