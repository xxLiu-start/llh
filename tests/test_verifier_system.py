import importlib.util
import json
from pathlib import Path

import pytest

from lh_harness import verifier_system
from lh_harness.verifier_runtime import SUPPORT


def test_system_plan_includes_test_helper_dependencies_and_preserves_input(tmp_path):
    task = tmp_path / "fixture"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('[environment]\ndocker_image="fixture:test"\n')
    script = task / "tests/test.sh"
    script.write_text('apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl expect\n')
    (task / "tests/test_outputs.py").write_text('setup="""apt-get install -y git openssh-client\n"""')
    (task / "tests/helpers").mkdir()
    (task / "tests/helpers/verify.sh").write_text('apt-get install -y sshpass\n')
    before = script.read_bytes()
    plan = verifier_system.plan(tmp_path, ["fixture"])
    assert plan["fixture"]["packages"] == ["curl", "expect", "git", "openssh-client", "procps", "sshpass"]
    assert script.read_bytes() == before


def test_missing_system_cache_prevents_benchmark(tmp_path):
    with pytest.raises(ValueError, match="prepare-verifier-runtime"):
        verifier_system.check(tmp_path, ["fixture"], tmp_path)


def test_debian_archive_transport_keeps_apt_version_and_rejects_changed_bytes(tmp_path, monkeypatch):
    import hashlib
    import io
    (tmp_path / "debs").mkdir()
    content = b"same package bytes from the authenticated apt index"
    sha = hashlib.sha256(content).hexdigest()
    filename = "curl_7.74.0-1.3+deb11u16_amd64.deb"
    relative = "pool/updates/main/c/curl/" + filename
    (tmp_path / "package-uris.txt").write_text(
        f"'http://deb.debian.org/debian-security/{relative}' {filename} {len(content)} \n")
    (tmp_path / "package-index.txt").write_text(
        f"Package: curl\nVersion: 7.74.0-1.3+deb11u16\nFilename: {relative}\nSize: {len(content)}\nSHA256: {sha}\n\n")
    class Archive:
        body = b"invalid package bytes"
        def open(self, url, **kwargs):
            assert url == "https://archive.debian.org/debian-security/" + relative
            return io.BytesIO(self.body)
    archive = Archive()
    monkeypatch.setattr(verifier_system.urllib.request, "build_opener", lambda *a: archive)
    with pytest.raises(ValueError, match="does not match apt metadata"):
        verifier_system.fetch_archived_debs(tmp_path)
    assert not (tmp_path / "debs" / filename).exists()
    archive.body = content
    recovered = verifier_system.fetch_archived_debs(tmp_path)
    assert (tmp_path / "debs" / filename).read_bytes() == content
    assert recovered[0]["version"] == "7.74.0-1.3+deb11u16" and recovered[0]["sha256"] == sha


def test_old_cache_cannot_hide_missing_native_python_prerequisite(tmp_path, monkeypatch):
    from types import SimpleNamespace
    (tmp_path / "system").mkdir()
    spec = {"image": "fixture:test", "packages": ["procps"]}
    (tmp_path / "system/manifest.json").write_text(json.dumps({"images": {"fixture": {"spec": spec, "image_id": "sha256:fixture", "debs": {}}}}))
    monkeypatch.setattr(verifier_system, "plan", lambda *a: {"fixture": spec})
    monkeypatch.setattr(verifier_system.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps([
        {"Id": "sha256:fixture", "RepoTags": ["fixture:test"]}])))
    with pytest.raises(ValueError, match="offline Python prerequisite preparation"):
        verifier_system.check(tmp_path, ["fixture"], tmp_path)


def test_cached_asset_transport_serves_original_bytes_without_network(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("asset_transport_test", SUPPORT / "asset_transport.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    url = verifier_system.ASSETS["filter-js-from-html"][0]
    asset = tmp_path / "system/assets/filter-js-from-html/payload"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"original downloaded bytes")
    (tmp_path / "system/manifest.json").write_text(json.dumps({"assets": {"filter-js-from-html": {"url": url}}}))
    called = []
    monkeypatch.setattr(transport.urllib.request, "urlretrieve", lambda *a: called.append(a))
    transport.install(tmp_path)
    target = tmp_path / "download.zip"
    transport.urllib.request.urlretrieve(url, target)
    assert target.read_bytes() == asset.read_bytes() and called == []
    transport.urllib.request.urlretrieve("http://localhost:8080/task", target)
    assert len(called) == 1


def test_hidden_test_assets_are_uploaded_only_with_original_verifier_tests(tmp_path):
    import asyncio
    import hashlib
    root = tmp_path / "cache"
    (root / "system").mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir()
    script = tests / "test.sh"
    script.write_bytes(b"original test script\n")
    (root / "manifest.json").write_text(json.dumps({"tasks": {"fix-ocaml-gc": {
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}}}))
    calls = []
    class Environment:
        async def upload_dir(self, source_dir, target_dir):
            calls.append((source_dir, target_dir))
            return "original-upload-result"
    environment = Environment()
    verifier_system.install_asset_upload(environment, "fix-ocaml-gc", root)
    assert calls == [], "hidden upstream tests must not be staged during agent setup"
    assert asyncio.run(environment.upload_dir(tests, "/tests")) == "original-upload-result"
    assert calls == [(tests, "/tests"), (root / "system/assets/fix-ocaml-gc", str(root / "system/assets/fix-ocaml-gc"))]
    script.write_bytes(b"changed tests")
    with pytest.raises(ValueError, match="prepared original tests"):
        asyncio.run(environment.upload_dir(tests, "/tests"))
    assert len(calls) == 3, "changed scripts must not receive hidden test assets"


def test_offline_apt_skips_index_download_and_installed_packages(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("system_dispatch_test", SUPPORT / "dispatch.py")
    dispatcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatcher)
    (tmp_path / "system").mkdir()
    (tmp_path / "system/manifest.json").write_text(json.dumps({"images": {"fixture": {"spec": {"packages": ["curl"]}}}}))
    monkeypatch.setenv("LLH_VERIFIER_TASK", "fixture")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == "dpkg-query", "offline bootstrap must not invoke apt or a network downloader"
        return SimpleNamespace(returncode=0, stdout="ii ")
    monkeypatch.setattr(dispatcher.subprocess, "run", run)
    dispatcher.system_tool("apt-get", ["update"], tmp_path, {"tasks": {}})
    dispatcher.system_tool("apt-get", ["install", "-y", "curl"], tmp_path, {"tasks": {}})
    assert len(calls) == 1
    with pytest.raises(SystemExit) as exc:
        monkeypatch.setattr(dispatcher, "fail", lambda message: (_ for _ in ()).throw(SystemExit(86)))
        dispatcher.system_tool("apt-get", ["install", "-y", "unprepared-package"], tmp_path, {"tasks": {}})
    assert exc.value.code == 86


def test_cached_git_clone_uses_private_repository_and_reports_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("git_dispatch_test", SUPPORT / "dispatch.py")
    dispatcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatcher)
    (tmp_path / "system").mkdir()
    url, ref = verifier_system.ASSETS["build-cython-ext"]
    (tmp_path / "system/manifest.json").write_text(json.dumps({"assets": {
        "build-cython-ext": {"url": url, "ref": ref}}}))
    repo = tmp_path / "system/assets/build-cython-ext/repo.git"
    repo.mkdir(parents=True)
    (repo / "HEAD").write_text("original cached HEAD")
    calls = []
    status = 0
    def run(argv, **kwargs):
        calls.append(argv)
        private = Path(next(x[7:] for x in argv if x.startswith("file://")))
        assert private != repo and (private / "HEAD").read_text() == "original cached HEAD"
        return SimpleNamespace(returncode=status)
    monkeypatch.setattr(dispatcher.subprocess, "run", run)
    monkeypatch.setattr(dispatcher, "fail", lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    argv = ["clone", "--depth", "1", "--branch", ref, url, "/tmp/probe"]
    dispatcher.system_tool("git", argv, tmp_path, {})
    assert calls[0][1:6] == ["clone", "--depth", "1", "--branch", ref]
    assert not any("safe.directory" in item for item in calls[0])
    assert url in argv, "original caller arguments remain unchanged"
    status = 128
    with pytest.raises(RuntimeError, match="no online fallback"):
        dispatcher.system_tool("git", argv, tmp_path, {})
