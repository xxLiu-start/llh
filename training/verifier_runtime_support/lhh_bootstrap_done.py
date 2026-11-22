"""Record successful bootstrap and restore normal tool behavior for tests.

No pytest hook or assertion is modified. Known dependency installation is
offline; functional subprocesses inside tests retain their original behavior.
"""
import json
import os
import sys
from pathlib import Path

root = os.environ.get("LLH_VERIFIER_RUNTIME")
support = os.environ.get("LLH_VERIFIER_SUPPORT")
if root and support:
    from asset_transport import install
    install(root)
    path = Path("/logs/verifier/bootstrap.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"state": "tests_started", "uv_version": "0.9.5",
                               "python_version": sys.version.split()[0], "python_executable": sys.executable,
                               "uv_python_downloads_during_grading": False}) + "\n")
    os.environ["PATH"] = str(Path(support) / "asset_tools") + ":" + str(Path(root) / "bin") + ":" + ":".join(
        entry for entry in os.environ.get("PATH", "").split(":") if entry != support)
    for name in ("UV_OFFLINE", "UV_PYTHON_DOWNLOADS", "UV_PYTHON_PREFERENCE", "BASH_ENV"):
        os.environ.pop(name, None)
