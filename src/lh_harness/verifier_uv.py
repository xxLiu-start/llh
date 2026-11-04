"""Supply the task verifier's pinned uv without editing benchmark tests."""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


UV_VERSION = "0.7.13"
INSTALLER_URL = f"https://astral.sh/uv/{UV_VERSION}/install.sh"
COMPOSE_NAME = "docker_verifier_uv_offline.yaml"
CONTAINER_BIN = "/opt/lhh-verifier-uv"
VERIFIER_PATH = f"{CONTAINER_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def needs_verifier_uv(dataset: Path, task: str) -> bool:
    """Limit the override to tasks whose unmodified verifier installs this uv."""
    script = dataset / task / "tests" / "test.sh"
    try:
        return INSTALLER_URL in script.read_text(encoding="utf-8")
    except OSError:
        return False


def add_verifier_uv(config: dict[str, Any], *, dataset: Path, task: str) -> None:
    if not needs_verifier_uv(dataset, task):
        return
    compose = Path(__file__).resolve().parents[2] / "training" / COMPOSE_NAME
    overrides = config.setdefault("environment", {}).setdefault("extra_docker_compose", [])
    if str(compose) not in overrides:
        overrides.append(str(compose))
    verifier_env = config.setdefault("verifier", {}).setdefault("env", {})
    verifier_env.setdefault("HOME", "/root")
    path = verifier_env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    if CONTAINER_BIN not in path.split(":"):
        verifier_env["PATH"] = f"{CONTAINER_BIN}:{path}"


def verifier_uv_host_env(config: dict[str, Any], base: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve the host mount source only when this trial uses the override."""
    env = dict(os.environ if base is None else base)
    overrides = config.get("environment", {}).get("extra_docker_compose", [])
    if not any(Path(path).name == COMPOSE_NAME for path in overrides):
        return env
    candidate = env.get("LH_VERIFIER_UV_BINARY") or shutil.which("uv", path=env.get("PATH"))
    if not candidate:
        raise RuntimeError(f"Verifier requires uv {UV_VERSION}; set LH_VERIFIER_UV_BINARY to a local executable")
    binary = Path(candidate).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"Verifier uv binary is missing or not executable: {binary}")
    try:
        version = subprocess.run([str(binary), "--version"], check=True, capture_output=True,
                                 text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not verify local uv binary: {binary}") from exc
    if version != f"uv {UV_VERSION}":
        raise RuntimeError(f"Verifier requires uv {UV_VERSION}; local binary reports {version!r}")
    support = Path(__file__).resolve().parents[2] / "training" / "verifier_uv_support"
    for name in ("curl", "env"):
        if not (support / name).is_file():
            raise RuntimeError(f"Verifier uv support file is missing: {support / name}")
    env["LH_VERIFIER_UV_BINARY"] = str(binary)
    env["LH_VERIFIER_UV_SUPPORT_DIR"] = str(support)
    return env


@contextmanager
def scoped_verifier_uv_host_env(config: dict[str, Any]) -> Iterator[None]:
    """Expose Compose mount variables while an existing Harbor runner executes."""
    resolved = verifier_uv_host_env(config)
    names = ("LH_VERIFIER_UV_BINARY", "LH_VERIFIER_UV_SUPPORT_DIR")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            if name in resolved:
                os.environ[name] = resolved[name]
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
