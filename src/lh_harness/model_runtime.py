"""Docker endpoint preflight and runtime connection monitoring, outside MEA."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .local_models import EndpointClient, EndpointError


def check_docker(url, model, profile, image, runtime, *, api_key=None, auth_scheme="none"):
    source = Path(__file__).parent
    python = Path(runtime) / "python/cpython-3.13.9-linux-x86_64-gnu/bin/python3.13"
    code = """import json,os,sys,types
p=types.ModuleType('lh_harness');p.__path__=['/probe'];sys.modules['lh_harness']=p
from lh_harness.local_models import check_endpoint
c=json.load(sys.stdin)
r=check_endpoint(c['url'],'claude_code',c['model'],c['profile'],api_key=os.getenv('LLH_DOCKER_PROBE_KEY'),auth_scheme=c['auth_scheme'])
print(json.dumps(r))
"""
    command = ["docker", "run", "--rm", "-i", "--pull", "never", "--network", "bridge", "--entrypoint", str(python),
               "--mount", f"type=bind,src={Path(runtime).resolve()},dst={Path(runtime).resolve()},readonly",
               "--mount", f"type=bind,src={source},dst=/probe,readonly", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "LLH_DOCKER_PROBE_KEY", image, "-c", code]
    env = dict(os.environ)
    env.pop("LLH_DOCKER_PROBE_KEY", None)
    if api_key:
        env["LLH_DOCKER_PROBE_KEY"] = api_key
    payload = {"url": url, "model": model, "profile": profile, "auth_scheme": auth_scheme}
    result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, env=env, timeout=180)
    if result.returncode:
        # Only our client-generated diagnostic is included, never an upstream
        # response body or Docker environment/credential dump.
        diagnostic = next((line for line in reversed(result.stderr.splitlines()) if "Endpoint" in line), "Docker endpoint probe failed")
        raise ValueError(f"{diagnostic}; benchmark not started; no fallback")
    record = json.loads(result.stdout)
    return {"status": "ok", "network": "Docker bridge", "image": image, "endpoint_check": record}


class Monitor:
    """Observe native proxy errors and verify the same service stays available.

    No inference retries, prompt changes, model substitution or score changes.
    Three consecutive failed inventory requests stop the batch. A native proxy
    connection/API error stops it as soon as its raw episode log is persisted.
    """

    def __init__(self, url, model, *, api_key=None, auth_scheme="none", interval=10):
        self.client = EndpointClient(url, "claude_code", timeout=3, api_key=api_key, auth_scheme=auth_scheme)
        self.model, self.interval = model, interval
        self.next_check, self.consecutive_failures = 0, 0
        self.offsets = {}
        self.failure = None

    def check(self, job):
        if self.failure:
            return self.failure
        for path in job.glob("*/agent/**/request_proxy_requests.jsonl"):
            size = path.stat().st_size
            offset = self.offsets.get(path, 0)
            if size < offset:
                offset = 0
            with path.open() as stream:
                stream.seek(offset)
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        stream.seek(start)
                        break
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("event") in {"upstream_error", "upstream_http_error"}:
                        self.failure = {"kind": "model_api_error", "event": event["event"], "http_status": event.get("status"),
                                        "log": str(path), "no_fallback": True}
                        return self.failure
                self.offsets[path] = stream.tell()
        if time.monotonic() < self.next_check:
            return None
        self.next_check = time.monotonic() + self.interval
        try:
            inventory = self.client.json("/models")
            if not any(item.get("id") == self.model for item in inventory.get("data", []) if isinstance(item, dict)):
                raise EndpointError("configured model is no longer loaded; no substitution")
        except (EndpointError, ValueError, TypeError, AttributeError):
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.failure = {"kind": "model_endpoint_unavailable_or_changed", "consecutive_failures": self.consecutive_failures,
                                "base_url": self.client.url, "model_id": self.model, "no_fallback": True}
        else:
            self.consecutive_failures = 0
        return self.failure
