"""Optional, explicit inference policy proxy; never selects a model/provider.

Both experimental tracks point their existing base_url at the same instance.
Protocols and streaming bytes are preserved; unsupported protocol combinations
are diagnosed by doctor, not translated or redirected to a cloud service.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import load_run_defaults
from .local_models import EndpointClient, EndpointError, base_url, validate_profile


def policy_payload(route: str, payload: dict, profile: dict, models: set[str]) -> dict:
    if payload.get("model") not in models:
        raise EndpointError("model is not explicitly permitted by this experiment")
    value = dict(payload)
    if "max_output_tokens" in profile:
        value.pop("max_completion_tokens", None)
        value["max_output_tokens" if route == "/responses" else "max_tokens"] = profile["max_output_tokens"]
    for name in ("temperature", "top_p"):
        if name in profile:
            value[name] = profile[name]
    reasoning = profile.get("reasoning", {})
    if reasoning:
        if route == "/messages":
            raise EndpointError("Messages reasoning policy needs an explicit protocol-specific adapter")
        if "enable_thinking" in reasoning:
            value["chat_template_kwargs"] = {**value.get("chat_template_kwargs", {}),
                                              "enable_thinking": reasoning["enable_thinking"]}
        if "effort" in reasoning:
            value["reasoning" if route == "/responses" else "reasoning_effort"] = (
                {"effort": reasoning["effort"]} if route == "/responses" else reasoning["effort"])
    if route == "/chat/completions" and value.get("stream"):
        value["stream_options"] = {**value.get("stream_options", {}), "include_usage": True}
    tool_policy = profile.get("tool_calling", {})
    if "parallel_tool_calls" in tool_policy and route != "/messages":
        value["parallel_tool_calls"] = tool_policy["parallel_tool_calls"]
    return value


def make_server(url: str, backend: str, models: set[str], profile: dict, *, port: int = 8011):
    profile = validate_profile(profile)
    client = EndpointClient(url, backend, timeout=profile.get("request_timeout", 120))
    semaphore = threading.BoundedSemaphore(profile.get("concurrency", 1))
    contract = {"upstream_base_url": base_url(url), "backend": backend, "models": sorted(models), "policy": profile}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass  # Never persist request bodies, credentials, or role prompts.

        def fail(self, status, message):
            encoded = json.dumps({"error": {"message": message, "type": "local_endpoint_error"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/llh-policy":
                encoded = json.dumps(contract).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(encoded)
            elif self.path in {"/v1/models", "/version"}:
                self.forward(None)
            else:
                self.fail(404, "unknown route")

        def do_POST(self):
            route = self.path.removeprefix("/v1")
            if route not in {"/responses", "/messages", "/chat/completions"}:
                self.fail(404, "unsupported inference route; no protocol fallback")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                payload = policy_payload(route, payload, profile, models)
            except (ValueError, TypeError, AttributeError) as exc:
                self.fail(400, str(exc))
                return
            self.forward(payload)

        def forward(self, payload):
            path = self.path.removeprefix("/v1")
            target = client
            if path == "/version":
                target = EndpointClient(url, backend)
                target.url = target.url.removesuffix("/v1")
            started = False
            with semaphore:
                try:
                    with target.request(path, payload) as response:
                        self.send_response(response.status)
                        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        self.end_headers()
                        started = True
                        # read1 preserves SSE latency, unlike read(n) which waits
                        # for n bytes or EOF. Propagate errors without retrying.
                        while chunk := response.read1(65536):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except (EndpointError, OSError):
                    if not started:
                        self.fail(502, "local upstream unavailable; no fallback")
                    self.close_connection = True

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args(argv)
    defaults = load_run_defaults(args.config)
    profile = defaults.get("local_model", {})
    url = profile.get("upstream_base_url")
    if not url:
        parser.error("local_model.upstream_base_url is required")
    models = {str(value) for key, value in defaults.items() if key == "model" or key.endswith("_model") and isinstance(value, str)}
    server = make_server(url, defaults.get("agent", "codex"), models, profile, port=args.port)
    print(f"Explicit local inference policy proxy on http://127.0.0.1:{server.server_port}/v1 -> {base_url(url)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
