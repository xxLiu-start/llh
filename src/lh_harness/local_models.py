"""Explicit endpoint contracts and active protocol probes, independent of MEA."""
from __future__ import annotations

import hashlib
import base64
import binascii
import ipaddress
import json
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
import zlib
from datetime import datetime, timezone

from .token_usage import normalize

PROTOCOLS = {"codex": "responses", "claude_code": "messages",
             "opencode": "chat_completions", "deepseek_harness": "chat_completions"}
KEY_ENVS = {"codex": ("OPENAI_API_KEY", "CODEX_API_KEY"),
            "claude_code": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
            "opencode": ("OPENCODE_API_KEY", "OPENAI_API_KEY"),
            "deepseek_harness": ("DEEPSEEK_API_KEY",)}


class EndpointError(ValueError):
    pass


class EndpointAuthenticationError(EndpointError):
    pass


class EndpointUnavailableError(EndpointError):
    pass


def base_url(value: str) -> str:
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise EndpointError("base_url must be an HTTP(S) API root without credentials/query/fragment")
    clean = value.rstrip("/")
    if clean.endswith(("/chat/completions", "/responses", "/messages")):
        raise EndpointError("base_url must be the API root, e.g. http://127.0.0.1:8010/v1, not a generation route")
    return clean if clean.endswith("/v1") else clean + "/v1"


def is_local(value: str | None) -> bool:
    host = urllib.parse.urlsplit(value or "").hostname
    if host and (host == "localhost" or host.endswith((".localhost", ".local", ".lan", ".internal"))
                 or "." not in host and ":" not in host):
        return True
    try:
        address = ipaddress.ip_address(host or "")
        return address.is_private or address.is_loopback
    except ValueError:
        return False


def credential(backend: str) -> str | None:
    return next((os.environ[k] for k in KEY_ENVS[backend] if os.environ.get(k)), None)


def validate_profile(profile: dict) -> dict:
    allowed = {"enabled", "protocol", "serving_framework", "serving_version", "weights", "weights_sha256",
               "quantization", "gpu_hardware", "context_length", "max_output_tokens", "temperature", "top_p",
               "reasoning", "tool_calling", "concurrency", "upstream_base_url", "tokenizer_path",
               "minimum_context_length", "minimum_output_tokens", "request_timeout", "server_evidence", "require_image",
               "image_probe_count"}
    if not isinstance(profile, dict) or set(profile) - allowed:
        raise EndpointError("invalid [local_model] key(s)")
    for name in ("context_length", "max_output_tokens", "concurrency", "minimum_context_length", "minimum_output_tokens", "request_timeout", "image_probe_count"):
        if name in profile and (type(profile[name]) is not int or profile[name] < 1):
            raise EndpointError(f"local_model.{name} must be a positive integer")
    for name, high in (("temperature", 2), ("top_p", 1)):
        if name in profile and (type(profile[name]) not in (int, float) or not 0 <= profile[name] <= high):
            raise EndpointError(f"local_model.{name} is out of range")
    if "enabled" in profile and type(profile["enabled"]) is not bool:
        raise EndpointError("local_model.enabled must be boolean")
    if "require_image" in profile and type(profile["require_image"]) is not bool:
        raise EndpointError("local_model.require_image must be boolean")
    if "image_probe_count" in profile and not profile.get("require_image"):
        raise EndpointError("local_model.image_probe_count requires require_image=true")
    if "protocol" in profile and profile["protocol"] not in set(PROTOCOLS.values()):
        raise EndpointError("unsupported local_model.protocol")
    for name in ("reasoning", "tool_calling"):
        if name in profile and not isinstance(profile[name], dict):
            raise EndpointError(f"local_model.{name} must be a table")
    return dict(profile)


class EndpointClient:
    def __init__(self, url: str, backend: str, *, timeout: int = 60, api_key: str | None = None, protocol: str | None = None,
                 auth_scheme: str | None = None):
        self.url, self.backend, self.timeout = base_url(url), backend, timeout
        self.protocol = protocol or PROTOCOLS[backend]
        self.key = api_key or credential(backend)
        self.bearer_auth = self.protocol != "messages" or (
            backend == "claude_code" and not api_key and not os.environ.get("ANTHROPIC_API_KEY")
            and bool(os.environ.get("ANTHROPIC_AUTH_TOKEN")))
        if auth_scheme not in {None, "bearer", "x-api-key", "none"}:
            raise EndpointError("auth_scheme must be bearer, x-api-key or none")
        self.auth_scheme = auth_scheme or ("bearer" if self.bearer_auth else "x-api-key")
        self.bearer_auth = self.auth_scheme == "bearer"
        if self.auth_scheme == "none":
            self.key = None
        # Loopback must not be sent to the shell's HTTP proxy. Never follow a
        # redirect: it could silently move inference or credentials to the cloud.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        handlers = [NoRedirect()]
        if is_local(url):
            handlers.append(urllib.request.ProxyHandler({}))
        self.opener = urllib.request.build_opener(*handlers)

    def request(self, path: str, payload: dict | None = None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers.update({"Authorization": "Bearer " + self.key} if self.auth_scheme == "bearer" else {"x-api-key": self.key})
        if self.protocol == "messages":
            headers["anthropic-version"] = "2023-06-01"
        req = urllib.request.Request(self.url + path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
        try:
            return self.opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            # Do not echo arbitrary upstream bodies, which can contain credentials.
            if exc.code in {401, 403}:
                raise EndpointAuthenticationError(
                    f"{path}: HTTP {exc.code}; authentication failed at {self.url} "
                    f"(auth_scheme={self.auth_scheme}, credential={'set' if self.key else 'unset'}; no fallback)"
                ) from None
            raise EndpointError(f"{path}: HTTP {exc.code}; endpoint/protocol/model compatibility check failed (no fallback)") from None
        except (OSError, urllib.error.URLError) as exc:
            raise EndpointUnavailableError(
                f"{path}: endpoint unavailable at {self.url} ({type(exc).__name__}); "
                "start the model service and wait for readiness; no fallback"
            ) from None

    def json(self, path: str, payload: dict | None = None):
        with self.request(path, payload) as response:
            try:
                return json.load(response)
            except (ValueError, UnicodeError):
                raise EndpointError(f"{path}: response is not JSON") from None


def stream_events(response):
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            text = "\n".join(data)
            data = []
            if text != "[DONE]":
                try:
                    yield json.loads(text)
                except ValueError:
                    raise EndpointError("malformed SSE JSON") from None
    if data:
        raise EndpointError("incomplete SSE frame")


def probe_image_base64() -> str:
    """A deterministic 56x56 PNG for active protocol checks, with no file I/O."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xffffffff)
    pixels = (b"\0" + b"\xff\0\0" * 56) * 56
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 56, 56, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def check_endpoint(url: str, backend: str, model: str, profile: dict | None = None, *, api_key: str | None = None,
                   auth_scheme: str | None = None) -> dict:
    profile = validate_profile(profile or {})
    protocol = profile.get("protocol", PROTOCOLS[backend])
    if protocol != PROTOCOLS[backend] and backend != "opencode":
        raise EndpointError(f"{backend} requires {PROTOCOLS[backend]}, not {protocol}; configure an explicit independent protocol adapter")
    if backend == "opencode":
        if "/" not in model or not model.partition("/")[0] or not model.partition("/")[2]:
            raise EndpointError("OpenCode requires provider/model; configure a local provider explicitly")
        wire_model = model.split("/", 1)[1]
    else:
        wire_model = model
    client = EndpointClient(url, backend, timeout=profile.get("request_timeout", 60), api_key=api_key, protocol=protocol,
                            auth_scheme=auth_scheme)
    models = client.json("/models")
    if not isinstance(models, dict) or not isinstance(models.get("data"), list) or not all(isinstance(item, dict) for item in models["data"]):
        raise EndpointError("/models: malformed model inventory")
    found = next((m for m in models.get("data", []) if m.get("id") == wire_model), None)
    if found is None:
        raise EndpointError(f"model ID {wire_model!r} is not in /models; available IDs: {[m.get('id') for m in models.get('data', [])]}; no substitution")
    context = found.get("max_model_len") or profile.get("context_length")
    if found.get("max_model_len") and profile.get("context_length") and profile["context_length"] != found["max_model_len"]:
        raise EndpointError("configured context_length differs from the loaded model's advertised limit")
    minimum = profile.get("minimum_context_length", 8192)
    output = profile.get("max_output_tokens")
    if type(context) is not int or context < minimum:
        raise EndpointError(f"context length is unknown or below required {minimum}; supply verified local_model.context_length")
    if output is not None and (output < profile.get("minimum_output_tokens", 2048) or output >= context):
        raise EndpointError("max_output_tokens does not meet the role prompt requirement/context capacity")
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
    common = {"model": wire_model, "stream": True}
    prompt = "Call endpoint_probe once with value OK. Do not call any other tool."
    function = {"name": "endpoint_probe", "description": "Report a harmless probe value", "parameters": parameters}
    if protocol == "responses":
        route = "/responses"
        payload = {**common, "input": prompt, "store": False, "max_output_tokens": 256,
                   "tools": [{"type": "function", **function}], "tool_choice": {"type": "function", "name": "endpoint_probe"}}
    elif protocol == "messages":
        route = "/messages"
        payload = {**common, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
                   "tools": [{"name": "endpoint_probe", "description": function["description"], "input_schema": parameters}],
                   "tool_choice": {"type": "tool", "name": "endpoint_probe"}}
        # Exercise the configured reasoning policy, including the native
        # Messages extension used by local vLLM. Server defaults alone do not
        # guarantee that CLI requests will use the same template settings.
        reasoning = profile.get("reasoning", {})
        if "enable_thinking" in reasoning:
            if type(reasoning["enable_thinking"]) is not bool:
                raise EndpointError("reasoning.enable_thinking must be boolean")
            payload["chat_template_kwargs"] = {"enable_thinking": reasoning["enable_thinking"]}
            if reasoning["enable_thinking"] is False:
                payload["thinking"] = {"type": "disabled"}
        if "effort" in reasoning:
            payload["output_config"] = {"effort": reasoning["effort"]}
    else:
        route = "/chat/completions"
        payload = {**common, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
                   "tools": [{"type": "function", "function": function}],
                   "tool_choice": {"type": "function", "function": {"name": "endpoint_probe"}},
                   "stream_options": {"include_usage": True}}
    if profile.get("require_image"):
        encoded = probe_image_base64()
        image_count = profile.get("image_probe_count", 1)
        if protocol == "messages":
            payload["messages"][0]["content"] = [
                *[{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}
                  for _ in range(image_count)],
                {"type": "text", "text": prompt}]
        elif protocol == "responses":
            payload["input"] = [{"role": "user", "content": [
                *[{"type": "input_image", "image_url": "data:image/png;base64," + encoded}
                  for _ in range(image_count)],
                {"type": "input_text", "text": prompt}]}]
        else:
            payload["messages"][0]["content"] = [
                *[{"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
                  for _ in range(image_count)],
                {"type": "text", "text": prompt}]
    tool, terminal, frames, usage_data, arguments = False, False, 0, {}, ""
    with client.request(route, payload) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise EndpointError("endpoint did not return an SSE stream")
        for event in stream_events(response):
            frames += 1
            kind = event.get("type", "")
            if kind in {"error", "response.failed"} or event.get("error"):
                raise EndpointError("endpoint returned an error inside the stream")
            if protocol == "responses":
                if kind == "response.output_item.done" and event.get("item", {}).get("type") == "function_call":
                    item = event["item"]
                    tool = item.get("name") == "endpoint_probe"
                    arguments = item.get("arguments", "")
                if kind == "response.completed":
                    terminal = True
                    usage_data = event.get("response", {}).get("usage") or {}
            elif protocol == "messages":
                if kind == "content_block_start":
                    item = event.get("content_block", {})
                    tool |= item.get("type") == "tool_use" and item.get("name") == "endpoint_probe"
                    if item.get("input"):
                        arguments = json.dumps(item["input"])
                if kind == "content_block_delta":
                    arguments += event.get("delta", {}).get("partial_json", "")
                usage_data.update(event.get("message", {}).get("usage") or event.get("usage") or {})
                terminal |= kind == "message_stop"
            else:
                for choice in event.get("choices", []):
                    terminal |= choice.get("finish_reason") is not None
                    for call in choice.get("delta", {}).get("tool_calls", []):
                        fn = call.get("function", {})
                        tool |= fn.get("name") == "endpoint_probe"
                        arguments += fn.get("arguments", "")
                if event.get("usage"):
                    usage_data = event["usage"]
    try:
        valid_arguments = json.loads(arguments).get("value") == "OK"
    except (ValueError, AttributeError):
        valid_arguments = False
    if not tool or not valid_arguments or not terminal or frames < 2:
        raise EndpointError("stream/tool-call check failed: missing complete endpoint_probe(value=OK) or terminal event")
    # Invalid models must be rejected, rather than served by a default model.
    bad = dict(payload, model="llh-intentionally-missing-model")
    try:
        with client.request(route, bad) as response:
            response.read(4096)
    except EndpointError as exc:
        if "HTTP 404" not in str(exc) and "HTTP 400" not in str(exc):
            raise EndpointError("invalid-model error handling could not be verified") from None
    else:
        raise EndpointError("endpoint silently accepted a nonexistent model")
    version = None
    try:
        origin = client.url.removesuffix("/v1")
        version_client = EndpointClient(origin, backend, api_key=api_key, auth_scheme=auth_scheme)
        version_client.url = origin
        version = version_client.json("/version").get("version")
    except EndpointError:
        pass
    usage = normalize(usage_data)
    return {"status": "ok", "checked_at": datetime.now(timezone.utc).isoformat(),
            "model_source": "local", "backend": backend, "model_id": model, "wire_model_id": wire_model,
            "base_url": client.url, "protocol": protocol, "context_length": context,
            "auth_scheme": client.auth_scheme,
            "context_length_source": "endpoint" if found.get("max_model_len") else "operator_declared",
            "serving_framework": found.get("owned_by") or profile.get("serving_framework"),
            "serving_version": version or profile.get("serving_version"),
            "quantization": profile.get("quantization"), "weights": profile.get("weights"),
            "weights_sha256": profile.get("weights_sha256"), "gpu_hardware": profile.get("gpu_hardware"),
            "max_output_tokens": output, "temperature": profile.get("temperature"), "top_p": profile.get("top_p"),
            "reasoning": profile.get("reasoning"), "tool_calling": profile.get("tool_calling"),
            "concurrency": profile.get("concurrency"), "streaming": "verified", "tool_probe": "verified",
            "image_probe": "verified" if profile.get("require_image") else "not_requested",
            "image_probe_count": profile.get("image_probe_count", 1) if profile.get("require_image") else 0,
            "invalid_model_rejected": True, "usage": usage,
            "validation_scope": "active API protocol probe; CLI tool execution requires a separate acceptance run",
            "unverified_metadata": [key for key in ("weights_sha256", "quantization", "gpu_hardware", "max_output_tokens", "temperature", "top_p", "reasoning", "concurrency") if profile.get(key) is None]}


def fingerprint(value: dict) -> str:
    keys = ("backend", "model_id", "wire_model_id", "base_url", "protocol", "context_length", "serving_framework",
            "serving_version", "quantization", "weights", "weights_sha256", "gpu_hardware", "max_output_tokens",
            "temperature", "top_p", "reasoning", "reasoning_effort", "tool_calling", "concurrency")
    return hashlib.sha256(json.dumps({key: value.get(key) for key in keys}, sort_keys=True).encode()).hexdigest()
