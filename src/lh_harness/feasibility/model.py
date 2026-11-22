"""JSON-only AgentAdapter using the existing configured provider, without tools.

CommandAgentAdapter requires Environment.exec, so it cannot safely back a truly
tool-less Reflector. This small adapter implements the same run_episode protocol
and reuses provider URL normalization/configuration, but supplies no tools.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.request

from ..harbor_url import claude_code_upstream_base_url
from ..types import EpisodeResult


def model_output(result: EpisodeResult) -> str:
    from ..agent_logs import visible_output
    for key in ("assistant_visible_output", "visible_output", "output"):
        if isinstance(result.metadata.get(key), str):
            return result.metadata[key]
    return visible_output(result.actions_log)


class JsonModelAgent:
    def __init__(self, *, model: str, base_url: str | None = None, api_key: str | None = None,
                 protocol: str = "openai", max_tokens: int = 1800):
        if protocol not in {"openai", "anthropic"}:
            raise ValueError("invalid JSON model protocol")
        self.model, self.protocol, self.max_tokens = model, protocol, max_tokens
        prefix = "ANTHROPIC" if protocol == "anthropic" else "OPENAI"
        self.base_url = base_url or os.getenv(prefix+"_BASE_URL") or ("https://api.anthropic.com" if protocol == "anthropic" else "https://api.openai.com")
        self.api_key = api_key or os.getenv(prefix+"_API_KEY")

    async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
        # env is deliberately ignored. No command, plugin, MCP, file, or GUI tool.
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def resolve(value, error):
            if not future.done():
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(value)
        def request():
            try:
                value, error = self._request(prompt, budget.max_duration_seconds), None
            except Exception as exc:
                value, error = None, exc
            try:
                loop.call_soon_threadsafe(resolve, value, error)
            except (RuntimeError, OSError):
                pass  # Optional call timed out and its owning loop already closed.
        # An optional DNS/HTTP failure must not delay the original loop's shutdown
        # by waiting for a default-executor worker. This worker has no tool access.
        threading.Thread(target=request, daemon=True, name="feasibility-json").start()
        return await asyncio.wait_for(future, budget.max_duration_seconds)

    def _request(self, prompt, timeout):
        if not self.model or not self.api_key:
            raise ValueError("Reflector JSON model credentials/model unavailable")
        url = claude_code_upstream_base_url(self.base_url)
        headers = {"Content-Type": "application/json"}
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if self.protocol == "openai":
            url += "/v1/chat/completions"
            headers["Authorization"] = "Bearer "+self.api_key
        else:
            url += "/v1/messages"
            headers.update({"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        started = time.monotonic()
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read(2_000_000))
        if self.protocol == "openai":
            message = payload["choices"][0]["message"]
            if message.get("tool_calls") or message.get("function_call"):
                raise ValueError("tool calls forbidden")
            output = message.get("content") or ""
            usage = payload.get("usage") or {}
            tokens = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
            finish = payload["choices"][0].get("finish_reason")
        else:
            if any(b.get("type") == "tool_use" for b in payload.get("content", [])):
                raise ValueError("tool calls forbidden")
            output = "\n".join(b["text"] for b in payload.get("content", []) if b.get("type") == "text")
            usage = payload.get("usage") or {}
            inp = usage.get("input_tokens")
            tokens = {"input_tokens": inp + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) if type(inp) is int else None,
                      "output_tokens": usage.get("output_tokens")}
            finish = payload.get("stop_reason")
        return EpisodeResult(status="done", metadata={"assistant_visible_output": output,
            "usage": tokens, "finish_reason": finish}, duration_ms=int((time.monotonic()-started)*1000))


class NoToolEnvironment:
    async def exec(self, *args, **kwargs):
        raise PermissionError("Reflector cannot execute tools")

    async def screenshot(self, *args, **kwargs):
        raise PermissionError("Reflector cannot inspect live environment")

    async def upload(self, *args, **kwargs):
        raise PermissionError("Reflector cannot upload")

    async def download(self, *args, **kwargs):
        raise PermissionError("Reflector cannot download")


def parse_json(text: str):
    import re
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    text = text.lstrip("\ufeff")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value
