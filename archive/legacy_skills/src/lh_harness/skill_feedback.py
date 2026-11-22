"""Post-verifier Skill learning, using the AgentAdapter protocol and no tools.

Only saved evidence is supplied. This adapter cannot execute commands or touch
the task workspace after Harbor has completed its independent verification.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from .harbor_url import claude_code_upstream_base_url
from .skill_learning import SkillBank, SkillSession, token_usage
from .types import EpisodeBudget, EpisodeResult


class SkillFeedbackAgent:
    def __init__(self, template: dict[str, Any], log_dir: Path) -> None:
        config = template["agents"][0]
        self.kwargs = config.get("kwargs") or {}
        self.model = self.kwargs.get("verifier_model") or self.kwargs.get("harness_model") or config.get("model_name")
        configured_env = config.get("env") or {}
        self.protocol = os.getenv("LH_SKILL_FEEDBACK_PROTOCOL") or (
            "openai" if str(configured_env.get("LH_HARNESS_PROXY_OPENAI_BRIDGE", os.getenv("LH_HARNESS_PROXY_OPENAI_BRIDGE", "0"))) == "1" else "anthropic")
        self.bearer_auth = str(configured_env.get("LH_HARNESS_PROXY_BEARER_AUTH", os.getenv("LH_HARNESS_PROXY_BEARER_AUTH", "0"))) == "1"
        if self.protocol not in {"openai", "anthropic"}:
            raise ValueError("Skill feedback protocol must be openai or anthropic")
        self.log_dir = log_dir
        self.usage: list[dict[str, Any]] = []

    async def run_episode(self, prompt, env, budget, live_trajectory_path=None):
        del env, live_trajectory_path
        return await asyncio.to_thread(self._request, prompt, budget.max_duration_seconds)

    def _request(self, prompt: str, timeout: int) -> EpisodeResult:
        base = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not base or not key or not self.model:
            raise ValueError("Skill feedback needs the configured model API environment")
        base = claude_code_upstream_base_url(base)
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "max_tokens": min(int(self.kwargs.get("max_tokens", 8192)), 8192),
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": self.kwargs.get("temperature", 1.0)}
        if self.protocol == "openai":
            url = base + "/v1/chat/completions"
            headers["Authorization"] = "Bearer " + key
            payload.update((self.kwargs.get("llm_call_kwargs") or {}).get("extra_body") or {})
        else:
            url = base + "/v1/messages"
            headers.update({"anthropic-version": "2023-06-01"})
            headers["Authorization" if self.bearer_auth else "x-api-key"] = "Bearer " + key if self.bearer_auth else key
        # Deliberately do not provide tools or allow a tool-call execution loop.
        started = time.monotonic()
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        if self.protocol == "openai":
            message = body["choices"][0]["message"]
            if message.get("tool_calls"):
                raise ValueError("Skill feedback does not permit tool calls")
            output = message.get("content") or ""
            raw_usage = body.get("usage") or {}
            usage = {"input_tokens": raw_usage.get("prompt_tokens"), "output_tokens": raw_usage.get("completion_tokens")}
        else:
            if any(b.get("type") == "tool_use" for b in body.get("content", [])):
                raise ValueError("Skill feedback does not permit tool calls")
            output = "\n".join(b["text"] for b in body.get("content", []) if b.get("type") == "text")
            raw_usage = body.get("usage") or {}
            inp = raw_usage.get("input_tokens")
            usage = {"input_tokens": inp + (raw_usage.get("cache_read_input_tokens") or 0) +
                     (raw_usage.get("cache_creation_input_tokens") or 0) if type(inp) is int else None,
                     "output_tokens": raw_usage.get("output_tokens")}
        self.usage.append(usage)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Headers/credentials never enter trace artifacts.
        (self.log_dir/f"episode_{len(self.usage):03d}.json").write_text(json.dumps({
            "prompt": prompt, "output": output, "model": self.model, "protocol": self.protocol,
            "usage": usage, "elapsed_seconds": time.monotonic()-started}, ensure_ascii=False, indent=2)+"\n")
        return EpisodeResult(status="done", metadata={"assistant_visible_output": output, "usage": usage})


async def learn_from_verifier(*, report: dict[str, Any], routes: list[dict[str, Any]],
        outcome: dict[str, Any], bank_path: Path, log_dir: Path, language: str,
        agent: Any, token_budget: int = 2000) -> dict[str, Any]:
    if type(outcome.get("correct")) is not bool:
        return {"mode": "post_verifier", "learning_phase": "skipped", "skip_reason": "external_outcome_unavailable"}
    saved = copy.deepcopy(report)
    saved["skill_external_outcome"] = copy.deepcopy(outcome)
    class MeteredAgent:
        counts: list[int | None]
        def __init__(self):
            self.counts = []
        async def run_episode(self, prompt, env, budget):
            if len(self.counts) >= 8:
                raise RuntimeError("post-verifier Skill model-call budget exhausted")
            index = len(self.counts)
            self.counts.append(None)  # Failed calls have unknown usage, not zero cost.
            result = await agent.run_episode(prompt, env, budget)
            self.counts[index] = token_usage(result)
            return result
    metered = MeteredAgent()
    session = SkillSession(SkillBank.load(bank_path), language, metered, None, EpisodeBudget(max_duration_seconds=600),
                           log_dir, token_budget=token_budget, benchmark_id="terminal_bench_2_1",
                           interaction_mode="cli", dataset_id="terminal_bench_pro")
    session.records = copy.deepcopy(routes)
    # Usage here is exclusively post-verifier learning. Original task usage
    # remains in Harbor's immutable result and is not overwritten.
    result = await session.finish(saved)
    task_tokens = (report.get("skill_learning") or {}).get("task_tokens")
    result.update(mode="post_verifier", external_correct=outcome["correct"],
                  task_tokens=task_tokens, token_usage_missing=type(task_tokens) is not int,
                  learning_tokens=sum(metered.counts) if all(n is not None for n in metered.counts) else None,
                  learning_llm_calls=len(metered.counts), external_result_path=outcome.get("result_path"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir/'feedback.json').write_text(json.dumps({"external_outcome": outcome, "learning": result},
                                                   ensure_ascii=False, indent=2)+"\n")
    return result
