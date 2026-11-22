"""Tool-free offline learner using the explicitly configured existing endpoint."""
from __future__ import annotations

import json
import copy
import os
import time
from pathlib import Path
from ..local_models import EndpointClient
from ..token_usage import normalize
from .library import atomic_json
from .schema import validate_object


class JsonClient:
    def __init__(self, *, backend, model, base_url, profile=None, output=None, api_key=None, auth_scheme="none"):
        if not model or not base_url:
            raise ValueError("offline learning requires an explicit model and endpoint; no defaults or fallback")
        self.client = EndpointClient(base_url, backend, api_key=api_key,
                                     auth_scheme=auth_scheme, timeout=180)
        self.model, self.profile, self.output = model, profile or {}, Path(output) if output else None
        self.records = []

    def __call__(self, stage, role, instructions, payload, schema):
        payload = copy.deepcopy(payload)
        bundle = payload.get("bundle")
        if bundle:
            from .trajectory_compactor import redact
            import re
            for ref in bundle.get("evidence", {}).values():
                text = ref["text"]
                if ref["role"] not in {"auditor", "benchmark"} and len(text) > 4000:
                    important = [line for line in text.splitlines() if re.search(r"error|fail|exception|timeout|blocked|错误|失败|阻塞",line,re.I)]
                    ref["text"] = text[:1400] + "\n[ordinary output omitted; consult evidence references]\n" + "\n".join(important) + "\n" + text[-1400:]
            # Canonical references retain all audit/error evidence; don't send
            # duplicate large state/output fields in addition to those refs.
            for role_name in ("manager", "executor", "auditor"):
                if role_name in bundle:
                    bundle[role_name] = {key:value for key,value in bundle[role_name].items()
                        if key not in {"plan", "output", "report", "report_text", "task_state", "task_contract"}}
            for summaries in bundle.get("trajectories", {}).values():
                for summary in summaries:
                    summary["tool_calls"] = summary["tool_calls"][:8] + summary["tool_calls"][-8:]
        prompt = instructions + "\nOutput JSON schema:\n" + json.dumps(schema) + "\nEvidence data:\n" + json.dumps(payload, ensure_ascii=False)
        started = time.monotonic()
        common = {"model": self.model, "stream": False}
        protocol = self.client.protocol
        if protocol == "messages":
            route = "/messages"
            body = {**common, "max_tokens": self.profile.get("max_output_tokens", 4096),
                    "messages": [{"role": "user", "content": prompt}], "thinking": {"type": "disabled"},
                    "chat_template_kwargs": {"enable_thinking": False},
                    "output_config": {"format": {"type": "json_schema", "schema": schema}}}
        elif protocol == "responses":
            route = "/responses"
            body = {**common, "input": prompt, "max_output_tokens": self.profile.get("max_output_tokens", 4096),
                    "text": {"format": {"type": "json_schema", "name": "role_skill_decision", "schema": schema, "strict": True}}}
        else:
            route = "/chat/completions"
            body = {**common, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.profile.get("max_output_tokens", 4096),
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {"type": "json_schema", "json_schema": {"name": "role_skill_decision", "schema": schema, "strict": True}}}
        for name in ("temperature", "top_p"):
            if name in self.profile:
                body[name] = self.profile[name]
        row = {"role": stage, "target_role": role, "model": self.model, "backend_protocol": protocol}
        try:
            result = self.client.json(route, body)
            row["usage"] = normalize(result.get("usage"))
            if protocol == "messages":
                text = "\n".join(b["text"] for b in result.get("content", []) if b.get("type") == "text")
            elif protocol == "responses":
                text = "\n".join(b["text"] for item in result.get("output", []) for b in item.get("content", []) if b.get("type") == "output_text")
            else:
                message = result["choices"][0]["message"]
                if message.get("tool_calls"):
                    raise ValueError("offline learner must not call tools")
                text = message.get("content", "")
            row["raw_output"] = text
            # A single Markdown JSON fence is an envelope, not a field parser.
            # The enclosed object still passes the exact closed JSON schema.
            text = text.strip()
            if text.startswith("```json\n") and text.endswith("```"):
                text = text[8:-3].strip()
            elif text.startswith("```\n") and text.endswith("```"):
                text = text[4:-3].strip()
            value = json.loads(text)
            validate_object(value, schema)
            row.update(status="ok", output=value)
            return value
        except Exception as exc:
            row.update(status="error", error=type(exc).__name__)
            raise
        finally:
            row["duration_ms"] = int((time.monotonic() - started) * 1000)
            row.setdefault("usage", normalize(None))
            self.records.append(row)
            if self.output:
                atomic_json(self.output / f"{len(self.records):04d}-{stage}-{role}.json", row)
                from .statistics import summarize
                atomic_json(self.output / "learning_cost.json", summarize(self.records))
