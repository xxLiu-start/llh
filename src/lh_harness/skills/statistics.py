from __future__ import annotations
from ..token_usage import aggregate


def summarize(records, *, prices=None, training_tokens=None, successful_tasks=0):
    categories = {role: aggregate([r["usage"] for r in records if r["role"] == role])
                  for role in ("manager", "executor", "auditor", "auxiliary", "adherence", "attribution", "proposer", "curator", "general_curator")}
    runtime = aggregate([r["usage"] for r in records if r["role"] in {"manager", "executor", "auditor", "auxiliary"}])
    learning = aggregate([r["usage"] for r in records if r["role"] in {"adherence", "attribution", "proposer", "curator", "general_curator"}])
    total = aggregate([r["usage"] for r in records])
    price = prices or {}
    cost = None
    if total.get("status") == "reported" and all(k in price for k in ("input_per_million", "output_per_million")):
        cost = (total["input_tokens"] * price["input_per_million"] + total["output_tokens"] * price["output_per_million"]) / 1e6
    value = runtime.get("total_tokens")
    return {"categories": categories, "runtime": runtime, "offline_learning": learning, "total": total,
            "api_cost": cost, "currency": price.get("currency"),
            "duration_ms": sum(r.get("duration_ms", 0) for r in records),
            "tokens_per_successful_task": value / successful_tasks if value is not None and successful_tasks else None,
            "amortized_tokens_per_successful_task": (value + training_tokens) / successful_tasks
             if value is not None and training_tokens is not None and successful_tasks else None,
            "injection_accounting": "subset of role input; never added to total twice"}
