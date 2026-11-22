"""Sidecar reporting; original task reports and benchmark rewards are untouched."""
from __future__ import annotations

import json
from pathlib import Path

from .bank import atomic_json


def write_summary(rounds_dir: Path, target: Path):
    rows = []
    for path in sorted(rounds_dir.glob("round_*/feasibility.json")):
        value = json.loads(path.read_text())
        rows.append({"reference": str(path.relative_to(rounds_dir.parent)),
                     "decision": value["result"]["decision"], "dispatch": value["dispatch"],
                     "skill_ids": value["result"]["retrieved_skills"],
                     "contract": value["final_contract"], "trace": value["trace"],
                     "auditor_report_reference": (value.get("subsequent_auditor") or {}).get("reference")})
    def total(key):
        values = [r["trace"].get(key) for r in rows]
        return sum(values) if all(type(v) is int for v in values) else None
    atomic_json(target, {"schema_version": 1, "rounds": rows,
        "llm_calls": total("llm_calls"), "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"), "runtime_ms": total("runtime_ms"),
        "note": "Reflector overhead only; UNKNOWN/missing provider usage is not counted as zero."})
