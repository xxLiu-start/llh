"""Adapter at the existing dispatch boundary. No live environment reads."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ..types import ManagedRound
from .schema import Assessment, ReflectorConfig, Snapshot


def snapshot_for_contract(*, task: str, contract: str, plan: str, manager_state: str,
                          rounds: list[ManagedRound], route: str, tools=(), observations=()) -> Snapshot:
    # Include prior auditor reports only; never the current/future Executor result.
    verified = tuple({"round": r.round_index, "report": r.auditor_report,
                      "auditor_status": r.auditor_status,
                      "reference": f"rounds/round_{r.round_index:03d}/auditor_report.txt"}
                     for r in rounds if r.auditor_report)
    reference = hashlib.sha256(json.dumps(verified, sort_keys=True).encode()).hexdigest()
    return Snapshot(task, contract, plan, manager_state, verified, tuple(tools), False,
                    tuple(observations), "sha256:"+reference, route)


def disposition(config: ReflectorConfig, result: Assessment, trace: dict[str, Any], revisions: int) -> str:
    if not config.active or config.mode == "shadow":
        return "dispatch"
    if trace.get("runtime_error"):
        return "dispatch" if config.runtime_error_policy == "fail_open" else "block"
    if result.decision == "ACCEPT":
        return "dispatch"
    if result.decision == "UNKNOWN":
        return "dispatch" if config.unknown_policy == "pass_through" else "block"
    if result.decision == "BLOCK":
        return "block"
    if revisions < config.max_revision_rounds:
        return "revise"
    return "dispatch" if config.revision_exhausted_policy == "pass_through" else "block"


def feedback(result: Assessment) -> str:
    return ("Feasibility feedback (pre-execution, NOT an Auditor verification). "
            "Regenerate the next contract with minimal corrections; preserve the original user task, "
            "role protocol, execution tools and Auditor authority.\n"+
            json.dumps(result.to_dict(), ensure_ascii=False))
