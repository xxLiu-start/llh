from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from lh_harness.feasibility import FeasibilityReflector, ReflectorConfig
from lh_harness.feasibility.shadow_run import collect, main
from tests.test_feasibility import JsonAgent, snapshot, verdict


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["shadow", "gate"])
async def test_empty_bank_assesses_without_inventing_skill_usage(tmp_path, mode):
    config = ReflectorConfig(enabled=True, mode=mode,
                             skill_bank_path=str(tmp_path/"empty-bank"))
    model = JsonAgent([verdict("ACCEPT")])
    result, trace = await FeasibilityReflector(config, model).assess(snapshot())
    assert result.decision == "ACCEPT" and result.retrieved_skills == []
    assert trace["llm_calls"] == 1 and trace["used_skill_guidance"] is False
    assert not (tmp_path/"empty-bank").exists()


def test_shadow_launcher_dry_run_and_reserved_flags(tmp_path):
    output = tmp_path/"preview"
    assert main(["--output", str(output), "--dry-run"]) == 0
    config = json.loads((output/"reflector_config.json").read_text())
    assert config["mode"] == "shadow" and config["enabled"] is True
    assert config["max_revision_rounds"] == 0
    assert config["enable_skill_learning"] is False
    assert not (output/"runtime").exists()
    with pytest.raises(SystemExit): main(["--output", str(output), "--dry-run"])
    with pytest.raises(SystemExit): main(["--dry-run", "--feasibility-reflector-config", "gate.json"])
    with pytest.raises(SystemExit): main(["--dry-run", "--skip-without-skills"])


def test_launcher_streams_logs_collects_rounds_and_preserves_native_exit(tmp_path, monkeypatch):
    import lh_harness.feasibility.shadow_run as module
    output = tmp_path/"run"
    observed = []
    class Process:
        pid = 99999
        stdout = io.StringIO("original MEA progress\n[feasibility] REVISE · dispatch\n")
        def __init__(self, command, **kwargs):
            observed.append(command)
            assert kwargs["start_new_session"] is True
            role = output/"runtime/main/lh_harness/role_orchestration"
            directory = role/"rounds/round_001"; directory.mkdir(parents=True)
            (directory/"manager_plan.txt").write_text("unchanged manager plan")
            (directory/"task_contract.txt").write_text("unchanged contract")
            (directory/"auditor_report.txt").write_text("actual fixture audit")
            sidecar = {"result": {"decision": "REVISE", "issues": [], "retrieved_skills": []},
                "dispatch": "dispatch", "trace": {"llm_calls": 1, "input_tokens": 10, "output_tokens": 5, "runtime_ms": 1}}
            (directory/"feasibility.json").write_text(json.dumps(sidecar))
            last = role/"rounds/round_002"; last.mkdir()
            (last/"task_contract.txt").write_text("unchanged contract")
            (role/"rounds.jsonl").write_text(json.dumps({"round_index":1,"next_step":"cli"})+'\n'+
                json.dumps({"round_index":2,"next_step":"done"})+'\n')
            (role.parent/"report.json").write_text(json.dumps({"status":"incomplete","completion_satisfied":False}))
        def wait(self): return 7
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    result = main(["--output", str(output), "--task", "original objective", "--api-key", "TEST_PRIVATE_KEY"])
    assert result == 7
    assert observed[0][:4] == [module.sys.executable, "-u", "-m", "lh_harness"]
    assert "original objective" in observed[0]
    assert "original MEA progress" in (output/"console_logs/run.log").read_text()
    assert "TEST_PRIVATE_KEY" not in ''.join(p.read_text() for p in output.rglob('*') if p.is_file())
    summary = json.loads((output/"shadow_summary.json").read_text())
    assert summary["harness_exit_code"] == 7 and summary["assessed_rounds"] == 1
    rows = [json.loads(l) for l in (output/"manager_contracts.jsonl").read_text().splitlines()]
    assert rows[0]["contract"] == "unchanged contract" and rows[0]["dispatch"] == "dispatch"
    assert rows[1]["next_step"] == "done" and rows[1]["assessed"] is False
    assert rows[1]["decision"] is None


def test_partial_run_summary_does_not_invent_assessments(tmp_path):
    directory = tmp_path/"runtime/main/lh_harness/role_orchestration/rounds/round_001"
    directory.mkdir(parents=True)
    (directory/"task_contract.txt").write_text("contract before interrupt")
    summary = collect(tmp_path, -2, 2)
    assert summary["interrupted_signal"] == 2 and summary["assessed_rounds"] == 0
    assert summary["manager_rounds"] == 1 and summary["task_status"] is None
