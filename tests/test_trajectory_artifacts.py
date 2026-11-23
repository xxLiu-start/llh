from __future__ import annotations

import base64
import json
from pathlib import Path

from lh_harness.trajectory_artifacts import (
    StreamingTrajectoryArtifactWriter,
    persist_trajectory_artifacts,
)


def _codex_image_trajectory(payloads: list[bytes]) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {
                "id": "shot-1",
                "type": "mcp_tool_call",
                "result": {
                    "content": [
                        {
                            "type": "image",
                            "data": base64.b64encode(payload).decode("ascii"),
                            "mimeType": "image/png",
                        }
                        for payload in payloads
                    ],
                },
            },
        }
    )


def test_compactor_preserves_bounded_textual_tool_results(tmp_path: Path) -> None:
    from lh_harness.skills.trajectory_compactor import compact

    stream = tmp_path / "claude_stream.jsonl"
    events = [
        {"type": "assistant", "content": [{"type": "tool_use", "id": "tool-1",
            "name": "Bash", "input": {"command": "cat report.txt"}}]},
        {"type": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1",
            "content": [{"type": "text", "text": "report validated successfully"}]}]},
        {"type": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-2",
            "content": [{"type": "image", "data": "c2VjcmV0LWltYWdl", "mimeType": "image/png"}]}]},
    ]
    stream.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    summary = compact(stream)

    assert len(summary["tool_results"]) == 2
    assert summary["tool_results"][0]["tool_use_id"] == "tool-1"
    assert "report validated successfully" in summary["tool_results"][0]["detail"]
    assert "c2VjcmV0LWltYWdl" not in json.dumps(summary["tool_results"])


def test_embedded_screenshots_become_files_and_file_references(tmp_path: Path) -> None:
    round_dir = tmp_path / "run" / "round_001"
    payloads = [b"\x89PNG\r\nfirst", b"\x89PNG\r\nsecond"]

    summary = persist_trajectory_artifacts(
        _codex_image_trajectory(payloads),
        round_dir=round_dir,
        role_name="executor",
    )

    assert summary["screenshot_count"] == 2
    manifest = json.loads((round_dir / "executor_screenshots.json").read_text(encoding="utf-8"))
    names = [item["screenshot_file"] for item in manifest["screenshots"]]
    assert names == ["executor_step_0002_01.png", "executor_step_0002_02.png"]
    for name, payload in zip(names, payloads, strict=True):
        assert (round_dir / name).read_bytes() == payload

    assert not (tmp_path / "workspace").exists()

    normalized = (round_dir / "executor_trajectory.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in normalized.splitlines()]
    assert records[-1]["screenshot_file"] == names[0]
    assert records[-1]["screenshot_files"] == names
    assert "data:image" not in normalized
    assert base64.b64encode(payloads[0]).decode("ascii") not in normalized


def test_invalid_inline_image_is_not_written(tmp_path: Path) -> None:
    raw = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "id": "shot-1",
                "type": "mcp_tool_call",
                "result": {"content": [{"type": "image", "data": "not-base64", "mimeType": "image/png"}]},
            },
        }
    )

    summary = persist_trajectory_artifacts(raw, round_dir=tmp_path, role_name="executor")

    assert summary["screenshot_count"] == 0
    assert list(tmp_path.glob("*.png")) == []


def test_streaming_writer_persists_screenshot_before_role_finishes(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_001"
    live_path = round_dir / "executor_raw_trajectory.jsonl"

    writer = StreamingTrajectoryArtifactWriter.from_live_path(live_path)
    assert writer is not None
    writer.consume_line(_codex_image_trajectory([b"\x89PNG\r\nlive"]))

    screenshot = round_dir / "executor_step_0002_01.png"
    assert screenshot.read_bytes() == b"\x89PNG\r\nlive"
    manifest = json.loads((round_dir / "executor_screenshots.json").read_text(encoding="utf-8"))
    assert manifest["live"] is True
    assert manifest["screenshot_count"] == 1
    normalized = (round_dir / "executor_trajectory.jsonl").read_text(encoding="utf-8")
    assert "executor_step_0002_01.png" in normalized
    assert "data:image" not in normalized


def test_streaming_writer_pairs_codex_tool_start_and_completion_once(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_001"
    live_path = round_dir / "executor_raw_trajectory.jsonl"
    writer = StreamingTrajectoryArtifactWriter.from_live_path(live_path)
    assert writer is not None
    started = {
        "type": "item.started",
        "item": {
            "id": "command-1",
            "type": "command_execution",
            "command": "/bin/zsh -lc pwd",
            "status": "in_progress",
        },
    }
    completed = {
        "type": "item.completed",
        "item": {
            "id": "command-1",
            "type": "command_execution",
            "command": "/bin/zsh -lc pwd",
            "status": "completed",
            "exit_code": 0,
            "aggregated_output": "/tmp/workspace\n",
        },
    }

    writer.consume_line(json.dumps(started))
    writer.consume_line(json.dumps(completed))

    records = [
        json.loads(line)
        for line in (round_dir / "executor_trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["kind"] for record in records] == ["tool_use", "tool_result"]
    assert records[0]["id"] == records[1]["tool_use_id"] == "command-1"
    assert records[1]["status"] == "completed"

def test_compactor_extracts_codex_mcp_gui_calls_without_image_payloads(tmp_path: Path) -> None:
    from lh_harness.skills.trajectory_compactor import compact

    stream = tmp_path / "codex.jsonl"
    events = [
        {
            "type": "item.started",
            "item": {"id": "mcp-1", "type": "mcp_tool_call", "server": "computer",
                     "tool": "click", "arguments": {"x": 10, "y": 20}},
        },
        {
            "type": "item.completed",
            "item": {"id": "mcp-1", "type": "mcp_tool_call", "server": "computer",
                     "tool": "click", "arguments": {"x": 10, "y": 20},
                     "result": {"content": [
                         {"type": "text", "text": "Clicked the target"},
                         {"type": "image", "data": "c2VjcmV0LWltYWdl", "mimeType": "image/png"},
                     ]}},
        },
    ]
    stream.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    summary = compact(stream)

    assert summary["tool_call_count"] == 1
    assert summary["tool_calls"][0]["tool"] == "computer/click"
    assert '"x": 10' in summary["tool_calls"][0]["input"]
    assert len(summary["tool_results"]) == 1
    assert "Clicked the target" in summary["tool_results"][0]["detail"]
    assert "c2VjcmV0LWltYWdl" not in json.dumps(summary)


def test_compactor_extracts_opencode_tool_results(tmp_path: Path) -> None:
    from lh_harness.skills.trajectory_compactor import compact

    stream = tmp_path / "opencode.jsonl"
    events = [{
        "type": "tool_use",
        "part": {"type": "tool", "id": "tool-1", "tool": "computer_click",
                 "state": {"status": "completed", "input": {"x": 10, "y": 20},
                           "output": "Click completed"}},
    }]
    stream.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    summary = compact(stream)

    assert summary["tool_call_count"] == 1
    assert summary["tool_calls"][0]["tool"] == "computer_click"
    assert "Click completed" in summary["tool_results"][0]["detail"]
