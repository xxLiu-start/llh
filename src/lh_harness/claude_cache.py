"""Find a matching local Claude Code binary for offline Harbor setup."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def find_cached_claude_binary(output_root: Path, version: str) -> Path | None:
    if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        return None
    preferred = output_root / "jobs" / "claude_code_cache" / version / "claude"
    siblings = sorted(output_root.parent.glob(f"*/jobs/claude_code_cache/{version}/claude"))
    archived = sorted(output_root.parent.glob(f"*/*/jobs/claude_code_cache/{version}/claude"))
    for candidate in dict.fromkeys([preferred, *siblings, *archived]):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        try:
            completed = subprocess.run(
                [str(candidate), "--version"], capture_output=True, text=True,
                timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        match = re.search(r"\b(\d+\.\d+\.\d+)\b", completed.stdout)
        if completed.returncode == 0 and match and match.group(1) == version:
            return candidate.resolve()
    return None
