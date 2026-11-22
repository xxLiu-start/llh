"""Launch the existing a-evolve vLLM recipe with Docker-reachable listening.

No source/model files are modified. The server has no API-key authentication;
the local client placeholder is not a server secret.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import shlex
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_MAX_IMAGES = 32


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-evolve-root", type=Path, default=PROJECT / "a-evolve")
    parser.add_argument("--host", default="0.0.0.0", help="Listen on host interfaces so Docker can reach the same model server.")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--enable-vision", action="store_true", help="Keep the same Qwen weights/recipe but enable the vision encoder for image-reading tasks.")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES,
                        help=f"Image limit for the entire request, including images retained in tool history (default: {DEFAULT_MAX_IMAGES}).")
    parser.add_argument("--dry-run", action="store_true", help="Print the original vLLM command without loading weights or starting a server.")
    args, forwarded = parser.parse_known_args(argv)
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not args.dry_run and (not gpu_ids or gpu_ids == "-1"):
        parser.error("set CUDA_VISIBLE_DEVICES to the GPUs allocated to this experiment before starting the server")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.max_images < 1:
        parser.error("--max-images must be positive")
    if not (args.a_evolve_root / "agent_evolve/llm/local.py").is_file():
        parser.error(f"a-evolve local launcher not found under {args.a_evolve_root}")
    if gpu_ids and not any(value == "--tensor-parallel-size" or value.startswith("--tensor-parallel-size=") for value in forwarded):
        forwarded.extend(["--tensor-parallel-size", str(len(gpu_ids.split(",")))])
    # Keep the original context/YaRN/tool parser/thinking recipe. This wrapper
    # changes only network binding and explicitly disables inherited server auth.
    old_path = list(sys.path)
    old_argv = sys.argv
    old_port = os.environ.get("AEVOLVE_LOCAL_PORT")
    old_key = os.environ.get("VLLM_API_KEY")
    local = None
    old_host = None
    try:
        sys.path.insert(0, str(args.a_evolve_root.resolve()))
        local = importlib.import_module("agent_evolve.llm.local")
        old_host = local.LOCAL_HOST
        local.LOCAL_HOST = args.host
        os.environ["AEVOLVE_LOCAL_PORT"] = str(args.port)
        os.environ.pop("VLLM_API_KEY", None)
        sys.argv = ["agent_evolve.llm.local", *forwarded, *(["--dry-run"] if args.dry_run or args.enable_vision else [])]
        if args.enable_vision:
            # Ask the existing recipe to build/validate the exact command. Only
            # the explicit language-only switch and modality limits change.
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                local.main()
            command = shlex.split(capture.getvalue().strip())
            prefix = []
            if command and command[0] == "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1":
                prefix.append(command.pop(0))
            command = vision_command(command, args.max_images)
            if args.dry_run:
                print(shlex.join([*prefix, *command]))
            else:
                os.execv(command[0], command)
        else:
            local.main()
    finally:
        # The dry-run path returns. A real server replaces this process via
        # the original launcher's execv, preserving the selected environment.
        sys.path[:] = old_path
        sys.argv = old_argv
        if local is not None:
            local.LOCAL_HOST = old_host
        for key, value in (("AEVOLVE_LOCAL_PORT", old_port), ("VLLM_API_KEY", old_key)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return 0


def vision_command(command: list[str], max_images: int) -> list[str]:
    if command.count("--language-model-only") != 1 or max_images < 1:
        raise ValueError("unexpected local recipe or invalid image limit; refusing an implicit serving change")
    return [part for part in command if part != "--language-model-only"] + [
        "--limit-mm-per-prompt", json.dumps({"image": max_images, "video": 0}, separators=(",", ":"))]


if __name__ == "__main__":
    raise SystemExit(main())
