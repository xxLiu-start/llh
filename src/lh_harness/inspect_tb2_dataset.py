"""Copy Inspect TB2 definitions into a separate Harbor dataset, without edits.

Only file layout and task configuration are translated. Instructions, tests,
solutions, image references, CPU/memory limits and timeouts are preserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

DEFAULT_SOURCE = Path("/data0/liuxule/project/.cache/a-evolve-tb2/challenges")
DEFAULT_OUTPUT = Path("eval/TB-harness/datasets/a-evolve-tb20-harbor")
MANIFEST = "dataset_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tree_hashes(root: Path, *, exclude_manifest: bool = False) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not supported in dataset definitions: {path}")
        if path.is_file() and not (exclude_manifest and path == root / MANIFEST):
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def memory_mb(value) -> int:
    match = re.fullmatch(r"(\d+)([gm])", str(value).lower())
    if not match:
        raise ValueError(f"unsupported Compose mem_limit: {value!r}")
    return int(match[1]) * (1024 if match[2] == "g" else 1)


def copy_mapping(task: Path, destination: Path, mappings: dict, prefix: str) -> list[dict]:
    records = []
    if not isinstance(mappings, dict):
        raise ValueError(f"{task.name}: files must be a mapping")
    for container_path, relative_source in sorted(mappings.items()):
        target = PurePosixPath(container_path)
        try:
            relative_target = target.relative_to(prefix)
        except ValueError:
            raise ValueError(f"{task.name}: unsupported file target {container_path!r}; expected {prefix}/...") from None
        if not relative_target.parts or ".." in target.parts:
            raise ValueError(f"invalid mapped destination: {container_path!r}")
        source = (task / relative_source).resolve()
        if not source.is_relative_to(task.resolve()) or not source.is_file():
            raise ValueError(f"{task.name}: missing or external mapped source {relative_source!r}")
        target_path = destination / relative_target.as_posix()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            raise ValueError(f"duplicate mapped destination: {container_path}")
        shutil.copy2(source, target_path)
        records.append({"source": source.relative_to(task).as_posix(),
                        "target": target_path.relative_to(destination.parent).as_posix(),
                        "container_path": container_path, "sha256": digest(source)})
    return records


def convert_task(task: Path, output: Path) -> dict:
    import toml
    import yaml

    definition = yaml.safe_load((task / "eval.yaml").read_text())
    compose = yaml.safe_load((task / "compose.yaml").read_text())
    if set(compose) != {"services"} or set(compose["services"]) != {"default"}:
        raise ValueError(f"{task.name}: unsupported multi-service Compose definition")
    service = copy.deepcopy(compose["services"]["default"])
    supported = {"image", "platform", "command", "init", "mem_limit", "cpus"}
    if set(service) != supported or service["platform"] != "linux/amd64":
        raise ValueError(f"{task.name}: unsupported Compose fields/platform; refusing a lossy conversion")
    if type(service["cpus"]) is not int or service["cpus"] < 1:
        raise ValueError(f"{task.name}: integer positive CPU limit required")
    image = service["image"]
    if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9_.:/@-]+", image):
        raise ValueError(f"{task.name}: invalid image reference")
    variants = definition["variants"]
    if set(variants) != {"default", "solution"} or set(variants["default"]) != {"prompt"}:
        raise ValueError(f"{task.name}: unsupported variants; default prompt must not be changed")
    prompt = variants["default"]["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{task.name}: empty task instruction")
    metadata = copy.deepcopy(definition.get("metadata", {}))
    agent_timeout = metadata.get("agent_timeout_sec", 900)
    verifier_timeout = metadata.get("verifier_timeout_sec", 900)
    if any(type(value) not in (int, float) or value <= 0 for value in (agent_timeout, verifier_timeout)):
        raise ValueError(f"{task.name}: invalid original timeouts")
    output.mkdir()
    (output / "instruction.md").write_text(prompt, encoding="utf-8")
    (output / "environment").mkdir()
    # A FROM-only fallback supports Harbor versions that require a Dockerfile.
    # Normal execution selects the identical prebuilt image in task.toml.
    (output / "environment" / "Dockerfile").write_text(f"FROM {image}\n")
    (output / "environment" / "docker-compose.yaml").write_text(
        yaml.safe_dump({"services": {"main": service}}, sort_keys=False))
    tests = copy_mapping(task, output / "tests", definition["files"], "/tests")
    solutions = copy_mapping(task, output / "solution", variants["solution"]["files"], "/solution")
    if not (output / "tests" / "test.sh").is_file() or not (output / "solution" / "solve.sh").is_file():
        raise ValueError(f"{task.name}: missing original test.sh or solve.sh mapping")
    config = {"version": "1.0", "metadata": metadata,
              "agent": {"timeout_sec": agent_timeout}, "verifier": {"timeout_sec": verifier_timeout},
              "environment": {"docker_image": image, "cpus": service["cpus"],
                              "memory_mb": memory_mb(service["mem_limit"]), "gpus": 0}}
    (output / "task.toml").write_text(toml.dumps(config), encoding="utf-8")
    from harbor.models.task.task import Task
    if not Task.is_valid_dir(output):
        raise ValueError(f"{task.name}: converted task failed Harbor validation")
    return {"name": task.name, "image": image, "prompt_sha256": digest(output / "instruction.md"),
            "agent_timeout_sec": agent_timeout, "verifier_timeout_sec": verifier_timeout,
            "cpus": service["cpus"], "memory_mb": config["environment"]["memory_mb"],
            "tests": tests, "solutions": solutions}


def verify_dataset(source: Path, output: Path) -> dict:
    manifest = json.loads((output / MANIFEST).read_text())
    if manifest.get("source_root") != str(source.resolve()):
        raise ValueError("existing dataset was generated from a different source; choose a new output directory")
    if manifest.get("source_files") != tree_hashes(source):
        raise ValueError("source definitions changed; refusing to silently reuse/overwrite the adapted dataset")
    if manifest.get("output_files") != tree_hashes(output, exclude_manifest=True):
        raise ValueError("adapted dataset was modified; refusing to use changed instructions/tests/configuration")
    from harbor.models.task.task import Task
    for record in manifest["tasks"]:
        if not Task.is_valid_dir(output / record["name"]):
            raise ValueError(f"invalid Harbor task: {record['name']}")
    return manifest


def prepare_dataset(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT, *, expected_tasks: int | None = None) -> dict:
    source, output = source.resolve(), output.absolute()
    if not source.is_dir():
        raise ValueError(f"source dataset does not exist: {source}")
    if output.is_symlink() or output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("source and output must be separate trees; symlink outputs are not supported")
    if output.exists():
        manifest = verify_dataset(source, output)
        if expected_tasks is not None and len(manifest["tasks"]) != expected_tasks:
            raise ValueError("existing dataset has an unexpected task count")
        return manifest
    tasks = sorted(p for p in source.iterdir() if p.is_dir() and (p / "eval.yaml").is_file())
    if not tasks or expected_tasks is not None and len(tasks) != expected_tasks:
        raise ValueError(f"expected {expected_tasks or 'nonempty'} tasks, found {len(tasks)}")
    before = tree_hashes(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inspect-tb2-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        records = [convert_task(task, staging / task.name) for task in tasks]
        if tree_hashes(source) != before:
            raise ValueError("source changed while converting; no adapted dataset was published")
        manifest = {"schema_version": 1, "dataset_label": "a-evolve Terminal-Bench 2.0 (Inspect layout adapted to Harbor)",
                    "benchmark_version": "2.0", "source_root": str(source), "output_root": str(output),
                    "task_count": len(records), "instructions_and_verifiers_modified": False,
                    "tests_uploaded_only_at_verification": True,
                    "declared_source": {"repository": "https://github.com/UKGovernmentBEIS/inspect_evals",
                                        "revision": "6e30b2de72e98dd5cc342eb9ba545ae27d2f63d7",
                                        "revision_source": "a-evolve download_challenges.sh; not independently attested"},
                    "tasks": records, "source_files": before,
                    "output_files": tree_hashes(staging, exclude_manifest=True)}
        (staging / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.rename(staging, output)
    return verify_dataset(source, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-tasks", type=int, default=89)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_dataset(args.source, args.output, expected_tasks=args.expected_tasks)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Dataset preparation failed: {exc}\n")
    print(json.dumps({"dataset": str(args.output.absolute()), "version": "2.0", "tasks": manifest["task_count"],
                      "source_and_test_hashes_verified": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
