from __future__ import annotations
import hashlib
import json
from pathlib import Path


def protected_sources(root):
    root = Path(root)
    record = json.loads((root / "experiments/skill_evolution/manifests/baseline-protection.json").read_text())
    protected = {**record["protected_files"], **record.get("application_kernels", {})}
    changed = [name for name, expected in protected.items()
               if not (root / name).exists() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
    if changed:
        raise ValueError(f"frozen evaluation files changed: {changed}")
    return {"protected_files": len(protected), "original_commit": record["original_commit"]}


def parity(left, right):
    # Artifact destinations and the explicit skill adapter are the only
    # non-semantic differences; all other task/runtime options must match.
    import copy
    def normalized(value):
        value = copy.deepcopy(value)
        for key in ("job_name", "jobs_dir"):
            value.pop(key, None)
        agent = value["agents"][0]
        agent["import_path"] = "pristine-original-plus-optional-skills"
        agent.get("kwargs", {}).pop("skills_config", None)
        return value
    if normalized(left) != normalized(right):
        raise ValueError("paired configuration differs outside skills/output paths")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    print(json.dumps(protected_sources(args.root), indent=2))
