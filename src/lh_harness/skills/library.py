from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schema import Skill, ROLES, sha, canonical


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class SkillLibrary:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def create(self):
        if (self.root / "index.json").exists():
            return
        for role in ROLES:
            for modality in (("shared",) if role == "manager" else ("gui", "cli", "shared")):
                (self.root / role / modality).mkdir(parents=True, exist_ok=True)
            (self.root / "candidates" / role).mkdir(parents=True, exist_ok=True)
        atomic_json(self.root / "index.json", {"schema_version": 1, "revision": 0, "skills": {}})
        atomic_json(self.root / "stats.json", {"roles": {r: {"active": 0, "candidate": 0} for r in ROLES}})

    @contextmanager
    def lock(self):
        self.create()
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (self.root / "frozen.json").exists():
                raise ValueError("skill library is frozen; train in a separate library")
            yield

    def index(self):
        return json.loads((self.root / "index.json").read_text())

    def list(self, role=None, *, candidates=False):
        if not (self.root / "index.json").exists():
            return []
        result = []
        for key, record in sorted(self.index()["skills"].items()):
            skill = Skill.from_dict(json.loads((self.root / record["path"]).read_text()))
            if sha(skill.to_dict()) != record["sha256"]:
                raise ValueError(f"skill integrity mismatch: {key}")
            if (role is None or skill.role == role) and (candidates or skill.status == "active"):
                result.append(skill)
        return result

    def get(self, key):
        found = [s for s in self.list(candidates=True) if s.key == key or s.name == key]
        if len(found) != 1:
            raise ValueError("skill lookup must identify exactly one role/modality/name")
        return found[0]

    def get_version(self, key, version):
        """Load the immutable Skill version recorded when it was injected."""
        parts = key.split("/") if isinstance(key, str) else []
        if len(parts) != 3 or parts[0] not in ROLES or parts[1] not in {"shared", "cli", "gui"} or type(version) is not int or version < 1:
            raise ValueError("invalid skill key or version")
        path = self.root / ".history" / parts[0] / parts[1] / parts[2] / f"{version:06d}.json"
        if path.is_file():
            skill = Skill.from_dict(json.loads(path.read_text()))
        else:
            skill = self.get(key)
            if skill.version != version:
                raise KeyError(f"skill version unavailable: {key}@{version}")
        if skill.key != key or skill.version != version:
            raise ValueError("skill history identity mismatch")
        return skill

    def put(self, skill: Skill, *, min_support=2):
        if type(min_support) is not int or min_support < 1:
            raise ValueError("minimum task support must be a positive integer")
        with self.lock():
            index = self.index()
            prior = index["skills"].get(skill.key)
            skill = Skill.from_dict(skill.to_dict())
            if skill.status == "active" and skill.support < min_support:
                skill.status = "candidate"
            skill.version = (prior["version"] + 1) if prior else 1
            skill.updated_at = datetime.now(timezone.utc).isoformat()
            path = Path(skill.role) / skill.modality / f"{skill.name}.json" if skill.status == "active" else Path("candidates") / skill.role / f"{skill.modality}--{skill.name}.json"
            # History is append-only; active/candidate files are replaceable
            # indexes of immutable versions, never the only saved copy.
            history = self.root / ".history" / skill.key / f"{skill.version:06d}.json"
            if history.exists():
                raise ValueError("skill version already exists")
            atomic_json(history, skill.to_dict())
            atomic_json(self.root / path, skill.to_dict())
            index["revision"] += 1
            index["skills"][skill.key] = {"path": str(path), "version": skill.version, "sha256": sha(skill.to_dict())}
            atomic_json(self.root / "index.json", index)
            if prior and prior["path"] != str(path):
                (self.root / prior["path"]).unlink()
            all_skills = self.list(candidates=True)
            atomic_json(self.root / "stats.json", {"roles": {r: {state: sum(s.role == r and s.status == state for s in all_skills)
                        for state in ("active", "candidate")} for r in ROLES}})
            return skill

    def rollback(self, key, version):
        current = self.get(key)
        source = self.root / ".history" / current.key / f"{int(version):06d}.json"
        previous = Skill.from_dict(json.loads(source.read_text()))
        return self.put(previous)

    def digest(self):
        files = {}
        if self.root.exists():
            import hashlib
            for path in sorted(self.root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("symlinks are forbidden in skill libraries")
                if path.is_file() and path.name not in {".lock", "frozen.json"} and not path.name.endswith(".tmp"):
                    files[str(path.relative_to(self.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return sha(files)

    def freeze(self, *, test_task_ids=()):
        with self.lock():
            training_tasks = {e.task_id for s in self.list(candidates=True) for e in s.evidence}
            if training_tasks & set(test_task_ids):
                raise ValueError("test leakage: library evidence includes held-out test tasks")
            record = {"version": self.index()["revision"], "sha256": self.digest(),
                      "frozen_at": datetime.now(timezone.utc).isoformat(), "test_task_ids": sorted(test_task_ids)}
            atomic_json(self.root / "frozen.json", record)
            return record

    def verify_frozen(self):
        record = json.loads((self.root / "frozen.json").read_text())
        if record["sha256"] != self.digest() or record["version"] != self.index()["revision"]:
            raise ValueError("frozen skill library was changed")
        return record
