"""Build opt-in Weave/OSWorld source overlays without editing their eval trees."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {"weave": "eval/WeaveBench-harness/cua-harness", "osworld": "eval/OSWorldv2-harness/cua-harness"}
WRAPPER = '''"""Independent role-skill wrapper; the original implementation is preserved."""
import json,os
from pathlib import Path
from ._original_orchestrator import *
from ._original_orchestrator import run as original_run

async def run(*args, **kwargs):
    path=os.environ.get("LLH_SKILL_SETTINGS")
    if not path:
        raise ValueError("enhanced GUI adapter requires explicit LLH_SKILL_SETTINGS")
    settings=json.loads(Path(path).read_text())
    if not settings.get("enabled",False):
        return await original_run(*args,**kwargs)
    from lh_harness.skills.integration import SkillSession
    config=kwargs["config"]
    session=SkillSession(settings,Path(config.log_dir)/"skill_evolution")
    mapping={"agent":"cli_executor","orchestrator_agent":"manager",
             "gui_task_agent":"gui_executor","cli_task_agent":"cli_executor",
             "gui_verifier_agent":"gui_auditor","cli_verifier_agent":"cli_auditor"}
    for key,role in mapping.items():
        if kwargs.get(key) is not None:
            kwargs[key]=session.wrap(role,kwargs[key])
    result=await original_run(*args,**kwargs)
    session.write()
    return result
'''


def prepare(benchmark, output):
    source = ROOT / "experiments/skill_evolution/worktrees/original" / SOURCES[benchmark] / "src/cua_harness"
    if not source.is_dir():
        raise ValueError("official frozen GUI source is missing from the preserved worktree")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError("choose a new overlay destination; existing data is never replaced")
    package = destination / "src/cua_harness"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (package / "orchestrator.py").rename(package / "_original_orchestrator.py")
    (package / "orchestrator.py").write_text(WRAPPER)
    record = {"benchmark": benchmark, "original_source": str(source), "enhanced_source": str(destination),
              "original_files_sha256": {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')},
              "original_orchestrator_sha256": hashlib.sha256((package/'_original_orchestrator.py').read_bytes()).hexdigest(),
              "verifier_and_scoring_modified": False, "actual_gui_benchmark_run": False}
    (destination / "overlay_manifest.json").write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('benchmark',choices=SOURCES);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.benchmark,a.output),indent=2))
