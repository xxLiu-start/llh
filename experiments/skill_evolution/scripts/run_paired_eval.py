"""Run exactly Original and Skill-enhanced using the same frozen TB evaluator."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / "experiments/skill_evolution/worktrees/original"
# Resolve unqualified frozen modules from the official, clean commit, never
# from the locally modified evaluation copy. Both tracks use this same kernel.
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/skill_evolution/adapters"), str(ORIGINAL / "eval/TB-harness/Harness/src")]
from lh_harness import eval_inspect_tb2_original as native
from lh_harness import verifier_runtime
from lh_harness.skills.schema import Settings
from lh_harness.skills.library import SkillLibrary, atomic_json
from lh_harness.skills.experience import load_split
from lh_harness.local_models import check_endpoint
from lh_harness.model_runtime import Monitor, check_docker
from validate_manifest import parity, protected_sources


def online_provenance(library, test_tasks):
    path = library.root / "online_provenance.json"
    if not path.exists():
        return None
    source = json.loads(path.read_text())
    if not isinstance(source.get("observed_task_ids"), list):
        raise ValueError("online Skill library provenance is invalid")
    if set(source["observed_task_ids"]) & set(test_tasks):
        raise ValueError("test leakage: online library has already observed held-out test tasks, even if no skill was accepted")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("paired", "baseline", "skill_enhanced"), default="paired")
    parser.add_argument("--variant-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, default=native.DEFAULT_CONFIG,
                        help="Original frozen TB YAML; model/backend/budgets are shared by both tracks")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", help="Explicit endpoint override for BOTH tracks")
    parser.add_argument("--serving-manifest", type=Path, default=ROOT / "evaluation/validation/qwen27b-vision-serving.json")
    args = parser.parse_args(argv)
    try:
        import tomllib
        import yaml
        raw = tomllib.loads(args.variant_config.read_text())
        args.base_url = args.base_url or os.environ.get("LLH_LOCAL_BASE_URL") or raw.get("run",{}).get("base_url") or "http://172.17.0.1:8010"
        native.model_url({"LLH_LOCAL_BASE_URL":args.base_url})
        skills = Settings.from_dict(raw["skills"]) if args.track != "baseline" else Settings()
        skills.phase = "test"
        library = SkillLibrary(skills.library_dir)
        frozen = library.verify_frozen() if args.track != "baseline" else None
        split = load_split(args.split_manifest)
        online_origin = online_provenance(library, split["test"]) if frozen else None
        if not set(args.task) <= set(split["test"]):
            raise ValueError("paired final evaluation accepts held-out test IDs only")
        used = {e.task_id for s in library.list(candidates=True) for e in s.evidence} if frozen else set()
        if used & set(split["test"]):
            raise ValueError("test tasks occur in library training provenance")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        source_record = protected_sources(ROOT)
        native.prepare_dataset(native.DEFAULT_SOURCE, ROOT / native.DEFAULT_OUTPUT, expected_tasks=89)
        dataset = ROOT / native.DEFAULT_OUTPUT
        config = native.resolved_config(args.baseline_config, dataset)
        if args.repetitions < 1:
            raise ValueError("--repetitions must be positive")
        config["n_attempts"] = args.repetitions
        run = raw.get("run", {})
        if run.get("agent", "claude_code") != "claude_code" or run.get("model", config["agents"][0]["model_name"]) != config["agents"][0]["model_name"]:
            raise ValueError("the frozen TB backend/model must match the experiment TOML; use native Claude Code settings")
        if run.get('roles'):
            raise ValueError('this frozen TB launcher uses its original global model binding; role overrides remain available in the root CLI and GUI adapters')
        kwargs=config['agents'][0]['kwargs']
        if run.get('max_rounds',kwargs['max_rounds'])!=kwargs['max_rounds']:
            raise ValueError('TOML max_rounds differs from the shared original YAML')
        for key,actual in [('temperature',kwargs.get('temperature')),('top_p',kwargs.get('top_p')),('max_output_tokens',kwargs.get('max_tokens'))]:
            if key in raw.get('local_model',{}) and raw['local_model'][key]!=actual:
                raise ValueError(f'TOML inference setting differs from shared original YAML: {key}')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LLH_LOCAL_BASE_URL=args.base_url)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments/skill_evolution/adapters"), str(ORIGINAL / "eval/TB-harness/Harness/src")])
        env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost," + native.urlsplit(args.base_url).hostname
        key, _ = native.configure_auth(env, config, "none", None)
        env["ANTHROPIC_BASE_URL"] = args.base_url
        native.configure_installation(config, env)
        runtime = verifier_runtime.configure(config, env, dataset, args.task)
        needs_image = bool(set(args.task) & {"code-from-image", "video-processing"})
        profile = native.endpoint_profile(config, require_image=needs_image)
        check = check_endpoint(args.base_url, "claude_code", config["agents"][0]["model_name"], profile, auth_scheme="none")
        if raw.get('local_model',{}).get('context_length',check.get('context_length'))!=check.get('context_length'):
            raise ValueError('declared context length does not match the actual endpoint')
        first_image = tomllib.loads((dataset / args.task[0] / "task.toml").read_text())["environment"]["docker_image"]
        docker_check = check_docker(args.base_url, config["agents"][0]["model_name"], profile, first_image,
                                    verifier_runtime.DEFAULT_ROOT, auth_scheme="none")
        serving = native.serving_evidence(args.serving_manifest, check, config["agents"][0]["kwargs"])
        model_hashes = {}
        # Weights are shared by the single retained server process. Capture
        # actual local file hashes, rather than inventing a weights version.
        model_dir = Path(config["agents"][0]["model_name"])
        for path in sorted(model_dir.glob("*.safetensors")):
            model_hashes[path.name] = native.digest(path)
        if not model_hashes:
            raise ValueError("weight files unavailable for the required local identity manifest")
        pristine = ORIGINAL / "eval/TB-harness/Harness/src"
        sources = {str(p.relative_to(pristine)): native.digest(p) for p in pristine.rglob("*.py")}
        manifest = {"schema_version": 1, "benchmark": "a-evolve TB2.0 Inspect-to-Harbor (not TB2.1)",
            "tracks": [], "task_ids": sorted(args.task), "split": split, "library": frozen,
            "library_online_training_origin": online_origin,
            "original_commit": source_record["original_commit"], "original_sources": sources,
            "dataset_sha256": native.digest(dataset / native.MANIFEST), "endpoint": check,
            "serving": serving, "docker_endpoint_check": docker_check, "weights_files_sha256": model_hashes,
            "weights_identity_sha256": hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest(),
            "schedule": "sequential; same retained endpoint", "verifier_runtime": runtime,
            "fair_comparison_verified": False, "full_suite": False, "repetitions": args.repetitions}
        from dataclasses import asdict
        baseline, variant = copy.deepcopy(config), copy.deepcopy(config)
        baseline["agents"][0]["import_path"] = "harbor_skill_agent:PristineOriginalAgent"
        variant["agents"][0]["import_path"] = "harbor_skill_agent:SkillEnhancedAgent"
        variant["agents"][0]["kwargs"]["skills_config"] = asdict(skills)
        parity(baseline, variant)
        lock_id = hashlib.sha256(args.base_url.encode()).hexdigest()[:20]
        with (ROOT / ".cache" / f"skill-pair-{lock_id}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for track, current in (("baseline", baseline), ("skill_enhanced", variant)):
                if args.track != "paired" and track != args.track:
                    continue
                protected_sources(ROOT)
                native.prepare_dataset(native.DEFAULT_SOURCE,dataset,expected_tasks=89)
                from lh_harness.verifier_system import check as check_system
                check_system(dataset,args.task,verifier_runtime.DEFAULT_ROOT)
                if frozen:
                    library.verify_frozen()
                # Current model/context/version must still agree before each
                # track; no cloud fallback, backend switch or role budget edits.
                current_check = check_endpoint(args.base_url, "claude_code", config["agents"][0]["model_name"], profile, auth_scheme="none")
                for key in ("model_id", "base_url", "context_length", "serving_version"):
                    if current_check.get(key) != check.get(key):
                        raise ValueError("model service changed between paired tracks")
                path = output / track
                path.mkdir()
                cfg = path / "config.yaml"
                native.write_runtime_config(current, cfg)
                job_name = "paired-" + uuid.uuid4().hex[:8]
                command = [str(Path(sys.executable).with_name("harbor")), "run", "-c", str(cfg), "--path", str(dataset),
                           "--jobs-dir", str(path), "--job-name", job_name, "--allow-agent-host", native.urlsplit(args.base_url).hostname]
                for task in args.task:
                    command += ["--include-task-name", task]
                monitor = Monitor(args.base_url, config["agents"][0]["model_name"], auth_scheme="none")
                status = verifier_runtime.run_harbor(command, env, ROOT, path / job_name, model_monitor=monitor)
                record = {"track": track, "job": str(path / job_name), "returncode": status, "config_sha256": native.digest(cfg)}
                manifest["tracks"].append(record)
                atomic_json(output / "manifest.json", manifest)
                if status != 0:
                    raise ValueError(f"{track} failed infrastructure/startup checks; inspect preserved raw results")
                results = [json.loads(p.read_text()) for p in (path / job_name).glob("*/result.json")]
                if len(results) != len(args.task) * args.repetitions or any(not ((r.get("verifier_result") or {}).get("rewards")) for r in results):
                    raise ValueError(f"{track} did not produce authoritative verifier rewards for every task")
            if frozen:
                library.verify_frozen()
            protected_sources(ROOT)
            manifest["fair_comparison_verified"] = args.track == "paired"
            atomic_json(output / "manifest.json", manifest)
        print(json.dumps({"output": str(output), "tracks": manifest["tracks"]}, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Paired evaluation refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
