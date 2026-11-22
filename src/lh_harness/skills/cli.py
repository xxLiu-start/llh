from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from .schema import Settings, ROLES
from .library import SkillLibrary, atomic_json


def register(add_command):
    parser = add_command("skills", "Inspect, freeze, route and learn role-specific skills offline")
    sub = parser.add_subparsers(dest="skills_command", required=True)
    for name in ("list", "inspect", "stats", "init", "freeze", "rollback", "learn"):
        p = sub.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--library")
        if name in {"inspect", "rollback"}:
            p.add_argument("skill_name")
        if name == "rollback":
            p.add_argument("--version", type=int, required=True)
        if name in {"learn", "freeze"}:
            p.add_argument("--split-manifest", type=Path, required=True)
        if name == "learn":
            p.add_argument("--roles", default="manager,executor,auditor")
            p.add_argument("--stage", choices=("all", "propose", "curate"), default="all",
                           help="Per-task reflection/proposals without library writes, or batch-only curation")
            p.add_argument("--proposal-input", type=Path, action="append", default=[],
                           help="Saved proposals.json from a completed task; repeat for batch curation")
            p.add_argument("--phase", choices=("train", "online"), default="train",
                           help="Explicitly label post-task online learning separately from offline training")
            p.add_argument("--run", action="append", default=[], metavar="TASK_ID=RUN_DIRECTORY")
            p.add_argument("--run-index", type=Path, help="JSON object mapping training task IDs to individual run directories")
            p.add_argument("--output", type=Path, default=Path("experiments/skill_evolution/results/skill_enhanced/learning"))
            p.add_argument("--flush", action="store_true", help="Process a final partial batch without lowering independent support requirements")
            p.add_argument("--auth-scheme", choices=("none","bearer","x-api-key"), default="none")
            p.add_argument("--api-key-env", help="Credential environment-variable name; never pass the key itself")


def command(args):
    if args.skills_command == 'learn':
        import fcntl
        # Hold a shared lock for the entire learner call. Deployments acquire
        # the exclusive lock so a Batch never imports half an updated policy.
        directory = Path(__file__).resolve().parents[3]/'.cache'
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'skill-learning-policy.lock').open('a') as policy_lock:
            fcntl.flock(policy_lock, fcntl.LOCK_SH)
            return _command(args)
    return _command(args)


def _command(args):
    try:
        raw = {}
        if args.config:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            raw = tomllib.loads(Path(args.config).read_text())
        settings = Settings.from_dict(raw.get("skills", {}))
        library = SkillLibrary(args.library or settings.library_dir)
        action = args.skills_command
        if action == "init":
            library.create()
            value = {"library": str(library.root), "status": "created"}
        elif action == "list":
            value = [s.to_dict() for s in library.list(candidates=True)]
        elif action == "inspect":
            value = library.get(args.skill_name).to_dict()
        elif action == "rollback":
            value = library.rollback(args.skill_name, args.version).to_dict()
        elif action == "stats":
            value = json.loads((library.root / "stats.json").read_text())
        elif action == "freeze":
            from .experience import load_split
            split = load_split(args.split_manifest)
            library.create()
            value = library.freeze(test_task_ids=split["test"])
        else:
            from .experience import extract_run, load_split
            from .learning import learn, prepare_proposals, curate_proposals
            from .client import JsonClient
            roles = [] if args.roles in {"none", ""} else args.roles.split(",")
            if any(r not in ROLES for r in roles):
                raise ValueError("--roles must contain manager, executor, auditor, or none")
            settings.enabled, settings.phase = True, getattr(args, "phase", "train")
            settings.training.update(enabled=bool(roles), train_roles=sorted(set(roles)))
            mapping = json.loads(args.run_index.read_text()) if args.run_index else {}
            for item in args.run:
                key, path = item.split("=", 1)
                mapping[key] = path
            stage = getattr(args, "stage", "all")
            proposal_inputs = getattr(args, "proposal_input", [])
            if stage == "curate" and (mapping or not proposal_inputs):
                raise ValueError("curate requires --proposal-input and must not reread task trajectories")
            if stage != "curate" and proposal_inputs:
                raise ValueError("--proposal-input is only valid for --stage curate")
            if roles and not mapping and stage != "curate":
                raise ValueError("provide --run TASK_ID=RUN_DIRECTORY or --run-index JSON")
            splits = load_split(args.split_manifest)
            bundles = [extract_run(path, task_id=key, dataset=splits["dataset"]) for key, path in mapping.items()]
            # No model is constructed or called for disabled training.
            clients = {}
            def model(stage, role, prompt, payload, schema):
                binding = "manager" if role == "shared" else role
                gui = payload.get("bundle", {}).get("modality") == "gui"
                child = (("gui_" if gui else "cli_") + binding) if binding in {"executor","auditor"} else binding
                if child not in clients:
                    from ..config import load_run_defaults
                    flat = load_run_defaults(args.config) if args.config else {}
                    backend = flat.get(child + "_agent") or flat.get(binding + "_agent") or flat.get("agent")
                    model_id = flat.get(child + "_model") or flat.get(binding + "_model") or flat.get("model")
                    import os
                    key = os.environ.get(args.api_key_env) if args.api_key_env else None
                    clients[child] = JsonClient(backend=backend, model=model_id, base_url=flat.get("base_url"),
                        profile=raw.get("local_model", {}), output=args.output / child,
                        auth_scheme=args.auth_scheme, api_key=key)
                return clients[child](stage, role, prompt, payload, schema)
            if stage == "propose":
                value = prepare_proposals(bundles, settings, library, model, splits, output=args.output.resolve())
            elif stage == "curate":
                value = curate_proposals([json.loads(path.read_text()) for path in proposal_inputs],
                    settings, library, model, splits, output=args.output.resolve(), flush=args.flush)
            else:
                value = learn(bundles, settings, library, model, splits, output=args.output.resolve(), flush=args.flush)
            from .statistics import summarize
            costs = summarize([r for client in clients.values() for r in client.records])
            costs["experiment"] = "online_skill_learning" if settings.phase == "online" else "offline_skill_learning"
            atomic_json(args.output / "learning_cost.json", costs)
            if stage == "propose":
                value = {key: item for key, item in value.items() if key != "bundles"}
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return 1 if isinstance(value, dict) and value.get("errors") else 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"Skill operation refused: {exc}")
        return 2
