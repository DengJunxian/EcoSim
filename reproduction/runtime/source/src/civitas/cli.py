import argparse
import asyncio
import json
from pathlib import Path
import sys

from core.experiments.strict_types import strict_json_loads, strict_yaml_loads
from pydantic import TypeAdapter

from civitas.application import build_world
from civitas.application.run import run_to_directory
from civitas.contracts.base import content_hash
from civitas.contracts.config import RunConfig
from civitas.contracts.legacy_config import LegacyRunConfig
from civitas.contracts.multiasset_config import MultiAssetRunConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m civitas")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a strictly typed offline scenario configuration")
    validate.add_argument("config", type=Path)
    run = commands.add_parser("run", help="Run an offline scenario and export a new directory")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument('--durable',action='store_true')
    run.add_argument('--until',type=int)
    run.add_argument('--budget-limits',type=Path)
    checkpoint = commands.add_parser('checkpoint',help='Save a complete private checkpoint of a durable run')
    checkpoint.add_argument('root',type=Path)
    checkpoint.add_argument('--tick',type=int)
    checkpoint.add_argument('--parquet',type=Path)
    resume = commands.add_parser('resume',help='Resume an offline durable run without changing its frozen horizon')
    resume.add_argument('root',type=Path)
    resume.add_argument('--until',type=int)
    replay = commands.add_parser('replay',help='Offline read-only state or execution verification replay')
    replay.add_argument('root',type=Path)
    replay.add_argument('--mode',choices=('state','verification'),required=True)
    replay.add_argument('--output',type=Path,required=True)
    branch = commands.add_parser('branch',help='Create a planned branch with immutable parent history')
    branch.add_argument('root',type=Path)
    branch.add_argument('--tick',type=int,required=True)
    branch.add_argument('--output',type=Path,required=True)
    branch.add_argument('--intervention',type=Path)
    branch.add_argument('--allow-cache-reuse',action='store_true')
    batch = commands.add_parser('batch',help='Execute or resume a bounded offline config by seed batch')
    batch.add_argument('root',type=Path)
    batch.add_argument('--spec',type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command not in ('run','validate'):
            from civitas.storage.commit_store import CommitStore
            from civitas.application.persistent_run import resume_run,replay_run
            if args.command=='checkpoint':
                from civitas.storage.checkpoint import save
                from civitas.application.persistent_run import validated_snapshot
                store = CommitStore(args.root)
                validated_snapshot(store,tick=args.tick)
                result = save(store,tick=args.tick)
                if args.parquet:
                    from civitas.storage.parquet_sink import export_parquet
                    result['parquet'] = export_parquet(CommitStore(args.root),args.parquet)
            elif args.command=='resume':
                result = asyncio.run(resume_run(args.root,until=args.until))
            elif args.command=='replay':
                result = asyncio.run(replay_run(args.root,mode=args.mode,destination=args.output))
            elif args.command=='branch':
                from civitas.experiments.branch import branch_run,parse_intervention
                from civitas.experiments.registry import history
                intervention = parse_intervention(strict_json_loads(args.intervention.read_text())) if args.intervention else None
                child = branch_run(args.root,tick=args.tick,destination=args.output,intervention=intervention,
                                   allow_cache_reuse=args.allow_cache_reuse)
                result = {**history(child.root)[-1],'path':str(child.root),'run_id':child.identity.run_id}
            else:
                from civitas.experiments.batch import parse_batch_spec,run_batch
                spec = parse_batch_spec(strict_json_loads(args.spec.read_text())) if args.spec else None
                result = asyncio.run(run_batch(args.root,spec=spec))
            print(json.dumps(result,ensure_ascii=False,sort_keys=True))
            return 1 if result.get('status') in ('different','blocked','incomplete') else 0
        if args.config.suffix.lower() not in (".json", ".yaml", ".yml"):
            raise ValueError("Scenario inputs must use .json, .yaml or .yml")
        parse = strict_json_loads if args.config.suffix.lower() == ".json" else strict_yaml_loads
        raw = parse(args.config.read_text(encoding='utf-8'))
        if isinstance(raw,dict) and raw.get('scenario') in ('market_society','kdd27_v1'):
            from civitas.application.persistent_run import parse_config
            from civitas.application.society_run import main as society_main
            config = parse_config(raw)
            if args.command=='validate':
                print(json.dumps({'status':'valid','config_hash':content_hash(config),'scenario':config.scenario},sort_keys=True))
            else:
                if args.durable:
                    from civitas.application.persistent_run import create_run,advance
                    from civitas.cognition.budget import BudgetLimits
                    from civitas.experiments.registry import run_lock
                    limits = BudgetLimits.model_validate_json(args.budget_limits.read_text()) if args.budget_limits else None
                    world,store,runtime = create_run(config,args.output,limits=limits)
                    with run_lock(store.root):
                        result = asyncio.run(advance(world,store,runtime,until=args.until))
                    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
                    return 0
                if config.scenario=='kdd27_v1':
                    raise ValueError('kdd27_v1 requires --durable to preserve research evidence')
                if args.until is not None or args.budget_limits:
                    raise ValueError('--until/--budget-limits require --durable')
                if config.cognition_mode not in ('rule','synthetic_test'):
                    raise ValueError('Use the scenario runner with explicit provider and budget arguments')
                society_main([str(args.config),'--output',str(args.output)])
            return 0
        config = TypeAdapter(RunConfig | LegacyRunConfig | MultiAssetRunConfig).validate_python(raw)
        if args.command=='run' and (args.durable or args.until is not None or args.budget_limits):
            raise ValueError('The durable CLI currently requires a market_society configuration')
        world = build_world(config)
        if args.command == "validate":
            result = {"status": "valid", "config_hash": content_hash(config), "scenario": config.scenario}
        else:
            result = asyncio.run(run_to_directory(world, config, args.output)).model_dump()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
