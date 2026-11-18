"""WP07 local run artifacts; no checkpoint/restart protocol or implicit live calls."""
import argparse
import asyncio
import json
import sqlite3
from pathlib import Path
import time

from core.experiments.strict_types import strict_json_loads, strict_yaml_loads
from civitas.analytics.society import society_metrics
from civitas.application.market_society import build_market_society
from civitas.application.run import implementation_identity, mechanism_versions, write_json
from civitas.cognition.budget import BudgetController, BudgetLimits
from civitas.cognition.invocations import ModelInvocationStore
from civitas.cognition.metrics import invocation_metrics
from civitas.cognition.providers import ArchivedProvider, LiveProvider, ProviderConfig, SyntheticProvider
from civitas.cognition.runtime import CognitiveRuntime
from civitas.contracts.base import content_hash
from civitas.scenarios.society_config import MarketSocietyConfig


def load_config(path):
    path = Path(path)
    parse = strict_json_loads if path.suffix=='.json' else strict_yaml_loads
    return MarketSocietyConfig.model_validate(parse(path.read_text()))


async def run_society(config, destination, *, runtime=None, policy=None, phase_hook=None):
    config = MarketSocietyConfig.model_validate(config)
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    write_json(destination/'config.json',config.model_dump())
    implementation = implementation_identity(financial=True)
    write_json(destination/'implementation.json',implementation)
    manifest = {'schema_version':1,'scenario':'market_society','backend':config.backend,'status':'running',
        **{key:implementation[key] for key in ('code_commit','code_dirty','source_hash','dependency_hash')},
        'run_id':runtime.run_id if runtime else destination.name,'parent_run_id':None,'branch_id':None,
        'model_versions':runtime.config.model_dump() if runtime else {},
        'budget':runtime.budget.snapshot() if runtime else None,
        'data_origin':'simulation','initialization_origin':'synthetic','config_hash':content_hash(config),
        'planned_ticks':config.ticks,'actors':config.n_agents,'cognition_mode':config.cognition_mode,
        'interpretation':'engineering integration; no market-realism or causal-effect claim'}
    write_json(destination/'manifest.json',manifest)
    world = None
    try:
        world = build_market_society(config,runtime=runtime,policy=policy,phase_hook=phase_hook)
        policy_config = getattr(getattr(world.policy,'slow',None),'config',getattr(world.policy,'config',None))
        manifest.update(scenario_id=world.scenario.scenario_id,scenario_version=world.scenario.version,
            policy_id=world.policy.policy_id,policy_version=world.policy.version,
            mechanism_versions=mechanism_versions(world),
            cognition_policy=policy_config.model_dump() if runtime and policy_config is not None else None)
        manifest['initial_state_hash'] = content_hash(world.snapshot)
        write_json(destination/'initial_state.json',world.snapshot.model_dump())
        write_json(destination/'manifest.json',manifest)
        with (destination/'tick-hashes.jsonl').open('x') as stream:
            for _ in range(config.ticks):
                prepared = await world.step()
                stream.write(json.dumps({'tick':prepared.tick,'world_version':prepared.snapshot.world_version,'state_hash':prepared.state_hash})+'\n')
                stream.flush()
                if prepared.tick%20==0:
                    print(json.dumps({'tick':prepared.tick,'planned':config.ticks,'elapsed_seconds':time.perf_counter()-started}),flush=True)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            if runtime is not None:
                records = [r for r in runtime.store.records() if r['run_id']==runtime.run_id]
                write_json(destination/'invocations.json',records)
                write_json(destination/'budget-after.json',runtime.budget.snapshot())
                manifest['model_calls'] = invocation_metrics(records)
                manifest['budget'] = runtime.budget.snapshot()
            if world is not None:
                snapshot = world.snapshot
                manifest.update(committed_ticks=snapshot.tick,final_state_hash=content_hash(snapshot))
                write_json(destination/'state.json',snapshot.model_dump())
                write_json(destination/'failures.json',list(world.failures))
                write_json(destination/'settlement-diagnostics.json',(world.scenario.adapter if world.scenario.legacy else world.scenario.market).diagnostics)
                for name in ('events','decisions','receipts','messages','memories'):
                    with (destination/(name+'.jsonl')).open('x') as stream:
                        for value in getattr(snapshot,name):
                            stream.write(value.model_dump_json()+'\n')
                write_json(destination/'trades.json',world.scenario.trades())
                write_json(destination/'relations.json',snapshot.data['typed_relations'])
                write_json(destination/'decision-traces.json',snapshot.data['society_traces']['records'])
                write_json(destination/'metrics.json',society_metrics(world))
                if not world.scenario.legacy:
                    with (destination/'ledger-journal.jsonl').open('x') as stream:
                        for entry in world.scenario.market.ledger.journal:
                            stream.write(entry.model_dump_json()+'\n')
                manifest['trade_count'] = len(world.scenario.trades())
                manifest['rejected_commands'] = sum(r.status=='rejected' for r in snapshot.receipts)
        except BaseException as exc:
            manifest.update(status='failed',export_error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            manifest['elapsed_seconds'] = time.perf_counter()-started
            write_json(destination/'manifest.json',manifest)
            write_json(destination/'summary.json',manifest)
    return manifest


async def command(args):
    config = load_config(args.config)
    if args.mode is not None:
        config = config.model_copy(update={'cognition_mode':args.mode})
    runtime, provider = None, None
    destination = Path(args.output)
    if config.cognition_mode!='rule':
        provider_config = ProviderConfig.model_validate_json(Path(args.provider_config).read_text()) if args.provider_config else ProviderConfig()
        if config.cognition_mode=='live':
            if not args.provider_config or not args.project_ledger or provider_config.source!='live':
                raise ValueError('Live requires verified provider config and the shared cumulative project ledger')
            ledger_path = Path(args.project_ledger).resolve(strict=True)
            with sqlite3.connect(ledger_path.as_uri()+'?mode=ro',uri=True) as db:
                existing = db.execute("SELECT body FROM contracts WHERE name='project-budget-v1'").fetchone()
            if existing is None or BudgetLimits.model_validate_json(existing[0])!=BudgetLimits():
                raise ValueError('Live must reuse an existing configured cumulative budget; creation/reset is unavailable')
            store = ModelInvocationStore(args.project_ledger)
            limits = BudgetController.project_limits(store)
            provider = LiveProvider(provider_config)
        else:
            if destination.exists():
                raise FileExistsError(destination)
            store = ModelInvocationStore(destination.parent/(destination.name+'-invocations.sqlite'))
            limits = BudgetLimits(usd_nano=0,cny_nano=0,max_attempts=10000,max_tokens=1_000_000_000)
            provider_config = provider_config.model_copy(update={'source':config.cognition_mode})
            provider = ArchivedProvider(strict_json_loads(Path(args.tape).read_text())) if config.cognition_mode=='archived_response' and args.tape else SyntheticProvider()
        budget = BudgetController(store,limits)
        before = budget.snapshot()
        attempts = min(args.attempt_cap,limits.max_attempts-before['attempts'])
        tokens = min(attempts*(provider_config.max_input_tokens+provider_config.max_output_tokens),limits.max_tokens-before['tokens'])
        per_attempt = provider_config.max_input_tokens*provider_config.price.input_nano+provider_config.max_output_tokens*provider_config.price.output_nano
        cost = attempts*per_attempt if config.cognition_mode=='live' else 0
        remaining = getattr(limits,provider_config.price.currency.lower()+'_nano')-before[provider_config.price.currency]['accounted_nano']-before[provider_config.price.currency]['reserved_nano']
        if attempts<1 or tokens<provider_config.max_input_tokens+provider_config.max_output_tokens or cost>remaining:
            raise ValueError('Run allocation exceeds remaining cumulative budget')
        run_limits = BudgetLimits(usd_nano=cost if provider_config.price.currency=='USD' else 0,
            cny_nano=cost if provider_config.price.currency=='CNY' else 0,max_attempts=attempts,max_tokens=tokens)
        budget.register_run(args.run_id,run_limits)
        plan = {'provider':provider_config.model_dump(),'before':before,'run_limits':run_limits.model_dump(),
            'maximum_reserved_cost_nano':cost,'run_id':args.run_id,'data_origin':'synthetic_initial_world'}
        plan_path = destination.parent/(destination.name+'-budget-plan.json')
        if plan_path.exists():
            raise FileExistsError(plan_path)
        write_json(plan_path,plan)
        runtime = CognitiveRuntime(store=store,budget=budget,provider=provider,config=provider_config,run_id=args.run_id,
            concurrency=1 if config.cognition_mode=='live' else 8)
    try:
        result = await run_society(config,destination,runtime=runtime)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        if isinstance(provider,LiveProvider):
            await provider.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['rule','synthetic_test','archived_response','live'])
    parser.add_argument('--provider-config')
    parser.add_argument('--project-ledger')
    parser.add_argument('--tape')
    parser.add_argument('--run-id',default='wp07-scene')
    parser.add_argument('--attempt-cap',type=int,default=180)
    args = parser.parse_args(argv)
    if args.attempt_cap<1:
        parser.error('attempt-cap must be positive')
    asyncio.run(command(args))


if __name__=='__main__':
    main()
