"""Opt-in WP08 offline runner. The WP07 export runner remains unchanged."""
from pathlib import Path
import asyncio
import sys
import time
import tempfile
import uuid

from civitas.application.market_society import build_market_society
from civitas.application.run import implementation_identity
from civitas.cognition.budget import BudgetController, BudgetLimits
from civitas.cognition.invocations import ModelInvocationStore
from civitas.cognition.providers import ArchivedProvider, ProviderConfig, SyntheticProvider
from civitas.cognition.runtime import CognitiveRuntime
from civitas.contracts.base import content_hash
from civitas.experiments import registry
from civitas.scenarios.society_config import MarketSocietyConfig
from civitas.storage.checkpoint import attach, restore, save
from civitas.storage.commit_store import CommitStore, atomic_json


def parse_config(config):
    from civitas.scenarios.kdd27_config import parse_paper_config
    name=config.get('scenario') if isinstance(config,dict) else config.scenario
    return parse_paper_config(config) if name=='kdd27_v1' else MarketSocietyConfig.model_validate(config)


def build_persistent_world(config, *, runtime=None):
    if config.scenario=='kdd27_v1':
        from civitas.application.kdd27 import build_kdd27
        return build_kdd27(config,runtime=runtime)
    return build_market_society(config,runtime=runtime)


def offline_limits():
    return BudgetLimits(usd_nano=0,cny_nano=0,max_attempts=10000,max_tokens=2_000_000_000)


def runtime_for(config, metadata, run_id, lineage, branch, *, replay_path=None, tape=None, reopening=False):
    if config.cognition_mode == 'rule':
        return None
    provider_config = ProviderConfig.model_validate(metadata['provider_config'])
    limits = BudgetLimits.model_validate(metadata['budget_limits'])
    if replay_path is not None:
        provider_config = provider_config.model_copy(update={'source':'archived_response'})
        provider = ArchivedProvider(tape)
        ledger = replay_path
    else:
        if config.cognition_mode != 'synthetic_test' or provider_config.source != 'synthetic_test':
            raise ValueError('WP08 execution is offline synthetic only; live branches remain blocked')
        if provider_config.price.input_nano or provider_config.price.output_nano or limits.usd_nano or limits.cny_nano:
            raise ValueError('WP08 offline budgets must have zero monetary limits and prices')
        provider = SyntheticProvider()
        ledger = metadata['control_ledger']
    store = ModelInvocationStore(ledger,existing=reopening and replay_path is None)
    if reopening and replay_path is None:
        if BudgetController.project_limits(store)!=limits:
            raise ValueError('Existing invocation ledger project budget differs')
        with store.transaction() as db:
            saved = db.execute('SELECT body FROM contracts WHERE name=?',('run:'+run_id,)).fetchone()
        if saved is None or BudgetLimits.model_validate_json(saved[0])!=limits:
            raise ValueError('Missing or changed invocation ledger run budget')
    budget = BudgetController(store,limits)
    budget.register_run(run_id,limits)
    return CognitiveRuntime(store=store,budget=budget,provider=provider,config=provider_config,run_id=run_id,
        request_namespace={'lineage':lineage,'branch':branch},allow_cache_reuse=metadata.get('allow_cache_reuse',False))


def verify_implementation(store):
    current = implementation_identity(financial=True)
    if (store.identity.code_hash != current['source_hash'] or
            store.identity.metadata['implementation']['dependency_hash'] != current['dependency_hash']):
        raise ValueError('Code/dependency identity differs; explicit migration to new artifacts required')
    return current


def create_run(config, destination, *, provider_config=None, limits=None, run_id=None):
    config = parse_config(config)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if config.cognition_mode not in ('rule','synthetic_test'):
        raise ValueError('New durable runs must be offline rule or synthetic_test')
    control = destination.with_name(destination.name+'-control.sqlite')
    if control.exists():
        raise FileExistsError('Control ledger already exists; use resume instead')
    implementation = implementation_identity(financial=True)
    metadata = {'implementation':implementation,'data_origin':'synthetic_scenario',
        'control_ledger':str(control),'provider_config':(provider_config or ProviderConfig(max_input_tokens=200000)).model_dump(mode='json'),
        'budget_limits':(limits or offline_limits()).model_dump(mode='json'),'allow_cache_reuse':False}
    run_id = run_id or destination.name+'-'+uuid.uuid4().hex
    runtime = runtime_for(config,metadata,run_id,run_id,run_id)
    world = build_persistent_world(config,runtime=runtime)
    store = CommitStore.create(destination,world,config=config.model_dump(mode='json'),code_hash=implementation['source_hash'],
        run_id=run_id,metadata=metadata)
    attach(world,store)
    if runtime is not None:
        store.receipt_reader = runtime.store.records
    return world, store, runtime


def verify_model_ledger(lineage, ledger):
    persisted = {r['attempt_id']:r for r in ledger.records()}
    for source in lineage:
        for row in source.archived_records():
            if row['attempt_id'] not in persisted or row != persisted[row['attempt_id']]:
                raise ValueError('Model fact ledger differs from immutable commit receipt')


def open_world(store, *, replay_path=None):
    from civitas.experiments.branch import configure_branch, validate_parent
    verify_implementation(store)
    lineage = validate_parent(store)
    config = parse_config(store.identity.config)
    metadata = store.identity.metadata
    tape = None
    if replay_path is not None and config.cognition_mode != 'rule':
        tape = store.archived_records()
    runtime = runtime_for(config,metadata,store.identity.run_id,store.identity.lineage_id,store.identity.branch_id,
                          replay_path=replay_path,tape=tape,reopening=True)
    if replay_path is not None and runtime is not None:
        config = config.model_copy(update={'cognition_mode':'archived_response'})
    world = build_persistent_world(config,runtime=runtime)
    configure_branch(world,store)
    if runtime is not None and replay_path is None:
        verify_model_ledger(lineage,runtime.store)
        for row in store.archived_records():
            runtime.emit(row)
        store.receipt_reader = runtime.store.records
    return world, runtime


def validated_snapshot(store, *, tick=None):
    # Construct validators without consuming decisions or modifying the source billing log.
    with tempfile.TemporaryDirectory(prefix='civitas-state-validation-') as folder:
        world,_ = open_world(store,replay_path=Path(folder)/'validation.sqlite')
        marker = restore(world,store,tick=tick)
        return marker,world.snapshot


async def advance(world, store, runtime, *, until=None):
    config = parse_config(store.identity.config)
    until = config.ticks if until is None else until
    if type(until) is not int or not world.snapshot.tick <= until <= config.ticks:
        raise ValueError('Target tick must be between committed head and the frozen horizon')
    started = time.perf_counter()
    before = world.snapshot.tick
    status, error = 'completed', None
    registry.record(store.root,{'status':'running','from_tick':before,'until':until,'command':sys.argv})
    try:
        while world.snapshot.tick < until:
            await world.step()
        status = 'completed' if until == config.ticks else 'checkpointed'
    except BaseException as exc:
        status, error = ('cancelled' if isinstance(exc,(KeyboardInterrupt,asyncio.CancelledError)) else 'failed'), type(exc).__name__+': '+str(exc)
        raise
    finally:
        receipt = {'status':status,'error':error,'from_tick':before,'tick':world.snapshot.tick,'until':until,
            'state_hash':content_hash(world.snapshot),'elapsed_seconds':time.perf_counter()-started,
            'run_id':store.identity.run_id,'command':sys.argv,'exit_code':0 if error is None else 1}
        try:
            receipt['checkpoint'] = save(store)
            if runtime is not None:
                export = store.root/'diagnostics'/('model-facts-'+uuid.uuid4().hex+'.json')
                atomic_json(export,{'invocations':runtime.store.records(),'budget':runtime.budget.snapshot()})
                receipt['model_facts'] = str(export)
        except BaseException as exc:
            receipt.update(status='export_failed',export_error=type(exc).__name__+': '+str(exc),exit_code=1)
            raise
        finally:
            registry.record(store.root,receipt)
    return receipt


async def resume_run(root, *, until=None):
    store = CommitStore(root)
    with registry.run_lock(store.root):
        world, runtime = open_world(store)
        restore(world,store)
        store.repair_index()
        return await advance(world,store,runtime,until=until)


async def replay_run(root, *, mode, destination):
    from civitas.storage.replay import state_replay, verification_replay
    if mode not in ('state','verification'):
        raise ValueError('Replay requires exactly one mode')
    store = CommitStore(root)
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    if mode == 'state':
        _,snapshot = validated_snapshot(store)
        atomic_json(destination/'state.json',snapshot.model_dump(mode='json'))
        report = {'status':'matched','mode':mode,'tick':snapshot.tick,'state_hash':content_hash(snapshot),'executed_ticks':0}
    else:
        world, runtime = open_world(store,replay_path=destination/'verification-control.sqlite')
        report = await verification_replay(store,lambda:world)
        report['mode'] = mode
        if runtime is not None:
            report['budget'] = runtime.budget.snapshot()
            atomic_json(destination/'invocations.json',runtime.store.records())
    atomic_json(destination/'report.json',report)
    return report
