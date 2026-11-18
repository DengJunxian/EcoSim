"""Immutable archive-v3 envelopes, semantic checks and frozen-source reruns."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
from typing import Literal
import uuid

from pydantic import Field,model_validator
from core.experiments.strict_types import StrictResearchModel,strict_json_loads
from core.experiments.identity import (HASH_PATTERN,ImplementationManifest,IdentityNode,file_sha256,
    research_hash,scientific_projection,verify_implementation_manifest,validate_identity_graph)
from core.experiments.design_config import ResearchRunSpec,MeasurementSpec
from core.experiments.world import InitialWorldSnapshot,validate_initial_world
from core.experiments.measurement import (StepMeasurement,MarketSnapshot,StepActivity,MeasurementContext,
    SettledTrade,compute_step_measurement,aggregate_run_metrics)
from core.experiments.randomness import RandomDraw,RandomSpec


def durable_json(path:Path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class ArchiveFile(StrictResearchModel):
    path:str
    sha256:str=Field(pattern=HASH_PATTERN)
    bytes:int=Field(ge=0)

    @model_validator(mode='after')
    def _path(self):
        path=PurePosixPath(self.path)
        if (path.is_absolute() or '..' in path.parts or '\\' in self.path
                or path.as_posix()!=self.path or self.path in {'','.'}):
            raise ValueError('Archive paths must be canonical contained relative paths')
        return self


class ArchiveManifest(StrictResearchModel):
    archive_schema_version:Literal[3]=3
    run_id:str=Field(min_length=1)
    attempt_id:str=Field(min_length=1)
    execution_status:Literal['completed','failed']
    completed_steps:int=Field(ge=0)
    source_hash:str=Field(pattern=HASH_PATTERN)
    effective_spec_hash:str=Field(pattern=HASH_PATTERN)
    initial_world_hash:str=Field(pattern=HASH_PATTERN)
    result_content_hash:str|None=Field(pattern=HASH_PATTERN)
    scientific_eligibility:Literal['NOT_EVALUATED']='NOT_EVALUATED'
    files:list[ArchiveFile]=Field(min_length=1)

    @model_validator(mode='after')
    def _entries(self):
        names=[row.path for row in self.files]
        if names!=sorted(set(names)) or {'archive_manifest.json','COMMITTED.json'}&set(names):
            raise ValueError('Manifest needs unique sorted file paths and cannot contain itself or its later commit marker')
        if (self.result_content_hash is not None)!=(self.execution_status=='completed'):
            raise ValueError('Only a completed execution has a complete scientific result identity')
        return self


def rows(path:Path):
    return [strict_json_loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def plain_order(order):
    value=asdict(order)
    for key in ('side','order_type','status'):value[key]=getattr(order,key).value
    return value


def finalize_archive(runner,status):
    root=runner.output_dir;spec=runner.research_spec;stepper=runner.research_stepper;world=stepper.services.world
    if (root/'archive_manifest.json').exists() or (root/'COMMITTED.json').exists():
        raise ValueError('Archive attempts are immutable and finalized once')
    for name,payload in [('config.json',spec.base.model_dump(mode='json')),
        ('research_spec.json',spec.model_dump(mode='json')),('world.json',world.snapshot.model_dump(mode='json')),
        ('population.json',world.snapshot.population.model_dump(mode='json')),
        ('random_spec.json',spec.random.model_dump(mode='json')),('behavior_spec.json',spec.behavior.model_dump(mode='json')),
        ('measurement_spec.json',spec.measurement.model_dump(mode='json')),
        ('compiled_policy.json',stepper.services.policy.compiled.model_dump(mode='json')),
        ('model_calls.json',[]),('run_state.json',status),('summary.json',status)]:
        durable_json(root/name,payload)
    if not world.clearing.failed:
        durable_json(root/'orders.json',[plain_order(order) for _,order in sorted(world.clearing.orders.items())])
    else:
        durable_json(root/'failed_clearing_state.json',world.clearing.diagnostic_snapshot())
    durable_json(root/'step_receipts.json',rows(root/'step_commits.jsonl') if (root/'step_commits.jsonl').exists() else [])
    draws=rows(root/'random.jsonl') if (root/'random.jsonl').exists() else []
    durable_json(root/'draw_audit.json',{'algorithm':spec.random.algorithm,'requested_audit_mode':spec.random.audit_mode,
        'storage':'full_records_with_channel_step_digests','record_count':len(draws),
        'record_hashes':[row['record_hash'] for row in draws],
        'channel_step_digests':{f'{channel}:{step}':research_hash([row['draw'] for row in draws
            if row['draw']['key']['channel']==channel and row['draw']['key']['step']==step])
            for channel,step in sorted({(row['draw']['key']['channel'],row['draw']['key']['step']) for row in draws})}})
    environment=strict_json_loads((root/'environment.json').read_text())
    (root/'requirements-lock.txt').write_text(''.join(f'{name}=={version}\n' for name,version in sorted(environment['installed_packages'].items())))
    files=[]
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Archive cannot include symlinks')
        if path.is_file():files.append(ArchiveFile(path=path.relative_to(root).as_posix(),sha256=file_sha256(path),bytes=path.stat().st_size))
    manifest=ArchiveManifest(run_id=status['run_id'],attempt_id=status['attempt_id'],execution_status=status['status'],
        completed_steps=status['completed_steps'],source_hash=status['source_hash'],effective_spec_hash=research_hash(spec),
        initial_world_hash=world.snapshot.world_hash,result_content_hash=status.get('result_content_hash'),files=files)
    durable_json(root/'archive_manifest.json',manifest.model_dump(mode='json'))
    if status['status']=='completed':
        durable_json(root/'COMMITTED.json',{'run_id':status['run_id'],'completed_steps':status['completed_steps'],
            'manifest_sha256':file_sha256(root/'archive_manifest.json'),'result_content_hash':status['result_content_hash']})
    return manifest


def resolve_run_reference(reference:str|Path)->Path:
    path=Path(reference).resolve(strict=True)
    if path.is_dir():return path
    index=strict_json_loads(path.read_text())
    if set(index)!={'schema_version','runs'} or index['schema_version']!=1 or len(index['runs'])!=1:
        raise ValueError('RUN_INDEX_AMBIGUOUS: single-run operation requires exactly one immutable archive')
    record=index['runs'][0]
    if set(record)!={'path','manifest_sha256'}:raise ValueError('Invalid run index record')
    root=(path.parent/record['path']).resolve(strict=True)
    if file_sha256(root/'archive_manifest.json')!=record['manifest_sha256']:
        raise ValueError('RUN_INDEX_MANIFEST_MISMATCH')
    return root


def write_run_index(path:Path,run_dirs:list[Path]):
    if path.exists():raise ValueError('Run index is immutable')
    durable_json(path,{'schema_version':1,'runs':[{'path':os.path.relpath(root,path.parent),
        'manifest_sha256':file_sha256(root/'archive_manifest.json')} for root in run_dirs]})


class ArtifactIntegrityError(ValueError):
    """A persisted scientific artifact cannot be verified (CLI exit 5)."""


def verify_archive_v3(reference:str|Path,*,allow_failed=False):
    try:return _verify_archive_v3(reference,allow_failed=allow_failed)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        raise ArtifactIntegrityError(str(exc)) from exc


def _verify_archive_v3(reference:str|Path,*,allow_failed=False):
    root=resolve_run_reference(reference)
    manifest=ArchiveManifest.model_validate_json((root/'archive_manifest.json').read_text())
    if manifest.execution_status!='completed' and not allow_failed:
        raise ValueError('EXECUTION_NOT_COMPLETED: forensic integrity is not a usable sample')
    expected={row.path for row in manifest.files}|{'archive_manifest.json'}
    if manifest.execution_status=='completed':
        marker=strict_json_loads((root/'COMMITTED.json').read_text())
        if marker!={'run_id':manifest.run_id,'completed_steps':manifest.completed_steps,
            'manifest_sha256':file_sha256(root/'archive_manifest.json'),'result_content_hash':manifest.result_content_hash}:
            raise ValueError('COMMIT_MARKER_MISMATCH')
        expected.add('COMMITTED.json')
    actual={path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    if actual!=expected:raise ValueError('ARCHIVE_FILE_SET_MISMATCH')
    for entry in manifest.files:
        path=root/entry.path
        if path.is_symlink() or not path.resolve().is_relative_to(root) or file_sha256(path)!=entry.sha256 or path.stat().st_size!=entry.bytes:
            raise ValueError(f'ARCHIVE_CONTENT_MISMATCH: {entry.path}')
    spec=ResearchRunSpec.model_validate_json((root/'research_spec.json').read_text())
    if research_hash(spec)!=manifest.effective_spec_hash:raise ValueError('EFFECTIVE_SPEC_IDENTITY_MISMATCH')
    source=ImplementationManifest.model_validate_json((root/'implementation_manifest.json').read_text())
    verify_implementation_manifest(root/'source_bundle',source)
    if source.source_hash!=manifest.source_hash:raise ValueError('SOURCE_IDENTITY_MISMATCH')
    initial=InitialWorldSnapshot.model_validate_json((root/'initial_world.json').read_text())
    validate_initial_world(initial,spec=spec,source_hash=manifest.source_hash)
    if initial.world_hash!=manifest.initial_world_hash:raise ValueError('INITIAL_WORLD_IDENTITY_MISMATCH')
    from core.interventions.compiler import compile_policy
    compiled=compile_policy(spec.policy,constraints=spec.policy_constraints,world=spec.world,market=spec.base.market)
    if strict_json_loads((root/'compiled_policy.json').read_text())!=compiled.model_dump(mode='json'):
        raise ValueError('COMPILED_POLICY_IDENTITY_MISMATCH')
    pairs=[('effective_spec.json',spec.model_dump(mode='json')),('config.json',spec.base.model_dump(mode='json')),
        ('random_spec.json',spec.random.model_dump(mode='json')),('measurement_spec.json',spec.measurement.model_dump(mode='json')),
        ('behavior_spec.json',spec.behavior.model_dump(mode='json')),('world.json',initial.model_dump(mode='json')),
        ('population.json',initial.population.model_dump(mode='json'))]
    for name,value in pairs:
        if strict_json_loads((root/name).read_text())!=value:raise ValueError(f'ARCHIVE_SCOPE_MISMATCH: {name}')
    environment=strict_json_loads((root/'environment.json').read_text())
    lock=''.join(f'{name}=={version}\n' for name,version in sorted(environment['installed_packages'].items()))
    if (root/'requirements-lock.txt').read_text()!=lock:raise ValueError('ENVIRONMENT_LOCK_MISMATCH')
    status=strict_json_loads((root/'run_status.json').read_text())
    if (status!=strict_json_loads((root/'run_state.json').read_text()) or status!=strict_json_loads((root/'summary.json').read_text())
            or status['source_hash']!=manifest.source_hash or status['status']!=manifest.execution_status
            or status['completed_steps']!=manifest.completed_steps):raise ValueError('RUN_STATE_MISMATCH')
    if strict_json_loads((root/'model_calls.json').read_text())!=[] or status['lower_llm_calls']!=0:
        raise ValueError('LOWER_LLM_SCOPE_MISMATCH')
    streams={}
    for path in sorted(root.glob('*.jsonl')):
        records=rows(path)
        for row in records:
            declared=row.get('record_hash');body={key:value for key,value in row.items() if key!='record_hash'}
            if declared!=research_hash(scientific_projection(body)):
                raise ValueError(f'JOURNAL_RECORD_IDENTITY_MISMATCH: {path.name}')
        streams[path.stem]=records
    for row in streams.get('random',[]):
        draw=RandomDraw.model_validate(row['draw']);key=draw.key
        if (key.root_seed,key.exogenous_family_id,key.replication_id,key.coupling_spec_hash)!=(spec.random.root_seed,
            spec.random.exogenous_family_id,spec.random.replication_id,spec.random.coupling_spec_hash):
            raise ValueError('DRAW_SCOPE_MISMATCH')
    addresses=[research_hash(row['draw']['key']) for row in streams.get('random',[])]
    if len(addresses)!=len(set(addresses)):raise ValueError('DUPLICATE_DRAW_ADDRESS')
    draws=streams.get('random',[])
    audit={'algorithm':spec.random.algorithm,'requested_audit_mode':spec.random.audit_mode,
        'storage':'full_records_with_channel_step_digests','record_count':len(draws),
        'record_hashes':[row['record_hash'] for row in draws],
        'channel_step_digests':{f'{channel}:{step}':research_hash([row['draw'] for row in draws
            if row['draw']['key']['channel']==channel and row['draw']['key']['step']==step])
            for channel,step in sorted({(row['draw']['key']['channel'],row['draw']['key']['step']) for row in draws})}}
    if strict_json_loads((root/'draw_audit.json').read_text())!=audit:raise ValueError('DRAW_AUDIT_MISMATCH')
    if manifest.execution_status!='completed':
        return {'status':'verified_forensic_attempt','execution_status':'failed','scientific_eligibility':'NOT_ELIGIBLE',
                'run_id':manifest.run_id,'checked_files':len(manifest.files)}
    required={'information','policy_due','social','random','measurements','market','step_activity','decisions',
              'observations','accounts','agents','social_state','policy_events','policy_execution','step_commits'}
    if not required<=set(streams):raise ValueError('ARCHIVE_REQUIRED_STREAM_MISSING')
    clock=list(range(1,spec.base.steps+1))
    if manifest.completed_steps!=spec.base.steps:raise ValueError('INCOMPLETE_CLOCK')
    for name in ('information','policy_due','social','measurements','market','step_activity','decisions','accounts','social_state','policy_events','step_commits'):
        if [row['step'] for row in streams[name]]!=clock:raise ValueError(f'JOURNAL_CLOCK_MISMATCH: {name}')
    if strict_json_loads((root/'step_receipts.json').read_text())!=streams['step_commits']:
        raise ValueError('STEP_RECEIPT_ALIAS_MISMATCH')
    for index,commit in enumerate(streams['step_commits']):
        from core.experiments.stepper import ResearchStepper
        if (commit['completed_step']!=index+1 or commit['market_record_hash']!=streams['measurements'][index]['record_hash']
                or commit['policy_record_hash']!=streams['policy_events'][index]['record_hash']
                or commit['phase_order']!=list(ResearchStepper.phase_names)):
            raise ValueError('STEP_COMMIT_LINEAGE_MISMATCH')
    roster=set(spec.world.stable_actor_ids)|{spec.world.policy_account_initial.actor_id}
    for record in streams['accounts']:
        if set(record['accounts'])!=roster:raise ValueError('ACCOUNT_ROSTER_MISMATCH')
    for t in [0]+clock:
        actors=[row['actor_id'] for row in streams['agents'] if row['step']==t]
        if len(actors)!=len(roster) or set(actors)!=roster:raise ValueError('AGENT_ROSTER_MISMATCH')
    # Independently replay settled cash/share transfers; this does not reuse the
    # clearing implementation's invariant verdict or substitute last prices.
    accounts={r['actor_id']:dict(r['account']) for r in streams['agents'] if r['step']==0}
    fees=0.0;seen_trades=set();fills={}
    for record in streams['accounts']:
        t=record['step']
        for trade in streams.get('trades',[]):
            if trade['step']!=t:continue
            if trade['trade_id'] in seen_trades:raise ValueError('DUPLICATE_SETTLED_TRADE')
            seen_trades.add(trade['trade_id']);q=trade['quantity'];notional=trade['price']*q
            buyer=accounts[trade['buyer_agent_id']];seller=accounts[trade['seller_agent_id']]
            buyer['cash']-=notional+trade['buyer_fee'];buyer['shares']+=q
            seller['cash']+=notional-trade['seller_fee']-trade['seller_tax'];seller['shares']-=q
            fees+=trade['buyer_fee']+trade['seller_fee']+trade['seller_tax']
            for key in ('maker_id','taker_id'):fills[trade[key]]=fills.get(trade[key],0)+q
        if not math.isclose(fees,record['fees'],rel_tol=1e-12,abs_tol=1e-9):raise ValueError('ARCHIVE_FEE_RECONCILIATION_FAILED')
        for aid,actual in record['accounts'].items():
            if accounts[aid]['shares']!=actual['shares'] or not math.isclose(accounts[aid]['cash'],actual['cash'],rel_tol=1e-12,abs_tol=1e-8):
                raise ValueError('ARCHIVE_ACCOUNT_RECONCILIATION_FAILED')
            agent=next(r for r in streams['agents'] if r['step']==t and r['actor_id']==aid)
            if agent['account']!=actual:raise ValueError('AGENT_ACCOUNT_ALIAS_MISMATCH')
    orders=strict_json_loads((root/'orders.json').read_text())
    if len({o['order_id'] for o in orders})!=len(orders) or not set(fills)<={o['order_id'] for o in orders}:
        raise ValueError('SETTLED_ORDER_LINEAGE_MISMATCH')
    for order in orders:
        if order['filled_qty']!=fills.get(order['order_id'],0):raise ValueError('ORDER_FILL_RECONCILIATION_FAILED')
    measurements=[]
    for index,row in enumerate(streams['measurements']):
        measurement=StepMeasurement.model_validate({k:v for k,v in row.items() if k!='record_hash'})
        if measurement.policy_hash!=compiled.policy_hash:raise ValueError('MEASUREMENT_POLICY_SCOPE_MISMATCH')
        trade_rows=[{k:v for k,v in trade.items() if k not in {'step','record_hash'}}
                    for trade in streams.get('trades',[]) if trade['step']==measurement.step]
        activity=StepActivity.model_validate(streams['step_activity'][index]['activity'])
        recomputed=compute_step_measurement(snapshot=MarketSnapshot.model_validate(streams['market'][index]['snapshot']),
            trade_tape=[SettledTrade.model_validate(trade) for trade in trade_rows],activity=activity,
            context=MeasurementContext(spec=spec.measurement,world_id=initial.world_hash,run_id=manifest.run_id,
                step=measurement.step,session_id=measurement.session_id,policy_hash=measurement.policy_hash,
                market_open=measurement.market_open,observation_complete=True,execution_status='completed',
                origin='market',policy_activity=measurement.policy_activity))
        if recomputed!=measurement:raise ValueError('MEASUREMENT_RECOMPUTATION_MISMATCH')
        measurements.append(measurement)
    metrics=aggregate_run_metrics(measurements,spec=spec.measurement,run_ref=research_hash(spec))
    if metrics.model_dump(mode='json')!=strict_json_loads((root/'run_metrics.json').read_text()):
        raise ValueError('RUN_AGGREGATION_MISMATCH')
    nodes={key:IdentityNode.model_validate(value) for key,value in strict_json_loads((root/'identity_graph.json').read_text()).items()}
    validate_identity_graph(nodes)
    if manifest.result_content_hash not in nodes:raise ValueError('RESULT_IDENTITY_UNRESOLVED')
    body={'streams':{name:[scientific_projection(row) for row in values] for name,values in streams.items()},
          'run_metrics':metrics.model_dump(mode='json')}
    if nodes[manifest.result_content_hash].body!=body:raise ValueError('RESULT_CONTENT_MISMATCH')
    return {'status':'verified','archive_schema_version':3,'run_id':manifest.run_id,'source_hash':manifest.source_hash,
        'result_content_hash':manifest.result_content_hash,'checked_files':len(manifest.files),
        'scientific_eligibility':'NOT_EVALUATED','scope':'integrity_scope_recomputation_only'}


def reproduce_archive_v3(reference:str|Path,output_root:str|Path):
    root=resolve_run_reference(reference);original=verify_archive_v3(root)
    output=Path(output_root).resolve();output.mkdir(parents=True,exist_ok=True)
    receipt_dir=output/f'reproduction-{uuid.uuid4().hex[:16]}';receipt_dir.mkdir()
    bundle=root/'source_bundle'
    script="""
import json,sys
from core.experiments.design_config import ResearchRunSpec,load_research_model
from core.experiments.runner import ExperimentRunner
spec=load_research_model(sys.argv[1],ResearchRunSpec)
run=ExperimentRunner(spec.base.to_legacy(),research_spec=spec).run(sys.argv[2])
print(json.dumps({'rerun':str(run)}))
"""
    environment=dict(os.environ,PYTHONPATH=str(bundle),PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-c',script,str(root/'research_spec.json'),str(receipt_dir/'runs')]
    result=subprocess.run(command,cwd=bundle,env=environment,capture_output=True,text=True)
    (receipt_dir/'stdout.log').write_text(result.stdout);(receipt_dir/'stderr.log').write_text(result.stderr)
    report={'original':str(root),'original_manifest_sha256':file_sha256(root/'archive_manifest.json'),
        'source_hash':original['source_hash'],'execution_source':'verified_original_source_bundle',
        'exit_code':result.returncode,'command':command,'created_at':datetime.now(timezone.utc).isoformat()}
    if result.returncode!=0:
        report['status']='rerun_failed';durable_json(receipt_dir/'reproduction.json',report)
        raise ValueError(f'Frozen-source reproduction failed: {receipt_dir}')
    rerun=Path(strict_json_loads(result.stdout.strip().splitlines()[-1])['rerun'])
    verified=verify_archive_v3(rerun)
    matched=verified['result_content_hash']==original['result_content_hash'] and verified['source_hash']==original['source_hash']
    report.update(status='matched' if matched else 'different',rerun=str(rerun),rerun_verification=verified)
    durable_json(receipt_dir/'reproduction.json',report)
    if not matched:raise ValueError(f'Frozen-source reproduction differs: {receipt_dir}')
    return dict(report,report=str(receipt_dir/'reproduction.json'))
