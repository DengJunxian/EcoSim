"""Strict research input validation and the Runner's explicit v2 composition.

Phase 2 exposes validate only. Actual run entry remains ExperimentRunner;
archive-v3 commands are introduced with the complete Phase 3 verifier.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import traceback
import uuid

from core.experiments.design_config import ResearchRunSpec, load_research_model, strict_json_loads
from core.experiments.identity import (IdentityNode, build_implementation_manifest, file_sha256,
    research_hash, scientific_projection, validate_identity_graph, verify_implementation_manifest)
from core.experiments.measurement import aggregate_run_metrics
from core.experiments.randomness import KeyedRandomSource
from core.experiments.stepper import (DiagnosticJournal, NoOpInformationService, NoOpPolicyService,
    ResearchStepper, RuntimeServices, validate_research_execution)
from core.experiments.world import build_initial_world, restore_initial_world
from agents.research_kernel import ResearchDecisionKernel

ROOT = Path(__file__).resolve().parents[2]


def write_research_json(path: Path, payload) -> None:
    from core.experiments.archive import durable_json
    durable_json(path,payload)


def local_import_closure(root_paths: list[str]) -> list[str]:
    """Conservative local Python dependency closure from explicit runtime roots.

    Includes package initializers and imports in optional branches, never tests,
    outputs, environment secrets, or an unbounded repository directory scan.
    External libraries are bound separately by the environment lock.
    """
    pending, found = list(root_paths), set()
    def add_module(name):
        if not name or name.startswith('.'):
            return
        parts = name.split('.')
        for size in range(1, len(parts)+1):
            stem = Path(*parts[:size])
            for candidate in (stem.with_suffix('.py'), stem/'__init__.py'):
                if (ROOT/candidate).is_file() and str(candidate) not in found:
                    pending.append(candidate.as_posix())
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        found.add(relative)
        path = ROOT / relative
        module_parts = list(Path(relative).with_suffix('').parts)
        package = module_parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                name = '.'.join(prefix + ([node.module] if node.module else []))
                add_module(name)
                for alias in node.names:
                    if alias.name != '*':
                        add_module(name + '.' + alias.name)
    return sorted(found)


def validate_resources(spec: ResearchRunSpec) -> None:
    refs = set(spec.provenance_refs) | set(spec.world.data_origin_refs) | set(spec.scenario.origin_refs)
    refs |= {row.origin_ref for row in spec.scenario.initial_information}
    if not refs <= set(spec.services.resource_hashes):
        raise ValueError('UNRESOLVED_PROVENANCE: every input origin needs a bound resource')
    if spec.base.historical_input is not None:
        selected = spec.base.historical_input
        if spec.services.resource_hashes.get(selected.path) != selected.sha256:
            raise ValueError('UNRESOLVED_HISTORICAL_RESOURCE: archive the actual selected data file')
    coupling_matches = []
    for relative, digest in spec.services.resource_hashes.items():
        path = (ROOT/relative).resolve(strict=True)
        if not path.is_relative_to(ROOT) or path.is_dir() or Path(relative).is_absolute():
            raise ValueError('Resources must use repository-relative file paths')
        if file_sha256(path) != digest:
            raise ValueError(f'RESOURCE_HASH_MISMATCH: {relative}')
        if path.suffix == '.json':
            payload = strict_json_loads(path.read_text(encoding='utf-8'))
            if research_hash(payload) == spec.random.coupling_spec_hash:
                from core.experiments.randomness import CHANNEL_REGISTRY
                declared=payload.get('channels',{}) if isinstance(payload,dict) else {}
                if any(declared.get(channel)!={'market_phase':CHANNEL_REGISTRY[channel][0],
                        'distribution':CHANNEL_REGISTRY[channel][1]} for channel in spec.random.channels):
                    raise ValueError('COUPLING_CHANNEL_MISMATCH: actual declared mechanism domains differ')
                coupling_matches.append(relative)
    if len(coupling_matches) != 1:
        raise ValueError('UNRESOLVED_COUPLING: one actual coupling descriptor must match its semantic hash')


def runtime_implementation_manifest(spec: ResearchRunSpec):
    paths = local_import_closure(['core/experiments/runner.py', 'core/experiments/research_cli.py'])
    roles = {path: 'code' for path in paths}
    for path in spec.services.resource_hashes:
        if path not in roles:
            paths.append(path)
        roles[path] = 'resource'
    lock = 'requirements-cscwd.lock'
    paths.append(lock)
    roles[lock] = 'configuration'
    return build_implementation_manifest(ROOT, paths, roles=roles)


def validate_research_spec(spec: ResearchRunSpec) -> dict:
    validate_research_execution(spec)
    validate_resources(spec)
    # A historical first price must already have been resolved into an explicit
    # effective specification; run initialization never silently changes P0.
    if spec.base.historical_input is not None:
        from core.validation.calibration import validate_historical_initial_price
        validate_historical_initial_price(spec, ROOT)
    manifest = runtime_implementation_manifest(spec)
    world = build_initial_world(spec, source_hash=manifest.source_hash) if spec.world.initialization_mode == 'generated' else None
    if world is not None and spec.services.scenario_version=='causal-scenarios-v1':
        from core.interventions.runtime import ScenarioRuntime
        components=restore_initial_world(world,spec=spec,source_hash=manifest.source_hash)
        ScenarioRuntime(spec,components,KeyedRandomSource(spec.random))
    return {'status': 'VALID', 'research_schema_version': 2, 'archive_scope': 'archive_v3',
        'effective_spec_hash': research_hash(spec), 'source_hash': manifest.source_hash,
        'initial_world_hash': None if world is None else world.world_hash,
        'lower_llm_calls': 0, 'scientific_gate_status': 'NOT_EVALUATED'}


def run_research(runner, output_root: str | Path) -> Path:
    spec = runner.research_spec
    validate_research_spec(spec)
    spec.require_base_match(runner.config)
    if hasattr(runner, 'output_dir'):
        raise RuntimeError('ExperimentRunner is single-use; create a new runner')
    manifest = runtime_implementation_manifest(spec)
    source_hash = manifest.source_hash
    rng = KeyedRandomSource(spec.random)
    initial = runner.research_initial_world
    if initial is None:
        initial = build_initial_world(spec, source_hash=source_hash, random_source=rng)
    world = restore_initial_world(initial, spec=spec, source_hash=source_hash)
    run_id = f'{spec.base.name}-{uuid.uuid4().hex[:16]}'
    output = Path(output_root).resolve()/run_id
    output.mkdir(parents=True, exist_ok=False)
    runner.output_dir = output
    started = datetime.now(timezone.utc).isoformat()
    write_research_json(output/'effective_spec.json', spec.model_dump(mode='json'))
    write_research_json(output/'initial_world.json', initial.model_dump(mode='json'))
    write_research_json(output/'implementation_manifest.json', manifest.model_dump(mode='json'))
    for entry in manifest.files:
        target = output/'source_bundle'/entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/entry.path).read_bytes())
    from core.experiments.counterfactual import execution_environment
    environment=execution_environment()
    write_research_json(output/'environment.json',environment)
    # In the v2 composition these exact runtime objects belong to the same
    # ExperimentRunner, not a separate matching engine or legacy price replay.
    runner.population, runner.graph, runner.diffusion = world.population, world.social, world.diffusion
    runner.book, runner.clearing = world.book, world.clearing
    journal = DiagnosticJournal(output)
    from core.interventions.runtime import ScenarioRuntime, PolicyRuntime
    scenario=ScenarioRuntime(spec,world,rng) if spec.services.scenario_version=='causal-scenarios-v1' else NoOpInformationService(spec)
    policy=PolicyRuntime(spec,world.clearing) if spec.services.policy_version=='funded-policy-v1' else NoOpPolicyService(spec)
    services = RuntimeServices(world, rng, scenario, policy,
                               ResearchDecisionKernel(), journal)
    stepper = ResearchStepper(spec, services, run_id=run_id)
    runner.research_stepper = stepper
    status = {'archive_schema_version':3,'run_id':run_id,'attempt_id':run_id,
        'status':'running','completed_steps':0,'source_hash':source_hash,'world_hash':initial.world_hash,
        'effective_spec_hash':research_hash(spec),'origin':'simulation_output','purpose':spec.purpose,
        'data_phase':spec.data_phase,'started_at':started,'lower_llm_calls':0,
        'scientific_gate_status':'NOT_EVALUATED'}
    write_research_json(output/'run_status.json', status)
    try:
        while stepper.completed_step < spec.base.steps:
            stepper.advance_one_step()
        verify_implementation_manifest(ROOT, manifest)
        verify_implementation_manifest(output/'source_bundle', manifest)
        status['status'] = 'completed'
        result = aggregate_run_metrics(stepper.measurements, spec=spec.measurement, run_ref=research_hash(spec))
        write_research_json(output/'run_metrics.json', result.model_dump(mode='json'))
        streams = {}
        for path in sorted(output.glob('*.jsonl')):
            streams[path.stem] = [scientific_projection(strict_json_loads(line))
                                 for line in path.read_text(encoding='utf-8').splitlines()]
        result_body = {'streams':streams, 'run_metrics':result.model_dump(mode='json')}
        nodes = {}
        def node(kind, body, dependencies=()):
            record = IdentityNode(kind=kind, body=body, dependencies=list(dependencies))
            nodes[record.content_hash] = record
            return record.content_hash
        source = node('implementation',manifest.model_dump(mode='json'))
        environment_ref=node('environment',environment)
        effective = node('effective_spec',spec.model_dump(mode='json'),[source,environment_ref])
        initial_ref = node('initial_world',initial.model_dump(mode='json'),[source])
        result_ref = node('run_result', result_body, [initial_ref,effective])
        validate_identity_graph(nodes)
        write_research_json(output/'identity_graph.json', {key:value.model_dump(mode='json') for key,value in nodes.items()})
        status['result_content_hash'] = result_ref
    except BaseException as exc:
        status.update(status='failed', failure_type=type(exc).__name__, failure_message=str(exc),
                      failed_phase=stepper.active_phase)
        write_research_json(output/'failure.json', {'error_type':type(exc).__name__,'message':str(exc),
            'phase':stepper.active_phase,'completed_step':stepper.completed_step,
            'traceback':traceback.format_exc(),'clearing':world.clearing.diagnostic_snapshot()})
        raise
    finally:
        status['completed_steps'] = stepper.completed_step
        status['ended_at'] = datetime.now(timezone.utc).isoformat()
        write_research_json(output/'run_status.json',status)
        write_research_json(output/'phase_trace.json',stepper.phase_trace)
        from core.experiments.archive import finalize_archive
        try:
            finalize_archive(runner,status)
        except BaseException as archive_error:
            if status['status']=='completed':
                status.update(status='failed',failure_type=type(archive_error).__name__,failure_message=str(archive_error),
                              failed_phase='archive_finalization')
                write_research_json(output/'run_status.json',status)
                raise
            (output/'archive_finalization_error.txt').write_text(type(archive_error).__name__+': '+str(archive_error)+'\n')
    return output


def engineering_budget(path,*,run_count,confirmation=False):
    """Finite local diagnostic allowance; all language resources stay zero."""
    from core.codesign.budget import BudgetSpec,BudgetLedger
    spec=BudgetSpec(llm_calls=0,input_tokens=0,output_tokens=0,unique_market_runs=run_count,
        max_attempts=max(1,run_count*2),candidate_submissions=0,information_queries=0,messages=0,rounds=0,
        confirmation_batches=int(confirmation),wall_timeout_seconds=600.0)
    return BudgetLedger(path,spec,owner='engineering-rule',episode=str(Path(path).resolve()),alpha_episode=0.05)


def execute_confirmation_example(path,output,workers):
    from core.experiments.evidence import (ConfirmationRequest,ConfirmationAtom,engineering_scope,
        register_confirmation,certify_confirmation)
    from core.experiments.jobs import Job,JobCoordinator
    from core.experiments.archive import durable_json
    path=Path(path).resolve();data=strict_json_loads(path.read_text())
    expected={'schema_version','purpose','plan_id','model_id','spec_refs','replication_ids','quantity_id','direction','threshold','alpha'}
    if set(data)!=expected or data['schema_version']!=1 or data['purpose']!='engineering_pipeline_only':
        raise ValueError('Only the explicit engineering confirmation adapter is registered; financial confirmation requires scoped gates')
    specs=[load_research_model(path.parent/ref,ResearchRunSpec) for ref in data['spec_refs']]
    if len(specs)!=1:raise ValueError('Example adapter declares one absolute quantity; full API supports planned comparisons')
    model=data['model_id'];scope,descriptors=engineering_scope({f'{model}/policy-{i}':s for i,s in enumerate(specs)})
    atom=ConfirmationAtom(atom_id='pipeline-quote-coverage',kind='absolute_utility',policy_hashes=list(scope.policy_hashes),
        model_id=model,quantity_id=data['quantity_id'],direction=data['direction'],threshold=data['threshold'],
        alpha=data['alpha'],support_lower=0.0,support_upper=1.0)
    request=ConfirmationRequest(plan_id=data['plan_id'],scope=scope,n_complete_pairs_or_runs=len(data['replication_ids']),
        alpha_episode=0.05,alpha_batch=data['alpha'],atoms=[atom],planned_replication_ids=data['replication_ids'])
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'evidence.json').exists():raise ValueError('Immutable confirmation output already exists')
    port=engineering_budget(output/'budget.json',run_count=len(data['replication_ids'])*len(specs),confirmation=True)
    plan,registration=register_confirmation(request,models={model:specs},reservation_port=port,output=output/'registration')
    jobs=[Job(job_id=f'run-{i:04d}',spec=ResearchRunSpec.model_validate(binding['spec']))
          for i,binding in enumerate(registration['semantic_body']['bindings'])]
    batch=JobCoordinator(output/'jobs',port).execute(jobs,workers=workers,plan_hash=registration['plan_hash'])
    if batch['status']!='COMMITTED':
        result={'status':'INCOMPLETE','batch':batch,'financial_eligibility':'NOT_EVALUATED'}
        durable_json(output/'incomplete.json',result);return result
    records=certify_confirmation(plan,registration,batch,reservation_port=port)
    result={'status':'COMPLETED_ENGINEERING_ONLY','records':[r.model_dump(mode='json') for r in records],
            'financial_eligibility':'NOT_EVALUATED','llm_calls':0}
    durable_json(output/'evidence.json',result);return result


def main(argv=None) -> int:
    from core.experiments.archive import ArtifactIntegrityError
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    validate = sub.add_parser('validate')
    validate.add_argument('--spec',required=True)
    run=sub.add_parser('run');run.add_argument('--spec',required=True);run.add_argument('--out',required=True)
    for name in ('pair','confirm'):
        command=sub.add_parser(name);command.add_argument('--plan',required=True);command.add_argument('--out',required=True)
        command.add_argument('--workers',type=int,default=1)
    args = parser.parse_args(argv)
    executing=False
    try:
        if args.command=='pair':
            from core.experiments.counterfactual import PairPlan,evaluate_batch
            plan=load_research_model(args.plan,PairPlan)
            destination=Path(args.out).resolve()
            if destination.exists():raise ValueError('Pair output already exists')
            port=engineering_budget(destination.parent/(destination.name+'-budget.json'),run_count=len(plan.replication_ids)*2)
            executing=True
            result=evaluate_batch(plan,base_dir=Path(args.plan).resolve().parent,output=destination,
                reservation_port=port,workers=args.workers)
            print(json.dumps(result,ensure_ascii=False));return 0 if result['status']=='COMMITTED' else 4
        if args.command=='confirm':
            result=execute_confirmation_example(args.plan,args.out,args.workers)
            print(json.dumps(result,ensure_ascii=False));return 0 if result['status']=='COMPLETED_ENGINEERING_ONLY' else 3
        spec = load_research_model(args.spec,ResearchRunSpec)
        if args.command=='validate':result=validate_research_spec(spec)
        else:
            from core.experiments.runner import ExperimentRunner
            from core.experiments.archive import verify_archive_v3,write_run_index
            destination=Path(args.out).resolve()
            if (destination/'run-index.json').exists():raise ValueError('Immutable run index already exists')
            runner=ExperimentRunner(spec.base.to_legacy(),research_spec=spec)
            validate_research_spec(spec);executing=True
            archive=runner.run(destination/'runs')
            result=verify_archive_v3(archive)
            write_run_index(destination/'run-index.json',[archive])
            result['run_index']=str(destination/'run-index.json')
        print(json.dumps(result,ensure_ascii=False,sort_keys=True))
        return 0
    except ArtifactIntegrityError as exc:
        print(json.dumps({'status':'INVALID_ARTIFACT','reason':str(exc)},ensure_ascii=False));return 5
    except (ValueError, OSError, TypeError,RuntimeError) as exc:
        print(json.dumps({'status':'INVALID_INPUT','error_type':type(exc).__name__,'reason':str(exc)},ensure_ascii=False))
        if type(exc).__name__=='BudgetExceeded':return 3
        return 4 if executing else 2


if __name__ == '__main__':
    raise SystemExit(main())
