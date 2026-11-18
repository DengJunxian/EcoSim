"""Authoritative evidence types and fixed-batch, preregistered bounded inference."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
from typing import Annotated,Literal

from pydantic import Field,model_validator
from core.experiments.strict_types import StrictResearchModel,strict_json_loads
from core.experiments.identity import HASH_PATTERN,research_hash,file_sha256
from core.experiments.design_config import ResearchRunSpec
from core.experiments.archive import durable_json,rows,verify_archive_v3

Hash=Annotated[str,Field(pattern=HASH_PATTERN)]


class EvidenceScope(StrictResearchModel):
    task_hash:Hash
    policy_hashes:list[Hash]=Field(min_length=1)
    model_family_hash:Hash
    source_hash:Hash
    backend_hash:Hash
    world_distribution_hash:Hash
    scenario_hash:Hash
    behavior_hash:Hash
    timing_version:Literal['research-step-v1']
    horizon_steps:int=Field(ge=1)
    measurement_hash:Hash
    missing_rule_hash:Hash
    random_coupling_hash:Hash
    preference_hash:Hash
    acceptance_rule_hash:Hash

    @model_validator(mode='after')
    def _unique(self):
        if len(set(self.policy_hashes))!=len(self.policy_hashes):raise ValueError('Duplicate policy scope')
        return self


class ConfirmationAtom(StrictResearchModel):
    atom_id:str=Field(min_length=1)
    kind:Literal['paired_difference','absolute_utility','violation_probability']
    policy_hashes:list[Hash]=Field(min_length=1,max_length=2)
    model_id:str=Field(min_length=1)
    quantity_id:str=Field(min_length=1)
    direction:Literal['le','ge']
    threshold:float
    alpha:float=Field(gt=0,lt=1)
    support_lower:float
    support_upper:float

    @model_validator(mode='after')
    def _semantics(self):
        paired=self.kind=='paired_difference'
        if len(set(self.policy_hashes))!=len(self.policy_hashes) or len(self.policy_hashes)!=(2 if paired else 1):
            raise ValueError('Atom policy cardinality mismatch')
        if (self.support_lower,self.support_upper)!=((-1.0,1.0) if paired else (0.0,1.0)):
            raise ValueError('Frozen Hoeffding adapter requires certified [-1,1] differences or [0,1] absolutes')
        if not self.support_lower<=self.threshold<=self.support_upper:raise ValueError('Threshold outside support')
        return self


class ConfirmationRequest(StrictResearchModel):
    schema_version:Literal[1]=1
    plan_id:str=Field(min_length=1)
    scope:EvidenceScope
    data_phase:Literal['confirmation']='confirmation'
    n_complete_pairs_or_runs:int=Field(ge=1)
    method:Literal['fixed_batch_hoeffding']='fixed_batch_hoeffding'
    alpha_episode:float=Field(gt=0,lt=1)
    alpha_batch:float=Field(gt=0,lt=1)
    atoms:list[ConfirmationAtom]=Field(min_length=1)
    planned_replication_ids:list[str]=Field(min_length=1)

    @model_validator(mode='after')
    def _family(self):
        if len(set(self.planned_replication_ids))!=len(self.planned_replication_ids) or any(not r for r in self.planned_replication_ids):
            raise ValueError('Exact unique replication IDs required')
        if len(self.planned_replication_ids)!=self.n_complete_pairs_or_runs:raise ValueError('Fixed n must match complete ID list')
        if len({a.atom_id for a in self.atoms})!=len(self.atoms):raise ValueError('Duplicate atom')
        if not set(p for a in self.atoms for p in a.policy_hashes)<=set(self.scope.policy_hashes):raise ValueError('Atom outside policy scope')
        if sum((Fraction.from_float(a.alpha) for a in self.atoms),Fraction())>Fraction.from_float(self.alpha_batch):
            raise ValueError('Atom alpha exceeds batch allocation')
        if self.alpha_batch>self.alpha_episode:raise ValueError('Batch alpha exceeds episode')
        return self


class ConfirmationPlan(ConfirmationRequest):
    status:Literal['registered_before_visibility']='registered_before_visibility'
    registration_receipt_hash:Hash


class EvidenceRecord(StrictResearchModel):
    evidence_id:Hash
    atom_id:str
    scope:EvidenceScope
    plan_hash:Hash
    registration_receipt_hash:Hash
    data_phase:Literal['exploration','confirmation','audit']
    raw_run_refs:list[str]
    replication_ids:list[str]
    estimate:float|None
    interval:list[float]|None
    support:list[float]
    n:int=Field(ge=0)
    alpha:float|None=Field(gt=0,lt=1)
    method:Literal['fixed_batch_hoeffding','exact_analytic']
    failure_count:int=Field(ge=0)
    missing_count:int=Field(ge=0)
    validity:Literal['VALID','INCOMPLETE','SCOPE_MISMATCH','INVALID']
    verdict:Literal['SUPPORTED','REFUTED','UNRESOLVED','INAPPLICABLE','INVALID_ARTIFACT']
    data_visibility:dict

    @model_validator(mode='after')
    def _validity(self):
        if len(set(self.replication_ids))!=len(self.replication_ids) or self.n!=len(self.replication_ids):raise ValueError('Evidence n counts unique complete IDs')
        if len(self.support)!=2 or self.support[0]>=self.support[1]:raise ValueError('Invalid evidence support')
        if self.interval is not None and (len(self.interval)!=2 or not self.support[0]<=self.interval[0]<=self.interval[1]<=self.support[1]):
            raise ValueError('Evidence interval outside support')
        if self.method=='fixed_batch_hoeffding' and self.alpha is None:raise ValueError('Statistical evidence needs alpha')
        if self.method=='exact_analytic' and (self.alpha is not None or self.n or self.replication_ids or self.data_phase!='audit'):
            raise ValueError('Analytic proof has no statistical samples or alpha')
        if self.verdict in {'SUPPORTED','REFUTED'} and (self.validity!='VALID' or self.interval is None or self.failure_count or self.missing_count):
            raise ValueError('Only complete valid evidence can support or refute')
        body=self.model_dump(mode='json',exclude={'evidence_id'})
        if self.evidence_id!=research_hash(body):raise ValueError('Evidence identity mismatch')
        return self


class ExactAtom(StrictResearchModel):
    atom_id:str=Field(min_length=1)
    model_id:str=Field(min_length=1)
    quantity_id:str=Field(min_length=1)
    kind:Literal['absolute_utility','violation_probability','static_feasible','paired_difference']
    direction:Literal['le','ge']
    threshold:float=Field(ge=-1.0,le=1.0)


def issue_exact_evidence(*,scope:EvidenceScope,atom:ExactAtom,plan:dict,registration:dict,
                         input_ref:str,evaluator) -> EvidenceRecord:
    """Versioned pure evaluator injection; the lower layer knows no roles or protocol.

    The caller must verify input/source provenance and recompute this function when
    consuming a proof. A self-consistent hash alone is not an integrity certificate.
    """
    value=evaluator(atom)
    support=[-1.0,1.0] if atom.kind=='paired_difference' else [0.0,1.0]
    if type(value) is not float or not support[0]<=value<=support[1]:
        raise ValueError('Exact evaluator must supply a finite supported value')
    supported=value>=atom.threshold if atom.direction=='ge' else value<=atom.threshold
    body=dict(atom_id=atom.atom_id,scope=scope.model_dump(mode='json'),plan_hash=research_hash(plan),
        registration_receipt_hash=research_hash(registration),data_phase='audit',raw_run_refs=[input_ref],
        replication_ids=[],estimate=value,interval=[value,value],support=support,n=0,alpha=None,
        method='exact_analytic',failure_count=0,missing_count=0,validity='VALID',
        verdict='SUPPORTED' if supported else 'REFUTED',
        data_visibility={'origin':'analytic_fixture','atom':atom.model_dump(mode='json'),
                         'proof':'complete_finite_deterministic_input_v1','input_hash':registration['input_hash']})
    return EvidenceRecord(evidence_id=research_hash(body),**body)


QUANTITIES={
    'quote_coverage_v1':{'version':1,'support':[0.0,1.0],'definition':'fraction of evaluation-clock steps with actual two-sided end quotes'},
    'halt_fraction_v1':{'version':1,'support':[0.0,1.0],'definition':'fraction of evaluation-clock steps with market_open false'},
    'trading_step_fraction_v1':{'version':1,'support':[0.0,1.0],'definition':'fraction of evaluation-clock steps containing at least one settled trade'},
}


def evaluate_quantity(archive:Path,quantity_id:str):
    if quantity_id not in QUANTITIES:raise ValueError('UNREGISTERED_BOUNDED_QUANTITY')
    spec=ResearchRunSpec.model_validate_json((archive/'research_spec.json').read_text())
    records=[r for r in rows(archive/'measurements.jsonl') if spec.measurement.evaluation_start<=r['step']<spec.measurement.evaluation_end]
    expected=spec.measurement.evaluation_end-spec.measurement.evaluation_start
    if len(records)!=expected:raise ValueError('QUANTITY_CLOCK_INCOMPLETE')
    if quantity_id=='quote_coverage_v1':value=sum(r['quote_valid'] for r in records)/expected
    elif quantity_id=='halt_fraction_v1':value=sum(not r['market_open'] for r in records)/expected
    else:
        trades=rows(archive/'trades.jsonl') if (archive/'trades.jsonl').exists() else []
        steps={r['step'] for r in trades};value=sum(r['step'] in steps for r in records)/expected
    return float(value)


def bounded_interval(values:list[float],atom:ConfirmationAtom):
    if not values or any(type(x) is not float or not math.isfinite(x) or not atom.support_lower<=x<=atom.support_upper for x in values):
        raise ValueError('EMPTY_OR_OUT_OF_SUPPORT_SAMPLE')
    n=len(values);estimate=math.fsum(values)/n
    h=math.sqrt(2*math.log(2/atom.alpha)/n) if atom.kind=='paired_difference' else math.sqrt(math.log(2/atom.alpha)/(2*n))
    lower=max(atom.support_lower,estimate-h);upper=min(atom.support_upper,estimate+h)
    if atom.direction=='le':verdict='SUPPORTED' if upper<=atom.threshold else 'REFUTED' if lower>atom.threshold else 'UNRESOLVED'
    else:verdict='SUPPORTED' if lower>=atom.threshold else 'REFUTED' if upper<atom.threshold else 'UNRESOLVED'
    return {'estimate':estimate,'interval':[lower,upper],'half_width_unclipped':h,'verdict':verdict}


def engineering_scope(specs:dict[str,ResearchRunSpec]):
    """Content-backed engineering descriptors, without inventing role utility."""
    from core.experiments.counterfactual import execution_environment
    from core.experiments.research_cli import runtime_implementation_manifest
    from core.interventions.compiler import compile_policy
    first=next(iter(specs.values()))
    common=first.model_dump(mode='json');common.pop('policy');common['random'].pop('replication_id')
    policies=[]
    for spec in specs.values():
        check=spec.model_dump(mode='json');check.pop('policy');check['random'].pop('replication_id')
        if check!=common:raise ValueError('Engineering scope adapter supports one structural model with multiple policies')
        policies.append(compile_policy(spec.policy,constraints=spec.policy_constraints,world=spec.world,market=spec.base.market).policy_hash)
    descriptors={'task':{'version':'engineering-pipeline-v1','purpose':'execution_and_inference_validation',
        'scientific_policy_efficacy':False,'quantities':QUANTITIES},
        'preference':{'version':'engineering-no-role-preferences-v1','role_utilities':None},
        'acceptance_rule':{'version':'fixed-batch-hoeffding-v1','missing':'refuse_incomplete_batch','direction':'per_registered_atom'},
        'model_family':{'version':'single-explicit-rule-model-v1','model_ids':sorted(specs),'market':first.base.market.model_dump(mode='json')},
        'backend':execution_environment(),'world_distribution':first.world.model_dump(mode='json'),
        'scenario':first.scenario.model_dump(mode='json'),'behavior':first.behavior.model_dump(mode='json'),
        'measurement':first.measurement.model_dump(mode='json'),
        'missing_rule':{'version':'measurement-v2','unquoted':'null_spread','no_trade':'null_vwap','failed_run':'refuse_confirmation'}}
    scope=EvidenceScope(**{k+'_hash':research_hash(v) for k,v in descriptors.items()},policy_hashes=sorted(set(policies)),
        source_hash=runtime_implementation_manifest(first).source_hash,timing_version='research-step-v1',
        horizon_steps=first.base.steps,random_coupling_hash=first.random.coupling_spec_hash)
    return scope,descriptors


def register_confirmation(request:ConfirmationRequest,*,models:dict[str,list[ResearchRunSpec]],reservation_port,output:Path):
    from core.experiments.counterfactual import with_replication,raw_cache_identity
    from core.interventions.compiler import compile_policy
    # First implementation is intentionally one structural model; replicates
    # cannot silently stand in for uncertainty over unimplemented model families.
    if len(models)!=1:raise ValueError('UNSUPPORTED_MODEL_FAMILY_ADAPTER')
    model_id,policies=next(iter(models.items()))
    scope,descriptors=engineering_scope({f'{model_id}/policy-{i}':s for i,s in enumerate(policies)})
    if scope!=request.scope:raise ValueError('CONFIRMATION_SCOPE_MISMATCH')
    if any(a.model_id!=model_id or a.quantity_id not in QUANTITIES for a in request.atoms):raise ValueError('UNBOUND_MODEL_OR_QUANTITY')
    if any(a.kind=='violation_probability' for a in request.atoms):raise ValueError('No registered binary violation evaluator in this engineering adapter')
    bindings=[]
    for replication_id in request.planned_replication_ids:
        for spec in policies:
            if spec.data_phase!='confirmation':raise ValueError('CONFIRMATION_REQUIRES_EXPLICIT_DATA_PHASE')
            effective=with_replication(spec,replication_id)
            policy=compile_policy(effective.policy,constraints=effective.policy_constraints,world=effective.world,market=effective.base.market)
            bindings.append({'model_id':model_id,'policy_hash':policy.policy_hash,'replication_id':replication_id,
                'run_key':raw_cache_identity(effective)['run_key'],'spec':effective.model_dump(mode='json')})
    if len({b['run_key'] for b in bindings})!=len(bindings):raise ValueError('Duplicate baseline/policy cannot increase n')
    body={'request':request.model_dump(mode='json'),'bindings':bindings,'descriptors':descriptors}
    plan_hash=research_hash(body)
    receipt=reservation_port.reserve_confirmation(plan_hash,plan_id=request.plan_id,
        replication_ids=list(request.planned_replication_ids),run_keys=[b['run_key'] for b in bindings],
        alpha=request.alpha_batch,alpha_episode=request.alpha_episode)
    # Registration publication is durable before any job can be issued. A crash
    # after reserving alpha can repeat this exact transaction but cannot refund it.
    registration={'plan_hash':plan_hash,'semantic_body':body,'reservation':receipt}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'registration.json'
    if path.exists() and strict_json_loads(path.read_text())!=registration:raise ValueError('Immutable registration conflict')
    if not path.exists():durable_json(path,registration)
    registered=ConfirmationPlan(**request.model_dump(mode='json'),registration_receipt_hash=research_hash(registration))
    plan_path=output/'registered_plan.json'
    if not plan_path.exists():durable_json(plan_path,registered.model_dump(mode='json'))
    elif strict_json_loads(plan_path.read_text())!=registered.model_dump(mode='json'):raise ValueError('Registered plan tampered')
    return registered,registration


def certify_confirmation(plan:ConfirmationPlan,registration:dict,batch:dict,*,reservation_port):
    from core.experiments.counterfactual import raw_cache_identity,validate_pair
    if research_hash(registration)!=plan.registration_receipt_hash or research_hash(registration['semantic_body'])!=registration['plan_hash']:
        raise ValueError('REGISTRATION_IDENTITY_MISMATCH')
    request=plan.model_dump(mode='json',exclude={'status','registration_receipt_hash'})
    if request!=registration['semantic_body']['request']:raise ValueError('REGISTERED_PLAN_CHANGED')
    state=reservation_port.snapshot();plan_hash=registration['plan_hash']
    receipt=state['transactions'].get('plan:'+plan_hash)
    if receipt!=registration['reservation']:raise ValueError('REGISTRATION_NOT_IN_LEDGER')
    if batch.get('status')!='COMMITTED' or batch.get('visibility')!='whole_batch_barrier':raise ValueError('INCOMPLETE_CONFIRMATION_BATCH')
    bindings=registration['semantic_body']['bindings'];expected={b['run_key']:b for b in bindings}
    results=batch['results'];keys=[r['run_key'] for r in results]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('MISSING_DUPLICATED_OR_REPLACED_CONFIRMATION_RUN')
    archives={};refs={}
    for row in results:
        key=row['run_key'];binding=expected[key];archive=Path(row['archive'])
        if row.get('cache_hit') or key not in state['visible_runs']:raise ValueError('FRESHNESS_OR_VISIBILITY_VIOLATION')
        run_reservation=state['transactions'].get('run:'+key)
        if not run_reservation or run_reservation['sequence']<=receipt['sequence'] or run_reservation['details']['plan_hash']!=plan_hash:
            raise ValueError('RUN_NOT_ISSUED_AFTER_REGISTRATION')
        verify_archive_v3(archive)
        spec=ResearchRunSpec.model_validate_json((archive/'research_spec.json').read_text())
        if spec.model_dump(mode='json')!=binding['spec'] or raw_cache_identity(spec)['run_key']!=key:
            raise ValueError('CONFIRMATION_ARCHIVE_SCOPE_MISMATCH')
        if strict_json_loads((archive/'archive_manifest.json').read_text())['source_hash']!=plan.scope.source_hash:
            raise ValueError('CONFIRMATION_SOURCE_MISMATCH')
        archives[(binding['replication_id'],binding['model_id'],binding['policy_hash'])]=archive
        refs[str(archive)]=file_sha256(archive/'archive_manifest.json')
    records=[]
    for atom in plan.atoms:
        values=[];raw=[]
        for replication_id in plan.planned_replication_ids:
            roots=[archives[(replication_id,atom.model_id,p)] for p in atom.policy_hashes]
            if len(roots)==2:validate_pair(*roots)
            quantities=[evaluate_quantity(root,atom.quantity_id) for root in roots]
            values.append(quantities[1]-quantities[0] if len(quantities)==2 else quantities[0]);raw.extend(map(str,roots))
        stats=bounded_interval(values,atom)
        body=dict(atom_id=atom.atom_id,scope=plan.scope.model_dump(mode='json'),plan_hash=plan_hash,
            registration_receipt_hash=plan.registration_receipt_hash,data_phase='confirmation',raw_run_refs=raw,
            replication_ids=list(plan.planned_replication_ids),estimate=stats['estimate'],interval=stats['interval'],
            support=[atom.support_lower,atom.support_upper],n=len(values),alpha=atom.alpha,method=plan.method,
            failure_count=0,missing_count=0,validity='VALID',verdict=stats['verdict'],
            data_visibility={'registration_sequence':receipt['sequence'],'batch_barrier':True,'archive_hashes':refs,
                             'purpose':'engineering_pipeline_only','financial_eligibility':'NOT_EVALUATED'})
        records.append(EvidenceRecord(evidence_id=research_hash(body),**body))
    return records


def issue_evidence(plan,registration,batch,*,reservation_port):
    return certify_confirmation(plan,registration,batch,reservation_port=reservation_port)


def verify_evidence(record:EvidenceRecord,*,plan,registration,batch,reservation_port):
    checked=certify_confirmation(plan,registration,batch,reservation_port=reservation_port)
    if record not in checked:raise ValueError('EVIDENCE_NOT_REPRODUCIBLE_FROM_REGISTERED_BATCH')
    return {'status':'verified','evidence_id':record.evidence_id,'verdict':record.verdict,'scope':record.scope.model_dump(mode='json')}
