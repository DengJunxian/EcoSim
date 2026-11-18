"""One utility and eligibility boundary for all design methods and owner decisions."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal
from pydantic import Field

from core.experiments.strict_types import StrictResearchModel,strict_json_loads
from core.experiments.identity import research_hash, file_sha256
from core.experiments.measurement import StepMeasurement
from core.experiments.evidence import EvidenceScope, ExactAtom, EvidenceRecord, issue_exact_evidence
from core.experiments.archive import durable_json
from core.interventions.compiler import compile_policy, PolicyCompileError
from .schema import TaskSpec, PolicyCandidate, AssentReceipt, LossVector


class UtilityInput(StrictResearchModel):
    execution_status: Literal['completed', 'failed', 'incomplete']
    records: list[StepMeasurement]
    hidden_f_by_step: dict[str, float]
    cumulative_buy_cash: float = Field(ge=0.0)
    raw_refs: list[str] = Field(min_length=1)


def role_utilities(task, losses):
    losses=LossVector.model_validate(losses)
    return {r.role_id:float(1.0-math.fsum(getattr(r.weights,k)*v for k,v in losses.model_dump().items())) for r in task.roles}


def evaluate_task_utility(task:TaskSpec,run_metrics:UtilityInput) -> dict:
    """Pure, versioned task aggregation; raw v2 measurements are never altered."""
    task=TaskSpec.model_validate(task); data=UtilityInput.model_validate(run_metrics)
    if data.execution_status!='completed': raise ValueError('FAILED_RUN_HAS_NO_UTILITY')
    expected=list(range(task.measurement.evaluation_start,task.measurement.evaluation_end))
    records=[r for r in data.records if r.step in expected]
    if [r.step for r in records]!=expected: raise ValueError('UTILITY_CLOCK_INCOMPLETE')
    if len({(r.world_id,r.run_id,r.policy_hash) for r in data.records})!=1:
        raise ValueError('MIXED_RUN_UTILITY_INPUT')
    components={k:[] for k in ('L_Q','L_S','L_D','L_E')};clips={k:0 for k in components}
    valid_domains={k:[] for k in components}
    need_f=any(r.weights.L_E>0.0 for r in task.roles)
    for row in records:
        if row.measurement_hash!=research_hash(task.measurement) or not row.observation_complete or row.execution_status!='completed' or not row.snapshot_valid:
            raise ValueError('UTILITY_MEASUREMENT_INVALID_OR_INCOMPLETE')
        available=row.market_open and row.quote_valid
        components['L_Q'].append(0.0 if available else 1.0);valid_domains['L_Q'].append(row.step)
        if not available:
            components['L_S'].append(1.0);components['L_D'].append(1.0)
        else:
            spread=row.metrics[task.utility.spread_metric]
            db,da=row.metrics['depth_bid_top_k'],row.metrics['depth_ask_top_k']
            if spread.status!='VALID' or db.status!='VALID' or da.status!='VALID':
                raise ValueError('MISSING_REQUIRED_SERVICE_MEASUREMENT')
            s=spread.value/task.utility.s_cap; d=min(db.value,da.value)/task.utility.q_ref
            components['L_S'].append(min(1.0,s));components['L_D'].append(max(0.0,1.0-min(d,1.0)))
            clips['L_S']+=int(s>1.0);clips['L_D']+=int(d>1.0)
            valid_domains['L_S'].append(row.step);valid_domains['L_D'].append(row.step)
        price=row.metrics['price_proxy']
        f=data.hidden_f_by_step.get(str(row.step))
        if need_f and (f is None or f<=0.0): raise ValueError('MISSING_REQUIRED_HIDDEN_F')
        if f is not None and f<=0.0: raise ValueError('NONPOSITIVE_HIDDEN_F')
        proxy_age=({'current_mid':None if row.quote_as_of_step is None else row.step-row.quote_as_of_step,
                    'last_trade':row.price_age_steps,'history_reference':row.history_reference_age_steps}[task.measurement.price_proxy])
        if price.status!='VALID' or proxy_age is None or proxy_age>task.measurement.max_price_age_steps:
            components['L_E'].append(1.0)
        elif f is None:
            # Unused component is explicitly unavailable, not a substitute target.
            components['L_E'].append(1.0)
        else:
            error=abs(math.log(price.value/f))/task.utility.e_cap
            components['L_E'].append(min(1.0,error));clips['L_E']+=int(error>1.0)
            valid_domains['L_E'].append(row.step)
    losses={k:float(math.fsum(v)/len(records)) for k,v in components.items()}
    losses['L_B']=min(1.0,data.cumulative_buy_cash/task.utility.B_task)
    return {'evaluator_version':task.utility.version,'losses':losses,'utilities':role_utilities(task,losses),
        'valid_domain_steps':valid_domains,'clip_fractions':{**{k:v/len(records) for k,v in clips.items()},
        'L_B':float(data.cumulative_buy_cash>task.utility.B_task)},'raw_refs':list(data.raw_refs),
        'task_hash':task.task_hash,'input_hash':research_hash(data),'clock_denominator':len(records),
        'L_E_applicable':need_f or bool(data.hidden_f_by_step)}


def compile_candidate(task,candidate):
    return compile_policy(candidate.policy,constraints=task.static_constraints,world=task.compiler_world,market=task.compiler_market)


def scope_for(task,candidate,source_hash):
    return EvidenceScope(task_hash=task.task_hash,policy_hashes=[compile_candidate(task,candidate).policy_hash],
        model_family_hash=research_hash(task.model_family),source_hash=source_hash,
        backend_hash=research_hash({'origin':task.origin,'compiler':task.candidate_language}),
        world_distribution_hash=research_hash([m.world_distribution for m in task.model_family]),
        scenario_hash=research_hash([m.scenario_distribution for m in task.model_family]),
        behavior_hash=research_hash([m.behavior for m in task.model_family]),timing_version='research-step-v1',
        horizon_steps=task.horizon_steps,measurement_hash=research_hash(task.measurement),
        missing_rule_hash=research_hash({'measurement':task.measurement.missing_policy,'utility':task.utility.missing_rule}),
        random_coupling_hash=research_hash({'kind':'no_random_samples' if task.origin=='analytic_fixture' else 'registered_fresh_plan'}),
        preference_hash=task.preference_hash,acceptance_rule_hash=task.acceptance_rule_hash)


def required_atoms(task):
    atoms=[]
    for model in task.model_family:
        for role in task.roles:
            atoms.append(ExactAtom(atom_id=f'{model.model_id}:utility:{role.role_id}',model_id=model.model_id,
                quantity_id=role.role_id,kind='absolute_utility',direction='ge',threshold=role.rho))
        for constraint in task.outcome_constraints:
            atoms.append(ExactAtom(atom_id=f'{model.model_id}:constraint:{constraint.constraint_id}',model_id=model.model_id,
                quantity_id=constraint.constraint_id,kind='violation_probability',direction='le',threshold=constraint.upper_bound))
    return atoms


class ExactEvidenceService:
    """Register immutable finite input before evaluation; verify by recomputation."""
    def __init__(self,task:TaskSpec,source_hash:str,output:Path):
        if task.evidence_mode!='exact_analytic' or task.origin!='analytic_fixture':raise ValueError('ANALYTIC_SCOPE_REQUIRED')
        self.task,self.source_hash,self.output=task,source_hash,Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.input_path=self.output/'finite_problem.json'
        durable_json(self.input_path,task.model_dump(mode='json'))
        self.plan={'kind':'exact-finite-v1','task_hash':task.task_hash,'source_hash':source_hash,
                   'atoms':[a.model_dump(mode='json') for a in required_atoms(task)],
                   'additional_exact_family':{'static':'compiler_feasibility_ge_one',
                       'pairs':'all_ordered_distinct_legal_pool_candidates','quantities':[r.role_id for r in task.roles],
                       'models':[m.model_id for m in task.model_family],'difference_thresholds':[0.0],'directions':['ge','le']}}
        self.registration={'input_hash':task.task_hash,'input_sha256':file_sha256(self.input_path),
            'plan_hash':research_hash(self.plan),'registered_before_evaluation':True}
        durable_json(self.output/'registration.json',self.registration)
        self.issued={}

    def _evaluate(self,candidate,atom):
        outcome=next(a for a in self.task.analytic_outcomes if a.candidate_id==candidate.candidate_id and a.model_id==atom.model_id)
        return (role_utilities(self.task,outcome.losses)[atom.quantity_id] if atom.kind=='absolute_utility'
                else outcome.violations[atom.quantity_id])

    def _record(self,candidate,atom):
        return issue_exact_evidence(scope=scope_for(self.task,candidate,self.source_hash),atom=atom,
            plan=self.plan,registration=self.registration,input_ref=str(self.input_path.resolve()),
            evaluator=lambda a:self._evaluate(candidate,a))

    def issue(self,candidate):
        compile_candidate(self.task,candidate)
        result=[self._record(candidate,a) for a in required_atoms(self.task)]
        for record in result:
            path=self.output/(record.evidence_id+'.json')
            if not path.exists():durable_json(path,record.model_dump(mode='json'))
            self.issued[record.evidence_id]=path
        return result

    def verify(self,record,candidate):
        try:
            record=EvidenceRecord.model_validate(record)
            if file_sha256(self.input_path)!=self.registration['input_sha256']:return False
            if strict_json_loads((self.output/'registration.json').read_text())!=self.registration:return False
            path=self.issued.get(record.evidence_id)
            if path is None or EvidenceRecord.model_validate_json(path.read_text())!=record:return False
            if record.data_visibility.get('declared_claim') is not None:
                from .schema import ClaimSpec
                claim=ClaimSpec.model_validate(record.data_visibility['declared_claim'])
                if compile_candidate(self.task,candidate).policy_hash!=claim.policy_hashes[-1]:return False
                return self._record_claim(claim)==record
            atom=next(a for a in required_atoms(self.task) if a.atom_id==record.atom_id)
            return self._record(candidate,atom)==record
        except (ValueError,KeyError,StopIteration,OSError):return False

    def validate_additional_claim(self,claim):
        candidates=[]
        for ph in claim.policy_hashes:
            found=None
            for c in self.task.candidate_pool:
                try:
                    if compile_candidate(self.task,c).policy_hash==ph:found=c;break
                except PolicyCompileError:continue
            if found is None:raise ValueError('UNKNOWN_CLAIM_POLICY')
            candidates.append(found)
        expected=scope_for(self.task,candidates[0],self.source_hash).model_dump(mode='json')
        expected['policy_hashes']=list(claim.policy_hashes)
        if claim.scope!=EvidenceScope.model_validate(expected) or claim.registered_plan_hash!=research_hash(self.plan):
            raise ValueError('ADDITIONAL_CLAIM_SCOPE_MISMATCH')
        if claim.evidence_mode!='exact_analytic':raise ValueError('EXACT_CLAIM_MODE_REQUIRED')
        if claim.kind=='paired_difference':
            if (len(candidates)!=2 or claim.model_id not in [m.model_id for m in self.task.model_family]
                or claim.quantity_id not in [r.role_id for r in self.task.roles] or claim.threshold!=0.0):
                raise ValueError('UNREGISTERED_PAIR_PREDICATE')
        elif claim.kind=='static_feasible':
            if (claim.model_id,claim.quantity_id,claim.direction,claim.threshold)!=('all','static_feasible','ge',1.0):
                raise ValueError('UNREGISTERED_STATIC_PREDICATE')
        else:raise ValueError('NOT_AN_ADDITIONAL_EXACT_PREDICATE')
        return candidates

    def _record_claim(self,claim):
        candidates=self.validate_additional_claim(claim)
        atom=ExactAtom(atom_id=claim.claim_id,model_id=claim.model_id,quantity_id=claim.quantity_id,
                       kind=claim.kind,direction=claim.direction,threshold=claim.threshold)
        if claim.kind=='static_feasible':value=1.0
        else:
            values=[]
            for c in candidates:
                o=next(r for r in self.task.analytic_outcomes if r.candidate_id==c.candidate_id and r.model_id==claim.model_id)
                values.append(role_utilities(self.task,o.losses)[claim.quantity_id])
            value=float(values[1]-values[0])
        record=issue_exact_evidence(scope=claim.scope,atom=atom,plan=self.plan,registration=self.registration,
            input_ref=str(self.input_path.resolve()),evaluator=lambda _:value)
        body=record.model_dump(mode='json',exclude={'evidence_id'});body['data_visibility']['declared_claim']=claim.model_dump(mode='json')
        return EvidenceRecord(evidence_id=research_hash(body),**body)

    def issue_claim(self,claim):
        record=self._record_claim(claim);path=self.output/(record.evidence_id+'.json')
        if not path.exists():durable_json(path,record.model_dump(mode='json'))
        self.issued[record.evidence_id]=path;return record


def evaluate_eligibility(task:TaskSpec,candidate:PolicyCandidate,evidence:list[EvidenceRecord],*,evidence_service) -> dict:
    """Integrity and exact scope precede all interval and preference comparisons."""
    result={'candidate_id':candidate.candidate_id,'policy_hash':None,'status':'UNRESOLVED','reasons':[], 'roles':{},'constraints':{}}
    if candidate not in task.candidate_pool:
        return {**result,'status':'INAPPLICABLE','reasons':['CANDIDATE_OUTSIDE_REGISTERED_POOL']}
    try: compiled=compile_candidate(task,candidate)
    except PolicyCompileError as exc:return {**result,'status':'REFUTED','reasons':['STATIC_ILLEGAL'], 'compile_diagnostic':exc.diagnostic}
    result['policy_hash']=compiled.policy_hash
    if task.origin=='market' and task.evidence_mode=='diagnostic_only':
        return {**result,'status':'INAPPLICABLE','reasons':['NO_SCOPED_FINANCIAL_CERTIFICATE']}
    expected=scope_for(task,candidate,evidence_service.source_hash)
    by_atom={}
    for record in evidence:
        if record.atom_id in by_atom or record.scope!=expected or not evidence_service.verify(record,candidate):
            return {**result,'status':'INVALID_ARTIFACT','reasons':['EVIDENCE_INTEGRITY_OR_SCOPE_MISMATCH']}
        by_atom[record.atom_id]=record
    required=required_atoms(task)
    if set(by_atom)-{a.atom_id for a in required}:
        return {**result,'status':'INVALID_ARTIFACT','reasons':['UNREGISTERED_ATOM']}
    statuses=[]
    for atom in required:
        record=by_atom.get(atom.atom_id)
        status='UNRESOLVED'
        if record is not None:
            if record.validity!='VALID':status='INVALID_ARTIFACT'
            elif record.verdict=='INAPPLICABLE':status='INAPPLICABLE'
            elif record.interval is not None:
                lo,hi=record.interval
                status=('SUPPORTED' if lo>=atom.threshold else 'REFUTED' if hi<atom.threshold else 'UNRESOLVED') if atom.direction=='ge' else (
                    'SUPPORTED' if hi<=atom.threshold else 'REFUTED' if lo>atom.threshold else 'UNRESOLVED')
        statuses.append(status)
        target=result['roles'] if atom.kind=='absolute_utility' else result['constraints']
        target[atom.atom_id]={'status':status,'interval':None if record is None else record.interval,'threshold':atom.threshold,'direction':atom.direction}
    result['status']=next((s for s in ['INVALID_ARTIFACT','INAPPLICABLE','REFUTED','UNRESOLVED'] if s in statuses),'SUPPORTED')
    return result


def certify_agreement(task,candidate,evidence,assents,*,evidence_service,assent_service):
    eligibility=evaluate_eligibility(task,candidate,evidence,evidence_service=evidence_service)
    status='NOT_ELIGIBLE' if eligibility['status']!='SUPPORTED' else 'CERTIFIED'
    owners={}
    for receipt in assents:
        if (receipt.owner in owners or receipt.policy_hash!=eligibility['policy_hash'] or receipt.task_hash!=task.task_hash
            or receipt.preference_hash!=task.preference_hash or receipt.acceptance_rule_hash!=task.acceptance_rule_hash
            or not assent_service.verify(receipt,candidate,evidence)):
            status='INVALID_ASSENT';break
        owners[receipt.owner]=receipt
    if status=='CERTIFIED' and any(r not in owners or owners[r].decision!='ASSENT' for r in task.required_signers):status='NO_ASSENT'
    return {'status':status,'eligibility':eligibility,'owner_receipts':{k:v.receipt_id for k,v in owners.items()}}


def select_candidate(task,eligible,evidence_by_candidate):
    def key(candidate):
        compiled=compile_candidate(task,candidate)
        by_atom={r.atom_id:r for r in evidence_by_candidate[candidate.candidate_id]}
        margin=min((by_atom[f'{m.model_id}:utility:{r.role_id}'].interval[0]-r.rho)/(1.0-r.rho) for m in task.model_family for r in task.roles)
        cost=compiled.deterministic_bounds['cash_authorization']
        complexity=len(compiled.allowed_actions)
        return (-margin,cost,complexity,compiled.policy_hash)
    return min(eligible,key=key) if eligible else None


def role_eligibility(task,candidate,owner,evidence,*,evidence_service):
    """A role's own absolute rule plus all outcome constraints, not other roles' rho."""
    result=evaluate_eligibility(task,candidate,evidence,evidence_service=evidence_service)
    if owner not in task.required_signers:raise ValueError('UNKNOWN_REQUIRED_OWNER')
    if result['status'] in {'INVALID_ARTIFACT','INAPPLICABLE'} or result['reasons']:
        return {'owner':owner,'status':result['status'],'reasons':result['reasons']}
    needed=[v['status'] for k,v in result['roles'].items() if k.endswith(':utility:'+owner)]+[v['status'] for v in result['constraints'].values()]
    status=next((s for s in ['INVALID_ARTIFACT','INAPPLICABLE','REFUTED','UNRESOLVED'] if s in needed),'SUPPORTED') if needed else 'UNRESOLVED'
    return {'owner':owner,'status':status,'reasons':[], 'policy_hash':result['policy_hash']}


def robust_interval_envelope(intervals):
    if not intervals or any(v is None or len(v)!=2 or not 0.0<=v[0]<=v[1]<=1.0 for v in intervals):
        raise ValueError('COMPLETE_VALID_MODEL_INTERVALS_REQUIRED')
    return [min(v[0] for v in intervals),min(v[1] for v in intervals)]


def credible_shortlist(task,evidence_by_candidate,*,evidence_service):
    """Current-evidence dominance using [min-model L, min-model U]."""
    rows={};envelopes={};excluded={};role_sets={r.role_id:[] for r in task.roles}
    for c in task.candidate_pool:
        try:ph=compile_candidate(task,c).policy_hash
        except ValueError:continue
        records=evidence_by_candidate.get(c.candidate_id,[])
        e=evaluate_eligibility(task,c,records,evidence_service=evidence_service)
        rows[ph]={'candidate_id':c.candidate_id,'status':e['status'],'reason':'current_evidence_not_dominated'}
        for r in task.roles:
            if role_eligibility(task,c,r.role_id,records,evidence_service=evidence_service)['status']=='SUPPORTED':role_sets[r.role_id].append(ph)
        if e['status'] in {'INVALID_ARTIFACT','INAPPLICABLE'}:continue
        if any(v['status']=='REFUTED' for v in e['constraints'].values()):
            excluded[ph]='outcome_constraint_refuted';continue
        if any(v['status']!='SUPPORTED' for v in e['constraints'].values()):continue
        if any(v['interval'] is None for v in e['roles'].values()):continue
        envelopes[ph]={r.role_id:robust_interval_envelope([e['roles'][f'{m.model_id}:utility:{r.role_id}']['interval']
            for m in task.model_family]) for r in task.roles}
    dominated={}
    for q,qenv in envelopes.items():
        dominators=[p for p,penv in envelopes.items() if p!=q and all(penv[r][0]>=qenv[r][1] for r in qenv)
            and any(penv[r][0]>qenv[r][1] for r in qenv)]
        if dominators:dominated[q]=sorted(dominators)
    return {'scope':'current_evidence_finite_pool','nondominated_hashes':sorted(set(rows)-set(dominated)-set(excluded)),
        'excluded_as_infeasible':excluded,
        'credible_dominators':dominated,'robust_envelopes':envelopes,'role_acceptance_sets':role_sets,
        'candidates':rows,'claim':'not a true or global Pareto frontier'}
