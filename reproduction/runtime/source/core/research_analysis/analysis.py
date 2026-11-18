"""F72: task-paired pilot planning, without formal freeze or effect selection.

Joint Bernoulli task vectors preserve method pairing. Stratified multinomial
simulation propagates all four CDR cells together; messages are not samples.
"""
from __future__ import annotations
import itertools
import math
import shutil
import importlib.metadata
from pathlib import Path
from typing import Literal
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import beta, norm, t
from pydantic import Field, model_validator
from core.experiments.strict_types import StrictResearchModel, strict_json_loads, load_research_model
from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256, research_hash
from core.codesign.pilot import load_pilot_plan, load_pilot_inputs, PilotPlanV2, PilotPlanV3, METHODS
from core.codesign.runner import verify_episode
from core.codesign.schema import TaskSpec

JOINT_METHODS=['cdr_s0_v0','cdr_s0_v1','cdr_s1_v0','cdr_s1_v1','single']
PATTERNS=np.array(list(itertools.product([0.,1.],repeat=5)))

class PlanningContrast(StrictResearchModel):
    id: str
    rq: Literal['RQ2','RQ3']
    weights: dict[str,float]
    delta: float = Field(gt=0,lt=1)

class PlanningDesign(StrictResearchModel):
    schema_version: Literal[1]
    design_id: str
    pilot_plan_id: str
    analysis_population: str
    independent_unit: Literal['task_table']
    strata: Literal[3]
    design_repeats_per_method_task: Literal[1]
    market_repeats_per_policy: Literal[0]
    family_alpha: float = Field(gt=0,lt=1)
    target_power: float = Field(gt=0.5,lt=1)
    target_half_width: float = Field(gt=0,lt=1)
    equivalence_margin: float = Field(gt=0,lt=1)
    contrasts: list[PlanningContrast] = Field(min_length=1)
    candidate_task_counts: list[int] = Field(min_length=1)
    simulation_draws: int = Field(ge=1000)
    simulation_seed: int = Field(ge=0)
    bootstrap_draws: int = Field(ge=1000)
    joint_pattern_pseudocounts: list[float] = Field(min_length=1)
    delta_sensitivity: list[float] = Field(min_length=1)
    continuous_cost_delta_messages: float = Field(gt=0)
    continuous_variance_inflation: float = Field(ge=1)
    zero_event_rate_target: float = Field(gt=0,lt=1)
    event_confidence: float = Field(gt=0.5,lt=1)
    missing_trial_rule: str
    planning_test: str
    precision_rule: str
    effect_rationale: str
    precision_rationale: str
    cost_rationale: str
    sensitivity_rationale: str
    formal_execution_authorized_by_this_file: Literal[False]
    financial_status: Literal['BLOCKED_NO_HEALTH_FREEZE']

    @model_validator(mode='after')
    def valid(self):
        if self.candidate_task_counts!=sorted(set(self.candidate_task_counts)) or any(n<12 or n%3 for n in self.candidate_task_counts):
            raise ValueError('BALANCED_INDEPENDENT_TASK_GRID_REQUIRED')
        if len({c.id for c in self.contrasts})!=len(self.contrasts):raise ValueError('DUPLICATE_CONTRAST')
        for c in self.contrasts:
            if not set(c.weights)<=set(JOINT_METHODS) or abs(sum(c.weights.values()))>1e-12 or not any(c.weights.values()):
                raise ValueError('INVALID_PAIRED_CONTRAST')
        if any(p<=0 for p in self.joint_pattern_pseudocounts):raise ValueError('POSITIVE_SPARSE_SENSITIVITY_REQUIRED')
        if any(d<=0 or d>=1 for d in self.delta_sensitivity):raise ValueError('INVALID_EFFECT_SENSITIVITY')
        return self


def wilson_interval(successes,n,confidence=.95):
    z=norm.ppf((1+confidence)/2);p=successes/n;denom=1+z*z/n
    mid=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.,float(mid-half)),min(1.,float(mid+half))]


def event_upper_bound(events,n,confidence=.95):
    if n<1:return None
    if not 0<=events<=n:raise ValueError('INVALID_EVENT_COUNT')
    return 1. if events==n else float(beta.ppf(confidence,events+1,n-events))


def balanced_size(value,strata=3):return max(12,strata*math.ceil(value/strata))


def joint_probabilities(matrix,strata,pseudocount):
    """One row per independent task, not one row per message or method."""
    matrix=np.asarray(matrix);strata=np.asarray(strata)
    if matrix.ndim!=2 or matrix.shape[1]!=5 or not np.isin(matrix,[0,1]).all():raise ValueError('BINARY_TASK_VECTORS_REQUIRED')
    if set(strata)!=set(range(3)) or len(strata)!=len(matrix):raise ValueError('THREE_STRATA_REQUIRED')
    index=(matrix @ (2**np.arange(4,-1,-1))).astype(int)
    return np.array([(np.bincount(index[strata==s],minlength=32)+pseudocount/32)/(sum(strata==s)+pseudocount) for s in range(3)])


def tilt_joint(probabilities,values,target):
    """A common exponential tilt fixes the declared mean effect, not pilot effect."""
    logp=np.log(probabilities)
    def distribution(lam):
        x=logp+lam*values;return np.exp(x-logsumexp(x,axis=1)[:,None])
    lam=brentq(lambda q:float((distribution(q)@values).mean()-target),-100,100)
    answer=distribution(lam)
    if not np.isclose((answer@values).mean(),target,atol=1e-10):raise ValueError('TILT_TARGET_MISMATCH')
    return answer


def paired_interval(matrix,strata,weights,alpha=.05):
    values=np.asarray(matrix)@np.asarray(weights);strata=np.asarray(strata)
    groups=[values[strata==s] for s in range(3)]
    if any(len(g)<2 for g in groups):raise ValueError('NEED_TWO_TASKS_PER_STRATUM')
    components=np.array([np.var(g,ddof=1)/len(g)/9 for g in groups]);v=float(sum(components))
    denominator=sum(a*a/(len(g)-1) for a,g in zip(components,groups))
    mean=float(np.mean([np.mean(g) for g in groups]))
    half=None if v==0 else float(t.ppf(1-alpha/2,v*v/denominator)*math.sqrt(v))
    return {'estimate':mean,'within_stratum_variance':float(np.mean([np.var(g,ddof=1) for g in groups])),
        'standard_error':math.sqrt(v),'approximate_interval':None if half is None else [mean-half,mean+half],
        'zero_variance_caution':v==0,'independent_tasks':len(values)}


def simulate_contrast(probabilities,values,n,alpha,draws,rng):
    if n%3 or n<6:raise ValueError('BALANCED_TASK_COUNT_REQUIRED')
    per=n//3
    counts=np.stack([rng.multinomial(per,p,size=draws) for p in probabilities],axis=1)
    means=(counts@values)/per
    variances=np.maximum(0,(counts@(values*values)-per*means*means)/(per-1))
    components=variances/(per*9);variance=components.sum(axis=1)
    denominator=(components*components/(per-1)).sum(axis=1)
    df=np.divide(variance*variance,denominator,out=np.full(draws,np.inf),where=denominator>0)
    half=t.ppf(1-alpha/2,df)*np.sqrt(variance)
    # Degenerate draws do not create free statistical certainty.
    half[variance==0]=np.inf
    estimates=means.mean(axis=1)
    positive=estimates>half;two_sided=np.abs(estimates)>half
    count=int(positive.sum());null_count=int(two_sided.sum())
    ordered=np.sort(half)
    # The order-statistic quantile handles infinity without inf-minus-inf.
    median=ordered[math.ceil(.5*draws)-1];p90=ordered[math.ceil(.9*draws)-1]
    return {'task_count':n,'power':count/draws,'power_mc_95_interval':wilson_interval(count,draws),
        'two_sided_rejection_rate':null_count/draws,'two_sided_mc_95_interval':wilson_interval(null_count,draws),
        'half_width_median':float(median) if np.isfinite(median) else None,
        'half_width_p90':float(p90) if np.isfinite(p90) else None,
        'degenerate_fraction':float(np.mean(variance==0))}


def load_completed_pilot(index_path,design_path):
    path=Path(index_path).resolve();root=path.parent;index=strict_json_loads(path.read_text())
    marker=strict_json_loads((root/'COMPLETED.json').read_text());reg=strict_json_loads((root/'registration.json').read_text())
    if marker['index_sha256']!=file_sha256(path) or marker['registration_sha256']!=file_sha256(root/'registration.json'):
        raise ValueError('PILOT_COMPLETION_HASH_MISMATCH')
    if not marker['complete_planned_index'] or index['study_stage']!='pilot' or index['stopping_reason']:
        raise ValueError('COMPLETED_INDEPENDENT_PILOT_REQUIRED')
    plan=load_pilot_plan(root/'plan.json');tasks,check=load_pilot_inputs(plan)
    if not isinstance(plan,PilotPlanV2) or plan.plan_id!=load_research_model(design_path,PlanningDesign).pilot_plan_id:raise ValueError('PLANNING_PILOT_VERSION_MISMATCH')
    if file_sha256(design_path)!=plan.planning_design_sha256 or file_sha256(design_path)!=file_sha256(root/'planning_design.json'):
        # durable_json uses canonical indentation; compare semantic objects as well,
        # but the original externally supplied design must match its pre-call hash.
        if file_sha256(design_path)!=plan.planning_design_sha256 or research_hash(strict_json_loads(Path(design_path).read_text()))!=research_hash(strict_json_loads((root/'planning_design.json').read_text())):
            raise ValueError('DESIGN_NOT_BOUND_BEFORE_PILOT')
    if research_hash(plan)!=index['plan_hash'] or reg['plan_hash']!=index['plan_hash'] or reg['source_hash']!=index['source_hash']:
        raise ValueError('PILOT_REGISTRATION_MISMATCH')
    schedule=[(x['episode_id'],x['task_id'],x['method']) for x in reg['schedule']]
    actual=[(x['episode_id'],x['task_id'],x['method']) for x in index['episodes']]
    if schedule!=actual or len(set(actual))!=len(actual) or len(actual)!=len(tasks)*len(plan.methods):raise ValueError('PILOT_SCHEDULE_INCOMPLETE')
    results={};costs={};refusals={}
    for trial in index['episodes']:
        if trial['status']!='RECORDED':raise ValueError('UNRUN_TRIAL_BLOCKS_PLANNING')
        p=Path(trial['path']);verify_episode(p)
        if file_sha256(p/'episode_result.json')!=trial['result_sha256']:raise ValueError('PILOT_RESULT_HASH_MISMATCH')
        r=strict_json_loads((p/'episode_result.json').read_text())
        if (r['episode_id'],r['task_id'],r['method'])!=(trial['episode_id'],trial['task_id'],trial['method']):
            raise ValueError('TRIAL_RESULT_IDENTITY_MISMATCH')
        if r['source_hash']!=index['source_hash'] or r['split']!='development':raise ValueError('PILOT_SCOPE_MISMATCH')
        key=(trial['task_id'],trial['method']);results[key]=r
        calls=strict_json_loads((p/'llm/call_index.json').read_text()) if trial['method']!='central_rule' else []
        amounts=[c.get('spend_settlement',{}).get('peak_price_cost_upper_bound_nano') for c in calls]
        costs[key]=sum(amounts)/1e9 if all(a is not None for a in amounts) else None
        feedback=strict_json_loads((p/'protocol/operation-feedback.json').read_text()) if trial['method']!='central_rule' else {'receipts':[]}
        refusals[key]=sum(not x['applied'] for x in feedback['receipts'])
    return plan,tasks,index,results,costs,refusals,check


def plan_study(pilot_path,design_path,output):
    design=load_research_model(design_path,PlanningDesign)
    plan,tasks,index,results,costs,refusals,check=load_completed_pilot(pilot_path,design_path)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parents[2]
    snapshots=[]
    for name in ('core/research_analysis/analysis.py','scripts/research/prepare_study.py','requirements-cscwd.lock'):
        target=output/'planning_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_root/name,target);snapshots.append(target)
    durable_json(output/'planning_environment.json',{'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','pydantic')},
        'source_scope':'planning algorithm and CLI snapshots, plus pilot frozen codesign source bundle and dependency lock'})
    task_ids=[x.task_id for x in tasks]
    strata=np.array([int(x.prompt_resources['stratum_index']) if isinstance(plan,PilotPlanV3) else
        int(np.argmin(np.abs(np.array([.35,.55,.75])-np.mean([r.rho for r in x.roles])))) for x in tasks])
    if len(set(np.bincount(strata)))!=1:raise ValueError('UNEQUAL_PILOT_STRATA')
    matrix=np.array([[float(bool(results[(tid,m)]['A_cert'])) for m in JOINT_METHODS] for tid in task_ids])
    alpha=design.family_alpha/len(design.contrasts);rng=np.random.default_rng(design.simulation_seed)
    details=[];summaries=[]
    for contrast in design.contrasts:
        weights=np.array([contrast.weights.get(m,0.) for m in JOINT_METHODS]);values=PATTERNS@weights
        observed=paired_interval(matrix,strata,weights,alpha)
        # Whole task vectors are resampled together within each threshold stratum.
        boot=[]
        for s in range(3):
            block=(matrix@weights)[strata==s]
            boot.append(np.var(rng.choice(block,size=(design.bootstrap_draws,len(block)),replace=True),axis=1,ddof=1))
        observed['bootstrap_variance_p90']=float(np.quantile(np.mean(boot,axis=0),.9))
        scenarios=[]
        for pseudo in design.joint_pattern_pseudocounts:
            base=joint_probabilities(matrix,strata,pseudo)
            for delta in sorted(set(design.delta_sensitivity+[contrast.delta])):
                alternate=tilt_joint(base,values,delta);null=tilt_joint(base,values,0.)
                grid=[]
                for n in design.candidate_task_counts:
                    alt=simulate_contrast(alternate,values,n,alpha,design.simulation_draws,rng)
                    zero=simulate_contrast(null,values,n,alpha,design.simulation_draws,rng)
                    alt['null_type1_rate']=zero['two_sided_rejection_rate']
                    alt['null_type1_mc_95_interval']=zero['two_sided_mc_95_interval']
                    alt['null_half_width_p90']=zero['half_width_p90']
                    widths=[alt['half_width_p90'],zero['half_width_p90']]
                    alt['planning_precision_half_width_p90']=max(widths) if all(v is not None for v in widths) else None
                    # Reject a scenario if simulation shows alpha inflation beyond
                    # its Monte Carlo uncertainty at the proposed task count.
                    alt['null_calibration_ok']=zero['two_sided_mc_95_interval'][0]<=alpha
                    grid.append(alt)
                power=next((r['task_count'] for r in grid if r['power_mc_95_interval'][0]>=design.target_power and r['null_calibration_ok']),None)
                precision=next((r['task_count'] for r in grid if r['planning_precision_half_width_p90'] is not None and r['planning_precision_half_width_p90']<=design.target_half_width and r['null_calibration_ok']),None)
                scenarios.append({'pseudocount_per_stratum':pseudo,'target_delta':delta,'power_task_count':power,'precision_task_count':precision,'grid':grid,
                    'joint_base_probabilities':base.tolist(),'joint_alternative_probabilities':alternate.tolist()})
        primary=[s for s in scenarios if s['target_delta']==contrast.delta]
        pnums=[s['power_task_count'] for s in primary];hnums=[s['precision_task_count'] for s in primary]
        width=float(values.max()-values.min())
        bound=balanced_size(width*width*math.log(2*len(design.contrasts)/design.family_alpha)/(2*design.target_half_width**2))
        summary={'contrast':contrast.id,'rq':contrast.rq,'delta':contrast.delta,'alpha':alpha,**observed,
            'power_task_count_sensitivity_max':max(pnums) if all(n is not None for n in pnums) else None,
            'precision_task_count_sensitivity_max':max(hnums) if all(n is not None for n in hnums) else None,
            'distribution_free_precision_task_count':bound}
        summaries.append(summary);details.append({'summary':summary,'scenarios':scenarios})
        durable_json(output/(contrast.id+'-simulation.json'),details[-1])
        print(contrast.id+' planning complete',flush=True)
    messages=np.array([[results[(tid,m)]['budget_snapshot']['used']['messages'] for m in ['cdr_s1_v1','single']] for tid in task_ids])
    continuous=paired_interval(messages,strata,[1.,-1.],alpha)
    delta=design.continuous_cost_delta_messages
    variance=continuous['within_stratum_variance']*design.continuous_variance_inflation
    continuous.update(unit='protocol_messages',delta=delta,variance_inflation=design.continuous_variance_inflation,
        approximate_power_task_count=None if variance==0 else balanced_size(variance*(norm.ppf(1-alpha/2)+norm.ppf(design.target_power))**2/delta**2),
        approximate_precision_task_count=None if variance==0 else balanced_size(variance*norm.ppf(1-alpha/2)**2/delta**2))
    spend=index['spend_snapshot'];reserved=sum(x['reserved_nano'] for x in spend['reservations'].values())
    remaining=spend['contract']['cap_nano']-reserved
    per_task=(len(plan.methods)-1)*(plan.design_turns+3)*(plan.model.input_reservation_tokens*plan.input_price_nano+plan.model.max_output_tokens*plan.output_price_nano)
    cost_by_task=[None if any(costs[(tid,m)] is None for m in plan.methods) else sum(costs[(tid,m)] for m in plan.methods) for tid in task_ids]
    unknown_cost=any(c is None for c in cost_by_task)
    expected_task_cost=None if unknown_cost else float(np.mean(cost_by_task))
    high_task_cost=None if unknown_cost else float(np.mean(cost_by_task)+2*np.std(cost_by_task,ddof=1))
    recommendations=[max(s['power_task_count_sensitivity_max'],s['precision_task_count_sensitivity_max']) if s['power_task_count_sensitivity_max'] is not None and s['precision_task_count_sensitivity_max'] is not None else None for s in summaries]
    recommended=max(recommendations) if all(n is not None for n in recommendations) else None
    budget_grid=[{'tasks':n,'llm_design_repeats':1,'market_repeats':0,'live_calls':n*8*9,'episode_count':n*9,
        'maximum_design_messages':n*9*tasks[0].budget.messages,
        'maximum_candidate_attempts':n*9*tasks[0].budget.candidate_submissions,
        'maximum_input_token_reservation':n*8*9*plan.model.input_reservation_tokens,
        'maximum_output_token_reservation':n*8*9*plan.model.max_output_tokens,
        'unique_market_simulations':0,'analytic_audit_llm_calls':0,
        'maximum_additional_reservation_cny':n*per_task/1e9,'expected_usage_price_upper_cny':None if unknown_cost else n*expected_task_cost,
        'two_sd_cost_sensitivity_cny':None if unknown_cost else n*high_task_cost,'fits_current_remaining_authorization':n*per_task<=remaining}
        for n in design.candidate_task_counts]
    events={}
    for m in METHODS:
        r=[results[(tid,m)] for tid in task_ids];cert=sum(bool(x['A_cert']) for x in r);failed=sum(x['terminal_state']=='METHOD_FAILED' for x in r)
        rejected=sum(refusals[(tid,m)]>0 for tid in task_ids)
        events[m]={'tasks':len(tasks),'certified':cert,'completion_wilson_95':wilson_interval(cert,len(tasks)),
            'method_failed':failed,'failure_rate_upper_95':event_upper_bound(failed,len(tasks)),
            'episodes_with_operation_rejection':rejected if m!='central_rule' else None,
            'operation_rejections':sum(refusals[(tid,m)] for tid in task_ids) if m!='central_rule' else None,
            'evaluated_candidates':sum(len(x['common_contract']['exact_evaluated_candidate_ids']) for x in r),
            'calls':sum(x['common_contract']['live_calls'] for x in r)}
    blockers=['Formal PlanCore/AnalysisPlan, held-out AuditPlan and executable freeze are not implemented by F72.']
    if recommended is None:blockers.append('No task count in the registered grid met every power/precision/calibration criterion.')
    elif recommended*per_task>remaining:blockers.append('Recommended formal design exceeds remaining authorization under the existing no-refund full-cap reservation rule.')
    if unknown_cost or spend['blocked_reason']:blockers.append('Unknown or blocked usage requires reconciliation before any further live execution.')
    candidate={'schema_version':1,'study_stage':'planning','formal_status':'NOT_FROZEN','design_id':design.design_id,
        'pilot_plan_id':plan.plan_id,'pilot_independent_tasks':len(tasks),'source_hash':index['source_hash'],
        'primary_contrasts':summaries,'continuous_confirmation_communication_cost':continuous,
        'recommended_balanced_task_count':recommended,'recommended_llm_repeats_per_method_task':1,'recommended_market_repeats_per_policy':0,
        'method_diagnostics':events,'resource_grid':budget_grid,
        'cumulative_reserved_cny':reserved/1e9,'remaining_reservation_cny':remaining/1e9,
        'maximum_additional_complete_tasks':remaining//per_task,'maximum_additional_balanced_tasks':3*((remaining//per_task)//3),
        'observed_cost_upper_cny_per_complete_task':expected_task_cost,'cost_two_sd_sensitivity_per_task':high_task_cost,
        'pilot_cost_upper_cny':None if unknown_cost else sum(cost_by_task),
        'zero_event_upper_95_at_pilot_N':event_upper_bound(0,len(tasks),design.event_confidence),
        'zero_event_required_tasks_for_5pct_upper':math.ceil(math.log(1-design.event_confidence)/math.log(1-design.zero_event_rate_target)),
        'effect_rationale':design.effect_rationale,'precision_rationale':design.precision_rationale,
        'assumptions':[design.analysis_population,design.planning_test,design.sensitivity_rationale,
            'Per-method Wilson bands are descriptive binomial reference intervals; the paired planning simulation explicitly retains the three threshold strata.',
            'One trajectory per task/method estimates the compound task-plus-generation variance. It cannot identify within-task LLM repeat variance.',
            'Pilot v1 is preserved but excluded from v2 nuisance estimates because the feedback intervention changed.',
            'Simulation alpha calibration is a diagnostic, not a finite-sample guarantee; bounded Hoeffding precision is a separate conservative target.'],
        'blockers':blockers,'formal_execution_permitted':False}
    durable_json(output/'candidate_plan.json',candidate)
    durable_json(output/'task_vectors.json',{'task_ids':task_ids,'strata':strata.tolist(),'methods':JOINT_METHODS,'certification_matrix':matrix.tolist(),
        'cost_upper_cny_by_task':cost_by_task,'paired_message_counts_full_single':messages.tolist()})
    durable_json(output/'design.json',design.model_dump(mode='json'))
    inputs={str(Path(p).resolve()):file_sha256(p) for p in [pilot_path,design_path,Path(pilot_path).parent/'COMPLETED.json',Path(pilot_path).parent/'registration.json',*snapshots]}
    artifacts={str(p.resolve()):file_sha256(p) for p in sorted(output.glob('*.json'))}
    durable_json(output/'power_precision_receipt.json',{'schema_version':1,'status':'PLANNING_COMPLETED_FORMAL_BLOCKED',
        'input_hashes':inputs,'artifact_hashes':artifacts,'split_check':check,'simulation_draws_per_scenario_N':design.simulation_draws,
        'random_seed':design.simulation_seed,'family_alpha':design.family_alpha,'formal_freeze_created':False,
        'thresholds_bound_by_pre_call_plan':True,'recommended_task_count':recommended,'blockers':blockers})
    return candidate
