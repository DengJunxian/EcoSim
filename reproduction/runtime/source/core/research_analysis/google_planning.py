"""Google F72: same task-paired planning kernel, explicit USD resources.

This prospective provider branch does not change DeepSeek plans or cost ledgers.
"""
from pathlib import Path
import importlib.metadata
import math
import shutil
import numpy as np
from scipy.stats import norm
from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256, research_hash
from core.experiments.strict_types import strict_json_loads, load_research_model
from core.codesign.pilot import METHODS
from core.codesign.runner import verify_episode
from .study_freeze import FreezeGateError
from .analysis import (PlanningDesign, JOINT_METHODS, PATTERNS, paired_interval,
    joint_probabilities, tilt_joint, simulate_contrast, balanced_size,
    wilson_interval, event_upper_bound)


def load_completed_google_pilot(index_path, design_path):
    from .google_pilot import load_google_pilot, load_google_inputs
    path=Path(index_path).resolve();root=path.parent
    index=strict_json_loads(path.read_text())
    marker=strict_json_loads((root/'COMPLETED.json').read_text())
    reg=strict_json_loads((root/'registration.json').read_text())
    if marker['index_sha256']!=file_sha256(path) or marker['registration_sha256']!=file_sha256(root/'registration.json'):
        raise ValueError('PILOT_COMPLETION_HASH_MISMATCH')
    if not marker['complete_planned_index'] or index['study_stage']!='pilot' or index['stopping_reason']:
        raise FreezeGateError('COMPLETED_INDEPENDENT_GOOGLE_PILOT_REQUIRED')
    plan=load_google_pilot(root/'plan.json');tasks,check=load_google_inputs(plan)
    design=load_research_model(design_path,PlanningDesign)
    if plan.plan_id!=design.pilot_plan_id:raise ValueError('GOOGLE_PLANNING_PILOT_VERSION_MISMATCH')
    if file_sha256(design_path)!=plan.planning_design_sha256 or research_hash(design)!=research_hash(strict_json_loads((root/'planning_design.json').read_text())):
        raise ValueError('DESIGN_NOT_BOUND_BEFORE_PILOT')
    if research_hash(plan)!=index['plan_hash'] or reg['plan_hash']!=index['plan_hash'] or reg['source_hash']!=index['source_hash']:
        raise ValueError('PILOT_REGISTRATION_MISMATCH')
    scheduled=[(x['episode_id'],x['task_id'],x['method']) for x in reg['schedule']]
    actual=[(x['episode_id'],x['task_id'],x['method']) for x in index['episodes']]
    if scheduled!=actual or len(set(actual))!=len(actual) or len(actual)!=len(tasks)*len(plan.methods):
        raise ValueError('PILOT_SCHEDULE_INCOMPLETE')
    results={};costs={};refusals={}
    for trial in index['episodes']:
        if trial['status']!='RECORDED':raise FreezeGateError('UNRUN_TRIAL_BLOCKS_PLANNING')
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


def plan_google_study(pilot_path,design_path,output):
    design=load_research_model(design_path,PlanningDesign)
    plan,tasks,index,results,costs,refusals,check=load_completed_google_pilot(pilot_path,design_path)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parents[2]
    snapshots=[]
    for name in ('core/research_analysis/google_planning.py','core/research_analysis/analysis.py','requirements-cscwd.lock'):
        target=output/'planning_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_root/name,target);snapshots.append(target)
    durable_json(output/'planning_environment.json',{'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','pydantic')},
        'source_scope':'planning algorithm and CLI snapshots, plus pilot frozen codesign source bundle and dependency lock'})
    task_ids=[x.task_id for x in tasks]
    strata=np.array([int(x.prompt_resources['stratum_index']) for x in tasks])
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
    remaining=None  # User authorized no monetary cap; finite plan allocations stay separate.
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
        'maximum_additional_reservation_usd':n*per_task/1e9,'expected_usage_price_upper_usd':None if unknown_cost else n*expected_task_cost,
        'two_sd_cost_sensitivity_usd':None if unknown_cost else n*high_task_cost,'fits_monetary_authorization':True,'requires_finite_formal_call_allocation':True}
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
    blockers=['Formal PlanCore/AnalysisPlan, AuditPlan and finite call allocation must pass the separate F73 gate.']
    if recommended is None:blockers.append('No task count in the registered grid met every power/precision/calibration criterion.')
    if unknown_cost or spend['blocked_reason']:blockers.append('Unknown or blocked usage requires reconciliation before any further live execution.')
    candidate={'schema_version':2,'provider':'gemini','currency':'USD','monetary_cap':None,'study_stage':'planning','formal_status':'NOT_FROZEN','design_id':design.design_id,
        'pilot_plan_id':plan.plan_id,'pilot_independent_tasks':len(tasks),'source_hash':index['source_hash'],
        'primary_contrasts':summaries,'continuous_confirmation_communication_cost':continuous,
        'recommended_balanced_task_count':recommended,'recommended_llm_repeats_per_method_task':1,'recommended_market_repeats_per_policy':0,
        'method_diagnostics':events,'resource_grid':budget_grid,
        'cumulative_reserved_usd':reserved/1e9,'remaining_monetary_authorization':None,
        'finite_formal_live_call_allocation_required':None if recommended is None else recommended*72,
        'observed_cost_upper_usd_per_complete_task':expected_task_cost,'cost_two_sd_sensitivity_per_task':high_task_cost,
        'pilot_cost_upper_usd':None if unknown_cost else sum(cost_by_task),
        'zero_event_upper_95_at_pilot_N':event_upper_bound(0,len(tasks),design.event_confidence),
        'zero_event_required_tasks_for_5pct_upper':math.ceil(math.log(1-design.event_confidence)/math.log(1-design.zero_event_rate_target)),
        'effect_rationale':design.effect_rationale,'precision_rationale':design.precision_rationale,
        'assumptions':[design.analysis_population,design.planning_test,design.sensitivity_rationale,
            'Per-method Wilson bands are descriptive binomial reference intervals; the paired planning simulation explicitly retains the three prespecified conflict strata.',
            'One trajectory per task/method estimates the compound task-plus-generation variance. It cannot identify within-task LLM repeat variance.',
            'All prior DeepSeek pilots are preserved and excluded from Google nuisance estimates; provider and output cap changed before the independent Google pilot.',
            'Simulation alpha calibration is a diagnostic, not a finite-sample guarantee; bounded Hoeffding precision is a separate conservative target.'],
        'blockers':blockers,'formal_execution_permitted':False}
    durable_json(output/'candidate_plan.json',candidate)
    durable_json(output/'task_vectors.json',{'task_ids':task_ids,'strata':strata.tolist(),'methods':JOINT_METHODS,'certification_matrix':matrix.tolist(),
        'cost_upper_usd_by_task':cost_by_task,'paired_message_counts_full_single':messages.tolist()})
    durable_json(output/'design.json',design.model_dump(mode='json'))
    inputs={str(Path(p).resolve()):file_sha256(p) for p in [pilot_path,design_path,Path(pilot_path).parent/'COMPLETED.json',Path(pilot_path).parent/'registration.json',*snapshots]}
    artifacts={str(p.resolve()):file_sha256(p) for p in sorted(output.glob('*.json'))}
    durable_json(output/'power_precision_receipt.json',{'schema_version':1,'status':'PLANNING_COMPLETED_REQUIRES_FORMAL_FREEZE',
        'input_hashes':inputs,'artifact_hashes':artifacts,'split_check':check,'simulation_draws_per_scenario_N':design.simulation_draws,
        'random_seed':design.simulation_seed,'family_alpha':design.family_alpha,'formal_freeze_created':False,
        'thresholds_bound_by_pre_call_plan':True,'recommended_task_count':recommended,'blockers':blockers})
    return candidate
