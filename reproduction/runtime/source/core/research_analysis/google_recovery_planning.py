"""F72 for the prospectively amended, complete Google administrative recovery.

No transport or write-capable ledger is constructed by the loader. Operational
observations remain immutable; quota completions are planning nuisance only.
"""
from datetime import datetime
from pathlib import Path
import itertools
import math
import shutil
import sqlite3

import numpy as np
from scipy.stats import norm

from core.codesign.schema import EpisodeResult, TaskSpec
from core.codesign.runner import verify_episode, design_implementation_manifest
from core.codesign.pilot import METHODS
from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256, research_hash
from core.experiments.strict_types import strict_json_loads, load_research_model
from .google_pilot import load_google_pilot, load_google_inputs, validate_episode_identity, verify_history
from .google_recovery import validate_recovery_scope
from .study_freeze import FreezeGateError, FreezeIntegrityError
from .analysis import (PlanningDesign, JOINT_METHODS, PATTERNS, joint_probabilities,
    tilt_joint, simulate_contrast, paired_interval, balanced_size, wilson_interval, event_upper_bound)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SHA256 = 'ac3835e2ad2c6858541485de09356c40217db3513b17ab51fd0eecdafb7d4fbb'
LOCKED = [('analytic-google-pilot-v4-development-0000', 0),
          ('analytic-google-pilot-v4-development-0001', 1)]
CONTRASTS = ['full_vs_single', 'factor_S', 'factor_V', 'interaction_SV']
COUNTS = [12,24,48,72,96,144,192,288,384,576,768,1152,1536,2304,3072,4608]


def read(path):
    return strict_json_loads(Path(path).read_text())


def ref(path):
    return {'path':str(Path(path).resolve()), 'sha256':file_sha256(path)}


def require(condition, message, *, integrity=True):
    if not condition:
        raise (FreezeIntegrityError if integrity else FreezeGateError)(message)


def checked(reference):
    path=Path(reference['path'])
    require(path.is_absolute() and path.is_file() and file_sha256(path)==reference['sha256'],
            'RECOVERY_PLANNING_BOUND_CONTENT_CHANGED')
    return path


def _read_ledger(path):
    """Read-only SQLite access; never create or reconcile a ledger."""
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as db:
        state={k:strict_json_loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
        for name in ('reservations','settlements'):
            state[name]={k:strict_json_loads(v) for k,v in db.execute(f'SELECT identity,body FROM {name}')}
    return state


def validate_amendment(record, original_design, original_index):
    path=checked(record['planning_amendment']); amendment=read(path)
    protocol=checked(amendment['protocol'])
    require(file_sha256(protocol)==PROTOCOL_SHA256,'RECOVERY_POWER_PROTOCOL_CHANGED')
    for item in ('protocol','original_planning_design','original_pilot_index','free_tier_authorization'):
        checked(amendment[item])
    require(amendment['original_planning_design']==ref(original_design)
            and amendment['original_pilot_index']==ref(original_index),'AMENDMENT_ORIGINAL_INPUT_MISMATCH')
    require(amendment['locked_quota_task_ids']==[{'task_id':t,'stratum':s} for t,s in LOCKED]
            and amendment['joint_completion_scenarios']==1024 and amendment['projected_scenario_count']==84
            and amendment['primary_delta']==.1 and amendment['operational_outcomes_unchanged'] is True
            and amendment['same_N_required_for_all_operational_and_quota_nuisance_primary_scenarios'] is True
            and amendment['no_automatic_formal_permission'] is True
            and amendment['stage']=='BEFORE_ANY_RECOVERY_REQUEST','RECOVERY_SENSITIVITY_CONTRACT_CHANGED')
    registered=datetime.fromisoformat(amendment['registered_at_utc']).timestamp()
    require(registered<=record['registered_at'],'RECOVERY_AMENDMENT_NOT_PROSPECTIVE')
    require(record['preserved_hashes'].get(str(protocol))==PROTOCOL_SHA256
            and record['preserved_hashes'].get(str(path))==file_sha256(path),'AMENDMENT_NOT_FROZEN_WITH_RECOVERY')
    require(amendment['pacing']['minimum_interval_seconds']==record['minimum_interval_seconds']
            and record['minimum_interval_seconds']>=5.0,'AMENDMENT_PACING_MISMATCH')
    return amendment


def validate_allocation_and_pacing(state, before, parent, record, pacing):
    verify_history(state,before)
    expected={'authorization_id':record['authorization_id'],'plan_id':record['allocation_plan_id'],
        'plan_hash':research_hash(record),'parent_reference':record['parent_reference'],
        'maximum_calls':2016,'maximum_input_tokens':2016*32768,'maximum_output_tokens':2016*4096,
        'per_call_input_cap':32768,'per_call_output_cap':4096,
        'input_nano':300,'output_nano':2500,'cached_nano':30,'provider':'gemini','currency':'USD','monetary_cap':None}
    require(all(state['contract'].get(k)==v for k,v in expected.items()),'RECOVERY_ALLOCATION_CONTRACT_CHANGED')
    require(not before['reservations'] and not before['settlements'],'RECOVERY_ALLOCATION_WAS_PREUSED')
    require(not state['blocked_reason'] and not state['pending_block_reasons'] and state['active_batch'] is None,
            'RECOVERY_NEW_ALLOCATION_STOPPED',integrity=False)
    require(set(state['reservations'])==set(state['settlements']) and
            all(s['usage_status']=='known' and s['peak_price_cost_upper_bound_nano'] is not None
                for s in state['settlements'].values()),'RECOVERY_NEW_USAGE_INCOMPLETE',integrity=False)
    require(len(state['reservations'])<=2016 and len(parent['reservations'])+len(state['reservations'])<=2160,
            'RECOVERY_FINITE_CALL_CAP_CHANGED')
    expected_totals={'calls':len(state['reservations']),
        'input_tokens':sum(r['input_cap'] for r in state['reservations'].values()),
        'output_tokens':sum(r['output_cap'] for r in state['reservations'].values()),
        'reserved_nano':sum(r['reserved_nano'] for r in state['reservations'].values())}
    require(state['totals']==expected_totals,'RECOVERY_LEDGER_TOTALS_CHANGED')
    for r in state['reservations'].values():
        require(r['input_cap']==32768 and r['output_cap']==4096 and r['reserved_nano']==20070400,
                'RECOVERY_PER_CALL_RESERVATION_CHANGED')
    grants=pacing['grants']; interval=record['minimum_interval_seconds']
    require(pacing['contract']['minimum_interval_seconds']==interval and interval>=5.0
            and pacing['contract']['retry'] is False
            and pacing['contract']['queue_rule']=='durable_fifo_tickets_v1'
            and pacing['contract']['queue_time_counts_toward_original_episode_wall_budget'] is True,
            'RECOVERY_PACING_CONTRACT_CHANGED')
    tickets=pacing['tickets']
    require([t['sequence'] for t in tickets]==list(range(1,len(tickets)+1))
            and len({t['identity'] for t in tickets})==len(tickets)
            and all(t['status'] in {'expired','granted'} for t in tickets)
            and [t['identity'] for t in tickets if t['status']=='granted']==[g['identity'] for g in grants],
            'RECOVERY_PACING_FIFO_HISTORY_CHANGED')
    require([g['sequence'] for g in grants]==list(range(1,len(grants)+1))
            and len({g['identity'] for g in grants})==len(grants)
            and {g['identity'] for g in grants}==set(state['reservations']),'RECOVERY_PACING_CALL_COVERAGE')
    require(all(b['granted_monotonic']-a['granted_monotonic']>=interval-1e-9 for a,b in zip(grants,grants[1:])),
            'RECOVERY_PACING_INTERVAL_VIOLATION')
    require(all(g['granted_at']>=record['registered_at'] and
                state['reservations'][g['identity']]['time']>=g['granted_at'] for g in grants),
            'RECOVERY_REQUEST_PRECEDES_REGISTRATION_OR_PACING')
    pending_ids={r['episode_id'] for r in record['pending_schedule']}
    require(all(k.rsplit(':',1)[0] in pending_ids for k in state['reservations']),
            'RECOVERY_REEXECUTED_OR_FOREIGN_EPISODE')


def load_completed_recovery(index_path,design_path):
    """Return the original F72 seven-object interface only after chain validation."""
    index_path=Path(index_path).resolve(); root=index_path.parent; index=read(index_path)
    marker=read(root/'COMPLETED.json'); record=read(root/'recovery_registration.json')
    for key,name in [('index_sha256','benchmark_index.json'),('registration_sha256','registration.json'),
                     ('recovery_registration_sha256','recovery_registration.json')]:
        require(marker[key]==file_sha256(root/name),'RECOVERY_COMPLETION_HASH_MISMATCH')
    require(marker['complete_planned_index'] is True and marker['complete_original_schedule_execution'] is True
            and index['study_stage']=='pilot_recovery' and index['stopping_reason'] is None,
            'COMPLETED_ORIGINAL_RECOVERY_SCHEDULE_REQUIRED',integrity=False)
    require(index['recovery_registration']==ref(root/'recovery_registration.json'), 'RECOVERY_REGISTRATION_POINTER_CHANGED')
    original_index_path=checked(record['original_index']); original=read(original_index_path)
    original_registration=read(checked(record['original_registration']))
    parent_envelope=read(checked(record['parent_reference'])); parent=parent_envelope['state']
    require(research_hash(parent)==parent_envelope['sha256'] and parent==original['spend_snapshot'],
            'RECOVERY_PARENT_SNAPSHOT_CHANGED')
    original_rows,pending,unknown=validate_recovery_scope(original,original_registration,parent)
    require(record['original_recorded_rows']==original_rows and record['pending_schedule']==pending
            and record['retained_unknown_liabilities_nano']==unknown and record['unknown_usage_reconciled'] is False
            and record['original_failed_episodes_rerun'] is False and record['original_tasks_or_seeds_replaced'] is False,
            'RECOVERY_ORIGINAL_HISTORY_CHANGED')
    for path,sha in {**record['preserved_hashes'],**record['recovery_source_hashes']}.items():
        checked({'path':path,'sha256':sha})
    require(research_hash(record['recovery_source_hashes'])==record['recovery_runtime_hash']
            and research_hash(original_registration['runtime_source_hashes'])==record['original_runtime_hash'],
            'RECOVERY_RUNTIME_HASH_MISMATCH')
    require(index['episodes'][:18]==original_rows and len(index['episodes'])==270
            and index['planned_trials']==index['indexed_trials']==270 and index['independent_task_count']==30,
            'RECOVERY_ORIGINAL_ROWS_OR_FULL_DENOMINATOR_CHANGED')
    identity=lambda r:tuple(r[k] for k in ('episode_id','task_id','method','batch_id'))
    expected=[identity(r) for r in original_registration['schedule']]
    require([identity(r) for r in index['episodes']]==expected and len(set(expected))==270,
            'RECOVERY_ORIGINAL_SCHEDULE_MISMATCH')
    require(all(r['status']=='RECORDED' for r in index['episodes']), 'RECOVERY_UNRUN_OR_EXECUTION_FAILURE',integrity=False)
    for name in ('plan.json','registration.json','planning_design.json','experiment_spec.json'):
        require(file_sha256(root/name)==record['preserved_hashes'].get(str(original_index_path.parent/name)),
                'RECOVERY_ORIGINAL_CONFIGURATION_COPY_CHANGED')
    plan=load_google_pilot(root/'plan.json'); tasks,split_check=load_google_inputs(plan)
    design=load_research_model(design_path,PlanningDesign)
    require(file_sha256(design_path)==plan.planning_design_sha256
            and research_hash(design)==research_hash(read(root/'planning_design.json')),
            'RECOVERY_ORIGINAL_PRECALL_DESIGN_CHANGED')
    require(plan.plan_id==design.pilot_plan_id and research_hash(plan)==index['plan_hash']==original['plan_hash']
            and index['source_hash']==original['source_hash']==record['original_source_hash']
            ==design_implementation_manifest().source_hash,'RECOVERY_SCIENTIFIC_IDENTITY_CHANGED')
    require([c.id for c in design.contrasts]==CONTRASTS and design.candidate_task_counts==COUNTS
            and design.joint_pattern_pseudocounts==[.5,1.,2.] and design.simulation_seed==2026102344
            and (design.family_alpha,design.target_power,design.target_half_width,design.simulation_draws)==(.05,.8,.1,10000)
            and all(c.delta==.1 for c in design.contrasts),'RECOVERY_REGISTERED_TARGETS_CHANGED')
    amendment=validate_amendment(record,design_path,original_index_path)
    started=read(root/'EXECUTION_STARTED.json')
    require(started['recovery_registration']==ref(root/'recovery_registration.json')
            and started['started_at']>=record['registered_at'],'RECOVERY_STARTED_BEFORE_REGISTRATION')
    before=read(root/'allocation_before.json'); after=read(root/'allocation_final_snapshot.json')
    require(all(research_hash(e['state'])==e['sha256'] for e in (before,after))
            and after['state']==index['spend_snapshot']==_read_ledger(record['allocation_path']),
            'RECOVERY_ALLOCATION_SNAPSHOT_MISMATCH')
    require(marker['pacing_snapshot']==index['pacing_snapshot']==ref(root/'pacing_snapshot.json'),
            'RECOVERY_PACING_SNAPSHOT_NOT_BOUND')
    pacing=read(checked(index['pacing_snapshot']))
    validate_allocation_and_pacing(after['state'],before['state'],parent,record,pacing)
    state=after['state']; result_map={}; costs={}; refusals={}; calls_seen=[set(),set()]
    task_map={t.task_id:t for t in tasks}
    require(len(tasks)==30 and len(task_map)==30,'RECOVERY_THIRTY_INDEPENDENT_TASKS_REQUIRED')
    for n,row in enumerate(index['episodes']):
        path=Path(row['path']); verify_episode(path)
        require(file_sha256(path/'episode_result.json')==row['result_sha256'],'RECOVERY_EPISODE_HASH_CHANGED')
        result=load_research_model(path/'episode_result.json',EpisodeResult)
        task=task_map[row['task_id']]
        if row['method']=='central_rule':
            body=task.model_dump(mode='json'); body.update(signing_layer='explicit_rule_owners_v1',model_configurations=[])
            task=TaskSpec.model_validate(body)
        validate_episode_identity(result,task,row,index['source_hash'])
        require(result.split=='development' and (n<18 or result.started_at>=started['started_at']),
                'RECOVERY_TASK_OR_EPISODE_TIME_MISMATCH')
        key=(row['task_id'],row['method']); result_map[key]=result.model_dump(mode='json')
        calls=read(path/'llm/call_index.json') if row['method']!='central_rule' else []
        require(len(calls)==result.common_contract['live_calls'],'RECOVERY_EPISODE_CALL_COUNT_CHANGED')
        book=parent if n<18 else state; seen=calls_seen[0 if n<18 else 1]
        amounts=[]
        for call in calls:
            call_id=row['episode_id']+':'+call['attempt_id']
            require(call_id not in seen and book['reservations'].get(call_id,{}).get('request_hash')==call['request_hash']
                    and book['settlements'].get(call_id)==call.get('spend_settlement'),
                    'RECOVERY_RAW_CALL_LEDGER_MISMATCH')
            seen.add(call_id); amounts.append(call['spend_settlement']['peak_price_cost_upper_bound_nano'])
        costs[key]=sum(amounts)/1e9 if all(a is not None for a in amounts) else None
        feedback=read(path/'protocol/operation-feedback.json') if row['method']!='central_rule' else {'receipts':[]}
        refusals[key]=sum(not r['applied'] for r in feedback['receipts'])
    require(calls_seen==[set(parent['reservations']),set(state['reservations'])], 'RECOVERY_LEDGER_HAS_UNINDEXED_CALLS')
    for tid,stratum in LOCKED:
        require(int(task_map[tid].prompt_resources['stratum_index'])==stratum
                and all(not result_map[(tid,m)]['A_cert'] and result_map[(tid,m)]['terminal_state']=='METHOD_FAILED'
                        for m in JOINT_METHODS),'LOCKED_QUOTA_TASK_OR_ORIGINAL_FAILURE_CHANGED')
    require(index['parent_spend_snapshot']==record['parent_reference']
            and index['cumulative_usage']['unknown_request_count']==16
            and index['cumulative_usage']['unknown_full_reservation_usd']==sum(unknown.values())/1e9
            and index['cumulative_usage']['total_calls']==35+len(state['reservations']),
            'RECOVERY_CUMULATIVE_PARENT_LIABILITY_CHANGED')
    checks={'split_check':split_check,'amendment':record['planning_amendment'],'protocol':amendment['protocol'],
        'parent_index':record['original_index'],'original_rows_retained':18,'resumed_rows':252,
        'independent_tasks':30,'pacing_grants':len(pacing['grants']),'retained_parent_unknowns':16,
        'raw_call_accounting_complete':True,'same_model_tasks_budget_and_science':True}
    return plan,tasks,index,result_map,costs,refusals,checks


def project_distribution(probabilities,weights):
    values=PATTERNS@np.asarray(weights); unique=np.unique(values)
    return unique,np.stack([probabilities[:,values==v].sum(axis=1) for v in unique],axis=1)


def nuisance_scenarios(matrix,strata,weights,pseudocount,locked_indices):
    """One representative per exact projected pair, retaining joint prior mass."""
    require(len(locked_indices)==2 and [int(strata[i]) for i in locked_indices]==[0,1],
            'LOCKED_QUOTA_STRATA_CHANGED')
    values=PATTERNS@np.asarray(weights); unique=np.unique(values)
    for i,j in itertools.product(range(len(unique)),repeat=2):
        representatives=[int(np.flatnonzero(values==unique[k])[0]) for k in (i,j)]
        completed=np.array(matrix,copy=True)
        completed[locked_indices]=PATTERNS[representatives]
        base=joint_probabilities(completed,strata,pseudocount)
        projected,probabilities=project_distribution(base,weights)
        yield {'pair_indices':[i,j],'projected_values':[float(unique[i]),float(unique[j])],
               'representative_patterns':representatives,'joint_probabilities':base.tolist(),
               'values':projected,'probabilities':probabilities}


def completion_mapping(contrasts):
    maps={}
    for c in contrasts:
        values=PATTERNS@np.array([c.weights.get(m,0.) for m in JOINT_METHODS])
        maps[c.id]=np.searchsorted(np.unique(values),values)
    return [{'joint_pattern_indices':[a,b],
             'projected_pair_indices':{name:[int(v[a]),int(v[b])] for name,v in maps.items()}}
            for a,b in itertools.product(range(32),repeat=2)]


def _simulate_grid(base,values,delta,design,alpha,rng_for):
    alternative=tilt_joint(base,values,delta); null=tilt_joint(base,values,0.)
    rows=[]
    for i,n in enumerate(design.candidate_task_counts):
        alt=simulate_contrast(alternative,values,n,alpha,design.simulation_draws,rng_for(i,0))
        zero=simulate_contrast(null,values,n,alpha,design.simulation_draws,rng_for(i,1))
        widths=[alt['half_width_p90'],zero['half_width_p90']]
        alt.update(null_type1_rate=zero['two_sided_rejection_rate'],null_type1_mc_95_interval=zero['two_sided_mc_95_interval'],
            null_half_width_p90=zero['half_width_p90'],null_degenerate_fraction=zero['degenerate_fraction'],
            planning_precision_half_width_p90=max(widths) if all(v is not None for v in widths) else None,
            null_calibration_ok=zero['two_sided_mc_95_interval'][0]<=alpha)
        rows.append(alt)
    return {'grid':rows,'joint_alternative_probabilities':alternative.tolist(),'joint_null_probabilities':null.tolist()}


def common_passing_counts(scenarios,design):
    """Intersect predicates at the same N; never assume MC monotonicity."""
    require(bool(scenarios),'NO_REGISTERED_PLANNING_SCENARIOS')
    good=[]
    for n in design.candidate_task_counts:
        matches=[[r for r in s['grid'] if r['task_count']==n] for s in scenarios]
        require(all(len(x)==1 for x in matches),'SCENARIO_GRID_INCOMPLETE_OR_DUPLICATED')
        if all(x[0]['power_mc_95_interval'][0]>=design.target_power and
               x[0]['planning_precision_half_width_p90'] is not None and
               x[0]['planning_precision_half_width_p90']<=design.target_half_width and
               x[0]['null_calibration_ok'] for x in matches):good.append(n)
    return good


def plan_recovered_google_study(index_path,design_path,output):
    plan,tasks,index,results,costs,refusals,checks=load_completed_recovery(index_path,design_path)
    design=load_research_model(design_path,PlanningDesign); output=Path(output).resolve()
    require(not output.exists(),'RECOVERY_PLANNING_OUTPUT_EXISTS'); output.mkdir(parents=True)
    ids=[t.task_id for t in tasks]; strata=np.array([int(t.prompt_resources['stratum_index']) for t in tasks])
    matrix=np.array([[float(bool(results[(tid,m)]['A_cert'])) for m in JOINT_METHODS] for tid in ids])
    locked=[ids.index(tid) for tid,_ in LOCKED]
    require(np.bincount(strata).tolist()==[10,10,10] and not matrix[locked].any(),'ORIGINAL_OPERATIONAL_VECTOR_CHANGED')
    durable_json(output/'task_vectors.json',{'task_ids':ids,'strata':strata.tolist(),'methods':JOINT_METHODS,
        'certification_matrix':matrix.tolist(),'source':'observed_operational_only','nuisance_imputation_applied':False})
    durable_json(output/'joint_completion_mapping.json',{'patterns':PATTERNS.tolist(),'mappings':completion_mapping(design.contrasts)})
    alpha=design.family_alpha/4; rng=np.random.default_rng(design.simulation_seed)
    operational=[]; summaries=[]; continuous=None
    # This order matches the original operational kernel's RNG consumption.
    for contrast in design.contrasts:
        weights=np.array([contrast.weights.get(m,0.) for m in JOINT_METHODS]); values=PATTERNS@weights
        observed=paired_interval(matrix,strata,weights,alpha); boot=[]
        for s in range(3):
            block=(matrix@weights)[strata==s]
            boot.append(np.var(rng.choice(block,size=(design.bootstrap_draws,len(block)),replace=True),axis=1,ddof=1))
        observed['bootstrap_variance_p90']=float(np.quantile(np.mean(boot,axis=0),.9))
        scenarios=[]
        for pseudo in design.joint_pattern_pseudocounts:
            base=joint_probabilities(matrix,strata,pseudo)
            for delta in sorted(set(design.delta_sensitivity+[contrast.delta])):
                detail=_simulate_grid(base,values,delta,design,alpha,lambda *_:rng)
                detail.update(contrast=contrast.id,pseudocount_per_stratum=pseudo,target_delta=delta,
                    joint_base_probabilities=base.tolist(),source='observed_operational')
                scenarios.append(detail)
                if delta==contrast.delta:operational.append(detail)
        summary={'contrast':contrast.id,'rq':contrast.rq,'delta':contrast.delta,'alpha':alpha,**observed,
            'distribution_free_precision_task_count':balanced_size(float(np.ptp(values))**2*math.log(8/design.family_alpha)/(2*design.target_half_width**2))}
        summaries.append(summary)
        durable_json(output/(contrast.id+'-operational.json'),{'summary':summary,'scenarios':scenarios})
        print(contrast.id+' operational planning completed',flush=True)
    operational_counts=common_passing_counts(operational,design)
    nuisance=[]; scenario_counts={}; (output/'nuisance').mkdir()
    for cidx,contrast in enumerate(design.contrasts):
        weights=np.array([contrast.weights.get(m,0.) for m in JOINT_METHODS]); scenario_counts[contrast.id]=0
        for pidx,pseudo in enumerate(design.joint_pattern_pseudocounts):
            for scenario in nuisance_scenarios(matrix,strata,weights,pseudo,locked):
                pair=scenario['pair_indices']
                def substream(nidx,phase):
                    return np.random.default_rng(np.random.SeedSequence(
                        [design.simulation_seed,72731024,cidx,pidx,*pair,nidx,phase]))
                detail=_simulate_grid(scenario['probabilities'],scenario['values'],.1,design,alpha,substream)
                detail.update(contrast=contrast.id,pseudocount_per_stratum=pseudo,target_delta=.1,
                    source='quota_nuisance_only',pair_indices=pair,projected_values=scenario['projected_values'],
                    representative_patterns=scenario['representative_patterns'],projection_support=scenario['values'].tolist(),
                    joint_base_probabilities=scenario['joint_probabilities'],projected_base_probabilities=scenario['probabilities'].tolist())
                nuisance.append(detail); scenario_counts[contrast.id]+=1
                durable_json(output/'nuisance'/f'{contrast.id}-p{pidx}-a{pair[0]}-b{pair[1]}.json',detail)
        print(contrast.id+' quota nuisance planning completed',flush=True)
    require(scenario_counts==dict(zip(CONTRASTS,[27,75,75,75])),'NUISTANCE_SCENARIO_COUNT_CHANGED')
    all_scenarios=operational+nuisance; common=common_passing_counts(all_scenarios,design)
    for summary in summaries:
        rows=[s for s in all_scenarios if s['contrast']==summary['contrast']]
        summary['common_passing_task_counts']=common_passing_counts(rows,design)
        summary['common_task_count']=next(iter(summary['common_passing_task_counts']),None)
    messages=np.array([[results[(tid,m)]['budget_snapshot']['used']['messages'] for m in ['cdr_s1_v1','single']] for tid in ids])
    continuous=paired_interval(messages,strata,[1.,-1.],alpha)
    variance=continuous['within_stratum_variance']*design.continuous_variance_inflation
    continuous.update(unit='protocol_messages',source='observed_operational_only',paced_cost_variance_identified=False,
        delta=design.continuous_cost_delta_messages,variance_inflation=design.continuous_variance_inflation,
        approximate_power_task_count=None if variance==0 else balanced_size(variance*(norm.ppf(1-alpha/2)+norm.ppf(design.target_power))**2/design.continuous_cost_delta_messages**2),
        approximate_precision_task_count=None if variance==0 else balanced_size(variance*norm.ppf(1-alpha/2)**2/design.target_half_width**2))
    methods={m:{'tasks':30,'certified':sum(bool(results[(tid,m)]['A_cert']) for tid in ids),
        'method_failed':sum(results[(tid,m)]['terminal_state']=='METHOD_FAILED' for tid in ids),
        'operation_rejections':sum(refusals[(tid,m)] for tid in ids),'calls':sum(results[(tid,m)]['common_contract']['live_calls'] for tid in ids)} for m in METHODS}
    per_task=8*9*(32768*300+4096*2500)
    candidate={'schema_version':3,'study_stage':'planning','provider':'gemini','pilot_plan_id':plan.plan_id,
        'planning_kind':'prospectively_amended_administrative_recovery','source_hash':index['source_hash'],
        'protocol':checks['protocol'],'planning_amendment':checks['amendment'],'currency':'USD','monetary_cap':None,
        'pilot_independent_tasks':30,'observed_operational_rows':270,'locked_quota_tasks':[t for t,_ in LOCKED],
        'primary_contrasts':summaries,'joint_completion_count':1024,'projected_classes_without_pseudocount':84,
        'quota_scenario_count':len(nuisance),'operational_primary_scenario_count':len(operational),
        'operational_common_passing_task_counts':operational_counts,'common_passing_task_counts':common,
        'recommended_balanced_task_count':next(iter(common),None),'continuous_confirmation_communication_cost':continuous,
        'method_diagnostics':methods,'parent_unknown_usage_retained':16,'actual_provider_bill_usd':None,
        'resource_grid':[{'tasks':n,'live_calls':n*72,'episodes':n*9,'maximum_input_token_reservation':n*72*32768,
            'maximum_output_token_reservation':n*72*4096,'maximum_additional_reservation_usd':n*per_task/1e9,
            'independent_audit_candidate_evaluations':n*9,'market_runs':0} for n in design.candidate_task_counts],
        'same_pacing_required_for_formal':True,'formal_status':'NOT_FROZEN','formal_execution_permitted':False,
        'blockers':([] if common else ['NO_COMMON_REGISTERED_N_PASSES_ALL_SCENARIOS'])+['F73_INDEPENDENT_TEST_FREEZE_REQUIRED'],
        'inference_scope':'conditional planning under registered nuisance family; not a confirmatory effect or arbitrary distribution guarantee'}
    durable_json(output/'candidate_plan.json',candidate); durable_json(output/'design.json',design.model_dump(mode='json'))
    snapshots=[]
    for name in ('core/research_analysis/google_recovery_planning.py','core/research_analysis/google_planning.py',
                 'core/research_analysis/analysis.py','requirements-cscwd.lock'):
        target=output/'planning_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target);snapshots.append(target)
    bound=[Path(index_path),Path(design_path),Path(index_path).parent/'COMPLETED.json',
        Path(index_path).parent/'recovery_registration.json',Path(index_path).parent/'pacing_snapshot.json',
        Path(index_path).parent/'allocation_final_snapshot.json',Path(checks['amendment']['path']),Path(checks['protocol']['path']),*snapshots]
    durable_json(output/'power_precision_receipt.json',{'schema_version':3,'status':'RECOVERY_PLANNING_COMPLETED_REQUIRES_FORMAL_FREEZE',
        'input_hashes':{str(p.resolve()):file_sha256(p) for p in bound},
        'artifact_hashes':{str(p.resolve()):file_sha256(p) for p in sorted(output.rglob('*.json'))},
        'checks':checks,'simulation_draws_per_scenario_N':design.simulation_draws,
        'seed':design.simulation_seed,'nuisance_domain_tag':72731024,'recommended_task_count':candidate['recommended_balanced_task_count'],
        'formal_freeze_created':False,'observed_records_modified':False})
    return candidate
