"""Provider-separated Phase 7 planning: GLM primary and descriptive replications."""
from pathlib import Path
import math
import shutil
import numpy as np
from scipy.stats import t,norm
from core.codesign.schema import TaskSpec,EpisodeResult
from core.codesign.runner import verify_episode,design_implementation_manifest
from core.codesign.pilot import METHODS
from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256,research_hash
from core.experiments.strict_types import load_research_model
from .provider_pilot import load_provider_pilot,load_provider_inputs,validate_episode_identity,verify_history,PROVIDERS
from .analysis import PlanningDesign,JOINT_METHODS,PATTERNS,joint_probabilities,paired_interval,balanced_size,wilson_interval,event_upper_bound
from .google_recovery_planning import read,ref,checked,require,_simulate_grid,common_passing_counts

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL_SHA256='900f55c68ebd5f09ef3b6ea1f44f16bb20eac356796f8d01c4b7e5af3d019db5'


def validate_admission(plan,admission,state,pacing=None):
    """Audit all admission events, including expired requests with no allocation."""
    require(admission['maximum_concurrent_requests']==plan.maximum_concurrent_requests
        and admission['queue_time_counts_toward_original_episode_wall_budget'] is True
        and not admission['active'] and not admission['pending'],'PROVIDER_ADMISSION_NOT_CLOSED')
    queued={};active=set();admitted=set();released=set();expired=set()
    for event in admission['events']:
        identity=event['identity'];kind=event['kind']
        if kind=='queued':
            require(identity not in queued,'PROVIDER_DUPLICATE_ADMISSION_TICKET')
            queued[identity]=event['ticket']
        elif kind=='expired':
            require(identity in queued and identity not in admitted,'PROVIDER_INVALID_EXPIRED_TICKET')
            expired.add(identity)
        elif kind=='admitted':
            waiting=[key for key in queued if key not in admitted and key not in expired]
            require(waiting and identity==min(waiting,key=queued.get),'PROVIDER_ADMISSION_FIFO_VIOLATION')
            require(identity not in active,'PROVIDER_DUPLICATE_ACTIVE_REQUEST')
            active.add(identity);admitted.add(identity)
            require(len(active)<=plan.maximum_concurrent_requests and event['active_count']==len(active),
                'PROVIDER_CONCURRENCY_LIMIT_EXCEEDED')
        elif kind=='released':
            require(identity in active,'PROVIDER_RELEASE_WITHOUT_ADMISSION')
            active.remove(identity);released.add(identity)
            require(event['active_count']==len(active),'PROVIDER_ADMISSION_COUNT_CHANGED')
        else:require(False,'PROVIDER_UNKNOWN_ADMISSION_EVENT')
    require(not active and admitted==released and set(queued)==admitted|expired,
        'PROVIDER_ADMISSION_HISTORY_INCOMPLETE')
    require(set(state['reservations'])<=admitted,'PROVIDER_UNADMITTED_REQUEST')
    if plan.provider=='gemini':
        require(pacing is not None and pacing['contract']['minimum_interval_seconds']==5.1
            and pacing['contract']['queue_rule']=='durable_fifo_tickets_v1'
            and pacing['contract']['queue_time_counts_toward_original_episode_wall_budget'] is True,
            'GOOGLE_REGISTERED_PACING_REQUIRED')
        grants=pacing['grants'];tickets=pacing['tickets']
        require([g['sequence'] for g in grants]==list(range(1,len(grants)+1))
            and len({g['identity'] for g in grants})==len(grants)
            and [x['identity'] for x in tickets if x['status']=='granted']==[g['identity'] for g in grants],
            'GOOGLE_PACING_FIFO_HISTORY_CHANGED')
        require(all(b['granted_monotonic']-a['granted_monotonic']>=5.1-1e-9 for a,b in zip(grants,grants[1:])),
            'GOOGLE_PACING_INTERVAL_VIOLATION')
        require(set(state['reservations'])<={g['identity'] for g in grants}<=admitted,
            'GOOGLE_PACING_RESERVATION_COVERAGE')
    elif pacing is not None:require(False,'FOREIGN_PROVIDER_PACING_ATTACHED')


def load_completed_provider_pilot(index_path,design_path=None):
    root=Path(index_path).resolve().parent;index=read(index_path);marker=read(root/'COMPLETED.json')
    require(marker['index_sha256']==file_sha256(index_path)
        and marker['registration_sha256']==file_sha256(root/'registration.json'),'PROVIDER_COMPLETION_HASH_MISMATCH')
    require(marker['complete_planned_index'] is True and index['study_stage']=='provider_pilot'
        and index['stopping_reason'] is None,'COMPLETED_INDEPENDENT_PROVIDER_PILOT_REQUIRED',integrity=False)
    plan=load_provider_pilot(root/'plan.json');tasks,split_check=load_provider_inputs(plan)
    design_path=Path(design_path or plan.planning_design_path).resolve()
    design=load_research_model(design_path,PlanningDesign);registration=read(root/'registration.json')
    require(plan.provider_protocol.sha256==PROTOCOL_SHA256
        and file_sha256(plan.provider_protocol.path)==PROTOCOL_SHA256,'PROVIDER_PROTOCOL_CHANGED')
    require(file_sha256(design_path)==plan.planning_design_sha256
        and research_hash(design)==research_hash(read(root/'planning_design.json'))
        and design.pilot_plan_id==plan.plan_id,'PROVIDER_PRECALL_PLANNING_DESIGN_CHANGED')
    for path,sha in {**registration['runtime_source_hashes'],**registration['input_hashes']}.items():
        checked({'path':path,'sha256':sha})
    require(research_hash(registration['runtime_source_hashes'])==registration['runtime_hash']==index['runtime_hash']
        and registration['source_hash']==index['source_hash']==design_implementation_manifest().source_hash,
        'PROVIDER_REGISTERED_IMPLEMENTATION_CHANGED')
    require(research_hash(plan)==registration['plan_hash']==index['plan_hash']
        and registration['provider']==index['provider']==plan.provider
        and registration['study_role']==index['study_role']==plan.study_role
        and registration['no_cross_provider_power_pooling'] is True and index['no_cross_provider_power_pooling'] is True,
        'PROVIDER_ROLE_OR_ALLOCATION_CHANGED')
    expected=[]
    for i,task in enumerate(tasks):
        order=plan.methods[i%9:]+plan.methods[:i%9]
        expected.extend({'episode_id':task.task_id+'--'+method,'task_id':task.task_id,'method':method,'batch_id':f'batch-{i//2:03d}'} for method in order)
    identity=lambda row:tuple(row[k] for k in ('episode_id','task_id','method','batch_id'))
    require(registration['schedule']==expected and [identity(r) for r in index['episodes']]==[identity(r) for r in expected]
        and len({r['episode_id'] for r in index['episodes']})==len(expected)
        and index['planned_trials']==index['indexed_trials']==len(expected)
        and index['independent_task_count']==len(tasks),'PROVIDER_FIXED_COMPLETE_SCHEDULE_CHANGED')
    require(all(r['status']=='RECORDED' for r in index['episodes']),'PROVIDER_UNRUN_TRIAL_BLOCKS_PLANNING',integrity=False)
    require(registration['maximum_concurrent_requests']==plan.maximum_concurrent_requests
        and registration['minimum_request_interval_seconds']==plan.minimum_request_interval_seconds
        and registration['host_queue_time_counts_toward_original_episode_wall_budget'] is True,
        'PROVIDER_REGISTERED_TRANSPORT_CHANGED')
    before=read(root/'spend_before.json');final=read(root/'spend_final_snapshot.json');state=final['state']
    require(research_hash(state)==final['sha256'] and state==index['spend_snapshot'],'PROVIDER_SPEND_SNAPSHOT_CHANGED')
    verify_history(state,before)
    require(state['contract']==registration['spend_contract'] and not before['reservations']
        and not before['settlements'] and state['blocked_reason'] is None and state['active_batch'] is None
        and not state['pending_block_reasons'],'PROVIDER_ALLOCATION_NOT_COMPLETE',integrity=False)
    contract=state['contract'];expected_contract={'plan_id':plan.plan_id,'plan_hash':research_hash(plan),
        'authorization_id':plan.authorization_id,'provider':plan.provider,'currency':plan.currency,
        'maximum_calls':plan.maximum_generation_calls,'per_call_input_cap':24576,'per_call_output_cap':768,
        'maximum_input_tokens':plan.maximum_generation_calls*24576,'maximum_output_tokens':plan.maximum_generation_calls*768,
        'input_nano':plan.input_price_nano,'output_nano':plan.output_price_nano,'cached_nano':plan.cached_price_nano,
        'monetary_cap_nano':plan.monetary_cap_nano}
    require(all(contract.get(k)==v for k,v in expected_contract.items()),'PROVIDER_FINITE_ALLOCATION_CONTRACT_CHANGED')
    expected_parent=None if plan.parent_liability_reference is None else plan.parent_liability_reference.model_dump(mode='json')
    require(contract['parent_reference']==expected_parent,'PROVIDER_PARENT_LIABILITY_CHANGED')
    if expected_parent:
        parent=read(checked(expected_parent))
        require(parent['retired_no_further_execution'] is True and parent['allowed_successor_plan_id']==plan.plan_id
            and contract['inherited_liability_nano']==parent['known_settled_cost_nano']+parent['unknown_full_reservation_nano']
            and contract['inherited_liability_nano']+state['totals']['reserved_nano']<=100_000_000_000,
            'DEEPSEEK_RELEASE_CAP_OR_RETIRED_PARENT_CHANGED')
    else:require(contract['inherited_liability_nano']==0,'PROVIDER_UNREGISTERED_INHERITED_LIABILITY')
    require(set(state['reservations'])==set(state['settlements'])
        and all(r['usage_status']=='known' and r['peak_price_cost_upper_bound_nano'] is not None for r in state['settlements'].values()),
        'PROVIDER_USAGE_UNKNOWN_OR_UNSETTLED',integrity=False)
    totals={'calls':len(state['reservations']),'input_tokens':sum(r['input_cap'] for r in state['reservations'].values()),
        'output_tokens':sum(r['output_cap'] for r in state['reservations'].values()),
        'reserved_nano':sum(r['reserved_nano'] for r in state['reservations'].values())}
    require(totals==state['totals'] and totals['calls']<=plan.maximum_generation_calls,'PROVIDER_LEDGER_TOTALS_OR_CAP_CHANGED')
    for reservation in state['reservations'].values():
        require(reservation['input_cap']==24576 and reservation['output_cap']==768
            and reservation['reserved_nano']==24576*plan.input_price_nano+768*plan.output_price_nano,
            'PROVIDER_RESERVATION_BUDGET_CHANGED')
    require(file_sha256(root/'request_admission.json')==index['request_admission_sha256'],'PROVIDER_ADMISSION_HASH_CHANGED')
    pacing=None
    if plan.provider=='gemini':
        require(file_sha256(root/'pacing_snapshot.json')==index['pacing_snapshot_sha256'],'GOOGLE_PACING_HASH_CHANGED')
        pacing=read(root/'pacing_snapshot.json')
    validate_admission(plan,read(root/'request_admission.json'),state,pacing)
    task_map={t.task_id:t for t in tasks};results={};costs={};refusals={};seen=set()
    for row in index['episodes']:
        path=Path(row['path']);verify_episode(path)
        require(file_sha256(path/'episode_result.json')==row['result_sha256'],'PROVIDER_RESULT_HASH_CHANGED')
        result=load_research_model(path/'episode_result.json',EpisodeResult);task=task_map[row['task_id']]
        if row['method']=='central_rule':
            body=task.model_dump(mode='json');body.update(signing_layer='explicit_rule_owners_v1',model_configurations=[])
            task=TaskSpec.model_validate(body)
        validate_episode_identity(result,task,row,index['source_hash'])
        require(result.split=='development' and result.started_at>=registration['registered_at'],'PROVIDER_TASK_SPLIT_OR_START_CHANGED')
        key=(row['task_id'],row['method']);results[key]=result.model_dump(mode='json')
        calls=read(path/'llm/call_index.json') if row['method']!='central_rule' else []
        require(len(calls)==result.common_contract['live_calls'],'PROVIDER_RAW_CALL_COUNT_CHANGED')
        for call in calls:
            call_id=row['episode_id']+':'+call['attempt_id']
            require(call_id not in seen and state['reservations'].get(call_id,{}).get('request_hash')==call['request_hash']
                and state['settlements'].get(call_id)==call.get('spend_settlement'),'PROVIDER_RAW_CALL_LEDGER_MISMATCH')
            seen.add(call_id)
        costs[key]=sum(c['spend_settlement']['peak_price_cost_upper_bound_nano'] for c in calls)/1e9
        feedback=read(path/'protocol/operation-feedback.json') if row['method']!='central_rule' else {'receipts':[]}
        refusals[key]=sum(not r['applied'] for r in feedback['receipts'])
    require(seen==set(state['reservations']),'PROVIDER_LEDGER_HAS_UNINDEXED_REQUESTS')
    checks={'source_index':ref(index_path),'design':ref(design_path),'protocol':plan.provider_protocol.model_dump(mode='json'),
        'provider':plan.provider,'independent_tasks':len(tasks),'split_check':split_check,
        'all_planned_records_retained':True,'raw_call_accounting_complete':True,'provider_role':plan.study_role}
    return plan,tasks,index,results,costs,refusals,checks


def _vectors(tasks,results):
    ids=[t.task_id for t in tasks]
    strata=np.array([int(t.prompt_resources['stratum_index']) for t in tasks])
    matrix=np.array([[float(bool(results[(tid,m)]['A_cert'])) for m in JOINT_METHODS] for tid in ids])
    return ids,strata,matrix


def plan_primary_provider(index_path,design_path,output):
    plan,tasks,index,results,costs,refusals,checks=load_completed_provider_pilot(index_path,design_path)
    require(plan.provider=='zhipu' and plan.study_role=='primary_power_planning' and len(tasks)==30,
        'GLM_THIRTY_TASK_PRIMARY_ONLY',integrity=False)
    design=load_research_model(design_path,PlanningDesign);ids,strata,matrix=_vectors(tasks,results)
    require(np.bincount(strata).tolist()==[10,10,10],'GLM_PRIMARY_STRATUM_COUNTS_CHANGED')
    output=Path(output).resolve();require(not output.exists(),'IMMUTABLE_PROVIDER_PLANNING_OUTPUT_EXISTS')
    output.mkdir(parents=True);durable_json(output/'task_vectors.json',{'task_ids':ids,'strata':strata.tolist(),
        'methods':JOINT_METHODS,'certification_matrix':matrix.tolist(),'provider':'zhipu','source':'actual_primary_only'})
    rng=np.random.default_rng(design.simulation_seed);alpha=design.family_alpha/4;primary=[];summaries=[]
    for contrast in design.contrasts:
        weights=np.array([contrast.weights.get(m,0.) for m in JOINT_METHODS]);values=PATTERNS@weights
        observed=paired_interval(matrix,strata,weights,alpha);boot=[]
        for s in range(3):
            block=(matrix@weights)[strata==s]
            boot.append(np.var(rng.choice(block,size=(design.bootstrap_draws,len(block)),replace=True),axis=1,ddof=1))
        observed['bootstrap_variance_p90']=float(np.quantile(np.mean(boot,axis=0),.9));scenarios=[]
        for pseudo in design.joint_pattern_pseudocounts:
            base=joint_probabilities(matrix,strata,pseudo)
            for delta in sorted(set(design.delta_sensitivity+[contrast.delta])):
                detail=_simulate_grid(base,values,delta,design,alpha,lambda *_:rng)
                detail.update(contrast=contrast.id,pseudocount_per_stratum=pseudo,target_delta=delta,
                    joint_base_probabilities=base.tolist(),source='zhipu_primary_only')
                scenarios.append(detail)
                if delta==contrast.delta:primary.append(detail)
        common=common_passing_counts([s for s in scenarios if s['target_delta']==contrast.delta],design)
        summary={'contrast':contrast.id,'rq':contrast.rq,'delta':contrast.delta,'alpha':alpha,**observed,
            'common_passing_task_counts':common,'common_task_count':next(iter(common),None),
            'distribution_free_precision_task_count':balanced_size(float(np.ptp(values))**2*math.log(8/design.family_alpha)/(2*design.target_half_width**2))}
        summaries.append(summary);durable_json(output/(contrast.id+'-simulation.json'),{'summary':summary,'scenarios':scenarios})
        print(contrast.id+' GLM primary planning complete',flush=True)
    common=common_passing_counts(primary,design)
    messages=np.array([[results[(tid,m)]['budget_snapshot']['used']['messages'] for m in ['cdr_s1_v1','single']] for tid in ids])
    continuous=paired_interval(messages,strata,[1.,-1.],alpha);variance=continuous['within_stratum_variance']*design.continuous_variance_inflation
    continuous.update(unit='protocol_messages',delta=design.continuous_cost_delta_messages,variance_inflation=design.continuous_variance_inflation,
        approximate_power_task_count=None if variance==0 else balanced_size(variance*(norm.ppf(1-alpha/2)+norm.ppf(design.target_power))**2/design.continuous_cost_delta_messages**2),
        approximate_precision_task_count=None if variance==0 else balanced_size(variance*norm.ppf(1-alpha/2)**2/design.target_half_width**2))
    method_diagnostics={}
    for m in METHODS:
        certified=sum(bool(results[(tid,m)]['A_cert']) for tid in ids)
        failed=sum(results[(tid,m)]['terminal_state']=='METHOD_FAILED' for tid in ids)
        method_diagnostics[m]={'tasks':30,'certified':certified,'completion_wilson_95':wilson_interval(certified,30),
            'method_failed':failed,'failure_rate_upper_95':event_upper_bound(failed,30),
            'operation_rejections':sum(refusals[(tid,m)] for tid in ids),
            'calls':sum(results[(tid,m)]['common_contract']['live_calls'] for tid in ids)}
    candidate={'schema_version':5,'study_stage':'planning','planning_kind':'provider_separated_glm_primary',
        'provider':'zhipu','model_id':plan.model.model_id,'pilot_plan_id':plan.plan_id,'source_hash':index['source_hash'],
        'protocol':checks['protocol'],'pilot_independent_tasks':30,'primary_contrasts':summaries,
        'common_passing_task_counts':common,'recommended_balanced_task_count':next(iter(common),None),
        'recommended_llm_repeats_per_method_task':1,'recommended_market_repeats_per_policy':0,
        'continuous_confirmation_communication_cost':continuous,'method_diagnostics':method_diagnostics,
        'currency':plan.currency,'price_estimated_token_cost':sum(costs.values()),'actual_provider_bill':None,
        'resource_grid':[{'tasks':n,'episodes':n*9,'live_calls':n*72,'maximum_input_token_reservation':n*72*24576,
            'maximum_output_token_reservation':n*72*768,'independent_audit_candidate_evaluations':n*9,'market_runs':0,
            'maximum_additional_reservation':n*72*(24576*plan.input_price_nano+768*plan.output_price_nano)/1e9,
            'currency':plan.currency,'requires_finite_allocation':True} for n in design.candidate_task_counts],
        'no_cross_provider_power_pooling':True,'secondary_provider_tasks_in_primary_N':0,
        'formal_status':'NOT_FROZEN','formal_execution_permitted':False,
        'blockers':([] if common else ['NO_COMMON_REGISTERED_N_PASSES_ALL_PRIMARY_SCENARIOS'])+['INDEPENDENT_GLM_F73_FREEZE_REQUIRED']}
    durable_json(output/'candidate_plan.json',candidate);durable_json(output/'design.json',design.model_dump(mode='json'))
    snapshots=[]
    for name in ('core/research_analysis/provider_planning.py','core/research_analysis/google_recovery_planning.py',
                 'core/research_analysis/analysis.py','scripts/research/plan_provider_study.py','requirements-cscwd.lock'):
        target=output/'planning_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target);snapshots.append(target)
    sources=[Path(index_path),Path(design_path),Path(index_path).parent/'COMPLETED.json',
        Path(index_path).parent/'registration.json',Path(index_path).parent/'spend_final_snapshot.json',Path(checks['protocol']['path']),*snapshots]
    durable_json(output/'power_precision_receipt.json',{'schema_version':5,'status':'PRIMARY_PLANNING_COMPLETED_REQUIRES_FORMAL_FREEZE',
        'input_hashes':{str(p.resolve()):file_sha256(p) for p in sources},
        'artifact_hashes':{str(p.resolve()):file_sha256(p) for p in sorted(output.rglob('*.json'))},
        'checks':checks,'seed':design.simulation_seed,'simulation_draws':design.simulation_draws,
        'same_N_joint_criterion':True,'provider':'zhipu','secondary_tasks_used':0,'formal_freeze_created':False})
    return candidate


def equal_provider_contrast(blocks,weights):
    """Nine equally weighted provider-by-stratum means, never a 48-row pool."""
    require(set(blocks)==set(PROVIDERS),'THREE_PROVIDER_SECONDARY_REQUIRED')
    groups=[]
    for provider in ('zhipu','deepseek','gemini'):
        matrix,strata=blocks[provider];values=np.asarray(matrix)@weights
        for s in range(3):
            group=values[np.asarray(strata)==s]
            require(len(group)>=2,'SECONDARY_STRATUM_TOO_SMALL',integrity=False);groups.append(group)
    components=np.array([np.var(g,ddof=1)/(81*len(g)) for g in groups]);variance=float(components.sum())
    estimate=float(np.mean([np.mean(g) for g in groups]))
    if variance==0:return {'estimate':estimate,'approximate_interval':None,'zero_variance_caution':True,'descriptive_only':True}
    denominator=sum(c*c/(len(g)-1) for c,g in zip(components,groups));df=variance*variance/denominator
    half=float(t.ppf(.975,df)*math.sqrt(variance))
    return {'estimate':estimate,'standard_error':math.sqrt(variance),'degrees_of_freedom':df,
        'approximate_interval':[estimate-half,estimate+half],'zero_variance_caution':False,'descriptive_only':True}


def secondary_provider_report(index_paths,output):
    require(len(index_paths)==3,'THREE_DISTINCT_PROVIDER_INDEXES_REQUIRED');loaded={}
    for path in index_paths:
        data=load_completed_provider_pilot(path);provider=data[0].provider
        require(provider not in loaded,'DUPLICATE_PROVIDER_INDEX');loaded[provider]=data
    require(set(loaded)==set(PROVIDERS),'THREE_PROVIDER_SECONDARY_REQUIRED')
    blocks={};individual={};design=None
    for provider,data in loaded.items():
        plan,tasks,index,results,costs,refusals,checks=data
        current=load_research_model(plan.planning_design_path,PlanningDesign)
        if design is None:design=current
        require(current.contrasts==design.contrasts,'SECONDARY_COMPARISON_FAMILY_CHANGED')
        ids,strata,matrix=_vectors(tasks,results);blocks[provider]=(matrix,strata)
        individual[provider]={'independent_tasks':len(tasks),'provider_model':plan.model.model_id,
            'scope':'descriptive_provider_specific','contrasts':{c.id:paired_interval(matrix,strata,
                [c.weights.get(m,0.) for m in JOINT_METHODS],.05) for c in design.contrasts},
            'price_estimated_token_cost':sum(costs.values()),'currency':plan.currency}
    contrasts={c.id:equal_provider_contrast(blocks,np.array([c.weights.get(m,0.) for m in JOINT_METHODS])) for c in design.contrasts}
    report={'schema_version':5,'study_stage':'secondary_provider_report','providers':individual,
        'equal_provider_contrasts':contrasts,'weighting':'one_third_per_provider_one_third_per_conflict_stratum',
        'primary_N_increased':False,'powered_claim':False,'confirmatory_test_performed':False,
        'source_indexes':[ref(p) for p in index_paths],'protocol_sha256':PROTOCOL_SHA256}
    output=Path(output).resolve();require(not output.exists(),'IMMUTABLE_SECONDARY_REPORT_EXISTS')
    output.mkdir(parents=True);durable_json(output/'provider_report.json',report)
    return report

