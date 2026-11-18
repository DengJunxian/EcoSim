"""Provider formal execution with fixed batches and post-selection exact audit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import time
import importlib.metadata

from core.codesign.schema import TaskSpec
from core.codesign.runner import run_design_episode, verify_episode, freeze_design_source, design_implementation_manifest
from core.codesign.live_protocol import SharedEvidenceProtocolMethod
from core.codesign.llm_adapter import DesignAgentClient
from core.codesign.baselines import CentralRuleMethod
from .provider_pilot import load_provider_settings,ProviderRequestGate,ControlledProviderLedger,validate_episode_identity
from .provider_planning import validate_admission
from core.codesign.traces import export_trace_collection
from core.experiments.archive import durable_json
from core.experiments.identity import research_hash,file_sha256
from core.experiments.strict_types import load_research_model,strict_json_loads
from .study_freeze import FileRef,check_ref,reference,FreezeIntegrityError,FreezeGateError
from .formal_execution import EvaluatorLedger,freeze_selection,audit_frozen_selection,_write_immutable
from .provider_freeze import verify_provider_envelope,validate_natural_registration


def verify_history(current,previous):
    if current['contract']!=previous['contract']:raise FreezeIntegrityError('PROVIDER_AUTHORIZATION_CHANGED')
    if current['created_at']!=previous['created_at']:raise FreezeIntegrityError('PROVIDER_ALLOCATION_ORIGIN_CHANGED')
    for group in ('reservations','settlements'):
        if any(current[group].get(k)!=v for k,v in previous[group].items()):
            raise FreezeIntegrityError('PROVIDER_HISTORY_CHANGED')
    if previous['blocked_reason'] and current['blocked_reason']!=previous['blocked_reason']:
        raise FreezeIntegrityError('PROVIDER_BLOCK_RESET')


def audit_has_execution_failure(audit):
    """An evaluator exception remains a failure even when its envelope is recorded.

    UNRESOLVED truth and NO_APPLICABLE_OBJECT are retained semantic outcomes;
    they are not silently converted into execution failures.
    """
    return audit['status']=='FAILED' or any(obj['status']=='FAILED' for obj in audit.get('objects',[]))


def run_provider_formal(envelope_path,output,*,credential_path):
    from .provider_usage import ProviderUsageLedger
    core,analysis,receipt=verify_provider_envelope(envelope_path)
    out=Path(output).resolve()
    if out.exists():raise FreezeIntegrityError('IMMUTABLE_FORMAL_OUTPUT_ALREADY_EXISTS')
    if importlib.metadata.version('openai')!=core.model.sdk_version:raise FreezeIntegrityError('PROVIDER_RUNTIME_SDK_CHANGED')
    validate_natural_registration(core)
    settings=load_provider_settings(core,credential_path)
    available={'model_id':core.model.model_id,'generation_check':'same_config_completed_independent_glm_primary_pilot',
        'new_generation_calls':0}
    tasks={t.task_id:t for t in [load_research_model(check_ref(r),TaskSpec) for r in core.tasks]}
    for x in core.schedule:
        if Path(x['episode_id']).name!=x['episode_id'] or '\\' in x['episode_id']:
            raise FreezeIntegrityError('UNSAFE_EPISODE_ID')
    startup_manifest=design_implementation_manifest()
    if startup_manifest.source_hash!=core.source_hash:raise FreezeIntegrityError('FORMAL_SOURCE_CHANGED')
    runtime_source_hashes={r.path:r.sha256 for r in core.scientific_inputs if r.role in {'source','dependency_lock'}}
    runtime_hash=research_hash(runtime_source_hashes)
    def source_guard():
        # Unchanged entry/import files cannot introduce a different static closure.
        # Every file from the parsed closure and all other bindings is rehashed.
        # This avoids repeating AST parsing under the GIL for each request.
        for r in core.scientific_inputs:check_ref(r)
        check_ref(core.authorization.ledger)
    source_guard()
    auth=core.authorization
    spend=ProviderUsageLedger(auth.usage_ledger_path,authorization_id=auth.authorization_id,plan_id=core.plan_id,
        plan_hash=research_hash(core),maximum_calls=auth.maximum_generation_calls,
        input_cap=core.model.input_reservation_tokens,output_cap=core.model.max_output_tokens,
        input_nano=auth.input_price_nano,output_nano=auth.output_price_nano,cached_nano=auth.cached_price_nano,
        source_guard=source_guard,parent_reference=auth.ledger.model_dump(mode='json'),provider='zhipu',currency='CNY',
        monetary_cap_nano=None,inherited_liability_nano=0)
    if spend.snapshot()['reservations']:raise FreezeIntegrityError('FORMAL_ALLOCATION_ALREADY_USED')
    out.mkdir(parents=True,exist_ok=False);manifest=freeze_design_source(out)
    gate=ProviderRequestGate(core.maximum_concurrent_requests,out/'request_admission.json');gate._save()
    core_ref=reference(Path(envelope_path).parent/'PlanCore.json')
    _write_immutable(out/'registration.json',{'schema_version':5,'study_stage':'formal',
        'origin':'real_execution','source_integrity_algorithm':core.source_integrity_algorithm,
        'runtime_source_hashes':runtime_source_hashes,'runtime_hash':runtime_hash,
        'scope_declaration':core.scope_declaration,'envelope':reference(envelope_path).model_dump(mode='json'),
        'plan_core':core_ref.model_dump(mode='json'),'source_hash':manifest.source_hash,'schedule':core.schedule,
        'registered_at':time.time(),'model_check':available,'provider':'zhipu',
        'maximum_concurrent_requests':2,'minimum_request_interval_seconds':0.0,
        'host_queue_time_counts_toward_original_episode_wall_budget':True,
        'source_reservation':core.source_reservation.model_dump(mode='json'),
        'trace_collection_registration':core.trace_collection_registration.model_dump(mode='json'),'execution_workers':16,'tasks_per_batch':2,
        'audit_plan_hash':research_hash(core.audit_plan),'scope':'exact_analytic','scientific_effects_status':'NOT_ANALYZED'})
    _write_immutable(out/'spend_before.json',spend.snapshot())
    evaluator=EvaluatorLedger(out/'evaluator_budget.json',core.audit_plan.evaluator_budget.candidate_evaluations)
    episodes=[];audits=[];stopping=None;previous=spend.snapshot();jobs=out/'jobs.jsonl';jobs.touch(exist_ok=False)
    def job(body):
        with jobs.open('a') as f:
            f.write(json.dumps(body,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    def save():
        nonlocal previous,stopping
        current=spend.snapshot();verify_history(current,previous);previous=current
        objects=[x for a in audits for x in a.get('objects',[])];raw=[x for a in audits for x in a.get('raw_claim_coverage',[])]
        durable_json(out/'audit_index.json',{'schema_version':5,'study_stage':'formal','episodes':audits,
            'audit_plan_hash':research_hash(core.audit_plan),'design_feedback':False,'evaluator_budget':evaluator.state,
            'coverage':{'policy_object_denominator':len(objects),'decidable_policy_objects':sum(x['truth']!='UNKNOWN' for x in objects),
                'unknown_policy_objects':sum(x['truth']=='UNKNOWN' for x in objects),
                'failed_policy_objects':sum(x['status']=='FAILED' for x in objects),
                'episode_audit_failures':sum(a['status']=='FAILED' for a in audits),'raw_policy_mention_denominator':len(raw),
                'uncovered_raw_policy_mentions':sum(not x['covered_policy'] for x in raw),
                'no_applicable_object_episodes':sum(a['status']=='NO_APPLICABLE_OBJECT' for a in audits)}})
        unrun=sum(r['status']=='NOT_RUN_STOPPING_RULE' for r in episodes)
        # Completing the last scheduled batch cannot erase unknown resource
        # usage or a provider/model identity violation. Recorded language
        # METHOD_FAILED outcomes remain valid terminal observations.
        allowed_stop={None,'THREE_FAILED_EPISODES_IN_FIXED_BATCH'}
        provider_integrity=(current['blocked_reason'] in allowed_stop
            and all(reason in allowed_stop for reason in current['pending_block_reasons'])
            and current['active_batch'] is None
            and set(current['reservations'])==set(current['settlements'])
            and all(s['usage_status']=='known' for s in current['settlements'].values()))
        integrity_complete=(provider_integrity and len(episodes)==len(core.schedule) and unrun==0
            and not any(r['status']=='EXECUTION_FAILED' for r in episodes)
            and len(audits)==len(core.schedule) and not any(audit_has_execution_failure(a) for a in audits))
        index={'schema_version':5,'study_stage':'formal','scope':'exact_analytic','plan_id':core.plan_id,'source_hash':core.source_hash,
            'registration':reference(out/'registration.json').model_dump(mode='json'),'episodes':episodes,
            'planned_trials':len(core.schedule),'indexed_trials':len(episodes),'started_trials':len(episodes)-unrun,'not_run_trials':unrun,
            'recorded_trials':sum(r['status']=='RECORDED' for r in episodes),
            'execution_failed_trials':sum(r['status']=='EXECUTION_FAILED' for r in episodes),
            'method_failed_trials':sum(r.get('terminal_state')=='METHOD_FAILED' for r in episodes),
            'all_registered_trials_have_terminal_records':len(episodes)==len(core.schedule) and unrun==0,
            'formal_status':'EXECUTED' if integrity_complete else 'INCOMPLETE_FORMAL',
            'all_design_and_audit_integrity_checks_passed':integrity_complete,
            'all_provider_resource_and_identity_checks_passed':provider_integrity,
            'scientific_effects_status':'NOT_ANALYZED','stopping_reason':stopping,
            'provider':'zhipu','runtime_hash':runtime_hash,'scope_declaration':core.scope_declaration,'request_admission_sha256':file_sha256(out/'request_admission.json'),'spend_snapshot':current,
            'spend_history_integrity':{'status':'VERIFIED'},'jobs':reference(jobs).model_dump(mode='json'),
            'audit_index':reference(out/'audit_index.json').model_dump(mode='json')}
        durable_json(out/'episode_index.json',{'episodes':episodes});durable_json(out/'benchmark_index.json',index)
        return index
    def run_one(item):
        task=tasks[item['task_id']];method=item['method'];source='rule' if method=='central_rule' else 'live'
        dest=out/'episodes'/item['episode_id'];row={**item,'source':source}
        try:
            actual=task
            if source=='rule':
                b=task.model_dump(mode='json');b.update(signing_layer='explicit_rule_owners_v1',model_configurations=[])
                actual=TaskSpec.model_validate(b);implementation=CentralRuleMethod(protocol_enabled=True)
            else:
                implementation=SharedEvidenceProtocolMethod(method,lambda ledger,path:DesignAgentClient(core.model,ledger,path,
                    allow_live=True,settings=settings,spend_ledger=ControlledProviderLedger(spend,gate,ledger,source_guard)),turns=core.design_turns)
            result=run_design_episode(actual,implementation,dest,episode_id=item['episode_id']);verification=verify_episode(dest)
            validate_episode_identity(result,actual,item,core.source_hash)
            terminal=dest/'episode_result.json';row.update(status='RECORDED',path=str(dest),terminal_state=result.terminal_state,
                verification=verification,live_calls=result.common_contract['live_calls'],result_sha256=file_sha256(terminal),
                natural_trace=source=='live' and result.live_status=='RUN',reference_only=source=='rule')
        except Exception as exc:
            terminal=out/'failed_attempts'/(item['episode_id']+'.json')
            _write_immutable(terminal,{**item,'status':'EXECUTION_FAILED','error_type':type(exc).__name__,'reason':str(exc)})
            row.update(status='EXECUTION_FAILED',reason=str(exc),error_type=type(exc).__name__)
        finally:
            gate.finish_episode(item['episode_id'])
        selected=freeze_selection(task,terminal,out/'selections'/(item['episode_id']+'.json'),episode_id=item['episode_id'],
            source=source,plan_core=core_ref,disposition=row['status'])
        row['selection_record']=selected.model_dump(mode='json');return row
    size=core.tasks_per_batch*len(core.methods)
    for offset in range(0,len(core.schedule),size):
        batch=core.schedule[offset:offset+size];batch_id=f'batch-{offset//size:05d}'
        if not stopping:
            try:
                source_guard();verify_provider_envelope(envelope_path)
                spend.begin_batch(batch_id,[x['episode_id'] for x in batch])
            except Exception as exc:
                stopping='PRE_BATCH_INTEGRITY_FAILURE:'+type(exc).__name__
        if not stopping:
            for item in batch:job({**item,'stage':'DESIGN_STARTED','recorded_at':time.time()})
            with ThreadPoolExecutor(max_workers=core.execution_workers) as pool:rows=list(pool.map(run_one,batch))
            failures=sum(r['status']=='EXECUTION_FAILED' or r.get('terminal_state')=='METHOD_FAILED' for r in rows)
            reason='THREE_FAILED_EPISODES_IN_FIXED_BATCH' if failures>=3 else None
            if any(r['status']=='EXECUTION_FAILED' for r in rows):reason='EXECUTION_INTEGRITY_OR_SERVICE_FAILURE'
            spend.close_batch(batch_id,stop_reason=reason);stopping=spend.snapshot()['blocked_reason']
        else:
            rows=[]
            for item in batch:
                source='rule' if item['method']=='central_rule' else 'live';terminal=out/'unrun'/(item['episode_id']+'.json')
                _write_immutable(terminal,{**item,'status':'NOT_RUN_STOPPING_RULE','reason':stopping})
                selected=freeze_selection(tasks[item['task_id']],terminal,out/'selections'/(item['episode_id']+'.json'),
                    episode_id=item['episode_id'],source=source,plan_core=core_ref,disposition='NOT_RUN_STOPPING_RULE')
                rows.append({**item,'source':source,'status':'NOT_RUN_STOPPING_RULE','reason':stopping,'selection_record':selected.model_dump(mode='json')})
        # Every selection in the batch is immutable before its independent audit is opened.
        for row in rows:
            auditdir=out/'audits'/row['episode_id'];selection=FileRef.model_validate(row['selection_record'])
            try:
                audit=audit_frozen_selection(selection,tasks[row['task_id']],lambda:core.audit_plan.model_dump(mode='json'),
                    auditdir,source_hash=core.source_hash,evaluator_ledger=evaluator)
                row['audit_result']=reference(auditdir/'audit_result.json').model_dump(mode='json')
            except Exception as exc:
                stopping='INDEPENDENT_AUDIT_INTEGRITY_FAILURE';audit={'episode_id':row['episode_id'],'status':'FAILED','reason':str(exc),'truth':'UNKNOWN','design_feedback':False}
                _write_immutable(auditdir/'audit_failure.json',audit);row['audit_result']=reference(auditdir/'audit_failure.json').model_dump(mode='json')
            if audit_has_execution_failure(audit):stopping='INDEPENDENT_AUDIT_INTEGRITY_FAILURE'
            audits.append(audit);episodes.append(row);job({'episode_id':row['episode_id'],'stage':'TERMINAL_AND_AUDIT_RECORDED','status':row['status'],'recorded_at':time.time()})
        index=save();print(json.dumps({'batch':batch_id,'indexed':len(episodes),'planned':len(core.schedule),'stopping':stopping}),flush=True)
    index=save()
    validate_admission(core,strict_json_loads((out/'request_admission.json').read_text()),spend.snapshot())
    collection=Path(core.trace_collection_plan.path).parent
    source_rows=[r for r in episodes if r['source']=='live']
    try:
        if not all(r['status']=='RECORDED' and r.get('natural_trace') for r in source_rows):
            raise FreezeGateError('COMPLETE_REGISTERED_NATURAL_SOURCE_REQUIRED')
        export_trace_collection(collection,[r['path'] for r in source_rows])
        natural={'status':'COLLECTED','trace_index':reference(collection/'trace-index.json').model_dump(mode='json')}
    except Exception as exc:
        natural={'status':'INCOMPLETE','reason':str(exc),'planned_live_sources':len(source_rows),
            'source_records':[{'episode_id':r['episode_id'],'status':r['status'],'natural_trace':r.get('natural_trace',False)} for r in source_rows]}
    _write_immutable(out/'natural_collection_receipt.json',natural)
    index['natural_collection']=reference(out/'natural_collection_receipt.json').model_dump(mode='json')
    durable_json(out/'benchmark_index.json',index)
    _write_immutable(out/'COMPLETED.json',{'index_sha256':file_sha256(out/'benchmark_index.json'),
        'audit_index_sha256':file_sha256(out/'audit_index.json'),'complete_planned_index':len(episodes)==len(core.schedule),
        'formal_execution_complete':index['formal_status']=='EXECUTED'})
    return index
