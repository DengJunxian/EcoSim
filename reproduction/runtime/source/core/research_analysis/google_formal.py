"""Google formal execution with fixed batches and post-selection exact audit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import time
import urllib.request

from core.codesign.schema import TaskSpec
from core.codesign.runner import run_design_episode, verify_episode, freeze_design_source, design_implementation_manifest
from core.codesign.live_protocol import SharedEvidenceProtocolMethod
from core.codesign.llm_adapter import DesignAgentClient
from core.codesign.baselines import CentralRuleMethod
from core.llm.gemini_client import load_gemini_settings
from core.experiments.archive import durable_json
from core.experiments.identity import research_hash,file_sha256
from core.experiments.strict_types import load_research_model,strict_json_loads
from .study_freeze import FileRef,check_ref,reference,FreezeIntegrityError,FreezeGateError
from .formal_execution import EvaluatorLedger,freeze_selection,audit_frozen_selection,_write_immutable
from .google_freeze import verify_google_envelope


def verify_history(current,previous):
    if current['contract']!=previous['contract']:raise FreezeIntegrityError('GOOGLE_AUTHORIZATION_CHANGED')
    if current['created_at']!=previous['created_at']:raise FreezeIntegrityError('GOOGLE_ALLOCATION_ORIGIN_CHANGED')
    for group in ('reservations','settlements'):
        if any(current[group].get(k)!=v for k,v in previous[group].items()):
            raise FreezeIntegrityError('GOOGLE_HISTORY_CHANGED')
    if previous['blocked_reason'] and current['blocked_reason']!=previous['blocked_reason']:
        raise FreezeIntegrityError('GOOGLE_BLOCK_RESET')


def model_preflight(settings,config):
    url=settings.base_url+'/models/'+config.model_id
    request=urllib.request.Request(url,headers={'x-goog-api-key':settings.api_key})
    with urllib.request.urlopen(request,timeout=20) as response:
        body=json.loads(response.read())
    if body.get('name')!='models/'+config.model_id or 'generateContent' not in body.get('supportedGenerationMethods',[]):
        raise FreezeGateError('GOOGLE_MODEL_NOT_AVAILABLE')
    return {'endpoint':url,'http_status':200,'model':body,'generation_check':'same_config_completed_independent_pilot'}


def run_google_formal(envelope_path,output,*,credential_path):
    from .google_usage import GoogleUsageLedger
    core,analysis,receipt=verify_google_envelope(envelope_path)
    out=Path(output).resolve()
    if out.exists():raise FreezeIntegrityError('IMMUTABLE_FORMAL_OUTPUT_ALREADY_EXISTS')
    settings=load_gemini_settings(credential_path,timeout_seconds=core.model.timeout_seconds)
    available=model_preflight(settings,core.model)
    tasks={t.task_id:t for t in [load_research_model(check_ref(r),TaskSpec) for r in core.tasks]}
    for x in core.schedule:
        if Path(x['episode_id']).name!=x['episode_id'] or '\\' in x['episode_id']:
            raise FreezeIntegrityError('UNSAFE_EPISODE_ID')
    def source_guard():
        if design_implementation_manifest().source_hash!=core.source_hash:raise FreezeIntegrityError('FORMAL_SOURCE_CHANGED')
        for r in core.scientific_inputs:check_ref(r)
        check_ref(core.authorization.ledger)
    source_guard()
    auth=core.authorization
    spend=GoogleUsageLedger(auth.usage_ledger_path,authorization_id=auth.authorization_id,plan_id=core.plan_id,
        plan_hash=research_hash(core),maximum_calls=auth.maximum_generation_calls,
        input_cap=core.model.input_reservation_tokens,output_cap=core.model.max_output_tokens,
        input_nano=auth.input_price_nano,output_nano=auth.output_price_nano,cached_nano=auth.cached_price_nano,
        source_guard=source_guard,parent_reference=auth.ledger.model_dump(mode='json'))
    if spend.snapshot()['reservations']:raise FreezeIntegrityError('FORMAL_ALLOCATION_ALREADY_USED')
    out.mkdir(parents=True,exist_ok=False);manifest=freeze_design_source(out)
    core_ref=reference(Path(envelope_path).parent/'PlanCore.json')
    _write_immutable(out/'registration.json',{'schema_version':2,'study_stage':'formal',
        'origin':'real_execution','envelope':reference(envelope_path).model_dump(mode='json'),
        'plan_core':core_ref.model_dump(mode='json'),'source_hash':manifest.source_hash,'schedule':core.schedule,
        'registered_at':time.time(),'model_check':available,'execution_workers':16,'tasks_per_batch':2,
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
        durable_json(out/'audit_index.json',{'schema_version':2,'study_stage':'formal','episodes':audits,
            'audit_plan_hash':research_hash(core.audit_plan),'design_feedback':False,'evaluator_budget':evaluator.state,
            'coverage':{'policy_object_denominator':len(objects),'decidable_policy_objects':sum(x['truth']!='UNKNOWN' for x in objects),
                'unknown_policy_objects':sum(x['truth']=='UNKNOWN' for x in objects),'raw_policy_mention_denominator':len(raw),
                'uncovered_raw_policy_mentions':sum(not x['covered_policy'] for x in raw),
                'no_applicable_object_episodes':sum(a['status']=='NO_APPLICABLE_OBJECT' for a in audits)}})
        unrun=sum(r['status']=='NOT_RUN_STOPPING_RULE' for r in episodes)
        integrity_complete=(len(episodes)==len(core.schedule) and unrun==0
            and not any(r['status']=='EXECUTION_FAILED' for r in episodes)
            and len(audits)==len(core.schedule) and not any(a['status']=='FAILED' for a in audits))
        index={'schema_version':2,'study_stage':'formal','scope':'exact_analytic','plan_id':core.plan_id,'source_hash':core.source_hash,
            'registration':reference(out/'registration.json').model_dump(mode='json'),'episodes':episodes,
            'planned_trials':len(core.schedule),'indexed_trials':len(episodes),'started_trials':len(episodes)-unrun,'not_run_trials':unrun,
            'recorded_trials':sum(r['status']=='RECORDED' for r in episodes),
            'execution_failed_trials':sum(r['status']=='EXECUTION_FAILED' for r in episodes),
            'method_failed_trials':sum(r.get('terminal_state')=='METHOD_FAILED' for r in episodes),
            'all_registered_trials_have_terminal_records':len(episodes)==len(core.schedule) and unrun==0,
            'formal_status':'EXECUTED' if integrity_complete else 'INCOMPLETE_FORMAL',
            'all_design_and_audit_integrity_checks_passed':integrity_complete,
            'scientific_effects_status':'NOT_ANALYZED','stopping_reason':stopping,'spend_snapshot':current,
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
                    allow_live=True,settings=settings,spend_ledger=spend),turns=core.design_turns)
            result=run_design_episode(actual,implementation,dest,episode_id=item['episode_id']);verification=verify_episode(dest)
            if result.source_hash!=core.source_hash or result.task_hash!=actual.task_hash:raise FreezeIntegrityError('EXECUTED_TASK_OR_SOURCE_MISMATCH')
            if (result.episode_id,result.task_id,result.method)!=(item['episode_id'],item['task_id'],method):
                raise FreezeIntegrityError('EXECUTED_EPISODE_IDENTITY_MISMATCH')
            if source=='live' and result.common_contract['live_calls'] and result.live_status!='RUN':raise FreezeIntegrityError('LIVE_SOURCE_MISMATCH')
            if source=='live' and not result.common_contract['live_calls'] and result.terminal_state not in {'METHOD_FAILED','BUDGET_UNRESOLVED','BUDGET_EXHAUSTED_UNRESOLVED'}:
                raise FreezeIntegrityError('LIVE_TERMINAL_WITHOUT_PROVIDER_ATTEMPT')
            if source=='rule' and result.common_contract['live_calls']:
                raise FreezeIntegrityError('RULE_METHOD_CONTAINS_LIVE_CALLS')
            terminal=dest/'episode_result.json';row.update(status='RECORDED',path=str(dest),terminal_state=result.terminal_state,
                verification=verification,live_calls=result.common_contract['live_calls'],result_sha256=file_sha256(terminal),
                natural_trace=source=='live' and result.live_status=='RUN',reference_only=source=='rule')
        except (ValueError,RuntimeError,OSError) as exc:
            terminal=out/'failed_attempts'/(item['episode_id']+'.json')
            _write_immutable(terminal,{**item,'status':'EXECUTION_FAILED','error_type':type(exc).__name__,'reason':str(exc)})
            row.update(status='EXECUTION_FAILED',reason=str(exc),error_type=type(exc).__name__)
        selected=freeze_selection(task,terminal,out/'selections'/(item['episode_id']+'.json'),episode_id=item['episode_id'],
            source=source,plan_core=core_ref,disposition=row['status'])
        row['selection_record']=selected.model_dump(mode='json');return row
    size=core.tasks_per_batch*len(core.methods)
    for offset in range(0,len(core.schedule),size):
        batch=core.schedule[offset:offset+size];batch_id=f'batch-{offset//size:05d}'
        if not stopping:
            source_guard();verify_google_envelope(envelope_path)
            spend.begin_batch(batch_id,[x['episode_id'] for x in batch])
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
            except (ValueError,RuntimeError,OSError) as exc:
                stopping='INDEPENDENT_AUDIT_INTEGRITY_FAILURE';audit={'episode_id':row['episode_id'],'status':'FAILED','reason':str(exc),'truth':'UNKNOWN','design_feedback':False}
                _write_immutable(auditdir/'audit_failure.json',audit);row['audit_result']=reference(auditdir/'audit_failure.json').model_dump(mode='json')
            audits.append(audit);episodes.append(row);job({'episode_id':row['episode_id'],'stage':'TERMINAL_AND_AUDIT_RECORDED','status':row['status'],'recorded_at':time.time()})
        index=save();print(json.dumps({'batch':batch_id,'indexed':len(episodes),'planned':len(core.schedule),'stopping':stopping}),flush=True)
    index=save();_write_immutable(out/'COMPLETED.json',{'index_sha256':file_sha256(out/'benchmark_index.json'),
        'audit_index_sha256':file_sha256(out/'audit_index.json'),'complete_planned_index':len(episodes)==len(core.schedule),
        'formal_execution_complete':index['formal_status']=='EXECUTED'})
    return index
