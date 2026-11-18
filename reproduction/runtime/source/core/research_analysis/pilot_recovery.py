"""Administrative continuation of an interrupted, immutable pilot schedule.

This does not reconcile provider usage. Unknown calls retain their entire
reservation and unchanged raw settlement. The predecessor remains blocked;
an exclusive successor marker prevents a second continuation ledger.
"""
from __future__ import annotations
import copy
import fcntl
import json
from pathlib import Path
import shutil
import time

from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256, research_hash
from core.codesign.pilot import load_pilot_plan, load_pilot_inputs
from core.codesign.runner import design_implementation_manifest, run_design_episode, verify_episode
from core.codesign.schema import TaskSpec
from core.codesign.live_budget import SpendLedger, load_deepseek_settings
from core.codesign.live_protocol import SharedEvidenceProtocolMethod
from core.codesign.llm_adapter import DesignAgentClient
from core.codesign.baselines import CentralRuleMethod


def admission(index, registration, state, *, calls_per_episode, input_cap, output_cap):
    if index['stopping_reason'] != 'UNKNOWN_USAGE_REQUIRES_RECONCILIATION' or state['blocked_reason'] != index['stopping_reason']:
        raise ValueError('ONLY_UNKNOWN_USAGE_ADMINISTRATIVE_RECOVERY')
    if state != index['spend_snapshot']:
        raise ValueError('PREDECESSOR_LEDGER_CHANGED')
    fields = ('episode_id', 'task_id', 'method')
    expected = [tuple(x[k] for k in fields) for x in registration['schedule']]
    actual = [tuple(x[k] for k in fields) for x in index['episodes']]
    if expected != actual or len(set(actual)) != len(actual):
        raise ValueError('ORIGINAL_SCHEDULE_CHANGED')
    if any(x['status'] not in ('RECORDED', 'NOT_RUN_STOPPING_RULE') for x in index['episodes']):
        raise ValueError('UNKNOWN_EPISODE_STATE')
    pending = [x for x in index['episodes'] if x['status'] == 'NOT_RUN_STOPPING_RULE']
    if not pending or any(x['reason'] != index['stopping_reason'] for x in pending):
        raise ValueError('ONLY_ORIGINAL_UNRUN_TRIALS')
    if set(state['reservations']) != set(state['settlements']):
        raise ValueError('UNSETTLED_ATTEMPT_REQUIRES_SEPARATE_REVIEW')
    unknown = [k for k,v in state['settlements'].items() if v['peak_price_cost_upper_bound_nano'] is None]
    if not unknown:
        raise ValueError('NO_UNKNOWN_LIABILITY')
    pending_ids = {x['episode_id'] for x in pending}
    if any(k.rsplit(':',1)[0] in pending_ids for k in state['reservations']):
        raise ValueError('UNRUN_TRIAL_ALREADY_CHARGED')
    contract = state['contract']
    if contract['reservation_rule'] != 'peak-price-full-cap-no-refund-v1' or contract['cap_nano'] > 100_000_000_000:
        raise ValueError('UNCHANGED_AUTHORIZATION_REQUIRED')
    prior = sum(x['reserved_nano'] for x in state['reservations'].values())
    count = sum(x['method'] != 'central_rule' for x in pending)*calls_per_episode
    extra = count*(input_cap*contract['input_nano']+output_cap*contract['output_nano'])
    if prior+extra > contract['cap_nano']:
        raise ValueError('RECOVERY_EXCEEDS_ORIGINAL_CAP')
    return {'pending_schedule': [{k:x[k] for k in fields} for x in pending],
        'maximum_additional_calls':count, 'maximum_additional_reservation_nano':extra,
        'cumulative_maximum_reservation_nano':prior+extra,
        'unknown_liabilities':{k:state['reservations'][k]['reserved_nano'] for k in unknown},
        'unknown_usage_reconciled':False, 'historical_settlements_modified':False,
        'failed_recorded_episodes_replayed':False}


def create_successor(old_path, new_path, amendment_path):
    old_path,new_path,amendment_path = map(Path,(old_path,new_path,amendment_path))
    amendment = json.loads(amendment_path.read_text())
    contract = json.loads(old_path.read_text())['state']['contract']
    ledger = SpendLedger(old_path,authorization_id=contract['authorization_id'],cap_cny=contract['cap_nano']//10**9,
        input_nano=contract['input_nano'],output_nano=contract['output_nano'],cached_nano=contract['cached_nano'])
    with ledger.lock():
        if file_sha256(old_path) != amendment['predecessor_ledger']['sha256']:
            raise ValueError('PREDECESSOR_LEDGER_CHANGED')
        if new_path.exists():raise ValueError('SUCCESSOR_ALREADY_EXISTS')
        state=ledger.read()
        if state['blocked_reason'] != 'UNKNOWN_USAGE_REQUIRES_RECONCILIATION':raise ValueError('PREDECESSOR_NOT_BLOCKED')
        marker=old_path.with_suffix('.successor.json')
        # O_EXCL gives a single successor even if concurrent processes race.
        with marker.open('x') as handle:
            json.dump({'successor_path':str(new_path.resolve()),'amendment_path':str(amendment_path.resolve()),
                'amendment_sha256':file_sha256(amendment_path),'predecessor_sha256':file_sha256(old_path)},handle,indent=2)
            handle.flush()
        state=copy.deepcopy(state)
        state['administrative_continuation']={'amendment_path':str(amendment_path.resolve()),
            'amendment_sha256':file_sha256(amendment_path),'predecessor_path':str(old_path.resolve()),
            'prior_blocked_reason':state['blocked_reason'], 'unknown_usage_reconciled':False}
        state['blocked_reason']=None
        durable_json(new_path,{'state':state,'sha256':research_hash(state)})
    return new_path


def prepare_recovery(pilot, old_ledger, new_ledger, out):
    pilot,old_ledger,new_ledger,out = map(lambda p:Path(p).resolve(),(pilot,old_ledger,new_ledger,out))
    if out.exists():raise ValueError('RECOVERY_ALREADY_REGISTERED')
    index=json.loads((pilot/'benchmark_index.json').read_text());registration=json.loads((pilot/'registration.json').read_text())
    marker=json.loads((pilot/'COMPLETED.json').read_text())
    if marker['index_sha256']!=file_sha256(pilot/'benchmark_index.json') or marker['registration_sha256']!=file_sha256(pilot/'registration.json'):
        raise ValueError('PREDECESSOR_INDEX_CHANGED')
    plan=load_pilot_plan(pilot/'plan.json');load_pilot_inputs(plan)
    if design_implementation_manifest().source_hash!=index['source_hash']:raise ValueError('FROZEN_RUNTIME_CHANGED')
    raw=json.loads(old_ledger.read_text());state=raw['state']
    if research_hash(state)!=raw['sha256']:raise ValueError('SPEND_HASH_MISMATCH')
    scope=admission(index,registration,state,calls_per_episode=plan.design_turns+3,
        input_cap=plan.model.input_reservation_tokens,output_cap=plan.model.max_output_tokens)
    for trial in index['episodes']:
        if trial['status']=='RECORDED':verify_episode(trial['path'])
    # Narrow administrative exception: empty-usage connection errors only.
    for sid in scope['unknown_liabilities']:
        eid,attempt=sid.rsplit(':',1)
        row=next(x for x in index['episodes'] if x['episode_id']==eid)
        response=Path(row['path'])/'llm'/(attempt+'-response.json')
        obj=json.loads(response.read_text())
        if obj['ok'] or obj['usage'] or obj.get('raw',{}).get('error')!='Connection error.':
            raise ValueError('UNKNOWN_RESPONSE_REQUIRES_SEPARATE_REVIEW')
    out.mkdir(parents=True)
    for name in ('plan.json','registration.json','planning_design.json','experiment_spec.json','implementation_manifest.json','spend_before.json'):
        shutil.copyfile(pilot/name,out/name)
    shutil.copytree(pilot/'inputs',out/'inputs')
    shutil.copytree(pilot/'source_bundle',out/'source_bundle')
    preserve={str(p):file_sha256(p) for p in sorted(pilot.rglob('*')) if p.is_file()}
    durable_json(out/'predecessor_manifest.json',{'files':preserve})
    root=Path(__file__).resolve().parents[2]
    amendment={'schema_version':1,'kind':'administrative_original_batch_continuation_v1','registered_at':time.time(),
        'authorization':'User requested resolving Phase7; original cumulative CNY100 authorization retained.',
        'original_pilot':str(pilot),'original_index_sha256':file_sha256(pilot/'benchmark_index.json'),
        'predecessor_ledger':{'path':str(old_ledger),'sha256':file_sha256(old_ledger)},
        'successor_ledger':str(new_ledger),'runtime_source_hash':index['source_hash'],
        'recovery_source_hashes':{str(p):file_sha256(p) for p in (Path(__file__).resolve(),root/'scripts/research/recover_pilot_v3.py')},
        'unchanged_scientific_inputs':{'plan_hash':index['plan_hash'],'registration_sha256':file_sha256(pilot/'registration.json'),
            'planning_design_sha256':plan.planning_design_sha256,'experiment_spec_sha256':plan.experiment_spec_sha256},
        'policy':'Carry full unknown liability; no release/refund/zero imputation; only original unrun items; stop on any new unknown usage or three consecutive failed episodes.',
        'price_verification':{'date':'2026-10-23','url':'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
            'model':'deepseek-flash','documented_version':'DeepSeek-V4.1-Flash-1017','peak_cny_per_million':{'input':2,'cached_input':0.04,'output':8}},
        **scope}
    durable_json(out/'recovery_registration.json',amendment)
    create_successor(old_ledger,new_ledger,out/'recovery_registration.json')
    return amendment


class RecoveryLedger(SpendLedger):
    def __init__(self,path,amendment,plan):
        self.amendment=amendment;self.plan=plan
        self.allowed={x['episode_id'] for x in amendment['pending_schedule'] if x['method']!='central_rule'}
        super().__init__(path,authorization_id=plan.authorization_id,cap_cny=plan.monetary_cap_cny,
            input_nano=plan.input_price_nano,output_nano=plan.output_price_nano,cached_nano=plan.cached_price_nano)

    def reserve(self,identity,incoming,outgoing,request_hash):
        if identity.rsplit(':',1)[0] not in self.allowed:raise ValueError('RECOVERY_EPISODE_NOT_AUTHORIZED')
        if (incoming,outgoing)!=(self.plan.model.input_reservation_tokens,self.plan.model.max_output_tokens):
            raise ValueError('RECOVERY_REQUEST_CAP_CHANGED')
        count=sum(k.rsplit(':',1)[0] in self.allowed for k in self.snapshot()['reservations'])
        if count>=self.amendment['maximum_additional_calls']:raise ValueError('RECOVERY_CALL_CAP_EXHAUSTED')
        return super().reserve(identity,incoming,outgoing,request_hash)


def execute_recovery(out, credential_path):
    out=Path(out).resolve()
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COMPLETED.json').exists():raise ValueError('RECOVERY_ALREADY_COMPLETED')
        amendment=json.loads((out/'recovery_registration.json').read_text())
        revision_path=out/'pre_call_recovery_revision.json'
        revision=json.loads(revision_path.read_text()) if revision_path.exists() else None
        if revision and revision['original_registration_sha256']!=file_sha256(out/'recovery_registration.json'):
            raise ValueError('RECOVERY_REVISION_PARENT_CHANGED')
        hashes=revision['recovery_source_hashes'] if revision else amendment['recovery_source_hashes']
        for p,h in hashes.items():
            if file_sha256(p)!=h:raise ValueError('RECOVERY_SOURCE_CHANGED')
        old=Path(amendment['original_pilot']);old_index=json.loads((old/'benchmark_index.json').read_text())
        if file_sha256(old/'benchmark_index.json')!=amendment['original_index_sha256']:raise ValueError('PREDECESSOR_INDEX_CHANGED')
        pointer=json.loads(Path(amendment['predecessor_ledger']['path']).with_suffix('.successor.json').read_text())
        if pointer['successor_path']!=amendment['successor_ledger'] or pointer['amendment_sha256']!=file_sha256(out/'recovery_registration.json'):
            raise ValueError('LEDGER_SUCCESSION_MISMATCH')
        plan=load_pilot_plan(out/'plan.json');tasks,_=load_pilot_inputs(plan);task_map={t.task_id:t for t in tasks}
        ledger=RecoveryLedger(amendment['successor_ledger'],amendment,plan)
        settings=load_deepseek_settings(credential_path)
        previous=json.loads((out/'benchmark_index.json').read_text()) if (out/'benchmark_index.json').exists() else old_index
        entries=copy.deepcopy(previous['episodes']);stopping=None;consecutive_failed=0
        def save():
            result={**old_index,'episodes':entries,'stopping_reason':stopping,'spend_snapshot':ledger.snapshot(),
                'administrative_recovery':{'registration_path':str(out/'recovery_registration.json'),
                    'sha256':file_sha256(out/'recovery_registration.json'),'predecessor_index':str(old/'benchmark_index.json'),
                    'pre_call_revision':{'path':str(revision_path),'sha256':file_sha256(revision_path)} if revision else None}}
            durable_json(out/'benchmark_index.json',result)
            return result
        for n,trial in enumerate(entries):
            if trial['status']=='RECORDED':
                verify_episode(trial['path'])
                consecutive_failed=consecutive_failed+1 if trial['terminal_state']=='METHOD_FAILED' else 0
                if consecutive_failed>=3:stopping='THREE_CONSECUTIVE_FAILED_EPISODES'
                continue
            if stopping or ledger.snapshot()['blocked_reason']:
                stopping=stopping or ledger.snapshot()['blocked_reason'];save();break
            if design_implementation_manifest().source_hash!=amendment['runtime_source_hash']:raise ValueError('FROZEN_RUNTIME_CHANGED')
            task=task_map[trial['task_id']];method=trial['method'];destination=out/'episodes'/trial['episode_id']
            if destination.exists():raise ValueError('PARTIAL_RECOVERY_EPISODE_REQUIRES_REVIEW_NO_REPLAY')
            if method=='central_rule':
                body=task.model_dump(mode='json');body.update(signing_layer='explicit_rule_owners_v1',model_configurations=[])
                task=TaskSpec.model_validate(body);implementation=CentralRuleMethod(protocol_enabled=True)
            else:
                implementation=SharedEvidenceProtocolMethod(method,lambda budget,path:DesignAgentClient(plan.model,budget,path,
                    allow_live=True,settings=settings,spend_ledger=ledger),turns=plan.design_turns)
            result=run_design_episode(task,implementation,destination,episode_id=trial['episode_id'])
            entries[n]={k:trial[k] for k in ('episode_id','task_id','method')}
            entries[n].update(status='RECORDED',path=str(destination),terminal_state=result.terminal_state,
                live_calls=result.common_contract['live_calls'],result_sha256=file_sha256(destination/'episode_result.json'),
                verification=verify_episode(destination),natural_trace=method!='central_rule' and result.live_status=='RUN',
                reference_only=method=='central_rule',source='rule' if method=='central_rule' else 'live')
            consecutive_failed=consecutive_failed+1 if result.terminal_state=='METHOD_FAILED' else 0
            if consecutive_failed>=3:stopping='THREE_CONSECUTIVE_FAILED_EPISODES'
            stopping=stopping or ledger.snapshot()['blocked_reason']
            save();print(json.dumps({'episode_id':trial['episode_id'],'terminal':result.terminal_state,'calls':result.common_contract['live_calls']}),flush=True)
        stopping=stopping or ledger.snapshot()['blocked_reason']
        result=save()
        durable_json(out/'COMPLETED.json',{'index_sha256':file_sha256(out/'benchmark_index.json'),
            'registration_sha256':file_sha256(out/'registration.json'),'complete_planned_index':len(entries)==len(old_index['episodes'])})
        return result
