"""Prospectively registered continuation of untouched formal trial batches.

The old ledger stays blocked. Unknown usage stays unknown and consumes its
entire original allocation. Completion of the census is a separate fact from
passing the original strict F70 usage gate.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import copy
import fcntl
import importlib.metadata
import json
import os
import sqlite3
import time

from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256, research_hash
from core.experiments.strict_types import load_research_model, strict_json_loads
from core.codesign.schema import TaskSpec, EpisodeResult
from core.codesign.runner import run_design_episode, verify_episode, design_implementation_manifest
from core.codesign.live_protocol import SharedEvidenceProtocolMethod
from core.codesign.llm_adapter import DesignAgentClient
from core.codesign.baselines import CentralRuleMethod
from .study_freeze import reference, check_ref, FileRef, FreezeIntegrityError
from .formal_execution import EvaluatorLedger, freeze_selection, audit_frozen_selection, _write_immutable
from .provider_freeze import verify_provider_envelope, validate_natural_registration
from .provider_formal import verify_history, audit_has_execution_failure
from .provider_pilot import load_provider_settings, ProviderRequestGate, ControlledProviderLedger, validate_episode_identity
from .provider_usage import ProviderUsageLedger
from .provider_planning import validate_admission

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/research/cscwd/PROVIDER_FORMAL_ADMINISTRATIVE_CONTINUATION_V1.md'
OWN_SOURCES = [Path(__file__).resolve(), ROOT/'scripts/research/recover_provider_formal.py',
               ROOT/'scripts/research/audit_provider_pilot.py', PROTOCOL]
UNKNOWN = 'UNKNOWN_USAGE_REQUIRES_RECONCILIATION'


def require(value, code):
    if not value:
        raise FreezeIntegrityError(code)


def read(path):
    return strict_json_loads(Path(path).read_text())


def ref(path):
    return reference(path).model_dump(mode='json')


def snapshot(path):
    db = sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        body = {k: strict_json_loads(v) for k, v in db.execute('SELECT key,value FROM meta')}
        for name in ('reservations', 'settlements'):
            body[name] = {k: strict_json_loads(v) for k, v in db.execute('SELECT identity,body FROM '+name)}
        return body
    finally:
        db.close()


def admission(core, index, states, *, for_continuation=True):
    """Pure administrative boundary; no outcome-based admission criterion."""
    fields = ('episode_id', 'task_id', 'method')
    identity = lambda row: tuple(row[k] for k in fields)
    require([identity(r) for r in index['episodes']] == [identity(r) for r in core.schedule],
            'ORIGINAL_ORDERED_CENSUS_CHANGED')
    require(len({r['episode_id'] for r in core.schedule}) == len(core.schedule), 'DUPLICATE_ORIGINAL_TRIAL')
    if for_continuation:
        require(index['stopping_reason'] == UNKNOWN and states[-1]['blocked_reason'] == UNKNOWN,
                'ONLY_UNKNOWN_USAGE_ADMINISTRATIVE_CONTINUATION')
    require(index['execution_failed_trials'] == 0, 'EXECUTION_FAILURE_REQUIRES_REVIEW')
    pending = [r for r in index['episodes'] if r['status'] == 'NOT_RUN_STOPPING_RULE']
    require((bool(pending) if for_continuation else not pending)
            and all(r['status'] in {'RECORDED', 'NOT_RUN_STOPPING_RULE'} for r in index['episodes']),
            'ONLY_RECORDED_OR_UNTOUCHED_TRIALS')
    require(all(r['reason'] == UNKNOWN for r in pending), 'UNRUN_STOP_REASON_CHANGED')
    pending_ids = {r['episode_id'] for r in pending}
    first_pending = next((i for i,r in enumerate(index['episodes']) if r['episode_id'] in pending_ids),len(index['episodes']))
    require(all(r['episode_id'] in pending_ids for r in index['episodes'][first_pending:]), 'ONLY_UNTOUCHED_SCHEDULE_SUFFIX')
    size = core.tasks_per_batch*len(core.methods)
    for start in range(0, len(core.schedule), size):
        batch = core.schedule[start:start+size]
        require(not pending_ids.intersection(r['episode_id'] for r in batch)
                or all(r['episode_id'] in pending_ids for r in batch), 'ONLY_COMPLETE_ORIGINAL_BATCHES')
    seen = set()
    for ordinal, state in enumerate(states):
        c = state['contract']
        require(c['provider'] == 'zhipu' and c['currency'] == 'CNY'
                and (c['input_nano'], c['output_nano'], c['cached_nano']) == (0, 0, 0)
                and c['per_call_input_cap'] == core.model.input_reservation_tokens
                and c['per_call_output_cap'] == core.model.max_output_tokens
                and c['authorization_id'] == core.authorization.authorization_id,
                'ORIGINAL_PROVIDER_ALLOCATION_CHANGED')
        allowed = {UNKNOWN} if for_continuation or ordinal < len(states)-1 else {None,UNKNOWN,'THREE_FAILED_EPISODES_IN_FIXED_BATCH'}
        require(state['active_batch'] is None and state['blocked_reason'] in allowed
                and set(state['pending_block_reasons']) <= allowed-{None}, 'UNREVIEWED_PARENT_STOP')
        if state['blocked_reason'] == UNKNOWN:
            require(state['pending_block_reasons'] == [UNKNOWN], 'PARENT_UNKNOWN_STOP_RESET')
        require(set(state['reservations']) == set(state['settlements']), 'UNSETTLED_PARENT_REQUEST')
        require(all((r['input_cap'],r['output_cap'],r['reserved_nano']) ==
                    (core.model.input_reservation_tokens,core.model.max_output_tokens,0)
                    for r in state['reservations'].values()), 'PARENT_PER_REQUEST_ALLOCATION_CHANGED')
        require(not seen.intersection(state['reservations']), 'DUPLICATE_PARENT_REQUEST')
        seen.update(state['reservations'])
        require(not any(k.rsplit(':', 1)[0] in pending_ids for k in state['reservations']), 'UNRUN_ALREADY_STARTED')
        require(state['totals'] == dict(calls=len(state['reservations']),
            input_tokens=len(state['reservations'])*core.model.input_reservation_tokens,
            output_tokens=len(state['reservations'])*core.model.max_output_tokens, reserved_nano=0),
            'PARENT_FULL_RESERVATION_TOTALS_CHANGED')
    additional = sum(r['method'] != 'central_rule' for r in pending)*(core.design_turns+3)
    require(len(seen)+additional <= core.authorization.maximum_generation_calls,
            'CONTINUATION_EXCEEDS_ORIGINAL_CALL_OR_TOKEN_CAP')
    require(any(v['usage_status'] == 'unknown' for s in states for v in s['settlements'].values()), 'NO_UNKNOWN_PARENT_USAGE')
    return dict(pending_schedule=[copy.deepcopy(r) for r in core.schedule if r['episode_id'] in pending_ids],
        maximum_additional_calls=additional, prior_reserved_calls=len(seen),
        cumulative_maximum_calls=len(seen)+additional,
        original_maximum_calls=core.authorization.maximum_generation_calls,
        cumulative_input_reservation=(len(seen)+additional)*core.model.input_reservation_tokens,
        cumulative_output_reservation=(len(seen)+additional)*core.model.max_output_tokens)


def audit_parent(envelope, parent_index, output, *, for_continuation=True):
    """Independently inspect the complete terminal census and every native call."""
    from scripts.research.audit_provider_pilot import independent_usage, verify_admission
    output = Path(output).resolve()
    require(not output.exists(), 'IMMUTABLE_PARENT_AUDIT_EXISTS')
    core, _, _ = verify_provider_envelope(envelope)
    index = read(parent_index)
    original = Path(parent_index).parent
    marker = read(original/'COMPLETED.json')
    require(marker['index_sha256'] == file_sha256(parent_index), 'PARENT_COMPLETION_HASH_CHANGED')
    require(index['plan_id'] == core.plan_id and index['source_hash'] == core.source_hash, 'PARENT_PLAN_CHANGED')
    ledgers = index.get('administrative_ledger_chain', [core.authorization.usage_ledger_path])
    require(len(ledgers) == len(set(ledgers)) and ledgers[0] == core.authorization.usage_ledger_path, 'DUPLICATE_OR_FOREIGN_LEDGER_CHAIN')
    registered = read(check_ref(FileRef.model_validate(index['registration'])))
    if 'administrative_continuation' in index:
        require(registered['original_envelope'] == ref(envelope)
                and ledgers == registered['ledger_chain']+[registered['usage_ledger_path']], 'PARENT_RECOVERY_REGISTRATION_CHANGED')
        for r in registered['bindings']: check_ref(FileRef.model_validate(r))
    else:
        require(registered['envelope'] == ref(envelope) and registered['schedule'] == core.schedule, 'PARENT_FORMAL_REGISTRATION_CHANGED')
    states = [snapshot(p) for p in ledgers]
    require(states[-1] == index['spend_snapshot'], 'PARENT_FINAL_LEDGER_CHANGED')
    scope = admission(core, index, states, for_continuation=for_continuation)
    tasks = {t.task_id: t for t in [load_research_model(check_ref(r), TaskSpec) for r in core.tasks]}
    reservations, settlements = {}, {}
    for state in states:
        reservations.update(state['reservations']); settlements.update(state['settlements'])
    seen, unknown, anchors, raw_files = set(), [], {}, []
    audits = read(check_ref(FileRef.model_validate(index['audit_index'])))['episodes']
    require([a['episode_id'] for a in audits] == [r['episode_id'] for r in index['episodes']], 'PARENT_AUDIT_CENSUS_CHANGED')
    require(not any(audit_has_execution_failure(a) for a in audits), 'PARENT_AUDIT_FAILED')
    for ordinal, (row, audit) in enumerate(zip(index['episodes'], audits)):
        selected = read(check_ref(FileRef.model_validate(row['selection_record'])))
        check_ref(FileRef.model_validate(selected['terminal_record']))
        require(selected['episode_id'] == row['episode_id'] and selected['disposition'] == row['status'], 'PARENT_SELECTION_CHANGED')
        actual_audit = read(check_ref(FileRef.model_validate(row['audit_result'])))
        require(actual_audit == audit, 'PARENT_AUDIT_ENTRY_CHANGED')
        if row['status'] != 'RECORDED':
            continue
        dest = Path(row['path']); verify_episode(dest)
        require(file_sha256(dest/'episode_result.json') == row['result_sha256'], 'PARENT_RESULT_CHANGED')
        result = load_research_model(dest/'episode_result.json', EpisodeResult)
        actual = tasks[row['task_id']]
        if row['method'] == 'central_rule':
            b = actual.model_dump(mode='json'); b.update(signing_layer='explicit_rule_owners_v1', model_configurations=[])
            actual = TaskSpec.model_validate(b)
        validate_episode_identity(result, actual, row, core.source_hash)
        require(result.terminal_state == row['terminal_state'], 'PARENT_TERMINAL_CHANGED')
        calls = read(dest/'llm/call_index.json') if (dest/'llm/call_index.json').exists() else []
        require(len(calls) == row['live_calls'] == result.common_contract['live_calls'], 'PARENT_CALL_COUNT_CHANGED')
        for call in calls:
            reqpath = Path(call['request_ref'])
            request = read(reqpath); response_path = reqpath.with_name(reqpath.name.replace('-request.json', '-response.json'))
            require(str(response_path) == call['response_ref'], 'PARENT_RESPONSE_REFERENCE_CHANGED')
            response = read(response_path); identity = row['episode_id']+':'+request['attempt_id']
            require(identity in reservations and identity not in seen, 'UNACCOUNTED_OR_DUPLICATE_NATIVE_CALL')
            seen.add(identity)
            require(research_hash({k:v for k,v in request.items() if k not in {'source','attempt_id'}})
                    == call['request_hash'] == reservations[identity]['request_hash'], 'PARENT_REQUEST_HASH_CHANGED')
            require(response['provider'] == 'zhipu' and response['model'] == core.model.model_id,
                    'PARENT_PROVIDER_IDENTITY_CHANGED')
            require(request['configuration'] == core.model.model_dump(mode='json'), 'PARENT_MODEL_CONFIG_CHANGED')
            settlement = settlements[identity]; usage = independent_usage('zhipu', response)
            require(usage == (settlement['input_tokens'], settlement['output_tokens'], settlement['cached_tokens']),
                    'PARENT_NATIVE_USAGE_MISMATCH')
            if settlement['usage_status'] == 'unknown':
                require(response['ok'] is False and response['usage'] == {}
                        and response['raw'] == {'error': 'Connection error.'}, 'UNREVIEWED_UNKNOWN_RESPONSE')
                unknown.append(dict(identity=identity, response=ref(response_path),
                    full_input_reservation=reservations[identity]['input_cap'],
                    full_output_reservation=reservations[identity]['output_cap'],
                    usage_status='unknown', actual_bill=None))
            else:
                require(settlement['usage_status'] == 'known' and all(type(v) is int and v >= 0 for v in usage[:2])
                        and usage[0] <= core.model.input_reservation_tokens and usage[1] <= core.model.max_output_tokens
                        and settlement['peak_price_cost_upper_bound_nano'] == 0, 'PARENT_KNOWN_USAGE_OR_CAP')
            raw_files.extend([ref(reqpath), ref(response_path)])
        anchors[str(dest/'COMMITTED.json')] = file_sha256(dest/'COMMITTED.json')
        if (ordinal+1) % 90 == 0:
            print(json.dumps({'parent_audit_rows': ordinal+1, 'raw_calls': len(seen)}), flush=True)
    require(seen == set(reservations), 'PARENT_NATIVE_CALL_CENSUS_MISMATCH')
    admission_paths = index.get('administrative_admission_chain', [str(original/'request_admission.json')])
    require(len(admission_paths) == len(states), 'PARENT_ADMISSION_CHAIN_CHANGED')
    require(file_sha256(admission_paths[-1]) == index['request_admission_sha256'], 'PARENT_ADMISSION_HASH_CHANGED')
    for admission_path, state in zip(admission_paths, states):
        verify_admission(read(admission_path), 2, state['reservations'])
    output.mkdir(parents=True, exist_ok=False)
    _write_immutable(output/'raw_call_index.json', {'files': raw_files})
    _write_immutable(output/'parent_spend_snapshot.json', {'states': states, 'sha256': research_hash(states)})
    body = dict(schema_version=1, status=('VERIFIED_FOR_ORIGINAL_UNRUN_CONTINUATION' if for_continuation
        else 'VERIFIED_COMPLETE_CENSUS_USAGE_UNRESOLVED'),
        audited_at_utc=datetime.now(timezone.utc).isoformat(), parent_index=ref(parent_index),
        original_envelope=ref(envelope), raw_call_index=ref(output/'raw_call_index.json'),
        spend_snapshot=ref(output/'parent_spend_snapshot.json'), ledger_chain=ledgers,
        raw_requests_verified=len(seen), recorded_trials=sum(r['status']=='RECORDED' for r in index['episodes']),
        unknown_requests=unknown, unknown_usage_reconciled=False, original_f70_strict_gate_passed=False,
        failed_trials_replayed=False, parent_anchors=anchors,
        source_hashes={str(p): file_sha256(p) for p in OWN_SOURCES}, **scope)
    body['admission_chain'] = admission_paths
    _write_immutable(output/'audit_receipt.json', body)
    return body


def prepare_recovery(envelope, parent_index, audit_path, output):
    output = Path(output).resolve()
    require(not output.exists(), 'IMMUTABLE_RECOVERY_REGISTRATION_EXISTS')
    core, _, _ = verify_provider_envelope(envelope)
    audit = read(audit_path)
    require(audit['status'] == 'VERIFIED_FOR_ORIGINAL_UNRUN_CONTINUATION'
            and audit['original_envelope'] == ref(envelope) and audit['parent_index'] == ref(parent_index), 'AUDIT_PARENT_MISMATCH')
    for p, sha in audit['source_hashes'].items(): require(file_sha256(p) == sha, 'RECOVERY_AUDIT_SOURCE_CHANGED')
    states = [snapshot(p) for p in audit['ledger_chain']]
    require(states == read(check_ref(FileRef.model_validate(audit['spend_snapshot'])))['states'], 'AUDITED_LEDGER_CHANGED')
    scope = admission(core, read(parent_index), states)
    require(all(scope[k] == audit[k] for k in scope), 'AUDITED_RECOVERY_SCOPE_CHANGED')
    output.mkdir(parents=True, exist_ok=False)
    bindings = [ref(p) for p in OWN_SOURCES]+[ref(audit_path), ref(parent_index), ref(envelope), audit['spend_snapshot'], audit['raw_call_index']]
    for p in audit['ledger_chain']:
        bindings.append(ref(p))
        wal = Path(str(p)+'-wal')
        if wal.exists(): bindings.append(ref(wal))
    registration = dict(schema_version=1, kind='original_unrun_formal_administrative_continuation_v1',
        registered_at_utc=datetime.now(timezone.utc).isoformat(), original_plan_id=core.plan_id,
        original_envelope=ref(envelope), parent_index=ref(parent_index), parent_audit=ref(audit_path),
        protocol=ref(PROTOCOL), bindings=bindings, ledger_chain=audit['ledger_chain'],
        admission_chain=audit['admission_chain'],
        usage_ledger_path=str(output/'usage.sqlite'), source_hash=core.source_hash,
        unknown_usage_reconciled=False, original_f70_strict_gate_passed=False,
        sensitivity_plan=['original_ITT_failures_zero','unknown_service_outcome_extreme_0_1_bounds',
                          'whole_affected_task_cluster_exclusion_nonprimary'], **scope)
    _write_immutable(output/'recovery_registration.json', registration)
    return registration


def run_recovery(registration_path, *, credential_path):
    registration_path = Path(registration_path).resolve(); out = registration_path.parent
    reg = read(registration_path)
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (out/'STARTED.json').exists(), 'RECOVERY_ALREADY_STARTED')
        core, _, _ = verify_provider_envelope(reg['original_envelope']['path'])
        require(ref(reg['original_envelope']['path']) == reg['original_envelope'], 'ORIGINAL_ENVELOPE_CHANGED')
        parent = read(check_ref(FileRef.model_validate(reg['parent_index'])))
        def source_guard():
            require(ref(registration_path) == registration_ref, 'RECOVERY_REGISTRATION_CHANGED')
            for r in reg['bindings']: check_ref(FileRef.model_validate(r))
            for r in core.scientific_inputs: check_ref(r)
            check_ref(core.authorization.ledger)
        registration_ref = ref(registration_path); source_guard()
        states = [snapshot(p) for p in reg['ledger_chain']]
        scope = admission(core, parent, states)
        require(all(reg[k] == v for k, v in scope.items()), 'RECOVERY_SCOPE_CHANGED')
        require(design_implementation_manifest().source_hash == core.source_hash, 'FROZEN_DESIGN_CHANGED')
        validate_natural_registration(core)
        require(importlib.metadata.version('openai') == core.model.sdk_version, 'PROVIDER_RUNTIME_SDK_CHANGED')
        settings = load_provider_settings(core, credential_path)
        # O_EXCL parent marker prevents a second allocation for the same stopped census.
        successor = Path(reg['parent_index']['path']).with_suffix('.successor.json')
        with successor.open('x') as handle:
            json.dump(dict(registration=registration_ref, created_at=time.time()), handle, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        spend = ProviderUsageLedger(reg['usage_ledger_path'], authorization_id=core.authorization.authorization_id,
            plan_id=core.plan_id+'-continuation-'+research_hash(reg)[:12], plan_hash=research_hash(reg),
            maximum_calls=reg['maximum_additional_calls'], input_cap=core.model.input_reservation_tokens,
            output_cap=core.model.max_output_tokens, provider='zhipu', currency='CNY', input_nano=0,
            output_nano=0, cached_nano=0, source_guard=source_guard, parent_reference=reg['parent_audit'])
        require(not spend.snapshot()['reservations'], 'RECOVERY_ALLOCATION_ALREADY_USED')
        _write_immutable(out/'STARTED.json', dict(started_at=time.time(), process_id=os.getpid(), registration=registration_ref))
        gate = ProviderRequestGate(2, out/'request_admission.json'); gate._save()
        evaluator = EvaluatorLedger(out/'evaluator_budget.json', len(reg['pending_schedule']))
        tasks = {t.task_id: t for t in [load_research_model(check_ref(r), TaskSpec) for r in core.tasks]}
        original_core = reference(Path(reg['original_envelope']['path']).parent/'PlanCore.json')
        entries = copy.deepcopy(parent['episodes']); by_id = {r['episode_id']: i for i, r in enumerate(entries)}
        audits = copy.deepcopy(read(check_ref(FileRef.model_validate(parent['audit_index'])))['episodes'])
        previous = spend.snapshot(); stopping = None; segments = reg['ledger_chain']+[reg['usage_ledger_path']]
        jobs = out/'jobs.jsonl'; jobs.touch(exist_ok=False)
        def job(body):
            with jobs.open('a') as handle:
                handle.write(json.dumps({**body,'recorded_at':time.time()},sort_keys=True)+'\n')
                handle.flush(); os.fsync(handle.fileno())
        def save():
            nonlocal previous
            current = spend.snapshot(); verify_history(current, previous); previous = current
            remaining = sum(r['status']=='NOT_RUN_STOPPING_RULE' for r in entries)
            child_ok = (current['blocked_reason'] in {None, 'THREE_FAILED_EPISODES_IN_FIXED_BATCH'}
                and set(current['pending_block_reasons']) <= {'THREE_FAILED_EPISODES_IN_FIXED_BATCH'}
                and current['active_batch'] is None and set(current['reservations']) == set(current['settlements'])
                and all(s['usage_status']=='known' for s in current['settlements'].values()))
            design_ok = not any(r['status']=='EXECUTION_FAILED' for r in entries) and not any(audit_has_execution_failure(a) for a in audits)
            complete = remaining == 0 and design_ok
            durable_json(out/'audit_index.json', dict(schema_version=6, episodes=audits, design_feedback=False,
                audit_plan_hash=research_hash(core.audit_plan), evaluator_budget=evaluator.state))
            index = {**parent, 'schema_version':6, 'episodes':entries, 'registration':registration_ref,
                'planned_trials':len(core.schedule), 'indexed_trials':len(entries),
                'recorded_trials':sum(r['status']=='RECORDED' for r in entries), 'not_run_trials':remaining,
                'started_trials':len(entries)-remaining, 'execution_failed_trials':sum(r['status']=='EXECUTION_FAILED' for r in entries),
                'method_failed_trials':sum(r.get('terminal_state')=='METHOD_FAILED' for r in entries),
                'formal_status':'ALL_ORIGINAL_TRIALS_RECORDED_USAGE_UNRESOLVED' if complete else 'INCOMPLETE_FORMAL',
                'all_registered_trials_have_terminal_records':complete,
                'all_design_and_audit_integrity_checks_passed':design_ok,
                'all_provider_resource_and_identity_checks_passed':False, 'original_f70_strict_gate_passed':False,
                'current_segment_usage_verified':child_ok, 'parent_unknown_usage_reconciled':False,
                'scientific_effects_status':'NOT_ANALYZED', 'stopping_reason':stopping,
                'administrative_ledger_chain':segments, 'administrative_continuation':registration_ref,
                'administrative_admission_chain':reg['admission_chain']+[str(out/'request_admission.json')],
                'cumulative_reserved_calls':reg['prior_reserved_calls']+current['totals']['calls'],
                'cumulative_maximum_calls':reg['cumulative_maximum_calls'],
                'spend_snapshot':current, 'request_admission_sha256':file_sha256(out/'request_admission.json'),
                'audit_index':ref(out/'audit_index.json'), 'jobs':ref(jobs)}
            index.pop('natural_collection', None)
            durable_json(out/'benchmark_index.json', index)
            return index
        def run_one(item):
            task = tasks[item['task_id']]; method = item['method']; source = 'rule' if method=='central_rule' else 'live'
            row = {**item, 'source':source}; dest = out/'episodes'/item['episode_id']
            try:
                actual = task
                if source == 'rule':
                    b = task.model_dump(mode='json'); b.update(signing_layer='explicit_rule_owners_v1', model_configurations=[])
                    actual = TaskSpec.model_validate(b); implementation = CentralRuleMethod(protocol_enabled=True)
                else:
                    implementation = SharedEvidenceProtocolMethod(method, lambda ledger,path: DesignAgentClient(core.model,
                        ledger,path,allow_live=True,settings=settings,
                        spend_ledger=ControlledProviderLedger(spend,gate,ledger,source_guard)), turns=core.design_turns)
                result = run_design_episode(actual, implementation, dest, episode_id=item['episode_id'])
                verification = verify_episode(dest); validate_episode_identity(result,actual,item,core.source_hash)
                terminal = dest/'episode_result.json'
                row.update(status='RECORDED',path=str(dest),terminal_state=result.terminal_state,
                    verification=verification,live_calls=result.common_contract['live_calls'],result_sha256=file_sha256(terminal),
                    natural_trace=source=='live' and result.live_status=='RUN',reference_only=source=='rule')
            except Exception as exc:
                terminal = out/'failed_attempts'/(item['episode_id']+'.json')
                _write_immutable(terminal,{**item,'status':'EXECUTION_FAILED','error_type':type(exc).__name__,'reason':str(exc)})
                row.update(status='EXECUTION_FAILED',reason=str(exc),error_type=type(exc).__name__)
            finally:
                gate.finish_episode(item['episode_id'])
            selected = freeze_selection(task,terminal,out/'selections'/(item['episode_id']+'.json'),
                episode_id=item['episode_id'],source=source,plan_core=original_core,disposition=row['status'])
            row['selection_record'] = selected.model_dump(mode='json')
            return row
        size = core.tasks_per_batch*len(core.methods); pending = reg['pending_schedule']
        for offset in range(0,len(pending),size):
            if stopping: break
            batch = pending[offset:offset+size]; batch_id = 'continuation-'+str(offset//size).zfill(5)
            try:
                source_guard(); verify_provider_envelope(reg['original_envelope']['path'])
            except Exception as exc:
                stopping = 'PRE_BATCH_INTEGRITY_FAILURE:'+type(exc).__name__
                break
            spend.begin_batch(batch_id,[x['episode_id'] for x in batch])
            for item in batch: job({**item,'stage':'DESIGN_STARTED'})
            with ThreadPoolExecutor(max_workers=core.execution_workers) as pool:
                rows = list(pool.map(run_one,batch))
            failed = sum(r['status']=='EXECUTION_FAILED' or r.get('terminal_state')=='METHOD_FAILED' for r in rows)
            reason = 'THREE_FAILED_EPISODES_IN_FIXED_BATCH' if failed >= 3 else None
            if any(r['status']=='EXECUTION_FAILED' for r in rows): reason = 'EXECUTION_INTEGRITY_OR_SERVICE_FAILURE'
            spend.close_batch(batch_id,stop_reason=reason); stopping = spend.snapshot()['blocked_reason']
            for row in rows:
                auditdir = out/'audits'/row['episode_id']
                try:
                    audit = audit_frozen_selection(FileRef.model_validate(row['selection_record']),tasks[row['task_id']],
                        lambda:core.audit_plan.model_dump(mode='json'),auditdir,source_hash=core.source_hash,evaluator_ledger=evaluator)
                    row['audit_result'] = ref(auditdir/'audit_result.json')
                except Exception as exc:
                    stopping = 'INDEPENDENT_AUDIT_INTEGRITY_FAILURE'
                    audit = dict(episode_id=row['episode_id'],status='FAILED',reason=str(exc),truth='UNKNOWN',design_feedback=False)
                    _write_immutable(auditdir/'audit_failure.json',audit); row['audit_result'] = ref(auditdir/'audit_failure.json')
                if audit_has_execution_failure(audit): stopping = 'INDEPENDENT_AUDIT_INTEGRITY_FAILURE'
                i = by_id[row['episode_id']]; entries[i] = row; audits[i] = audit
                job(dict(episode_id=row['episode_id'],stage='TERMINAL_AND_AUDIT_RECORDED',status=row['status']))
            index = save()
            print(json.dumps(dict(batch=batch_id,recorded=index['recorded_trials'],planned=len(entries),stopping=stopping)),flush=True)
        index = save(); validate_admission(core,read(out/'request_admission.json'),spend.snapshot())
        _write_immutable(out/'COMPLETED.json',dict(index_sha256=file_sha256(out/'benchmark_index.json'),
            audit_index_sha256=file_sha256(out/'audit_index.json'),
            complete_planned_index=True, all_original_trials_recorded=index['all_registered_trials_have_terminal_records'],
            formal_execution_complete=False, original_f70_strict_gate_passed=False, parent_unknown_usage_reconciled=False))
        return index
