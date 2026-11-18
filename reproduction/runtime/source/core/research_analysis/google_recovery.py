"""Explicit administrative continuation of Google's untouched pilot schedule.

This outer coordinator preserves the original model, protocol, tasks and failed
episodes. It never rewrites the original SQLite ledger, completion marker, or
raw evidence. A separate finite allocation retains the parent liability.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import math
import os
import shutil
import sqlite3
import time
from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256, research_hash
from core.experiments.strict_types import strict_json_loads
from core.codesign.schema import TaskSpec
from core.codesign.runner import run_design_episode, verify_episode, design_implementation_manifest
from core.codesign.live_protocol import SharedEvidenceProtocolMethod
from core.codesign.llm_adapter import DesignAgentClient
from core.codesign.baselines import CentralRuleMethod
from core.codesign.budget import BudgetExceeded
from core.llm.base import redact_sensitive
from core.llm.gemini_client import load_gemini_settings
from .google_pilot import load_google_pilot, load_google_inputs, validate_episode_identity, verify_history
from .google_usage import GoogleUsageLedger

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_FILES = ['core/research_analysis/google_recovery.py', 'scripts/research/recover_google_pilot.py']
INTERVAL_SECONDS = 5.1


def read(path):
    return strict_json_loads(Path(path).read_text())


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}


def check_ref(ref):
    path = Path(ref['path'])
    if not path.is_absolute() or file_sha256(path) != ref['sha256']:
        raise ValueError('GOOGLE_RECOVERY_BOUND_REFERENCE_CHANGED')
    return path


def immutable(path, body):
    path = Path(path)
    if path.exists():
        raise ValueError('GOOGLE_RECOVERY_IMMUTABLE_OUTPUT_EXISTS')
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_json(path, body)
    os.chmod(path, 0o444)


def validate_recovery_scope(index, registration, parent_state):
    identity = lambda row: tuple(row[key] for key in ('episode_id', 'task_id', 'method', 'batch_id'))
    if (len(index['episodes']) != 270 or [identity(r) for r in index['episodes']] !=
            [identity(r) for r in registration['schedule']] or len({identity(r) for r in index['episodes']}) != 270):
        raise ValueError('GOOGLE_RECOVERY_ORIGINAL_COMPLETE_SCHEDULE_REQUIRED')
    if index['stopping_reason'] != 'UNKNOWN_USAGE_REQUIRES_RECONCILIATION':
        raise ValueError('GOOGLE_RECOVERY_ORIGINAL_STOP_MISMATCH')
    original = index['episodes'][:18]
    pending = index['episodes'][18:]
    if any(r['status'] != 'RECORDED' for r in original) or any(r['status'] != 'NOT_RUN_STOPPING_RULE' for r in pending):
        raise ValueError('GOOGLE_RECOVERY_ONLY_ORIGINAL_252_UNRUN_ALLOWED')
    if sum(r['method'] != 'central_rule' for r in pending) != 224:
        raise ValueError('GOOGLE_RECOVERY_ORIGINAL_224_LIVE_EPISODES_REQUIRED')
    if (len(parent_state['reservations']) != 35 or set(parent_state['reservations']) != set(parent_state['settlements'])
            or parent_state['blocked_reason'] != index['stopping_reason'] or parent_state['active_batch'] is not None):
        raise ValueError('GOOGLE_RECOVERY_ORIGINAL_LEDGER_STATE_REQUIRED')
    unknown = {key: parent_state['reservations'][key]['reserved_nano'] for key, value in parent_state['settlements'].items()
               if value['peak_price_cost_upper_bound_nano'] is None}
    if len(unknown) != 16 or sum(unknown.values()) != 321_126_400:
        raise ValueError('GOOGLE_RECOVERY_UNKNOWN_LIABILITY_CHANGED')
    if parent_state['contract']['maximum_calls'] != 2160 or 35+2016 > 2160:
        raise ValueError('GOOGLE_RECOVERY_ORIGINAL_FINITE_CALL_CAP_REQUIRED')
    return original, [{key: row[key] for key in ('episode_id', 'task_id', 'method', 'batch_id')} for row in pending], unknown


class SQLiteRequestPacer:
    """One durable host-wide admission gate, shared by all recovery workers.

    Durable FIFO tickets prevent an already waiting worker from being overtaken.
    The transaction serializes each grant; slots are never refunded or reused.
    """
    def __init__(self, path, *, minimum_interval_seconds=INTERVAL_SECONDS, clock=time.monotonic, sleeper=time.sleep):
        if (type(minimum_interval_seconds) is not float or not math.isfinite(minimum_interval_seconds)
                or minimum_interval_seconds < 5.0):
            raise ValueError('PACING_AT_MOST_TWELVE_REQUESTS_PER_MINUTE_REQUIRED')
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock, self.sleeper, self.interval = clock, sleeper, minimum_interval_seconds
        db = self._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS pacing_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS grants (sequence INTEGER PRIMARY KEY,identity TEXT UNIQUE NOT NULL,monotonic REAL NOT NULL,wall_time REAL NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS tickets (sequence INTEGER PRIMARY KEY AUTOINCREMENT,identity TEXT UNIQUE NOT NULL,requested_at REAL NOT NULL,status TEXT NOT NULL)')
            contract = json.dumps({'schema_version': 1, 'provider': 'gemini', 'scope': 'all_requests_using_this_shared_admission_ledger',
                'minimum_interval_seconds': self.interval, 'queue_rule': 'durable_fifo_tickets_v1',
                'queue_time_counts_toward_original_episode_wall_budget': True,
                'retry': False, 'external_project_traffic_controlled': False}, sort_keys=True)
            old = db.execute('SELECT value FROM pacing_meta WHERE key="contract"').fetchone()
            if old and old[0] != contract:
                raise ValueError('PACING_CONTRACT_CHANGED')
            db.execute('INSERT OR IGNORE INTO pacing_meta VALUES ("contract",?)', (contract,))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=600.0)
        db.execute('PRAGMA busy_timeout=600000')
        db.execute('PRAGMA synchronous=FULL')
        return db

    def grant(self, identity, *, wall_deadline=None):
        db = self._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM tickets WHERE identity=?', (identity,)).fetchone():
                raise ValueError('PACING_REQUEST_ID_REUSED')
            db.execute('INSERT INTO tickets(identity,requested_at,status) VALUES (?, ?, "queued")', (identity, time.time()))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
        while True:
            db = self._connect()
            try:
                db.execute('BEGIN IMMEDIATE')
                wall = time.time()
                if wall_deadline is not None and wall >= wall_deadline:
                    db.execute('UPDATE tickets SET status="expired" WHERE identity=?', (identity,))
                    db.commit()
                    raise BudgetExceeded('WALL_DEADLINE_EXCEEDED_DURING_HOST_REQUEST_QUEUE')
                first = db.execute('SELECT identity FROM tickets WHERE status="queued" ORDER BY sequence LIMIT 1').fetchone()
                last = db.execute('SELECT sequence,monotonic FROM grants ORDER BY sequence DESC LIMIT 1').fetchone()
                now = self.clock()
                if last and now < last[1]:
                    raise ValueError('PACING_MONOTONIC_CLOCK_RESET_REQUIRES_REVIEW')
                due = now if last is None else last[1]+self.interval
                if first and first[0] == identity and now >= due:
                    sequence = 1 if last is None else last[0]+1
                    db.execute('INSERT INTO grants VALUES (?,?,?,?)', (sequence, identity, now, wall))
                    db.execute('UPDATE tickets SET status="granted" WHERE identity=?', (identity,))
                    db.commit()
                    return dict(sequence=sequence, identity=identity, granted_monotonic=now, granted_at=wall)
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()
            self.sleeper(min(.25, max(.01, due-now)))

    def snapshot(self):
        db = self._connect()
        try:
            contract = strict_json_loads(db.execute('SELECT value FROM pacing_meta WHERE key="contract"').fetchone()[0])
            grants = [dict(sequence=s, identity=i, granted_monotonic=m, granted_at=w)
                      for s, i, m, w in db.execute('SELECT sequence,identity,monotonic,wall_time FROM grants ORDER BY sequence')]
            tickets = [dict(sequence=s, identity=i, requested_at=w, status=status)
                       for s, i, w, status in db.execute('SELECT sequence,identity,requested_at,status FROM tickets ORDER BY sequence')]
        finally:
            db.close()
        if any(b['granted_monotonic']-a['granted_monotonic'] < self.interval-1e-9 for a, b in zip(grants, grants[1:])):
            raise ValueError('PACING_HISTORY_INTERVAL_VIOLATION')
        if [row['identity'] for row in tickets if row['status'] == 'granted'] != [row['identity'] for row in grants]:
            raise ValueError('PACING_HISTORY_FIFO_VIOLATION')
        return {'contract': contract, 'grants': grants, 'tickets': tickets}


class PacedUsageLedger:
    def __init__(self, ledger, pacer, source_guard, episode_ledger=None):
        self.ledger, self.pacer, self.source_guard = ledger, pacer, source_guard
        self.episode_ledger = episode_ledger

    def reserve(self, identity, incoming, outgoing, request_hash):
        self.source_guard()
        deadline = None if self.episode_ledger is None else self.episode_ledger.snapshot()['deadline']
        self.pacer.grant(identity, wall_deadline=deadline)
        if deadline is not None and self.episode_ledger.clock() >= deadline:
            raise BudgetExceeded('WALL_DEADLINE_EXCEEDED_DURING_HOST_REQUEST_QUEUE')
        return self.ledger.reserve(identity, incoming, outgoing, request_hash)

    def settle(self, *args, **kwargs): return self.ledger.settle(*args, **kwargs)
    def snapshot(self): return self.ledger.snapshot()
    def block(self, reason): return self.ledger.block(reason)


def prepare_recovery(pilot, output, *, allocation_path, pacing_path, planning_amendment_path, free_tier_authorization_path):
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    allocation_path, pacing_path = Path(allocation_path).resolve(), Path(pacing_path).resolve()
    if (output.exists() or allocation_path.exists() or pacing_path.exists()
            or len({output, allocation_path, pacing_path}) != 3):
        raise ValueError('GOOGLE_RECOVERY_REQUIRES_NEW_DISTINCT_OUTPUT_AND_LEDGERS')
    amendment_path = Path(planning_amendment_path).resolve()
    amendment = read(amendment_path)
    if type(amendment) is not dict or 'protocol' not in amendment:
        raise ValueError('GOOGLE_RECOVERY_EXPLICIT_PLANNING_AMENDMENT_REQUIRED')
    protocol_path = check_ref(amendment['protocol'])
    authorization_path = Path(free_tier_authorization_path).resolve()
    if type(read(authorization_path)) is not dict:
        raise ValueError('GOOGLE_RECOVERY_FREE_TIER_AUTHORIZATION_REQUIRED')
    amendment_refs = {}
    for key in ('protocol', 'free_tier_authorization', 'original_planning_design', 'original_pilot_index'):
        checked = check_ref(amendment[key])
        amendment_refs[str(checked)] = amendment[key]['sha256']
    if (amendment['free_tier_authorization'] != reference(authorization_path)
            or amendment['original_pilot_index'] != reference(pilot/'benchmark_index.json')):
        raise ValueError('GOOGLE_RECOVERY_AMENDMENT_REFERENCE_SCOPE_MISMATCH')
    marker, index, registration = (read(pilot/name) for name in ('COMPLETED.json', 'benchmark_index.json', 'registration.json'))
    if (not marker['complete_planned_index'] or marker['index_sha256'] != file_sha256(pilot/'benchmark_index.json')
            or marker['registration_sha256'] != file_sha256(pilot/'registration.json')):
        raise ValueError('GOOGLE_ORIGINAL_COMPLETION_CHANGED')
    plan = load_google_pilot(pilot/'plan.json')
    load_google_inputs(plan)
    parent_path = pilot/'spend_final_snapshot.json'
    parent = read(parent_path)
    if research_hash(parent['state']) != parent['sha256'] or parent['state'] != index['spend_snapshot']:
        raise ValueError('GOOGLE_RECOVERY_PARENT_SNAPSHOT_CHANGED')
    original, pending, unknown = validate_recovery_scope(index, registration, parent['state'])
    if (research_hash(plan) != index['plan_hash'] or registration['plan_hash'] != index['plan_hash']
            or research_hash(registration['runtime_source_hashes']) != registration['runtime_hash']
            or registration['source_hash'] != index['source_hash']
            or design_implementation_manifest().source_hash != index['source_hash']):
        raise ValueError('GOOGLE_RECOVERY_SCIENTIFIC_SOURCE_CHANGED')
    originals = {str(pilot/name): file_sha256(pilot/name) for name in ('COMPLETED.json', 'benchmark_index.json',
        'registration.json', 'plan.json', 'planning_design.json', 'experiment_spec.json', 'spend_final_snapshot.json')}
    for row in original:
        path = Path(row['path'])
        verify_episode(path)
        if file_sha256(path/'episode_result.json') != row['result_sha256']:
            raise ValueError('GOOGLE_ORIGINAL_RECORDED_EPISODE_CHANGED')
        originals[str(path/'COMMITTED.json')] = file_sha256(path/'COMMITTED.json')
        originals[str(path/'episode_result.json')] = file_sha256(path/'episode_result.json')
    for identity in unknown:
        episode_id, attempt = identity.rsplit(':', 1)
        raw_path = pilot/'episodes'/episode_id/'llm'/(attempt+'-response.json')
        raw = read(raw_path)
        if raw.get('ok') or raw.get('raw', {}).get('http_status') != 429:
            raise ValueError('GOOGLE_RECOVERY_EXPECTS_RETAINED_QUOTA_FAILURES')
        originals[str(raw_path)] = file_sha256(raw_path)
    runtime = {str(ROOT/name): file_sha256(ROOT/name) for name in RECOVERY_FILES}
    bound = {**registration['runtime_source_hashes'], **registration['input_hashes'], **originals,
             **amendment_refs,
             str(amendment_path): file_sha256(amendment_path), str(protocol_path): file_sha256(protocol_path),
             str(authorization_path): file_sha256(authorization_path)}
    for path, sha in bound.items():
        if file_sha256(path) != sha:
            raise ValueError('GOOGLE_RECOVERY_ORIGINAL_BOUND_CONTENT_CHANGED')
    record = dict(schema_version=1, amendment='google-pilot-v4-original-schedule-paced-recovery-v1',
        registered_at=time.time(), original_pilot=str(pilot), original_index=reference(pilot/'benchmark_index.json'),
        original_registration=reference(pilot/'registration.json'), original_source_hash=index['source_hash'],
        original_runtime_hash=registration['runtime_hash'], recovery_runtime_hash=research_hash(runtime),
        recovery_source_hashes=runtime, preserved_hashes=bound, parent_reference=reference(parent_path),
        planning_amendment=reference(amendment_path),
        planning_protocol=reference(protocol_path), free_tier_authorization=reference(authorization_path),
        authorization_id=plan.authorization_id, allocation_plan_id=plan.plan_id+'-paced-original-schedule-recovery-v1',
        allocation_path=str(Path(allocation_path).resolve()), pacing_path=str(Path(pacing_path).resolve()),
        minimum_interval_seconds=INTERVAL_SECONDS, maximum_recovery_generation_calls=2016,
        original_attempted_calls=35, original_plan_maximum_calls=2160, maximum_cumulative_attempted_calls=2051,
        pending_schedule=pending, original_recorded_rows=original, retained_unknown_liabilities_nano=unknown,
        retained_unknown_full_reservation_usd=sum(unknown.values())/1e9, unknown_usage_reconciled=False,
        original_failed_episodes_rerun=False, original_tasks_or_seeds_replaced=False,
        cumulative_accounting_rule='parent_full_reservations_and_unknown_usage_retained_plus_new_finite_allocation',
        stop_rule=plan.stop_rule, batch_task_count=2, maximum_workers=16,
        host_queue_rule='durable_fifo_tickets_v1', host_queue_time_counts_toward_original_episode_wall_budget=True,
        scientific_protocol_changed=False, outer_coordinator_changed=True,
        F72_automatic_acceptance=False,
        F72_boundary='Retain the two original quota-failed tasks under the original failure rule. Report an administrative sensitivity for initial unpaced versus remaining paced service conditions; do not silently pool them as an unchanged service distribution.',
        execution_started=False)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('plan.json', 'registration.json', 'planning_design.json', 'experiment_spec.json'):
        shutil.copyfile(pilot/name, output/name)
        os.chmod(output/name, 0o444)
    for path in runtime:
        target = output/'recovery_source'/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        os.chmod(target, 0o444)
    immutable(output/'recovery_registration.json', record)
    return record


def execute_recovery(output, *, credential_path, workers=16):
    output = Path(output).resolve()
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('GOOGLE_RECOVERY_WORKERS_MUST_BE_ONE_TO_SIXTEEN')
    record = read(output/'recovery_registration.json')
    pilot = Path(record['original_pilot'])
    for name in ('plan.json', 'registration.json', 'planning_design.json', 'experiment_spec.json'):
        if file_sha256(output/name) != record['preserved_hashes'].get(str(pilot/name)):
            raise ValueError('GOOGLE_RECOVERY_ORIGINAL_CONFIGURATION_COPY_CHANGED')
    if (record['maximum_recovery_generation_calls'] != 2016 or record['minimum_interval_seconds'] != INTERVAL_SECONDS
            or record['unknown_usage_reconciled'] is not False or record['original_failed_episodes_rerun'] is not False
            or record['original_tasks_or_seeds_replaced'] is not False or record['F72_automatic_acceptance'] is not False):
        raise ValueError('GOOGLE_RECOVERY_REGISTERED_CONTRACT_CHANGED')
    plan = load_google_pilot(output/'plan.json')
    tasks, _ = load_google_inputs(plan)
    original_index = read(check_ref(record['original_index']))
    parent = read(check_ref(record['parent_reference']))['state']
    original, pending, unknown = validate_recovery_scope(original_index, read(check_ref(record['original_registration'])), parent)
    if pending != record['pending_schedule'] or original != record['original_recorded_rows'] or unknown != record['retained_unknown_liabilities_nano']:
        raise ValueError('GOOGLE_RECOVERY_REGISTERED_SCOPE_CHANGED')
    def unchanged():
        for path, sha in {**record['preserved_hashes'], **record['recovery_source_hashes']}.items():
            if file_sha256(path) != sha:
                raise ValueError('GOOGLE_RECOVERY_SOURCE_OR_INPUT_CHANGED:'+path)
    unchanged()
    if research_hash(record['recovery_source_hashes']) != record['recovery_runtime_hash']:
        raise ValueError('GOOGLE_RECOVERY_RUNTIME_MANIFEST_CHANGED')
    check_ref(record['planning_amendment'])
    check_ref(record['planning_protocol'])
    check_ref(record['free_tier_authorization'])
    if (Path(record['allocation_path']).exists() or Path(record['pacing_path']).exists()
            or record['allocation_path'] == record['pacing_path'] or (output/'EXECUTION_STARTED.json').exists()):
        raise ValueError('GOOGLE_RECOVERY_ALREADY_STARTED_NO_AUTOMATIC_RETRY')
    settings = load_gemini_settings(credential_path, timeout_seconds=plan.model.timeout_seconds)
    pointer = pilot/'administrative-successor.json'
    pointer_body = dict(recovery_registration=reference(output/'recovery_registration.json'),
                        output=str(output), allocation_path=record['allocation_path'])
    # Exclusive sidecar leaves every original pilot file and SQLite byte untouched.
    with pointer.open('x', encoding='utf-8') as handle:
        json.dump(pointer_body, handle, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    immutable(output/'EXECUTION_STARTED.json', dict(started_at=time.time(), recovery_registration=reference(output/'recovery_registration.json'), workers=workers))
    book = GoogleUsageLedger(record['allocation_path'], authorization_id=record['authorization_id'],
        plan_id=record['allocation_plan_id'], plan_hash=research_hash(record), maximum_calls=2016,
        input_cap=32768, output_cap=4096, input_nano=300, output_nano=2500, cached_nano=30,
        source_guard=unchanged, parent_reference=record['parent_reference'])
    pacer = SQLiteRequestPacer(record['pacing_path'], minimum_interval_seconds=record['minimum_interval_seconds'])
    before = book.snapshot(); prior = before
    immutable(output/'allocation_before.json', {'state': before, 'sha256': research_hash(before)})
    (output/'failed_attempts').mkdir()
    task_map = {task.task_id: task for task in tasks}
    episodes, stopping = list(original), None
    def save():
        state = book.snapshot()
        verify_history(state, prior)
        durable_json(output/'pacing_snapshot.json', pacer.snapshot())
        known = sum(v['peak_price_cost_upper_bound_nano'] for s in (parent, state) for v in s['settlements'].values()
                    if v['peak_price_cost_upper_bound_nano'] is not None)
        unresolved = {key: s['reservations'][key]['reserved_nano'] for s in (parent, state)
                      for key, value in s['settlements'].items() if value['peak_price_cost_upper_bound_nano'] is None}
        merged = dict(schema_version=5, study_stage='pilot_recovery', original_study_stage='pilot',
            plan_hash=research_hash(plan), source_hash=record['original_source_hash'],
            original_runtime_hash=record['original_runtime_hash'], recovery_runtime_hash=record['recovery_runtime_hash'],
            registration_sha256=file_sha256(output/'registration.json'), recovery_registration=reference(output/'recovery_registration.json'),
            pacing_snapshot=reference(output/'pacing_snapshot.json'),
            original_index=record['original_index'], episodes=episodes, planned_trials=270, indexed_trials=len(episodes),
            independent_task_count=30, market_runs=0, llm_repeats_per_task_method=1, stopping_reason=stopping,
            historical_stopping_reason=original_index['stopping_reason'], spend_snapshot=state,
            parent_spend_snapshot=record['parent_reference'], formal_status='NOT_RUN', F72_automatic_acceptance=False,
            cumulative_usage=dict(currency='USD', known_usage_price_upper_usd=known/1e9,
                unknown_request_count=len(unresolved), unknown_full_reservation_usd=sum(unresolved.values())/1e9,
                cumulative_full_reservation_usd=sum(v['reserved_nano'] for s in (parent, state) for v in s['reservations'].values())/1e9,
                total_calls=len(parent['reservations'])+len(state['reservations']), actual_provider_bill_usd=None))
        durable_json(output/'benchmark_index.json', merged)
        return merged
    def execute(item):
        destination = output/'episodes'/item['episode_id']
        try:
            unchanged()
            task = task_map[item['task_id']]
            if item['method'] == 'central_rule':
                body = task.model_dump(mode='json'); body.update(signing_layer='explicit_rule_owners_v1', model_configurations=[])
                task = TaskSpec.model_validate(body)
                method = CentralRuleMethod(protocol_enabled=True)
            else:
                method = SharedEvidenceProtocolMethod(item['method'], lambda ledger, path: DesignAgentClient(
                    plan.model, ledger, path, allow_live=True, settings=settings,
                    spend_ledger=PacedUsageLedger(book, pacer, unchanged, episode_ledger=ledger)), turns=plan.design_turns)
            result = run_design_episode(task, method, destination, episode_id=item['episode_id'])
            verification = verify_episode(destination)
            validate_episode_identity(result, task, item, record['original_source_hash'])
            return {**item, 'status': 'RECORDED', 'path': str(destination), 'terminal_state': result.terminal_state,
                'live_calls': result.common_contract['live_calls'], 'result_sha256': file_sha256(destination/'episode_result.json'),
                'verification': verification, 'natural_trace': item['method'] != 'central_rule' and result.live_status == 'RUN',
                'reference_only': item['method'] == 'central_rule', 'source': 'rule' if item['method'] == 'central_rule' else 'live'}
        except Exception as exc:
            body = {**item, 'status': 'EXECUTION_FAILED', 'reason': redact_sensitive(exc),
                    'error_type': type(exc).__name__, 'attempt_directory': str(destination)}
            immutable(output/'failed_attempts'/(item['episode_id']+'.json'), body)
            return body
    for start in range(0, len(pending), 18):
        batch = pending[start:start+18]
        if stopping:
            episodes.extend({**row, 'status': 'NOT_RUN_STOPPING_RULE', 'reason': stopping} for row in batch)
            save(); continue
        try:
            unchanged()
        except (ValueError, OSError):
            stopping = 'GOOGLE_RECOVERY_SOURCE_OR_INPUT_CHANGED'; book.block(stopping)
            episodes.extend({**row, 'status': 'NOT_RUN_STOPPING_RULE', 'reason': stopping} for row in batch)
            save(); continue
        book.begin_batch(batch[0]['batch_id'], [row['episode_id'] for row in batch])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(execute, batch))
        episodes.extend(results)
        failures = sum(row['status'] == 'EXECUTION_FAILED' or row.get('terminal_state') == 'METHOD_FAILED' for row in results)
        reason = 'THREE_FAILED_EPISODES_IN_REGISTERED_BATCH' if failures >= 3 else None
        if any(row['status'] == 'EXECUTION_FAILED' for row in results): reason = 'EXECUTION_INTEGRITY_OR_SERVICE_FAILURE'
        try: unchanged()
        except (ValueError, OSError): reason = 'GOOGLE_RECOVERY_SOURCE_OR_INPUT_CHANGED'
        after = book.close_batch(batch[0]['batch_id'], stop_reason=reason)
        verify_history(after, prior); prior = after; stopping = after['blocked_reason']
        save()
        print(json.dumps({'batch_id': batch[0]['batch_id'], 'failures': failures,
            'new_calls': after['totals']['calls'], 'stopping_reason': stopping}), flush=True)
    index = save()
    immutable(output/'allocation_final_snapshot.json', book.snapshot_envelope())
    immutable(output/'COMPLETED.json', dict(index_sha256=file_sha256(output/'benchmark_index.json'),
        registration_sha256=file_sha256(output/'registration.json'), recovery_registration_sha256=file_sha256(output/'recovery_registration.json'),
        recovery_registration=reference(output/'recovery_registration.json'), pacing_snapshot=reference(output/'pacing_snapshot.json'),
        complete_planned_index=len(episodes) == 270, complete_original_schedule_execution=all(r['status'] == 'RECORDED' for r in episodes),
        F72_automatic_acceptance=False))
    unchanged()
    return index
