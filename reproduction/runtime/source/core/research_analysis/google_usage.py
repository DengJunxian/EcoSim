"""SQLite Google usage ledger: USD estimates, finite plan resources, no amount cap.

Unknown counts remain unknown. A fixed batch may finish after a pending stop;
closing that batch makes the stop durable before another batch can start.
"""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from core.codesign.budget import BudgetExceeded
from core.experiments.identity import research_hash
from core.experiments.strict_types import strict_json_loads


class GoogleUsageLedger:
    def __init__(self, path, *, authorization_id, plan_id, plan_hash,
                 maximum_calls, input_cap, output_cap,
                 input_nano=300, output_nano=2500, cached_nano=30,
                 source_guard=None, parent_reference=None):
        quantities = (maximum_calls, input_cap, output_cap, input_nano, output_nano)
        if any(type(x) is not int or x < 1 for x in quantities):
            raise ValueError('FINITE_POSITIVE_GOOGLE_PLAN_RESOURCES_REQUIRED')
        if type(cached_nano) is not int or not 0 <= cached_nano <= input_nano:
            raise ValueError('INVALID_GOOGLE_CACHE_PRICE')
        if not authorization_id or not plan_id or not plan_hash:
            raise ValueError('EXPLICIT_GOOGLE_PLAN_AUTHORIZATION_REQUIRED')
        if parent_reference is not None:
            from core.experiments.identity import file_sha256
            if (set(parent_reference) != {'path', 'sha256'}
                    or not Path(parent_reference['path']).is_absolute()
                    or file_sha256(parent_reference['path']) != parent_reference['sha256']):
                raise ValueError('GOOGLE_PARENT_HISTORY_REFERENCE_INVALID')
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source_guard = source_guard
        self.contract = dict(schema_version=1, authorization_id=authorization_id,
            plan_id=plan_id, plan_hash=plan_hash, provider='gemini', currency='USD',
            monetary_cap=None, authorization_basis='user_no_monetary_cap_finite_registered_plan',
            maximum_calls=maximum_calls, maximum_input_tokens=maximum_calls*input_cap,
            maximum_output_tokens=maximum_calls*output_cap,
            per_call_input_cap=input_cap, per_call_output_cap=output_cap,
            input_nano=input_nano, output_nano=output_nano, cached_nano=cached_nano,
            reservation_rule='google-plan-full-token-caps-no-refund-v1',
            stop_rule='pending_stop_effective_after_registered_batch_v1',
            parent_reference=parent_reference)
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS reservations (identity TEXT PRIMARY KEY, body TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS settlements (identity TEXT PRIMARY KEY, body TEXT NOT NULL)')
            old = self._get(db, 'contract')
            if old is None:
                self._put(db, 'contract', self.contract)
                self._put(db, 'created_at', time.time())
                self._put(db, 'blocked_reason', None)
                self._put(db, 'pending_block_reasons', [])
                self._put(db, 'active_batch', None)
                self._put(db, 'closed_batches', [])
                self._put(db, 'totals', dict(calls=0, input_tokens=0, output_tokens=0, reserved_nano=0))
            elif old != self.contract:
                raise ValueError('GOOGLE_USAGE_CONTRACT_MISMATCH')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=60.0)
        try:
            db.execute('PRAGMA busy_timeout=60000')
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _get(db, key):
        row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return None if row is None else strict_json_loads(row[0])

    @staticmethod
    def _put(db, key, value):
        db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)',
                   (key, json.dumps(value, allow_nan=False, sort_keys=True)))

    def _pending(self, db, reason):
        reasons = self._get(db, 'pending_block_reasons')
        if reason not in reasons:
            reasons.append(reason)
            self._put(db, 'pending_block_reasons', reasons)
        if self._get(db, 'active_batch') is None and not self._get(db, 'blocked_reason'):
            self._put(db, 'blocked_reason', reason)

    def begin_batch(self, batch_id, episode_ids):
        if not batch_id or not episode_ids or len(episode_ids) != len(set(episode_ids)):
            raise ValueError('UNIQUE_EXPLICIT_GOOGLE_BATCH_REQUIRED')
        with self._db() as db:
            if self._get(db, 'blocked_reason') or self._get(db, 'pending_block_reasons'):
                raise BudgetExceeded('GOOGLE_USAGE_BLOCKED')
            if self._get(db, 'active_batch') is not None:
                raise ValueError('GOOGLE_BATCH_ALREADY_ACTIVE')
            if any(b['batch_id'] == batch_id for b in self._get(db, 'closed_batches')):
                raise ValueError('GOOGLE_BATCH_ID_REUSED')
            self._put(db, 'active_batch', dict(batch_id=batch_id, episode_ids=list(episode_ids), started_at=time.time()))

    def close_batch(self, batch_id, *, stop_reason=None):
        with self._db() as db:
            batch = self._get(db, 'active_batch')
            if batch is None or batch['batch_id'] != batch_id:
                raise ValueError('GOOGLE_ACTIVE_BATCH_MISMATCH')
            if stop_reason:
                self._pending(db, stop_reason)
            missing = db.execute('SELECT COUNT(*) FROM reservations r LEFT JOIN settlements s ON r.identity=s.identity WHERE s.identity IS NULL').fetchone()[0]
            if missing:
                self._pending(db, 'GOOGLE_UNSETTLED_REQUESTS')
            reasons = self._get(db, 'pending_block_reasons')
            if reasons:
                self._put(db, 'blocked_reason', reasons[0])
            closed = self._get(db, 'closed_batches')
            closed.append({**batch, 'closed_at': time.time(), 'stop_reasons': reasons})
            self._put(db, 'closed_batches', closed)
            self._put(db, 'active_batch', None)
        return self.snapshot()

    def reserve(self, identity, incoming, outgoing, request_hash):
        if self.source_guard:
            self.source_guard()
        if any(type(x) is not int or x < 1 for x in (incoming, outgoing)):
            raise ValueError('INVALID_GOOGLE_REQUEST_RESERVATION')
        if incoming != self.contract['per_call_input_cap'] or outgoing != self.contract['per_call_output_cap']:
            raise ValueError('GOOGLE_REQUEST_CAP_CHANGED')
        with self._db() as db:
            if self._get(db, 'contract') != self.contract:
                raise ValueError('GOOGLE_USAGE_CONTRACT_CHANGED')
            if self._get(db, 'blocked_reason'):
                raise BudgetExceeded('GOOGLE_USAGE_BLOCKED')
            batch = self._get(db, 'active_batch')
            if batch is None or identity.rsplit(':', 1)[0] not in batch['episode_ids']:
                raise ValueError('REQUEST_OUTSIDE_REGISTERED_ACTIVE_BATCH')
            if db.execute('SELECT 1 FROM reservations WHERE identity=?', (identity,)).fetchone():
                raise ValueError('GOOGLE_REQUEST_ID_REUSED')
            totals = self._get(db, 'totals')
            charges = dict(calls=1, input_tokens=incoming, output_tokens=outgoing,
                           reserved_nano=incoming*self.contract['input_nano']+outgoing*self.contract['output_nano'])
            for key, limit in [('calls', 'maximum_calls'), ('input_tokens', 'maximum_input_tokens'), ('output_tokens', 'maximum_output_tokens')]:
                if totals[key] + charges[key] > self.contract[limit]:
                    raise BudgetExceeded('GOOGLE_FINITE_PLAN_CAP_EXHAUSTED:' + key)
            body = dict(input_cap=incoming, output_cap=outgoing, request_hash=request_hash,
                        reserved_nano=charges['reserved_nano'], currency='USD', batch_id=batch['batch_id'], time=time.time())
            db.execute('INSERT INTO reservations VALUES (?,?)', (identity, json.dumps(body, sort_keys=True)))
            self._put(db, 'totals', {key: totals[key]+charges[key] for key in totals})
        return body

    def settle(self, identity, incoming, outgoing, cached=None):
        with self._db() as db:
            row = db.execute('SELECT body FROM reservations WHERE identity=?', (identity,)).fetchone()
            if row is None:
                raise ValueError('GOOGLE_SETTLEMENT_WITHOUT_RESERVATION')
            if db.execute('SELECT 1 FROM settlements WHERE identity=?', (identity,)).fetchone():
                raise ValueError('DUPLICATE_GOOGLE_SETTLEMENT')
            reservation = strict_json_loads(row[0])
            known = all(type(x) is int and x >= 0 for x in (incoming, outgoing))
            if not known:
                self._pending(db, 'UNKNOWN_USAGE_REQUIRES_RECONCILIATION')
            if cached is not None and (type(cached) is not int or cached < 0 or (known and cached > incoming)):
                known = False
                self._pending(db, 'INVALID_CACHE_USAGE')
            if known and (incoming > reservation['input_cap'] or outgoing > reservation['output_cap']):
                self._pending(db, 'PROVIDER_EXCEEDED_RESERVED_TOKEN_CAP')
            cost = None if not known else ((incoming-(cached or 0))*self.contract['input_nano']
                   +(cached or 0)*self.contract['cached_nano']+outgoing*self.contract['output_nano'])
            body = dict(input_tokens=incoming, output_tokens=outgoing, cached_tokens=cached,
                peak_price_cost_upper_bound_nano=cost, currency='USD', billed_cost_usd=None,
                usage_status='known' if known else 'unknown', time=time.time())
            db.execute('INSERT INTO settlements VALUES (?,?)', (identity, json.dumps(body, allow_nan=False, sort_keys=True)))
        return body

    def block(self, reason):
        with self._db() as db:
            self._pending(db, reason)

    def snapshot(self):
        with self._db() as db:
            state = {row[0]: strict_json_loads(row[1]) for row in db.execute('SELECT key,value FROM meta')}
            for name in ('reservations', 'settlements'):
                state[name] = {r[0]: strict_json_loads(r[1]) for r in db.execute(f'SELECT identity,body FROM {name} ORDER BY identity')}
        if state['contract'] != self.contract:
            raise ValueError('GOOGLE_USAGE_CONTRACT_CHANGED')
        totals = dict(calls=len(state['reservations']),
            input_tokens=sum(r['input_cap'] for r in state['reservations'].values()),
            output_tokens=sum(r['output_cap'] for r in state['reservations'].values()),
            reserved_nano=sum(r['reserved_nano'] for r in state['reservations'].values()))
        if totals != state['totals'] or not set(state['settlements']) <= set(state['reservations']):
            raise ValueError('GOOGLE_USAGE_HISTORY_INCONSISTENT')
        return state

    def snapshot_envelope(self):
        state = self.snapshot()
        return {'state': state, 'sha256': research_hash(state)}
