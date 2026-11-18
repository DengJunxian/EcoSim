"""SQLite Provider usage ledger: currency-separated estimates, finite plan resources, no amount cap.

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
from core.experiments.identity import research_hash, file_sha256
from core.experiments.strict_types import strict_json_loads


def verify_deepseek_retirement(reference):
    """Recompute inherited liabilities from immutable retired snapshots, not claims."""
    def checked(ref):
        if (set(ref) != {'path', 'sha256'} or not Path(ref['path']).is_absolute()
                or file_sha256(ref['path']) != ref['sha256']):
            raise ValueError('DEEPSEEK_RETIRED_HISTORY_CHANGED')
        return strict_json_loads(Path(ref['path']).read_text())
    def quantity(value):
        if type(value) is not int or value < 0:
            raise ValueError('DEEPSEEK_NONNEGATIVE_INTEGER_ACCOUNTING_REQUIRED')
        return value
    receipt = checked(reference)
    for field in ('known_settled_cost_nano', 'unknown_full_reservation_nano', 'released_unused_reservation_nano',
                  'available_for_new_allocations_nano', 'cap_nano', 'maximum_successor_calls',
                  'maximum_successor_full_reservation_nano'):
        quantity(receipt[field])
    if receipt['retired_no_further_execution'] is not True:
        raise ValueError('DEEPSEEK_RETIRED_PARENT_MUST_NOT_EXECUTE')
    refs, seen_files, seen_requests, known, unknown, reserved = {}, set(), set(), 0, 0, 0
    if not receipt['retired_ledger_snapshots']:
        raise ValueError('DEEPSEEK_RETIRED_PARENT_SNAPSHOT_REQUIRED')
    for ref in receipt['retired_ledger_snapshots']:
        absolute = str(Path(ref['path']).resolve())
        if absolute in seen_files:
            raise ValueError('DEEPSEEK_DUPLICATE_RETIRED_SNAPSHOT')
        seen_files.add(absolute)
        envelope = checked(ref); state = envelope['state']
        if research_hash(state) != envelope['sha256']:
            raise ValueError('DEEPSEEK_RETIRED_ENVELOPE_CHANGED')
        refs[ref['path']] = ref['sha256']
        contract = state['contract']
        if (contract['authorization_id'], contract['currency'], contract['cap_nano']) != (receipt['authorization_id'], 'CNY', 100_000_000_000):
            raise ValueError('DEEPSEEK_RETIRED_AUTHORIZATION_MISMATCH')
        prices = [quantity(contract[name]) for name in ('input_nano', 'output_nano', 'cached_nano')]
        if prices[2] > prices[0] or not set(state['settlements']) <= set(state['reservations']):
            raise ValueError('DEEPSEEK_RETIRED_ACCOUNTING_INCONSISTENT')
        for identity, reservation in state['reservations'].items():
            if identity in seen_requests:
                raise ValueError('DEEPSEEK_DUPLICATE_RETIRED_REQUEST_ID')
            seen_requests.add(identity)
            amount = quantity(reservation['reserved_nano'])
            if amount != quantity(reservation['input_cap'])*prices[0]+quantity(reservation['output_cap'])*prices[1]:
                raise ValueError('DEEPSEEK_RETIRED_FULL_RESERVATION_MISMATCH')
            reserved += amount
            settlement = state['settlements'].get(identity)
            cost = None if settlement is None else settlement['peak_price_cost_upper_bound_nano']
            if cost is None:
                unknown += amount
            else:
                quantity(cost)
                incoming, outgoing = quantity(settlement['input_tokens']), quantity(settlement['output_tokens'])
                cached = settlement.get('cached_tokens')
                cached = 0 if cached is None else quantity(cached)
                if cached > incoming or cost != (incoming-cached)*prices[0]+outgoing*prices[1]+cached*prices[2]:
                    raise ValueError('DEEPSEEK_RETIRED_KNOWN_USAGE_MISMATCH')
                known += cost
    if (known, unknown, reserved-known-unknown, 100_000_000_000-known-unknown) != (
            receipt['known_settled_cost_nano'], receipt['unknown_full_reservation_nano'],
            receipt['released_unused_reservation_nano'], receipt['available_for_new_allocations_nano']):
        raise ValueError('DEEPSEEK_RELEASE_LIABILITY_MISMATCH')
    audit = checked(receipt['audit']); refs[receipt['audit']['path']] = receipt['audit']['sha256']
    if (audit['status'], audit['ledger_known_upper_nano'], audit['unknown_full_reservation_nano'],
            audit['ledger_reservations'], audit['raw_call_records_verified'], audit['old_full_reserved_nano']) != (
            'VERIFIED', known, unknown, len(seen_requests), len(seen_requests), reserved):
        raise ValueError('DEEPSEEK_RELEASE_AUDIT_MISMATCH')
    for name in ('ledger_known_upper_nano', 'unknown_full_reservation_nano', 'ledger_reservations',
                 'raw_call_records_verified', 'old_full_reserved_nano'):
        quantity(audit[name])
    return receipt, refs


class ProviderUsageLedger:
    def __init__(self, path, *, authorization_id, plan_id, plan_hash,
                 maximum_calls, input_cap, output_cap,
                 provider, currency, input_nano, output_nano, cached_nano,
                 monetary_cap_nano=None, inherited_liability_nano=0,
                 source_guard=None, parent_reference=None):
        quantities = (maximum_calls, input_cap, output_cap)
        if any(type(x) is not int or x < 1 for x in quantities):
            raise ValueError('FINITE_POSITIVE_PROVIDER_PLAN_RESOURCES_REQUIRED')
        if any(type(x) is not int or x < 0 for x in (input_nano, output_nano, cached_nano)) or cached_nano > input_nano:
            raise ValueError('INVALID_PROVIDER_CACHE_PRICE')
        if not authorization_id or not plan_id or not plan_hash:
            raise ValueError('EXPLICIT_PROVIDER_PLAN_AUTHORIZATION_REQUIRED')
        if provider not in {'zhipu', 'deepseek', 'gemini'} or currency != ('USD' if provider == 'gemini' else 'CNY'):
            raise ValueError('PROVIDER_CURRENCY_SCOPE_REQUIRED')
        if type(inherited_liability_nano) is not int or inherited_liability_nano < 0:
            raise ValueError('PROVIDER_INVALID_INHERITED_LIABILITY')
        if monetary_cap_nano is not None and (type(monetary_cap_nano) is not int or monetary_cap_nano < 1):
            raise ValueError('PROVIDER_INVALID_MONETARY_CAP')
        if provider == 'deepseek' and (monetary_cap_nano != 100_000_000_000 or parent_reference is None):
            raise ValueError('DEEPSEEK_CNY100_RETIRED_PARENT_REQUIRED')
        if provider != 'deepseek' and (monetary_cap_nano is not None or inherited_liability_nano != 0):
            raise ValueError('PROVIDER_UNCAPPED_MONETARY_SCOPE_REQUIRED')
        maximum_reservation = maximum_calls*(input_cap*input_nano+output_cap*output_nano)
        if monetary_cap_nano is not None and inherited_liability_nano+maximum_reservation > monetary_cap_nano:
            raise ValueError('PROVIDER_FULL_PLAN_EXCEEDS_REMAINING_MONETARY_CAP')
        if parent_reference is not None:
            from core.experiments.identity import file_sha256
            if (set(parent_reference) != {'path', 'sha256'}
                    or not Path(parent_reference['path']).is_absolute()
                    or file_sha256(parent_reference['path']) != parent_reference['sha256']):
                raise ValueError('PROVIDER_PARENT_HISTORY_REFERENCE_INVALID')
        if provider == 'deepseek':
            receipt, _ = verify_deepseek_retirement(parent_reference)
            if (receipt['provider'], receipt['currency'], receipt['cap_nano'], receipt['retired_no_further_execution'],
                    receipt['allowed_successor_plan_id'], receipt['allowed_successor_ledger_path']) != (
                    'deepseek', 'CNY', 100_000_000_000, True, plan_id, str(Path(path).resolve())):
                raise ValueError('DEEPSEEK_RELEASE_SUCCESSOR_SCOPE_MISMATCH')
            if (receipt['known_settled_cost_nano']+receipt['unknown_full_reservation_nano'] != inherited_liability_nano
                    or receipt['available_for_new_allocations_nano'] != monetary_cap_nano-inherited_liability_nano
                    or receipt['authorization_id'] != authorization_id):
                raise ValueError('DEEPSEEK_RELEASE_LIABILITY_MISMATCH')
            if (receipt['maximum_successor_calls'] != maximum_calls
                    or receipt['maximum_successor_full_reservation_nano'] != maximum_reservation):
                raise ValueError('DEEPSEEK_RELEASE_ALLOCATION_CAP_MISMATCH')
            for ref in receipt['retired_ledger_snapshots']:
                if not Path(ref['path']).is_absolute() or file_sha256(ref['path']) != ref['sha256']:
                    raise ValueError('DEEPSEEK_RETIRED_HISTORY_CHANGED')
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source_guard = source_guard
        self.contract = dict(schema_version=1, authorization_id=authorization_id,
            plan_id=plan_id, plan_hash=plan_hash, provider=provider, currency=currency,
            monetary_cap_nano=monetary_cap_nano, inherited_liability_nano=inherited_liability_nano,
            authorization_basis='explicit_provider_scoped_finite_plan',
            maximum_calls=maximum_calls, maximum_input_tokens=maximum_calls*input_cap,
            maximum_output_tokens=maximum_calls*output_cap,
            per_call_input_cap=input_cap, per_call_output_cap=output_cap,
            input_nano=input_nano, output_nano=output_nano, cached_nano=cached_nano,
            reservation_rule='provider-plan-full-token-caps-no-refund-v1',
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
                raise ValueError('PROVIDER_USAGE_CONTRACT_MISMATCH')

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
            raise ValueError('UNIQUE_EXPLICIT_PROVIDER_BATCH_REQUIRED')
        with self._db() as db:
            if self._get(db, 'blocked_reason') or self._get(db, 'pending_block_reasons'):
                raise BudgetExceeded('PROVIDER_USAGE_BLOCKED')
            if self._get(db, 'active_batch') is not None:
                raise ValueError('PROVIDER_BATCH_ALREADY_ACTIVE')
            if any(b['batch_id'] == batch_id for b in self._get(db, 'closed_batches')):
                raise ValueError('PROVIDER_BATCH_ID_REUSED')
            self._put(db, 'active_batch', dict(batch_id=batch_id, episode_ids=list(episode_ids), started_at=time.time()))

    def close_batch(self, batch_id, *, stop_reason=None):
        with self._db() as db:
            batch = self._get(db, 'active_batch')
            if batch is None or batch['batch_id'] != batch_id:
                raise ValueError('PROVIDER_ACTIVE_BATCH_MISMATCH')
            if stop_reason:
                self._pending(db, stop_reason)
            missing = db.execute('SELECT COUNT(*) FROM reservations r LEFT JOIN settlements s ON r.identity=s.identity WHERE s.identity IS NULL').fetchone()[0]
            if missing:
                self._pending(db, 'PROVIDER_UNSETTLED_REQUESTS')
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
            raise ValueError('INVALID_PROVIDER_REQUEST_RESERVATION')
        if incoming != self.contract['per_call_input_cap'] or outgoing != self.contract['per_call_output_cap']:
            raise ValueError('PROVIDER_REQUEST_CAP_CHANGED')
        with self._db() as db:
            if self._get(db, 'contract') != self.contract:
                raise ValueError('PROVIDER_USAGE_CONTRACT_CHANGED')
            if self._get(db, 'blocked_reason'):
                raise BudgetExceeded('PROVIDER_USAGE_BLOCKED')
            batch = self._get(db, 'active_batch')
            if batch is None or identity.rsplit(':', 1)[0] not in batch['episode_ids']:
                raise ValueError('REQUEST_OUTSIDE_REGISTERED_ACTIVE_BATCH')
            if db.execute('SELECT 1 FROM reservations WHERE identity=?', (identity,)).fetchone():
                raise ValueError('PROVIDER_REQUEST_ID_REUSED')
            totals = self._get(db, 'totals')
            charges = dict(calls=1, input_tokens=incoming, output_tokens=outgoing,
                           reserved_nano=incoming*self.contract['input_nano']+outgoing*self.contract['output_nano'])
            cap = self.contract['monetary_cap_nano']
            if cap is not None and self.contract['inherited_liability_nano']+totals['reserved_nano']+charges['reserved_nano'] > cap:
                raise BudgetExceeded('PROVIDER_CUMULATIVE_MONETARY_CAP_EXHAUSTED')
            for key, limit in [('calls', 'maximum_calls'), ('input_tokens', 'maximum_input_tokens'), ('output_tokens', 'maximum_output_tokens')]:
                if totals[key] + charges[key] > self.contract[limit]:
                    raise BudgetExceeded('PROVIDER_FINITE_PLAN_CAP_EXHAUSTED:' + key)
            body = dict(input_cap=incoming, output_cap=outgoing, request_hash=request_hash,
                        reserved_nano=charges['reserved_nano'], currency=self.contract['currency'], batch_id=batch['batch_id'], time=time.time())
            db.execute('INSERT INTO reservations VALUES (?,?)', (identity, json.dumps(body, sort_keys=True)))
            self._put(db, 'totals', {key: totals[key]+charges[key] for key in totals})
        return body

    def settle(self, identity, incoming, outgoing, cached=None):
        with self._db() as db:
            row = db.execute('SELECT body FROM reservations WHERE identity=?', (identity,)).fetchone()
            if row is None:
                raise ValueError('PROVIDER_SETTLEMENT_WITHOUT_RESERVATION')
            if db.execute('SELECT 1 FROM settlements WHERE identity=?', (identity,)).fetchone():
                raise ValueError('DUPLICATE_PROVIDER_SETTLEMENT')
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
                peak_price_cost_upper_bound_nano=cost, currency=self.contract['currency'], billed_cost=None,
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
            raise ValueError('PROVIDER_USAGE_CONTRACT_CHANGED')
        totals = dict(calls=len(state['reservations']),
            input_tokens=sum(r['input_cap'] for r in state['reservations'].values()),
            output_tokens=sum(r['output_cap'] for r in state['reservations'].values()),
            reserved_nano=sum(r['reserved_nano'] for r in state['reservations'].values()))
        if totals != state['totals'] or not set(state['settlements']) <= set(state['reservations']):
            raise ValueError('PROVIDER_USAGE_HISTORY_INCONSISTENT')
        return state

    def snapshot_envelope(self):
        state = self.snapshot()
        return {'state': state, 'sha256': research_hash(state)}
