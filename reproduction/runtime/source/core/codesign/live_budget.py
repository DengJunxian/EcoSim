"""Study-wide CNY ceiling: integer nanoyuan, durable reservations, no refunds."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import time
from core.experiments.archive import durable_json
from core.experiments.identity import research_hash
from core.experiments.strict_types import strict_json_loads
from .budget import BudgetExceeded


class SpendLedger:
    def __init__(self, path, *, authorization_id, cap_cny=100, input_nano=2000, output_nano=8000, cached_nano=40):
        if type(cap_cny) is not int or not 0 < cap_cny <= 100:
            raise ValueError('INVALID_CNY_CAP')
        if any(type(v) is not int or v < 0 for v in (input_nano, output_nano, cached_nano)):
            raise ValueError('INVALID_PRICES')
        if input_nano < cached_nano:raise ValueError('INVALID_CACHE_PRICE')
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        self.contract={'authorization_id':authorization_id,'cap_nano':cap_cny*10**9,
            'input_nano':input_nano,'output_nano':output_nano,'cached_nano':cached_nano,
            'currency':'CNY','reservation_rule':'peak-price-full-cap-no-refund-v1'}
        with self.lock():
            if self.path.exists():
                if self.read()['contract']!=self.contract:raise ValueError('SPEND_CONTRACT_MISMATCH')
            else:self.write({'contract':self.contract,'reservations':{},'settlements':{},'blocked_reason':None,'created_at':time.time()})

    @contextmanager
    def lock(self):
        with self.path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(handle,fcntl.LOCK_UN)

    def read(self):
        obj=strict_json_loads(self.path.read_text())
        if research_hash(obj['state'])!=obj['sha256']:raise ValueError('SPEND_HASH_MISMATCH')
        return obj['state']

    def write(self,state):durable_json(self.path,{'state':state,'sha256':research_hash(state)})

    def snapshot(self):
        with self.lock():return self.read()

    def reserve(self, identity, incoming, outgoing, request_hash):
        if any(type(v) is not int or v < 1 for v in (incoming,outgoing)):raise ValueError('INVALID_SPEND_RESERVATION')
        body={'input_cap':incoming,'output_cap':outgoing,'request_hash':request_hash,
            'reserved_nano':incoming*self.contract['input_nano']+outgoing*self.contract['output_nano']}
        with self.lock():
            state=self.read()
            if identity in state['reservations']:raise ValueError('SPEND_CALL_ID_REUSED')
            if state['blocked_reason']:raise BudgetExceeded('SPEND_BLOCKED:'+state['blocked_reason'])
            if sum(r['reserved_nano'] for r in state['reservations'].values())+body['reserved_nano']>self.contract['cap_nano']:
                raise BudgetExceeded('TOTAL_CNY_CAP_EXHAUSTED')
            state['reservations'][identity]={**body,'time':time.time()};self.write(state)
        return body

    def settle(self,identity,incoming,outgoing,cached=None):
        with self.lock():
            state=self.read();reserved=state['reservations'][identity]
            if identity in state['settlements']:raise ValueError('DUPLICATE_SPEND_SETTLEMENT')
            known=all(type(x) is int and x>=0 for x in (incoming,outgoing))
            if not known:state['blocked_reason']='UNKNOWN_USAGE_REQUIRES_RECONCILIATION'
            if cached is not None and (type(cached) is not int or cached<0 or (known and cached>incoming)):
                known=False;state['blocked_reason']='INVALID_CACHE_USAGE'
            if known and (incoming>reserved['input_cap'] or outgoing>reserved['output_cap']):
                state['blocked_reason']='PROVIDER_EXCEEDED_RESERVED_TOKEN_CAP'
            bound=None if not known else ((incoming-(cached or 0))*self.contract['input_nano']+
                (cached or 0)*self.contract['cached_nano']+outgoing*self.contract['output_nano'])
            body={'input_tokens':incoming,'output_tokens':outgoing,'cached_tokens':cached,
                'peak_price_cost_upper_bound_nano':bound,'billed_cost_cny':None,'time':time.time()}
            state['settlements'][identity]=body;self.write(state);return body

    def block(self,reason):
        with self.lock():
            state=self.read();state['blocked_reason']=reason;self.write(state)


def load_deepseek_settings(path=None):
    """Read one explicitly selected private env file. Never eval/export/print it."""
    from core.llm.config import LLMSettings
    key=''
    if path is not None:
        target=Path(path).expanduser()
        if not target.is_file():raise ValueError('LOCAL_CREDENTIAL_FILE_MISSING')
        if stat.S_IMODE(target.stat().st_mode)&0o077:raise ValueError('CREDENTIAL_FILE_REQUIRES_MODE_600')
        lines=[line.strip() for line in target.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        matches=[line.split('=',1)[1].strip().strip('\"').strip("'") for line in lines if line.startswith('DEEPSEEK_API_KEY=')]
        if len(matches)!=1:raise ValueError('ONE_DEEPSEEK_API_KEY_REQUIRED')
        key=matches[0]
    else:key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not key or not key.startswith('sk-') or any(c.isspace() for c in key):raise ValueError('LOCAL_CREDENTIAL_NOT_CONFIGURED')
    return LLMSettings(deepseek_api_key=key,zhipu_api_key='',default_provider='deepseek',
        deepseek_base_url='https://api.deepseek.com',max_retries=0,timeout_seconds=60.0)
