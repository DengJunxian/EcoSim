"""Durable, atomic episode accounting. Issued resources and alpha never refund."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Callable

from pydantic import Field
from core.experiments.strict_types import StrictResearchModel, strict_json_loads
from core.experiments.identity import research_hash


class BudgetSpec(StrictResearchModel):
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    unique_market_runs: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    candidate_submissions: int = Field(ge=0)
    information_queries: int = Field(ge=0)
    messages: int = Field(ge=0)
    rounds: int = Field(ge=0)
    confirmation_batches: int = Field(ge=0)
    wall_timeout_seconds: float = Field(gt=0)


class ResourceVector(StrictResearchModel):
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    unique_market_runs: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=0, ge=0)
    candidate_submissions: int = Field(default=0, ge=0)
    information_queries: int = Field(default=0, ge=0)
    messages: int = Field(default=0, ge=0)
    rounds: int = Field(default=0, ge=0)
    confirmation_batches: int = Field(default=0, ge=0)


class BudgetExceeded(ValueError):
    pass


class BudgetLedger:
    """One owner/episode, shared by processes through an OS lock and atomic file.

    Only the composition root constructs this object. Consumers receive its
    reservation methods as a port. Pending reservations occupy the full cap.
    Token reservations are conservative maxima; actual usage is logged separately.
    """
    def __init__(self, path: str | Path, spec: BudgetSpec, *, owner: str, episode: str,
                 alpha_episode: float = 0.05, clock: Callable[[], float] = time.time):
        if not owner or not episode or type(alpha_episode) is not float or not 0 < alpha_episode < 1:
            raise ValueError('Explicit owner, episode and finite alpha in (0,1) required')
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec, self.owner, self.episode, self.clock = spec, owner, episode, clock
        with self._locked():
            if self.path.exists():
                state = self._read()
                if (state['spec'], state['owner'], state['episode'], state['alpha_episode']) != (
                        spec.model_dump(mode='json'), owner, episode, alpha_episode):
                    raise ValueError('BUDGET_RESTORE_SCOPE_MISMATCH')
            else:
                now = clock()
                self._write(dict(schema_version=1, spec=spec.model_dump(mode='json'), owner=owner,
                    episode=episode, started_at=now, deadline=now+spec.wall_timeout_seconds,
                    used=ResourceVector().model_dump(), alpha_episode=alpha_episode, alpha_reserved=0.0,
                    transactions={}, run_keys={}, replications={}, plans={}, visible_runs={}, actual_usage=[]))

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(self.path.suffix+'.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self):
        envelope = strict_json_loads(self.path.read_text())
        if research_hash(envelope['state']) != envelope['state_hash']:
            raise ValueError('BUDGET_STATE_HASH_MISMATCH')
        return envelope['state']

    def _write(self, state):
        temp = self.path.with_suffix('.tmp')
        with temp.open('w') as handle:
            json.dump({'state':state,'state_hash':research_hash(state)}, handle, allow_nan=False, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, self.path)
        fd=os.open(self.path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def snapshot(self):
        with self._locked(): return self._read()

    def _transaction(self, transaction_id, request, prepare, *, check_deadline=True):
        if type(transaction_id) is not str or not transaction_id: raise ValueError('Transaction ID required')
        with self._locked():
            state=self._read(); request_hash=research_hash(request)
            prior=state['transactions'].get(transaction_id)
            if prior:
                if prior['request_hash'] != request_hash: raise ValueError('TRANSACTION_ID_REUSED_WITH_DIFFERENT_INPUT')
                return prior
            if check_deadline and self.clock() >= state['deadline']: raise BudgetExceeded('WALL_DEADLINE_EXCEEDED')
            charges, details = prepare(state)
            vector=ResourceVector.model_validate(charges).model_dump()
            for name, amount in vector.items():
                if state['used'][name]+amount > getattr(self.spec,name):
                    raise BudgetExceeded('BUDGET_EXCEEDED: '+name)
            for name,amount in vector.items(): state['used'][name]+=amount
            receipt=dict(transaction_id=transaction_id,request_hash=request_hash,charges=vector,
                owner=self.owner,episode=self.episode,sequence=len(state['transactions'])+1,details=details)
            receipt['receipt_hash']=research_hash(receipt)
            state['transactions'][transaction_id]=receipt;self._write(state)
            return receipt

    def reserve(self, transaction_id: str, resources: ResourceVector, *, purpose: str):
        # Specialized units cannot be minted through the generic resource path.
        if resources.unique_market_runs or resources.max_attempts or resources.confirmation_batches:
            raise ValueError('Use the run, attempt or plan reservation port for specialized resources')
        return self._transaction(transaction_id,dict(resources=resources.model_dump(),purpose=purpose),
            lambda state:(resources.model_dump(),{'purpose':purpose}))

    def reserve_run(self, run_key: str, replication_id: str, *, plan_hash: str | None = None):
        request=dict(run_key=run_key,replication_id=replication_id,plan_hash=plan_hash)
        def prepare(state):
            if plan_hash is not None:
                plan=state['plans'].get(plan_hash)
                if not plan or replication_id not in plan['replication_ids'] or run_key not in plan['run_keys']:
                    raise ValueError('RUN_OUTSIDE_REGISTERED_PLAN')
            elif state['replications'].get(replication_id,{}).get('plan_hash') is not None:
                raise ValueError('CONFIRMATION_REPLICATION_RESERVED')
            first=run_key not in state['run_keys']
            if not first and state['run_keys'][run_key] != request: raise ValueError('RUN_KEY_SCOPE_MISMATCH')
            state['run_keys'][run_key]=request
            state['replications'].setdefault(replication_id,{'plan_hash':plan_hash})
            return {'unique_market_runs':int(first)}, request
        return self._transaction('run:'+run_key,request,prepare)

    def reserve_attempt(self, run_key: str, attempt_id: str):
        def prepare(state):
            if run_key not in state['run_keys']: raise ValueError('ATTEMPT_WITHOUT_RUN_RESERVATION')
            return {'max_attempts':1},dict(run_key=run_key,attempt_id=attempt_id)
        return self._transaction('attempt:'+attempt_id,dict(run_key=run_key,attempt_id=attempt_id),prepare)

    def reserve_confirmation(self, plan_hash: str, *, plan_id: str, replication_ids: list[str],
                             run_keys: list[str], alpha: float, alpha_episode: float):
        if type(alpha) is not float or not 0 < alpha < 1 or not replication_ids or not run_keys:
            raise ValueError('Invalid confirmation reservation')
        if len(set(replication_ids))!=len(replication_ids) or len(set(run_keys))!=len(run_keys):
            raise ValueError('Duplicate planned replication or run')
        request=dict(plan_hash=plan_hash,plan_id=plan_id,replication_ids=replication_ids,run_keys=run_keys,
                     alpha=alpha,alpha_episode=alpha_episode)
        def prepare(state):
            from fractions import Fraction
            if alpha_episode!=state['alpha_episode']: raise ValueError('EPISODE_ALPHA_SCOPE_MISMATCH')
            if any(p['plan_id']==plan_id for p in state['plans'].values()): raise ValueError('PLAN_ID_REUSED')
            if any(r in state['replications'] for r in replication_ids) or any(k in state['run_keys'] for k in run_keys):
                raise ValueError('CONFIRMATION_NOT_FRESH')
            total=sum((Fraction.from_float(p['alpha']) for p in state['plans'].values()),Fraction())+Fraction.from_float(alpha)
            if total>Fraction.from_float(alpha_episode): raise BudgetExceeded('ALPHA_EXCEEDED')
            state['alpha_reserved']=float(total); state['plans'][plan_hash]=request
            for replication_id in replication_ids: state['replications'][replication_id]={'plan_hash':plan_hash}
            return {'confirmation_batches':1},request
        return self._transaction('plan:'+plan_hash,request,prepare)

    def publish(self, transaction_id: str, run_keys: list[str]):
        def prepare(state):
            if not set(run_keys)<=set(state['run_keys']): raise ValueError('UNRESERVED_VISIBILITY')
            for key in run_keys: state['visible_runs'][key]=transaction_id
            return {},{'run_keys':run_keys}
        return self._transaction(transaction_id,{'publish':run_keys},prepare)

    def record_usage(self, transaction_id: str, usage: dict):
        def prepare(state):
            state['actual_usage'].append({'transaction_id':transaction_id,**usage})
            return {},usage
        return self._transaction('usage:'+transaction_id,usage,prepare,check_deadline=False)

    def reserve_llm(self, attempt_id: str, *, input_cap: int, output_cap: int, request_hash: str):
        return self.reserve('llm:'+attempt_id,ResourceVector(llm_calls=1,input_tokens=input_cap,
            output_tokens=output_cap),purpose='llm_request:'+request_hash)

    def settle_llm(self, attempt_id: str, *, input_tokens: int|None, output_tokens: int|None,
                   breakdown: dict, status: str):
        for value in (input_tokens,output_tokens):
            if value is not None and (type(value) is not int or value<0):raise ValueError('Invalid actual token usage')
        request=dict(attempt_id=attempt_id,input_tokens=input_tokens,output_tokens=output_tokens,
            breakdown=breakdown,status=status)
        def prepare(state):
            reservation=state['transactions'].get('llm:'+attempt_id)
            if not reservation:raise ValueError('LLM_USAGE_WITHOUT_RESERVATION')
            unknown=input_tokens is None or output_tokens is None
            excess={name:max(0,value-reservation['charges'][name]) for name,value in (
                ('input_tokens',input_tokens),('output_tokens',output_tokens)) if value is not None}
            # Actual excess is an accounting debt, not a reason to discard usage.
            for name,value in excess.items():state['used'][name]+=value
            usage={**request,'usage_status':'unknown' if unknown else 'reported',
                'reserved_input':reservation['charges']['input_tokens'],'reserved_output':reservation['charges']['output_tokens'],
                'over_reservation':any(excess.values()),'monetary_cost':None}
            state['actual_usage'].append(usage)
            return {},usage
        # No positive new resources: the debt remains even if a provider exceeded a
        # cap. Record before raising on the next reservation; never drop late usage.
        with self._locked():
            state=self._read();key='settle:'+attempt_id;digest=research_hash(request)
            prior=state['transactions'].get(key)
            if prior:
                if prior['request_hash']!=digest:raise ValueError('CONFLICTING_LLM_SETTLEMENT')
                return prior
            _,details=prepare(state)
            receipt=dict(transaction_id=key,request_hash=digest,charges=ResourceVector().model_dump(),
                owner=self.owner,episode=self.episode,sequence=len(state['transactions'])+1,details=details)
            receipt['receipt_hash']=research_hash(receipt);state['transactions'][key]=receipt;self._write(state)
            return receipt
