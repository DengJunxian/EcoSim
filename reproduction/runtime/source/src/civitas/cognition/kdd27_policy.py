"""Common activity gate and resource-relative strategy cache; no live provider here."""
import math
from typing import Literal

from pydantic import Field, model_validator

from civitas.contracts.base import NonNegative, Record
from civitas.contracts.decisions import HoldIntent
from civitas.contracts.observations import ObservationFact
from civitas.contracts.portfolio import PortfolioDecisionBundle, PortfolioOrderIntent
from civitas.contracts.time import SimTime
from civitas.core.observation import ObservationBuilder
from civitas.core.rng import AddressedRNG
from civitas.mechanisms.kdd27 import max_quantity, quote_ticks, rule_quantity
from civitas.scenarios.kdd27_config import Probability
from .memory import AgentMemory
from .observations import validate_bundle
from .policy import empty_bundle
from .runtime import RunControlError
from .kdd27_persona import INSTRUCTION, PROMPT_VERSION, PARSER_VERSION, heuristic_allocation, persona_contract


class StrategyCache(Record):
    action: Literal['buy','sell','hold']
    fraction: Probability
    created_tick: NonNegative
    attempt_id: str

    @model_validator(mode='after')
    def hold_fraction(self):
        if self.action=='hold' and self.fraction!=0.0:
            raise ValueError('HOLD cannot cache a resource allocation')
        return self


class PaperObserver(ObservationBuilder):
    compact_snapshots=True
    window=1

    def __init__(self, config=None):
        self.config=config if getattr(config,'cognition_contract',None) else None
        if self.config is not None:self.window=None

    def select_memories(self, actor_id, snapshot):
        if self.config is None:return []
        now=snapshot.tick
        receipts=sorted((f for f in snapshot.memories if f.owner_id==actor_id and f.fact_id.startswith('memory:')
            and f.visible_at.key<=SimTime(tick=now,phase='OBSERVE').key),
            key=lambda f:(f.visible_at.key,f.fact_id),reverse=True)[:self.config.memory_items]
        # Kernel receipts become visible one tick after their creation.
        return [f.model_copy(update={'value':{**f.value,'memory_layer':'episodic',
            'memory_age':now-f.visible_at.tick+1,'memory_weight':math.exp(-(now-f.visible_at.tick+1)/self.config.memory_decay_steps)}})
            for f in receipts]

    def observe(self, actor_id, snapshot, candidates):
        if self.config is not None:
            candidates=[f.model_copy(update={'value':{**f.value,'persona_contract':persona_contract(f.value['profile']['archetype'])}})
                if f.fact_id.startswith('paper-own:') else f for f in candidates]
        hidden=getattr(self,'feed_hidden_from',None)
        if hidden is not None and snapshot.tick>=hidden:
            candidates=[f.model_copy(update={'value':{**f.value,'feed':[]}}) if f.fact_id.startswith('paper-own:') else f
                        for f in candidates]
        memories=[f for f in snapshot.memories if f.owner_id==actor_id and f.fact_id.startswith('belief:')
            and f.visible_at.tick<=snapshot.tick]
        latest=max(memories,key=lambda f:f.visible_at.key) if memories else None
        cache=None if latest is None else latest.value.get('paper_cache')
        if cache is not None:
            cache=StrategyCache.model_validate(cache)
            if cache.created_tick>=snapshot.tick:
                raise ValueError('Strategy cache cannot originate in the current or future decision')
        candidate=ObservationFact(fact_id=f'paper-cache:{snapshot.tick}:{actor_id}',category='memory',owner_id=actor_id,
            visible_at={'tick':snapshot.tick,'phase':'OBSERVE'},value={'strategy':None if cache is None else cache.model_dump()})
        return super().observe(actor_id,snapshot,[*candidates,candidate])


class PaperPolicy:
    policy_id='kdd27_policy'
    version='1'

    def __init__(self, config, runtime=None):
        self.config,self.runtime=config,runtime
        if getattr(config,'cognition_contract',None):self.version='2'
        self.rng=AddressedRNG(config.root_seed,config.replica)
        self.mechanism_versions={'paper':config.mechanism_version,'rng':config.rng_version}
        if (runtime is None)!=(config.cognition_mode=='rule'):
            raise ValueError('Paper cognition source must match the supplied runtime')
        if runtime is not None and runtime.config.source!=config.cognition_mode:
            raise ValueError('Paper runtime provenance mismatch')

    def intent(self, actor, tick, side, quantity, price):
        identifier=f'paper-intent:{tick}:{actor}'
        if side=='hold' or quantity==0:
            return HoldIntent(intent_id=identifier)
        return PortfolioOrderIntent(intent_id=identifier,order_id=f'paper-order:{tick}:{actor}',
            asset_id=self.config.market.asset_id,side=side,quantity_units=quantity,price_ticks=price)

    async def decide(self, observation):
        tick=observation.visible_until.tick; actor=observation.actor_id; cfg=self.config
        account=observation.account_view[0].value
        market=next(f.value for f in observation.market_view if f.fact_id.startswith('paper-market:'))
        own=next(f.value for f in observation.private_memory_refs if f.fact_id.startswith('paper-own:'))
        cached=next(f.value['strategy'] for f in observation.private_memory_refs if f.fact_id.startswith('paper-cache:'))
        cache=None if cached is None else StrategyCache.model_validate(cached)
        price=quote_ticks(cfg.behavior,market['historical_quote']['last_price_ticks'],self.rng,actor,tick)
        active=own['active']; probabilities=own['behavior']['action_probabilities']
        action_draw=self.rng.uniform('kdd27.decision',actor,tick,'action')
        fraction_draw=self.rng.uniform('kdd27.decision',actor,tick,'quantity')
        side='buy' if action_draw<probabilities[0] else 'sell' if action_draw<sum(probabilities[:2]) else 'hold'
        scale=own['state']['risk'] if side=='buy' else max(-own['behavior']['intention'],0.0)
        fraction=fraction_draw*(cfg.behavior.resource_fraction_min+(cfg.behavior.resource_fraction_max-cfg.behavior.resource_fraction_min)*scale)
        quantity=rule_quantity(side,fraction,account,price,cfg.market,market['commission'])
        intent=self.intent(actor,tick,side,quantity,price)
        path='rule'; reason=None; attempt=None; exact_hit=False; covered=False
        heuristic=None
        if cfg.policy_backend=='heuristic':
            side,fraction,heuristic=heuristic_allocation(own,account,market,cfg.market)
            quantity=rule_quantity(side,fraction,account,price,cfg.market,market['commission'])
            intent=self.intent(actor,tick,side,quantity,price);path='heuristic'
        scheduled=active and (cfg.policy_backend=='llm' or
            (cfg.policy_backend=='hybrid' and own['strategic'] and tick%cfg.slow_period==0))
        bundle=empty_bundle(observation,self.policy_id,self.version,PortfolioDecisionBundle)
        if not active:
            intent=self.intent(actor,tick,'hold',0,price); path='inactive'
        elif scheduled:
            def validate(value):
                validate_bundle(value,observation,self.policy_id,self.version)
                if (value.message_drafts or value.belief_patch or len(value.economic_intents)!=1
                        or not isinstance(value.economic_intents[0],(HoldIntent,PortfolioOrderIntent))):
                    raise ValueError('Paper direct action requires exactly one order/HOLD and no state overlay')
                if isinstance(value.economic_intents[0],PortfolioOrderIntent) and value.economic_intents[0].asset_id!=cfg.market.asset_id:
                    raise ValueError('Paper action names an unknown asset')
            try:
                result=await self.runtime.request(observation,AgentMemory(actor_id=actor),policy_id=self.policy_id,
                    policy_version=self.version,task='direct',sample_id=getattr(cfg,'cognitive_sample_id','paper-method-check:0'),
                    response_model=PortfolioDecisionBundle,validator=validate,
                    prompt_version=PROMPT_VERSION if self.version=='2' else 'kdd27-interface-check-v1',parser_version=PARSER_VERSION,
                    mechanism_versions=self.mechanism_versions,institution_versions={'market':'kdd27-market-v1'},
                    task_context={'identity':bundle.model_dump(),'instruction':INSTRUCTION if self.version=='2' else
                        'Return one typed order or HOLD from the visible facts. No belief or message overlay.'})
                intent=result.value.economic_intents[0]
                if isinstance(intent,PortfolioOrderIntent) and intent.quantity_units==0:
                    intent=self.intent(actor,tick,'hold',0,price)
                bundle=bundle.model_copy(update={'evidence_ids':result.value.evidence_ids})
                attempt=result.receipt.attempt_id; exact_hit=result.cache_hit; path='model'; covered=True
                side=intent.side if isinstance(intent,PortfolioOrderIntent) else 'hold'
                maximum=max_quantity(side,account,intent.price_ticks if side!='hold' else price,cfg.market,market['commission'])
                ratio=min(1.0,intent.quantity_units/maximum) if maximum and side!='hold' else 0.0
                cache=StrategyCache(action=side,fraction=ratio,created_tick=tick,attempt_id=attempt)
            except RunControlError as exc:
                if str(exc).split(':',1)[0] not in {'invalid_response','timeout','response_incomplete','response_output_invalid'}:
                    raise
                intent=self.intent(actor,tick,'hold',0,price); cache=None; path='degraded_hold'; reason=str(exc)
        elif cfg.policy_backend=='hybrid' and own['strategic']:
            if cache is not None and tick-cache.created_tick<=cfg.maximum_cache_age:
                maximum=max_quantity(cache.action,account,price,cfg.market,market['commission'])
                intent=self.intent(actor,tick,cache.action,int(cache.fraction*maximum),price)
                path='strategy_cache'; covered=True
            else:
                path='rule_fallback'; reason='missing_cache' if cache is None else 'expired_cache'
        trace={'active':active,'path':path,'scheduled':scheduled,'degraded':path=='degraded_hold','reason':reason,
            'covered':covered,'attempt_id':attempt,'exact_response_cache_hit':exact_hit,
            'strategy_age':None if cache is None else tick-cache.created_tick,'shared_quote_ticks':price,
            'action_draw':action_draw,'quantity_draw':fraction_draw,'rule_fraction':fraction,
            'model_origin':'rule' if cfg.cognition_mode=='rule' else 'synthetic_test',
            'decision_tick':tick,'observation_id':observation.observation_id,
            'rng_slots':['kdd27.decision/action','kdd27.decision/quantity','kdd27.decision/quote:radius','kdd27.decision/quote:angle']}
        if self.version=='2':
            trace.update(cognition_contract=cfg.cognition_contract,heuristic=heuristic,
                visible_receipt_memories=sum(f.value.get('memory_layer')=='episodic' for f in observation.private_memory_refs))
        return bundle.model_copy(update={'economic_intents':[intent],
            'belief_patch':{'paper_decision':trace,'paper_cache':None if cache is None else cache.model_dump()}})
