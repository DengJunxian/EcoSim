"""Paper timing and state, with the existing market as sole settlement authority."""
import math
from fractions import Fraction
from typing import Literal

from pydantic import Field, model_validator

from civitas.contracts.base import Id, NonNegative, Record, content_hash
from civitas.contracts.events import Event
from civitas.contracts.observations import ObservationFact
from civitas.contracts.portfolio import AssetSpec, FeeSchedule, PortfolioExecutionReceipt, PortfolioOrderIntent
from civitas.contracts.state import BatchExecutionResult, ExecutionResult, Feedback
from civitas.core.rng import AddressedRNG
from civitas.markets.multiasset import MultiAssetMarket
from civitas.mechanisms.kdd27 import behavior_update, social_update, rq2_parameters
from civitas.population.kdd27 import build_population
from civitas.resources.ledger import PortfolioLedger
from .kdd27_config import parse_paper_config, RQ2Config, RQ3Config, Probability, PositiveFloat, NonnegativeFloat


class PaperAgentState(Record):
    emotion: float = Field(ge=-1.0,le=1.0)
    confidence: Probability
    risk: Probability
    infection: Literal['S','I','R']
    duration: NonNegative
    references: list[PositiveFloat] = Field(min_length=4,max_length=4)
    cost: NonnegativeFloat

    @model_validator(mode='after')
    def duration_state(self):
        if (self.infection=='I')!=(self.duration>0):
            raise ValueError('Infection duration does not match state')
        return self


class PaperWorldState(Record):
    version: Literal['kdd27-state-v1'] = 'kdd27-state-v1'
    config_hash: Id
    profiles: dict[Id,dict]
    states: dict[Id,PaperAgentState]
    edges: dict[Id,dict[Id,PositiveFloat]]
    exposure: dict[Id,dict[Id,Probability]]
    strategic: list[Id]
    initial_infected: list[Id]
    price_history: list[PositiveFloat] = Field(min_length=1)
    last_quote: dict
    previous_return: float
    day_anchor_ticks: int = Field(gt=0)
    halted: bool
    band: float = Field(gt=0.0,le=1.0)
    same_day_lock: bool
    policy_signal: float = Field(ge=-1.0,le=1.0)
    current_events: list[dict]
    direct_injected: list[Id]
    last_trace: dict
    last_market_trace: dict
    cost_journal_count: NonNegative


class KDD27Scenario:
    scenario_id='kdd27_v1'
    version='kdd27-method-v1'
    owned_fields=frozenset({'kdd27'})

    def __init__(self, config):
        self.config=config=parse_paper_config(config)
        self.version=config.mechanism_version
        self.config_hash=content_hash(config.model_dump(exclude={'concurrency','cognition_mode'}))
        self.rng=AddressedRNG(config.root_seed,config.replica)
        values=build_population(config,self.rng)
        accounts,self.profiles,self.initial_states,self.edges,self.exposure,self.strategic,self.infected=values
        self.actors=list(self.profiles)
        locks={1:config.market.same_day_lock}
        locks.update({e.tick:e.value for e in config.interventions if e.kind=='lock'})
        asset=AssetSpec(asset_id=config.market.asset_id,quote_currency=config.market.currency,
            price_tick_minor=config.market.tick_minor,settlement_rule='T+0')
        self.market=MultiAssetMarket(PortfolioLedger([asset],accounts,fees=config.market.fees(),
            cash_scale=config.market.cash_scale,purchase_lock_schedule=sorted(locks.items())),
            initial_prices={asset.asset_id:config.market.initial_ticks},cancel_incoming_remainder=True)
        self.participants=(self.market,)
        self.events=[]
        for e in config.interventions:
            event_id=f'kdd:{1 if e.kind=="reset_sirs" else 3 if e.kind=="infect" else 2}:{e.event_id}'
            self.events.append(Event(event_id=event_id,root_event_id=event_id,tick=e.tick,phase='EVENTS',kind=e.kind,
                actors=e.actors,visible_at={'tick':e.tick,'phase':'EVENTS'},payload=e.model_dump()))
        count=int(config.n_agents*config.background_infection_fraction_per_day)
        if count:
            for tick in range(1,config.ticks+1,config.market.ticks_per_day):
                if isinstance(config,RQ2Config) and tick>=config.event_tick-config.pre_quiet_ticks:
                    break
                if getattr(config,'quiet_after_burn_in',False) and tick>config.burn_in_ticks:
                    break
                actors=sorted(self.actors,key=lambda a:(self.rng.u64('kdd27.infection',a,tick,'background'),a))[:count]
                event_id=f'kdd:0:background:{tick}'
                self.events.append(Event(event_id=event_id,root_event_id=event_id,tick=tick,phase='EVENTS',kind='infect',
                    actors=actors,visible_at={'tick':tick,'phase':'EVENTS'},payload={'actors':actors,'source':'background'}))
        if isinstance(config,RQ2Config):
            actors=sorted(self.actors,key=lambda a:(self.rng.u64('kdd27.infection',a,config.event_tick,'rq2-direct'),a))[:int(config.n_agents*config.direct_infection_fraction)]
            for order,kind,chosen in [(1,'reset_sirs',[]),(3,'infect',actors)]:
                event_id=f'kdd:{order}:rq2-{kind}'
                self.events.append(Event(event_id=event_id,root_event_id=event_id,tick=config.event_tick,phase='EVENTS',kind=kind,
                    actors=chosen,visible_at={'tick':config.event_tick,'phase':'EVENTS'},payload={'actors':chosen,'source':'rq2-shared'}))
        if isinstance(config,RQ3Config):
            actors=sorted(self.actors,key=lambda a:(self.rng.u64('kdd27.infection',a,config.event_tick,'rq3-direct'),a))[:config.direct_infected_count]
            event_id='kdd:3:rq3-infect'
            self.events.append(Event(event_id=event_id,root_event_id=event_id,tick=config.event_tick,phase='EVENTS',kind='infect',
                actors=actors,visible_at={'tick':config.event_tick,'phase':'EVENTS'},payload={'actors':actors,'source':'rq3-shared'}))

    def quote(self, halted=False):
        venue=self.market.venues[self.config.market.asset_id]
        quote=venue.quote()
        bids=sum(o.remaining_units for o in venue.orders.values() if o.status=='active' and o.command.side=='buy')
        asks=sum(o.remaining_units for o in venue.orders.values() if o.status=='active' and o.command.side=='sell')
        both=not halted and quote['best_bid_ticks'] is not None and quote['best_ask_ticks'] is not None
        return {**quote,'spread_ticks':quote['best_ask_ticks']-quote['best_bid_ticks'] if both else None,
            'depth_imbalance':(bids-asks)/(bids+asks) if both else None,'bid_depth':bids,'ask_depth':asks,
            'two_sided':both,'halted':halted}

    def initialize(self):
        cfg=self.config
        state=PaperWorldState(config_hash=self.config_hash,profiles=self.profiles,states=self.initial_states,
            edges=self.edges,exposure=self.exposure,strategic=self.strategic,initial_infected=self.infected,
            price_history=[cfg.market.initial_price],last_quote=self.quote(),previous_return=0.0,
            day_anchor_ticks=cfg.market.initial_ticks,halted=False,band=cfg.market.band,
            same_day_lock=cfg.market.same_day_lock,policy_signal=0.0,current_events=[],direct_injected=[],
            last_trace={},last_market_trace={},cost_journal_count=0)
        return {'kdd27':state.model_dump()}

    def boundary(self, context):
        data=context.model_dump()['data']; state=data['kdd27']; cfg=self.config.market
        self.market.begin_tick(context.tick,context.world_version)
        if context.tick>1 and (context.tick-1)%cfg.ticks_per_day==0:
            state['day_anchor_ticks']=state['last_quote']['last_price_ticks']
            self.market.ledger.new_day(self.market.ledger.day+1)
        self.market.expire(cfg.order_ttl)
        state['current_events']=[]
        state['direct_injected']=[]
        return data

    def apply_event(self, context, event):
        data=context.model_dump()['data']; state=data['kdd27']
        payload=event.payload
        if event.kind in ('halt','tax','fees'):
            self.market.cancel_all(reason=event.kind)
        if event.kind=='band' and payload.get('source')=='rq2':
            # Cancel every old order and reservation before admitting orders under the new band.
            self.market.cancel_all(reason='rq2_band_change')
        if event.kind=='reset_sirs':
            for actor in self.actors:
                state['states'][actor].update(infection='S',duration=0)
            state['direct_injected']=[]
        elif event.kind=='infect':
            for actor in payload['actors']:
                state['states'][actor].update(infection='I',duration=1)
            state['direct_injected']=sorted(set(state['direct_injected'])|set(payload['actors']))
        elif event.kind in ('halt','resume'):
            state['halted']=event.kind=='halt'
        elif event.kind in ('tax','fees'):
            fees=self.config.market.fees(payload['value']) if event.kind=='tax' else FeeSchedule.model_validate(payload['fees'])
            if Fraction(fees.commission.numerator,fees.commission.denominator)+Fraction(fees.tax.numerator,fees.tax.denominator)>=1:
                raise ValueError('Paper seller fees must leave positive proceeds')
            self.market.change_fees(fees)
        elif event.kind in ('lock','band','policy'):
            state[{'lock':'same_day_lock','band':'band','policy':'policy_signal'}[event.kind]]=payload['value']
        elif event.kind not in ('text','contagion_off','feed_off','rq2'):
            raise ValueError('Unknown paper intervention')
        state['current_events'].append(event.model_dump())
        return data

    async def prepare_observations(self, context):
        data=context.model_dump()['data']; state=data['kdd27']; cfg=self.config
        disabled=getattr(self,'contagion_disabled_from',None)
        if disabled is not None and context.tick>=disabled:
            cfg=cfg.model_copy(update={'social':cfg.social.model_copy(update={'infection_beta':0.0})})
        rq2=getattr(self,'rq2_branch',None);changes={}
        if rq2 is not None and context.tick>=rq2[0]:
            cfg,changes=rq2_parameters(cfg,rq2[1])
        price=state['last_quote']['last_price_ticks']*cfg.market.price_tick
        accounts={a:(self.market.ledger.accounts[a].balances[cfg.market.currency]/cfg.market.cash_scale,
            self.market.ledger.accounts[a].balances[cfg.market.asset_id]) for a in self.actors}
        states,edges,exposure,social=social_update(cfg,state['states'],self.profiles,state['edges'],state['exposure'],
            accounts,price,state['previous_return'],context.tick,self.rng,state['direct_injected'])
        traces={}
        for actor in self.actors:
            profile=self.profiles[actor]
            if 'population.loss_aversion' in changes:profile={**profile,'loss_aversion':cfg.population.loss_aversion}
            states[actor],behavior=behavior_update(cfg.behavior,states[actor],profile,price,accounts[actor][1],
                max(state['price_history']),social[actor]['neighbor_emotion'],state['policy_signal'],
                baseline=getattr(cfg,'baseline','full'))
            activation_draw=self.rng.uniform('kdd27.activation',actor,context.tick,'active')
            traces[actor]={'tick':context.tick,'market_as_of_tick':context.tick-1,'social':social[actor],'behavior':behavior,
                'activation_draw':activation_draw,'active':activation_draw<cfg.activation_probability,
                'arrival_key':self.rng.u64('scheduler.arrival',actor,context.tick,'decision'),
                'rng_address':{'root_seed':cfg.root_seed,'replica':cfg.replica,'actor':actor,'tick':context.tick,
                    'version':cfg.rng_version}}
            if changes:traces[actor]['effective_parameter_changes']=changes
        state.update(states=states,edges=edges,exposure=exposure,last_trace=traces)
        return data

    def project_observation(self, context, actor_id):
        state=context.data['kdd27']; now={'tick':context.tick,'phase':'OBSERVE'}
        cfg=self.config.market
        fees=self.market.ledger.fees
        return [ObservationFact(fact_id=f'paper-account:{context.tick}:{actor_id}',category='account',owner_id=actor_id,
            visible_at=now,value=self.market.ledger.account_view(actor_id,version=context.world_version).model_dump()),
            ObservationFact(fact_id=f'paper-market:{context.tick}',category='market',public=True,visible_at=now,
            value={'as_of_tick':context.tick-1,'historical_quote':state['last_quote'],'previous_return':state['previous_return'],
                'halted':state['halted'],'day_anchor_ticks':state['day_anchor_ticks'],'band':state['band'],
                'commission':fees.commission.numerator/fees.commission.denominator,
                'seller_tax':fees.tax.numerator/fees.tax.denominator,'same_day_lock':state['same_day_lock'],
                'policy_signal':state['policy_signal'],'events':[{'event_id':e['event_id'],'kind':e['kind'],
                    'value':e['payload'].get('value')} for e in state['current_events']],
                'asset_id':cfg.asset_id,'price_tick_minor':cfg.tick_minor,'cash_scale':cfg.cash_scale}),
            ObservationFact(fact_id=f'paper-own:{context.tick}:{actor_id}',category='memory',owner_id=actor_id,visible_at=now,
            value={'profile':self.profiles[actor_id],'state':state['states'][actor_id],
                'active':state['last_trace'].get(actor_id,{}).get('active',False),
                'behavior':state['last_trace'].get(actor_id,{}).get('behavior',{}),
                'feed':state['last_trace'].get(actor_id,{}).get('social',{}).get('feed',[]),
                'strategic':actor_id in state['strategic']})]

    def execute_batch(self, context, commands):
        state=context.data['kdd27']; receipts=[]
        if len({c.actor_id for c in commands})!=len(commands):
            raise ValueError('Paper actors may submit at most one decision per tick')
        band=Fraction(str(state['band']))
        for offset,command in enumerate(commands,1):
            if (command.tick,command.batch_id,command.world_version,command.observation_version)!=(context.tick,
                    f'tick:{context.tick}:base:{context.world_version}',context.world_version,context.world_version):
                raise ValueError('Stale paper observation batch')
            intent=command.intent; reason=None
            if isinstance(intent,PortfolioOrderIntent):
                if intent.order_id in self.market._order_ids:
                    raise ValueError('Paper order ID already used')
                if state['halted']:
                    reason='market_halted'
                elif abs(intent.price_ticks-state['day_anchor_ticks'])*band.denominator>state['day_anchor_ticks']*band.numerator:
                    reason='daily_price_limit'
            if reason:
                receipt=PortfolioExecutionReceipt(command_id=command.command_id,decision_id=command.decision_id,
                    actor_id=command.actor_id,execution_seq=context.execution_seq+offset,status='rejected',
                    reason_codes=[reason],executed_quantity=0,order_ids=[intent.order_id])
                self.market._event(intent.order_id,'rejected',intent.quantity_units,reason)
                self.market._record(command,receipt)
                receipts.append(receipt)
            else:
                receipts.extend(self.market.submit_orders([command],execution_seq_start=context.execution_seq+offset-1))
        return BatchExecutionResult(data=context.model_dump()['data'],receipts=receipts)

    def execute(self, context, command):
        result=self.execute_batch(context,[command])
        return ExecutionResult(data=result.data,receipt=result.receipts[0])

    def feedback(self, context, receipts):
        data=context.model_dump()['data']; state=data['kdd27']; cfg=self.config.market
        before={a['account_id']:a['balances'][cfg.asset_id] for a in data['multiasset_market']['ledger']['accounts']}
        transactions=self.market.ledger.journal[state['cost_journal_count']:]
        for tx in transactions:
            trade=tx.trade; buyer,seller=trade.buyer_id,trade.seller_id; quantity=trade.quantity_units
            cost=state['states'][buyer]['cost']
            state['states'][buyer]['cost']=(before[buyer]*cost+trade.price_ticks*cfg.price_tick*quantity)/(before[buyer]+quantity)
            before[buyer]+=quantity; before[seller]-=quantity
            if not before[seller]:
                state['states'][seller]['cost']=0.0
        quote=self.quote(state['halted'])
        old=state['last_quote']['last_price_ticks']; price=quote['last_price_ticks']*cfg.price_tick
        state.update(previous_return=math.log(quote['last_price_ticks']/old),last_quote=quote,
            price_history=[*state['price_history'],price][-self.config.behavior.rolling_high_window:],
            cost_journal_count=len(self.market.ledger.journal))
        state['last_market_trace']={'tick':context.tick,'trade_ids':[tx.trade.trade_id for tx in transactions],
            'no_trade':not transactions,'halted':state['halted'],'quotes':quote,
            'fees_minor':self.market.ledger.accounts['fees'].balances[cfg.currency],
            'resource_residuals':self.market.ledger.assert_invariants(),
            'rejected_commands':sum(r.status=='rejected' for r in receipts),
            'market_as_of_tick':context.tick,'observation_market_as_of_tick':context.tick-1}
        return Feedback(data=data)

    def validate(self, context):
        self.market.validate_state()
        state=PaperWorldState.model_validate(context.data['kdd27'])
        if (list(context.actors)!=sorted(self.actors) or state.config_hash!=self.config_hash
                or state.profiles!=self.profiles or state.strategic!=self.strategic or set(state.states)!=set(self.actors)
                or set(state.edges)!=set(self.actors) or set(state.exposure)!=set(self.actors)):
            raise ValueError('Paper identity/profile/state mismatch')
        if state.cost_journal_count!=len(self.market.ledger.journal) or state.last_quote!=self.quote(state.halted):
            raise ValueError('Paper cost/quote differs from committed market')
        if len(state.price_history)>self.config.behavior.rolling_high_window:
            raise ValueError('Rolling high exceeds its finite window')
        if context.tick and state.same_day_lock!=self.market.ledger.locks_purchase(self.config.market.asset_id,context.tick):
            raise ValueError('Paper lock observation differs from its frozen settlement schedule')
        costs={a:s['cost'] for a,s in self.initial_states.items()}
        quantities={a:self.market.ledger._initial[a].balances[self.config.market.asset_id] for a in self.actors}
        for tx in self.market.ledger.journal:
            t=tx.trade; b,s,n=t.buyer_id,t.seller_id,t.quantity_units
            costs[b]=(quantities[b]*costs[b]+t.price_ticks*self.config.market.price_tick*n)/(quantities[b]+n)
            quantities[b]+=n; quantities[s]-=n
            if not quantities[s]:
                costs[s]=0.0
        for actor in self.actors:
            if (set(state.edges[actor])!=set(self.edges[actor]) or set(state.exposure[actor])!=set(self.edges[actor])
                    or any(not self.config.network.weight_min<=w<=self.config.network.weight_max for w in state.edges[actor].values())):
                raise ValueError('Fixed graph/exposure contract mismatch')
            if not self.market.ledger.accounts[actor].balances[self.config.market.asset_id] and state.states[actor].cost!=0.0:
                raise ValueError('Empty paper position has nonzero cost')
            if state.states[actor].cost!=costs[actor]:
                raise ValueError('Paper cost differs from initial cost and actual fills')
        if context.tick and (set(state.last_trace)!=set(self.actors) or state.last_market_trace.get('tick')!=context.tick):
            raise ValueError('Missing paper trace for committed tick')

    def metrics(self, context):
        return {'data_origin':'simulation','purpose':'method_integration','last_step':context.data['kdd27']['last_market_trace']}
