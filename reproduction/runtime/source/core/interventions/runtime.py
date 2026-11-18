"""Announced, bounded policy state and causal external scenario inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from typing import Literal
from pydantic import Field, model_validator

from agents.decision_context import (ActorObservation, ExecutionRulesView, OrderIntent,
    PrivateValuation, PublicPolicyObservation, select_public_reference, stable_order_id,
    AccountObservation, QuoteObservation, CompletedPublicObservation, CompletedTradeObservation)
from core.exchange.market_rules import floor_to_tick, ceil_to_tick, execution_fees, worst_buy_cash
from core.experiments.identity import research_hash, scientific_projection
from core.experiments.measurement import StepMeasurement
from core.interventions.compiler import compile_policy
from core.types import OrderSide, OrderStatus
from core.interventions.schema import InitialInformation
from core.society.network import SentimentState
from core.experiments.strict_types import StrictResearchModel


class PolicyObservation(StrictResearchModel):
    actor_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    account: AccountObservation
    initial_price: float = Field(gt=0.0)
    current_quote: QuoteObservation
    completed_public_history: list[CompletedPublicObservation]
    last_completed_trade: CompletedTradeObservation|None
    private_valuation: Literal[None] = None
    public_policy: PublicPolicyObservation

    @model_validator(mode='after')
    def _public_clock(self):
        if (self.current_quote.available_step>self.step or self.public_policy.available_step>self.step
                or any(row.step>=self.step or row.available_step>self.step for row in self.completed_public_history)
                or self.last_completed_trade is not None and self.last_completed_trade.available_step>self.step):
            raise ValueError('Policy observation cannot consume unfinished/future prices')
        return self


@dataclass
class TriggerState:
    previous_condition: bool|None = None
    armed: bool = True
    last_trigger_step: int|None = None
    episodes: int = 0
    active_start: int|None = None
    active_end: int|None = None
    cooldown_until: int = 0


class PolicyRuntime:
    implementation_version='funded-policy-v1'

    def __init__(self,spec,clearing):
        self.spec,self.clearing=spec,clearing
        self.compiled=compile_policy(spec.policy,constraints=spec.policy_constraints,world=spec.world,market=spec.base.market)
        self.policy_hash=self.compiled.policy_hash
        self.account_id=spec.world.policy_account_initial.actor_id
        self.states={name:TriggerState() for name in ('halt','liquidity')}
        self.pending=[];self.announcements=[];self.public_history=[]
        self.market_open=True;self.liquidity_active=False;self.last_due_step=0;self.last_evaluation_step=0
        self.budget=self.compiled.deterministic_bounds['cash_authorization']
        self.spent_terms=[];self.buy_authorizations={};self.sell_authorizations={};self.seen_trades={}
        self.sale_proceeds_terms=[];self.fee_terms=[]

    @property
    def spent_buy_cash(self):return math.fsum(self.spent_terms)

    def _worst(self,price,quantity):
        # Cover the actual book fee operation and the legacy reservation's
        # binary64 operation order; they can differ by one final rounding unit.
        fee=execution_fees(price,quantity,self.spec.base.market.commission,self.spec.base.market.stamp_duty)[0]
        return max(worst_buy_cash(price,quantity,self.spec.base.market.commission),price*quantity+fee)

    def _spent_exact(self):return sum((Fraction.from_float(x) for x in self.spent_terms),Fraction(0))
    def _reserved_exact(self):
        return sum((Fraction.from_float(self._worst(row['price'],row['remaining_qty']))
                    for row in self.buy_authorizations.values()),Fraction(0))

    def authorization_snapshot(self):
        exact=self._spent_exact()+self._reserved_exact()
        if exact>Fraction.from_float(self.budget):
            raise RuntimeError('POLICY_CASH_AUTHORIZATION_EXCEEDED')
        return {'budget':self.budget,'spent_buy_cash':self.spent_buy_cash,
            'reserved_worst_case_buy_cash':float(self._reserved_exact()),
            'available_budget':float(Fraction.from_float(self.budget)-exact),
            'sale_proceeds':math.fsum(self.sale_proceeds_terms),'fees':math.fsum(self.fee_terms),
            'buy_authorizations':{oid:dict(row) for oid,row in sorted(self.buy_authorizations.items())},
            'sell_authorizations':{oid:dict(row) for oid,row in sorted(self.sell_authorizations.items())},
            'processed_trade_ids':sorted(self.seen_trades)}

    def reconcile_orders(self):
        """Release only unfilled canceled/rejected reservations; never refund spent."""
        for oid,row in list(self.buy_authorizations.items())+list(self.sell_authorizations.items()):
            order=self.clearing.orders.get(oid)
            if order is None:
                # An authorized not-yet-submitted order still reserves its full
                # amount. Absence alone is not evidence of a cancellation.
                continue
            row['remaining_qty']=order.remaining_qty if order.status in {OrderStatus.PENDING,OrderStatus.PARTIAL} else 0
        return self.authorization_snapshot()

    def consume_settled_trades(self,trades):
        """Called for every Clearing.submit return, including other actors' orders."""
        for trade in trades:
            digest=research_hash(asdict(trade))
            if trade.trade_id in self.seen_trades:
                if self.seen_trades[trade.trade_id]!=digest:raise RuntimeError('DUPLICATE_TRADE_CONTENT_MISMATCH')
                continue
            self.seen_trades[trade.trade_id]=digest
            if trade.buyer_agent_id==self.account_id:
                oid=trade.maker_id if trade.maker_agent_id==self.account_id else trade.taker_id
                if oid not in self.buy_authorizations:raise RuntimeError('UNAUTHORIZED_POLICY_BUY')
                auth=self.buy_authorizations[oid]
                if trade.quantity>auth['remaining_qty'] or trade.price>auth['price']:
                    raise RuntimeError('POLICY_FILL_EXCEEDS_AUTHORIZATION')
                auth['remaining_qty']-=trade.quantity
                self.spent_terms.append(trade.price*trade.quantity+trade.buyer_fee)
                self.fee_terms.append(trade.buyer_fee)
            if trade.seller_agent_id==self.account_id:
                oid=trade.maker_id if trade.maker_agent_id==self.account_id else trade.taker_id
                if oid not in self.sell_authorizations:raise RuntimeError('UNAUTHORIZED_POLICY_SELL')
                auth=self.sell_authorizations[oid]
                if trade.quantity>auth['remaining_qty'] or trade.price<auth['price']:
                    raise RuntimeError('POLICY_SELL_EXCEEDS_AUTHORIZATION')
                auth['remaining_qty']-=trade.quantity
                self.sale_proceeds_terms.append(trade.price*trade.quantity-trade.seller_fee-trade.seller_tax)
                self.fee_terms.append(trade.seller_fee+trade.seller_tax)
        self.reconcile_orders()

    def authorize_intent(self,intent):
        if intent.actor_id!=self.account_id:return
        if (intent.policy_hash!=self.policy_hash or not self.market_open or not self.liquidity_active
                or self.budget==0.0 or intent.order_type!='POST_ONLY' or intent.step!=self.last_due_step
                or intent.quantity_shares%self.spec.base.market.board_lot
                or intent.quantity_shares>self.spec.policy.liquidity.quote_qty_lots*self.spec.base.market.board_lot):
            raise RuntimeError('POLICY_INTENT_OUTSIDE_AUTHORIZED_WINDOW')
        if intent.order_id in self.buy_authorizations or intent.order_id in self.sell_authorizations:
            raise RuntimeError('DUPLICATE_POLICY_AUTHORIZATION')
        account=self.clearing.accounts[self.account_id];bounds=self.compiled.deterministic_bounds
        if intent.side=='buy':
            pending=sum(row['remaining_qty'] for row in self.buy_authorizations.values())
            if account.shares+pending+intent.quantity_shares>bounds['inventory_upper_shares']:
                raise RuntimeError('POLICY_BUY_EXCEEDS_INVENTORY_BOUND')
        else:
            pending=sum(row['remaining_qty'] for row in self.sell_authorizations.values())
            if account.shares-pending-intent.quantity_shares<bounds['inventory_lower_shares']:
                raise RuntimeError('POLICY_SELL_EXCEEDS_INVENTORY_BOUND')
        if intent.side=='buy':
            required=Fraction.from_float(self._worst(intent.limit_price,intent.quantity_shares))
            if self._spent_exact()+self._reserved_exact()+required>Fraction.from_float(self.budget):
                raise RuntimeError('POLICY_BUY_EXCEEDS_REMAINING_BUDGET')
            self.buy_authorizations[intent.order_id]={'price':intent.limit_price,'remaining_qty':intent.quantity_shares,
                                                    'authorized_qty':intent.quantity_shares}
        else:
            self.sell_authorizations[intent.order_id]={'price':intent.limit_price,'remaining_qty':intent.quantity_shares,
                                                     'authorized_qty':intent.quantity_shares}
        self.authorization_snapshot()

    def apply_due(self,step):
        if step!=self.last_due_step+1:raise RuntimeError('POLICY_DUE_CLOCK_MISMATCH')
        actions=[]
        for name,state in self.states.items():
            if state.active_end is not None and step>state.active_end:
                actions.append({'kind':'END','mechanism':name,'step':step})
                if name=='liquidity':
                    for order in list(self.clearing.active_orders(self.account_id)):
                        self.clearing.cancel(order.order_id,reason='policy_window_end')
                state.active_start=state.active_end=None
        for command in list(self.pending):
            if command['effective_step']<step:raise RuntimeError('MISSED_ANNOUNCED_POLICY_COMMAND')
            if command['effective_step']==step:
                state=self.states[command['mechanism']]
                if state.active_start is not None:raise RuntimeError('OVERLAPPING_POLICY_WINDOW')
                state.active_start=step;state.active_end=command['end_step']
                actions.append(dict(command,kind='START'))
                if command['mechanism']=='halt':
                    for order in list(self.clearing.active_orders()):
                        self.clearing.cancel(order.order_id,reason='halt')
                self.pending.remove(command)
        self.market_open=self.states['halt'].active_start is None
        self.liquidity_active=self.states['liquidity'].active_start is not None
        self.last_due_step=step
        return {'step':step,'phase':3,'kind':'APPLY_DUE','actions':actions,'market_open':self.market_open,
            'liquidity_active':self.liquidity_active,'policy_hash':self.policy_hash,
            'authorization':self.reconcile_orders()}

    def public_observation(self,step):
        return PublicPolicyObservation(market_open=self.market_open,liquidity_active=self.liquidity_active,
            as_of_step=step,available_step=step,announcement_refs=[row['announcement_hash'] for row in self.announcements])

    def evaluate_trigger(self,rule,step,history):
        # lookback_steps denotes the number of consecutive public prices,
        # including t: [t-L+1,t]. It never silently switches measurement proxy.
        window=list(range(step-rule.lookback_steps+1,step+1))
        by_step={row['step']:row for row in history}
        if any(t<1 or t not in by_step or by_step[t]['price'] is None
               or by_step[t]['age_steps'] is None or by_step[t]['age_steps']>rule.max_price_age_steps for t in window):
            return {'status':'TRIGGER_UNDEFINED','condition':None,'drawdown':None,'window':window}
        values=[by_step[t]['price'] for t in window]
        drawdown=1.0-values[-1]/max(values)
        return {'status':'VALID','condition':drawdown>=rule.threshold_bps/10000.0,
                'drawdown':drawdown,'window':window}

    def evaluate_completed_market(self,step,measurement:StepMeasurement):
        if step!=self.last_evaluation_step+1 or self.last_due_step!=step or measurement.step!=step:
            raise RuntimeError('POLICY_EVALUATION_CLOCK_MISMATCH')
        metric=measurement.metrics['price_proxy'];mode=self.spec.measurement.price_proxy
        age=(0 if mode=='current_mid' else measurement.price_age_steps if mode=='last_trade'
             else measurement.history_reference_age_steps)
        point={'step':step,'price':metric.value if metric.status=='VALID' else None,
            'age_steps':age,'proxy':mode,'market_record_hash':research_hash(scientific_projection(measurement.model_dump(mode='json')))}
        history=self.public_history+[point]
        triggers=[];announced=[]
        for name in ('halt','liquidity'):
            if (name=='halt' and 'halt' not in self.compiled.allowed_actions
                    or name=='liquidity' and 'funded_liquidity' not in self.compiled.allowed_actions):continue
            rule=getattr(self.spec.policy,name);state=self.states[name]
            observation=self.evaluate_trigger(rule.trigger,step,history)
            eligible=(state.active_start is None and not any(row['mechanism']==name for row in self.pending)
                      and step>=state.cooldown_until)
            limit=rule.max_episodes if name=='halt' else rule.max_activations
            if observation['condition'] is not None:
                state.previous_condition=observation['condition']
                if eligible and not observation['condition']:state.armed=True
                if eligible and state.armed and observation['condition'] and state.episodes<limit:
                    command={'mechanism':name,'announced_step':step,'effective_step':step+1,
                        'end_step':step+rule.duration_steps,'episode':state.episodes+1}
                    command['announcement_hash']=research_hash(dict(command,policy_hash=self.policy_hash))
                    self.pending.append(command);self.announcements.append(command);announced.append(command)
                    state.episodes+=1;state.armed=False;state.last_trigger_step=step
                    state.cooldown_until=command['end_step']+(rule.cooldown_steps if name=='halt' else 1)
            triggers.append({'mechanism':name,**observation,'eligible_window':eligible,'state':asdict(state)})
        self.public_history.append(point);self.last_evaluation_step=step
        return {'step':step,'phase':10,'kind':'POLICY_EVALUATION','policy_hash':self.policy_hash,
            'public_point':point,'triggers':triggers,'announcements':announced,
            'queued_commands':list(self.pending),'authorization':self.authorization_snapshot()}

    def quote(self,observation:PolicyObservation,rules:ExecutionRulesView):
        if not isinstance(observation,PolicyObservation) or observation.actor_id!=self.account_id:
            raise ValueError('POLICY_OBSERVATION_MUST_BE_PUBLIC_AND_ACCOUNT_LOCAL')
        record={'step':observation.step,'actor_id':self.account_id,'policy_hash':self.policy_hash,
            'observation_hash':research_hash(observation),'authorization':self.authorization_snapshot(),
            'reference':None,'reason':'INACTIVE','intents':[]}
        if not self.market_open or not self.liquidity_active or self.budget==0.0:
            record['reason']='HALTED' if not self.market_open else 'INACTIVE'
            return [],record
        lp=self.spec.policy.liquidity
        policy=self.spec.behavior.reference_policy
        limit=lp.max_reference_age_steps
        policy=policy.model_copy(update={name:min(getattr(policy,name),limit) for name in
            ('max_reference_age_steps','max_quote_age_steps','max_trade_age_steps')})
        reference=select_public_reference(observation,policy)
        record['reference']=reference.model_dump(mode='json')
        if reference.value is None:
            record['reason']='REFERENCE_UNAVAILABLE';return [],record
        raw_bid=reference.value-lp.half_spread_ticks*rules.tick_size
        bid=0.0 if raw_bid<=0.0 else floor_to_tick(raw_bid,rules.tick_size)
        ask=ceil_to_tick(reference.value+lp.half_spread_ticks*rules.tick_size,rules.tick_size)
        quote=observation.current_quote
        if observation.step-quote.as_of_step<=policy.max_quote_age_steps:
            if quote.best_ask is not None:
                bound=quote.best_ask-rules.tick_size
                bid=0.0 if bound<=0.0 else min(bid,floor_to_tick(bound,rules.tick_size))
            if quote.best_bid is not None:ask=max(ask,ceil_to_tick(quote.best_bid+rules.tick_size,rules.tick_size))
        account=observation.account;bounds=self.compiled.deterministic_bounds;lot=rules.board_lot
        available=Fraction.from_float(self.budget)-self._spent_exact()-self._reserved_exact()
        intents=[];reasons={}
        for side,price,slot in [('buy',bid,0),('sell',ask,1)]:
            if price<=0.0 or not rules.lower_price_limit<=price<=rules.upper_price_limit:
                reasons[side]='PRICE_OUTSIDE_RULES';continue
            caps=[lp.quote_qty_lots*lot]
            if side=='buy':
                caps += [max(0,bounds['inventory_upper_shares']-account.shares-account.pending_buy_qty),
                         max(0.0,account.available_cash)/self._worst(price,1),float(available)/self._worst(price,1)]
            else:
                caps += [max(0,account.shares-account.pending_sell_qty-bounds['inventory_lower_shares']),account.available_shares]
            quantity=lot*math.floor(max(0.0,min(caps))/lot)
            while side=='buy' and quantity>0 and Fraction.from_float(self._worst(price,quantity))>available:
                quantity-=lot
            if quantity==0:reasons[side]='BELOW_LOT_OR_RESOURCE_CAP';continue
            oid=stable_order_id(observation.step,self.account_id,slot)
            intents.append(OrderIntent(intent_id=oid,order_id=oid,actor_id=self.account_id,step=observation.step,
                slot=slot,side=side,order_type='POST_ONLY',limit_price=price,quantity_shares=quantity,
                reason_code='FUNDED_POLICY_LP_QUOTE',observation_hash=research_hash(observation),
                behavior_version='funded-policy-lp-v1',policy_hash=self.policy_hash))
        record.update(reason='QUOTE' if intents else 'NO_FEASIBLE_SIDE',rejected_sides=reasons,
                      intents=[intent.model_dump(mode='json') for intent in intents])
        return intents,record


class ScenarioRuntime:
    """External shocks alter information and registered participation, not prices.

    relative-event-private-v1 preserves each initial valuation / initial F ratio
    and releases F(t)*ratio after its registered actor delay. No new noise law is
    silently introduced; initial heterogeneity is explicit in the world.
    """
    implementation_version='causal-scenarios-v1'

    def __init__(self,spec,world,random_source):
        self.spec,self.world,self.random=spec,world,random_source
        self.private={};self.released=set();self.processed=set();self.last_step=0
        self.queue=list(spec.scenario.initial_information);self.withdrawals=[]
        self.fundamental=spec.scenario.fundamental_process.initial_value
        self.initial_private={row.actor_id:row.value for row in self.queue if row.kind=='private_valuation'}
        if (any(row.as_of_step!=0 for row in self.queue)
                or len(self.initial_private)!=sum(row.kind=='private_valuation' for row in self.queue)):
            raise ValueError('CAUSAL_SCENARIO_REQUIRES_UNIQUE_STEP_ZERO_INITIAL_INFORMATION')
        self.targets={};self.target_draws={}
        kinds={actor.actor_id:actor.actor_kind for actor in world.snapshot.actors}
        events=spec.scenario.events
        if any(event.kind=='constrained_selling_need' for event in events):
            raise ValueError('UNSUPPORTED_SCENARIO: constrained_selling_need')
        if any(event.kind=='fundamental_shock' for event in events) and (
                spec.scenario.fundamental_process.version!='piecewise-fundamental-v1'
                or spec.scenario.private_signal_process.version!='relative-event-private-v1'):
            raise ValueError('Fundamental events require the registered event information process')
        intervals={}
        for event in events:
            selector=event.target_selector
            targets=(list(selector.actor_ids) if selector.mode=='explicit' else
                [aid for aid in spec.world.stable_actor_ids if selector.mode!='normal_mm' or kinds[aid]=='normal_mm'])
            if set(targets)-set(kinds) or not targets:raise ValueError('Scenario targets must resolve to existing ordinary/MM actors')
            draws=[]
            if event.kind=='panic_seed':
                if event.parameters.count>len(targets):raise ValueError('Panic count exceeds the registered target population')
                if 'scenario_target' not in spec.random.channels:raise ValueError('Panic subset requires registered scenario_target channel')
                addressed=[]
                for aid in sorted(targets):
                    draw=random_source.draw_receipt(step=event.release_step,entity_id=event.event_id+'/'+aid,
                        channel='scenario_target',distribution='uniform')
                    draws.append(draw)
                    addressed.append((draw.value,aid))
                targets=[aid for _,aid in sorted(addressed)[:event.parameters.count]]
            elif event.kind=='fundamental_shock':
                if set(targets)-set(self.initial_private):raise ValueError('Every fundamental target needs its declared initial private signal')
            elif event.kind=='normal_mm_withdrawal':
                if any(kinds[aid]!='normal_mm' for aid in targets):raise ValueError('Withdrawal may only target normal MM actors')
                for aid in targets:
                    profile=world.population.research_profiles[aid]
                    profile.model_copy(update={'half_spread_fraction':profile.half_spread_fraction*event.parameters.half_spread_multiplier})
                    interval=(event.release_step,event.release_step+event.parameters.duration_steps-1)
                    if any(max(interval[0],a)<=min(interval[1],b) for a,b in intervals.get(aid,[])):
                        raise ValueError('Overlapping MM withdrawal modifiers require a separately registered composition')
                    intervals.setdefault(aid,[]).append(interval)
            self.targets[event.event_id]=sorted(targets)
            self.target_draws[event.event_id]=draws

    def apply(self,step):
        if step!=self.last_step+1:raise RuntimeError('SCENARIO_RELEASE_CLOCK_MISMATCH')
        events=[];draws=[]
        for event in sorted(self.spec.scenario.events,key=lambda e:(e.release_step,e.event_id)):
            if event.release_step!=step:continue
            targets=self.targets[event.event_id]
            record={'event_id':event.event_id,'kind':event.kind,'release_step':step,'targets':targets,
                'parameters':event.parameters.model_dump(mode='json'),'origin_refs':list(self.spec.scenario.origin_refs)}
            if event.kind=='fundamental_shock':
                before=self.fundamental;self.fundamental*=1.0+event.parameters.relative_change
                if not math.isfinite(self.fundamental) or self.fundamental<=0.0:raise ValueError('Invalid cumulative fundamental value')
                for aid in targets:
                    delay=self.world.population.research_profiles[aid].signal_delay_steps
                    value=self.fundamental*(self.initial_private[aid]/self.spec.scenario.fundamental_process.initial_value)
                    self.queue.append(InitialInformation(information_id=f'{event.event_id}/{aid}',kind='private_valuation',
                        actor_id=aid,value=value,as_of_step=step,available_step=step+delay,
                        origin_ref=self.spec.scenario.origin_refs[0]))
                record['fundamental_before']=before;record['fundamental_after']=self.fundamental
            elif event.kind=='panic_seed':
                changed=[]
                by_id={node.agent_id:node for node in self.world.social.agents.values()}
                for aid in targets:
                    node=by_id[aid];before=node.sentiment_state
                    if before!=SentimentState.INFECTED:
                        node.sentiment_history.append(before);node.sentiment_state=SentimentState.INFECTED;node.state_duration=0
                    changed.append({'actor_id':aid,'before':before.value,'after':node.sentiment_state.value})
                record['state_differences']=changed
            elif event.kind=='normal_mm_withdrawal':
                withdrawal={'start':step,'end':step+event.parameters.duration_steps-1,'targets':targets,
                    'participation_multiplier':event.parameters.participation_multiplier,
                    'half_spread_multiplier':event.parameters.half_spread_multiplier,'event_id':event.event_id}
                self.withdrawals.append(withdrawal);record['withdrawal']=withdrawal
            self.processed.add(event.event_id);events.append(record)
            draws.extend(row.model_dump(mode='json') for row in self.target_draws[event.event_id])
        released=[]
        for row in sorted(self.queue,key=lambda row:(row.available_step,row.as_of_step,row.information_id)):
            if row.available_step>step or row.information_id in self.released:continue
            self.released.add(row.information_id);released.append(row.model_dump(mode='json'))
            if row.kind=='private_valuation':
                current=self.private.get(row.actor_id)
                if current is None or row.as_of_step>=current.as_of_step:
                    self.private[row.actor_id]=PrivateValuation(owner_actor_id=row.actor_id,value=row.value,
                        as_of_step=row.as_of_step,available_step=row.available_step,information_id=row.information_id,
                        source='initial_private_signal' if row.as_of_step==0 else 'released_private_signal')
        self.last_step=step
        return {'step':step,'phase':2,'kind':'CAUSAL_INFORMATION_RELEASE','events':events,'released':released,
                'target_draws':draws}

    def release(self,step):return self.apply(step)

    def effective_profile(self,actor,step,profile):
        applicable=[row for row in self.withdrawals if row['start']<=step<=row['end'] and actor in row['targets']]
        if not applicable:return profile
        if len(applicable)!=1:raise RuntimeError('Ambiguous withdrawal composition')
        modifier=applicable[0]
        return profile.model_copy(update={'participation_rate':profile.participation_rate*modifier['participation_multiplier'],
            'half_spread_fraction':profile.half_spread_fraction*modifier['half_spread_multiplier']})
