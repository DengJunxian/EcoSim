"""A deterministic accounting stress scene, not a calibrated behavioral model."""
from civitas.contracts.events import Event
from civitas.contracts.observations import ObservationFact
from civitas.contracts.portfolio import PortfolioDecisionBundle, PortfolioOrderIntent
from civitas.contracts.state import BatchExecutionResult, ExecutionResult, Feedback
from civitas.contracts.time import SimTime
from civitas.core.rng import AddressedRNG
from civitas.markets.multiasset import MultiAssetMarket
from civitas.population.factory import PopulationFactory
from civitas.resources.ledger import PortfolioLedger


class MultiAssetAccounting:
    scenario_id='multiasset_accounting'
    version='accounting-orders-v1'
    owned_fields=frozenset({'population_initialization','quotes','accounting_metrics'})

    def __init__(self, config):
        self.config=config
        self.population=PopulationFactory.build(config.population,AddressedRNG(config.root_seed,config.replica))
        self.actors=[p.actor_id for p in self.population.profiles]
        self.market=MultiAssetMarket(PortfolioLedger(config.population.assets,self.population.accounts,
            fees=config.fees,cash_scale=config.population.cash_scale),initial_prices=dict(config.population.initial_price_ticks))
        self.participants=(self.market,)
        self.events=[Event(event_id=e.event_id,root_event_id=e.event_id,tick=e.tick,phase='EVENTS',kind='fee_change',
            actors=[],occurred_at=SimTime(tick=e.tick,phase='EVENTS'),visible_at=SimTime(tick=e.tick,phase='EVENTS'),
            payload={'fees':e.fees.model_dump()}) for e in config.fee_changes]

    def initialize(self):
        return {'population_initialization':self.population.model_dump(),'quotes':self._quotes(),'accounting_metrics':[]}

    def _quotes(self):
        return {a:v.quote() for a,v in self.market.venues.items()}

    def boundary(self, context):
        self.market.begin_tick(context.tick,context.world_version)
        if context.tick>1 and (context.tick-1)%self.config.ticks_per_day==0:
            self.market.ledger.new_day(self.market.ledger.day+1)
        self.market.expire(self.config.order_ttl)
        return dict(context.data)

    def apply_event(self, context, event):
        if event.kind!='fee_change':
            raise ValueError('Unknown accounting event')
        self.market.change_fees(event.payload['fees'])
        return dict(context.data)

    async def prepare_observations(self, context):
        return {**context.data,'quotes':self._quotes()}

    def project_observation(self, context, actor_id):
        now=SimTime(tick=context.tick,phase='OBSERVE')
        return [ObservationFact(fact_id=f'quotes:{context.tick}',category='market',public=True,visible_at=now,value=context.data['quotes']),
            ObservationFact(fact_id=f'account:{actor_id}:{context.tick}',category='account',owner_id=actor_id,visible_at=now,
                value=self.market.ledger.account_view(actor_id,version=context.world_version).model_dump())]

    def execute_batch(self, context, commands):
        receipts=self.market.submit_orders(commands,execution_seq_start=context.execution_seq,atomic=self.config.atomic_bundles)
        return BatchExecutionResult(data=dict(context.data),receipts=receipts)

    def execute(self, context, command):
        result=self.execute_batch(context,[command])
        return ExecutionResult(data=result.data,receipt=result.receipts[0])

    def feedback(self, context, receipts):
        ledger=self.market.ledger
        metric={'tick':context.tick,'trade_count':len(ledger.journal),'fees_minor':ledger.account_view('fees').balances[ledger.currency],
                'resource_residuals':ledger.assert_invariants(),'rejected_commands':sum(r.status=='rejected' for r in receipts)}
        return Feedback(data={**context.data,'quotes':self._quotes(),'accounting_metrics':[*context.data['accounting_metrics'],metric]})

    def validate(self, context):
        self.market.validate_state()
        if list(context.actors)!=sorted(self.actors):
            raise ValueError('Population actor mismatch')

    def metrics(self, context):
        return {'data_origin':'simulation','purpose':'accounting_stress','steps':context.data['accounting_metrics']}


class AccountingPolicy:
    policy_id='accounting_orders'
    version='v1'

    def __init__(self, config):
        self.rng=AddressedRNG(config.root_seed,config.replica)
        self.max_quantity=config.max_order_quantity

    def order_intent(self, actor, tick, market):
        assets=sorted(market)
        draw=lambda slot:self.rng.u64('accounting.orders.v1',actor,tick,slot)
        asset=assets[draw('asset')%len(assets)]
        price=max(1,market[asset]['last_price_ticks']+draw('quote')%5-2)
        return PortfolioOrderIntent(intent_id=f'intent:{tick}:{actor}',order_id=f'order:{tick}:{actor}',asset_id=asset,
            side='buy' if draw('side')%2 else 'sell',price_ticks=price,quantity_units=1+draw('quantity')%self.max_quantity)

    async def decide(self, observation):
        tick=observation.visible_until.tick
        actor=observation.actor_id
        intent=self.order_intent(actor,tick,observation.market_view[0].value)
        return PortfolioDecisionBundle(decision_id=f'decision:{tick}:{actor}',actor_id=actor,
            observation_id=observation.observation_id,batch_id=observation.batch_id,
            observation_version=observation.observation_version,world_version=observation.world_version,
            policy_id=self.policy_id,policy_version=self.version,tick=tick,economic_intents=[intent])
