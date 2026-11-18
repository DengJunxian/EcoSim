"""Social-market orchestration; the selected settlement backend owns all money."""
from dataclasses import asdict, replace
from fractions import Fraction

from core.exchange.clearing import Account, ClearingHouse
from core.exchange.market_rules import MarketRuleSchema
from core.exchange.order_book import OrderBook
from civitas.contracts.events import Event
from civitas.contracts.financial import FinancialExecutionReceipt, LegacyOrderBatch, OrderIntent
from civitas.contracts.observations import ObservationFact
from civitas.contracts.portfolio import PortfolioExecutionReceipt, PortfolioOrderIntent
from civitas.contracts.state import BatchExecutionResult, ExecutionResult, Feedback
from civitas.core.rng import AddressedRNG
from civitas.markets.legacy import LegacySettlementAdapter
from civitas.population.factory import PopulationFactory
from .legacy_market import LegacyMarket
from .multiasset_accounting import MultiAssetAccounting
from .society_relations import initial_relations, information_facts, resolutions, resolved_relations


class MarketSociety:
    scenario_id, version = 'market_society', '1'
    owned_fields = frozenset({'society'})

    def __init__(self, config):
        self.config = config
        self._projection = None
        self.legacy = config.backend == 'legacy_float64'
        if not self.legacy:
            base = MultiAssetAccounting(config.market)
            self.population, self.market, self.actors = base.population, base.market, base.actors
            self.participants = base.participants
            self.initialization = self.population.model_dump()
        elif config.initializer == 'legacy_initializer':
            base = LegacyMarket(config.market)
            self.adapter, self.actors = base.adapter, base.runner.ids
            self.participants = (self.adapter,)
            self.initialization = base.initial
        else:
            spec = config.population
            self.population = PopulationFactory.build(spec, AddressedRNG(config.root_seed, config.replica))
            asset = spec.assets[0]
            accounts = {a.account_id: Account(a.balances[asset.quote_currency]/spec.cash_scale,
                a.balances[asset.asset_id], cost_basis=float(config.market.experiment.market.initial_price))
                for a in self.population.accounts}
            cfg = config.market.experiment.market
            rules = MarketRuleSchema(symbol=cfg.symbol,prev_close=cfg.initial_price,price_limit_pct=cfg.price_limit,
                min_price_tick=cfg.tick_size,board_lot=cfg.board_lot,enforce_board_lot=True,
                allow_odd_lots=cfg.board_lot==1,commission_rate=cfg.commission,stamp_duty_rate=cfg.stamp_duty,
                t_plus_one=cfg.t_plus_one,feature_flags={'market_rules_v1':True,'strict_queue_timestamps':True},
                metadata={'session_model':'discrete_continuous_auction','holiday_calendar':'none'})
            self.adapter = LegacySettlementAdapter(ClearingHouse(OrderBook(cfg.symbol,cfg.initial_price,market_rules=rules),
                accounts,t_plus_one=cfg.t_plus_one))
            self.actors, self.participants = list(accounts), (self.adapter,)
            self.initialization = self.population.model_dump()
        self.events = [Event(event_id=e.event_id,root_event_id=e.event_id,tick=e.tick,phase='EVENTS',kind=e.kind,
            actors=[],visible_at={'tick':e.tick,'phase':'EVENTS'},payload={'public':True,**e.model_dump()})
            for e in config.interventions]

    def quotes(self):
        if self.legacy:
            return {self.adapter.clearing.book.symbol:self.adapter.quote()}
        return {a:v.quote() for a,v in self.market.venues.items()}

    def initialize(self):
        return {'typed_relations':initial_relations(self.config,self.actors).model_dump(),
            'society':{'initializer':self.config.initializer,'initialization':self.initialization,
            'quotes':self.quotes(),'halted':False,'steps':[],'institution_rejections':[],
            'order_ids':[],'interventions':[],'resolutions':[],'forecast_anchors':{}}}

    def boundary(self, context):
        cfg = self.config.market
        if self.legacy:
            self.adapter.begin_tick(context.tick,context.world_version)
            if context.tick>1 and (context.tick-1)%cfg.experiment.market.steps_per_day==0:
                self.adapter.apply_session_boundary(self.adapter.clearing.book.last_price)
            self.adapter.expire(context.tick,cfg.experiment.market.order_ttl)
        else:
            self.market.begin_tick(context.tick,context.world_version)
            if context.tick>1 and (context.tick-1)%cfg.ticks_per_day==0:
                self.market.ledger.new_day(self.market.ledger.day+1)
            self.market.expire(cfg.order_ttl)
        return dict(context.data)

    def apply_event(self, context, event):
        data = context.model_dump()['data']
        state = data['society']
        if event.kind in ('halt','fees'):
            if self.legacy:
                for order in self.adapter.clearing.active_orders():
                    self.adapter.cancel(order.order_id,reason=event.kind)
            else:
                self.market.cancel_all(reason=event.kind)
        if event.kind in ('halt','resume'):
            state['halted'] = event.kind=='halt'
        elif event.kind=='fees':
            from civitas.contracts.portfolio import FeeSchedule
            fees = FeeSchedule.model_validate(event.payload['fees'])
            if self.legacy:
                book = self.adapter.clearing.book
                book.market_rules = replace(book.market_rules,
                    commission_rate=fees.commission.numerator/fees.commission.denominator,
                    stamp_duty_rate=fees.tax.numerator/fees.tax.denominator)
            else:
                self.market.change_fees(fees)
        else:
            raise ValueError('Unknown society intervention')
        state['interventions'].append(event.model_dump())
        return data

    async def prepare_observations(self, context):
        data = context.model_dump()['data']
        data['society']['quotes'] = self.quotes()
        resolutions(self.config,self.actors,data['society'],context.tick)
        return data

    def project_observation(self, context, actor_id):
        now = {'tick':context.tick,'phase':'OBSERVE'}
        account = (self.adapter.account_view(actor_id,version=context.world_version) if self.legacy else
                   self.market.ledger.account_view(actor_id,version=context.world_version))
        facts = [ObservationFact(fact_id=f'account:{actor_id}:{context.tick}',category='account',owner_id=actor_id,
                    visible_at=now,value=account.model_dump()),
                 ObservationFact(fact_id=f'quotes:{context.tick}',category='market',public=True,visible_at=now,
                    value=context.data['society']['quotes']),
                 ObservationFact(fact_id='society:institution',category='market',public=True,visible_at=now,
                    value={'halted':context.data['society']['halted'],'backend':self.config.backend})]
        if self.legacy:
            facts.append(ObservationFact(fact_id='society:legacy-rules',category='market',public=True,visible_at=now,
                value=self.adapter.clearing.book.market_rules.to_dict()))
        if self._projection is None or self._projection[0] is not context:
            self._projection = (context,resolved_relations(context.data,context.tick))
        return [*facts,*information_facts(self.config,self.actors,context,actor_id,self._projection[1])]

    def execute_batch(self, context, commands):
        data = context.model_dump()['data']
        state = data['society']
        receipts, orders, accepted = [], set(state['order_ids']), []
        institution_reasons = []
        for command in commands:
            if (command.tick,command.batch_id,command.world_version,command.observation_version)!=(context.tick,
                    f'tick:{context.tick}:base:{context.world_version}',context.world_version,context.world_version):
                raise ValueError('Stale society observation batch')
            intent, reason = command.intent, None
            if isinstance(intent,(OrderIntent,PortfolioOrderIntent)):
                if intent.order_id in orders:
                    raise ValueError('Duplicate society order ID')
                orders.add(intent.order_id)
                if state['halted']:
                    reason = 'market_halted'
                elif not self.legacy and intent.asset_id in self.market.venues:
                    initial = self.config.market.population.initial_price_ticks[intent.asset_id]
                    band = Fraction(str(self.config.portfolio_price_limit))
                    if abs(intent.price_ticks-initial)*band.denominator>initial*band.numerator:
                        reason = 'price_limit'
            institution_reasons.append(reason)
        abort = not self.legacy and self.config.market.atomic_bundles and any(institution_reasons)
        def flush():
            if not accepted:
                return
            sequence = context.execution_seq+len(receipts)
            if self.legacy:
                receipts.extend(self.adapter.submit_orders(LegacyOrderBatch(commands=accepted,execution_seq_start=sequence)))
            else:
                receipts.extend(self.market.submit_orders(accepted,execution_seq_start=sequence,atomic=self.config.market.atomic_bundles))
            accepted.clear()
        for offset, (command,reason) in enumerate(zip(commands,institution_reasons,strict=True),1):
            intent = command.intent
            reason = reason or ('atomic_bundle_rejected' if abort else None)
            if reason:
                flush()
                receipt_type = FinancialExecutionReceipt if self.legacy else PortfolioExecutionReceipt
                receipt = receipt_type(command_id=command.command_id,decision_id=command.decision_id,
                    actor_id=command.actor_id,execution_seq=context.execution_seq+offset,status='rejected',
                    reason_codes=[reason],executed_quantity=0,order_ids=[intent.order_id] if hasattr(intent,'order_id') else [])
                state['institution_rejections'].append(receipt.model_dump())
                receipts.append(receipt)
            else:
                accepted.append(command)
        flush()
        state['order_ids'] = sorted(orders)
        return BatchExecutionResult(data=data,receipts=receipts)

    def execute(self, context, command):
        result = self.execute_batch(context,[command])
        return ExecutionResult(data=result.data,receipt=result.receipts[0])

    def trades(self):
        return ([asdict(t) for t in self.adapter.clearing.book.trades_history] if self.legacy else
                [t.model_dump() for t in self.market.trades])

    def feedback(self, context, receipts):
        data = context.model_dump()['data']
        state = data['society']
        trades = [t for t in self.trades() if (int(t['timestamp']) if self.legacy else t['tick'])==context.tick]
        state['quotes'] = self.quotes()
        state['steps'].append({'tick':context.tick,'trades':trades,'quotes':state['quotes'],
            'halted':state['halted'],'rejected_commands':sum(r.status=='rejected' for r in receipts),
            'fees':self.adapter.clearing.fees if self.legacy else self.market.ledger.account_view('fees').balances[self.market.ledger.currency]})
        memories = []
        for actor in context.actors:
            own = [t for t in trades if actor in (t['buyer_agent_id'],t['seller_agent_id'])] if self.legacy else [t for t in trades if actor in (t['buyer_id'],t['seller_id'])]
            if own:
                memories.append(ObservationFact(fact_id=f'fills:{context.tick}:{actor}',category='memory',owner_id=actor,
                    visible_at={'tick':context.tick+1,'phase':'OBSERVE'},value={'fills':own}))
        return Feedback(data=data,memories=memories)

    def validate(self, context):
        (self.adapter if self.legacy else self.market).validate_state()
        if list(context.actors) != sorted(self.actors):
            raise ValueError('Society actor mismatch')

    def metrics(self, context):
        return {'data_origin':'simulation','backend':self.config.backend,'steps':context.data['society']['steps']}
