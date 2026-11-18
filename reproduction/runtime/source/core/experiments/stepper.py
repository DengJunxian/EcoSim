"""The research-step-v1 clock, using the existing Python book and clearing.

An attempt is fail-stop. A completed market phase is not a completed step until
the final policy journal and commit record have both reached durable storage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from agents.decision_context import (AccountObservation, ActorObservation,
    CompletedPublicObservation, CompletedTradeObservation, DecisionBatch,
    DecisionContext, DecisionNoise, ExecutionRulesView, PrivateValuation,
    PublicPolicyObservation, QuoteObservation, SocialObservation)
from agents.research_kernel import ResearchDecisionKernel, active_parameter_fields
from core.experiments.design_config import ResearchRunSpec
from core.experiments.identity import research_hash, scientific_projection
from core.experiments.measurement import (CancellationObservation, DepthLevel,
    MarketSnapshot, MeasurementContext, PriceReference, RejectionObservation,
    StepActivity, StepMeasurement, compute_step_measurement)
from core.experiments.randomness import KeyedRandomSource
from core.experiments.world import WorldComponents
from core.types import Order, OrderSide, OrderStatus, OrderType


class UnsupportedResearchRequest(ValueError):
    pass


def validate_phase2_execution(spec: ResearchRunSpec) -> None:
    """This phase has a real NoOp implementation and no dynamic pressure service."""
    if any(event.kind == 'constrained_selling_need' for event in spec.scenario.events):
        raise UnsupportedResearchRequest('UNSUPPORTED_SCENARIO: constrained_selling_need')
    if spec.policy.halt is not None or spec.policy.liquidity is not None:
        raise UnsupportedResearchRequest('UNSUPPORTED_POLICY: Phase 2 executes only NoOp')
    if spec.scenario.events:
        raise UnsupportedResearchRequest('UNSUPPORTED_SCENARIO: dynamic pressure requires Phase 3')
    validate_common_execution(spec)


def validate_common_execution(spec: ResearchRunSpec) -> None:
    profiles = {profile.profile_id: profile for profile in spec.behavior.parameter_profiles}
    if {key: active_parameter_fields(value) for key, value in profiles.items()} != spec.behavior.active_parameter_map:
        raise ValueError('ACTIVE_PARAMETER_MAP_MISMATCH: registered consumers differ from the kernel')
    for row in spec.scenario.initial_information:
        if row.kind == 'private_valuation':
            profile = profiles[spec.behavior.actor_profile_map[row.actor_id]]
            if row.available_step != row.as_of_step + profile.signal_delay_steps:
                raise ValueError('SIGNAL_DELAY_MISMATCH: availability must equal source step plus registered delay')
        elif (row.value != spec.world.initial_price or row.as_of_step != 0 or row.available_step != 0):
            raise UnsupportedResearchRequest('UNSUPPORTED_INFORMATION: Phase 2 public reference is the declared P0')


def validate_research_execution(spec: ResearchRunSpec) -> None:
    validate_common_execution(spec)
    from core.interventions.compiler import compile_policy
    compile_policy(spec.policy,constraints=spec.policy_constraints,world=spec.world,market=spec.base.market)
    if any(event.kind=='constrained_selling_need' for event in spec.scenario.events):
        raise UnsupportedResearchRequest('UNSUPPORTED_SCENARIO: constrained_selling_need')
    if spec.services.policy_version=='noop-policy-v1' and (spec.policy.halt or spec.policy.liquidity):
        raise UnsupportedResearchRequest('Non-NoOp policy requires registered funded-policy-v1 service')
    if spec.services.scenario_version=='noop-information-v1' and spec.scenario.events:
        raise UnsupportedResearchRequest('Dynamic pressure requires registered causal-scenarios-v1 service')


def json_body(value: Any) -> Any:
    return value.model_dump(mode='json') if hasattr(value, 'model_dump') else value


class DiagnosticJournal:
    """Append-only, finite JSON with an explicit fsync at each evidence record.

    record_hash hashes the declared scientific projection of one record. The
    exact bytes, including run metadata, are separately covered by the archive.
    """
    implementation_version = 'diagnostic-jsonl-v1'

    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)

    def append(self, stream: str, value: Any) -> str:
        if not stream.replace('_', '').isalnum():
            raise ValueError('Journal stream must be a simple registered name')
        body = json_body(value)
        digest = research_hash(scientific_projection(body))
        record = dict(body, record_hash=digest)
        data = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n'
        with (self.output / f'{stream}.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return digest


class NoOpInformationService:
    implementation_version = 'noop-information-v1'

    def __init__(self, spec: ResearchRunSpec):
        self.spec = spec
        self.released: set[str] = set()
        self.private: dict[str, PrivateValuation] = {}

    def release(self, step: int) -> dict:
        rows = []
        for info in sorted(self.spec.scenario.initial_information,
                           key=lambda row: (row.available_step, row.as_of_step, row.information_id)):
            if info.available_step > step or info.information_id in self.released:
                continue
            self.released.add(info.information_id)
            rows.append(info.model_dump(mode='json'))
            if info.kind == 'private_valuation':
                current = self.private.get(info.actor_id)
                if current is None or info.as_of_step >= current.as_of_step:
                    self.private[info.actor_id] = PrivateValuation(owner_actor_id=info.actor_id,
                        value=info.value, as_of_step=info.as_of_step, available_step=info.available_step,
                        source='initial_private_signal' if info.as_of_step == 0 else 'released_private_signal',
                        information_id=info.information_id)
        return {'step': step, 'phase': 2, 'kind': 'INFORMATION_RELEASE', 'released': rows}


class NoOpPolicyService:
    implementation_version = 'noop-policy-v1'

    def __init__(self, spec: ResearchRunSpec):
        from core.interventions.compiler import compile_policy
        self.compiled = compile_policy(spec.policy,constraints=spec.policy_constraints,world=spec.world,market=spec.base.market)
        self.policy_hash = self.compiled.policy_hash
        self.market_open, self.liquidity_active = True, False

    def apply_due(self, step: int) -> dict:
        return {'step': step, 'phase': 3, 'kind': 'NO_ACTION', 'market_open': True,
                'liquidity_active': False, 'policy_hash': self.policy_hash}

    def public_observation(self, step: int) -> PublicPolicyObservation:
        return PublicPolicyObservation(market_open=True, liquidity_active=False,
            as_of_step=step, available_step=step, announcement_refs=[])

    def evaluate_completed_market(self, step: int, measurement: StepMeasurement) -> dict:
        return {'step': step, 'phase': 10, 'kind': 'NO_COMMAND', 'triggers': [],
                'announcements': [], 'queued_commands': [], 'policy_hash': self.policy_hash,
                'market_record_hash': research_hash(scientific_projection(measurement.model_dump(mode='json')))}


@dataclass
class RuntimeServices:
    world: WorldComponents
    random: KeyedRandomSource
    scenario: NoOpInformationService
    policy: NoOpPolicyService
    kernel: ResearchDecisionKernel
    journal: DiagnosticJournal
    measurement: Any = compute_step_measurement

    def validate(self, spec: ResearchRunSpec) -> None:
        from agents.population import StratifiedPopulation
        from core.exchange.order_book import OrderBook
        from core.exchange.clearing import ClearingHouse
        from core.society.network import SocialGraph, InformationDiffusion
        from core.interventions.runtime import PolicyRuntime, ScenarioRuntime
        if (type(self.world.population) is not StratifiedPopulation
                or type(self.world.book) is not OrderBook or type(self.world.clearing) is not ClearingHouse
                or type(self.world.social) is not SocialGraph or type(self.world.diffusion) is not InformationDiffusion
                or type(self.random) is not KeyedRandomSource or type(self.kernel) is not ResearchDecisionKernel
                or type(self.scenario) not in {NoOpInformationService,ScenarioRuntime}
                or type(self.policy) not in {NoOpPolicyService,PolicyRuntime}
                or not isinstance(self.journal, DiagnosticJournal) or self.measurement is not compute_step_measurement):
            raise ValueError('RUNTIME_IMPLEMENTATION_MISMATCH')
        if (self.random.spec != spec.random or self.scenario.spec != spec
                or self.scenario.implementation_version != spec.services.scenario_version
                or self.policy.implementation_version != spec.services.policy_version
                or self.journal.implementation_version != spec.services.journal_version
                or self.world.clearing.book is not self.world.book
                or self.world.diffusion.graph is not self.world.social):
            raise ValueError('RUNTIME_BINDING_MISMATCH')


class ResearchStepper:
    implementation_version = 'research-step-v1'
    phase_names = ('session_ttl_mm_refresh', 'information_release', 'apply_due_policy',
                   'synchronous_social', 'freeze_observations', 'advance_and_decide',
                   'ordered_submission', 'reconcile_and_measure', 'market_journal', 'policy_and_commit')

    def __init__(self, spec: ResearchRunSpec, services: RuntimeServices, *, run_id: str):
        validate_research_execution(spec)
        services.validate(spec)
        self.spec, self.services, self.run_id = spec, services, run_id
        self.completed_step, self.failed, self.active_phase = 0, False, 0
        self.history: list[CompletedPublicObservation] = []
        self.last_completed_trade: CompletedTradeObservation | None = None
        self.measurements: list[StepMeasurement] = []
        self.contexts: dict[str, DecisionContext] = {}
        self.order_steps: dict[str, int] = {}
        self.last_batch: DecisionBatch | None = None
        self.phase_trace: list[dict] = []
        self._initial_journal_written=False

    def _phase(self, step: int, number: int):
        self.active_phase = number
        self.phase_trace.append({'step': step, 'phase': number, 'name': self.phase_names[number-1]})

    def _draw(self, step: int, actor: str, channel: str, distribution: str = 'uniform', slot: int = 0):
        draw = self.services.random.draw_receipt(step=step, entity_id=actor, channel=channel,
            draw_slot=slot, distribution=distribution)
        self.services.journal.append('random', {'step': step, 'draw': draw.model_dump(mode='json')})
        return draw

    def _book_view(self, step: int):
        # get_best_bid/ask clean legacy heap tombstones. Observation instead
        # derives the same live levels without changing any book representation.
        levels = {OrderSide.BUY: {}, OrderSide.SELL: {}}
        for order in self.services.world.clearing.active_orders():
            side = levels[order.side]
            side[order.price] = side.get(order.price, 0) + order.remaining_qty
        bid = max(levels[OrderSide.BUY], default=None)
        ask = min(levels[OrderSide.SELL], default=None)
        quote = QuoteObservation(best_bid=bid, best_ask=ask, quote_valid=bid is not None and ask is not None,
            as_of_step=step, available_step=step)
        depth = {side: [DepthLevel(price=p, quantity=values[p]) for p in
                       sorted(values, reverse=side == OrderSide.BUY)[:self.spec.measurement.top_k]]
                 for side, values in levels.items()}
        return quote, depth

    def observe(self, step: int, social_receipt_ref: str) -> dict[str, DecisionContext]:
        """Detached immutable observations; this function commits no runtime state."""
        if self.failed or step != self.completed_step + 1:
            raise RuntimeError('INVALID_OBSERVATION_CLOCK')
        world = self.services.world
        quote, _ = self._book_view(step)
        lower, upper = world.book.get_limit_prices()
        rules = world.book.market_rules
        rule_view = ExecutionRulesView(symbol=rules.symbol, tick_size=rules.min_price_tick,
            board_lot=rules.board_lot, lower_price_limit=lower, upper_price_limit=upper,
            commission_rate=rules.commission_rate, stamp_duty_rate=rules.stamp_duty_rate,
            seller_only_stamp_duty=rules.seller_only_stamp_duty)
        profiles = {p.profile_id: p for p in self.spec.behavior.parameter_profiles}
        social = world.diffusion.research_signals()
        public_policy = self.services.policy.public_observation(step)
        result = {}
        for aid in sorted(self.spec.world.stable_actor_ids):
            account = world.clearing.accounts[aid]
            free_cash, free_shares = world.clearing.available(aid)
            pending = world.clearing.active_orders(aid)
            obs = ActorObservation(actor_id=aid, step=step, as_of_state_version=self.completed_step,
                account=AccountObservation(cash=account.cash, shares=account.shares,
                    available_cash=free_cash, available_shares=free_shares,
                    pending_buy_qty=sum(o.remaining_qty for o in pending if o.side == OrderSide.BUY),
                    pending_sell_qty=sum(o.remaining_qty for o in pending if o.side == OrderSide.SELL),
                    acquired_today=account.acquired_today, cost_basis=account.cost_basis,
                    t_plus_one=world.clearing.t_plus_one), initial_wealth=world.actor_initial_wealth[aid],
                initial_price=self.spec.world.initial_price, max_inventory_shares=world.actor_max_inventory[aid],
                current_quote=quote, completed_public_history=self.history,
                last_completed_trade=self.last_completed_trade, private_valuation=self.services.scenario.private.get(aid),
                social=SocialObservation(**social[aid], delivery_receipt_ref=social_receipt_ref),
                public_policy=public_policy)
            profile=profiles[world.actor_profile_ids[aid]]
            if hasattr(self.services.scenario,'effective_profile'):
                profile=self.services.scenario.effective_profile(aid,step,profile)
            result[aid] = DecisionContext(step=step, completed_step=self.completed_step,
                state_version=self.completed_step, observation=obs,
                profile=profile, reference_policy=self.spec.behavior.reference_policy,
                rules=rule_view, behavior_spec_hash=research_hash(self.spec.behavior),
                world_spec_hash=world.snapshot.initialization_spec_hash)
        return result

    def observe_policy(self,step):
        from core.interventions.runtime import PolicyObservation
        world=self.services.world;aid=self.spec.world.policy_account_initial.actor_id
        account=world.clearing.accounts[aid];free_cash,free_shares=world.clearing.available(aid)
        pending=world.clearing.active_orders(aid);quote,_=self._book_view(step)
        return PolicyObservation(actor_id=aid,step=step,
            account=AccountObservation(cash=account.cash,shares=account.shares,
                available_cash=free_cash,available_shares=free_shares,
                pending_buy_qty=sum(o.remaining_qty for o in pending if o.side==OrderSide.BUY),
                pending_sell_qty=sum(o.remaining_qty for o in pending if o.side==OrderSide.SELL),
                acquired_today=account.acquired_today,cost_basis=account.cost_basis,t_plus_one=world.clearing.t_plus_one),
            initial_price=self.spec.world.initial_price,current_quote=quote,completed_public_history=self.history,
            last_completed_trade=self.last_completed_trade,public_policy=self.services.policy.public_observation(step))

    def _sync_population(self):
        world = self.services.world
        for index, aid in enumerate(self.spec.world.stable_actor_ids):
            account, state = world.clearing.accounts[aid], world.behavior_states[aid]
            world.population.state[index, :3] = account.cash, account.shares, account.cost_basis
            world.population.reference_points[index, :] = state.reference_anchor
            world.population.loss_aversion_intensity_state[index] = state.lambda_effective
        sentiments = {'S': 0.0, 'I': -1.0, 'R': 0.0, 'B': 1.0}
        for index, node in world.social.agents.items():
            world.population.state[index, 3] = sentiments[node.sentiment_state.value]

    def _write_agents(self,step):
        world=self.services.world
        for aid,account in sorted(world.clearing.accounts.items()):
            state=world.behavior_states.get(aid)
            self.services.journal.append('agents',{'step':step,'actor_id':aid,
                'kind':'policy_account' if state is None else 'ordinary_or_normal_mm',
                'account':asdict(account),'behavior_state':None if state is None else state.model_dump(mode='json')})

    def advance_one_step(self) -> StepMeasurement:
        if self.failed:
            raise RuntimeError('FAILED_ATTEMPT: restart the full job from its complete initial world')
        if self.completed_step >= self.spec.base.steps:
            raise RuntimeError('HORIZON_EXHAUSTED')
        step, services = self.completed_step + 1, self.services
        world, journal = services.world, services.journal
        try:
            if not self._initial_journal_written:
                self._write_agents(0)
                for draw in world.snapshot.initialization_draws:
                    journal.append('random',{'step':0,'draw':draw.model_dump(mode='json')})
                self._initial_journal_written=True
            self._phase(step, 1)
            world.clearing.ensure_usable()
            cancel_start = len(world.clearing.cancellation_history)
            if step > 1 and (step-1) % self.spec.base.market.steps_per_day == 0:
                # Session mark is the latest genuine completed trade, or the
                # previous declared close when no trade has ever occurred.
                close = self.last_completed_trade.price if self.last_completed_trade else world.book.prev_close
                receipt = world.clearing.apply_session_boundary(close, step)
                boundary = receipt.to_dict()
                boundary['unlocked_accounts'] = list(receipt.unlocked_accounts)
                boundary['cancellations'] = [asdict(row) for row in receipt.cancellations]
                boundary['cancelled_order_ids'] = list(receipt.cancelled_order_ids)
                journal.append('session_boundaries', boundary)
            kinds = {actor.actor_id: actor.actor_kind for actor in world.snapshot.actors}
            for order in list(world.clearing.active_orders()):
                age = step - self.order_steps[order.order_id]
                reason = ('ttl' if age >= self.spec.base.market.order_ttl else
                          'policy_lp_refresh' if order.agent_id == self.spec.world.policy_account_initial.actor_id else
                          'normal_mm_refresh' if kinds.get(order.agent_id) == 'normal_mm' else None)
                if reason is not None:
                    world.clearing.cancel(order.order_id, reason=reason)

            self._phase(step, 2)
            information=services.scenario.release(step)
            journal.append('information', information)
            for draw in information.get('target_draws',[]):journal.append('random',{'step':step,'draw':draw})
            self._phase(step, 3)
            due=services.policy.apply_due(step)
            journal.append('policy_due', due)
            journal.append('policy_execution',due)
            self._phase(step, 4)
            social_draws = {aid: self._draw(step, aid, 'social_transition').value
                            for aid in sorted(self.spec.world.stable_actor_ids)}
            social = world.diffusion.advance_research_social(step, social_draws)
            social_ref = journal.append('social', social)
            self._phase(step, 5)
            contexts = self.observe(step, social_ref)
            self.contexts = contexts
            policy_observation=self.observe_policy(step) if hasattr(services.policy,'quote') else None
            self._phase(step, 6)
            batches, advances, new_states = [], [], {}
            for aid, context in contexts.items():
                noise = DecisionNoise(actor_id=aid, step=step,
                    participation_draw=self._draw(step, aid, 'participation'),
                    forecast_draw=self._draw(step, aid, 'forecast_noise', 'normal'))
                state, advance = services.kernel.advance(world.behavior_states[aid], context, noise)
                new_states[aid] = state
                advances.append(advance)
                batches.append(services.kernel.decide(state, context, noise))
            batch = DecisionBatch.combine(batches, expected_actor_ids=list(self.spec.world.stable_actor_ids),
                                          advance_receipts=advances)
            world.behavior_states = new_states
            self.last_batch = batch
            policy_intents,policy_record=(services.policy.quote(policy_observation,next(iter(contexts.values())).rules)
                if policy_observation is not None else ([],None))
            self._phase(step, 7)
            all_intents=list(batch.intents)+policy_intents
            prioritized = [(self._draw(step, intent.actor_id, 'submission_priority', slot=intent.slot).value,
                            intent.actor_id, intent.slot, intent) for intent in all_intents]
            trades, submissions, rejections = [], [], []
            for priority, aid, slot, intent in sorted(prioritized, key=lambda row: row[:3]):
                if not services.policy.market_open:
                    submissions.append({'step':step,'priority':priority,'actor_id':aid,'slot':slot,
                        'order_id':intent.order_id,'status':'HALT_SUPPRESSED','reason':'HALT_ACTIVE','trade_ids':[]})
                    continue
                if hasattr(services.policy,'authorize_intent'):services.policy.authorize_intent(intent)
                order = Order(symbol=self.spec.base.market.symbol, price=intent.limit_price,
                    quantity=intent.quantity_shares, side=OrderSide(intent.side),
                    order_type=OrderType.POST_ONLY if intent.order_type == 'POST_ONLY' else OrderType.LIMIT,
                    agent_id=aid, timestamp=float(step), order_id=intent.order_id,
                    metadata={'research_step': step, 'intent_id': intent.intent_id})
                returned = world.clearing.submit(order)
                if hasattr(services.policy,'consume_settled_trades'):services.policy.consume_settled_trades(returned)
                self.order_steps[order.order_id] = step
                trades.extend(returned)
                submissions.append({'step': step, 'priority': priority, 'actor_id': aid, 'slot': slot,
                    'order_id': order.order_id, 'status': order.status.value, 'reason': order.reason,
                    'trade_ids': [trade.trade_id for trade in returned]})
                if order.status == OrderStatus.REJECTED:
                    rejections.append(RejectionObservation(attempt_id=order.order_id, reason=order.reason))

            self._phase(step, 8)
            invariant = world.clearing.check_invariants()
            quote, depth = self._book_view(step)
            last = (CompletedTradeObservation(trade_id=trades[-1].trade_id, price=trades[-1].price,
                     as_of_step=step, available_step=step+1) if trades else self.last_completed_trade)
            historical_quote = next((row for row in reversed(self.history) if row.quote_valid), None)
            history_ref = (PriceReference(price=historical_quote.quote_mid, source='completed_quote',
                           as_of_step=historical_quote.step) if historical_quote else
                           PriceReference(price=self.spec.world.initial_price, source='initial_condition', as_of_step=0))
            lower, upper = world.book.get_limit_prices()
            snapshot = MarketSnapshot(captured=True, best_bid=quote.best_bid, best_ask=quote.best_ask,
                bid_depth=depth[OrderSide.BUY], ask_depth=depth[OrderSide.SELL],
                depth_levels=self.spec.measurement.top_k, current_quote_source='book_end_market', quote_as_of_step=step,
                last_trade_price=None if last is None else last.price,
                last_trade_step=None if last is None else last.as_of_step,
                history_reference=history_ref, lower_price_limit=lower, upper_price_limit=upper)
            cancels = [CancellationObservation(order_id=r.order_id, reason=r.reason)
                       for r in world.clearing.cancellation_history[cancel_start:]]
            activity=StepActivity(settled_trade_ids=[t.trade_id for t in trades], cancel_events=cancels,
                    reject_events=rejections, ioc_remainder_count=0,
                    halt_suppressed_count=len(all_intents) if not services.policy.market_open else 0)
            measurement = services.measurement(snapshot=snapshot, trade_tape=trades,activity=activity,
                context=MeasurementContext(spec=self.spec.measurement, world_id=world.snapshot.world_hash,
                    run_id=self.run_id, step=step, session_id=f'session-{(step-1)//self.spec.base.market.steps_per_day}',
                    policy_hash=services.policy.policy_hash, market_open=services.policy.market_open, observation_complete=True,
                    execution_status='completed', origin='market', policy_activity={
                        'halt_active':not services.policy.market_open,'liquidity_active':services.policy.liquidity_active}))
            self._sync_population()
            self._phase(step, 9)
            market_ref = journal.append('measurements', measurement)
            journal.append('step_activity',{'step':step,'activity':activity.model_dump(mode='json')})
            invariant_record = asdict(invariant)
            invariant_record['account_cash_residuals'] = dict(invariant.account_cash_residuals)
            journal.append('market', {'step': step, 'snapshot': snapshot.model_dump(mode='json'),
                                      'measurement_record_hash': market_ref, 'invariants': invariant_record})
            journal.append('decisions', batch)
            if policy_record is not None:
                journal.append('policy_decisions',policy_record)
                journal.append('policy_observations',policy_observation)
                journal.append('policy_execution',{'phase':6,**policy_record})
            for aid, context in contexts.items():
                journal.append('observations', {'step': step, 'actor_id': aid, 'context': context.model_dump(mode='json')})
            for trade in trades:
                journal.append('trades', {'step': step, **asdict(trade)})
            for row in submissions:
                journal.append('submissions', row)
            journal.append('accounts', {'step': step, 'fees': world.clearing.fees,
                'accounts': {aid: asdict(account) for aid, account in sorted(world.clearing.accounts.items())}})
            self._write_agents(step)
            journal.append('social_state',{'step':step,'state':world.diffusion.research_snapshot().model_dump(mode='json')})
            self._phase(step, 10)
            proxy, mid = measurement.metrics['price_proxy'], measurement.metrics['mid_price']
            completed_observation = CompletedPublicObservation(step=step, price=proxy.value,
                price_status=proxy.status, quote_mid=mid.value if measurement.quote_valid else None,
                quote_valid=measurement.quote_valid, available_step=step+1)
            policy_event=services.policy.evaluate_completed_market(step, measurement)
            policy_ref = journal.append('policy_events', policy_event)
            journal.append('policy_execution',policy_event)
            journal.append('step_commits', {'step': step, 'completed_step': step, 'market_record_hash': market_ref,
                'policy_record_hash': policy_ref, 'phase_order': list(self.phase_names)})
            # No operation that can perform I/O or user code follows this commit.
            self.completed_step = step
            self.last_completed_trade = last
            self.history.append(completed_observation)
            self.measurements.append(measurement)
            return measurement
        except BaseException:
            self.failed = True
            raise
