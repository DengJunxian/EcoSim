"""Immutable actor-local inputs and outputs for prospect-three-point-v1.

No runtime/world objects cross this boundary. Historical rows must be authorized
by the stepper's completed_step, not merely present in a market journal.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from core.experiments.strict_types import StrictResearchModel
from core.experiments.design_config import BehaviorParameterProfile, ReferencePolicy
from core.experiments.identity import research_hash
from core.experiments.randomness import RandomDraw
from core.exchange.clearing import CASH_ATOL
from core.exchange.market_rules import floor_to_tick

BEHAVIOR_VERSION = "prospect-three-point-v1"
HASH_PATTERN = r"^[0-9a-f]{64}$"


class AccountObservation(StrictResearchModel):
    cash: float
    shares: int = Field(ge=0)
    available_cash: float
    available_shares: int = Field(ge=0)
    pending_buy_qty: int = Field(ge=0)
    pending_sell_qty: int = Field(ge=0)
    acquired_today: int = Field(ge=0)
    cost_basis: float = Field(ge=0.0)
    t_plus_one: bool

    @model_validator(mode="after")
    def _resources(self):
        locked = self.acquired_today if self.t_plus_one else 0
        if (self.cash < -CASH_ATOL or self.available_cash < -CASH_ATOL
                or self.available_cash > self.cash + CASH_ATOL
                or self.available_shares != self.shares - locked - self.pending_sell_qty):
            raise ValueError("INVALID_CONTEXT: inconsistent authoritative clearing resources")
        return self


class ExecutionRulesView(StrictResearchModel):
    symbol: str = Field(min_length=1)
    tick_size: float = Field(gt=0.0)
    board_lot: int = Field(ge=1)
    lower_price_limit: float = Field(gt=0.0)
    upper_price_limit: float = Field(gt=0.0)
    commission_rate: float = Field(ge=0.0)
    stamp_duty_rate: float = Field(ge=0.0)
    seller_only_stamp_duty: bool

    @model_validator(mode="after")
    def _limits(self):
        if self.lower_price_limit > self.upper_price_limit:
            raise ValueError("INVALID_CONTEXT: inverted session limits")
        return self


class QuoteObservation(StrictResearchModel):
    best_bid: float | None = Field(gt=0.0)
    best_ask: float | None = Field(gt=0.0)
    quote_valid: bool
    source: Literal["pre_decision_book"] = "pre_decision_book"
    as_of_step: int = Field(ge=0)
    available_step: int = Field(ge=0)

    @model_validator(mode="after")
    def _quote(self):
        both = self.best_bid is not None and self.best_ask is not None
        if both and self.best_bid > self.best_ask:
            raise ValueError("INVALID_CONTEXT: crossed current BBO")
        if self.quote_valid != both or self.available_step < self.as_of_step:
            raise ValueError("INVALID_CONTEXT: current quote flags or times disagree")
        return self


class CompletedPublicObservation(StrictResearchModel):
    step: int = Field(ge=1)
    price: float | None = Field(gt=0.0)
    price_status: Literal["VALID", "UNDEFINED", "NOT_OBSERVED", "INVALID_INPUT", "NOT_IMPLEMENTED"]
    quote_mid: float | None = Field(gt=0.0)
    quote_valid: bool
    source: Literal["completed_measurement_v2"] = "completed_measurement_v2"
    available_step: int = Field(ge=1)

    @model_validator(mode="after")
    def _history(self):
        if ((self.price is not None) != (self.price_status == "VALID")
                or (self.quote_mid is not None) != self.quote_valid
                or self.available_step <= self.step):
            raise ValueError("INVALID_CONTEXT: completed history value, validity or release time disagree")
        return self


class CompletedTradeObservation(StrictResearchModel):
    trade_id: str = Field(min_length=1)
    price: float = Field(gt=0.0)
    as_of_step: int = Field(ge=0)
    available_step: int = Field(ge=1)
    source: Literal["completed_settled_trade"] = "completed_settled_trade"

    @model_validator(mode="after")
    def _trade_time(self):
        if self.available_step <= self.as_of_step:
            raise ValueError("INVALID_CONTEXT: current unsettled step cannot supply trade history")
        return self


class PrivateValuation(StrictResearchModel):
    owner_actor_id: str = Field(min_length=1)
    value: float = Field(gt=0.0)
    source: Literal["initial_private_signal", "released_private_signal"]
    as_of_step: int = Field(ge=0)
    available_step: int = Field(ge=0)
    information_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _arrival(self):
        if self.available_step < self.as_of_step:
            raise ValueError("INVALID_CONTEXT: private signal arrives before its source")
        return self


class SocialObservation(StrictResearchModel):
    signal: float = Field(ge=-1.0, le=1.0)
    stress: float = Field(ge=0.0, le=1.0)
    source: Literal["synchronous_social", "no_neighbors"]
    as_of_step: int = Field(ge=0)
    available_step: int = Field(ge=0)
    delivery_receipt_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def _stress(self):
        if (self.stress != max(0.0, -self.signal) or self.available_step < self.as_of_step
                or (self.source == "no_neighbors" and self.signal != 0.0)):
            raise ValueError("INVALID_CONTEXT: social stress must derive once from the delivered signal")
        return self


class PublicPolicyObservation(StrictResearchModel):
    market_open: bool
    liquidity_active: bool
    source: Literal["announced_public_policy"] = "announced_public_policy"
    as_of_step: int = Field(ge=0)
    available_step: int = Field(ge=0)
    announcement_refs: list[str]

    @model_validator(mode="after")
    def _policy_time(self):
        if self.available_step < self.as_of_step or any(not item for item in self.announcement_refs):
            raise ValueError("INVALID_CONTEXT: policy timing or public references are invalid")
        return self


class ActorObservation(StrictResearchModel):
    actor_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    as_of_state_version: int = Field(ge=0)
    account: AccountObservation
    initial_wealth: float = Field(gt=0.0)
    initial_price: float = Field(gt=0.0)
    max_inventory_shares: int = Field(ge=0)
    current_quote: QuoteObservation
    completed_public_history: list[CompletedPublicObservation]
    last_completed_trade: CompletedTradeObservation | None
    private_valuation: PrivateValuation | None
    social: SocialObservation
    public_policy: PublicPolicyObservation

    @model_validator(mode="after")
    def _visibility(self):
        data = [self.current_quote, self.social, self.public_policy]
        if self.private_valuation is not None:
            if self.private_valuation.owner_actor_id != self.actor_id:
                raise ValueError("INVALID_CONTEXT: another actor's private valuation")
            data.append(self.private_valuation)
        if self.last_completed_trade is not None:
            data.append(self.last_completed_trade)
        if any(value.available_step > self.step or value.as_of_step > self.step for value in data):
            raise ValueError("INVALID_CONTEXT: future information is not visible")
        steps = [row.step for row in self.completed_public_history]
        if steps != sorted(set(steps)) or any(row.available_step > self.step for row in self.completed_public_history):
            raise ValueError("INVALID_CONTEXT: history must have unique sorted released steps")
        if self.social.as_of_step != self.step:
            raise ValueError("INVALID_CONTEXT: social observation must be this step's synchronous delivery")
        return self


class PublicReference(StrictResearchModel):
    value: float | None = Field(gt=0.0)
    source: Literal["observed_mid", "completed_quote_history", "fresh_last_trade", "initial_reference_only", "unavailable"]
    as_of_step: int | None = Field(ge=0)
    available_step: int | None = Field(ge=0)
    age_steps: int | None = Field(ge=0)
    reason: str | None

    @model_validator(mode="after")
    def _reference(self):
        if self.source == "unavailable":
            if any(v is not None for v in (self.value, self.as_of_step, self.available_step, self.age_steps)) or not self.reason:
                raise ValueError("Unavailable reference must preserve null and a reason")
        elif any(v is None for v in (self.value, self.as_of_step, self.available_step, self.age_steps)) or self.reason is not None:
            raise ValueError("Available reference needs value/source/time/age without a missing reason")
        return self


class DecisionContext(StrictResearchModel):
    schema_version: Literal[1] = 1
    step: int = Field(ge=1)
    phase: Literal["decision"] = "decision"
    completed_step: int = Field(ge=0)
    state_version: int = Field(ge=0)
    observation: ActorObservation
    profile: BehaviorParameterProfile
    reference_policy: ReferencePolicy
    rules: ExecutionRulesView
    behavior_spec_hash: str = Field(pattern=HASH_PATTERN)
    world_spec_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def _context(self):
        obs = self.observation
        if (obs.step != self.step or obs.as_of_state_version != self.state_version
                or self.completed_step != self.step - 1
                or obs.max_inventory_shares != self.profile.max_inventory_shares):
            raise ValueError("INVALID_CONTEXT: state/step/inventory scope mismatch")
        if any(row.step > self.completed_step for row in obs.completed_public_history):
            raise ValueError("INVALID_CONTEXT: unfinished market rows are not completed history")
        if obs.last_completed_trade is not None and obs.last_completed_trade.as_of_step > self.completed_step:
            raise ValueError("INVALID_CONTEXT: unfinished trade cannot be consumed")
        for price in (obs.current_quote.best_bid, obs.current_quote.best_ask):
            if price is not None and (not self.rules.lower_price_limit <= price <= self.rules.upper_price_limit
                                      or floor_to_tick(price, self.rules.tick_size) != price):
                raise ValueError("INVALID_CONTEXT: current BBO violates registered price rules")
        return self

    @property
    def public_reference(self) -> PublicReference:
        return select_public_reference(self.observation, self.reference_policy)


def select_public_reference(observation: ActorObservation, policy: ReferencePolicy) -> PublicReference:
    """Pure priority selection; old values keep their original time and source."""
    step, quote = observation.step, observation.current_quote
    def selected(value, source, as_of, available):
        return PublicReference(value=value, source=source, as_of_step=as_of,
                               available_step=available, age_steps=step-as_of, reason=None)
    if quote.quote_valid and step - quote.as_of_step <= policy.max_quote_age_steps:
        return selected(quote.best_bid / 2.0 + quote.best_ask / 2.0, "observed_mid", quote.as_of_step, quote.available_step)
    for row in reversed(observation.completed_public_history):
        if row.quote_valid and step - row.step <= min(policy.max_reference_age_steps, policy.max_quote_age_steps):
            return selected(row.quote_mid, "completed_quote_history", row.step, row.available_step)
    trade = observation.last_completed_trade
    if trade is not None and step - trade.as_of_step <= policy.max_trade_age_steps:
        return selected(trade.price, "fresh_last_trade", trade.as_of_step, trade.available_step)
    if step <= policy.initial_reference_grace_steps:
        return selected(observation.initial_price, "initial_reference_only", 0, 0)
    return PublicReference(value=None, source="unavailable", as_of_step=None, available_step=None,
                           age_steps=None, reason="REFERENCE_UNAVAILABLE")


class DecisionNoise(StrictResearchModel):
    schema_version: Literal[1] = 1
    algorithm_version: Literal["sha256-boxmuller-v1"] = "sha256-boxmuller-v1"
    actor_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    participation_draw: RandomDraw
    forecast_draw: RandomDraw

    @model_validator(mode="after")
    def _noise_scope(self):
        for draw, channel, domain in ((self.participation_draw, "participation", "u"),
                                      (self.forecast_draw, "forecast_noise", "n")):
            key = draw.key
            if (key.entity_or_event_id != self.actor_id or key.step != self.step
                    or key.channel != channel or key.market_phase != "decision"
                    or key.distribution_domain != domain or key.slot != 0
                    or key.algorithm != self.algorithm_version):
                raise ValueError("INVALID_CONTEXT: decision draw scope/domain mismatch")
        p, f = self.participation_draw.key, self.forecast_draw.key
        if (p.root_seed, p.exogenous_family_id, p.replication_id, p.coupling_spec_hash) != (
                f.root_seed, f.exogenous_family_id, f.replication_id, f.coupling_spec_hash):
            raise ValueError("INVALID_CONTEXT: mixed decision random scopes")
        return self


class BehaviorState(StrictResearchModel):
    schema_version: Literal[1] = 1
    behavior_version: Literal["prospect-three-point-v1"] = BEHAVIOR_VERSION
    actor_id: str = Field(min_length=1)
    reference_anchor: float = Field(gt=0.0)
    lambda_effective: float = Field(ge=1.0)
    state_version: int = Field(ge=0)
    last_advanced_step: int = Field(ge=0)
    last_context_hash: str | None = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def _state(self):
        if self.state_version != self.last_advanced_step or (self.last_context_hash is None) != (self.last_advanced_step == 0):
            raise ValueError("Behavior state must retain its unique sequential advance receipt")
        return self


class BehaviorAdvanceReceipt(StrictResearchModel):
    actor_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    previous_state_hash: str = Field(pattern=HASH_PATTERN)
    new_state_hash: str = Field(pattern=HASH_PATTERN)
    context_hash: str = Field(pattern=HASH_PATTERN)
    noise_hash: str = Field(pattern=HASH_PATTERN)
    old_anchor: float = Field(gt=0.0)
    new_anchor: float = Field(gt=0.0)
    old_lambda: float = Field(ge=1.0)
    new_lambda: float = Field(ge=1.0)
    reference: PublicReference
    social_signal: float = Field(ge=-1.0, le=1.0)
    stress: float = Field(ge=0.0, le=1.0)
    source: Literal["kernel.advance"] = "kernel.advance"


class OrderIntent(StrictResearchModel):
    schema_version: Literal[1] = 1
    intent_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    slot: Literal[0, 1]
    side: Literal["buy", "sell"]
    order_type: Literal["LIMIT", "POST_ONLY"]
    limit_price: float = Field(gt=0.0)
    quantity_shares: int = Field(gt=0)
    reason_code: str = Field(min_length=1)
    observation_hash: str = Field(pattern=HASH_PATTERN)
    behavior_version: Literal["prospect-three-point-v1", "funded-policy-lp-v1"] = BEHAVIOR_VERSION
    policy_hash: str | None = Field(default=None, pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def _identity(self):
        if self.slot != (0 if self.side == "buy" else 1) or self.order_id != stable_order_id(self.step, self.actor_id, self.slot):
            raise ValueError("Order identity must be the stable step/actor/side-slot identity")
        if self.intent_id != self.order_id:
            raise ValueError("Initial kernel uses one intent per stable order ID")
        if self.behavior_version == 'funded-policy-lp-v1' and (self.policy_hash is None or self.order_type != 'POST_ONLY'):
            raise ValueError('Funded policy LP requires exact policy identity and POST_ONLY')
        return self


def stable_order_id(step: int, actor_id: str, slot: int) -> str:
    return "research-order-" + research_hash({"step": step, "actor_id": actor_id, "slot": slot})


class SignalComponent(StrictResearchModel):
    value: float | None
    status: Literal["VALID", "INACTIVE", "UNAVAILABLE"]
    source: str = Field(min_length=1)
    reason: str | None

    @model_validator(mode="after")
    def _signal(self):
        if ((self.status == "VALID") != (self.value is not None)
                or (self.status == "VALID" and self.reason is not None)
                or (self.status != "VALID" and not self.reason)):
            raise ValueError("Signal status must distinguish active values from inactive/missing channels")
        return self


class CandidateScore(StrictResearchModel):
    signed_quantity_shares: int
    limit_price: float = Field(gt=0.0)
    fee: float = Field(ge=0.0)
    hypothetical_cash: float
    hypothetical_shares: int = Field(ge=0)
    normalized_outcomes: list[float] = Field(min_length=3, max_length=3)
    score: float


class DecisionRecord(StrictResearchModel):
    actor_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    actor_kind: Literal["fundamental", "trend", "behavioral_social", "normal_mm"]
    action: Literal["BUY", "SELL", "HOLD", "QUOTE"]
    reason_code: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    context_hash: str = Field(pattern=HASH_PATTERN)
    observation_hash: str = Field(pattern=HASH_PATTERN)
    noise_hash: str = Field(pattern=HASH_PATTERN)
    state_hash: str = Field(pattern=HASH_PATTERN)
    reference: PublicReference
    reference_anchor: float = Field(gt=0.0)
    lambda_effective: float = Field(ge=1.0)
    signal_components: dict[str, SignalComponent]
    subjective_mu: float | None
    subjective_prices: list[float]
    subjective_probabilities: list[float]
    candidate_scores: list[CandidateScore]
    rejected_candidates: dict[str, str]
    mm_center_source: Literal["private_valuation", "public_reference"] | None
    mm_inventory_gap: float | None = Field(ge=-1.0, le=1.0)
    mm_center: float | None = Field(gt=0.0)
    mm_raw_bid: float | None
    mm_raw_ask: float | None


class DecisionBatch(StrictResearchModel):
    schema_version: Literal[1] = 1
    scope: Literal["actor", "step"]
    step: int = Field(ge=1)
    records: list[DecisionRecord] = Field(min_length=1)
    intents: list[OrderIntent]
    cancel_requests: list[str]
    state_advance_receipts: list[BehaviorAdvanceReceipt]

    @model_validator(mode="after")
    def _batch(self):
        actors = [r.actor_id for r in self.records]
        if len(actors) != len(set(actors)) or any(r.step != self.step for r in self.records):
            raise ValueError("Decision batch must retain one record per actor and a single step")
        if self.scope == "actor" and (len(actors) != 1 or self.state_advance_receipts):
            raise ValueError("Pure decide returns one actor; stepper separately owns advance receipts")
        if self.cancel_requests:
            raise ValueError("Additional kernel cancellations are unsupported; refresh belongs to phase 1")
        ids = [o.intent_id for o in self.intents]
        if len(ids) != len(set(ids)) or any(o.actor_id not in actors or o.step != self.step for o in self.intents):
            raise ValueError("Order intents require unique IDs in their own actor/step scope")
        for record in self.records:
            orders = [o for o in self.intents if o.actor_id == record.actor_id]
            if record.actor_kind == "normal_mm":
                if len(orders) > 2 or any(o.order_type != "POST_ONLY" for o in orders):
                    raise ValueError("Normal MM can issue at most two POST_ONLY orders")
            elif len(orders) > 1 or any(o.order_type != "LIMIT" for o in orders):
                raise ValueError("Directional actors can issue at most one LIMIT order")
            if any(o.observation_hash != record.observation_hash for o in orders):
                raise ValueError("Order observation provenance differs from its decision")
            if (record.action == "HOLD") != (len(orders) == 0):
                raise ValueError("HOLD and order-list emptiness must agree")
        if self.scope == "step":
            receipts = [r.actor_id for r in self.state_advance_receipts]
            if sorted(receipts) != sorted(actors) or any(r.step != self.step for r in self.state_advance_receipts):
                raise ValueError("A full step needs exactly one advance receipt for every actor")
        return self

    @classmethod
    def combine(cls, batches: list[DecisionBatch], *, expected_actor_ids: list[str],
                advance_receipts: list[BehaviorAdvanceReceipt]) -> DecisionBatch:
        if not batches or any(b.scope != "actor" or b.step != batches[0].step for b in batches):
            raise ValueError("Combine requires nonempty same-step actor decisions")
        records = [r for batch in batches for r in batch.records]
        if sorted(r.actor_id for r in records) != sorted(expected_actor_ids) or len(expected_actor_ids) != len(set(expected_actor_ids)):
            raise ValueError("Full decision batch must cover the exact registered ordinary/MM roster")
        return cls(scope="step", step=batches[0].step, records=sorted(records, key=lambda r:r.actor_id),
                   intents=sorted([o for b in batches for o in b.intents], key=lambda o:(o.actor_id,o.slot)),
                   cancel_requests=[], state_advance_receipts=sorted(advance_receipts, key=lambda r:r.actor_id))
