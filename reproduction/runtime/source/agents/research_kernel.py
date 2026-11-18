"""Pure finite rule decisions; all actual execution belongs to ClearingHouse."""
from __future__ import annotations

import math

from agents.decision_context import (
    BehaviorAdvanceReceipt, BehaviorState, CandidateScore, DecisionBatch,
    DecisionContext, DecisionNoise, DecisionRecord, OrderIntent, SignalComponent,
    stable_order_id,
)
from core.experiments.design_config import BehaviorParameterProfile
from core.experiments.identity import research_hash
from core.exchange.market_rules import execution_fees, worst_buy_cash, floor_to_tick, ceil_to_tick


# Registered consumers, including declared bounds and state updates. Fixed
# alpha/lambda in fundamental/trend are not a claim of heterogeneous preferences.
_COMMON = frozenset({"participation_rate", "trade_fraction", "max_inventory_shares",
                     "reference_adaptation", "lambda_base", "social_lambda_gain", "lambda_max"})
_DIRECTIONAL = frozenset({"beta_f", "beta_t", "beta_s", "forecast_noise_std", "signal_max",
                          "mu_max", "forecast_sigma", "sigma_max", "alpha", "aggressiveness",
                          "aggressiveness_max", "indifference_epsilon"})
_MM = frozenset({"signal_delay_steps", "q_target_shares", "inventory_skew", "half_spread_fraction", "quote_qty_lots"})


def active_parameter_fields(profile: BehaviorParameterProfile) -> list[str]:
    """Actual profile consumers in the kernel and its information-release input.

    signal_delay_steps is consumed by the registered information release
    service, not reapplied to an already delivered signal by decide().
    """
    fields = set(_COMMON | (_MM if profile.actor_kind == "normal_mm" else _DIRECTIONAL))
    if profile.actor_kind != "normal_mm":
        if profile.beta_f > 0.0:
            fields.add("signal_delay_steps")
        if profile.beta_t > 0.0:
            fields.add("trend_window")
    return sorted(fields)


def inactive_parameter_fields(profile: BehaviorParameterProfile) -> list[str]:
    return sorted(set(type(profile).model_fields) - set(active_parameter_fields(profile)) - {"profile_id", "actor_kind"})


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _signal(value: float | None, status: str, source: str, reason: str | None = None) -> SignalComponent:
    return SignalComponent(value=value, status=status, source=source, reason=reason)


def _check_scope(state: BehaviorState, context: DecisionContext, noise: DecisionNoise) -> None:
    if not isinstance(state, BehaviorState) or not isinstance(context, DecisionContext) or not isinstance(noise, DecisionNoise):
        raise TypeError("Kernel requires strict BehaviorState, DecisionContext and DecisionNoise")
    if state.actor_id != context.observation.actor_id or state.actor_id != noise.actor_id or noise.step != context.step:
        raise ValueError("INVALID_CONTEXT: state, observation and draw actor/step must match")


def _intent(context: DecisionContext, side: str, price: float, quantity: int, reason: str) -> OrderIntent:
    if quantity <= 0 or quantity % context.rules.board_lot:
        raise ValueError("INVALID_CONTEXT: orders require positive integer lots")
    if not context.rules.lower_price_limit <= price <= context.rules.upper_price_limit or floor_to_tick(price, context.rules.tick_size) != price:
        raise ValueError("INVALID_CONTEXT: intent violates the observed market price rules")
    slot = 0 if side == "buy" else 1
    oid = stable_order_id(context.step, context.observation.actor_id, slot)
    return OrderIntent(intent_id=oid, order_id=oid, actor_id=context.observation.actor_id,
        step=context.step, slot=slot, side=side,
        order_type="POST_ONLY" if context.profile.actor_kind == "normal_mm" else "LIMIT",
        limit_price=price, quantity_shares=quantity, reason_code=reason,
        observation_hash=research_hash(context.observation), policy_hash=None)


def _record_values(state: BehaviorState, context: DecisionContext, noise: DecisionNoise) -> dict:
    return dict(actor_id=state.actor_id, step=context.step, actor_kind=context.profile.actor_kind,
        profile_id=context.profile.profile_id, context_hash=research_hash(context),
        observation_hash=research_hash(context.observation), noise_hash=research_hash(noise),
        state_hash=research_hash(state), reference=context.public_reference,
        reference_anchor=state.reference_anchor, lambda_effective=state.lambda_effective,
        signal_components={}, subjective_mu=None, subjective_prices=[], subjective_probabilities=[],
        candidate_scores=[], rejected_candidates={}, mm_center_source=None,
        mm_inventory_gap=None, mm_center=None, mm_raw_bid=None, mm_raw_ask=None)


def _batch(context: DecisionContext, values: dict, action: str, reason: str, intents: list[OrderIntent]) -> DecisionBatch:
    record = DecisionRecord(**values, action=action, reason_code=reason)
    return DecisionBatch(scope="actor", step=context.step, records=[record], intents=intents,
                         cancel_requests=[], state_advance_receipts=[])


def _fresh_private(context: DecisionContext):
    value = context.observation.private_valuation
    if value is None or context.step - value.as_of_step > context.reference_policy.max_private_signal_age_steps:
        return None
    return value


def _signals(context: DecisionContext, noise: DecisionNoise, pref: float) -> dict[str, SignalComponent]:
    profile, obs = context.profile, context.observation
    result = {}
    if profile.beta_f == 0.0:
        result["fundamental"] = _signal(None, "INACTIVE", "profile", "BETA_F_ZERO")
    else:
        private = _fresh_private(context)
        result["fundamental"] = (_signal(None, "UNAVAILABLE", "own_private_valuation", "PRIVATE_SIGNAL_UNAVAILABLE")
            if private is None else _signal(_clip(private.value / pref - 1.0, -profile.signal_max, profile.signal_max), "VALID", private.source))
    if profile.beta_t == 0.0:
        result["trend"] = _signal(None, "INACTIVE", "profile", "BETA_T_ZERO")
    else:
        history = {row.step: row for row in obs.completed_public_history}
        steps = range(context.completed_step - profile.trend_window, context.completed_step + 1)
        rows = [history.get(t) for t in steps]
        if any(row is None or row.price_status != "VALID" for row in rows):
            result["trend"] = _signal(None, "UNAVAILABLE", "completed_adjacent_returns", "TREND_HISTORY_INSUFFICIENT")
        else:
            returns = [b.price / a.price - 1.0 for a, b in zip(rows[:-1], rows[1:])]
            if any(not math.isfinite(value) for value in returns):
                raise ValueError("INVALID_CONTEXT: nonfinite trend return")
            trend = math.fsum(value / len(returns) for value in returns)
            result["trend"] = _signal(_clip(trend, -profile.signal_max, profile.signal_max), "VALID", "completed_adjacent_returns")
    result["social"] = (_signal(None, "INACTIVE", "profile", "BETA_S_ZERO") if profile.beta_s == 0.0
                        else _signal(obs.social.signal, "VALID", obs.social.source))
    result["forecast_noise"] = _signal(profile.forecast_noise_std * noise.forecast_draw.value, "VALID", "addressed_forecast_noise")
    return result


def _quantity(context: DecisionContext, side: str, price: float, *, mm: bool = False) -> int:
    obs, profile, rules = context.observation, context.profile, context.rules
    account = obs.account
    caps = [profile.trade_fraction * obs.initial_wealth / price]
    if mm:
        caps.append(profile.quote_qty_lots * rules.board_lot)
    if side == "buy":
        caps += [account.available_cash / worst_buy_cash(price, 1, rules.commission_rate),
                 max(0, obs.max_inventory_shares - account.shares - account.pending_buy_qty)]
    else:
        caps.append(account.available_shares)
    # The raw authoritative account remains intact; a tiny allowed negative
    # free-cash residual has zero purchasing capacity, never a negative order.
    return rules.board_lot * math.floor(max(0.0, min(caps)) / rules.board_lot)


class ResearchDecisionKernel:
    """No stateful constructor, RNG, network, account mutation or hidden fallback."""

    @staticmethod
    def advance(previous_state: BehaviorState, context: DecisionContext, noise: DecisionNoise) -> tuple[BehaviorState, BehaviorAdvanceReceipt]:
        _check_scope(previous_state, context, noise)
        if (previous_state.last_advanced_step != context.step - 1
                or previous_state.state_version != context.state_version):
            raise ValueError("DUPLICATE_OR_OUT_OF_ORDER_ADVANCE: exactly one sequential advance is required")
        profile, ref = context.profile, context.public_reference
        old_anchor = previous_state.reference_anchor
        anchor = (old_anchor if ref.value is None else
                  (1.0 - profile.reference_adaptation) * old_anchor + profile.reference_adaptation * ref.value)
        effective = profile.lambda_base * (1.0 + profile.social_lambda_gain * context.observation.social.stress)
        if not math.isfinite(effective) or effective > profile.lambda_max:
            raise ValueError("INVALID_CONTEXT: effective lambda exceeds the registered bound")
        new_state = BehaviorState(actor_id=previous_state.actor_id, reference_anchor=anchor,
            lambda_effective=effective, state_version=previous_state.state_version + 1,
            last_advanced_step=context.step, last_context_hash=research_hash(context))
        receipt = BehaviorAdvanceReceipt(actor_id=new_state.actor_id, step=context.step,
            previous_state_hash=research_hash(previous_state), new_state_hash=research_hash(new_state),
            context_hash=research_hash(context), noise_hash=research_hash(noise),
            old_anchor=old_anchor, new_anchor=anchor, old_lambda=previous_state.lambda_effective,
            new_lambda=effective, reference=ref, social_signal=context.observation.social.signal,
            stress=context.observation.social.stress)
        return new_state, receipt

    @staticmethod
    def decide(state: BehaviorState, context: DecisionContext, noise: DecisionNoise) -> DecisionBatch:
        _check_scope(state, context, noise)
        if state.last_advanced_step != context.step or state.last_context_hash != research_hash(context):
            raise ValueError("INVALID_CONTEXT: decide requires this context's already advanced state")
        values = _record_values(state, context, noise)
        if noise.participation_draw.value >= context.profile.participation_rate:
            return _batch(context, values, "HOLD", "NOT_PARTICIPATING", [])
        if context.profile.actor_kind == "normal_mm":
            return ResearchDecisionKernel._market_maker(state, context, values)
        ref = context.public_reference
        if ref.value is None:
            return _batch(context, values, "HOLD", "REFERENCE_UNAVAILABLE", [])
        profile, rules, obs = context.profile, context.rules, context.observation
        signals = _signals(context, noise, ref.value)
        values["signal_components"] = signals
        missing = [item.reason for item in signals.values() if item.status == "UNAVAILABLE"]
        if missing:
            return _batch(context, values, "HOLD", missing[0], [])
        weighted = [signals["forecast_noise"].value]
        for name, weight in (("fundamental", profile.beta_f), ("trend", profile.beta_t), ("social", profile.beta_s)):
            if weight != 0.0:
                weighted.append(weight * signals[name].value)
        mu = _clip(math.fsum(weighted), -profile.mu_max, profile.mu_max)
        prices = [ref.value * (1.0 + mu - profile.forecast_sigma),
                  ref.value * (1.0 + mu), ref.value * (1.0 + mu + profile.forecast_sigma)]
        if any(not math.isfinite(price) or price <= 0.0 for price in prices):
            raise ValueError("INVALID_CONTEXT: subjective price support must be positive and finite")
        values.update(subjective_mu=mu, subjective_prices=prices, subjective_probabilities=[0.25, 0.5, 0.25])
        buy = floor_to_tick(ref.value * (1.0 + profile.aggressiveness), rules.tick_size)
        sell = ceil_to_tick(ref.value * (1.0 - profile.aggressiveness), rules.tick_size)
        candidates = [(0, ref.value)]
        for side, price in (("sell", sell), ("buy", buy)):
            if price <= 0.0 or not rules.lower_price_limit <= price <= rules.upper_price_limit:
                values["rejected_candidates"][side] = "PRICE_OUTSIDE_RULES"
                continue
            quantity = _quantity(context, side, price)
            if quantity == 0:
                values["rejected_candidates"][side] = "BELOW_LOT_OR_RESOURCE_CAP"
                continue
            candidates.append((quantity if side == "buy" else -quantity, price))
        reference_wealth = obs.account.cash + obs.account.shares * state.reference_anchor
        scale = max(obs.account.cash + obs.account.shares * ref.value, rules.board_lot * ref.value)
        scored = []
        for quantity, limit in candidates:
            buyer_fee, seller_fee, seller_tax = execution_fees(limit, abs(quantity), rules.commission_rate,
                rules.stamp_duty_rate, seller_only_stamp_duty=rules.seller_only_stamp_duty)
            fee = buyer_fee if quantity > 0 else seller_fee + seller_tax if quantity < 0 else 0.0
            cash = obs.account.cash - quantity * limit - fee
            shares = obs.account.shares + quantity
            outcomes = [(cash + shares * price - reference_wealth) / scale for price in prices]
            prospect = [z ** profile.alpha if z >= 0.0 else -state.lambda_effective * (-z) ** profile.alpha for z in outcomes]
            score = math.fsum(p * v for p, v in zip((0.25, 0.5, 0.25), prospect))
            scored.append(CandidateScore(signed_quantity_shares=quantity, limit_price=limit, fee=fee,
                hypothetical_cash=cash, hypothetical_shares=shares, normalized_outcomes=outcomes, score=score))
        values["candidate_scores"] = scored
        hold = scored[0]
        # Exact ties use smaller absolute position change, then SELL before BUY.
        best = min(scored, key=lambda item:(-item.score, abs(item.signed_quantity_shares), item.signed_quantity_shares > 0))
        if best.signed_quantity_shares == 0 or best.score <= hold.score + profile.indifference_epsilon:
            return _batch(context, values, "HOLD", "NO_SCORE_IMPROVEMENT", [])
        side = "buy" if best.signed_quantity_shares > 0 else "sell"
        order = _intent(context, side, best.limit_price, abs(best.signed_quantity_shares), "PROSPECT_SCORE_IMPROVEMENT")
        return _batch(context, values, side.upper(), "PROSPECT_SCORE_IMPROVEMENT", [order])

    @staticmethod
    def _market_maker(state: BehaviorState, context: DecisionContext, values: dict) -> DecisionBatch:
        profile, obs, rules = context.profile, context.observation, context.rules
        private, public = _fresh_private(context), context.public_reference
        if private is not None:
            valuation, source = private.value, "private_valuation"
        elif public.value is not None:
            valuation, source = public.value, "public_reference"
        else:
            return _batch(context, values, "HOLD", "REFERENCE_UNAVAILABLE", [])
        gap = _clip((obs.account.shares - profile.q_target_shares) / max(profile.q_target_shares, rules.board_lot), -1.0, 1.0)
        center = valuation * (1.0 - profile.inventory_skew * gap)
        raw_bid, raw_ask = center * (1.0 - profile.half_spread_fraction), center * (1.0 + profile.half_spread_fraction)
        if any(not math.isfinite(value) or value <= 0.0 for value in (center, raw_bid, raw_ask)):
            raise ValueError("INVALID_CONTEXT: MM center and quotes must be positive and finite")
        values.update(mm_center_source=source, mm_inventory_gap=gap, mm_center=center,
                      mm_raw_bid=raw_bid, mm_raw_ask=raw_ask)
        bid, ask = floor_to_tick(raw_bid, rules.tick_size), ceil_to_tick(raw_ask, rules.tick_size)
        # Only current quotes constrain executable POST_ONLY prices. A historical
        # public reference can set V but cannot masquerade as a current BBO.
        quote = obs.current_quote
        if context.step - quote.as_of_step <= context.reference_policy.max_quote_age_steps:
            if quote.best_ask is not None:
                bound = quote.best_ask - rules.tick_size
                bid = 0.0 if bound <= 0.0 else min(bid, floor_to_tick(bound, rules.tick_size))
            if quote.best_bid is not None:
                ask = max(ask, ceil_to_tick(quote.best_bid + rules.tick_size, rules.tick_size))
        if bid >= ask:
            raise ValueError("INVALID_CONTEXT: MM sides must have strictly ordered prices")
        intents = []
        for side, price in (("buy", bid), ("sell", ask)):
            if price <= 0.0 or not rules.lower_price_limit <= price <= rules.upper_price_limit:
                values["rejected_candidates"][side] = "PRICE_OUTSIDE_RULES"
                continue
            quantity = _quantity(context, side, price, mm=True)
            if quantity == 0:
                values["rejected_candidates"][side] = "BELOW_LOT_OR_RESOURCE_CAP"
                continue
            intents.append(_intent(context, side, price, quantity, "NORMAL_MM_QUOTE"))
        return _batch(context, values, "QUOTE" if intents else "HOLD",
                      "NORMAL_MM_QUOTE" if intents else "NO_FEASIBLE_MM_SIDE", intents)
