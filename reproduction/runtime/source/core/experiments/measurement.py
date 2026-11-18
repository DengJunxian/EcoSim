"""Pure, immutable v2 market measurements; no policy utility or zero filling.

Inputs are snapshots and the caller's explicitly identified *settled* trades.
The order book's unconfirmed trade buffer is not an admissible source. This
boundary checks matching settlement IDs and completeness; actual settlement is
owned by ClearingHouse and must be attested by the runner that supplies them.
Every serialized StepMeasurement retains nulls, validity and the market clock.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
import math
import statistics
from typing import Literal

from pydantic import Field, model_validator

from core.experiments.design_config import MeasurementSpec, StrictResearchModel
from core.types import Trade

MetricStatus = Literal["VALID", "UNDEFINED", "NOT_OBSERVED", "INVALID_INPUT", "NOT_IMPLEMENTED"]
Origin = Literal["market", "analytic_fixture", "synthetic_fixture"]


class MetricValue(StrictResearchModel):
    value: float | None
    status: MetricStatus
    reason: str | None
    unit: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    denominator_count: int = Field(ge=0)
    definition_version: Literal["measurement-v2"] = "measurement-v2"

    @model_validator(mode="after")
    def _status_value(self):
        if self.status == "VALID":
            if self.value is None or self.reason is not None:
                raise ValueError("VALID metrics require a finite value and no missingness reason")
        elif self.value is not None or not self.reason:
            raise ValueError("Non-VALID metrics require null value and an explicit reason")
        return self


class DepthLevel(StrictResearchModel):
    price: float = Field(gt=0)
    quantity: int = Field(gt=0)


class PriceReference(StrictResearchModel):
    price: float = Field(gt=0)
    source: Literal["initial_condition", "last_trade", "completed_quote"]
    as_of_step: int = Field(ge=0)


class MarketSnapshot(StrictResearchModel):
    """Current BBO is separate from a historical reference or initial mark.

    Finite nonpositive/crossed BBO values can be retained as invalid-input
    diagnostics. NaN/Inf and scalar coercions are rejected at the type boundary.
    None depth means not captured; an empty list means an observed empty side.
    """

    captured: bool
    best_bid: float | None
    best_ask: float | None
    bid_depth: list[DepthLevel] | None
    ask_depth: list[DepthLevel] | None
    depth_levels: int = Field(ge=1)
    current_quote_source: Literal["book_end_market", "not_observed"]
    quote_as_of_step: int | None = Field(ge=0)
    last_trade_price: float | None = Field(default=None, gt=0)
    last_trade_step: int | None = Field(default=None, ge=0)
    history_reference: PriceReference | None = None
    lower_price_limit: float | None = Field(default=None, gt=0)
    upper_price_limit: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _snapshot_semantics(self):
        if (self.last_trade_price is None) != (self.last_trade_step is None):
            raise ValueError("A real last trade requires both price and step; initial price is a reference")
        if (self.lower_price_limit is None) != (self.upper_price_limit is None):
            raise ValueError("Both price limits must be present or absent")
        if self.lower_price_limit is not None and self.lower_price_limit > self.upper_price_limit:
            raise ValueError("Inverted price limits")
        if self.captured:
            if self.current_quote_source != "book_end_market" or self.quote_as_of_step is None:
                raise ValueError("Captured current quote needs its book source and sampling step")
        elif (self.current_quote_source != "not_observed" or self.quote_as_of_step is not None
              or self.best_bid is not None or self.best_ask is not None
              or self.bid_depth is not None or self.ask_depth is not None):
            raise ValueError("Uncaptured snapshot cannot masquerade as current observations")
        for levels, descending in ((self.bid_depth, True), (self.ask_depth, False)):
            if levels is not None:
                if len(levels) > self.depth_levels:
                    raise ValueError("Depth contains more levels than its declared top-k capture")
                prices = [row.price for row in levels]
                if len(prices) != len(set(prices)) or prices != sorted(prices, reverse=descending):
                    raise ValueError("Depth must contain distinct ordered price levels")
        return self


class CancellationObservation(StrictResearchModel):
    order_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RejectionObservation(StrictResearchModel):
    attempt_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class StepActivity(StrictResearchModel):
    """Explicit event capture; None is missing, [] is captured and empty.

    cancel_events includes successful explicit cancellations only. Failed or
    duplicate cancel requests belong in reject_events. IOC remainder is separate.
    """

    settled_trade_ids: list[str] | None
    cancel_events: list[CancellationObservation] | None
    reject_events: list[RejectionObservation] | None
    ioc_remainder_count: int | None = Field(default=None, ge=0)
    halt_suppressed_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _event_uniqueness(self):
        for values in (
            self.settled_trade_ids,
            None if self.cancel_events is None else [e.order_id for e in self.cancel_events],
            None if self.reject_events is None else [e.attempt_id for e in self.reject_events],
        ):
            if values is not None and (any(not value for value in values) or len(values) != len(set(values))):
                raise ValueError("Captured events require nonempty unique IDs")
        return self


class MeasurementContext(StrictResearchModel):
    spec: MeasurementSpec
    world_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    session_id: str = Field(min_length=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    market_open: bool
    observation_complete: bool
    execution_status: Literal["completed", "incomplete", "failed"]
    origin: Origin
    policy_activity: dict[str, bool | int | float | str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _completion(self):
        if self.observation_complete and self.execution_status != "completed":
            raise ValueError("A failed/incomplete market observation cannot be marked complete")
        return self


class SettledTrade(StrictResearchModel):
    """An immutable copy of one completed ClearingHouse trade return."""

    trade_id: str = Field(min_length=1)
    price: float = Field(gt=0)
    quantity: int = Field(gt=0)
    maker_id: str = Field(min_length=1)
    taker_id: str = Field(min_length=1)
    maker_agent_id: str = Field(min_length=1)
    taker_agent_id: str = Field(min_length=1)
    buyer_agent_id: str = Field(min_length=1)
    seller_agent_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    buyer_fee: float = Field(ge=0)
    seller_fee: float = Field(ge=0)
    seller_tax: float = Field(ge=0)

    @model_validator(mode="after")
    def _parties(self):
        if self.buyer_agent_id == self.seller_agent_id:
            raise ValueError("A settled trade cannot be a self trade")
        if {self.maker_agent_id, self.taker_agent_id} != {self.buyer_agent_id, self.seller_agent_id}:
            raise ValueError("Maker/taker and buyer/seller parties disagree")
        if self.maker_id == self.taker_id:
            raise ValueError("A settled trade needs two distinct orders")
        return self

    @classmethod
    def from_clearing_trade(cls, trade: Trade) -> SettledTrade:
        if type(trade) is not Trade:
            raise TypeError("Expected the real ClearingHouse Trade return")
        return cls.model_validate(asdict(trade))


METRIC_UNITS = {
    "spread": "price", "relative_spread": "ratio", "mid_price": "price",
    "depth_bid_top_k": "shares", "depth_ask_top_k": "shares", "depth_imbalance": "ratio",
    "trade_count": "trades", "traded_quantity": "shares", "vwap": "price",
    "avg_trade_size": "shares/trade", "explicit_cancel_count": "orders",
    "cancel_to_trade_ratio": "orders/trade", "reject_count": "attempts",
    "ioc_remainder_count": "orders", "halt_suppressed_count": "intents",
    "last_trade_price": "price", "price_proxy": "price",
    "execution_cost": "basis_points", "herding_proxy": "ratio",
}


class StepMeasurement(StrictResearchModel):
    schema_version: Literal[2] = 2
    definition_version: Literal["measurement-v2"] = "measurement-v2"
    measurement_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    session_id: str = Field(min_length=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    market_open: bool
    snapshot_stage: Literal["end_market_phase"] = "end_market_phase"
    observation_complete: bool
    execution_status: Literal["completed", "incomplete", "failed"]
    snapshot_valid: bool
    quote_valid: bool
    best_bid: float | None
    best_ask: float | None
    current_quote_source: Literal["book_end_market", "not_observed"]
    quote_as_of_step: int | None = Field(ge=0)
    history_reference_source: Literal["initial_condition", "last_trade", "completed_quote"] | None
    history_reference_price: float | None
    history_reference_as_of_step: int | None = Field(ge=0)
    history_reference_age_steps: int | None = Field(ge=0)
    last_trade_price: float | None
    last_trade_step: int | None = Field(ge=0)
    price_age_steps: int | None = Field(ge=0)
    reference_only: bool
    depth_levels: int = Field(ge=1)
    settled_trade_ids: list[str] | None
    settled_trade_hash: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    cancel_reason_counts: dict[str, int] | None
    reject_reason_counts: dict[str, int] | None
    price_band_status: Literal["WITHIN_BOUNDS", "VIOLATION", "NOT_CAPTURED", "NOT_OBSERVED"]
    policy_activity: dict[str, bool | int | float | str | None]
    origin: Origin
    metrics: dict[str, MetricValue]

    @model_validator(mode="after")
    def _record_semantics(self):
        if set(self.metrics) != set(METRIC_UNITS):
            raise ValueError("StepMeasurement must retain the full v2 metric family")
        if any(metric.unit != METRIC_UNITS[name] for name, metric in self.metrics.items()):
            raise ValueError("Metric units do not match the v2 definitions")
        complete = self.execution_status == "completed" and self.observation_complete
        if self.observation_complete and self.execution_status != "completed":
            raise ValueError("Failed/incomplete execution cannot claim a complete observation")
        if not complete:
            if self.snapshot_valid or self.quote_valid or any(m.status == "VALID" for m in self.metrics.values()):
                raise ValueError("Incomplete/failed observation cannot issue valid measurements")
        captured = self.current_quote_source == "book_end_market"
        if captured:
            if self.quote_as_of_step != self.step:
                raise ValueError("Current quote source must retain the current sampling step")
        elif self.quote_as_of_step is not None or self.best_bid is not None or self.best_ask is not None:
            raise ValueError("Unobserved quotes cannot carry current BBO data")
        bid, ask = self.best_bid, self.best_ask
        invalid_bbo = ((bid is not None and bid <= 0) or (ask is not None and ask <= 0)
                       or (bid is not None and ask is not None and ask < bid))
        if self.snapshot_valid and (not complete or not captured or invalid_bbo
                                   or self.price_band_status in ("VIOLATION", "NOT_OBSERVED")
                                   or self.metrics["spread"].status == "INVALID_INPUT"):
            raise ValueError("Snapshot validity conflicts with its observation or price bounds")
        expected_quote = (self.snapshot_valid and self.market_open and bid is not None and ask is not None
                          and not invalid_bbo)
        if self.quote_valid != expected_quote:
            raise ValueError("Quote validity must agree with the current complete legal two-sided BBO")
        quote_metrics = ("spread", "relative_spread", "mid_price")
        if any((self.metrics[name].status == "VALID") != self.quote_valid for name in quote_metrics):
            raise ValueError("Quote metrics cannot disagree with current quote validity")
        if self.quote_valid:
            mid = bid / 2.0 + ask / 2.0
            expected = {"spread": ask - bid, "mid_price": mid, "relative_spread": (ask - bid) / mid}
            if any(self.metrics[name].value != value for name, value in expected.items()):
                raise ValueError("Quote metrics must be computed from the current BBO")
        history = (self.history_reference_source, self.history_reference_price,
                   self.history_reference_as_of_step, self.history_reference_age_steps)
        if any(v is None for v in history) and not all(v is None for v in history):
            raise ValueError("Historical reference needs source, price, as-of and age together")
        if self.history_reference_source is not None:
            if (self.history_reference_price <= 0 or self.history_reference_as_of_step > self.step
                    or self.history_reference_age_steps != self.step - self.history_reference_as_of_step):
                raise ValueError("Historical reference has invalid price, time or age")
            if self.history_reference_source == "completed_quote" and self.history_reference_as_of_step >= self.step:
                raise ValueError("A completed quote history must precede the unfinished current step")
        last = (self.last_trade_price, self.last_trade_step, self.price_age_steps)
        if any(v is None for v in last) and not all(v is None for v in last):
            raise ValueError("Last true trade needs price, step and age together")
        if self.last_trade_step is not None and (
                self.last_trade_price <= 0 or self.last_trade_step > self.step
                or self.price_age_steps != self.step - self.last_trade_step):
            raise ValueError("Last true trade has invalid price, time or age")
        if self.reference_only != (self.last_trade_price is None and self.history_reference_source is not None):
            raise ValueError("reference_only must distinguish an initial/reference mark from a real trade")
        if (self.settled_trade_ids is None) != (self.settled_trade_hash is None):
            raise ValueError("Settled trade IDs and tape digest must be captured together")
        if self.settled_trade_ids is not None:
            if any(not i for i in self.settled_trade_ids) or len(self.settled_trade_ids) != len(set(self.settled_trade_ids)):
                raise ValueError("Settled trade IDs must be nonempty and unique")
            count = self.metrics["trade_count"]
            if count.status == "VALID" and count.value != len(self.settled_trade_ids):
                raise ValueError("Valid trade count must agree with the captured settlement IDs")
            if complete and self.settled_trade_ids and (not self.market_open or self.last_trade_step != self.step):
                raise ValueError("Current settled trades need an open market and current last-trade metadata")
            if complete and not self.settled_trade_ids and self.last_trade_step == self.step:
                raise ValueError("Captured empty tape cannot claim a new trade")
        elif any(self.metrics[name].status == "VALID" for name in ("trade_count", "traded_quantity", "vwap", "avg_trade_size")):
            raise ValueError("Missing settlement provenance cannot supply valid trade metrics")
        if self.price_band_status == "VIOLATION" and any(m.status != "INVALID_INPUT" for m in self.metrics.values()):
            raise ValueError("Price-bound violation must invalidate the complete measurement")
        for counts in (self.cancel_reason_counts, self.reject_reason_counts):
            if counts is not None and any(type(n) is not int or n < 0 for n in counts.values()):
                raise ValueError("Reason counts require nonnegative strict integers")
        return self


def _digest(value) -> str:
    """Phase 2 content identity under research-canonical-v1, not a Gate.

    Phase 1 diagnostic archives retain their original recorded JSON digests.
    Reproduction binds the implementation manifest, never rewrites old receipts.
    """
    from core.experiments.identity import research_hash
    return research_hash(value)


def measurement_spec_hash(spec: MeasurementSpec) -> str:
    return _digest(spec.model_dump(mode="json"))


def _metric(value, unit, *, status="VALID", reason=None, sample_count=1, denominator_count=1):
    return MetricValue(value=None if value is None else float(value), status=status, reason=reason,
                       unit=unit, sample_count=sample_count, denominator_count=denominator_count)


def _missing(name, status, reason, *, denominator_count=1):
    return _metric(None, METRIC_UNITS[name], status=status, reason=reason,
                   sample_count=0, denominator_count=denominator_count)


def _reason_counts(events):
    if events is None:
        return None
    result = {}
    for event in events:
        result[event.reason] = result.get(event.reason, 0) + 1
    return result


def compute_step_measurement(*, snapshot: MarketSnapshot,
                             trade_tape: Sequence[Trade | SettledTrade] | None,
                             activity: StepActivity,
                             context: MeasurementContext) -> StepMeasurement:
    """Measure one step without touching the book, accounts or event buffers."""
    if not isinstance(snapshot, MarketSnapshot) or not isinstance(activity, StepActivity) or not isinstance(context, MeasurementContext):
        raise TypeError("v2 measurement requires typed snapshot, activity and context")
    spec = context.spec
    unsupported = set(spec.required_metrics) - set(METRIC_UNITS)
    if unsupported or (spec.recovery_metric is not None and spec.recovery_metric not in METRIC_UNITS):
        raise ValueError(f"Unknown registered measurement: {unsupported or spec.recovery_metric}")
    if snapshot.depth_levels != spec.top_k:
        raise ValueError("Captured top-k must exactly match the registered MeasurementSpec")
    if snapshot.quote_as_of_step is not None and snapshot.quote_as_of_step != context.step:
        raise ValueError("Current BBO must be captured at this step; history uses its own field")
    for prior in (snapshot.last_trade_step,
                  None if snapshot.history_reference is None else snapshot.history_reference.as_of_step):
        if prior is not None and prior > context.step:
            raise ValueError("Measurement cannot reference future observations")
    if (snapshot.history_reference is not None and snapshot.history_reference.source == "completed_quote"
            and snapshot.history_reference.as_of_step >= context.step):
        raise ValueError("A completed quote history must precede the unfinished current step")
    trades = None
    if trade_tape is not None:
        if not isinstance(trade_tape, Sequence) or isinstance(trade_tape, (str, bytes)):
            raise TypeError("Settled trade tape must be an explicit finite sequence")
        trades = [t if isinstance(t, SettledTrade) else SettledTrade.from_clearing_trade(t) for t in trade_tape]
        ids = [t.trade_id for t in trades]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate settled trade IDs cannot increase volume or sample count")
        if activity.settled_trade_ids is None or ids != list(activity.settled_trade_ids):
            raise ValueError("Trade tape must exactly match the captured settlement return IDs and order")
    elif activity.settled_trade_ids is not None:
        raise ValueError("A settlement ID list without its trade tape is incomplete provenance")
    complete = context.execution_status == "completed" and context.observation_complete
    if complete and trades:
        if not context.market_open:
            raise ValueError("A completed halted step cannot have settled trades")
        if snapshot.last_trade_step != context.step or snapshot.last_trade_price != trades[-1].price:
            raise ValueError("Last real trade metadata must match the last settled trade in this step")
    elif complete and trades == [] and snapshot.last_trade_step == context.step:
        raise ValueError("A captured empty tape cannot claim a new trade at this step")

    reason = "RUN_FAILED" if context.execution_status == "failed" else "STEP_INCOMPLETE"
    metrics = {name: _missing(name, "NOT_OBSERVED", reason) for name in METRIC_UNITS}
    quote_valid = False
    invalid_quote = False
    observed = complete and snapshot.captured
    cancel_counts = _reason_counts(activity.cancel_events)
    reject_counts = _reason_counts(activity.reject_events)
    age = None if snapshot.last_trade_step is None else context.step - snapshot.last_trade_step
    ref = snapshot.history_reference
    ref_age = None if ref is None else context.step - ref.as_of_step
    band = "NOT_OBSERVED"
    if observed:
        bid, ask = snapshot.best_bid, snapshot.best_ask
        invalid_quote = ((bid is not None and bid <= 0) or (ask is not None and ask <= 0)
                         or (bid is not None and ask is not None and ask < bid))
        for levels, best in ((snapshot.bid_depth, bid), (snapshot.ask_depth, ask)):
            if levels is not None and ((levels and levels[0].price != best) or (not levels and best is not None)):
                invalid_quote = True
        quote_valid = context.market_open and not invalid_quote and bid is not None and ask is not None
        quote_reason = "INVALID_QUOTE" if invalid_quote else "MARKET_CLOSED" if not context.market_open else "NO_TWO_SIDED_QUOTE"
        for name in ("spread", "relative_spread", "mid_price"):
            metrics[name] = _missing(name, "INVALID_INPUT" if invalid_quote else "UNDEFINED", quote_reason)
        if quote_valid:
            mid = bid / 2.0 + ask / 2.0
            spread = ask - bid
            for name, val in (("spread", spread), ("relative_spread", spread / mid), ("mid_price", mid)):
                metrics[name] = _metric(val, METRIC_UNITS[name])
        for name, levels in (("depth_bid_top_k", snapshot.bid_depth), ("depth_ask_top_k", snapshot.ask_depth)):
            metrics[name] = (_missing(name, "NOT_OBSERVED", "CAPTURE_MISSING") if levels is None
                             else _metric(sum(row.quantity for row in levels), METRIC_UNITS[name]))
        db, da = metrics["depth_bid_top_k"], metrics["depth_ask_top_k"]
        if db.value is None or da.value is None:
            metrics["depth_imbalance"] = _missing("depth_imbalance", "NOT_OBSERVED", "CAPTURE_MISSING")
        elif db.value + da.value == 0:
            metrics["depth_imbalance"] = _missing("depth_imbalance", "UNDEFINED", "ZERO_DENOMINATOR", denominator_count=0)
        else:
            metrics["depth_imbalance"] = _metric((db.value - da.value) / (db.value + da.value), "ratio")
        band = "NOT_CAPTURED"
    elif complete:
        for name in ("spread", "relative_spread", "mid_price", "depth_bid_top_k", "depth_ask_top_k", "depth_imbalance"):
            metrics[name] = _missing(name, "NOT_OBSERVED", "CAPTURE_MISSING")

    if complete:
        if trades is None:
            for name in ("trade_count", "traded_quantity", "vwap", "avg_trade_size"):
                metrics[name] = _missing(name, "NOT_OBSERVED", "SETTLEMENT_TAPE_MISSING")
        else:
            count = len(trades)
            qty = sum(t.quantity for t in trades)
            metrics["trade_count"] = _metric(count, "trades")
            metrics["traded_quantity"] = _metric(qty, "shares")
            # Weighted components avoid overflowing an otherwise finite VWAP.
            metrics["vwap"] = (_metric(math.fsum(t.price * (t.quantity / qty) for t in trades), "price", sample_count=count, denominator_count=count)
                               if qty else _missing("vwap", "UNDEFINED", "NO_TRADES", denominator_count=0))
            metrics["avg_trade_size"] = (_metric(qty / count, "shares/trade", sample_count=count, denominator_count=count)
                                         if count else _missing("avg_trade_size", "UNDEFINED", "NO_TRADES", denominator_count=0))
        for name, counts in (("explicit_cancel_count", cancel_counts), ("reject_count", reject_counts)):
            metrics[name] = (_missing(name, "NOT_OBSERVED", "CAPTURE_MISSING") if counts is None
                             else _metric(sum(counts.values()), METRIC_UNITS[name]))
        cancels, count = metrics["explicit_cancel_count"].value, metrics["trade_count"].value
        if cancels is None or count is None:
            metrics["cancel_to_trade_ratio"] = _missing("cancel_to_trade_ratio", "NOT_OBSERVED", "CAPTURE_MISSING")
        elif count == 0:
            metrics["cancel_to_trade_ratio"] = _missing("cancel_to_trade_ratio", "UNDEFINED", "ZERO_DENOMINATOR", denominator_count=0)
        else:
            metrics["cancel_to_trade_ratio"] = _metric(cancels / count, "orders/trade", denominator_count=int(count))
        for name in ("ioc_remainder_count", "halt_suppressed_count"):
            value = getattr(activity, name)
            metrics[name] = (_missing(name, "NOT_OBSERVED", "CAPTURE_MISSING") if value is None
                             else _metric(value, METRIC_UNITS[name]))
        metrics["last_trade_price"] = (_missing("last_trade_price", "NOT_OBSERVED", "NO_TRADE_HISTORY")
                                       if snapshot.last_trade_price is None else _metric(snapshot.last_trade_price, "price"))
        metrics["execution_cost"] = _missing("execution_cost", "NOT_OBSERVED" if spec.arrival_cost_capture else "NOT_IMPLEMENTED",
                                              "ARRIVAL_QUOTES_NOT_CAPTURED" if spec.arrival_cost_capture else "ARRIVAL_COST_NOT_IMPLEMENTED")
        metrics["herding_proxy"] = _missing("herding_proxy", "NOT_IMPLEMENTED", "BEHAVIOR_FLOW_NOT_IMPLEMENTED")
        if spec.price_proxy == "current_mid":
            metrics["price_proxy"] = metrics["mid_price"]
        else:
            value, proxy_age = ((snapshot.last_trade_price, age) if spec.price_proxy == "last_trade"
                                else (None if ref is None else ref.price, ref_age))
            metrics["price_proxy"] = (_missing("price_proxy", "NOT_OBSERVED", "NO_PRICE_REFERENCE") if value is None
                                      else _missing("price_proxy", "UNDEFINED", "STALE_PRICE_REFERENCE") if proxy_age > spec.max_price_age_steps
                                      else _metric(value, "price"))

    if complete and snapshot.lower_price_limit is not None:
        prices = [v for v in (snapshot.best_bid, snapshot.best_ask) if v is not None]
        prices += [row.price for levels in (snapshot.bid_depth, snapshot.ask_depth) if levels is not None for row in levels]
        prices += [t.price for t in (trades or [])]
        band = "WITHIN_BOUNDS" if all(snapshot.lower_price_limit <= p <= snapshot.upper_price_limit for p in prices) else "VIOLATION"
    if band == "VIOLATION":
        # Retain raw provenance, but certify no component of an illegal market
        # observation. The run aggregator will exclude this original-clock row.
        quote_valid, invalid_quote = False, True
        metrics = {name: _missing(name, "INVALID_INPUT", "PRICE_BOUND_VIOLATION") for name in METRIC_UNITS}

    return StepMeasurement(
        measurement_hash=measurement_spec_hash(spec), world_id=context.world_id, run_id=context.run_id,
        step=context.step, session_id=context.session_id, policy_hash=context.policy_hash,
        market_open=context.market_open, observation_complete=context.observation_complete,
        execution_status=context.execution_status, snapshot_valid=observed and not invalid_quote,
        quote_valid=quote_valid, best_bid=snapshot.best_bid, best_ask=snapshot.best_ask,
        current_quote_source=snapshot.current_quote_source, quote_as_of_step=snapshot.quote_as_of_step,
        history_reference_source=None if ref is None else ref.source,
        history_reference_price=None if ref is None else ref.price,
        history_reference_as_of_step=None if ref is None else ref.as_of_step,
        history_reference_age_steps=ref_age, last_trade_price=snapshot.last_trade_price,
        last_trade_step=snapshot.last_trade_step, price_age_steps=age,
        reference_only=snapshot.last_trade_price is None and ref is not None,
        depth_levels=spec.top_k, settled_trade_ids=None if trades is None else [t.trade_id for t in trades],
        settled_trade_hash=None if trades is None else _digest([t.model_dump(mode="json") for t in trades]),
        cancel_reason_counts=cancel_counts, reject_reason_counts=reject_counts,
        price_band_status=band, policy_activity=context.policy_activity, origin=context.origin, metrics=metrics,
    )


class TimedReturn(StrictResearchModel):
    previous_step: int = Field(ge=0)
    step: int = Field(ge=1)
    metric: MetricValue

    @model_validator(mode="after")
    def _adjacent(self):
        if self.step != self.previous_step + 1 or self.metric.unit != "ratio":
            raise ValueError("Returns must retain adjacent original clock steps and ratio units")
        return self


class RecoveryObservation(StrictResearchModel):
    status: Literal["NOT_REQUESTED", "RECOVERED", "RIGHT_CENSORED", "INCOMPLETE_OBSERVATION"]
    start_step: int | None = Field(ge=0)
    event_step: int | None = Field(ge=0)
    confirmed_at_step: int | None = Field(ge=0)
    elapsed_steps: int | None = Field(ge=0)
    censor_step: int | None = Field(ge=0)
    observed_followup_steps: int | None = Field(ge=0)

    @model_validator(mode="after")
    def _censoring(self):
        fields = (self.start_step, self.event_step, self.confirmed_at_step,
                  self.elapsed_steps, self.censor_step, self.observed_followup_steps)
        if self.status == "NOT_REQUESTED":
            if any(value is not None for value in fields):
                raise ValueError("Unrequested recovery cannot contain an event or follow-up")
        elif self.start_step is None or self.observed_followup_steps is None:
            raise ValueError("Requested recovery needs start and observed follow-up")
        elif self.status == "RECOVERED":
            if (any(value is None for value in (self.event_step, self.confirmed_at_step, self.elapsed_steps))
                    or self.censor_step is not None
                    or not self.start_step <= self.event_step <= self.confirmed_at_step
                    or self.elapsed_steps != self.event_step - self.start_step
                    or self.observed_followup_steps != self.confirmed_at_step - self.start_step + 1):
                raise ValueError("Recovered event must retain onset, confirmation and elapsed time")
        elif (any(value is not None for value in (self.event_step, self.confirmed_at_step, self.elapsed_steps))
              or self.censor_step is None or self.censor_step < self.start_step
              or self.observed_followup_steps > self.censor_step - self.start_step + 1):
            raise ValueError("Nonrecovery cannot become an exact zero or horizon event time")
        elif self.status == "RIGHT_CENSORED" and self.observed_followup_steps != self.censor_step - self.start_step + 1:
            raise ValueError("Right censoring requires complete follow-up through the censor step")
        return self


class RunMetrics(StrictResearchModel):
    schema_version: Literal[2] = 2
    measurement_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_ref: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Origin
    clock_steps: list[int]
    missing_steps: list[int]
    invalid_steps: list[int]
    unavailable_required_steps: list[int]
    status: Literal["COMPLETE", "INCOMPLETE"]
    required_metric_valid_counts: dict[str, int]
    metrics: dict[str, MetricValue]
    returns: list[TimedReturn]
    recovery: RecoveryObservation

    @model_validator(mode="after")
    def _run_semantics(self):
        if (not self.clock_steps or self.clock_steps[0] < 0
                or self.clock_steps != list(range(self.clock_steps[0], self.clock_steps[-1] + 1))):
            raise ValueError("Run metrics must retain a nonempty contiguous evaluation clock")
        for steps in (self.missing_steps, self.invalid_steps, self.unavailable_required_steps):
            if steps != sorted(set(steps)) or not set(steps) <= set(self.clock_steps):
                raise ValueError("Observation status steps must be unique ordered members of the clock")
        if set(self.missing_steps) & (set(self.invalid_steps) | set(self.unavailable_required_steps)):
            raise ValueError("Absent observations cannot also be captured invalid/required-missing records")
        expected = "INCOMPLETE" if self.missing_steps or self.invalid_steps or self.unavailable_required_steps else "COMPLETE"
        if self.status != expected:
            raise ValueError("Run completeness must agree with observation and required capture status")
        if any(n < 0 or n > len(self.clock_steps) for n in self.required_metric_valid_counts.values()):
            raise ValueError("Required metric counts must respect the original clock denominator")
        if [(r.previous_step, r.step) for r in self.returns] != list(zip(self.clock_steps[:-1], self.clock_steps[1:])):
            raise ValueError("Run return records must retain every original adjacent clock pair")
        return self


def aggregate_run_metrics(measurements: Sequence[StepMeasurement], *, spec: MeasurementSpec,
                          run_ref: str) -> RunMetrics:
    """Aggregate on the original clock, preserving conditional denominators.

    Conditional means describe observed components. ``status``/coverage and
    required_metric_valid_counts expose incomplete observations; they do not
    grant policy eligibility. MDD requires a complete valid price window.
    """
    if not measurements or any(not isinstance(row, StepMeasurement) for row in measurements):
        raise ValueError("Aggregation requires typed v2 observations and at least one run reference row")
    ordered = [row.step for row in measurements]
    if ordered != sorted(set(ordered)):
        raise ValueError("Measurements require unique increasing original clock steps")
    first = measurements[0]
    digest = measurement_spec_hash(spec)
    for row in measurements:
        if (row.world_id, row.run_id, row.policy_hash, row.origin, row.measurement_hash) != (
                first.world_id, first.run_id, first.policy_hash, first.origin, digest):
            raise ValueError("Cannot aggregate mixed run, policy, world, origin or measurement scopes")
    if set(spec.required_metrics) - set(METRIC_UNITS) or (spec.recovery_metric is not None and spec.recovery_metric not in METRIC_UNITS):
        raise ValueError("Unknown registered measurement")
    clock = list(range(spec.evaluation_start, spec.evaluation_end))
    by_step = {row.step: row for row in measurements}
    rows = [by_step[t] for t in clock if t in by_step]
    complete = [row for row in rows if row.observation_complete and row.execution_status == "completed" and row.snapshot_valid]
    open_rows = [row for row in complete if row.market_open]
    trade_rows = [row for row in complete if row.metrics["trade_count"].status == "VALID"]
    open_trade_rows = [row for row in trade_rows if row.market_open]
    missing = [t for t in clock if t not in by_step]
    invalid = [row.step for row in rows if row not in complete]
    unavailable_required = [row.step for row in rows if any(
        row.metrics[name].status in ("NOT_OBSERVED", "INVALID_INPUT", "NOT_IMPLEMENTED")
        for name in spec.required_metrics)]
    output = {}

    def ratio(name, numerator, denominator):
        output[name] = (_metric(numerator / denominator, "ratio", sample_count=numerator, denominator_count=denominator)
                        if denominator else _metric(None, "ratio", status="UNDEFINED", reason="ZERO_DENOMINATOR", sample_count=0, denominator_count=0))

    ratio("observation_coverage", len(complete), len(clock))
    ratio("step_end_quote_coverage", sum(row.quote_valid for row in complete), len(complete))
    ratio("open_step_end_quote_coverage", sum(row.quote_valid for row in open_rows), len(open_rows))
    ratio("halt_fraction", sum(not row.market_open for row in complete), len(complete))
    ratio("zero_trade_rate", sum(row.metrics["trade_count"].value == 0 for row in trade_rows), len(trade_rows))
    ratio("open_zero_trade_rate", sum(row.metrics["trade_count"].value == 0 for row in open_trade_rows), len(open_trade_rows))
    ratio("trade_capture_coverage", len(trade_rows), len(clock))
    ratio("required_capture_coverage", len(clock) - len(set(missing + invalid + unavailable_required)), len(clock))
    for name, unit in METRIC_UNITS.items():
        values = [row.metrics[name].value for row in complete if row.metrics[name].status == "VALID"]
        output[f"mean_{name}"] = (_metric(math.fsum(v / len(values) for v in values), unit, sample_count=len(values), denominator_count=len(clock))
                                     if values else _metric(None, unit, status="NOT_OBSERVED", reason="NO_VALID_OBSERVATIONS", sample_count=0, denominator_count=len(clock)))
    returns = []
    for t in clock[1:]:
        previous, current = by_step.get(t - 1), by_step.get(t)
        if (previous is not None and current is not None and previous in complete and current in complete
                and previous.metrics["price_proxy"].status == current.metrics["price_proxy"].status == "VALID"):
            value = current.metrics["price_proxy"].value / previous.metrics["price_proxy"].value - 1.0
            metric = (_metric(value, "ratio") if math.isfinite(value) else
                      _metric(None, "ratio", status="INVALID_INPUT", reason="NUMERIC_OVERFLOW", sample_count=0, denominator_count=1))
        else:
            metric = _metric(None, "ratio", status="NOT_OBSERVED", reason="NON_ADJACENT_VALID_PRICES", sample_count=0, denominator_count=1)
        returns.append(TimedReturn(previous_step=t - 1, step=t, metric=metric))
    valid_returns = [row.metric.value for row in returns if row.metric.status == "VALID"]
    output["adjacent_price_return_std"] = (
        _metric(statistics.stdev(valid_returns), "ratio", sample_count=len(valid_returns), denominator_count=len(returns))
        if len(valid_returns) >= 2 else _metric(None, "ratio", status="UNDEFINED", reason="INSUFFICIENT_ADJACENT_RETURNS", sample_count=len(valid_returns), denominator_count=len(returns)))
    prices = [row.metrics["price_proxy"].value for row in complete if row.metrics["price_proxy"].status == "VALID"]
    if len(prices) == len(clock):
        peak, drawdown = prices[0], 0.0
        for price in prices:
            peak = max(peak, price)
            drawdown = max(drawdown, 1.0 - price / peak)
        output["max_drawdown"] = _metric(drawdown, "ratio", sample_count=len(prices), denominator_count=len(clock))
    else:
        output["max_drawdown"] = _metric(None, "ratio", status="NOT_OBSERVED", reason="INCOMPLETE_PRICE_WINDOW", sample_count=len(prices), denominator_count=len(clock))

    recovery = RecoveryObservation(status="NOT_REQUESTED", start_step=None, event_step=None,
                                   confirmed_at_step=None, elapsed_steps=None, censor_step=None, observed_followup_steps=None)
    if spec.recovery_predicate != "none":
        start = spec.recovery_start_step
        consecutive = 0
        event = confirmation = None
        unknown = False
        observed_followup = 0
        for t in range(start, spec.evaluation_end):
            row = by_step.get(t)
            value = None if row is None or row not in complete else row.metrics[spec.recovery_metric].value
            if value is None:
                unknown, consecutive = True, 0
                continue
            observed_followup += 1
            passes = value <= spec.recovery_threshold if spec.recovery_predicate == "metric_le" else value >= spec.recovery_threshold
            consecutive = consecutive + 1 if passes else 0
            if consecutive >= spec.recovery_k:
                event, confirmation = t - spec.recovery_k + 1, t
                break
        if unknown:
            status = "INCOMPLETE_OBSERVATION"
            event = confirmation = None
        else:
            status = "RECOVERED" if event is not None else "RIGHT_CENSORED"
        recovery = RecoveryObservation(status=status, start_step=start, event_step=event,
            confirmed_at_step=confirmation, elapsed_steps=None if event is None else event - start,
            censor_step=spec.evaluation_end - 1 if event is None else None, observed_followup_steps=observed_followup)
    counts = {name: sum(row.metrics[name].status == "VALID" for row in complete) for name in spec.required_metrics}
    return RunMetrics(measurement_hash=digest, run_ref=run_ref, world_id=first.world_id,
        policy_hash=first.policy_hash, origin=first.origin, clock_steps=clock, missing_steps=missing,
        invalid_steps=invalid, unavailable_required_steps=unavailable_required,
        status="INCOMPLETE" if missing or invalid or unavailable_required else "COMPLETE",
        required_metric_valid_counts=counts, metrics=output, returns=returns, recovery=recovery)
