"""Reuse research measurements, adding event totals and explicit legacy aliases."""
import math
from typing import Literal

from pydantic import Field
from core.market_metrics import MarketMetrics
from core.experiments.measurement import (CancellationObservation, RejectionObservation, StepActivity, StepMeasurement)
from civitas.contracts.base import NonNegative, Record
from civitas.contracts.financial import OrderEvent


class MarketMetricsV2(Record):
    schema_version: Literal[2] = 2
    definition_version: Literal["legacy-market-v2"] = "legacy-market-v2"
    step: NonNegative
    has_two_sided_quote: bool
    missing_side: Literal["bid", "ask", "both"] | None
    spread: float | None
    spread_pct: float | None
    slippage_bps: float | None
    slippage_definition: Literal["absolute VWAP distance from step-end midpoint; descriptive, not arrival cost"] = "absolute VWAP distance from step-end midpoint; descriptive, not arrival cost"
    cancel_count_step: NonNegative
    cancel_count_cumulative: NonNegative
    reject_count_step: NonNegative
    cancel_to_trade_ratio: float | None
    herding_proxy: float | None = Field(ge=0, le=1)
    quality_reason: dict[str, str]
    measurement: StepMeasurement


def compute_market_v2(*, snapshot, trade_tape, order_events, context, role_flows=None, halt_suppressed_count=0):
    events = {}
    for raw in order_events:
        event = OrderEvent.model_validate(raw)
        if event.step > context.step:
            raise ValueError("Future order event in current metric window")
        if event.event_id in events and events[event.event_id] != event:
            raise ValueError("Conflicting duplicate order event")
        events[event.event_id] = event
    current = [event for event in events.values() if event.step == context.step]
    cancelled = [event for event in current if event.event == "cancel"]
    rejected = [event for event in current if event.event == "rejected"]
    activity = StepActivity(settled_trade_ids=[t.trade_id for t in trade_tape],
        cancel_events=[CancellationObservation(order_id=e.order_id, reason=e.reason) for e in cancelled],
        reject_events=[RejectionObservation(attempt_id=e.event_id, reason=e.reason or "market_rule_rejection") for e in rejected],
        ioc_remainder_count=sum(e.event == "cancelled" and e.order_type == "ioc" for e in current),
        halt_suppressed_count=halt_suppressed_count)
    measured = MarketMetrics.compute_v2(snapshot=snapshot, trade_tape=trade_tape, activity=activity, context=context)
    value = lambda name: measured.metrics[name].value
    mid, vwap = value("mid_price"), value("vwap")
    reasons = {name: measured.metrics[name].reason for name in ("spread", "relative_spread", "vwap", "execution_cost")
               if measured.metrics[name].reason is not None}
    if mid is None or vwap is None:
        reasons["slippage_bps"] = measured.metrics["mid_price" if mid is None else "vwap"].reason
    herding = None
    if role_flows is None:
        reasons["herding_proxy"] = "ROLE_FLOWS_NOT_CAPTURED"
    else:
        if (not isinstance(role_flows, dict) or any(type(k) is not str or type(v) is not float or not math.isfinite(v)
                                                  for k, v in role_flows.items())):
            raise ValueError("Role flows require finite signed quantities by role")
        gross = sum(abs(v) for v in role_flows.values())
        if gross:
            herding = abs(sum(role_flows.values())) / gross
        else:
            reasons["herding_proxy"] = "ZERO_GROSS_ROLE_FLOW"
    missing = ("both" if snapshot.best_bid is None and snapshot.best_ask is None else
               "bid" if snapshot.best_bid is None else "ask" if snapshot.best_ask is None else None)
    return MarketMetricsV2(step=context.step, has_two_sided_quote=snapshot.best_bid is not None and snapshot.best_ask is not None,
        missing_side=missing, spread=value("spread"), spread_pct=value("relative_spread"),
        slippage_bps=abs(vwap - mid) / mid * 10000.0 if mid is not None and vwap is not None else None,
        cancel_count_step=len(cancelled), cancel_count_cumulative=sum(e.event == "cancel" for e in events.values()),
        reject_count_step=len(rejected), cancel_to_trade_ratio=value("cancel_to_trade_ratio"),
        herding_proxy=herding, quality_reason=reasons, measurement=measured)
