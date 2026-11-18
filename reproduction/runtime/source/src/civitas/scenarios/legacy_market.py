"""Compatibility orchestration; accounting stays inside the original ClearingHouse."""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from core.experiments.config import ExperimentConfig
from core.experiments.design_config import MeasurementSpec
from core.experiments.measurement import DepthLevel, MarketSnapshot, MeasurementContext, PriceReference
from core.experiments.runner import ExperimentRunner
from core.market_metrics import MarketMetrics
from core.society.network import SentimentState, capture_social_state
from civitas.analytics.market_v2 import compute_market_v2
from civitas.contracts.base import content_hash
from civitas.contracts.decisions import HoldIntent
from civitas.contracts.events import Event
from civitas.contracts.financial import FinancialDecisionBundle, LegacyOrderBatch, OrderIntent
from civitas.contracts.legacy_config import LegacyRunConfig
from civitas.contracts.observations import ObservationFact
from civitas.contracts.state import BatchExecutionResult, ExecutionResult, Feedback
from civitas.contracts.time import SimTime
from civitas.markets.legacy import LegacySettlementAdapter, json_values
from .legacy_state import LegacySocietyParticipant


class LegacyMarket:
    scenario_id = "legacy_market"
    version = "legacy-vectorized-v1"
    owned_fields = frozenset({"initial", "frame", "steps", "session_boundaries", "event_effects"})

    def __init__(self, config: LegacyRunConfig):
        self.config = config
        self.runner = ExperimentRunner(ExperimentConfig.model_validate(config.experiment.model_dump()))
        with TemporaryDirectory(prefix="civitas-legacy-init-") as directory:
            self.initial = json_values(self.runner._initialize(Path(directory)))
        self.adapter = LegacySettlementAdapter(self.runner.clearing)
        self.runner.order_steps = self.adapter.order_steps
        self.society = LegacySocietyParticipant(self.runner)
        self.participants = (self.adapter, self.society)
        self.events = [Event(event_id=f"legacy-event:{i:08d}", root_event_id=f"legacy-event:{i:08d}",
                       tick=e.step, phase="EVENTS", kind=e.kind, actors=self.runner.ids,
                       visible_at=SimTime(tick=e.step, phase="OBSERVE"), payload={"value": e.value})
                       for i, e in enumerate(config.experiment.events)]

    def initialize(self):
        return {"initial": self.initial, "frame": self._frame(0), "steps": [], "session_boundaries": [], "event_effects": []}

    def _frame(self, version):
        return {"price": float(self.runner.price), "halted": self.runner.halted, "proposals": {},
                "accounts": {aid: self.adapter.account_view(aid, version=version).model_dump() for aid in self.runner.ids}}

    def boundary(self, context):
        r, cfg = self.runner, self.runner.config
        r.clearing.ensure_usable()
        self.adapter.begin_tick(context.tick, context.world_version)
        data = context.model_dump()["data"]
        if context.tick > 1 and (context.tick - 1) % cfg.market.steps_per_day == 0:
            data["session_boundaries"].append(json_values(self.adapter.apply_session_boundary(r.price).to_dict()))
        self.adapter.expire(context.tick, cfg.market.order_ttl)
        return data

    def apply_event(self, context, event):
        r, value = self.runner, event.payload["value"]
        affected = []
        if event.kind == "panic":
            affected = r.diffusion.inject_panic(int(value))
        elif event.kind == "sentiment":
            r.population.state[:, 3] = np.clip(r.population.state[:, 3] + value, -1, 1)
        elif event.kind in {"halt", "resume"}:
            r.halted = event.kind == "halt"
            if r.halted:
                for order in r.clearing.active_orders():
                    self.adapter.cancel(order.order_id, reason="halt")
        elif event.kind == "tax_change":
            r.book.market_rules = replace(r.book.market_rules, stamp_duty_rate=value)
        else:
            raise ValueError("Unknown legacy event")
        data = context.model_dump()["data"]
        data["event_effects"].append({"step": context.tick, "kind": event.kind, "value": value, "affected_nodes": affected})
        return data

    async def prepare_observations(self, context):
        r, cfg = self.runner, self.runner.config
        for i, aid in enumerate(r.ids):
            r.social.update_holdings(i, {cfg.market.symbol: r.clearing.accounts[aid].shares})
        social_stats = r.diffusion.update_sentiment_propagation()
        signals = [-1 if r.social.agents[i].sentiment_state == SentimentState.INFECTED else 0 for i in range(cfg.n_agents)]
        r.population.update_tier2_sentiment(signals, (r.price - r.previous_price) / r.previous_price)
        actions, quantities, prices = await r._decisions(context.tick)
        if (any(np.shape(v) != (cfg.n_agents,) for v in (actions, quantities, prices))
                or not np.isin(actions, [-1, 0, 1]).all() or not np.isfinite(prices).all()
                or not np.isfinite(quantities).all() or (quantities < 0).any()
                or (quantities != np.floor(quantities)).any() or (prices <= 0).any()):
            raise ValueError("Decision batch violates agent-market contract")
        frame = self._frame(context.world_version)
        frame["social"] = social_stats
        frame["trade_start"] = len(r.book.trades_history)
        frame["proposals"] = {aid: {"step": context.tick, "agent_id": aid, "action": int(action),
            "quantity": int(qty), "price": float(price), "source": cfg.decision_mode,
            "order_id": f"s{context.tick:08d}-a{i:06d}", "symbol": cfg.market.symbol, "halted": r.halted}
            for i, (aid, action, qty, price) in enumerate(zip(r.ids, actions, quantities, prices, strict=True))}
        data = context.model_dump()["data"]
        data["frame"] = frame
        return data

    def project_observation(self, context, actor_id):
        frame = context.data["frame"]
        now = SimTime(tick=context.tick, phase="OBSERVE")
        facts = [ObservationFact(fact_id=f"account:{actor_id}", category="account", owner_id=actor_id,
                                 visible_at=now, value=frame["accounts"][actor_id]),
                 ObservationFact(fact_id=f"price:{context.tick}", category="market", public=True,
                                 visible_at=now, value={"price": frame["price"], "halted": frame["halted"]})]
        if actor_id in frame["proposals"]:
            facts.append(ObservationFact(fact_id=f"proposal:{context.tick}:{actor_id}", category="memory",
                owner_id=actor_id, visible_at=now, value=frame["proposals"][actor_id]))
        return facts

    def execute_batch(self, context, commands):
        receipts = self.adapter.submit_orders(LegacyOrderBatch(commands=commands, execution_seq_start=context.execution_seq)) if commands else []
        return BatchExecutionResult(data=context.data, receipts=receipts)

    def execute(self, context, command):
        result = self.execute_batch(context, [command])
        return ExecutionResult(data=result.data, receipt=result.receipts[0])

    def feedback(self, context, receipts):
        r, step = self.runner, context.tick
        r.previous_price, r.price = r.price, r.book.last_price
        r.clearing.check_invariants()
        # Preserve the legacy end-step lazy heap cleanup, outside observation reads.
        quote = {"last_price": r.price, "best_bid": r.book.get_best_bid(), "best_ask": r.book.get_best_ask(), "depth": r.book.get_depth(5)}
        trades = r.book.trades_history[context.data["frame"]["trade_start"]:]
        tape = [asdict(t) for t in trades]
        metrics_v1 = {"step": step, "price": r.price, "return": r.price / r.previous_price - 1,
            "volume": sum(t.quantity for t in trades), "halted": r.halted, "fees_cumulative": r.clearing.fees,
            "infected": context.data["frame"]["social"]["infected"], **MarketMetrics.compute(snapshot=quote, trade_tape=tape)}
        last = r.book.trades_history[-1] if r.book.trades_history else None
        snapshot = MarketSnapshot(captured=True, best_bid=quote["best_bid"], best_ask=quote["best_ask"],
            bid_depth=[DepthLevel(price=row["price"], quantity=row["qty"]) for row in quote["depth"]["bids"]],
            ask_depth=[DepthLevel(price=row["price"], quantity=row["qty"]) for row in quote["depth"]["asks"]],
            depth_levels=5, current_quote_source="book_end_market", quote_as_of_step=step,
            last_trade_price=last.price if last else None, last_trade_step=int(last.timestamp) if last else None,
            history_reference=PriceReference(price=last.price if last else self.initial["world"]["initial_price"],
                source="last_trade" if last else "initial_condition", as_of_step=int(last.timestamp) if last else 0),
            lower_price_limit=r.book.market_rules.get_limit_prices()[0], upper_price_limit=r.book.market_rules.get_limit_prices()[1])
        measurement_context = MeasurementContext(spec=MeasurementSpec(evaluation_end=self.config.ticks + 1),
            world_id=content_hash(self.initial), run_id=content_hash(self.config), step=step,
            session_id=str((step - 1) // r.config.market.steps_per_day), policy_hash=content_hash({"mechanism": self.version}),
            market_open=not r.halted, observation_complete=True, execution_status="completed", origin="market")
        proposals = context.data["frame"]["proposals"]
        metrics_v2 = compute_market_v2(snapshot=snapshot, trade_tape=trades, order_events=self.adapter.order_events,
            context=measurement_context, halt_suppressed_count=sum(p["action"] != 0 and p["quantity"] > 0 and p["halted"] for p in proposals.values()))
        data = context.model_dump()["data"]
        data["steps"].append({"step": step, "trades": tape, "agents": list(r._agent_rows(step)),
            "social": data["frame"]["social"], "social_state_hash": content_hash(capture_social_state(r.diffusion)),
            "decisions": [{key: p[key] for key in ("step", "agent_id", "action", "quantity", "price", "source")} for p in proposals.values()],
            "metrics_v1": metrics_v1, "metrics_v2": metrics_v2.model_dump()})
        data["frame"] = self._frame(context.world_version + 1)
        return Feedback(data=data)

    def validate(self, context):
        self.adapter.validate_state()
        self.society.validate_state()

    def metrics(self, context):
        return {"metrics_v1": [s["metrics_v1"] for s in context.data["steps"]],
                "metrics_v2": [s["metrics_v2"] for s in context.data["steps"]]}


class LegacyPolicy:
    policy_id = "legacy_vectorized"
    version = "1"

    async def decide(self, observation):
        proposal = next(f.value for f in observation.private_memory_refs if f.fact_id.startswith("proposal:"))
        tick, actor = observation.visible_until.tick, observation.actor_id
        intent_id = f"intent:{tick}:{actor}"
        if proposal["action"] == 0 or proposal["quantity"] <= 0 or proposal["halted"]:
            intent = HoldIntent(intent_id=intent_id)
        else:
            intent = OrderIntent(intent_id=intent_id, order_id=proposal["order_id"], symbol=proposal["symbol"],
                price=proposal["price"], quantity=proposal["quantity"], side="buy" if proposal["action"] == 1 else "sell")
        return FinancialDecisionBundle(tick=tick, decision_id=f"decision:{tick}:{actor}", actor_id=actor,
            observation_id=observation.observation_id, batch_id=observation.batch_id,
            observation_version=observation.observation_version, world_version=observation.world_version,
            policy_id=self.policy_id, policy_version=self.version, economic_intents=[intent])
