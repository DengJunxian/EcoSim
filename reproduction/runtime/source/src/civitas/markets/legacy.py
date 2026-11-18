"""The original ClearingHouse owns every reservation, match, fee and settlement."""
from dataclasses import asdict
import json

from core.exchange.clearing import (Account, CancellationReceipt, ClearingHouse,
                                    FailureDiagnostic, InvariantReceipt)
from core.exchange.market_rules import MarketRuleSchema, SessionRule
from core.exchange.order_book import OrderBook
from core.types import Order, OrderSide, OrderStatus, OrderType, Trade

from civitas.contracts.financial import (CancelIntent, FinancialExecutionReceipt, LegacyAccountView,
                                        LegacyOrderBatch, LegacySettlementState, OrderEvent, OrderIntent)
from civitas.contracts.time import SimTime
from civitas.contracts.base import content_hash


def json_values(value):
    """Legacy dataclass enums/tuples have declared JSON equivalents; reject NaN."""
    return json.loads(json.dumps(value, allow_nan=False))


class LegacySettlementAdapter:
    participant_id = "legacy_settlement"
    state_schema_version = 1
    _clear_scalars = ("t_plus_one", "fees", "initial_cash", "initial_shares", "_expected_fees",
                      "_expected_last_price", "_settled_volume")
    _clear_maps = ("_order_specs", "_order_fills", "_order_status", "_arrival_sequences", "_settled_trades")
    _book_scalars = ("symbol", "prev_close", "last_price", "total_volume", "submitted_orders",
                     "cancelled_orders", "trade_count")

    def __init__(self, clearing: ClearingHouse):
        if type(clearing.book) is not OrderBook:
            raise ValueError("Legacy transactional adapter supports the Python book only; C++ restore is unavailable")
        self.clearing = clearing
        self.tick = self.world_version = 0
        self.order_steps = {}
        self.order_decisions = {}
        self.receipts = {}
        self.order_events: list[OrderEvent] = []
        self.diagnostics = []  # Control-plane failures survive import_state/rollback.
        self.validate_state()

    def begin_tick(self, tick: int, world_version: int) -> None:
        SimTime(tick=tick, phase="BOUNDARY")
        SimTime(tick=world_version, phase="BOUNDARY")
        if tick != self.tick + 1 or world_version != tick - 1:
            raise ValueError("Legacy clock must advance once per committed tick")
        self.tick, self.world_version = tick, world_version

    def _event(self, order: Order, event: str, reason: str) -> None:
        self.order_events.append(OrderEvent(event_id=f"order-event:{len(self.order_events)}", step=self.tick,
            order_id=order.order_id, decision_id=self.order_decisions[order.order_id], event=event,
            reason=reason, remaining_quantity=order.remaining_qty, order_type=order.order_type.value))

    def submit_orders(self, batch: LegacyOrderBatch) -> list[FinancialExecutionReceipt]:
        batch = LegacyOrderBatch.model_validate(batch)
        order_ids = set(self.clearing.orders)
        for command in batch.commands:
            if command.command_id in self.receipts:
                raise ValueError("Duplicate command receipt")
            if (command.tick, command.batch_id, command.observation_version, command.world_version) != (
                    self.tick, f"tick:{self.tick}:base:{self.world_version}", self.world_version, self.world_version):
                raise ValueError("Stale observation batch")
            if isinstance(command.intent, OrderIntent):
                if command.intent.order_id in order_ids:
                    raise ValueError("Duplicate order ID")
                order_ids.add(command.intent.order_id)
        result = []
        for offset, command in enumerate(batch.commands, 1):
            intent, seq = command.intent, batch.execution_seq_start + offset
            if isinstance(intent, CancelIntent):
                receipt = self.cancel(intent.order_id, actor_id=command.actor_id, request_id=command.command_id,
                                      decision_id=command.decision_id, execution_seq=seq)
            else:
                trades, orders, status, reasons = [], [], "accepted", []
                if isinstance(intent, OrderIntent):
                    order = Order(intent.symbol, intent.price, intent.quantity, OrderSide(intent.side),
                                  OrderType(intent.order_type), command.actor_id, float(command.tick), order_id=intent.order_id)
                    self.order_steps[order.order_id] = self.tick
                    self.order_decisions[order.order_id] = command.decision_id
                    try:
                        trades = self.clearing.submit(order)
                    except BaseException:
                        self.diagnostics.append(self.clearing.diagnostic_snapshot())
                        raise
                    orders = [order.order_id]
                    self._event(order, order.status.value, order.reason)
                    status = ("rejected" if order.status == OrderStatus.REJECTED else
                              "partially_executed" if 0 < order.filled_qty < order.quantity else
                              "cancelled" if order.status == OrderStatus.CANCELLED else "accepted")
                    reasons = [order.reason or "market_rule_rejection"] if status == "rejected" else ([order.reason] if order.reason else [])
                receipt = FinancialExecutionReceipt(command_id=command.command_id, decision_id=command.decision_id,
                    actor_id=command.actor_id, execution_seq=seq, status=status, reason_codes=reasons,
                    executed_quantity=sum(t.quantity for t in trades), order_ids=orders,
                    trade_ids=[t.trade_id for t in trades], ledger_tx_ids=[t.trade_id for t in trades])
                self.receipts[receipt.command_id] = receipt
            result.append(receipt)
        return result

    def cancel(self, order_id: str, *, reason="manual", actor_id=None, request_id=None,
               decision_id=None, execution_seq=0) -> FinancialExecutionReceipt:
        order = self.clearing.orders.get(order_id)
        request_id = request_id or f"cancel:{self.tick}:{len(self.receipts)}:{order_id}"
        if request_id in self.receipts:
            raise ValueError("Duplicate cancellation receipt")
        actor = actor_id or (order.agent_id if order else "system")
        decision = decision_id or self.order_decisions.get(order_id, request_id)
        # Validate the receipt envelope before a successful cancellation mutates the book.
        FinancialExecutionReceipt(command_id=request_id, decision_id=decision, actor_id=actor,
            execution_seq=execution_seq, status="cancelled", reason_codes=[reason], executed_quantity=0,
            order_ids=[order_id])
        allowed = order is not None and (actor_id is None or actor_id == order.agent_id)
        cancelled = allowed and self.clearing.cancel(order_id, reason=reason)
        if cancelled:
            self._event(order, "cancel", reason)
        receipt = FinancialExecutionReceipt(command_id=request_id, decision_id=decision, actor_id=actor,
            execution_seq=execution_seq, status="cancelled" if cancelled else "rejected",
            reason_codes=[reason if cancelled else "not_active_or_not_owner"], executed_quantity=0,
            order_ids=[order_id])
        self.receipts[request_id] = receipt
        return receipt

    def expire(self, tick: int, ttl: int) -> list[FinancialExecutionReceipt]:
        if type(tick) is not int or tick != self.tick or type(ttl) is not int or ttl < 1:
            raise ValueError("Expiry requires current tick and positive integer TTL")
        return [self.cancel(order.order_id, reason="time_to_live") for order in self.clearing.active_orders()
                if tick - self.order_steps[order.order_id] >= ttl]

    def apply_session_boundary(self, price: float):
        receipt = self.clearing.apply_session_boundary(price, self.tick)
        for cancellation in receipt.cancellations:
            self._event(self.clearing.orders[cancellation.order_id], "cancel", cancellation.reason)
        return receipt

    def account_view(self, actor_id: str, *, version: int) -> LegacyAccountView:
        account = self.clearing.accounts[actor_id]
        cash, shares = self.clearing.available(actor_id)
        locked = account.acquired_today if self.clearing.t_plus_one else 0
        return LegacyAccountView(actor_id=actor_id, account_id=actor_id, version=version, cash=float(account.cash),
            available_cash=float(cash), reserved_cash=float(account.cash - cash), shares=account.shares,
            reserved_shares=account.shares - shares - locked, available_shares=shares, locked_shares=locked,
            acquired_today=account.acquired_today, cost_basis=float(account.cost_basis))

    def quote(self, levels=5) -> dict:
        # The legacy best-quote getters clean heaps; read-only projections use depth.
        depth = self.clearing.book.get_depth(levels)
        return {"last_price": self.clearing.book.last_price, "depth": depth,
                "best_bid": depth["bids"][0]["price"] if depth["bids"] else None,
                "best_ask": depth["asks"][0]["price"] if depth["asks"] else None}

    def validate_state(self) -> None:
        self.clearing.check_invariants()

    def export_state(self) -> dict:
        c, b = self.clearing, self.clearing.book
        clear_known = {*self._clear_scalars, *self._clear_maps, "book", "accounts", "orders", "_expected_accounts",
                       "_rejected_ids", "_cancellations", "_failure", "last_invariant_report"}
        book_known = {*self._book_scalars, "market_rules", "orders", "bids", "asks", "trades_history",
                      "_step_trades", "_arrival_sequence"}
        if set(vars(c)) != clear_known or set(vars(b)) != book_known:
            raise ValueError("Unregistered legacy market state; update the versioned snapshot before use")
        clearing = {name: getattr(c, name) for name in (*self._clear_scalars, *self._clear_maps)}
        clearing.update(accounts={k: asdict(v) for k, v in c.accounts.items()}, account_ids=list(c.accounts),
            _expected_accounts={k: asdict(v) for k, v in c._expected_accounts.items()},
            _rejected_ids=sorted(c._rejected_ids), _cancellations=[asdict(v) for v in c._cancellations],
            _failure=asdict(c._failure) if c._failure else None,
            last_invariant_report=asdict(c.last_invariant_report) if c.last_invariant_report else None)
        book = {name: getattr(b, name) for name in self._book_scalars}
        book.update(market_rules=b.market_rules.to_dict(), order_ids=list(b.orders), bids=b.bids, asks=b.asks,
            trades=[asdict(t) for t in b.trades_history], step_trade_ids=[t.trade_id for t in b._step_trades],
            _arrival_sequence=b._arrival_sequence)
        return LegacySettlementState(orders=json_values([asdict(o) for o in c.orders.values()]),
            clearing=json_values(clearing), book=json_values(book), adapter={"tick": self.tick, "world_version": self.world_version,
                "order_steps": self.order_steps, "order_decisions": self.order_decisions,
                "receipts": {key: value.model_dump() for key, value in self.receipts.items()},
                "order_events": [event.model_dump() for event in self.order_events]}).model_dump()

    def import_state(self, state: dict) -> None:
        state = LegacySettlementState.model_validate(state).model_dump()
        expected_hash = content_hash(state)
        raw, book = state["clearing"], state["book"]
        orders = {}
        for value in state["orders"]:
            value.update(side=OrderSide(value["side"]), order_type=OrderType(value["order_type"]), status=OrderStatus(value["status"]))
            order = Order(**value)
            if order.order_id in orders:
                raise ValueError("Duplicate canonical order ID")
            orders[order.order_id] = order
        rule = book["market_rules"]
        rule["sessions"] = tuple(SessionRule(**row) for row in rule["sessions"])
        rule["supported_order_types"] = tuple(rule["supported_order_types"])
        restored_book = OrderBook(book["symbol"], book["prev_close"], market_rules=MarketRuleSchema(**rule))
        for name in self._book_scalars:
            setattr(restored_book, name, book[name])
        restored_book.orders = {oid: orders[oid] for oid in book["order_ids"]}
        restored_book.bids = [(price, tuple(priority), oid) for price, priority, oid in book["bids"]]
        restored_book.asks = [(price, tuple(priority), oid) for price, priority, oid in book["asks"]]
        restored_book.trades_history = [Trade(**row) for row in book["trades"]]
        trades = {trade.trade_id: trade for trade in restored_book.trades_history}
        restored_book._step_trades = [trades[tid] for tid in book["step_trade_ids"]]
        restored_book._arrival_sequence = dict(book["_arrival_sequence"])
        restored = ClearingHouse.__new__(ClearingHouse)
        restored.book, restored.orders = restored_book, orders
        restored.accounts = {key: Account(**raw["accounts"][key]) for key in raw["account_ids"]}
        restored._expected_accounts = {key: Account(**raw["_expected_accounts"][key]) for key in raw["account_ids"]}
        for name in self._clear_scalars:
            setattr(restored, name, raw[name])
        for name in self._clear_maps:
            setattr(restored, name, dict(raw[name]))
        restored._order_specs = {key: tuple(value[:5] + [OrderSide(value[5]), OrderType(value[6])] + value[7:])
                                 for key, value in raw["_order_specs"].items()}
        restored._order_status = {key: OrderStatus(value) for key, value in raw["_order_status"].items()}
        restored._settled_trades = {trade.trade_id: tuple(raw["_settled_trades"][trade.trade_id]) for trade in restored_book.trades_history}
        restored._rejected_ids = set(raw["_rejected_ids"])
        restored._cancellations = [CancellationReceipt(**row) for row in raw["_cancellations"]]
        restored._failure = FailureDiagnostic(**raw["_failure"]) if raw["_failure"] else None
        report = raw["last_invariant_report"]
        if report:
            report["account_cash_residuals"] = tuple(tuple(row) for row in report["account_cash_residuals"])
        restored.last_invariant_report = InvariantReceipt(**report) if report else None
        restored.check_invariants()
        adapter = state["adapter"]
        candidate = LegacySettlementAdapter(restored)
        SimTime(tick=adapter["tick"], phase="BOUNDARY")
        SimTime(tick=adapter["world_version"], phase="BOUNDARY")
        candidate.tick, candidate.world_version = adapter["tick"], adapter["world_version"]
        candidate.order_steps, candidate.order_decisions = adapter["order_steps"], adapter["order_decisions"]
        candidate.receipts = {key: FinancialExecutionReceipt.model_validate(value) for key, value in adapter["receipts"].items()}
        candidate.order_events = [OrderEvent.model_validate(value) for value in adapter["order_events"]]
        if (set(candidate.order_steps) != set(orders) or set(candidate.order_decisions) != set(orders)
                or any(type(v) is not int or not 1 <= v <= candidate.tick for v in candidate.order_steps.values())
                or any(key != receipt.command_id for key, receipt in candidate.receipts.items())
                or len({e.event_id for e in candidate.order_events}) != len(candidate.order_events)
                or any(e.order_id not in orders or e.step > candidate.tick for e in candidate.order_events)
                or content_hash(candidate.export_state()) != expected_hash):
            raise ValueError("Incomplete or noncanonical settlement snapshot")
        # Keep Runner's ClearingHouse and OrderBook aliases, replacing only their state.
        original_book = self.clearing.book
        original_book.__dict__.clear()
        original_book.__dict__.update(vars(restored_book))
        restored.book = original_book
        self.clearing.__dict__.clear()
        self.clearing.__dict__.update(vars(restored))
        self.tick, self.world_version = candidate.tick, candidate.world_version
        self.order_steps.clear()
        self.order_steps.update(candidate.order_steps)
        self.order_decisions = candidate.order_decisions
        self.receipts, self.order_events = candidate.receipts, candidate.order_events
