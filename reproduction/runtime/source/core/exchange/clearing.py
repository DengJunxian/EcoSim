"""Closed cash-market settlement, dynamic reservations and fail-stop integrity.

The existing order book owns matching. Clearing owns accounts and records every
completed settlement so aggregate conservation cannot hide a wrong participant,
fill or account delta. Unexpected mutation failures preserve the scene and stop
all later writes to this world.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, fields, is_dataclass, replace

from core.types import Order, OrderSide, OrderStatus, OrderType, Trade
from core.exchange.market_rules import execution_fees, worst_buy_cash

CASH_RTOL = 1e-12
CASH_ATOL = 1e-7


def _number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass
class Account:
    cash: float
    shares: int
    acquired_today: int = 0
    cost_basis: float = 0.0


@dataclass(frozen=True)
class CancellationReceipt:
    order_id: str
    agent_id: str
    remaining_qty: int
    reason: str


@dataclass(frozen=True)
class SessionBoundaryReceipt:
    step: int
    old_reference: float
    new_reference: float
    lower_limit: float
    upper_limit: float
    unlocked_accounts: tuple[str, ...]
    cancellations: tuple[CancellationReceipt, ...]

    @property
    def cancelled_order_ids(self) -> tuple[str, ...]:
        return tuple(record.order_id for record in self.cancellations)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["cancelled_order_ids"] = self.cancelled_order_ids
        return result


@dataclass(frozen=True)
class FailureDiagnostic:
    operation: str
    error_type: str
    message: str
    book_trade_count: int
    settled_trade_count: int


@dataclass(frozen=True)
class InvariantReceipt:
    cash_residual: float
    shares_residual: int
    book_trade_count: int
    settled_trade_count: int
    account_cash_residuals: tuple[tuple[str, float], ...]


class ClearingHouse:
    def __init__(self, book, accounts: dict[str, Account], *, t_plus_one: bool = False):
        if not isinstance(t_plus_one, bool):
            raise ValueError("t_plus_one must be bool")
        if (not isinstance(accounts, dict) or not accounts
                or any(not isinstance(key, str) or not key.strip() for key in accounts)
                or len({id(account) for account in accounts.values()}) != len(accounts)):
            raise ValueError("Accounts require distinct accounts and nonempty string IDs")
        for account in accounts.values():
            self._validate_account(account, t_plus_one=t_plus_one)
        if (book.orders or book.trades_history or book.submitted_orders or book.trade_count
                or book.total_volume or book.cancelled_orders or book._arrival_sequence or book.bids or book.asks
                or not _number(book.prev_close) or book.prev_close <= 0 or book.last_price != book.prev_close):
            raise ValueError("Clearing requires a fresh order book")
        self.book = book
        self.accounts = accounts
        self.t_plus_one = t_plus_one
        self.fees = 0.0
        self.orders: dict[str, Order] = {}
        try:
            self.initial_cash = math.fsum(a.cash for a in accounts.values())
        except OverflowError as exc:
            raise ValueError("Initial cash must have a finite total") from exc
        self.initial_shares = sum(a.shares for a in accounts.values())
        if not math.isfinite(self.initial_cash):
            raise ValueError("Initial cash must have a finite total")
        self._expected_accounts = {key: replace(account) for key, account in accounts.items()}
        self._expected_fees = 0.0
        self._expected_last_price = book.last_price
        self._settled_volume = 0
        self._order_specs: dict[str, tuple] = {}
        self._order_fills: dict[str, int] = {}
        self._order_status: dict[str, OrderStatus] = {}
        self._arrival_sequences: dict[str, int] = {}
        self._rejected_ids: set[str] = set()
        self._settled_trades: dict[str, tuple] = {}
        self._cancellations: list[CancellationReceipt] = []
        self._failure: FailureDiagnostic | None = None
        self.last_invariant_report: InvariantReceipt | None = None
        self.check_invariants()

    @staticmethod
    def _validate_account(account, *, t_plus_one: bool, tolerance: float = 0.0) -> None:
        if (not isinstance(account, Account)
                or not _number(account.cash) or account.cash < -tolerance
                or not _number(account.cost_basis) or account.cost_basis < 0
                or not _integer(account.shares) or account.shares < 0
                or not _integer(account.acquired_today) or account.acquired_today < 0
                or (t_plus_one and account.acquired_today > account.shares)):
            raise ValueError("Accounts require finite non-negative cash/cost and integer shares/acquired_today")

    @property
    def failed(self) -> bool:
        return self._failure is not None

    @property
    def failure(self) -> FailureDiagnostic | None:
        return self._failure

    @property
    def cancellation_history(self) -> tuple[CancellationReceipt, ...]:
        return tuple(self._cancellations)

    def ensure_usable(self) -> None:
        if self.failed:
            raise RuntimeError(f"Clearing world is failed (fail-stop): {self._failure.operation}: {self._failure.message}")

    def _fail(self, operation: str, exc: BaseException) -> None:
        if self._failure is None:
            self._failure = FailureDiagnostic(operation, type(exc).__name__, str(exc),
                                               self.book.trade_count, len(self._settled_trades))

    def diagnostic_snapshot(self) -> dict:
        """JSON-safe forensic observation without changing any market state.

        Non-finite values remain explicit forensic objects; they never become
        valid zero balances or measurements.
        """
        visiting: set[int] = set()
        def forensic(value):
            if isinstance(value, float) and not math.isfinite(value):
                return {"forensic_non_finite": repr(value)}
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if id(value) in visiting:
                return {"forensic_reference": "cycle", "forensic_type": type(value).__name__}
            visiting.add(id(value))
            try:
                if is_dataclass(value) and not isinstance(value, type):
                    return {field.name: forensic(getattr(value, field.name)) for field in fields(value)}
                if isinstance(value, dict):
                    return {str(key): forensic(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [forensic(item) for item in value]
                return {"forensic_type": type(value).__name__, "repr": repr(value)}
            finally:
                visiting.remove(id(value))
        return forensic({"failed": self.failed,
                "failure": self._failure,
                "accounts": self.accounts,
                "expected_accounts": self._expected_accounts,
                "orders": self.orders,
                "book": {"prev_close": self.book.prev_close, "last_price": self.book.last_price,
                         "trade_count": self.book.trade_count, "total_volume": self.book.total_volume,
                         "submitted_orders": self.book.submitted_orders, "cancelled_orders": self.book.cancelled_orders,
                         "bids": self.book.bids, "asks": self.book.asks,
                         "arrival_sequence": self.book._arrival_sequence,
                         "orders": self.book.orders},
                "book_trades": self.book.trades_history,
                "settled_trade_ids": tuple(self._settled_trades),
                "settled_trades": [dict(zip(Trade.__dataclass_fields__, values))
                                   for values in self._settled_trades.values()],
                "fees": self.fees,
                "last_invariant_report": self.last_invariant_report,
                "cancellations": self._cancellations})

    def diagnostics(self) -> dict:
        return self.diagnostic_snapshot()

    def active_orders(self, agent_id: str | None = None):
        return [o for o in self.orders.values() if
                o.status in {OrderStatus.PENDING, OrderStatus.PARTIAL} and o.remaining_qty > 0
                and (agent_id is None or o.agent_id == agent_id)]

    def available(self, agent_id: str) -> tuple[float, int]:
        account = self.accounts[agent_id]
        orders = self.active_orders(agent_id)
        reserved_cash = math.fsum(worst_buy_cash(o.price, o.remaining_qty, self.book.market_rules.commission_rate)
                                  for o in orders if o.side == OrderSide.BUY)
        reserved_shares = sum(o.remaining_qty for o in orders if o.side == OrderSide.SELL)
        locked = account.acquired_today if self.t_plus_one else 0
        return account.cash - reserved_cash, account.shares - reserved_shares - locked

    @staticmethod
    def _order_spec(order: Order) -> tuple:
        return (order.order_id, order.agent_id, order.symbol, order.price, order.quantity,
                order.side, order.order_type, order.timestamp)

    @staticmethod
    def _trade_record(trade: Trade) -> tuple:
        return tuple(getattr(trade, name) for name in Trade.__dataclass_fields__)

    @staticmethod
    def _input_reason(order: Order) -> str:
        if (any(not isinstance(value, str) or not value.strip()
                for value in (order.order_id, order.agent_id, order.symbol))
                or not _integer(order.quantity) or order.quantity <= 0
                or not _integer(order.filled_qty) or order.filled_qty != 0
                or not _number(order.price) or order.price <= 0
                or not _number(order.timestamp)
                or order.status != OrderStatus.PENDING
                or not isinstance(order.metadata, dict)):
            return "invalid order fields"
        try:
            OrderSide(order.side)
            OrderType(order.order_type)
        except (ValueError, TypeError):
            return "invalid side or order type"
        for key in ("position_acquired_ts", "inventory_acquired_ts"):
            if key in order.metadata and order.metadata[key] is not None:
                if not _number(order.metadata[key]):
                    return "invalid inventory acquisition timestamp"
                try:
                    time.localtime(order.metadata[key])
                    time.localtime(order.timestamp)
                except (OverflowError, ValueError, OSError):
                    return "invalid inventory acquisition timestamp"
        return ""

    def _reject(self, order: Order, reason: str) -> list[Trade]:
        order.status, order.reason = OrderStatus.REJECTED, reason
        if isinstance(order.order_id, str) and order.order_id.strip():
            self.orders[order.order_id] = order
            self._rejected_ids.add(order.order_id)
        return []

    def submit(self, order: Order) -> list[Trade]:
        self.ensure_usable()
        if not isinstance(order, Order):
            raise TypeError("submit requires an Order")
        if isinstance(order.order_id, str) and order.order_id in self.orders:
            raise ValueError(f"Duplicate order_id: {order.order_id}")
        self.check_invariants()
        reason = self._input_reason(order)
        if reason:
            return self._reject(order, reason)
        if order.agent_id not in self.accounts:
            return self._reject(order, "unknown account")
        try:
            # MARKET normalization precedes worst-case cash reservation.
            if not self.book._preprocess_order(order):
                return self._reject(order, order.reason or "market rule rejection")
            # The legacy book only checks LIMIT here. Every cleared order must
            # obey the same current-session limits before economic mutation.
            if not self.book.market_rules.check_price_limit(order.price):
                return self._reject(order, "price outside limit")
            cash, shares = self.available(order.agent_id)
            try:
                worst_cash = worst_buy_cash(order.price, order.quantity, self.book.market_rules.commission_rate)
            except OverflowError:
                return self._reject(order, "non-finite order notional")
            reason = ""
            if not math.isfinite(worst_cash):
                reason = "non-finite order notional"
            elif order.side == OrderSide.BUY and worst_cash > cash + 1e-8:
                reason = "insufficient unreserved cash"
            elif order.side == OrderSide.SELL and order.quantity > shares:
                reason = "insufficient sellable unreserved shares"
            for other in self.active_orders(order.agent_id):
                if other.side != order.side and ((order.side == OrderSide.BUY and order.price >= other.price)
                                                or (order.side == OrderSide.SELL and order.price <= other.price)):
                    reason = "self-cross prevention"
            if reason:
                return self._reject(order, reason)
            self.orders[order.order_id] = order
            self._order_specs[order.order_id] = self._order_spec(order)
            self._order_fills[order.order_id] = 0
            self._order_status[order.order_id] = OrderStatus.PENDING
            self._arrival_sequences[order.order_id] = len(self._order_specs) - 1
            trades = self.book.add_order(order)
            for trade in trades:
                self._settle_trade(trade)
            touched = {order.order_id}
            for trade in trades:
                touched.update((trade.maker_id, trade.taker_id))
            for oid in touched:
                current = self.orders[oid]
                filled = self._order_fills[oid]
                status = OrderStatus.FILLED if filled == current.quantity else (
                    OrderStatus.PARTIAL if filled else OrderStatus.PENDING)
                if current.order_type in {OrderType.IOC, OrderType.MARKET, OrderType.FOK} and filled < current.quantity:
                    status = OrderStatus.CANCELLED
                self._order_status[oid] = status
            self.check_invariants()
            return trades
        except BaseException as exc:
            self._fail("submit", exc)
            raise

    def _settle_trade(self, trade: Trade) -> None:
        """Validate and settle one real book trade; never hide a partial exception."""
        if (not isinstance(trade, Trade) or not isinstance(trade.trade_id, str) or not trade.trade_id
                or trade.trade_id in self._settled_trades
                or not _integer(trade.quantity) or trade.quantity <= 0
                or not _number(trade.price) or trade.price <= 0
                or not _number(trade.timestamp)
                or any(not _number(value) or value < 0
                       for value in (trade.buyer_fee, trade.seller_fee, trade.seller_tax))):
            raise RuntimeError("Invalid or duplicate trade in settlement")
        maker = self.orders.get(trade.maker_id)
        taker = self.orders.get(trade.taker_id)
        if (maker is None or taker is None or maker is taker or maker.side == taker.side
                or trade.maker_agent_id != maker.agent_id or trade.taker_agent_id != taker.agent_id):
            raise RuntimeError("Trade participant/order reconciliation failed")
        bid, ask = (maker, taker) if maker.side == OrderSide.BUY else (taker, maker)
        if (trade.buyer_agent_id != bid.agent_id or trade.seller_agent_id != ask.agent_id
                or trade.buyer_agent_id == trade.seller_agent_id
                or trade.price != maker.price or not ask.price <= trade.price <= bid.price
                or not self.book.market_rules.check_price_limit(trade.price)
                or any(self._order_fills[o.order_id] + trade.quantity > o.quantity for o in (bid, ask))):
            raise RuntimeError("Trade price/quantity/account reconciliation failed")
        rules = self.book.market_rules
        expected_buyer_fee, expected_seller_fee, expected_tax = execution_fees(
            trade.price, trade.quantity, rules.commission_rate, rules.stamp_duty_rate,
            seller_only_stamp_duty=rules.seller_only_stamp_duty)
        if (not math.isclose(trade.buyer_fee, expected_buyer_fee, rel_tol=CASH_RTOL, abs_tol=CASH_ATOL)
                or not math.isclose(trade.seller_fee, expected_seller_fee, rel_tol=CASH_RTOL, abs_tol=CASH_ATOL)
                or not math.isclose(trade.seller_tax, expected_tax, rel_tol=CASH_RTOL, abs_tol=CASH_ATOL)):
            raise RuntimeError("Trade fee reconciliation failed")
        buyer = self.accounts[bid.agent_id]
        seller = self.accounts[ask.agent_id]
        cost, proceeds = trade.buyer_total_cost, trade.seller_net_proceeds
        sellable = seller.shares - (seller.acquired_today if self.t_plus_one else 0)
        if buyer is seller or buyer.cash + 1e-8 < cost or sellable < trade.quantity:
            raise RuntimeError("Settlement invariant violated before account mutation")
        # Independent expected balances permit detection of conservation-preserving
        # misallocations and later direct account mutations.
        for buy, sell in ((buyer, seller), (self._expected_accounts[bid.agent_id], self._expected_accounts[ask.agent_id])):
            buy.cash -= cost
            buy.cost_basis = (buy.cost_basis * buy.shares + cost) / (buy.shares + trade.quantity)
            buy.shares += trade.quantity
            buy.acquired_today += trade.quantity
            sell.cash += proceeds
            sell.shares -= trade.quantity
            if sell.shares == 0:
                sell.cost_basis = 0.0
        fee = trade.buyer_fee + trade.seller_fee + trade.seller_tax
        self.fees += fee
        self._expected_fees += fee
        self._order_fills[bid.order_id] += trade.quantity
        self._order_fills[ask.order_id] += trade.quantity
        self._settled_trades[trade.trade_id] = self._trade_record(trade)
        self._settled_volume += trade.quantity
        self._expected_last_price = trade.price

    def _cancel_order(self, order_id: str, reason: str) -> bool:
        order = self.orders.get(order_id)
        if order is None or order.status not in {OrderStatus.PENDING, OrderStatus.PARTIAL}:
            return False
        remaining = order.remaining_qty
        if not self.book.cancel_order(order_id):
            raise RuntimeError("Active clearing order missing from cancellation path")
        order.reason = reason
        self._order_status[order_id] = OrderStatus.CANCELLED
        self._cancellations.append(CancellationReceipt(order_id, order.agent_id, remaining, reason))
        return True

    def cancel(self, order_id: str, *, reason: str = "manual") -> bool:
        self.ensure_usable()
        if not isinstance(order_id, str) or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Cancellation requires a string order_id and a nonempty reason")
        self.check_invariants()
        try:
            result = self._cancel_order(order_id, reason)
            self.check_invariants()
            return result
        except BaseException as exc:
            self._fail("cancel", exc)
            raise

    def _unlock(self) -> tuple[str, ...]:
        unlocked = tuple(key for key, account in self.accounts.items() if account.acquired_today)
        for key, account in self.accounts.items():
            account.acquired_today = 0
            self._expected_accounts[key].acquired_today = 0
        return unlocked

    def new_day(self) -> None:
        """Legacy explicit unlock. Runners use apply_session_boundary for a full day."""
        self.ensure_usable()
        self.check_invariants()
        try:
            self._unlock()
            self.check_invariants()
        except BaseException as exc:
            self._fail("new_day", exc)
            raise

    def apply_session_boundary(self, prev_close: float, step: int) -> SessionBoundaryReceipt:
        self.ensure_usable()
        if not _number(prev_close) or prev_close <= 0 or not _integer(step) or step < 1:
            raise ValueError("Session boundary requires a finite positive reference and positive integer step")
        proposed_rules = replace(self.book.market_rules, prev_close=float(prev_close))
        lower, upper = proposed_rules.get_limit_prices()
        if not _number(lower) or not _number(upper) or not 0 < lower <= upper:
            raise ValueError("Session boundary produces invalid price limits")
        self.check_invariants()
        old_reference = self.book.prev_close
        try:
            unlocked = self._unlock()
            self.book.update_prev_close(prev_close)
            cancellation_start = len(self._cancellations)
            for order in self.active_orders():
                if not lower <= order.price <= upper:
                    self._cancel_order(order.order_id, "session_price_limit")
            self.check_invariants()
            return SessionBoundaryReceipt(step, old_reference, float(prev_close), lower, upper,
                                          unlocked, tuple(self._cancellations[cancellation_start:]))
        except BaseException as exc:
            self._fail("apply_session_boundary", exc)
            raise

    def check_invariants(self) -> InvariantReceipt:
        self.ensure_usable()
        try:
            return self._check_invariants()
        except BaseException as exc:
            self._fail("check_invariants", exc)
            raise

    def _check_invariants(self) -> InvariantReceipt:
        if set(self.accounts) != set(self._expected_accounts):
            raise RuntimeError("Account registry changed outside clearing")
        for account in self.accounts.values():
            self._validate_account(account, t_plus_one=self.t_plus_one, tolerance=CASH_ATOL)
        if not _number(self.fees) or self.fees < 0:
            raise RuntimeError("Invalid fee sink")
        cash_total = math.fsum(a.cash for a in self.accounts.values()) + self.fees
        share_total = sum(a.shares for a in self.accounts.values())
        report = InvariantReceipt(cash_total - self.initial_cash, share_total - self.initial_shares,
            self.book.trade_count, len(self._settled_trades),
            tuple((key, account.cash - self._expected_accounts[key].cash) for key, account in self.accounts.items()))
        self.last_invariant_report = report
        if report.shares_residual != 0:
            raise RuntimeError("Share conservation violated")
        if not math.isclose(cash_total, self.initial_cash, rel_tol=CASH_RTOL, abs_tol=CASH_ATOL):
            raise RuntimeError("Cash plus fee-sink conservation violated")
        for key, actual in self.accounts.items():
            expected = self._expected_accounts[key]
            if (actual.shares != expected.shares or actual.acquired_today != expected.acquired_today
                    or not math.isclose(actual.cash, expected.cash, rel_tol=CASH_RTOL, abs_tol=CASH_ATOL)
                    or not math.isclose(actual.cost_basis, expected.cost_basis, rel_tol=CASH_RTOL, abs_tol=CASH_ATOL)):
                raise RuntimeError(f"Order-trade-account reconciliation failed: {key}")
            cash, shares = self.available(key)
            if not math.isfinite(cash) or cash < -CASH_ATOL or shares < 0:
                raise RuntimeError(f"Over-reserved account: {key}")
        if not math.isclose(self.fees, self._expected_fees, rel_tol=CASH_RTOL, abs_tol=CASH_ATOL):
            raise RuntimeError("Fee sink reconciliation failed")
        if set(self.orders) != set(self._order_specs) | self._rejected_ids:
            raise RuntimeError("Order registry changed outside clearing")
        for oid in self._rejected_ids:
            if self.orders[oid].status != OrderStatus.REJECTED:
                raise RuntimeError("Rejected order reactivated outside clearing")
        for oid, spec in self._order_specs.items():
            order = self.orders[oid]
            if (self._order_spec(order) != spec or not _integer(order.filled_qty)
                    or order.filled_qty != self._order_fills[oid] or order.status != self._order_status[oid]):
                raise RuntimeError(f"Order-trade reconciliation failed: {oid}")
            if order.status in {OrderStatus.PENDING, OrderStatus.PARTIAL}:
                if self.book.orders.get(oid) is not order:
                    raise RuntimeError("Active order bypassed clearing/book registry")
                if not self.book.market_rules.check_price_limit(order.price):
                    raise RuntimeError("Active order outside current session price limits")
        if any(oid not in self._order_specs or order is not self.orders[oid] for oid, order in self.book.orders.items()):
            raise RuntimeError("Order book changed outside clearing")
        if self.book.submitted_orders != len(self._order_specs) or self.book.cancelled_orders != len(self._cancellations):
            raise RuntimeError("Book order activity reconciliation failed")
        if self.book._arrival_sequence != self._arrival_sequences:
            raise RuntimeError("Order arrival priority changed outside clearing")
        if (any(not _integer(value) for value in (self.book.submitted_orders, self.book.cancelled_orders,
                                                 self.book.trade_count, self.book.total_volume))
                or self.book.total_volume != self._settled_volume
                or not _number(self.book.last_price) or self.book.last_price != self._expected_last_price):
            raise RuntimeError("Book price/volume reconciliation failed")
        # Python heaps retain lazy cancelled/filled entries; only active orders
        # must occur exactly once. Their prices and queue positions remain fixed.
        # The optional native wrapper has its own index and is outside S1 scope.
        if not hasattr(self.book, "_cpp_lob"):
            indexed: dict[str, int] = {}
            for side, heap in ((OrderSide.BUY, self.book.bids), (OrderSide.SELL, self.book.asks)):
                for i, entry in enumerate(heap):
                    price, priority, oid = entry
                    order = self.orders.get(oid)
                    if order is None or oid not in self._order_specs or order.side != side:
                        raise RuntimeError("Unknown or wrong-side book heap entry")
                    expected_price = -order.price if side == OrderSide.BUY else order.price
                    timestamp = (self.book.market_rules.normalize_timestamp(order.timestamp)
                                 if self.book.market_rules.is_feature_enabled("strict_queue_timestamps")
                                 else order.timestamp)
                    if price != expected_price or priority != (timestamp, self._arrival_sequences[oid]):
                        raise RuntimeError("Book heap price/priority reconciliation failed")
                    if i and heap[(i - 1) // 2] > entry:
                        raise RuntimeError("Book heap ordering violated")
                    if order.status in {OrderStatus.PENDING, OrderStatus.PARTIAL}:
                        indexed[oid] = indexed.get(oid, 0) + 1
            if indexed != {order.order_id: 1 for order in self.active_orders()}:
                raise RuntimeError("Active orders disagree with book heap index")
        if (self.book.trade_count != len(self._settled_trades)
                or len(self.book.trades_history) != len(self._settled_trades)
                or tuple(t.trade_id for t in self.book.trades_history) != tuple(self._settled_trades)):
            raise RuntimeError("Unsettled or missing book trades")
        if any(self._trade_record(trade) != self._settled_trades[trade.trade_id] for trade in self.book.trades_history):
            raise RuntimeError("Settled trade record changed")
        return report
