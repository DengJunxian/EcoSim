from fractions import Fraction
from itertools import combinations
from typing import Literal

from pydantic import TypeAdapter, model_validator

from civitas.contracts.base import Id, NonNegative, Positive, Record, require_unique
from civitas.contracts.portfolio import LedgerTransaction, PortfolioAccountView
from .types import PerceivedSimilarity, Probability


class HoldingSnapshot(Record):
    actors: list[Id]
    prices_minor: dict[Id, Positive]
    quantities: dict[Id, dict[Id, NonNegative]]
    wealth_minor: dict[Id, NonNegative]
    as_of_tick: NonNegative
    origin: Literal["ledger_projection", "synthetic_test"]
    visibility: Literal["researcher_only"] = "researcher_only"
    epsilon_minor: Positive = 1

    @model_validator(mode="after")
    def structure(self):
        require_unique(self.actors, "holding actor")
        if set(self.quantities) != set(self.actors) or set(self.wealth_minor) != set(self.actors):
            raise ValueError("Holding projection actor mismatch")
        for actor, quantities in self.quantities.items():
            if not quantities.keys() <= self.prices_minor.keys():
                raise ValueError("Unknown holding asset")
            if sum(self.prices_minor[k] * q for k, q in quantities.items()) > self.wealth_minor[actor]:
                raise ValueError("Asset value exceeds nonnegative-cash wealth")
        return self


class HoldingDisclosure(Record):
    disclosure_id: Id
    actor_id: Id
    visible_tick: NonNegative
    weights: dict[Id, Probability]
    public: Literal[True] = True

    @model_validator(mode="after")
    def allocation(self):
        if sum(Fraction(str(w)) for w in self.weights.values()) > 1:
            raise ValueError("Disclosed asset weights exceed wealth")
        return self


class TradeLink(Record):
    trade_id: Id
    tx_id: Id
    buyer_id: Id
    seller_id: Id
    asset_id: Id
    quantity_units: Positive
    notional_minor: Positive
    tick: Positive
    visibility: Literal["researcher_only"] = "researcher_only"


def project_trades(journal) -> list[TradeLink]:
    transactions = [LedgerTransaction.model_validate(tx) for tx in journal]
    require_unique([t.tx_id for t in transactions], "transaction")
    require_unique([t.trade.trade_id for t in transactions], "trade")
    return [TradeLink(trade_id=t.trade.trade_id, tx_id=t.tx_id, buyer_id=t.trade.buyer_id,
        seller_id=t.trade.seller_id, asset_id=t.trade.asset_id, quantity_units=t.trade.quantity_units,
        notional_minor=t.notional_minor, tick=t.trade.tick) for t in transactions]


def project_holdings(account_views, prices_minor: dict, *, currency: str, tick: int) -> HoldingSnapshot:
    """Consume WP03 immutable views; projections never reserve, settle or own balances."""
    views = [PortfolioAccountView.model_validate(view) for view in account_views]
    require_unique([v.account_id for v in views], "account view")
    quantities, wealth = {}, {}
    for view in views:
        if set(view.balances) != {currency, *prices_minor}:
            raise ValueError("Every ledger asset requires a price in the same cash unit")
        quantities[view.account_id] = {k: view.balances[k] for k in prices_minor}
        wealth[view.account_id] = view.balances[currency] + sum(prices_minor[k] * view.balances[k] for k in prices_minor)
    return HoldingSnapshot(actors=sorted(quantities), quantities=quantities, prices_minor=prices_minor,
                           wealth_minor=wealth, as_of_tick=tick, origin="ledger_projection")


def _weights(snapshot):
    return {a: {k: Fraction(snapshot.prices_minor[k] * q, max(snapshot.wealth_minor[a], snapshot.epsilon_minor))
                for k, q in snapshot.quantities[a].items() if q} for a in snapshot.actors}


def full_exposure(snapshot: HoldingSnapshot) -> dict:
    """Small-world exact reference; production updates use the inverted asset index."""
    snapshot = HoldingSnapshot.model_validate(snapshot)
    values = {}
    for a, b in combinations(sorted(snapshot.actors), 2):
        overlap = sum((min(Fraction(snapshot.prices_minor[k] * snapshot.quantities[a].get(k, 0),
                                    max(snapshot.wealth_minor[a], snapshot.epsilon_minor)),
                           Fraction(snapshot.prices_minor[k] * snapshot.quantities[b].get(k, 0),
                                    max(snapshot.wealth_minor[b], snapshot.epsilon_minor)))
                       for k in snapshot.prices_minor), Fraction())
        if overlap:
            values[a, b] = overlap
    return values


class ExposureIndex:
    """Research-only derived weights; no balances, message permissions or trust writes."""

    def __init__(self, snapshot: HoldingSnapshot):
        self._weights, self._holders, self._pairs = {}, {}, {}
        self._tick = -1
        self.update(snapshot)

    def update(self, snapshot: HoldingSnapshot) -> None:
        snapshot = HoldingSnapshot.model_validate(snapshot)
        if snapshot.as_of_tick < self._tick:
            raise ValueError("Holding projection cannot move backwards")
        weights = _weights(snapshot)
        holders = {}
        for actor, row in weights.items():
            for asset in row:
                holders.setdefault(asset, set()).add(actor)
        changed = {a for a in self._weights.keys() | weights.keys() if self._weights.get(a) != weights.get(a)}
        affected = set()
        for a in changed:
            for index, rows in ((self._holders, self._weights), (holders, weights)):
                for k in rows.get(a, {}):
                    affected.update(tuple(sorted((a, b))) for b in index[k] if a != b)
        for a, b in affected:
            left, right = weights.get(a, {}), weights.get(b, {})
            overlap = sum((min(left[k], right[k]) for k in left.keys() & right.keys()), Fraction())
            if overlap:
                self._pairs[a, b] = overlap
            else:
                self._pairs.pop((a, b), None)
        self._weights, self._holders, self._tick = weights, holders, snapshot.as_of_tick
        self._assets = frozenset(snapshot.prices_minor)

    def weight(self, actor: str, asset: str) -> Fraction:
        if actor not in self._weights:
            raise ValueError("Unknown holding actor")
        if asset not in self._assets:
            raise ValueError("Unknown holding asset")
        return self._weights[actor].get(asset, Fraction())

    def exposure(self, a: str, b: str) -> Fraction:
        if a not in self._weights or b not in self._weights:
            raise ValueError("Unknown holding actor")
        if a == b:
            return sum(self._weights[a].values(), Fraction())
        return self._pairs.get(tuple(sorted((a, b))), Fraction())

    def pairs(self) -> dict:
        return dict(sorted(self._pairs.items()))

    def disclosed_similarity(self, actor: str, disclosures, *, tick: int) -> list[PerceivedSimilarity]:
        TypeAdapter(NonNegative).validate_python(tick, strict=True)
        if tick < self._tick:
            raise ValueError("Disclosure query cannot use future own holdings")
        if actor not in self._weights:
            raise ValueError("Unknown disclosure observer")
        latest = {}
        for raw in disclosures:
            disclosure = HoldingDisclosure.model_validate(raw)
            if disclosure.actor_id not in self._weights:
                raise ValueError("Unknown disclosure actor")
            if not disclosure.weights.keys() <= self._assets:
                raise ValueError("Unknown disclosure asset")
            if disclosure.visible_tick <= tick and disclosure.actor_id != actor:
                previous = latest.get(disclosure.actor_id)
                if previous is None or (disclosure.visible_tick, disclosure.disclosure_id) > (previous.visible_tick, previous.disclosure_id):
                    latest[disclosure.actor_id] = disclosure
        return [PerceivedSimilarity(observer_id=actor, target_id=other, value=float(sum((
            min(weight, Fraction(str(disclosure.weights.get(asset, 0.0))))
            for asset, weight in self._weights[actor].items()), Fraction())), available_tick=max(self._tick, disclosure.visible_tick),
            origin="public_disclosure", evidence_refs=[disclosure.disclosure_id]) for other, disclosure in sorted(latest.items())]
