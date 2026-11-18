"""Declared engineering archetypes; these are not calibrated institutional models."""
from typing import NamedTuple

from civitas.contracts.base import content_hash


VERSION = 'kdd27-persona-v1'
PROMPT_VERSION = 'kdd27-cognition-v1'
PARSER_VERSION = 'kdd27-direct-v1'
INSTRUCTION = (
    'Return exactly one typed order or HOLD using only this visible observation. '
    'Use the actor profile and declared persona contract; public framing adds no numeric facts. '
    'Historical quotes describe the previous committed market, not executable current inventory. '
    'Respect available cash, commission, sellable inventory, halt and the day-anchor price band. '
    'Do not overlay beliefs, write messages, or return reasoning. Cite only visible evidence IDs. '
    'Execution may reject an otherwise valid typed order; do not invent a fill.'
)


class PersonaRule(NamedTuple):
    target_exposure: float
    value_weight: float
    trend_weight: float
    emotion_weight: float
    deadband: float
    max_resource_fraction: float


PERSONAS = {
    'retail': PersonaRule(.50, .5, 2., .10, .03, .10),
    'mutual_fund': PersonaRule(.65, 2., 0., 0., .04, .08),
    'pension': PersonaRule(.45, 1., 0., 0., .10, .02),
    'insurance': PersonaRule(.35, .5, 0., 0., .10, .02),
    'proprietary': PersonaRule(.50, 0., 5., .05, .02, .10),
    'market_maker': PersonaRule(.50, 0., 0., 0., .01, .05),
    'stabilization': PersonaRule(.50, 1., -5., 0., .03, .08),
    'rumor_sensitive': PersonaRule(.50, 0., 1., .30, .02, .10),
    'arbitrage': PersonaRule(.50, 5., 0., 0., .01, .10),
}


def persona_contract(archetype):
    rule = PERSONAS[archetype]._asdict()
    return {'version': VERSION, 'parameters': rule, 'hash': content_hash({'version': VERSION, **rule})}


def heuristic_allocation(own, account, market, config):
    rule = PERSONAS[own['profile']['archetype']]
    price = market['historical_quote']['last_price_ticks'] * config.price_tick
    stock = account['balances'][config.asset_id] * price
    wealth = stock + account['balances'][config.currency] / config.cash_scale
    exposure = stock / wealth if wealth else 0.0
    target = max(0., min(1., rule.target_exposure
        + rule.value_weight * (own['state']['references'][2] / price - 1.)
        + rule.trend_weight * market['previous_return'] + rule.emotion_weight * own['state']['emotion']))
    gap = target - exposure
    side = 'buy' if gap > 0 else 'sell'
    cost = market['commission'] + (market['seller_tax'] if side == 'sell' else 0.)
    fraction = min(rule.max_resource_fraction, abs(gap))
    if market['halted'] or abs(gap) <= rule.deadband + cost:
        side, fraction = 'hold', 0.0
    return side, fraction, {'contract': persona_contract(own['profile']['archetype']),
        'target': target, 'exposure': exposure, 'gap': gap, 'cost_threshold': rule.deadband + cost}
