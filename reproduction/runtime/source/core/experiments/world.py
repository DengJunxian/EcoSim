"""Complete, immutable initial worlds and explicit restoration of real engines.

Only step-zero snapshots are supported here. Runtime state is never recovered
from a summary, a pickle, a Runner copy, or a fresh random draw.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Literal

import networkx as nx
import numpy as np
from pydantic import Field, model_validator

from agents.decision_context import BehaviorState
from agents.population import StratifiedPopulation
from core.exchange.clearing import Account, ClearingHouse
from core.exchange.market_rules import MarketRuleSchema, SessionRule
from core.exchange.order_book import OrderBook
from core.experiments.design_config import ResearchRunSpec, BehaviorParameterProfile
from core.experiments.identity import HASH_PATTERN, ResearchIntegrityError, research_hash
from core.experiments.randomness import KeyedRandomSource, RandomDraw, RandomSpec
from core.experiments.strict_types import StrictResearchModel
from core.interventions.schema import InitialInformation
from core.society.network import (AgentNode, SocialGraph, InformationDiffusion, SentimentState,
    SocialStateSnapshot, capture_social_state, restore_social_state)


class ArraySnapshot(StrictResearchModel):
    dtype: Literal['float64']
    shape: list[int]
    values: list[float]

    @model_validator(mode='after')
    def _shape(self):
        if not self.shape or any(dim < 0 for dim in self.shape) or math.prod(self.shape) != len(self.values):
            raise ValueError('Array shape does not match its complete flattened values')
        return self

    @classmethod
    def capture(cls, array: np.ndarray) -> ArraySnapshot:
        if array.dtype != np.dtype('float64'):
            raise ValueError('Research population arrays require binary64')
        return cls(dtype='float64', shape=list(array.shape), values=array.ravel().tolist())

    def restore(self) -> np.ndarray:
        return np.array(self.values, dtype=np.float64).reshape(self.shape).copy()


class InitialAccount(StrictResearchModel):
    actor_id: str = Field(min_length=1)
    cash: float = Field(ge=0.0)
    shares: int = Field(ge=0)
    acquired_today: Literal[0]
    cost_basis: float = Field(ge=0.0)


class InitialActor(StrictResearchModel):
    actor_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    actor_kind: Literal['fundamental', 'trend', 'behavioral_social', 'normal_mm']
    parameter_profile: BehaviorParameterProfile
    active_parameters: list[str]
    initial_wealth: float = Field(gt=0.0)
    wealth_weight: float = Field(gt=0.0)
    cash_fraction: float = Field(ge=0.0, le=1.0)
    relative_log_anchor: float
    wealth_residual: float
    behavior_state: BehaviorState


class PopulationSnapshot(StrictResearchModel):
    implementation: Literal['civitas-stratified-v1'] = 'civitas-stratified-v1'
    active_decision_kernel: Literal['research_rule_v1'] = 'research_rule_v1'
    account_view: Literal['non_authoritative_clearing_copy'] = 'non_authoritative_clearing_copy'
    arrays: dict[str, ArraySnapshot]
    active_parameter_map: dict[str, list[str]]
    inactive_legacy_state: dict[str, Any]


class InitialWorldSnapshot(StrictResearchModel):
    schema_version: Literal[1] = 1
    definition_version: Literal['initial-world-v1'] = 'initial-world-v1'
    world_hash: str = Field(pattern=HASH_PATTERN)
    initialization_spec_hash: str = Field(pattern=HASH_PATTERN)
    source_hash: str = Field(pattern=HASH_PATTERN)
    world_family_id: str = Field(min_length=1)
    step: Literal[0] = 0
    completed_step: Literal[0] = 0
    initial_price: float = Field(gt=0.0)
    initial_total_trader_wealth: float = Field(gt=0.0)
    total_wealth_residual: float
    actors: list[InitialActor]
    accounts: list[InitialAccount]
    policy_account_id: str = Field(min_length=1)
    policy_state: dict[str, Any]
    population: PopulationSnapshot
    social: SocialStateSnapshot
    market_rules: dict[str, Any]
    market_state: dict[str, Any]
    initial_information: list[InitialInformation]
    public_initial_reference: dict[str, Any]
    random_spec: RandomSpec
    initialization_draws: list[RandomDraw]
    data_origin_refs: list[str]

    def semantic_body(self) -> dict[str, Any]:
        return self.model_dump(mode='json', exclude={'world_hash'})


@dataclass
class WorldComponents:
    population: StratifiedPopulation
    social: SocialGraph
    diffusion: InformationDiffusion
    book: OrderBook
    clearing: ClearingHouse
    behavior_states: dict[str, BehaviorState]
    actor_initial_wealth: dict[str, float]
    actor_max_inventory: dict[str, int]
    actor_profile_ids: dict[str, str]
    initial_information: list[InitialInformation]
    snapshot: InitialWorldSnapshot


def initialization_projection(spec: ResearchRunSpec) -> dict[str, Any]:
    """Exclude policy actions and loader location, retain all initial economics."""
    world = spec.world.model_dump(mode='json')
    world.pop('initialization_mode')
    world.pop('snapshot_ref')
    return {'world': world, 'behavior': spec.behavior.model_dump(mode='json'),
        'market': spec.base.market.model_dump(mode='json'), 'social': spec.base.social.model_dump(mode='json'),
        'market_regime': spec.base.market_regime, 'random': spec.random.model_dump(mode='json'),
        'initial_information': [row.model_dump(mode='json') for row in spec.scenario.initial_information],
        'fundamental_process': spec.scenario.fundamental_process.model_dump(mode='json'),
        'private_signal_process': spec.scenario.private_signal_process.model_dump(mode='json')}


def _initial_rules(spec: ResearchRunSpec) -> MarketRuleSchema:
    market = spec.base.market
    return MarketRuleSchema(symbol=market.symbol, prev_close=spec.world.initial_price,
        price_limit_pct=market.price_limit, min_price_tick=market.tick_size, board_lot=market.board_lot,
        enforce_board_lot=True, allow_odd_lots=market.board_lot == 1,
        commission_rate=market.commission, stamp_duty_rate=market.stamp_duty,
        t_plus_one=market.t_plus_one,
        feature_flags={'market_rules_v1': True, 'strict_queue_timestamps': True},
        metadata={'session_model': 'discrete_continuous_auction', 'holiday_calendar': 'none'})


def _initial_social(spec: ResearchRunSpec, actors: list[InitialActor], accounts: list[InitialAccount]) -> SocialStateSnapshot:
    declared, ids = spec.world.social_graph_spec, spec.world.stable_actor_ids
    graph = SocialGraph.__new__(SocialGraph)
    graph.n_agents, graph.seed, graph.rng, graph.tick = len(ids), 0, None, 0
    graph.k, graph.p = declared.neighbors, 0.0
    graph.graph = nx.Graph()
    graph.graph.add_nodes_from(range(len(ids)))
    if declared.topology == 'ring':
        for index in range(len(ids)):
            for distance in range(1, declared.neighbors // 2 + 1):
                graph.graph.add_edge(index, (index + distance) % len(ids), weight=1.0)
    elif declared.topology == 'explicit':
        indices = {aid: index for index, aid in enumerate(ids)}
        for edge in declared.edges:
            graph.graph.add_edge(indices[edge.source_actor_id], indices[edge.target_actor_id], weight=edge.weight)
    graph.agents = {}
    by_id = {row.actor_id: row for row in accounts}
    for actor in actors:
        graph.agents[actor.row_index] = AgentNode(node_id=actor.row_index, agent_id=actor.actor_id,
            sentiment_state=SentimentState(declared.initial_states[actor.actor_id]),
            base_lambda=actor.parameter_profile.lambda_base, lambda_coeff=actor.parameter_profile.lambda_base,
            conformity=declared.conformity, influence=declared.influence, feed_capacity=declared.feed_capacity,
            holdings={spec.base.market.symbol: by_id[actor.actor_id].shares})
    diffusion = InformationDiffusion(graph, beta=spec.base.social.beta, gamma=spec.base.social.gamma,
        delta=spec.base.social.delta, lambda_amplifier=0.0)
    return capture_social_state(diffusion)


def build_initial_world(spec: ResearchRunSpec, *, source_hash: str,
                        random_source: KeyedRandomSource | None = None) -> InitialWorldSnapshot:
    spec = ResearchRunSpec.model_validate(spec)
    if type(source_hash) is not str or not re.fullmatch(HASH_PATTERN, source_hash):
        raise ValueError('A world requires a SHA-256 implementation content identity')
    if spec.world.initialization_mode != 'generated':
        raise ValueError('Snapshot mode requires the content resolver and restore_initial_world')
    rng = KeyedRandomSource(spec.random) if random_source is None else random_source
    if rng.spec != spec.random:
        raise ValueError('World randomness scope differs from its effective specification')
    world, resources, anchors = spec.world, spec.world.initial_resource_distribution, spec.world.reference_anchor_distribution
    draws: list[RandomDraw] = []
    def draw(actor: str, channel: str, distribution: str) -> float:
        record = rng.draw_receipt(step=0, entity_id=actor, channel=channel, draw_slot=0, distribution=distribution)
        draws.append(record)
        return record.value
    if resources.mode == 'explicit':
        weights, fractions = list(resources.wealth_weights), list(resources.cash_fractions)
    else:
        # A common shift preserves softmax weights and avoids exponential overflow.
        log_weights = [resources.wealth_log_sigma * draw(aid, 'world_wealth', 'normal') for aid in world.stable_actor_ids]
        shift = max(log_weights)
        raw_weights = [math.exp(value - shift) for value in log_weights]
        denominator = math.fsum(raw_weights)
        weights = [value / denominator for value in raw_weights]
        fractions = [resources.cash_fraction_min + (resources.cash_fraction_max-resources.cash_fraction_min)
            * draw(aid, 'world_cash_fraction', 'uniform') for aid in world.stable_actor_ids]
    offsets = (list(anchors.relative_log_offsets) if anchors.mode == 'explicit' else
        [anchors.offset_min + (anchors.offset_max-anchors.offset_min)
            * draw(aid, 'world_reference_anchor', 'uniform') for aid in world.stable_actor_ids])
    profiles = {profile.profile_id: profile for profile in spec.behavior.parameter_profiles}
    actors, accounts, rows, references, lambdas = [], [], [], [], []
    lot, price = spec.base.market.board_lot, world.initial_price
    sentiment_values = {'S': 0.0, 'I': -1.0, 'R': 0.0, 'B': 1.0}
    for index, aid in enumerate(world.stable_actor_ids):
        profile = profiles[spec.behavior.actor_profile_map[aid]]
        wealth = weights[index] * world.initial_total_trader_wealth
        shares = lot * math.floor((1.0-fractions[index]) * wealth / (lot * price))
        cash = wealth - shares * price
        if shares > profile.max_inventory_shares:
            raise ValueError(f'Initial shares exceed registered inventory limit for {aid}')
        anchor = price * math.exp(offsets[index])
        state = BehaviorState(actor_id=aid, reference_anchor=anchor, lambda_effective=profile.lambda_base,
            state_version=0, last_advanced_step=0, last_context_hash=None)
        actors.append(InitialActor(actor_id=aid, row_index=index, actor_kind=profile.actor_kind,
            parameter_profile=profile, active_parameters=list(spec.behavior.active_parameter_map[profile.profile_id]),
            initial_wealth=wealth, wealth_weight=weights[index], cash_fraction=fractions[index],
            relative_log_anchor=offsets[index], wealth_residual=cash+shares*price-wealth, behavior_state=state))
        accounts.append(InitialAccount(actor_id=aid, cash=cash, shares=shares, acquired_today=0, cost_basis=price if shares else 0.0))
        rows.append([cash, float(shares), price if shares else 0.0,
            sentiment_values[world.social_graph_spec.initial_states[aid]], 0.0, 0.0])
        references.append([anchor, anchor, anchor, anchor])
        lambdas.append(profile.lambda_base)
    policy = world.policy_account_initial
    accounts.append(InitialAccount(actor_id=policy.actor_id, cash=policy.cash, shares=policy.shares,
        acquired_today=0, cost_basis=policy.cost_basis))
    n = len(actors)
    arrays = {'state': np.array(rows, dtype=np.float64), 'reference_points': np.array(references, dtype=np.float64),
        'risk_appetite_state': np.zeros(n, dtype=np.float64), 'trading_intent_state': np.zeros(n, dtype=np.float64),
        'loss_aversion_intensity_state': np.array(lambdas, dtype=np.float64)}
    population = PopulationSnapshot(arrays={name: ArraySnapshot.capture(array) for name, array in arrays.items()},
        active_parameter_map=spec.behavior.active_parameter_map,
        inactive_legacy_state={'n_smart': 0, 'smart_agents': [], 'compat_agents': [],
            'influence_map': {}, 'neighbor_network': {}, 'smart_social_network': {},
            'last_smart_actions': {}, 'smart_genomes': {}, 'ambient_rng_enabled': False,
            'legacy_behavior_enabled': False, 'legacy_persona_generation_enabled': False})
    body = dict(initialization_spec_hash=research_hash(initialization_projection(spec)), source_hash=source_hash,
        world_family_id=world.world_family_id, initial_price=price,
        initial_total_trader_wealth=world.initial_total_trader_wealth,
        total_wealth_residual=math.fsum(actor.initial_wealth for actor in actors)-world.initial_total_trader_wealth,
        actors=actors, accounts=accounts, policy_account_id=policy.actor_id,
        policy_state={'status': 'idle', 'halted': False, 'queued_commands': [], 'announcement_history': []},
        population=population, social=_initial_social(spec, actors, accounts), market_rules=_initial_rules(spec).to_dict(),
        market_state={'prev_close': price, 'last_price': price, 'total_volume': 0, 'orders': [], 'trades': [],
            'active_reservations': {}, 'fees': 0.0, 'submitted_orders': 0, 'cancelled_orders': 0, 'trade_count': 0,
            'order_steps': {}, 'arrival_sequence': {}},
        initial_information=list(spec.scenario.initial_information),
        public_initial_reference={'value': price, 'source': 'initial_reference_only', 'as_of_step': 0, 'available_step': 0},
        random_spec=spec.random, initialization_draws=draws, data_origin_refs=list(world.data_origin_refs))
    provisional = InitialWorldSnapshot(world_hash='0'*64, **body)
    return provisional.model_copy(update={'world_hash': research_hash(provisional.semantic_body())})


def validate_initial_world(snapshot: InitialWorldSnapshot | dict[str, Any], *, spec: ResearchRunSpec,
                           source_hash: str) -> InitialWorldSnapshot:
    snapshot = InitialWorldSnapshot.model_validate(snapshot)
    spec = ResearchRunSpec.model_validate(spec)
    if snapshot.world_hash != research_hash(snapshot.semantic_body()):
        raise ResearchIntegrityError('Initial world content hash mismatch')
    if snapshot.source_hash != source_hash or snapshot.initialization_spec_hash != research_hash(initialization_projection(spec)):
        raise ResearchIntegrityError('Initial world implementation or initialization specification mismatch')
    if spec.world.snapshot_ref is not None and spec.world.snapshot_ref != snapshot.world_hash:
        raise ResearchIntegrityError('Snapshot content reference mismatch')
    # Reassemble from recorded, independently validated Draw receipts. No new
    # KeyedRandomSource draw, ambient RNG, or call-count state is consumed.
    generated_spec = spec.model_copy(update={'world': spec.world.model_copy(update={
        'initialization_mode': 'generated', 'snapshot_ref': None})})
    class RecordedInitialDraws:
        def __init__(self):
            self.spec = spec.random
            self.used = []
            self.records = {}
            for record in snapshot.initialization_draws:
                key = record.key
                if (key.root_seed, key.exogenous_family_id, key.replication_id, key.coupling_spec_hash) != (
                        spec.random.root_seed, spec.random.exogenous_family_id, spec.random.replication_id,
                        spec.random.coupling_spec_hash):
                    raise ResearchIntegrityError('Initial random receipt has a different scope')
                address = (key.step, key.entity_or_event_id, key.channel, key.slot, key.distribution_domain)
                if address in self.records:
                    raise ResearchIntegrityError('Initial random receipt has duplicate addresses')
                self.records[address] = record

        def draw_receipt(self, *, step, entity_id, channel, draw_slot, distribution):
            address = (step, entity_id, channel, draw_slot, 'u' if distribution == 'uniform' else 'n')
            if address not in self.records:
                raise ResearchIntegrityError('Required actual initial random receipt is missing')
            self.used.append(address)
            return self.records[address]

    recorded = RecordedInitialDraws()
    expected = build_initial_world(generated_spec, source_hash=source_hash, random_source=recorded)
    if set(recorded.used) != set(recorded.records):
        raise ResearchIntegrityError('Snapshot has inactive or unexpected initial random draws')
    if snapshot != expected:
        raise ResearchIntegrityError('Snapshot fields disagree with the complete registered initial world')
    return snapshot


def restore_initial_world(snapshot: InitialWorldSnapshot | dict[str, Any], *, spec: ResearchRunSpec,
                          source_hash: str) -> WorldComponents:
    snapshot = validate_initial_world(snapshot, spec=spec, source_hash=source_hash)
    # Build the real population container directly from complete effective state.
    # The legacy constructor would load ambient defaults and redraw resources.
    population = StratifiedPopulation.__new__(StratifiedPopulation)
    population.n_smart, population.n_vectorized = 0, len(snapshot.actors)
    population.seed, population._api_key = spec.random.root_seed, None
    population.feature_flags = {'population_protocol_v1': True, 'research_rule_v1': True}
    population.market_regime = spec.base.market_regime
    population.market_composition_path, population.market_composition = None, {}
    for name, array in snapshot.population.arrays.items():
        setattr(population, name, array.restore())
    for name, value in snapshot.population.model_dump()['inactive_legacy_state'].items():
        setattr(population, name, value)
    population.research_actor_ids = [actor.actor_id for actor in snapshot.actors]
    population.research_profiles = {actor.actor_id: actor.parameter_profile for actor in snapshot.actors}
    population.active_parameter_map = snapshot.population.model_dump()['active_parameter_map']
    population.snapshot_info = {'world_hash': snapshot.world_hash, 'initialization': 'complete_snapshot'}
    population.data_snapshot_info = dict(population.snapshot_info)
    population.config_hash = snapshot.initialization_spec_hash
    population.runtime_metadata = {'seed': population.seed, 'config_hash': population.config_hash,
        'snapshot': dict(population.snapshot_info), 'feature_flags': dict(population.feature_flags)}
    population.market_distribution_report = {'source': 'explicit_world_profiles', 'actors':
        [actor.model_dump(mode='json') for actor in snapshot.actors]}
    rule_data = snapshot.model_dump()['market_rules']
    rule_data['supported_order_types'] = tuple(rule_data['supported_order_types'])
    rule_data['sessions'] = tuple(SessionRule(**row) for row in rule_data['sessions'])
    rules = MarketRuleSchema(**rule_data)
    book = OrderBook(symbol=rules.symbol, prev_close=snapshot.initial_price, market_rules=rules)
    accounts = {row.actor_id: Account(cash=row.cash, shares=row.shares, acquired_today=row.acquired_today,
        cost_basis=row.cost_basis) for row in snapshot.accounts}
    clearing = ClearingHouse(book, accounts, t_plus_one=rules.t_plus_one)
    diffusion = restore_social_state(snapshot.social)
    return WorldComponents(population=population, social=diffusion.graph, diffusion=diffusion, book=book, clearing=clearing,
        behavior_states={actor.actor_id: actor.behavior_state for actor in snapshot.actors},
        actor_initial_wealth={actor.actor_id: actor.initial_wealth for actor in snapshot.actors},
        actor_max_inventory={actor.actor_id: actor.parameter_profile.max_inventory_shares for actor in snapshot.actors},
        actor_profile_ids={actor.actor_id: actor.parameter_profile.profile_id for actor in snapshot.actors},
        initial_information=list(snapshot.initial_information), snapshot=snapshot)
