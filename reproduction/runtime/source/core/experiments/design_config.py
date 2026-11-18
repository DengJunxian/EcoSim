"""Versioned research configuration and compatibility exports for strict records."""
from __future__ import annotations

import math
from typing import Literal
from pydantic import Field, model_validator

# Phase 0 public names retain their import path; implementation has one owner.
from core.experiments.strict_types import (
    StrictResearchModel, ResearchInputError, ResearchModelT,
    strict_json_loads, strict_yaml_loads, load_research_model,
)


class MeasurementSpec(StrictResearchModel):
    """Versioned raw measurements, with an explicit half-open evaluation clock.

    ``evaluation_start <= step < evaluation_end`` is the evaluation window.
    Caps, penalties, role weights and task utilities belong to later TaskSpec.
    Recovery uses one registered measured quantity and a consecutive-step rule.
    """

    schema_version: Literal[2] = 2
    definition_version: Literal["measurement-v2"] = "measurement-v2"
    snapshot_stage: Literal["end_market_phase"] = "end_market_phase"
    top_k: int = Field(default=5, ge=1)
    warmup_steps: int = Field(default=0, ge=0)
    evaluation_start: int = Field(default=0, ge=0)
    evaluation_end: int = Field(ge=1)
    price_proxy: Literal["current_mid", "last_trade", "history_reference"] = "current_mid"
    max_price_age_steps: int = Field(default=0, ge=0)
    return_gap_rule: Literal["adjacent_only"] = "adjacent_only"
    recovery_predicate: Literal["none", "metric_le", "metric_ge"] = "none"
    recovery_metric: str | None = None
    recovery_threshold: float | None = None
    recovery_start_step: int | None = Field(default=None, ge=0)
    recovery_k: int = Field(default=1, ge=1)
    required_metrics: list[str] = Field(default_factory=lambda: [
        "spread", "relative_spread", "vwap", "traded_quantity", "trade_count",
        "depth_bid_top_k", "depth_ask_top_k",
    ])
    arrival_cost_capture: bool = False
    missing_policy: Literal["explicit-null-v2"] = "explicit-null-v2"

    @model_validator(mode="after")
    def _measurement_semantics(self):
        if not self.warmup_steps <= self.evaluation_start < self.evaluation_end:
            raise ValueError("Require warmup_steps <= evaluation_start < evaluation_end")
        if (not self.required_metrics or any(not name for name in self.required_metrics)
                or len(set(self.required_metrics)) != len(self.required_metrics)):
            raise ValueError("required_metrics must be nonempty unique names")
        recovery_fields = (self.recovery_metric, self.recovery_threshold, self.recovery_start_step)
        if self.recovery_predicate == "none":
            if any(value is not None for value in recovery_fields):
                raise ValueError("No recovery predicate may carry recovery parameters")
        elif (any(value is None for value in recovery_fields) or not self.recovery_metric
              or not self.evaluation_start <= self.recovery_start_step < self.evaluation_end):
            raise ValueError("Recovery needs metric, finite threshold and start in evaluation window")
        return self


ActorKind = Literal['fundamental', 'trend', 'behavioral_social', 'normal_mm']


class PopulationAllocation(StrictResearchModel):
    actor_type: ActorKind
    count: int = Field(ge=0)
    parameter_profile_id: str = Field(min_length=1)


class InitialResourceDistribution(StrictResearchModel):
    mode: Literal['explicit', 'addressed']
    wealth_weights: list[float] | None
    wealth_log_sigma: float = Field(ge=0.0)
    cash_fractions: list[float] | None
    cash_fraction_min: float = Field(ge=0.0, le=1.0)
    cash_fraction_max: float = Field(ge=0.0, le=1.0)
    correlation_mode: Literal['independent', 'explicit_arrays']

    @model_validator(mode='after')
    def _distribution(self):
        if self.cash_fraction_min > self.cash_fraction_max:
            raise ValueError('Cash fraction bounds are inverted')
        if self.mode == 'explicit':
            if not self.wealth_weights or self.cash_fractions is None:
                raise ValueError('Explicit resources require both arrays')
            if any(w <= 0.0 for w in self.wealth_weights) or not math.isclose(math.fsum(self.wealth_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError('Wealth weights must be positive and sum to one')
            if any(not self.cash_fraction_min <= c <= self.cash_fraction_max for c in self.cash_fractions):
                raise ValueError('Cash fractions lie outside registered bounds')
            if self.correlation_mode != 'explicit_arrays' or self.wealth_log_sigma != 0.0:
                raise ValueError('Explicit resources require explicit_arrays and no inactive distribution variance')
        elif (self.wealth_weights is not None or self.cash_fractions is not None
              or self.correlation_mode != 'independent'):
            raise ValueError('Addressed resources use the registered independent generator')
        return self


class ReferenceAnchorDistribution(StrictResearchModel):
    mode: Literal['explicit', 'addressed']
    relative_log_offsets: list[float] | None
    offset_min: float
    offset_max: float

    @model_validator(mode='after')
    def _distribution(self):
        if self.offset_min > self.offset_max:
            raise ValueError('Anchor bounds are inverted')
        try:
            if not math.isfinite(math.exp(self.offset_max)) or math.exp(self.offset_min) == 0.0:
                raise ValueError('Anchor bounds exceed positive binary64 support')
        except OverflowError as exc:
            raise ValueError('Anchor bounds overflow binary64') from exc
        if self.mode == 'explicit':
            if self.relative_log_offsets is None or any(not self.offset_min <= x <= self.offset_max for x in self.relative_log_offsets):
                raise ValueError('Explicit anchors require bounded offsets')
        elif self.relative_log_offsets is not None:
            raise ValueError('Addressed anchors cannot also carry explicit offsets')
        return self


class SocialEdgeSpec(StrictResearchModel):
    source_actor_id: str = Field(min_length=1)
    target_actor_id: str = Field(min_length=1)
    weight: float = Field(gt=0.0)


class SocialGraphSpec(StrictResearchModel):
    topology: Literal['empty', 'ring', 'explicit']
    neighbors: int = Field(ge=0)
    edges: list[SocialEdgeSpec]
    conformity: float = Field(ge=0.0, le=1.0)
    influence: float = Field(ge=0.0)
    feed_capacity: int = Field(ge=0)
    initial_states: dict[str, Literal['S', 'I', 'R', 'B']]
    semantic_enabled: Literal[False]
    portfolio_weight_updates: Literal[False]

    @model_validator(mode='after')
    def _graph(self):
        if self.topology != 'explicit' and self.edges:
            raise ValueError('Only explicit topology may carry edges')
        if self.topology != 'ring' and self.neighbors != 0:
            raise ValueError('Only ring topology uses neighbors')
        if self.topology == 'ring' and (self.neighbors < 2 or self.neighbors % 2):
            raise ValueError('Ring neighbors must be a positive even count')
        keys = [tuple(sorted((edge.source_actor_id, edge.target_actor_id))) for edge in self.edges]
        if len(keys) != len(set(keys)) or any(a == b for a, b in keys):
            raise ValueError('Explicit undirected graph forbids self edges and duplicates')
        return self


class PolicyAccountInitial(StrictResearchModel):
    actor_id: str = Field(min_length=1)
    cash: float = Field(ge=0.0)
    shares: int = Field(ge=0)
    cost_basis: float = Field(ge=0.0)


class WorldSpec(StrictResearchModel):
    schema_version: Literal[1]
    world_family_id: str = Field(min_length=1)
    initialization_mode: Literal['generated', 'snapshot']
    generator_version: Literal['explicit-world-v1']
    n_ordinary_and_normal_mm: int = Field(ge=2)
    stable_actor_ids: list[str]
    initial_price: float = Field(gt=0.0)
    initial_total_trader_wealth: float = Field(gt=0.0)
    population_allocation: list[PopulationAllocation]
    initial_resource_distribution: InitialResourceDistribution
    reference_anchor_distribution: ReferenceAnchorDistribution
    social_graph_spec: SocialGraphSpec
    policy_account_initial: PolicyAccountInitial
    snapshot_ref: str | None = Field(pattern=r'^[0-9a-f]{64}$')
    data_origin_refs: list[str]

    @model_validator(mode='after')
    def _world(self):
        ids = self.stable_actor_ids
        if (len(ids) != self.n_ordinary_and_normal_mm or len(ids) != len(set(ids))
                or any(not aid for aid in ids) or self.policy_account_initial.actor_id in ids):
            raise ValueError('World requires unique stable ordinary/MM IDs and a separate policy account')
        if sum(row.count for row in self.population_allocation) != len(ids):
            raise ValueError('Population allocation must sum to N')
        if self.initialization_mode == 'snapshot' and self.snapshot_ref is None:
            raise ValueError('Snapshot initialization requires a content reference')
        if self.initialization_mode == 'generated' and self.snapshot_ref is not None:
            raise ValueError('Generated initialization cannot hide a snapshot reference')
        for values in (self.initial_resource_distribution.wealth_weights,
                       self.initial_resource_distribution.cash_fractions,
                       self.reference_anchor_distribution.relative_log_offsets):
            if values is not None and len(values) != len(ids):
                raise ValueError('Every explicit initial array must match the actor IDs')
        graph = self.social_graph_spec
        if set(graph.initial_states) != set(ids) or graph.neighbors >= len(ids):
            raise ValueError('Social initial states must contain exactly N actors, with legal degree')
        if any(edge.source_actor_id not in ids or edge.target_actor_id not in ids for edge in graph.edges):
            raise ValueError('Social edges cannot refer to absent actors or the policy account')
        if not self.data_origin_refs or any(not ref for ref in self.data_origin_refs):
            raise ValueError('World requires explicit origin references')
        return self


class BehaviorParameterProfile(StrictResearchModel):
    profile_id: str = Field(min_length=1)
    actor_kind: ActorKind
    participation_rate: float = Field(ge=0.0, le=1.0)
    trade_fraction: float = Field(ge=0.0, le=1.0)
    max_inventory_shares: int = Field(ge=0)
    beta_f: float = Field(ge=0.0)
    beta_t: float = Field(ge=0.0)
    beta_s: float = Field(ge=0.0)
    trend_window: int = Field(ge=1)
    forecast_noise_std: float = Field(ge=0.0)
    signal_delay_steps: int = Field(ge=0)
    signal_max: float = Field(ge=0.0)
    mu_max: float = Field(ge=0.0)
    forecast_sigma: float = Field(ge=0.0)
    sigma_max: float = Field(ge=0.0)
    alpha: float = Field(gt=0.0, le=1.0)
    lambda_base: float = Field(ge=1.0)
    social_lambda_gain: float = Field(ge=0.0)
    lambda_max: float = Field(ge=1.0)
    reference_adaptation: float = Field(ge=0.0, le=1.0)
    aggressiveness: float = Field(ge=0.0)
    aggressiveness_max: float = Field(ge=0.0, lt=1.0)
    indifference_epsilon: float = Field(ge=0.0)
    q_target_shares: int = Field(ge=0)
    inventory_skew: float = Field(ge=0.0, lt=1.0)
    half_spread_fraction: float = Field(gt=0.0, lt=1.0)
    quote_qty_lots: int = Field(ge=1)

    @model_validator(mode='after')
    def _profile(self):
        if self.forecast_sigma > self.sigma_max or 1.0 - self.mu_max - self.sigma_max <= 0.0:
            raise ValueError('Registered three-point forecasts must stay positive')
        if self.aggressiveness > self.aggressiveness_max:
            raise ValueError('Aggressiveness exceeds its registered bound')
        if self.lambda_base * (1.0 + self.social_lambda_gain) > self.lambda_max:
            raise ValueError('Worst-case effective lambda exceeds its registered bound')
        if self.q_target_shares > self.max_inventory_shares:
            raise ValueError('MM target inventory exceeds inventory limit')
        if self.actor_kind in ('fundamental', 'trend'):
            if self.alpha != 1.0 or self.lambda_base != 1.0 or self.social_lambda_gain != 0.0:
                raise ValueError('Fundamental/trend base family fixes alpha=lambda=1 and no lambda modifier')
        if self.actor_kind == 'fundamental' and not (self.beta_f > 0.0 and self.beta_t == self.beta_s == 0.0):
            raise ValueError('Fundamental family requires only the fundamental forecast channel')
        if self.actor_kind == 'trend' and not (self.beta_t > 0.0 and self.beta_f == self.beta_s == 0.0):
            raise ValueError('Trend family requires only the trend forecast channel')
        if self.actor_kind == 'behavioral_social' and self.beta_f + self.beta_t + self.beta_s <= 0.0:
            raise ValueError('Behavioral-social family requires an active forecast channel')
        return self


class ReferencePolicy(StrictResearchModel):
    priority: Literal['current_bbo_completed_quote_fresh_trade_initial_else_hold']
    max_quote_age_steps: int = Field(ge=0)
    max_reference_age_steps: int = Field(ge=0)
    max_trade_age_steps: int = Field(ge=0)
    max_private_signal_age_steps: int = Field(ge=0)
    initial_reference_grace_steps: Literal[1]


class BehaviorSpec(StrictResearchModel):
    version: Literal['prospect-three-point-v1']
    parameter_profiles: list[BehaviorParameterProfile]
    actor_profile_map: dict[str, str]
    active_parameter_map: dict[str, list[str]]
    reference_policy: ReferencePolicy
    submission_priority_version: Literal['addressed-priority-v1']
    social_transition_version: Literal['research-sync-sirs-v1']

    @model_validator(mode='after')
    def _profiles(self):
        profile_ids = [row.profile_id for row in self.parameter_profiles]
        if not profile_ids or len(profile_ids) != len(set(profile_ids)):
            raise ValueError('Behavior profile IDs must be nonempty and unique')
        if any(profile not in profile_ids for profile in self.actor_profile_map.values()):
            raise ValueError('Unknown actor behavior profile')
        if set(self.active_parameter_map) != set(profile_ids):
            raise ValueError('Active parameter map must cover all profiles')
        fields = set(BehaviorParameterProfile.model_fields) - {'profile_id', 'actor_kind'}
        for names in self.active_parameter_map.values():
            if len(names) != len(set(names)) or set(names) - fields:
                raise ValueError('Active parameter names must be unique registered fields')
        return self


class RuntimeServicesSpec(StrictResearchModel):
    stepper_version: Literal['research-step-v1']
    decision_kernel_version: Literal['research_rule_v1']
    population_version: Literal['civitas-stratified-v1']
    social_version: Literal['research-sync-sirs-v1']
    order_book_version: Literal['civitas-python-lob-v1']
    clearing_version: Literal['cscwd-clearing-v1']
    scenario_version: Literal['noop-information-v1','causal-scenarios-v1']
    policy_version: Literal['noop-policy-v1','funded-policy-v1']
    measurement_version: Literal['measurement-v2']
    journal_version: Literal['diagnostic-jsonl-v1']
    lower_llm_calls: Literal[0]
    resource_hashes: dict[str, str]

    @model_validator(mode='after')
    def _resources(self):
        import re
        if any(not key or not re.fullmatch('[0-9a-f]{64}', value) for key, value in self.resource_hashes.items()):
            raise ValueError('Runtime resources require named sha256 content references')
        return self


# Inherit the existing v1 field definitions and validators, with strict nested
# adapters. Validate the raw research input before a legacy parser can coerce it.
from core.experiments.config import (ExperimentConfig, MarketConfig, SocialConfig,
    LLMConfig, HistoricalInput, EventConfig)
from core.experiments.randomness import RandomSpec
from core.interventions.schema import ScenarioSpec, PolicySpec, PolicyConstraints
from core.exchange.market_rules import floor_to_tick


class ResearchMarketConfig(StrictResearchModel, MarketConfig):
    pass


class ResearchSocialConfig(StrictResearchModel, SocialConfig):
    pass


class ResearchLLMConfig(StrictResearchModel, LLMConfig):
    pass


class ResearchHistoricalInput(StrictResearchModel, HistoricalInput):
    pass


class ResearchEventConfig(StrictResearchModel, EventConfig):
    pass


class ResearchBaseConfig(StrictResearchModel, ExperimentConfig):
    market: ResearchMarketConfig = Field(default_factory=ResearchMarketConfig)
    social: ResearchSocialConfig = Field(default_factory=ResearchSocialConfig)
    llm: ResearchLLMConfig = Field(default_factory=ResearchLLMConfig)
    historical_input: ResearchHistoricalInput | None = None
    events: list[ResearchEventConfig] = Field(default_factory=list)

    def to_legacy(self) -> ExperimentConfig:
        return ExperimentConfig.model_validate(self.model_dump(mode='json'))


class ResearchRunSpec(StrictResearchModel):
    schema_version: Literal[2]
    base: ResearchBaseConfig
    world: WorldSpec
    behavior: BehaviorSpec
    scenario: ScenarioSpec
    policy: PolicySpec
    policy_constraints: PolicyConstraints
    measurement: MeasurementSpec
    random: RandomSpec
    services: RuntimeServicesSpec
    data_phase: Literal['development', 'validation', 'confirmation', 'test', 'ood', 'diagnostic']
    purpose: Literal['engineering_test', 'scientific_validation', 'exploration', 'confirmation', 'final_evaluation']
    provenance_refs: list[str]

    @model_validator(mode='after')
    def _effective_spec(self):
        base, world, behavior = self.base, self.world, self.behavior
        if base.events or base.market.backend != 'python' or base.decision_mode != 'vectorized':
            raise ValueError('Research v2 requires empty legacy events, Python backend and vectorized compatibility input')
        if base.llm.max_calls != 0 or base.llm.max_retries != 0 or base.llm.model:
            raise ValueError('The lower research market has zero LLM calls and no configured lower model')
        if base.n_agents != world.n_ordinary_and_normal_mm or base.market.initial_price != world.initial_price:
            raise ValueError('Research base and effective WorldSpec must agree exactly on N and P0')
        if floor_to_tick(world.initial_price, base.market.tick_size) != world.initial_price:
            raise ValueError('Effective initial price must be tick aligned')
        if self.policy_constraints.horizon != base.steps:
            raise ValueError('Policy constraints horizon must match the run')
        if self.policy_constraints.policy_account_cash != world.policy_account_initial.cash:
            raise ValueError('All policies use the same declared initial policy resources')
        if self.measurement.evaluation_start < 1 or self.measurement.evaluation_end > base.steps + 1:
            raise ValueError('Research market steps use 1..T; evaluation must lie in [1,T+1)')
        if set(behavior.actor_profile_map) != set(world.stable_actor_ids):
            raise ValueError('Behavior map must cover exactly the ordinary/MM actor IDs')
        profiles = {p.profile_id: p for p in behavior.parameter_profiles}
        position = 0
        for allocation in world.population_allocation:
            profile = profiles.get(allocation.parameter_profile_id)
            if profile is None or profile.actor_kind != allocation.actor_type:
                raise ValueError('Population allocation and actual behavior profile disagree')
            for actor in world.stable_actor_ids[position:position + allocation.count]:
                if behavior.actor_profile_map[actor] != profile.profile_id:
                    raise ValueError('Allocation order and explicit actor profile map disagree')
            position += allocation.count
        for profile in profiles.values():
            if (profile.max_inventory_shares % base.market.board_lot
                    or profile.q_target_shares % base.market.board_lot):
                raise ValueError('Inventory limits and targets must use whole board lots')
        if world.policy_account_initial.shares % base.market.board_lot:
            raise ValueError('Initial policy shares must use whole board lots')
        for info in self.scenario.initial_information:
            if info.actor_id is not None and info.actor_id not in world.stable_actor_ids:
                raise ValueError('Private information can only belong to its ordinary/MM actor')
        if any(event.release_step > base.steps for event in self.scenario.events):
            raise ValueError('Scenario event lies outside the run horizon')
        if not self.provenance_refs or any(not ref for ref in self.provenance_refs):
            raise ValueError('Research inputs require explicit provenance references')
        if self.random.root_seed != base.seed:
            raise ValueError('Legacy seed metadata and addressed root seed must agree')
        return self

    def require_base_match(self, config: ExperimentConfig) -> None:
        """Compare exact typed values, including normally inactive v1 metadata."""
        from core.experiments.identity import canonical_research_bytes
        if canonical_research_bytes(config.model_dump(mode='json')) != canonical_research_bytes(self.base.model_dump(mode='json')):
            raise ValueError('Passed Runner v1 config must exactly match research_spec.base')
