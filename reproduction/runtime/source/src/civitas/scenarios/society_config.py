"""Explicit scenario composition; each run has exactly one economic backend."""
from typing import Literal

from pydantic import Field, model_validator

from civitas.contracts.base import Id, NonNegative, Positive, Record, require_unique
from civitas.contracts.legacy_config import LegacyRunConfig
from civitas.contracts.multiasset_config import MultiAssetRunConfig
from civitas.contracts.population import PopulationSpec
from civitas.contracts.portfolio import FeeSchedule
from civitas.mechanisms.behavior import BehaviorConfig
from civitas.relations.types import RelationsConfig


class SocietyNews(Record):
    asset_id: Id = 'A'
    delivery_tick: Positive = 20
    clarification_tick: Positive = 35
    community_size: Positive = 8
    probabilities: dict[Id,float]
    fact_outcome: Literal[0,1] = 0
    forecast_delay: Positive = 2
    forecast_horizon: Positive = 12


class SocietyIntervention(Record):
    event_id: Id
    tick: Positive
    kind: Literal['halt','resume','fees']
    fees: FeeSchedule | None = None

    @model_validator(mode='after')
    def fee_payload(self):
        if (self.kind == 'fees') != (self.fees is not None):
            raise ValueError('Only a fee intervention supplies a fee schedule')
        return self


class MarketSocietyConfig(Record):
    schema_version: Literal[1] = 1
    scenario: Literal['market_society'] = 'market_society'
    market: LegacyRunConfig | MultiAssetRunConfig
    initializer: Literal['wealth_weights_v1','legacy_initializer'] = 'wealth_weights_v1'
    population: PopulationSpec | None = None
    behavior: BehaviorConfig
    cognition_mode: Literal['rule','synthetic_test','archived_response','live'] = 'rule'
    slow_agents: NonNegative = 8
    slow_period: Positive = 12
    reflection_period: NonNegative = 60
    max_messages_per_activation: Literal[1] = 1
    memory_window: Positive = 2
    communication_neighbors: NonNegative = 2
    relations: RelationsConfig = Field(default_factory=RelationsConfig)
    news: SocietyNews | None = None
    interventions: list[SocietyIntervention] = Field(default_factory=list)
    quantity_rule: Literal['fraction','one_unit'] = 'fraction'
    arrival_order: Literal['event_key','actor_id'] = 'event_key'
    portfolio_price_limit: float = Field(default=0.1,gt=0.0,le=1.0)
    sirs_baseline: Literal[False] = False

    @property
    def backend(self):
        return self.market.backend

    @property
    def ticks(self):
        return self.market.ticks

    @property
    def root_seed(self):
        return self.market.root_seed

    @property
    def replica(self):
        return self.market.replica

    @property
    def population_spec(self):
        return self.market.population if isinstance(self.market,MultiAssetRunConfig) else self.population

    @property
    def n_agents(self):
        return self.market.experiment.n_agents if isinstance(self.market,LegacyRunConfig) else self.market.population.n_agents

    @model_validator(mode='after')
    def composition(self):
        legacy = isinstance(self.market,LegacyRunConfig)
        if self.slow_agents > self.n_agents or self.communication_neighbors >= self.n_agents:
            raise ValueError('Slow-agent and communication counts must fit the population')
        if not legacy and (self.population is not None or self.initializer != 'wealth_weights_v1'):
            raise ValueError('Portfolio initialization is owned by its population specification')
        if legacy and self.market.experiment.events:
            raise ValueError('Society interventions have one owner; legacy events must be empty')
        if not legacy and self.market.fee_changes:
            raise ValueError('Use society interventions for fee changes')
        if legacy and self.initializer == 'wealth_weights_v1':
            if self.population is None or self.population.n_agents != self.n_agents or len(self.population.assets) != 1:
                raise ValueError('Legacy wealth initialization requires one matching population asset')
            asset = self.population.assets[0]
            if asset.quote_currency!='CNY':
                raise ValueError('Legacy society amounts are in CNY')
            price = self.population.initial_price_ticks[asset.asset_id]*asset.price_tick_minor/self.population.cash_scale
            if (asset.asset_id,price) != (self.market.experiment.market.symbol,self.market.experiment.market.initial_price):
                raise ValueError('Population and legacy initial quote disagree')
        if legacy and self.initializer == 'legacy_initializer' and self.population is not None:
            raise ValueError('Legacy initializer cannot also use a new population')
        assets = ({self.market.experiment.market.symbol} if legacy else {a.asset_id for a in self.market.population.assets})
        joint = self.behavior.joint_outcomes
        if self.behavior.influence.owner!='rule_social' or self.behavior.influence.model_reads_peers:
            raise ValueError('Scene config supplies rule ownership; the designated slow policy exclusively owns its model belief')
        if legacy and self.behavior.parameters.norm_weight:
            raise ValueError('Legacy society has no group-norm projection')
        if set(joint.scenarios[0].asset_returns) != assets or joint.scenarios[0].horizon_ticks < self.slow_period:
            raise ValueError('Joint assets/horizon must cover the scene and slow period')
        if not any(s.execution == 'unfilled' for s in joint.scenarios):
            raise ValueError('Society decisions require an explicit no-fill scenario')
        if (self.behavior.root_seed,self.behavior.replica) != (self.root_seed,self.replica):
            raise ValueError('Scene and policy seeds must agree')
        require_unique([e.event_id for e in self.interventions],'intervention ID')
        if any(e.tick > self.ticks for e in self.interventions):
            raise ValueError('Intervention lies outside the run')
        if self.news:
            from civitas.mechanisms.influence import Opinion
            news = self.news
            if (news.asset_id not in assets or news.delivery_tick < 2 or news.clarification_tick <= news.delivery_tick
                    or news.clarification_tick > self.ticks or news.community_size*2 > self.n_agents):
                raise ValueError('News schedule, asset or community is invalid')
            Opinion(proposition_id=self.behavior.influence.proposition_id,horizon_ticks=joint.scenarios[0].horizon_ticks,
                    probabilities=news.probabilities)
            if set(news.probabilities) != {s.scenario_id for s in joint.scenarios}:
                raise ValueError('News must use the same joint scenario space')
        return self
