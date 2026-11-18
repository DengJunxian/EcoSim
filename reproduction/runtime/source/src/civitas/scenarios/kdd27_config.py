"""Strict, opt-in single-asset method profile; experiment arms remain separate."""
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from civitas.contracts.base import Id, NonNegative, Positive, Record, require_unique
from civitas.contracts.portfolio import FeeSchedule, Rate
from civitas.resources.fixedpoint import decimal_minor

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonnegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


def normalized(values):
    if any(v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-12:
        raise ValueError('Nonnegative normalized weights required')


class PaperPopulation(Record):
    archetype_proportions: dict[Id, Probability] = Field(default_factory=lambda:dict(
        retail=0.70, mutual_fund=0.08, pension=0.05, insurance=0.04, proprietary=0.04,
        market_maker=0.04, stabilization=0.01, rumor_sensitive=0.02, arbitrage=0.02))
    cognitive_type_proportions: dict[Literal['technical','news','herding'], Probability] = Field(
        default_factory=lambda:dict(technical=0.3333333333333333, news=0.3333333333333333, herding=0.3333333333333334))
    cash_log_mean: float = 11.512925464970229
    cash_log_sd: NonnegativeFloat = 0.6
    inventory_max: NonNegative = 2000
    cost_normal_mean: PositiveFloat = 100.0
    cost_normal_sd: NonnegativeFloat = 5.0
    emotion_beta: list[PositiveFloat] = Field(default_factory=lambda:[2.0,2.0], min_length=2,max_length=2)
    confidence_mean: Probability = 0.6
    confidence_sd: NonnegativeFloat = 0.1
    initial_risk: Probability = 0.5
    loss_aversion: Annotated[float, Field(ge=1.0)] = 2.25
    herding_beta: list[PositiveFloat] = Field(default_factory=lambda:[2.0,2.0],min_length=2,max_length=2)
    topics: list[Id] = Field(default_factory=lambda:['fundamentals','technical','macro','policy'],min_length=1)
    topic_probabilities: list[Probability] = Field(default_factory=lambda:[0.25]*4,min_length=1)
    initial_infection_fraction: Probability = 0.01

    @model_validator(mode='after')
    def distributions(self):
        for v in (self.archetype_proportions.values(),self.cognitive_type_proportions.values(),self.topic_probabilities):
            normalized(list(v))
        require_unique(self.topics,'topic')
        if len(self.topics)!=len(self.topic_probabilities):
            raise ValueError('Topic labels and probabilities differ')
        return self


class PaperMarket(Record):
    asset_id: Id = 'A'
    currency: Id = 'CNY'
    cash_scale: Literal[1000000] = 1000000
    initial_price: PositiveFloat = 100.0
    price_tick: PositiveFloat = 0.01
    commission: Probability = 0.00025
    seller_tax: Probability = 0.0005
    band: Annotated[float, Field(gt=0.0,le=1.0)] = 0.1
    order_ttl: Positive = 1
    same_day_lock: bool = False
    ticks_per_day: Positive = 100
    fee_rounding: Literal['floor_per_fill'] = 'floor_per_fill'
    quote_rounding: Literal['half_even'] = 'half_even'
    self_trade_prevention: Literal['cancel_incoming_remainder'] = 'cancel_incoming_remainder'

    @property
    def tick_minor(self):
        return decimal_minor(str(self.price_tick),self.cash_scale)

    @property
    def initial_ticks(self):
        return decimal_minor(str(self.initial_price),self.cash_scale)//self.tick_minor

    def fees(self, tax=None):
        def rate(value):
            f=Fraction(str(value))
            return Rate(numerator=f.numerator,denominator=f.denominator)
        return FeeSchedule(commission=rate(self.commission),tax=rate(self.seller_tax if tax is None else tax))

    @model_validator(mode='after')
    def units(self):
        if self.asset_id==self.currency or self.commission+self.seller_tax>=1.0:
            raise ValueError('Distinct resources and positive seller net proceeds required')
        if not self.tick_minor or decimal_minor(str(self.initial_price),self.cash_scale)%self.tick_minor:
            raise ValueError('Initial price must align to a representable positive tick')
        return self


class PaperNetwork(Record):
    topology: Literal['watts_strogatz','ring','empty'] = 'watts_strogatz'
    neighbors: NonNegative = 6
    rewiring: Probability = 0.1
    weight_initial: PositiveFloat = 1.0
    weight_min: PositiveFloat = 1.0
    weight_max: PositiveFloat = 2.0
    weight_smoothing: Probability = 0.1
    exposure_epsilon: PositiveFloat = 1e-12
    attention_capacity: NonNegative = 3
    exposure_ema: Probability = 0.05
    ranking_weights: list[Probability] = Field(default_factory=lambda:[0.35,0.25,0.15,0.15,0.1],min_length=5,max_length=5)

    @model_validator(mode='after')
    def weights(self):
        normalized(self.ranking_weights)
        if not self.weight_min<=self.weight_initial<=self.weight_max or self.neighbors%2:
            raise ValueError('Ordered edge bounds and an even WS degree required')
        return self


class PaperSocial(Record):
    infection_beta: Probability = 0.3
    recovery_gamma: Probability = 0.1
    resusceptibility_delta: Probability = 0.05
    pressure_coefficients: list[NonnegativeFloat] = Field(default_factory=lambda:[0.4,0.4,0.2],min_length=3,max_length=3)
    duration_recovery: NonnegativeFloat = 0.02
    echo_recovery: Probability = 0.5
    pressure_resusceptibility: NonnegativeFloat = 0.5
    pessimism_threshold: Probability = 0.2
    pessimistic_neighbor_threshold: Probability = 0.5
    herding_weight_transfer: Probability = 0.5
    confidence_drop: Probability = 0.05
    confidence_recovery: Probability = 0.02
    confidence_target: Probability = 0.6
    return_scale: PositiveFloat = 0.01
    emotion_noise_sd: NonnegativeFloat = 0.02
    emotion_weights: dict[Literal['technical','news','herding'],list[Probability]] = Field(default_factory=lambda:dict(
        technical=[0.6,0.1,0.1,0.2],news=[0.35,0.35,0.2,0.1],herding=[0.25,0.15,0.5,0.1]))

    @model_validator(mode='after')
    def weights(self):
        if set(self.emotion_weights)!={'technical','news','herding'}:
            raise ValueError('All cognitive types require emotion weights')
        for weights in self.emotion_weights.values():
            if len(weights)!=4:
                raise ValueError('Four emotion weights required')
            normalized(weights)
        return self


class PaperBehavior(Record):
    prospect_alpha: Annotated[float, Field(gt=0.0,le=1.0)] = 0.88
    direction_scale: PositiveFloat = 10.0
    reference_update: list[Probability] = Field(default_factory=lambda:[0.03,0.1,0.16,0.18],min_length=4,max_length=4)
    reference_weight: list[Probability] = Field(default_factory=lambda:[0.34,0.26,0.2,0.2],min_length=4,max_length=4)
    rolling_high_window: Positive = 100
    peer_price_amplitude: Annotated[float, Field(ge=0.0,lt=1.0)] = 0.05
    policy_price_amplitude: Annotated[float, Field(ge=0.0,lt=1.0)] = 0.05
    loss_pressure_multiplier: NonnegativeFloat = 0.5
    loss_pressure_min: PositiveFloat = 1.0
    loss_pressure_max: PositiveFloat = 5.0
    loss_pressure_scale: PositiveFloat = 4.0
    risk_weights: list[NonnegativeFloat] = Field(default_factory=lambda:[0.35,0.45,0.1,0.1,0.15],min_length=5,max_length=5)
    intention_weights: list[NonnegativeFloat] = Field(default_factory=lambda:[0.45,0.4,0.15],min_length=3,max_length=3)
    action_logistic_scale: PositiveFloat = 5.0
    hold_threshold: NonnegativeFloat = 0.25
    resource_fraction_min: Annotated[float, Field(gt=0.0,le=1.0)] = 0.01
    resource_fraction_max: Annotated[float, Field(gt=0.0,le=1.0)] = 0.1
    quote_log_noise_sd: NonnegativeFloat = 0.002
    reverse_retrospective: bool = False

    @model_validator(mode='after')
    def ordering(self):
        normalized(self.reference_weight)
        p,h,n,e=self.reference_update
        if not (0<p<h<n<e<=1) or sum(self.reference_weight[:2])<=0:
            raise ValueError('Ordered reference speeds and positive retrospective weight required')
        if self.resource_fraction_min>self.resource_fraction_max or self.loss_pressure_min>self.loss_pressure_max:
            raise ValueError('Reversed behavior bounds')
        return self


class PaperIntervention(Record):
    event_id: Id
    tick: Positive
    kind: Literal['infect','reset_sirs','halt','resume','tax','lock','band','policy','text']
    actors: list[Id] = Field(default_factory=list)
    value: float | bool | str | None = None

    @model_validator(mode='after')
    def payload(self):
        require_unique(self.actors,'direct infection actor')
        if self.kind=='infect':
            valid=bool(self.actors) and self.value is None
        elif self.actors:
            valid=False
        elif self.kind in ('reset_sirs','halt','resume'):
            valid=self.value is None
        elif self.kind=='lock':
            valid=type(self.value) is bool
        elif self.kind=='text':
            valid=type(self.value) is str and bool(self.value)
        else:
            valid=type(self.value) is float and ((-1.0<=self.value<=1.0) if self.kind=='policy' else
                (0.0<self.value<=1.0) if self.kind=='band' else (0.0<=self.value<1.0))
        if not valid:
            raise ValueError('Intervention payload differs from its declared kind')
        return self


class KDD27Config(Record):
    scenario: Literal['kdd27_v1'] = 'kdd27_v1'
    schema_version: Literal[1] = 1
    protocol_id: Literal['kdd27-method-v1'] = 'kdd27-method-v1'
    mechanism_version: Literal['kdd27-method-v1'] = 'kdd27-method-v1'
    metric_version: Literal['kdd27-trace-v1-not-formal-metrics'] = 'kdd27-trace-v1-not-formal-metrics'
    rng_version: Literal['kdd27-addressed-pcg64-v1'] = 'kdd27-addressed-pcg64-v1'
    execution_backend: Literal['python_reference_v1'] = 'python_reference_v1'
    backend: Literal['fixedpoint_multiasset'] = 'fixedpoint_multiasset'
    n_agents: Positive = 10000
    ticks: Positive = 20000
    root_seed: NonNegative = 20261110
    replica: NonNegative = 0
    concurrency: Positive = 8
    activation_probability: Probability = 0.05
    background_infection_fraction_per_day: Probability = 0.002
    population: PaperPopulation = Field(default_factory=PaperPopulation)
    market: PaperMarket = Field(default_factory=PaperMarket)
    network: PaperNetwork = Field(default_factory=PaperNetwork)
    social: PaperSocial = Field(default_factory=PaperSocial)
    behavior: PaperBehavior = Field(default_factory=PaperBehavior)
    cognition_mode: Literal['rule','synthetic_test','archived_response'] = 'rule'
    policy_backend: Literal['rule','llm','hybrid'] = 'rule'
    slow_agents: NonNegative = 10
    slow_period: Positive = 10
    maximum_cache_age: NonNegative = 10
    memory_policy: Literal['no_extra_memory_kdd01'] = 'no_extra_memory_kdd01'
    event_triggered_calls: Literal[False] = False
    interventions: list[PaperIntervention] = Field(default_factory=list)

    @model_validator(mode='after')
    def composition(self):
        if self.network.topology!='empty' and self.network.neighbors>=self.n_agents:
            raise ValueError('Network degree must fit the population')
        if self.slow_agents>self.n_agents or (self.cognition_mode=='rule')!=(self.policy_backend in ('rule','heuristic')):
            raise ValueError('Backend/source or strategic population mismatch')
        if self.policy_backend=='hybrid' and not self.slow_agents:
            raise ValueError('Hybrid requires at least one strategic actor')
        require_unique([e.event_id for e in self.interventions],'event')
        require_unique([(e.tick,e.kind) for e in self.interventions],'event kind per tick')
        actors={f'agent_{i:04d}' for i in range(self.n_agents)}
        for event in self.interventions:
            if event.tick>self.ticks or not set(event.actors)<=actors:
                raise ValueError('Event time/actor outside the run')
            if event.kind=='tax' and self.market.commission+event.value>=1.0:
                raise ValueError('Tax leaves nonpositive net proceeds')
        return self


class RQ1Config(KDD27Config):
    schema_version: Literal[2] = 2
    protocol_id: Literal['kdd27-rq1-v1'] = 'kdd27-rq1-v1'
    mechanism_version: Literal['kdd27-rq1-v1'] = 'kdd27-rq1-v1'
    baseline: Literal['full','zero_intelligence','homogeneous_linear_value'] = 'full'
    burn_in_ticks: NonNegative = 5000
    quiet_after_burn_in: bool = False
    cognition_mode: Literal['rule'] = 'rule'
    policy_backend: Literal['rule'] = 'rule'

    @model_validator(mode='after')
    def analysis_window(self):
        if self.burn_in_ticks>=self.ticks or self.burn_in_ticks%self.market.ticks_per_day:
            raise ValueError('RQ1 burn-in must end on a day boundary before the horizon')
        return self


def parse_paper_config(value):
    raw=value.model_dump(mode='json') if isinstance(value,Record) else value
    model={2:RQ1Config,3:RQ2Config,4:PaperCognitionConfig,5:RQ3Config}.get(raw.get('schema_version'),KDD27Config) if isinstance(raw,dict) else KDD27Config
    return model.model_validate(raw)


RQ2Condition = Literal['full_control','no_secondary_contagion','no_herding_feedback','both_off',
    'full_loss_off','lambda_only_1','reverse_retrospective_mapping','halt25','tax0015','same_day_lock','narrow_band005']


class RQ2Config(KDD27Config):
    schema_version: Literal[3] = 3
    protocol_id: Literal['kdd27-rq2-v1'] = 'kdd27-rq2-v1'
    mechanism_version: Literal['kdd27-rq2-v1'] = 'kdd27-rq2-v1'
    cognition_mode: Literal['rule'] = 'rule'
    policy_backend: Literal['rule'] = 'rule'
    ticks: Positive = 10999
    event_tick: Positive = 10000
    pre_quiet_ticks: Positive = 200
    direct_infection_fraction: Probability = 0.05

    @model_validator(mode='after')
    def event_window(self):
        if not 1<=self.event_tick-self.pre_quiet_ticks<self.event_tick or self.event_tick+25>self.ticks:
            raise ValueError('RQ2 requires a quiet prefix and a complete halt/resumption window')
        if self.interventions:
            raise ValueError('RQ2 shared shock and branch policies own the event schedule')
        if self.market.seller_tax!=.0005 or self.market.band!=.1 or self.market.same_day_lock:
            raise ValueError('RQ2 control requires the frozen tax, band and unlocked initial inventory')
        return self


class PaperCognitionConfig(KDD27Config):
    schema_version: Literal[4] = 4
    protocol_id: Literal['kdd27-cognition-v1'] = 'kdd27-cognition-v1'
    mechanism_version: Literal['kdd27-cognition-v1'] = 'kdd27-cognition-v1'
    n_agents: Positive = 100
    ticks: Positive = 4000
    background_infection_fraction_per_day: Literal[0.0] = 0.0
    policy_backend: Literal['rule','heuristic','llm','hybrid'] = 'rule'
    memory_policy: Literal['visible_receipts_decay_kdd06'] = 'visible_receipts_decay_kdd06'
    memory_decay_steps: Positive = 100
    memory_items: Positive = 8
    cognition_contract: Literal['kdd27-cognition-v1'] = 'kdd27-cognition-v1'
    persona_version: Literal['kdd27-persona-v1'] = 'kdd27-persona-v1'
    cognitive_sample_id: Id = 'kdd27-cognition:0'

    @model_validator(mode='after')
    def known_archetypes(self):
        from civitas.cognition.kdd27_persona import PERSONAS
        if set(self.population.archetype_proportions)!=set(PERSONAS):
            raise ValueError('Cognition profile requires the nine declared archetypes')
        return self


class RQ3Config(PaperCognitionConfig):
    schema_version: Literal[5] = 5
    protocol_id: Literal['kdd27-rq3-v1'] = 'kdd27-rq3-v1'
    mechanism_version: Literal['kdd27-rq3-v1'] = 'kdd27-rq3-v1'
    burn_in_ticks: NonNegative = 1000
    event_tick: Positive = 1200
    direct_infected_count: Positive = 5

    @model_validator(mode='after')
    def paired_event(self):
        if not self.burn_in_ticks<self.event_tick<=self.ticks or self.direct_infected_count>self.n_agents:
            raise ValueError('RQ3 requires a post-burn-in event and a feasible fixed infection count')
        if self.interventions:
            raise ValueError('RQ3 shared shock and wording branches own the event schedule')
        return self
