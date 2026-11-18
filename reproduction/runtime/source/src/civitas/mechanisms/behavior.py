"""Dimensionless reference-dependent values; proposals use the existing decision contract."""
from fractions import Fraction
import math
from typing import Literal
import numpy as np

from pydantic import Field, model_validator

from core.behavioral_finance import ProspectTheoryParams, prospect_value
from civitas.contracts.base import Id, NonNegative, Positive, Record, content_hash, require_unique
from civitas.contracts.decisions import HoldIntent
from civitas.contracts.portfolio import PortfolioDecisionBundle, PortfolioIntent, PortfolioOrderIntent, Rate
from civitas.core.rng import AddressedRNG
from civitas.cognition.beliefs import JointBelief
from civitas.cognition.policy import empty_bundle
from civitas.relations.types import Membership
from .outcomes import CandidateAction, ExactAmount, JointOutcomes, OutcomeProjector, Projection, market_inputs
from .norms import GroupNorm, norm_distance
from .influence import InfluenceConfig, Opinion, social_update


class ReferenceWealth(Record):
    amount: ExactAmount
    currency: Id
    cash_scale: Positive
    observed_tick: NonNegative
    evidence_ids: list[Id] = Field(min_length=1)

    @model_validator(mode='after')
    def positive_wealth(self):
        require_unique(self.evidence_ids,'reference evidence')
        if self.amount.numerator <= 0:
            raise ValueError('Reference wealth must be positive')
        return self


class BehaviorParameters(Record):
    alpha: float = Field(default=1.0, gt=0.0, le=1.0)
    beta: float = Field(default=1.0, gt=0.0, le=1.0)
    loss_aversion: float = Field(default=1.0, ge=1.0)
    temperature: float = Field(default=0.0, ge=0.0)
    effort_weight: float = Field(default=0.0, ge=0.0)
    norm_weight: float = Field(default=0.0, ge=0.0)
    reference_rho: float = Field(default=0.0, ge=0.0, le=1.0)
    epsilon_currency: float = Field(default=0.000001, gt=0.0)


class BehaviorConfig(Record):
    version: Literal['reference_value_v1'] = 'reference_value_v1'
    horizon_semantics: Literal['rolling_fixed_horizon'] = 'rolling_fixed_horizon'
    joint_outcomes: JointOutcomes
    parameters: BehaviorParameters = Field(default_factory=BehaviorParameters)
    trade_fraction: Rate = Field(default_factory=lambda: Rate(numerator=1, denominator=10))
    influence: InfluenceConfig = Field(default_factory=lambda: InfluenceConfig(proposition_id='joint_market'))
    root_seed: NonNegative = 42
    replica: NonNegative = 0


class ValueComponent(Record):
    scenario_id: Id
    normalized_outcome: float
    value: float
    probability: float
    weighted_value: float


class ActionScore(Record):
    action_key: Id
    projection: Projection
    components: list[ValueComponent]
    effort_cost: float
    norm_cost: float
    q: float


class DecisionScores(Record):
    observation_id: Id
    parameters: BehaviorParameters
    reference: ReferenceWealth
    joint_outcomes: JointOutcomes
    scores: list[ActionScore]
    rejected: list[Projection]


class ActionChoice(Record):
    action_key: Id
    probabilities: dict[Id, float]
    draw: float
    economic_intents: list[PortfolioIntent]


def _check_reference(observation, reference, account, assets):
    if (reference.currency != next(iter(assets.values())).quote_currency or reference.cash_scale != account.cash_scale
            or reference.observed_tick > observation.visible_until.tick):
        raise ValueError('Reference units or time disagree with observation')
    witnessed = set(reference.evidence_ids) <= observation.evidence_ids
    if not witnessed and not any(f.value.get('behavior', {}).get('reference_next') == reference.model_dump()
                                 for f in observation.private_memory_refs):
        raise ValueError('Reference must come from witnessed evidence')


def observed_reference(observation):
    account, assets, prices, _ = market_inputs(observation)
    currency = next(iter(assets.values())).quote_currency
    wealth = account.balances[currency] + sum(account.balances[a]*prices[a]*assets[a].price_tick_minor for a in assets)
    evidence = [f.fact_id for f in observation.account_view + observation.market_view
                if f.category == 'account' or f.fact_id.startswith('quotes:') or f.fact_id == 'wp05-institution:multiasset']
    return ReferenceWealth(amount=ExactAmount.of(wealth),currency=currency,cash_scale=account.cash_scale,
                          observed_tick=observation.visible_until.tick,evidence_ids=sorted(evidence))


def update_reference(observation, reference, rho):
    reference = ReferenceWealth.model_validate(reference)
    rho = BehaviorParameters(reference_rho=rho).reference_rho
    account, assets, _, _ = market_inputs(observation)
    _check_reference(observation,reference,account,assets)
    anchor = observed_reference(observation)
    weight = Fraction(str(rho))
    return anchor.model_copy(update={'amount':ExactAmount.of((1-weight)*reference.amount.fraction + weight*anchor.amount.fraction)})


class BehaviorPolicy:
    policy_id, version, mechanism_id = 'behavior', '1', 'reference_value_v1'

    def __init__(self, joint_outcomes, *, parameters=None, root_seed=42, replica=0, trade_fraction=None, influence=None):
        self.joint_outcomes = JointOutcomes.model_validate(joint_outcomes)
        self.parameters = BehaviorParameters.model_validate(parameters or BehaviorParameters())
        self.rng = AddressedRNG(root_seed, replica)
        self.trade_fraction = Rate.model_validate(trade_fraction or Rate(numerator=1,denominator=10))
        self.influence = InfluenceConfig.model_validate(influence or InfluenceConfig(proposition_id='joint_market'))
        self.configuration = BehaviorConfig(joint_outcomes=self.joint_outcomes,parameters=self.parameters,
            trade_fraction=self.trade_fraction,influence=self.influence,root_seed=root_seed,replica=replica)
        self.mechanism_versions = {self.mechanism_id:'1:' + content_hash(self.configuration)}

    def validate_scenarios(self, scenarios):
        if scenarios != {s.scenario_id:s.asset_returns for s in self.joint_outcomes.scenarios}:
            raise ValueError('Model and numeric policy must use the same joint scenario returns')

    @property
    def belief_contract(self):
        return {'horizon_ticks':self.joint_outcomes.scenarios[0].horizon_ticks,
                'horizon_semantics':'rolling_fixed_horizon'}

    def validate_belief(self, observation, belief):
        tick = observation.visible_until.tick
        if (belief.horizon_tick - belief.as_of_tick != self.belief_contract['horizon_ticks']
                or not belief.as_of_tick <= tick <= belief.valid_until_tick < belief.horizon_tick):
            raise ValueError('Belief must use the configured rolling horizon and expire before its horizon tick')

    def score(self, observation, candidates, reference, *, joint_outcomes=None, group_norms=()):
        return self._score(observation,candidates,reference,joint_outcomes=joint_outcomes,
                           group_norms=group_norms,vectorized=False)

    def score_batch(self, requests):
        return [self._score(obs,actions,reference,joint_outcomes=None,group_norms=(),vectorized=True)
                for obs,actions,reference in requests]

    def _score(self, observation, candidates, reference, *, joint_outcomes, group_norms, vectorized):
        reference = ReferenceWealth.model_validate(reference)
        account, assets, prices, _ = market_inputs(observation)
        groups = [GroupNorm.model_validate(g) for g in group_norms]
        facts = {f.fact_id:f for f in observation.market_view}
        relation = facts.get('relations:' + observation.actor_id)
        memberships = [Membership.model_validate(m) for m in relation.value.get('memberships', [])] if relation else []
        own_groups = {m.target_id for m in memberships if m.source_id == observation.actor_id}
        groups = [g for g in groups if g.group_id in own_groups] if self.parameters.norm_weight else []
        if any(g.evidence_id not in facts or facts[g.evidence_id].value.get('group_norm') != g.model_dump() for g in groups):
            raise ValueError('Norm must be present in a visible rule fact')
        _check_reference(observation,reference,account,assets)
        outcomes = JointOutcomes.model_validate(joint_outcomes or self.joint_outcomes)
        unique = {}
        for raw in [*candidates, CandidateAction(action_id='hold')]:
            action = CandidateAction.model_validate(raw)
            if set(action.delta_units) - set(assets):
                raise ValueError('Unknown action asset')
            key = action.economic_key
            if key in unique and unique[key].effort != action.effort:
                raise ValueError('Equivalent economic actions have conflicting effort costs')
            unique[key] = CandidateAction(action_id=key,delta_units={k:v for k,v in action.delta_units.items() if v},effort=action.effort)
        scores, rejected = [], []
        p = self.parameters
        params = ProspectTheoryParams(alpha=p.alpha, beta=p.beta, lambda_=p.loss_aversion)
        denominator = max(reference.amount.fraction, Fraction(str(p.epsilon_currency)) * account.cash_scale)
        wealth = account.balances[reference.currency] + sum(account.balances[a]*prices[a]*assets[a].price_tick_minor for a in assets)
        if wealth <= 0:
            raise ValueError('Positive observed wealth required for normalized actions')
        for key, action in sorted(unique.items()):
            projection = OutcomeProjector().project(observation, action, outcomes)
            if not projection.feasible:
                rejected.append(projection)
                continue
            xs = [float((row.wealth.fraction-reference.amount.fraction)/denominator) for row in projection.outcomes]
            if vectorized:
                array = np.asarray(xs,dtype=np.float64)
                values = np.where(array >= 0,np.power(np.abs(array),p.alpha),-p.loss_aversion*np.power(np.abs(array),p.beta)).tolist()
            else:
                values = [float(prospect_value(x,params=params)) for x in xs]
            components = []
            for row,x,value in zip(projection.outcomes,xs,values,strict=True):
                components.append(ValueComponent(scenario_id=row.scenario_id,normalized_outcome=x,value=value,
                    probability=row.probability,weighted_value=row.probability * value))
            effort = p.effort_weight * action.effort
            normalized = {a:float(Fraction(action.delta_units.get(a,0)*prices[a]*assets[a].price_tick_minor,wealth)) for a in assets}
            norm = p.norm_weight * norm_distance(normalized, groups)
            scores.append(ActionScore(action_key=key,projection=projection,components=components,
                effort_cost=effort,norm_cost=norm,q=math.fsum(c.weighted_value for c in components) - effort - norm))
        return DecisionScores(observation_id=observation.observation_id, parameters=p,reference=reference,
                              joint_outcomes=outcomes,scores=scores,rejected=rejected)

    def sample(self, scores, observation):
        scores = DecisionScores.model_validate(scores)
        if scores.observation_id != observation.observation_id or scores.parameters != self.parameters or not scores.scores:
            raise ValueError('Choice requires current scores and parameters')
        ordered = sorted(scores.scores, key=lambda s:s.action_key)
        maximum = max(s.q for s in ordered)
        tau = self.parameters.temperature
        weights = [(1.0 if s.q == maximum else 0.0) if tau == 0.0 else math.exp((s.q - maximum) / tau) for s in ordered]
        total = math.fsum(weights)
        probabilities = {s.action_key:w / total for s,w in zip(ordered,weights,strict=True)}
        tick = observation.visible_until.tick
        draw = self.rng.uniform('wp06.choice',observation.actor_id,tick,'economic_action')
        cumulative, chosen = 0.0, ordered[-1]
        for score in ordered:
            cumulative += probabilities[score.action_key]
            if draw < cumulative:
                chosen = score
                break
        prefix = f'wp06:{tick}:{observation.actor_id}:'
        intents = [PortfolioOrderIntent(intent_id=prefix+a,order_id='order:'+prefix+a,asset_id=a,
            side='buy' if q > 0 else 'sell',price_ticks=chosen.projection.assumed_price_ticks[a],quantity_units=abs(q))
            for a,q in sorted(chosen.projection.candidate.delta_units.items()) if q]
        return ActionChoice(action_key=chosen.action_key,probabilities=probabilities,draw=draw,
                            economic_intents=intents or [HoldIntent(intent_id=prefix+'hold')])

    def candidate_actions(self, observation):
        account, assets, prices, fees = market_inputs(observation)
        currency = next(iter(assets.values())).quote_currency
        ratio = self.trade_fraction
        actions = []
        for asset,spec in sorted(assets.items()):
            price = prices[asset] * spec.price_tick_minor
            rate = fees.commission
            # A per-unit ceiling keeps the candidate grid affordable under floor-per-fill execution fees.
            per_unit = price + (price*rate.numerator + rate.denominator-1)//rate.denominator
            buy = (account.available[currency]*ratio.numerator//ratio.denominator)//per_unit
            sell = account.available[asset]*ratio.numerator//ratio.denominator
            for name,quantity in [('buy',buy),('sell',-sell)]:
                if quantity:
                    actions.append(CandidateAction(action_id=name+':'+asset,delta_units={asset:quantity}))
        return actions

    def _decide(self, observation, outcomes):
        references = [f for f in observation.private_memory_refs if 'reference_next' in f.value.get('behavior',{})]
        reference = (ReferenceWealth.model_validate(max(references,key=lambda f:(f.visible_at.key,f.fact_id)).value['behavior']['reference_next'])
                     if references else observed_reference(observation))
        own = Opinion(proposition_id=self.influence.proposition_id,horizon_ticks=outcomes.scenarios[0].horizon_ticks,
                      probabilities={s.scenario_id:s.probability for s in outcomes.scenarios})
        pooled = social_update(observation,own,self.influence)
        outcomes = outcomes.model_copy(update={'scenarios':[s.model_copy(update={'probability':pooled.probabilities[s.scenario_id]}) for s in outcomes.scenarios]})
        norms = [GroupNorm.model_validate(f.value['group_norm']) for f in observation.market_view if 'group_norm' in f.value] if self.parameters.norm_weight else []
        scores = self.score(observation,self.candidate_actions(observation),reference,joint_outcomes=outcomes,group_norms=norms)
        choice = self.sample(scores,observation)
        trace = {'scores':scores.model_dump(),'choice':choice.model_dump(),
                 'reference_next':update_reference(observation,reference,self.parameters.reference_rho).model_dump(),
                 'influence':pooled.model_dump(),'belief_update_owner':self.influence.owner,
                 'horizon_semantics':self.belief_contract['horizon_semantics'],
                 'mechanism_versions':self.mechanism_versions}
        return empty_bundle(observation,self.policy_id,self.version,PortfolioDecisionBundle).model_copy(update={
            'economic_intents':choice.economic_intents,'belief_patch':{'behavior':trace},
            'evidence_ids':sorted(observation.evidence_ids)})

    async def decide(self, observation):
        if self.influence.owner != 'rule_social':
            raise ValueError('Model-owned beliefs require the explicit decide_from_belief port')
        return self._decide(observation,self.joint_outcomes)

    async def fallback_decide(self, observation):
        bundle = empty_bundle(observation,self.policy_id,self.version,PortfolioDecisionBundle)
        return bundle.model_copy(update={'economic_intents':[HoldIntent(intent_id='intent:'+bundle.decision_id)]})

    async def decide_from_belief(self, observation, belief):
        belief = JointBelief.model_validate(belief)
        if self.influence.owner != 'llm_integrated':
            raise ValueError('WP05 reads peer messages: numeric policy must declare llm_integrated ownership')
        self.validate_belief(observation, belief)
        if belief.abstain:
            return await self.fallback_decide(observation)
        if set(belief.probabilities) != {s.scenario_id for s in self.joint_outcomes.scenarios}:
            raise ValueError('Joint belief scenarios disagree with numeric policy')
        outcomes = self.joint_outcomes.model_copy(update={'scenarios':[s.model_copy(update={
            'probability':belief.probabilities[s.scenario_id]}) for s in self.joint_outcomes.scenarios]})
        return self._decide(observation,outcomes)
