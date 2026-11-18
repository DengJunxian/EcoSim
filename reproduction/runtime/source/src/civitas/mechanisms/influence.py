"""Convex opinion pooling over witnessed, deduplicated WP04 message content."""
import math
from typing import Literal
from pydantic import Field, model_validator

from civitas.contracts.base import Id, Positive, Record
from civitas.contracts.observations import Observation, ObservationFact
from civitas.cognition.observations import CognitionObservationBuilder
from civitas.relations.integration import FIELD, active_relations
from civitas.relations.messages import message_weight
from civitas.relations.types import TypedMessageContent
from .outcomes import MASS_TOLERANCE
from .registry import BeliefOwner


class Opinion(Record):
    proposition_id: Id
    horizon_ticks: Positive
    probabilities: dict[Id, float]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def distribution(self):
        if (not self.probabilities or any(not 0.0 <= p <= 1.0 for p in self.probabilities.values())
                or not math.isclose(math.fsum(self.probabilities.values()),1.0,rel_tol=0,abs_tol=MASS_TOLERANCE)):
            raise ValueError('Opinion probabilities must be a finite distribution')
        return self


class InfluenceConfig(BeliefOwner):
    mu: float = Field(default=0.0, ge=0.0, le=1.0)
    nu: float = Field(default=0.0, ge=0.0, le=1.0)
    public_reference_id: Id | None = None

    @model_validator(mode='after')
    def one_update(self):
        if self.mu + self.nu > 1.0:
            raise ValueError('Social/public coefficients must sum to at most one')
        if (self.owner == 'llm_integrated' or self.model_reads_peers) and (self.mu or self.nu):
            raise ValueError('Peer-reading model and numeric social update cannot both own a proposition')
        return self


class InfluenceResult(Opinion):
    owner: str
    self_weight: float
    peer_weight: float
    public_weight: float
    evidence_ids: list[Id]
    message_ids: list[Id]


class MessageWeights(Record):
    version: Literal['wp04-message-weight-v1'] = 'wp04-message-weight-v1'
    weights: dict[Id, float]

    @model_validator(mode='after')
    def bounded(self):
        if any(not 0.0 <= weight <= 1.0 for weight in self.weights.values()):
            raise ValueError('Message weights must lie in [0,1]')
        return self


class InfluenceObservationBuilder(CognitionObservationBuilder):
    """Expose WP04's effective weights without duplicating its trust or commit rules."""

    def __init__(self, relations=None):
        self.relations = relations
        self.effect_spec = relations.effect_spec if relations is not None else None

    def select_messages(self, actor_id, snapshot, messages):
        if self.relations is not None:
            return self.relations.select_messages(actor_id, snapshot, messages)
        if FIELD in snapshot.data:
            raise ValueError('Typed relations require a composed RelationObserver')
        return super().select_messages(actor_id, snapshot, messages)

    def observe(self, actor_id, snapshot, candidates):
        observed = super().observe(actor_id, snapshot, candidates)
        if self.relations is None:
            return observed
        state = active_relations(snapshot.data, snapshot.tick)
        weights = MessageWeights(weights={m.message_id:message_weight(
            state, TypedMessageContent.model_validate(m.body), actor_id) for m in observed.delivered_messages})
        fact = ObservationFact(fact_id='wp06-message-weights:' + actor_id, category='market', owner_id=actor_id,
            visible_at=observed.visible_until, value=weights.model_dump())
        return super().observe(actor_id, snapshot, [*candidates, fact])

    def prepare_commit(self, snapshot, observations):
        return self.relations.prepare_commit(snapshot, observations) if self.relations is not None else {}


def social_update(observation, own, config):
    observation = Observation.model_validate(observation)
    own, config = Opinion.model_validate(own), InfluenceConfig.model_validate(config)
    if own.proposition_id != config.proposition_id:
        raise ValueError('Belief proposition owner mismatch')
    facts = {f.fact_id:f for f in observation.market_view}
    peers, roots, used = [], {}, []
    def compatible(raw):
        opinion = Opinion.model_validate(raw)
        if (opinion.proposition_id != own.proposition_id or opinion.horizon_ticks != own.horizon_ticks
                or set(opinion.probabilities) != set(own.probabilities)):
            raise ValueError('Social opinions require the same proposition, horizon and joint scenarios')
        return opinion
    weights = None
    fact = facts.get('wp06-message-weights:' + observation.actor_id)
    if fact is not None:
        if fact.owner_id != observation.actor_id or fact.public:
            raise ValueError('Effective message weights must be private to the observer')
        weights = MessageWeights.model_validate(fact.value).weights
        if set(weights) != {m.message_id for m in observation.delivered_messages}:
            raise ValueError('Effective weights must match delivered messages')
    if config.mu and observation.delivered_messages and 'relations:' + observation.actor_id in facts and weights is None:
        raise ValueError('Compose InfluenceObservationBuilder to consume WP04 message weights')
    if config.mu:
        for message in sorted(observation.delivered_messages,key=lambda m:m.message_id):
            if 'root_claim_id' not in message.body:
                continue
            body = TypedMessageContent.model_validate(message.body)
            fact = facts.get(body.content_ref)
            if body.author_id == observation.actor_id or fact is None or 'joint_opinion' not in fact.value:
                continue
            raw = fact.value['joint_opinion']
            if raw.get('proposition_id') != own.proposition_id:
                continue
            opinion = compatible(raw)
            signature = (body.author_id,opinion)
            if body.root_claim_id in roots:
                if roots[body.root_claim_id] != signature:
                    raise ValueError('Conflicting repeated root claim')
                continue
            roots[body.root_claim_id] = signature
            weight = weights[message.message_id] if weights is not None else 1.0
            if weight > 0:
                peers.append((opinion,weight,message.message_id))
                used.extend([message.message_id,body.content_ref])
    public = None
    fact = facts.get(config.public_reference_id)
    if config.nu and fact is not None:
        if not fact.public:
            raise ValueError('Public reference must be publicly observed')
        public = compatible(fact.value['joint_opinion'])
        used.append(fact.fact_id)
    mu, nu = config.mu if peers else 0.0, config.nu if public else 0.0
    peer_mass = math.fsum(weight for _,weight,_ in peers)
    probabilities = {}
    for key,p in sorted(own.probabilities.items()):
        neighbor = math.fsum(v.probabilities[key]*w for v,w,_ in peers)/peer_mass if peers else 0.0
        probabilities[key] = math.fsum([(1.0-mu-nu)*p,mu*neighbor,nu*public.probabilities[key] if public else 0.0])
    return InfluenceResult(proposition_id=own.proposition_id,horizon_ticks=own.horizon_ticks,
        probabilities=probabilities,confidence=own.confidence,owner=config.owner,
        self_weight=1.0-mu-nu,peer_weight=mu,public_weight=nu,
        evidence_ids=sorted(set(used)),message_ids=[m for _,_,m in peers])
