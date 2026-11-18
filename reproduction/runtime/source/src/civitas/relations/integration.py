from civitas.application import IntentCompiler
from civitas.contracts.base import content_hash
from civitas.contracts.config import MechanismSpec
from civitas.contracts.observations import ObservationFact
from civitas.core.observation import ObservationBuilder

from .membership import visible_memberships
from .messages import exposure_events, select_messages, validate_messages
from .rewiring import propose_rewiring
from .store import RelationStore
from .trust import update_fact_trust, update_prediction_trust
from .types import EdgeDelta, RelationState, ScoredClaim, TypedMessageContent

FIELD = "typed_relations"


def active_relations(data, tick):
    return RelationStore(RelationState.model_validate(data[FIELD])).advance(tick)


def project_relations(state: RelationState, actor: str, tick: int) -> list[ObservationFact]:
    store = RelationStore(state)
    values = {"outgoing": list(store.neighbors(actor)),
        "incoming": sorted(e.source_id for e in state.communication if e.target_id == actor),
        "fact_trust": [t.model_dump() for t in state.fact_trust if t.source_id == actor],
        "prediction_trust": [t.model_dump() for t in state.prediction_trust if t.source_id == actor],
        "memberships": [m.model_dump() for m in visible_memberships(state, actor)],
        "perceived_similarity": [s.model_dump() for s in state.similarities if s.observer_id == actor and s.available_tick <= tick]}
    facts = [ObservationFact(fact_id="relations:" + actor, category="market", owner_id=actor,
                            visible_at={"tick": tick, "phase": "OBSERVE"}, value=values)]
    for evidence in state.evidence:
        if evidence.observer_id == actor and actor in evidence.visible_to:
            facts.append(ObservationFact(fact_id=evidence.evidence_id, category="market", owner_id=actor,
                visible_at={"tick": max(evidence.matured_tick, evidence.visible_tick), "phase": "OBSERVE"},
                value=evidence.model_dump()))
    return facts


class RelationScenario:
    """Compose relation initialization/projection with an existing economic scenario."""

    def __init__(self, base, initial: RelationState):
        self.base, self.initial = base, RelationState.model_validate(initial)
        if self.initial.tick != 0 or self.initial.exposures or self.initial.applied:
            raise ValueError("New relation world requires an uncommitted initial state")
        if FIELD in base.owned_fields:
            raise ValueError("Relation state already has a scenario owner")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def initialize(self):
        data = self.base.initialize()
        if FIELD in data:
            raise ValueError("Duplicate relation state")
        return {**data, FIELD: self.initial.model_dump()}

    def project_observation(self, context, actor_id):
        return [*self.base.project_observation(context, actor_id),
                *project_relations(active_relations(context.data, context.tick), actor_id, context.tick)]

    def validate(self, context):
        self.base.validate(context)
        state = RelationState.model_validate(context.data[FIELD])
        if set(state.actors) != set(context.actors) or state.tick != context.tick:
            raise ValueError("Relation actors/clock disagree with world")


class RelationIntentCompiler(IntentCompiler):
    def compile_messages(self, bundle, tick):
        for draft in bundle.message_drafts:
            body = TypedMessageContent.model_validate(draft.body)
            if draft.recipient_id not in body.audience:
                raise ValueError("Draft recipient is outside its audience")
        return super().compile_messages(bundle, tick)


class RelationObserver(ObservationBuilder):
    effect_spec = MechanismSpec(mechanism_id="typed_relations_feedback", version="1", phase="FEEDBACK", reads=[FIELD], writes=[FIELD])

    def __init__(self, rng):
        self.rng = rng

    def select_messages(self, actor_id, snapshot, messages):
        return select_messages(active_relations(snapshot.data, snapshot.tick), messages, actor_id, snapshot.tick)

    def prepare_commit(self, snapshot, observations):
        state = active_relations(snapshot.data, snapshot.tick)
        if state.feedback_tick == snapshot.tick:
            return {}
        by_actor = {o.actor_id: o for o in observations}
        if len(by_actor) != len(observations) or set(by_actor) != set(state.actors):
            raise ValueError("Exposure commit requires one observation per actor")
        events = {e.exposure_id: e for e in state.exposures}
        for actor in sorted(by_actor):
            actual = by_actor[actor]
            expected = select_messages(state, snapshot.messages, actor, snapshot.tick)
            if sorted(m.message_id for m in actual.delivered_messages) != sorted(m.message_id for m in expected):
                raise ValueError("Exposure must describe the actual frozen observation")
            for event in exposure_events(state, snapshot.messages, actor, snapshot.tick):
                if event.exposure_id in events and event != events[event.exposure_id]:
                    raise ValueError("Conflicting exposure identity")
                events[event.exposure_id] = event
        state = state.model_copy(update={"exposures": sorted(events.values(), key=lambda e: (e.tick, e.actor_id, e.rank))})
        validate_messages(state, snapshot.messages)
        roots = {TypedMessageContent.model_validate(m.body).root_claim_id: TypedMessageContent.model_validate(m.body)
                 for m in snapshot.messages}
        for evidence in state.evidence:
            if evidence.kind == "prediction" and evidence.root_claim_id in roots:
                if roots[evidence.root_claim_id].probability != evidence.probability:
                    raise ValueError("Prediction resolution must score the original probability")
        deltas = []
        scored = list(state.scored_claims)
        used = {(s.observer_id, s.root_claim_id, s.kind) for s in scored}
        if state.config.trust_enabled:
            for trust in (*state.fact_trust, *state.prediction_trust):
                observation = by_actor[trust.source_id]
                evidence = [e for e in state.evidence if e.evidence_id in observation.evidence_ids
                            and (e.observer_id, e.root_claim_id, e.kind) not in used]
                seen = {(e.root_claim_id, e.author_id, e.claim_kind) for e in state.exposures if e.actor_id == trust.source_id}
                if trust.kind == "fact_trust":
                    updated = update_fact_trust(trust, evidence, tick=state.tick, decay=state.config.fact_decay, seen=seen)
                else:
                    updated = update_prediction_trust(trust, evidence, tick=state.tick, alpha=state.config.prediction_alpha, seen=seen)
                if updated != trust:
                    for root in sorted(set(updated.used_roots) - set(trust.used_roots)):
                        kind = "fact" if trust.kind == "fact_trust" else "prediction"
                        scored.append(ScoredClaim(observer_id=trust.source_id, author_id=trust.target_id,
                                                  root_claim_id=root, kind=kind, tick=state.tick))
                        used.add((trust.source_id, root, kind))
                    deltas.append(EdgeDelta(delta_id="trust:" + content_hash([trust.kind, trust.source_id, trust.target_id, state.tick]),
                        proposed_tick=state.tick, effective_tick=state.tick + 1, layer=trust.kind,
                        source_id=trust.source_id, target_id=trust.target_id, operation="upsert", value=updated))
        plan = propose_rewiring(state, self.rng, state.similarities)
        state = RelationStore(state).propose([*deltas, *plan.deltas], tick=state.tick)
        return {FIELD: state.model_copy(update={"feedback_tick": snapshot.tick, "scored_claims": scored,
                                              "rewiring_trace": [*state.rewiring_trace, *plan.traces]}).model_dump()}
