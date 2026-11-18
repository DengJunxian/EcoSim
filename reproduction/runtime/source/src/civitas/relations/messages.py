import math

from civitas.contracts.base import content_hash, require_unique
from civitas.contracts.observations import Message
from civitas.contracts.time import SimTime
from .store import RelationStore
from .types import ExposureEvent, RelationState, TypedMessageContent


def validate_messages(state: RelationState, messages) -> None:
    known = {m.message_id: Message.model_validate(m) for m in messages}
    require_unique([m.message_id for m in messages], "message")
    roots = {}
    seen = {(e.actor_id, e.message_id): e.tick for e in state.exposures}
    for message in sorted(known.values(), key=lambda m: (m.created_at.key, m.message_id)):
        body = TypedMessageContent.model_validate(message.body)
        if not {message.sender_id, message.recipient_id, body.author_id, *body.audience} <= set(state.actors):
            raise ValueError("Unknown message actor")
        if message.recipient_id not in body.audience or message.visible_at.tick <= message.created_at.tick:
            raise ValueError("Message needs an authorized audience and t+1 delivery")
        identity = (body.author_id, body.content_ref, body.claim_kind, body.signal, body.probability)
        if body.root_claim_id in roots and identity != roots[body.root_claim_id]:
            raise ValueError("Conflicting root claim")
        roots[body.root_claim_id] = identity
        if body.parent_message_id is None:
            if body.author_id != message.sender_id:
                raise ValueError("Root author must be its sender")
        else:
            parent = known.get(body.parent_message_id)
            if parent is None or parent.visible_at.key >= message.created_at.key:
                raise ValueError("Message lineage needs an earlier delivered parent")
            original = TypedMessageContent.model_validate(parent.body)
            if (original.root_claim_id, original.author_id, original.content_ref, original.claim_kind,
                original.signal, original.probability) != (body.root_claim_id, *identity):
                raise ValueError("Forward lineage must preserve root claim")
            seen_at = seen.get((message.sender_id, parent.message_id))
            if parent.recipient_id != message.sender_id or seen_at is None or seen_at > message.created_at.tick:
                raise ValueError("A forwarded parent must have been actually seen")


def message_weight(state, body, actor):
    if not state.config.trust_enabled:
        return 1.0
    if body.claim_kind == "prediction":
        trust = next((t for t in state.prediction_trust if (t.source_id, t.target_id) == (actor, body.author_id)), None)
        return 1.0 - trust.brier_ewma if trust else 0.75
    return RelationStore(state).trust_weight(actor, body.author_id)


def select_messages(state: RelationState, messages, actor: str, tick: int) -> list[Message]:
    store = RelationStore(state)
    store.neighbors(actor)
    now = SimTime(tick=tick, phase="OBSERVE")
    seen = {(e.actor_id, e.message_id): e.tick for e in state.exposures}
    eligible = []
    for raw in messages:
        message = Message.model_validate(raw)
        body = TypedMessageContent.model_validate(message.body)
        if (message.recipient_id != actor or actor not in body.audience or message.visible_at.key > now.key
                or message.created_at.tick >= tick or (message.expires_at and message.expires_at.key <= now.key)
                or seen.get((actor, message.message_id), tick) < tick):
            continue
        if store.can_deliver(message.sender_id, actor):
            eligible.append(message)
    eligible.sort(key=lambda m: (-message_weight(state, TypedMessageContent.model_validate(m.body), actor),
                                 m.visible_at.key, m.message_id))
    return eligible[:state.config.attention_budget]


def exposure_events(state: RelationState, messages, actor: str, tick: int) -> list[ExposureEvent]:
    events = []
    for rank, message in enumerate(select_messages(state, messages, actor, tick)):
        body = TypedMessageContent.model_validate(message.body)
        events.append(ExposureEvent(exposure_id="exposure:" + content_hash([actor, message.message_id]),
            message_id=message.message_id, root_claim_id=body.root_claim_id, actor_id=actor,
            author_id=body.author_id, sender_id=message.sender_id, tick=tick, rank=rank,
            weight=message_weight(state, body, actor), claim_kind=body.claim_kind))
    return events


def weighted_signal(state: RelationState, messages, actor: str, tick: int) -> float:
    selected = select_messages(state, messages, actor, tick)
    pairs = [(message_weight(state, TypedMessageContent.model_validate(m.body), actor),
              TypedMessageContent.model_validate(m.body).signal) for m in selected]
    total = math.fsum(w for w, _ in pairs)
    return math.fsum(w * signal for w, signal in pairs) / total if total else 0.0
