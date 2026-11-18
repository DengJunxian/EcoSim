import math

from pydantic import Field

from civitas.contracts.base import Record, content_hash
from .membership import public_peers
from .store import RelationStore
from .types import CommunicationEdge, EdgeDelta, PerceivedSimilarity, RelationState, RewireTrace


class RewirePlan(Record):
    deltas: list[EdgeDelta] = Field(default_factory=list)
    traces: list[RewireTrace] = Field(default_factory=list)


def propose_rewiring(state: RelationState, rng, similarities=()) -> RewirePlan:
    state = RelationState.model_validate(state)
    config, tick = state.config.rewiring, state.tick
    if not config.enabled or tick % config.period or not config.max_changes:
        return RewirePlan()
    perceived = {}
    for raw in similarities:
        s = PerceivedSimilarity.model_validate(raw)
        if s.observer_id not in state.actors or s.target_id not in state.actors:
            raise ValueError("Unknown perceived-similarity actor")
        if s.available_tick <= tick:
            key = (s.observer_id, s.target_id)
            if key in perceived:
                raise ValueError("Duplicate perceived similarity")
            perceived[key] = s.value
    store = RelationStore(state)
    incoming = {a: {} for a in state.actors}
    for edge in state.communication:
        incoming[edge.target_id][edge.source_id] = edge
    actors, deltas, traces = sorted(state.actors), [], []
    for actor in actors:
        history = [d.effective_tick for d in state.applied if d.layer == "communication" and d.target_id == actor]
        if (history and tick - max(history) < config.cooldown) or any(
                d.layer == "communication" and d.target_id == actor for d in state.pending):
            continue
        current = dict(incoming[actor])
        changed = 0

        def consider(source, origin, operation, cost):
            score = math.fsum([config.intercept, config.trust_coefficient * store.trust_weight(actor, source),
                config.similarity_coefficient * perceived.get((actor, source), 0.0), -config.cost_coefficient * cost,
                config.exploration_bonus if origin == "exploration" else 0.0])
            probability = 1.0 / (1.0 + math.exp(-score)) if score >= 0 else math.exp(score) / (1.0 + math.exp(score))
            draw = rng.uniform("relations.rewiring", actor, tick, f"{operation}:{source}")
            selected = draw < probability if operation == "upsert" else draw > probability
            traces.append(RewireTrace(observer_id=actor, candidate_id=source, tick=tick, origin=origin,
                probability=probability, draw=draw, selected=selected, operation=operation))
            if selected:
                deltas.append(EdgeDelta(delta_id="rewire:" + content_hash([actor, source, tick, operation]),
                    proposed_tick=tick, effective_tick=tick + 1, layer="communication", source_id=source,
                    target_id=actor, operation=operation,
                    value=CommunicationEdge(source_id=source, target_id=actor, cost=cost) if operation == "upsert" else None))
            return selected

        for source in sorted(current):
            if changed == config.max_changes:
                break
            if consider(source, "existing", "delete", current[source].cost):
                changed += 1
                del current[source]
        candidates = {source: "introduction" for friend in incoming[actor] for source in incoming[friend]
                      if source != actor and source not in incoming[actor]}
        for source in sorted(public_peers(state, actor)):
            if source not in incoming[actor]:
                candidates.setdefault(source, "public_group")
        for slot in range(config.exploration_candidates):
            source = actors[rng.u64("relations.exploration", actor, tick, str(slot)) % len(actors)]
            if source != actor and source not in incoming[actor]:
                candidates.setdefault(source, "exploration")
        for source in sorted(candidates):
            if changed == config.max_changes or len(current) >= config.max_degree:
                break
            if consider(source, candidates[source], "upsert", config.new_edge_cost):
                current[source] = None
                changed += 1
    return RewirePlan(deltas=deltas, traces=traces)
