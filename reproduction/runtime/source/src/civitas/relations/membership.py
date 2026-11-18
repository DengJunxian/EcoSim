from .types import RelationState


def visible_memberships(state: RelationState, actor: str) -> tuple:
    if actor not in state.actors:
        raise ValueError("Unknown membership observer")
    joined = {m.target_id for m in state.memberships if m.source_id == actor}
    public_groups = {g.group_id for g in state.groups if g.public}
    return tuple(sorted((m for m in state.memberships if m.source_id == actor
        or (m.visibility == "public" and m.target_id in public_groups)
        or (m.visibility == "members" and m.target_id in joined)), key=lambda m: (m.target_id, m.source_id)))


def public_peers(state: RelationState, actor: str) -> set[str]:
    visible = visible_memberships(state, actor)
    public_groups = {g.group_id for g in state.groups if g.public}
    joined = {m.target_id for m in state.memberships if m.source_id == actor} & public_groups
    return {m.source_id for m in visible if m.target_id in joined and m.source_id != actor}
