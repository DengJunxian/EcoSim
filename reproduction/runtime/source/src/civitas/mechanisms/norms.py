"""Optional normalized action distance; a norm contributes exactly one cost term."""
import math
from pydantic import Field, model_validator

from civitas.contracts.base import Id, Record, require_unique
from .outcomes import MASS_TOLERANCE


class NormalizedAction(Record):
    values: dict[Id, float]

    @model_validator(mode='after')
    def bounded(self):
        if any(not -1.0 <= x <= 1.0 for x in self.values.values()):
            raise ValueError('Normalized actions must lie in [-1,1]')
        return self


class GroupNorm(Record):
    group_id: Id
    weight: float = Field(ge=0.0, le=1.0)
    target: dict[Id, float]
    evidence_id: Id

    @model_validator(mode='after')
    def action_space(self):
        NormalizedAction(values=self.target)
        return self


def norm_distance(action, groups):
    action = NormalizedAction(values=action).values
    groups = [GroupNorm.model_validate(g) for g in groups]
    require_unique([g.group_id for g in groups], 'norm group')
    if not groups:
        return 0.0
    if not action or any(set(g.target) != set(action) for g in groups):
        raise ValueError('Norm and action dimensions must match')
    if not math.isclose(math.fsum(g.weight for g in groups), 1.0, rel_tol=0, abs_tol=MASS_TOLERANCE):
        raise ValueError('Membership weights must sum to one')
    return math.fsum(g.weight * math.fsum(abs(action[a] - g.target[a]) for a in sorted(action)) / (2*len(action)) for g in groups)
