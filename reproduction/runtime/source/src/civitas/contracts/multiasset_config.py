from typing import Literal
from pydantic import Field, model_validator
from .base import Id, NonNegative, Positive, Record, require_unique
from .population import PopulationSpec
from .portfolio import FeeSchedule


class FeeChange(Record):
    event_id: Id
    tick: Positive
    fees: FeeSchedule


class MultiAssetRunConfig(Record):
    schema_version: Literal[3] = 3
    scenario: Literal['multiasset_accounting'] = 'multiasset_accounting'
    backend: Literal['fixedpoint_multiasset'] = 'fixedpoint_multiasset'
    mechanism_version: Literal['accounting-orders-v1'] = 'accounting-orders-v1'
    root_seed: NonNegative = 42
    replica: NonNegative = 0
    ticks: Positive = 300
    concurrency: Positive = Field(default=8,le=64)
    population: PopulationSpec
    fees: FeeSchedule = Field(default_factory=FeeSchedule)
    order_ttl: Positive = 3
    ticks_per_day: Positive = 20
    max_order_quantity: Positive = 5
    atomic_bundles: bool = False
    fee_changes: list[FeeChange] = Field(default_factory=list)

    @model_validator(mode='after')
    def changes(self):
        require_unique([e.event_id for e in self.fee_changes],'fee event ID')
        if any(e.tick>self.ticks for e in self.fee_changes):
            raise ValueError('Fee changes must be within run')
        return self
