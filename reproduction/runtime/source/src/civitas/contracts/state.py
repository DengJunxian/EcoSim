"""Complete logical tick state, separate from external invocation facts."""
from typing import Any, Literal

from pydantic import Field, model_validator

from .base import Digest, Id, NonNegative, Record, content_hash, require_unique
from .decisions import DecisionBundle
from .events import Event
from .financial import FinancialDecisionBundle, FinancialExecutionReceipt
from .portfolio import PortfolioDecisionBundle, PortfolioExecutionReceipt
from .observations import Message, ObservationFact
from .receipts import ExecutionReceipt
from .time import SimTime


class TickContext(Record):
    tick: NonNegative
    world_version: NonNegative
    execution_seq: NonNegative
    actors: list[Id]
    data: dict[str, Any]


class RelationDelta(Record):
    delta_id: Id
    relation_id: Id
    source_id: Id
    target_id: Id
    layer: Id
    effective_at: SimTime
    value: dict[str, Any]


class Feedback(Record):
    data: dict[str, Any]
    memories: list[ObservationFact] = Field(default_factory=list)
    relations: list[RelationDelta] = Field(default_factory=list)


class ExecutionResult(Record):
    data: dict[str, Any]
    receipt: ExecutionReceipt | FinancialExecutionReceipt | PortfolioExecutionReceipt


class BatchExecutionResult(Record):
    data: dict[str, Any]
    receipts: list[ExecutionReceipt | FinancialExecutionReceipt | PortfolioExecutionReceipt]


class WorldSnapshot(TickContext):
    schema_version: Literal[1] = 1
    pending_events: list[Event] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    decisions: list[DecisionBundle] = Field(default_factory=list)
    receipts: list[ExecutionReceipt] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    memories: list[ObservationFact] = Field(default_factory=list)
    beliefs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    relations: dict[str, RelationDelta] = Field(default_factory=dict)
    pending_relations: list[RelationDelta] = Field(default_factory=list)
    used_ids: list[Id] = Field(default_factory=list)

    @model_validator(mode="after")
    def identities(self):
        require_unique(self.actors, "actor ID")
        require_unique(self.used_ids, "world ID")
        for field, key in ((self.events + self.pending_events, "event_id"), (self.decisions, "decision_id"),
                           (self.receipts, "command_id"), (self.messages, "message_id"),
                           (self.memories, "fact_id"), (self.pending_relations, "delta_id")):
            require_unique([getattr(item, key) for item in field], key)
        if not set(self.beliefs) <= set(self.actors):
            raise ValueError("Unknown belief owner")
        return self


class PreparedTick(Record):
    base_version: NonNegative
    tick: NonNegative
    snapshot: WorldSnapshot
    state_hash: Digest
    invocation_receipt_refs: list[Id] = Field(default_factory=list)

    @model_validator(mode="after")
    def identity(self):
        if (self.snapshot.world_version != self.base_version + 1 or self.snapshot.tick != self.tick
                or self.state_hash != content_hash(self.snapshot)):
            raise ValueError("Prepared tick version/hash mismatch")
        return self


class FinancialWorldSnapshot(WorldSnapshot):
    schema_version: Literal[2] = 2
    decisions: list[FinancialDecisionBundle] = Field(default_factory=list)
    receipts: list[FinancialExecutionReceipt] = Field(default_factory=list)


class FinancialPreparedTick(PreparedTick):
    snapshot: FinancialWorldSnapshot


class PortfolioWorldSnapshot(WorldSnapshot):
    schema_version: Literal[3] = 3
    decisions: list[PortfolioDecisionBundle] = Field(default_factory=list)
    receipts: list[PortfolioExecutionReceipt] = Field(default_factory=list)


class PortfolioPreparedTick(PreparedTick):
    snapshot: PortfolioWorldSnapshot
