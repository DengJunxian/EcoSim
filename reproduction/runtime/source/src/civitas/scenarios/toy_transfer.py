from typing import Any, Literal

from pydantic import Field

from civitas.contracts.base import Record
from civitas.contracts.config import RunConfig
from civitas.contracts.decisions import Command, DecisionBundle, TransferIntent
from civitas.contracts.events import Event
from civitas.contracts.observations import AccountView, Observation, ObservationFact
from civitas.contracts.receipts import ExecutionReceipt
from civitas.contracts.state import BatchExecutionResult, ExecutionResult, Feedback, TickContext
from civitas.contracts.time import SimTime


class ToyNotice(Record):
    text: str = Field(min_length=1)
    public: Literal[True] = True


class ToyTransfer:
    scenario_id = "toy_transfer"
    version = "1"
    owned_fields = frozenset({"balances", "notices"})

    def __init__(self, config: RunConfig):
        self.config = config
        for event in config.events:
            if event.kind != "notice":
                raise ValueError("Toy only supports notice events")
            ToyNotice.model_validate(event.payload)

    def initialize(self) -> dict:
        return {"balances": dict(self.config.initial_balances), "notices": []}

    def boundary(self, context: TickContext) -> dict:
        return context.data

    async def prepare_observations(self, context: TickContext) -> dict:
        return context.data

    def apply_event(self, context: TickContext, event: Event) -> dict:
        ToyNotice.model_validate(event.payload)
        if event.kind != "notice":
            raise ValueError("Unknown toy event")
        data = context.model_dump()["data"]
        data["notices"].append(event.model_dump())
        return data

    def project_observation(self, context: TickContext, actor_id: str) -> list[ObservationFact]:
        facts = [ObservationFact(fact_id=f"account:{actor}", category="account", owner_id=actor,
                  visible_at=SimTime(tick=context.tick, phase="OBSERVE"),
                  value=AccountView(actor_id=actor, account_id=actor, version=context.world_version,
                                    balance=balance, reserved=0, available=balance).model_dump())
                 for actor, balance in context.data["balances"].items()]
        facts.extend(ObservationFact(fact_id=raw["event_id"], category="market", public=True,
                                    visible_at=raw["visible_at"], value=raw["payload"])
                     for raw in context.data["notices"])
        return facts

    def execute(self, context: TickContext, command: Command) -> ExecutionResult:
        data = context.model_dump()["data"]
        balances = data["balances"]
        intent = command.intent
        reason, quantity = None, 0
        if isinstance(intent, TransferIntent):
            if intent.recipient_id not in balances:
                reason = "unknown_recipient"
            elif intent.recipient_id == command.actor_id:
                reason = "self_transfer"
            elif balances[command.actor_id] < intent.amount:
                reason = "insufficient_resources"
            else:
                balances[command.actor_id] -= intent.amount
                balances[intent.recipient_id] += intent.amount
                quantity = intent.amount
        receipt = ExecutionReceipt(command_id=command.command_id, decision_id=command.decision_id,
            actor_id=command.actor_id, execution_seq=context.execution_seq + 1,
            status="rejected" if reason else "accepted", reason_codes=[reason] if reason else [],
            executed_quantity=quantity, ledger_tx_ids=[f"transfer:{command.command_id}"] if quantity else [])
        return ExecutionResult(data=data, receipt=receipt)

    def feedback(self, context: TickContext, receipts: list[ExecutionReceipt]) -> Feedback:
        return Feedback(data=context.data)

    def execute_batch(self, context: TickContext, commands: list[Command]) -> BatchExecutionResult:
        receipts = []
        for command in commands:
            result = self.execute(context, command)
            receipts.append(result.receipt)
            context = context.model_copy(update={"data": result.data, "execution_seq": context.execution_seq + 1})
        return BatchExecutionResult(data=context.data, receipts=receipts)

    def validate(self, context: TickContext) -> None:
        balances = context.data["balances"]
        if (set(balances) != set(self.config.initial_balances)
                or any(type(v) is not int or v < 0 for v in balances.values())
                or sum(balances.values()) != sum(self.config.initial_balances.values())):
            raise ValueError("Toy account membership, nonnegativity or conservation failed")

    def metrics(self, context: TickContext) -> dict[str, Any]:
        return {"total_tokens": sum(context.data["balances"].values()), "balances": dict(context.data["balances"])}


class TransferPolicy:
    policy_id = "toy_transfer_rule"
    version = "1"

    def __init__(self, config: RunConfig):
        self.actors = tuple(sorted(config.initial_balances))
        self.amount = config.transfer_amount

    async def decide(self, observation: Observation) -> DecisionBundle:
        tick, actor = observation.visible_until.tick, observation.actor_id
        recipient = next(a for a in self.actors if a != actor)
        return DecisionBundle(decision_id=f"decision:{tick}:{actor}", actor_id=actor,
            observation_id=observation.observation_id, batch_id=observation.batch_id,
            observation_version=observation.observation_version, world_version=observation.world_version,
            policy_id=self.policy_id, policy_version=self.version,
            economic_intents=[TransferIntent(intent_id=f"intent:{tick}:{actor}", recipient_id=recipient, amount=self.amount)])
