"""The only composition root importing both the kernel and concrete scenarios."""
from civitas.contracts.config import RunConfig
from civitas.contracts.decisions import Command, DecisionBundle
from civitas.contracts.financial import FinancialCommand, FinancialDecisionBundle
from civitas.contracts.legacy_config import LegacyRunConfig
from civitas.contracts.multiasset_config import MultiAssetRunConfig
from civitas.contracts.portfolio import PortfolioCommand, PortfolioDecisionBundle
from pydantic import TypeAdapter
from civitas.core.observation import ObservationBuilder
from civitas.core.registry import MechanismRegistry
from civitas.core.rng import AddressedRNG
from civitas.core.world import WorldKernel
from civitas.scenarios.toy_transfer import ToyTransfer, TransferPolicy


class IntentCompiler:
    def compile_messages(self, bundle, tick):
        from civitas.core.messages import compile_messages
        return compile_messages(bundle, tick)

    def compile(self, bundle: DecisionBundle, arrival_key: int) -> list[Command]:
        command_type, extra = Command, {}
        if isinstance(bundle, FinancialDecisionBundle):
            command_type = PortfolioCommand if isinstance(bundle, PortfolioDecisionBundle) else FinancialCommand
            extra = {key: getattr(bundle, key) for key in ("tick", "batch_id", "observation_version", "world_version")}
        return [command_type(**extra, command_id=f"command:{intent.intent_id}", decision_id=bundle.decision_id,
                        actor_id=bundle.actor_id, local_order_index=index, arrival_key=arrival_key, intent=intent)
                for index, intent in enumerate(bundle.economic_intents)]


def build_world(config: RunConfig, *, policy=None, scenario=None, mechanisms=(), compositors=None, phase_hook=None,
                participants=(), observer=None) -> WorldKernel:
    config = TypeAdapter(RunConfig | LegacyRunConfig | MultiAssetRunConfig).validate_python(config)
    observer = observer if observer is not None else ObservationBuilder()
    if isinstance(config, MultiAssetRunConfig):
        from civitas.scenarios.multiasset_accounting import MultiAssetAccounting, AccountingPolicy
        scenario = scenario if scenario is not None else MultiAssetAccounting(config)
        participants = (*scenario.participants, *participants)
        return WorldKernel(scenario=scenario, policy=policy if policy is not None else AccountingPolicy(config),
            compiler=IntentCompiler(), observer=observer,
            registry=MechanismRegistry(list(mechanisms), compositors,
                scenario.owned_fields | frozenset(p.participant_id for p in participants)),
            rng=AddressedRNG(config.root_seed, config.replica), actors=scenario.actors, events=scenario.events,
            concurrency=config.concurrency, phase_hook=phase_hook, participants=participants, schema_version=3)
    if isinstance(config, LegacyRunConfig):
        from civitas.scenarios.legacy_market import LegacyMarket, LegacyPolicy
        scenario = scenario if scenario is not None else LegacyMarket(config)
        participants = (*scenario.participants, *participants)
        return WorldKernel(scenario=scenario, policy=policy if policy is not None else LegacyPolicy(),
            compiler=IntentCompiler(), observer=observer,
            registry=MechanismRegistry(list(mechanisms), compositors,
                scenario.owned_fields | frozenset(p.participant_id for p in participants)),
            rng=AddressedRNG(config.root_seed, config.replica), actors=scenario.runner.ids, events=scenario.events,
            concurrency=config.concurrency, phase_hook=phase_hook, participants=participants,
            schema_version=2, arrival_key=lambda actor: int(actor.removeprefix("agent_")))
    scenario = scenario if scenario is not None else ToyTransfer(config)
    return WorldKernel(scenario=scenario, policy=policy if policy is not None else TransferPolicy(config),
                       compiler=IntentCompiler(), observer=observer,
                       registry=MechanismRegistry(list(mechanisms), compositors,
                           scenario.owned_fields | frozenset(p.participant_id for p in participants)),
                       rng=AddressedRNG(config.root_seed, config.replica), actors=list(config.initial_balances),
                       events=list(config.events), concurrency=config.concurrency, phase_hook=phase_hook,
                       participants=participants)
