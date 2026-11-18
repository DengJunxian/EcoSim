"""Explicit composition root for the WP07 scenario."""
from civitas.relations.integration import RelationIntentCompiler
from civitas.core.registry import MechanismRegistry
from civitas.core.rng import AddressedRNG
from civitas.core.world import WorldKernel
from civitas.scenarios.market_society import MarketSociety
from civitas.scenarios.society_config import MarketSocietyConfig
from civitas.scenarios.society_policy import SocietyPolicy
from civitas.scenarios.society_relations import SocietyObserver
from civitas.scenarios.society_cognition import RecentMemoryBook, SocietyHybrid, TraceBook
from civitas.scenarios.society_policy import LegacySocietyBehavior, SocietyBehavior
from civitas.cognition.policy import PolicyConfig
from civitas.contracts.financial import FinancialDecisionBundle
from civitas.contracts.portfolio import PortfolioDecisionBundle


def build_market_society(config, *, policy=None, phase_hook=None, runtime=None):
    config = MarketSocietyConfig.model_validate(config)
    scenario = MarketSociety(config)
    traces = TraceBook(config,scenario.actors)
    participants = (*scenario.participants,traces)
    slow = None
    if config.cognition_mode!='rule':
        if runtime is None or runtime.config.source!=config.cognition_mode or config.slow_agents==0:
            raise ValueError('Cognition mode requires a matching runtime and designated slow actors')
        memory = RecentMemoryBook(scenario.actors[:config.slow_agents],config.memory_window)
        numeric = (LegacySocietyBehavior if scenario.legacy else SocietyBehavior)(config,model_owned=True)
        slow = SocietyHybrid(runtime,memory,fast_rule=numeric,
            bundle_model=FinancialDecisionBundle if scenario.legacy else PortfolioDecisionBundle,
            config=PolicyConfig(period=config.slow_period,reflection_period=config.reflection_period,
                prompt_version='wp07-society-json-v9',
                max_horizon=config.behavior.joint_outcomes.scenarios[0].horizon_ticks),
            scenarios={s.scenario_id:s.asset_returns for s in config.behavior.joint_outcomes.scenarios})
        participants = (*participants,memory)
    elif runtime is not None:
        raise ValueError('Rule mode cannot silently use a model runtime')
    world = WorldKernel(scenario=scenario,policy=policy or SocietyPolicy(config,scenario.actors,slow=slow,traces=traces),compiler=RelationIntentCompiler(),
        observer=SocietyObserver(AddressedRNG(config.root_seed,config.replica),config.memory_window),registry=MechanismRegistry([],None,
            scenario.owned_fields | frozenset(p.participant_id for p in participants)),
        rng=AddressedRNG(config.root_seed,config.replica),actors=scenario.actors,events=scenario.events,
        concurrency=config.market.concurrency,phase_hook=phase_hook,participants=participants,
        schema_version=2 if scenario.legacy else 3,
        arrival_key=(lambda actor:int(actor.removeprefix('agent_'))) if config.arrival_order=='actor_id' else None)
    if runtime is not None:
        def record(receipt):
            if receipt.attempt_id not in {r.attempt_id for r in world.invocations}:
                world.record_invocation(receipt)
        runtime.on_receipt = record
    return world
