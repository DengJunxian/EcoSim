"""Composition only: paper method, existing transaction/market/storage facilities."""
from civitas.application import IntentCompiler
from civitas.cognition.kdd27_policy import PaperObserver, PaperPolicy
from civitas.core.registry import MechanismRegistry
from civitas.core.rng import AddressedRNG
from civitas.core.world import WorldKernel
from civitas.scenarios.kdd27 import KDD27Scenario
from civitas.scenarios.kdd27_config import parse_paper_config


def build_kdd27(config, *, runtime=None, policy=None, phase_hook=None):
    config=parse_paper_config(config)
    scenario=KDD27Scenario(config)
    world=WorldKernel(scenario=scenario,policy=policy or PaperPolicy(config,runtime),compiler=IntentCompiler(),
        observer=PaperObserver(config),registry=MechanismRegistry([],None,scenario.owned_fields|{'multiasset_market'}),
        rng=AddressedRNG(config.root_seed,config.replica),actors=scenario.actors,events=scenario.events,
        concurrency=config.concurrency,participants=scenario.participants,phase_hook=phase_hook,schema_version=3)
    if runtime is not None:
        def record(receipt):
            if receipt.attempt_id not in {r.attempt_id for r in world.invocations}:
                world.record_invocation(receipt)
        runtime.on_receipt=record
    return world
