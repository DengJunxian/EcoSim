"""Explicit opt-in assembly; the old CLI and default policies remain untouched."""
from . import build_world
from civitas.cognition.observations import CognitionObservationBuilder


def build_cognitive_world(config, *, policy, **kwargs):
    world = build_world(config, policy=policy, participants=(policy.memory,), observer=CognitionObservationBuilder(), **kwargs)

    def record(receipt):
        if receipt.attempt_id not in {r.attempt_id for r in world.invocations}:
            world.record_invocation(receipt)

    policy.runtime.on_receipt = record
    return world
