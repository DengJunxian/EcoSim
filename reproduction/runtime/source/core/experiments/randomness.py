"""Pure addressed SHA-256 / binary64 Box-Muller randomness.

Queries have no mutable stream, cached normal component or hidden audit counter.
Callers explicitly persist RandomDraw receipts when a mechanism consumes them.
Same keys give the same values independently of policy, workers and query order.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from core.experiments.identity import HASH_PATTERN, canonical_research_bytes, research_hash
from core.experiments.strict_types import ResearchInputError, StrictResearchModel

ALGORITHM = "sha256-boxmuller-v1"
CHANNEL_REGISTRY_VERSION = "research-channels-v2"
# This registry is part of the implementation identity, not an extra RandomSpec
# field. A channel is one mechanism and one declared distribution family.
CHANNEL_REGISTRY = MappingProxyType({
    "world_wealth": ("initial_world", "normal"),
    "world_cash_fraction": ("initial_world", "uniform"),
    "world_reference_anchor": ("initial_world", "uniform"),
    "private_signal": ("information", "normal"),
    "scenario_target": ("information", "uniform"),
    "social_transition": ("social", "uniform"),
    "social_delivery": ("social", "uniform"),
    "participation": ("decision", "uniform"),
    "forecast_noise": ("decision", "normal"),
    "submission_priority": ("submission", "uniform"),
})
MarketPhase = Literal["initial_world", "information", "social", "decision", "submission"]
Distribution = Literal["uniform", "normal"]
DistributionDomain = Literal["u", "n", "n/u1", "n/u2"]


class RandomSpec(StrictResearchModel):
    """The eight frozen fields from critical_records.schema.json, defined once."""
    schema_version: Literal[1] = 1
    algorithm: Literal["sha256-boxmuller-v1"] = ALGORITHM
    root_seed: int = Field(ge=0, le=2**32 - 1)
    exogenous_family_id: str = Field(min_length=1)
    replication_id: str = Field(min_length=1)
    coupling_spec_hash: str = Field(pattern=HASH_PATTERN)
    channels: list[str] = Field(min_length=1)
    audit_mode: Literal["full", "channel_step_digest"] = "full"

    @model_validator(mode="after")
    def _registered_channels(self):
        if not self.exogenous_family_id.strip() or not self.replication_id.strip():
            raise ValueError("Random namespaces must be nonempty stable IDs")
        if len(self.channels) != len(set(self.channels)) or set(self.channels) - set(CHANNEL_REGISTRY):
            raise ValueError("Random channels must be unique names in the frozen registry")
        return self


class DrawKey(StrictResearchModel):
    algorithm: Literal["sha256-boxmuller-v1"] = ALGORITHM
    root_seed: int = Field(ge=0, le=2**32 - 1)
    exogenous_family_id: str = Field(min_length=1)
    replication_id: str = Field(min_length=1)
    coupling_spec_hash: str = Field(pattern=HASH_PATTERN)
    market_phase: MarketPhase
    step: int = Field(ge=0)
    entity_or_event_id: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    slot: int = Field(ge=0)
    distribution_domain: DistributionDomain

    @model_validator(mode="after")
    def _registered_address(self):
        if any(not value.strip() for value in (self.exogenous_family_id, self.replication_id, self.entity_or_event_id)):
            raise ValueError("Random addresses require stable nonempty IDs")
        registered = CHANNEL_REGISTRY.get(self.channel)
        if registered is None or registered[0] != self.market_phase:
            raise ValueError("Random market_phase must be derived from its registered channel")
        if (registered[1] == "uniform") != (self.distribution_domain == "u"):
            raise ValueError("Distribution domain conflicts with the registered channel")
        if self.market_phase == "initial_world" and self.step != 0:
            raise ValueError("Initial-world addresses use step 0")
        return self


def _uniform_component(key: DrawKey) -> tuple[str, float]:
    if key.distribution_domain == "n":
        raise ResearchInputError("Normal result keys have two independent component domains")
    digest = hashlib.sha256(canonical_research_bytes(key)).digest()
    k = int.from_bytes(digest[:8], "big") >> 12
    value = (k + 0.5) * 2.0**-52
    return digest.hex(), value


class UniformComponent(StrictResearchModel):
    key: DrawKey
    sha256: str = Field(pattern=HASH_PATTERN)
    value: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _recompute(self):
        digest, expected = _uniform_component(self.key)
        if self.sha256 != digest or self.value != expected:
            raise ValueError("Uniform component must match its actual SHA-256 address")
        return self


class RandomDraw(StrictResearchModel):
    key: DrawKey
    value: float
    components: list[UniformComponent]

    @model_validator(mode="after")
    def _recompute(self):
        if self.key.distribution_domain == "u":
            if len(self.components) != 1 or self.components[0].key != self.key or self.value != self.components[0].value:
                raise ValueError("Uniform receipt must retain its single addressed component")
        elif self.key.distribution_domain == "n":
            if len(self.components) != 2:
                raise ValueError("Normal receipt needs the independent n/u1 and n/u2 components")
            for component, domain in zip(self.components, ("n/u1", "n/u2")):
                if component.key != self.key.model_copy(update={"distribution_domain": domain}):
                    raise ValueError("Normal component changed its address or distribution domain")
            u1, u2 = (component.value for component in self.components)
            expected = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            if self.value != expected:
                raise ValueError("Normal receipt disagrees with the frozen Box-Muller cosine component")
        else:
            raise ValueError("A draw receipt uses u or n, not an internal normal component domain")
        return self


@dataclass(frozen=True)
class KeyedRandomSource:
    spec: RandomSpec

    def __post_init__(self):
        if not isinstance(self.spec, RandomSpec):
            raise TypeError("KeyedRandomSource requires the unique validated RandomSpec")

    def key(self, *, step: int, entity_id: str, channel: str, draw_slot: int = 0,
            distribution_domain: DistributionDomain = "u") -> DrawKey:
        if channel not in self.spec.channels:
            raise ResearchInputError("Requested random channel was not enabled in RandomSpec")
        phase = CHANNEL_REGISTRY[channel][0]
        return DrawKey(algorithm=self.spec.algorithm, root_seed=self.spec.root_seed,
            exogenous_family_id=self.spec.exogenous_family_id, replication_id=self.spec.replication_id,
            coupling_spec_hash=self.spec.coupling_spec_hash, market_phase=phase, step=step,
            entity_or_event_id=entity_id, channel=channel, slot=draw_slot,
            distribution_domain=distribution_domain)

    def draw_receipt(self, *, step: int, entity_id: str, channel: str, draw_slot: int = 0,
                     distribution: Distribution) -> RandomDraw:
        if distribution not in ("uniform", "normal"):
            raise ResearchInputError("Unknown random distribution")
        key = self.key(step=step, entity_id=entity_id, channel=channel, draw_slot=draw_slot,
                       distribution_domain="u" if distribution == "uniform" else "n")
        domains = ("u",) if distribution == "uniform" else ("n/u1", "n/u2")
        components = []
        for domain in domains:
            component_key = key.model_copy(update={"distribution_domain": domain})
            digest, value = _uniform_component(component_key)
            components.append(UniformComponent(key=component_key, sha256=digest, value=value))
        value = components[0].value if distribution == "uniform" else (
            math.sqrt(-2.0 * math.log(components[0].value)) * math.cos(2.0 * math.pi * components[1].value))
        return RandomDraw(key=key, value=value, components=components)

    def uniform(self, *, step: int, entity_id: str, channel: str, draw_slot: int = 0) -> float:
        return self.draw_receipt(step=step, entity_id=entity_id, channel=channel,
                                 draw_slot=draw_slot, distribution="uniform").value

    def normal(self, *, step: int, entity_id: str, channel: str, draw_slot: int = 0) -> float:
        return self.draw_receipt(step=step, entity_id=entity_id, channel=channel,
                                 draw_slot=draw_slot, distribution="normal").value

    def uniforms(self, *, step: int, entity_ids: tuple[str, ...], channel: str) -> np.ndarray:
        if (type(entity_ids) is not tuple or any(type(entity) is not str for entity in entity_ids)
                or len(entity_ids) != len(set(entity_ids))):
            raise ResearchInputError("Batch draws require a tuple of unique stable entity IDs")
        values = [self.uniform(step=step, entity_id=entity, channel=channel) for entity in entity_ids]
        # Immutable bytes backing prevents a caller re-enabling writeability.
        return np.frombuffer(np.asarray(values, dtype=np.float64).tobytes(), dtype=np.float64)


def summarize_draws(draws: list[RandomDraw]) -> dict:
    """Explicit order-independent audit of consumed unique addresses.

    Repeating a diagnostic query is not a new exogenous draw. Full receipt
    sequences may retain all consumers; the digest groups unique addressed
    values and records that scope explicitly.
    """
    if not isinstance(draws, list) or any(not isinstance(draw, RandomDraw) for draw in draws):
        raise TypeError("Draw audit requires actual RandomDraw receipts")
    unique = {research_hash(draw.key): draw for draw in draws}
    groups = {}
    for address in sorted(unique):
        draw = unique[address]
        group = f"{draw.key.channel}/{draw.key.step}"
        groups.setdefault(group, []).append(draw.model_dump(mode="json"))
    return {key: {"unique_key_count": len(rows), "digest": research_hash(rows)}
            for key, rows in sorted(groups.items())}
