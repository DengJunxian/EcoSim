"""Snapshot the existing vectorized model without changing its update rules."""
import random
from copy import copy
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator
from core.society.network import SocialStateSnapshot, capture_social_state, restore_social_state
from civitas.contracts.base import NonNegative, Record, content_hash
from civitas.markets.legacy import json_values


class PopulationArray(Record):
    dtype: Literal["float32", "int64"]
    shape: list[NonNegative]
    values: list[float | int]

    @model_validator(mode="after")
    def dimensions(self):
        expected = float if self.dtype == "float32" else int
        if np.prod(self.shape) != len(self.values) or any(type(v) is not expected for v in self.values):
            raise ValueError("Population array shape/dtype mismatch")
        return self

    def array(self):
        return np.asarray(self.values, dtype=self.dtype).reshape(self.shape).copy()


class LegacySocietyState(Record):
    state_schema_version: Literal[1] = 1
    population: dict[str, PopulationArray]
    population_rng: dict[str, Any]
    social: SocialStateSnapshot
    social_rng: list[Any]
    diffusion_rng: list[Any]
    adjacency_order: dict[str, list[int]]
    price: float = Field(gt=0)
    previous_price: float = Field(gt=0)
    halted: bool


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value


class LegacySocietyParticipant:
    participant_id = "legacy_society"
    state_schema_version = 1
    array_fields = frozenset({"state", "reference_points", "risk_appetite_state", "trading_intent_state",
                              "loss_aversion_intensity_state", "neighbor_network", "influence_map"})

    def __init__(self, runner):
        self.runner = runner
        self.population_fields = frozenset(vars(runner.population))
        self.graph_fields = frozenset(vars(runner.social))
        self.diffusion_fields = frozenset(vars(runner.diffusion)) | {"last_research_step"}

    def validate_state(self):
        p = self.runner.population
        if (p.n_smart != 0 or p.smart_agents or p.n_vectorized != self.runner.config.n_agents
                or frozenset(vars(p)) != self.population_fields
                or {k for k, v in vars(p).items() if isinstance(v, np.ndarray)} != self.array_fields):
            raise ValueError("Unregistered population state; only the frozen vectorized mode is supported")
        if (set(vars(self.runner.social)) != self.graph_fields
                or set(vars(self.runner.diffusion)) - self.diffusion_fields):
            raise ValueError("Unregistered social state")
        if any(not np.isfinite(getattr(p, key)).all() for key in self.array_fields):
            raise ValueError("Nonfinite population state")

    def export_state(self):
        self.validate_state()
        r, p = self.runner, self.runner.population
        rng = p.np_rng.get_state()
        arrays = {name: PopulationArray(dtype=str(getattr(p, name).dtype), shape=list(getattr(p, name).shape),
                   values=getattr(p, name).ravel().tolist()) for name in sorted(self.array_fields)}
        return LegacySocietyState(population=arrays,
            population_rng={"python": json_values(p.py_rng.getstate()),
                "numpy": [rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])]},
            social=capture_social_state(r.diffusion), social_rng=json_values(r.social.rng.getstate()),
            diffusion_rng=json_values(r.diffusion.rng.getstate()),
            adjacency_order={str(i): list(r.social.graph.neighbors(i)) for i in r.social.graph},
            price=float(r.price), previous_price=float(r.previous_price), halted=r.halted).model_dump()

    def import_state(self, state):
        state = LegacySocietyState.model_validate(state)
        if (set(state.population) != self.array_fields or state.social.n_agents != self.runner.config.n_agents
                or [node.agent_id for node in state.social.nodes] != self.runner.ids):
            raise ValueError("Incomplete population snapshot")
        arrays = {name: record.array() for name, record in state.population.items()}
        for name, value in arrays.items():
            original = getattr(self.runner.population, name)
            if value.shape != original.shape or value.dtype != original.dtype or not np.isfinite(value).all():
                raise ValueError("Restored population layout differs")
        diffusion = restore_social_state(state.social)
        graph = diffusion.graph.graph
        if set(state.adjacency_order) != {str(i) for i in graph}:
            raise ValueError("Missing social adjacency order")
        for key, neighbors in state.adjacency_order.items():
            node = int(key)
            if len(neighbors) != len(set(neighbors)) or set(neighbors) != set(graph[node]):
                raise ValueError("Social adjacency membership differs")
            # NetworkX preserves insertion order; legacy feed ranking relies on it.
            graph._adj[node] = {other: graph._adj[node][other] for other in neighbors}
        diffusion.rng.setstate(tuples(state.diffusion_rng))
        diffusion.graph.rng = random.Random()
        diffusion.graph.rng.setstate(tuples(state.social_rng))
        py_rng, np_rng = random.Random(), np.random.RandomState()
        py_rng.setstate(tuples(state.population_rng["python"]))
        kind, keys, pos, gaussian, cached = state.population_rng["numpy"]
        np_rng.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
        candidate_runner = copy(self.runner)
        candidate_runner.population = copy(self.runner.population)
        candidate_runner.diffusion, candidate_runner.social = diffusion, diffusion.graph
        for name, value in arrays.items():
            setattr(candidate_runner.population, name, value)
        candidate_runner.population.py_rng, candidate_runner.population.np_rng = py_rng, np_rng
        candidate_runner.price, candidate_runner.previous_price, candidate_runner.halted = state.price, state.previous_price, state.halted
        if content_hash(LegacySocietyParticipant(candidate_runner).export_state()) != content_hash(state):
            raise ValueError("Noncanonical society snapshot")
        r, p = self.runner, self.runner.population
        for name, value in arrays.items():
            setattr(p, name, value)
        p.py_rng, p.np_rng = py_rng, np_rng
        r.diffusion, r.social = diffusion, diffusion.graph
        r.price, r.previous_price, r.halted = state.price, state.previous_price, state.halted
