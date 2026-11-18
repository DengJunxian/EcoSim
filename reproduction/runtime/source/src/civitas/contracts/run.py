from typing import Literal

from pydantic import Field

from .base import Digest, Id, Record


class RunManifest(Record):
    schema_version: Literal[1] = 1
    status: Literal["running", "completed", "failed"]
    code_commit: Id | None
    code_dirty: bool | None
    source_hash: Digest
    config_hash: Digest
    initial_state_hash: Digest
    final_state_hash: Digest
    dependency_hash: Digest
    scenario_id: Id
    scenario_version: Id
    policy_id: Id
    policy_version: Id
    mechanism_versions: dict[Id, Id] = Field(default_factory=dict)
    rng_algorithm: Id
    data_origin: Literal["synthetic_toy"] = "synthetic_toy"
    api_calls: Literal[0] = 0
    api_spend_usd: Literal[0] = 0
    api_spend_cny: Literal[0] = 0
    error: str | None = None


class FinancialRunManifest(RunManifest):
    schema_version: Literal[2] = 2
    data_origin: Literal["simulation"] = "simulation"
    backend: Literal["legacy_float64"] = "legacy_float64"


class PortfolioRunManifest(RunManifest):
    schema_version: Literal[3] = 3
    data_origin: Literal["simulation"] = "simulation"
    backend: Literal["fixedpoint_multiasset"] = "fixedpoint_multiasset"
    population_origin: Literal["synthetic_model_population"] = "synthetic_model_population"
