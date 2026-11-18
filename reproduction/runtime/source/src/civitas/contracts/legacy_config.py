"""Strict v2 envelope reusing the legacy configuration and its validators."""
from typing import Literal

from pydantic import Field
from core.experiments.config import ExperimentConfig, EventConfig, LLMConfig, MarketConfig, SocialConfig
from .base import Positive, Record


class LegacyMarketConfig(MarketConfig, Record):
    backend: Literal["python"] = "python"


class LegacySocialConfig(SocialConfig, Record):
    pass


class LegacyEventConfig(EventConfig, Record):
    pass


class OfflineLLMConfig(LLMConfig, Record):
    max_calls: Literal[0] = 0


class LegacyExperimentConfig(ExperimentConfig, Record):
    decision_mode: Literal["vectorized"] = "vectorized"
    historical_input: None = None
    market: LegacyMarketConfig = Field(default_factory=LegacyMarketConfig)
    social: LegacySocialConfig = Field(default_factory=LegacySocialConfig)
    llm: OfflineLLMConfig = Field(default_factory=OfflineLLMConfig)
    events: list[LegacyEventConfig] = Field(default_factory=list)


class LegacyRunConfig(Record):
    schema_version: Literal[2] = 2
    scenario: Literal["legacy_market"] = "legacy_market"
    backend: Literal["legacy_float64"] = "legacy_float64"
    mechanism_version: Literal["legacy-vectorized-v1"] = "legacy-vectorized-v1"
    concurrency: Positive = Field(default=8, le=64)
    experiment: LegacyExperimentConfig = Field(default_factory=LegacyExperimentConfig)

    @property
    def ticks(self):
        return self.experiment.steps

    @property
    def root_seed(self):
        return self.experiment.seed

    @property
    def replica(self):
        return 0
