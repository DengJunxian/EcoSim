"""Strict structural policy/scenario types; execution belongs to later services."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator

from core.experiments.strict_types import StrictResearchModel


class PublicDrawdownTrigger(StrictResearchModel):
    kind: Literal["public_drawdown"]
    lookback_steps: int = Field(ge=1)
    threshold_bps: int = Field(ge=1, le=9999)
    max_price_age_steps: int = Field(ge=0)


class HaltRule(StrictResearchModel):
    trigger: PublicDrawdownTrigger
    duration_steps: int = Field(ge=1)
    cooldown_steps: int = Field(ge=1)
    max_episodes: int = Field(ge=1)


class LiquidityRule(StrictResearchModel):
    trigger: PublicDrawdownTrigger
    duration_steps: int = Field(ge=1)
    max_activations: int = Field(ge=1)
    cash_cap_ppm: int = Field(ge=0, le=1_000_000)
    max_inventory_deviation_lots: int = Field(ge=0)
    half_spread_ticks: int = Field(ge=1)
    quote_qty_lots: int = Field(ge=1)
    max_reference_age_steps: int = Field(ge=0)


class PolicySpec(StrictResearchModel):
    schema_version: Literal[1]
    label: str = Field(min_length=1)
    halt: HaltRule | None
    liquidity: LiquidityRule | None
    activation_latency_steps: Literal[1]
    announcement_rule: Literal["after_trigger_before_effect"]


class PolicyConstraints(StrictResearchModel):
    schema_version: Literal[1]
    horizon: int = Field(ge=1)
    max_total_halt_steps: int = Field(ge=0)
    max_halt_duration_steps: int = Field(ge=0)
    min_cooldown_steps: int = Field(ge=1)
    max_halt_episodes: int = Field(ge=0)
    max_cash_cap_ppm: int = Field(ge=0, le=1_000_000)
    policy_account_cash: float = Field(ge=0)
    max_inventory_deviation_lots: int = Field(ge=0)
    max_liquidity_duration_steps: int = Field(ge=0)
    max_liquidity_activations: int = Field(ge=0)
    max_quote_qty_lots: int = Field(ge=0)
    max_half_spread_ticks: int = Field(ge=1)
    max_reference_age_steps: int = Field(ge=0)
    max_trigger_age_steps: int = Field(ge=0)
    max_lookback_steps: int = Field(ge=1)
    allow_combination: bool
    parameter_grids: dict[str, list[int]]

    @model_validator(mode="after")
    def _finite_grid(self):
        permitted = {"duration_steps", "cooldown_steps", "max_episodes", "max_activations",
            "cash_cap_ppm", "max_inventory_deviation_lots", "half_spread_ticks", "quote_qty_lots",
            "max_reference_age_steps", "lookback_steps", "threshold_bps", "max_price_age_steps"}
        if set(self.parameter_grids) - permitted:
            raise ValueError("Unknown policy parameter grid")
        if any(not values or len(values) != len(set(values)) or min(values) < 0
               for values in self.parameter_grids.values()):
            raise ValueError("Policy grids require nonempty unique nonnegative integers")
        return self


class TargetSelector(StrictResearchModel):
    mode: Literal["all_ordinary", "normal_mm", "explicit"]
    actor_ids: list[str]

    @model_validator(mode="after")
    def _selection(self):
        if len(self.actor_ids) != len(set(self.actor_ids)) or any(not x for x in self.actor_ids):
            raise ValueError("Target IDs must be nonempty and unique")
        if self.mode == "explicit" and not self.actor_ids:
            raise ValueError("Explicit target selection must list actors")
        if self.mode != "explicit" and self.actor_ids:
            raise ValueError("Non-explicit target selection cannot hide an actor list")
        return self


class FundamentalShockParameters(StrictResearchModel):
    relative_change: float = Field(gt=-1.0)


class PanicSeedParameters(StrictResearchModel):
    count: int = Field(ge=1)


class MMWithdrawalParameters(StrictResearchModel):
    duration_steps: int = Field(ge=1)
    participation_multiplier: float = Field(ge=0.0, le=1.0)
    half_spread_multiplier: float = Field(ge=1.0)


class UnsupportedSellingParameters(StrictResearchModel):
    """Reserved name only: no behavioral equation is authorized for execution."""


class _ScenarioEventBase(StrictResearchModel):
    event_id: str = Field(min_length=1)
    release_step: int = Field(ge=1)
    target_selector: TargetSelector


class FundamentalShockEvent(_ScenarioEventBase):
    kind: Literal["fundamental_shock"]
    parameters: FundamentalShockParameters


class PanicSeedEvent(_ScenarioEventBase):
    kind: Literal["panic_seed"]
    parameters: PanicSeedParameters


class MMWithdrawalEvent(_ScenarioEventBase):
    kind: Literal["normal_mm_withdrawal"]
    parameters: MMWithdrawalParameters


class UnsupportedSellingEvent(_ScenarioEventBase):
    kind: Literal["constrained_selling_need"]
    parameters: UnsupportedSellingParameters


ScenarioEvent = FundamentalShockEvent | PanicSeedEvent | MMWithdrawalEvent | UnsupportedSellingEvent


class FundamentalProcessSpec(StrictResearchModel):
    version: Literal["constant-fundamental-v1","piecewise-fundamental-v1"]
    initial_value: float = Field(gt=0.0)


class PrivateSignalProcessSpec(StrictResearchModel):
    version: Literal["explicit-initial-information-v1","relative-event-private-v1"]


class InitialInformation(StrictResearchModel):
    information_id: str = Field(min_length=1)
    kind: Literal["private_valuation", "public_reference"]
    actor_id: str | None
    value: float = Field(gt=0.0)
    as_of_step: int = Field(ge=0)
    available_step: int = Field(ge=0)
    origin_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def _availability(self):
        if self.as_of_step > self.available_step:
            raise ValueError("Information cannot arrive before it exists")
        if (self.kind == "private_valuation") != (self.actor_id is not None):
            raise ValueError("Only a private valuation has one actor owner")
        if self.actor_id == "":
            raise ValueError("Actor IDs cannot be empty")
        return self


class ScenarioSpec(StrictResearchModel):
    schema_version: Literal[1]
    scenario_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    events: list[ScenarioEvent]
    fundamental_process: FundamentalProcessSpec
    private_signal_process: PrivateSignalProcessSpec
    initial_information: list[InitialInformation]
    origin_refs: list[str]

    @model_validator(mode="after")
    def _unique_records(self):
        for ids in ([row.event_id for row in self.events],
                    [row.information_id for row in self.initial_information]):
            if len(ids) != len(set(ids)):
                raise ValueError("Scenario record IDs must be unique")
        if not self.origin_refs or any(not ref for ref in self.origin_refs):
            raise ValueError("Scenario requires explicit origin references")
        return self
