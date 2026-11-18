"""Finite policy compilation; no clipping, retries, account creation or execution."""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Literal

from pydantic import ValidationError,Field,model_validator
from core.experiments.strict_types import StrictResearchModel
from core.experiments.identity import research_hash,HASH_PATTERN
from core.interventions.schema import PolicySpec, PolicyConstraints


class PolicyCompileError(ValueError):
    def __init__(self,code:str,path:str,expected:Any,observed:Any):
        self.diagnostic={'code':code,'path':path,'expected':expected,'observed':observed}
        super().__init__(f'{code} at {path}: expected {expected!r}, observed {observed!r}')


def conservative_money_round(total_wealth:float,ppm:int) -> float:
    if type(total_wealth) is not float or not math.isfinite(total_wealth) or total_wealth<0.0:
        raise ValueError('Wealth must be nonnegative finite binary64')
    if type(ppm) is not int or not 0<=ppm<=1_000_000:
        raise ValueError('Cash authorization requires integer ppm in [0,1000000]')
    exact=Fraction.from_float(total_wealth)*ppm/1_000_000
    rounded=float(exact)
    if Fraction.from_float(rounded)>exact:
        rounded=math.nextafter(rounded,-math.inf)
    return rounded


def canonicalize_policy(policy:PolicySpec,*,effective_cash_budget:float|None=None) -> dict:
    policy=PolicySpec.model_validate(policy)
    body=policy.model_dump(mode='json',exclude={'label'})
    if policy.liquidity is not None and (policy.liquidity.cash_cap_ppm==0 or effective_cash_budget==0.0):
        body['liquidity']=None
    return body


class CompiledPolicy(StrictResearchModel):
    raw_policy: PolicySpec
    semantic_body: dict[str,Any]
    policy_hash: str=Field(pattern=HASH_PATTERN)
    constraints_hash: str=Field(pattern=HASH_PATTERN)
    compile_version: Literal['finite-policy-compiler-v1']='finite-policy-compiler-v1'
    allowed_actions: list[str]
    deterministic_bounds: dict[str,int|float]
    diagnostics: list[dict[str,Any]]

    @model_validator(mode='after')
    def _identity(self):
        if self.policy_hash!=research_hash(self.semantic_body):raise ValueError('Compiled policy semantic hash mismatch')
        return self


def compile_policy(policy:PolicySpec|dict,*,constraints:PolicyConstraints,world,market) -> CompiledPolicy:
    try:
        policy=PolicySpec.model_validate(policy)
        constraints=PolicyConstraints.model_validate(constraints)
    except ValidationError as exc:
        error=exc.errors(include_url=False)[0]
        raise PolicyCompileError('INVALID_POLICY_SCHEMA','.'.join(map(str,error['loc'])),error['type'],error.get('input')) from exc
    def upper(value,limit,path):
        if value>limit:raise PolicyCompileError('ABOVE_BOUND',path,{'maximum':limit},value)
    if policy.halt is not None and policy.liquidity is not None and not constraints.allow_combination:
        raise PolicyCompileError('COMBINATION_FORBIDDEN','policy',{'allow_combination':False},'halt+liquidity')
    if constraints.policy_account_cash!=world.policy_account_initial.cash:
        raise PolicyCompileError('INITIAL_RESOURCE_MISMATCH','policy_account_cash',world.policy_account_initial.cash,constraints.policy_account_cash)
    for name,rule in [('halt',policy.halt),('liquidity',policy.liquidity)]:
        if rule is None:continue
        upper(rule.trigger.lookback_steps,min(constraints.horizon,constraints.max_lookback_steps),f'{name}.trigger.lookback_steps')
        upper(rule.trigger.max_price_age_steps,constraints.max_trigger_age_steps,f'{name}.trigger.max_price_age_steps')
        fields=rule.model_dump(mode='json');fields.update(fields.pop('trigger'))
        for field,grid in constraints.parameter_grids.items():
            if field in fields and fields[field] not in grid:
                raise PolicyCompileError('OUTSIDE_REGISTERED_GRID',f'{name}.{field}',list(grid),fields[field])
    if policy.halt is not None:
        halt=policy.halt
        upper(halt.duration_steps,min(constraints.horizon,constraints.max_halt_duration_steps),'halt.duration_steps')
        upper(halt.max_episodes,constraints.max_halt_episodes,'halt.max_episodes')
        upper(halt.duration_steps*halt.max_episodes,constraints.max_total_halt_steps,'halt.total_steps')
        if halt.cooldown_steps<constraints.min_cooldown_steps:
            raise PolicyCompileError('BELOW_BOUND','halt.cooldown_steps',{'minimum':constraints.min_cooldown_steps},halt.cooldown_steps)
    budget=0.0
    lower=upper_inventory=world.policy_account_initial.shares
    if policy.liquidity is not None:
        lp=policy.liquidity
        for field,limit in [('cash_cap_ppm',constraints.max_cash_cap_ppm),
            ('duration_steps',min(constraints.horizon,constraints.max_liquidity_duration_steps)),
            ('max_activations',constraints.max_liquidity_activations),
            ('max_inventory_deviation_lots',constraints.max_inventory_deviation_lots),
            ('quote_qty_lots',constraints.max_quote_qty_lots),('half_spread_ticks',constraints.max_half_spread_ticks),
            ('max_reference_age_steps',constraints.max_reference_age_steps)]:
            upper(getattr(lp,field),limit,f'liquidity.{field}')
        budget=conservative_money_round(world.initial_total_trader_wealth,lp.cash_cap_ppm)
        upper(budget,world.policy_account_initial.cash,'liquidity.cash_authorization')
        deviation=lp.max_inventory_deviation_lots*market.board_lot
        lower=max(0,world.policy_account_initial.shares-deviation)
        upper_inventory=world.policy_account_initial.shares+deviation
    semantic=canonicalize_policy(policy,effective_cash_budget=budget)
    actions=[]
    if semantic['halt'] is not None:actions.append('halt')
    if semantic['liquidity'] is not None:actions.append('funded_liquidity')
    return CompiledPolicy(raw_policy=policy,semantic_body=semantic,policy_hash=research_hash(semantic),
        constraints_hash=research_hash(constraints),allowed_actions=actions,
        deterministic_bounds={'cash_authorization':budget,'inventory_lower_shares':lower,
            'inventory_upper_shares':upper_inventory,'max_total_halt_steps':0 if policy.halt is None else policy.halt.duration_steps*policy.halt.max_episodes},
        diagnostics=[])
