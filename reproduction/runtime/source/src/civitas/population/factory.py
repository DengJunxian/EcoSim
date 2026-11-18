"""Fixed total endowments constrain allocation; mismatch is reported or rejected."""
from fractions import Fraction
from civitas.contracts.base import content_hash
from civitas.contracts.population import PopulationInitialization, PopulationSpec, Profile
from civitas.contracts.portfolio import AccountState
from civitas.core.rng import AddressedRNG
from civitas.resources.fixedpoint import checked, decimal_minor, notional
from .allocation import allocate


class PopulationFactory:
    @staticmethod
    def build(spec: PopulationSpec, rng: AddressedRNG) -> PopulationInitialization:
        s=PopulationSpec.model_validate(spec)
        decimal_minor('0',s.cash_scale)
        currency=s.assets[0].quote_currency
        weights={}
        width=s.wealth_prior.max_weight-s.wealth_prior.min_weight+1
        cutoff=2**64-(2**64 % width)
        for i in range(s.n_agents):
            actor=f'agent_{i:04d}'
            attempt=0
            draw=rng.u64('population.wealth.v1',actor,0,'weight')
            while draw>=cutoff:
                attempt+=1
                draw=rng.u64('population.wealth.v1',actor,0,f'weight.retry:{attempt}')
            weights[actor]=s.wealth_prior.min_weight+draw%width
        totals={currency:s.cash_total_minor,**s.asset_supplies}
        prices={currency:1,**{a.asset_id:notional(s.initial_price_ticks[a.asset_id],a.price_tick_minor,1) for a in s.assets}}
        value={r:checked(totals[r]*prices[r]) for r in totals}
        wealth_total=checked(sum(value.values()))
        if wealth_total==0:
            raise ValueError('Population requires a positive total endowment')
        target_total=sum(s.target_weights.values())
        mismatch=any(value[r]*target_total!=wealth_total*s.target_weights[r] for r in totals)
        if mismatch and s.mismatch_policy=='reject':
            raise ValueError('Supplies/prices are incompatible with target allocation weights')
        journal=[entry for r,total in totals.items() for entry in allocate(total,weights,r)]
        balances={actor:{} for actor in weights}
        for e in journal:
            balances[e.account_id][e.resource_id]=e.allocated_units
        accounts=[AccountState(account_id=actor,balances=balance,
                  cost_minor={a.asset_id:balance[a.asset_id]*prices[a.asset_id] for a in s.assets})
                  for actor,balance in balances.items()]
        wealth={actor:sum(balance[r]*prices[r] for r in totals) for actor,balance in balances.items()}
        cognition={actor:{'anchor_price_ticks':dict(s.initial_price_ticks),
                           'valuation_price_ticks':dict(s.initial_price_ticks),'wealth_reference_minor':wealth[actor],
                           'sentiment':0,'leverage_enabled':False} for actor in weights}
        realized={actor:{r:str(Fraction(balance[r]*prices[r],wealth[actor])) if wealth[actor] else None
                         for r in totals} for actor,balance in balances.items()}
        deviations={actor:{r:str(Fraction(balance[r]*prices[r])-Fraction(wealth[actor]*s.target_weights[r],target_total))
                           for r in totals} for actor,balance in balances.items()}
        shares=[Fraction(w,wealth_total) for w in wealth.values()]
        stats={'allocation_rule':'largest_remainder_stable_account_id','cash_unallocated_minor':0,
            'supply_mismatch':mismatch,'wealth_total_minor':wealth_total,'resource_totals':totals,
            'resource_values_minor':value,'target_weight_fractions':{r:str(Fraction(s.target_weights[r],target_total)) for r in totals},
            'realized_wealth_minor':wealth,'realized_weight_fractions':realized,'deviation_sign':'realized_minus_target',
            'deviation_minor_fractions':deviations,'wealth_hhi':str(sum(w*w for w in shares)),
            'cash_ratio_fractions':{a:realized[a][currency] for a in weights},
            'wealth_weight_prior':s.wealth_prior.model_dump(),
            'correlation':'All resources use the same independent per-actor wealth weight; no institutional capital multiplier',
            'zero_wealth_accounts':[a for a,w in wealth.items() if w==0],
            'rng_algorithm':rng._root.algorithm,'root_seed':rng._root.root_seed,'replica':rng._root.replica}
        return PopulationInitialization(initialization_id='population:'+content_hash({'spec':s.model_dump(),'weights':weights}),
            profiles=[Profile(profile_id=f'{s.population_id}:{a}',actor_id=a,wealth_weight=w) for a,w in weights.items()],
            accounts=accounts,cognition=cognition,journal=journal,statistics=stats)
