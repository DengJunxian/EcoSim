"""WP06 choices on current observations; legacy projections retain binary64 units."""
from fractions import Fraction
import math

from core.behavioral_finance import ProspectTheoryParams, prospect_value
from core.exchange.market_rules import execution_fees, worst_buy_cash
from civitas.contracts.base import content_hash
from civitas.contracts.decisions import HoldIntent
from civitas.contracts.financial import FinancialDecisionBundle, LegacyAccountView, OrderIntent
from civitas.contracts.portfolio import PortfolioDecisionBundle
from civitas.cognition.policy import empty_bundle
from civitas.mechanisms.behavior import BehaviorPolicy, ReferenceWealth
from civitas.mechanisms.influence import Opinion, social_update
from civitas.mechanisms.outcomes import ExactAmount
from .society_relations import scheduled_drafts


class SocietyBehavior(BehaviorPolicy):
    def __init__(self, config, *, model_owned=False):
        self.scene = config
        b = config.behavior
        influence = b.influence.model_copy(update={'owner':'llm_integrated','model_reads_peers':True,'mu':0.0,'nu':0.0}) if model_owned else b.influence
        super().__init__(b.joint_outcomes,parameters=b.parameters,root_seed=b.root_seed,replica=b.replica,
                         trade_fraction=b.trade_fraction,influence=influence)

    def zero_wealth_hold(self, observation, bundle_type):
        value = empty_bundle(observation,self.policy_id,self.version,bundle_type)
        return value.model_copy(update={'economic_intents':[HoldIntent(intent_id='intent:'+value.decision_id)],
            'belief_patch':{'financial_constraint':'zero_wealth'},'evidence_ids':[observation.account_view[0].fact_id]})

    def _decide(self, observation, outcomes):
        if not any(observation.account_view[0].value['balances'].values()):
            return self.zero_wealth_hold(observation,PortfolioDecisionBundle)
        return super()._decide(observation,outcomes)

    def candidate_actions(self, observation):
        actions = super().candidate_actions(observation)
        if self.scene.quantity_rule=='one_unit':
            actions = [a.model_copy(update={'delta_units':{k:1 if v>0 else -1 for k,v in a.delta_units.items()}}) for a in actions]
        return actions

    async def decide(self, observation):
        # Persist a rule-owned opinion between activations; model-owned plans use the WP05 port.
        outcomes = self.joint_outcomes
        previous = [f for f in observation.private_memory_refs if 'influence' in f.value.get('behavior',{})]
        if previous:
            probabilities = max(previous,key=lambda f:(f.visible_at.key,f.fact_id)).value['behavior']['influence']['probabilities']
            outcomes = outcomes.model_copy(update={'scenarios':[s.model_copy(update={'probability':probabilities[s.scenario_id]}) for s in outcomes.scenarios]})
        if self.influence.owner!='rule_social':
            raise ValueError('Model-owned opinion requires a current WP05 belief')
        return self._decide(observation,outcomes)


class LegacySocietyBehavior(SocietyBehavior):
    """Counterfactual valuation only: orders still settle in the original ClearingHouse."""

    async def fallback_decide(self, observation):
        bundle = empty_bundle(observation,self.policy_id,self.version,FinancialDecisionBundle)
        return bundle.model_copy(update={'economic_intents':[HoldIntent(intent_id='intent:'+bundle.decision_id)]})

    def _decide(self, observation, outcomes):
        account = LegacyAccountView.model_validate(observation.account_view[0].value)
        facts = {f.fact_id:f for f in observation.market_view}
        rules = facts['society:legacy-rules'].value
        quote = next(f.value for f in observation.market_view if f.fact_id.startswith('quotes:'))
        asset, tick = rules['symbol'], observation.visible_until.tick
        price = quote[asset]['last_price']
        wealth = account.cash+account.shares*price
        if wealth<=0:
            return self.zero_wealth_hold(observation,FinancialDecisionBundle)
        evidence = [observation.account_view[0].fact_id,next(f.fact_id for f in observation.market_view if f.fact_id.startswith('quotes:'))]
        anchor = ReferenceWealth(amount=ExactAmount.of(Fraction.from_float(wealth)),currency='CNY',cash_scale=1,
            observed_tick=tick,evidence_ids=evidence)
        old = [f for f in observation.private_memory_refs if 'reference_next' in f.value.get('behavior',{})]
        reference = ReferenceWealth.model_validate(max(old,key=lambda f:(f.visible_at.key,f.fact_id)).value['behavior']['reference_next']) if old else anchor
        if reference.cash_scale!=1 or reference.currency!='CNY' or reference.observed_tick>tick:
            raise ValueError('Legacy reference unit/time mismatch')
        p = self.parameters
        if p.norm_weight:
            raise ValueError('Legacy society supports no group-norm penalty')
        own = Opinion(proposition_id=self.influence.proposition_id,horizon_ticks=outcomes.scenarios[0].horizon_ticks,
            probabilities={s.scenario_id:s.probability for s in outcomes.scenarios})
        pooled = social_update(observation,own,self.influence)
        ratio, lot = self.trade_fraction, rules['board_lot']
        buy = math.floor(account.available_cash*ratio.numerator/ratio.denominator/(price*(1+rules['commission_rate'])))
        while buy>0 and worst_buy_cash(price,buy,rules['commission_rate'])>account.available_cash:
            buy-=1
        sell = account.available_shares*ratio.numerator//ratio.denominator
        if self.scene.quantity_rule=='one_unit':
            buy, sell = min(buy,lot), min(sell,lot)
        quantities = sorted({0,buy//lot*lot,-(sell//lot*lot)})
        scores = []
        denominator = max(float(reference.amount.fraction),p.epsilon_currency)
        params = ProspectTheoryParams(alpha=p.alpha,beta=p.beta,lambda_=p.loss_aversion)
        for quantity in quantities:
            buy_fee,sell_fee,tax = execution_fees(price,abs(quantity),rules['commission_rate'],rules['stamp_duty_rate'])
            fees = buy_fee if quantity>0 else sell_fee+tax if quantity<0 else 0.0
            components = []
            for scenario in outcomes.scenarios:
                filled = quantity if scenario.execution=='filled' else 0
                projected = account.cash-filled*price-(fees if filled else 0.0)+(account.shares+filled)*price*(1+scenario.asset_returns[asset])
                normalized = (projected-float(reference.amount.fraction))/denominator
                value = float(prospect_value(normalized,params=params))
                components.append({'scenario_id':scenario.scenario_id,'wealth':projected,'normalized_outcome':normalized,
                    'value':value,'probability':pooled.probabilities[scenario.scenario_id]})
            scores.append({'quantity':quantity,'action_key':content_hash({asset:quantity} if quantity else {}),
                'q':math.fsum(c['value']*c['probability'] for c in components),'components':components})
        scores.sort(key=lambda s:s['action_key'])
        maximum = max(s['q'] for s in scores)
        weights = [(1.0 if s['q']==maximum else 0.0) if p.temperature==0.0 else math.exp((s['q']-maximum)/p.temperature) for s in scores]
        total = math.fsum(weights)
        draw = self.rng.uniform('wp06.choice',observation.actor_id,tick,'economic_action')
        cumulative, chosen = 0.0,scores[-1]
        for score,weight in zip(scores,weights,strict=True):
            cumulative+=weight/total
            if draw<cumulative:
                chosen = score
                break
        prefix = f'wp07:{tick}:{observation.actor_id}'
        quantity = chosen['quantity']
        intent = (OrderIntent(intent_id=prefix,order_id='order:'+prefix,symbol=asset,side='buy' if quantity>0 else 'sell',
            price=price,quantity=abs(quantity)) if quantity else HoldIntent(intent_id=prefix))
        rho = Fraction(str(p.reference_rho))
        updated = anchor.model_copy(update={'amount':ExactAmount.of((1-rho)*reference.amount.fraction+rho*anchor.amount.fraction)})
        trace = {'scores':scores,'choice':{'action_key':chosen['action_key'],'draw':draw,
            'probabilities':{s['action_key']:w/total for s,w in zip(scores,weights,strict=True)}},
            'reference_next':updated.model_dump(),'influence':pooled.model_dump(),'belief_update_owner':self.influence.owner,
            'horizon_semantics':self.belief_contract['horizon_semantics'],'projection_units':'legacy_float64_currency',
            'execution_assumption':'single_fill_at_visible_last_price_or_explicit_unfilled'}
        return empty_bundle(observation,self.policy_id,self.version,FinancialDecisionBundle).model_copy(update={
            'economic_intents':[intent],'belief_patch':{'behavior':trace},'evidence_ids':sorted(observation.evidence_ids)})


class SocietyPolicy:
    policy_id, version = 'market_society', '1'

    def __init__(self, config, actors, *, slow=None, traces=None):
        self.config, self.actors, self.slow = config, actors, slow
        cls = LegacySocietyBehavior if config.backend=='legacy_float64' else SocietyBehavior
        self.fast = cls(config)
        self.mechanism_versions = self.fast.mechanism_versions
        self.configuration = config
        self.traces = traces

    async def decide(self, observation):
        delegate = self.slow if self.slow is not None and observation.actor_id in self.actors[:self.config.slow_agents] else self.fast
        bundle = await delegate.decide(observation)
        patch = bundle.model_dump()['belief_patch']
        numeric = patch.get('behavior',{}).copy()
        if 'behavior' in patch:
            # Detailed scores are exported for selected audit chains, not recursively fed back.
            patch['behavior'].pop('scores',None)
            patch['behavior'].pop('mechanism_versions',None)
        drafts = [*bundle.message_drafts,*scheduled_drafts(self.config,self.actors,observation.actor_id,observation.visible_until.tick)]
        if len(drafts)>self.config.max_messages_per_activation:
            raise ValueError('Society activation exceeds the message cap')
        result = bundle.model_copy(update={'policy_id':self.policy_id,'policy_version':self.version,'belief_patch':patch,'message_drafts':drafts})
        if self.traces is not None:
            self.traces.capture(observation,result,numeric)
        return result
