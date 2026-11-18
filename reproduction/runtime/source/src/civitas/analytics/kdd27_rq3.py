"""RQ3 projections and seed-paired contrasts; no scientific target fitting."""
from collections import Counter, deque
import math

from .kdd27 import ratio
from .kdd27_statistics import paired_summary


BACKENDS = ('rule','heuristic','llm','hybrid')
FRAMES = ('neutral','negative')


def decision_metrics(facts, decisions):
    active=[d for d in decisions if d['active']]
    paths=Counter(d['path'] for d in active)
    if len(active)!=sum(r['active'] for r in facts):
        raise ValueError('Decision/fact active denominators differ')
    if len({(d['tick'],d['actor_id']) for d in decisions})!=len(decisions):
        raise ValueError('Duplicate actor decision')
    orders=sum(r['orders'] for r in facts);rejected=sum(r['rejected_orders'] for r in facts)
    if not 0<=rejected<=orders:raise ValueError('Invalid order counts')
    return {'active_decisions':len(active),'paths':dict(paths),
        'fresh_model_coverage':ratio(paths['model'],len(active),'NO_ACTIVE_DECISIONS'),
        'strategy_cache_coverage':ratio(paths['strategy_cache'],len(active),'NO_ACTIVE_DECISIONS'),
        'total_cognitive_coverage':ratio(paths['model']+paths['strategy_cache'],len(active),'NO_ACTIVE_DECISIONS'),
        'rule_fallback_coverage':ratio(paths['rule_fallback'],len(active),'NO_ACTIVE_DECISIONS'),
        'degraded_coverage':ratio(paths['degraded_hold'],len(active),'NO_ACTIVE_DECISIONS'),
        'accepted_orders':orders-rejected,'rejected_orders':rejected,
        'accepted_order_rate':ratio(orders-rejected,orders,'NO_SUBMITTED_ORDERS'),
        'filled_order_rate':ratio(sum(r['filled_orders'] for r in facts),orders,'NO_SUBMITTED_ORDERS'),
        'definition':'fresh means model decision at this tick, including exact receipt recovery; new API requests come from owned attempts'}


def attempt_metrics(records):
    if len({r['attempt_id'] for r in records})!=len(records):raise ValueError('Duplicate model attempt')
    errors=Counter(r['error']['code'] for r in records if r['error'])
    return {'attempts':len(records),'requests':len({r['request_id'] for r in records}),
        'retries':sum(r['number']>1 for r in records),'states':dict(Counter(r['state'] for r in records)),
        'errors':dict(errors),'valid_response_rate':ratio(sum(r['state']=='success' for r in records),len(records),'NO_MODEL_ATTEMPTS'),
        'latency_ms_sum':sum(r['latency_ms'] or 0 for r in records) if all(r['latency_ms'] is not None for r in records) else None,
        'scope':'Owned attempts only, including uncommitted failures; exact response recovery is not another attempt'}


def event_response(facts, *, event_tick, horizon, pre_price, pre_confidence):
    if type(event_tick) is not int or type(horizon) is not int or min(event_tick,horizon)<1:
        raise ValueError('Positive event tick and horizon required')
    if [r['tick'] for r in facts]!=list(range(event_tick,event_tick+horizon)):
        raise ValueError('Complete committed event window required')
    if not math.isfinite(pre_price) or pre_price<=0 or not math.isfinite(pre_confidence):
        raise ValueError('Verified finite pre-event anchors required')
    return {'event_tick':event_tick,'horizon':horizon,'pre_price':pre_price,'pre_confidence':pre_confidence,
        'log_return':math.log(facts[-1]['close']/pre_price),
        'net_sell_quantity':sum(r['net_sell_quantity'] for r in facts),
        'confidence_change':facts[-1]['confidence_mean']-pre_confidence,
        'traded_quantity':sum(r['volume'] for r in facts),
        'definition':'End-of-window log return and confidence change from committed t_event-1; sum of submitted sell minus buy quantity'}


def resource_outcomes(initial, final, *, market):
    """FIFO for within-window buys; opening inventory has unknown acquisition age."""
    if initial.tick>=final.tick or initial.actors!=final.actors:raise ValueError('Aligned ordered snapshots required')
    asset,currency=market['asset_id'],market['currency']
    tick_minor=next(a['price_tick_minor'] for a in final.data['multiasset_market']['ledger']['assets'] if a['asset_id']==asset)
    start={r['account_id']:r for r in initial.data['multiasset_market']['ledger']['accounts']}
    end={r['account_id']:r for r in final.data['multiasset_market']['ledger']['accounts']}
    initial_price=initial.data['kdd27']['last_quote']['last_price_ticks']*tick_minor
    final_price=final.data['kdd27']['last_quote']['last_price_ticks']*tick_minor
    trades=[r['trade'] for r in final.data['multiasset_market']['ledger']['journal'] if initial.tick<r['trade']['tick']<=final.tick]
    if len({r['trade_id'] for r in trades})!=len(trades) or any(a['tick']>b['tick'] for a,b in zip(trades,trades[1:])):
        raise ValueError('Unique chronological settlement tape required')
    rows=[]
    for actor in initial.actors:
        opening=start[actor]['balances'][asset];unknown=opening;lots=deque();closed=[];notional=count=sold_unknown=0
        for t in trades:
            if actor not in (t['buyer_id'],t['seller_id']):continue
            q=t['quantity_units'];count+=1;notional+=q*t['price_ticks']*tick_minor
            if t['buyer_id']==actor:lots.append([t['tick'],q]);continue
            used=min(unknown,q);unknown-=used;q-=used;sold_unknown+=used
            while q:
                if not lots:raise ValueError('Sale exceeds observed FIFO inventory')
                bought,quantity=lots[0];used=min(quantity,q)
                closed.append({'buy_tick':bought,'sell_tick':t['tick'],'quantity':used,'holding_steps':t['tick']-bought})
                lots[0][1]-=used;q-=used
                if not lots[0][1]:lots.popleft()
        cash=end[actor]['balances'][currency];shares=end[actor]['balances'][asset]
        if unknown+sum(q for _,q in lots)!=shares:raise ValueError('FIFO inventory differs from final ledger')
        wealth=cash+shares*final_price;opening_wealth=start[actor]['balances'][currency]+opening*initial_price
        observed=sum(r['quantity'] for r in closed)
        rows.append({'actor_id':actor,'trade_count':count,'opening_inventory_left_censored':opening,
            'initial_units_sold_unknown_age':sold_unknown,'initial_units_still_open_unknown_age':unknown,
            'closed_lots':closed,'closed_units':observed,
            'closed_lot_weighted_mean_steps':sum(r['quantity']*r['holding_steps'] for r in closed)/observed if observed else None,
            'right_censored_lots':[{'buy_tick':t,'quantity':q,'age_at_end':final.tick-t} for t,q in lots],
            'right_censored_units':sum(q for _,q in lots),
            'gross_turnover_initial_wealth':notional/opening_wealth if opening_wealth else None,
            'end_resource_exposure':shares*final_price/wealth if wealth else None,
            'persona_score':None,'persona_reason':'TARGETS_NOT_FROZEN' if count else 'NO_TRADES'})
    return {'window':[initial.tick+1,final.tick],'actors':rows,'actors_with_trades':sum(r['trade_count']>0 for r in rows),
        'persona_scored':0,'persona_coverage':ratio(0,len(rows)),
        'definition':'Gross buy+sell notional / opening marked wealth; final share value / marked wealth; FIFO is an accounting convention, not inferred intent',
        'holding_limit':'Closed-lot mean excludes censored lots; not an estimate of overall holding time'}


def paired_rq3(rows, *, bootstrap_seed=935401):
    """Common complete-seed blocks keep all eight cells aligned for every contrast."""
    lookup={};worlds=set();cells=[f'{b}:{f}' for b in BACKENDS for f in FRAMES]
    for row in rows:
        key=(row['world_seed'],row['condition_id'])
        if row['condition_id'] not in cells or key in lookup:raise ValueError('Unexpected/duplicate RQ3 seed cell')
        if row['value'] is not None and (row['status']!='completed' or not math.isfinite(row['value'])):
            raise ValueError('Nonfinite or incomplete-cell outcome')
        lookup[key]=row;worlds.add(row['world_seed'])
    selected=[];excluded=[]
    for world in sorted(worlds):
        block=[lookup.get((world,c)) for c in cells];present=[r for r in block if r is not None]
        if len({(r['pair_id'],r['protocol_version']) for r in present})!=1:raise ValueError('RQ3 block identity mismatch')
        complete=all(r is not None and r['status']=='completed' and r['value'] is not None for r in block)
        if not complete:excluded.append({'world_seed':world,'incomplete_cells':[c for c,r in zip(cells,block) if r is None or r['status']!='completed' or r['value'] is None]})
        for cell,row in zip(cells,block):
            selected.append({**present[0],'world_seed':world,'condition_id':cell,
                'status':'completed' if complete else 'incomplete','value':row['value'] if complete else None})
    contrasts=[(f'{b}:neutral',f'{b}:negative') for b in BACKENDS]
    contrasts += [(f'{control}:{f}',f'{b}:{f}') for f in FRAMES for control,b in
        [('rule','heuristic'),('rule','llm'),('rule','hybrid'),('llm','hybrid')]]
    return {'planned_seeds':len(worlds),'complete_seed_blocks':len(worlds)-len(excluded),
        'planned_cells':8*len(worlds),'provided_cells':len(rows),'raw_statuses':dict(Counter(r['status'] for r in rows)),
        'excluded_blocks':excluded,'contrasts':[paired_summary([r for r in selected if r['condition_id'] in (a,b)],
            control=a,treatment=b,seed=bootstrap_seed) for a,b in contrasts],
        'resampling':'Identically seeded bootstrap on the same complete world blocks for all contrasts; pointwise 95% intervals',
        'equivalence':None,'equivalence_reason':'INDEPENDENT_MARGIN_AND_PRECISION_NOT_FROZEN'}
