"""Complete event-window adapters over KDD03 metrics; no new inference engine."""
import math

from .kdd27 import recovery,summarize_facts


def event_metrics(facts,*,event_tick,horizon,pre_price,initial_cash,initial_shares,ticks_per_day):
    if type(event_tick) is not int or event_tick<1 or type(horizon) is not int or horizon<1:
        raise ValueError('Positive event tick and horizon required')
    if [r['tick'] for r in facts]!=list(range(event_tick,event_tick+horizon)):
        raise ValueError('Event outcomes require every committed step in the declared window')
    if type(pre_price) is not float or not math.isfinite(pre_price) or pre_price<=0:
        raise ValueError('Verified positive pre-event price required')
    summary=summarize_facts(facts,initial_cash=initial_cash,initial_shares=initial_shares,
        ticks_per_day=ticks_per_day,initial_price=pre_price,initial_tick=event_tick-1)
    prices=[pre_price,*[r['close'] for r in facts]]
    jumps=[{'u':i+1,'tick':r['tick'],'log_return':math.log(r['close']/facts[i-1]['close'])}
        for i,r in enumerate(facts) if i and facts[i-1]['halted'] and not r['halted']]
    return {'summary':summary,'recovery':recovery(prices,horizon=horizon),
        'price_path':[{'u':u,'tick':event_tick+u-1,'price':p} for u,p in enumerate(prices)],
        'tradable_steps':sum(not r['halted'] for r in facts),
        'traded_notional':sum(r['notional'] for r in facts) if all('notional' in r for r in facts) else None,
        'reopening_jumps':jumps,'formal_eligible':False,
        'temporal_chain_status':'AVAILABLE' if horizon>=1000 else 'INCOMPLETE_PRESPECIFIED_WINDOWS',
        'temporal_windows_u':{'social':[0,40],'orders':[41,100],'later_drawdown':[101,1000]}}
