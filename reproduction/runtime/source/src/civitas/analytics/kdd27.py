"""Versioned paper metrics from committed facts; original definitions stay intact."""
import math
import statistics
from collections import defaultdict
from typing import Literal

from core.experiments.measurement import MetricValue
from core.validation.statistics import _finite_optional


class PaperMetric(MetricValue):
    definition_version: Literal['kdd27-metrics-v1']='kdd27-metrics-v1'


def metric(value, count, total, unit, reason='INSUFFICIENT_SAMPLE'):
    return PaperMetric(value=None if value is None else float(value),status='VALID' if value is not None else 'UNDEFINED',
        reason=None if value is not None else reason,unit=unit,sample_count=count,denominator_count=total).model_dump()


def ratio(numerator,denominator,reason='ZERO_DENOMINATOR'):
    if numerator<0 or denominator<0:raise ValueError('Counts must be nonnegative')
    return metric(numerator/denominator if denominator else None,numerator,denominator,'ratio',reason)


def daily_statistics(clock,prices,volumes):
    max_lag=20
    if (not clock or any(type(t) is not int for t in clock) or clock!=list(range(clock[0],clock[-1]+1))
            or len(clock)!=len(prices) or len(clock)!=len(volumes) or type(max_lag) is not int or max_lag<1):
        raise ValueError('Complete explicit daily clock and aligned arrays required')
    _finite_optional(prices);_finite_optional(volumes)
    if any(v is not None and v<=0 for v in prices) or any(v is not None and v<0 for v in volumes):raise ValueError('Invalid price/volume')
    returns=[None if a is None or b is None else math.log(b/a) for a,b in zip(prices,prices[1:])]
    return return_statistics(returns,volumes[1:])


def return_statistics(returns,volumes):
    """Use precomputed adjacent returns when resampling date blocks."""
    _finite_optional(returns);_finite_optional(volumes)
    if len(returns)!=len(volumes) or any(v is not None and v<0 for v in volumes):
        raise ValueError('Returns and nonnegative daily volume must align')
    max_lag=20
    n=len(returns);complete=all(v is not None for v in returns);valid=[v for v in returns if v is not None]
    center=statistics.fmean(valid) if valid else 0.0
    m2=statistics.fmean((x-center)**2 for x in valid) if valid else 0.0
    kurt=None
    if complete and n>=4 and m2>0:
        g2=statistics.fmean((x-center)**4 for x in valid)/(m2*m2)-3
        kurt=(n-1)/((n-2)*(n-3))*((n+1)*g2+6)
    def acf(values):
        if not complete or n<max_lag+2:return None
        mean=statistics.fmean(values);den=math.fsum((x-mean)**2 for x in values)
        if den==0:return None
        return [math.fsum((values[t]-mean)*(values[t-lag]-mean) for t in range(lag,n))/den for lag in range(1,max_lag+1)]
    raw=acf(valid);absolute=acf([abs(v) for v in valid]);corr=None
    if complete and n>=2 and all(v is not None for v in volumes):
        xs=[abs(v) for v in valid];ys=volumes;xm=statistics.fmean(xs);ym=statistics.fmean(ys)
        den=math.sqrt(math.fsum((x-xm)**2 for x in xs)*math.fsum((y-ym)**2 for y in ys))
        if den>0:corr=math.fsum((x-xm)*(y-ym) for x,y in zip(xs,ys))/den
    reason='MISSING_DAY' if not complete else 'INSUFFICIENT_OR_CONSTANT_SAMPLE'
    return {'definition_version':'kdd27-metrics-v1','return_clock':'daily log return; no gap compression',
        'returns':returns,'lags':list(range(1,max_lag+1)),'raw_acf':raw,'absolute_acf':absolute,
        'metrics':{'excess_kurtosis':metric(kurt,len(valid),n,'ratio',reason),
        'mean_abs_return_acf_lags_1_20':metric(None if absolute is None else statistics.fmean(absolute),len(valid),n,'ratio',reason),
        'maximum_abs_raw_return_acf_lags_1_20':metric(None if raw is None else max(abs(x) for x in raw),len(valid),n,'ratio',reason),
        'volume_abs_return_correlation':metric(corr,len(valid),n,'ratio',reason)}}


def spread_bps(bid,ask,halted):
    _finite_optional([bid,ask])
    if any(v is not None and v<=0 for v in (bid,ask)) or (bid is not None and ask is not None and bid>ask):
        raise ValueError('Invalid/crossed quote')
    reason='HALTED' if halted else 'MISSING_QUOTE_SIDE'
    return metric(None if halted or bid is None or ask is None else 20000*(ask-bid)/(ask+bid),
        int(not halted and bid is not None and ask is not None),1,'bp',reason)


def daily_facts(rows,*,ticks_per_day,initial_price,initial_tick=0):
    if type(ticks_per_day) is not int or ticks_per_day<=0:raise ValueError('Invalid day clock')
    ticks=[r['tick'] for r in rows]
    if ticks and ticks!=list(range(initial_tick+1,ticks[-1]+1)):raise ValueError('Gap or duplicate committed tick')
    groups=defaultdict(list)
    for row in rows:groups[(row['tick']-1)//ticks_per_day+1].append(row)
    daily=[{'day':initial_tick//ticks_per_day,'close':float(initial_price) if initial_tick%ticks_per_day==0 else None,
        'volume':0.0,'complete':initial_tick%ticks_per_day==0,'reason':None if initial_tick%ticks_per_day==0 else 'PARTIAL_PREFIX_DAY'}]
    for day,group in groups.items():
        complete=len(group)==ticks_per_day and group[-1]['tick']==day*ticks_per_day
        daily.append({'day':day,'close':float(group[-1]['close']) if complete else None,
            'volume':float(sum(r['volume'] for r in group)) if complete else None,
            'complete':complete,'reason':None if complete else 'INCOMPLETE_MODEL_DAY'})
    return daily


def recovery(prices,*,horizon,consecutive=20,threshold=.98):
    if len(prices)!=horizon+1 or horizon<1 or consecutive<1:raise ValueError('Full pre-price plus declared horizon required')
    _finite_optional(prices)
    if any(p is None for p in prices):return {'mdd':None,'drop_from_pre':None,'confirmation':None,'restricted_time':None,'censored':None,'reason':'MISSING_PRICE'}
    if any(p<=0 for p in prices) or not 0<threshold<=1:raise ValueError('Invalid recovery inputs')
    peak=prices[0];mdd=0.0
    for price in prices:
        peak=max(peak,price);mdd=max(mdd,1-price/peak)
    trough=prices.index(min(prices));streak=0;confirmation=None
    for t in range(trough+1,len(prices)):
        streak=streak+1 if prices[t]>=threshold*prices[0] else 0
        if streak==consecutive:confirmation=t;break
    return {'mdd':mdd,'drop_from_pre':max(0.0,1-min(prices)/prices[0]),'trough':trough,
        'confirmation':confirmation,'restricted_time':horizon if confirmation is None else confirmation,
        'censored':confirmation is None,'reason':None if confirmation is not None else 'RIGHT_CENSORED'}


def persona_score(values,targets,*,trade_count,minimum_trades):
    if minimum_trades<1 or trade_count<0:raise ValueError('Explicit positive eligibility threshold required')
    if set(values)!=set(targets) or len(values)!=3:raise ValueError('All three persona dimensions required')
    _finite_optional(list(values.values()))
    if trade_count<minimum_trades or any(v is None for v in values.values()):return metric(None,0,1,'score','PERSONA_NOT_EVALUABLE')
    distances=[]
    for key,value in values.items():
        lo,hi,scale=targets[key];_finite_optional([lo,hi,scale])
        if lo is None or hi is None or lo>hi:raise ValueError('Invalid target interval')
        width=hi-lo if hi>lo else scale
        if width is None or width<=0:raise ValueError('Zero-width target needs a frozen positive scale')
        distances.append(min(1.0,max(lo-value,0.0,value-hi)/width))
    return metric(1-statistics.fmean(distances),1,1,'score')


def relative_turnover(treatment,control):
    _finite_optional([treatment,control])
    if any(v is not None and v<0 for v in (treatment,control)):raise ValueError('Negative turnover')
    return metric(None if treatment is None or control in (None,0.0) else treatment/control,1,1,'ratio',
        'ZERO_CONTROL_TURNOVER' if control==0.0 else 'MISSING_TURNOVER')


def summarize_facts(rows,*,initial_cash,initial_shares,ticks_per_day,initial_price,initial_tick=0):
    total=len(rows)
    if not total:raise ValueError('At least one committed tick required')
    spread=[r['spread_bps'] for r in rows if r['spread_bps'] is not None]
    active=sum(r['active'] for r in rows);orders=sum(r['orders'] for r in rows)
    metrics={'no_trade_ratio':ratio(sum(r['trade_count']==0 for r in rows),total),
        'two_sided_coverage':ratio(sum(r['two_sided'] for r in rows),total),
        'one_sided_ratio':ratio(sum(r['one_sided'] for r in rows),total),
        'empty_book_ratio':ratio(sum(r['empty_book'] for r in rows),total),
        'halt_ratio':ratio(sum(r['halted'] for r in rows),total),
        'two_sided_spread_bps':metric(statistics.fmean(spread) if spread else None,len(spread),total,'bp','NO_TWO_SIDED_OBSERVATIONS'),
        'rejected_order_ratio':ratio(sum(r['rejected_orders'] for r in rows),orders),
        'executed_order_ratio':ratio(sum(r['filled_orders'] for r in rows),orders),
        'cognitive_active_coverage':ratio(sum(r['covered'] for r in rows),active,'NO_ACTIVE_DECISIONS'),
        'degraded_active_ratio':ratio(sum(r['degraded'] for r in rows),active,'NO_ACTIVE_DECISIONS'),
        'traded_quantity':metric(sum(r['volume'] for r in rows),total,total,'shares'),
        'turnover':metric(sum(r['volume'] for r in rows)/initial_shares if initial_shares else None,total,total,'fraction_initial_shares','ZERO_INITIAL_SHARES'),
        'fees_minor':metric(sum(r['fees_step_minor'] for r in rows),total,total,'minor_currency'),
        'fee_cash_fraction':metric(sum(r['fees_step_minor'] for r in rows)/initial_cash if initial_cash else None,total,total,'fraction_initial_tradable_cash','ZERO_INITIAL_CASH'),
        'peak_infected_count':metric(max(r['infected'] for r in rows),total,total,'actors'),
        'infected_person_steps':metric(sum(r['infected'] for r in rows),total,total,'actor_steps'),
        'unique_infected_actors':metric(len(set(a for r in rows for a in r['infected_actor_ids'])),total,total,'actors'),
        'mean_confidence':metric(statistics.fmean(r['confidence_mean'] for r in rows),total,total,'score'),
        'exposure_items':metric(sum(r['exposure_items'] for r in rows),total,total,'selected_source_actor_steps'),
        'price_band_touch_ratio':ratio(sum(r['price_at_band'] for r in rows),total),
        'tradable_cash_end_minor':metric(rows[-1]['tradable_cash_minor'],total,total,'minor_currency')}
    for side in ('bid','ask'):
        values=[r[f'depth_{side}5'] for r in rows if r[f'depth_{side}5'] is not None]
        metrics[f'depth_{side}5']=metric(statistics.fmean(values) if values else None,len(values),total,'shares','NO_TWO_SIDED_OBSERVATIONS')
    daily=daily_facts(rows,ticks_per_day=ticks_per_day,initial_price=initial_price,initial_tick=initial_tick)
    daily_result=daily_statistics([r['day'] for r in daily],[r['close'] for r in daily],[r['volume'] for r in daily])
    return {'definition_version':'kdd27-metrics-v1','window':[rows[0]['tick'],rows[-1]['tick']],
        'metrics':{**metrics,**daily_result['metrics']},'daily':daily,'daily_statistics':daily_result}
