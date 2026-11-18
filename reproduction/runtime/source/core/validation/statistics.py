"""Reference statistics on the original declared clock, with missingness kept."""
from __future__ import annotations

import math
from core.experiments.strict_types import ResearchInputError


def _finite_optional(values):
    if not isinstance(values,list) or any(value is not None and
        (type(value) is not float or not math.isfinite(value)) for value in values):
        raise ResearchInputError('Statistics need explicit finite binary64 values or null')


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def _quantile(values,probability):
    if not values:
        return None
    ordered=sorted(values);position=(len(ordered)-1)*probability
    low=math.floor(position);high=math.ceil(position)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def compute_reference_statistics(*, clock: list[int], prices: list[float|None],
                                 volumes: list[float|None]|None = None,
                                 frequency: str, lags: list[int] = [1,2,5]) -> dict:
    if (not isinstance(clock,list) or not clock or any(type(t) is not int for t in clock)
            or clock!=sorted(set(clock)) or len(clock)!=len(prices)
            or frequency not in {'daily','tick','step'}):
        raise ResearchInputError('A unique increasing explicit clock and frequency are required')
    if (not isinstance(lags,list) or not lags or any(type(lag) is not int or lag<1 for lag in lags)
            or len(set(lags))!=len(lags)):
        raise ResearchInputError('ACF lags must be unique positive clock offsets')
    _finite_optional(prices)
    if any(p is not None and p<=0.0 for p in prices):
        raise ResearchInputError('Observed prices must be positive')
    if volumes is not None:
        _finite_optional(volumes)
        if len(volumes)!=len(clock) or any(v is not None and v<0.0 for v in volumes):
            raise ResearchInputError('Volumes must align and remain nonnegative')
    pmap=dict(zip(clock,prices))
    vmap={} if volumes is None else dict(zip(clock,volumes))
    returns={t:(pmap[t]/pmap[t-1]-1.0 if pmap.get(t) is not None and pmap.get(t-1) is not None else None)
             for t in range(clock[0]+1,clock[-1]+1)}
    if any(r is not None and not math.isfinite(r) for r in returns.values()):
        raise ResearchInputError('Nonfinite derived return')
    valid=[r for r in returns.values() if r is not None]
    def metric(value,n,denominator,unit,reason=None):
        return {'value':value,'status':'VALID' if value is not None else 'UNDEFINED',
            'reason':None if value is not None else reason or 'INSUFFICIENT_OR_CONSTANT_SAMPLE',
            'sample_count':n,'denominator_count':denominator,'unit':unit,
            'uncertainty':{'status':'NOT_ESTIMATED','interval':None,'reason':'DESCRIPTIVE_REFERENCE_ONLY'}}
    n,total=len(valid),len(returns);mean=_mean(valid)
    variance=math.fsum((r-mean)**2 for r in valid)/(n-1) if n>1 else None
    metrics={'mean_return':metric(mean,n,total,'ratio'),
        'std_return':metric(None if variance is None else math.sqrt(variance),n,total,'ratio'),
        'return_q05':metric(_quantile(valid,0.05),n,total,'ratio'),
        'return_q95':metric(_quantile(valid,0.95),n,total,'ratio'),
        'mean_abs_return':metric(_mean([abs(r) for r in valid]),n,total,'ratio')}
    second=math.fsum((r-mean)**2 for r in valid)/n if n else None
    kurtosis=(math.fsum((r-mean)**4 for r in valid)/n/(second*second)-3.0
              if n>=4 and second>0.0 else None)
    metrics['excess_kurtosis']=metric(kurtosis,n,total,'ratio')
    for transform,name in [(lambda r:r,'return_acf'),(abs,'absolute_return_acf'),(lambda r:r*r,'squared_return_acf')]:
        series={t:transform(r) for t,r in returns.items() if r is not None}
        center=_mean(list(series.values()))
        divisor=math.fsum((r-center)**2 for r in series.values()) if series else 0.0
        for lag in lags:
            pairs=[(series[t-lag],r) for t,r in series.items() if t-lag in series]
            value=math.fsum((x-center)*(y-center) for x,y in pairs)/divisor if pairs and divisor>0.0 else None
            metrics[f'{name}_{lag}']=metric(value,len(pairs),max(0,total-lag),'ratio')
    observed_volumes=[v for v in vmap.values() if v is not None]
    metrics['mean_volume']=metric(_mean(observed_volumes),len(observed_volumes),clock[-1]-clock[0]+1,'shares')
    pairs=[(abs(r),vmap[t]) for t,r in returns.items() if r is not None and vmap.get(t) is not None]
    correlation=None
    if len(pairs)>=2:
        xm,ym=_mean([p[0] for p in pairs]),_mean([p[1] for p in pairs])
        denominator=math.sqrt(math.fsum((x-xm)**2 for x,y in pairs)*math.fsum((y-ym)**2 for x,y in pairs))
        if denominator>0.0:
            correlation=math.fsum((x-xm)*(y-ym) for x,y in pairs)/denominator
    metrics['volume_absolute_return_correlation']=metric(correlation,len(pairs),total,'ratio')
    return {'definition_version':'reference-statistics-v1','frequency':frequency,
        'clock_rule':'original_adjacent_only','quantile_rule':'linear_n_minus_one',
        'acf_rule':'center_on_all_observed_values_fixed_variance_sum',
        'returns':[{'step':t,'value':value} for t,value in returns.items()], 'metrics':metrics}
