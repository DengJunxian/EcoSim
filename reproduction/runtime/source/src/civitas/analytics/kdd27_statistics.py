"""World-level inference; missing worlds remain in the planned denominator."""
import math
import statistics

import numpy as np

from core.eval.stat_tests import bootstrap_ci
from core.validation.statistics import _finite_optional


def world_summary(rows, *, seed=931401):
    """One already-selected observation per independent world, never per actor."""
    identities=[r['world_seed'] for r in rows]
    if len(set(identities))!=len(identities):
        raise ValueError('Duplicate world: select attempts and reduce nested model samples explicitly')
    values=[];excluded=[]
    for row in rows:
        value=row['value'];_finite_optional([value])
        if row['status']!='completed' and value is not None:
            raise ValueError('Failed/incomplete worlds cannot contribute numeric outcomes')
        if row['status']=='completed' and value is not None:values.append(value)
        else:excluded.append({'world_seed':row['world_seed'],'status':row['status'],'reason':row.get('reason') or 'MISSING_OUTCOME'})
    interval=bootstrap_ci(values,n_bootstrap=5000,ci=.95,seed=seed) if len(values)>1 else None
    return {'planned_worlds':len(rows),'observed_worlds':len(values),'excluded':excluded,
        'mean':statistics.fmean(values) if values else None,
        'sd_between_worlds':statistics.stdev(values) if len(values)>1 else None,
        'ci95':None if interval is None else [interval['lower'],interval['upper']],
        'ci_reason':None if interval else 'FEWER_THAN_TWO_WORLDS','resampling_unit':'world_seed',
        'bootstrap_draws':5000,'bootstrap_seed':seed,'interval_kind':'percentile_pointwise',
        'selection_warning':'Complete-world subset; failures retained and not imputed' if excluded else None}


def paired_summary(rows, *, control, treatment, seed=931401):
    lookup={};worlds=set()
    for r in rows:
        if r['condition_id'] not in {control,treatment}:raise ValueError('Unexpected condition')
        key=(r['world_seed'],r['condition_id'])
        if key in lookup:raise ValueError('Duplicate world/condition, including unselected retries')
        lookup[key]=r;worlds.add(r['world_seed'])
    differences=[]
    for world in sorted(worlds):
        a,b=[lookup.get((world,c)) for c in (control,treatment)]
        good=a is not None and b is not None
        if good and (a['pair_id']!=b['pair_id'] or a['protocol_version']!=b['protocol_version']):
            raise ValueError('Paired worlds must share pair and protocol identity')
        for r in (a,b):
            if r is not None:
                _finite_optional([r['value']])
                if r['status']!='completed' and r['value'] is not None:raise ValueError('Failed world has a value')
        good=good and all(r['status']=='completed' and r['value'] is not None for r in (a,b))
        differences.append({'world_seed':world,'status':'completed' if good else 'incomplete',
            'value':b['value']-a['value'] if good else None,'reason':None if good else 'INCOMPLETE_PAIR'})
    return {'contrast':f'{treatment} - {control}','differences':differences,**world_summary(differences,seed=seed)}


def factorial_interaction(cells):
    if set(cells)!={'00','01','10','11'}:raise ValueError('All four paired cells are required')
    _finite_optional(list(cells.values()))
    return None if any(v is None for v in cells.values()) else (cells['11']-cells['10'])-(cells['01']-cells['00'])


def holm(p_values):
    _finite_optional(list(p_values.values()))
    if any(v is not None and not 0<=v<=1 for v in p_values.values()):raise ValueError('Invalid p value')
    result={key:None for key in p_values};previous=0.0
    for rank,(key,p) in enumerate(sorted(((k,v) for k,v in p_values.items() if v is not None),key=lambda x:x[1])):
        previous=max(previous,min(1.0,(len(p_values)-rank)*p));result[key]=previous
    return {'adjusted':result,'family_size':len(p_values),'family_complete':all(v is not None for v in p_values.values())}


def synchronous_blocks(panel, *, window_days, block_days=20, draws=5000, seed=931402, valid_starts=None):
    """Resample already-computed daily features on common trading-date indices."""
    array=np.asarray(panel,dtype=float)
    if array.ndim not in (2,3):raise ValueError('Aligned date x asset [x feature] panel required')
    if any(type(v) is not int or v<=0 for v in (window_days,block_days,draws)):
        raise ValueError('Positive integer block/window/draw counts required')
    if len(array)<max(block_days,window_days):raise ValueError('Insufficient observed design/holdout window')
    if valid_starts is None:
        if not np.isfinite(array).all():raise ValueError('Finite aligned panel required')
        valid_starts=list(range(len(array)-block_days+1))
    if (not valid_starts or len(set(valid_starts))!=len(valid_starts) or any(type(i) is not int or
            not 0<=i<=len(array)-block_days or not np.isfinite(array[i:i+block_days]).all() for i in valid_starts)):
        raise ValueError('Eligible blocks must be unique, contiguous and fully observed')
    rng=np.random.default_rng(seed);blocks=math.ceil(window_days/block_days)
    starts=np.asarray(valid_starts)[rng.integers(0,len(valid_starts),size=(draws,blocks))]
    indices=(starts[:,:,None]+np.arange(block_days)).reshape(draws,-1)[:,:window_days]
    return indices,array[indices]


def empirical_distance(simulated, medians, design_iqr):
    if set(simulated)!=set(medians) or set(simulated)!=set(design_iqr) or len(simulated)!=5:
        raise ValueError('Original H1 always requires all five primary metrics')
    for values in (simulated,medians,design_iqr):_finite_optional(list(values.values()))
    invalid={k:('MISSING_COMPONENT' if any(d[k] is None for d in (simulated,medians,design_iqr)) else 'ZERO_DESIGN_IQR')
        for k in simulated if any(d[k] is None for d in (simulated,medians,design_iqr)) or design_iqr[k]==0}
    if any(v is not None and v<0 for v in design_iqr.values()):raise ValueError('Negative IQR')
    return {'value':None if invalid else statistics.fmean(abs(simulated[k]-medians[k])/design_iqr[k] for k in simulated),
        'required_metrics':5,'excluded_components':invalid,'reason':'INCOMPLETE_H1' if invalid else None}


def equivalent(interval, *, margin):
    if margin is None:return {'equivalent':None,'reason':'MARGIN_NOT_FROZEN'}
    if type(margin) is not float or not math.isfinite(margin) or margin<=0:raise ValueError('Positive frozen margin required')
    if interval is None:return {'equivalent':None,'reason':'INTERVAL_NOT_ESTIMATED'}
    _finite_optional(interval)
    if len(interval)!=2 or any(v is None for v in interval) or interval[0]>interval[1]:raise ValueError('Invalid interval')
    return {'equivalent':-margin<interval[0] and interval[1]<margin,'reason':None}
