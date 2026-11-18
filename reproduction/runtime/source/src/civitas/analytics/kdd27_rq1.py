"""RQ1 window and data adapters over the frozen KDD03 metrics; no data fetching."""
from datetime import date,datetime
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from core.experiments.identity import file_sha256
from core.validation.calibration import DataSource
from civitas.contracts.base import content_hash,require_unique
from .kdd27 import return_statistics,summarize_facts
from .kdd27_data import load_daily_csv
from .kdd27_statistics import synchronous_blocks,empirical_distance

PRIMARY=('excess_kurtosis','mean_abs_return_acf_lags_1_20','maximum_abs_raw_return_acf_lags_1_20',
         'volume_abs_return_correlation','two_sided_spread_bps')


def rq1_run_metrics(facts,*,config,initial_cash,initial_shares):
    """Retain the burn-in close as the first return anchor, without sampling it."""
    ticks=[r['tick'] for r in facts]
    if ticks!=list(range(1,config.ticks+1)):
        raise ValueError('RQ1 complete outcomes require every committed tick through the horizon')
    start=config.burn_in_ticks
    result=summarize_facts(facts[start:],initial_cash=initial_cash,initial_shares=initial_shares,
        ticks_per_day=config.market.ticks_per_day,initial_tick=start,
        initial_price=facts[start-1]['close'] if start else config.market.initial_price)
    result.update(burn_in_ticks=start,expected_daily_returns=(config.ticks-start)//config.market.ticks_per_day,
        formal_eligible=False,admission_note='Window extraction only; source/data/campaign gates must be checked separately')
    return result


def import_daily_reference(path,*,mapping,manifest,as_of,calendar):
    """Bind supplied files and PIT universe evidence; never guess a vendor header."""
    source=DataSource.model_validate(manifest['source'])
    frozen=datetime.fromisoformat(manifest['universe_fixed_at'])
    acquired=datetime.fromisoformat(source.acquired_at)
    cutoff=datetime.fromisoformat(as_of)
    if (frozen.tzinfo is None or cutoff.tzinfo is None or
            frozen>=datetime(2019,1,1,tzinfo=ZoneInfo('Asia/Shanghai')) or not frozen<=acquired<=cutoff):
        raise ValueError('Universe must be fixed before2019; source acquisition cannot be in the future')
    evidence=Path(manifest['universe_evidence'])
    if file_sha256(evidence)!=manifest['universe_evidence_sha256']:
        raise ValueError('Universe evidence content mismatch')
    if source.license!=manifest['license_id']:
        raise ValueError('Source license differs from the file contract')
    rows=load_daily_csv(path,mapping=mapping,manifest=manifest,as_of=as_of)
    panel=daily_panel(rows,assets=manifest['universe'],calendar=calendar)
    return {**panel,'source':source.model_dump(),'data_sha256':manifest['sha256'],
        'universe_evidence_sha256':manifest['universe_evidence_sha256'],'calendar_hash':content_hash(calendar),
        'output_kind':manifest['output_kind'],'universe_selection_audit':'REQUIRED_BEFORE_FORMAL_USE',
        'empirical_ready':False,'reason':'Daily import alone does not qualify PIT selection, license scope or intraday sampling'}


def daily_panel(rows,*,assets,calendar):
    require_unique(assets,'fixed stock universe')
    if (len(assets)!=20 or any(type(a) is not str or not a for a in assets) or
            calendar!=sorted(set(calendar)) or len(calendar)<2):
        raise ValueError('Exactly20 fixed stocks and a unique increasing trading calendar required')
    days=[date.fromisoformat(d) for d in calendar];lookup={};asset_set=set(assets);day_set=set(calendar)
    for row in rows:
        key=(row['asset_id'],row['date'])
        if key in lookup or key[0] not in asset_set or key[1] not in day_set:
            raise ValueError('Duplicate or outside-universe/calendar daily record')
        lookup[key]=row
    features=[];missing=[]
    for previous,current in zip(calendar,calendar[1:]):
        group=[]
        for asset in assets:
            a,b=lookup.get((asset,previous)),lookup.get((asset,current))
            pa=None if a is None else a['close_adjusted'];pb=None if b is None else b['close_adjusted']
            volume=None if b is None else b['volume']
            for p in (pa,pb):
                if p is not None and (not math.isfinite(p) or p<=0):raise ValueError('Invalid adjusted price')
            if volume is not None and (not math.isfinite(volume) or volume<0):raise ValueError('Invalid volume')
            ret=None if pa is None or pb is None else math.log(pb/pa)
            group.append([ret,volume])
            if ret is None or volume is None:missing.append({'asset':asset,'date':current,'reason':'MISSING_ADJACENT_PRICE_OR_VOLUME'})
        features.append(group)
    return {'assets':assets,'return_dates':calendar[1:],'features':features,'missing':missing,
        'halted_observations':sum(r['halted'] for r in rows),'raw_rows':len(rows),
        'split_years':{'design':[2019,2020,2021],'holdout':[2022,2023]},'intraday_status':'NOT_PROVIDED',
        'required_primary_metrics':5,'price_clock':'supplied trading calendar; no gap compression'}


def empirical_daily_blocks(panel,*,split,window_days=150,draws=5000,seed=931402):
    if split not in {'design','holdout'}:raise ValueError('Use the frozen design or holdout partition')
    years=set(panel['split_years'][split]);selection=[i for i,d in enumerate(panel['return_dates']) if date.fromisoformat(d).year in years]
    if selection and selection!=list(range(selection[0],selection[-1]+1)):
        raise ValueError('Split cannot concatenate disjoint date ranges')
    data=np.asarray([panel['features'][i] for i in selection],dtype=float)
    if len(data)<max(20,window_days):
        return {'status':'pending','reason':'INSUFFICIENT_TRADING_DAYS','observed_days':len(data),'split':split,'required_metrics':5}
    starts=[i for i in range(len(data)-19) if np.isfinite(data[i:i+20]).all()]
    if not starts:return {'status':'pending','reason':'NO_COMPLETE_SYNCHRONOUS_BLOCK','split':split,'required_metrics':5}
    indices,samples=synchronous_blocks(data,window_days=window_days,draws=draws,seed=seed,valid_starts=starts)
    distribution={k:[] for k in PRIMARY}
    for sample in samples:
        statistics_by_asset=[return_statistics(sample[:,a,0].tolist(),sample[:,a,1].tolist())['metrics'] for a in range(len(panel['assets']))]
        for key in PRIMARY:
            values=[m[key]['value'] for m in statistics_by_asset] if key in statistics_by_asset[0] else [None]
            distribution[key].append(None if any(v is None for v in values) else math.fsum(values)/len(values))
    summaries={}
    for key,values in distribution.items():
        valid=[v for v in values if v is not None]
        q=np.quantile(valid,[.025,.25,.5,.75,.975]).tolist() if len(valid)==draws else None
        summaries[key]={'median':None if q is None else q[2],'interval95':None if q is None else [q[0],q[4]],
            'iqr':None if q is None else q[3]-q[1],'valid_draws':len(valid),'planned_draws':draws}
    dates=[panel['return_dates'][i] for i in selection]
    return {'status':'daily_only','split':split,'split_input_hash':content_hash({'dates':dates,'features':[panel['features'][i] for i in selection]}),
        'dates':dates,'eligible_block_starts':starts,'blocks_count':len(starts),'all_possible_blocks':len(data)-19,
        'block_days':20,'window_days':window_days,'draws':draws,'seed':seed,'synchronous_indices_hash':content_hash(indices.tolist()),
        'distribution':distribution,'metrics':summaries,'required_metrics':5,'h1_status':'INCOMPLETE_H1',
        'missing_component':'two_sided_spread_bps','formal_eligible':False}


def h1_decision(*,simulated,holdout_medians,holdout_intervals,design_iqr,paired_distance_intervals,
                paired_health_intervals,health_margins,campaign_ready):
    """No positive H1 result can bypass data, paired contrasts or health gates."""
    if any(set(d)!=set(PRIMARY) for d in (simulated,holdout_medians,holdout_intervals,design_iqr)):
        raise ValueError('H1 retains exactly five named primary metrics')
    baselines={'zero_intelligence','homogeneous_linear_value'}
    if set(paired_distance_intervals)!=baselines or set(paired_health_intervals)!=baselines:
        raise ValueError('Both prespecified paired baselines are required')
    distance=empirical_distance(simulated,holdout_medians,design_iqr)
    def interval(v):
        if v is None:return False
        if len(v)!=2 or any(type(x) is not float or not math.isfinite(x) for x in v) or v[0]>v[1]:raise ValueError('Invalid frozen interval')
        return True
    issues=[]
    if not campaign_ready:issues.append('CAMPAIGN_NOT_READY')
    if distance['value'] is None:issues.append('INCOMPLETE_H1')
    observed=[]
    for key in PRIMARY:
        bounds=holdout_intervals[key]
        observed.append(bounds[0]<=simulated[key]<=bounds[1] if interval(bounds) and simulated[key] is not None else None)
    if any(v is None for v in observed):issues.append('MISSING_PRIMARY_INTERVAL_OR_VALUE')
    improved=[];healthy=[]
    for baseline in sorted(baselines):
        ci=paired_distance_intervals[baseline]
        improved.append(ci[1]<0 if interval(ci) else None)
        if not interval(ci):issues.append('MISSING_PAIRED_DISTANCE')
        if set(paired_health_intervals[baseline])!={'no_trade_ratio','one_sided_ratio'} or set(health_margins)!={'no_trade_ratio','one_sided_ratio'}:
            raise ValueError('Both frozen market-health endpoints are required')
        for endpoint,ci in paired_health_intervals[baseline].items():
            margin=health_margins[endpoint]
            if margin is None or not interval(ci):issues.append('HEALTH_THRESHOLD_OR_INTERVAL_NOT_FROZEN');healthy.append(None)
            elif type(margin) is not float or not math.isfinite(margin) or margin<0:raise ValueError('Invalid health margin')
            else:healthy.append(ci[1]<=margin)
    return {'status':'pending' if issues else 'supported' if sum(observed)>=4 and all(improved) and all(healthy) else 'not_supported',
        'reasons':sorted(set(issues)),'required_metrics':5,'inside_count':sum(v is True for v in observed),
        'distance':distance,'distance_contrast':'full minus baseline; entire95% CI below0 required',
        'health_rule':'upper paired95% CI <= independent frozen noninferiority margin',
        'condition_results':{'distance_improved':improved,'health_not_worse':healthy}}
