"""Read-only paper projections over the existing commit store and attempt registry."""
import csv
from datetime import date,datetime
import hashlib
import math
from pathlib import Path
import statistics
from fractions import Fraction
from typing import Literal

from pydantic import Field

from civitas.contracts.base import Record,Id,Digest,NonNegative,require_unique
from civitas.storage.commit_store import CommitStore
from civitas.experiments.registry import history
from .kdd27 import spread_bps,summarize_facts


class MetricSpec(Record):
    metric_id: Id
    definition_version: Literal['kdd27-metrics-v1']='kdd27-metrics-v1'
    unit: Id
    denominator: Id
    raw_fields: list[Id]=Field(min_length=1)
    clock: Id
    window: Id
    aggregation: Id
    missing_rule: Id
    statistical_unit: Literal['world_seed']='world_seed'
    paper_anchor: Id


class RunEvidence(Record):
    schema_version: Literal[1]=1
    experiment_id: Id
    condition_id: Id
    protocol_version: Id
    pair_id: Id
    world_seed: NonNegative
    model_sample_id: str|None
    run_id: str|None
    parent_run_id: str|None
    branch_point: int|None
    code_commit: str|None
    source_hash: Digest|None
    resolved_config_hash: Digest|None
    initial_state_hash: Digest|None
    data_version: Id
    model_id: str|None
    prompt_version: str|None
    parser_version: str|None
    world_schema: int|None
    status: Literal['planned','running','completed','failed','cancelled','budget_blocked','not_applicable']
    raw_status: Id
    failure_reason: str|None
    output_kind: Literal['measured_empirical','measured_simulation','measured_online_llm','replayed_llm_tape','mock_fixture','synthetic_illustration','cost_projection']
    input_kind: Literal['synthetic_initialization','historical_observations','analytic_fixture']
    execution_phase: Literal['smoke','pilot','confirmatory','exploratory','replication']
    evidence_scope: Literal['engineering','research','not_eligible']
    raw_path: str|None
    verified_integrity: bool
    attempts: list[dict]
    unknown_fields: list[str]


def require_figure_eligible(records):
    for raw in records:
        r=RunEvidence.model_validate(raw)
        if (r.output_kind not in {'measured_empirical','measured_simulation','measured_online_llm'}
                or r.execution_phase not in {'confirmatory','exploratory','replication'}
                or r.evidence_scope!='research' or r.status!='completed' or not r.verified_integrity
                or r.unknown_fields or any(x is None for x in (r.source_hash,r.resolved_config_hash,r.initial_state_hash,r.code_commit))):
            raise ValueError('Ineligible source/phase/status for a research figure')


def run_reference(store,*,experiment_id,condition_id,protocol_version,phase='smoke',scope='engineering'):
    identity=store.identity;cfg=identity.config
    if cfg.get('scenario')!='kdd27_v1':raise ValueError('Paper projections require kdd27_v1')
    if scope=='research' and cfg['protocol_id']!=protocol_version:raise ValueError('Run was not executed under this research protocol')
    if identity.participants.get('multiasset_market')!=3:raise ValueError('Unknown paper market participant schema')
    markers=store.verified_markers();records=history(store.root)
    last=next((r for r in reversed(records) if 'status' in r),{})
    raw=last.get('status','planned');mapped={'blocked':'budget_blocked','skipped':'not_applicable','initialized':'planned'}.get(raw,raw)
    if mapped not in {'planned','running','completed','failed','cancelled','budget_blocked','not_applicable'}:raise ValueError('Unknown execution status')
    mode=cfg['cognition_mode'];origins={'rule':'measured_simulation','synthetic_test':'mock_fixture','archived':'replayed_llm_tape','live':'measured_online_llm'}
    if mode not in origins:raise ValueError('Unknown model provenance')
    implementation=identity.metadata['implementation'];parent=identity.metadata.get('parent',{})
    provider=identity.metadata.get('provider_config') or {}
    return RunEvidence(experiment_id=experiment_id,condition_id=condition_id,protocol_version=protocol_version,
        pair_id=f"{experiment_id}:{cfg['root_seed']}",world_seed=cfg['root_seed'],model_sample_id=None,
        run_id=identity.run_id,parent_run_id=parent.get('run_id'),branch_point=parent.get('tick'),
        code_commit=implementation['code_commit'],source_hash=identity.code_hash,resolved_config_hash=identity.config_hash,
        initial_state_hash=markers[0].state_hash,data_version='synthetic_initialization:'+identity.config_hash,
        model_id=provider.get('model') if mode!='rule' else None,prompt_version=None,parser_version=None,
        world_schema=next(store.states())[1].schema_version,status=mapped,raw_status=raw,failure_reason=last.get('error') or last.get('reason'),
        output_kind=origins[mode],input_kind='synthetic_initialization',execution_phase=phase,evidence_scope=scope,
        raw_path=str(store.root),verified_integrity=True,attempts=records,
        unknown_fields=[] if mode=='rule' else ['model_sample_id','prompt_version','parser_version']).model_dump()


def project_snapshot(snapshot,*,price_tick,currency,asset):
    tick=snapshot.tick;data=snapshot.data;paper=data['kdd27'];market=data['multiasset_market']
    actors=set(snapshot.actors);accounts={r['account_id']:r for r in market['ledger']['accounts']}
    require_unique([r['account_id'] for r in market['ledger']['accounts']],'account primary key')
    require_unique([c['command_id'] for c in market['commands']],'command primary key')
    require_unique([r['command_id'] for r in market['receipts']],'receipt primary key')
    commands={c['command_id']:c for c in market['commands']};receipts={r['command_id']:r for r in market['receipts']}
    if commands.keys()!=receipts.keys():raise ValueError('Orphan command/receipt')
    decisions={d.decision_id:d for d in snapshot.decisions}
    require_unique([d.decision_id for d in snapshot.decisions],'decision primary key')
    for c in commands.values():
        if c['decision_id'] not in decisions or c['actor_id'] not in actors or c['tick']>tick:raise ValueError('Orphan/future order')
        if decisions[c['decision_id']].actor_id!=c['actor_id']:raise ValueError('Order actor differs from decision')
    all_orders={c['intent']['order_id'] for c in commands.values() if c['intent']['kind']=='portfolio_order'}
    journal=market['ledger']['journal'];require_unique([t['tx_id'] for t in journal],'transaction primary key')
    require_unique([t['trade']['trade_id'] for t in journal],'trade primary key')
    trades=[tx for tx in journal if tx['trade']['tick']==tick]
    filled_ids={tx['trade'][key] for tx in trades for key in ('buy_order_id','sell_order_id')}
    for tx in journal:
        t=tx['trade']
        if (t['buy_order_id'] not in all_orders or t['sell_order_id'] not in all_orders or t['tick']>tick
                or t['buyer_id'] not in actors or t['seller_id'] not in actors or t['buyer_id']==t['seller_id']):raise ValueError('Orphan/future/self trade')
        for resource in (currency,asset):
            if sum(p['amount'] for p in tx['postings'] if p['resource_id']==resource)!=0:raise ValueError('Unbalanced settlement postings')
        for p in tx['postings']:
            if p['account_id'] not in accounts:raise ValueError('Orphan ledger account')
    quote=paper['last_market_trace']['quotes'];bid,ask=quote['best_bid_ticks'],quote['best_ask_ticks']
    spread=spread_bps(None if bid is None else float(bid*price_tick),None if ask is None else float(ask*price_tick),paper['halted'])
    current=[c for c in commands.values() if c['tick']==tick and c['intent']['kind']=='portfolio_order']
    decision_rows=[]
    for d in snapshot.decisions:
        if d.tick!=tick:continue
        if 'paper_decision' not in d.belief_patch:raise ValueError('Missing paper decision trace')
        trace=d.belief_patch['paper_decision'];intent=d.economic_intents[0]
        if trace['decision_tick']!=tick or trace['observation_id']!=d.observation_id:raise ValueError('Decision clock/input mismatch')
        decision_rows.append({'tick':tick,'actor_id':d.actor_id,'decision_id':d.decision_id,'observation_id':d.observation_id,
            'active':trace['active'],'path':trace['path'],'covered':trace['covered'],'degraded':trace['degraded'],
            'attempt_id':trace['attempt_id'],'kind':intent.kind,'quantity':getattr(intent,'quantity_units',0)})
    if {d['actor_id'] for d in decision_rows}!=actors or len(decision_rows)!=len(actors):raise ValueError('Incomplete actor decision barrier')
    levels={'buy':{},'sell':{}}
    for venue in market['venues']:
        if venue['asset_id']!=asset:raise ValueError('Paper metric expects one asset')
        for order in venue['orders']:
            if order['status']!='active':continue
            side=levels[order['command']['side']];p=order['command']['limit_price_ticks']
            side[p]=side.get(p,0)+order['remaining_units']
    depth={side:sum(orders[p] for p in sorted(orders,reverse=side=='buy')[:5]) for side,orders in levels.items()}
    buys=sum(c['intent']['quantity_units'] for c in current if c['intent']['side']=='buy')
    sells=sum(c['intent']['quantity_units'] for c in current if c['intent']['side']=='sell')
    infected=[a for a,s in paper['states'].items() if s['infection']=='I']
    exposed=set(infected)|set(paper['direct_injected'])
    exposures=sum(len(t['social']['feed']) for t in paper['last_trace'].values())
    band=Fraction(str(paper['band']))
    reserved={}
    for r in market['ledger']['reservations']:
        key=(r['command']['account_id'],r['resource_id']);reserved[key]=reserved.get(key,0)+r['amount']
    available_cash={a:accounts[a]['balances'][currency]-reserved.get((a,currency),0) for a in actors}
    available_shares={a:accounts[a]['balances'][asset]-accounts[a]['locked'].get(asset,0)-reserved.get((a,asset),0) for a in actors}
    if min([*available_cash.values(),*available_shares.values()])<0:raise ValueError('Negative free resources')
    tick_minor=next(a['price_tick_minor'] for a in market['ledger']['assets'] if a['asset_id']==asset)
    price_minor=quote['last_price_ticks']*tick_minor;rate=market['ledger']['fees']['commission']
    one_share_cost=price_minor+price_minor*rate['numerator']//rate['denominator']
    row={'tick':tick,'close':float(quote['last_price_ticks']*price_tick),'trade_count':len(trades),
        'volume':sum(t['trade']['quantity_units'] for t in trades),'fees_step_minor':sum(t['buyer_fee_minor']+t['seller_fee_minor']+t['tax_minor'] for t in trades),
        'fees_cumulative_minor':accounts['fees']['balances'][currency],'spread_bps':spread['value'],'spread_reason':spread['reason'],
        'two_sided':spread['value'] is not None,'one_sided':(bid is None)!=(ask is None),'empty_book':bid is None and ask is None,
        'depth_bid5':depth['buy'] if spread['value'] is not None else None,'depth_ask5':depth['sell'] if spread['value'] is not None else None,
        'halted':paper['halted'],'orders':len(current),'rejected_orders':sum(receipts[c['command_id']]['status']=='rejected' for c in current),
        'filled_orders':sum(c['intent']['order_id'] in filled_ids for c in current),
        'rejection_reasons':{reason:sum(reason in receipts[c['command_id']]['reason_codes'] for c in current)
            for reason in {v for c in current for v in receipts[c['command_id']]['reason_codes']}},
        'active':sum(d['active'] for d in decision_rows),'covered':sum(d['active'] and d['covered'] for d in decision_rows),
        'degraded':sum(d['active'] and d['degraded'] for d in decision_rows),'infected':len(infected),
        'infected_actor_ids':sorted(exposed),'exposure_items':exposures,
        'confidence_mean':statistics.fmean(s['confidence'] for s in paper['states'].values()),
        'order_imbalance':(buys-sells)/(buys+sells) if buys+sells else None,
        'net_sell_quantity':sells-buys,
        'price_at_band':abs(quote['last_price_ticks']-paper['day_anchor_ticks'])*band.denominator>=paper['day_anchor_ticks']*band.numerator,
        'tradable_cash_minor':sum(accounts[a]['balances'][currency] for a in actors),
        'free_cash_minor':sum(available_cash.values()),'free_shares':sum(available_shares.values()),
        'buy_feasible_actors':sum(v>=one_share_cost for v in available_cash.values()),
        'sell_feasible_actors':sum(v>=1 for v in available_shares.values()),
        'marked_wealth_minor':sum(accounts[a]['balances'][currency]+accounts[a]['balances'][asset]*price_minor for a in actors)}
    return row,decision_rows,trades


def analyze_run(root,*,experiment_id,condition_id,protocol_version):
    store=CommitStore(root);reference=run_reference(store,experiment_id=experiment_id,condition_id=condition_id,protocol_version=protocol_version)
    if reference['status']!='completed':raise ValueError('Incomplete worlds may be indexed but not given complete outcomes')
    cfg=store.identity.config;market=cfg['market'];rows=[];decisions=[];trades=[];hashes=[];initial=None
    for marker,snapshot in store.states():
        if initial is None:initial=snapshot;continue
        row,ds,ts=project_snapshot(snapshot,price_tick=market['price_tick'],currency=market['currency'],asset=market['asset_id'])
        rows.append(row);decisions.extend(ds);trades.extend(ts)
        hashes.append({'tick':marker.tick,'commit_hash':marker.commit_hash,'state_hash':marker.state_hash})
    accounts=initial.data['multiasset_market']['ledger']['initial_accounts'];actors=set(initial.actors)
    summary=summarize_facts(rows,initial_cash=sum(a['balances'][market['currency']] for a in accounts if a['account_id'] in actors),
        initial_shares=sum(a['balances'][market['asset_id']] for a in accounts if a['account_id'] in actors),
        ticks_per_day=market['ticks_per_day'],initial_tick=initial.tick,
        initial_price=initial.data['kdd27']['last_quote']['last_price_ticks']*market['price_tick'])
    return {'run':reference,'facts':rows,'decisions':decisions,'trades':trades,'committed_hashes':hashes,'summary':summary,
        'model_cost':invocation_costs(store),
        'analysis_is_new_simulation':False,'formal_eligible':False}


def invocation_costs(store):
    if store.identity.config['cognition_mode']=='rule':
        return {'owned_attempts':0,'input_tokens':0,'output_tokens':0,'unknown_usage_attempts':0,
            'known_accounted_nano':{'USD':0,'CNY':0},'total_accounted_nano':{'USD':0,'CNY':0},
            'scope':'No provider configured; not an online cost benchmark'}
    from civitas.cognition.invocations import ModelInvocationStore
    ledger=ModelInvocationStore(store.identity.metadata['control_ledger'],read_only=True)
    records=[r for r in ledger.records() if r['run_id']==store.identity.run_id]
    with ledger.transaction() as db:
        reservations=[dict(r) for r in db.execute('SELECT * FROM reservations WHERE run_id=?',(store.identity.run_id,))]
    unknown=sum(r['incoming'] is None for r in reservations)
    known={c:sum(r['accounted'] or 0 for r in reservations if r['currency']==c) for c in ('USD','CNY')}
    totals={c:None if any(r['accounted'] is None for r in reservations if r['currency']==c) else known[c] for c in known}
    return {'owned_attempts':len(records),'unique_owned_requests':len({r['request_id'] for r in records}),
        'input_tokens':None if unknown else sum(r['incoming'] for r in reservations),
        'output_tokens':None if unknown else sum(r['outgoing'] for r in reservations),
        'unknown_usage_attempts':unknown,'known_accounted_nano':known,'total_accounted_nano':totals,
        'reserved_unknown_nano':{c:sum(r['reserved'] for r in reservations if r['currency']==c and r['accounted'] is None) for c in known},
        'valid_response_attempts':sum(r['state']=='success' for r in records),
        'scope':'Owned attempts only, including world-uncommitted facts; inherited tape/cache use is not a new call',
        'source':store.identity.config['cognition_mode']}


def load_daily_csv(path,*,mapping,manifest,as_of):
    """Explicit real headers and metadata only; no guessed vendor codes or fetch."""
    path=Path(path)
    if manifest['output_kind']!='measured_empirical' or not manifest['license_id']:raise ValueError('Empirical provenance and license required')
    if manifest['price_unit']!='CNY/share' or manifest['volume_unit']!='shares':raise ValueError('Unsupported/ambiguous units')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Raw file hash mismatch')
    fields={'asset_id','date','close_raw','close_adjusted','adjustment_factor','volume','halted','source_time','available_at'}
    if set(mapping)!=fields or len(set(mapping.values()))!=len(mapping):raise ValueError('Explicit unique concept/header mapping required')
    cutoff=datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:raise ValueError('As-of timestamp must include timezone')
    rows=[];seen=set()
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        require_unique(reader.fieldnames or [],'CSV header')
        if not set(mapping.values())<=set(reader.fieldnames or []):raise ValueError('Unknown real CSV headers')
        for raw in reader:
            row={k:raw[v] for k,v in mapping.items()};key=(row['asset_id'],row['date'])
            if key in seen:raise ValueError('Duplicate asset/date primary key')
            seen.add(key);date.fromisoformat(row['date'])
            source,available=[datetime.fromisoformat(row[k]) for k in ('source_time','available_at')]
            if source.tzinfo is None or available.tzinfo is None or not source<=available<=cutoff:raise ValueError('Future/unordered availability')
            if row['asset_id'] not in manifest['universe']:raise ValueError('Asset outside frozen universe')
            for key in ('close_raw','close_adjusted','adjustment_factor','volume'):
                value=None if row[key]=='' else float(row[key])
                if value is not None and (not math.isfinite(value) or value<0 or (key!='volume' and value==0)):
                    raise ValueError('Invalid market value')
                row[key]=value
            if all(row[k] is not None for k in ('close_raw','close_adjusted','adjustment_factor')) and not math.isclose(
                    row['close_adjusted'],row['close_raw']*row['adjustment_factor'],rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Adjusted price differs from declared multiplicative factor')
            if row['halted'] not in {'0','1'}:raise ValueError('Explicit 0/1 halt flag required')
            row['halted']=row['halted']=='1';rows.append(row)
    return rows
