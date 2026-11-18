"""Post-run projections reuse the existing nullable research measurements."""
from collections import defaultdict
from core.experiments.design_config import MeasurementSpec
from core.experiments.measurement import MarketSnapshot, MeasurementContext, PriceReference, SettledTrade, StepActivity
from core.market_metrics import MarketMetrics
from civitas.contracts.base import content_hash


def society_metrics(world):
    scene, config = world.scenario, world.scenario.config
    by_tick = defaultdict(list)
    if scene.legacy:
        asset = config.market.experiment.market.symbol
        for raw in scene.trades():
            by_tick[(int(raw['timestamp']),asset)].append(SettledTrade.model_validate(raw))
        initials = {asset:config.market.experiment.market.initial_price}
        factors = {asset:1.0}
    else:
        factors = {a.asset_id:a.price_tick_minor/config.market.population.cash_scale for a in config.market.population.assets}
        initials = {a:p*factors[a] for a,p in config.market.population.initial_price_ticks.items()}
        scale = config.market.population.cash_scale
        for tx in scene.market.ledger.journal:
            t = tx.trade
            maker = t.buyer_id if t.maker_order_id==t.buy_order_id else t.seller_id
            taker = t.seller_id if maker==t.buyer_id else t.buyer_id
            by_tick[(t.tick,t.asset_id)].append(SettledTrade(trade_id=t.trade_id,price=t.price_ticks*factors[t.asset_id],
                quantity=t.quantity_units,maker_id=t.maker_order_id,taker_id=t.taker_order_id,
                maker_agent_id=maker,taker_agent_id=taker,buyer_agent_id=t.buyer_id,seller_agent_id=t.seller_id,
                timestamp=float(t.tick),buyer_fee=tx.buyer_fee_minor/scale,seller_fee=tx.seller_fee_minor/scale,
                seller_tax=tx.tax_minor/scale))
    last, rows = {}, []
    identity = content_hash(config)
    for step in world.snapshot.data['society']['steps']:
        tick = step['tick']
        for asset,quote in step['quotes'].items():
            tape = by_tick[(tick,asset)]
            if tape:
                last[asset] = tape[-1]
            previous = last.get(asset)
            suffix = '' if scene.legacy else '_ticks'
            bid, ask = [quote['best_'+side+suffix] for side in ('bid','ask')]
            bid, ask = [None if value is None else value*factors[asset] for value in (bid,ask)]
            depth = quote.get('depth')
            levels = lambda side: [{'price':float(r['price']),'quantity':r['qty']} for r in depth[side]] if depth is not None else None
            snapshot = MarketSnapshot(captured=True,best_bid=bid,best_ask=ask,
                bid_depth=levels('bids'),ask_depth=levels('asks'),depth_levels=5,
                current_quote_source='book_end_market',quote_as_of_step=tick,
                last_trade_price=previous.price if previous else None,last_trade_step=int(previous.timestamp) if previous else None,
                history_reference=PriceReference(price=previous.price if previous else initials[asset],
                    source='last_trade' if previous else 'initial_condition',as_of_step=int(previous.timestamp) if previous else 0))
            context = MeasurementContext(spec=MeasurementSpec(evaluation_end=config.ticks+1),world_id=identity,run_id=identity,
                step=tick,session_id=str((tick-1)//(config.market.experiment.market.steps_per_day if scene.legacy else config.market.ticks_per_day)),
                policy_hash=content_hash(config.behavior),market_open=not step['halted'],observation_complete=True,
                execution_status='completed',origin='market')
            measured = MarketMetrics.compute_v2(snapshot=snapshot,trade_tape=tape,
                activity=StepActivity(settled_trade_ids=[t.trade_id for t in tape],cancel_events=None,reject_events=None),context=context)
            rows.append({'tick':tick,'asset_id':asset,'has_two_sided_quote':bid is not None and ask is not None,
                'missing_side':'both' if bid is None and ask is None else 'bid' if bid is None else 'ask' if ask is None else None,
                'metrics':{k:v.model_dump() for k,v in measured.metrics.items()},
                'run_rejected_commands_step':step['rejected_commands'],'fees_cumulative':step['fees']})
    return {'definition':'existing research measurement v2','price_unit':'quote currency; integer ticks converted only in this report',
        'fee_unit':'currency' if scene.legacy else 'minor currency units','data_origin':'simulation',
        'backend':config.backend,'steps':rows,'two_sided_availability':sum(r['has_two_sided_quote'] for r in rows)/len(rows) if rows else None}
