"""Executable semantic witnesses attached to a scoped gate decision.

These checks produce their inputs and observed values. They supplement complete
archive verification and independent health acceptance; a test exit code is not
an input to certification.
"""
from pathlib import Path
import math

from core.experiments.identity import research_hash
from core.experiments.strict_types import strict_json_loads
from core.experiments.design_config import ResearchRunSpec,MeasurementSpec
from core.experiments.archive import rows,verify_archive_v3


def measurement_witness(spec):
    from core.experiments.measurement import (MarketSnapshot,DepthLevel,StepActivity,MeasurementContext,
        compute_step_measurement,aggregate_run_metrics)
    from core.interventions.compiler import compile_policy
    policy=compile_policy(spec.policy,constraints=spec.policy_constraints,world=spec.world,market=spec.base.market)
    ms=spec.measurement.model_dump(mode='json');ms.update(evaluation_start=1,evaluation_end=4)
    measurement=MeasurementSpec.model_validate(ms);cases=[];all_pass=True
    def compute(step,bid,ask,market_open=True):
        snapshot=MarketSnapshot(captured=True,best_bid=bid,best_ask=ask,
            bid_depth=[] if bid is None else [DepthLevel(price=bid,quantity=10)],
            ask_depth=[] if ask is None else [DepthLevel(price=ask,quantity=20)],depth_levels=measurement.top_k,
            current_quote_source='book_end_market',quote_as_of_step=step)
        context=MeasurementContext(spec=measurement,world_id='semantic-measurement-fixture',run_id='semantic-gate',
            step=step,session_id='unit',policy_hash=policy.policy_hash,market_open=market_open,
            observation_complete=True,execution_status='completed',origin='analytic_fixture')
        activity=StepActivity(settled_trade_ids=[],cancel_events=[],reject_events=[],ioc_remainder_count=0,halt_suppressed_count=0)
        return compute_step_measurement(snapshot=snapshot,trade_tape=[],activity=activity,context=context)
    for name,bid,ask,expected in [('empty',None,None,{'spread':None,'vwap':None,'traded_quantity':0.0}),
        ('bid_only',99.0,None,{'spread':None,'depth_bid_top_k':10.0,'depth_ask_top_k':0.0}),
        ('quoted',99.0,101.0,{'spread':2.0,'mid_price':100.0,'depth_imbalance':-1/3}),
        ('crossed',101.0,100.0,{'spread':None})]:
        result=compute(1,bid,ask)
        actual={key:result.metrics[key].value for key in expected}
        passed=actual==expected and (name!='crossed' or not result.snapshot_valid)
        cases.append({'case':name,'expected':expected,'actual':actual,'passed':passed});all_pass &= passed
    sequence=[compute(1,99.0,101.0),compute(2,None,None),compute(3,109.0,111.0)]
    aggregate=aggregate_run_metrics(sequence,spec=measurement,run_ref=research_hash({'fixture':'gapped-series'}))
    # No price return bridges the missing clock step.
    valid_returns=[r for r in aggregate.returns if r.metric.value is not None]
    gap_pass=not valid_returns;all_pass &= gap_pass
    cases.append({'case':'no_gap_bridging','returns':aggregate.model_dump(mode='json')['returns'],'passed':gap_pass})
    halt=compute(1,None,None,False)
    halt_pass=halt.metrics['spread'].value is None and halt.metrics['traded_quantity'].value==0.0
    all_pass &= halt_pass;cases.append({'case':'halt_unavailable','passed':halt_pass,'measurement':halt.model_dump(mode='json')})
    return {'status':'PASS' if all_pass else 'FAIL','origin':'analytic_fixture','cases':cases}


def clearing_witness():
    from core.exchange.clearing import Account,ClearingHouse
    from core.exchange.order_book import OrderBook
    from core.exchange.market_rules import MarketRuleSchema
    from core.types import Order,OrderSide,OrderType
    rules=MarketRuleSchema('GATE',100,commission_rate=0.001,stamp_duty_rate=0,
        feature_flags={'market_rules_v1':True,'strict_queue_timestamps':True})
    clearing=ClearingHouse(OrderBook('GATE',100,market_rules=rules),{'buyer':Account(2000,3),'seller':Account(0,10)},t_plus_one=False)
    def order(name,side,price,qty,owner):return Order('GATE',price,qty,OrderSide(side),OrderType.LIMIT,owner,1.0,order_id=name)
    clearing.submit(order('maker','sell',100.0,2,'seller'))
    trades=clearing.submit(order('taker','buy',101.0,1,'buyer'))
    invariant=clearing.check_invariants();free_before=clearing.available('seller');clearing.cancel('maker');free_after=clearing.available('seller')
    ordinary_ok=len(trades)==1 and clearing.accounts['buyer'].shares==4 and free_after[1]==free_before[1]+1
    clearing.submit(order('fault-maker','sell',100.0,1,'seller'))
    actual=clearing.book.add_order
    def interrupted(order):
        actual(order)
        raise RuntimeError('semantic matching interruption')
    clearing.book.add_order=interrupted
    try:clearing.submit(order('fault-taker','buy',101.0,1,'buyer'))
    except RuntimeError:pass
    snapshot=clearing.diagnostic_snapshot();rejected=0
    for action in (clearing.ensure_usable,lambda:clearing.cancel('fault-maker'),clearing.check_invariants):
        try:action()
        except RuntimeError:rejected+=1
    failed_ok=snapshot['failed'] and rejected==3 and clearing.diagnostic_snapshot()==snapshot
    return {'status':'PASS' if ordinary_ok and failed_ok else 'FAIL','origin':'synthetic_fixture',
        'ordinary_trade_count':len(trades),'reserved_share_release':free_after[1]-free_before[1],
        'fail_stop_rejections':rejected,'failure_snapshot':snapshot}


def observation_witness(archive):
    from core.experiments.world import InitialWorldSnapshot,restore_initial_world
    from core.society.network import capture_social_state
    from core.experiments.randomness import KeyedRandomSource
    archive=Path(archive);spec=ResearchRunSpec.model_validate_json((archive/'research_spec.json').read_text())
    source=strict_json_loads((archive/'archive_manifest.json').read_text())['source_hash']
    initial=InitialWorldSnapshot.model_validate_json((archive/'initial_world.json').read_text())
    world=restore_initial_world(initial,spec=spec,source_hash=source)
    before=capture_social_state(world.diffusion)
    for _ in range(3):
        observed=world.diffusion.research_signals()
        if observed:observed[next(iter(observed))]['signal']=0.77
    unchanged=before==capture_social_state(world.diffusion)
    rng=KeyedRandomSource(spec.random);ids=list(spec.world.stable_actor_ids)
    left={aid:rng.uniform(step=1,entity_id=aid,channel='participation') for aid in ids}
    right={aid:rng.uniform(step=1,entity_id=aid,channel='participation') for aid in reversed(ids)}
    return {'status':'PASS' if unchanged and left==right else 'FAIL','initial_world_hash':initial.world_hash,
        'observation_state_hash':research_hash(before),'detached_observation':unchanged,'query_order_invariant':left==right}


def behavior_witness(archive):
    from agents.decision_context import DecisionContext,DecisionNoise,BehaviorState
    from agents.research_kernel import ResearchDecisionKernel
    from core.experiments.randomness import KeyedRandomSource
    archive=Path(archive);spec=ResearchRunSpec.model_validate_json((archive/'research_spec.json').read_text())
    rng=KeyedRandomSource(spec.random);states={(r['step'],r['actor_id']):r['behavior_state']
        for r in rows(archive/'agents.jsonl') if r['behavior_state'] is not None}
    decisions={(r['step'],record['actor_id']):record for r in rows(archive/'decisions.jsonl') for record in r['records']}
    checked=0;selected=None
    for row in rows(archive/'observations.jsonl'):
        ctx=DecisionContext.model_validate(row['context']);aid=row['actor_id'];step=row['step']
        noise=DecisionNoise(actor_id=aid,step=step,
            participation_draw=rng.draw_receipt(step=step,entity_id=aid,channel='participation',distribution='uniform'),
            forecast_draw=rng.draw_receipt(step=step,entity_id=aid,channel='forecast_noise',distribution='normal'))
        state=BehaviorState.model_validate(states[(step-1,aid)])
        advanced,receipt=ResearchDecisionKernel.advance(state,ctx,noise)
        batch=ResearchDecisionKernel.decide(advanced,ctx,noise)
        if batch.records[0].model_dump(mode='json')!=decisions[(step,aid)]:raise ValueError('BEHAVIOR_RECOMPUTATION_MISMATCH')
        if advanced.model_dump(mode='json')!=states[(step,aid)]:raise ValueError('BEHAVIOR_STATE_ADVANCE_MISMATCH')
        if ResearchDecisionKernel.decide(advanced,ctx,noise)!=batch:raise ValueError('IMPURE_DECISION')
        checked+=1
        if selected is None and ctx.profile.actor_kind=='behavioral_social' and step==1:selected=(ctx,noise,state)
    if selected is None:return {'status':'INCONCLUSIVE','recomputed_decisions':checked,'reason':'No applicable loss-aversion channel context'}
    ctx,noise,state=selected
    # Analytic finite-choice probe: expected BUY at lambda=2, HOLD at lambda=4.
    # Both observations and hypothetical parameters are archived in this witness.
    value=ctx.model_dump(mode='json');value['profile'].update(lambda_base=2.0,social_lambda_gain=0.0,beta_s=0.0,
        forecast_sigma=0.15,mu_max=0.2,aggressiveness=0.0,trade_fraction=0.5)
    value['observation']['account'].update(cash=200.0,shares=0,available_cash=200.0,available_shares=0,
        pending_buy_qty=0,pending_sell_qty=0,acquired_today=0,cost_basis=0.0,t_plus_one=False)
    value['observation'].update(initial_wealth=200.0)
    value['observation']['private_valuation']['value']=105.0
    value['rules'].update(commission_rate=0.0,stamp_duty_rate=0.0)
    actions=[];probes=[]
    for lam in (2.0,4.0):
        value['profile']['lambda_base']=lam;context=DecisionContext.model_validate(value)
        advanced,_=ResearchDecisionKernel.advance(state,context,noise)
        result=ResearchDecisionKernel.decide(advanced,context,noise);actions.append(result.records[0].action)
        probes.append({'context':context.model_dump(mode='json'),'noise':noise.model_dump(mode='json'),
            'before':state.model_dump(mode='json'),'result':result.model_dump(mode='json')})
    return {'status':'PASS' if actions==['BUY','HOLD'] else 'INCONCLUSIVE','recomputed_decisions':checked,
        'loss_branch_probe':probes,'expected_actions':['BUY','HOLD'],'observed_actions':actions}


def semantic_gate_witness(gate_id,report,root):
    root=Path(root);archives=[Path(r['archive']) for r in report['runs']]
    if not archives:return {'status':'INCONCLUSIVE','reason':'No complete independent validation archives'}
    for path in archives:verify_archive_v3(path)
    if gate_id=='S0':
        spec=ResearchRunSpec.model_validate_json((archives[0]/'research_spec.json').read_text());return measurement_witness(spec)
    if gate_id=='S1':return clearing_witness()
    if gate_id=='S3':return behavior_witness(archives[0])
    if gate_id=='S2':
        from core.experiments.counterfactual import validate_pair
        witness=observation_witness(archives[0]);pair_file=root/'noop-pair/batch.json'
        if not pair_file.exists():return dict(witness,status='INCONCLUSIVE',reason='Scoped independent NoOp pairs absent')
        pairs=strict_json_loads(pair_file.read_text())['pairs']
        for pair in pairs:
            checked=validate_pair(pair['left'],pair['right'])
            source=strict_json_loads((Path(pair['left'])/'archive_manifest.json').read_text())['source_hash']
            if not checked['noop_identity'] or source!=report['source_hash']:return dict(witness,status='INCONCLUSIVE',reason='Pair source/model compatibility absent')
        return dict(witness,pairs=pairs,status=witness['status'] if pairs else 'INCONCLUSIVE')
    if gate_id=='S6':
        from core.experiments.evidence import ConfirmationPlan,certify_confirmation
        folder=root/'confirmation';required=[folder/'registration/registered_plan.json',folder/'registration/registration.json',
            folder/'jobs/VISIBLE.json',folder/'budget.json',root/'recovery/witness.json']
        if not all(p.exists() for p in required):return {'status':'INCONCLUSIVE','reason':'Scoped confirmation/recovery witnesses incomplete'}
        plan=ConfirmationPlan.model_validate_json(required[0].read_text());registration=strict_json_loads(required[1].read_text())
        batch=strict_json_loads(required[2].read_text());envelope=strict_json_loads(required[3].read_text())
        if research_hash(envelope['state'])!=envelope['state_hash']:raise ValueError('Budget integrity mismatch')
        class ReadOnlyPort:
            def snapshot(self):return envelope['state']
        records=certify_confirmation(plan,registration,batch,reservation_port=ReadOnlyPort())
        recovery=strict_json_loads(required[4].read_text())
        if plan.scope.source_hash!=report['source_hash'] or recovery['source_hash']!=report['source_hash']:
            return {'status':'INCONCLUSIVE','reason':'Confirmation/recovery source scope mismatch'}
        failed=Path(recovery['interrupted_attempt']);rerun=Path(recovery['recovered_archive']);verify_archive_v3(rerun)
        marker=failed/'COMMITTED.json'
        if marker.exists() or recovery['exit_code']!=-9 or not list(failed.rglob('step_commits.jsonl')):
            raise ValueError('Recovery interruption witness invalid')
        if not list(failed.rglob('research_spec.json')) or strict_json_loads(next(failed.rglob('research_spec.json')).read_text())!=strict_json_loads((rerun/'research_spec.json').read_text()):
            raise ValueError('Recovery changed full input')
        return {'status':'PASS','confirmation_evidence_ids':[r.evidence_id for r in records],'recovery':recovery}
    raise ValueError('Unsupported semantic gate')
