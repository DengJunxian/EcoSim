"""Explicit constructed observations; an archive template does not make these real snapshots."""
from copy import deepcopy
from itertools import product

from civitas.contracts.base import content_hash
from civitas.contracts.observations import Observation
from civitas.contracts.portfolio import PortfolioAccountView
from .kdd27_persona import PERSONAS, persona_contract

VERSION = 'kdd27-frozen-observations-v1'
FRAMES = {'neutral':'Consider the recorded facts.',
    'pessimistic':'Consider the downside of the recorded facts.',
    'optimistic':'Consider the upside of the recorded facts.'}


def observation_from_payload(payload):
    digest=content_hash(payload)
    return Observation(**payload,observation_id='observation:'+digest,payload_hash=digest)


def frozen_cases(template, config, *, split='engineering'):
    if split not in ('engineering','holdout'):
        raise ValueError('Unknown frozen observation split')
    if config.cognition_contract!='kdd27-cognition-v1':
        raise ValueError('Frozen cases require the declared cognition contract')
    base=template.payload(); base['delivered_messages']=[]
    base['private_memory_refs']=[f for f in base['private_memory_refs'] if f['fact_id'].startswith(('paper-own:','paper-cache:'))]
    pairs={'price':(9500,10500),'locked_inventory':(0,500),'cash':(50000,0)} if split=='engineering' else {
        'price':(9000,11000),'locked_inventory':(250,500),'cash':(1000,0)}
    for archetype,pair,state,frame in product(sorted(PERSONAS),pairs,range(2),FRAMES):
        body=deepcopy(base);account=body['account_view'][0]['value']
        account.update(balances={config.market.currency:50000*config.market.cash_scale,config.market.asset_id:500},
            available={config.market.currency:50000*config.market.cash_scale,config.market.asset_id:500},
            locked={config.market.asset_id:0},reserved={config.market.currency:0,config.market.asset_id:0},
            cost_minor={config.market.asset_id:50000*config.market.cash_scale})
        market=next(f['value'] for f in body['market_view'] if f['fact_id'].startswith('paper-market:'))
        market.update(previous_return=0.0,halted=False,day_anchor_ticks=10000,band=.1,policy_signal=0.0,
            commission=.00025,seller_tax=.0005,same_day_lock=True,
            events=[{'event_id':'benchmark-framing','kind':'text','value':FRAMES[frame]}])
        market['historical_quote'].update(last_price_ticks=10000,best_bid_ticks=None,best_ask_ticks=None,
            spread_ticks=None,depth_imbalance=None,bid_depth=0,ask_depth=0,two_sided=False,halted=False)
        own=next(f['value'] for f in body['private_memory_refs'] if f['fact_id'].startswith('paper-own:'))
        own['profile']['archetype']=archetype;own.update(active=True,strategic=True,feed=[],persona_contract=persona_contract(archetype))
        own['state'].update(emotion=0.0,confidence=.6,risk=.5,references=[100.0]*4,cost=100.0,infection='S',duration=0)
        own['behavior'].update(action_probabilities=[.5,.5,0.0],intention=-.5)
        next(f['value'] for f in body['private_memory_refs'] if f['fact_id'].startswith('paper-cache:'))['strategy']=None
        value=pairs[pair][state]
        if pair=='price':market['historical_quote']['last_price_ticks']=value
        elif pair=='locked_inventory':account['locked'][config.market.asset_id]=value;account['available'][config.market.asset_id]=500-value
        else:account['balances'][config.market.currency]=value*config.market.cash_scale;account['available'][config.market.currency]=value*config.market.cash_scale
        PortfolioAccountView.model_validate(account)
        observation=observation_from_payload(body)
        numeric=deepcopy(body);next(f['value'] for f in numeric['market_view'] if f['fact_id'].startswith('paper-market:'))['events'][0]['value']='FRAME_REMOVED'
        yield {'case_id':f'{VERSION}:{split}:{archetype}:{pair}:{state}:{frame}','split':split,
            'origin_kind':'mock_fixture','template_observation_hash':template.payload_hash,
            'archetype':archetype,'fact_pair':pair,'state':state,'frame':frame,
            'block_id':f'{archetype}:{pair}','fact_hash':content_hash(numeric),
            'observation':observation.model_dump(mode='json'),'formal_eligible':False}
