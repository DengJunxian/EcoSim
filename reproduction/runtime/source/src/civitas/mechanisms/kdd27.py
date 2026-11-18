"""Paper equations; pure synchronous updates, independent of execution and LLMs."""
import math
from fractions import Fraction

from core.behavioral_finance import ProspectTheoryParams, prospect_value


def clip(x, low=0.0, high=1.0):
    return max(low,min(high,x))


def normal(rng, process, actor, tick, slot):
    u=rng.uniform(process,actor,tick,slot+':radius')
    v=rng.uniform(process,actor,tick,slot+':angle')
    return math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)


def social_update(config, states, profiles, edges, exposure, accounts, price, previous_return, tick, rng, direct=()):
    net,soc=config.network,config.social
    fractions={a:price*q/(c+price*q+net.exposure_epsilon) for a,(c,q) in accounts.items()}
    weights={a:{b:clip((1-net.weight_smoothing)*w+net.weight_smoothing*(2-abs(fractions[a]-fractions[b])),
        net.weight_min,net.weight_max) for b,w in neighbors.items()} for a,neighbors in edges.items()}
    updated,new_exposure,trace={},{},{}
    for actor,old in states.items():
        neighbors=weights[actor]
        total=sum(neighbors.values())
        avg=sum(w*states[b]['emotion'] for b,w in neighbors.items())/total if total else old['emotion']
        rho=sum(w for b,w in neighbors.items() if states[b]['infection']=='I')/total if total else 0.0
        pessimism=sum(w for b,w in neighbors.items() if states[b]['emotion'] < -soc.pessimism_threshold)/total if total else 0.0
        ranked=[]
        for source in neighbors:
            if states[source]['infection']!='I':
                continue
            similarity=1-abs(old['emotion']-states[source]['emotion'])/2
            terms=[similarity,float(profiles[actor]['topic']==profiles[source]['topic']),
                1-abs(old['risk']-states[source]['risk']),profiles[source]['influence'],exposure[actor][source]]
            heat=sum(w*x for w,x in zip(net.ranking_weights,terms,strict=True))
            ranked.append({'source':source,'score':heat,'similarity':similarity,
                'message_id':f'paper-message:{tick}:{source}','signal':-1.0})
        feed=sorted(ranked,key=lambda x:(-x['score'],x['source']))[:net.attention_capacity]
        selected={row['source'] for row in feed}
        new_exposure[actor]={b:(1-net.exposure_ema)*exposure[actor][b]+net.exposure_ema*int(b in selected) for b in neighbors}
        exposure_total=sum(new_exposure[actor].values())
        echo=sum((v/exposure_total)**2 for v in new_exposure[actor].values()) if exposure_total else 0.0
        mean=sum(row['score']*row['similarity'] for row in feed)/len(feed) if feed else 0.0
        k0,k1,k2=soc.pressure_coefficients
        pressure=clip(mean*(k0+k1*rho)+k2*rho)
        previous=old['infection']
        probability=clip(soc.infection_beta*pressure*profiles[actor]['herding']) if previous=='S' else (
            clip(soc.recovery_gamma*(1+soc.duration_recovery*old['duration'])*(1-soc.echo_recovery*echo)) if previous=='I' else
            clip(soc.resusceptibility_delta*(1+soc.pressure_resusceptibility*pressure)))
        draw=rng.uniform('kdd27.infection',actor,tick,'transition')
        current=({'S':'I','I':'R','R':'S'}[previous] if draw<probability else previous)
        duration=(old['duration']+1 if previous=='I' and actor not in direct else 1) if current=='I' else 0
        herd=bool(neighbors) and pessimism>soc.pessimistic_neighbor_threshold
        wd,wg,wn,wm=soc.emotion_weights[profiles[actor]['cognitive_type']]
        transfer=float(herd)*soc.herding_weight_transfer*(1-old['confidence'])*wd
        effective=[wd-transfer,wg,wn+transfer,wm]
        noise=soc.emotion_noise_sd*normal(rng,'kdd27.emotion',actor,tick,'noise')
        emotion=clip(effective[0]*old['emotion']-wg*int(current=='I')+effective[2]*avg+
            wm*clip(previous_return/soc.return_scale,-1.0,1.0)+noise,-1.0,1.0)
        confidence=clip(old['confidence']-float(herd)*soc.confidence_drop*(1-old['confidence'])+
            float(not herd)*soc.confidence_recovery*(soc.confidence_target-old['confidence']))
        updated[actor]={**old,'infection':current,'duration':duration,'emotion':emotion,'confidence':confidence}
        trace[actor]={'pre_transition':previous,'direct_injection':actor in direct,'infection':current,'duration':duration,
            'transition_probability':probability,'infection_draw':draw,'rho':rho,'pressure':pressure,'echo':echo,
            'feed':feed,'neighbor_emotion':avg,'pessimism':pessimism,'herding':herd,'emotion_weights':effective,
            'emotion_noise':noise,'emotion':emotion,'confidence':confidence}
    return updated,weights,new_exposure,trace


def sigmoid(value):
    return 1/(1+math.exp(-value)) if value>=0 else math.exp(value)/(1+math.exp(value))


def behavior_update(cfg, old, profile, price, quantity, high, neighbor_emotion, policy, *, baseline='full'):
    if baseline not in {'full','zero_intelligence','homogeneous_linear_value'}:
        raise ValueError('Unknown RQ1 baseline')
    if baseline=='homogeneous_linear_value':
        p,h=cfg.reference_weight[:2]
        cfg=cfg.model_copy(update={'prospect_alpha':1.0,'loss_pressure_multiplier':0.0,
            'risk_weights':[*cfg.risk_weights[:4],0.0],'reference_weight':[p/(p+h),h/(p+h),0.0,0.0]})
        profile={**profile,'loss_aversion':1.0}
        old={**old,'emotion':0.0,'risk':0.5}
    targets=[old['cost'] if quantity else price,high,price*(1+cfg.peer_price_amplitude*neighbor_emotion),
        price*(1+cfg.policy_price_amplitude*policy)]
    references=[(1-k)*r+k*target for k,r,target in zip(cfg.reference_update,old['references'],targets,strict=True)]
    deviations=[(price-r)/r if i<2 else (r-price)/price for i,r in enumerate(references)]
    if cfg.reverse_retrospective:
        deviations[:2]=[-v for v in deviations[:2]]
    z=sum(w*x for w,x in zip(cfg.reference_weight,deviations,strict=True))
    value=float(prospect_value(z,params=ProspectTheoryParams(alpha=cfg.prospect_alpha,beta=cfg.prospect_alpha,lambda_=profile['loss_aversion'])))
    direction=math.tanh(cfg.direction_scale*value)
    loss_fraction=sum(cfg.reference_weight[i]*int(price<references[i]) for i in (0,1))/sum(cfg.reference_weight[:2])
    pressure=clip(profile['loss_aversion']*(1+cfg.loss_pressure_multiplier*loss_fraction),cfg.loss_pressure_min,cfg.loss_pressure_max)
    terms=[old['confidence'],old['risk'],old['emotion'],direction,-max(0.0,(pressure-1)/cfg.loss_pressure_scale)]
    risk=0.5 if baseline!='full' else clip(sum(w*x for w,x in zip(cfg.risk_weights,terms,strict=True)))
    intent=0.0 if baseline=='zero_intelligence' else clip(sum(w*x for w,x in zip(cfg.intention_weights,[direction,old['emotion'],risk-0.5],strict=True)),-1.0,1.0)
    buy=sigmoid(cfg.action_logistic_scale*(intent-cfg.hold_threshold))
    sell=sigmoid(-cfg.action_logistic_scale*(intent+cfg.hold_threshold))
    trace={'targets':targets,'references':references,'deviations':deviations,
        'z':z,'value':value,'direction':direction,'loss_fraction':loss_fraction,'loss_pressure':pressure,
        'risk':risk,'intention':intent,'action_probabilities':[buy,sell,max(0.0,1-buy-sell)]}
    if baseline!='full':
        trace['baseline_override']={'baseline':baseline,'alpha':cfg.prospect_alpha,'lambda':profile['loss_aversion'],
            'reference_weight':cfg.reference_weight,'loss_pressure_multiplier':cfg.loss_pressure_multiplier,
            'risk_loss_weight':cfg.risk_weights[-1],'fixed_risk':0.5}
    return {**old,'references':references,'risk':risk},trace


def quote_ticks(cfg, price_ticks, rng, actor, tick):
    noise=cfg.quote_log_noise_sd*normal(rng,'kdd27.decision',actor,tick,'quote')
    return max(1,round(price_ticks*math.exp(noise-cfg.quote_log_noise_sd**2/2)))


def rq2_parameters(config, condition):
    """Effective views only: preserve the checkpoint's latent profiles and state."""
    social={};behavior={};population={}
    if condition in ('no_secondary_contagion','both_off'):social['infection_beta']=0.0
    if condition in ('no_herding_feedback','both_off'):social.update(herding_weight_transfer=0.0,confidence_drop=0.0)
    if condition in ('full_loss_off','lambda_only_1'):population['loss_aversion']=1.0
    if condition=='full_loss_off':
        behavior.update(loss_pressure_multiplier=0.0,risk_weights=[*config.behavior.risk_weights[:4],0.0])
    if condition=='reverse_retrospective_mapping':behavior['reverse_retrospective']=True
    changes={f'{group}.{key}':{'before':getattr(getattr(config,group),key),'after':value}
        for group,values in [('social',social),('behavior',behavior),('population',population)] for key,value in values.items()}
    effective=config.model_copy(update={group:getattr(config,group).model_copy(update=values)
        for group,values in [('social',social),('behavior',behavior),('population',population)] if values})
    return effective,changes


def max_quantity(side, account, price_ticks, market, commission):
    if side=='hold':
        return 0
    if side=='sell':
        return account['available'][market.asset_id]
    rate=Fraction(str(commission))
    return account['available'][market.currency]*rate.denominator//(price_ticks*market.tick_minor*(rate.denominator+rate.numerator))


def rule_quantity(side, fraction, account, price_ticks, market, commission):
    ratio=Fraction(str(fraction))
    if side=='hold':
        return 0
    if side=='sell':
        return ratio.numerator*account['available'][market.asset_id]//ratio.denominator
    rate=Fraction(str(commission))
    return ratio.numerator*account['available'][market.currency]*rate.denominator//(
        ratio.denominator*price_ticks*market.tick_minor*(rate.denominator+rate.numerator))
