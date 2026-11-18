"""Independent paper endowments and fixed profiles, using semantic RNG addresses."""
from decimal import Decimal, ROUND_HALF_EVEN

import networkx as nx
import numpy as np

from civitas.contracts.portfolio import AccountState
from civitas.population.allocation import allocate


def generator(rng, actor, purpose):
    return np.random.Generator(np.random.PCG64(rng.u64('kdd27.population',actor,0,purpose)))


def categorical(rng, actor, purpose, probabilities):
    labels=sorted(probabilities)
    return str(generator(rng,actor,purpose).choice(labels,p=[probabilities[label] for label in labels]))


def truncated_normal(draw, mean, sd, low, high):
    for _ in range(10000):
        value=float(draw.normal(mean,sd))
        if low<value<high:
            return value
    raise ValueError('Truncated initial distribution could not generate an admissible value')


def build_population(config, rng):
    spec, market=config.population,config.market
    actors=[f'agent_{i:04d}' for i in range(config.n_agents)]
    profiles, states, accounts={},{},[]
    for actor in actors:
        cash=float(generator(rng,actor,'cash').lognormal(spec.cash_log_mean,spec.cash_log_sd))
        cash_minor=int((Decimal(str(cash))*market.cash_scale).quantize(Decimal(1),rounding=ROUND_HALF_EVEN))
        shares=int(generator(rng,actor,'inventory').integers(0,spec.inventory_max+1))
        cost=truncated_normal(generator(rng,actor,'cost'),spec.cost_normal_mean,spec.cost_normal_sd,0.0,float('inf')) if shares else 0.0
        confidence=(spec.confidence_mean if spec.confidence_sd==0 else
            truncated_normal(generator(rng,actor,'confidence'),spec.confidence_mean,spec.confidence_sd,0.0,1.0))
        profiles[actor]={'archetype':categorical(rng,actor,'archetype',spec.archetype_proportions),
            'cognitive_type':categorical(rng,actor,'cognitive_type',spec.cognitive_type_proportions),
            'loss_aversion':spec.loss_aversion,
            'herding':float(generator(rng,actor,'herding').beta(*spec.herding_beta)),
            'influence':float(generator(rng,actor,'influence').uniform()),
            'topic':categorical(rng,actor,'topic',dict(zip(spec.topics,spec.topic_probabilities,strict=True)))}
        states[actor]={'emotion':2*float(generator(rng,actor,'emotion').beta(*spec.emotion_beta))-1,
            'confidence':confidence,'risk':spec.initial_risk,'infection':'S','duration':0,
            'references':[market.initial_price]*4,'cost':cost}
        accounts.append(AccountState(account_id=actor,balances={market.currency:cash_minor,market.asset_id:shares},
            cost_minor={market.asset_id:int((Decimal(str(cost))*shares*market.cash_scale).quantize(Decimal(1),rounding=ROUND_HALF_EVEN))}))
    infected=sorted(actors,key=lambda a:(rng.u64('kdd27.infection',a,0,'initial'),a))[:int(config.n_agents*spec.initial_infection_fraction)]
    for actor in infected:
        states[actor].update(infection='I',duration=1)
    network=config.network
    graph=(nx.empty_graph(len(actors)) if network.topology=='empty' else nx.watts_strogatz_graph(
        len(actors),network.neighbors,network.rewiring if network.topology=='watts_strogatz' else 0.0,
        seed=rng.u64('kdd27.network','world',0,'topology')))
    edges={a:{actors[j]:network.weight_initial for j in sorted(graph.neighbors(i))} for i,a in enumerate(actors)}
    exposure={a:{b:0.0 for b in edges[a]} for a in actors}
    strata={kind:[a for a in actors if profiles[a]['archetype']==kind] for kind in sorted({p['archetype'] for p in profiles.values()})}
    counts=allocate(config.slow_agents,{k:len(v) for k,v in strata.items()},'strategic')
    strategic=sorted(a for entry in counts for a in sorted(strata[entry.account_id],key=lambda a:(
        rng.u64('kdd27.population',a,0,'strategic'),a))[:entry.allocated_units])
    return accounts,profiles,states,edges,exposure,strategic,infected
