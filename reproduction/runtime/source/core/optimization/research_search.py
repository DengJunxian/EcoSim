"""Budgeted finite-policy search. NSGA-II is the pinned official pymoo algorithm."""
from __future__ import annotations

import importlib.metadata
import math
import random


class EnumeratedReference:
    def run(self,port,*,seed=0):
        for candidate_id in port.candidate_ids:
            if not port.can_submit(1):break
            port.submit(candidate_id)
        return {'algorithm':'enumeration','complete_pool':set(port.evaluated_ids)==set(port.candidate_ids),
                'optimality_scope':'finite_analytic_pool' if port.origin=='analytic_fixture' else 'finite_pool_at_recorded_precision'}


class RandomSearch:
    def run(self,port,*,seed=0):
        rng=random.Random(seed)
        while port.can_submit(1):port.submit(rng.choice(port.candidate_ids))
        return {'algorithm':'random_with_replacement','search_seed':seed,'duplicates_charged':True,
                'random_source':'private_python_random_MT19937_not_market_address_space'}


class NSGA2Adapter:
    def __init__(self,*,population_size=8,generations=3):
        if type(population_size) is not int or population_size<4 or type(generations) is not int or generations<1:
            raise ValueError('Explicit NSGA-II population >=4 and generations >=1 required')
        self.population_size,self.generations=population_size,generations

    def run(self,port,*,seed=0):
        if importlib.metadata.version('pymoo')!='0.6.2':raise ValueError('NSGA2_REQUIRES_PINNED_PYMOO_0_6_2')
        import numpy as np
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.repair.rounding import RoundingRepair
        if len(port.candidate_ids)<2:raise ValueError('NSGA2_REQUIRES_AT_LEAST_TWO_REGISTERED_CODES')
        problem=Problem(n_var=1,n_obj=len(port.objective_ids),n_ieq_constr=1,
            xl=0,xu=len(port.candidate_ids)-1,vtype=int)
        algorithm=NSGA2(pop_size=self.population_size,sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9,eta=15,vtype=float,repair=RoundingRepair()),
            mutation=PM(prob=0.9,eta=20,vtype=float,repair=RoundingRepair()),eliminate_duplicates=False)
        algorithm.setup(problem,termination=('n_gen',self.generations),seed=seed,verbose=False)
        history=[]
        while algorithm.has_next():
            if not port.can_submit(self.population_size):break
            infills=algorithm.ask()
            if len(infills)>self.population_size:raise RuntimeError('UNREGISTERED_POPULATION_EXPANSION')
            scores=[];violations=[];evaluations=[]
            for individual in infills:
                index=int(individual.X[0])
                if index<0 or index>=len(port.candidate_ids):raise RuntimeError('DISCRETE_ENCODING_OUT_OF_RANGE')
                evaluation=port.submit(port.candidate_ids[index]);evaluations.append(evaluation)
                if evaluation['utilities'] is None:
                    # A constrained optimizer placeholder is never exported as a utility.
                    scores.append([1.0]*len(port.objective_ids));violations.append([1.0])
                else:
                    scores.append([-evaluation['utilities'][r] for r in port.objective_ids])
                    violations.append([evaluation['constraint_violation']])
            infills.set('F',np.asarray(scores));infills.set('G',np.asarray(violations))
            algorithm.tell(infills=infills)
            population=[]
            for member in algorithm.pop:
                crowd=member.get('crowding');rank=member.get('rank')
                population.append({'candidate_id':port.candidate_ids[int(member.X[0])],
                    'objectives':member.F.tolist(),'constraint_violation':float(member.CV[0]),
                    'rank':None if rank is None else int(rank),
                    'crowding_distance':None if crowd is None or not math.isfinite(float(crowd)) else float(crowd),
                    'crowding_status':'not_ranked' if crowd is None else 'boundary_infinite' if math.isinf(float(crowd)) else 'finite'})
            history.append({'generation':len(history)+1,'submissions':[e['submission_id'] for e in evaluations],
                            'population_after_official_survival':population})
        front=[] if algorithm.opt is None else sorted({port.candidate_ids[int(x[0])] for x in algorithm.opt.get('X')})
        return {'algorithm':'pymoo.algorithms.moo.nsga2.NSGA2','pymoo_version':importlib.metadata.version('pymoo'),
            'population_size':self.population_size,'planned_generations':self.generations,'completed_generations':len(history),
            'search_seed':seed,'encoding':'one_integer_registered_pool_index','duplicate_elimination':False,
            'selection':'official_binary_tournament_rank_and_crowding','crossover':{'name':'SBX','prob':0.9,'eta':15,'repair':'RoundingRepair'},
            'mutation':{'name':'PM','prob':0.9,'eta':20,'repair':'RoundingRepair'},
            'termination':'planned_generations_or_before_unfunded_generation','history':history,'optimizer_front':front}
