"""Noncollaborative controls share ports; none receives hidden task preferences."""
from __future__ import annotations
from core.optimization.research_search import EnumeratedReference, RandomSearch, NSGA2Adapter


class PolicySearchMethod:
    def __init__(self,name,*,seed=0,population_size=8,generations=3):
        self.name,self.seed=name,seed
        self.algorithm={'enumerated':EnumeratedReference,'random':RandomSearch,
            'nsga2':lambda:NSGA2Adapter(population_size=population_size,generations=generations)}.get(name)
        if name not in {'noop','fixed_rule','enumerated','random','nsga2'}:raise ValueError('Unknown search baseline')

    def run(self,port):
        port.acquire_shared_information()
        if self.name in {'noop','fixed_rule'}:
            candidate_id=port.noop_candidate_id if self.name=='noop' else port.fixed_rule_candidate_id
            port.submit(candidate_id)
            return {'algorithm':self.name,'source':'rule'}
        return self.algorithm().run(port,seed=self.seed)


class CentralRuleMethod:
    name='central_rule'
    def __init__(self,*,protocol_enabled=False):self.protocol_enabled=protocol_enabled
    def run(self,port):
        if self.protocol_enabled:return _run_rule_protocol(port,'central_rule')
        port.acquire_shared_information()
        receipt=EnumeratedReference().run(port)
        return {**receipt,'algorithm':'central_rule','decision_rule':'exhaust_finite_pool_then_common_max_min',
                'information_access':'same_designer_initial_and_charged_query_ports','oracle':False}


class SingleAgentMethod:
    def __init__(self,client=None,*,mode,client_factory=None):
        if mode not in {'proposal_only','simulation_guided'}:raise ValueError('Explicit single-agent mode required')
        self.client,self.mode,self.client_factory=client,mode,client_factory
        self.name='single_'+mode

    def bind(self,ledger,output):
        if self.client is None:
            if self.client_factory is None:raise ValueError('Explicit client configuration required')
            self.client=self.client_factory(ledger,output)
        if self.client.ledger.path.resolve()!=ledger.path.resolve():raise ValueError('SINGLE_AGENT_MUST_USE_EPISODE_LEDGER')

    async def run_async(self,port):
        messages=[{'role':'system','content':port.prompt(self.mode)}]
        round_index=0
        while port.can_round():
            port.reserve_round();round_index+=1
            result=await self.client.request(messages,request_id=f'round-{round_index}')
            raw_ref=self.client.calls[-1]['response_ref']
            port.observe_raw_claim(result,raw_ref)
            if result.action=='stop':break
            if result.action=='query_information':
                card=port.query_information(result.information_id)
                feedback={'information':card.model_dump(mode='json')}
            else:
                feedback=port.submit(result.candidate_id,raw_ref=raw_ref,claimed_agreement=result.claimed_agreement)
                if self.mode=='proposal_only':break
            messages.append({'role':'assistant','content':__import__('json').dumps(result.model_dump(mode='json'))})
            messages.append({'role':'user','content':__import__('json').dumps(feedback,ensure_ascii=False)})
        return {'algorithm':self.name,'mode':self.mode,'source':self.client.config.source,
                'calls':len(self.client.calls),'feedback':'none_after_proposal' if self.mode=='proposal_only' else 'common_service'}


def _run_rule_protocol(port,kind):
    """Executable rule controls for protocol structure, with complete common tools."""
    from .budget import BudgetExceeded
    from core.experiments.archive import durable_json
    session=port.protocol_session();roles=port.objective_ids
    try:
        port.acquire_shared_information()
        assignments=[(cid,roles[i%len(roles)] if kind!='central_rule' else 'designer') for i,cid in enumerate(port.candidate_ids)]
        if kind=='independent':
            # All drafts are produced from their owner's initial cards before any
            # peer proposal is released. Fixed-pool coverage is shared by design.
            drafts=[{'candidate_id':cid,'principal':owner,'visible_cards':port._information.view(owner),
                     'visible_peer_proposals':[],'origin':'rule','stage':'independent_before_publication'} for cid,owner in assignments]
            durable_json(session.root/'independent-drafts.json',drafts)
            for draft in drafts:session.propose(draft['candidate_id'],principal=draft['principal'])
            for cid,owner in assignments:session.register_and_request(cid,principal=owner)
        else:
            for cid,owner in assignments:
                if not port.can_submit(1):raise BudgetExceeded('CANDIDATE_QUOTA_EXHAUSTED')
                text=('Rule diagnostic: propose '+cid+' for shared evaluation; discussion text remains uncompiled.' if kind=='free_debate' else '')
                session.propose(cid,principal=owner,raw_text=text)
                session.register_and_request(cid,principal=owner)
        session.flush()
        if kind=='abn':
            from .schema import Objection
            from .verifier import scope_for
            relations=[]
            for cid,result in port.eligibility.items():
                if cid not in session.state.candidate_aliases:continue
                c=port._candidates[cid];scope=scope_for(port._task,c,port._evidence.source_hash)
                for atom_id,record in result['roles'].items():
                    owner=atom_id.rsplit(':',1)[1];claim_id=cid+':'+atom_id
                    if record['status']=='REFUTED':
                        objection=Objection(objection_id='objection:'+claim_id,owner=owner,kind='objective',
                            policy_hash=scope.policy_hashes[0],scope=scope,reason='absolute utility below owner threshold',
                            claim_ids=[claim_id],status='OPEN')
                        session.emit('CHALLENGE',{'objections':[objection.model_dump(mode='json')]},principal=owner)
                        relations.append({'attack':claim_id,'target':scope.policy_hashes[0],'type':'preference_objection'})
                    elif record['status']=='SUPPORTED':
                        relations.append({'support':claim_id,'target':scope.policy_hashes[0],'type':'fact_support_not_assent'})
            durable_json(session.root/'abn-relations.json',{'adaptation':'finite-ABN-proCLAIM-style-v1','relations':relations,
                'resolution_order':'integrity_scope_then_atom_then_owner_then_common_gate',
                'version_handling':'exact_policy_scope_no_transfer','paper_line_by_line_replication':False})
        session.explicit_decisions()
        receipt=session.finish()
    except BudgetExceeded:
        receipt=session.finish(reason='budget')
    replay=session.replay_transitions()
    return {**receipt,'transition_replay':replay['status'],'algorithm':kind,'controller_source':'rule','scheduler':'fixed_role_rotation_stable_batch_publication_v1',
        'tools':['compile','information_query','registered_evidence','explicit_owner_decision','common_final_gate'],
        'feedback_timing':'after_complete_fixed_batch','schema_repair':'NA_deterministic_rule_control',
        'language_error_rate':None,'language_metric_status':'NA_RULE'}


class IndependentVotingMethod:
    name='independent'
    def run(self,port):return _run_rule_protocol(port,self.name)


class FreeDebateMethod:
    name='free_debate'
    def run(self,port):return _run_rule_protocol(port,self.name)


class ABNControlMethod:
    name='abn'
    def run(self,port):return _run_rule_protocol(port,self.name)


class AllAbstainMethod:
    name='all_abstain'
    def run(self,port):return port.protocol_session().finish(reason='abstain')
