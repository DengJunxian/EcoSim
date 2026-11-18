"""Two explicit local treatments over the common authenticated finite protocol.

This is a controlled semantic intervention, not another optimizer, evidence
service or final gate. Every local result remains subordinate to that gate.
"""
from __future__ import annotations
from pydantic import Field
from core.experiments.strict_types import StrictResearchModel
from core.experiments.identity import research_hash
from core.experiments.archive import durable_json
from .schema import CDRConfig,ClaimSpec,SupportJustification
from .protocol import _acyclic,_claim_allowed,_fact_status
from .verifier import required_atoms,compile_candidate,evaluate_eligibility,role_eligibility,credible_shortlist


class SupportGraph(StrictResearchModel):
    claims:dict[str,ClaimSpec]=Field(default_factory=dict)
    justifications:dict[str,SupportJustification]=Field(default_factory=dict)
    withdrawals:dict[str,str]=Field(default_factory=dict)
    justification_status:dict[str,str]=Field(default_factory=dict)
    verdicts:dict[str,str]=Field(default_factory=dict)
    receipts:list[dict]=Field(default_factory=list)


def add_justification(graph,justification,claims,services):
    """Register only the finite claim's actual evidence or declared conjunction."""
    graph=SupportGraph.model_validate(graph);j=SupportJustification.model_validate(justification)
    all_claims={**graph.claims,**claims}
    if any(k in graph.claims and graph.claims[k]!=v for k,v in claims.items()):raise ValueError('SUPPORT_CLAIM_OVERWRITE')
    if j.conclusion_claim_id not in all_claims:raise ValueError('UNKNOWN_SUPPORT_CONCLUSION')
    claim=all_claims[j.conclusion_claim_id];_claim_allowed(claim,services)
    if j.scope!=claim.scope:raise ValueError('SUPPORT_SCOPE_MISMATCH')
    if j.justification_id in graph.justifications:
        if graph.justifications[j.justification_id]!=j:raise ValueError('JUSTIFICATION_OVERWRITE')
        return graph
    if j.rule=='positive_conjunction':
        if claim.kind!='model_conjunction' or set(j.premise_claim_ids)!=set(claim.dependencies):
            raise ValueError('UNREGISTERED_CONJUNCTION')
        if any(k not in all_claims or all_claims[k].scope!=claim.scope for k in j.premise_claim_ids):
            raise ValueError('CONJUNCTION_SCOPE_OR_PREMISE')
        expected={(a.kind,a.model_id,a.quantity_id,a.direction,a.threshold) for a in required_atoms(services.task)}
        actual={(all_claims[k].kind,all_claims[k].model_id,all_claims[k].quantity_id,all_claims[k].direction,all_claims[k].threshold) for k in j.premise_claim_ids}
        if actual!=expected:raise ValueError('CONJUNCTION_MISSING_REQUIRED_MODEL_ATOMS')
    elif claim.kind=='model_conjunction':raise ValueError('CONJUNCTION_REQUIRES_EXPLICIT_PREMISES')
    else:
        # Evidence integrity is checked now and again whenever a source changes.
        status=_fact_status(claim,j.evidence_ids,services)
        if status in {'INVALID_ARTIFACT','INAPPLICABLE','UNRESOLVED'}:raise ValueError('INVALID_OR_MISSING_SUPPORT_SOURCE')
    data=graph.model_dump(mode='json');data['claims']={k:v.model_dump(mode='json') for k,v in all_claims.items()}
    data['justifications'][j.justification_id]=j.model_dump(mode='json')
    edges={k:list(v.dependencies) for k,v in all_claims.items()}
    for value in data['justifications'].values():edges.setdefault(value['conclusion_claim_id'],[]).extend(value['premise_claim_ids'])
    _acyclic(edges)
    return SupportGraph.model_validate(data)


def refresh_support(graph,services,*,changed_evidence_ids=None,withdrawals=None):
    """Re-evaluate only affected conclusions and their positive dependants."""
    graph=SupportGraph.model_validate(graph);data=graph.model_dump(mode='json');withdrawals=withdrawals or {}
    if any(k not in graph.justifications or not v for k,v in withdrawals.items()):raise ValueError('UNKNOWN_WITHDRAWAL_OR_MISSING_REASON')
    data['withdrawals'].update(withdrawals)
    changed=None if changed_evidence_ids is None else set(changed_evidence_ids)
    affected=set(graph.claims) if changed is None else set()
    for k,j in graph.justifications.items():
        if k in withdrawals or (changed is not None and set(j.evidence_ids)&changed):affected.add(j.conclusion_claim_id)
    while True:
        extra={j.conclusion_claim_id for j in graph.justifications.values() if set(j.premise_claim_ids)&affected}
        if extra<=affected:break
        affected|=extra
    edges={k:list(c.dependencies) for k,c in graph.claims.items()};_acyclic(edges)
    visited=set();order=[]
    def visit(key):
        if key in visited:return
        for p in edges.get(key,[]):visit(p)
        visited.add(key);order.append(key)
    for key in sorted(affected):visit(key)
    updated=[]
    for key in order:
        if key not in affected:continue
        claim=graph.claims[key];proofs=[j for j in graph.justifications.values() if j.conclusion_claim_id==key]
        active=[];verdicts=[]
        for j in proofs:
            if j.justification_id in data['withdrawals']:status='INAPPLICABLE'
            elif j.rule=='positive_conjunction':
                values=[data['verdicts'].get(p,'UNRESOLVED') for p in j.premise_claim_ids]
                status=next((s for s in ['INVALID_ARTIFACT','INAPPLICABLE','REFUTED','UNRESOLVED'] if s in values),'SUPPORTED')
            else:
                status=_fact_status(claim,j.evidence_ids,services);active.extend(j.evidence_ids)
            data['justification_status'][j.justification_id]=status
            if j.justification_id not in data['withdrawals']:verdicts.append(status)
        if claim.kind!='model_conjunction':
            # Evaluate all still-active records together: no favorable record selection.
            status=_fact_status(claim,list(dict.fromkeys(active)),services) if active else 'UNRESOLVED'
        elif not verdicts:status='UNRESOLVED'
        elif len(set(verdicts))>1:status='UNRESOLVED'
        else:status=verdicts[0]
        before=data['verdicts'].get(key,'UNRESOLVED');data['verdicts'][key]=status
        updated.append({'claim_id':key,'before':before,'after':status,
            'retained_justifications':[j.justification_id for j in proofs if data['justification_status'][j.justification_id]=='SUPPORTED'],
            'reason':'recompute_affected_positive_support'})
    receipt={'changed_evidence_ids':None if changed is None else sorted(changed),'withdrawals':withdrawals,
        'affected_claim_ids':sorted(affected),'updates':updated,'new_samples':0}
    data['receipts'].append(receipt)
    return SupportGraph.model_validate(data),receipt


def support_lineage(graph,services):
    ids=sorted({i for j in graph.justifications.values() for i in j.evidence_ids})
    records=[services.evidence_records[i] for i in ids if i in services.evidence_records]
    return {'unique_evidence_ids':ids,'unique_raw_refs':sorted({r for e in records for r in e.raw_run_refs}),
        'replication_ids':sorted({r for e in records for r in e.replication_ids}),
        'statistical_n':0 if all(e.method=='exact_analytic' for e in records) else None,
        'effective_n_rule':'No new independent sample from copying or adding a justification.'}


class CDRProtocol:
    name='cdr'
    def __init__(self,*,separate_fact_assent,maintain_scope_support):
        self.flags=CDRConfig(separate_fact_assent=separate_fact_assent,maintain_scope_support=maintain_scope_support)
        self.graph=SupportGraph();self.positions={};self.local_history=[];self.feedback=[]

    @property
    def configuration(self):return {'name':self.name,'controller':'fixed_rule_rotation_v1',**self.flags.model_dump(mode='json')}

    def prompt_difference(self):
        return {'separation':('Distinguish supported facts, your absolute eligibility, and your explicit policy assent.' if self.flags.separate_fact_assent else ''),
            'scope_support':('Use exact policy/scope support; retain other valid justifications when one is withdrawn.' if self.flags.maintain_scope_support else '')}

    def observe(self,before,message,services,common_receipt,after):
        """After raw persistence; affects only method state and declared feedback."""
        local={'operation_id':message.operation_id,'flags':self.flags.model_dump(mode='json'),
            'common_applied':common_receipt['applied'],'reminders':[],'method_guard':None,'new_queries':0}
        identity=(message.speaker==services.auth.principal and message.source==services.auth.source and
            (message.raw_ref,message.request_hash,message.budget_receipt_hash)==
            (services.auth.raw_ref,services.auth.request_hash,services.auth.budget_receipt_hash))
        reserved=services.budget_receipts.get(message.budget_receipt_hash)
        if (not identity or before.terminal!='ACTIVE' or reserved is None or reserved['charges']['messages']!=1
            or any(h['message']['budget_receipt_hash']==message.budget_receipt_hash for h in before.history)
            or message.operation_id in before.processed_operations):return local
        p=message.parsed_payload
        if self.flags.separate_fact_assent and message.type in {'ACKNOWLEDGE_FACT','DECLARE_ASSENT'}:
            local['reminders'].append(self.prompt_difference()['separation'])
        if message.type=='DECLARE_ASSENT' and (p.owner==message.speaker or
            (message.speaker=='designer' and p.owner in services.task.delegated_role_ids)) and p.policy_hash in before.candidates:
            cid=before.candidates[p.policy_hash]['candidate']['candidate_id']
            candidate=next(c for c in services.task.candidate_pool if c.candidate_id==cid)
            records=[r for i,r in services.evidence_records.items() if i in before.evidence_ids and r.scope.policy_hashes==[p.policy_hash]
                and r.atom_id in {a.atom_id for a in required_atoms(services.task)}]
            eligible=role_eligibility(services.task,candidate,p.owner,records,evidence_service=services.evidence_service)
            accepted=not self.flags.separate_fact_assent or eligible['status']=='SUPPORTED'
            local['method_guard']={'owner':p.owner,'eligible':eligible['status'],'accepted_locally':accepted}
            if accepted:self.positions[p.owner+':'+p.policy_hash]={'owner':p.owner,'policy_hash':p.policy_hash,'raw_ref':message.raw_ref,
                'status':'DECLARED','eligible':eligible['status'],'support_ids':[r.evidence_id for r in records]}
        elif message.type=='DECLARE_DISSENT' and p.owner==message.speaker:
            key=p.owner+':'+p.policy_hash
            if key in self.positions:self.positions[key]={**self.positions[key],'status':'WITHDRAWN'}
        if self.flags.maintain_scope_support and common_receipt['applied']:
            if message.type in {'REGISTER_CLAIM','TEST_RESULT'}:
                for key,claim in after.claims.items():
                    if claim.kind=='model_conjunction':
                        j=SupportJustification(justification_id='and:'+key,conclusion_claim_id=key,rule='positive_conjunction',
                            premise_claim_ids=list(claim.dependencies),evidence_ids=[],scope=claim.scope)
                        self.graph=add_justification(self.graph,j,after.claims,services)
                    elif message.type=='TEST_RESULT':
                        # Single-record proofs keep provenance; the refresh also checks
                        # their union for conflicts and never aggregates sample counts.
                        for identity in p.evidence_ids:
                            record=services.evidence_records[identity]
                            atom=next((a for a in required_atoms(services.task) if (a.kind,a.model_id,a.quantity_id,a.direction,a.threshold)==
                                (claim.kind,claim.model_id,claim.quantity_id,claim.direction,claim.threshold)),None)
                            target=atom.atom_id if atom else claim.claim_id
                            if record.scope!=claim.scope or record.atom_id!=target:continue
                            j=SupportJustification(justification_id=research_hash({'claim':key,'evidence':identity}),conclusion_claim_id=key,
                                rule='registered_evidence',premise_claim_ids=[],evidence_ids=[identity],scope=claim.scope)
                            self.graph=add_justification(self.graph,j,after.claims,services)
                if message.type=='TEST_RESULT':self.graph,local['support_refresh']=refresh_support(self.graph,services,changed_evidence_ids=p.evidence_ids)
            elif message.type=='REVISE':
                ph=after.candidate_aliases.get(p.candidate.candidate_id)
                local['reminders'].append({'scope_support':self.prompt_difference()['scope_support'],'old_policy':p.parent_policy_hash,
                    'new_policy':ph,'semantic_change':ph!=p.parent_policy_hash,'assent_transfer':False})
                self.graph,local['support_refresh']=refresh_support(self.graph,services)
            if 'support_refresh' in local:
                local['commitment_applicability_updates']=self.refresh_positions(services)
        self.local_history.append(local)
        return local

    def refresh_positions(self,services):
        """Preserve assent history; recompute applicability of its cited sources."""
        updates=[]
        for key,position in self.positions.items():
            if position['status']=='WITHDRAWN' or not position['support_ids']:continue
            candidate=next((c for c in services.task.candidate_pool
                if self._policy_hash(services.task,c)==position['policy_hash']),None)
            valid=candidate is not None and all(i in services.evidence_records and
                services.evidence_records[i].scope.policy_hashes==[position['policy_hash']] and
                services.evidence_service.verify(services.evidence_records[i],candidate) for i in position['support_ids'])
            status='DECLARED' if valid else 'INAPPLICABLE'
            if status!=position['status']:
                updates.append({'owner_policy':key,'before':position['status'],'after':status,'raw_ref':position['raw_ref']})
                self.positions[key]={**position,'status':status}
        return updates

    @staticmethod
    def _policy_hash(task,candidate):
        try:return compile_candidate(task,candidate).policy_hash
        except ValueError:return None

    def method_agreements(self,task):
        policies={p['policy_hash'] for p in self.positions.values()}
        return sorted(ph for ph in policies if all(self.positions.get(owner+':'+ph,{}).get('status')=='DECLARED' for owner in task.required_signers))

    def run(self,port):
        from .budget import BudgetExceeded
        session=port.protocol_session(local_controller=self)
        try:
            port.acquire_shared_information()
            for index,cid in enumerate(port.candidate_ids):
                if not port.can_submit(1):raise BudgetExceeded('CANDIDATE_QUOTA_EXHAUSTED')
                owner=port.objective_ids[index%len(port.objective_ids)]
                session.propose(cid,principal=owner);session.register_and_request(cid,principal=owner)
            session.flush();session.explicit_decisions();receipt=session.finish()
        except BudgetExceeded:receipt=session.finish(reason='budget')
        port.method_claims=self.method_agreements(port._task)
        report={'configuration':self.configuration,'prompt_difference':self.prompt_difference(),'source':'rule',
            'graph':self.graph.model_dump(mode='json'),'positions':self.positions,'local_history':self.local_history,
            'current_evidence_shortlist':credible_shortlist(port._task,port.evidence_by_candidate,evidence_service=port._evidence),
            'new_local_evidence_queries':0,'prompt_tokens':'NA_RULE','statistical_n':0}
        durable_json(session.root/'cdr-local-state.json',report)
        replay=session.replay_transitions()
        return {**receipt,'configuration':self.configuration,'transition_replay':replay['status'],
            'scheduler':'fixed_role_rotation_stable_batch_publication_v1','source':'rule','new_local_evidence_queries':0}
