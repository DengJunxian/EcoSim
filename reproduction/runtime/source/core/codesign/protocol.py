"""Finite, immutable protocol transitions. The runner executes returned requests.

No network, market, random generation or ledger mutations occur here. Services
are read-only capabilities supplied by the composition root, not by a message.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from pydantic import Field
from core.experiments.strict_types import StrictResearchModel
from core.experiments.identity import research_hash
from .schema import AgentMessage,ClaimSpec,Objection,Commitment
from .verifier import compile_candidate,scope_for,required_atoms,evaluate_eligibility,role_eligibility

Terminal=Literal['ACTIVE','AGREEMENT_CERTIFIED','SHORTLIST_WITH_DISSENT',
    'BUDGET_EXHAUSTED_UNRESOLVED','NO_ACCEPTABLE_POLICY_IN_ENUMERATED_SET','INVALID_WORLD']


class ProtocolState(StrictResearchModel):
    episode_id:str
    task_hash:str
    mode:Literal['analytic_fixture','fixed_pool_replay','online']
    round:int=Field(default=0,ge=0)
    sequence:int=Field(default=0,ge=0)
    candidates:dict=Field(default_factory=dict)
    candidate_aliases:dict[str,str]=Field(default_factory=dict)
    claims:dict[str,ClaimSpec]=Field(default_factory=dict)
    fact_verdicts:dict[str,str]=Field(default_factory=dict)
    evidence_ids:list[str]=Field(default_factory=list)
    objections:dict[str,Objection]=Field(default_factory=dict)
    commitments:dict[str,Commitment]=Field(default_factory=dict)
    dissents:dict=Field(default_factory=dict)
    dependencies:dict[str,list[str]]=Field(default_factory=dict)
    visible_information:dict[str,list[str]]=Field(default_factory=dict)
    pending_requests:dict=Field(default_factory=dict)
    processed_operations:dict[str,str]=Field(default_factory=dict)
    history:list[dict]=Field(default_factory=list)
    budget_state:dict=Field(default_factory=dict)
    alpha_reservations:list[str]=Field(default_factory=list)
    terminal:Terminal='ACTIVE'
    terminal_reason:str=''


class AuthenticatedInput(StrictResearchModel):
    principal:str
    source:Literal['live','rule','mock','recorded']
    raw_ref:str
    request_hash:str
    budget_receipt_hash:str
    service_event_hash:str|None=None


@dataclass(frozen=True)
class ProtocolServices:
    task:object
    source_hash:str
    evidence_service:object
    assent_service:object
    information_service:object
    evidence_records:dict
    assent_records:dict
    service_events:dict
    budget_receipts:dict
    budget_state:dict
    auth:AuthenticatedInput
    replay_catalog:dict|None=None


def _acyclic(edges):
    visiting=set();done=set()
    def visit(key):
        if key in visiting:raise ValueError('CYCLIC_DEPENDENCY')
        if key in done:return
        visiting.add(key)
        for other in edges.get(key,[]):visit(other)
        visiting.remove(key);done.add(key)
    for key in edges:visit(key)


def _candidate(state,policy_hash,services):
    for candidate in services.task.candidate_pool:
        try:
            if compile_candidate(services.task,candidate).policy_hash==policy_hash:return candidate
        except ValueError:continue
    raise ValueError('UNKNOWN_POLICY_VERSION')


def _claim_allowed(claim,services):
    task=services.task
    if claim.kind in {'paired_difference','static_feasible'}:
        services.evidence_service.validate_additional_claim(claim);return
    candidate=_candidate(None,claim.policy_hashes[0],services)
    if claim.scope!=scope_for(task,candidate,services.source_hash):raise ValueError('CLAIM_SCOPE_MISMATCH')
    if claim.evidence_mode!=('exact_analytic' if task.evidence_mode=='exact_analytic' else 'fresh_fixed_batch'):
        raise ValueError('CLAIM_EVIDENCE_MODE_MISMATCH')
    if claim.registered_plan_hash!=research_hash(services.evidence_service.plan):raise ValueError('CLAIM_PLAN_MISMATCH')
    if claim.kind=='model_conjunction':
        if not claim.dependencies or claim.model_id!='all' or claim.quantity_id!='all_registered_atoms' or claim.direction!='ge' or claim.threshold!=1.0:
            raise ValueError('INVALID_CONJUNCTION')
    else:
        atom=next((a for a in required_atoms(task) if (a.kind,a.model_id,a.quantity_id,a.direction,a.threshold)==
                   (claim.kind,claim.model_id,claim.quantity_id,claim.direction,claim.threshold)),None)
        if atom is None:raise ValueError('UNREGISTERED_FINITE_PREDICATE')


def _fact_status(claim,ids,services):
    candidate=_candidate(None,claim.policy_hashes[-1],services)
    records=[services.evidence_records.get(i) for i in ids]
    if any(r is None for r in records):return 'INVALID_ARTIFACT'
    atom=next((a for a in required_atoms(services.task) if (a.kind,a.model_id,a.quantity_id,a.direction,a.threshold)==
        (claim.kind,claim.model_id,claim.quantity_id,claim.direction,claim.threshold)),None)
    if claim.kind in {'paired_difference','static_feasible'}:
        records=[r for r in records if r.atom_id==claim.claim_id]
    elif claim.kind=='model_conjunction':
        base_ids={a.atom_id for a in required_atoms(services.task)}
        records=[r for r in records if r.atom_id in base_ids]
    elif atom is not None:records=[r for r in records if r.atom_id==atom.atom_id]
    if not records:return 'UNRESOLVED'
    if any(r is None or r.scope!=claim.scope or not services.evidence_service.verify(r,candidate) for r in records):
        return 'INVALID_ARTIFACT'
    if claim.kind=='model_conjunction':
        base_ids={a.atom_id for a in required_atoms(services.task)}
        return evaluate_eligibility(services.task,candidate,[r for r in records if r.atom_id in base_ids],evidence_service=services.evidence_service)['status']
    if atom is None and claim.kind not in {'paired_difference','static_feasible'}:return 'INAPPLICABLE'
    matched=[r for r in records if r.atom_id==(claim.claim_id if atom is None else atom.atom_id)]
    if not matched:return 'UNRESOLVED'
    verdicts=[]
    for r in matched:
        if r.interval is None:verdicts.append('UNRESOLVED');continue
        low,high=r.interval
        verdicts.append(('SUPPORTED' if low>=claim.threshold else 'REFUTED' if high<claim.threshold else 'UNRESOLVED')
            if claim.direction=='ge' else ('SUPPORTED' if high<=claim.threshold else 'REFUTED' if low>claim.threshold else 'UNRESOLVED'))
    return verdicts[0] if len(set(verdicts))==1 else 'UNRESOLVED'


def advance_protocol(state:ProtocolState,message:AgentMessage,services:ProtocolServices):
    """Return (new state, transition receipt, finite requests), retaining rejections."""
    state=ProtocolState.model_validate(state);message=AgentMessage.model_validate(message)
    before=research_hash(state);data=state.model_dump(mode='json');requests=[];reason='APPLIED';applied=True
    prior=state.processed_operations.get(message.operation_id)
    identity_valid=(message.speaker==services.auth.principal and message.source==services.auth.source
        and (message.raw_ref,message.request_hash,message.budget_receipt_hash)==
            (services.auth.raw_ref,services.auth.request_hash,services.auth.budget_receipt_hash))
    if prior==message.operation_fingerprint and identity_valid:
        return state,{'before_hash':before,'after_hash':before,'applied':False,'reason':'DUPLICATE_DELIVERY',
                      'raw_ref':message.raw_ref,'operation_id':message.operation_id},[]
    try:
        auth=services.auth;task=services.task
        if prior is not None:raise ValueError('OPERATION_ID_REUSED_WITH_DIFFERENT_CONTENT')
        if state.task_hash!=task.task_hash or message.episode_id!=state.episode_id:raise ValueError('EPISODE_SCOPE_MISMATCH')
        if state.terminal!='ACTIVE':raise ValueError('PROTOCOL_ALREADY_TERMINAL')
        if not identity_valid:
            raise ValueError('TRANSPORT_IDENTITY_MISMATCH')
        if message.speaker not in [r.role_id for r in task.roles]+['designer','evidence_service']:
            raise ValueError('UNKNOWN_SPEAKER')
        if message.budget_receipt_hash not in services.budget_receipts:raise ValueError('UNRESERVED_MESSAGE')
        reservation=services.budget_receipts[message.budget_receipt_hash]
        if reservation['charges']['messages']!=1:raise ValueError('MESSAGE_RESOURCE_NOT_RESERVED')
        if any(h['message']['budget_receipt_hash']==message.budget_receipt_hash for h in state.history):
            raise ValueError('MESSAGE_RESERVATION_REUSED')
        if message.sequence!=state.sequence+1 or message.round<state.round:raise ValueError('MESSAGE_ORDER_MISMATCH')
        if message.round>task.budget.rounds:raise ValueError('ROUND_BUDGET_EXCEEDED')
        payload=message.parsed_payload
        if message.type=='TEST_RESULT':
            if message.speaker!='evidence_service' or auth.service_event_hash!=research_hash(payload) or services.service_events.get(auth.service_event_hash)!=payload.model_dump(mode='json'):
                raise ValueError('ONLY_EVIDENCE_SERVICE_CAN_PUBLISH')
        elif message.speaker=='evidence_service':raise ValueError('SERVICE_CANNOT_SPEAK_FOR_A_ROLE')
        if message.type in {'PROPOSE','REVISE'}:
            candidate=payload.candidate
            # Allow a display-label revision only when its entire semantic policy
            # matches a registered candidate; unknown fields are already rejected.
            registered=next((c for c in task.candidate_pool if c.candidate_id==candidate.candidate_id),None)
            if registered is None:raise ValueError('CANDIDATE_OUTSIDE_FINITE_POOL')
            compiled=compile_candidate(task,candidate)
            if compiled.policy_hash!=compile_candidate(task,registered).policy_hash:raise ValueError('REVISION_OUTSIDE_REGISTERED_POOL')
            if message.type=='REVISE' and payload.parent_policy_hash not in state.candidates:raise ValueError('UNKNOWN_PARENT_POLICY')
            ph=compiled.policy_hash
            if ph not in data['candidates']:
                data['candidates'][ph]={'candidate':candidate.model_dump(mode='json'),'semantic_body':compiled.semantic_body,
                    'parent_hash':payload.parent_policy_hash if message.type=='REVISE' else None,'first_sequence':message.sequence}
            data['candidate_aliases'][candidate.candidate_id]=ph
            requests.append({'action':'record_candidate','candidate_id':candidate.candidate_id,'policy_hash':ph,
                             'semantic_change':message.type=='REVISE' and payload.parent_policy_hash!=ph})
        elif message.type=='REGISTER_CLAIM':
            claim=payload.claim;_claim_allowed(claim,services)
            if claim.claim_id in state.claims and state.claims[claim.claim_id]!=claim:raise ValueError('CLAIM_CANNOT_BE_OVERWRITTEN')
            if not set(claim.policy_hashes)<=set(state.candidates):raise ValueError('CLAIM_WITHOUT_PROPOSED_POLICY')
            if not set(claim.dependencies)<=set(state.claims):raise ValueError('UNKNOWN_CLAIM_DEPENDENCY')
            if claim.kind=='model_conjunction':
                deps=[state.claims[c] for c in claim.dependencies]
                expected={(a.kind,a.model_id,a.quantity_id,a.direction,a.threshold) for a in required_atoms(task)}
                observed={(c.kind,c.model_id,c.quantity_id,c.direction,c.threshold) for c in deps}
                if observed!=expected or any(c.scope!=claim.scope for c in deps):raise ValueError('CONJUNCTION_MUST_COVER_ALL_REQUIRED_MODEL_ATOMS')
            data['claims'][claim.claim_id]=claim.model_dump(mode='json');data['dependencies'][claim.claim_id]=list(claim.dependencies)
            _acyclic(data['dependencies']);data['fact_verdicts'].setdefault(claim.claim_id,'UNRESOLVED')
        elif message.type=='DISCLOSE':
            card=services.information_service.cards.get(payload.information_id)
            if card is None or message.speaker not in card.disclosable_by:raise ValueError('DISCLOSURE_PERMISSION_DENIED')
            if not services.information_service.verify(payload.information_receipt_id,message.speaker):raise ValueError('DISCLOSURE_WITHOUT_SOURCE')
            receipt=services.information_service.receipts[payload.information_receipt_id]
            if receipt['card_id']!=payload.information_id:raise ValueError('DISCLOSURE_SOURCE_MISMATCH')
            if any(r not in card.readable_by or r not in services.information_service.entitlements for r in payload.recipients):raise ValueError('RECIPIENT_PERMISSION_DENIED')
            for recipient in payload.recipients:
                visible=data['visible_information'].setdefault(recipient,[])
                if payload.information_id not in visible:visible.append(payload.information_id)
            requests.append({'action':'publish_information','information_id':payload.information_id,'recipients':payload.recipients,
                'principal':message.speaker,'source_receipt_id':payload.information_receipt_id,'budget_receipt_hash':message.budget_receipt_hash})
        elif message.type=='CHALLENGE':
            for objection in payload.objections:
                if objection.owner!=message.speaker:raise ValueError('OBJECTION_OWNER_MISMATCH')
                if objection.objection_id in state.objections:raise ValueError('OBJECTION_CANNOT_BE_OVERWRITTEN')
                if objection.policy_hash not in state.candidates:raise ValueError('UNKNOWN_OBJECTION_POLICY')
                c=_candidate(state,objection.policy_hash,services)
                if objection.scope!=scope_for(task,c,services.source_hash) or not set(objection.claim_ids)<=set(state.claims):raise ValueError('OBJECTION_SCOPE_MISMATCH')
                if objection.status!='OPEN':raise ValueError('OBJECTION_CANNOT_SELF_CERTIFY_RESOLUTION')
                data['objections'][objection.objection_id]=objection.model_dump(mode='json')
        elif message.type=='REQUEST_TEST':
            if payload.candidate_id not in state.candidate_aliases:raise ValueError('TEST_WITHOUT_CANDIDATE')
            ph=state.candidate_aliases[payload.candidate_id]
            if any(c not in state.claims or state.claims[c].policy_hashes[-1]!=ph for c in payload.claim_ids):raise ValueError('TEST_OUTSIDE_REGISTERED_CLAIMS')
            if state.mode=='fixed_pool_replay' and payload.candidate_id not in (services.replay_catalog or {}):raise ValueError('REPLAY_CANNOT_CREATE_NEW_EVIDENCE')
            request={'action':'evaluate_registered_candidate','candidate_id':payload.candidate_id,
                'claim_ids':list(payload.claim_ids),'operation_id':message.operation_id}
            data['pending_requests'][message.operation_id]=request;requests.append(request)
        elif message.type=='TEST_RESULT':
            pending=state.pending_requests.get(payload.request_operation_id)
            if pending is None or pending['candidate_id']!=payload.candidate_id:raise ValueError('UNREQUESTED_RESULT')
            c=_candidate(state,state.candidate_aliases[payload.candidate_id],services)
            for identity in payload.evidence_ids:
                record=services.evidence_records.get(identity)
                if record is None or not services.evidence_service.verify(record,c):raise ValueError('INVALID_EVIDENCE_RESULT')
                if identity not in data['evidence_ids']:data['evidence_ids'].append(identity)
            for claim_id in pending['claim_ids']:
                data['fact_verdicts'][claim_id]=_fact_status(state.claims[claim_id],payload.evidence_ids,services)
            # A fact update can discharge only its explicitly dependent empirical
            # objection; preferences and structural assumptions have separate owners.
            for objection in data['objections'].values():
                if objection['kind']=='empirical' and objection['claim_ids'] and all(data['fact_verdicts'].get(c)=='SUPPORTED' for c in objection['claim_ids']):
                    objection['status']='RESOLVED_BY_EVIDENCE'
            del data['pending_requests'][payload.request_operation_id]
        elif message.type=='ACKNOWLEDGE_FACT':
            if payload.claim_id not in state.claims or not set(payload.evidence_ids)<=set(state.evidence_ids):raise ValueError('ACK_WITHOUT_VISIBLE_EVIDENCE')
            actual=_fact_status(state.claims[payload.claim_id],payload.evidence_ids,services)
            if payload.verdict!=actual:raise ValueError('FACT_ACK_VERDICT_MISMATCH')
            data['fact_verdicts'][payload.claim_id]=actual
            # Fact acknowledgement never creates or alters role commitments.
        elif message.type=='DECLARE_ASSENT':
            if payload.owner!=message.speaker and not (message.speaker=='designer' and payload.owner in task.delegated_role_ids):
                raise ValueError('ASSENT_OWNER_PERMISSION_DENIED')
            c=_candidate(state,payload.policy_hash,services)
            receipt=services.assent_records.get(payload.receipt_id)
            if receipt is None or receipt.owner!=payload.owner or receipt.policy_hash!=payload.policy_hash or receipt.decision!='ASSENT':
                raise ValueError('EXPLICIT_OWNER_RECEIPT_REQUIRED')
            records=[services.evidence_records.get(i) for i in receipt.evidence_ids]
            if any(r is None for r in records) or not services.assent_service.verify(receipt,c,records):raise ValueError('INVALID_ASSENT_RECEIPT')
            if not set(receipt.evidence_ids)<=set(state.evidence_ids):raise ValueError('ASSENT_BEFORE_EVIDENCE_VISIBILITY')
            eligibility=role_eligibility(task,c,payload.owner,records,evidence_service=services.evidence_service)
            commitment=Commitment(owner=payload.owner,policy_hash=payload.policy_hash,scope=scope_for(task,c,services.source_hash),
                preference_hash=task.preference_hash,acceptance_rule_hash=task.acceptance_rule_hash,eligible=eligibility['status'],
                explicit_assent_receipt_id=receipt.receipt_id,raw_ref=message.raw_ref,support_ids=list(receipt.evidence_ids),
                sequence=message.sequence,current_status='DECLARED',status_reason='explicit_authenticated_owner_receipt')
            data['commitments'][payload.owner+':'+payload.policy_hash]=commitment.model_dump(mode='json')
        elif message.type=='DECLARE_DISSENT':
            if payload.owner!=message.speaker:raise ValueError('DISSENT_OWNER_PERMISSION_DENIED')
            if payload.policy_hash not in state.candidates:raise ValueError('UNKNOWN_DISSENT_POLICY')
            if any(i not in state.objections or state.objections[i].owner!=payload.owner for i in payload.objection_ids):raise ValueError('DISSENT_OBJECTION_OWNER_MISMATCH')
            key=payload.owner+':'+payload.policy_hash;data['dissents'][key]=payload.model_dump(mode='json')
            if key in data['commitments']:
                data['commitments'][key]['current_status']='WITHDRAWN';data['commitments'][key]['status_reason']=payload.reason
        elif message.type=='STOP':
            requests.append({'action':'common_terminal_evaluation','reason':payload.reason})
    except (ValueError,KeyError,StopIteration) as exc:
        data=state.model_dump(mode='json');requests=[];applied=False;reason=str(exc)
    data['sequence']=state.sequence+1;data['round']=max(state.round,min(message.round,services.task.budget.rounds))
    data['budget_state']=services.budget_state
    data['alpha_reservations']=list(services.budget_state.get('plans',{}))
    if prior is None:data['processed_operations'][message.operation_id]=message.operation_fingerprint
    data['history'].append({'message':message.model_dump(mode='json'),'applied':applied,'reason':reason})
    result=ProtocolState.model_validate(data)
    transition={'before_hash':before,'after_hash':research_hash(result),'raw_ref':message.raw_ref,
        'operation_id':message.operation_id,'applied':applied,'reason':reason,'budget_receipt_hash':message.budget_receipt_hash}
    return result,transition,requests
