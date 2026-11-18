"""Bounded live analytic negotiation on the existing protocol and common verifier."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
from pydantic import Field
from core.experiments.archive import durable_json
from core.experiments.strict_types import StrictResearchModel, strict_json_loads
from core.experiments.identity import research_hash, file_sha256
from .schema import AssentReceipt, Objection
from .services import RuleOwnerService
from .verifier import compile_candidate, scope_for
from .budget import ResourceVector, BudgetExceeded
from .llm_adapter import DesignCallError
from .cdr import CDRProtocol


class TurnAction(StrictResearchModel):
    action: Literal['propose','revise','acknowledge_fact','assent','decline','challenge','query_information','abstain']
    candidate_id: str | None
    parent_candidate_id: str | None
    claim_id: str | None
    verdict: Literal['SUPPORTED','REFUTED','UNRESOLVED'] | None
    information_id: str | None
    objection_kind: Literal['empirical','objective','structural_model','scope_measurement'] | None = None
    claimed_agreement: bool
    rationale: str = Field(max_length=2000)


class OwnerDecision(StrictResearchModel):
    candidate_id: str
    decision: Literal['ASSENT','DECLINE','UNRESOLVED']
    rationale: str = Field(max_length=1000)


class OwnerResponse(StrictResearchModel):
    decisions: list[OwnerDecision] = Field(max_length=6)


class LLMOwnerService(RuleOwnerService):
    """Authenticates an actual owner response; never substitutes rule-based assent."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.bindings={}

    def decide(self,*args,**kwargs):raise ValueError('LLM_OWNER_REQUIRES_EXPLICIT_RESPONSE')

    @staticmethod
    def _contains(raw, decision):
        if 'decisions' in raw:return decision.model_dump(mode='json') in raw['decisions']
        return raw.get('candidate_id')==decision.candidate_id and raw.get('action')=={
            'ASSENT':'assent','DECLINE':'decline','UNRESOLVED':'abstain'}[decision.decision] and raw.get('rationale')==decision.rationale

    def record(self,owner,candidate,evidence,decision,call,source):
        if self.task.signing_layer!='delegated_explicit_llm_v1' or owner not in self.task.required_signers:
            raise ValueError('LLM_OWNER_NOT_AUTHORIZED')
        request=strict_json_loads(Path(call['request_ref']).read_text())
        response=strict_json_loads(Path(call['response_ref']).read_text())
        parsed=strict_json_loads(response['text'])
        parsed_model=OwnerResponse.model_validate(parsed) if 'decisions' in parsed else TurnAction.model_validate(parsed)
        if (request['source']!=source or response['source']!=source or not response.get('ok') or
            request['request_id']!=call['request_id'] or not request['request_id'].endswith(':'+owner) or
            research_hash(parsed_model)!=call.get('parsed_hash') or not self._contains(parsed,decision) or
            candidate.candidate_id!=decision.candidate_id):raise ValueError('OWNER_RESPONSE_BINDING_MISMATCH')
        sequence=len(self.issued)+1
        # Original provider response already exists, before the message or eligibility guard.
        self.ledger.reserve(f'owner-message:{sequence}',ResourceVector(messages=1),purpose='explicit_llm_owner_decision')
        body=dict(episode_id=self.ledger.episode,owner=owner,policy_hash=compile_candidate(self.task,candidate).policy_hash,
            task_hash=self.task.task_hash,preference_hash=self.task.preference_hash,acceptance_rule_hash=self.task.acceptance_rule_hash,
            decision=decision.decision,source=source,decision_rule='explicit_llm_response_v1',
            evidence_ids=[e.evidence_id for e in evidence],information_receipt_ids=list(self.information.visible[owner].values()),
            raw_decision_ref=call['response_ref'],sequence=sequence)
        receipt=AssentReceipt(receipt_id=research_hash(body),**body)
        binding={'call':call,'decision':decision.model_dump(mode='json'),'response_sha256':file_sha256(call['response_ref']),
            'request_sha256':file_sha256(call['request_ref'])}
        durable_json(self.output/(receipt.receipt_id+'-binding.json'),binding)
        durable_json(self.output/(receipt.receipt_id+'.json'),receipt.model_dump(mode='json'))
        self.bindings[receipt.receipt_id]=binding;self.issued[receipt.receipt_id]=receipt
        return receipt

    def verify(self,receipt,candidate,evidence):
        try:
            r=AssentReceipt.model_validate(receipt);b=self.bindings[r.receipt_id]
            if self.issued[r.receipt_id]!=r or AssentReceipt.model_validate_json((self.output/(r.receipt_id+'.json')).read_text())!=r:return False
            if strict_json_loads((self.output/(r.receipt_id+'-binding.json')).read_text())!=b:return False
            if file_sha256(r.raw_decision_ref)!=b['response_sha256'] or file_sha256(b['call']['request_ref'])!=b['request_sha256']:return False
            raw=strict_json_loads(Path(r.raw_decision_ref).read_text())
            decision=OwnerDecision.model_validate(b['decision'])
            return (r.owner in self.task.required_signers and self.roles[r.owner].assent_permission and
                r.task_hash==self.task.task_hash and r.episode_id==self.ledger.episode and
                r.policy_hash==compile_candidate(self.task,candidate).policy_hash and
                r.preference_hash==self.task.preference_hash and r.acceptance_rule_hash==self.task.acceptance_rule_hash and
                r.decision==decision.decision and candidate.candidate_id==decision.candidate_id and r.source==raw['source'] and
                self._contains(strict_json_loads(raw['text']),decision) and
                r.evidence_ids==[e.evidence_id for e in evidence] and
                all(self.information.verify(i,r.owner) for i in r.information_receipt_ids) and
                self.information.visible[r.owner][self.roles[r.owner].preference_card_id] in r.information_receipt_ids)
        except (ValueError,OSError,KeyError):return False


class LiveProtocolMethod:
    def __init__(self,method_id,client_factory,*,turns=6):
        self.name=method_id;self.factory=client_factory;self.turns=turns
        self.operation_feedback=[]
        self.drafts_released=method_id!='independent'
        self.controller=None
        if method_id.startswith('cdr_s'):
            self.controller=CDRProtocol(separate_fact_assent=method_id[5]=='1',maintain_scope_support=method_id[-1]=='1')

    @property
    def configuration(self):
        return {'method_id':self.name,'controller':'bounded_llm_protocol_v2','design_turns':self.turns,
            'operation_feedback':'common_operation_receipts_v1',
            'closing_owner_calls':3,'independent_history':self.name=='independent',
            'cdr_flags':None if self.controller is None else self.controller.flags.model_dump(mode='json')}

    def bind(self,ledger,output):self.client=self.factory(ledger,output)

    def _prompt_feedback(self):
        return [{k:r[k] for k in ('principal','request_id','action','candidate_id','applied','reason')} for r in self.operation_feedback]

    def _setup(self,s):
        pass

    def _feedback_extension(self,s,item,transitions):
        return {}

    def _messages(self,s,principal,history,closing=False):
        port=s.port
        notes={'single':'Use a single central designer to solve this finite policy problem with the same tools and preferences.',
            'independent':'Develop an independent proposal using your own preceding statements and the shared tool results.',
            'free_debate':'Discuss and revise policy proposals with the other roles.',
            'abn':'Use explicit support and attack arguments; distinguish empirical objections from preference or model objections.'}
        method_text=notes.get(self.name,'Discuss and revise policy proposals with the other roles.')
        if self.controller:method_text+=' '+' '.join(self.controller.prompt_difference().values())
        model=OwnerResponse if closing else TurnAction
        system='You are participating in a controlled finite analytic policy co-design task. Return ONLY a JSON object matching the supplied schema. Keep each rationale under 25 words. '+method_text
        if closing:system+=' Give your own explicit decisions on any of the proposed candidates. You may assent to several, decline, remain unresolved, or return an empty list. Do not speak for another role.'
        facts=[]
        for cid,records in port.evidence_by_candidate.items():
            facts.append({'candidate_id':cid,'policy_hash':s.state.candidate_aliases.get(cid),
                'evidence':[{'atom':r.atom_id,'interval':r.interval,'evidence_id':r.evidence_id} for r in records]})
        visible_history=history if self.name!='independent' else [h for h in history if h['principal']==principal]
        body={'principal':principal,'phase':'owner_decision' if closing else 'design',
            'task_id':s.task.task_id,'risk_rule':'Every registered model must meet each owner threshold and all constraints.',
            'candidates':[c.model_dump(mode='json') for c in s.task.candidate_pool],
            'information':port._information.view(principal),'queryable_ids':port._information.entitlements[principal][1],
            'constraints':[c.model_dump(mode='json') for c in s.task.outcome_constraints],
            'proposed':dict(s.state.candidate_aliases),'facts':facts,
            'claim_columns':['claim_id','verdict','quantity'],
            'claims':[[k,s.state.fact_verdicts.get(k),v.quantity_id] for k,v in s.state.claims.items()],
            'history':visible_history,'schema':model.model_json_schema(),
            'operation_feedback':self._prompt_feedback() if self.drafts_released else [],
            'feedback_rule':'Operation receipts are produced by the common tool boundary. A rejected operation has not succeeded; use its reason and the current proposed/claims lists on a later scheduled turn. An applied operation is not a certificate of agreement. No extra retry turns are provided.',
            'tools':'propose/revise automatically registers and evaluates the finite candidate; facts are exact conditional analytic values, not market samples. ACK acknowledges a fact only. Only an owner can assent. claimed_agreement states your raw agreement claim. abstain consumes a turn. All actions use exact candidate IDs.'}
        if self.controller:
            aliases={ph:cid for cid,ph in s.state.candidate_aliases.items()}
            body['method_feedback']={'positions':[
                {'owner':p['owner'],'candidate_id':aliases.get(p['policy_hash']),'status':p['status'],'eligible':p['eligible']}
                for p in self.controller.positions.values()],
                'support_statuses':self.controller.graph.verdicts if self.controller.flags.maintain_scope_support else {},
                'recent_guards':[{k:h[k] for k in ('reminders','method_guard')} for h in self.controller.local_history[-2:]]}
        return [{'role':'system','content':system},{'role':'user','content':json.dumps(body,ensure_ascii=False,separators=(',',':'))}]

    def _feedback(self,s,principal,request_id,action,candidate_id,transitions,reason=None):
        # Only trusted transition receipts enter this channel. Model prose is never
        # interpreted as a receipt, and raw provider bytes remain in their archive.
        operations=[]
        for receipt in transitions:
            raw=strict_json_loads(Path(receipt['raw_ref']).read_text())
            operations.append({'type':raw.get('type'),'applied':receipt['applied'],'reason':receipt.get('reason')})
        rejected=[r['reason'] for r in operations if not r['applied']]
        item={'principal':principal,'request_id':request_id,'action':action,'candidate_id':candidate_id,
            'applied':reason is None and not rejected,'reason':reason or (rejected[0] if rejected else 'APPLIED'),
            'operations':operations}
        item.update(self._feedback_extension(s,item,transitions))
        item['receipt_id']=research_hash(item)
        self.operation_feedback.append(item)
        durable_json(s.root/'operation-feedback.json',{'version':'common_operation_receipts_v1','receipts':self.operation_feedback})

    def _with_feedback(self,s,principal,call,action,candidate_id,operation):
        start=len(s.transition_log);reason=None
        try:return operation()
        except (ValueError,BudgetExceeded) as exc:
            reason=str(exc);raise
        finally:self._feedback(s,principal,call['request_id'],action,candidate_id,s.transition_log[start:],reason)

    def _apply_with_feedback(self,s,principal,action,call):
        return self._with_feedback(s,principal,call,action.action,action.candidate_id,
            lambda:self._apply(s,principal,action,call))

    def _raw_decisions(self,s,principal,decisions,call):
        for decision in decisions:
            c=s.port._candidates.get(decision.candidate_id)
            if decision.decision=='ASSENT':
                s.port.protocol_raw_assents.append({'owner':principal,'policy_hash':None if c is None else compile_candidate(s.task,c).policy_hash,
                    'principal':principal,'source':self.client.config.source,'raw_ref':call['response_ref'],'operation_id':call['request_id']})

    def _decision(self,s,principal,decision,call,*,record_raw=True):
        port=s.port;c=port._candidates.get(decision.candidate_id)
        if record_raw:self._raw_decisions(s,principal,[decision],call)
        if principal not in s.task.required_signers or c is None or c.candidate_id not in s.state.candidate_aliases:
            raise ValueError('OWNER_DECISION_REQUIRES_OWN_ROLE_AND_PROPOSED_POLICY')
        records=port.evidence_by_candidate.get(c.candidate_id,[])
        receipt=port._owners.record(principal,c,records,decision,call,self.client.config.source)
        reservation=port.ledger.snapshot()['transactions']['owner-message:'+str(receipt.sequence)]
        if decision.decision=='ASSENT':
            s.emit('DECLARE_ASSENT',{'owner':principal,'policy_hash':receipt.policy_hash,'receipt_id':receipt.receipt_id},
                principal=principal,source=self.client.config.source,raw_text=decision.rationale,existing_reservation=reservation)
        elif decision.decision=='DECLINE':
            s.emit('DECLARE_DISSENT',{'owner':principal,'policy_hash':receipt.policy_hash,'reason':decision.rationale or 'declined','objection_ids':[]},
                principal=principal,source=self.client.config.source,raw_text=decision.rationale,existing_reservation=reservation)

    def _apply(self,s,principal,action,call):
        port=s.port;port.observe_raw_claim(action,call['response_ref'])
        cid=action.candidate_id;c=port._candidates.get(cid);raw={'principal':principal,'source':self.client.config.source,'raw_text':action.rationale}
        if action.action in {'propose','revise'}:
            if c is None:raise ValueError('UNKNOWN_PROPOSED_CANDIDATE')
            if action.action=='revise':
                parent=s.state.candidate_aliases.get(action.parent_candidate_id)
                if parent is None:raise ValueError('UNKNOWN_REVISION_PARENT')
                receipt=s.emit('REVISE',{'candidate':c.model_dump(mode='json'),'parent_policy_hash':parent},**raw)
            else:receipt=s.emit('PROPOSE',{'candidate':c.model_dump(mode='json')},**raw)
            if receipt['applied'] and cid not in port.evidence_by_candidate:
                s.register_and_request(cid,principal=principal);s.flush()
        elif action.action=='query_information':port._information.query(principal,action.information_id)
        elif action.action in {'assent','decline'}:
            self._decision(s,principal,OwnerDecision(candidate_id=cid or '',decision='ASSENT' if action.action=='assent' else 'DECLINE',rationale=action.rationale),call)
        elif action.action=='acknowledge_fact':
            claim=s.state.claims.get(action.claim_id)
            ids=[r.evidence_id for r in s.evidence_records.values() if claim and r.scope==claim.scope]
            s.emit('ACKNOWLEDGE_FACT',{'claim_id':action.claim_id,'verdict':action.verdict,'evidence_ids':ids},**raw)
        elif action.action=='challenge':
            if c is None or cid not in s.state.candidate_aliases:raise ValueError('UNKNOWN_CHALLENGED_POLICY')
            objection=Objection(objection_id='llm:'+call['request_id'],owner=principal,kind=action.objection_kind or 'objective',
                policy_hash=s.state.candidate_aliases[cid],scope=scope_for(s.task,c,port._evidence.source_hash),
                reason=action.rationale or 'unspecified objection',claim_ids=[action.claim_id] if action.claim_id else [],status='OPEN')
            s.emit('CHALLENGE',{'objections':[objection.model_dump(mode='json')]},**raw)

    async def run_async(self,port):
        s=port.protocol_session(local_controller=self.controller);history=[];errors=[];reason='completed_round';drafts=[]
        try:
            self._setup(s)
            for turn in range(self.turns):
                principal='designer' if self.name=='single' else port.objective_ids[turn%len(port.objective_ids)]
                port.reserve_round()
                try:
                    action=await self.client.request(self._messages(s,principal,history),request_id=f'turn-{turn:02d}:'+principal,response_model=TurnAction)
                    call=self.client.calls[-1]
                    history.append({'principal':principal,'response':action.model_dump(mode='json')})
                    if self.name=='independent' and turn<3:
                        drafts.append((principal,action,call))
                    else:self._apply_with_feedback(s,principal,action,call)
                except DesignCallError as exc:
                    errors.append({'turn':turn,'reason':exc.code,'raw_refs':exc.raw_refs})
                    self._feedback(s,principal,f'turn-{turn:02d}:'+principal,'response',None,[],exc.code)
                    if exc.code!='INVALID_RESPONSE':raise
                except BudgetExceeded:raise
                except ValueError as exc:
                    errors.append({'turn':turn,'reason':str(exc),'raw_ref':self.client.calls[-1].get('response_ref')})
                if self.name=='independent' and turn==2:
                    for speaker,draft,raw_call in drafts:
                        try:self._apply_with_feedback(s,speaker,draft,raw_call)
                        except BudgetExceeded:raise
                        except ValueError as exc:errors.append({'draft_owner':speaker,'reason':str(exc),'raw_ref':raw_call['response_ref']})
                    self.drafts_released=True
                durable_json(s.root/'live-history.json',history)
            # Each required owner independently signs its own exact candidates.
            closing_messages={owner:self._messages(s,owner,history,closing=True) for owner in s.task.required_signers}
            for owner in s.task.required_signers:
                try:
                    answer=await self.client.request(closing_messages[owner],request_id='closing:'+owner,response_model=OwnerResponse)
                    call=self.client.calls[-1];seen=set()
                    self._raw_decisions(s,owner,answer.decisions,call)
                    for decision in answer.decisions:
                        def apply_decision():
                            if decision.candidate_id in seen:raise ValueError('DUPLICATE_OWNER_DECISION')
                            seen.add(decision.candidate_id)
                            self._decision(s,owner,decision,call,record_raw=False)
                        try:
                            self._with_feedback(s,owner,call,decision.decision,decision.candidate_id,apply_decision)
                        except BudgetExceeded:raise
                        except ValueError as exc:errors.append({'owner':owner,'reason':str(exc),'candidate_id':decision.candidate_id})
                    history.append({'principal':owner,'response':answer.model_dump(mode='json')})
                except DesignCallError as exc:
                    errors.append({'owner':owner,'reason':exc.code,'raw_refs':exc.raw_refs})
                    self._feedback(s,owner,'closing:'+owner,'response',None,[],exc.code)
                    if exc.code!='INVALID_RESPONSE':raise
                except BudgetExceeded:raise
                except ValueError as exc:errors.append({'owner':owner,'reason':str(exc)})
        except BudgetExceeded:reason='budget'
        finally:
            durable_json(s.root/'live-history.json',history);durable_json(s.root/'live-errors.json',errors)
            durable_json(s.root/'operation-feedback.json',{'version':'common_operation_receipts_v1','receipts':self.operation_feedback})
            port.failures.extend(errors)
            sdk=getattr(self.client.transport,'_client',None)
            if sdk is not None:await sdk.close()
        report=s.finish(reason=reason)
        if self.controller:
            port.method_claims=self.controller.method_agreements(s.task)
            durable_json(s.root/'cdr-local-state.json',{'configuration':self.configuration,
                'graph':self.controller.graph.model_dump(mode='json'),'positions':self.controller.positions,'history':self.controller.local_history})
        return {**report,'source':self.client.config.source,'natural_trace_status':'COLLECTED' if self.client.config.source=='live' else 'NOT_LIVE',
            'configuration':self.configuration,'response_errors':len(errors)}


class SharedEvidenceProtocolMethod(LiveProtocolMethod):
    """V3 fixed-pool protocol diagnostic, with identical charged evidence for all."""
    @property
    def configuration(self):
        return {**super().configuration,'controller':'bounded_llm_protocol_v3',
            'initial_evidence':'all_registered_candidates_before_design_v1',
            'finite_policy_semantics':'immutable_candidates_explicit_noop_revision_v1'}

    def _setup(self,s):
        before=s.port.ledger.snapshot()
        for candidate in s.task.candidate_pool:
            receipt=s.propose(candidate.candidate_id)
            if not receipt['applied']:raise ValueError('SHARED_POOL_PROPOSAL_REJECTED:'+receipt['reason'])
            s.register_and_request(candidate.candidate_id)
        s.flush()
        if set(s.port.evidence_by_candidate)!=set(s.port.candidate_ids):raise ValueError('INCOMPLETE_SHARED_POOL')
        durable_json(s.root/'shared-setup.json',{'version':'all_registered_candidates_before_design_v1',
            'candidate_ids':list(s.port.candidate_ids),'evidence_ids':{k:[r.evidence_id for r in v] for k,v in s.port.evidence_by_candidate.items()},
            'state_hash':research_hash(s.state),'budget_before':before,'budget_after':s.port.ledger.snapshot(),
            'source':'rule','owner_decisions_generated':0,'statistical_market_n':0})

    def _messages(self,s,principal,history,closing=False):
        messages=super()._messages(s,principal,history,closing)
        body=strict_json_loads(messages[-1]['content'])
        body['finite_pool_rule']='Candidate IDs select immutable registered policies. REVISE selects a registered target with a parent; using the same policy changes no parameters. Text in rationale never edits policy content. Synthetic outcomes come from the supplied exact table; do not infer them from economic names. All methods receive this complete pool before design, with the common resource charge.'
        body['fact_columns']=['atom','interval','evidence_id']
        for row in body['facts']:
            row['evidence']=[[r[k] for k in body['fact_columns']] for r in row['evidence']]
        messages[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'))
        return messages

    def _prompt_feedback(self):
        base=super()._prompt_feedback()
        for item,raw in zip(base,self.operation_feedback):
            if 'policy_effect' in raw:item['policy_effect']=raw['policy_effect']
        return base

    def _feedback_extension(self,s,item,transitions):
        if item['action']!='revise':return {}
        for receipt in transitions:
            raw=strict_json_loads(Path(receipt['raw_ref']).read_text())
            if raw.get('type')=='REVISE' and receipt['applied']:
                target=s.state.candidate_aliases[item['candidate_id']]
                return {'policy_effect':'UNCHANGED' if raw['payload']['parent_policy_hash']==target else 'DIFFERENT_REGISTERED_POLICY'}
        return {'policy_effect':'NOT_APPLIED'}
