"""Auditable information access and explicit role-owned decisions, outside verifier."""
from __future__ import annotations
from pathlib import Path
from core.experiments.archive import durable_json
from core.experiments.strict_types import strict_json_loads
from core.experiments.identity import research_hash
from .budget import ResourceVector
from .schema import AssentReceipt
from .verifier import compile_candidate, evaluate_eligibility


class InformationService:
    def __init__(self,task,ledger,output):
        self.task,self.ledger,self.output=task,ledger,Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.cards={c.card_id:c for c in task.information_cards};self.receipts={};self.visible={}
        self.entitlements={'designer':(task.method_initial_information_ids,task.method_queryable_information_ids)}
        self.entitlements.update({r.role_id:(r.initial_information_ids,r.queryable_information_ids) for r in task.roles})
        for principal,(initial,_) in self.entitlements.items():
            self.visible[principal]={}
            for card_id in initial:self._grant(principal,card_id,initial=True)

    def _grant(self,principal,card_id,*,initial):
        card=self.cards[card_id]
        if principal not in card.readable_by:raise ValueError('INFORMATION_PERMISSION_DENIED')
        body={'principal':principal,'card_id':card_id,'card_hash':research_hash(card),'initial':initial,
              'episode':self.ledger.episode,'sequence':len(self.receipts)+1}
        reservation=self.ledger.reserve('info:'+research_hash(body),ResourceVector(information_queries=0 if initial else 1),purpose='information_access')
        body['budget_receipt_hash']=reservation['receipt_hash'];body['receipt_id']=research_hash(body)
        durable_json(self.output/(body['receipt_id']+'.json'),body)
        self.receipts[body['receipt_id']]=body;self.visible[principal][card_id]=body['receipt_id']
        return card

    def query(self,principal,card_id):
        if principal not in self.entitlements or card_id not in self.cards:raise ValueError('UNKNOWN_INFORMATION_PRINCIPAL_OR_CARD')
        if card_id in self.visible[principal]:return self.cards[card_id]
        if card_id not in self.entitlements[principal][1]:raise ValueError('INFORMATION_PERMISSION_DENIED')
        return self._grant(principal,card_id,initial=False)

    def view(self,principal):
        return {k:self.cards[k].model_dump(mode='json') for k in self.visible[principal]}

    def publish(self,principal,card_id,recipients,source_receipt_id,budget_receipt_hash):
        """Deliver an authenticated disclosure; its message already paid the cost."""
        card=self.cards[card_id]
        if principal not in card.disclosable_by or not self.verify(source_receipt_id,principal):
            raise ValueError('DISCLOSURE_PERMISSION_DENIED')
        if self.receipts[source_receipt_id]['card_id']!=card_id:raise ValueError('DISCLOSURE_SOURCE_MISMATCH')
        if any(p not in self.entitlements or p not in card.readable_by for p in recipients):
            raise ValueError('RECIPIENT_PERMISSION_DENIED')
        reservations=self.ledger.snapshot()['transactions'].values()
        if not any(r['receipt_hash']==budget_receipt_hash and r['charges']['messages']==1 for r in reservations):
            raise ValueError('DISCLOSURE_MESSAGE_NOT_CHARGED')
        for recipient in recipients:
            if card_id in self.visible[recipient]:continue
            body={'principal':recipient,'card_id':card_id,'card_hash':research_hash(card),'initial':False,
                'episode':self.ledger.episode,'sequence':len(self.receipts)+1,'source':'disclosure',
                'disclosed_by':principal,'source_receipt_id':source_receipt_id,'budget_receipt_hash':budget_receipt_hash}
            body['receipt_id']=research_hash(body)
            durable_json(self.output/(body['receipt_id']+'.json'),body)
            self.receipts[body['receipt_id']]=body;self.visible[recipient][card_id]=body['receipt_id']

    def verify(self,receipt_id,principal):
        try:
            receipt=self.receipts[receipt_id]
            return receipt['principal']==principal and strict_json_loads((self.output/(receipt_id+'.json')).read_text())==receipt
        except (KeyError,ValueError,OSError):return False


class RuleOwnerService:
    def __init__(self,task,ledger,information,evidence_service,output):
        self.task,self.ledger,self.information,self.evidence_service=task,ledger,information,evidence_service
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.issued={};self.roles={r.role_id:r for r in task.roles}

    def _decision(self,owner,candidate,evidence):
        role=self.roles[owner]
        if not role.assent_permission:raise ValueError('ROLE_CANNOT_SIGN')
        self.information.query(owner,role.preference_card_id)
        eligibility=evaluate_eligibility(self.task,candidate,evidence,evidence_service=self.evidence_service)
        if eligibility['status'] in {'INVALID_ARTIFACT','INAPPLICABLE'}:return 'UNRESOLVED'
        required=[v['status'] for k,v in eligibility['roles'].items() if k.endswith(':utility:'+owner)]
        constraints=[v['status'] for v in eligibility['constraints'].values()]
        if not required:return 'UNRESOLVED'
        if any(s=='REFUTED' for s in required+constraints):return 'DECLINE'
        if any(s!='SUPPORTED' for s in required+constraints):return 'UNRESOLVED'
        return 'ASSENT'

    def decide(self,owner,candidate,evidence):
        if self.task.signing_layer!='explicit_rule_owners_v1':raise ValueError('RULE_SIGNING_NOT_REGISTERED')
        if owner not in self.task.required_signers:raise ValueError('UNAUTHORIZED_OWNER')
        decision=self._decision(owner,candidate,evidence)
        sequence=len(self.issued)+1
        raw={'owner':owner,'policy_hash':compile_candidate(self.task,candidate).policy_hash,'decision':decision,
             'source':'rule','sequence':sequence,'episode':self.ledger.episode}
        self.ledger.reserve(f'owner-message:{sequence}',ResourceVector(messages=1),purpose='explicit_owner_decision')
        path=self.output/f'{sequence:05d}-raw.json';durable_json(path,raw)
        body=dict(episode_id=self.ledger.episode,owner=owner,policy_hash=raw['policy_hash'],task_hash=self.task.task_hash,
            preference_hash=self.task.preference_hash,acceptance_rule_hash=self.task.acceptance_rule_hash,
            decision=decision,source='rule',decision_rule=self.roles[owner].decision_rule,
            evidence_ids=[r.evidence_id for r in evidence],
            information_receipt_ids=list(self.information.visible[owner].values()),raw_decision_ref=str(path.resolve()),sequence=sequence)
        receipt=AssentReceipt(receipt_id=research_hash(body),**body)
        durable_json(self.output/(receipt.receipt_id+'.json'),receipt.model_dump(mode='json'))
        self.issued[receipt.receipt_id]=receipt
        return receipt

    def verify(self,receipt,candidate,evidence):
        try:
            receipt=AssentReceipt.model_validate(receipt)
            if self.issued.get(receipt.receipt_id)!=receipt or receipt.owner not in self.task.required_signers:return False
            if AssentReceipt.model_validate_json((self.output/(receipt.receipt_id+'.json')).read_text())!=receipt:return False
            raw=strict_json_loads(Path(receipt.raw_decision_ref).read_text())
            return (receipt.source=='rule' and receipt.decision==self._decision(receipt.owner,candidate,evidence)
                and receipt.evidence_ids==[r.evidence_id for r in evidence]
                and raw=={'owner':receipt.owner,'policy_hash':receipt.policy_hash,'decision':receipt.decision,
                    'source':receipt.source,'sequence':receipt.sequence,'episode':receipt.episode_id}
                and all(self.information.verify(i,receipt.owner) for i in receipt.information_receipt_ids)
                and self.information.visible[receipt.owner][self.roles[receipt.owner].preference_card_id] in receipt.information_receipt_ids)
        except (ValueError,OSError,KeyError):return False
