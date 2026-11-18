"""Composition root for shared finite design episodes; no implicit live calls."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import time

from core.experiments.archive import durable_json
from core.experiments.identity import research_hash, build_implementation_manifest, file_sha256
from core.experiments.research_cli import local_import_closure, ROOT
from .schema import TaskSpec, EpisodeResult
from .budget import BudgetLedger, BudgetExceeded, ResourceVector
from .verifier import ExactEvidenceService, compile_candidate, evaluate_eligibility, certify_agreement, select_candidate
from .services import InformationService, RuleOwnerService
from core.interventions.compiler import PolicyCompileError


def design_implementation_manifest():
    roots=[str(p.relative_to(ROOT)) for p in (ROOT/'core/codesign').glob('*.py')]
    roots+=['core/optimization/research_search.py']
    return build_implementation_manifest(ROOT,local_import_closure(roots)+['requirements-cscwd.lock'])


def freeze_design_source(output):
    output=Path(output);manifest=design_implementation_manifest()
    for entry in manifest.files:
        target=output/'source_bundle'/entry.path;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/entry.path,target)
    durable_json(output/'implementation_manifest.json',manifest.model_dump(mode='json'))
    return manifest


class EpisodePort:
    """The method sees only this restricted view and explicitly acquired information."""
    def __init__(self,task,ledger,evidence,information,owners,output):
        self._task,self.ledger,self._evidence,self._information,self._owners,self.output=task,ledger,evidence,information,owners,Path(output)
        self.candidate_ids=[c.candidate_id for c in task.candidate_pool]
        self._candidates={c.candidate_id:c for c in task.candidate_pool}
        self.fixed_rule_candidate_id=task.fixed_rule_candidate_id
        self.noop_candidate_id=next((c.candidate_id for c in task.candidate_pool if c.policy.halt is None and c.policy.liquidity is None),None)
        self.objective_ids=[r.role_id for r in task.roles];self.origin=task.origin
        self.submissions=[];self.evaluated_ids=[];self.evidence_by_candidate={};self.eligibility={};self.certificates={}
        self.assents={};self.raw_claims=[];self.protocol_raw_assents=[];self.method_claims=[];self.failures=[]

    def can_submit(self,count):
        state=self.ledger.snapshot()
        return time.time()<state['deadline'] and state['used']['candidate_submissions']+count<=self.ledger.spec.candidate_submissions

    def can_round(self):
        state=self.ledger.snapshot()
        return self.can_submit(1) and state['used']['rounds']<self.ledger.spec.rounds

    def reserve_round(self):
        sequence=self.ledger.snapshot()['used']['rounds']+1
        self.ledger.reserve(f'round:{sequence}',ResourceVector(rounds=1),purpose='method_round')

    def acquire_shared_information(self):
        for card_id in self._task.method_queryable_information_ids:self.query_information(card_id)

    def query_information(self,card_id):return self._information.query('designer',card_id)

    def protocol_session(self,*,mode=None,replay_catalog=None,local_controller=None):
        if hasattr(self,'replay_bundle'):
            mode=mode or 'fixed_pool_replay';replay_catalog=replay_catalog or self.replay_bundle.catalog
        mode=mode or 'analytic_fixture'
        return ProtocolSession(self,mode=mode,replay_catalog=replay_catalog,local_controller=local_controller)

    def prompt(self,mode):
        return json.dumps({'task_id':self._task.task_id,'mode':mode,'candidates':[
            c.model_dump(mode='json') for c in self._task.candidate_pool],
            'initial_cards':self._information.view('designer'),'queryable_ids':self._task.method_queryable_information_ids,
            'action_schema':{'action':'propose|query_information|stop','candidate_id':'string or null',
                'information_id':'string or null','claimed_agreement':'boolean','rationale':'string'},
            'instructions':'Use registered candidate IDs. Agreement requires explicit owner decisions; eligibility alone is not agreement.'},ensure_ascii=False)

    def observe_raw_claim(self,result,raw_ref):
        # Called before candidate or policy semantic validation, including invalid IDs.
        raw={'raw_ref':raw_ref,'action':result.model_dump(mode='json')}
        durable_json(self.output/f'raw-claim-{len(self.raw_claims)+1:05d}.json',raw)
        self.raw_claims.append(raw)

    def submit(self,candidate_id,*,raw_ref=None,claimed_agreement=False):
        sequence=len(self.submissions)+1;submission_id=f'submission-{sequence:05d}'
        reservation=self.ledger.reserve(submission_id,ResourceVector(candidate_submissions=1),purpose='candidate_submission')
        raw={'candidate_id':candidate_id,'raw_ref':raw_ref,'claimed_agreement':claimed_agreement,
             'submission_id':submission_id,'reservation':reservation}
        path=self.output/(submission_id+'-raw.json');durable_json(path,raw)
        self.submissions.append(raw)
        feedback={'submission_id':submission_id,'candidate_id':candidate_id,'utilities':None,
                  'constraint_violation':1.0,'status':'INVALID_CANDIDATE'}
        candidate=self._candidates.get(candidate_id)
        if candidate is None:
            self.failures.append({'kind':'INVALID_CANDIDATE','candidate_id':candidate_id,'raw_ref':str(path)})
            durable_json(self.output/(submission_id+'-evaluation.json'),feedback);return feedback
        if candidate_id not in self.evaluated_ids:
            self.evaluated_ids.append(candidate_id)
            try:records=self._evidence.issue(candidate)
            except PolicyCompileError:
                records=[]
            self.evidence_by_candidate[candidate_id]=records
            self.eligibility[candidate_id]=evaluate_eligibility(self._task,candidate,records,evidence_service=self._evidence)
        records=self.evidence_by_candidate[candidate_id];eligibility=self.eligibility[candidate_id]
        feedback['status']=eligibility['status']
        if records:
            # An optimizer needs all objectives. It must first acquire the same
            # role cards through the common port, never read hidden TaskSpec weights.
            visible=self._information.visible['designer']
            if all(r.preference_card_id in visible for r in self._task.roles):
                feedback['utilities']={r.role_id:min(x.interval[0] for x in records if x.atom_id.endswith(':utility:'+r.role_id)) for r in self._task.roles}
                feedback['constraint_violation']=max([0.0]+[v['interval'][1]-v['threshold'] for v in eligibility['constraints'].values()])
        if claimed_agreement and eligibility['status']=='SUPPORTED':
            self.method_claims.append(eligibility['policy_hash'])
        if (eligibility['status']=='SUPPORTED' and self._task.signing_layer=='explicit_rule_owners_v1'
                and candidate_id not in self.certificates):
            decisions=[self._owners.decide(owner,candidate,records) for owner in self._task.required_signers]
            self.assents[candidate_id]=decisions
            if all(r.decision=='ASSENT' for r in decisions):self.method_claims.append(eligibility['policy_hash'])
            self.certificates[candidate_id]=certify_agreement(self._task,candidate,records,decisions,
                evidence_service=self._evidence,assent_service=self._owners)
        durable_json(self.output/(submission_id+'-evaluation.json'),{'feedback':feedback,'eligibility':eligibility,
            'evidence_ids':[r.evidence_id for r in records],'certificate':self.certificates.get(candidate_id),
            'duplicate':sum(x['candidate_id']==candidate_id for x in self.submissions)>1})
        return feedback


def run_design_episode(task:TaskSpec,method,output,*,episode_id,method_config=None,replay_bundle=None):
    task=TaskSpec.model_validate(task);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    if hasattr(method,'configuration'):method_config={**(method_config or {}),'treatment':method.configuration}
    started=time.time();manifest=freeze_design_source(output)
    durable_json(output/'task.json',task.model_dump(mode='json'))
    if task.evidence_mode!='exact_analytic':raise ValueError('Financial episodes require a scoped gate and registered evaluator; diagnostic LOB runs use the benchmark diagnostic port')
    ledger=BudgetLedger(output/'budget.json',task.budget,owner=method.name,episode=episode_id)
    if replay_bundle is None:evidence=ExactEvidenceService(task,manifest.source_hash,output/'evidence')
    else:
        if replay_bundle.task_hash!=task.task_hash or replay_bundle.source_hash!=manifest.source_hash:
            raise ValueError('REPLAY_TASK_OR_SOURCE_SCOPE_MISMATCH')
        replay_bundle.verify();evidence=replay_bundle.service
        durable_json(output/'external_evidence_references.json',replay_bundle.references)
    information=InformationService(task,ledger,output/'information')
    if task.signing_layer=='delegated_explicit_llm_v1':
        from .live_protocol import LLMOwnerService
        owners=LLMOwnerService(task,ledger,information,evidence,output/'assents')
    else:owners=RuleOwnerService(task,ledger,information,evidence,output/'assents')
    port=EpisodePort(task,ledger,evidence,information,owners,output)
    if replay_bundle is not None:port.replay_bundle=replay_bundle
    failed=False;budget_stopped=False
    try:
        if hasattr(method,'bind'):method.bind(ledger,output/'llm')
        if getattr(method,'client',None) and method.client.config not in task.model_configurations:
            raise ValueError('MODEL_CONFIGURATION_NOT_REGISTERED_IN_TASK')
        if hasattr(method,'run_async'):method_receipt=asyncio.run(method.run_async(port))
        else:method_receipt=method.run(port)
    except BudgetExceeded as exc:
        budget_stopped=True;method_receipt={'status':'BUDGET_STOPPED','reason':str(exc)};port.failures.append(method_receipt)
    except (ValueError,RuntimeError,OSError) as exc:
        failed=True;method_receipt={'status':'METHOD_FAILED','error_type':type(exc).__name__,'reason':str(exc)};port.failures.append(method_receipt)
    durable_json(output/'method_receipt.json',method_receipt)
    eligible=[port._candidates[c] for c,e in port.eligibility.items() if e['status']=='SUPPORTED']
    certified=[c for c in eligible if port.certificates.get(c.candidate_id,{}).get('status')=='CERTIFIED']
    selectable=eligible if task.signing_layer=='candidate_only' else certified
    chosen=select_candidate(task,selectable,port.evidence_by_candidate)
    all_excluded=(set(port.evaluated_ids)==set(port.candidate_ids) and all(e['status']=='REFUTED' for e in port.eligibility.values()))
    terminal=('METHOD_FAILED' if failed else 'CANDIDATE_ONLY' if chosen and task.signing_layer=='candidate_only' else
        'SELECTED' if chosen else 'NO_ACCEPTABLE_IN_FINITE_POOL' if all_excluded else
        'BUDGET_UNRESOLVED' if budget_stopped or not port.can_submit(1) else 'SHORTLIST')
    if hasattr(port,'protocol_state') and not failed:terminal=port.protocol_state.terminal
    def hashes(candidates):return [compile_candidate(task,c).policy_hash for c in candidates]
    raw_hashes=[]
    for claim in port.raw_claims:
        if claim['action']['claimed_agreement']:
            candidate=port._candidates.get(claim['action']['candidate_id'])
            if candidate:
                # Raw claim identity can remain an uncompiled syntax hash if illegal.
                try:raw_hashes.append(compile_candidate(task,candidate).policy_hash)
                except ValueError:raw_hashes.append(research_hash(candidate.policy))
    # Individual raw owner declarations are retained before permissions and
    # local guards. This set is not a claim that every required owner signed.
    import re
    raw_hashes.extend(a['policy_hash'] for a in port.protocol_raw_assents
        if isinstance(a['policy_hash'],str) and re.fullmatch('[0-9a-f]{64}',a['policy_hash']))
    durable_json(output/'raw-assent-index.json',port.protocol_raw_assents)
    rule=not hasattr(method,'client')
    client=getattr(method,'client',None)
    received=[] if client is None else [c for c in client.calls if c.get('response_ref')]
    invalid=sum(f.get('reason')=='INVALID_RESPONSE' for f in port.failures)
    result=EpisodeResult(episode_id=episode_id,task_id=task.task_id,underlying_cluster_id=task.underlying_cluster_id,
        task_hash=task.task_hash,method=method.name,method_config_hash=research_hash(method_config or {'name':method.name}),
        source_hash=manifest.source_hash,split=task.split,origin=task.origin,live_status='RUN' if client and client.config.source=='live' and client.calls else 'NOT_RUN',
        candidate_refs=[str(p.resolve()) for p in sorted(output.glob('submission-*-raw.json'))]+
            [s['raw_ref'] for s in port.submissions if s.get('raw_ref') and str(output/'protocol') in s['raw_ref']],
        evidence_refs=[str(evidence.issued[i].resolve()) for i in sorted(
            {r.evidence_id for records in port.evidence_by_candidate.values() for r in records} |
            set(port.protocol_state.evidence_ids if hasattr(port,'protocol_state') else []))],
        message_refs=[str(p.resolve()) for pattern in ('*-raw.json','*-transport.json') for p in sorted((output/'protocol').glob(pattern))],A_raw=list(dict.fromkeys(raw_hashes)),
        A_method=list(dict.fromkeys(port.method_claims)),A_cert=hashes(certified),role_eligibility=port.eligibility,
        assent_refs=[str(p.resolve()) for p in sorted((output/'assents').glob('*.json'))],
        selected_policy_hash=compile_candidate(task,chosen).policy_hash if chosen else None,shortlist=hashes(eligible),
        terminal_state=terminal,budget_snapshot=ledger.snapshot(),failures=port.failures,
        language_error_rate=None if rule or not received else invalid/len(received),
        language_metric_status='NA_RULE' if rule else 'OBSERVED' if received else 'NO_RESPONSES',started_at=started,ended_at=time.time(),
        common_contract={'compiler':task.candidate_language,'utility':research_hash(task.utility),'information':research_hash(task.information_cards),
            'initial_information':list(task.method_initial_information_ids),'queryable_information':list(task.method_queryable_information_ids),
            'budget':research_hash(task.budget),'verifier':'scope-absolute-explicit-assent-v1','selection':task.final_selection,
            'underlying_n':0,'market_runs':0,'exact_evaluated_candidate_ids':port.evaluated_ids,'live_calls':0 if client is None else sum(1 for _ in client.calls if client.config.source=='live')})
    durable_json(output/'episode_result.json',result.model_dump(mode='json'))
    if design_implementation_manifest().source_hash!=manifest.source_hash:raise RuntimeError('IMPLEMENTATION_CHANGED_DURING_EPISODE')
    files={str(p.relative_to(output)):file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='COMMITTED.json'}
    durable_json(output/'COMMITTED.json',{'schema_version':1,'result_hash':research_hash(result),'files':files})
    return result


def verify_episode(path):
    from core.experiments.strict_types import strict_json_loads
    path=Path(path);marker=strict_json_loads((path/'COMMITTED.json').read_text())
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name!='COMMITTED.json'}
    if set(marker['files'])!=actual:raise ValueError('EPISODE_FILE_SET_MISMATCH')
    for relative,digest in marker['files'].items():
        target=(path/relative).resolve()
        if not target.is_relative_to(path.resolve()) or file_sha256(target)!=digest:raise ValueError('EPISODE_FILE_HASH_MISMATCH')
    if (path/'external_evidence_references.json').exists():
        refs=strict_json_loads((path/'external_evidence_references.json').read_text())
        for item in refs:
            if file_sha256(item['path'])!=item['sha256']:raise ValueError('EXTERNAL_EVIDENCE_HASH_MISMATCH')
    result=EpisodeResult.model_validate_json((path/'episode_result.json').read_text())
    if research_hash(result)!=marker['result_hash']:raise ValueError('EPISODE_RESULT_HASH_MISMATCH')
    return {'status':'INTEGRITY_VERIFIED','episode_id':result.episode_id,'origin':result.origin,
        'files':len(marker['files']),'terminal_state':result.terminal_state,'live_status':result.live_status}


class ProtocolSession:
    """Trusted composition boundary. Method callbacks receive finite operations only."""
    def __init__(self,port,*,mode='analytic_fixture',replay_catalog=None,local_controller=None):
        from .protocol import ProtocolState
        self.port=port;self.task=port._task;self.root=port.output/'protocol';self.root.mkdir(exist_ok=False)
        if mode=='online':raise ValueError('ONLINE_PROTOCOL_REQUIRES_SCOPED_FINANCIAL_GATE')
        if mode=='fixed_pool_replay' and replay_catalog is None:raise ValueError('FROZEN_REPLAY_CATALOG_REQUIRED')
        self.state=ProtocolState(episode_id=port.ledger.episode,task_hash=self.task.task_hash,mode=mode,
            visible_information={p:list(cards) for p,cards in port._information.visible.items()})
        durable_json(self.root/'initial-state.json',self.state.model_dump(mode='json'))
        self.replay_catalog=replay_catalog;self.evidence_records={};self.service_events={};self.transition_log=[]
        self.raw_count=0;self.pending=[];self.batch_count=0;self.stop_requested=False
        self.local_controller=local_controller

    def _services(self,auth):
        from .protocol import ProtocolServices
        snapshot=self.port.ledger.snapshot()
        return ProtocolServices(self.task,self.port._evidence.source_hash,self.port._evidence,self.port._owners,
            self.port._information,dict(self.evidence_records),dict(self.port._owners.issued),dict(self.service_events),
            {v['receipt_hash']:v for v in snapshot['transactions'].values()},snapshot,auth,self.replay_catalog)

    def emit(self,kind,payload,*,principal='designer',source='rule',raw_text='',operation_id=None,
             existing_reservation=None,wire_overrides=None,service_event_hash=None):
        from .schema import AgentMessage
        from .protocol import AuthenticatedInput,advance_protocol,ProtocolState
        self.raw_count+=1;operation_id=operation_id or f'op-{self.raw_count:05d}'
        transport_path=self.root/f'{self.raw_count:05d}-transport.json'
        durable_json(transport_path,{'type':kind,'payload':payload,'principal':principal,'source':source,
            'raw_text':raw_text,'operation_id':operation_id,'wire_overrides':wire_overrides})
        if kind=='DECLARE_ASSENT':
            self.port.protocol_raw_assents.append({'owner':payload.get('owner'),'policy_hash':payload.get('policy_hash'),
                'principal':principal,'source':source,'raw_ref':str(transport_path),'operation_id':operation_id})
        try:
            if self.state.round==0:self.port.reserve_round()
            reservation=existing_reservation or self.port.ledger.reserve('protocol-message:'+str(self.raw_count),ResourceVector(messages=1),purpose='protocol_message')
            if kind in {'PROPOSE','REVISE'}:
                self.port.ledger.reserve('protocol-candidate:'+str(self.raw_count),ResourceVector(candidate_submissions=1),purpose='protocol_candidate_attempt')
        except BudgetExceeded as exc:
            durable_json(self.root/f'{self.raw_count:05d}-budget-refusal.json',{'raw_ref':str(transport_path),
                'reason':str(exc),'budget_state':self.port.ledger.snapshot()})
            raise
        path=self.root/f'{self.raw_count:05d}-raw.json'
        request_hash=research_hash({'episode':self.state.episode_id,'principal':principal,'kind':kind,'operation_id':operation_id,'payload':payload})
        wire=dict(message_id=f'message-{self.raw_count:05d}',operation_id=operation_id,episode_id=self.state.episode_id,
            round=max(1,self.state.round),sequence=self.state.sequence+1,speaker=principal,type=kind,payload=payload,
            source=source,raw_ref=str(transport_path),raw_text=raw_text,request_hash=request_hash,budget_receipt_hash=reservation['receipt_hash'])
        if wire_overrides:wire.update(wire_overrides)
        durable_json(path,wire)  # Before schema, identity, semantics or method-local guards.
        auth=AuthenticatedInput(principal=principal,source=source,raw_ref=str(transport_path),request_hash=request_hash,
            budget_receipt_hash=reservation['receipt_hash'],service_event_hash=service_event_hash)
        service_view=self._services(auth)
        durable_json(self.root/f'{self.raw_count:05d}-context.json',{'auth':auth.model_dump(mode='json'),
            'budget_state':service_view.budget_state,'evidence_ids':list(service_view.evidence_records),
            'assent_ids':list(service_view.assent_records),'service_event_ids':list(service_view.service_events)})
        if kind in {'PROPOSE','REVISE'}:
            cid=payload.get('candidate',{}).get('candidate_id')
            self.port.submissions.append({'submission_id':operation_id,'candidate_id':cid,'raw_ref':str(path)})
        before_state=self.state;message=None
        try:
            message=AgentMessage.model_validate(wire)
            self.state,receipt,requests=advance_protocol(self.state,message,service_view)
        except ValueError as exc:
            receipt={'applied':False,'reason':'SCHEMA_REJECTED','detail':str(exc),'raw_ref':str(path),
                'before_hash':research_hash(self.state),'after_hash':research_hash(self.state)};requests=[]
        if self.local_controller is not None and message is not None:
            local=self.local_controller.observe(before_state,message,service_view,receipt,self.state)
            durable_json(self.root/f'{self.raw_count:05d}-local-treatment.json',local)
        self.transition_log.append(receipt);durable_json(self.root/f'{self.raw_count:05d}-transition.json',receipt)
        if not receipt['applied'] and kind in {'PROPOSE','REVISE'}:
            cid=payload.get('candidate',{}).get('candidate_id');c=self.port._candidates.get(cid)
            if c is not None:
                result=evaluate_eligibility(self.task,c,[],evidence_service=self.port._evidence)
                if result['reasons']==['STATIC_ILLEGAL']:
                    self.port.eligibility[cid]=result
                    if cid not in self.port.evaluated_ids:self.port.evaluated_ids.append(cid)
                    self.port.evidence_by_candidate[cid]=[]
        for request in requests:
            if request['action']=='evaluate_registered_candidate':self.pending.append(request)
            elif request['action']=='common_terminal_evaluation':self.stop_requested=True
            elif request['action']=='publish_information':
                self.port._information.publish(request['principal'],request['information_id'],request['recipients'],
                    request['source_receipt_id'],request['budget_receipt_hash'])
        durable_json(self.root/'state.json',self.state.model_dump(mode='json'))
        return receipt

    def propose(self,candidate_id,*,principal='designer',raw_text=''):
        c=self.port._candidates[candidate_id]
        return self.emit('PROPOSE',{'candidate':c.model_dump(mode='json')},principal=principal,raw_text=raw_text)

    def register_and_request(self,candidate_id,*,principal='designer'):
        from .schema import ClaimSpec
        from .verifier import required_atoms,scope_for
        c=self.port._candidates[candidate_id]
        if candidate_id not in self.state.candidate_aliases:return
        scope=scope_for(self.task,c,self.port._evidence.source_hash);claim_ids=[]
        for atom in required_atoms(self.task):
            claim_id=candidate_id+':'+atom.atom_id;claim_ids.append(claim_id)
            claim=ClaimSpec(claim_id=claim_id,kind=atom.kind,policy_hashes=scope.policy_hashes,model_id=atom.model_id,
                quantity_id=atom.quantity_id,direction=atom.direction,threshold=atom.threshold,scope=scope,
                evidence_mode='exact_analytic',registered_plan_hash=research_hash(self.port._evidence.plan),dependencies=[])
            self.emit('REGISTER_CLAIM',{'claim':claim.model_dump(mode='json')},principal=principal)
        self.emit('REQUEST_TEST',{'candidate_id':candidate_id,'claim_ids':claim_ids},principal=principal)

    def flush(self):
        """Evaluate the entire fixed queue before publishing any result."""
        from .schema import TestResultPayload
        if not self.pending:return
        queue=list(self.pending);self.pending=[];self.batch_count+=1;complete=[]
        for request in queue:
            cid=request['candidate_id'];candidate=self.port._candidates[cid]
            if self.state.mode=='fixed_pool_replay':
                records=self.replay_catalog[cid]
                if any(not self.port._evidence.verify(r,candidate) for r in records):raise ValueError('INVALID_REPLAY_EVIDENCE')
            else:records=self.port._evidence.issue(candidate)
            records=list(records)
            for claim_id in request['claim_ids']:
                claim=self.state.claims[claim_id]
                if claim.kind in {'paired_difference','static_feasible'}:
                    if self.state.mode=='fixed_pool_replay':
                        if not any(r.atom_id==claim.claim_id for r in records):raise ValueError('REPLAY_CANNOT_ADD_UNREGISTERED_CLAIM')
                    else:records.append(self.port._evidence.issue_claim(claim))
            complete.append((request,records))
        batch={'batch':self.batch_count,'visibility':'whole_batch_barrier','source_mode':self.state.mode,
            'requests':[r for r,_ in complete],'evidence_ids':[[r.evidence_id for r in records] for _,records in complete],
            'statistical_n':0 if self.task.origin=='analytic_fixture' else None}
        durable_json(self.root/f'batch-{self.batch_count:03d}.json',batch)
        for request,records in complete:
            cid=request['candidate_id'];self.evidence_records.update({r.evidence_id:r for r in records})
            payload=TestResultPayload(request_operation_id=request['operation_id'],candidate_id=cid,
                evidence_ids=[r.evidence_id for r in records],batch_receipt_hash=research_hash(batch))
            event_hash=research_hash(payload);self.service_events[event_hash]=payload.model_dump(mode='json')
            receipt=self.emit('TEST_RESULT',payload.model_dump(mode='json'),principal='evidence_service',service_event_hash=event_hash)
            if not receipt['applied']:raise ValueError('EVIDENCE_PUBLICATION_REJECTED: '+receipt['reason'])
            from .verifier import required_atoms
            base_ids={a.atom_id for a in required_atoms(self.task)}
            base_records=[r for r in records if r.atom_id in base_ids]
            self.port.evidence_by_candidate[cid]=base_records
            self.port.eligibility[cid]=evaluate_eligibility(self.task,self.port._candidates[cid],base_records,evidence_service=self.port._evidence)
            if cid not in self.port.evaluated_ids:self.port.evaluated_ids.append(cid)

    def explicit_decisions(self):
        for cid,eligibility in self.port.eligibility.items():
            if eligibility['status']!='SUPPORTED':continue
            candidate=self.port._candidates[cid];records=self.port.evidence_by_candidate[cid];assents=[]
            for owner in self.task.required_signers:
                receipt=self.port._owners.decide(owner,candidate,records);assents.append(receipt)
                reservation=self.port.ledger.snapshot()['transactions']['owner-message:'+str(receipt.sequence)]
                self.emit('DECLARE_ASSENT',{'owner':owner,'policy_hash':receipt.policy_hash,'receipt_id':receipt.receipt_id},
                    principal=owner,existing_reservation=reservation)
            self.port.assents[cid]=assents

    def finish(self,*,reason='completed_round'):
        from .protocol import ProtocolState
        if reason!='budget':
            if self.pending:self.flush()
            self.emit('STOP',{'reason':reason})
        self.port.certificates={};self.port.method_claims=[]
        for cid,records in self.port.evidence_by_candidate.items():
            c=self.port._candidates[cid]
            try:ph=compile_candidate(self.task,c).policy_hash
            except ValueError:continue
            commitments=[self.state.commitments.get(owner+':'+ph) for owner in self.task.required_signers]
            current=[commit for commit in commitments if commit and commit.current_status=='DECLARED']
            assents=[self.port._owners.issued[commit.explicit_assent_receipt_id] for commit in current]
            if len(current)==len(self.task.required_signers):self.port.method_claims.append(ph)
            self.port.certificates[cid]=certify_agreement(self.task,c,records,assents,
                evidence_service=self.port._evidence,assent_service=self.port._owners)
        certified=any(v['status']=='CERTIFIED' for v in self.port.certificates.values())
        excluded=(set(self.port.evaluated_ids)==set(self.port.candidate_ids)
            and all(e['status']=='REFUTED' for e in self.port.eligibility.values()))
        terminal=('AGREEMENT_CERTIFIED' if certified else 'NO_ACCEPTABLE_POLICY_IN_ENUMERATED_SET' if excluded else
                  'BUDGET_EXHAUSTED_UNRESOLVED' if reason=='budget' else 'SHORTLIST_WITH_DISSENT')
        data=self.state.model_dump(mode='json');data.update(terminal=terminal,terminal_reason=reason)
        self.state=ProtocolState.model_validate(data);self.port.protocol_state=self.state
        durable_json(self.root/'state.json',self.state.model_dump(mode='json'))
        durable_json(self.root/'transitions.json',self.transition_log)
        return {'protocol_terminal':terminal,'state_hash':research_hash(self.state),'transitions':len(self.transition_log),
            'rejected':sum(not r['applied'] for r in self.transition_log),'source':'rule','mode':self.state.mode,
            'natural_trace_status':'NOT_COLLECTED','statistical_n':0,'whole_batch_visibility':True}

    def replay_transitions(self):
        """Replay saved context without new reservations, messages, calls or worlds."""
        from dataclasses import replace
        from .protocol import ProtocolState,AuthenticatedInput,advance_protocol
        from .schema import AgentMessage
        from core.experiments.strict_types import strict_json_loads
        state=ProtocolState.model_validate_json((self.root/'initial-state.json').read_text())
        ledger_before=research_hash(self.port.ledger.snapshot());checks=[]
        for raw in sorted(self.root.glob('*-raw.json')):
            prefix=raw.name.split('-')[0]
            context=strict_json_loads((self.root/(prefix+'-context.json')).read_text())
            expected=strict_json_loads((self.root/(prefix+'-transition.json')).read_text())
            auth=AuthenticatedInput.model_validate(context['auth']);services=self._services(auth)
            services=replace(services,budget_state=context['budget_state'],
                budget_receipts={v['receipt_hash']:v for v in context['budget_state']['transactions'].values()},
                evidence_records={i:self.evidence_records[i] for i in context['evidence_ids']},
                assent_records={i:self.port._owners.issued[i] for i in context['assent_ids']},
                service_events={i:self.service_events[i] for i in context['service_event_ids']})
            try:
                message=AgentMessage.model_validate_json(raw.read_text())
                state,receipt,_=advance_protocol(state,message,services)
            except ValueError:
                if expected['reason']!='SCHEMA_REJECTED':raise
                receipt=expected
            for field in ('before_hash','after_hash','applied','reason'):
                if receipt[field]!=expected[field]:raise ValueError('REPLAY_TRANSITION_MISMATCH: '+field)
            checks.append({'raw_ref':str(raw),'state_hash':research_hash(state)})
        if research_hash(self.port.ledger.snapshot())!=ledger_before:raise ValueError('REPLAY_MUTATED_BUDGET')
        final=self.state.model_dump(mode='json');final.update(terminal=state.terminal,terminal_reason=state.terminal_reason)
        if ProtocolState.model_validate(final)!=state:raise ValueError('REPLAY_METHOD_STATE_MISMATCH')
        report={'status':'DETERMINISTIC_TRANSITIONS_VERIFIED','events':len(checks),'new_market_runs':0,'new_calls':0,
            'new_reservations':0,'checks':checks,'terminal_note':'terminal selection separately recomputed by common verifier'}
        durable_json(self.root/'replay-receipt.json',report);return report


class AnalyticReplayBundle:
    """One frozen finite family shared across episodes, not new online sampling."""
    def __init__(self,task,output):
        self.task_hash=task.task_hash;self.source_hash=design_implementation_manifest().source_hash
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.service=ExactEvidenceService(task,self.source_hash,self.output/'evidence');self.catalog={}
        for c in task.candidate_pool:
            try:self.catalog[c.candidate_id]=self.service.issue(c)
            except PolicyCompileError:continue
        index={'origin':'analytic_fixture','mode':'fixed_pool_replay','task_hash':self.task_hash,
            'source_hash':self.source_hash,'visibility_order':[c.candidate_id for c in task.candidate_pool if c.candidate_id in self.catalog],
            'evidence_by_candidate':{cid:[r.evidence_id for r in records] for cid,records in self.catalog.items()},
            'statistical_n':0,'alpha':None,'family_coverage':'complete_exact_finite_table'}
        durable_json(self.output/'replay-index.json',index)
        self.references=[{'path':str(p),'sha256':file_sha256(p)} for p in sorted(self.output.rglob('*')) if p.is_file()]
        durable_json(self.output/'FROZEN.json',{'references':self.references})

    def verify(self):
        for item in self.references:
            if file_sha256(item['path'])!=item['sha256']:raise ValueError('FROZEN_REPLAY_HASH_MISMATCH')
        return True
