"""Whole-job recovery with immutable attempts and an atomic batch visibility barrier."""
from __future__ import annotations

import multiprocessing
import fcntl
import os
from pathlib import Path
import signal
import time
import traceback
from typing import Protocol
import uuid

from pydantic import Field
from core.experiments.strict_types import StrictResearchModel,strict_json_loads
from core.experiments.design_config import ResearchRunSpec
from core.experiments.identity import research_hash,file_sha256
from core.experiments.archive import durable_json,verify_archive_v3


class ReservationPort(Protocol):
    def reserve_run(self,run_key,replication_id,*,plan_hash=None): ...
    def reserve_attempt(self,run_key,attempt_id): ...
    def publish(self,transaction_id,run_keys): ...
    def snapshot(self): ...
    def record_usage(self,transaction_id,usage): ...


class Job(StrictResearchModel):
    job_id:str=Field(pattern=r'^[A-Za-z0-9_-]+$')
    spec:ResearchRunSpec


def reserve_job(job:Job,reservation_port,*,plan_hash=None):
    from core.experiments.counterfactual import raw_cache_identity
    identity=raw_cache_identity(job.spec)
    receipt=reservation_port.reserve_run(identity['run_key'],job.spec.random.replication_id,plan_hash=plan_hash)
    return identity,receipt


def commit_attempt(attempt_path,*,archive,run_key,cpu_seconds):
    from core.experiments.counterfactual import raw_cache_identity
    root=Path(attempt_path);archive=Path(archive)
    if (root/'COMMITTED.json').exists():raise ValueError('Attempt commits are immutable')
    reservation=strict_json_loads((root/'reservation.json').read_text())
    if reservation['details']['run_key']!=run_key:raise ValueError('Attempt reservation scope mismatch')
    if not archive.resolve().is_relative_to(root.resolve()):raise ValueError('Attempt cannot commit another attempt archive')
    verified=verify_archive_v3(archive)
    spec=ResearchRunSpec.model_validate_json((archive/'research_spec.json').read_text())
    if raw_cache_identity(spec)['run_key']!=run_key:raise ValueError('Attempt full execution identity mismatch')
    record={'run_key':run_key,'archive':str(archive),'manifest_sha256':file_sha256(archive/'archive_manifest.json'),
            'verification':verified,'cpu_seconds':cpu_seconds}
    durable_json(root/'COMMITTED.json',record);return record


def run_job(job:Job,output,*,reservation_port,plan_hash=None):
    return JobCoordinator(Path(output),reservation_port).execute([job],workers=1,plan_hash=plan_hash)


def resume_incomplete_jobs(output,*,reservation_port,workers=1):
    root=Path(output);contract=strict_json_loads((root/'batch_input.json').read_text())
    jobs=[Job.model_validate(row) for row in contract['jobs']]
    return JobCoordinator(root,reservation_port).execute(jobs,workers=workers,plan_hash=contract['plan_hash'])


def _worker(spec_data,attempt_path,run_key,kill_after_step):
    root=Path(attempt_path);started=time.process_time()
    durable_json(root/'worker_started.json',{'pid':os.getpid(),'run_key':run_key})
    try:
        if kill_after_step is not None:
            # Explicit infrastructure-failure fixture, never a scientific spec.
            from core.experiments.stepper import DiagnosticJournal
            original=DiagnosticJournal.append
            def fault(self,stream,record):
                value=original(self,stream,record)
                if stream=='step_commits' and record.get('step')==kill_after_step:
                    os.kill(os.getpid(),signal.SIGKILL)
                return value
            DiagnosticJournal.append=fault
        from core.experiments.runner import ExperimentRunner
        spec=ResearchRunSpec.model_validate(spec_data)
        runner=ExperimentRunner(spec.base.to_legacy(),research_spec=spec)
        archive=runner.run(root/'runs')
        commit_attempt(root,archive=archive,run_key=run_key,cpu_seconds=time.process_time()-started)
    except BaseException as exc:
        durable_json(root/'FAILED.json',{'run_key':run_key,'error':type(exc).__name__,'reason':str(exc),
            'traceback':traceback.format_exc(),'cpu_seconds':time.process_time()-started})
        raise


class JobCoordinator:
    def __init__(self,root:Path,reservation_port:ReservationPort):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.port=reservation_port

    def execute(self,jobs:list[Job],*,workers=1,plan_hash=None,faults=None,cache=None):
        with (self.root/'coordinator.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise ValueError('COORDINATOR_ALREADY_RUNNING') from exc
            try:return self._execute(jobs,workers=workers,plan_hash=plan_hash,faults=faults,cache=cache)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def _execute(self,jobs,*,workers,plan_hash,faults,cache):
        from core.experiments.counterfactual import raw_cache_identity
        if type(workers) is not int or workers<1 or not jobs or len({j.job_id for j in jobs})!=len(jobs):
            raise ValueError('Invalid worker count or duplicate/empty jobs')
        faults=faults or {};cache=cache or {}
        if plan_hash is not None and (cache or faults):raise ValueError('Confirmation jobs require fresh executions without injected faults')
        contract={'jobs':[j.model_dump(mode='json') for j in jobs],'plan_hash':plan_hash}
        contract_path=self.root/'batch_input.json'
        if contract_path.exists():
            if strict_json_loads(contract_path.read_text())!=contract:raise ValueError('RECOVERY_INPUT_CHANGED')
        else:durable_json(contract_path,contract)
        published=self.root/'VISIBLE.json'
        if published.exists():
            prior=strict_json_loads(published.read_text())
            for row in prior['results']:verify_archive_v3(row['archive'])
            return prior
        states=[];pending=[]
        for job in jobs:
            identity,reservation=reserve_job(job,self.port,plan_hash=plan_hash)
            key=identity['run_key'];root=self.root/job.job_id
            root.mkdir(exist_ok=True)
            input_path=root/'input.json'
            if input_path.exists():
                if strict_json_loads(input_path.read_text())!=identity:raise ValueError('RECOVERY_EXECUTION_IDENTITY_CHANGED')
            else:durable_json(input_path,identity)
            history=[];committed=None
            for attempt in sorted(root.glob('attempt-*')):
                marker=attempt/'COMMITTED.json'
                if marker.exists():
                    record=strict_json_loads(marker.read_text());verify_archive_v3(record['archive'])
                    if record['run_key']!=key or file_sha256(Path(record['archive'])/'archive_manifest.json')!=record['manifest_sha256']:
                        raise ValueError('ATTEMPT_COMMIT_MISMATCH')
                    committed=record;state='COMMITTED'
                elif (attempt/'FAILED.json').exists():state='FAILED'
                else:
                    started=attempt/'worker_started.json'
                    if started.exists():
                        pid=strict_json_loads(started.read_text())['pid']
                        try:os.kill(pid,0)
                        except ProcessLookupError:pass
                        else:raise ValueError('PRIOR_WORKER_STILL_ALIVE; recovery cannot duplicate a running attempt')
                    state='INTERRUPTED'
                history.append({'attempt':str(attempt),'status':state})
            state={'job_id':job.job_id,'run_key':key,'history':history,'status':'PLANNED'}
            if committed:
                state.update(status='COMMITTED',**{k:v for k,v in committed.items() if k!='run_key'})
            elif key in cache:
                archive=Path(cache[key]);verify_archive_v3(archive)
                actual=ResearchRunSpec.model_validate_json((archive/'research_spec.json').read_text())
                if raw_cache_identity(actual)!=identity:raise ValueError('CACHE_SCOPE_MISMATCH')
                archived_source=strict_json_loads((archive/'archive_manifest.json').read_text())['source_hash']
                if archived_source!=identity['body']['source_hash'] or strict_json_loads((archive/'environment.json').read_text())!=identity['body']['environment']:
                    raise ValueError('CACHE_SOURCE_ENVIRONMENT_MISMATCH')
                state.update(status='COMMITTED',archive=str(archive),cache_hit=True)
            else:
                attempt_id=f'{job.job_id}-{uuid.uuid4().hex}'
                reservation=self.port.reserve_attempt(key,attempt_id)
                attempt=root/f'attempt-{len(history)+1:04d}';attempt.mkdir()
                durable_json(attempt/'reservation.json',reservation)
                state.update(status='RESERVED',attempt=str(attempt),attempt_id=attempt_id)
                pending.append((job,state))
            states.append(state)
        ctx=multiprocessing.get_context('spawn');running=[]
        while pending or running:
            while pending and len(running)<workers:
                job,state=pending.pop(0)
                process=ctx.Process(target=_worker,args=(job.spec.model_dump(mode='json'),state['attempt'],
                    state['run_key'],faults.get(job.job_id)))
                process.start();state.update(status='RUNNING',pid=process.pid);running.append((process,state))
            for process,state in list(running):
                if time.time()>=self.port.snapshot()['deadline'] and process.is_alive():process.kill()
                if process.is_alive():continue
                process.join();running.remove((process,state));root=Path(state['attempt'])
                if (root/'COMMITTED.json').exists() and process.exitcode==0:
                    record=strict_json_loads((root/'COMMITTED.json').read_text());verify_archive_v3(record['archive'])
                    state.update(status='COMMITTED',**{k:v for k,v in record.items() if k!='run_key'})
                else:state['status']='FAILED' if (root/'FAILED.json').exists() else 'INTERRUPTED'
                state['exit_code']=process.exitcode
                # Timing is observed work, separate from the reservation cap. A
                # killed worker has unknown CPU time, never zero by inference.
                try:self.port.record_usage(state['attempt_id'],{'run_key':state['run_key'],
                    'cpu_seconds':state.get('cpu_seconds'),'exit_code':process.exitcode})
                except ValueError as exc:
                    if type(exc).__name__!='BudgetExceeded' or str(exc)!='WALL_DEADLINE_EXCEEDED':raise
                    durable_json(root/'usage_after_deadline.json',{'cpu_seconds':state.get('cpu_seconds'),'exit_code':process.exitcode})
            if running:time.sleep(0.02)
        report={'status':'COMMITTED' if all(s['status']=='COMMITTED' for s in states) else 'INCOMPLETE',
                'input_hash':research_hash(contract),'results':states,'visibility':'whole_batch_barrier'}
        attempt_report=self.root/f'batch-attempt-{len(list(self.root.glob("batch-attempt-*.json")))+1:04d}.json'
        durable_json(attempt_report,report)
        if report['status']=='COMMITTED':
            self.port.publish('batch:'+str(self.root.resolve()),[s['run_key'] for s in states])
            durable_json(published,report)
        return report
