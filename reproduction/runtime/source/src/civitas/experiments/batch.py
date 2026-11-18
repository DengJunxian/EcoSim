"""Bounded offline subprocesses with append-only attempts and conservative quotas."""
import asyncio
from pathlib import Path
import sys
from typing import Literal

from pydantic import Field, model_validator
from civitas.cognition.budget import BudgetLimits
from civitas.contracts.base import NonNegative, Positive, Record, content_hash, require_unique
from civitas.scenarios.society_config import MarketSocietyConfig
from civitas.scenarios.kdd27_config import KDD27Config, RQ1Config, RQ2Config, PaperCognitionConfig, RQ3Config, parse_paper_config
from civitas.storage.commit_store import atomic_json, read_json
from . import registry


class BatchSpec(Record):
    schema_version: Literal[1] = 1
    configs: list[MarketSocietyConfig] = Field(min_length=1)
    seeds: list[NonNegative] = Field(min_length=1)
    concurrency: Positive = Field(default=1,le=8)
    max_run_attempts: NonNegative = 1000
    budget: BudgetLimits = Field(default_factory=lambda:BudgetLimits(usd_nano=0,cny_nano=0,max_attempts=10000,max_tokens=2_000_000_000))

    @model_validator(mode='after')
    def offline(self):
        require_unique(self.seeds,'batch seed')
        if self.budget.usd_nano or self.budget.cny_nano:
            raise ValueError('WP08 batch has zero monetary budget')
        return self


class PaperBatchSpec(BatchSpec):
    schema_version: Literal[2] = 2
    configs: list[KDD27Config] = Field(min_length=1)


class RQ1BatchSpec(BatchSpec):
    schema_version: Literal[3] = 3
    configs: list[RQ1Config] = Field(min_length=1)


class RQ2BatchSpec(BatchSpec):
    schema_version: Literal[4] = 4
    configs: list[RQ2Config] = Field(min_length=1)


class PaperCognitionBatchSpec(BatchSpec):
    schema_version: Literal[5] = 5
    configs: list[PaperCognitionConfig] = Field(min_length=1)


class RQ3BatchSpec(BatchSpec):
    schema_version: Literal[6] = 6
    configs: list[RQ3Config] = Field(min_length=1)


def parse_batch_spec(value):
    raw=value.model_dump(mode='json') if isinstance(value,Record) else value
    model={2:PaperBatchSpec,3:RQ1BatchSpec,4:RQ2BatchSpec,5:PaperCognitionBatchSpec,6:RQ3BatchSpec}.get(raw.get('schema_version'),BatchSpec) if isinstance(raw,dict) else BatchSpec
    return model.model_validate(raw)


def call_bound(config):
    if config.cognition_mode=='rule':
        return 0
    if config.scenario=='kdd27_v1':
        # Reserve both allowed attempts, assuming every eligible actor is active.
        return 2*(config.n_agents*config.ticks if config.policy_backend=='llm' else
                  config.slow_agents*(config.ticks//config.slow_period))
    return 4*config.slow_agents*config.ticks


def seeded(config, seed):
    raw = config.model_dump(mode='json')
    if config.scenario=='kdd27_v1':
        raw['root_seed']=seed
        return parse_paper_config(raw)
    if config.backend=='legacy_float64':
        raw['market']['experiment']['seed'] = seed
    else:
        raw['market']['root_seed'] = seed
    raw['behavior']['root_seed'] = seed
    return MarketSocietyConfig.model_validate(raw)


def manifest(root, spec):
    jobs, seen = [], set()
    for index, config in enumerate(spec.configs):
        for seed in spec.seeds:
            value = seeded(config,seed)
            key = content_hash(value)
            duplicate = key in seen
            seen.add(key)
            jobs.append({'job_id':f'config-{index:03d}-seed-{seed}','config_hash':key,'seed':seed,
                         'status':'skipped' if duplicate else 'planned','reason':'duplicate config/seed' if duplicate else None,
                         'history':[]})
    by_id = {j['job_id']:j for j in jobs}
    attempts, tokens, calls = 0, 0, 0
    for row in registry.history(root):
        job = by_id[row['job_id']]
        if row['kind']=='started':
            attempts += 1
            calls += row['allocation']['max_attempts']
            tokens += row['allocation']['max_tokens']
            job['history'].append(row)
            job.update(status='cancelled',reason='coordinator ended without an exit receipt')
        elif row['kind']=='finished':
            job['history'].append(row)
            job.update(status=row['status'],reason=row['reason'])
        elif row['kind']=='blocked':
            job.update(status='blocked',reason=row['reason'])
    return {'schema_version':spec.schema_version,'jobs':jobs,'run_attempts':attempts,'reserved_calls':calls,'reserved_tokens':tokens,
            'concurrency':spec.concurrency,'budget':spec.budget.model_dump(mode='json')}


async def run_batch(root, *, spec=None):
    root = Path(root).resolve()
    if spec is not None:
        spec = parse_batch_spec(spec)
        root.mkdir(parents=True,exist_ok=False)
        atomic_json(root/'spec.json',spec.model_dump(mode='json'))
    else:
        spec = parse_batch_spec(read_json(root/'spec.json'))
    with registry.run_lock(root):
        report = manifest(root,spec)
        slots = asyncio.Semaphore(spec.concurrency)
        quotas = asyncio.Lock()

        async def execute(job):
            if job['status'] in ('completed','skipped'):
                return
            config_index = int(job['job_id'].split('-')[1])
            config = seeded(spec.configs[config_index],job['seed'])
            calls = call_bound(config)
            tokens = calls*202048
            async with slots:
                async with quotas:
                    current = manifest(root,spec)
                    reason = None
                    if config.cognition_mode not in ('rule','synthetic_test'):
                        reason = 'online/archived input is not authorized for a new offline batch'
                    elif (current['run_attempts']>=spec.max_run_attempts or
                          current['reserved_calls']+calls>spec.budget.max_attempts or
                          current['reserved_tokens']+tokens>spec.budget.max_tokens):
                        reason = 'declared cumulative batch budget exhausted'
                    if reason:
                        registry.record(root,{'kind':'blocked','job_id':job['job_id'],'reason':reason})
                        return
                    allocation = BudgetLimits(usd_nano=0,cny_nano=0,max_attempts=calls,max_tokens=tokens)
                    folder = root/job['job_id']
                    folder.mkdir(exist_ok=True)
                    config_path, budget_path = folder/'config.json',folder/'budget.json'
                    if not config_path.exists():
                        atomic_json(config_path,config.model_dump(mode='json'))
                        atomic_json(budget_path,allocation.model_dump(mode='json'))
                    elif read_json(config_path)!=config.model_dump(mode='json') or read_json(budget_path)!=allocation.model_dump(mode='json'):
                        raise ValueError('Batch child inputs changed')
                    destination = folder/'run'
                    command = ([sys.executable,'-m','civitas','resume',str(destination)] if destination.exists() else
                        [sys.executable,'-m','civitas','run',str(config_path),'--durable','--output',str(destination),
                         '--budget-limits',str(budget_path)])
                    number = len([r for r in registry.history(root) if r.get('job_id')==job['job_id'] and r['kind']=='started'])+1
                    prefix = folder/f'attempt-{number:04d}'
                    registry.record(root,{'kind':'started','job_id':job['job_id'],'attempt':number,
                        'command':command,'allocation':allocation.model_dump(mode='json'),
                        'stdout':str(prefix)+'.stdout.txt','stderr':str(prefix)+'.stderr.txt'})
                process, code, status, reason = None, None, 'failed', None
                try:
                    with Path(str(prefix)+'.stdout.txt').open('xb') as out, Path(str(prefix)+'.stderr.txt').open('xb') as err:
                        process = await asyncio.create_subprocess_exec(*command,stdout=out,stderr=err)
                        code = await process.wait()
                    status = 'completed' if code==0 else ('cancelled' if code<0 else 'failed')
                    reason = None if code==0 else 'child process failed; retained stderr and run diagnostics'
                except BaseException as exc:
                    status = 'cancelled' if isinstance(exc,(KeyboardInterrupt,asyncio.CancelledError)) else 'failed'
                    reason = type(exc).__name__
                    if process is not None and process.returncode is None:
                        process.terminate()
                        code = await process.wait()
                    raise
                finally:
                    registry.record(root,{'kind':'finished','job_id':job['job_id'],'attempt':number,
                        'status':status,'reason':reason,'command':command,'exit_code':code})

        tasks = [asyncio.create_task(execute(job)) for job in report['jobs']]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            raise
        finally:
            report = manifest(root,spec)
            report['status'] = 'completed' if all(j['status'] in ('completed','skipped') for j in report['jobs']) else 'incomplete'
            atomic_json(root/'manifest.json',report)
        return report
