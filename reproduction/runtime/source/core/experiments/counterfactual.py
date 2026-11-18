"""Independent Runner pairs with complete input identity and public-prefix checks."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field,model_validator
from core.experiments.strict_types import StrictResearchModel,load_research_model,strict_json_loads
from core.experiments.design_config import ResearchRunSpec
from core.experiments.identity import research_hash,scientific_projection,file_sha256
from core.experiments.archive import durable_json,rows,verify_archive_v3,resolve_run_reference


def execution_environment():
    return {'python':sys.version,'platform':platform.platform(),'backend':'civitas-python-lob-v1',
        'packages':{name:importlib.metadata.version(name) for name in ('numpy','networkx','pandas','pydantic','PyYAML')},
        'installed_packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}


def raw_cache_identity(spec:ResearchRunSpec):
    from core.experiments.research_cli import validate_research_spec,runtime_implementation_manifest
    from core.experiments.world import build_initial_world
    validate_research_spec(spec)
    source=runtime_implementation_manifest(spec).source_hash
    if spec.world.initialization_mode!='generated':
        raise ValueError('Batch adapter requires generated complete worlds; snapshot runs remain available through Runner')
    world=build_initial_world(spec,source_hash=source)
    body={'version':'complete-raw-cache-v1','source_hash':source,'environment':execution_environment(),
          'effective_spec':spec.model_dump(mode='json'),'initial_world_hash':world.world_hash}
    return {'run_key':research_hash(body),'body':body}


class PairPlan(StrictResearchModel):
    schema_version:Literal[1]=1
    purpose:Literal['engineering_identity','development','validation']
    baseline_spec:str
    treatment_spec:str
    replication_ids:list[str]=Field(min_length=1)

    @model_validator(mode='after')
    def _unique(self):
        if len(set(self.replication_ids))!=len(self.replication_ids) or any(not x for x in self.replication_ids):
            raise ValueError('Unique nonempty replication IDs required')
        return self


def with_replication(spec,replication_id,*,data_phase=None):
    value=spec.model_dump(mode='json');value['random']['replication_id']=replication_id
    if data_phase is not None:value['data_phase']=data_phase
    return ResearchRunSpec.model_validate(value)


def validate_pair(left,right):
    left=resolve_run_reference(left);right=resolve_run_reference(right)
    verified=[verify_archive_v3(p) for p in (left,right)]
    specs=[ResearchRunSpec.model_validate_json((p/'research_spec.json').read_text()) for p in (left,right)]
    a,b=[s.model_dump(mode='json') for s in specs]
    a.pop('policy');b.pop('policy')
    if a!=b:raise ValueError('PAIR_SPEC_SCOPE_MISMATCH: only the policy may differ')
    for name in ('environment.json','initial_world.json','implementation_manifest.json'):
        if strict_json_loads((left/name).read_text())!=strict_json_loads((right/name).read_text()):
            raise ValueError('PAIR_SCOPE_MISMATCH: '+name)
    if left==right:raise ValueError('PAIR_REQUIRES_INDEPENDENT_RUNNER_ARCHIVES')
    policy=[rows(p/'policy_events.jsonl') for p in (left,right)]
    # Configuration labels and empty bookkeeping do not constitute announcements.
    public=lambda row:row.get('announcements',[])
    first=next((i+1 for i,(x,y) in enumerate(zip(*policy)) if public(x)!=public(y)),None)
    # Public economic records before an announcement must coincide. Measurement
    # metadata includes assigned policy IDs and is removed only by this projection.
    def projection(row):
        if isinstance(row,dict):return {k:projection(v) for k,v in row.items()
            if k not in {'record_hash','run_id','policy_hash','observation_hash','measurement_record_hash'}}
        if isinstance(row,list):return [projection(v) for v in row]
        return row
    for name in ('market','measurements','accounts','social_state'):
        x=[projection(r) for r in rows(left/(name+'.jsonl')) if first is None or r['step']<first]
        y=[projection(r) for r in rows(right/(name+'.jsonl')) if first is None or r['step']<first]
        if x!=y:raise ValueError('PUBLIC_PREFIX_DIVERGED: '+name)
    draws=[]
    for root in (left,right):
        draws.append({research_hash(r['draw']['key']):r['draw'] for r in rows(root/'random.jsonl')})
    common=set(draws[0])&set(draws[1])
    if any(draws[0][key]!=draws[1][key] for key in common):raise ValueError('PAIR_DRAW_MISMATCH')
    noop=specs[0].policy.halt is None and specs[0].policy.liquidity is None and specs[0].policy==specs[1].policy
    if noop and verified[0]['result_content_hash']!=verified[1]['result_content_hash']:
        raise ValueError('NOOP_IDENTITY_FAILED')
    return {'status':'verified_pair','left':str(left),'right':str(right),
        'manifest_sha256':[file_sha256(p/'archive_manifest.json') for p in (left,right)],
        'replication_id':specs[0].random.replication_id,'first_public_policy_difference_step':first,
        'shared_draw_addresses':len(common),'draw_counts':[len(d) for d in draws],
        'noop_identity':noop,'scientific_eligibility':'NOT_EVALUATED'}


def run_pair(baseline,treatment,output,*,reservation_port,workers=1):
    from core.experiments.jobs import Job,JobCoordinator
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    jobs=[Job(job_id=side,spec=spec) for side,spec in [('baseline',baseline),('treatment',treatment)]]
    coordinator=JobCoordinator(output/'jobs',reservation_port)
    results=coordinator.execute(jobs,workers=workers)
    if results['status']!='COMMITTED':return results
    report=validate_pair(results['results'][0]['archive'],results['results'][1]['archive'])
    durable_json(output/'pair.json',report)
    return report


def paired_covariance(pairs:list[tuple[float,float]]):
    n=len(pairs)
    if n<2:return {'n':n,'covariance':None,'variance_difference':None,'reason':'NEED_AT_LEAST_TWO_COMPLETE_PAIRS'}
    ax=sum(x for x,y in pairs)/n;ay=sum(y for x,y in pairs)/n
    covariance=sum((x-ax)*(y-ay) for x,y in pairs)/(n-1)
    difference_variance=sum(((y-x)-(ay-ax))**2 for x,y in pairs)/(n-1)
    return {'n':n,'covariance':covariance,'variance_difference':difference_variance,
        'interpretation':'empirical pairing covariance; no universal variance-reduction claim'}


def evaluate_batch(plan:PairPlan,*,base_dir:Path,output:Path,reservation_port,workers=1):
    baseline=load_research_model(base_dir/plan.baseline_spec,ResearchRunSpec)
    treatment=load_research_model(base_dir/plan.treatment_spec,ResearchRunSpec)
    output.mkdir(parents=True,exist_ok=False);results=[];quantities=[]
    for index,replication_id in enumerate(plan.replication_ids):
        report=run_pair(with_replication(baseline,replication_id),with_replication(treatment,replication_id),
            output/f'pair-{index:04d}',reservation_port=reservation_port,workers=workers)
        results.append(report)
        if report['status']=='verified_pair':
            quantities.append(tuple(float(sum(r['quote_valid'] for r in rows(Path(report[side])/'measurements.jsonl'))/baseline.base.steps)
                                    for side in ('left','right')))
    result={'status':'COMMITTED' if all(r['status']=='verified_pair' for r in results) else 'INCOMPLETE',
        'purpose':plan.purpose,'pairs':results,'quote_coverage_pairing':paired_covariance(quantities),
        'scientific_eligibility':'NOT_EVALUATED'}
    durable_json(output/'batch.json',result);return result
