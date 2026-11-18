"""Development diagnostics, justified profile freezing and held-out health checks."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field,model_validator
from core.experiments.strict_types import StrictResearchModel,load_research_model,strict_json_loads
from core.experiments.identity import research_hash,HASH_PATTERN,file_sha256
from core.experiments.design_config import ResearchRunSpec
from core.experiments.archive import durable_json,rows,verify_archive_v3
from core.experiments.counterfactual import with_replication,raw_cache_identity
from core.experiments.jobs import Job,JobCoordinator

SCENARIOS={'normal','fundamental_shock','panic_seed','normal_mm_withdrawal','false_intervention','combined'}


class HealthCase(StrictResearchModel):
    case_id:str=Field(pattern=r'^[A-Za-z0-9_-]+$')
    scenario:Literal['normal','fundamental_shock','panic_seed','normal_mm_withdrawal','false_intervention','combined']
    spec:str
    replication_ids:list[str]=Field(min_length=1)


class HealthSuite(StrictResearchModel):
    version:Literal[1]=1
    namespace:Literal['development','validation']
    cases:list[HealthCase]=Field(min_length=1)

    @model_validator(mode='after')
    def _ids(self):
        ids=[r for c in self.cases for r in c.replication_ids]
        if len(ids)!=len(set(ids)) or len({c.case_id for c in self.cases})!=len(self.cases):raise ValueError('Duplicate health IDs')
        if any(not r.startswith(self.namespace+'/') for r in ids):raise ValueError('Health IDs must belong to the declared namespace')
        return self


class HealthCriterion(StrictResearchModel):
    criterion_id:str=Field(min_length=1)
    scenario:str=Field(min_length=1)
    statistic:Literal['quote_coverage','trading_step_fraction','lp_buy_spend','halt_fraction','mean_price_proxy']
    lower:float|None
    upper:float|None
    maximum_half_width:float|None
    alpha:float=Field(gt=0,lt=1)
    rationale:str
    source_refs:list[str]
    failure_rule:Literal['any_case_fails']='any_case_fails'

    @model_validator(mode='after')
    def _range(self):
        if self.lower is not None and self.upper is not None and self.lower>self.upper:raise ValueError('Reversed acceptance interval')
        if self.maximum_half_width is not None and self.maximum_half_width<=0:raise ValueError('Positive precision required')
        return self


class GateProfile(StrictResearchModel):
    profile_id:str=Field(min_length=1)
    version:int=Field(ge=1)
    purpose:Literal['synthetic_financial_health']
    source_hash:str|None=Field(pattern=HASH_PATTERN)
    model_scope_hash:str|None=Field(pattern=HASH_PATTERN)
    required_gate_scopes:dict[str,list[str]]
    criteria:list[HealthCriterion]
    family_alpha:float=Field(gt=0,lt=1)
    development_ids:list[str]
    validation_ids:list[str]
    calibration_claim_level:Literal['VALID_FOR_SYNTHETIC_TASK']='VALID_FOR_SYNTHETIC_TASK'
    freeze_receipt_hash:str|None=Field(pattern=HASH_PATTERN)

    @model_validator(mode='after')
    def _scope(self):
        if set(self.required_gate_scopes)!={f'S{i}' for i in range(7)}:raise ValueError('Explicit S0-S6 scopes required')
        if len(set(self.development_ids))!=len(self.development_ids) or len(set(self.validation_ids))!=len(self.validation_ids):raise ValueError('Repeated profile IDs')
        if set(self.development_ids)&set(self.validation_ids):raise ValueError('Validation contaminated by development')
        if any(not r.startswith('development/') for r in self.development_ids) or any(not r.startswith('validation/') for r in self.validation_ids):
            raise ValueError('Profile namespace mismatch')
        if len({c.criterion_id for c in self.criteria})!=len(self.criteria):raise ValueError('Duplicate criterion')
        return self


def profile_semantic(profile):return profile.model_dump(mode='json',exclude={'freeze_receipt_hash'})


def verify_frozen_profile(path:Path):
    path=Path(path);profile=load_research_model(path,GateProfile)
    if profile.freeze_receipt_hash is None:raise ValueError('PROFILE_NOT_FROZEN')
    receipt=strict_json_loads(path.with_suffix('.freeze.json').read_text())
    if research_hash(receipt)!=profile.freeze_receipt_hash or receipt['profile_hash']!=research_hash(profile_semantic(profile)):
        raise ValueError('FROZEN_PROFILE_IDENTITY_MISMATCH')
    return profile,receipt


def freeze_profile(draft_path,development_evidence,output):
    draft=load_research_model(draft_path,GateProfile);output=Path(output)
    if output.exists() or output.with_suffix('.freeze.json').exists():raise ValueError('Frozen profiles are immutable')
    report_path=Path(development_evidence)/'health.json'
    report=strict_json_loads(report_path.read_text());missing=[]
    if report['namespace']!='development' or report['status']!='DEVELOPMENT_ONLY' or not report['runs']:
        raise ValueError('Freeze requires complete actual development-only evidence')
    if set(report['replication_ids'])!=set(draft.development_ids):raise ValueError('Development ID mismatch')
    actual_ids=[]
    for item in report['runs']:
        verify_archive_v3(item['archive'])
        if file_sha256(Path(item['archive'])/'archive_manifest.json')!=item['manifest_sha256']:raise ValueError('Development evidence changed')
        spec=ResearchRunSpec.model_validate_json((Path(item['archive'])/'research_spec.json').read_text())
        source=strict_json_loads((Path(item['archive'])/'archive_manifest.json').read_text())['source_hash']
        if spec.random.replication_id!=item['replication_id'] or spec.data_phase!='development' or source!=report['source_hash']:
            raise ValueError('Development archive scope/namespace mismatch')
        if diagnostic_metrics(Path(item['archive']))!=item['metrics']:raise ValueError('Development statistic differs from actual archive')
        actual_ids.append(spec.random.replication_id)
    if sorted(actual_ids)!=sorted(report['replication_ids']):raise ValueError('Incomplete or repeated development runs')
    if not draft.validation_ids:missing.append('Independent validation IDs and sample count are absent')
    if not draft.criteria:missing.append('No scientific acceptance criteria')
    if set(c.scenario for c in draft.criteria)!=SCENARIOS:missing.append('Per-scenario acceptance coverage incomplete')
    for criterion in draft.criteria:
        if criterion.lower is None and criterion.upper is None:missing.append(criterion.criterion_id+': acceptance interval absent')
        if criterion.maximum_half_width is None:missing.append(criterion.criterion_id+': precision requirement absent')
        if not criterion.rationale.strip() or not criterion.source_refs:missing.append(criterion.criterion_id+': rationale/source absent')
        if criterion.statistic not in {'quote_coverage','trading_step_fraction','halt_fraction'}:
            missing.append(criterion.criterion_id+': diagnostic quantity lacks a preregistered bounded precision procedure')
    from fractions import Fraction
    if sum((Fraction.from_float(c.alpha) for c in draft.criteria),Fraction())>Fraction.from_float(draft.family_alpha):
        missing.append('Criterion alpha allocations exceed the frozen health family budget')
    if any(not v for v in draft.required_gate_scopes.values()):missing.append('Required gate evidence declarations are incomplete')
    if draft.source_hash!=report['source_hash'] or draft.model_scope_hash!=report['model_scope_hash']:
        missing.append('Draft does not bind the actual development implementation/model scope')
    result={'status':'INCONCLUSIVE' if missing else 'FROZEN','missing_requirements':missing,
        'development_report':str(report_path.resolve()),'development_sha256':file_sha256(report_path),
        'profile_hash':research_hash(profile_semantic(draft)),'source_hash':report['source_hash'],
        'validation_ids':list(draft.validation_ids),'empirical_calibration':'NOT_PERFORMED'}
    if missing:
        refusal=Path(development_evidence)/'freeze_refusal.json'
        if refusal.exists():raise ValueError('Freeze attempt receipt exists; use a fresh development attempt')
        durable_json(refusal,result);return result
    output.parent.mkdir(parents=True,exist_ok=True)
    durable_json(output.with_suffix('.freeze.json'),result)
    frozen=GateProfile(**dict(draft.model_dump(mode='json'),freeze_receipt_hash=research_hash(result)))
    durable_json(output,frozen.model_dump(mode='json'))
    return result


def diagnostic_metrics(archive:Path):
    spec=ResearchRunSpec.model_validate_json((archive/'research_spec.json').read_text())
    observed=[r for r in rows(archive/'measurements.jsonl') if spec.measurement.evaluation_start<=r['step']<spec.measurement.evaluation_end]
    trades=rows(archive/'trades.jsonl') if (archive/'trades.jsonl').exists() else []
    lp=spec.world.policy_account_initial.actor_id
    prices=[r['metrics']['price_proxy']['value'] for r in observed if r['metrics']['price_proxy']['value'] is not None]
    return {'quote_coverage':sum(r['quote_valid'] for r in observed)/len(observed),
        'trading_step_fraction':sum(any(t['step']==r['step'] for t in trades) for r in observed)/len(observed),
        'halt_fraction':sum(not r['market_open'] for r in observed)/len(observed),
        'lp_buy_spend':math.fsum(t['price']*t['quantity']+t['buyer_fee'] for t in trades if t['buyer_agent_id']==lp),
        'lp_trade_count':sum(lp in (t['buyer_agent_id'],t['seller_agent_id']) for t in trades),
        'settled_trade_count':len(trades),'mean_price_proxy':math.fsum(prices)/len(prices) if prices else None,
        'valid_price_steps':len(prices),'clock_steps':len(observed)}


def assess_criteria(profile:GateProfile,runs:list[dict]):
    results=[]
    for criterion in profile.criteria:
        # Distinct policies/cases are not interchangeable independent seeds.
        matching=[r for r in runs if r['scenario']==criterion.scenario]
        cases=sorted({r.get('case_id','unit-case') for r in matching}) or ['missing-case']
        for case in cases:
            values=[r['metrics'][criterion.statistic] for r in matching if r.get('case_id','unit-case')==case]
            results.append(_assess_one(criterion,case,values,criterion.alpha/len(cases)))
    return results


def _assess_one(criterion,case,values,alpha):
        reason=[];status='INCONCLUSIVE';mean=half=None
        if not values or any(v is None for v in values):reason.append('Missing scenario observations')
        elif criterion.lower is None and criterion.upper is None:reason.append('No acceptance interval')
        elif criterion.maximum_half_width is None:reason.append('No precision requirement')
        elif criterion.statistic not in {'quote_coverage','trading_step_fraction','halt_fraction'}:
            reason.append('Unbounded diagnostic quantity has no registered confidence procedure')
        else:
            mean=math.fsum(values)/len(values);half=math.sqrt(math.log(2/alpha)/(2*len(values)))
            lo=max(0.0,mean-half);hi=min(1.0,mean+half)
            if (criterion.lower is not None and hi<criterion.lower) or (criterion.upper is not None and lo>criterion.upper):status='FAIL'
            elif half<=criterion.maximum_half_width and (criterion.lower is None or lo>=criterion.lower) and (criterion.upper is None or hi<=criterion.upper):status='PASS'
            else:reason.append('Interval or precision is inconclusive')
        return {'criterion_id':criterion.criterion_id,'case_id':case,'status':status,'n':len(values),
            'estimate':mean,'half_width':half,'alpha':alpha,'reasons':reason}


def run_health(suite:HealthSuite,profile:GateProfile,*,base_dir:Path,output:Path,reservation_port,workers=1,development=False,freeze_receipt=None):
    if development!=(suite.namespace=='development'):raise ValueError('HEALTH_NAMESPACE_VIOLATION')
    if not development and profile.freeze_receipt_hash is None:raise ValueError('PROFILE_NOT_FROZEN')
    if not development and (freeze_receipt is None or research_hash(freeze_receipt)!=profile.freeze_receipt_hash
        or freeze_receipt.get('status')!='FROZEN' or freeze_receipt.get('profile_hash')!=research_hash(profile_semantic(profile))):
        raise ValueError('FROZEN_RECEIPT_REQUIRED')
    ids=[r for c in suite.cases for r in c.replication_ids]
    if set(ids)!=set(profile.development_ids if development else profile.validation_ids):raise ValueError('HEALTH_PLAN_IDS_MISMATCH')
    specs=[];jobs=[];metadata=[]
    for case in suite.cases:
        spec=load_research_model(base_dir/case.spec,ResearchRunSpec)
        if spec.data_phase!=suite.namespace:raise ValueError('Spec data_phase differs from suite namespace')
        kinds={event.kind for event in spec.scenario.events}
        if (case.scenario in {'normal','false_intervention'} and kinds
            or case.scenario in {'fundamental_shock','panic_seed','normal_mm_withdrawal'} and case.scenario not in kinds
            or case.scenario=='combined' and len(kinds)<2
            or case.scenario=='normal' and (spec.policy.halt is not None or spec.policy.liquidity is not None)
            or case.scenario=='false_intervention' and spec.policy.halt is None and spec.policy.liquidity is None):
            raise ValueError('Declared health scenario does not match effective mechanism inputs')
        specs.append(spec)
        for i,rid in enumerate(case.replication_ids):
            effective=with_replication(spec,rid);jobs.append(Job(job_id=f'{case.case_id}-{i:04d}',spec=effective))
            metadata.append({'case_id':case.case_id,'scenario':case.scenario,'replication_id':rid})
    identities=[raw_cache_identity(s) for s in specs];sources={r['body']['source_hash'] for r in identities}
    if len(sources)!=1:raise ValueError('Health scenarios must use a single bound implementation/resource set')
    def model_body(s):
        d=s.model_dump(mode='json');d['random'].pop('replication_id');d.pop('data_phase');return d
    model_scope_hash=research_hash([model_body(s) for s in specs]);source=next(iter(sources))
    if not development and (source!=profile.source_hash or model_scope_hash!=profile.model_scope_hash):
        raise ValueError('FROZEN_HEALTH_SCOPE_CHANGED')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    durable_json(output/'suite.json',suite.model_dump(mode='json'))
    durable_json(output/'profile.json',profile.model_dump(mode='json'))
    if freeze_receipt is not None:durable_json(output/'freeze_receipt.json',freeze_receipt)
    batch=JobCoordinator(output/'jobs',reservation_port).execute(jobs,workers=workers)
    records=[]
    for meta,row in zip(metadata,batch['results']):
        if row['status']=='COMMITTED':
            root=Path(row['archive']);verify_archive_v3(root)
            records.append({**meta,'archive':str(root),'manifest_sha256':file_sha256(root/'archive_manifest.json'),
                            'metrics':diagnostic_metrics(root)})
    criteria=[] if development else assess_criteria(profile,records)
    status='DEVELOPMENT_ONLY' if development else 'PASS' if criteria and all(c['status']=='PASS' for c in criteria) else 'FAIL' if any(c['status']=='FAIL' for c in criteria) else 'INCONCLUSIVE'
    if batch['status']!='COMMITTED':status='INCOMPLETE'
    report={'status':status,'namespace':suite.namespace,'source_hash':source,'model_scope_hash':model_scope_hash,
        'profile_hash':research_hash(profile_semantic(profile)),'replication_ids':ids,'runs':records,'criteria':criteria,
        'scientific_certificate_issued':False,'empirical_calibration':'NOT_PERFORMED'}
    durable_json(output/'health.json',report);return report


def run_health_suite(suite,profile,**execution):
    return run_health(suite,profile,**execution)


def evaluate_health_profile(profile:GateProfile,runs:list[dict]):
    return assess_criteria(profile,runs)
