"""Content-verified data, disjoint splits, and an explicit finite-fit boundary.

Phase 2 provides structural/numerical evidence and honest NOT_PERFORMED receipts.
Empirical fit requires real reference data and a health-checked standard Runner
adapter. Historical path replay is not a permissible evaluator.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator

from core.experiments.identity import HASH_PATTERN, file_sha256, research_hash
from core.experiments.strict_types import StrictResearchModel, load_research_model, strict_json_loads
from core.validation.statistics import compute_reference_statistics

DataOrigin = Literal['historical_input','calibration_data','evaluation_data','simulation_output','synthetic_fixture','mock']


class DataSource(StrictResearchModel):
    provider: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    license: str = Field(min_length=1)
    acquired_at: str = Field(min_length=1)
    description: str = Field(min_length=1)

    @model_validator(mode='after')
    def _time(self):
        if datetime.fromisoformat(self.acquired_at).tzinfo is None:
            raise ValueError('Acquisition time requires an explicit offset')
        return self


class DataManifest(StrictResearchModel):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    origin: DataOrigin
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=HASH_PATTERN)
    bytes: int = Field(ge=0)
    retrieval_status: Literal['available','failed']
    source: DataSource
    time_start: str
    time_end: str
    timezone: str
    frequency: Literal['daily','tick','step']
    clock_rule: Literal['calendar_day','event_index','declared_step']
    columns: dict[str,str]
    units: dict[str,str]
    field_origins: dict[str,DataOrigin]
    adjustment: Literal['unadjusted','split_adjusted','total_return','not_applicable']
    missing_policy: Literal['preserve_null','reject']
    split: Literal['train','validation','test','ood','confirmation','diagnostic','initialization']
    purpose: Literal['calibration_fit','calibration_validation','held_out_evaluation','initial_price','engineering_test','simulation_analysis']

    @model_validator(mode='after')
    def _semantics(self):
        ZoneInfo(self.timezone)
        if not {'timestamp','price'}<=set(self.columns) or set(self.columns)-{'timestamp','price','volume','spread','depth','step'}:
            raise ValueError('Data columns require timestamp/price and declared supported fields')
        if len(set(self.columns.values()))!=len(self.columns) or set(self.units)!=set(self.columns) or set(self.field_origins)!=set(self.columns):
            raise ValueError('Column mappings, units and origins must cover the same unique fields')
        if any(origin!=self.origin for origin in self.field_origins.values()):
            raise ValueError('MIXED_ORIGIN: reference volume cannot masquerade as simulated executions')
        expected={'timestamp':'time','price':'price','volume':'shares','spread':'price','depth':'shares','step':'step'}
        if any(unit!=expected[name] for name,unit in self.units.items()):
            raise ValueError('Data field units disagree with registered semantics')
        if (self.frequency=='daily' and self.clock_rule!='calendar_day'
                or self.frequency=='tick' and self.clock_rule!='event_index'
                or self.frequency=='step' and (self.clock_rule!='declared_step' or 'step' not in self.columns)):
            raise ValueError('Frequency requires its explicit clock; a simulation step is not a minute')
        if self.frequency!='tick' and {'spread','depth'}&set(self.columns):
            raise ValueError('FREQUENCY_MISMATCH: only tick data can support spread/depth here')
        if self.purpose=='calibration_fit' and self.split!='train':
            raise ValueError('Calibration fit may only read train data')
        if self.purpose=='calibration_validation' and self.split!='validation':
            raise ValueError('Calibration validation requires its separate validation split')
        if self.purpose=='held_out_evaluation' and self.split not in {'test','ood','confirmation'}:
            raise ValueError('Held-out reference must have a held-out split')
        if self.origin=='historical_input' and (self.purpose!='initial_price' or self.split!='initialization'):
            raise ValueError('Historical input is an initial condition, not calibration')
        if self.origin=='evaluation_data' and self.purpose!='held_out_evaluation':
            raise ValueError('Evaluation reference cannot be a calibration input')
        if self.origin in {'synthetic_fixture','mock'} and self.purpose!='engineering_test':
            raise ValueError('Synthetic/mock data cannot claim empirical calibration or evaluation')
        return self


@dataclass(frozen=True)
class ValidatedData:
    manifest: DataManifest
    manifest_path: Path
    source_path: Path
    rows: tuple[dict,...]
    timestamps: tuple[datetime,...]
    clock: tuple[int,...]


def validate_data_manifest(path: str|Path) -> ValidatedData:
    manifest_path=Path(path).resolve(strict=True)
    manifest=load_research_model(manifest_path,DataManifest)
    if manifest.retrieval_status!='available':
        raise ValueError('DATA_UNAVAILABLE: retrieval failed; no synthetic fallback')
    source=(manifest_path.parent/manifest.file).resolve(strict=True)
    if not source.is_file() or source.stat().st_size!=manifest.bytes or file_sha256(source)!=manifest.sha256:
        raise ValueError('DATA_CONTENT_MISMATCH')
    rows,times,clock=[],[],[]
    tz=ZoneInfo(manifest.timezone)
    def timestamp(value):
        t=datetime.fromisoformat(value)
        return t.replace(tzinfo=tz) if t.tzinfo is None else t.astimezone(tz)
    with source.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if (reader.fieldnames is None or len(reader.fieldnames)!=len(set(reader.fieldnames))
                or not set(manifest.columns.values())<=set(reader.fieldnames)):
            raise ValueError('DATA_COLUMNS_MISMATCH_OR_DUPLICATE')
        for index,row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed CSV record')
            t=timestamp(row[manifest.columns['timestamp']])
            record={'timestamp':t.isoformat()}
            for field,column in manifest.columns.items():
                if field=='timestamp':
                    continue
                raw=row[column]
                value=None if raw=='' else float(raw)
                if value is None:
                    if manifest.missing_policy=='reject' or field=='step':
                        raise ValueError('DATA_MISSING_VALUE')
                elif not math.isfinite(value) or (field=='price' and value<=0.0) or (field!='price' and value<0.0):
                    raise ValueError('DATA_INVALID_NUMERIC_VALUE')
                if field=='step' and (not value.is_integer() or value<0.0):
                    raise ValueError('Declared simulation clock must contain nonnegative integers')
                record[field]=value
            times.append(t);rows.append(record)
            clock.append(t.date().toordinal() if manifest.clock_rule=='calendar_day' else
                         int(record['step']) if manifest.clock_rule=='declared_step' else index)
    if not rows or times!=sorted(set(times)) or clock!=sorted(set(clock)):
        raise ValueError('DATA_CLOCK_NOT_UNIQUE_INCREASING')
    if times[0]!=timestamp(manifest.time_start) or times[-1]!=timestamp(manifest.time_end):
        raise ValueError('DATA_DECLARED_TIME_COVERAGE_MISMATCH')
    return ValidatedData(manifest,manifest_path,source,tuple(rows),tuple(times),tuple(clock))


def data_check(path: str|Path) -> dict:
    data=validate_data_manifest(path)
    stats=compute_reference_statistics(clock=list(data.clock),prices=[row['price'] for row in data.rows],
        volumes=[row['volume'] for row in data.rows] if 'volume' in data.manifest.columns else None,
        frequency=data.manifest.frequency)
    return {'status':'VALIDATED_DATA','manifest_hash':research_hash(data.manifest),
        'content_sha256':data.manifest.sha256,'origin':data.manifest.origin,'purpose':data.manifest.purpose,
        'split':data.manifest.split,'row_count':len(data.rows),'statistics':stats,
        'empirical_calibration':'NOT_PERFORMED'}


class CalibrationPlan(StrictResearchModel):
    schema_version: Literal[1] = 1
    plan_id: str = Field(min_length=1)
    train_manifests: list[str]
    validation_manifests: list[str]
    run_spec_ref: str = Field(min_length=1)
    health_evidence_ref: str|None
    target_frequency: Literal['daily','tick','step']
    simulation_steps_per_observation: int = Field(ge=1)
    temporal_mapping_justification: str = Field(min_length=1)
    parameter_candidates: list[dict[str,float|int]] = Field(min_length=1)
    target_statistics: list[str] = Field(min_length=1)
    loss_scales: dict[str,float]
    loss_weights: dict[str,float]
    replications: list[int] = Field(min_length=1)
    max_runner_attempts: int = Field(ge=1)
    selection_rule: Literal['minimum_train_loss_then_candidate_index']

    @model_validator(mode='after')
    def _finite_plan(self):
        if (len(self.train_manifests)!=len(set(self.train_manifests))
                or len(self.validation_manifests)!=len(set(self.validation_manifests))
                or set(self.train_manifests)&set(self.validation_manifests)):
            raise ValueError('SPLIT_LEAKAGE: manifests cannot be duplicated across fitting/validation')
        if (len(set(self.target_statistics))!=len(self.target_statistics)
                or set(self.loss_scales)!=set(self.target_statistics) or set(self.loss_weights)!=set(self.target_statistics)
                or any(v<=0.0 for v in self.loss_scales.values()) or any(v<=0.0 for v in self.loss_weights.values())):
            raise ValueError('Targets need unique names, explicit positive loss scales and weights')
        if len(self.replications)!=len(set(self.replications)) or any(seed<0 or seed>=2**32 for seed in self.replications):
            raise ValueError('Calibration replications need unique registered seeds')
        if any(not candidate for candidate in self.parameter_candidates):
            raise ValueError('Finite candidate settings cannot be empty')
        if len({research_hash(candidate) for candidate in self.parameter_candidates})!=len(self.parameter_candidates):
            raise ValueError('Repeated candidates cannot count as independent search')
        if len(self.parameter_candidates)*len(self.replications)>self.max_runner_attempts:
            raise ValueError('Finite search exceeds the registered Runner attempt budget')
        return self


def validate_calibration(plan: CalibrationPlan, *, base_dir: str|Path) -> dict:
    plan=CalibrationPlan.model_validate(plan);base=Path(base_dir)
    train=[validate_data_manifest(base/path) for path in plan.train_manifests]
    validation=[validate_data_manifest(base/path) for path in plan.validation_manifests]
    for data in train+validation:
        if data.manifest.frequency!=plan.target_frequency:
            raise ValueError('FREQUENCY_MISMATCH: reference and target frequency differ')
    for split,values in [('train',train),('validation',validation)]:
        for data in values:
            if data.manifest.split!=split or data.manifest.origin!='calibration_data':
                raise ValueError('SPLIT_OR_ORIGIN_LEAKAGE: calibration requires its owned empirical split')
    all_data=train+validation
    for index,a in enumerate(all_data):
        for b in all_data[index+1:]:
            if (a.manifest.sha256==b.manifest.sha256 or
                max(a.timestamps[0],b.timestamps[0])<=min(a.timestamps[-1],b.timestamps[-1])):
                raise ValueError('SPLIT_LEAKAGE: reference contents or temporal ranges overlap')
    if len({data.manifest.adjustment for data in all_data})>1:
        raise ValueError('ADJUSTMENT_MISMATCH')
    if plan.target_frequency!='tick' and any('spread' in target or 'depth' in target for target in plan.target_statistics):
        raise ValueError('FREQUENCY_MISMATCH: daily/step reference cannot certify tick liquidity')
    missing=[]
    if not train:missing.append('EMPIRICAL_TRAIN_DATA')
    if not validation:missing.append('EMPIRICAL_VALIDATION_DATA')
    if plan.health_evidence_ref is None or not (base/plan.health_evidence_ref).is_file():
        missing.append('PHASE3_HEALTH_QUALIFICATION')
    if not (base/plan.run_spec_ref).is_file():missing.append('RESOLVED_RESEARCH_RUN_SPEC')
    return {'status':'READY_FOR_ADAPTER_VALIDATION' if not missing else 'NOT_PERFORMED',
        'plan_hash':research_hash(plan),'missing':missing,'reference_manifest_hashes':[research_hash(d.manifest) for d in all_data],
        'empirical_calibration':'NOT_PERFORMED'}


class CalibrationEvaluator(Protocol):
    """A Phase 3 adapter must bind health/source/spec and expose only this Runner path."""
    implementation_version: str
    def validate_health_and_scope(self, plan: CalibrationPlan, *, base_dir: Path) -> None: ...
    def evaluate_parameters(self, parameters: dict, replication: int, plan: CalibrationPlan) -> dict[str,float]: ...


def fit_calibration(plan: CalibrationPlan, *, base_dir: str|Path, evaluator: CalibrationEvaluator|None=None) -> dict:
    base=Path(base_dir);readiness=validate_calibration(plan,base_dir=base)
    if evaluator is None:
        readiness['missing'].append('HEALTH_VERIFIED_STANDARD_RUNNER_ADAPTER')
    if readiness['missing']:
        return dict(readiness,status='NOT_PERFORMED',runner_attempts=0,parameter_sets=None)
    if evaluator.implementation_version!='research-runner-calibration-v1':
        raise ValueError('Only the registered standard Runner adapter can fit calibration')
    evaluator.validate_health_and_scope(plan,base_dir=base)
    train=[validate_data_manifest(base/path) for path in plan.train_manifests]
    references=[data_check(data.manifest_path)['statistics']['metrics'] for data in train]
    target={name:math.fsum(ref[name]['value'] for ref in references)/len(references) for name in plan.target_statistics}
    trials=[]
    for index,parameters in enumerate(plan.parameter_candidates):
        values=[evaluator.evaluate_parameters(dict(parameters),seed,plan) for seed in plan.replications]
        if any(set(row)!=set(target) or any(type(v) is not float or not math.isfinite(v) for v in row.values()) for row in values):
            raise ValueError('Invalid Runner calibration statistic; failed attempts cannot be dropped')
        mean={name:math.fsum(row[name] for row in values)/len(values) for name in target}
        loss=math.fsum(plan.loss_weights[name]*((mean[name]-target[name])/plan.loss_scales[name])**2 for name in target)
        trials.append({'candidate_index':index,'parameters':dict(parameters),'statistics':mean,'loss':loss})
    best=min(trials,key=lambda trial:(trial['loss'],trial['candidate_index']))
    return dict(readiness,status='FIT_COMPLETED_VALIDATION_PENDING',empirical_calibration='VALIDATION_PENDING',
        runner_attempts=len(trials)*len(plan.replications),trials=trials,
        parameter_sets=[best['parameters']],selection_uses='train_only')


def _historical_data(spec,repo_root):
    historical=spec.base.historical_input
    if historical is None:return None
    if historical.origin!='historical':
        raise ValueError('Historical initial price requires actual historical origin')
    matches=[]
    for ref in spec.provenance_refs:
        path=Path(repo_root)/ref
        if path.suffix=='.json':
            payload=strict_json_loads(path.read_text(encoding='utf-8'))
            if isinstance(payload,dict) and payload.get('origin')=='historical_input' and 'dataset_id' in payload:
                matches.append(validate_data_manifest(path))
    if len(matches)!=1:raise ValueError('Historical initial price requires one bound DataManifest')
    data=matches[0]
    if data.manifest.sha256!=historical.sha256 or data.source_path!=(Path(repo_root)/historical.path).resolve():
        raise ValueError('Historical source path/hash must match the validated manifest')
    if data.rows[0]['price'] is None:raise ValueError('Historical first price is missing')
    return data


def validate_historical_initial_price(spec,repo_root):
    data=_historical_data(spec,repo_root)
    if data and data.rows[0]['price']!=spec.world.initial_price:
        raise ValueError('UNRESOLVED_HISTORICAL_INITIAL_PRICE: build the explicit effective spec before hashing')


def resolve_historical_initial_price(spec,*,repo_root):
    """Return a new explicit spec and receipt before any world/identity is built.

    Only P0 fields change. Wealth, private valuations, fundamental and policy
    inventory cost remain declared independent economic inputs.
    """
    data=_historical_data(spec,repo_root)
    if data is None:raise ValueError('No historical input was selected')
    price=data.rows[0]['price']
    effective=spec.model_copy(update={'base':spec.base.model_copy(update={
        'market':spec.base.market.model_copy(update={'initial_price':price})}),
        'world':spec.world.model_copy(update={'initial_price':price})})
    validate_historical_initial_price(effective,repo_root)
    return effective,{'operation':'resolve_historical_initial_price','source_sha256':data.manifest.sha256,
        'manifest_hash':research_hash(data.manifest),'previous_initial_price':spec.world.initial_price,
        'effective_initial_price':price,'timestamp':data.rows[0]['timestamp'],'purpose':'initial_price_only'}
