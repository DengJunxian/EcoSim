"""Bounded administrative recovery of preselected interrupted qualification jobs.

The original experiment, its budget and its incomplete report remain immutable.
This amendment buys one whole-job attempt for every preselected interruption;
it cannot change scientific inputs, add worlds or rerun a committed observation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from core.codesign.budget import BudgetLedger, BudgetSpec
from core.experiments.archive import durable_json, verify_archive_v3
from core.experiments.counterfactual import raw_cache_identity
from core.experiments.design_config import ResearchRunSpec
from core.experiments.identity import file_sha256, research_hash
from core.experiments.jobs import Job, JobCoordinator
from core.experiments.strict_types import StrictResearchModel, load_research_model, strict_json_loads
from core.research_analysis.financial_qualification import (
    QualificationPlan, assess_binary, extract_observations, predicate_observation,
    validate_plan, with_replication,
)

ROOT = Path(__file__).resolve().parents[2]
HASH = r'^[0-9a-f]{64}$'
VERSION = 'phase7-financial-administrative-recovery-v1'


class BoundFile(StrictResearchModel):
    path: str
    sha256: str = Field(pattern=HASH)


class RecoveryJob(StrictResearchModel):
    job_id: str
    run_key: str = Field(pattern=HASH)
    original_input: BoundFile
    original_attempt: str
    original_attempt_id: str
    original_exit_code: Literal[-9] = -9
    complete_research_spec: ResearchRunSpec


class AdministrativeAmendment(StrictResearchModel):
    schema_version: Literal[1] = 1
    version: Literal['phase7-financial-administrative-recovery-v1'] = VERSION
    registered_at: str
    authorization: str
    original_directory: str
    original_plan: BoundFile
    original_freeze: BoundFile
    original_batch: BoundFile
    pre_result_candidate: BoundFile
    source_hash: str = Field(pattern=HASH)
    source_files: dict[str, str]
    jobs: list[RecoveryJob] = Field(min_length=94, max_length=94)
    maximum_attempts: Literal[94] = 94
    maximum_unique_in_recovery: Literal[94] = 94
    global_unique_scientific_runs: Literal[768] = 768
    global_maximum_attempts: Literal[862] = 862
    wall_timeout_seconds: Literal[7200.0] = 7200.0
    llm_calls: Literal[0] = 0
    selection_rule: Literal['all_original_INTERRUPTED_exit_minus9_no_outcome_selection'] = 'all_original_INTERRUPTED_exit_minus9_no_outcome_selection'
    execution_rule: Literal['exact_complete_input_fresh_start_one_attempt'] = 'exact_complete_input_fresh_start_one_attempt'
    preserve_original_incomplete_report: Literal[True] = True

    @model_validator(mode='after')
    def no_duplicates(self):
        if len({j.job_id for j in self.jobs}) != 94 or len({j.run_key for j in self.jobs}) != 94:
            raise ValueError('Recovery jobs must be unique')
        return self


def _load_bound(ref: BoundFile):
    if file_sha256(ref.path) != ref.sha256:
        raise ValueError('BOUND_FILE_CHANGED: ' + ref.path)
    return strict_json_loads(Path(ref.path).read_text())


def check_selection(amendment, batch, candidate):
    """Pure integrity check used by negative tests before starting any worker."""
    interrupted = {r['job_id']: r for r in batch['results'] if r['status'] == 'INTERRUPTED'}
    if len(batch['results']) != 768 or len(interrupted) != 94:
        raise ValueError('ORIGINAL_TERMINAL_COUNTS_CHANGED')
    expected = {j['job_id']: j for j in candidate['jobs']}
    if set(expected) != set(interrupted) or {j.job_id for j in amendment.jobs} != set(interrupted):
        raise ValueError('RECOVERY_SELECTION_CHANGED')
    for job in amendment.jobs:
        old, fixed = interrupted[job.job_id], expected[job.job_id]
        if old['exit_code'] != -9 or old['run_key'] != job.run_key or fixed['run_key'] != job.run_key:
            raise ValueError('RECOVERY_IDENTITY_CHANGED')
        if (old['attempt'] != job.original_attempt or old['attempt_id'] != job.original_attempt_id
                or fixed['original_input_sha256'] != job.original_input.sha256
                or fixed['original_input_path'] != job.original_input.path
                or fixed['complete_research_spec'] != job.complete_research_spec.model_dump(mode='json')):
            raise ValueError('RECOVERY_COMPLETE_INPUT_CHANGED')
    if candidate['outcome_exposure']['scientific_stats_reviewed_by_agent'] is not False:
        raise ValueError('RECOVERY_SELECTION_NOT_PREREGISTERED')
    return True


def validate_amendment(amendment, *, root=ROOT):
    for path, digest in amendment.source_files.items():
        if file_sha256(Path(root) / path) != digest:
            raise ValueError('RECOVERY_PROGRAM_CHANGED')
    if set(amendment.source_files) != {'core/research_analysis/financial_recovery.py', 'scripts/research/qualify_financial_phase7_recovery.py'}:
        raise ValueError('RECOVERY_PROGRAM_BINDING_INCOMPLETE')
    original = QualificationPlan.model_validate(_load_bound(amendment.original_plan))
    validate_plan(original, root=root)
    _load_bound(amendment.original_freeze)
    batch, candidate = _load_bound(amendment.original_batch), _load_bound(amendment.pre_result_candidate)
    check_selection(amendment, batch, candidate)
    if amendment.source_hash != original.source_hash or candidate['source_hash'] != original.source_hash:
        raise ValueError('RECOVERY_SOURCE_CHANGED')
    for job in amendment.jobs:
        old_identity = _load_bound(job.original_input)
        case_id, index = job.job_id.rsplit('-', 1)
        case = next(c for c in original.cases if c.case_id == case_id)
        spec = with_replication(case.spec, original.replication_ids[int(index)])
        if spec != job.complete_research_spec:
            raise ValueError('RECOVERY_REGISTERED_SPEC_CHANGED')
        identity = raw_cache_identity(spec)
        if identity != old_identity or identity['run_key'] != job.run_key:
            raise ValueError('RECOVERY_SOURCE_ENVIRONMENT_OR_INPUT_CHANGED')
    return original


def build_amendment(original_directory, authorization, *, root=ROOT):
    base = Path(original_directory)
    candidate_path = base / 'administrative_recovery_candidate_before_result_review.json'
    candidate = strict_json_loads(candidate_path.read_text())
    bound = lambda p: BoundFile(path=str(p), sha256=file_sha256(p))
    result = AdministrativeAmendment(
        registered_at=datetime.now(timezone.utc).isoformat(), authorization=authorization,
        original_directory=str(base), original_plan=bound(base / 'plan.json'),
        original_freeze=bound(base / 'freeze_receipt.json'), original_batch=bound(base / 'jobs/batch-attempt-0001.json'),
        pre_result_candidate=bound(candidate_path), source_hash=candidate['source_hash'],
        source_files={p: file_sha256(Path(root) / p) for p in (
            'core/research_analysis/financial_recovery.py', 'scripts/research/qualify_financial_phase7_recovery.py')},
        jobs=[RecoveryJob(job_id=j['job_id'], run_key=j['run_key'],
                          original_input=BoundFile(path=j['original_input_path'], sha256=j['original_input_sha256']),
                          original_attempt=j['original_attempt'], original_attempt_id=j['original_attempt_id'],
                          complete_research_spec=ResearchRunSpec.model_validate(j['complete_research_spec'])) for j in candidate['jobs']])
    validate_amendment(result, root=root)
    return result


def execute_recovery(amendment, output, *, workers=4):
    validate_amendment(amendment)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    durable_json(output / 'amendment.json', amendment.model_dump(mode='json'))
    durable_json(output / 'freeze_receipt.json', {
        'amendment_hash': research_hash(amendment), 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'source_hash': amendment.source_hash, 'original_contract_mutated': False,
        'selection_candidate_sha256': amendment.pre_result_candidate.sha256,
        'new_attempts': 94, 'global_unique_scientific_runs': 768, 'global_maximum_attempts': 862})
    ledger = BudgetLedger(output / 'budget.json', BudgetSpec(
        llm_calls=0, input_tokens=0, output_tokens=0, unique_market_runs=94, max_attempts=94,
        candidate_submissions=0, information_queries=0, messages=0, rounds=0,
        confirmation_batches=0, wall_timeout_seconds=7200.0), owner=VERSION, episode='administrative-recovery')
    result = JobCoordinator(output / 'jobs', ledger).execute(
        [Job(job_id=j.job_id, spec=j.complete_research_spec) for j in amendment.jobs], workers=workers)
    durable_json(output / 'execution_receipt.json', {
        'status': result['status'], 'amendment_hash': research_hash(amendment),
        'planned_attempts': 94, 'terminal_records': len(result['results']),
        'committed': sum(r['status'] == 'COMMITTED' for r in result['results']),
        'budget': ledger.snapshot(), 'original_terminal_records_preserved': True})
    return result


def combine_recovered_analysis(amendment, recovery_directory):
    """Keep original incomplete estimates and create a separately named analysis."""
    original = validate_amendment(amendment)
    out = Path(recovery_directory)
    target = out / 'recovered_qualification_report.json'
    if target.exists():
        raise ValueError('RECOVERED_REPORT_ALREADY_EXISTS')
    old_path = Path(amendment.original_directory) / 'qualification_report.json'
    old = strict_json_loads(old_path.read_text())
    assert old['status'] == 'INCOMPLETE' and old['committed_runs'] == 674
    batches = sorted((out / 'jobs').glob('batch-attempt-*.json'))
    if len(batches) != 1:
        raise ValueError('RECOVERY_MUST_HAVE_EXACTLY_ONE_BATCH_ATTEMPT')
    batch = strict_json_loads(batches[0].read_text())
    rows_by_id = {r['job_id']: r for r in old['runs']}
    links = []
    for row in batch['results']:
        prior = rows_by_id[row['job_id']]
        if prior['status'] != 'INTERRUPTED':
            raise ValueError('CANNOT_REPLACE_COMMITTED_RECORD')
        if row['status'] == 'COMMITTED':
            archive = Path(row['archive']); receipt = verify_archive_v3(archive)
            job = next(j for j in amendment.jobs if j.job_id == row['job_id'])
            if receipt['source_hash'] != amendment.source_hash or row['run_key'] != job.run_key:
                raise ValueError('RECOVERED_SOURCE_OR_RUN_KEY_CHANGED')
            prior = dict(prior, status='COMMITTED', archive=str(archive),
                archive_manifest_sha256=file_sha256(archive / 'archive_manifest.json'),
                metrics=extract_observations(archive), administrative_recovery=True)
            rows_by_id[row['job_id']] = prior
        links.append({'job_id': row['job_id'], 'original_status': 'INTERRUPTED',
                      'recovery_status': row['status'], 'recovery_attempt': row.get('attempt'),
                      'run_key': row['run_key']})
    records = [rows_by_id[r['job_id']] for r in old['runs']]
    by_world = {rid: {} for rid in original.replication_ids}
    for row in records:
        if row['status'] == 'COMMITTED': by_world[row['replication_id']][row['case_id']] = row['metrics']
    binary = {c.criterion_id: [predicate_observation(c, by_world[r]) for r in original.replication_ids] for c in original.criteria}
    criteria = [assess_binary(c, binary[c.criterion_id], alpha=original.family_alpha / len(original.criteria),
        maximum_half_width=original.maximum_interval_half_width) for c in original.criteria]
    complete = all(r['status'] == 'COMMITTED' for r in records)
    status = ('INCOMPLETE' if not complete else 'FAIL' if any(c['status'] == 'FAIL' for c in criteria)
              else 'PASS' if all(c['status'] == 'PASS' for c in criteria) else 'INCONCLUSIVE')
    report = dict(old, version=VERSION, status=status, runs=records, criteria=criteria,
        binary_observations=binary, committed_runs=sum(r['status'] == 'COMMITTED' for r in records),
        analysis_kind='recovered_model_health_same_scientific_inputs',
        original_operational_status='INCOMPLETE', original_report_sha256=file_sha256(old_path),
        administrative_amendment_sha256=file_sha256(out / 'amendment.json'),
        original_attempt_count=768, administrative_attempt_count=94, total_attempt_count=862,
        recovery_provenance=links, scientific_certificate_issued=False, financial_online_authorized=False)
    durable_json(target, report)
    return report
