"""Prospective qualification of unchanged, previously registered synthetic markets.

This supplement measures task identification; it cannot mint an S0--S6 GateCertificate.
The frozen validation batch uses the existing Runner and immutable JobCoordinator.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from scipy.stats import beta

from core.experiments.strict_types import StrictResearchModel, strict_json_loads
from core.experiments.identity import HASH_PATTERN, file_sha256, research_hash
from core.experiments.design_config import ResearchRunSpec
from core.experiments.archive import durable_json, rows, verify_archive_v3
from core.experiments.counterfactual import with_replication, validate_pair
from core.experiments.jobs import Job, JobCoordinator
from core.experiments.research_cli import runtime_implementation_manifest
from core.validation.health import diagnostic_metrics
from core.codesign.budget import BudgetLedger, BudgetSpec

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'phase7-financial-qualification-v1'
CASES = ('normal', 'fundamental-baseline', 'fundamental-halt', 'fundamental-lp',
         'panic', 'withdrawal', 'false-intervention', 'combined',
         'repair1-baseline', 'repair1-lp', 'repair2-baseline', 'repair2-lp')
SCIENTIFIC_SOURCE = 'docs/research/cscwd/FINANCIAL_QUALIFICATION_PHASE7.md'


class QualificationCase(StrictResearchModel):
    case_id: str
    original_path: str
    original_sha256: str = Field(pattern=HASH_PATTERN)
    original_field: Literal['whole', 'baseline', 'treatment']
    spec: ResearchRunSpec


class IdentificationCriterion(StrictResearchModel):
    criterion_id: str
    predicate: Literal['normal_health', 'shock_direction', 'panic_market_response',
        'withdrawal_market_response', 'halt_cost', 'false_activation', 'combined_response',
        'lp_funded_trade', 'lp_market_tradeoff', 'repair_price_direction']
    case_id: str
    reference_case: str | None
    acceptance_lower: float | None
    acceptance_upper: float | None
    rationale: str = Field(min_length=10)

    @model_validator(mode='after')
    def check_interval(self):
        if self.acceptance_lower is None and self.acceptance_upper is None:
            raise ValueError('Explicit identification requirement required')
        if any(x is not None and not 0 <= x <= 1 for x in (self.acceptance_lower, self.acceptance_upper)):
            raise ValueError('Probability bounds required')
        if self.acceptance_lower is not None and self.acceptance_upper is not None and self.acceptance_lower > self.acceptance_upper:
            raise ValueError('Reversed interval')
        return self


class QualificationPlan(StrictResearchModel):
    schema_version: Literal[1] = 1
    version: Literal['phase7-financial-qualification-v1'] = VERSION
    purpose: Literal['independent_synthetic_task_qualification'] = 'independent_synthetic_task_qualification'
    empirical_calibration: Literal['NOT_PERFORMED'] = 'NOT_PERFORMED'
    source_hash: str = Field(pattern=HASH_PATTERN)
    analysis_files: dict[str, str]
    cases: list[QualificationCase]
    replication_ids: list[str]
    criteria: list[IdentificationCriterion]
    family_alpha: float = Field(gt=0.0, lt=1.0)
    maximum_interval_half_width: float = Field(gt=0.0, lt=0.5)
    planned_unique_runs: int = Field(ge=1)
    maximum_attempts: int = Field(ge=1)
    wall_timeout_seconds: float = Field(gt=0.0)
    stop_rule: Literal['fixed_batch_all_results_no_model_or_seed_repair'] = 'fixed_batch_all_results_no_model_or_seed_repair'

    @model_validator(mode='after')
    def check_design(self):
        names = [c.case_id for c in self.cases]
        if tuple(names) != CASES:
            raise ValueError('All original and preexisting structural conditions are required')
        if len(self.replication_ids) != len(set(self.replication_ids)) or not self.replication_ids:
            raise ValueError('Duplicate or absent independent replications')
        if any(not r.startswith('validation/phase7-financial-qualification-v1/') for r in self.replication_ids):
            raise ValueError('Fresh validation namespace required')
        if self.planned_unique_runs != len(self.replication_ids) * len(self.cases):
            raise ValueError('Full fixed schedule must be budgeted')
        if self.maximum_attempts < self.planned_unique_runs:
            raise ValueError('Attempt budget cannot omit planned trials')
        if len({c.criterion_id for c in self.criteria}) != len(self.criteria) or not self.criteria:
            raise ValueError('Complete unique criterion family required')
        if any(c.case_id not in names or (c.reference_case is not None and c.reference_case not in names) for c in self.criteria):
            raise ValueError('Unknown criterion case')
        return self


def original_semantic(spec):
    body = spec.model_dump(mode='json')
    body.pop('data_phase')
    body['random'].pop('replication_id')
    return body


def validate_plan(plan: QualificationPlan, root=ROOT):
    """Reject changed science, source, development relabeling and invalid precision."""
    root = Path(root)
    required_files = {'core/research_analysis/financial_qualification.py',
                      'scripts/research/qualify_financial_phase7.py', SCIENTIFIC_SOURCE}
    if set(plan.analysis_files) != required_files:
        raise ValueError('COMPLETE_ANALYSIS_SOURCE_BINDING_REQUIRED')
    if plan.criteria != registered_criteria() or plan.family_alpha != 0.05 or plan.maximum_interval_half_width != 0.2:
        raise ValueError('REGISTERED_CRITERIA_CHANGED')
    expected_ids = [f'validation/phase7-financial-qualification-v1/world-{i:03d}' for i in range(64)]
    if plan.replication_ids != expected_ids or plan.maximum_attempts != 768:
        raise ValueError('REGISTERED_FIXED_BATCH_CHANGED')
    for path, digest in plan.analysis_files.items():
        if file_sha256(root / path) != digest:
            raise ValueError('QUALIFICATION_ANALYSIS_CHANGED: ' + path)
    for case in plan.cases:
        source = root / case.original_path
        if file_sha256(source) != case.original_sha256:
            raise ValueError('ORIGINAL_SPEC_CHANGED: ' + case.case_id)
        original = strict_json_loads(source.read_text())
        if case.original_field != 'whole':
            original = original[case.original_field]
        original = ResearchRunSpec.model_validate(original)
        if original_semantic(case.spec) != original_semantic(original):
            raise ValueError('MODEL_OR_EVENT_REPAIR_FORBIDDEN: ' + case.case_id)
        if case.spec.data_phase != 'validation':
            raise ValueError('VALIDATION_PHASE_REQUIRED')
        if runtime_implementation_manifest(case.spec).source_hash != plan.source_hash:
            raise ValueError('MARKET_IMPLEMENTATION_CHANGED')
    worst = max((exact_probability_interval(k, len(plan.replication_ids), plan.family_alpha / len(plan.criteria))[1]
                 - exact_probability_interval(k, len(plan.replication_ids), plan.family_alpha / len(plan.criteria))[0]) / 2
                for k in range(len(plan.replication_ids) + 1))
    if worst > plan.maximum_interval_half_width:
        raise ValueError('REGISTERED_SAMPLE_CANNOT_REACH_PRECISION')
    return {'status': 'VERIFIED', 'worst_case_half_width': worst,
            'independent_units': len(plan.replication_ids), 'planned_unique_runs': plan.planned_unique_runs}


def exact_probability_interval(k, n, alpha):
    if type(k) is not int or type(n) is not int or n < 1 or not 0 <= k <= n or not 0 < alpha < 1:
        raise ValueError('Exact binomial interval requires integer observations')
    return (0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)))


def assess_binary(criterion, observations, *, alpha, maximum_half_width):
    """Unknown observations widen the interval over all possible completions."""
    if not observations or any(x is not None and type(x) is not bool for x in observations):
        raise ValueError('Binary observations with explicit unknowns required')
    n, k, missing = len(observations), sum(x is True for x in observations), sum(x is None for x in observations)
    lo = exact_probability_interval(k, n, alpha)[0]
    hi = exact_probability_interval(k + missing, n, alpha)[1]
    half = (hi - lo) / 2
    lower, upper = criterion.acceptance_lower, criterion.acceptance_upper
    if (lower is not None and hi < lower) or (upper is not None and lo > upper):
        status = 'FAIL'
    elif not missing and half <= maximum_half_width and (lower is None or lo >= lower) and (upper is None or hi <= upper):
        status = 'PASS'
    else:
        status = 'INCONCLUSIVE'
    return dict(criterion_id=criterion.criterion_id, status=status, n=n, successes=k, unknown=missing,
                estimate=k / n if not missing else None, interval=[lo, hi], half_width=half,
                alpha=alpha, acceptance=[lower, upper], rationale=criterion.rationale)


def extract_observations(archive):
    root = Path(archive)
    spec = ResearchRunSpec.model_validate_json((root / 'research_spec.json').read_text())
    m = diagnostic_metrics(root)
    observed = rows(root / 'measurements.jsonl')
    post = [r for r in observed if r['step'] >= 10]
    quoted = [r['metrics']['price_proxy']['value'] for r in post if r['metrics']['price_proxy']['value'] is not None]
    m['post10_mean_mid'] = math.fsum(quoted) / len(quoted) if quoted else None
    m['post10_quoted_steps'] = len(quoted)
    m['funding_cap'] = (spec.policy.liquidity.cash_cap_ppm / 1_000_000 * spec.world.initial_total_trader_wealth
                        if spec.policy.liquidity is not None else 0.0)
    m['post10_quotes'] = {str(r['step']): r['metrics']['price_proxy']['value'] for r in post}
    return m


def predicate_observation(criterion, metrics):
    a = metrics.get(criterion.case_id)
    b = metrics.get(criterion.reference_case) if criterion.reference_case else None
    if a is None or (criterion.reference_case and b is None):
        return None
    kind = criterion.predicate
    if kind == 'normal_health':
        return a['quote_coverage'] >= 0.5 and a['trading_step_fraction'] >= 3 / 30 and a['halt_fraction'] == 0.0
    if kind in {'shock_direction', 'repair_price_direction', 'combined_response'}:
        if a['post10_mean_mid'] is None or b['post10_mean_mid'] is None:
            return None
        direction = b['post10_mean_mid'] - a['post10_mean_mid'] >= 2.5
        return direction and (kind != 'combined_response' or 1 <= round(a['halt_fraction'] * 30) <= 4)
    if kind == 'panic_market_response':
        # A one-clock service difference or ten basis points in price is visible
        # at the model's 30-clock horizon and tick resolution.
        rate = (abs(a['quote_coverage'] - b['quote_coverage']) >= 1 / 30 - 1e-12
                or abs(a['trading_step_fraction'] - b['trading_step_fraction']) >= 1 / 30 - 1e-12)
        if a['post10_mean_mid'] is None or b['post10_mean_mid'] is None:
            return True if rate else None
        return rate or abs(a['post10_mean_mid'] - b['post10_mean_mid']) >= 0.1
    if kind == 'withdrawal_market_response':
        return b['quote_coverage'] - a['quote_coverage'] >= 1 / 30 - 1e-12
    if kind == 'halt_cost':
        return 1 <= round(a['halt_fraction'] * 30) <= 4
    if kind == 'false_activation':
        return a['halt_fraction'] > 0 or a['lp_trade_count'] > 0 or a['lp_buy_spend'] > 0
    if kind == 'lp_funded_trade':
        return a['lp_trade_count'] > 0 and 0 < a['lp_buy_spend'] <= a['funding_cap'] + 1e-9
    if kind == 'lp_market_tradeoff':
        # Require a changed service outcome alongside actual finite expenditure.
        # Either sign qualifies; a benefit is never imposed as the test result.
        changed = (abs(a['quote_coverage'] - b['quote_coverage']) >= 1 / 30 - 1e-12
                   or abs(a['trading_step_fraction'] - b['trading_step_fraction']) >= 1 / 30 - 1e-12)
        return a['lp_trade_count'] > 0 and 0 < a['lp_buy_spend'] <= a['funding_cap'] + 1e-9 and changed
    raise ValueError('Unknown registered predicate')


def _criterion(name, kind, case, ref, lo, hi, reason):
    return IdentificationCriterion(criterion_id=name, predicate=kind, case_id=case, reference_case=ref,
                                  acceptance_lower=lo, acceptance_upper=hi, rationale=reason)


def registered_criteria():
    criteria = [
        _criterion('normal-health', 'normal_health', 'normal', None, 0.8, None,
                   'At least 80% of worlds must support quotes during half the horizon and trades on at least three distinct clocks; this excludes a dead-book task.'),
        _criterion('shock-direction', 'shock_direction', 'fundamental-baseline', 'normal', 0.8, None,
                   'A 5% fundamental decrease must move the quoted post-event mean by at least half its magnitude (2.5 price units) in at least 80% of paired worlds.'),
        _criterion('panic-identification', 'panic_market_response', 'panic', 'normal', 0.5, None,
                   'Panic must affect a service rate by at least one clock or a quoted price by ten basis points in a majority of paired worlds; internal sentiment alone is not task identification.'),
        _criterion('withdrawal-identification', 'withdrawal_market_response', 'withdrawal', 'normal', 0.5, None,
                   'Removing normal market makers must remove at least one quoted clock in a majority of paired worlds.'),
        _criterion('halt-opportunity-cost', 'halt_cost', 'fundamental-halt', None, 0.8, None,
                   'A stress-triggered halt must consume one through four market clocks in at least 80% of worlds, establishing the clock service cost under the frozen cap.'),
        _criterion('normal-false-activation', 'false_activation', 'false-intervention', None, None, 0.1,
                   'Normal-world intervention expenditure or halting may occur in at most 10% of independent worlds; rare legitimate noisy triggers remain visible.'),
        _criterion('combined-identification', 'combined_response', 'combined', 'normal', 0.8, None,
                   'The registered combined case must exhibit the half-shock price response and capped halt opportunity cost jointly in at least 80% of worlds.'),
    ]
    for suffix, case, ref in [('original', 'fundamental-lp', 'fundamental-baseline'),
                             ('delay', 'repair1-lp', 'repair1-baseline'),
                             ('early', 'repair2-lp', 'repair2-baseline')]:
        criteria.extend([
            _criterion('lp-funded-' + suffix, 'lp_funded_trade', case, None, 0.1, None,
                       'Funded liquidity must execute positive cash-funded trades within the original cap in at least one in ten worlds; merely posting an unused quote cannot identify the funding mechanism.'),
            _criterion('lp-tradeoff-' + suffix, 'lp_market_tradeoff', case, ref, 0.1, None,
                       'At least one in ten paired worlds must show actual bounded funding and a one-clock quote/trading service change, of either sign; unchanged outcomes cannot identify the service-resource tradeoff.'),
        ])
    for n in (1, 2):
        criteria.append(_criterion(f'structural-price-{n}', 'repair_price_direction', f'repair{n}-baseline', 'normal', 0.8, None,
                                   'The two previously registered structural conditions are assessed separately for the same half-shock response; they are not extra independent replications or newly tuned models.'))
    return criteria


def build_plan(*, root=ROOT):
    root = Path(root)
    cases = []
    for name in CASES:
        if name.startswith('repair'):
            n = int(name[6]); field = 'baseline' if name.endswith('baseline') else 'treatment'
            path = f'artifacts/research/phase_03/s4-repair-{n:02d}/plan-before-execution.json'
            value = strict_json_loads((root / path).read_text())[field]
        else:
            field = 'whole'; path = f'examples/research/health/specs/development-{name}.json'
            value = strict_json_loads((root / path).read_text())
        spec = with_replication(ResearchRunSpec.model_validate(value), 'validation/phase7-financial-qualification-v1/template', data_phase='validation')
        cases.append(QualificationCase(case_id=name, original_path=path, original_sha256=file_sha256(root / path), original_field=field, spec=spec))
    criteria = registered_criteria()
    files = [__file__.replace(str(ROOT) + '/', ''), 'scripts/research/qualify_financial_phase7.py', SCIENTIFIC_SOURCE]
    plan = QualificationPlan(source_hash=runtime_implementation_manifest(cases[0].spec).source_hash,
        analysis_files={p: file_sha256(root / p) for p in files}, cases=cases,
        replication_ids=[f'validation/phase7-financial-qualification-v1/world-{i:03d}' for i in range(64)],
        criteria=criteria, family_alpha=0.05, maximum_interval_half_width=0.2,
        planned_unique_runs=64 * len(cases), maximum_attempts=64 * len(cases), wall_timeout_seconds=7200.0)
    validate_plan(plan, root)
    return plan


def execute_qualification(plan, output, *, workers=4, root=ROOT):
    root, output = Path(root), Path(output)
    checked = validate_plan(plan, root)
    output.mkdir(parents=True, exist_ok=False)
    durable_json(output / 'plan.json', plan.model_dump(mode='json'))
    freeze = dict(version=VERSION, plan_hash=research_hash(plan), frozen_at=datetime.now(timezone.utc).isoformat(),
                  verification=checked, analysis_files=plan.analysis_files, source_hash=plan.source_hash,
                  validation_exposure='NONE_BEFORE_FREEZE', historical_models_unchanged=True,
                  stop_rule=plan.stop_rule)
    durable_json(output / 'freeze_receipt.json', freeze)
    budget = BudgetSpec(llm_calls=0, input_tokens=0, output_tokens=0, unique_market_runs=plan.planned_unique_runs,
        max_attempts=plan.maximum_attempts, candidate_submissions=0, information_queries=0, messages=0,
        rounds=0, confirmation_batches=0, wall_timeout_seconds=plan.wall_timeout_seconds)
    ledger = BudgetLedger(output / 'budget.json', budget, owner=VERSION, episode='independent-validation')
    jobs = [Job(job_id=f'{case.case_id}-{i:03d}', spec=with_replication(case.spec, rid))
            for i, rid in enumerate(plan.replication_ids) for case in plan.cases]
    result = JobCoordinator(output / 'jobs', ledger).execute(jobs, workers=workers)
    # Only a completed fixed batch is analyzed. Failure rows and original IDs remain.
    job_map = {r['job_id']: r for r in result['results']}
    observations = {c.criterion_id: [] for c in plan.criteria}
    run_records, pair_records = [], []
    for i, rid in enumerate(plan.replication_ids):
        metrics, archives = {}, {}
        for case in plan.cases:
            row = job_map[f'{case.case_id}-{i:03d}']
            record = {'case_id': case.case_id, 'replication_id': rid, 'status': row['status'], 'job_id': row['job_id']}
            if row['status'] == 'COMMITTED':
                archive = Path(row['archive']); verified = verify_archive_v3(archive)
                if verified['source_hash'] != plan.source_hash:
                    raise ValueError('FROZEN_MARKET_SOURCE_CHANGED_DURING_EXECUTION')
                metrics[case.case_id] = extract_observations(archive); archives[case.case_id] = archive
                record.update(archive=str(archive), archive_manifest_sha256=file_sha256(archive / 'archive_manifest.json'), metrics=metrics[case.case_id])
            run_records.append(record)
        for a, b in [('fundamental-baseline', 'fundamental-lp'), ('repair1-baseline', 'repair1-lp'), ('repair2-baseline', 'repair2-lp')]:
            if a in archives and b in archives:
                # Historical configuration labels differ. Preserve those labels;
                # exact economic pairing is checked below without relabeling inputs.
                sa = next(c.spec for c in plan.cases if c.case_id == a).model_dump(mode='json')
                sb = next(c.spec for c in plan.cases if c.case_id == b).model_dump(mode='json')
                sa.pop('policy'); sb.pop('policy')
                same = sa == sb
                pair_records.append({'replication_id': rid, 'baseline': a, 'treatment': b, 'same_nonpolicy_spec': same,
                                     'initial_world_equal': strict_json_loads((archives[a] / 'initial_world.json').read_text()) == strict_json_loads((archives[b] / 'initial_world.json').read_text())})
                if not same or not pair_records[-1]['initial_world_equal']:
                    raise ValueError('PAIR_WORLD_OR_NONPOLICY_INPUT_CHANGED')
        for criterion in plan.criteria:
            observations[criterion.criterion_id].append(predicate_observation(criterion, metrics))
    criteria = [assess_binary(c, observations[c.criterion_id], alpha=plan.family_alpha / len(plan.criteria),
                             maximum_half_width=plan.maximum_interval_half_width) for c in plan.criteria]
    complete = all(r['status'] == 'COMMITTED' for r in run_records)
    status = ('INCOMPLETE' if not complete else 'FAIL' if any(c['status'] == 'FAIL' for c in criteria)
              else 'PASS' if all(c['status'] == 'PASS' for c in criteria) else 'INCONCLUSIVE')
    report = dict(version=VERSION, status=status, plan_hash=research_hash(plan), freeze_sha256=file_sha256(output / 'freeze_receipt.json'),
                  source_hash=plan.source_hash, empirical_calibration='NOT_PERFORMED', origin='simulation_output',
                  independent_worlds=len(plan.replication_ids), planned_runs=len(jobs), recorded_runs=len(run_records),
                  committed_runs=sum(r['status'] == 'COMMITTED' for r in run_records),
                  criteria=criteria, runs=run_records, paired_world_checks=pair_records,
                  binary_observations=observations, scientific_certificate_issued=False,
                  financial_online_authorized=False,
                  qualification_scope='Supplementary S4 task identification of the original four-actor models; S0-S6 certification remains a separate exact-scope requirement.',
                  budget=ledger.snapshot(), remaining_interfaces=['Frozen health profile and exact-scope S0-S6 certificate integration',
                    'Financial benchmark pilot, model-specific planning, freeze and independent evaluator',
                    'Formal finite policy search including NoOp, rules, random, enumeration and NSGA-II'])
    durable_json(output / 'qualification_report.json', report)
    return report
