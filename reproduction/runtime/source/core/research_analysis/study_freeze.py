"""F73 content-addressed analytic study freeze.

The dependency graph is AnalysisPlan -> PlanCore -> FreezeReceipt -> envelope.
This module never calls a model, changes a spend ledger, or grants financial
qualification. Its positive tests are synthetic engineering fixtures.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import os

from pydantic import Field, model_validator
from core.experiments.strict_types import StrictResearchModel, strict_json_loads, load_research_model
from core.experiments.identity import file_sha256, research_hash
from core.experiments.archive import durable_json
from core.experiments.evidence import Hash
from core.codesign.budget import BudgetSpec
from core.codesign.schema import TaskSpec, ModelCallConfig
from core.codesign.pilot import METHODS, validate_prior_independence, load_pilot_plan, inspect_conflict_design
from core.codesign.runner import design_implementation_manifest
from .analysis import PlanningDesign, load_completed_pilot

FREEZE_RUNTIME_FILES = ('core/research_analysis/study_freeze.py',
    'core/research_analysis/formal_execution.py', 'core/research_analysis/analysis.py',
    'scripts/research/prepare_study.py', 'scripts/research/run_benchmark.py')


class FreezeGateError(ValueError):
    """A missing scientific/resource prerequisite (CLI exit 3)."""


class FreezeIntegrityError(ValueError):
    """A changed bound artifact or inconsistent graph (CLI exit 5)."""


class FileRef(StrictResearchModel):
    path: str = Field(min_length=1)
    sha256: Hash


class BoundInput(FileRef):
    role: Literal['source', 'research_document', 'prompt', 'method_config', 'dependency_lock']


class AnalysisPlan(StrictResearchModel):
    schema_version: Literal[1]
    analysis_id: str = Field(min_length=1)
    study_stage: Literal['formal']
    planning_design: PlanningDesign
    primary_outcome: Literal['A_cert']
    raw_method_cert_separate: Literal[True]
    failure_rule: Literal['retain_all_registered_trials_no_seed_replacement']
    missing_rule: Literal['incomplete_index_blocks_confirmatory_analysis']
    selection_rule: Literal['max_min_margin_cost_complexity_hash_v1']
    multiplicity: Literal['bonferroni_registered_four_contrasts']
    fixed_sample_no_optional_stopping: Literal[True]


class EvaluatorBudget(StrictResearchModel):
    candidate_evaluations: int = Field(ge=1)
    llm_calls: Literal[0]
    market_runs: Literal[0]
    max_attempts_per_object: Literal[1]


class AuditPlan(StrictResearchModel):
    schema_version: Literal[1]
    audit_id: str = Field(min_length=1)
    scope: Literal['exact_analytic']
    coverage: Literal['selected_only', 'frozen_shortlist']
    maximum_policies_per_episode: int = Field(ge=1)
    required_role_ids: list[str] = Field(min_length=1)
    required_model_ids: list[str] = Field(min_length=1)
    claim_scope: Literal['all_role_utilities_and_outcome_constraints']
    estimand: Literal['exact_joint_acceptability_of_frozen_selection']
    fixed_n_per_policy: Literal[0]
    hidden_replication_ids: list[str]
    support: Literal['losses-utilities-violations-unit-interval-v1']
    interval_rule: Literal['exact_finite_table_no_sampling_interval']
    multiplicity_rule: Literal['exact_all_registered_atoms']
    failure_rule: Literal['retain_failed_unresolved_uncovered_as_unknown']
    no_selection_rule: Literal['no_applicable_object_no_substitute_policy']
    visibility_rule: Literal['read_after_immutable_selection_never_return_to_designer']
    evaluator_budget: EvaluatorBudget

    @model_validator(mode='after')
    def _valid(self):
        if self.hidden_replication_ids:
            raise ValueError('EXACT_ANALYTIC_AUDIT_HAS_NO_MARKET_REPLICATIONS')
        if self.coverage == 'selected_only' and self.maximum_policies_per_episode != 1:
            raise ValueError('SELECTED_ONLY_REQUIRES_K_ONE')
        for ids in (self.required_role_ids, self.required_model_ids):
            if len(ids) != len(set(ids)):
                raise ValueError('DUPLICATE_AUDIT_SCOPE')
        return self


class AuthorizationBinding(StrictResearchModel):
    ledger: FileRef
    authorization_id: str = Field(min_length=1)
    cap_nano: int = Field(gt=0, le=100_000_000_000)
    input_price_nano: int = Field(gt=0)
    output_price_nano: int = Field(gt=0)
    cached_price_nano: int = Field(ge=0)


class PlanCoreDraft(StrictResearchModel):
    schema_version: Literal[1]
    study_stage: Literal['formal']
    scope: Literal['exact_analytic']
    plan_id: str = Field(min_length=1)
    source_hash: Hash
    scientific_inputs: list[BoundInput] = Field(min_length=1)
    tasks: list[FileRef] = Field(min_length=1)
    split_manifest: FileRef
    prior_split_manifests: list[FileRef] = Field(min_length=1)
    exposure_registry: FileRef
    model: ModelCallConfig
    methods: list[str]
    protocol_version: Literal['bounded_llm_protocol_v2', 'bounded_llm_protocol_v3']
    design_turns: int = Field(ge=3, le=12)
    design_budget: BudgetSpec
    budget_axis: Literal['fixed_tokens']
    audit_plan: AuditPlan
    authorization: AuthorizationBinding
    stop_rule: Literal['unknown_usage_or_identity_or_three_consecutive_failures_v1']
    llm_repeats_per_method_task: Literal[1]
    market_repeats_per_policy: Literal[0]

    @model_validator(mode='after')
    def _valid(self):
        if len(self.methods) != len(METHODS) or set(self.methods) != set(METHODS):
            raise ValueError('FORMAL_METHOD_SET_MISMATCH')
        if self.model.source != 'live' or self.model.max_request_attempts != 1:
            raise ValueError('LIVE_EXPLICIT_MODEL_WITHOUT_RETRIES_REQUIRED')
        if len({r.path for r in self.tasks}) != len(self.tasks):
            raise ValueError('DUPLICATE_FORMAL_TASK')
        if {r.role for r in self.scientific_inputs} != {'source', 'research_document', 'prompt', 'method_config', 'dependency_lock'}:
            raise ValueError('INCOMPLETE_SCIENTIFIC_INPUT_BINDING')
        return self


class FormalTrial(StrictResearchModel):
    episode_id: str
    task_id: str
    method: str
    rq_ids: list[Literal['RQ1', 'RQ2', 'RQ3']]
    status: Literal['REGISTERED']


class PlanCore(PlanCoreDraft):
    analysis_plan: FileRef
    candidate_plan: FileRef
    planning_receipt: FileRef
    schedule: list[FormalTrial] = Field(min_length=1)


class ExposureEvent(StrictResearchModel):
    stage: Literal['development', 'validation', 'diagnostic', 'pilot', 'formal']
    task_id: str
    underlying_cluster_id: str
    analytic_content_hash: Hash


class ExposureRegistry(StrictResearchModel):
    schema_version: Literal[1]
    registry_id: str = Field(min_length=1)
    events: list[ExposureEvent]
    evidence_files: list[FileRef]
    coverage_rule: Literal['all_prior_registered_design_and_evaluation_outputs']


class FreezeReceipt(StrictResearchModel):
    schema_version: Literal[1]
    status: Literal['FROZEN']
    created_at_utc: str
    plan_core: FileRef
    analysis_plan: FileRef
    candidate_plan: FileRef
    planning_receipt: FileRef
    spend_snapshot: FileRef
    input_hashes: dict[str, Hash]
    checks: dict[str, Literal[True]]
    task_count: int = Field(ge=1)
    trial_count: int = Field(ge=1)
    maximum_additional_reservation_nano: int = Field(gt=0)
    remaining_reservation_nano: int = Field(ge=0)
    scope: Literal['exact_analytic']
    financial_execution_permitted: Literal[False]


class ExecutionEnvelope(StrictResearchModel):
    schema_version: Literal[1]
    study_stage: Literal['formal']
    plan_id: str
    plan_core: FileRef
    analysis_plan: FileRef
    freeze_receipt: FileRef


def reference(path):
    path = Path(path).resolve()
    return FileRef(path=str(path), sha256=file_sha256(path))


def check_ref(ref):
    path = Path(ref.path)
    if not path.is_absolute():
        raise ValueError('ABSOLUTE_BOUND_PATH_REQUIRED')
    if not path.is_file() or file_sha256(path) != ref.sha256:
        raise FreezeIntegrityError('BOUND_CONTENT_CHANGED_OR_MISSING:' + str(path))
    return path


def _load_candidate(path, *, require_recommended=True):
    """F72 provenance is validated again; an edited candidate is not accepted."""
    path = Path(path).resolve()
    candidate = strict_json_loads(path.read_text())
    receipt_path = path.parent / 'power_precision_receipt.json'
    receipt = strict_json_loads(receipt_path.read_text())
    if receipt.get('artifact_hashes', {}).get(str(path)) != file_sha256(path):
        raise FreezeIntegrityError('CANDIDATE_NOT_BOUND_BY_PLANNING_RECEIPT')
    for group in ('input_hashes', 'artifact_hashes'):
        for name, sha in receipt[group].items():
            check_ref(FileRef(path=name, sha256=sha))
    design_path = path.parent / 'design.json'
    design = load_research_model(design_path, PlanningDesign)
    indexes = [p for p in receipt['input_hashes'] if Path(p).name == 'benchmark_index.json']
    designs = [p for p in receipt['input_hashes'] if Path(p).name != 'design.json'
               and Path(p).suffix in ('.json', '.yaml') and Path(p).name not in ('benchmark_index.json', 'COMPLETED.json', 'registration.json')]
    registered = [p for p in designs if strict_json_loads(Path(p).read_text()).get('design_id') == design.design_id]
    if len(indexes) != 1 or len(registered) != 1:
        raise FreezeIntegrityError('PLANNING_PILOT_DESIGN_BINDINGS_REQUIRED')
    plan, tasks, index, *_ = load_completed_pilot(indexes[0], registered[0])
    if candidate['pilot_plan_id'] != plan.plan_id or candidate['source_hash'] != index['source_hash']:
        raise FreezeIntegrityError('CANDIDATE_PILOT_IDENTITY_MISMATCH')
    if candidate['pilot_independent_tasks'] != len(tasks) or candidate['design_id'] != design.design_id:
        raise FreezeIntegrityError('CANDIDATE_DESIGN_IDENTITY_MISMATCH')
    n = candidate.get('recommended_balanced_task_count')
    if require_recommended and (type(n) is not int or n < 12 or n % 3):
        raise FreezeGateError('NO_FEASIBLE_CANDIDATE_SAMPLE_SIZE')
    if candidate['recommended_llm_repeats_per_method_task'] != 1 or candidate['recommended_market_repeats_per_policy'] != 0:
        raise FreezeGateError('UNSUPPORTED_PLANNING_SCOPE')
    if design.model_dump(mode='json') != load_research_model(registered[0], PlanningDesign).model_dump(mode='json'):
        raise FreezeIntegrityError('PLANNING_THRESHOLDS_CHANGED')
    return candidate, design, receipt_path


def preflight_candidate_resources(candidate_path, ledger_path, output=None):
    """Read-only F73 resource admission before generating any held-out tasks.

    This records a gate diagnosis, never PlanCore, a freeze receipt, or permission
    to execute. It recomputes full-cap costs from the already registered model
    and actual cumulative ledger, rather than trusting F72's historical balance.
    """
    candidate, design, receipt_path = _load_candidate(candidate_path, require_recommended=False)
    planning_receipt = strict_json_loads(Path(receipt_path).read_text())
    index_path = next(Path(p) for p in planning_receipt['input_hashes'] if Path(p).name == 'benchmark_index.json')
    plan = load_pilot_plan(index_path.parent / 'plan.json')
    pilot_index = strict_json_loads(index_path.read_text())
    tasks = [load_research_model(p, TaskSpec) for p in plan.task_paths]
    if len({len(t.required_signers) for t in tasks}) != 1:
        raise FreezeGateError('PLANNING_CALL_SCHEDULE_NOT_UNIFORM')
    wrapper = strict_json_loads(Path(ledger_path).read_text()); state = wrapper['state']
    if research_hash(state) != wrapper['sha256']:
        raise FreezeIntegrityError('SPEND_HASH_MISMATCH')
    contract = state['contract']
    expected = {'authorization_id': plan.authorization_id, 'cap_nano': plan.monetary_cap_cny * 10**9,
                'input_nano': plan.input_price_nano, 'output_nano': plan.output_price_nano,
                'cached_nano': plan.cached_price_nano, 'currency': 'CNY',
                'reservation_rule': 'peak-price-full-cap-no-refund-v1'}
    if contract != expected or contract != pilot_index['spend_snapshot']['contract']:
        raise FreezeIntegrityError('CUMULATIVE_AUTHORIZATION_CHANGED')
    previous = pilot_index['spend_snapshot']
    for group in ('reservations', 'settlements'):
        if any(state[group].get(k) != value for k, value in previous[group].items()):
            raise FreezeIntegrityError('PRIOR_SPEND_HISTORY_CHANGED')
    if set(state['reservations']) != set(state['settlements']):
        raise FreezeGateError('SPEND_UNSETTLED_CALLS')
    calls = plan.design_turns + len(tasks[0].required_signers)
    per_task = (len(plan.methods) - 1) * calls * (plan.model.input_reservation_tokens * contract['input_nano']
                                               + plan.model.max_output_tokens * contract['output_nano'])
    reserved = sum(r['reserved_nano'] for r in state['reservations'].values())
    remaining = contract['cap_nano'] - reserved
    n = candidate.get('recommended_balanced_task_count')
    minimum_n = min(design.candidate_task_counts)
    blockers = []
    if type(n) is not int or n < minimum_n or n % design.strata:
        blockers.append('NO_RECOMMENDED_SAMPLE_SIZE_MEETS_REGISTERED_POWER_AND_PRECISION')
        required = None
    else:
        required = n * per_task
        if required > remaining:
            blockers.append('RECOMMENDED_FORMAL_DESIGN_EXCEEDS_REMAINING_AUTHORIZATION')
    if minimum_n * per_task > remaining:
        blockers.append('EVEN_MINIMUM_REGISTERED_TASK_GRID_EXCEEDS_REMAINING_AUTHORIZATION')
    if state['blocked_reason']:
        blockers.append('SPEND_BLOCKED:' + state['blocked_reason'])
    if design_implementation_manifest().source_hash != candidate['source_hash']:
        blockers.append('CURRENT_DESIGN_SOURCE_DIFFERS_FROM_PLANNING_PILOT')
    result = {'schema_version': 1, 'stage': 'F73_resource_preflight',
        'status': 'BLOCKED' if blockers else 'RESOURCE_PRECHECK_PASSED',
        'candidate_plan': reference(candidate_path).model_dump(mode='json'),
        'planning_receipt': reference(receipt_path).model_dump(mode='json'),
        'cumulative_ledger': reference(ledger_path).model_dump(mode='json'),
        'pilot_plan': reference(index_path.parent / 'plan.json').model_dump(mode='json'),
        'authorization_id': contract['authorization_id'], 'cap_nano': contract['cap_nano'],
        'cumulative_reserved_nano': reserved, 'remaining_reservation_nano': remaining,
        'per_independent_task_full_cap_nano': per_task, 'recommended_task_count': n,
        'recommended_additional_reservation_nano': required,
        'minimum_registered_task_count': minimum_n,
        'minimum_grid_additional_reservation_nano': minimum_n * per_task,
        'maximum_additional_complete_tasks': max(0, remaining // per_task),
        'maximum_additional_balanced_tasks': max(0, design.strata * ((remaining // per_task) // design.strata)),
        'blockers': blockers, 'formal_freeze_created': False, 'formal_execution_permitted': False,
        'generated_heldout_tasks': 0, 'llm_generation_calls': 0,
        'remaining_checks': ['untouched held-out inputs', 'complete scientific bindings',
                             'independent evaluator coverage/budget', 'actual model preflight']}
    if output is not None:
        target = Path(output)
        if target.exists():
            raise FreezeIntegrityError('IMMUTABLE_PREFLIGHT_OUTPUT_ALREADY_EXISTS')
        target.parent.mkdir(parents=True, exist_ok=True)
        durable_json(target, result)
        os.chmod(target, 0o444)
    return result


def _validate_inputs(draft, analysis, candidate, design):
    if analysis.planning_design != design:
        raise FreezeIntegrityError('ANALYSIS_CHANGED_REGISTERED_DESIGN')
    if draft.source_hash != candidate['source_hash']:
        raise FreezeIntegrityError('FORMAL_SOURCE_CHANGED_FROM_PLANNING_PILOT')
    if [c.id for c in design.contrasts] != ['full_vs_single', 'factor_S', 'factor_V', 'interaction_SV']:
        raise FreezeGateError('REGISTERED_FOUR_CONTRASTS_REQUIRED')
    if len(draft.tasks) < candidate['recommended_balanced_task_count']:
        raise FreezeGateError('INSUFFICIENT_INDEPENDENT_TASK_COUNT')
    manifest = design_implementation_manifest()
    if manifest.source_hash != draft.source_hash:
        raise FreezeIntegrityError('SOURCE_HASH_CHANGED')
    refs = [*draft.scientific_inputs, *draft.tasks, draft.split_manifest, *draft.prior_split_manifests,
            draft.exposure_registry, draft.authorization.ledger]
    for ref in refs:
        check_ref(ref)
    if len({Path(r.path).resolve() for r in draft.scientific_inputs}) != len(draft.scientific_inputs):
        raise ValueError('DUPLICATE_SCIENTIFIC_INPUT')
    source_hashes = {Path(x.path).resolve(): x.sha256 for x in draft.scientific_inputs if x.role in ('source', 'dependency_lock')}
    root = Path(__file__).resolve().parents[2]
    if any(source_hashes.get(root / x.path) != x.sha256 for x in manifest.files):
        raise FreezeIntegrityError('EXECUTABLE_SOURCE_CLOSURE_NOT_BOUND')
    for name in FREEZE_RUNTIME_FILES:
        if not (root / name).is_file():
            raise FreezeGateError('MISSING_FORMAL_EXECUTABLE:' + name)
        if source_hashes.get(root / name) != file_sha256(root / name):
            raise FreezeIntegrityError('FORMAL_EXECUTABLE_NOT_BOUND:' + name)
    splits = strict_json_loads(Path(draft.split_manifest.path).read_text())
    # Reuse the existing provenance exclusion rule for all previously consumed splits.
    priors = [strict_json_loads(Path(x.path).read_text()) for x in draft.prior_split_manifests]
    validate_prior_independence(splits, priors)
    entries = splits['tasks']
    if {x['split'] for x in entries} != {'development', 'validation', 'test', 'ood'}:
        raise ValueError('ALL_FOUR_SPLITS_MUST_BE_REGISTERED')
    for key in ('task_id', 'underlying_cluster_id', 'analytic_content_hash', 'replication_namespace'):
        values = [x[key] for x in entries]
        if len(values) != len(set(values)):
            raise FreezeIntegrityError('SPLIT_INDEPENDENCE_VIOLATION:' + key)
    dependencies = [d for x in entries for d in x['source_dependencies']]
    if len(dependencies) != len(set(dependencies)):
        raise FreezeIntegrityError('SHARED_SPLIT_SOURCE_DEPENDENCY')
    # Bind all four split contents, including currently unexecuted validation/OOD.
    for entry in entries:
        ref = FileRef(path=entry['task_path'], sha256=entry['sha256'])
        check_ref(ref)
        refs.append(ref)
    exposure = load_research_model(draft.exposure_registry.path, ExposureRegistry)
    for ref in exposure.evidence_files:
        check_ref(ref)
    # A missing registry cannot be replaced by a free-form "unexposed=true" flag.
    if not exposure.evidence_files:
        raise FreezeGateError('EXPOSURE_REGISTRY_EVIDENCE_REQUIRED')
    event_ids = {e.task_id for e in exposure.events}
    for ref in exposure.evidence_files:
        evidence = strict_json_loads(Path(ref.path).read_text())
        if 'episodes' not in evidence or 'study_stage' not in evidence:
            raise FreezeIntegrityError('EXPOSURE_EVIDENCE_MUST_BE_BENCHMARK_INDEX')
        observed = {e['task_id'] for e in evidence['episodes'] if e.get('status') != 'NOT_RUN'}
        if not observed <= event_ids:
            raise FreezeIntegrityError('EXPOSURE_REGISTRY_OMITS_RECORDED_TASK')
    tasks = [load_research_model(x.path, TaskSpec) for x in draft.tasks]
    indexed = {x['task_id']: x for x in entries}
    strata = []
    for task, ref in zip(tasks, draft.tasks):
        entry = indexed.get(task.task_id)
        if task.split != 'test' or not entry or entry['split'] != 'test':
            raise FreezeGateError('FORMAL_HELDOUT_TEST_ONLY')
        if entry['task_hash'] != task.task_hash or entry['sha256'] != ref.sha256 or entry['analytic_content_hash'] != research_hash(task.analytic_outcomes):
            raise FreezeIntegrityError('FORMAL_TASK_SPLIT_CONTENT_MISMATCH')
        if entry['underlying_cluster_id'] != task.underlying_cluster_id:
            raise FreezeIntegrityError('FORMAL_CLUSTER_MISMATCH')
        if any(e.task_id == task.task_id or e.underlying_cluster_id == task.underlying_cluster_id
               or e.analytic_content_hash == entry['analytic_content_hash'] for e in exposure.events):
            raise FreezeGateError('HELDOUT_INPUT_ALREADY_EXPOSED')
        if task.origin != 'analytic_fixture' or task.evidence_mode != 'exact_analytic':
            raise FreezeGateError('FINANCIAL_SCOPE_REQUIRES_UNIMPLEMENTED_HEALTH_GATES')
        if draft.protocol_version == 'bounded_llm_protocol_v3':
            validate_v3_population(task)
        if task.model_configurations != [draft.model] or task.budget != draft.design_budget:
            raise FreezeIntegrityError('TASK_MODEL_OR_BUDGET_MISMATCH')
        if task.signing_layer != 'delegated_explicit_llm_v1':
            raise FreezeGateError('FORMAL_LIVE_SIGNERS_REQUIRED')
        if set(task.required_signers) != {r.role_id for r in task.roles}:
            raise FreezeGateError('ALL_REGISTERED_ROLE_OWNERS_MUST_SIGN')
        if task.budget.llm_calls != draft.design_turns + len(task.required_signers):
            raise FreezeGateError('TASK_CALL_SCHEDULE_BUDGET_MISMATCH')
        if set(draft.audit_plan.required_role_ids) != {r.role_id for r in task.roles} or set(draft.audit_plan.required_model_ids) != {m.model_id for m in task.model_family}:
            raise FreezeIntegrityError('AUDIT_SCOPE_MISSING_ROLE_OR_MODEL')
        strata.append(task.prompt_resources.get('stratum_index'))
    if set(strata) != {'0', '1', '2'} or len({strata.count(s) for s in set(strata)}) != 1:
        raise FreezeGateError('BALANCED_THREE_STRATUM_FORMAL_SAMPLE_REQUIRED')
    calls = draft.design_turns + len(tasks[0].required_signers)
    b, model = draft.design_budget, draft.model
    if (b.llm_calls != calls or b.input_tokens < calls * model.input_reservation_tokens
            or b.output_tokens < calls * model.max_output_tokens or b.unique_market_runs != 0
            or b.confirmation_batches != 0):
        raise FreezeGateError('DESIGN_RESOURCE_BUDGET_INSUFFICIENT_OR_WRONG_SCOPE')
    if draft.protocol_version == 'bounded_llm_protocol_v3':
        if any(b.candidate_submissions < len(t.candidate_pool) or
               b.messages < len(t.candidate_pool) * (3 + len(t.roles) * len(t.model_family) + len(t.outcome_constraints) * len(t.model_family))
               for t in tasks):
            raise FreezeGateError('SHARED_SETUP_RESOURCE_BUDGET_INSUFFICIENT')
    trials = len(tasks) * len(draft.methods)
    if draft.audit_plan.evaluator_budget.candidate_evaluations < trials * draft.audit_plan.maximum_policies_per_episode:
        raise FreezeGateError('INDEPENDENT_EVALUATOR_BUDGET_INSUFFICIENT')
    state_wrapper = strict_json_loads(Path(draft.authorization.ledger.path).read_text())
    state = state_wrapper['state']
    if research_hash(state) != state_wrapper['sha256']:
        raise FreezeIntegrityError('SPEND_HASH_MISMATCH')
    auth = draft.authorization
    expected = {'authorization_id': auth.authorization_id, 'cap_nano': auth.cap_nano,
                'input_nano': auth.input_price_nano, 'output_nano': auth.output_price_nano,
                'cached_nano': auth.cached_price_nano, 'currency': 'CNY',
                'reservation_rule': 'peak-price-full-cap-no-refund-v1'}
    if state['contract'] != expected:
        raise FreezeIntegrityError('SPEND_AUTHORIZATION_MISMATCH')
    if state['blocked_reason']:
        raise FreezeGateError('SPEND_BLOCKED:' + state['blocked_reason'])
    if set(state['reservations']) != set(state['settlements']):
        raise FreezeGateError('SPEND_UNSETTLED_CALLS')
    remaining = auth.cap_nano - sum(x['reserved_nano'] for x in state['reservations'].values())
    required = len(tasks) * (len(draft.methods) - 1) * calls * (
        model.input_reservation_tokens * auth.input_price_nano + model.max_output_tokens * auth.output_price_nano)
    if required > remaining:
        raise FreezeGateError('FORMAL_DESIGN_EXCEEDS_REMAINING_AUTHORIZATION')
    return refs + exposure.evidence_files, required, remaining


def validate_v3_population(task):
    """Check the registered conditional population, not only a stratum label."""
    result = inspect_conflict_design(task)
    stratum = result['stratum']
    mapping = {'role_conflict': ('0', 1), 'model_conflict': ('1', 1), 'infeasible': ('2', 0)}
    index, accepted = mapping[stratum]
    if (task.prompt_resources.get('stratum_index') != index or result['acceptable_count'] != accepted
            or len(task.candidate_pool) != 6):
        raise FreezeGateError('V3_REGISTERED_POPULATION_MISMATCH')
    changing = 0
    for candidate in task.candidate_pool:
        rows = [x for x in task.analytic_outcomes if x.candidate_id == candidate.candidate_id]
        if any(len({1-sum(w*x.losses.model_dump()[k] for k,w in r.weights.model_dump().items()) >= r.rho
                    for x in rows}) > 1 for r in task.roles):
            changing += 1
    if (stratum == 'role_conflict' and changing != 0) or (stratum == 'model_conflict' and changing != 2):
        raise FreezeGateError('V3_CONFLICT_TYPE_DIFFERS_FROM_REGISTERED_POPULATION')
    return result


def freeze_study(candidate_path, formal_plan_path, analysis_plan_path, output):
    """Validate every prerequisite before publishing an immutable output directory."""
    output = Path(output).resolve()
    if output.exists():
        raise FreezeIntegrityError('IMMUTABLE_FREEZE_OUTPUT_ALREADY_EXISTS')
    draft = load_research_model(formal_plan_path, PlanCoreDraft)
    analysis = load_research_model(analysis_plan_path, AnalysisPlan)
    candidate, design, planning_receipt_path = _load_candidate(candidate_path)
    planning_receipt = strict_json_loads(Path(planning_receipt_path).read_text())
    pilot_indexes = {str(Path(p).resolve()) for p in planning_receipt['input_hashes'] if Path(p).name == 'benchmark_index.json'}
    exposure = load_research_model(check_ref(draft.exposure_registry), ExposureRegistry)
    if not pilot_indexes <= {r.path for r in exposure.evidence_files}:
        raise FreezeIntegrityError('PLANNING_PILOT_MISSING_FROM_EXPOSURE_REGISTRY')
    ancestor_refs = []
    for index in pilot_indexes:
        prior_plan = strict_json_loads((Path(index).parent / 'plan.json').read_text())
        if (draft.protocol_version != prior_plan['protocol_version'] or
                draft.model.model_dump(mode='json') != prior_plan['model'] or
                draft.design_turns != prior_plan['design_turns']):
            raise FreezeIntegrityError('FORMAL_PROTOCOL_CHANGED_FROM_PLANNING_PILOT')
        if draft.protocol_version == 'bounded_llm_protocol_v3':
            spec_ref = FileRef(path=prior_plan['experiment_spec_path'], sha256=prior_plan['experiment_spec_sha256'])
            check_ref(spec_ref); ancestor_refs.append(spec_ref)
        # Enforce cumulative authorization on the actual freeze entry point too;
        # preflight is optional and cannot be a prerequisite for budget safety.
        pilot_spend = strict_json_loads(Path(index).read_text())['spend_snapshot']
        current_wrapper = strict_json_loads(check_ref(draft.authorization.ledger).read_text())
        current_spend = current_wrapper['state']
        if research_hash(current_spend) != current_wrapper['sha256']:
            raise FreezeIntegrityError('SPEND_HASH_MISMATCH')
        if (current_spend['contract'] != pilot_spend['contract'] or
                draft.authorization.authorization_id != prior_plan['authorization_id'] or
                draft.authorization.cap_nano != prior_plan['monetary_cap_cny'] * 10**9 or
                draft.authorization.input_price_nano != prior_plan['input_price_nano'] or
                draft.authorization.output_price_nano != prior_plan['output_price_nano'] or
                draft.authorization.cached_price_nano != prior_plan['cached_price_nano']):
            raise FreezeIntegrityError('CUMULATIVE_AUTHORIZATION_CHANGED')
        for group in ('reservations', 'settlements'):
            if any(current_spend[group].get(k) != v for k, v in pilot_spend[group].items()):
                raise FreezeIntegrityError('PRIOR_SPEND_HISTORY_CHANGED')
        expected_priors = {prior_plan['split_manifest_path']: prior_plan['split_manifest_sha256'],
                           **prior_plan.get('prior_split_manifests', {})}
        declared_priors = {r.path: r.sha256 for r in draft.prior_split_manifests}
        if any(declared_priors.get(p) != h for p, h in expected_priors.items()):
            raise FreezeIntegrityError('PILOT_ANCESTOR_SPLITS_MISSING_FROM_EXCLUSIONS')
    refs, required, remaining = _validate_inputs(draft, analysis, candidate, design)
    refs += [reference(p) for p in (candidate_path, formal_plan_path, analysis_plan_path, planning_receipt_path)]
    refs += [reference(Path(p).parent / 'plan.json') for p in pilot_indexes]
    refs += ancestor_refs
    if any(output == Path(r.path).resolve() or output in Path(r.path).resolve().parents for r in refs):
        raise ValueError('CYCLIC_OR_FUTURE_OUTPUT_REFERENCE')
    # mkdir is exclusive; no previously issued freeze is overwritten on rerun.
    output.mkdir(parents=True, exist_ok=False)
    durable_json(output / 'AnalysisPlan.json', analysis.model_dump(mode='json'))
    analysis_ref = reference(output / 'AnalysisPlan.json')
    schedule = []
    for i, ref in enumerate(draft.tasks):
        task = load_research_model(ref.path, TaskSpec)
        order = draft.methods[i % len(draft.methods):] + draft.methods[:i % len(draft.methods)]
        schedule.extend({'episode_id': task.task_id + '--' + method, 'task_id': task.task_id,
            'method': method, 'rq_ids': ['RQ1', 'RQ2', 'RQ3'], 'status': 'REGISTERED'} for method in order)
    core = PlanCore.model_validate({**draft.model_dump(mode='json'),
        'analysis_plan': analysis_ref.model_dump(mode='json'),
        'candidate_plan': reference(candidate_path).model_dump(mode='json'),
        'planning_receipt': reference(planning_receipt_path).model_dump(mode='json'), 'schedule': schedule})
    durable_json(output / 'PlanCore.json', core.model_dump(mode='json'))
    core_ref = reference(output / 'PlanCore.json')
    # Preserve exact ledger bytes, not merely the unkeyed JSON payload hash.
    (output / 'spend_at_freeze.json').write_bytes(Path(draft.authorization.ledger.path).read_bytes())
    receipt = FreezeReceipt(schema_version=1, status='FROZEN', created_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_core=core_ref, analysis_plan=analysis_ref, candidate_plan=reference(candidate_path),
        planning_receipt=reference(planning_receipt_path), spend_snapshot=reference(output / 'spend_at_freeze.json'),
        input_hashes={r.path: r.sha256 for r in refs},
        checks={k: True for k in ('completed_independent_pilot', 'unchanged_thresholds', 'acyclic_inputs',
            'source_closure_bound', 'unexposed_test_registry', 'independent_evaluator_budget', 'finite_authorized_budget')},
        task_count=len(core.tasks), trial_count=len(core.tasks) * len(core.methods),
        maximum_additional_reservation_nano=required, remaining_reservation_nano=remaining,
        scope='exact_analytic', financial_execution_permitted=False)
    durable_json(output / 'freeze_receipt.json', receipt.model_dump(mode='json'))
    envelope = ExecutionEnvelope(schema_version=1, study_stage='formal', plan_id=core.plan_id,
        plan_core=core_ref, analysis_plan=analysis_ref, freeze_receipt=reference(output / 'freeze_receipt.json'))
    durable_json(output / 'ExecutionEnvelope.json', envelope.model_dump(mode='json'))
    verify_execution_envelope(output / 'ExecutionEnvelope.json')
    for path in output.iterdir():
        os.chmod(path, 0o444)
    return envelope


def verify_execution_envelope(path, *, check_live_inputs=True):
    """Verify frozen graph; after execution only the mutable spend ledger may advance.

    check_live_inputs=False still verifies all other bound inputs and the immutable
    spend snapshot. The executor separately verifies the ledger's append-only
    successor history and authorization before/after paid calls.
    """
    envelope = load_research_model(path, ExecutionEnvelope)
    for ref in (envelope.plan_core, envelope.analysis_plan, envelope.freeze_receipt):
        check_ref(ref)
    core = load_research_model(envelope.plan_core.path, PlanCore)
    analysis = load_research_model(envelope.analysis_plan.path, AnalysisPlan)
    receipt = load_research_model(envelope.freeze_receipt.path, FreezeReceipt)
    check_ref(receipt.spend_snapshot)
    if receipt.spend_snapshot.sha256 != core.authorization.ledger.sha256:
        raise FreezeIntegrityError('FREEZE_SPEND_SNAPSHOT_MISMATCH')
    if (envelope.plan_id != core.plan_id or core.analysis_plan != envelope.analysis_plan
            or receipt.plan_core != envelope.plan_core or receipt.analysis_plan != envelope.analysis_plan
            or receipt.candidate_plan != core.candidate_plan or receipt.planning_receipt != core.planning_receipt):
        raise FreezeIntegrityError('FREEZE_GRAPH_IDENTITY_MISMATCH')
    if len({envelope.plan_core.path, envelope.analysis_plan.path, envelope.freeze_receipt.path, str(Path(path).resolve())}) != 4:
        raise FreezeIntegrityError('CYCLIC_FREEZE_GRAPH')
    for name, sha in receipt.input_hashes.items():
        if not check_live_inputs and name == core.authorization.ledger.path:
            continue
        check_ref(FileRef(path=name, sha256=sha))
    for ref in (core.candidate_plan, core.planning_receipt, *core.scientific_inputs, *core.tasks,
                core.split_manifest, *core.prior_split_manifests, core.exposure_registry, core.authorization.ledger):
        if receipt.input_hashes.get(ref.path) != ref.sha256:
            raise FreezeIntegrityError('UNBOUND_SCIENTIFIC_INPUT')
    if receipt.task_count != len(core.tasks) or receipt.trial_count != len(core.tasks) * len(core.methods):
        raise FreezeIntegrityError('FREEZE_SCHEDULE_COUNT_MISMATCH')
    expected=[]
    for i, ref in enumerate(core.tasks):
        task=load_research_model(ref.path, TaskSpec)
        order=core.methods[i % len(core.methods):] + core.methods[:i % len(core.methods)]
        expected.extend((task.task_id + '--' + m, task.task_id, m) for m in order)
    if expected != [(x.episode_id, x.task_id, x.method) for x in core.schedule]:
        raise FreezeIntegrityError('FREEZE_TRIAL_SCHEDULE_MISMATCH')
    return core, analysis, receipt
