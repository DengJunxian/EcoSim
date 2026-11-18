"""Versioned Google F73 freeze; finite USD call allocations, no monetary ceiling."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import os
from pydantic import Field,model_validator
from core.experiments.archive import durable_json
from core.experiments.identity import research_hash,file_sha256
from core.experiments.strict_types import StrictResearchModel,load_research_model,strict_json_loads
from core.codesign.schema import TaskSpec
from core.codesign.runner import design_implementation_manifest
from core.codesign.pilot import validate_prior_independence
from .analysis import PlanningDesign
from .google_planning import load_completed_google_pilot
from .study_freeze import (FileRef,AnalysisPlan,PlanCoreDraft,ExposureRegistry,FreezeGateError,
    FreezeIntegrityError,check_ref,reference,validate_v3_population,FormalTrial)

GOOGLE_RUNTIME_FILES=('core/research_analysis/google_pilot.py','core/research_analysis/google_usage.py',
    'core/research_analysis/google_planning.py','core/research_analysis/google_freeze.py',
    'core/research_analysis/google_formal.py','core/research_analysis/analysis.py',
    'core/research_analysis/formal_execution.py','core/research_analysis/study_freeze.py',
    'scripts/research/build_google_pilot.py','scripts/research/prepare_google_study.py',
    'scripts/research/run_google_formal.py')

class GoogleAuthorization(StrictResearchModel):
    authorization_id: Literal['google-phase7-user-unlimited-20261023']
    currency: Literal['USD']
    monetary_cap: None
    ledger: FileRef
    usage_ledger_path: str
    maximum_generation_calls: int = Field(gt=0)
    input_price_nano: Literal[300]
    output_price_nano: Literal[2500]
    cached_price_nano: Literal[30]

class GooglePlanCoreDraft(PlanCoreDraft):
    schema_version: Literal[2]
    authorization: GoogleAuthorization
    stop_rule: Literal['finish_fixed_task_batch_then_unknown_identity_or_three_failures_v1']
    execution_workers: Literal[16]
    tasks_per_batch: Literal[2]
    natural_collection: Literal['all_trials_all_sources_no_error_selection_v1']
    budget_axis_scope: Literal['fixed_tokens;unique_market_runs_not_applicable_exact_analytic']

class GooglePlanCore(GooglePlanCoreDraft):
    analysis_plan: FileRef
    candidate_plan: FileRef
    planning_receipt: FileRef
    schedule: list[dict] = Field(min_length=1)

    @model_validator(mode='after')
    def valid_schedule(self):
        for row in self.schedule:FormalTrial.model_validate(row)
        return self

class GoogleFreezeReceipt(StrictResearchModel):
    schema_version: Literal[2]
    status: Literal['FROZEN']
    created_at_utc: str
    plan_core: FileRef
    analysis_plan: FileRef
    candidate_plan: FileRef
    planning_receipt: FileRef
    spend_snapshot: FileRef
    input_hashes: dict[str,str]
    checks: dict[str,Literal[True]]
    task_count: int = Field(gt=0)
    trial_count: int = Field(gt=0)
    maximum_additional_reservation_nano: int = Field(gt=0)
    currency: Literal['USD']
    maximum_generation_calls: int = Field(gt=0)
    monetary_cap: None
    scope: Literal['exact_analytic']
    financial_execution_permitted: Literal[False]

class GoogleExecutionEnvelope(StrictResearchModel):
    schema_version: Literal[2]
    study_stage: Literal['formal']
    plan_id: str
    plan_core: FileRef
    analysis_plan: FileRef
    freeze_receipt: FileRef


def _load_candidate(path):
    path=Path(path).resolve();candidate=strict_json_loads(path.read_text())
    receipt_path=path.parent/'power_precision_receipt.json';receipt=strict_json_loads(receipt_path.read_text())
    if receipt['artifact_hashes'].get(str(path))!=file_sha256(path):raise FreezeIntegrityError('CANDIDATE_NOT_BOUND')
    for group in ('input_hashes','artifact_hashes'):
        for name,sha in receipt[group].items():check_ref(FileRef(path=name,sha256=sha))
    design=load_research_model(path.parent/'design.json',PlanningDesign)
    indexes=[p for p in receipt['input_hashes'] if Path(p).name=='benchmark_index.json']
    if len(indexes)!=1:raise FreezeIntegrityError('ONE_COMPLETED_PILOT_REQUIRED')
    original_design=next(p for p in receipt['input_hashes'] if Path(p).name in ('design.yaml','design.json') and Path(p).parent!=path.parent)
    plan,tasks,index,*_=load_completed_google_pilot(indexes[0],original_design)
    if research_hash(design)!=research_hash(load_research_model(original_design,PlanningDesign)):
        raise FreezeIntegrityError('PLANNING_DESIGN_CHANGED_FROM_PRECALL_REGISTRATION')
    if (candidate['schema_version']!=2 or candidate['currency']!='USD' or candidate['pilot_plan_id']!=plan.plan_id
        or candidate['source_hash']!=index['source_hash'] or design.pilot_plan_id!=plan.plan_id):
        raise FreezeIntegrityError('GOOGLE_CANDIDATE_PROVENANCE_MISMATCH')
    if [x['contrast'] for x in candidate['primary_contrasts']]!=[c.id for c in design.contrasts]:
        raise FreezeIntegrityError('PLANNING_CONTRASTS_CHANGED')
    recommendations=[max(x['power_task_count_sensitivity_max'],x['precision_task_count_sensitivity_max'])
        for x in candidate['primary_contrasts'] if x['power_task_count_sensitivity_max'] is not None and x['precision_task_count_sensitivity_max'] is not None]
    if len(recommendations)!=4 or candidate['recommended_balanced_task_count']!=max(recommendations):
        raise FreezeGateError('SUPPORTED_FOUR_CONTRAST_SAMPLE_SIZE_REQUIRED')
    if (design.family_alpha!=.05 or design.target_power!=.8 or design.target_half_width!=.1
        or any(c.delta!=.1 for c in design.contrasts)):
        raise FreezeGateError('ORIGINAL_SCIENTIFIC_TARGETS_REQUIRED')
    return candidate,design,receipt_path


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
    for name in GOOGLE_RUNTIME_FILES:
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
    state_wrapper=strict_json_loads(check_ref(draft.authorization.ledger).read_text())
    state=state_wrapper['state']
    if research_hash(state)!=state_wrapper['sha256']:raise FreezeIntegrityError('PILOT_SPEND_HASH_MISMATCH')
    if state['contract']['currency']!='USD' or state['blocked_reason']:
        raise FreezeGateError('PILOT_USAGE_BLOCKED_OR_WRONG_CURRENCY')
    if set(state['reservations'])!=set(state['settlements']):raise FreezeGateError('PILOT_UNSETTLED_USAGE')
    required_calls=len(tasks)*(len(draft.methods)-1)*calls
    if draft.authorization.maximum_generation_calls!=required_calls:
        raise FreezeGateError('FINITE_FORMAL_CALL_ALLOCATION_MISMATCH')
    if not Path(draft.authorization.usage_ledger_path).is_absolute() or Path(draft.authorization.usage_ledger_path).exists():
        raise FreezeIntegrityError('NEW_ABSOLUTE_ALLOCATION_LEDGER_REQUIRED')
    if draft.model.provider!='gemini' or draft.model.model_id!='gemini-3.5-flash-lite':
        raise FreezeIntegrityError('GOOGLE_FIXED_MODEL_REQUIRED')
    required=required_calls*(model.input_reservation_tokens*draft.authorization.input_price_nano+
        model.max_output_tokens*draft.authorization.output_price_nano)
    return refs+exposure.evidence_files,required,required_calls


def freeze_google_study(candidate_path, formal_plan_path, analysis_plan_path, output):
    """Validate every prerequisite before publishing an immutable output directory."""
    output = Path(output).resolve()
    if output.exists():
        raise FreezeIntegrityError('IMMUTABLE_FREEZE_OUTPUT_ALREADY_EXISTS')
    draft = load_research_model(formal_plan_path, GooglePlanCoreDraft)
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
        prior_registration=strict_json_loads((Path(index).parent/'registration.json').read_text())
        if (prior_plan.get('maximum_workers') != draft.execution_workers or prior_plan.get('batch_task_count') != draft.tasks_per_batch
                or prior_registration.get('worker_count') != draft.execution_workers or prior_plan.get('stop_rule') != draft.stop_rule):
            raise FreezeIntegrityError('FORMAL_CONCURRENCY_CHANGED_FROM_PILOT')
        # Enforce cumulative authorization on the actual freeze entry point too;
        # preflight is optional and cannot be a prerequisite for budget safety.
        pilot_spend = strict_json_loads(Path(index).read_text())['spend_snapshot']
        current_wrapper = strict_json_loads(check_ref(draft.authorization.ledger).read_text())
        current_spend = current_wrapper['state']
        if research_hash(current_spend) != current_wrapper['sha256']:
            raise FreezeIntegrityError('SPEND_HASH_MISMATCH')
        if (current_spend['contract'] != pilot_spend['contract'] or
                draft.authorization.authorization_id != prior_plan['authorization_id'] or
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
    core = GooglePlanCore.model_validate({**draft.model_dump(mode='json'),
        'analysis_plan': analysis_ref.model_dump(mode='json'),
        'candidate_plan': reference(candidate_path).model_dump(mode='json'),
        'planning_receipt': reference(planning_receipt_path).model_dump(mode='json'), 'schedule': schedule})
    durable_json(output / 'PlanCore.json', core.model_dump(mode='json'))
    core_ref = reference(output / 'PlanCore.json')
    # Preserve exact ledger bytes, not merely the unkeyed JSON payload hash.
    (output / 'spend_at_freeze.json').write_bytes(Path(draft.authorization.ledger.path).read_bytes())
    receipt = GoogleFreezeReceipt(schema_version=2, status='FROZEN', created_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_core=core_ref, analysis_plan=analysis_ref, candidate_plan=reference(candidate_path),
        planning_receipt=reference(planning_receipt_path), spend_snapshot=reference(output / 'spend_at_freeze.json'),
        input_hashes={r.path: r.sha256 for r in refs},
        checks={k: True for k in ('completed_independent_pilot', 'unchanged_thresholds', 'acyclic_inputs',
            'source_closure_bound', 'unexposed_test_registry', 'independent_evaluator_budget', 'finite_authorized_budget')},
        task_count=len(core.tasks), trial_count=len(core.tasks) * len(core.methods),
        maximum_additional_reservation_nano=required, maximum_generation_calls=remaining,currency='USD',monetary_cap=None,
        scope='exact_analytic', financial_execution_permitted=False)
    durable_json(output / 'freeze_receipt.json', receipt.model_dump(mode='json'))
    envelope = GoogleExecutionEnvelope(schema_version=2, study_stage='formal', plan_id=core.plan_id,
        plan_core=core_ref, analysis_plan=analysis_ref, freeze_receipt=reference(output / 'freeze_receipt.json'))
    durable_json(output / 'ExecutionEnvelope.json', envelope.model_dump(mode='json'))
    verify_google_envelope(output / 'ExecutionEnvelope.json')
    for path in output.iterdir():
        os.chmod(path, 0o444)
    return envelope


def verify_google_envelope(path, *, check_live_inputs=True):
    """Verify frozen graph; after execution only the mutable spend ledger may advance.

    check_live_inputs=False still verifies all other bound inputs and the immutable
    spend snapshot. The executor separately verifies the ledger's append-only
    successor history and authorization before/after paid calls.
    """
    envelope = load_research_model(path, GoogleExecutionEnvelope)
    for ref in (envelope.plan_core, envelope.analysis_plan, envelope.freeze_receipt):
        check_ref(ref)
    core = load_research_model(envelope.plan_core.path, GooglePlanCore)
    analysis = load_research_model(envelope.analysis_plan.path, AnalysisPlan)
    receipt = load_research_model(envelope.freeze_receipt.path, GoogleFreezeReceipt)
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
    if expected != [(x['episode_id'], x['task_id'], x['method']) for x in core.schedule]:
        raise FreezeIntegrityError('FREEZE_TRIAL_SCHEDULE_MISMATCH')
    return core, analysis, receipt
