"""Google independent development pilot, fixed complete-task batches and USD usage."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
import json
import os
import shutil
import time
from pydantic import Field, model_validator
from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256, research_hash
from core.experiments.strict_types import StrictResearchModel, strict_json_loads, load_research_model
from core.codesign.schema import TaskSpec, ModelCallConfig
from core.codesign.pilot import METHODS, validate_split_manifest, validate_prior_independence, inspect_conflict_design
from core.codesign.runner import design_implementation_manifest, freeze_design_source, run_design_episode, verify_episode
from core.codesign.live_protocol import SharedEvidenceProtocolMethod
from core.codesign.baselines import CentralRuleMethod
from core.codesign.llm_adapter import DesignAgentClient
from core.llm.gemini_client import load_gemini_settings, native_sdk_version
from .google_usage import GoogleUsageLedger

ROOT = Path(__file__).resolve().parents[2]
STRATA = ['role_conflict', 'model_conflict', 'infeasible']
RUNTIME_FILES = ['core/research_analysis/google_pilot.py', 'core/research_analysis/google_usage.py',
                 'scripts/research/build_google_pilot.py', 'scripts/research/run_google_pilot.py']


class GooglePilotPlanV4(StrictResearchModel):
    schema_version: Literal[4]
    study_stage: Literal['pilot']
    mode: Literal['live']
    plan_id: str = Field(min_length=1)
    authorization_id: Literal['google-phase7-user-unlimited-20261023']
    currency: Literal['USD']
    monetary_cap: None
    authorization_basis: Literal['user_no_monetary_cap_finite_registered_plan']
    maximum_generation_calls: Literal[2160]
    task_paths: list[str]
    task_sha256: dict[str, str]
    split_manifest_path: str
    split_manifest_sha256: str
    prior_split_manifests: dict[str, str]
    planning_design_path: str
    planning_design_sha256: str
    planning_ancestor_path: str
    planning_ancestor_sha256: str
    experiment_spec_path: str
    experiment_spec_sha256: str
    methods: list[str]
    model: ModelCallConfig
    protocol_version: Literal['bounded_llm_protocol_v3']
    feedback_version: Literal['common_operation_receipts_v1']
    design_turns: Literal[6]
    llm_repeats: Literal[1]
    batch_task_count: Literal[2]
    maximum_workers: Literal[16]
    input_price_nano: Literal[300]
    output_price_nano: Literal[2500]
    cached_price_nano: Literal[30]
    pricing_source: Literal['https://ai.google.dev/gemini-api/docs/pricing']
    price_checked_date: Literal['2026-10-23']
    model_version_as_documented: Literal['gemini-3.5-flash-lite']
    sampling_rationale: str
    stop_rule: Literal['finish_fixed_task_batch_then_unknown_identity_or_three_failures_v1']
    collection_rule: Literal['all_trials_including_failures_source_stratified_v1']
    exclusions: dict[str, str]

    @model_validator(mode='after')
    def check(self):
        if list(self.methods) != METHODS:
            raise ValueError('GOOGLE_PILOT_ORIGINAL_METHOD_ORDER_REQUIRED')
        if len(self.task_paths) != 30 or len(set(self.task_paths)) != 30 or set(self.task_paths) != set(self.task_sha256):
            raise ValueError('GOOGLE_PILOT_THIRTY_INDEPENDENT_TASKS_REQUIRED')
        m = self.model
        if (m.source, m.provider, m.model_id, m.temperature, m.thinking,
            m.max_output_tokens, m.input_reservation_tokens, m.timeout_seconds, m.max_request_attempts,
            m.adapter_version, m.provider_version) != ('live', 'gemini', 'gemini-3.5-flash-lite', .2, True,
                4096, 32768, 120.0, 1, 'design-agent-gemini-native-v1', 'gemini-native-v1'):
            raise ValueError('GOOGLE_PILOT_FIXED_MODEL_CONFIGURATION_REQUIRED')
        if m.model_aliases:
            raise ValueError('GOOGLE_PILOT_EXACT_MODEL_VERSION_REQUIRED')
        if len(self.prior_split_manifests) < 3:
            raise ValueError('GOOGLE_PILOT_ALL_THREE_PRIOR_SPLITS_REQUIRED')
        return self


def load_google_pilot(path):
    return load_research_model(path, GooglePilotPlanV4)


def _checked(path, sha):
    p = Path(path)
    if not p.is_absolute() or file_sha256(p) != sha:
        raise ValueError('GOOGLE_BOUND_INPUT_CHANGED:' + str(p))
    return strict_json_loads(p.read_text())


def validate_google_population(task):
    result = inspect_conflict_design(task)
    stratum = result['stratum']
    if result['complete_pool_size'] != 6 or result['acceptable_count'] != (0 if stratum == 'infeasible' else 1):
        raise ValueError('GOOGLE_REGISTERED_POPULATION_CHANGED')
    if task.prompt_resources.get('stratum_index') != str(STRATA.index(stratum)):
        raise ValueError('GOOGLE_STRATUM_LABEL_CHANGED')
    switching = 0
    for candidate in task.candidate_pool:
        rows = [row for row in task.analytic_outcomes if row.candidate_id == candidate.candidate_id]
        switching += any(len({1-sum(w*x.losses.model_dump()[k] for k, w in role.weights.model_dump().items()) >= role.rho
                             for x in rows}) > 1 for role in task.roles)
    if (stratum == 'role_conflict' and switching != 0) or (stratum == 'model_conflict' and switching != 2):
        raise ValueError('GOOGLE_CONFLICT_DISTRIBUTION_CHANGED')
    return result


def load_google_inputs(plan):
    plan = GooglePilotPlanV4.model_validate(plan)
    tasks = [TaskSpec.model_validate(_checked(path, plan.task_sha256[path])) for path in plan.task_paths]
    for task in tasks:
        if (task.origin, task.evidence_mode, task.signing_layer) != ('analytic_fixture', 'exact_analytic', 'delegated_explicit_llm_v1'):
            raise ValueError('GOOGLE_ANALYTIC_PILOT_SCOPE_ONLY')
        if task.model_configurations != [plan.model]:
            raise ValueError('GOOGLE_TASK_MODEL_CONFIG_MISMATCH')
        if (len(task.required_signers) != 3 or task.budget.llm_calls != 9
                or task.budget.input_tokens != 9*32768 or task.budget.output_tokens != 9*4096
                or task.budget.wall_timeout_seconds != 1500.0 or task.budget.unique_market_runs != 0):
            raise ValueError('GOOGLE_TASK_BUDGET_SCHEDULE_MISMATCH')
    splits = _checked(plan.split_manifest_path, plan.split_manifest_sha256)
    check = validate_split_manifest(splits, tasks)
    for field in ('task_id', 'underlying_cluster_id', 'analytic_content_hash', 'replication_namespace'):
        vals = [entry[field] for entry in splits['tasks']]
        if len(vals) != len(set(vals)):
            raise ValueError('GOOGLE_SPLIT_INDEPENDENCE_VIOLATION:' + field)
    dependencies = [dep for entry in splits['tasks'] for dep in entry['source_dependencies']]
    if len(dependencies) != len(set(dependencies)):
        raise ValueError('GOOGLE_SPLIT_SOURCE_DEPENDENCY_REUSED')
    for entry in splits['tasks']:
        task = TaskSpec.model_validate(_checked(entry['task_path'], entry['sha256']))
        if (entry['task_id'], entry['task_hash'], entry['analytic_content_hash'], entry['split']) != (
            task.task_id, task.task_hash, research_hash(task.analytic_outcomes), task.split):
            raise ValueError('GOOGLE_SPLIT_TASK_CONTENT_MISMATCH')
    priors = [_checked(path, sha) for path, sha in plan.prior_split_manifests.items()]
    check['prior_independence'] = validate_prior_independence(splits, priors)
    design = _checked(plan.planning_design_path, plan.planning_design_sha256)
    ancestor = _checked(plan.planning_ancestor_path, plan.planning_ancestor_sha256)
    from .analysis import PlanningDesign
    PlanningDesign.model_validate(design)
    if design['pilot_plan_id'] != plan.plan_id or design['simulation_seed'] != 2026102344:
        raise ValueError('GOOGLE_PLANNING_IDENTITY_CHANGED')
    for key in ('contrasts', 'family_alpha', 'target_power', 'target_half_width', 'candidate_task_counts',
                'simulation_draws', 'bootstrap_draws', 'planning_test', 'precision_rule', 'missing_trial_rule',
                'joint_pattern_pseudocounts', 'delta_sensitivity', 'equivalence_margin'):
        if design[key] != ancestor[key]:
            raise ValueError('GOOGLE_REGISTERED_PLANNING_TARGET_CHANGED:' + key)
    if (design['family_alpha'], design['target_power'], design['target_half_width'], design['simulation_draws']) != (.05, .8, .1, 10000):
        raise ValueError('GOOGLE_ORIGINAL_PLANNING_TARGETS_REQUIRED')
    spec = _checked(plan.experiment_spec_path, plan.experiment_spec_sha256)
    diagnostics = [validate_google_population(task) for task in tasks]
    counts = {name: sum(d['stratum'] == name for d in diagnostics) for name in STRATA}
    if counts != {name: 10 for name in STRATA} or spec.get('strata') != counts or spec.get('maximum_development_tasks') != 30:
        raise ValueError('GOOGLE_FIXED_THIRTY_TASK_DESIGN')
    check['prospective_conflict_design'] = {'stratum_counts': counts, 'tasks': diagnostics}
    return tasks, check


def runtime_manifest(extra_paths=()):
    manifest = design_implementation_manifest()
    paths = {ROOT / entry.path for entry in manifest.files}
    paths.update(ROOT / p for p in RUNTIME_FILES)
    paths.update(Path(p).resolve() for p in extra_paths)
    files = {str(path.resolve()): file_sha256(path) for path in sorted(paths)}
    return {'source_hash': manifest.source_hash, 'runtime_hash': research_hash(files), 'files': files}


def verify_history(current, previous):
    if current['contract'] != previous['contract'] or current['created_at'] != previous['created_at']:
        raise ValueError('GOOGLE_CUMULATIVE_AUTHORIZATION_OR_ORIGIN_CHANGED')
    for group in ('reservations', 'settlements'):
        if any(current[group].get(k) != v for k, v in previous[group].items()):
            raise ValueError('GOOGLE_CUMULATIVE_HISTORY_CHANGED:' + group)
    if previous['blocked_reason'] and current['blocked_reason'] != previous['blocked_reason']:
        raise ValueError('GOOGLE_BLOCK_STATE_REWRITTEN')


def validate_episode_identity(result, task, item, source_hash):
    if (result.source_hash != source_hash or result.task_hash != task.task_hash
            or (result.episode_id, result.task_id, result.method) != (item['episode_id'], item['task_id'], item['method'])):
        raise ValueError('GOOGLE_EPISODE_SOURCE_TASK_OR_SCHEDULE_MISMATCH')
    calls = result.common_contract['live_calls']
    if type(calls) is not int or not 0 <= calls <= task.budget.llm_calls:
        raise ValueError('GOOGLE_EPISODE_CALL_COUNT_INVALID')
    if item['method'] == 'central_rule':
        if calls != 0 or result.live_status != 'NOT_RUN':
            raise ValueError('GOOGLE_RULE_EPISODE_CANNOT_CONTAIN_LIVE_CALLS')
    elif calls > 0:
        if result.live_status != 'RUN':
            raise ValueError('GOOGLE_LIVE_EPISODE_SOURCE_MISMATCH')
    elif result.terminal_state not in {'METHOD_FAILED', 'BUDGET_UNRESOLVED', 'BUDGET_EXHAUSTED_UNRESOLVED'}:
        raise ValueError('GOOGLE_ZERO_CALL_LIVE_EPISODE_REQUIRES_EXPLICIT_FAILURE')


def run_google_pilot(plan_path, output, *, credential_path, spend_path, workers=16, extra_runtime_paths=()):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('GOOGLE_WORKERS_MUST_BE_ONE_TO_SIXTEEN')
    plan = load_google_pilot(plan_path)
    tasks, split_check = load_google_inputs(plan)
    if plan.model.sdk_version != native_sdk_version():
        raise ValueError('GOOGLE_RUNTIME_SDK_CHANGED')
    settings = load_gemini_settings(credential_path, timeout_seconds=plan.model.timeout_seconds)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('IMMUTABLE_GOOGLE_PILOT_OUTPUT_ALREADY_EXISTS')
    if Path(spend_path).exists():
        raise ValueError('GOOGLE_PILOT_ALLOCATION_ALREADY_EXISTS')
    runtime = runtime_manifest(extra_runtime_paths)
    splits = _checked(plan.split_manifest_path, plan.split_manifest_sha256)
    bound_inputs = {str(Path(plan_path).resolve()): file_sha256(plan_path), **plan.task_sha256,
        plan.split_manifest_path: plan.split_manifest_sha256, **plan.prior_split_manifests,
        plan.planning_design_path: plan.planning_design_sha256,
        plan.planning_ancestor_path: plan.planning_ancestor_sha256,
        plan.experiment_spec_path: plan.experiment_spec_sha256,
        **{e['task_path']: e['sha256'] for e in splits['tasks']}}
    def unchanged():
        for path, sha in {**runtime['files'], **bound_inputs}.items():
            if file_sha256(path) != sha:
                raise ValueError('GOOGLE_REGISTERED_SOURCE_OR_INPUT_CHANGED:' + path)
    spend = GoogleUsageLedger(spend_path, authorization_id=plan.authorization_id, plan_id=plan.plan_id,
        plan_hash=research_hash(plan), maximum_calls=plan.maximum_generation_calls,
        input_cap=plan.model.input_reservation_tokens, output_cap=plan.model.max_output_tokens,
        input_nano=plan.input_price_nano, output_nano=plan.output_price_nano, cached_nano=plan.cached_price_nano,
        source_guard=unchanged)
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir()
    (output/'failed_attempts').mkdir()
    freeze_design_source(output)
    for path in runtime['files']:
        source = Path(path)
        relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('external')/research_hash(str(source))/source.name
        destination = output/'runtime_source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    durable_json(output/'plan.json', plan.model_dump(mode='json'))
    durable_json(output/'planning_design.json', strict_json_loads(Path(plan.planning_design_path).read_text()))
    durable_json(output/'experiment_spec.json', strict_json_loads(Path(plan.experiment_spec_path).read_text()))
    schedule = []
    for i, task in enumerate(tasks):
        order = plan.methods[i % len(plan.methods):] + plan.methods[:i % len(plan.methods)]
        schedule.extend(dict(episode_id=task.task_id+'--'+method, task_id=task.task_id,
                             method=method, batch_id=f'batch-{i//2:03d}') for method in order)
    before = spend.snapshot()
    registration = dict(schema_version=4, study_stage='pilot', plan_hash=research_hash(plan),
        source_hash=runtime['source_hash'], runtime_hash=runtime['runtime_hash'],
        runtime_source_hashes=runtime['files'], input_hashes=bound_inputs, registered_at=time.time(),
        schedule=schedule, split_check=split_check, maximum_generation_calls=plan.maximum_generation_calls,
        batch_task_count=2, worker_count=workers, batch_stop_rule=plan.stop_rule,
        spend_contract=spend.contract, protocol_version=plan.protocol_version,
        feedback_version=plan.feedback_version, planning_design_sha256=plan.planning_design_sha256,
        analysis_scope='development feasibility, variance and cost only; not confirmatory effects',
        natural_collection='all scheduled live methods; deterministic reference separate')
    durable_json(output/'spend_before.json', before)
    durable_json(output/'registration.json', registration)
    os.chmod(output/'registration.json', 0o444)
    for task in tasks:
        durable_json(output/'inputs'/(task.task_id+'.json'), task.model_dump(mode='json'))
    task_map = {task.task_id: task for task in tasks}
    episodes, stopping, prior = [], None, before
    def save():
        current = spend.snapshot()
        verify_history(current, prior)
        index = dict(schema_version=4, study_stage='pilot', plan_hash=research_hash(plan),
            source_hash=runtime['source_hash'], runtime_hash=runtime['runtime_hash'],
            registration_sha256=file_sha256(output/'registration.json'), episodes=episodes,
            planned_trials=len(schedule), indexed_trials=len(episodes), independent_task_count=len(tasks),
            market_runs=0, llm_repeats_per_task_method=1, stopping_reason=stopping,
            spend_snapshot=current, formal_status='NOT_RUN', financial_status='SEPARATE_QUALIFICATION')
        durable_json(output/'benchmark_index.json', index)
        return index
    def execute(item):
        destination = output/'episodes'/item['episode_id']
        try:
            unchanged()
            task = task_map[item['task_id']]
            if item['method'] == 'central_rule':
                body = task.model_dump(mode='json')
                body.update(signing_layer='explicit_rule_owners_v1', model_configurations=[])
                task = TaskSpec.model_validate(body)
                implementation = CentralRuleMethod(protocol_enabled=True)
            else:
                implementation = SharedEvidenceProtocolMethod(item['method'],
                    lambda ledger, path: DesignAgentClient(plan.model, ledger, path, allow_live=True,
                                                          settings=settings, spend_ledger=spend), turns=plan.design_turns)
            result = run_design_episode(task, implementation, destination, episode_id=item['episode_id'])
            verification = verify_episode(destination)
            validate_episode_identity(result, task, item, runtime['source_hash'])
            return {**item, 'status': 'RECORDED', 'path': str(destination), 'terminal_state': result.terminal_state,
                'live_calls': result.common_contract['live_calls'], 'result_sha256': file_sha256(destination/'episode_result.json'),
                'verification': verification, 'natural_trace': item['method'] != 'central_rule' and result.live_status == 'RUN',
                'reference_only': item['method'] == 'central_rule', 'source': 'rule' if item['method'] == 'central_rule' else 'live'}
        except Exception as exc:
            from core.llm.base import redact_sensitive
            body = {**item, 'status': 'EXECUTION_FAILED', 'reason': redact_sensitive(exc),
                    'error_type': type(exc).__name__, 'attempt_directory': str(destination),
                    'source': 'rule' if item['method'] == 'central_rule' else 'live'}
            durable_json(output/'failed_attempts'/(item['episode_id']+'.json'), body)
            return body
    for start in range(0, len(schedule), 18):
        batch = schedule[start:start+18]
        if stopping:
            episodes.extend({**item, 'status': 'NOT_RUN_STOPPING_RULE', 'reason': stopping} for item in batch)
            save()
            continue
        try:
            unchanged()
        except (ValueError, OSError) as exc:
            stopping = 'REGISTERED_SOURCE_OR_INPUT_CHANGED'
            spend.block(stopping)
            episodes.extend({**item, 'status': 'NOT_RUN_STOPPING_RULE', 'reason': stopping} for item in batch)
            save()
            continue
        spend.begin_batch(batch[0]['batch_id'], [item['episode_id'] for item in batch])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map returns the registered order, never the completion order.
            batch_results = list(pool.map(execute, batch))
        episodes.extend(batch_results)
        failures = sum(row['status'] == 'EXECUTION_FAILED' or row.get('terminal_state') == 'METHOD_FAILED' for row in batch_results)
        reason = 'THREE_FAILED_EPISODES_IN_REGISTERED_BATCH' if failures >= 3 else None
        if any(row['status'] == 'EXECUTION_FAILED' for row in batch_results):
            reason = 'EXECUTION_INTEGRITY_OR_SERVICE_FAILURE'
        try:
            unchanged()
        except (ValueError, OSError):
            reason = 'REGISTERED_SOURCE_OR_INPUT_CHANGED'
        after = spend.close_batch(batch[0]['batch_id'], stop_reason=reason)
        verify_history(after, prior)
        prior = after
        stopping = after['blocked_reason']
        save()
        print(json.dumps({'batch_id': batch[0]['batch_id'], 'recorded': sum(row['status'] == 'RECORDED' for row in batch_results),
            'failures': failures, 'cumulative_calls': after['totals']['calls'], 'stopping_reason': stopping}), flush=True)
    index = save()
    durable_json(output/'spend_final_snapshot.json', spend.snapshot_envelope())
    durable_json(output/'COMPLETED.json', dict(index_sha256=file_sha256(output/'benchmark_index.json'),
        registration_sha256=file_sha256(output/'registration.json'), complete_planned_index=len(episodes)==len(schedule)))
    return index
