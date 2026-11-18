"""Provider-specific independent pilots with a powered GLM primary and separate secondary plans."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
import json
import os
import shutil
import stat
import threading
import time
from collections import deque
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
from core.codesign.budget import BudgetExceeded
from core.llm.gemini_client import load_gemini_settings, native_sdk_version
from .provider_usage import ProviderUsageLedger, verify_deepseek_retirement
from .google_recovery import SQLiteRequestPacer

ROOT = Path(__file__).resolve().parents[2]
STRATA = ['role_conflict', 'model_conflict', 'infeasible']
RUNTIME_FILES = ['core/research_analysis/provider_pilot.py', 'core/research_analysis/provider_usage.py',
    'scripts/research/build_provider_pilot.py', 'scripts/research/run_provider_pilot.py',
    'core/research_analysis/google_pilot.py', 'core/research_analysis/google_usage.py',
    'core/research_analysis/google_recovery.py', 'scripts/research/build_google_pilot.py',
    'examples/research/benchmarks/analytic_baseline_v1.json']
PROVIDERS = {
    'zhipu': dict(model='glm-4-flash-250414', count=30, role='primary_power_planning', currency='CNY',
                  prices=(0, 0, 0), concurrent=2, thinking=False),
    'deepseek': dict(model='deepseek-flash', count=9, role='secondary_provider_specific', currency='CNY',
                     prices=(2000, 8000, 40), concurrent=2, thinking=False),
    'gemini': dict(model='gemini-3.5-flash-lite', count=9, role='secondary_provider_specific', currency='USD',
                   prices=(300, 2500, 30), concurrent=16, thinking=True),
}


class ProviderFileRef(StrictResearchModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ProviderPlan(StrictResearchModel):
    schema_version: Literal[5]
    study_stage: Literal['provider_pilot']
    mode: Literal['live']
    plan_id: str = Field(min_length=1)
    provider: Literal['zhipu', 'deepseek', 'gemini']
    study_role: Literal['primary_power_planning', 'secondary_provider_specific']
    authorization_id: str = Field(min_length=1)
    user_authorization: ProviderFileRef
    provider_protocol: ProviderFileRef
    currency: Literal['USD', 'CNY']
    monetary_cap_nano: int | None
    parent_liability_reference: ProviderFileRef | None
    maximum_generation_calls: int = Field(gt=0)
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
    maximum_concurrent_requests: int = Field(ge=1)
    minimum_request_interval_seconds: float = Field(ge=0)
    input_price_nano: int = Field(ge=0)
    output_price_nano: int = Field(ge=0)
    cached_price_nano: int = Field(ge=0)
    pricing_source: str
    price_checked_date: Literal['2026-10-23']
    model_version_as_documented: str
    sampling_rationale: str
    stop_rule: Literal['finish_fixed_task_batch_then_unknown_identity_or_three_failures_v1']
    collection_rule: Literal['all_trials_including_failures_source_stratified_v1']
    no_cross_provider_power_pooling: Literal[True]
    exclusions: dict[str, str]

    @model_validator(mode='after')
    def check(self):
        p = PROVIDERS[self.provider]
        if self.plan_id != 'analytic-multiprovider-v1-'+self.provider or self.methods != METHODS:
            raise ValueError('PROVIDER_FIXED_PLAN_AND_METHOD_ORDER_REQUIRED')
        if (self.study_role, self.currency, self.maximum_generation_calls, self.maximum_concurrent_requests,
            self.minimum_request_interval_seconds, self.input_price_nano, self.output_price_nano, self.cached_price_nano) != (
                p['role'], p['currency'], p['count']*8*9, p['concurrent'], 5.1 if self.provider == 'gemini' else 0., *p['prices']):
            raise ValueError('PROVIDER_REGISTERED_ROLE_RESOURCES_OR_PRICE_CHANGED')
        if (len(self.task_paths) != p['count'] or len(set(self.task_paths)) != p['count']
                or set(self.task_paths) != set(self.task_sha256)):
            raise ValueError('PROVIDER_FIXED_INDEPENDENT_TASK_COUNT_REQUIRED')
        m = self.model
        if (m.source, m.provider, m.model_id, m.temperature, m.thinking, m.max_output_tokens,
            m.input_reservation_tokens, m.timeout_seconds, m.max_request_attempts) != (
                'live', self.provider, p['model'], .2, p['thinking'], 768, 24576, 120., 1):
            raise ValueError('PROVIDER_FIXED_COMMON_MODEL_BUDGET_CONFIGURATION_REQUIRED')
        if m.model_aliases:
            raise ValueError('PROVIDER_EXACT_MODEL_IDENTITY_REQUIRED')
        if self.provider == 'deepseek':
            if self.monetary_cap_nano != 100_000_000_000:
                raise ValueError('DEEPSEEK_ORIGINAL_CNY100_CAP_REQUIRED')
        elif self.monetary_cap_nano is not None or self.parent_liability_reference is not None:
            raise ValueError('PROVIDER_SEPARATE_CURRENCY_NO_INHERITED_DEEPSEEK_CAP')
        if self.provider != 'gemini' and m.sdk_version != '3.36.1':
            raise ValueError('PROVIDER_FIXED_OPENAI_SDK_VERSION_REQUIRED')
        if len(self.prior_split_manifests) < 6:
            raise ValueError('PROVIDER_ALL_FOUR_PRIORS_AND_TWO_PEER_SPLITS_REQUIRED')
        return self


def load_provider_pilot(path):
    return load_research_model(path, ProviderPlan)


def _checked(path, sha):
    p = Path(path)
    if not p.is_absolute() or file_sha256(p) != sha:
        raise ValueError('PROVIDER_BOUND_INPUT_CHANGED:' + str(p))
    return strict_json_loads(p.read_text())


def reference_paths(plan):
    refs = [plan.user_authorization, plan.provider_protocol]
    if plan.parent_liability_reference is not None:
        refs.append(plan.parent_liability_reference)
    result = {}
    for ref in refs:
        if not Path(ref.path).is_absolute() or file_sha256(ref.path) != ref.sha256:
            raise ValueError('PROVIDER_BOUND_AUTHORIZATION_OR_PROTOCOL_CHANGED')
        result[ref.path] = ref.sha256
    if plan.parent_liability_reference is not None:
        _, parent_refs = verify_deepseek_retirement(plan.parent_liability_reference.model_dump(mode='json'))
        result.update(parent_refs)
    return result


def validate_provider_population(task):
    result = inspect_conflict_design(task)
    stratum = result['stratum']
    if result['complete_pool_size'] != 6 or result['acceptable_count'] != (0 if stratum == 'infeasible' else 1):
        raise ValueError('PROVIDER_REGISTERED_POPULATION_CHANGED')
    if task.prompt_resources.get('stratum_index') != str(STRATA.index(stratum)):
        raise ValueError('PROVIDER_STRATUM_LABEL_CHANGED')
    switching = 0
    for candidate in task.candidate_pool:
        rows = [row for row in task.analytic_outcomes if row.candidate_id == candidate.candidate_id]
        switching += any(len({1-sum(w*x.losses.model_dump()[k] for k, w in role.weights.model_dump().items()) >= role.rho
                             for x in rows}) > 1 for role in task.roles)
    if (stratum == 'role_conflict' and switching != 0) or (stratum == 'model_conflict' and switching != 2):
        raise ValueError('PROVIDER_CONFLICT_DISTRIBUTION_CHANGED')
    return result


def load_provider_inputs(plan):
    plan = ProviderPlan.model_validate(plan)
    reference_paths(plan)
    tasks = [TaskSpec.model_validate(_checked(path, plan.task_sha256[path])) for path in plan.task_paths]
    for task in tasks:
        if (task.origin, task.evidence_mode, task.signing_layer) != ('analytic_fixture', 'exact_analytic', 'delegated_explicit_llm_v1'):
            raise ValueError('PROVIDER_ANALYTIC_PILOT_SCOPE_ONLY')
        if task.model_configurations != [plan.model]:
            raise ValueError('PROVIDER_TASK_MODEL_CONFIG_MISMATCH')
        if (len(task.required_signers) != 3 or task.budget.llm_calls != 9
                or task.budget.input_tokens != 9*24576 or task.budget.output_tokens != 9*768
                or task.budget.wall_timeout_seconds != 1500.0 or task.budget.unique_market_runs != 0):
            raise ValueError('PROVIDER_TASK_BUDGET_SCHEDULE_MISMATCH')
    splits = _checked(plan.split_manifest_path, plan.split_manifest_sha256)
    check = validate_split_manifest(splits, tasks)
    for field in ('task_id', 'underlying_cluster_id', 'analytic_content_hash', 'replication_namespace'):
        vals = [entry[field] for entry in splits['tasks']]
        if len(vals) != len(set(vals)):
            raise ValueError('PROVIDER_SPLIT_INDEPENDENCE_VIOLATION:' + field)
    dependencies = [dep for entry in splits['tasks'] for dep in entry['source_dependencies']]
    if len(dependencies) != len(set(dependencies)):
        raise ValueError('PROVIDER_SPLIT_SOURCE_DEPENDENCY_REUSED')
    for entry in splits['tasks']:
        task = TaskSpec.model_validate(_checked(entry['task_path'], entry['sha256']))
        if (entry['task_id'], entry['task_hash'], entry['analytic_content_hash'], entry['split']) != (
            task.task_id, task.task_hash, research_hash(task.analytic_outcomes), task.split):
            raise ValueError('PROVIDER_SPLIT_TASK_CONTENT_MISMATCH')
    priors = [_checked(path, sha) for path, sha in plan.prior_split_manifests.items()]
    check['prior_independence'] = validate_prior_independence(splits, priors)
    design = _checked(plan.planning_design_path, plan.planning_design_sha256)
    ancestor = _checked(plan.planning_ancestor_path, plan.planning_ancestor_sha256)
    from .analysis import PlanningDesign
    PlanningDesign.model_validate(design)
    if design['pilot_plan_id'] != plan.plan_id or design['simulation_seed'] != 2026102344:
        raise ValueError('PROVIDER_PLANNING_IDENTITY_CHANGED')
    for key in ('contrasts', 'family_alpha', 'target_power', 'target_half_width', 'candidate_task_counts',
                'simulation_draws', 'bootstrap_draws', 'planning_test', 'precision_rule', 'missing_trial_rule',
                'joint_pattern_pseudocounts', 'delta_sensitivity', 'equivalence_margin'):
        if design[key] != ancestor[key]:
            raise ValueError('PROVIDER_REGISTERED_PLANNING_TARGET_CHANGED:' + key)
    if (design['family_alpha'], design['target_power'], design['target_half_width'], design['simulation_draws']) != (.05, .8, .1, 10000):
        raise ValueError('PROVIDER_ORIGINAL_PLANNING_TARGETS_REQUIRED')
    spec = _checked(plan.experiment_spec_path, plan.experiment_spec_sha256)
    diagnostics = [validate_provider_population(task) for task in tasks]
    counts = {name: sum(d['stratum'] == name for d in diagnostics) for name in STRATA}
    if counts != {name: PROVIDERS[plan.provider]['count']//3 for name in STRATA} or spec.get('strata') != counts or spec.get('maximum_development_tasks') != PROVIDERS[plan.provider]['count']:
        raise ValueError('PROVIDER_FIXED_BALANCED_TASK_DESIGN')
    check['prospective_conflict_design'] = {'stratum_counts': counts, 'tasks': diagnostics}
    return tasks, check


def load_provider_settings(plan, path):
    if plan.provider == 'gemini':
        return load_gemini_settings(path, timeout_seconds=plan.model.timeout_seconds)
    from core.llm.config import LLMSettings
    path = Path(path)
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError('PROVIDER_CREDENTIAL_FILE_REQUIRES_MODE_600')
    names = {'deepseek': {'DEEPSEEK_API_KEY'}, 'zhipu': {'ZHIPU_API_KEY', 'ZHIPUAI_API_KEY'}}[plan.provider]
    values = [line.split('=', 1)[1].strip().strip('\"').strip("'") for line in path.read_text().splitlines()
              if '=' in line and line.split('=', 1)[0].strip() in names]
    if len(values) != 1 or not values[0] or any(char.isspace() for char in values[0]):
        raise ValueError('ONE_PRIVATE_PROVIDER_CREDENTIAL_REQUIRED')
    return LLMSettings(deepseek_api_key=values[0] if plan.provider == 'deepseek' else '',
        zhipu_api_key=values[0] if plan.provider == 'zhipu' else '', default_provider=plan.provider,
        timeout_seconds=plan.model.timeout_seconds, max_retries=0, deepseek_base_url='https://api.deepseek.com',
        zhipu_base_url='https://open.bigmodel.cn/api/paas/v4')


class ProviderRequestGate:
    """FIFO admission with a shared provider concurrency ceiling and durable events."""
    def __init__(self, maximum, path):
        self.maximum, self.path = maximum, Path(path)
        self.condition = threading.Condition()
        self.waiting, self.active, self.events = deque(), set(), []
        self.next_ticket = 0

    def _save(self):
        durable_json(self.path, {'maximum_concurrent_requests': self.maximum,
            'queue_time_counts_toward_original_episode_wall_budget': True, 'events': self.events,
            'active': sorted(self.active), 'pending': list(self.waiting)})

    def acquire(self, identity, deadline):
        with self.condition:
            if identity in self.active or any(row['identity'] == identity for row in self.waiting):
                raise ValueError('PROVIDER_ADMISSION_ID_REUSED')
            self.next_ticket += 1
            ticket = dict(ticket=self.next_ticket, identity=identity)
            self.waiting.append(ticket)
            self.events.append({**ticket, 'kind': 'queued', 'time': time.time()})
            self._save()
            while self.waiting[0] != ticket or len(self.active) >= self.maximum:
                if time.time() >= deadline:
                    self.waiting.remove(ticket)
                    self.events.append({**ticket, 'kind': 'expired', 'time': time.time()})
                    self._save(); self.condition.notify_all()
                    raise BudgetExceeded('WALL_DEADLINE_EXCEEDED_DURING_PROVIDER_QUEUE')
                self.condition.wait(timeout=min(.25, max(.001, deadline-time.time())))
            if time.time() >= deadline:
                self.waiting.popleft()
                self.events.append({**ticket, 'kind': 'expired', 'time': time.time()})
                self._save(); self.condition.notify_all()
                raise BudgetExceeded('WALL_DEADLINE_EXCEEDED_DURING_PROVIDER_QUEUE')
            self.waiting.popleft(); self.active.add(identity)
            self.events.append({**ticket, 'kind': 'admitted', 'time': time.time(), 'active_count': len(self.active)})
            self._save(); self.condition.notify_all()

    def release(self, identity):
        with self.condition:
            if identity not in self.active:
                raise ValueError('PROVIDER_ADMISSION_RELEASE_WITHOUT_ACTIVE_REQUEST')
            self.active.remove(identity)
            self.events.append({'identity': identity, 'kind': 'released', 'time': time.time(), 'active_count': len(self.active)})
            self._save(); self.condition.notify_all()

    def finish_episode(self, episode_id):
        # A filesystem/adapter exception may bypass settlement after reservation.
        # Release only admission capacity; the unresolved ledger charge remains.
        with self.condition:
            for identity in list(self.active):
                if identity.rsplit(':', 1)[0] == episode_id:
                    self.release(identity)


class ControlledProviderLedger:
    def __init__(self, book, gate, episode_ledger, source_guard, pacer=None):
        self.book, self.gate, self.episode_ledger, self.source_guard, self.pacer = book, gate, episode_ledger, source_guard, pacer

    def reserve(self, identity, incoming, outgoing, request_hash):
        self.source_guard()
        deadline = self.episode_ledger.snapshot()['deadline']
        self.gate.acquire(identity, deadline)
        try:
            if self.pacer is not None:
                self.pacer.grant(identity, wall_deadline=deadline)
            if self.episode_ledger.clock() >= deadline:
                raise BudgetExceeded('WALL_DEADLINE_EXCEEDED_DURING_PROVIDER_QUEUE')
            return self.book.reserve(identity, incoming, outgoing, request_hash)
        except BaseException:
            self.gate.release(identity)
            raise

    def settle(self, identity, *args, **kwargs):
        try: return self.book.settle(identity, *args, **kwargs)
        finally: self.gate.release(identity)

    def block(self, reason): return self.book.block(reason)
    def snapshot(self): return self.book.snapshot()


def runtime_manifest(extra_paths=()):
    manifest = design_implementation_manifest()
    paths = {ROOT / entry.path for entry in manifest.files}
    paths.update(ROOT / p for p in RUNTIME_FILES)
    paths.update(Path(p).resolve() for p in extra_paths)
    files = {str(path.resolve()): file_sha256(path) for path in sorted(paths)}
    return {'source_hash': manifest.source_hash, 'runtime_hash': research_hash(files), 'files': files}


def verify_history(current, previous):
    if current['contract'] != previous['contract'] or current['created_at'] != previous['created_at']:
        raise ValueError('PROVIDER_CUMULATIVE_AUTHORIZATION_OR_ORIGIN_CHANGED')
    for group in ('reservations', 'settlements'):
        if any(current[group].get(k) != v for k, v in previous[group].items()):
            raise ValueError('PROVIDER_CUMULATIVE_HISTORY_CHANGED:' + group)
    if previous['blocked_reason'] and current['blocked_reason'] != previous['blocked_reason']:
        raise ValueError('PROVIDER_BLOCK_STATE_REWRITTEN')


def validate_episode_identity(result, task, item, source_hash):
    if (result.source_hash != source_hash or result.task_hash != task.task_hash
            or (result.episode_id, result.task_id, result.method) != (item['episode_id'], item['task_id'], item['method'])):
        raise ValueError('PROVIDER_EPISODE_SOURCE_TASK_OR_SCHEDULE_MISMATCH')
    calls = result.common_contract['live_calls']
    if type(calls) is not int or not 0 <= calls <= task.budget.llm_calls:
        raise ValueError('PROVIDER_EPISODE_CALL_COUNT_INVALID')
    if item['method'] == 'central_rule':
        if calls != 0 or result.live_status != 'NOT_RUN':
            raise ValueError('PROVIDER_RULE_EPISODE_CANNOT_CONTAIN_LIVE_CALLS')
    elif calls > 0:
        if result.live_status != 'RUN':
            raise ValueError('PROVIDER_LIVE_EPISODE_SOURCE_MISMATCH')
    elif result.terminal_state not in {'METHOD_FAILED', 'BUDGET_UNRESOLVED', 'BUDGET_EXHAUSTED_UNRESOLVED'}:
        raise ValueError('PROVIDER_ZERO_CALL_LIVE_EPISODE_REQUIRES_EXPLICIT_FAILURE')


def run_provider_pilot(plan_path, output, *, credential_path, spend_path, workers=16, extra_runtime_paths=(), pacing_path=None):
    if type(workers) is not int or workers != 16:
        raise ValueError('PROVIDER_REGISTERED_SIXTEEN_WORKERS_REQUIRED')
    plan = load_provider_pilot(plan_path)
    tasks, split_check = load_provider_inputs(plan)
    import importlib.metadata
    actual_sdk = native_sdk_version() if plan.provider == 'gemini' else importlib.metadata.version('openai')
    if plan.model.sdk_version != actual_sdk:
        raise ValueError('PROVIDER_RUNTIME_SDK_CHANGED')
    if plan.provider == 'deepseek' and plan.parent_liability_reference is None:
        raise ValueError('DEEPSEEK_RETIREMENT_AND_RELEASE_RECEIPT_REQUIRED_BEFORE_LIVE')
    settings = load_provider_settings(plan, credential_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('IMMUTABLE_PROVIDER_PILOT_OUTPUT_ALREADY_EXISTS')
    if Path(spend_path).exists():
        raise ValueError('PROVIDER_PILOT_ALLOCATION_ALREADY_EXISTS')
    if plan.provider == 'gemini' and pacing_path is None:
        raise ValueError('GOOGLE_EXPLICIT_SHARED_PACING_LEDGER_REQUIRED')
    runtime = runtime_manifest(extra_runtime_paths)
    splits = _checked(plan.split_manifest_path, plan.split_manifest_sha256)
    bound_inputs = {str(Path(plan_path).resolve()): file_sha256(plan_path), **plan.task_sha256,
        plan.split_manifest_path: plan.split_manifest_sha256, **plan.prior_split_manifests,
        plan.planning_design_path: plan.planning_design_sha256,
        plan.planning_ancestor_path: plan.planning_ancestor_sha256,
        plan.experiment_spec_path: plan.experiment_spec_sha256,
        **{e['task_path']: e['sha256'] for e in splits['tasks']}, **reference_paths(plan)}
    def unchanged():
        for path, sha in {**runtime['files'], **bound_inputs}.items():
            if file_sha256(path) != sha:
                raise ValueError('PROVIDER_REGISTERED_SOURCE_OR_INPUT_CHANGED:' + path)
    parent_ref = None if plan.parent_liability_reference is None else plan.parent_liability_reference.model_dump(mode='json')
    parent = None if parent_ref is None else _checked(parent_ref['path'], parent_ref['sha256'])
    liability = 0 if parent is None else parent['known_settled_cost_nano']+parent['unknown_full_reservation_nano']
    spend = ProviderUsageLedger(spend_path, authorization_id=plan.authorization_id, plan_id=plan.plan_id,
        plan_hash=research_hash(plan), maximum_calls=plan.maximum_generation_calls,
        input_cap=plan.model.input_reservation_tokens, output_cap=plan.model.max_output_tokens,
        input_nano=plan.input_price_nano, output_nano=plan.output_price_nano, cached_nano=plan.cached_price_nano,
        provider=plan.provider, currency=plan.currency, monetary_cap_nano=plan.monetary_cap_nano,
        inherited_liability_nano=liability, source_guard=unchanged, parent_reference=parent_ref)
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir()
    (output/'failed_attempts').mkdir()
    gate = ProviderRequestGate(plan.maximum_concurrent_requests, output/'request_admission.json')
    pacer = SQLiteRequestPacer(pacing_path, minimum_interval_seconds=5.1) if plan.provider == 'gemini' else None
    gate._save()
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
    registration = dict(schema_version=5, study_stage='provider_pilot', provider=plan.provider, study_role=plan.study_role,
        no_cross_provider_power_pooling=True, plan_hash=research_hash(plan),
        source_hash=runtime['source_hash'], runtime_hash=runtime['runtime_hash'],
        runtime_source_hashes=runtime['files'], input_hashes=bound_inputs, registered_at=time.time(),
        schedule=schedule, split_check=split_check, maximum_generation_calls=plan.maximum_generation_calls,
        batch_task_count=2, worker_count=workers, batch_stop_rule=plan.stop_rule,
        spend_contract=spend.contract, protocol_version=plan.protocol_version,
        feedback_version=plan.feedback_version, planning_design_sha256=plan.planning_design_sha256,
        analysis_scope='development feasibility, variance and cost only; not confirmatory effects',
        natural_collection='all scheduled live methods; deterministic reference separate')
    registration.update(maximum_concurrent_requests=plan.maximum_concurrent_requests,
        pacing_ledger_path=str(Path(pacing_path).resolve()) if pacing_path else None,
        minimum_request_interval_seconds=plan.minimum_request_interval_seconds,
        host_queue_time_counts_toward_original_episode_wall_budget=True,
        request_admission_version='provider-fifo-concurrency-and-google-pacing-v1')
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
        index = dict(schema_version=5, study_stage='provider_pilot', provider=plan.provider, study_role=plan.study_role,
            no_cross_provider_power_pooling=True, plan_hash=research_hash(plan),
            source_hash=runtime['source_hash'], runtime_hash=runtime['runtime_hash'],
            registration_sha256=file_sha256(output/'registration.json'), episodes=episodes,
            planned_trials=len(schedule), indexed_trials=len(episodes), independent_task_count=len(tasks),
            market_runs=0, llm_repeats_per_task_method=1, stopping_reason=stopping,
            spend_snapshot=current, formal_status='NOT_RUN', financial_status='SEPARATE_QUALIFICATION')
        index['request_admission_sha256'] = file_sha256(output/'request_admission.json')
        if pacer is not None:
            durable_json(output/'pacing_snapshot.json', pacer.snapshot())
            index['pacing_snapshot_sha256'] = file_sha256(output/'pacing_snapshot.json')
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
                        settings=settings, spend_ledger=ControlledProviderLedger(spend, gate, ledger, unchanged, pacer)), turns=plan.design_turns)
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
        finally:
            gate.finish_episode(item['episode_id'])
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
