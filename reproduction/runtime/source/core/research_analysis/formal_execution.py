"""F70 analytic execution with immutable post-selection independent evaluation.

The design implementation remains the previously frozen protocol.  This outer
coordinator owns formal registration, terminal selection records, evaluator
quota, and complete trial accounting.  It never feeds audit results to a method.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import json
import os
import time

from pydantic import Field, model_validator

from core.experiments.archive import durable_json
from core.experiments.evidence import Hash
from core.experiments.identity import file_sha256, research_hash
from core.experiments.strict_types import StrictResearchModel, load_research_model, strict_json_loads
from core.codesign.schema import TaskSpec, EpisodeResult
from core.codesign.runner import run_design_episode, verify_episode, freeze_design_source, design_implementation_manifest
from core.codesign.verifier import compile_candidate, ExactEvidenceService, evaluate_eligibility
from core.codesign.live_budget import SpendLedger, load_deepseek_settings
from core.codesign.live_protocol import LiveProtocolMethod, SharedEvidenceProtocolMethod
from core.codesign.llm_adapter import DesignAgentClient
from core.codesign.baselines import CentralRuleMethod
from .study_freeze import (AuditPlan, FileRef, FreezeGateError, FreezeIntegrityError,
                           check_ref, reference, verify_execution_envelope)


class SelectionRecord(StrictResearchModel):
    schema_version: Literal[1]
    episode_id: str
    task_id: str
    original_task_hash: Hash
    executed_task_hash: Hash | None
    terminal_record: FileRef
    selected_policy_hash: Hash | None
    shortlist: list[Hash]
    raw_claim_policy_hashes: list[Hash]
    method_claim_policy_hashes: list[Hash]
    certified_policy_hashes: list[Hash]
    selection_rule: Literal['max_min_margin_cost_complexity_hash_v1']
    closed_at_utc: str
    disposition: Literal['RECORDED', 'EXECUTION_FAILED', 'NOT_RUN_STOPPING_RULE']
    source: Literal['live', 'rule']
    plan_core: FileRef

    @model_validator(mode='after')
    def _valid(self):
        if len(self.shortlist) != len(set(self.shortlist)):
            raise ValueError('DUPLICATE_SELECTION_SHORTLIST')
        if self.disposition != 'RECORDED' and (self.selected_policy_hash or self.shortlist):
            raise ValueError('FAILED_OR_UNRUN_TRIAL_HAS_NO_SELECTION')
        return self


class EvaluatorLedger:
    """Independent finite quota; an attempted audit is never refunded."""
    def __init__(self, path, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError('INVALID_EVALUATOR_LIMIT')
        self.path, self.limit = Path(path), limit
        if self.path.exists():
            raise FreezeIntegrityError('IMMUTABLE_EVALUATOR_LEDGER_ALREADY_EXISTS')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = {'schema_version': 1, 'limit': limit, 'attempts': {},
                      'llm_calls': 0, 'market_runs': 0, 'cost_cny': 0.0}
        self._save()

    def _save(self):
        durable_json(self.path, {'state': self.state, 'sha256': research_hash(self.state)})

    def reserve(self, object_id, policy_hash):
        if object_id in self.state['attempts']:
            raise FreezeIntegrityError('AUDIT_OBJECT_ATTEMPT_REUSED')
        if len(self.state['attempts']) >= self.limit:
            raise FreezeGateError('INDEPENDENT_EVALUATOR_BUDGET_EXHAUSTED')
        self.state['attempts'][object_id] = {'policy_hash': policy_hash, 'status': 'RESERVED'}
        self._save()

    def settle(self, object_id, status):
        if self.state['attempts'][object_id]['status'] != 'RESERVED':
            raise FreezeIntegrityError('AUDIT_OBJECT_ALREADY_SETTLED')
        self.state['attempts'][object_id]['status'] = status
        self._save()


def _write_immutable(path, body):
    path = Path(path)
    if path.exists():
        raise FreezeIntegrityError('IMMUTABLE_RECORD_ALREADY_EXISTS:' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_json(path, body)
    os.chmod(path, 0o444)
    return reference(path)


def freeze_selection(task, terminal_path, output, *, episode_id, source, plan_core,
                     disposition='RECORDED'):
    """Persist the actual output without inspecting the audit coverage or truth."""
    terminal_path = Path(terminal_path)
    result = load_research_model(terminal_path, EpisodeResult) if disposition == 'RECORDED' else None
    if result and (result.episode_id != episode_id or result.task_id != task.task_id):
        raise FreezeIntegrityError('SELECTION_EPISODE_IDENTITY_MISMATCH')
    record = SelectionRecord(schema_version=1, episode_id=episode_id, task_id=task.task_id,
        original_task_hash=task.task_hash, executed_task_hash=result.task_hash if result else None,
        terminal_record=reference(terminal_path),
        selected_policy_hash=result.selected_policy_hash if result else None,
        shortlist=list(result.shortlist) if result else [],
        raw_claim_policy_hashes=list(result.A_raw) if result else [],
        method_claim_policy_hashes=list(result.A_method) if result else [],
        certified_policy_hashes=list(result.A_cert) if result else [],
        selection_rule=task.final_selection, closed_at_utc=datetime.now(timezone.utc).isoformat(),
        disposition=disposition, source=source, plan_core=plan_core)
    return _write_immutable(output, record.model_dump(mode='json'))


def audit_frozen_selection(selection_ref, task, audit_plan_loader, output, *,
                           source_hash, evaluator_ledger):
    """The loader is invoked only after selection and terminal integrity checks.

    For exact analytic tasks there are no hidden replication IDs.  The separate
    evaluator reissues all registered atoms from the immutable finite table; its
    files and quota are disjoint from the designer's evidence and budget.
    """
    selection_path = check_ref(selection_ref)
    selection = load_research_model(selection_path, SelectionRecord)
    check_ref(selection.terminal_record)
    check_ref(selection.plan_core)
    if selection.original_task_hash != task.task_hash:
        raise FreezeIntegrityError('AUDIT_TASK_SELECTION_MISMATCH')
    # This line is the first access to the final evaluator's coverage/IDs.
    plan = AuditPlan.model_validate(audit_plan_loader())
    if (task.origin != 'analytic_fixture' or task.evidence_mode != 'exact_analytic'
            or set(plan.required_role_ids) != {r.role_id for r in task.roles}
            or set(plan.required_model_ids) != {m.model_id for m in task.model_family}):
        raise FreezeIntegrityError('AUDIT_EXACT_SCOPE_MISMATCH')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan_ref = _write_immutable(output / 'AuditPlan.json', plan.model_dump(mode='json'))
    selected = selection.selected_policy_hash
    objects = ([selected] if selected else []) if plan.coverage == 'selected_only' else list(selection.shortlist)
    coverage_error = None
    if plan.coverage == 'frozen_shortlist' and selected and selected not in objects:
        coverage_error = 'SELECTED_POLICY_MISSING_FROM_FROZEN_SHORTLIST'
    if len(objects) > plan.maximum_policies_per_episode:
        # Preserve the entire output; do not choose a convenient subset to fit K.
        coverage_error = 'FROZEN_SHORTLIST_EXCEEDS_REGISTERED_K'
    by_hash = {}
    for candidate in task.candidate_pool:
        try:
            ph = compile_candidate(task, candidate).policy_hash
        except ValueError:
            continue
        if ph in by_hash:
            raise FreezeIntegrityError('AMBIGUOUS_POLICY_HASH_IN_POOL')
        by_hash[ph] = candidate
    audit_rows = []
    for ordinal, ph in enumerate(objects):
        object_id = f'{selection.episode_id}--audit-{ordinal:02d}'
        row = {'object_id': object_id, 'policy_hash': ph, 'status': 'UNRESOLVED',
               'truth': 'UNKNOWN', 'selection_record': selection_ref.model_dump(mode='json'),
               'audit_plan': plan_ref.model_dump(mode='json'), 'evidence_refs': [],
               'claim_ids': [], 'attempt_count': 0, 'llm_calls': 0, 'market_runs': 0, 'cost_cny': 0.0}
        try:
            if coverage_error:
                raise FreezeIntegrityError(coverage_error)
            evaluator_ledger.reserve(object_id, ph)
            row['attempt_count'] = 1
            if ph not in by_hash:
                raise FreezeIntegrityError('SELECTED_POLICY_OUTSIDE_FROZEN_POOL')
            service = ExactEvidenceService(task, source_hash, output / object_id)
            evidence = service.issue(by_hash[ph])
            eligibility = evaluate_eligibility(task, by_hash[ph], evidence, evidence_service=service)
            row['evidence_refs'] = [reference(service.issued[e.evidence_id]).model_dump(mode='json') for e in evidence]
            row['claim_ids'] = [e.atom_id for e in evidence]
            row['eligibility'] = eligibility
            row['truth'] = {'SUPPORTED': 'TRUE', 'REFUTED': 'FALSE'}.get(eligibility['status'], 'UNKNOWN')
            row['status'] = 'COMPLETED' if row['truth'] != 'UNKNOWN' else 'UNRESOLVED'
        except (ValueError, RuntimeError, OSError) as exc:
            row.update(status='FAILED', error_type=type(exc).__name__, reason=str(exc))
        if row['attempt_count']:
            evaluator_ledger.settle(object_id, row['status'])
        audit_rows.append(row)
    # The primary raw-claim truth denominator is explicitly confined to coverage.
    truth = {r['policy_hash']: r['truth'] for r in audit_rows}
    raw = [{'policy_hash': ph, 'covered_policy': ph in truth, 'raw_assent_truth': 'UNKNOWN',
            'joint_policy_acceptability': truth.get(ph, 'UNKNOWN')}
           for ph in selection.raw_claim_policy_hashes]
    result = {'schema_version': 1, 'episode_id': selection.episode_id,
              'selection_record': selection_ref.model_dump(mode='json'),
              'audit_plan': plan_ref.model_dump(mode='json'), 'audit_plan_hash': research_hash(plan),
              'status': 'NO_APPLICABLE_OBJECT' if not objects else 'RECORDED',
              'no_selection_substitution': False, 'objects': audit_rows,
              'raw_claim_coverage': raw, 'design_feedback': False,
              'raw_assent_truth_scope': 'policy-level A_raw omits individual claim scope; no raw-assent truth imputation',
              'evaluator_budget': {'candidate_evaluations': sum(r['attempt_count'] for r in audit_rows),
                                   'llm_calls': 0, 'market_runs': 0, 'cost_cny': 0.0}}
    _write_immutable(output / 'audit_result.json', result)
    # Detect mutation by an evaluator as well as by an external writer.
    check_ref(selection_ref)
    check_ref(selection.terminal_record)
    return result


def _check_model_availability(settings, config):
    """Read-only provider preflight; retries/fallback stay disabled."""
    if config.provider != 'deepseek' or config.source != 'live' or config.max_request_attempts != 1:
        raise FreezeGateError('SUPPORTED_EXPLICIT_DEEPSEEK_LIVE_CONFIGURATION_REQUIRED')
    import openai
    try:
        with openai.OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
                           timeout=config.timeout_seconds, max_retries=0) as client:
            available = {m.id for m in client.models.list().data}
    except openai.APIError as exc:
        raise FreezeGateError('PROVIDER_PREFLIGHT_UNAVAILABLE:' + type(exc).__name__) from None
    if config.model_id not in available:
        raise FreezeGateError('REGISTERED_MODEL_UNAVAILABLE')
    return {'provider': config.provider, 'model': config.model_id, 'available': True,
            'check': 'read_only_models_list', 'llm_generation_calls': 0}


def _schedule(core, tasks):
    schedule = []
    for i, task in enumerate(tasks):
        order = core.methods[i % len(core.methods):] + core.methods[:i % len(core.methods)]
        schedule.extend({'episode_id': task.task_id + '--' + method,
                         'task_id': task.task_id, 'method': method} for method in order)
    return schedule


def verify_cumulative_spend(current, frozen, previous):
    """A self-consistent hash cannot authorize rewriting cumulative history."""
    for baseline in (frozen, previous):
        if current['contract'] != baseline['contract'] or current['created_at'] != baseline['created_at']:
            raise FreezeIntegrityError('CUMULATIVE_SPEND_AUTHORIZATION_OR_ORIGIN_CHANGED')
        for group in ('reservations', 'settlements'):
            if any(current[group].get(key) != value for key, value in baseline[group].items()):
                raise FreezeIntegrityError('CUMULATIVE_SPEND_HISTORY_CHANGED:' + group)
        if baseline['blocked_reason'] and current['blocked_reason'] != baseline['blocked_reason']:
            raise FreezeIntegrityError('CUMULATIVE_SPEND_STOP_STATE_CHANGED')
    if not set(current['settlements']) <= set(current['reservations']):
        raise FreezeIntegrityError('SETTLEMENT_WITHOUT_RESERVATION')
    contract = current['contract']
    for reservation in current['reservations'].values():
        if (any(type(reservation[k]) is not int or reservation[k] < 1 for k in ('input_cap', 'output_cap', 'reserved_nano'))
                or reservation['reserved_nano'] != reservation['input_cap'] * contract['input_nano']
                   + reservation['output_cap'] * contract['output_nano']):
            raise FreezeIntegrityError('INVALID_CUMULATIVE_SPEND_RESERVATION')
    if sum(r['reserved_nano'] for r in current['reservations'].values()) > contract['cap_nano']:
        raise FreezeIntegrityError('CUMULATIVE_SPEND_EXCEEDS_AUTHORIZATION')


def run_formal_benchmark(envelope_path, output, *, credential_path, workers=1):
    """Run a fully frozen analytic plan.  Scientific effects remain unanalyzed."""
    if type(workers) is not int or workers != 1:
        raise FreezeGateError('FORMAL_SERIAL_EXECUTION_REQUIRED_FOR_GLOBAL_STOP_RULE')
    core, analysis, freeze = verify_execution_envelope(envelope_path)
    if core.scope != 'exact_analytic':
        raise FreezeGateError('FINANCIAL_SCOPE_REQUIRES_HEALTH_QUALIFICATION')
    tasks = [load_research_model(check_ref(ref), TaskSpec) for ref in core.tasks]
    for trial in core.schedule:
        if (Path(trial.episode_id).name != trial.episode_id or trial.episode_id in ('.', '..')
                or '\\' in trial.episode_id or '\x00' in trial.episode_id):
            raise FreezeIntegrityError('FORMAL_EPISODE_ID_MUST_BE_SAFE_FILENAME')
    if any(t.evidence_mode != 'exact_analytic' or t.split != 'test' for t in tasks):
        raise FreezeIntegrityError('FORMAL_ANALYTIC_HELDOUT_SCOPE_REQUIRED')
    if design_implementation_manifest().source_hash != core.source_hash:
        raise FreezeIntegrityError('FORMAL_SOURCE_CHANGED')
    output = Path(output).resolve()
    if output.exists():
        raise FreezeIntegrityError('IMMUTABLE_FORMAL_OUTPUT_ALREADY_EXISTS')
    settings = load_deepseek_settings(credential_path)
    model_check = _check_model_availability(settings, core.model)
    auth = core.authorization
    if auth.cap_nano % 10**9:
        raise FreezeIntegrityError('UNSUPPORTED_FRACTIONAL_CNY_AUTHORIZATION')
    spend = SpendLedger(auth.ledger.path, authorization_id=auth.authorization_id,
        cap_cny=auth.cap_nano // 10**9, input_nano=auth.input_price_nano,
        output_nano=auth.output_price_nano, cached_nano=auth.cached_price_nano)
    before = spend.snapshot()
    frozen_wrapper = strict_json_loads(check_ref(freeze.spend_snapshot).read_text())
    frozen_spend = frozen_wrapper['state']
    if research_hash(frozen_spend) != frozen_wrapper['sha256']:
        raise FreezeIntegrityError('FROZEN_SPEND_HASH_MISMATCH')
    verify_cumulative_spend(before, frozen_spend, frozen_spend)
    if before['blocked_reason']:
        raise FreezeGateError('SPEND_BLOCKED:' + before['blocked_reason'])
    maximum = sum(core.design_turns + len(t.required_signers) for t in tasks) * (len(core.methods) - 1) * (
        core.model.input_reservation_tokens * auth.input_price_nano + core.model.max_output_tokens * auth.output_price_nano)
    if maximum + sum(r['reserved_nano'] for r in before['reservations'].values()) > auth.cap_nano:
        raise FreezeGateError('FORMAL_DESIGN_EXCEEDS_REMAINING_AUTHORIZATION')
    output.mkdir(parents=True, exist_ok=False)
    manifest = freeze_design_source(output)
    schedule = [r.model_dump(mode='json') for r in core.schedule]
    if [{k: r[k] for k in ('episode_id', 'task_id', 'method')} for r in schedule] != _schedule(core, tasks):
        raise FreezeIntegrityError('FORMAL_SCHEDULE_CHANGED')
    envelope = strict_json_loads(Path(envelope_path).read_text())
    core_ref = FileRef.model_validate(envelope['plan_core'])
    registration = {'schema_version': 1, 'study_stage': 'formal', 'origin': 'real_execution',
                    'envelope': reference(envelope_path).model_dump(mode='json'),
                    'plan_core': core_ref.model_dump(mode='json'), 'source_hash': manifest.source_hash,
                    'registered_at': time.time(), 'schedule': schedule, 'model_check': model_check,
                    'audit_plan_hash': research_hash(core.audit_plan),
                    'scope': 'exact_analytic', 'scientific_effects_status': 'NOT_ANALYZED'}
    _write_immutable(output / 'registration.json', registration)
    _write_immutable(output / 'spend_before.json', before)
    jobs_path = output / 'jobs.jsonl'
    jobs_path.touch(exist_ok=False)

    def append_job(body):
        with jobs_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    evaluator = EvaluatorLedger(output / 'evaluator_budget.json', core.audit_plan.evaluator_budget.candidate_evaluations)
    task_map = {t.task_id: t for t in tasks}
    episodes, audits, consecutive_failed, stopping = [], [], 0, None
    previous_spend = before
    observed_spend = before
    spend_integrity = {'status': 'VERIFIED', 'last_verified_state_hash': research_hash(before)}

    def refresh_spend():
        nonlocal stopping, previous_spend, observed_spend, spend_integrity
        try:
            current = spend.snapshot()
            observed_spend = current
            verify_cumulative_spend(current, frozen_spend, previous_spend)
            # Once a violation is observed, subsequent rewrites cannot erase it.
            if spend_integrity['status'] != 'FAILED':
                previous_spend = current
                spend_integrity = {'status': 'VERIFIED', 'last_verified_state_hash': research_hash(current)}
            stopping = stopping or current['blocked_reason']
        except (ValueError, KeyError, TypeError, OSError) as exc:
            stopping = 'CUMULATIVE_SPEND_INTEGRITY_FAILURE'
            if spend_integrity['status'] != 'FAILED':
                spend_integrity = {'status': 'FAILED', 'reason': str(exc),
                    'last_verified_state_hash': research_hash(previous_spend)}
                _write_immutable(output / 'spend_integrity_failure.json', {
                    **spend_integrity, 'last_verified_snapshot': previous_spend,
                    'observed_snapshot': observed_spend, 'observed_at': time.time()})

    def save():
        # Includes the last episode: there need not be a next loop iteration.
        refresh_spend()
        objects = [obj for audit in audits for obj in audit.get('objects', [])]
        raw_mentions = [r for audit in audits for r in audit.get('raw_claim_coverage', [])]
        durable_json(output / 'audit_index.json', {'schema_version': 1, 'study_stage': 'formal',
            'audit_plan_hash': research_hash(core.audit_plan), 'episodes': audits,
            'evaluator_budget': evaluator.state, 'design_feedback': False,
            'coverage': {'policy_object_denominator': len(objects),
                'decidable_policy_objects': sum(r['truth'] != 'UNKNOWN' for r in objects),
                'unknown_policy_objects': sum(r['truth'] == 'UNKNOWN' for r in objects),
                'failed_policy_objects': sum(r['status'] == 'FAILED' for r in objects),
                'episode_audit_failures': sum(r['status'] == 'FAILED' for r in audits),
                'no_applicable_object_episodes': sum(r['status'] == 'NO_APPLICABLE_OBJECT' for r in audits),
                'raw_policy_mention_denominator': len(raw_mentions),
                'uncovered_raw_policy_mentions': sum(not r['covered_policy'] for r in raw_mentions),
                'raw_assent_truth': 'UNKNOWN_SCOPE_NOT_IN_POLICY_LEVEL_A_RAW'}})
        recorded = sum(r['status'] == 'RECORDED' for r in episodes)
        failed = sum(r['status'] == 'EXECUTION_FAILED' for r in episodes)
        missing = sum(r['status'] == 'NOT_RUN_STOPPING_RULE' for r in episodes)
        index = {'schema_version': 1, 'study_stage': 'formal', 'scope': 'exact_analytic',
                 'plan_id': core.plan_id, 'source_hash': core.source_hash,
                 'registration': reference(output / 'registration.json').model_dump(mode='json'),
                 'episodes': episodes, 'planned_trials': len(schedule), 'indexed_trials': len(episodes),
                 'started_trials': recorded + failed, 'recorded_trials': recorded,
                 'execution_failed_trials': failed, 'not_run_trials': missing,
                 'method_failed_trials': sum(r.get('terminal_state') == 'METHOD_FAILED' for r in episodes),
                 'terminal_state_counts': {state: sum(r.get('terminal_state') == state for r in episodes)
                     for state in sorted({r['terminal_state'] for r in episodes if 'terminal_state' in r})},
                 'all_registered_trials_have_terminal_records': len(episodes) == len(schedule) and missing == 0,
                 'formal_status': 'EXECUTED' if len(episodes) == len(schedule) and missing == 0 else 'INCOMPLETE_FORMAL',
                 'scientific_effects_status': 'NOT_ANALYZED', 'stopping_reason': stopping,
                 'spend_snapshot': observed_spend, 'spend_history_integrity': spend_integrity,
                 'ledger_blocked_reason': observed_spend['blocked_reason'],
                 'jobs': reference(jobs_path).model_dump(mode='json'),
                 'audit_index': reference(output / 'audit_index.json').model_dump(mode='json')}
        durable_json(output / 'episode_index.json', {'episodes': episodes})
        durable_json(output / 'benchmark_index.json', index)
        return index

    for item in schedule:
        task, method = task_map[item['task_id']], item['method']
        source = 'rule' if method == 'central_rule' else 'live'
        destination = output / 'episodes' / item['episode_id']
        row = {**item, 'source': source, 'status': 'NOT_RUN_STOPPING_RULE'}
        refresh_spend()
        if not stopping and design_implementation_manifest().source_hash != core.source_hash:
            stopping = 'FORMAL_SOURCE_CHANGED'
        if not stopping:
            try:
                verify_execution_envelope(envelope_path, check_live_inputs=False)
            except (ValueError, OSError) as exc:
                stopping = 'FORMAL_BOUND_INPUT_CHANGED:' + type(exc).__name__
        if stopping:
            terminal = output / 'unrun' / (item['episode_id'] + '.json')
            _write_immutable(terminal, {**item, 'status': 'NOT_RUN_STOPPING_RULE', 'reason': stopping})
            row['reason'] = stopping
        else:
            append_job({**item, 'stage': 'DESIGN_STARTED', 'recorded_at': time.time()})
            try:
                actual_task = task
                if method == 'central_rule':
                    body = task.model_dump(mode='json')
                    body.update(signing_layer='explicit_rule_owners_v1', model_configurations=[])
                    actual_task = TaskSpec.model_validate(body)
                    implementation = CentralRuleMethod(protocol_enabled=True)
                else:
                    protocol = SharedEvidenceProtocolMethod if core.protocol_version == 'bounded_llm_protocol_v3' else LiveProtocolMethod
                    implementation = protocol(method, lambda ledger, path: DesignAgentClient(core.model, ledger, path,
                        allow_live=True, settings=settings, spend_ledger=spend), turns=core.design_turns)
                result = run_design_episode(actual_task, implementation, destination, episode_id=item['episode_id'])
                verification = verify_episode(destination)
                if result.source_hash != core.source_hash or result.task_hash != actual_task.task_hash:
                    raise FreezeIntegrityError('EXECUTED_TASK_OR_SOURCE_MISMATCH')
                if source == 'live' and result.common_contract['live_calls'] and result.live_status != 'RUN':
                    raise FreezeIntegrityError('FORMAL_LIVE_SOURCE_MISMATCH')
                terminal = destination / 'episode_result.json'
                row.update(status='RECORDED', path=str(destination), terminal_state=result.terminal_state,
                           verification=verification, live_calls=result.common_contract['live_calls'],
                           result_sha256=file_sha256(terminal))
                consecutive_failed = consecutive_failed + 1 if result.terminal_state == 'METHOD_FAILED' else 0
                if consecutive_failed >= 3:
                    stopping = 'THREE_CONSECUTIVE_FAILED_EPISODES'
            except (ValueError, RuntimeError, OSError) as exc:
                terminal = output / 'failed_attempts' / (item['episode_id'] + '.json')
                _write_immutable(terminal, {**item, 'status': 'EXECUTION_FAILED',
                    'error_type': type(exc).__name__, 'reason': str(exc), 'attempt_directory': str(destination)})
                row.update(status='EXECUTION_FAILED', reason=str(exc), error_type=type(exc).__name__)
                stopping = 'EXECUTION_INTEGRITY_OR_SERVICE_FAILURE'
        selection = freeze_selection(task, terminal, output / 'selections' / (item['episode_id'] + '.json'),
            episode_id=item['episode_id'], source=source, plan_core=core_ref, disposition=row['status'])
        row['selection_record'] = selection.model_dump(mode='json')
        # No design call occurs after this selection is saved for the episode.
        audit_directory = output / 'audits' / item['episode_id']
        try:
            audit = audit_frozen_selection(selection, task, lambda: core.audit_plan.model_dump(mode='json'),
                audit_directory, source_hash=core.source_hash, evaluator_ledger=evaluator)
        except (ValueError, RuntimeError, OSError) as exc:
            stopping = 'INDEPENDENT_AUDIT_INTEGRITY_FAILURE'
            audit = {'schema_version': 1, 'episode_id': item['episode_id'], 'status': 'FAILED',
                     'selection_record': selection.model_dump(mode='json'), 'truth': 'UNKNOWN',
                     'error_type': type(exc).__name__, 'reason': str(exc), 'design_feedback': False}
            audit_directory.mkdir(parents=True, exist_ok=True)
            _write_immutable(audit_directory / 'audit_failure.json', audit)
        audits.append(audit)
        audit_path = audit_directory / ('audit_failure.json' if audit['status'] == 'FAILED' else 'audit_result.json')
        row['audit_result'] = reference(audit_path).model_dump(mode='json')
        append_job({'episode_id': item['episode_id'], 'stage': 'TERMINAL_AND_AUDIT_RECORDED',
                    'status': row['status'], 'selection_record': row['selection_record'],
                    'audit_result': row['audit_result'], 'recorded_at': time.time()})
        episodes.append(row)
        save()
    index = save()
    _write_immutable(output / 'COMPLETED.json', {
        'index_sha256': file_sha256(output / 'benchmark_index.json'),
        'audit_index_sha256': file_sha256(output / 'audit_index.json'),
        'complete_planned_index': len(episodes) == len(schedule),
        'formal_execution_complete': index['all_registered_trials_have_terminal_records']})
    return index
