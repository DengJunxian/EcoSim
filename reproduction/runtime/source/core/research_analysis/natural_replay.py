"""Frozen natural-message replay of the four existing CDR local observers.

This is a controlled intervention on a fixed message stream.  It does not
simulate how an agent would respond to a different observer's feedback.  Raw
messages and the common transition path remain identical in all four arms.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
import json
import os

from pydantic import Field, model_validator

from core.codesign.cdr import CDRProtocol
from core.codesign.live_protocol import LLMOwnerService
from core.codesign.protocol import AuthenticatedInput, ProtocolServices, ProtocolState, advance_protocol
from core.codesign.runner import verify_episode
from core.codesign.schema import AgentMessage, AssentReceipt, TaskSpec
from core.codesign.services import InformationService
from core.codesign.traces import import_trace_collection
from core.codesign.verifier import ExactEvidenceService, compile_candidate, required_atoms
from core.experiments.archive import durable_json
from core.experiments.evidence import EvidenceRecord
from core.experiments.identity import file_sha256, research_hash
from core.experiments.research_cli import local_import_closure
from core.experiments.strict_types import StrictResearchModel, strict_json_loads

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('cdr_s0_v0', 'cdr_s0_v1', 'cdr_s1_v0', 'cdr_s1_v1')
SOURCE_METHODS = ('single', 'independent', 'free_debate', 'abn', *ARMS)
MECHANISM_FILES = ('core/codesign/cdr.py', 'core/codesign/protocol.py',
                   'core/codesign/verifier.py', 'core/codesign/services.py')


class ReplayIntegrityError(ValueError):
    pass


class ReplayGateError(ValueError):
    pass


class Ref(StrictResearchModel):
    path: str
    sha256: str = Field(pattern='^[0-9a-f]{64}$')


def ref(path):
    p = Path(path).resolve()
    return Ref(path=str(p), sha256=file_sha256(p)).model_dump(mode='json')


def checked(value):
    value = Ref.model_validate(value)
    p = Path(value.path)
    if not p.is_file() or file_sha256(p) != value.sha256:
        raise ReplayIntegrityError('REPLAY_INPUT_HASH_MISMATCH:' + value.path)
    return p


def write_once(path, body):
    path = Path(path)
    if path.exists():
        raise ReplayIntegrityError('REPLAY_IMMUTABLE_OUTPUT_EXISTS:' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_json(path, body)
    os.chmod(path, 0o444)
    return ref(path)


class SourceReservation(StrictResearchModel):
    """Required before source generation for a future confirmatory replay."""
    schema_version: Literal[1]
    purpose: Literal['prospective_natural_replay_holdout']
    registered_at: float
    planned_episode_ids: list[str] = Field(min_length=1)
    minimum_independent_clusters: int = Field(ge=2)
    sampling_rule: Literal['all_planned_episodes_no_outcome_selection']
    source_method_strata: list[str]
    replay_protocol_sha256: str = Field(pattern='^[0-9a-f]{64}$')
    analysis_family: Literal['S_V_SV_local_observer_only']


class ReplayPlan(StrictResearchModel):
    schema_version: Literal[1]
    study_id: str = Field(min_length=1)
    study_stage: Literal['diagnostic', 'formal']
    source_collection: str
    source_reservation: Ref | None
    minimum_independent_clusters: int = Field(ge=2)
    source_use: Literal['retrospective_development', 'prospectively_reserved_replay_holdout']
    arms: list[str]
    source_method_strata: list[str]
    inclusion_rule: Literal['all_collected_live_episodes_including_failures']
    cluster_rule: Literal['underlying_task_cluster_shared_across_source_methods']
    estimand: Literal['finite_source_census_local_filter_effect_given_fixed_messages']
    primary_outcome: Literal['false_local_method_agreement_count']
    inference: Literal['exact_finite_source_census_no_superpopulation_test']
    language_adaptation_claim: Literal[False]
    visibility: Literal['original_context_evidence_and_original_message_order']
    audit_coverage: Literal['complete_original_finite_candidate_pool']
    maximum_candidates_per_source: int = Field(ge=1)
    audit_candidate_budget: int = Field(ge=1)
    llm_calls: Literal[0]
    market_runs: Literal[0]

    @model_validator(mode='after')
    def valid(self):
        if self.arms != list(ARMS) or self.source_method_strata != list(SOURCE_METHODS):
            raise ValueError('REPLAY_FIXED_ARM_AND_SOURCE_STRATA_REQUIRED')
        if self.study_stage == 'formal' and (self.source_use != 'prospectively_reserved_replay_holdout' or self.source_reservation is None):
            raise ReplayGateError('RETROSPECTIVE_SOURCE_CANNOT_BECOME_CONFIRMATORY')
        return self


def _read(path):
    return strict_json_loads(Path(path).read_text())


def inspect_sources(collection):
    index = import_trace_collection(collection)
    if index['collection_kind'] != 'natural_pilot' or index['natural_trace_status'] != 'COLLECTED':
        raise ReplayGateError('NATURAL_LIVE_COLLECTION_REQUIRED')
    if set(index['included_episode_ids']) != {e['episode_id'] for e in index['episodes']} or index['excluded_episode_reasons']:
        raise ReplayGateError('ALL_SOURCE_EPISODES_REQUIRED')
    rows = []
    for entry in index['episodes']:
        path = Path(entry['episode_path'])
        task = TaskSpec.model_validate(_read(path / 'task.json'))
        result = _read(path / 'episode_result.json')
        if entry['source'] != 'live' or result['live_status'] != 'RUN':
            raise ReplayGateError('MOCK_OR_RULE_CANNOT_BECOME_NATURAL_SOURCE')
        if task.origin != 'analytic_fixture' or task.evidence_mode != 'exact_analytic':
            raise ReplayGateError('REPLAY_EXACT_FINITE_SOURCE_REQUIRED')
        # A source archive remains under its original implementation identity.
        # The actual semantic observer and common gate must still be identical.
        for name in MECHANISM_FILES:
            if file_sha256(path / 'source_bundle' / name) != file_sha256(ROOT / name):
                raise ReplayGateError('REPLAY_MECHANISM_CHANGED_FROM_SOURCE:' + name)
        rows.append({'episode_id': entry['episode_id'], 'episode_path': str(path),
            'source_method': entry['method'], 'underlying_cluster_id': entry['underlying_cluster_id'],
            'task_id': result['task_id'], 'source_terminal': result['terminal_state'],
            'source_started_at': result['started_at'], 'source_ended_at': result['ended_at'],
            'candidate_count': len(task.candidate_pool), 'task': ref(path / 'task.json'),
            'marker': ref(path / 'COMMITTED.json'), 'result': ref(path / 'episode_result.json'),
            'claim_family_hash': research_hash([a.model_dump(mode='json') for a in required_atoms(task)]),
            'source_hash': result['source_hash']})
    clusters = defaultdict(list)
    for row in rows:
        clusters[row['underlying_cluster_id']].append(row['source_method'])
    if any(sorted(v) != sorted(SOURCE_METHODS) for v in clusters.values()):
        raise ReplayGateError('INCOMPLETE_OR_DUPLICATED_SOURCE_METHOD_CLUSTER')
    return rows, clusters


def freeze_replay(plan_path, output):
    plan = ReplayPlan.model_validate(_read(plan_path))
    output = Path(output).resolve()
    if output.exists():
        raise ReplayIntegrityError('REPLAY_FREEZE_ALREADY_EXISTS')
    rows, clusters = inspect_sources(plan.source_collection)
    if len(clusters) < plan.minimum_independent_clusters:
        raise ReplayGateError('INSUFFICIENT_INDEPENDENT_SOURCE_CLUSTERS')
    if any(r['candidate_count'] > plan.maximum_candidates_per_source for r in rows):
        raise ReplayGateError('REPLAY_CANDIDATE_COVERAGE_CAP_TOO_SMALL')
    needed = sum(r['candidate_count'] for r in rows)
    if plan.audit_candidate_budget < needed:
        raise ReplayGateError('REPLAY_INDEPENDENT_AUDIT_BUDGET_INSUFFICIENT')
    source_refs = [ref(Path(plan.source_collection) / name) for name in
                   ('plan.json', 'registration.json', 'trace-index.json', 'FROZEN.json')]
    source_refs += [r[k] for r in rows for k in ('task', 'marker', 'result')]
    if plan.study_stage == 'formal':
        reservation = SourceReservation.model_validate(_read(checked(plan.source_reservation)))
        collection_registration = _read(Path(plan.source_collection) / 'registration.json')
        # Merely adding a backdated standalone reservation after looking at a
        # trace cannot qualify it.  The original source collection registration
        # must have bound this exact reservation before its first source call.
        if collection_registration.get('natural_replay_reservation') != plan.source_reservation.model_dump(mode='json'):
            raise ReplayGateError('SOURCE_COLLECTION_DID_NOT_PREREGISTER_REPLAY_HOLDOUT')
        if (set(reservation.planned_episode_ids) != {r['episode_id'] for r in rows}
                or len(reservation.planned_episode_ids) != len(rows)
                or reservation.registered_at >= min(r['source_started_at'] for r in rows)
                or reservation.minimum_independent_clusters != plan.minimum_independent_clusters
                or reservation.source_method_strata != plan.source_method_strata
                or collection_registration['registered_at'] < reservation.registered_at
                or reservation.replay_protocol_sha256 != file_sha256(__file__)):
            raise ReplayGateError('PROSPECTIVE_REPLAY_RESERVATION_MISMATCH')
        source_refs.append(plan.source_reservation.model_dump(mode='json'))
    runtime = [ref(ROOT / name) for name in sorted(set(local_import_closure([
        'core/research_analysis/natural_replay.py', 'scripts/research/run_natural_replay.py'])) | {'requirements-cscwd.lock'})]
    output.mkdir(parents=True, exist_ok=False)
    core = {'schema_version': 1, 'plan': plan.model_dump(mode='json'), 'source_refs': source_refs,
            'runtime_refs': runtime, 'source_rows': rows, 'independent_clusters': len(clusters),
            'planned_trials': len(rows) * len(ARMS), 'audit_objects': needed,
            'source_method_counts': dict(Counter(r['source_method'] for r in rows)),
            'raw_source_outcomes_known': plan.source_use == 'retrospective_development',
            'rq_ids': ['RQ1', 'RQ2'], 'full_vs_single_not_identified': True,
            'fixed_message_feedback_never_returned_to_original_model': True}
    core_ref = write_once(output / 'ReplayPlanCore.json', core)
    receipt = {'schema_version': 1, 'status': 'FROZEN', 'plan_core': core_ref,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'input_plan': ref(plan_path),
        'checks': {'complete_natural_source_collection': True, 'task_level_cluster_accounting': True,
                   'all_four_observers_identical_input': True, 'full_finite_claim_family': True,
                   'independent_audit_budget': True},
        'confirmatory': plan.study_stage == 'formal'}
    receipt_ref = write_once(output / 'freeze_receipt.json', receipt)
    return write_once(output / 'ExecutionEnvelope.json', {'schema_version': 1,
        'plan_core': core_ref, 'freeze_receipt': receipt_ref})


def _archive_services(path, task, result):
    """Rehydrate read-only services; constructors would create new evidence."""
    evidence = object.__new__(ExactEvidenceService)
    evidence.task, evidence.source_hash, evidence.output = task, result['source_hash'], path / 'evidence'
    evidence.input_path = evidence.output / 'finite_problem.json'
    evidence.registration = _read(evidence.output / 'registration.json')
    evidence.plan = {'kind': 'exact-finite-v1', 'task_hash': task.task_hash, 'source_hash': result['source_hash'],
        'atoms': [a.model_dump(mode='json') for a in required_atoms(task)],
        'additional_exact_family': {'static': 'compiler_feasibility_ge_one',
            'pairs': 'all_ordered_distinct_legal_pool_candidates', 'quantities': [r.role_id for r in task.roles],
            'models': [m.model_id for m in task.model_family], 'difference_thresholds': [0.0], 'directions': ['ge', 'le']}}
    if research_hash(evidence.plan) != evidence.registration['plan_hash']:
        raise ReplayIntegrityError('SOURCE_EVIDENCE_PLAN_MISMATCH')
    records = {}
    evidence.issued = {}
    for p in evidence.output.glob('*.json'):
        if p.name in {'finite_problem.json', 'registration.json'}:
            continue
        record = EvidenceRecord.model_validate(_read(p))
        records[record.evidence_id] = record
        evidence.issued[record.evidence_id] = p
    information = object.__new__(InformationService)
    information.task, information.output = task, path / 'information'
    information.cards = {c.card_id: c for c in task.information_cards}
    information.receipts = {p.stem: _read(p) for p in information.output.glob('*.json')}
    information.entitlements = {'designer': (task.method_initial_information_ids, task.method_queryable_information_ids)}
    information.entitlements.update({r.role_id: (r.initial_information_ids, r.queryable_information_ids) for r in task.roles})
    information.visible = {owner: {} for owner in information.entitlements}
    for identity, receipt in sorted(information.receipts.items(), key=lambda v: v[1]['sequence']):
        information.visible[receipt['principal']][receipt['card_id']] = identity
    owner = object.__new__(LLMOwnerService)
    owner.task, owner.output, owner.information, owner.evidence_service = task, path / 'assents', information, evidence
    owner.roles = {r.role_id: r for r in task.roles}
    owner.ledger = SimpleNamespace(episode=result['episode_id'])
    owner.issued = {p.stem: AssentReceipt.model_validate(_read(p)) for p in owner.output.glob('*.json') if not p.stem.endswith('-binding')}
    owner.bindings = {p.stem.removesuffix('-binding'): _read(p) for p in owner.output.glob('*-binding.json')}
    events = {}
    for p in (path / 'protocol').glob('*-raw.json'):
        d = _read(p)
        if d.get('type') == 'TEST_RESULT':
            events[research_hash(d['payload'])] = d['payload']
    return evidence, information, owner, records, events


def replay_source(row):
    path = Path(row['episode_path'])
    verify_episode(path)
    for key in ('task', 'marker', 'result'):
        checked(row[key])
    task = TaskSpec.model_validate(_read(path / 'task.json'))
    result = _read(path / 'episode_result.json')
    evidence, info, owners, records, events = _archive_services(path, task, result)
    state = ProtocolState.model_validate(_read(path / 'protocol/initial-state.json'))
    controllers = {arm: CDRProtocol(separate_fact_assent='_s1_' in arm,
                                    maintain_scope_support=arm.endswith('v1')) for arm in ARMS}
    history = []
    for raw in sorted((path / 'protocol').glob('*-raw.json')):
        prefix = raw.name.split('-')[0]
        context = _read(raw.with_name(prefix + '-context.json'))
        expected = _read(raw.with_name(prefix + '-transition.json'))
        auth = AuthenticatedInput.model_validate(context['auth'])
        services = ProtocolServices(task, result['source_hash'], evidence, owners, info,
            {i: records[i] for i in context['evidence_ids']},
            {i: owners.issued[i] for i in context['assent_ids']},
            {i: events[i] for i in context['service_event_ids']},
            {v['receipt_hash']: v for v in context['budget_state']['transactions'].values()},
            context['budget_state'], auth)
        before = state
        try:
            message = AgentMessage.model_validate(_read(raw))
        except ValueError:
            if expected['reason'] != 'SCHEMA_REJECTED' or expected['before_hash'] != research_hash(state) or expected['after_hash'] != research_hash(state):
                raise ReplayIntegrityError('SOURCE_SCHEMA_REJECTION_MISMATCH')
            history.append({'sequence': prefix, 'source_raw': ref(raw), 'schema_rejected': True})
            continue
        state, actual, _ = advance_protocol(state, message, services)
        if any(actual[k] != expected[k] for k in ('before_hash', 'after_hash', 'applied', 'reason')):
            raise ReplayIntegrityError('SOURCE_COMMON_TRANSITION_MISMATCH:' + prefix)
        local = {arm: c.observe(before, message, services, actual, state) for arm, c in controllers.items()}
        history.append({'sequence': prefix, 'source_raw': ref(raw),
                        'visible_evidence_ids': list(context['evidence_ids']),
                        'common_transition': actual, 'local_observers': local})
    final = _read(path / 'protocol/state.json')
    final.update(terminal=state.terminal, terminal_reason=state.terminal_reason)
    if ProtocolState.model_validate(final) != state:
        raise ReplayIntegrityError('SOURCE_TERMINAL_STATE_MISMATCH')
    return {'episode_id': row['episode_id'], 'source_method': row['source_method'],
        'underlying_cluster_id': row['underlying_cluster_id'], 'source_terminal': row['source_terminal'],
        'original_selected_policy_hash': result['selected_policy_hash'],
        'original_A_raw': result['A_raw'], 'original_A_cert': result['A_cert'],
        'arms': {arm: {'A_method': c.method_agreements(task), 'positions': c.positions,
                      'support_verdicts': c.graph.verdicts,
                      'support_refresh_count': len(c.graph.receipts),
                      'new_queries': 0, 'new_calls': 0} for arm, c in controllers.items()},
        'history': history, 'raw_generation_effect': 'NOT_IDENTIFIED',
        'terminal_selection_rule': 'retain_actual_source_selection_no_counterfactual_reselection'}


def audit_selection(selection_ref, row, output):
    """Scalar full-pool audit is evaluated only after the selection is durable."""
    selection = _read(checked(selection_ref))
    if selection['episode_id'] != row['episode_id']:
        raise ReplayIntegrityError('AUDIT_SOURCE_SELECTION_MISMATCH')
    task = TaskSpec.model_validate(_read(checked(row['task'])))
    by_candidate = defaultdict(list)
    for outcome in task.analytic_outcomes:
        by_candidate[outcome.candidate_id].append(outcome)
    truth = {}
    for candidate in task.candidate_pool:
        policy = compile_candidate(task, candidate).policy_hash
        atoms = []
        for outcome in by_candidate[candidate.candidate_id]:
            for role in task.roles:
                utility = 1.0 - sum(role.weights.model_dump()[k] * v for k, v in outcome.losses.model_dump().items())
                atoms.append({'model': outcome.model_id, 'kind': 'utility', 'quantity': role.role_id,
                              'value': utility, 'threshold': role.rho, 'supported': utility >= role.rho})
            for constraint in task.outcome_constraints:
                value = outcome.violations[constraint.quantity]
                atoms.append({'model': outcome.model_id, 'kind': 'constraint', 'quantity': constraint.constraint_id,
                              'value': value, 'threshold': constraint.upper_bound, 'supported': value <= constraint.upper_bound})
        if len(atoms) != len(required_atoms(task)):
            raise ReplayIntegrityError('AUDIT_INCOMPLETE_FINITE_ATOM_FAMILY')
        truth[policy] = {'candidate_id': candidate.candidate_id, 'joint_acceptable': all(a['supported'] for a in atoms), 'atoms': atoms}
    if selection['original_selected_policy_hash'] is not None and selection['original_selected_policy_hash'] not in truth:
        raise ReplayIntegrityError('ORIGINAL_SELECTION_OUTSIDE_AUDIT_FAMILY')
    audit = {'schema_version': 1, 'selection_record': selection_ref, 'source_episode_id': row['episode_id'],
        'coverage': 'complete_original_finite_candidate_pool', 'truth': truth,
        'audit_status': 'COMPLETED', 'audit_candidate_evaluations': len(truth),
        'original_selected_policy_hash': selection['original_selected_policy_hash'],
        'no_selection_status': 'NO_APPLICABLE_SELECTED_OBJECT' if selection['original_selected_policy_hash'] is None else 'SELECTED_OBJECT_COVERED',
        'no_substitute_policy': True, 'design_feedback': False,
        'llm_calls': 0, 'market_runs': 0, 'new_independent_observations': 0}
    audit['arms'] = {}
    for arm, local in selection['arms'].items():
        unknown = set(local['A_method']) - set(truth)
        if unknown:
            raise ReplayIntegrityError('REPLAY_LOCAL_POLICY_OUTSIDE_FROZEN_FAMILY')
        audit['arms'][arm] = {'method_agreements': len(local['A_method']),
            'false_local_method_agreement_count': sum(not truth[p]['joint_acceptable'] for p in local['A_method']),
            'true_local_method_agreement_count': sum(truth[p]['joint_acceptable'] for p in local['A_method'])}
    write_once(output, audit)
    checked(selection_ref)
    return audit


def summarize_replay(rows, audits):
    """Aggregate paired arms inside original task clusters, not 48 fake N."""
    cluster_values = defaultdict(lambda: defaultdict(list))
    source_totals = {source: {arm: 0 for arm in ARMS} for source in SOURCE_METHODS}
    for row, audit in zip(rows, audits):
        for arm in ARMS:
            value = audit['arms'][arm]['false_local_method_agreement_count']
            cluster_values[row['underlying_cluster_id']][arm].append(value)
            source_totals[row['source_method']][arm] += value
    paired = []
    for cluster, values in sorted(cluster_values.items()):
        means = {arm: sum(v) / len(v) for arm, v in values.items()}
        a, b, c, d = [means[arm] for arm in ARMS]
        paired.append({'cluster': cluster, 'arm_source_averages': means,
            'S': (c + d - a - b) / 2, 'V': (b + d - a - c) / 2, 'SV': d - c - b + a})
    return {'source_episode_count': len(rows), 'independent_task_clusters': len(paired),
        'source_method_totals': source_totals, 'task_cluster_vectors': paired,
        'finite_population_contrasts': {k: sum(x[k] for x in paired) / len(paired) for k in ('S', 'V', 'SV')},
        'confidence_intervals': None, 'p_values': None,
        'inference': 'exact conditional census of the frozen source population',
        'raw_error_reduction_by_observer': 'NOT_IDENTIFIED_FIXED_RAW_STREAM',
        'adaptive_language_policy_effect': 'NOT_IDENTIFIED',
        'dynamic_withdrawal_effect': 'NOT_IDENTIFIED_WITHOUT_NATURAL_WITHDRAWAL_EVENTS',
        'formal_superpopulation_power': 'NOT_PLANNED_FOR_THIS_FINITE_CENSUS',
        'new_llm_calls': 0, 'new_market_runs': 0}


def execute_replay(envelope_path, output):
    envelope = _read(envelope_path)
    core = _read(checked(envelope['plan_core']))
    receipt = _read(checked(envelope['freeze_receipt']))
    if receipt['plan_core'] != envelope['plan_core'] or receipt['status'] != 'FROZEN':
        raise ReplayIntegrityError('REPLAY_FREEZE_CHAIN_MISMATCH')
    for item in core['runtime_refs'] + core['source_refs']:
        checked(item)
    plan = ReplayPlan.model_validate(core['plan'])
    output = Path(output).resolve()
    if output.exists():
        raise ReplayIntegrityError('REPLAY_EXECUTION_ALREADY_EXISTS')
    output.mkdir(parents=True, exist_ok=False)
    rows = core['source_rows']; results = []; audits = []; failed = []
    write_once(output / 'execution_registration.json', {'envelope': ref(envelope_path),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_episode_ids': [r['episode_id'] for r in rows], 'planned_trials': core['planned_trials']})
    for row in rows:
        destination = output / 'episodes' / row['episode_id']
        try:
            replayed = replay_source(row)
            selection = {k: v for k, v in replayed.items() if k != 'history'}
            selection_ref = write_once(destination / 'SelectionRecord.json', selection)
            history_ref = write_once(destination / 'replay_events.json', replayed['history'])
            audit = audit_selection(selection_ref, row, destination / 'audit.json')
            audits.append(audit)
            results.extend({'episode_id': row['episode_id'], 'arm': arm, 'status': 'RECORDED',
                'selection_record': selection_ref, 'events': history_ref, 'audit': ref(destination / 'audit.json')}
                for arm in ARMS)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            failure = {'source_episode_id': row['episode_id'], 'status': 'FAILED',
                       'error_type': type(exc).__name__, 'reason': str(exc)}
            failure_ref = write_once(destination / 'failure.json', failure)
            failed.append(failure)
            results.extend({'episode_id': row['episode_id'], 'arm': arm, 'status': 'FAILED', 'failure': failure_ref} for arm in ARMS)
        durable_json(output / 'benchmark_index.json', {'study_stage': plan.study_stage,
            'planned_trials': core['planned_trials'], 'indexed_trials': len(results), 'trials': results})
    # A source mutation after execution invalidates the whole dependent archive.
    for item in core['runtime_refs'] + core['source_refs']:
        checked(item)
    index = {'schema_version': 1, 'study_stage': plan.study_stage, 'source_use': plan.source_use,
        'status': 'COMPLETED' if not failed else 'COMPLETED_WITH_REPLAY_FAILURES',
        'planned_trials': core['planned_trials'], 'indexed_trials': len(results), 'trials': results,
        'independent_clusters': core['independent_clusters'], 'failures': failed,
        'summary': summarize_replay(rows, audits) if not failed else None,
        'llm_calls': 0, 'market_runs': 0, 'formal_agent_effect_completed': False}
    durable_json(output / 'benchmark_index.json', index)
    write_once(output / 'audit_index.json', {'planned_source_episodes': len(rows),
        'completed_source_audits': len(audits), 'failed_sources': failed,
        'candidate_budget': plan.audit_candidate_budget,
        'candidate_evaluations': sum(a['audit_candidate_evaluations'] for a in audits),
        'audits': [r['audit'] for r in results if r['arm'] == ARMS[0] and r['status'] == 'RECORDED']})
    write_once(output / 'COMPLETED.json', {'index': ref(output / 'benchmark_index.json'),
        'audit_index': ref(output / 'audit_index.json'), 'complete_planned_index': len(results) == core['planned_trials'],
        'source_failures_preserved': len(failed), 'confirmatory': receipt['confirmatory']})
    return index
