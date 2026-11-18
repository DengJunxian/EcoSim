"""Original GLM formal AnalysisPlan, gated on a complete audited source census.

No provider calls, market runs, new contrasts, selection, or sample-size planning.
The original F72 equal-stratum paired_interval kernel is reused unchanged.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import subprocess

import numpy as np

from core.codesign.schema import EpisodeResult, TaskSpec
from core.codesign.verifier import compile_candidate
from core.experiments.archive import durable_json
from core.experiments.identity import file_sha256, research_hash
from core.experiments.research_cli import ROOT, local_import_closure
from core.experiments.strict_types import load_research_model
from .analysis import JOINT_METHODS, paired_interval
from .provider_freeze import verify_provider_envelope
from .study_freeze import check_ref


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha256(path))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def checked_reference(value):
    require(ref(value['path']) == value, 'ANALYSIS_INPUT_REFERENCE_CHANGED')
    return Path(value['path'])


def complete_census_gate(core, envelope, index_path, audit_path, post_path):
    """Validate the complete original schedule before opening any result outcomes."""
    index, audit, post = read(index_path), read(audit_path), read(post_path)
    marker = read(Path(index_path).parent / 'COMPLETED.json')
    fields = ('episode_id', 'task_id', 'method')
    keys = lambda rows: [tuple(row[k] for k in fields) for row in rows]
    require(marker['index_sha256'] == file_sha256(index_path), 'FORMAL_TERMINAL_INDEX_HASH_CHANGED')
    require(keys(index['episodes']) == keys(core.schedule)
            and len(set(keys(index['episodes']))) == len(core.schedule), 'ORIGINAL_ANALYSIS_SCHEDULE_CHANGED')
    require(index['plan_id'] == core.plan_id and index['source_hash'] == core.source_hash,
            'ORIGINAL_ANALYSIS_IDENTITY_CHANGED')
    require(index['recorded_trials'] == len(core.schedule) and index['not_run_trials'] == 0
            and all(row['status'] == 'RECORDED' for row in index['episodes'])
            and index['all_registered_trials_have_terminal_records'] is True
            and index['all_design_and_audit_integrity_checks_passed'] is True
            and index['execution_failed_trials'] == 0
            and marker['all_original_trials_recorded'] is True,
            'COMPLETE_ORIGINAL_AUDITED_CENSUS_REQUIRED')
    require(index['original_f70_strict_gate_passed'] is False
            and marker['formal_execution_complete'] is False,
            'HISTORICAL_STRICT_F70_GATE_MUST_REMAIN_VISIBLE')
    require(audit['status'] == 'VERIFIED_COMPLETE_CENSUS_USAGE_UNRESOLVED'
            and audit['recorded_trials'] == len(core.schedule)
            and audit['maximum_additional_calls'] == 0
            and audit['parent_index'] == ref(index_path)
            and audit['original_envelope'] == ref(envelope)
            and audit['recovery_protocol_version'] == 2
            and audit['original_f70_strict_gate_passed'] is False
            and audit['unknown_usage_reconciled'] is False,
            'MATCHING_COMPLETE_TRANSPORT_V2_AUDIT_REQUIRED')
    require(post['status'] == 'CENSUS_AUDITED_AND_NATURAL_REPLAY_COMPLETED_EFFECTS_NOT_ANALYZED'
            and post['terminal_parent_index'] == ref(index_path)
            and [c['stage'] for c in post['commands']] == [
                'COMPLETE_CENSUS_AUDIT', 'ALL_SOURCE_EXPORT', 'NATURAL_REPLAY_FREEZE', 'NATURAL_REPLAY_RUN']
            and all(c.get('exit_code') == 0 for c in post['commands'])
            and post['new_generation_calls'] == post['new_market_runs'] == 0,
            'ORIGINAL_NATURAL_REPLAY_CHAIN_MUST_FINISH_BEFORE_EFFECT_ANALYSIS')
    for value in (audit['raw_call_index'], audit['spend_snapshot'], audit['human_approval']):
        checked_reference(value)
    for path, expected in audit['source_hashes'].items():
        require(file_sha256(path) == expected, 'COMPLETE_AUDIT_IMPLEMENTATION_CHANGED')
    return index, audit, post


def original_contrasts(matrix, strata, contrasts, family_alpha):
    """Equal weight for each original stratum, task pairing inside every contrast."""
    matrix, strata = np.asarray(matrix, dtype=float), np.asarray(strata, dtype=int)
    require(matrix.shape == (len(strata), len(JOINT_METHODS))
            and np.isin(matrix, [0, 1]).all(), 'BINARY_ORIGINAL_METHOD_TASK_MATRIX_REQUIRED')
    require(set(strata) == {0, 1, 2}, 'THREE_ORIGINAL_STRATA_REQUIRED')
    alpha = family_alpha / len(contrasts)
    rows = []
    for contrast in contrasts:
        weights = [contrast['weights'].get(method, 0.) for method in JOINT_METHODS]
        base = dict(contrast=contrast['id'], rq=contrast['rq'], weights=contrast['weights'],
                    registered_delta=contrast['delta'], per_contrast_alpha=alpha,
                    interval_confidence=1-alpha, stratum_counts=[int(sum(strata == s)) for s in range(3)])
        if min(base['stratum_counts']) < 2:
            rows.append(dict(**base, estimate=None, approximate_interval=None, standard_error=None,
                             independent_tasks=len(strata), interval_decision='NOT_ESTIMABLE_INSUFFICIENT_STRATUM'))
            continue
        summary = paired_interval(matrix, strata, weights, alpha)
        interval = summary['approximate_interval']
        decision = ('NOT_ESTIMABLE_ZERO_VARIANCE' if interval is None else
                    'POSITIVE_INTERVAL_EXCLUDES_ZERO' if interval[0] > 0 else
                    'NEGATIVE_INTERVAL_EXCLUDES_ZERO' if interval[1] < 0 else 'INTERVAL_INCLUDES_ZERO')
        rows.append(dict(**base, **summary, interval_decision=decision))
    return rows


def sensitivity_analysis(matrix, strata, unknown_mask, affected_tasks, contrasts, family_alpha):
    """Predeclared service extremes plus whole affected-task exclusion, never new N selection."""
    matrix = np.asarray(matrix, dtype=float)
    strata = np.asarray(strata, dtype=int)
    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    affected_tasks = np.asarray(affected_tasks, dtype=bool)
    require(matrix.shape == unknown_mask.shape and len(affected_tasks) == len(matrix),
            'SENSITIVITY_MASK_SHAPE_MISMATCH')
    require(np.all(matrix[unknown_mask] == 0), 'PRIMARY_UNKNOWN_SERVICE_FAILURES_MUST_BE_ZERO')
    require(np.all(affected_tasks[np.any(unknown_mask, axis=1)]), 'ENTIRE_AFFECTED_TASK_MUST_BE_IDENTIFIED')
    extremes = {}
    for value in (0, 1):
        scenario = matrix.copy()
        scenario[unknown_mask] = value
        extremes[str(value)] = original_contrasts(scenario, strata, contrasts, family_alpha)
    endpoints = []
    for contrast in contrasts:
        w = np.array([contrast['weights'].get(method, 0.) for method in JOINT_METHODS])
        fixed = np.where(unknown_mask, 0, matrix) @ w
        lower = fixed + unknown_mask @ np.minimum(w, 0)
        upper = fixed + unknown_mask @ np.maximum(w, 0)
        mean = lambda values: float(np.mean([np.mean(values[strata == s]) for s in range(3)]))
        endpoints.append(dict(contrast=contrast['id'], logical_point_estimate_bounds=[mean(lower), mean(upper)],
                              meaning='Coefficient-sign 0/1 endpoints under the same registered contrast; no additional hypothesis test.'))
    retained = ~affected_tasks
    exclusion = (original_contrasts(matrix[retained], strata[retained], contrasts, family_alpha)
                 if set(strata[retained]) == {0, 1, 2} else [])
    return dict(primary_failure_rule='Original ITT: METHOD_FAILED counts as uncertified, including unknown-service failures.',
                unknown_method_cells=int(unknown_mask.sum()), excluded_whole_task_clusters=int(affected_tasks.sum()),
                unknown_service_all_zero_or_all_one=extremes, unknown_service_logical_point_bounds=endpoints,
                whole_affected_task_exclusion_nonprimary=dict(retained_tasks=int(retained.sum()), contrasts=exclusion,
                    status='CALCULATED' if exclusion else 'NOT_ESTIMABLE_MISSING_STRATUM'),
                sensitivity_results_are_not_replacement_primary_tests=True)


def eligible_policies(task):
    policies = set()
    for candidate in task.candidate_pool:
        rows = [r for r in task.analytic_outcomes if r.candidate_id == candidate.candidate_id]
        eligible = all(all(1-sum(weight*r.losses.model_dump()[name] for name, weight in role.weights.model_dump().items()) >= role.rho
                           for role in task.roles)
                       and all(r.violations[c.quantity] <= c.upper_bound for c in task.outcome_constraints) for r in rows)
        if eligible:
            policies.add(compile_candidate(task, candidate).policy_hash)
    return policies


def collect_complete_inputs(core, index, audit):
    tasks = [load_research_model(check_ref(t), TaskSpec) for t in core.tasks]
    require(len(tasks) == 384 and all(t.split == 'test' for t in tasks), 'ORIGINAL_384_HELDOUT_TASKS_REQUIRED')
    ids = [t.task_id for t in tasks]
    strata = np.array([int(t.prompt_resources['stratum_index']) for t in tasks])
    require(np.bincount(strata, minlength=3).tolist() == [128, 128, 128], 'ORIGINAL_EQUAL_STRATUM_ALLOCATION_CHANGED')
    task_map = {t.task_id: t for t in tasks}
    allowed = {tid: eligible_policies(t) for tid, t in task_map.items()}
    unknown_ids = {r['identity'].split(':', 1)[0] for r in audit['unknown_requests']}
    planned_ids = {r['episode_id'] for r in index['episodes']}
    require(unknown_ids <= planned_ids, 'UNKNOWN_SERVICE_REQUEST_OUTSIDE_ORIGINAL_SCHEDULE')
    results, episode_rows, evidence = {}, [], []
    for row in index['episodes']:
        path = Path(row['path']) / 'episode_result.json'
        require(file_sha256(path) == row['result_sha256'], 'ANALYSIS_EPISODE_RESULT_CHANGED')
        result = load_research_model(path, EpisodeResult)
        require(result.episode_id == row['episode_id'] and result.task_id == row['task_id']
                and result.method == row['method'] and result.source_hash == core.source_hash
                and result.terminal_state == row['terminal_state'] and result.split == 'test',
                'ORIGINAL_ANALYSIS_EPISODE_IDENTITY_CHANGED')
        require(set(result.A_cert) <= allowed[result.task_id], 'INDEPENDENT_CERTIFICATE_TRUTH_MISMATCH')
        failed = result.terminal_state == 'METHOD_FAILED'
        unresolved = row['episode_id'] in unknown_ids
        require(not unresolved or failed, 'UNKNOWN_SERVICE_EPISODE_FAILURE_MUST_BE_RETAINED')
        value = int(not failed and bool(result.A_cert))
        results[(result.task_id, result.method)] = value
        used = result.budget_snapshot['used']
        episode_rows.append(dict(episode_id=result.episode_id, task_id=result.task_id, method=result.method,
            stratum=int(task_map[result.task_id].prompt_resources['stratum_index']), terminal_state=result.terminal_state,
            primary_A_cert_nonempty_ITT=value, observed_A_cert_nonempty=int(bool(result.A_cert)), method_failed=int(failed),
            unknown_service=int(unresolved), false_A_method_claims=sum(h not in allowed[result.task_id] for h in result.A_method),
            false_A_cert_claims=0, protocol_messages=used['messages'], live_calls=result.common_contract['live_calls'],
            local_reserved_input_tokens=used['input_tokens'], local_reserved_output_tokens=used['output_tokens'],
            wall_seconds=result.ended_at-result.started_at))
        evidence.append(ref(path))
    require(len(results) == 3456 and len(episode_rows) == 3456, 'ALL_ORIGINAL_METHOD_TASK_CELLS_REQUIRED')
    matrix = np.array([[results[(tid, m)] for m in JOINT_METHODS] for tid in ids], dtype=float)
    unknown_cells = {(r['task_id'], r['method']) for r in episode_rows if r['unknown_service']}
    unknown_mask = np.array([[(tid, m) in unknown_cells for m in JOINT_METHODS] for tid in ids])
    affected_ids = {r['task_id'] for r in episode_rows if r['unknown_service']}
    affected = np.array([tid in affected_ids for tid in ids])
    return ids, strata, matrix, unknown_mask, affected, episode_rows, evidence


def usage_components(audit):
    states = read(checked_reference(audit['spend_snapshot']))['states']
    reservations, settlements = {}, {}
    for state in states:
        require(not (set(reservations) & set(state['reservations'])), 'DUPLICATE_FORMAL_USAGE_IDENTITY')
        reservations.update(state['reservations'])
        settlements.update(state['settlements'])
    require(set(reservations) == set(settlements) and len(reservations) == audit['raw_requests_verified'],
            'COMPLETE_FORMAL_USAGE_CENSUS_REQUIRED')
    unknown = {key for key, value in settlements.items() if value['usage_status'] == 'unknown'}
    require(unknown == {r['identity'] for r in audit['unknown_requests']}, 'AUDITED_UNKNOWN_USAGE_SET_CHANGED')
    known = [v for v in settlements.values() if v['usage_status'] == 'known']
    require(len(known)+len(unknown) == len(settlements), 'INVALID_FORMAL_USAGE_STATUS')
    return dict(formal_requests=len(reservations), known_usage_requests=len(known), unknown_usage_requests=len(unknown),
                known_input_tokens=sum(v['input_tokens'] for v in known), known_output_tokens=sum(v['output_tokens'] for v in known),
                unknown_input_token_full_cap=sum(reservations[k]['input_cap'] for k in unknown),
                unknown_output_token_full_cap=sum(reservations[k]['output_cap'] for k in unknown),
                total_actual_input_tokens=None if unknown else sum(v['input_tokens'] for v in known),
                total_actual_output_tokens=None if unknown else sum(v['output_tokens'] for v in known),
                actual_supplier_bill=None, currency='CNY', frozen_all_call_price_cap_CNY=0,
                original_f70_strict_gate_passed=False, unknown_usage_reconciled=False)


def register(envelope, index, audit, post, output):
    output = Path(output).resolve()
    require(not output.exists(), 'IMMUTABLE_FORMAL_ANALYSIS_REGISTRATION_EXISTS')
    core, analysis, _ = verify_provider_envelope(envelope)
    require(len(core.tasks) == 384 and core.provider == 'zhipu' and analysis.primary_outcome == 'A_cert'
            and len(analysis.planning_design.contrasts) == 4, 'ORIGINAL_GLM384_ANALYSIS_PLAN_REQUIRED')
    source_files = local_import_closure(['core/research_analysis/provider_effects.py', 'scripts/research/analyze_provider_formal.py'])
    bindings = [ref(ROOT / f) for f in sorted(set(source_files + ['requirements-cscwd.lock']))]
    registration = dict(schema_version=1, kind='original_glm_formal_analysis_implementation_v1',
        registered_at_utc=datetime.now(timezone.utc).isoformat(), original_envelope=ref(envelope),
        original_analysis_plan=core.analysis_plan.model_dump(mode='json'), original_design_source_hash=core.source_hash,
        complete_index_path=str(Path(index).resolve()), complete_audit_path=str(Path(audit).resolve()),
        completed_postprocessing_status_path=str(Path(post).resolve()), bindings=bindings,
        analysis_implementation_hash=research_hash(bindings), original_scientific_plan_changed=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        maximum_generation_calls=0, maximum_market_runs=0,
        primary_kernel='core.research_analysis.analysis.paired_interval',
        primary_failure_rule='METHOD_FAILED=0', unknown_service_sensitivity='original_0_1_extremes_and_whole_task_exclusion',
        original_f70_strict_gate_passed=False, financial_online_authorized=False)
    output.mkdir(parents=True)
    durable_json(output / 'registration.json', registration)
    os.chmod(output / 'registration.json', 0o444)
    return registration


def execute(registration_path, readiness_only=False):
    registration_path = Path(registration_path).resolve()
    output = registration_path.parent
    registration_ref = ref(registration_path)
    registration = read(registration_path)
    for binding in registration['bindings']:
        checked_reference(binding)
    envelope = checked_reference(registration['original_envelope'])
    core, analysis, _ = verify_provider_envelope(envelope)
    checked_reference(registration['original_analysis_plan'])
    require(core.source_hash == registration['original_design_source_hash'] and len(core.tasks) == 384,
            'ORIGINAL_DESIGN_OR_TASK_COUNT_CHANGED')
    paths = [registration[k] for k in ['complete_index_path','complete_audit_path','completed_postprocessing_status_path']]
    index, audit, _ = complete_census_gate(core, envelope, *paths)
    if readiness_only:
        return dict(status='READY_COMPLETE_AUDITED_CENSUS_AND_REPLAY', scientific_effects_status='NOT_ANALYZED')
    with (output / 'STARTED.json').open('x') as handle:
        json.dump(dict(process_id=os.getpid(), registration=registration_ref,
                       started_at_utc=datetime.now(timezone.utc).isoformat()), handle)
        handle.flush()
        os.fsync(handle.fileno())
    ids, strata, matrix, unknown_mask, affected, episode_rows, evidence = collect_complete_inputs(core, index, audit)
    design = analysis.planning_design.model_dump(mode='json')
    contrasts = original_contrasts(matrix, strata, design['contrasts'], design['family_alpha'])
    sensitivities = sensitivity_analysis(matrix, strata, unknown_mask, affected, design['contrasts'], design['family_alpha'])
    methods = []
    for method in core.methods:
        rows = [r for r in episode_rows if r['method'] == method]
        methods.append(dict(method=method, independent_tasks=len(rows), certified_ITT=sum(r['primary_A_cert_nonempty_ITT'] for r in rows),
            terminal_states=dict(Counter(r['terminal_state'] for r in rows)), method_failed=sum(r['method_failed'] for r in rows),
            unknown_service=sum(r['unknown_service'] for r in rows), false_A_method_claims=sum(r['false_A_method_claims'] for r in rows),
            false_A_cert_claims=0, protocol_messages=sum(r['protocol_messages'] for r in rows), live_calls=sum(r['live_calls'] for r in rows),
            mean_wall_seconds=float(np.mean([r['wall_seconds'] for r in rows]))))
    usage = usage_components(audit)
    require(usage['formal_requests'] == sum(r['live_calls'] for r in episode_rows), 'FORMAL_RESULTS_USAGE_COUNT_MISMATCH')
    result = dict(schema_version=1, status='ORIGINAL_REGISTERED_ANALYSIS_COMPLETED_USAGE_UNRESOLVED',
        analysis_id=analysis.analysis_id, generated_at_utc=datetime.now(timezone.utc).isoformat(), registration=registration_ref,
        original_envelope=ref(envelope), complete_index=ref(paths[0]), complete_audit=ref(paths[1]), completed_postprocessing=ref(paths[2]),
        primary_outcome='nonempty A_cert with original ITT METHOD_FAILED=0', independent_tasks=len(ids), trials=len(episode_rows),
        stratum_counts=np.bincount(strata).tolist(), family_alpha=design['family_alpha'], primary_contrasts=contrasts,
        sensitivities=sensitivities, methods=methods, usage=usage, all_registered_trials_included=True,
        equivalence_test_performed=False, posthoc_power_or_sample_size_selection=False, generation_calls=0, market_runs=0,
        original_f70_strict_gate_passed=False, financial_online_authorized=False,
        scope_declaration=core.scope_declaration,
        inference_scope='Original GLM heldout exact-analytic four contrasts. Natural fixed-stream, financial, secondary-provider and OOD scopes remain separate.')
    durable_json(output/'task_vectors.json', dict(task_ids=ids, strata=strata.tolist(), methods=JOINT_METHODS,
        certification_matrix=matrix.tolist(), unknown_service_matrix=unknown_mask.tolist(), affected_whole_task=affected.tolist()))
    with (output/'all_episode_outcomes.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    durable_json(output/'analysis_result.json', result)
    durable_json(output/'episode_evidence.json', dict(files=evidence))
    for binding in registration['bindings']:
        checked_reference(binding)
    require(ref(registration_path) == registration_ref, 'FORMAL_ANALYSIS_REGISTRATION_CHANGED')
    complete_census_gate(core, envelope, *paths)
    durable_json(output/'COMPLETED.json', dict(status=result['status'], result=ref(output/'analysis_result.json'),
        all_episodes=ref(output/'all_episode_outcomes.csv'), task_vectors=ref(output/'task_vectors.json'),
        episode_evidence=ref(output/'episode_evidence.json'), original_f70_strict_gate_passed=False,
        generation_calls=0, market_runs=0, all_original_trials_analyzed=True))
    return result
