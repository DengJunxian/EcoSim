"""User-directed Phase 8 from the complete formal census, with experiments paused.

The original statistical functions and all frozen execution files remain unchanged.
Natural replay is a separate, paused study and is not an analysis input here.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import socket
import subprocess

import numpy as np

from . import provider_effects as e


def assert_paused(pause):
    e.require(pause['status'] == 'PAUSED_BY_EXPLICIT_USER_REQUEST'
              and pause['automation']['status'] == 'PAUSED', 'EXPLICIT_USER_PAUSE_REQUIRED')
    for row in pause['after']:
        process = subprocess.run(['ps', '-p', str(row['pid']), '-o', 'stat=,command='],
                                 capture_output=True, text=True)
        if process.returncode == 0:
            state = process.stdout.strip().split()[0]
            e.require('T' in state, 'EXPERIMENT_PROCESS_NO_LONGER_PAUSED')
    for reference in pause['current_evidence_snapshots']:
        e.require(e.file_sha256(reference['snapshot']) == reference['sha256'], 'PAUSE_SNAPSHOT_CHANGED')
        if reference['path'].endswith('provider-formal-v1-natural-replay/benchmark_index.json'):
            e.require(e.file_sha256(reference['path']) == reference['sha256'], 'PAUSED_REPLAY_INDEX_CHANGED')


def complete_formal_gate(core, envelope, index_path, audit_path):
    """Only the separate replay ordering dependency is omitted in this new scope."""
    index, audit = e.read(index_path), e.read(audit_path)
    marker = e.read(Path(index_path).parent / 'COMPLETED.json')
    keys = lambda rows: [tuple(r[k] for k in ('episode_id', 'task_id', 'method')) for r in rows]
    e.require(marker['index_sha256'] == e.file_sha256(index_path), 'FORMAL_INDEX_CHANGED')
    e.require(keys(index['episodes']) == keys(core.schedule)
              and len(set(keys(index['episodes']))) == len(core.schedule), 'ORIGINAL_SCHEDULE_REQUIRED')
    e.require(index['plan_id'] == core.plan_id and index['source_hash'] == core.source_hash,
              'ORIGINAL_IDENTITY_REQUIRED')
    e.require(index['recorded_trials'] == len(core.schedule) and index['not_run_trials'] == 0
              and all(r['status'] == 'RECORDED' for r in index['episodes'])
              and index['all_registered_trials_have_terminal_records'] is True
              and index['all_design_and_audit_integrity_checks_passed'] is True
              and index['execution_failed_trials'] == 0 and marker['all_original_trials_recorded'] is True,
              'COMPLETE_FORMAL_CENSUS_REQUIRED')
    e.require(index['original_f70_strict_gate_passed'] is False
              and marker['formal_execution_complete'] is False, 'STRICT_F70_MUST_REMAIN_UNRESOLVED')
    e.require(audit['status'] == 'VERIFIED_COMPLETE_CENSUS_USAGE_UNRESOLVED'
              and audit['recorded_trials'] == len(core.schedule) and audit['maximum_additional_calls'] == 0
              and audit['parent_index'] == e.ref(index_path) and audit['original_envelope'] == e.ref(envelope)
              and audit['recovery_protocol_version'] == 2 and audit['original_f70_strict_gate_passed'] is False
              and audit['unknown_usage_reconciled'] is False, 'MATCHING_COMPLETE_AUDIT_REQUIRED')
    for value in (audit['raw_call_index'], audit['spend_snapshot'], audit['human_approval']):
        e.checked_reference(value)
    for path, sha in {**audit['source_hashes'], **audit.get('parent_anchors', {})}.items():
        e.require(e.file_sha256(path) == sha, 'AUDITED_SOURCE_OR_COMMIT_CHANGED')
    return index, audit


def register(envelope, index, audit, pause, out):
    out = Path(out).resolve()
    e.require(not out.exists(), 'IMMUTABLE_REGISTRATION_EXISTS')
    p = e.read(pause)
    assert_paused(p)
    core, analysis, _ = e.verify_provider_envelope(envelope)
    e.require(len(core.tasks) == 384 and len(core.schedule) == 3456 and core.provider == 'zhipu'
              and analysis.primary_outcome == 'A_cert' and len(analysis.planning_design.contrasts) == 4,
              'ORIGINAL_GLM384_FOUR_CONTRASTS_REQUIRED')
    complete_formal_gate(core, envelope, index, audit)
    files = e.local_import_closure(['core/research_analysis/phase8_snapshot.py', 'core/research_analysis/__main__.py'])
    files += ['requirements-cscwd.lock', 'docs/research/cscwd/ADR_PHASE8_PAUSED_EXPERIMENTS_V1.md']
    bindings = [e.ref(e.ROOT / f) for f in sorted(set(files))]
    value = dict(schema_version=1, kind='phase8_complete_formal_census_under_user_pause_v1',
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        original_envelope=e.ref(envelope), original_analysis_plan=core.analysis_plan.model_dump(mode='json'),
        complete_index=e.ref(index), complete_audit=e.ref(audit), user_pause=e.ref(pause),
        bindings=bindings, implementation_hash=e.research_hash(bindings),
        original_design_source_hash=core.source_hash,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=e.ROOT,text=True).strip(),
        original_scientific_plan_changed=False, dependency_change='Separate paused replay is not consumed.',
        maximum_generation_calls=0, maximum_market_runs=0, maximum_replay_cells=0)
    out.mkdir(parents=True)
    e.durable_json(out/'registration.json', value)
    os.chmod(out/'registration.json', 0o444)
    return value


def execute(registration, out):
    registration, out = Path(registration).resolve(), Path(out).resolve()
    e.require(not out.exists(), 'ANALYSIS_OUTPUT_EXISTS')
    reg_ref = e.ref(registration)
    reg = e.read(registration)
    for ref in reg['bindings'] + [reg[k] for k in ('original_envelope','original_analysis_plan',
                                                  'complete_index','complete_audit','user_pause')]:
        e.checked_reference(ref)
    assert_paused(e.read(reg['user_pause']['path']))
    envelope = reg['original_envelope']['path']
    core, analysis, _ = e.verify_provider_envelope(envelope)
    index, audit = complete_formal_gate(core, envelope, reg['complete_index']['path'], reg['complete_audit']['path'])
    out.mkdir(parents=True)
    e.durable_json(out/'STARTED.json', dict(registration=reg_ref, process_id=os.getpid(),
                   started_at_utc=datetime.now(timezone.utc).isoformat()))
    ids, strata, matrix, unknown, affected, rows, evidence = e.collect_complete_inputs(core, index, audit)
    design = analysis.planning_design.model_dump(mode='json')
    contrasts = e.original_contrasts(matrix, strata, design['contrasts'], design['family_alpha'])
    sensitivity = e.sensitivity_analysis(matrix, strata, unknown, affected, design['contrasts'], design['family_alpha'])
    methods = []
    for method in core.methods:
        group = [r for r in rows if r['method'] == method]
        methods.append(dict(method=method, independent_tasks=len(group),
            certified_ITT=sum(r['primary_A_cert_nonempty_ITT'] for r in group),
            terminal_states=dict(Counter(r['terminal_state'] for r in group)),
            **{k:sum(r[k] for r in group) for k in ['method_failed','unknown_service','false_A_method_claims',
                'false_A_cert_claims','protocol_messages','live_calls']},
            mean_wall_seconds=float(np.mean([r['wall_seconds'] for r in group]))))
    usage = e.usage_components(audit)
    e.require(usage['formal_requests'] == sum(r['live_calls'] for r in rows), 'USAGE_COUNT_MISMATCH')
    e.durable_json(out/'task_vectors.json', dict(task_ids=ids, strata=strata.tolist(), methods=e.JOINT_METHODS,
        certification_matrix=matrix.tolist(), unknown_service_matrix=unknown.tolist(), affected_whole_task=affected.tolist()))
    with (out/'all_episode_outcomes.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    e.durable_json(out/'episode_evidence.json', dict(files=evidence))
    result = dict(schema_version=1, status='COMPLETE_ORIGINAL_FORMAL_ANALYSIS_IN_PAUSED_SCOPE',
        study_stage='formal', analysis_id=analysis.analysis_id, registration=reg_ref,
        original_envelope=e.ref(envelope), original_analysis_plan=reg['original_analysis_plan'],
        complete_index=reg['complete_index'], complete_audit=reg['complete_audit'],
        original_design_source_hash=core.source_hash, analysis_implementation_hash=reg['implementation_hash'],
        primary_outcome='nonempty A_cert with original ITT METHOD_FAILED=0', independent_tasks=len(ids),
        trials=len(rows), stratum_counts=np.bincount(strata).tolist(), family_alpha=design['family_alpha'],
        primary_contrasts=contrasts, sensitivities=sensitivity, methods=methods, usage=usage,
        all_registered_trials_included=True, equivalence_test_performed=False,
        posthoc_power_or_sample_size_selection=False, generation_calls=0, market_runs=0, replay_cells=0,
        original_f70_strict_gate_passed=False, financial_online_authorized=False,
        natural_replay_status='PAUSED_PARTIAL_EXCLUDED_FROM_THIS_ANALYSIS',
        scope_declaration=core.scope_declaration,
        inference_scope='Original GLM exact-analytic heldout task comparison; natural, financial and OOD remain separate.',
        tables={k:e.ref(out/f) for k,f in [('episodes','all_episode_outcomes.csv'),('vectors','task_vectors.json'),
                                         ('evidence','episode_evidence.json')]})
    e.durable_json(out/'summary.json', result)
    for ref in reg['bindings']:
        e.checked_reference(ref)
    e.require(e.ref(registration) == reg_ref, 'REGISTRATION_CHANGED')
    assert_paused(e.read(reg['user_pause']['path']))
    complete_formal_gate(core, envelope, reg['complete_index']['path'], reg['complete_audit']['path'])
    e.durable_json(out/'COMPLETED.json', dict(status=result['status'], summary=e.ref(out/'summary.json'),
        tables=result['tables'], original_f70_strict_gate_passed=False, generation_calls=0, market_runs=0, replay_cells=0))
    return result


def deny_network():
    def denied(*args, **kwargs):
        raise RuntimeError('PHASE8_NETWORK_FORBIDDEN')
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
