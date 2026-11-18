"""Read-only KDD27 evidence build; engineering outputs never satisfy research admission."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from civitas.analytics.kdd27 import metric, relative_turnover
from civitas.analytics.kdd27_data import RunEvidence, analyze_run, require_figure_eligible
from civitas.analytics.kdd27_rq2 import event_metrics
from civitas.analytics.kdd27_statistics import factorial_interaction, paired_summary, world_summary
from civitas.contracts.base import content_hash
from civitas.storage.commit_store import CommitStore

VERSION = 'kdd27-measured-build-v1'
METRIC_VERSION = 'kdd27-metrics-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def table(path, rows):
    require(bool(rows), 'EMPTY_TABLE_REQUIRES_EXPLICIT_SCHEMA')
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def checked_reference(ref):
    require(set(ref) == {'path', 'sha256', 'bytes'}, 'REFERENCE_SCHEMA')
    path = Path(ref['path'])
    require(path.is_file() and path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'], 'INPUT_HASH_CHANGED')
    return path


def admit(record, scope):
    record = RunEvidence.model_validate(record)
    require(scope in {'engineering', 'research'}, 'UNKNOWN_EVIDENCE_SCOPE')
    if scope == 'research':
        require_figure_eligible([record.model_dump()])
    else:
        require(record.evidence_scope == 'engineering' and record.output_kind == 'measured_simulation'
                and record.status == 'completed' and record.verified_integrity and not record.unknown_fields
                and record.world_schema == 3 and all(x is not None for x in
                    (record.source_hash, record.code_commit, record.resolved_config_hash, record.initial_state_hash, record.raw_path)),
                'INELIGIBLE_ENGINEERING_MEASUREMENT')
    return record


def load_selection(path, expected_hash):
    require(sha(path) == expected_hash, 'SELECTION_HASH_CHANGED')
    selection = read(path)
    require(selection['schema_version'] == 'kdd27-selection-v1', 'UNKNOWN_SELECTION_SCHEMA')
    require(selection['metric_version'] == METRIC_VERSION, 'UNKNOWN_METRIC_VERSION')
    require(selection['new_simulation_steps_allowed'] == selection['new_model_calls_allowed'] == 0, 'OFFLINE_SCOPE_REQUIRED')
    require(selection['formal_runs_selected'] == 0, 'FORMAL_RQ_ADAPTER_NOT_QUALIFIED')
    seen = set()
    for entry in selection['runs']:
        record = RunEvidence.model_validate(entry['record'])
        require(record.run_id not in seen, 'DUPLICATE_RUN_SELECTION')
        seen.add(record.run_id)
        checked_reference(entry['registry_source'])
        frozen = {Path(f['path']).resolve() for f in entry['archive_files']}
        actual = {p.resolve() for p in Path(record.raw_path).rglob('*') if p.is_file() and not p.is_symlink()}
        require(actual == frozen, 'ARCHIVE_FILE_SET_CHANGED')
        for ref in entry['archive_files']:
            checked_reference(ref)
        if entry['analysis_action'] == 'recompute_engineering':
            admit(record.model_dump(), 'engineering')
        else:
            require(entry['analysis_action'] == 'register_only' and record.output_kind == 'mock_fixture', 'UNKNOWN_SELECTION_ACTION')
    for family in selection['planned_families']:
        source = read(checked_reference(family['source']))
        require(family['rows'] == source.get('rows', source.get('run_rows')), 'PLAN_CENSUS_CHANGED')
    for entry in selection['performance']:
        checked_reference(entry['measurement'])
    for key in ('historical_sources', 'method_sources', 'paper_sources', 'analysis_sources'):
        for ref in selection[key]:
            checked_reference(ref)
    return selection


def metric_rows(payload, study):
    record = payload['run']
    values = dict(payload['summary']['metrics'])
    if study == 'RQ2':
        rec = payload['event']['recovery']
        for key, unit in [('mdd', 'fraction'), ('restricted_time', 'model_steps'), ('censored', 'indicator')]:
            values[key] = metric(rec[key], len(payload['facts']), len(payload['facts']), unit)
    return [{'study': study, 'run_id': record['run_id'], 'condition_id': record['condition_id'],
             'world_seed': record['world_seed'], 'pair_id': record['pair_id'], 'protocol_version': record['protocol_version'],
             'status': record['status'], 'evidence_scope': 'engineering', 'output_kind': record['output_kind'],
             'metric_id': name, 'metric_version': value['definition_version'], 'value': value['value'],
             'unit': value['unit'], 'n_valid': value['sample_count'], 'n_total': value['denominator_count'],
             'reason': value['reason'], 'window_start': payload['summary']['window'][0],
             'window_end': payload['summary']['window'][1], 'clock': 'model_day' if name in payload['summary']['daily_statistics']['metrics'] else 'committed_step',
             'raw_path': record['raw_path'], 'source_hash': record['source_hash'], 'config_hash': record['resolved_config_hash'],
             'code_commit':record['code_commit'],'raw_manifest_hash':payload['source_archive_manifest_hash'],
             'initial_state_hash': record['initial_state_hash'], 'projection': 'projections/' + record['run_id'] + '.json'}
            for name, value in values.items()]


def build(selection_path, selection_hash, out, *, scope='engineering'):
    selection_path, out = Path(selection_path).resolve(), Path(out).resolve()
    selection = load_selection(selection_path, selection_hash)
    require(len(selection['runs']) == 22 and sum(e['analysis_action']=='recompute_engineering' for e in selection['runs']) == 18,
            'ENGINEERING_CENSUS_CHANGED')
    require(scope == 'engineering', 'FORMAL_RESULTS_PENDING_NO_RESEARCH_BUILD')
    require(not out.exists(), 'OUTPUT_EXISTS')
    require(not any(out.is_relative_to(Path(e['record']['raw_path']).resolve()) for e in selection['runs']), 'OUTPUT_INSIDE_RAW_ARCHIVE')
    out.mkdir(parents=True)
    (out/'projections').mkdir()
    run_rows, failure_rows, params, rq2, selected_records = [], [], [], {}, []
    for entry in selection['runs']:
        ref = entry['record']; selected_records.append(ref)
        if entry['analysis_action'] == 'register_only':
            failure_rows.append({'id': ref['run_id'], 'category': 'excluded_source', 'status': ref['status'],
                'reason': entry['reason'], 'source': ref['raw_path'], 'metric_value': None})
            continue
        payload = analyze_run(ref['raw_path'], experiment_id=ref['experiment_id'],
            condition_id=ref['condition_id'], protocol_version=ref['protocol_version'])
        require(payload['run'] == ref, 'RUN_IDENTITY_OR_STATUS_CHANGED')
        store = CommitStore(ref['raw_path']); cfg = store.identity.config
        require(content_hash(cfg) == ref['resolved_config_hash'], 'RESOLVED_CONFIG_CHANGED')
        if entry['study'] == 'RQ2':
            _, initial = next(store.states()); market = cfg['market']
            accounts = [a for a in initial.data['multiasset_market']['ledger']['initial_accounts'] if a['account_id'] in initial.actors]
            for fact in payload['facts']:
                fact['notional'] = sum(t['notional_minor'] for t in payload['trades'] if t['trade']['tick'] == fact['tick']) / market['cash_scale']
            payload['event'] = event_metrics(payload['facts'], event_tick=cfg['event_tick'],
                horizon=cfg['ticks'] - cfg['event_tick'] + 1,
                pre_price=float(initial.data['kdd27']['last_quote']['last_price_ticks'] * market['price_tick']),
                initial_cash=sum(a['balances'][market['currency']] for a in accounts),
                initial_shares=sum(a['balances'][market['asset_id']] for a in accounts), ticks_per_day=market['ticks_per_day'])
            rq2[ref['condition_id']] = payload
        payload['source_archive_manifest_hash'] = content_hash(entry['archive_files'])
        dump(out/'projections'/(ref['run_id']+'.json'), payload)
        run_rows.extend(metric_rows(payload, entry['study']))
        params.append({'run_id': ref['run_id'], 'study': entry['study'], 'condition_id': ref['condition_id'],
            'n_agents': cfg['n_agents'], 'ticks': cfg['ticks'], 'root_seed': cfg['root_seed'],
            'ticks_per_day': cfg['market']['ticks_per_day'], 'protocol_id': cfg['protocol_id'],
            'config_hash': ref['resolved_config_hash'], 'resolved_config': cfg})
    planned = []
    for family in selection['planned_families']:
        for raw in family['rows']:
            require(raw.get('run_id', raw.get('actual_run_id')) is None and not raw['attempts'], 'PLAN_ALREADY_EXECUTED')
            row = {'id': raw.get('plan_id', raw.get('candidate_id')), 'category': 'not_executed',
                'status': raw['status'], 'reason': 'Original formal prerequisites pending',
                'source': family['source']['path'], 'metric_value': None}
            failure_rows.append(row);planned.append({'family': family['family'], **raw})
    for ref in selection['historical_sources']:
        path = Path(ref['path'])
        if path.name == 'provenance.json':
            old = read(path)
            failure_rows.append({'id': 'historical-kdd08-n1000', 'category': 'partial_historical_capacity',
                'status': 'initial_exit_-9_then_partial_5_of_12', 'reason': old['qualification'],
                'source': str(path), 'metric_value': None})
        if 'attempt-diagnosis' in path.name:
            old = read(path)
            failure_rows.append({'id': old['failed_command'], 'category': 'historical_analysis_failure',
                'status': 'failed_analysis_not_failed_simulation', 'reason': old['cause'], 'source': str(path), 'metric_value': None})
    groups = {}
    for row in run_rows:
        groups.setdefault((row['study'], row['condition_id'], row['metric_id']), []).append(row)
    summaries = [{'study': study, 'condition_id': condition, 'metric_id': name, 'metric_version': METRIC_VERSION,
        'evidence_scope': 'engineering', **world_summary(rows)} for (study, condition, name), rows in groups.items()]
    paired = []
    for name in ('mdd', 'turnover', 'two_sided_spread_bps', 'restricted_time', 'censored'):
        for condition in rq2:
            if condition == 'full_control':continue
            rows = [r for r in run_rows if r['study'] == 'RQ2' and r['metric_id'] == name
                    and r['condition_id'] in {'full_control', condition}]
            paired.append({'study': 'RQ2', 'metric_id': name, 'metric_version': METRIC_VERSION,
                'evidence_scope': 'engineering', **paired_summary(rows, control='full_control', treatment=condition)})
    rq2_table = []
    control_turnover = rq2['full_control']['summary']['metrics']['turnover']['value']
    for condition, payload in rq2.items():
        result = payload['event'];rec = result['recovery'];metrics = result['summary']['metrics']
        relative = relative_turnover(metrics['turnover']['value'], control_turnover)
        rq2_table.append({'condition': condition, 'run_id': payload['run']['run_id'], 'mdd': rec['mdd'],
            'turnover': metrics['turnover']['value'], 'relative_turnover': relative['value'],
            'relative_turnover_reason': relative['reason'], 'restricted_time': rec['restricted_time'],
            'censored': rec['censored'], 'tradable_steps': result['tradable_steps'], 'total_steps': len(payload['facts']),
            'spread_bps': metrics['two_sided_spread_bps']['value'], 'spread_n_valid': metrics['two_sided_spread_bps']['sample_count'],
            'no_trade_ratio': metrics['no_trade_ratio']['value'], 'n_worlds': 1, 'ci95': None})
    cells = {cell: rq2[condition]['event']['recovery']['mdd'] for cell, condition in
        [('11', 'full_control'), ('10', 'no_herding_feedback'), ('01', 'no_secondary_contagion'), ('00', 'both_off')]}
    interaction = {'metric': 'mdd', 'cells': cells, 'value': factorial_interaction(cells), 'world_seed': 933001,
        'ci95': None, 'reason': 'ONE_ENGINEERING_WORLD', 'resampling_unit': 'world_seed'}
    measurements = [{**entry, 'value': read(entry['measurement']['path'])} for entry in selection['performance']]
    timing = []
    for pair in sorted({e['pair'] for e in measurements}):
        a, b = [next(e['value'] for e in measurements if e['pair'] == pair and e['variant'] == v) for v in 'AB']
        require(a['status'] == b['status'] == 'completed' and a['config_hash'] == b['config_hash'], 'TIMING_PAIR_MISMATCH')
        require(a['commits'] == b['commits'], 'TIMING_SEMANTICS_DIFFER')
        av, bv = a['total']['wall_seconds'], b['total']['wall_seconds']
        timing.append({'pair': pair, 'A_seconds': av, 'B_seconds': bv, 'B_over_A': bv/av,
            'reduction_percent': 100*(1-bv/av), 'world_seed': a['seed'], 'n_agents': a['n_agents'],
            'ticks': a['ticks_requested'], 'A_run_id': a['run_id'], 'B_run_id': b['run_id'], 'unit': 'timing_repeat_same_world'})
    traces = [
        {'id': 'T1', 'metric': 'two_sided_spread_bps_and_coverage', 'source_runs': [r['run_id'] for r in run_rows if r['study']=='RQ1' and r['condition_id']=='full'][:1],
         'values': {r['metric_id']: {'value':r['value'],'n_valid':r['n_valid'],'n_total':r['n_total']} for r in run_rows if r['study']=='RQ1' and r['condition_id']=='full' and r['metric_id'] in {'two_sided_spread_bps','two_sided_coverage'}},
         'method':'CommitStore.states -> project_snapshot -> summarize_facts; missing quote excluded from mean, kept in coverage denominator'},
        {'id': 'T2', 'metric':'paired_MDD_and_four_cell_interaction', 'source_runs':[rq2[c]['run']['run_id'] for c in ('full_control','no_secondary_contagion','no_herding_feedback','both_off')],
         'values':{'both_off_minus_control':rq2['both_off']['event']['recovery']['mdd']-rq2['full_control']['event']['recovery']['mdd'],'interaction':interaction['value']},
         'method':'Raw committed price path -> event_metrics/recovery -> paired_summary/factorial_interaction; one seed, CI null'},
        {'id':'T3','metric':'median_paired_wall_reduction_percent','source_measurements':selection['performance'],
         'values':{'median_paired_reduction_percent':statistics.median(r['reduction_percent'] for r in timing),
                   'ratio_of_means_reduction_percent':100*(1-statistics.mean(r['B_seconds'] for r in timing)/statistics.mean(r['A_seconds'] for r in timing))},
         'method':'Original per-run measured total wall seconds; compute each B/A, then median; no new timing runs, no confidence interval'}]
    table(out/'run_level.csv', run_rows);table(out/'paired_effects.csv', paired);table(out/'summary.csv', summaries)
    table(out/'failure.csv', failure_rows);table(out/'parameters.csv', params);table(out/'rq2_diagnostics.csv', rq2_table);table(out/'timing_pairs.csv', timing)
    dump(out/'selected_runs.json', selected_records);dump(out/'planned_rows.json', planned)
    dump(out/'traces.json', traces);dump(out/'interaction.json', interaction)
    with (out/'engineering_rows.tex').open('x') as handle:
        for row in rq2_table:
            name = row['condition'].replace('_', r'\_')
            handle.write(f"{name} & {100*row['mdd']:.4f} & {row['restricted_time']} & {row['tradable_steps']}/{row['total_steps']} "+r'\\'+'\n')
    with (out/'parameters.tex').open('x') as handle:
        for row in params:
            name=row['condition_id'].replace('_',r'\_')
            handle.write(f"{row['study']}: {name} & {row['n_agents']} & {row['ticks']} & {row['root_seed']} & {row['ticks_per_day']} "+r'\\'+'\n')
    # shortcut: official RQ evidence is pending; add a protocol-specific research adapter before admitting it.
    claims = [('C01','supported','Implemented mapped profile subset; 42 micro-verified and 6 partial items; method diagram only'),
              ('C02','supported','Three recorded same-platform read-only restores to tick12; no fresh trajectory execution'),
              ('C03','pending','Formal RQ1 and empirical daily/L1 reference absent'),
              ('C04','pending','One short engineering world; formal paired worlds and neighborhood effects absent'),
              ('C05','pending','No real-model increment experiment'),('C06','pending','No online paired quality/cost and equivalence result'),
              ('C07','supported','Three timing repeats of one N100/T12 world; no scale inference')]
    detail={
        'C01':('mapped implemented subset','method_contract_coverage','42 verified;6 partial',42,48,'method_contract',None),
        'C02':('KDD08 B repeats1-3','final_state_hash','read-only restore vs saved tick12',3,3,'restore_check',None),
        'C03':('full;zero_intelligence;homogeneous_linear_value','frozen five daily/L1 metrics and empirical distance','full vs both baselines',0,120,'formal_run',None),
        'C04':('original RQ2 eleven arms','MDD;turnover;spread;restricted_recovery','paired and four-cell contrasts',0,220,'formal_branch',None),
        'C05':('rule;heuristic;llm;hybrid','frozen cognition quality and coverage','backend and wording contrasts',0,160,'formal_branch',None),
        'C06':('llm;hybrid','quality equivalence and actual cost','paired backend cost/quality',0,160,'formal_branch',None),
        'C07':('KDD08 A/B same N100T12 world','median paired wall reduction percent','B vs A',3,3,'timing_repeat_same_world',traces[2]['values']['median_paired_reduction_percent'])}
    claim_rows=[]
    for c,status,reason in claims:
        condition,name,comparison,valid,total,unit,effect=detail[c]
        claim_rows.append({'claim_id':c,'status':status,'scope':reason,'condition':condition,'metric':name,'comparison':comparison,
            'sample_unit':unit,'n_valid':valid,'n_total':total,'n_failed':0,'n_partial':6 if c=='C01' else 0,
            'formal_worlds':0,'effect':effect,'ci95':None,'interval_reason':'DESCRIPTIVE_ENGINEERING_OR_NO_FORMAL_RUNS',
            'analysis_version':VERSION,'analysis_base_commit':selection['base_commit'],'analysis_script_sha256':sha(__file__),
            'selection_sha256':selection_hash,'effect_source':'traces.json' if c=='C07' else None,
            'engineering_context':'traces.json;selected_runs.json','raw_source_index':'frozen selection file hashes and planned_rows.json',
            'scientific_hypothesis':'pending'})
    table(out/'claim_evidence.csv',claim_rows)
    planned_figures = [{'id':x,'status':'pending','output':None,'reason':'Original formal RQ prerequisites not met'} for x in
        ('fig:rq1','fig:rq2','fig:rq3','tab:rq1-results','tab:rq2-results','tab:rq3-results')]
    dump(out/'figure_manifest.json',{'schema_version':1,'formal_results':planned_figures,
        'buildable':['fig:overview','eng:rq1-liquidity','eng:rq2-arms','eng:timing'],'scope':'method and engineering appendix only'})
    for entry in selection['runs']:
        for ref in entry['archive_files']:checked_reference(ref)
    files={p.relative_to(out).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
    receipt={'schema_version':VERSION,'metric_version':METRIC_VERSION,'selection':{'path':str(selection_path),'sha256':selection_hash},
        'analysis_script_sha256':sha(__file__),'analysis_sources':selection['analysis_sources'],'base_commit':selection['base_commit'],'files':files,
        'recomputed_runs':18,'registered_runs':22,'mock_register_only':4,'planned_rows':len(planned),'new_steps':0,'new_model_calls':0,
        'formal_figures_built':0,'archive_files_unchanged':True}
    dump(out/'BUILD.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--selection-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--scope',choices=['engineering','research'],default='engineering')
    args=parser.parse_args();result=build(args.selection,args.selection_sha256,args.out,scope=args.scope)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))


if __name__=='__main__':main()
