"""Pre-registered development analytic pilot; formal/financial execution is separate."""
from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Literal
from pydantic import Field, model_validator
from core.experiments.archive import durable_json
from core.experiments.identity import research_hash, file_sha256
from core.experiments.strict_types import StrictResearchModel, load_research_model, strict_json_loads
from .schema import TaskSpec, ModelCallConfig
from .runner import run_design_episode, verify_episode, freeze_design_source, design_implementation_manifest
from .live_budget import SpendLedger, load_deepseek_settings
from .live_protocol import LiveProtocolMethod,SharedEvidenceProtocolMethod
from .llm_adapter import DesignAgentClient
from .baselines import CentralRuleMethod

METHODS=['single','independent','free_debate','abn','cdr_s0_v0','cdr_s0_v1','cdr_s1_v0','cdr_s1_v1','central_rule']


class PilotPlan(StrictResearchModel):
    schema_version: Literal[1]
    study_stage: Literal['pilot']
    mode: Literal['live']
    plan_id: str
    authorization_id: str
    task_paths: list[str] = Field(min_length=1)
    task_sha256: dict[str,str]
    split_manifest_path: str
    split_manifest_sha256: str
    methods: list[str]
    model: ModelCallConfig
    design_turns: int = Field(ge=3,le=12)
    llm_repeats: Literal[1]
    monetary_cap_cny: int = Field(ge=1,le=100)
    input_price_nano: int = Field(gt=0)
    output_price_nano: int = Field(gt=0)
    cached_price_nano: int = Field(ge=0)
    pricing_source: str
    model_version_as_documented: str
    price_checked_date: str
    sampling_rationale: str
    stop_rule: Literal['unknown_usage_or_identity_or_three_consecutive_failures_v1']
    collection_rule: Literal['all_trials_including_failures_source_stratified_v1']
    exclusions: dict[str,str]

    @model_validator(mode='after')
    def check(self):
        if set(self.methods)!=set(METHODS) or len(self.methods)!=len(METHODS):raise ValueError('PILOT_METHOD_SET_MISMATCH')
        if set(self.task_paths)!=set(self.task_sha256) or len(self.task_paths)!=len(set(self.task_paths)):raise ValueError('TASK_INPUT_SET_MISMATCH')
        if self.model.source!='live' or self.model.provider!='deepseek' or self.model.model_id!='deepseek-flash':raise ValueError('PILOT_FIXED_MODEL_MISMATCH')
        if self.model.max_request_attempts!=1:raise ValueError('PILOT_NO_HIDDEN_RETRY')
        return self


class PilotPlanV2(PilotPlan):
    schema_version: Literal[2]
    protocol_version: Literal['bounded_llm_protocol_v2']
    feedback_version: Literal['common_operation_receipts_v1']
    prior_split_manifests: dict[str,str] = Field(min_length=1)
    planning_design_path: str
    planning_design_sha256: str


class MechanismPilotSpec(StrictResearchModel):
    schema_version: Literal[1]
    experiment_id: Literal['analytic-shared-evidence-v3']
    purpose: str
    strata: dict[Literal['role_conflict','model_conflict','infeasible'],int]
    maximum_development_tasks: Literal[6]
    initial_evidence: Literal['all_registered_candidates_before_design_v1']
    sampling_rule: Literal['prospective_conditional_conflict_draws_v1']
    endogenous_errors_only: Literal[True]
    prior_pilots_not_pooled: Literal[True]
    post_pilot_decision_rule: str
    population_rationale: str
    unchanged_primary_contrasts: list[str]
    forbidden_claims: list[str]

    @model_validator(mode='after')
    def check(self):
        if self.strata!={'role_conflict':2,'model_conflict':2,'infeasible':2}:raise ValueError('V3_FIXED_SIX_TASK_DESIGN')
        if set(self.unchanged_primary_contrasts)!={'full_vs_single','factor_S','factor_V','interaction_SV'}:raise ValueError('V3_ORIGINAL_CONTRASTS_REQUIRED')
        return self


class PilotPlanV3(PilotPlanV2):
    schema_version: Literal[3]
    protocol_version: Literal['bounded_llm_protocol_v3']
    feedback_version: Literal['common_operation_receipts_v1']
    experiment_spec_path: str
    experiment_spec_sha256: str


def inspect_conflict_design(task):
    """Independent scalar check of declared task conditions, never an agent hint."""
    supported={};role_supported={}
    for c in task.candidate_pool:
        rows=[x for x in task.analytic_outcomes if x.candidate_id==c.candidate_id]
        eligible={r.role_id:all(1-sum(w*x.losses.model_dump()[k] for k,w in r.weights.model_dump().items())>=r.rho for x in rows) for r in task.roles}
        constraints=all(x.violations[a.quantity]<=a.upper_bound for x in rows for a in task.outcome_constraints)
        supported[c.candidate_id]=all(eligible.values()) and constraints
        role_supported[c.candidate_id]=eligible
    stratum=task.prompt_resources.get('conflict_stratum')
    acceptable=[k for k,v in supported.items() if v]
    if stratum=='infeasible':
        if acceptable or not all(any(v[r.role_id] for v in role_supported.values()) for r in task.roles):raise ValueError('V3_INFEASIBLE_CONDITION_NOT_MET')
    elif stratum in {'role_conflict','model_conflict'}:
        if not acceptable:raise ValueError('V3_FEASIBLE_CONDITION_NOT_MET')
    else:raise ValueError('V3_UNKNOWN_CONFLICT_STRATUM')
    return {'stratum':stratum,'acceptable_count':len(acceptable),'complete_pool_size':len(supported)}


def load_pilot_plan(path):
    body=strict_json_loads(Path(path).read_text())
    return {1:PilotPlan,2:PilotPlanV2,3:PilotPlanV3}.get(body.get('schema_version'),PilotPlan).model_validate(body)


def validate_prior_independence(current,priors):
    """Changing labels cannot turn previously used or held-out data into new draws."""
    for previous in priors:
        for field in ('task_id','underlying_cluster_id','analytic_content_hash','source_dependencies','replication_namespace'):
            old={v for row in previous['tasks'] for v in (row[field] if field=='source_dependencies' else [row[field]])}
            new={v for row in current['tasks'] for v in (row[field] if field=='source_dependencies' else [row[field]])}
            if old & new:raise ValueError('PRIOR_PILOT_SOURCE_OVERLAP:'+field)
    return {'status':'DISJOINT_FROM_ALL_PRIOR_SPLITS','prior_manifests':len(priors)}


def validate_split_manifest(manifest,tasks):
    by_split={s:[] for s in ('development','validation','test','ood')};clusters={};dependencies={};contents={}
    for entry in manifest['tasks']:
        split=entry['split']
        if split not in by_split:raise ValueError('UNKNOWN_SPLIT')
        by_split[split].append(entry['task_id'])
        for mapping,keys in ((clusters,[entry['underlying_cluster_id']]),(dependencies,entry['source_dependencies']),
                             (contents,[entry['analytic_content_hash']])):
            for key in keys:
                if key in mapping and mapping[key]!=split:raise ValueError('CROSS_SPLIT_SOURCE_LEAKAGE')
                mapping[key]=split
    if not all(by_split.values()):raise ValueError('ALL_FOUR_SPLITS_MUST_BE_REGISTERED')
    if len({r['task_id'] for r in manifest['tasks']})!=len(manifest['tasks']):raise ValueError('DUPLICATE_SPLIT_TASK')
    indexed={r['task_id']:r for r in manifest['tasks']}
    for task in tasks:
        entry=indexed.get(task.task_id)
        if entry is None or task.split!='development' or entry['split']!='development':raise ValueError('PILOT_DEVELOPMENT_ONLY')
        if entry['task_hash']!=task.task_hash or entry['underlying_cluster_id']!=task.underlying_cluster_id:raise ValueError('SPLIT_TASK_CONTENT_MISMATCH')
        if entry['analytic_content_hash']!=research_hash(task.analytic_outcomes):raise ValueError('SPLIT_TABLE_MISMATCH')
    if len({t.underlying_cluster_id for t in tasks})!=len(tasks):raise ValueError('PILOT_REPEATS_ARE_NOT_INDEPENDENT_TASKS')
    return {'status':'VALIDATED','split_counts':{s:len(v) for s,v in by_split.items()},'pilot_independent_tasks':len(tasks)}


def load_pilot_inputs(plan):
    tasks=[]
    for path in plan.task_paths:
        if file_sha256(path)!=plan.task_sha256[path]:raise ValueError('PILOT_TASK_HASH_MISMATCH')
        task=load_research_model(path,TaskSpec)
        if task.origin!='analytic_fixture' or task.evidence_mode!='exact_analytic' or task.signing_layer!='delegated_explicit_llm_v1':
            raise ValueError('ANALYTIC_PILOT_SCOPE_ONLY')
        if task.model_configurations!=[plan.model]:raise ValueError('TASK_MODEL_CONFIG_MISMATCH')
        if task.budget.llm_calls!=plan.design_turns+len(task.required_signers):raise ValueError('CALL_BUDGET_SCHEDULE_MISMATCH')
        tasks.append(task)
    if file_sha256(plan.split_manifest_path)!=plan.split_manifest_sha256:raise ValueError('SPLIT_MANIFEST_CHANGED')
    splits=strict_json_loads(Path(plan.split_manifest_path).read_text())
    check=validate_split_manifest(splits,tasks)
    if isinstance(plan,PilotPlanV2):
        priors=[]
        for path,sha in plan.prior_split_manifests.items():
            if file_sha256(path)!=sha:raise ValueError('PRIOR_SPLIT_MANIFEST_CHANGED')
            priors.append(strict_json_loads(Path(path).read_text()))
        check['prior_independence']=validate_prior_independence(splits,priors)
        if file_sha256(plan.planning_design_path)!=plan.planning_design_sha256:raise ValueError('PLANNING_DESIGN_CHANGED')
    if isinstance(plan,PilotPlanV3):
        if file_sha256(plan.experiment_spec_path)!=plan.experiment_spec_sha256:raise ValueError('EXPERIMENT_SPEC_CHANGED')
        spec=load_research_model(plan.experiment_spec_path,MechanismPilotSpec)
        diagnostics=[inspect_conflict_design(t) for t in tasks]
        counts={s:sum(d['stratum']==s for d in diagnostics) for s in spec.strata}
        if counts!=spec.strata or len(tasks)!=spec.maximum_development_tasks:raise ValueError('V3_FIXED_SIX_TASK_DESIGN')
        check['prospective_conflict_design']={'stratum_counts':counts,'tasks':diagnostics}
    return tasks,check


def run_pilot(plan_path,output,*,credential_path,spend_path,workers=1):
    if workers!=1:raise ValueError('PILOT_SERIAL_EXECUTION_REQUIRED')
    plan=load_pilot_plan(plan_path)
    if not isinstance(plan,PilotPlanV2):raise ValueError('LEGACY_PILOT_REQUIRES_ITS_FROZEN_V1_SOURCE')
    if not isinstance(plan,PilotPlanV3):raise ValueError('LEGACY_PILOT_REQUIRES_ITS_FROZEN_V2_SOURCE')
    tasks,split_check=load_pilot_inputs(plan)
    settings=load_deepseek_settings(credential_path)
    spend=SpendLedger(spend_path,authorization_id=plan.authorization_id,cap_cny=plan.monetary_cap_cny,
        input_nano=plan.input_price_nano,output_nano=plan.output_price_nano,cached_nano=plan.cached_price_nano)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    manifest=freeze_design_source(output)
    durable_json(output/'plan.json',plan.model_dump(mode='json'))
    schedule=[]
    # Rotate method order within task before outcomes, to spread chronological provider drift.
    for i,task in enumerate(tasks):
        order=plan.methods[i%len(plan.methods):]+plan.methods[:i%len(plan.methods)]
        schedule.extend({'episode_id':task.task_id+'--'+method,'task_id':task.task_id,'method':method} for method in order)
    per_call=plan.model.input_reservation_tokens*plan.input_price_nano+plan.model.max_output_tokens*plan.output_price_nano
    maximum=(len(plan.methods)-1)*len(tasks)*(plan.design_turns+3)*per_call
    before=spend.snapshot()
    prior=sum(v['reserved_nano'] for v in before['reservations'].values())
    if maximum+prior>plan.monetary_cap_cny*10**9:raise ValueError('PLANNED_MAXIMUM_EXCEEDS_REMAINING_AUTHORIZATION')
    registration={'study_stage':'pilot','plan_hash':research_hash(plan),'source_hash':manifest.source_hash,
        'registered_at':time.time(),'schedule':schedule,'split_check':split_check,'reserved_worst_case_nano':maximum,
        'analysis_scope':'development feasibility, variance and cost only; not confirmatory effects',
        'natural_collection':'all scheduled live methods, grouped by method source; deterministic reference separate',
        'task_input_hashes':plan.task_sha256,'credential_state':'CONFIGURED_PRIVATE_LOCAL_FILE','spend_contract':spend.contract,
        'protocol_version':plan.protocol_version,'feedback_version':plan.feedback_version,
        'prior_reserved_nano':prior,'planning_design_sha256':plan.planning_design_sha256}
    durable_json(output/'spend_before.json',before)
    durable_json(output/'planning_design.json',strict_json_loads(Path(plan.planning_design_path).read_text()))
    durable_json(output/'experiment_spec.json',strict_json_loads(Path(plan.experiment_spec_path).read_text()))
    registration['experiment_spec_sha256']=plan.experiment_spec_sha256
    durable_json(output/'registration.json',registration)
    (output/'inputs').mkdir()
    for task in tasks:durable_json(output/'inputs'/(task.task_id+'.json'),task.model_dump(mode='json'))
    task_map={t.task_id:t for t in tasks};episodes=[];consecutive_failed=0;stopping=None
    def save():
        index={'schema_version':1,'study_stage':'pilot','plan_hash':research_hash(plan),'source_hash':manifest.source_hash,
            'registration_sha256':file_sha256(output/'registration.json'),'episodes':episodes,
            'planned_trials':len(schedule),'indexed_trials':len(episodes),'independent_task_count':len(tasks),
            'market_runs':0,'llm_repeats_per_task_method':1,'stopping_reason':stopping,
            'spend_snapshot':spend.snapshot(),'formal_status':'NOT_RUN','financial_status':'BLOCKED_NO_HEALTH_FREEZE'}
        durable_json(output/'benchmark_index.json',index);return index
    for item in schedule:
        if stopping or spend.snapshot()['blocked_reason']:
            stopping=stopping or spend.snapshot()['blocked_reason']
            episodes.append({**item,'status':'NOT_RUN_STOPPING_RULE','reason':stopping});save();continue
        if design_implementation_manifest().source_hash!=manifest.source_hash:raise RuntimeError('PILOT_SOURCE_CHANGED')
        task=task_map[item['task_id']];method=item['method'];destination=output/'episodes'/item['episode_id']
        if method=='central_rule':
            body=task.model_dump(mode='json');body.update(signing_layer='explicit_rule_owners_v1',model_configurations=[])
            task=TaskSpec.model_validate(body);implementation=CentralRuleMethod(protocol_enabled=True)
        else:
            implementation=SharedEvidenceProtocolMethod(method,lambda ledger,path:DesignAgentClient(plan.model,ledger,path,
                allow_live=True,settings=settings,spend_ledger=spend),turns=plan.design_turns)
        result=run_design_episode(task,implementation,destination,episode_id=item['episode_id'])
        verification=verify_episode(destination)
        episodes.append({**item,'status':'RECORDED','path':str(destination),'terminal_state':result.terminal_state,
            'live_calls':result.common_contract['live_calls'],'result_sha256':file_sha256(destination/'episode_result.json'),
            'verification':verification,'natural_trace':method!='central_rule' and result.live_status=='RUN',
            'reference_only':method=='central_rule','source':'rule' if method=='central_rule' else 'live'})
        consecutive_failed=consecutive_failed+1 if result.terminal_state=='METHOD_FAILED' else 0
        if consecutive_failed>=3:stopping='THREE_CONSECUTIVE_FAILED_EPISODES'
        save()
        print(json.dumps({'episode_id':item['episode_id'],'terminal':result.terminal_state,'live_calls':result.common_contract['live_calls']}),flush=True)
    index=save()
    durable_json(output/'COMPLETED.json',{'index_sha256':file_sha256(output/'benchmark_index.json'),
        'registration_sha256':file_sha256(output/'registration.json'),'complete_planned_index':len(episodes)==len(schedule)})
    return index
