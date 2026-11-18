"""Explicit research benchmark entrypoint; diagnostic, rule-only plan is first class."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Literal
from pydantic import Field
from core.experiments.strict_types import StrictResearchModel, load_research_model
from core.experiments.archive import durable_json, verify_archive_v3, rows
from core.experiments.identity import file_sha256, research_hash
from core.experiments.design_config import ResearchRunSpec
from .schema import TaskSpec
from .baselines import (PolicySearchMethod,CentralRuleMethod,SingleAgentMethod,IndependentVotingMethod,
                       FreeDebateMethod,ABNControlMethod,AllAbstainMethod)
from .runner import run_design_episode, verify_episode
from .verifier import UtilityInput, evaluate_task_utility


class MethodPlan(StrictResearchModel):
    name: Literal['noop','fixed_rule','enumerated','random','nsga2','central_rule','single_proposal_only','single_simulation_guided',
                  'independent','free_debate','abn','all_abstain','cdr']
    seed: int = Field(ge=0)
    population_size: int = Field(ge=4)
    generations: int = Field(ge=1)
    model_configuration_index: int | None = Field(default=None,ge=0)
    response_fixture_path: str | None = None
    protocol_enabled: bool = False
    separate_fact_assent: bool | None = None
    maintain_scope_support: bool | None = None


class BenchmarkPlan(StrictResearchModel):
    schema_version: Literal[1]
    study_stage: Literal['diagnostic']
    mode: Literal['rule_only','mock','recorded','live']
    task_paths: list[str] = Field(min_length=1)
    methods: list[MethodPlan] = Field(min_length=1)
    lob_diagnostic_spec: str | None


def run_lob_diagnostic(spec_path,task,output,workers):
    from .budget import BudgetLedger
    from core.experiments.jobs import Job,JobCoordinator
    from core.experiments.measurement import StepMeasurement
    spec=load_research_model(spec_path,ResearchRunSpec)
    if spec.data_phase!='diagnostic':raise ValueError('LOB_DIAGNOSTIC_NAMESPACE_REQUIRED')
    task_data=task.model_dump(mode='json')
    task_data.update(task_id='lob-noop-diagnostic-v1',underlying_cluster_id=spec.random.exogenous_family_id,
        task_family='market_engineering_diagnostic',origin='market',evidence_mode='diagnostic_only',analytic_outcomes=[],
        compiler_world=spec.world.model_dump(mode='json'),compiler_market=spec.base.market.model_dump(mode='json'),
        static_constraints=spec.policy_constraints.model_dump(mode='json'),measurement=spec.measurement.model_dump(mode='json'),
        horizon_steps=spec.base.steps,model_family=[dict(model_id='actual-civitas-rule-v1',family_version=spec.behavior.version,
            world_distribution=spec.world.model_dump(mode='json'),scenario_distribution=spec.scenario.model_dump(mode='json'),
            behavior=spec.behavior.model_dump(mode='json'),hidden_f_declared=True)])
    task=TaskSpec.model_validate(task_data)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    durable_json(output/'task.json',task.model_dump(mode='json'))
    ledger=BudgetLedger(output/'budget.json',task.budget,owner='lob_diagnostic',episode='phase4-lob-diagnostic')
    result=JobCoordinator(output/'jobs',ledger).execute([Job(job_id='noop',spec=spec)],workers=workers)
    if result['status']!='COMMITTED':raise RuntimeError('LOB_DIAGNOSTIC_NOT_COMMITTED')
    archive=Path(result['results'][0]['archive']);verification=verify_archive_v3(archive)
    # The v3 envelope hash has already been verified above; it is not a v2
    # measurement field. Strip only that declared envelope field, never extras.
    records=[StepMeasurement.model_validate({k:v for k,v in r.items() if k!='record_hash'})
             for r in rows(archive/'measurements.jsonl')]
    # This fixture's declared F is constant. Other processes require a declared
    # process evaluator; do not substitute the initial F through a shock.
    if spec.scenario.events:raise ValueError('DIAGNOSTIC_F_PROCESS_ADAPTER_REQUIRED')
    data=UtilityInput(execution_status='completed',records=records,
        hidden_f_by_step={str(r.step):spec.scenario.fundamental_process.initial_value for r in records},
        cumulative_buy_cash=0.0,raw_refs=[str(archive/'measurements.jsonl'),str(archive/'policy_execution.jsonl')])
    if spec.policy.halt is not None or spec.policy.liquidity is not None:raise ValueError('NOOP_DIAGNOSTIC_EXPECTED')
    utility=evaluate_task_utility(task,data)
    report={'origin':'market','scope':'diagnostic_only','eligibility':'INAPPLICABLE','verification':verification,
        'utility':utility,'archive':str(archive),'budget':ledger.snapshot(),'live_calls':0}
    durable_json(output/'diagnostic.json',report);return report


def benchmark(plan_path,output,workers=1,*,credential_path=None,spend_path=None):
    from core.experiments.strict_types import strict_yaml_loads
    if strict_yaml_loads(Path(plan_path).read_text()).get('study_stage')=='pilot':
        from .pilot import run_pilot
        return run_pilot(plan_path,output,workers=workers,
            credential_path=credential_path or str(Path.home()/'.config/cscwd/deepseek.env'),
            spend_path=spend_path or 'artifacts/research/deepseek_spend.json')
    plan=load_research_model(plan_path,BenchmarkPlan);output=Path(output).resolve()
    if plan.mode=='live':raise ValueError('LIVE_REQUIRES_REGISTERED_PILOT_PLAN')
    output.mkdir(parents=True,exist_ok=False);durable_json(output/'plan.json',plan.model_dump(mode='json'))
    episodes=[];tasks=[]
    for task_path in plan.task_paths:
        task=load_research_model(task_path,TaskSpec);tasks.append(task)
        for index,method in enumerate(plan.methods):
            if method.name=='cdr':
                from .cdr import CDRProtocol
                implementation=CDRProtocol(separate_fact_assent=method.separate_fact_assent,maintain_scope_support=method.maintain_scope_support)
            elif method.name.startswith('single_'):
                if plan.mode not in {'mock','recorded'} or method.model_configuration_index is None or not method.response_fixture_path:
                    raise ValueError('EXPLICIT_SINGLE_AGENT_SOURCE_AND_CONFIGURATION_REQUIRED')
                from .llm_adapter import DesignAgentClient
                from core.llm.base import LLMResponse
                from core.experiments.strict_types import strict_json_loads
                config=task.model_configurations[method.model_configuration_index]
                if config.source!=plan.mode:raise ValueError('SOURCE_MODE_MISMATCH')
                fixture=strict_json_loads(Path(method.response_fixture_path).read_text())
                if fixture.get('source')!=plan.mode:raise ValueError('FIXTURE_SOURCE_MISMATCH')
                class FixtureTransport:
                    def __init__(self):self.responses=iter(fixture['responses'])
                    async def complete(self,messages,**kwargs):return LLMResponse(**next(self.responses))
                def factory(ledger,destination):
                    client=DesignAgentClient(config,ledger,destination,transport=FixtureTransport())
                    durable_json(Path(destination)/'transport_fixture.json',fixture)
                    return client
                implementation=SingleAgentMethod(mode=method.name.removeprefix('single_'),client_factory=factory)
            else:
                protocol_types={'independent':IndependentVotingMethod,'free_debate':FreeDebateMethod,'abn':ABNControlMethod,'all_abstain':AllAbstainMethod}
                implementation=(protocol_types[method.name]() if method.name in protocol_types else
                    CentralRuleMethod(protocol_enabled=method.protocol_enabled) if method.name=='central_rule' else
                    PolicySearchMethod(method.name,seed=method.seed,population_size=method.population_size,generations=method.generations))
            episode_id=f'{task.task_id}--{index:02d}-{method.name}'
            destination=output/'episodes'/episode_id
            result=run_design_episode(task,implementation,destination,episode_id=episode_id,method_config=method.model_dump(mode='json'))
            verification=verify_episode(destination)
            episodes.append({'path':str(destination),'result_sha256':file_sha256(destination/'episode_result.json'),
                'verification':verification,'terminal_state':result.terminal_state,'selected_policy_hash':result.selected_policy_hash})
    diagnostic=None
    if plan.lob_diagnostic_spec:diagnostic=run_lob_diagnostic(plan.lob_diagnostic_spec,tasks[0],output/'lob-diagnostic',workers)
    index={'schema_version':1,'study_stage':plan.study_stage,'mode':plan.mode,'plan_hash':research_hash(plan),
        'episodes':episodes,'lob_diagnostic_ref':str(output/'lob-diagnostic/diagnostic.json') if diagnostic else None,
        'live_status':'NOT_RUN','live_calls':0,'statistical_n':0,'scientific_status':'ANALYTIC_AND_ENGINEERING_ONLY'}
    durable_json(output/'benchmark-index.json',index);return index


def main(argv=None):
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    bench=sub.add_parser('benchmark');bench.add_argument('--plan',required=True);bench.add_argument('--out',required=True)
    bench.add_argument('--workers',type=int,default=1)
    bench.add_argument('--credential-file');bench.add_argument('--spend-ledger')
    args=parser.parse_args(argv)
    try:
        result=benchmark(args.plan,args.out,args.workers,credential_path=args.credential_file,spend_path=args.spend_ledger)
        print(json.dumps(result,ensure_ascii=False));return 0
    except (ValueError,TypeError,OSError,RuntimeError) as exc:
        code=3 if type(exc).__name__=='BudgetExceeded' else 2 if isinstance(exc,(ValueError,TypeError)) else 4
        print(json.dumps({'status':'FAILED','error_type':type(exc).__name__,'reason':str(exc)}));return code


if __name__=='__main__':raise SystemExit(main())
