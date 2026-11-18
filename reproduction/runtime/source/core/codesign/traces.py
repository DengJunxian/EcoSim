"""Small collection manifest and integrity checks; no workflow or trace selection agent."""
from __future__ import annotations
from pathlib import Path
import time
from typing import Literal
from pydantic import Field,model_validator
from core.experiments.strict_types import StrictResearchModel,strict_json_loads
from core.experiments.identity import research_hash,file_sha256
from core.experiments.archive import durable_json
from .schema import EpisodeResult


class TraceCollectionPlan(StrictResearchModel):
    plan_id:str=Field(min_length=1)
    collection_kind:Literal['synthetic_fixture','natural_pilot']
    source_strata:list[Literal['rule','mock','recorded','live']]=Field(min_length=1)
    sampling_rule:Literal['all_planned_episodes_source_stratified_v1']
    planned_episode_ids:list[str]=Field(min_length=1)
    included_episode_ids:list[str]
    excluded_episode_reasons:dict[str,str]

    @model_validator(mode='after')
    def _complete(self):
        planned=set(self.planned_episode_ids);included=set(self.included_episode_ids);excluded=set(self.excluded_episode_reasons)
        if len(planned)!=len(self.planned_episode_ids) or len(included)!=len(self.included_episode_ids):raise ValueError('DUPLICATE_COLLECTION_ID')
        if included&excluded or included|excluded!=planned or any(not s for s in self.excluded_episode_reasons.values()):
            raise ValueError('COLLECTION_INCLUSION_PARTITION_INCOMPLETE')
        if self.collection_kind=='natural_pilot' and set(self.source_strata)!={'live'}:raise ValueError('NATURAL_PILOT_REQUIRES_LIVE_SOURCE')
        return self


def register_trace_collection(plan:TraceCollectionPlan,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    durable_json(output/'plan.json',plan.model_dump(mode='json'))
    receipt={'plan_hash':research_hash(plan),'plan_sha256':file_sha256(output/'plan.json'),'registered_at':time.time()}
    durable_json(output/'registration.json',receipt);return output


def _check_episode(path,plan,registration):
    from .runner import verify_episode
    path=Path(path).resolve();verify_episode(path)
    result=EpisodeResult.model_validate_json((path/'episode_result.json').read_text())
    config_path=path/'llm/configuration.json'
    source=strict_json_loads(config_path.read_text())['source'] if config_path.exists() else 'rule'
    if source not in plan.source_strata:raise ValueError('TRACE_SOURCE_STRATUM_MISMATCH')
    raw_paths=set(result.message_refs+result.candidate_refs)
    for message in result.message_refs:
        envelope=strict_json_loads(Path(message).read_text())
        if envelope.get('raw_ref'):raw_paths.add(envelope['raw_ref'])
    call_index=path/'llm/call_index.json'
    calls=strict_json_loads(call_index.read_text()) if call_index.exists() else []
    for call in calls:
        raw_paths.add(call['request_ref'])
        if call.get('response_ref'):raw_paths.add(call['response_ref'])
    if not raw_paths:raise ValueError('TRACE_MISSING_RAW')
    refs=[]
    for raw_path in sorted(raw_paths):
        actual=Path(raw_path).resolve()
        if not actual.is_relative_to(path) or not actual.is_file():raise ValueError('TRACE_MISSING_OR_FOREIGN_RAW')
        refs.append({'path':str(actual),'sha256':file_sha256(actual)})
    if plan.collection_kind=='natural_pilot':
        if source!='live' or result.live_status!='RUN' or not calls or not result.budget_snapshot['used']['llm_calls']:
            raise ValueError('SYNTHETIC_CANNOT_BECOME_NATURAL_LIVE_TRACE')
        if result.started_at<registration['registered_at']:raise ValueError('NATURAL_TRACE_PRECEDES_COLLECTION_REGISTRATION')
        for call in calls:
            if strict_json_loads(Path(call['request_ref']).read_text()).get('source')!='live':raise ValueError('RAW_SOURCE_MISMATCH')
            if call.get('response_ref') and strict_json_loads(Path(call['response_ref']).read_text()).get('source')!='live':raise ValueError('RAW_SOURCE_MISMATCH')
    return {'episode_id':result.episode_id,'episode_path':str(path),'result_sha256':file_sha256(path/'episode_result.json'),
        'marker_sha256':file_sha256(path/'COMMITTED.json'),'source':source,'method':result.method,
        'underlying_cluster_id':result.underlying_cluster_id,'terminal_state':result.terminal_state,'raw_refs':refs}


def export_trace_collection(collection,episode_paths):
    collection=Path(collection).resolve()
    if (collection/'trace-index.json').exists():raise ValueError('TRACE_COLLECTION_ALREADY_FROZEN')
    plan=TraceCollectionPlan.model_validate_json((collection/'plan.json').read_text())
    registration=strict_json_loads((collection/'registration.json').read_text())
    if registration['plan_hash']!=research_hash(plan) or registration['plan_sha256']!=file_sha256(collection/'plan.json'):
        raise ValueError('TRACE_COLLECTION_PLAN_TAMPERED')
    entries=[_check_episode(path,plan,registration) for path in episode_paths]
    ids=[e['episode_id'] for e in entries]
    if len(set(ids))!=len(ids) or set(ids)!=set(plan.planned_episode_ids):raise ValueError('COMPLETE_RAW_EPISODE_INDEX_REQUIRED')
    entries=sorted(entries,key=lambda e:plan.planned_episode_ids.index(e['episode_id']))
    index={'plan_hash':research_hash(plan),'registration_hash':research_hash(registration),'episodes':entries,
        'included_episode_ids':plan.included_episode_ids,'excluded_episode_reasons':plan.excluded_episode_reasons,
        'natural_trace_status':'COLLECTED' if plan.collection_kind=='natural_pilot' else 'NOT_COLLECTED',
        'collection_kind':plan.collection_kind,'provenance_limit':'local recorded source and complete planned index verified; no claim about unrecorded external selection'}
    durable_json(collection/'trace-index.json',index)
    durable_json(collection/'FROZEN.json',{'index_sha256':file_sha256(collection/'trace-index.json'),
        'plan_sha256':file_sha256(collection/'plan.json'),'registration_sha256':file_sha256(collection/'registration.json')})
    return index


def import_trace_collection(collection):
    collection=Path(collection).resolve();marker=strict_json_loads((collection/'FROZEN.json').read_text())
    for key,name in [('index_sha256','trace-index.json'),('plan_sha256','plan.json'),('registration_sha256','registration.json')]:
        if file_sha256(collection/name)!=marker[key]:raise ValueError('TRACE_COLLECTION_HASH_MISMATCH')
    index=strict_json_loads((collection/'trace-index.json').read_text())
    plan=TraceCollectionPlan.model_validate_json((collection/'plan.json').read_text())
    registration=strict_json_loads((collection/'registration.json').read_text())
    if index['plan_hash']!=research_hash(plan) or index['registration_hash']!=research_hash(registration):raise ValueError('TRACE_PLAN_IDENTITY_MISMATCH')
    if index['collection_kind']!=plan.collection_kind or index['natural_trace_status']!=('COLLECTED' if plan.collection_kind=='natural_pilot' else 'NOT_COLLECTED'):
        raise ValueError('TRACE_SOURCE_LABEL_MISMATCH')
    expected=[_check_episode(e['episode_path'],plan,registration) for e in index['episodes']]
    if expected!=index['episodes'] or [e['episode_id'] for e in expected]!=plan.planned_episode_ids:
        raise ValueError('TRACE_COMPLETE_INDEX_MISMATCH')
    if index['included_episode_ids']!=plan.included_episode_ids or index['excluded_episode_reasons']!=plan.excluded_episode_reasons:
        raise ValueError('TRACE_INCLUSION_RULE_MISMATCH')
    return index
