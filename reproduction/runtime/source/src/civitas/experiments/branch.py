"""Counterfactuals inherit history and recompute observations after the branch point."""
from pathlib import Path
from typing import Literal
import uuid

from pydantic import model_validator
from civitas.contracts.base import Record, content_hash
from civitas.contracts.events import Event
from civitas.contracts.observations import ObservationFact
from civitas.contracts.portfolio import FeeSchedule
from civitas.cognition.providers import ProviderConfig
from civitas.scenarios.society_relations import SocietyObserver
from civitas.scenarios.kdd27_config import RQ2Config, RQ2Condition, RQ3Config
from civitas.mechanisms.kdd27 import rq2_parameters
from civitas.storage.commit_store import CommitStore
from .registry import record


class Intervention(Record):
    schema_version: Literal[1] = 1
    kind: Literal['none','messages_off','fees','model'] = 'none'
    fees: FeeSchedule | None = None
    provider: ProviderConfig | None = None

    @model_validator(mode='after')
    def payload(self):
        if (self.kind=='fees') != (self.fees is not None) or (self.kind=='model') != (self.provider is not None):
            raise ValueError('Intervention payload does not match kind')
        return self


class PaperBranchIntervention(Record):
    schema_version: Literal[2] = 2
    kind: Literal['contagion_off','feed_off']


class RQ2BranchIntervention(Record):
    schema_version: Literal[3] = 3
    kind: Literal['rq2'] = 'rq2'
    condition: RQ2Condition


RQ3_FRAMES = {
    'neutral':'The same recorded public facts are presented for consideration.',
    'negative':'Consider the downside implications of the same recorded public facts.',
}


class RQ3BranchIntervention(Record):
    schema_version: Literal[4] = 4
    kind: Literal['rq3'] = 'rq3'
    condition: Literal['neutral','negative']


def parse_intervention(value):
    raw=value.model_dump(mode='json') if isinstance(value,Record) else value
    if raw is None:
        return Intervention()
    kind=raw.get('kind') if isinstance(raw,dict) else None
    model=RQ3BranchIntervention if kind=='rq3' else RQ2BranchIntervention if kind=='rq2' else PaperBranchIntervention if kind in ('contagion_off','feed_off') else Intervention
    return model.model_validate(raw)


def validate_parent(store):
    seen = {store.root}
    lineage = [store]
    while parent := store.identity.metadata.get('parent'):
        source = CommitStore(parent['path'])
        if source.root in seen:
            raise ValueError('Cyclic branch lineage')
        seen.add(source.root)
        markers = source.verified_markers()
        selected = next((r for r in markers if r.tick==parent['tick']),None)
        if (selected is None or selected.commit_hash != parent['commit_hash']
                or content_hash(source.identity) != parent['identity_hash']
                or selected.state_hash != store.markers()[0].state_hash):
            raise ValueError('Parent reference or branch prefix mismatch')
        store = source
        lineage.append(source)
    return lineage


def branch_run(parent, *, tick, destination, intervention=None, allow_cache_reuse=False):
    from civitas.application.persistent_run import validated_snapshot,verify_model_ledger
    source = CommitStore(parent)
    marker, snapshot = validated_snapshot(source,tick=tick)
    intervention = parse_intervention(intervention)
    if isinstance(intervention,RQ2BranchIntervention):
        cfg=RQ2Config.model_validate(source.identity.config)
        if tick!=cfg.event_tick-1 or source.identity.metadata.get('intervention_schedule'):
            raise ValueError('RQ2 arms require the common pristine prefix immediately before the shock')
    if isinstance(intervention,RQ3BranchIntervention):
        cfg=RQ3Config.model_validate(source.identity.config)
        if tick!=cfg.event_tick-1 or source.identity.metadata.get('intervention_schedule'):
            raise ValueError('RQ3 wording arms require their own backend pristine prefix immediately before the event')
    if isinstance(intervention,PaperBranchIntervention) and source.identity.config.get('scenario')!='kdd27_v1':
        raise ValueError('Paper interventions require kdd27_v1')
    if source.identity.config.get('scenario')=='kdd27_v1' and intervention.kind=='messages_off':
        raise ValueError('Use paper feed_off; WP04 messages_off is a different visibility contract')
    identity = source.identity.model_dump(mode='json')
    run_id = Path(destination).name+'-'+uuid.uuid4().hex
    identity.update(run_id=run_id,branch_id=run_id)
    metadata = identity['metadata']
    metadata.update(parent={'path':str(source.root),'identity_hash':content_hash(source.identity),
        'run_id':source.identity.run_id,'tick':tick,'commit_hash':marker.commit_hash},
        intervention=intervention.model_dump(mode='json'),allow_cache_reuse=allow_cache_reuse)
    metadata['intervention_schedule'] = [*metadata.get('intervention_schedule',[])]
    if intervention.kind!='none':
        metadata['intervention_schedule'] = [row for row in metadata['intervention_schedule']
            if not (row['tick']==tick+1 and row['intervention']['kind']==intervention.kind)]
        event_id = 'branch-event:'+content_hash({'parent_state_hash':marker.state_hash,'tick':tick+1,'intervention':intervention})
        metadata['intervention_schedule'].append({'tick':tick+1,'event_id':event_id,
                                                  'intervention':intervention.model_dump(mode='json')})
    if isinstance(intervention,RQ2BranchIntervention):
        metadata['rq2_parameter_changes']=rq2_parameters(cfg,intervention.condition)[1]
        policies={'halt25':{'halted':{'before':False,'after':True,'duration':25}},
            'tax0015':{'seller_tax':{'before':cfg.market.seller_tax,'after':.0015}},
            'same_day_lock':{'same_day_lock':{'before':False,'after':True}},
            'narrow_band005':{'band':{'before':cfg.market.band,'after':.05}}}
        metadata['rq2_policy_changes']=policies.get(intervention.condition,{})
    if intervention.kind == 'model':
        if identity['config']['cognition_mode'] == 'rule':
            raise ValueError('Model intervention requires existing slow cognition actors')
        metadata['provider_config'] = intervention.provider.model_dump(mode='json')
    budget = None
    if identity['config']['cognition_mode']!='rule':
        from civitas.cognition.invocations import ModelInvocationStore
        from civitas.cognition.budget import BudgetController,BudgetLimits
        ledger = ModelInvocationStore(metadata['control_ledger'],existing=True)
        limits = BudgetLimits.model_validate(metadata['budget_limits'])
        if BudgetController.project_limits(ledger)!=limits:
            raise ValueError('Existing invocation ledger project budget differs')
        verify_model_ledger(validate_parent(source),ledger)
        budget = BudgetController(ledger,limits)
    child = CommitStore.initialize(destination,identity,snapshot)
    if budget is not None:
        budget.register_run(run_id,BudgetLimits.model_validate(metadata['budget_limits']))
    blocked = intervention.kind=='model' and intervention.provider.source!='synthetic_test'
    record(child.root,{'status':'blocked' if blocked else 'planned','tick':tick,
        'reason':'no authorized online model budget; WP08 is offline' if blocked else 'new branch awaiting execution',
        'parent_commit_hash':marker.commit_hash,'shared_rng':identity['rng']})
    return child


class BranchObserver(SocietyObserver):
    def __init__(self, rng, window, schedule):
        super().__init__(rng,window)
        self.schedule = schedule
        self._projection = None

    def project(self, snapshot):
        if any(row['intervention']['kind']=='messages_off' and snapshot.tick>=row['tick'] for row in self.schedule):
            if self._projection is None or self._projection[0] is not snapshot:
                raw = snapshot.model_dump(mode='json')['data']['typed_relations']
                raw['config']['attention_budget'] = 0
                self._projection = snapshot, snapshot.model_copy(update={'data':{**snapshot.data,'typed_relations':raw}})
            return self._projection[1]
        return snapshot

    def observe(self, actor_id, snapshot, candidates):
        for row in self.schedule:
            if snapshot.tick==row['tick']:
                candidates = [*candidates,ObservationFact(fact_id=row['event_id']+':'+actor_id,
                    category='market',owner_id=actor_id,visible_at={'tick':snapshot.tick,'phase':'OBSERVE'},
                    value={'event_kind':row['intervention']['kind'],'plan_invalidated':True})]
        return super().observe(actor_id,self.project(snapshot),candidates)

    def prepare_commit(self, snapshot, observations):
        return super().prepare_commit(self.project(snapshot),observations)


def configure_branch(world, store):
    metadata = store.identity.metadata
    if 'parent' not in metadata:
        return
    schedule = metadata.get('intervention_schedule',[])
    if not schedule:
        return
    if world.scenario.scenario_id!='kdd27_v1':
        world.observer = BranchObserver(world.rng,world.observer.window,schedule)
        kinds={'fees'}
    else:
        kinds={'fees','contagion_off','feed_off'}
        for kind,target,attribute in [('contagion_off',world.scenario,'contagion_disabled_from'),
                                     ('feed_off',world.observer,'feed_hidden_from')]:
            ticks=[row['tick'] for row in schedule if row['intervention']['kind']==kind]
            if ticks:
                setattr(target,attribute,min(ticks))
    world.scheduled_events = tuple(Event(event_id=row['event_id'],root_event_id=row['event_id'],tick=row['tick'],phase='EVENTS',
        kind=row['intervention']['kind'],actors=[],visible_at={'tick':row['tick'],'phase':'EVENTS'},
        payload={'public':True,'fees':row['intervention']['fees']} if row['intervention']['kind']=='fees' else
                {'public':True,'contract':row['intervention']})
        for row in schedule if row['intervention']['kind'] in kinds)
    for row in schedule:
        if row['intervention']['kind']=='rq3':
            cfg=world.scenario.config
            if not isinstance(cfg,RQ3Config) or row['tick']!=cfg.event_tick:
                raise ValueError('RQ3 branch identity/window mismatch')
            condition=RQ3BranchIntervention.model_validate(row['intervention']).condition
            event_id='kdd:2:rq3-text'
            world.scheduled_events+=(Event(event_id=event_id,root_event_id=event_id,tick=row['tick'],phase='EVENTS',kind='text',
                actors=[],visible_at={'tick':row['tick'],'phase':'EVENTS'},
                payload={'public':True,'source':'rq3','value':RQ3_FRAMES[condition]}),)
        if row['intervention']['kind']!='rq2':continue
        cfg=world.scenario.config
        if not isinstance(cfg,RQ2Config) or row['tick']!=cfg.event_tick:
            raise ValueError('RQ2 branch identity/window mismatch')
        condition=RQ2BranchIntervention.model_validate(row['intervention']).condition
        world.scenario.rq2_branch=(row['tick'],condition)
        events=[]
        if condition=='halt25':events=[(row['tick'],'halt',None),(row['tick']+25,'resume',None)]
        elif condition=='tax0015':events=[(row['tick'],'tax',.0015)]
        elif condition=='same_day_lock':
            events=[(row['tick'],'lock',True)]
            world.scenario.market.ledger.purchase_lock_schedule=((1,False),(row['tick'],True))
        elif condition=='narrow_band005':events=[(row['tick'],'band',.05)]
        elif condition!='full_control':events=[(row['tick'],'rq2',None)]
        for tick,kind,value in events:
            event_id=f'kdd:2:rq2:{condition}:{tick}'
            world.scheduled_events+= (Event(event_id=event_id,root_event_id=event_id,tick=tick,phase='EVENTS',kind=kind,
                actors=[],visible_at={'tick':tick,'phase':'EVENTS'},payload={'public':True,'source':'rq2','condition':condition,'value':value}),)
