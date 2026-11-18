"""Finite prompt memory and selected decision traces inside PreparedTick."""
from pydantic import Field
from civitas.contracts.base import Record
from civitas.contracts.observations import Observation
from civitas.contracts.financial import FinancialDecisionBundle
from civitas.contracts.portfolio import PortfolioDecisionBundle
from civitas.cognition.memory import MemoryBook
from civitas.cognition.policy import HybridBeliefPolicy
from civitas.relations.types import TypedMessageContent


def _closed_object(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def _hybrid_response_schema(context, observation):
    fields = {name:{'type':'integer' if type(value) is int else 'string','enum':[value]}
        for name,value in context['identity'].items() if type(value) in (int,str)}
    evidence = sorted(observation.evidence_ids)
    evidence_schema = {'type':'array','items':{'type':'string','enum':evidence}} if evidence else {
        'type':'array','items':{'type':'string'},'maxItems':0}
    probabilities = _closed_object({name:{'type':'number','minimum':0,'maximum':1} for name in context['scenarios']})
    belief = {'evidence_ids':evidence_schema,
        **{name:{'type':'integer','enum':[value]} for name,value in context['required_belief_times'].items()}}
    fields['belief_patch'] = _closed_object({'joint_belief':{'anyOf':[
        _closed_object({**belief,'abstain':{'type':'boolean','enum':[False]},'probabilities':probabilities}),
        _closed_object({**belief,'abstain':{'type':'boolean','enum':[True]},'probabilities':_closed_object({})})]}})
    fields['economic_intents'] = {'type':'array','items':{'type':'string'},'maxItems':0}
    message = TypedMessageContent.model_json_schema()['properties']
    message = {name:{k:v for k,v in spec.items() if k!='default'} for name,spec in message.items()}
    fields['message_drafts'] = {'type':'array','items':_closed_object({
        'message_id':{'type':'string','minLength':1},'recipient_id':{'type':'string','minLength':1},
        'body':_closed_object(message)})}
    fields['evidence_ids'] = evidence_schema
    return _closed_object(fields)


class RecentMemoryBook(MemoryBook):
    def __init__(self, actors, window):
        super().__init__(actors)
        self.window = window

    def accept(self, observation, **kwargs):
        value = super().accept(observation,**kwargs)
        pinned = {e for note in (*value.semantic,*value.procedural) for e in note.evidence_ids}
        episodes = [e for e in value.episodic if e.seen_at>observation.visible_until.tick-self.window
            or e.source=='message' or e.evidence_id in pinned or e.evidence_id.startswith(('wp05-event:','society:clarification'))]
        self._actors[value.actor_id] = value.model_copy(update={'episodic':episodes})
        return self._actors[value.actor_id]


class SocietyHybrid(HybridBeliefPolicy):
    async def _call(self, observation, task, response_model, validator, context):
        context = {**context,'memory_prompt_projection':'episode_index_v2',
            'observation_prompt_projection':'audience_sidecar_v1',
            'instruction':context['instruction']+' observation_audience_refs maps fact IDs to observation_audiences '
                'indexes for omitted value.visible_to lists; restore only those fields. Other values are literal.'}
        if task=='reflection' and self.runtime.config.provider=='openai':
            schema = response_model.model_json_schema()
            fields = schema['properties']
            for name in ('actor_id','observation_id'):
                fields[name] = {'type':'string','enum':[getattr(observation,name)]}
            evidence = sorted(observation.evidence_ids)
            fields['evidence_ids'] = {'type':'array','items':{'type':'string','enum':evidence}} if evidence else {
                'type':'array','items':{'type':'string'},'maxItems':0}
            for name in ('semantic','procedural'):
                fields[name] = {'type':'array','items':{'type':'string','pattern':r'[\s\S]+'},'maxItems':16}
            schema['required'] = list(fields)
            context = {**context,'strict_response_schema':schema}
        if task=='hybrid':
            tick = observation.visible_until.tick
            context = {**context,'plan_validity_ticks':self.config.period,
                'required_belief_times':{'as_of_tick':tick,'horizon_tick':tick+self.config.max_horizon,
                    'valid_until_tick':tick+self.config.period-1},
                'allowed_evidence_ids':sorted(observation.evidence_ids),
                'instruction':context['instruction']+' Use outcome_contract.horizon_ticks for horizon_tick-as_of_tick. '
                'Copy required_belief_times exactly; memory contains past plans, not current timestamps. '
                'Cite only allowed_evidence_ids in both evidence lists. '
                'Include top-level evidence_ids, copying every joint_belief.evidence_ids entry into that list. '
                'Include every field of identity even when its value is empty or a schema default. Output compact JSON only.'}
            if self.runtime.config.provider=='openai':
                context = {**context,'strict_response_schema':_hybrid_response_schema(context,observation)}
        return await super()._call(observation,task,response_model,validator,context)


class DecisionTrace(Record):
    observation: Observation
    decision: FinancialDecisionBundle | PortfolioDecisionBundle
    numeric_trace: dict
    next_observation: Observation | None = None


class TraceState(Record):
    records: list[DecisionTrace] = Field(default_factory=list)


class TraceBook:
    participant_id, state_schema_version = 'society_traces', 1

    def __init__(self, config, actors):
        news = config.news
        self.ticks = {1,news.delivery_tick,news.delivery_tick+1,news.clarification_tick,news.clarification_tick+1} if news else {1,2,3,4,5}
        self.actors = set(actors[:2])
        self.state = TraceState()

    def capture(self, observation, decision, numeric_trace):
        if observation.actor_id not in self.actors:
            return
        if observation.visible_until.tick not in self.ticks and not any(r.next_observation is None
                and r.observation.actor_id==observation.actor_id for r in self.state.records):
            return
        rows = [r.model_copy(update={'next_observation':observation}) if r.next_observation is None
            and r.observation.actor_id==observation.actor_id and r.observation.visible_until.tick+1==observation.visible_until.tick else r
            for r in self.state.records]
        if observation.actor_id in self.actors and observation.visible_until.tick in self.ticks:
            rows.append(DecisionTrace(observation=observation,decision=decision,numeric_trace=numeric_trace))
        self.state = TraceState(records=sorted(rows,key=lambda r:(r.observation.visible_until.tick,r.observation.actor_id)))

    def export_state(self):
        return self.state.model_dump()

    def import_state(self, state):
        self.state = TraceState.model_validate(state)

    def validate_state(self):
        self.import_state(self.export_state())
