"""Scheduled synthetic claims and public resolutions, using WP04 scoring unchanged."""
from civitas.contracts.config import MechanismSpec
from civitas.contracts.decisions import MessageDraft
from civitas.contracts.observations import ObservationFact
from civitas.mechanisms.influence import InfluenceObservationBuilder
from civitas.relations.integration import FIELD, RelationObserver, active_relations, project_relations
from civitas.relations.types import (CommunicationEdge, FactTrust, PredictionTrust, Group, Membership,
                                     RelationState, TrustEvidence, TypedMessageContent)


def pairs(config, actors):
    size = config.news.community_size if config.news else 0
    return list(zip(actors[size:2*size],actors[:size],strict=True))


def initial_relations(config, actors):
    edges = {(actor,actors[(i+j+1)%len(actors)]) for i,actor in enumerate(actors) for j in range(config.communication_neighbors)}
    edges.update(pairs(config,actors))
    return RelationState(actors=actors,config=config.relations,
        groups=[Group(group_id='news-community')],
        memberships=[Membership(source_id=target,target_id='news-community',visibility='members') for _,target in pairs(config,actors)],
        communication=[CommunicationEdge(source_id=a,target_id=b) for a,b in sorted(edges)],
        fact_trust=[FactTrust(source_id=b,target_id=a) for a,b in pairs(config,actors)],
        prediction_trust=[PredictionTrust(source_id=b,target_id=a) for a,b in pairs(config,actors)])


def claim(config, author, target, kind):
    news = config.news
    probability = sum(news.probabilities[s.scenario_id] for s in config.behavior.joint_outcomes.scenarios if s.asset_returns[news.asset_id]>0.0)
    root = f'society:{kind}:{author}'
    return TypedMessageContent(root_claim_id=root,author_id=author,audience=[target],content_ref='content:'+root,
        claim_kind=kind,signal=1.0,probability=probability if kind=='prediction' else None)


def scheduled_drafts(config, actors, actor, tick):
    if config.news is None:
        return []
    news = config.news
    for author,target in pairs(config,actors):
        if actor!=author:
            continue
        for kind,delivery in [('fact',news.delivery_tick),('prediction',news.delivery_tick+news.forecast_delay)]:
            if tick==delivery-1:
                body = claim(config,author,target,kind)
                return [MessageDraft(message_id='message:'+body.root_claim_id,recipient_id=target,body=body.model_dump())]
    return []


def resolutions(config, actors, state, tick):
    if config.news is None:
        return
    news = config.news
    forecast_tick = news.delivery_tick+news.forecast_delay
    price_key = 'last_price' if config.backend=='legacy_float64' else 'last_price_ticks'
    price = state['quotes'][news.asset_id][price_key]
    if tick==forecast_tick:
        state['forecast_anchors'][news.asset_id] = price
    for author,target in pairs(config,actors):
        for kind,due in [('fact',news.clarification_tick),('prediction',forecast_tick+news.forecast_horizon)]:
            if tick!=due:
                continue
            body = claim(config,author,target,kind)
            outcome = news.fact_outcome if kind=='fact' else int(price>state['forecast_anchors'][news.asset_id])
            state['resolutions'].append(TrustEvidence(evidence_id='resolution:'+body.root_claim_id,
                root_claim_id=body.root_claim_id,author_id=author,observer_id=target,kind=kind,outcome=outcome,
                probability=body.probability,matured_tick=tick,visible_tick=tick,visible_to=actors,
                origin='public_resolution').model_dump())


def resolved_relations(data, tick):
    state = active_relations(data,tick)
    existing = {e.evidence_id:e for e in state.evidence}
    for raw in data['society']['resolutions']:
        evidence = TrustEvidence.model_validate(raw)
        if evidence.evidence_id in existing and existing[evidence.evidence_id]!=evidence:
            raise ValueError('Conflicting public resolution')
        existing[evidence.evidence_id] = evidence
    return state.model_copy(update={'evidence':list(existing.values())})


def information_facts(config, actors, context, actor, relations):
    tick = context.tick
    now = {'tick':tick,'phase':'OBSERVE'}
    result = project_relations(relations,actor,tick)
    if config.news is None:
        return result
    news = config.news
    for author,target in pairs(config,actors):
        for kind,delivery in [('fact',news.delivery_tick),('prediction',news.delivery_tick+news.forecast_delay)]:
            if actor==target and tick>=delivery:
                body = claim(config,author,target,kind)
                result.append(ObservationFact(fact_id=body.content_ref,category='market',owner_id=actor,visible_at={'tick':delivery,'phase':'OBSERVE'},
                    value={'claim_origin':'scheduled_synthetic','asset_id':news.asset_id,
                        'joint_opinion':{'proposition_id':config.behavior.influence.proposition_id,
                            'horizon_ticks':config.behavior.joint_outcomes.scenarios[0].horizon_ticks,
                            'probabilities':news.probabilities},
                        'pending_prediction':{'root_claim_id':body.root_claim_id,'probability':body.probability,
                            'due_tick':delivery+news.forecast_horizon} if kind=='prediction' and tick<delivery+news.forecast_horizon else None}))
    if tick>=news.clarification_tick:
        result.append(ObservationFact(fact_id='society:clarification',category='market',public=True,
            visible_at={'tick':news.clarification_tick,'phase':'OBSERVE'},
            value={'event_kind':'public_clarification','asset_id':news.asset_id,'fact_outcome':news.fact_outcome,
                'data_origin':'synthetic_scenario','joint_opinion':{'proposition_id':config.behavior.influence.proposition_id,
                    'horizon_ticks':config.behavior.joint_outcomes.scenarios[0].horizon_ticks,
                    'probabilities':{s.scenario_id:s.probability for s in config.behavior.joint_outcomes.scenarios}}}))
    return result


class SocietyRelationObserver(RelationObserver):
    effect_spec = MechanismSpec(mechanism_id='typed_relations_feedback',version='1',phase='FEEDBACK',
        reads=[FIELD,'society'],writes=[FIELD])

    def prepare_commit(self, snapshot, observations):
        state = resolved_relations(snapshot.data,snapshot.tick)
        return super().prepare_commit(snapshot.model_copy(update={'data':{**snapshot.data,FIELD:state.model_dump()}}),observations)


class SocietyObserver(InfluenceObservationBuilder):
    compact_snapshots = True
    def __init__(self, rng, window):
        super().__init__(SocietyRelationObserver(rng))
        self.window = window
        self._view = None

    def select_messages(self, actor_id, snapshot, messages):
        from civitas.relations.messages import select_messages
        raw = snapshot.data[FIELD]
        if self._view is None or self._view[0] is not raw or self._view[1]!=snapshot.tick or self._view[2] is not snapshot.messages:
            self._view = (raw,snapshot.tick,snapshot.messages,active_relations(snapshot.data,snapshot.tick),{})
        selected = self._view[4]
        if actor_id not in selected:
            selected[actor_id] = select_messages(self._view[3],messages,actor_id,snapshot.tick)
        return selected[actor_id]

    def observe(self, actor_id, snapshot, candidates):
        from civitas.cognition.observations import CognitionObservationBuilder
        from civitas.mechanisms.influence import MessageWeights
        from civitas.relations.messages import message_weight
        delivered = self.select_messages(actor_id,snapshot,snapshot.messages)
        state = self._view[3]
        witnessed = {e.message_id for e in state.exposures if e.actor_id==actor_id}
        witnessed.update(m.message_id for m in delivered)
        content = {m.body['content_ref'] for m in snapshot.messages if m.message_id in witnessed}
        weights = MessageWeights(weights={m.message_id:message_weight(state,TypedMessageContent.model_validate(m.body),actor_id) for m in delivered})
        fact = ObservationFact(fact_id='wp06-message-weights:'+actor_id,category='market',owner_id=actor_id,
            visible_at={'tick':snapshot.tick,'phase':'OBSERVE'},value=weights.model_dump())
        return CognitionObservationBuilder.observe(self,actor_id,snapshot,[fact,*[f for f in candidates
            if not f.fact_id.startswith('content:society:') or f.fact_id in content]])

    def select_memories(self, actor_id, snapshot):
        now = snapshot.tick
        own = [f for f in snapshot.memories if f.owner_id==actor_id and f.visible_at.tick<=now]
        beliefs = [f for f in own if f.fact_id.startswith('belief:')]
        recent = [f for f in own if not f.fact_id.startswith('belief:') and f.visible_at.tick>now-self.window]
        return recent+([max(beliefs,key=lambda f:(f.visible_at.key,f.fact_id))] if beliefs else [])
