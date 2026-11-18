"""Frozen task inputs and public result records. Evidence has one lower-layer owner."""
from __future__ import annotations

import math
from typing import Literal
from pydantic import Field, model_validator

from core.experiments.strict_types import StrictResearchModel
from core.experiments.identity import research_hash
from core.experiments.evidence import EvidenceRecord, EvidenceScope, ConfirmationPlan, Hash
from core.experiments.design_config import WorldSpec, ResearchMarketConfig, MeasurementSpec
from core.interventions.schema import PolicySpec, PolicyConstraints
from .budget import BudgetSpec


class LossVector(StrictResearchModel):
    L_Q: float = Field(ge=0.0, le=1.0)
    L_S: float = Field(ge=0.0, le=1.0)
    L_D: float = Field(ge=0.0, le=1.0)
    L_E: float = Field(ge=0.0, le=1.0)
    L_B: float = Field(ge=0.0, le=1.0)


class RoleSpec(StrictResearchModel):
    role_id: str = Field(min_length=1)
    kind: Literal['stability', 'liquidity', 'resource']
    private_preference_version: str = Field(min_length=1)
    public_preference_version: str = Field(min_length=1)
    weights: LossVector
    rho: float = Field(ge=0.0, lt=1.0)
    risk_rule: Literal['all_registered_models']
    initial_information_ids: list[str]
    queryable_information_ids: list[str]
    preference_card_id: str
    assent_permission: bool
    decision_rule: Literal['own_absolute_threshold_v1']

    @model_validator(mode='after')
    def _valid(self):
        if abs(math.fsum(self.weights.model_dump().values())-1.0) > 1e-12:
            raise ValueError('Role weights must sum to one')
        if self.kind == 'liquidity' and self.weights.L_D <= 0.0:
            raise ValueError('Liquidity requires positive depth weight')
        if self.preference_card_id not in self.initial_information_ids:
            raise ValueError('Each owner must initially know its preference card')
        return self

    @property
    def preference_hash(self):
        return research_hash(self)


class InformationCard(StrictResearchModel):
    card_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    version: str = Field(min_length=1)
    payload: dict
    readable_by: list[str] = Field(min_length=1)
    disclosable_by: list[str]


class UtilityDefinition(StrictResearchModel):
    version: Literal['cscwd-loss-v1']
    spread_metric: Literal['relative_spread']
    s_cap: float = Field(gt=0.0)
    q_ref: float = Field(gt=0.0)
    e_cap: float = Field(gt=0.0)
    B_task: float = Field(gt=0.0)
    missing_rule: Literal['unavailable-service-loss-one-v1']


class ModelBinding(StrictResearchModel):
    model_id: str = Field(min_length=1)
    family_version: str = Field(min_length=1)
    world_distribution: dict
    scenario_distribution: dict
    behavior: dict
    hidden_f_declared: bool


class OutcomeConstraint(StrictResearchModel):
    constraint_id: str = Field(min_length=1)
    quantity: str = Field(min_length=1)
    direction: Literal['le']
    upper_bound: float = Field(ge=0.0, le=1.0)


class PolicyCandidate(StrictResearchModel):
    candidate_id: str = Field(min_length=1)
    policy: PolicySpec


class AnalyticOutcome(StrictResearchModel):
    """An explicitly finite deterministic problem, never a simulated market sample."""
    candidate_id: str
    model_id: str
    losses: LossVector
    violations: dict[str, float]

    @model_validator(mode='after')
    def _bounds(self):
        if any(not 0.0 <= x <= 1.0 for x in self.violations.values()):
            raise ValueError('Analytic violation quantities require [0,1] support')
        return self


class ModelCallConfig(StrictResearchModel):
    source: Literal['live', 'mock', 'recorded', 'rule']
    provider: Literal['deepseek', 'zhipu', 'gemini']
    model_id: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    thinking: bool
    timeout_seconds: float = Field(gt=0.0)
    max_output_tokens: int = Field(ge=1)
    input_reservation_tokens: int = Field(ge=1)
    max_request_attempts: int = Field(ge=1, le=5)
    model_aliases: dict[str, list[str]]
    adapter_version: Literal['design-agent-v1', 'design-agent-gemini-native-v1']
    provider_version: Literal['civitas-router-v1', 'gemini-native-v1']
    sdk_version: str = Field(min_length=1)
    token_accounting: Literal['reserved-cap-no-refund-v1']

    @model_validator(mode='after')
    def _provider(self):
        if self.provider == 'gemini':
            if not self.model_id.startswith('gemini-'):
                raise ValueError('Fixed Gemini provider/model must agree')
            if (self.adapter_version, self.provider_version) != ('design-agent-gemini-native-v1', 'gemini-native-v1'):
                raise ValueError('Gemini requires the registered native adapter versions')
            if not self.thinking:
                raise ValueError('Gemini native v1 registers thinking LOW explicitly')
        elif (self.adapter_version, self.provider_version) != ('design-agent-v1', 'civitas-router-v1'):
            raise ValueError('Existing providers retain their registered adapter versions')
        if (self.provider == 'zhipu') != self.model_id.startswith('glm-'):
            raise ValueError('Fixed provider/model must agree with existing router routing')
        if set(self.model_aliases) - {self.model_id}:
            raise ValueError('Aliases must be explicitly bound to the requested full model ID')
        return self


class TaskSpec(StrictResearchModel):
    schema_version: Literal[1]
    task_id: str = Field(min_length=1)
    underlying_cluster_id: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    split: Literal['development', 'validation', 'test', 'ood', 'diagnostic']
    origin: Literal['analytic_fixture', 'market']
    candidate_language: Literal['finite-policy-compiler-v1']
    candidate_pool: list[PolicyCandidate] = Field(min_length=1)
    fixed_rule_candidate_id: str
    compiler_world: WorldSpec
    compiler_market: ResearchMarketConfig
    static_constraints: PolicyConstraints
    model_family: list[ModelBinding] = Field(min_length=1)
    horizon_steps: int = Field(ge=1)
    measurement: MeasurementSpec
    utility: UtilityDefinition
    roles: list[RoleSpec] = Field(min_length=1)
    required_signers: list[str] = Field(min_length=1)
    outcome_constraints: list[OutcomeConstraint]
    information_cards: list[InformationCard]
    method_initial_information_ids: list[str]
    method_queryable_information_ids: list[str]
    delegated_role_ids: list[str]
    evidence_mode: Literal['exact_analytic', 'fresh_fixed_batch', 'diagnostic_only']
    plan_family_version: str = Field(min_length=1)
    analytic_outcomes: list[AnalyticOutcome]
    budget: BudgetSpec
    termination: Literal['finite_pool_or_budget_v1']
    final_selection: Literal['max_min_margin_cost_complexity_hash_v1']
    signing_layer: Literal['explicit_rule_owners_v1', 'delegated_explicit_llm_v1', 'candidate_only']
    model_configurations: list[ModelCallConfig]
    prompt_resources: dict[str, str]
    support: Literal['losses-utilities-violations-unit-interval-v1']

    @model_validator(mode='after')
    def _cross_fields(self):
        groups = [([p.candidate_id for p in self.candidate_pool], 'candidate'),
                  ([r.role_id for r in self.roles], 'role'),
                  ([m.model_id for m in self.model_family], 'model'),
                  ([c.card_id for c in self.information_cards], 'card'),
                  ([c.constraint_id for c in self.outcome_constraints], 'constraint')]
        for ids, name in groups:
            if len(set(ids)) != len(ids): raise ValueError('Duplicate '+name)
        candidates, roles, models, cards, constraints = [set(ids) for ids, _ in groups]
        if self.fixed_rule_candidate_id not in candidates: raise ValueError('Unknown fixed rule')
        if (not set(self.required_signers) <= roles or len(set(self.required_signers)) != len(self.required_signers)
                or not set(self.delegated_role_ids) <= set(self.required_signers)):
            raise ValueError('Invalid signing permissions')
        if any(not r.assent_permission for r in self.roles if r.role_id in self.required_signers):
            raise ValueError('Required signer lacks permission')
        if self.horizon_steps != self.static_constraints.horizon or self.measurement.evaluation_end > self.horizon_steps+1:
            raise ValueError('Task clock mismatch')
        if any(r.weights.L_E > 0.0 for r in self.roles) and any(not m.hidden_f_declared for m in self.model_family):
            raise ValueError('Price discovery loss requires declared hidden F in every model')
        card_map = {c.card_id:c for c in self.information_cards}
        for principal, initial, queryable in [('designer', self.method_initial_information_ids, self.method_queryable_information_ids)] + [
                (r.role_id, r.initial_information_ids, r.queryable_information_ids) for r in self.roles]:
            if len(set(initial)) != len(initial) or len(set(queryable)) != len(queryable): raise ValueError('Duplicate information entitlement')
            if not set(initial+queryable) <= cards: raise ValueError('Unknown information card')
            if any(principal not in card_map[c].readable_by for c in initial+queryable): raise ValueError('Information permission mismatch')
        for role in self.roles:
            card = card_map[role.preference_card_id]
            if card.owner != role.role_id or card.payload != {'weights':role.weights.model_dump(), 'rho':role.rho,
                    'private_preference_version':role.private_preference_version}:
                raise ValueError('Preference card must match its owner and actual registered preferences')
        pairs = [(a.candidate_id,a.model_id) for a in self.analytic_outcomes]
        if self.evidence_mode == 'exact_analytic':
            if self.origin != 'analytic_fixture' or set(pairs) != {(c,m) for c in candidates for m in models} or len(pairs) != len(set(pairs)):
                raise ValueError('Exact analytic mode needs the complete finite problem')
            if any(set(a.violations) != constraints for a in self.analytic_outcomes): raise ValueError('Analytic constraint coverage incomplete')
        elif self.analytic_outcomes:
            raise ValueError('Market tasks cannot embed an analytic outcome table')
        return self

    @property
    def task_hash(self): return research_hash(self)

    @property
    def preference_hash(self): return research_hash(self.roles)

    @property
    def acceptance_rule_hash(self):
        return research_hash({'roles':self.roles,'constraints':self.outcome_constraints,'utility':self.utility,
                              'selection':self.final_selection, 'signers':self.required_signers})


class AssentReceipt(StrictResearchModel):
    receipt_id: Hash
    episode_id: str
    owner: str
    policy_hash: Hash
    task_hash: Hash
    preference_hash: Hash
    acceptance_rule_hash: Hash
    decision: Literal['ASSENT', 'DECLINE', 'UNRESOLVED']
    source: Literal['rule', 'live', 'mock', 'recorded']
    decision_rule: str
    evidence_ids: list[Hash]
    information_receipt_ids: list[Hash]
    raw_decision_ref: str
    sequence: int = Field(ge=1)

    @model_validator(mode='after')
    def _hash(self):
        if self.receipt_id != research_hash(self.model_dump(mode='json',exclude={'receipt_id'})):
            raise ValueError('Assent identity mismatch')
        return self


class EpisodeResult(StrictResearchModel):
    schema_version: Literal[1] = 1
    episode_id: str
    task_id: str
    underlying_cluster_id: str
    task_hash: Hash
    method: str
    method_config_hash: Hash
    source_hash: Hash
    split: str
    origin: Literal['analytic_fixture', 'market']
    live_status: Literal['NOT_RUN', 'RUN']
    candidate_refs: list[str]
    evidence_refs: list[str]
    message_refs: list[str]
    A_raw: list[Hash]
    A_method: list[Hash]
    A_cert: list[Hash]
    role_eligibility: dict
    assent_refs: list[str]
    selected_policy_hash: Hash | None
    shortlist: list[Hash]
    terminal_state: Literal['SELECTED', 'CANDIDATE_ONLY', 'SHORTLIST', 'BUDGET_UNRESOLVED',
                            'NO_ACCEPTABLE_IN_FINITE_POOL', 'WORLD_INVALID', 'METHOD_FAILED',
                            'AGREEMENT_CERTIFIED','SHORTLIST_WITH_DISSENT','BUDGET_EXHAUSTED_UNRESOLVED',
                            'NO_ACCEPTABLE_POLICY_IN_ENUMERATED_SET','INVALID_WORLD']
    budget_snapshot: dict
    failures: list[dict]
    language_error_rate: float | None
    language_metric_status: Literal['NA_RULE', 'OBSERVED', 'NO_RESPONSES']
    started_at: float
    ended_at: float
    common_contract: dict


class ClaimSpec(StrictResearchModel):
    claim_id: str = Field(min_length=1)
    kind: Literal['static_feasible','absolute_utility','violation_probability','paired_difference','model_conjunction']
    policy_hashes: list[Hash] = Field(min_length=1,max_length=2)
    model_id: str
    quantity_id: str
    direction: Literal['ge','le']
    threshold: float
    scope: EvidenceScope
    evidence_mode: Literal['exact_analytic','fixed_pool_replay','fresh_fixed_batch']
    registered_plan_hash: Hash
    dependencies: list[str]

    @model_validator(mode='after')
    def _scope(self):
        if self.policy_hashes!=self.scope.policy_hashes:raise ValueError('Claim policy scope mismatch')
        if len(set(self.policy_hashes))!=len(self.policy_hashes):raise ValueError('Duplicate claim policy')
        if (self.kind=='paired_difference')!=(len(self.policy_hashes)==2):raise ValueError('Ordered pair only for paired difference')
        if self.claim_id in self.dependencies or len(set(self.dependencies))!=len(self.dependencies):raise ValueError('Invalid claim dependency')
        return self


class Objection(StrictResearchModel):
    objection_id: str = Field(min_length=1)
    owner: str
    kind: Literal['empirical','objective','structural_model','scope_measurement']
    policy_hash: Hash
    scope: EvidenceScope
    reason: str = Field(min_length=1)
    claim_ids: list[str]
    status: Literal['OPEN','RESOLVED_BY_EVIDENCE','WITHDRAWN','INAPPLICABLE']


class Commitment(StrictResearchModel):
    owner: str
    policy_hash: Hash
    scope: EvidenceScope
    preference_hash: Hash
    acceptance_rule_hash: Hash
    eligible: Literal['SUPPORTED','REFUTED','UNRESOLVED','INAPPLICABLE','INVALID_ARTIFACT']
    explicit_assent_receipt_id: Hash
    raw_ref: str
    support_ids: list[Hash]
    sequence: int = Field(ge=1)
    current_status: Literal['DECLARED','WITHDRAWN','INAPPLICABLE']
    status_reason: str


class ProposePayload(StrictResearchModel):
    candidate: PolicyCandidate


class ClaimPayload(StrictResearchModel):
    claim: ClaimSpec


class DisclosePayload(StrictResearchModel):
    information_id: str
    information_receipt_id: Hash
    recipients: list[str] = Field(min_length=1)


class ChallengePayload(StrictResearchModel):
    objections: list[Objection] = Field(min_length=1)


class TestRequestPayload(StrictResearchModel):
    candidate_id: str
    claim_ids: list[str] = Field(min_length=1)


class TestResultPayload(StrictResearchModel):
    request_operation_id: str
    candidate_id: str
    evidence_ids: list[Hash] = Field(min_length=1)
    batch_receipt_hash: Hash


class RevisePayload(StrictResearchModel):
    parent_policy_hash: Hash
    candidate: PolicyCandidate


class FactPayload(StrictResearchModel):
    claim_id: str
    verdict: Literal['SUPPORTED','REFUTED','UNRESOLVED','INAPPLICABLE','INVALID_ARTIFACT']
    evidence_ids: list[Hash] = Field(min_length=1)


class AssentPayload(StrictResearchModel):
    owner: str
    policy_hash: Hash
    receipt_id: Hash


class DissentPayload(StrictResearchModel):
    owner: str
    policy_hash: Hash
    reason: str = Field(min_length=1)
    objection_ids: list[str]


class StopPayload(StrictResearchModel):
    reason: Literal['completed_round','budget','enumerated_pool','abstain','world_invalid']


MESSAGE_PAYLOADS={'PROPOSE':ProposePayload,'REGISTER_CLAIM':ClaimPayload,'DISCLOSE':DisclosePayload,
    'CHALLENGE':ChallengePayload,'REQUEST_TEST':TestRequestPayload,'TEST_RESULT':TestResultPayload,
    'REVISE':RevisePayload,'ACKNOWLEDGE_FACT':FactPayload,'DECLARE_ASSENT':AssentPayload,
    'DECLARE_DISSENT':DissentPayload,'STOP':StopPayload}


class AgentMessage(StrictResearchModel):
    message_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    episode_id: str
    round: int = Field(ge=1)
    sequence: int = Field(ge=1)
    speaker: str
    type: Literal['PROPOSE','REGISTER_CLAIM','DISCLOSE','CHALLENGE','REQUEST_TEST','TEST_RESULT','REVISE',
                  'ACKNOWLEDGE_FACT','DECLARE_ASSENT','DECLARE_DISSENT','STOP']
    payload: dict
    source: Literal['live','rule','mock','recorded']
    raw_ref: str
    raw_text: str
    request_hash: Hash
    budget_receipt_hash: Hash

    @model_validator(mode='after')
    def _payload(self):
        MESSAGE_PAYLOADS[self.type].model_validate(self.payload)
        return self

    @property
    def parsed_payload(self):return MESSAGE_PAYLOADS[self.type].model_validate(self.payload)

    @property
    def operation_fingerprint(self):
        return research_hash({'episode_id':self.episode_id,'speaker':self.speaker,'type':self.type,'payload':self.payload})


class CDRConfig(StrictResearchModel):
    separate_fact_assent: bool
    maintain_scope_support: bool


class SupportJustification(StrictResearchModel):
    justification_id: str = Field(min_length=1)
    conclusion_claim_id: str = Field(min_length=1)
    rule: Literal['registered_evidence','positive_conjunction']
    premise_claim_ids: list[str]
    evidence_ids: list[Hash]
    scope: EvidenceScope

    @model_validator(mode='after')
    def _finite(self):
        if len(set(self.premise_claim_ids))!=len(self.premise_claim_ids) or len(set(self.evidence_ids))!=len(self.evidence_ids):
            raise ValueError('DUPLICATE_SUPPORT_REFERENCE')
        if self.conclusion_claim_id in self.premise_claim_ids:raise ValueError('CYCLIC_SUPPORT')
        if self.rule=='registered_evidence' and (self.premise_claim_ids or not self.evidence_ids):raise ValueError('EVIDENCE_SUPPORT_SHAPE')
        if self.rule=='positive_conjunction' and (not self.premise_claim_ids or self.evidence_ids):raise ValueError('CONJUNCTION_SUPPORT_SHAPE')
        return self
