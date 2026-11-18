"""Deterministic public schemas; semantic invariants also run in Python validators."""
from civitas.contracts.config import MechanismSpec, RunConfig
from civitas.contracts.decisions import Command, DecisionBundle
from civitas.contracts.events import Event
from civitas.contracts.observations import AccountView, Observation
from civitas.contracts.receipts import ExecutionReceipt, ModelCallReceipt
from civitas.contracts.run import RunManifest
from civitas.contracts.run import FinancialRunManifest
from civitas.contracts.financial import (CancelIntent, FinancialCommand, FinancialDecisionBundle, FinancialExecutionReceipt,
                                        LegacyAccountView, LegacyOrderBatch, LegacySettlementState, OrderEvent, OrderIntent)
from civitas.contracts.legacy_config import LegacyRunConfig
from civitas.contracts.state import FinancialPreparedTick, FinancialWorldSnapshot
from civitas.analytics.market_v2 import MarketMetricsV2
from civitas.scenarios.legacy_state import LegacySocietyState
from civitas.contracts.state import PreparedTick, WorldSnapshot
from civitas.contracts.time import SimTime
from civitas.core.rng import DrawAddress


PUBLIC_MODELS = (SimTime, Event, AccountView, Observation, DecisionBundle, Command, ExecutionReceipt,
                 ModelCallReceipt, RunConfig, MechanismSpec, WorldSnapshot, PreparedTick, RunManifest, DrawAddress,
                 OrderIntent, CancelIntent, FinancialCommand, FinancialDecisionBundle, FinancialExecutionReceipt,
                 LegacyAccountView, LegacyOrderBatch, LegacySettlementState, OrderEvent, LegacyRunConfig,
                 FinancialPreparedTick, FinancialWorldSnapshot, FinancialRunManifest, LegacySocietyState, MarketMetricsV2)


def schemas() -> dict:
    return {model.__name__: model.model_json_schema() for model in PUBLIC_MODELS}


from civitas.contracts.portfolio import (AssetSpec, Rate, FeeSchedule, AccountState, PortfolioAccountView,
    ReserveCommand, Reservation, Posting, PortfolioTrade, LedgerTransaction, PortfolioLedgerState, PortfolioOrderIntent,
    PortfolioDecisionBundle, PortfolioCommand, PortfolioExecutionReceipt, PortfolioOrder, PortfolioOrderEvent,
    MatchingVenueState, MultiAssetState)
from civitas.contracts.population import WealthPrior, PopulationSpec, Profile, AllocationEntry, PopulationInitialization
from civitas.contracts.multiasset_config import FeeChange, MultiAssetRunConfig
from civitas.contracts.state import PortfolioWorldSnapshot, PortfolioPreparedTick
from civitas.contracts.run import PortfolioRunManifest

PUBLIC_MODELS += (AssetSpec, Rate, FeeSchedule, AccountState, PortfolioAccountView, ReserveCommand, Reservation, Posting,
    PortfolioTrade, LedgerTransaction, PortfolioLedgerState, PortfolioOrderIntent, PortfolioDecisionBundle, PortfolioCommand,
    PortfolioExecutionReceipt, PortfolioOrder, PortfolioOrderEvent, MatchingVenueState, MultiAssetState, WealthPrior,
    PopulationSpec, Profile, AllocationEntry, PopulationInitialization, FeeChange, MultiAssetRunConfig,
    PortfolioWorldSnapshot, PortfolioPreparedTick, PortfolioRunManifest)

from civitas.relations.types import (CommunicationEdge, FactTrust, PredictionTrust, TrustEvidence, Group, Membership,
    RewiringConfig, RelationsConfig, PerceivedSimilarity, TypedMessageContent, ExposureEvent, ScoredClaim, RewireTrace,
    EdgeDelta, RelationState)
from civitas.relations.exposure import HoldingSnapshot, HoldingDisclosure, TradeLink
from civitas.relations.rewiring import RewirePlan

PUBLIC_MODELS += (CommunicationEdge, FactTrust, PredictionTrust, TrustEvidence, Group, Membership,
    RewiringConfig, RelationsConfig, PerceivedSimilarity, TypedMessageContent, ExposureEvent, ScoredClaim, RewireTrace,
    EdgeDelta, RelationState, HoldingSnapshot, HoldingDisclosure, TradeLink, RewirePlan)

from civitas.cognition.beliefs import JointBelief
from civitas.cognition.budget import BudgetLimits, Price
from civitas.cognition.memory import AgentMemory, MemoryEpisode, MemoryNote, MemoryState
from civitas.cognition.policy import PolicyConfig
from civitas.cognition.providers import ProviderConfig, ProviderResponse
from civitas.cognition.reflection import ReflectionCandidate

PUBLIC_MODELS += (JointBelief, BudgetLimits, Price, AgentMemory, MemoryEpisode, MemoryNote, MemoryState,
                  PolicyConfig, ProviderConfig, ProviderResponse, ReflectionCandidate)

from civitas.mechanisms.outcomes import (ExactAmount, PriceAnchor, AccountingCost, JointOutcome, JointOutcomes,
    CandidateAction, OutcomeWealth, Projection)
from civitas.mechanisms.behavior import (ReferenceWealth, BehaviorParameters, BehaviorConfig, ValueComponent,
    ActionScore, DecisionScores, ActionChoice)
from civitas.mechanisms.norms import NormalizedAction, GroupNorm
from civitas.mechanisms.registry import BeliefOwner
from civitas.mechanisms.influence import Opinion, InfluenceConfig, InfluenceResult, MessageWeights
from civitas.mechanisms.games import (PublicGoodsConfig, ContributionChoice, PublicGroupHistory, LearningParameters,
    LearningState, ObservedGameReturn, ChosenActionUpdate)

PUBLIC_MODELS += (ExactAmount, PriceAnchor, AccountingCost, JointOutcome, JointOutcomes, CandidateAction,
    OutcomeWealth, Projection, ReferenceWealth, BehaviorParameters, BehaviorConfig, ValueComponent, ActionScore,
    DecisionScores, ActionChoice, NormalizedAction, GroupNorm, BeliefOwner, Opinion, InfluenceConfig,
    InfluenceResult, MessageWeights, PublicGoodsConfig, ContributionChoice, PublicGroupHistory,
    LearningParameters, LearningState, ObservedGameReturn, ChosenActionUpdate)

from civitas.scenarios.society_config import SocietyNews, SocietyIntervention, MarketSocietyConfig
from civitas.scenarios.society_cognition import DecisionTrace, TraceState

PUBLIC_MODELS += (SocietyNews, SocietyIntervention, MarketSocietyConfig, DecisionTrace, TraceState)

from civitas.storage.commit_store import StateChange, StateDelta, Segment, StoreIdentity, Commit
from civitas.storage.checkpoint import Checkpoint
from civitas.experiments.branch import Intervention
from civitas.experiments.batch import BatchSpec

PUBLIC_MODELS += (StateChange, StateDelta, Segment, StoreIdentity, Commit, Checkpoint, Intervention, BatchSpec)

from civitas.scenarios.kdd27_config import KDD27Config
from civitas.scenarios.kdd27 import PaperAgentState, PaperWorldState
from civitas.cognition.kdd27_policy import StrategyCache

PUBLIC_MODELS += (KDD27Config, PaperAgentState, PaperWorldState, StrategyCache)

from civitas.experiments.branch import PaperBranchIntervention
from civitas.experiments.batch import PaperBatchSpec

PUBLIC_MODELS += (PaperBranchIntervention, PaperBatchSpec)

from civitas.analytics.kdd27 import PaperMetric
from civitas.analytics.kdd27_data import MetricSpec, RunEvidence

PUBLIC_MODELS += (PaperMetric, MetricSpec, RunEvidence)

from civitas.scenarios.kdd27_config import RQ1Config
from civitas.experiments.batch import RQ1BatchSpec

PUBLIC_MODELS += (RQ1Config, RQ1BatchSpec)

from civitas.scenarios.kdd27_config import RQ2Config
from civitas.experiments.batch import RQ2BatchSpec
from civitas.experiments.branch import RQ2BranchIntervention
PUBLIC_MODELS += (RQ2Config, RQ2BatchSpec, RQ2BranchIntervention)

from civitas.scenarios.kdd27_config import PaperCognitionConfig
from civitas.experiments.batch import PaperCognitionBatchSpec
PUBLIC_MODELS += (PaperCognitionConfig, PaperCognitionBatchSpec)

from civitas.scenarios.kdd27_config import RQ3Config
from civitas.experiments.batch import RQ3BatchSpec
from civitas.experiments.branch import RQ3BranchIntervention
PUBLIC_MODELS += (RQ3Config, RQ3BatchSpec, RQ3BranchIntervention)
