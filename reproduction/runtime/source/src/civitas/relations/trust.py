from pydantic import TypeAdapter

from civitas.contracts.base import NonNegative
from .types import FactTrust, PredictionTrust, Probability, TrustEvidence


def _eligible(trust, evidence, tick, seen, kind):
    TypeAdapter(NonNegative).validate_python(tick, strict=True)
    roots = {}
    for raw in evidence:
        e = TrustEvidence.model_validate(raw)
        if (e.kind != kind or e.observer_id != trust.source_id or e.author_id != trust.target_id
                or e.root_claim_id in trust.used_roots or e.matured_tick > tick or e.visible_tick > tick
                or trust.source_id not in e.visible_to or (e.root_claim_id, e.author_id, kind) not in seen):
            continue
        if e.root_claim_id in roots:
            prior = roots[e.root_claim_id]
            if (e.outcome, e.probability) != (prior.outcome, prior.probability):
                raise ValueError("Conflicting evidence for one root claim")
        else:
            roots[e.root_claim_id] = e
    return [roots[key] for key in sorted(roots)]


def update_fact_trust(trust: FactTrust, evidence, *, tick: int, decay: float, seen: set) -> FactTrust:
    trust = FactTrust.model_validate(trust)
    TypeAdapter(Probability).validate_python(decay, strict=True)
    eligible = _eligible(trust, evidence, tick, seen, "fact")
    return trust.model_copy(update={"a": trust.a0 + decay * (trust.a - trust.a0) + float(sum(e.outcome for e in eligible)),
        "b": trust.b0 + decay * (trust.b - trust.b0) + float(sum(1 - e.outcome for e in eligible)),
        "used_roots": sorted([*trust.used_roots, *(e.root_claim_id for e in eligible)])})


def update_prediction_trust(trust: PredictionTrust, evidence, *, tick: int, alpha: float, seen: set) -> PredictionTrust:
    trust = PredictionTrust.model_validate(trust)
    TypeAdapter(Probability).validate_python(alpha, strict=True)
    eligible = _eligible(trust, evidence, tick, seen, "prediction")
    score = trust.brier_ewma
    # Stable root order defines EWMA when several forecasts mature in one tick.
    for e in eligible:
        score = (1.0 - alpha) * score + alpha * (e.probability - e.outcome) ** 2
    return trust.model_copy(update={"brier_ewma": score, "samples": trust.samples + len(eligible),
        "used_roots": sorted([*trust.used_roots, *(e.root_claim_id for e in eligible)])})
