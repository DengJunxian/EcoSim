"""Reuse the repository's strict finite values and canonical identities."""
from typing import Annotated

from pydantic import Field

from core.experiments.identity import research_hash as content_hash
from core.experiments.strict_types import StrictResearchModel as Record

Id = Annotated[str, Field(min_length=1)]
NonNegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(gt=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def require_unique(values, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")
