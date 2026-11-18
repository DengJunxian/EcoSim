"""Use WP01's sole field-writer registry; add proposition-level belief ownership."""
from typing import Literal
from pydantic import model_validator

from civitas.contracts.base import Id, Record, require_unique
from civitas.contracts.config import MechanismSpec
from civitas.contracts.ports import Mechanism
from civitas.core.registry import MechanismRegistry


class BeliefOwner(Record):
    proposition_id: Id
    owner: Literal['rule_social','llm_integrated','llm_private_then_numeric_social'] = 'rule_social'
    model_reads_peers: bool = False

    @model_validator(mode='after')
    def private_route(self):
        if self.owner == 'llm_private_then_numeric_social' and self.model_reads_peers:
            raise ValueError('Private-then-social model input cannot include peer messages')
        return self


def validate_belief_owners(declarations):
    declarations = [BeliefOwner.model_validate(d) for d in declarations]
    require_unique([d.proposition_id for d in declarations], 'belief update owner')
    return {d.proposition_id:d.owner for d in declarations}
