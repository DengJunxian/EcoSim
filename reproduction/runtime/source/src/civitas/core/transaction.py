from civitas.contracts.base import content_hash
from civitas.contracts.state import FinancialPreparedTick, PortfolioPreparedTick, PreparedTick, TickContext, WorldSnapshot


class TickTransaction:
    """Logical deep snapshot: no working state is published before prepare passes."""

    def __init__(self, base: WorldSnapshot, participants=()):
        self.base = base
        self.working = base.model_dump(mode="json")
        self.working["tick"] += 1
        self._used = set(self.working['used_ids'])
        self.participants = tuple(participants)
        self.before = {p.participant_id: p.export_state() for p in self.participants}
        if any(state != self.working["data"][key] for key, state in self.before.items()):
            raise RuntimeError("Participant changed outside a tick transaction")

    def context(self) -> TickContext:
        return TickContext(**{key: self.working[key] for key in TickContext.model_fields})

    def snapshot(self, *, data=None, projection=False, memory_window=None) -> WorldSnapshot:
        value = self.working
        if projection:
            # Read barriers need current inputs; the commit still validates the entire history.
            value = {key:value[key] for key in (*TickContext.model_fields,'schema_version','events','messages','memories')}
            if memory_window is not None:
                if type(memory_window) is not int or memory_window<1:
                    raise ValueError('Observation history window must be positive')
                beliefs, recent = {}, []
                for fact in value['memories']:
                    if fact['visible_at']['tick']>value['tick']:
                        continue
                    if fact['fact_id'].startswith('belief:'):
                        beliefs[fact['owner_id']] = fact
                    elif fact['visible_at']['tick']>value['tick']-memory_window:
                        recent.append(fact)
                value['memories'] = [*recent,*beliefs.values()]
        if data is not None:
            value = {**value,'data':data}
        return type(self.base).model_validate(value)

    def capture_participants(self) -> None:
        for participant in self.participants:
            participant.validate_state()
            self.working["data"][participant.participant_id] = participant.export_state()

    def rollback(self) -> None:
        for participant in reversed(self.participants):
            participant.import_state(self.before[participant.participant_id])

    def set_data(self, data: dict) -> None:
        validated = TickContext(**{key: data if key == "data" else self.working[key] for key in TickContext.model_fields})
        self.working["data"] = validated.model_dump()["data"]

    def claim(self, *identifiers: str) -> None:
        used = self.working["used_ids"]
        for identifier in identifiers:
            if identifier in self._used:
                raise ValueError(f"Duplicate world ID: {identifier}")
            used.append(identifier)
            self._used.add(identifier)

    def prepare(self, invocation_refs: list[str]) -> PreparedTick:
        snapshot = type(self.base).model_validate({**self.working,'world_version':self.base.world_version+1})
        prepared_type = {1: PreparedTick, 2: FinancialPreparedTick, 3: PortfolioPreparedTick}[snapshot.schema_version]
        return prepared_type(base_version=self.base.world_version, tick=snapshot.tick, snapshot=snapshot,
                            state_hash=content_hash(snapshot), invocation_receipt_refs=invocation_refs)
