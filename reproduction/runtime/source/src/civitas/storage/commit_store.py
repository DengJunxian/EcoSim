"""A fsynced marker is the sole authority; indexes never decide commitment."""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Literal
import uuid

from pydantic import Field, TypeAdapter
from core.experiments.strict_types import strict_json_loads
from civitas.contracts.base import Digest, Id, NonNegative, Record, content_hash
from civitas.contracts.state import WorldSnapshot, FinancialWorldSnapshot, PortfolioWorldSnapshot

SNAPSHOT = TypeAdapter(WorldSnapshot | FinancialWorldSnapshot | PortfolioWorldSnapshot)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')


def stored_json_bytes(value, *, compressed=False):
    raw = json_bytes(value)
    return gzip.compress(raw,compresslevel=1,mtime=0) if compressed else raw


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as handle:
        handle.write(stored_json_bytes(value,compressed=path.name.endswith('.gz')))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or unsafe artifact: {path.name}')
    raw = path.read_bytes()
    if path.name.endswith('.gz'):
        try:
            raw = gzip.decompress(raw)
        except (OSError,EOFError) as exc:
            raise ValueError('Invalid compressed JSON artifact') from exc
    return strict_json_loads(raw.decode('utf-8'))


class StateChange(Record):
    op: Literal['set', 'delete', 'append']
    path: list[str | NonNegative] = Field(min_length=1)
    value: Any = None
    offset: NonNegative | None = None


class StateDelta(Record):
    schema_version: Literal[1] = 1
    base_hash: Digest
    state_hash: Digest
    changes: list[StateChange]


def differences(before, after, path=()):
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        rows = [StateChange(op='delete', path=[*path, key]) for key in sorted(before.keys()-after.keys())]
        for key in sorted(after):
            rows.extend(differences(before[key], after[key], (*path, key)) if key in before else
                        [StateChange(op='set', path=[*path, key], value=after[key])])
        return rows
    if isinstance(before, list) and isinstance(after, list) and len(after) >= len(before):
        rows = []
        for index, value in enumerate(before):
            rows.extend(differences(value, after[index], (*path, index)))
        if len(after) > len(before):
            rows.append(StateChange(op='append', path=list(path), value=after[len(before):], offset=len(before)))
        return rows
    return [StateChange(op='set', path=list(path), value=after)]


def apply_delta(before, delta):
    delta = StateDelta.model_validate(delta)
    if content_hash(before) != delta.base_hash:
        raise ValueError('Delta base hash mismatch')
    value = deepcopy(before)
    for change in delta.changes:
        payload = change.model_dump(mode='json')['value']
        target = value
        for key in change.path[:-1]:
            target = target[key]
        key = change.path[-1]
        if change.op == 'set':
            target[key] = payload
        elif change.op == 'delete':
            del target[key]
        else:
            array = target[key]
            if not isinstance(array, list) or len(array) != change.offset or not isinstance(change.value, list):
                raise ValueError('Delta append offset/type mismatch')
            array.extend(payload)
    if content_hash(value) != delta.state_hash:
        raise ValueError('Delta result hash mismatch')
    return value


class Segment(Record):
    sha256: Digest
    bytes: NonNegative
    encoding: Literal['json','json+gzip'] = 'json'


class StoreIdentity(Record):
    schema_version: Literal[1] = 1
    run_id: Id
    lineage_id: Id
    branch_id: Id
    code_hash: Digest
    config_hash: Digest
    config: dict[str, Any]
    participants: dict[str, int]
    rng: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Commit(Record):
    schema_version: Literal[1] = 1
    kind: Literal['initial', 'tick']
    tick: NonNegative
    world_version: NonNegative
    parent_commit_hash: Digest | None
    identity_hash: Digest
    state_hash: Digest
    segment: Segment
    model_facts: Segment | None = None
    invocation_receipt_refs: list[Id] = Field(default_factory=list)
    commit_hash: Digest


class CommitStore:
    def __init__(self, root, *, fault_hook=None):
        self.root = Path(root).resolve()
        self.identity = StoreIdentity.model_validate(read_json(self.root/'identity.json'))
        if content_hash(self.identity.config) != self.identity.config_hash:
            raise ValueError('Configuration hash mismatch')
        self.fault_hook = fault_hook
        self.receipt_reader = None

    @classmethod
    def create(cls, root, world, *, config, code_hash, run_id=None, lineage_id=None, branch_id=None, metadata=None):
        from .checkpoint import participant_versions
        versions = participant_versions(world)
        run_id = run_id or Path(root).name
        identity = StoreIdentity(run_id=run_id, lineage_id=lineage_id or run_id, branch_id=branch_id or run_id,
            code_hash=code_hash, config_hash=content_hash(config), config=config, participants=versions,
            rng=world.rng._root.model_dump(mode='json'), metadata=metadata or {})
        return cls.initialize(root, identity, world.snapshot)

    @classmethod
    def initialize(cls, root, identity, snapshot):
        identity = StoreIdentity.model_validate(identity)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        for name in ('commits', 'segments', 'checkpoints', 'diagnostics'):
            (root/name).mkdir()
        atomic_json(root/'identity.json', identity.model_dump(mode='json'))
        sync_directory(root.parent)
        store = cls(root)
        store._publish(snapshot, snapshot.model_dump(mode='json'), kind='initial', parent=None, refs=[])
        return store

    @contextmanager
    def writer(self):
        with (self.root/'writer.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def fault(self, point):
        if self.fault_hook:
            self.fault_hook(point)

    def marker_path(self, tick):
        return self.root/'commits'/f'{tick:08d}.json'

    def _read_marker(self, path):
        raw = read_json(path)
        record = Commit.model_validate(raw)
        if (record.commit_hash != content_hash({k:v for k,v in raw.items() if k!='commit_hash'})
                or record.identity_hash != content_hash(self.identity) or path != self.marker_path(record.tick)):
            raise ValueError(f'Commit identity/hash mismatch: {path.name}')
        return record

    def markers(self):
        records = [self._read_marker(path) for path in sorted((self.root/'commits').glob('*.json'))]
        if not records or records[0].kind != 'initial' or records[0].parent_commit_hash is not None:
            raise ValueError('Missing initial commit')
        for previous, current in zip(records, records[1:]):
            if (current.kind != 'tick' or current.tick != previous.tick+1
                    or current.world_version != previous.world_version+1 or current.parent_commit_hash != previous.commit_hash):
                raise ValueError(f'Broken commit chain at tick {current.tick}')
        return records

    def segment(self, record):
        return self.read_segment(record.segment)

    def segment_path(self, reference):
        return self.root/'segments'/(reference.sha256+('.json.gz' if reference.encoding=='json+gzip' else '.json'))

    def read_segment(self, reference):
        path = self.segment_path(reference)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing referenced segment')
        raw = path.read_bytes()
        if len(raw) != reference.bytes or hashlib.sha256(raw).hexdigest() != reference.sha256:
            raise ValueError('Segment byte hash mismatch')
        if reference.encoding=='json+gzip':
            try:
                raw = gzip.decompress(raw)
            except (OSError,EOFError) as exc:
                raise ValueError('Invalid compressed segment') from exc
        value = strict_json_loads(raw.decode('utf-8'))
        if json_bytes(value) != raw:
            raise ValueError('Noncanonical segment JSON')
        return value

    def verified_markers(self):
        from .checkpoint import validate_versions
        validate_versions(self.identity.participants)
        records = self.markers()
        # Verify every byte reference before exposing any recovered state.
        for index, record in enumerate(records):
            payload = self.segment(record)
            self.model_records(record)
            if index:
                delta = StateDelta.model_validate(payload)
                if delta.base_hash != records[index-1].state_hash or delta.state_hash != record.state_hash:
                    raise ValueError('Delta hashes differ from commit chain')
            elif content_hash(SNAPSHOT.validate_python(payload)) != record.state_hash:
                raise ValueError('Initial snapshot hash mismatch')
        return records

    def states(self):
        records = self.verified_markers()
        state = None
        for record in records:
            payload = self.segment(record)
            state = payload if record.kind == 'initial' else apply_delta(state, payload)
            snapshot = SNAPSHOT.validate_python(state)
            if (snapshot.tick != record.tick or snapshot.world_version != record.world_version
                    or content_hash(snapshot) != record.state_hash):
                raise ValueError(f'Business state hash mismatch at tick {record.tick}')
            yield record, snapshot

    def load(self, tick=None):
        from .checkpoint import Checkpoint
        records = self.verified_markers()
        tick = records[-1].tick if tick is None else tick
        selected = next((r for r in records if r.tick==tick),None)
        if type(tick) is not int or selected is None:
            raise ValueError(f'No committed tick {tick}')
        by_tick = {r.tick:r for r in records}
        checkpoint = None
        for path in sorted(p for p in (self.root/'checkpoints').iterdir() if p.name.endswith(('.json','.json.gz'))):
            raw = read_json(path)
            candidate = Checkpoint.model_validate(raw)
            marker = by_tick.get(candidate.tick)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if (path.name not in (digest+'.json',digest+'.json.gz') or marker is None
                    or candidate.identity_hash!=content_hash(self.identity) or candidate.commit_hash!=marker.commit_hash
                    or candidate.state_hash!=marker.state_hash or content_hash(candidate.snapshot)!=marker.state_hash
                    or candidate.participants!=self.identity.participants or candidate.rng!=self.identity.rng):
                raise ValueError('Invalid checkpoint reference, version or hash')
            if candidate.tick<=tick and (checkpoint is None or candidate.tick>checkpoint.tick):
                checkpoint = candidate
        if checkpoint is None:
            state, start = self.segment(records[0]),records[0].tick
        else:
            state, start = checkpoint.model_dump(mode='json')['snapshot'],checkpoint.tick
        for record in records:
            if start<record.tick<=tick:
                state = apply_delta(state,self.segment(record))
        snapshot = SNAPSHOT.validate_python(state)
        if snapshot.tick!=tick or snapshot.world_version!=selected.world_version or content_hash(snapshot)!=selected.state_hash:
            raise ValueError('Recovered state does not match the selected commit')
        return selected,snapshot

    def write_segment(self, payload):
        raw = stored_json_bytes(payload,compressed=True)
        segment = Segment(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),encoding='json+gzip')
        path = self.segment_path(segment)
        if not path.exists():
            temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            with temporary.open('xb') as handle:
                midpoint = len(raw)//2
                handle.write(raw[:midpoint]); handle.flush(); os.fsync(handle.fileno())
                self.fault('segment_write')
                handle.write(raw[midpoint:]); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        elif path.is_symlink() or path.read_bytes() != raw:
            raise ValueError('Immutable segment conflict')
        return segment

    def model_records(self, record):
        rows = self.read_segment(record.model_facts) if record.model_facts is not None else []
        if (not isinstance(rows,list) or sorted(r['attempt_id'] for r in rows) != list(record.invocation_receipt_refs)
                or any(content_hash(r['request']) != r['cache_key'] for r in rows)):
            raise ValueError('Missing or conflicting model receipts')
        return rows

    def archived_records(self):
        rows = {}
        for marker in self.markers():
            for row in self.model_records(marker):
                key = row['attempt_id']
                if key in rows and rows[key] != row:
                    raise ValueError('Conflicting immutable model fact')
                rows[key] = row
        return list(rows.values())

    def _publish(self, snapshot, payload, *, kind, parent, refs):
        segment = self.write_segment(payload)
        facts = None
        if refs:
            if self.receipt_reader is None:
                raise ValueError('Invocation references require a durable receipt source')
            rows = {r['attempt_id']:r for r in self.receipt_reader()}
            facts = self.write_segment([rows[ref] for ref in refs])
        body = dict(schema_version=1, kind=kind, tick=snapshot.tick, world_version=snapshot.world_version,
                    parent_commit_hash=parent, identity_hash=content_hash(self.identity),
                    state_hash=content_hash(snapshot), segment=segment.model_dump(), model_facts=facts.model_dump() if facts else None,
                    invocation_receipt_refs=list(refs))
        record = Commit(**body, commit_hash=content_hash(body))
        marker = self.marker_path(snapshot.tick)
        if marker.exists():
            raise ValueError('Duplicate committed tick')
        self.fault('before_marker')
        atomic_json(marker, record.model_dump(mode='json'))
        self.fault('after_marker')
        return record

    def commit(self, base, prepared):
        with self.writer():
            head = self.markers()[-1]
            if head.tick != base.tick or head.state_hash != content_hash(base):
                raise ValueError('Commit base differs from durable head')
            old, new = base.model_dump(mode='json'), prepared.snapshot.model_dump(mode='json')
            delta = StateDelta(base_hash=head.state_hash, state_hash=prepared.state_hash, changes=differences(old, new))
            return self._publish(prepared.snapshot, delta.model_dump(mode='json'), kind='tick',
                                 parent=head.commit_hash, refs=prepared.invocation_receipt_refs)

    def is_committed(self, prepared):
        path = self.marker_path(prepared.tick)
        if not path.exists():
            return False
        record = self._read_marker(path)
        self.segment(record)
        self.model_records(record)
        if record.state_hash != prepared.state_hash:
            raise ValueError('Committed tick conflicts with prepared state')
        return True

    def update_index(self):
        self.fault('before_index')
        records = self.markers()
        with sqlite3.connect(self.root/'index.sqlite') as db:
            db.execute('CREATE TABLE IF NOT EXISTS commits (tick INTEGER PRIMARY KEY, commit_hash TEXT, state_hash TEXT)')
            db.execute('DELETE FROM commits')
            db.executemany('INSERT INTO commits VALUES (?,?,?)', [(r.tick,r.commit_hash,r.state_hash) for r in records])
        self.fault('after_index')

    def orphans(self):
        used = {self.segment_path(s).name for r in self.markers() for s in (r.segment,r.model_facts) if s is not None}
        return sorted(str(p.relative_to(self.root)) for folder in ('segments','commits','checkpoints') for p in (self.root/folder).iterdir()
                      if p.is_file() and (p.name.endswith('.tmp') or (folder=='segments' and p.name not in used)))

    def repair_index(self):
        record, _ = self.load()
        path = self.root/'index.sqlite'
        if path.exists():
            os.replace(path, self.root/'diagnostics'/('index-'+uuid.uuid4().hex+'.sqlite'))
        self.update_index()
        report = {'tick':record.tick,'commit_hash':record.commit_hash,'orphans':self.orphans()}
        atomic_json(self.root/'diagnostics'/('recovery-'+uuid.uuid4().hex+'.json'), report)
        return report
