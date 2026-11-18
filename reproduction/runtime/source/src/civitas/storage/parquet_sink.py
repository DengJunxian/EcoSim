"""Rebuildable researcher tables; never used to recover or decide world state."""
import hashlib
import os
from pathlib import Path

from .commit_store import atomic_json, json_bytes, sync_directory


def export_parquet(store, destination):
    import pyarrow as pa
    import pyarrow.parquet as pq
    head, snapshot = store.load()
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    markers = store.markers()
    tables = {
        'commits':pa.table({'tick':pa.array([r.tick for r in markers],type=pa.int64()),
            'commit_hash':pa.array([r.commit_hash for r in markers],type=pa.string()),
            'state_hash':pa.array([r.state_hash for r in markers],type=pa.string())}),
        'receipts':pa.table({'command_id':pa.array([r.command_id for r in snapshot.receipts],type=pa.string()),
            'actor_id':pa.array([r.actor_id for r in snapshot.receipts],type=pa.string()),
            'execution_seq':pa.array([r.execution_seq for r in snapshot.receipts],type=pa.int64()),
            'body_json':pa.array([json_bytes(r.model_dump(mode='json')).decode() for r in snapshot.receipts],type=pa.string())}),
    }
    artifacts = {}
    for name, table in tables.items():
        path = destination/(name+'.parquet')
        pq.write_table(table,path,compression='zstd')
        with path.open('rb') as handle:
            os.fsync(handle.fileno())
        artifacts[path.name] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
                                'rows':table.num_rows,'arrow_schema':str(table.schema)}
    sync_directory(destination)
    receipt = {'schema_version':1,'authority':False,'source_commit':head.commit_hash,'state_hash':head.state_hash,
        'pyarrow_version':pa.__version__,'audience':'researcher','files':artifacts}
    atomic_json(destination/'manifest.json',receipt)
    return receipt
