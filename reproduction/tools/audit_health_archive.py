"""Read-only terminal archive restoration, including an interrupted committed prefix."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import resource
import signal
import time

from civitas.application.persistent_run import open_world
from civitas.contracts.base import content_hash
from civitas.experiments.registry import history
from civitas.storage.checkpoint import restore
from civitas.storage.commit_store import CommitStore


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def financial_fields(data, currency):
    initial = {row["account_id"]: row for row in data["initial_accounts"]}
    accounts = {row["account_id"]: row for row in data["accounts"]}
    resources = sorted({resource for row in initial.values() for resource in row["balances"]})
    initial_totals = {resource: sum(row["balances"][resource] for row in initial.values()) for resource in resources}
    current_totals = {resource: sum(row["balances"][resource] for row in accounts.values()) for resource in resources}
    residuals = {resource: current_totals[resource] - initial_totals[resource] for resource in resources}
    journal_fees = sum(row["buyer_fee_minor"] + row["seller_fee_minor"] + row["tax_minor"] for row in data["journal"])
    fee_change = accounts["fees"]["balances"][currency] - initial["fees"]["balances"][currency]
    assert not any(residuals.values()) and fee_change == journal_fees
    return {"initial_totals": initial_totals, "terminal_totals": current_totals,
            "resource_residuals": residuals, "cash_residual_minor": residuals[currency],
            "asset_residual_units": {key: value for key, value in residuals.items() if key != currency},
            "cash_scale": data["cash_scale"], "journal_transactions": len(data["journal"]),
            "fees": {"journal_fee_tax_total_minor": journal_fees, "fee_account_change_minor": fee_change,
                     "residual_minor": fee_change - journal_fees},
            "reservations": {"records": len(data["reservations"]),
                             "nonzero_amount_records": sum(row["amount"] > 0 for row in data["reservations"]),
                             "reserved_by_resource": {resource: sum(row["amount"] for row in data["reservations"] if row["resource_id"] == resource)
                                                      for resource in resources}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-config-hash", required=True)
    parser.add_argument("--expected-source-hash", required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    start, cpu = time.perf_counter(), time.process_time()
    before = hashes(args.archive)
    store = CommitStore(args.archive)
    assert store.identity.config_hash == args.expected_config_hash
    assert store.identity.code_hash == args.expected_source_hash
    assert store.identity.config["cognition_mode"] == "rule"
    attempts = history(store.root)
    terminal = next(row for row in reversed(attempts) if "status" in row)
    assert terminal["status"] not in ("planned", "initialized", "running")
    markers = store.verified_markers()
    assert all(marker.model_facts is None and not marker.invocation_receipt_refs for marker in markers)
    world, runtime = open_world(store)
    assert runtime is None
    record = restore(world, store)
    assert record.tick == markers[-1].tick
    assert content_hash(world.snapshot) == record.state_hash == markers[-1].state_hash
    assert terminal["tick"] == record.tick and terminal["state_hash"] == record.state_hash
    for participant in world.participants:
        participant.validate_state()
    market_data = world.snapshot.data["multiasset_market"]
    currency = store.identity.config["market"]["currency"]
    financial = financial_fields(market_data["ledger"], currency)
    financial.update(status="passed_existing_runtime_validation",
                     minimum_available_by_resource={resource: min(world.scenario.market.ledger.available(actor, resource)
                                                                for actor in world.snapshot.actors)
                                                    for resource in financial["resource_residuals"]},
                     command_count=len(market_data["commands"]), receipt_count=len(market_data["receipts"]),
                     reservation_validation="passed_existing_ledger_and_market_validators",
                     historical_fee_posting_recomputation="passed_in_PortfolioLedger.import_state",
                     coverage_sources={"restore": "src/civitas/core/world.py:63",
                                       "cash_assets_available": "src/civitas/resources/ledger.py:220",
                                       "accounts_locks_cost_reservations": "src/civitas/resources/ledger.py:236",
                                       "historical_fees_and_postings": "src/civitas/resources/ledger.py:313",
                                       "market_command_receipt_trade_venue": "src/civitas/markets/multiasset.py:270"})
    guard_path = store.root.parent.parent / (store.root.parent.name + "-network.log")
    guard_lines = guard_path.read_text().splitlines() if guard_path.exists() else None
    model_proof = {"status": "passed_rule_runtime_and_durable_fact_checks", "cognition_mode": store.identity.config["cognition_mode"],
                   "runtime_is_none": runtime is None, "model_fact_segments": sum(marker.model_facts is not None for marker in markers),
                   "invocation_receipt_references": sum(len(marker.invocation_receipt_refs) for marker in markers),
                   "provider_price_metadata": store.identity.metadata.get("provider_config", {}).get("price"),
                   "control_ledger_file_exists": Path(store.identity.metadata["control_ledger"]).exists(),
                   "original_network_guard": str(guard_path),
                   "network_guard_status": "MISSING" if guard_lines is None else "installed" if "guard-installed" in guard_lines else "UNVERIFIED",
                   "blocked_network_attempts": None if guard_lines is None else [line for line in guard_lines if line.startswith("blocked ")],
                   "network_guard_sha256": hashlib.sha256(guard_path.read_bytes()).hexdigest() if guard_path.exists() else None,
                   "model_calls": 0, "paid_calls": 0,
                   "basis": "Rule config initializes no provider runtime; every verified commit has no model-fact segment or invocation receipt reference. Network evidence is reported separately and missing fields remain explicit."}
    checkpoints = []
    for path in sorted((store.root / "checkpoints").glob("*.json.gz")):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == path.name.removesuffix(".json.gz")
        checkpoint = json.loads(gzip.decompress(raw))
        assert checkpoint["identity_hash"] == content_hash(store.identity)
        assert checkpoint["participants"] == store.identity.participants and checkpoint["rng"] == store.identity.rng
        marker = next(m for m in markers if m.tick == checkpoint["tick"])
        assert checkpoint["state_hash"] == marker.state_hash == content_hash(checkpoint["snapshot"])
        assert checkpoint["commit_hash"] == marker.commit_hash
        checkpoints.append({"path": str(path), "tick": checkpoint["tick"], "sha256": hashlib.sha256(raw).hexdigest()})
    assert hashes(store.root) == before
    result = {"status": "passed", "runtime_status": terminal["status"], "runtime_terminal_receipt": terminal,
              "run_id": store.identity.run_id, "archive": str(store.root), "committed_tick": record.tick,
              "ticks_requested": store.identity.config["ticks"], "state_hash": record.state_hash,
              "source_hash": store.identity.code_hash, "config_hash": store.identity.config_hash,
              "dependency_hash": store.identity.metadata["implementation"]["dependency_hash"],
              "verified_markers": len(markers), "checkpoints": checkpoints,
              "commit_chain": {"status": "passed", "first_tick": markers[0].tick, "last_tick": record.tick,
                               "marker_count": len(markers), "head_commit_hash": record.commit_hash,
                               "head_state_hash": record.state_hash, "terminal_receipt_matches_head": True,
                               "validator": "CommitStore.markers/verified_markers/load"},
              "financial_checks": financial, "model_call_proof": model_proof,
              "restoration": {"status": "passed_same_platform", "participants": store.identity.participants,
                              "source_and_dependency_identity_verified": True,
                              "participant_hashes_match_snapshot": True, "forward_steps": 0},
              "wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
              "archive_files_unchanged": len(before), "archive_hashes": before,
              "archive_bytes_written": 0, "forward_steps": 0, "new_paid_calls": 0}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("archive_hashes", "runtime_terminal_receipt")}))


if __name__ == "__main__":
    main()
