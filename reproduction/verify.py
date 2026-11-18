"""Verify the published experiment projections offline."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from tools.health_final_arithmetic import check_tables, ledger_projection, project, read_csv

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_model_pilot(out):
    root = HERE / "model_pilot"
    rows = read_csv(root / "case_results.csv")
    groups = defaultdict(list)
    for row in rows:
        assert row["resource_executable"] == "True" and row["formal_eligible"] == "False"
        groups[row["backend"], row["block_id"]].append(row)
    expected = {(row["backend"], row["block_id"]): row for row in read_csv(root / "paired_actions.csv")}
    assert len(rows) == 162 and len(expected) == len(groups) == 81
    signatures = ("action", "signed_quantity_units", "price_ticks")
    changed = Counter()
    rebuilt = []
    for key, pair in sorted(groups.items()):
        a, b = sorted(pair, key=lambda row: row["state"])
        assert [a["state"], b["state"]] == ["0", "1"]
        assert all(a[k] == b[k] for k in ("backend", "block_id", "archetype", "fact_pair"))
        different = tuple(a[field] for field in signatures) != tuple(b[field] for field in signatures)
        changed[key[0]] += different
        row = {"backend": a["backend"], "block_id": a["block_id"], "archetype": a["archetype"],
               "fact_pair": a["fact_pair"], "state0_action": a["action"], "state1_action": b["action"],
               "economic_action_changed": str(different),
               "signed_quantity_delta": str(int(b["signed_quantity_units"]) - int(a["signed_quantity_units"])),
               "quote_delta_ticks": str(int(b["price_ticks"]) - int(a["price_ticks"]))
               if a["price_ticks"] and b["price_ticks"] else ""}
        assert row == expected[key], key
        rebuilt.append(row)
    with (out / "model_paired_actions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rebuilt[0]))
        writer.writeheader()
        writer.writerows(rebuilt)
    summary = read(root / "summary.json")
    for backend, expected_summary in summary["backend_summaries"].items():
        subset = [row for row in rows if row["backend"] == backend]
        assert len(subset) == expected_summary["cells"]
        assert Counter(row["action"] for row in subset) == expected_summary["action_counts"]
        assert changed[backend] == expected_summary["changed_economic_actions"]
    live = [row for row in rows if row["backend"] == "gpt6_luna_live"]
    assert len(live) == 54 and len({row["observation_hash"] for row in live}) == 45
    assert all(row["action"] == "hold" and row["output_origin"] == "live" for row in live)
    usage = {field: sum(int(row[field]) for row in live) for field in
             ("input_tokens", "output_tokens", "accounted_upper_bound_nano_USD")}
    assert usage["input_tokens"] == summary["live_input_tokens"]
    assert usage["output_tokens"] == summary["live_output_tokens"]
    assert usage["accounted_upper_bound_nano_USD"] / 1e9 == summary["live_accounted_upper_bound_USD"]
    consistency = read(root / "snapshot54_input_consistency_review.json")
    assert {row["case_id"] for row in live} == {row["case_id"] for row in consistency["per_case"]}
    assert consistency["risk_field_mismatch_count"] == consistency["reference_field_mismatch_count"] == 54
    mask = read(root / "rq3_exposure_audit.json")
    seed_rows = mask["per_seed_rows"]
    sums = {field: sum(row[field] for row in seed_rows) for field in
            ("llm_scheduled_text_requests", "hybrid_scheduled_text_requests",
             "hybrid_scheduled_directly_infected_requests")}
    assert len(seed_rows) == 20 and all(mask["summary"][key] == val for key, val in sums.items())
    zero_text = [row["seed"] for row in seed_rows if not row["hybrid_scheduled_text_requests"]]
    assert zero_text == mask["summary"]["hybrid_zero_text_seeds"]
    return {"cells": len(live), "unique_observations": 45, "comparison_rows": len(rows),
            "paired_rows": len(rebuilt), "changed_action_pairs": dict(changed), **usage,
            "risk_and_reference_mismatches": 54, "schedule_counts": sums, "scope": "frozen derived records"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run with assertions enabled.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    health = HERE / "health"
    plan = read(health / "plan.json")
    projected, admitted = {}, {}
    for job in plan["jobs"]:
        label = job["label"]
        config = health / job["config_file"]
        assert hashlib.sha256(config.read_bytes()).hexdigest() == job["config_file_sha256"]
        analysis = read(health / "projections" / (label + "-analysis.json"))
        audit = read(health / "projections" / (label + "-audit-result.json"))
        ledger = read(health / "terminal_ledgers" / (label + "-100.json"))
        assert ledger["state_hash"] == audit["state_hash"] == analysis["committed_hashes"][-1]["state_hash"]
        check = ledger_projection(ledger["ledger"], ledger["actors"])
        for key in ("cash_residual_minor", "asset_residual_units", "fees", "reservations"):
            assert check[key] == audit["financial_checks"][key], (label, key)
        projected[label] = project(analysis, read(config))
        admitted[label] = audit["financial_checks"]["status"] == "passed_existing_runtime_validation"
    results = {"health": check_tables(health / "reference", plan, projected, admitted),
               "model_pilot": check_model_pilot(out)}
    capacity = sorted((HERE / "capacity/terminal_ledgers").glob("*.json"))
    assert len(capacity) == 3
    for path in capacity:
        row = read(path)
        ledger_projection(row["ledger"], row["actors"])
    results["capacity_terminal_ledgers"] = len(capacity)
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Validated {len(projected)} health trajectories, 12 terminal ledgers, and 81 model contrasts.")


if __name__ == "__main__":
    main()
