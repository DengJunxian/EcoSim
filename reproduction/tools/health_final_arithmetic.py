"""Small independent arithmetic checks of existing health analysis JSON and CSV.

This module imports no simulator, runs no analyze_run, and never restores a world.
It verifies the seven descriptive outcomes from already exported per-tick facts,
trades, and decisions. Financial admission is a separate required gate.
"""
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import statistics

METRICS = ("no_trade_ratio", "one_sided_ratio", "two_sided_coverage", "trade_count",
           "fees_minor", "daily_log_return", "traded_quantity")
DAILY_UNDEFINED = ("excess_kurtosis", "mean_abs_return_acf_lags_1_20",
                   "maximum_abs_raw_return_acf_lags_1_20", "volume_abs_return_correlation")


def project(analysis, config):
    facts, decisions, trades = analysis["facts"], analysis["decisions"], analysis["trades"]
    assert analysis["run"]["status"] == analysis["run"]["raw_status"] == "completed"
    assert analysis["run"]["verified_integrity"] is True
    assert analysis["analysis_receipt"]["status"] == "passed"
    assert analysis["analysis_is_new_simulation"] is False and analysis["formal_eligible"] is False
    assert config["schema_version"] == 2 and config["ticks"] == config["market"]["ticks_per_day"] == 100
    assert config["cognition_mode"] == "rule"
    assert [row["tick"] for row in facts] == list(range(1, 101))
    assert [row["tick"] for row in analysis["committed_hashes"]] == list(range(1, 101))
    assert len(decisions) == 10000 and len({(d["tick"], d["actor_id"]) for d in decisions}) == 10000
    all_decisions = Counter(row["tick"] for row in decisions)
    active = Counter(row["tick"] for row in decisions if row["active"])
    by_tick = defaultdict(list)
    for trade in trades:
        by_tick[trade["trade"]["tick"]].append(trade)
        postings = defaultdict(int)
        for posting in trade["postings"]:
            postings[posting["resource_id"]] += posting["amount"]
        assert all(value == 0 for value in postings.values())
    cumulative_fees = 0
    for row in facts:
        tick_trades = by_tick[row["tick"]]
        assert all_decisions[row["tick"]] == 100
        assert active[row["tick"]] == row["active"]
        assert len(tick_trades) == row["trade_count"]
        assert sum(t["trade"]["quantity_units"] for t in tick_trades) == row["volume"]
        fees = sum(t["buyer_fee_minor"] + t["seller_fee_minor"] + t["tax_minor"] for t in tick_trades)
        assert fees == row["fees_step_minor"]
        cumulative_fees += fees
        assert cumulative_fees == row["fees_cumulative_minor"]
        assert sum(bool(row[key]) for key in ("one_sided", "two_sided", "empty_book")) == 1
    numerators = {"no_trade_ratio": sum(row["trade_count"] == 0 for row in facts),
                  "one_sided_ratio": sum(row["one_sided"] for row in facts),
                  "two_sided_coverage": sum(row["two_sided"] for row in facts)}
    metrics = {key: value / 100 for key, value in numerators.items()}
    metrics.update(trade_count=len(trades), fees_minor=cumulative_fees,
                   traded_quantity=sum(row["volume"] for row in facts),
                   daily_log_return=math.log(facts[-1]["close"] / config["market"]["initial_price"]))
    summary = analysis["summary"]
    assert summary["window"] == [1, 100]
    for name, value in metrics.items():
        if name == "trade_count":
            continue
        if name == "daily_log_return":
            assert summary["daily_statistics"]["returns"] == [value]
            continue
        record = summary["metrics"][name]
        assert record["value"] == value and record["status"] == "VALID"
        assert record["denominator_count"] == 100
        assert record["sample_count"] == numerators.get(name, 100)
    assert len(summary["daily"]) == 2
    assert [row["day"] for row in summary["daily"]] == [0, 1]
    assert all(row["complete"] for row in summary["daily"])
    assert summary["daily"][0]["close"] == config["market"]["initial_price"]
    assert summary["daily"][1]["close"] == facts[-1]["close"]
    assert summary["daily"][1]["volume"] == metrics["traded_quantity"]
    assert summary["daily_statistics"]["raw_acf"] is None
    assert summary["daily_statistics"]["absolute_acf"] is None
    for name in DAILY_UNDEFINED:
        for location in (summary["metrics"], summary["daily_statistics"]["metrics"]):
            record = location[name]
            assert record["value"] is None and record["status"] == "UNDEFINED"
            assert record["reason"] == "INSUFFICIENT_OR_CONSTANT_SAMPLE"
            assert record["sample_count"] == record["denominator_count"] == 1
    cost = analysis["model_cost"]
    assert all(cost[name] == 0 for name in ("owned_attempts", "input_tokens", "output_tokens", "unknown_usage_attempts"))
    assert cost["known_accounted_nano"] == cost["total_accounted_nano"] == {"USD": 0, "CNY": 0}
    return {"metrics": metrics, "ratio_numerators": numerators, "tick_denominator": 100,
            "decision_count": 10000, "active_decisions": sum(active.values()),
            "daily_return_count": 1, "undefined_daily_metrics": list(DAILY_UNDEFINED),
            "financial_admission": "NOT_DECIDED_BY_ARITHMETIC", "formal_eligible": False}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ledger_projection(ledger, actors, currency="CNY"):
    """Recompute terminal integer accounting without a ledger or world object."""
    initial = {row["account_id"]: row for row in ledger["initial_accounts"]}
    accounts = {row["account_id"]: row for row in ledger["accounts"]}
    assert set(initial) == set(accounts) == set(actors) | {"fees"}
    resources = sorted(next(iter(initial.values()))["balances"])
    assets = {row["asset_id"]: row for row in ledger["assets"]}
    initial_totals = {key: sum(row["balances"][key] for row in initial.values()) for key in resources}
    totals = {key: sum(row["balances"][key] for row in accounts.values()) for key in resources}
    assert initial_totals == totals
    reconstructed = {key: dict(value["balances"]) for key, value in initial.items()}
    fees = 0
    for tx in ledger["journal"]:
        trade = tx["trade"]
        notional = trade["price_ticks"] * assets[trade["asset_id"]]["price_tick_minor"] * trade["quantity_units"]
        assert notional == tx["notional_minor"]
        commission = tx["fees"]["commission"]
        tax = tx["fees"]["tax"]
        assert tx["buyer_fee_minor"] == tx["seller_fee_minor"] == notional * commission["numerator"] // commission["denominator"]
        assert tx["tax_minor"] == notional * tax["numerator"] // tax["denominator"]
        balanced = defaultdict(int)
        for posting in tx["postings"]:
            reconstructed[posting["account_id"]][posting["resource_id"]] += posting["amount"]
            balanced[posting["resource_id"]] += posting["amount"]
        assert all(value == 0 for value in balanced.values())
        fees += tx["buyer_fee_minor"] + tx["seller_fee_minor"] + tx["tax_minor"]
    assert reconstructed == {key: value["balances"] for key, value in accounts.items()}
    fee_change = accounts["fees"]["balances"][currency] - initial["fees"]["balances"][currency]
    assert fee_change == fees
    reserved = defaultdict(int)
    reserved_by_resource = {key: 0 for key in resources}
    for row in ledger["reservations"]:
        command = row["command"]
        quantity = row["remaining_units"]
        assert 0 <= quantity <= command["quantity_units"]
        expected_resource = currency if command["side"] == "buy" else command["asset_id"]
        assert row["resource_id"] == expected_resource
        if command["side"] == "sell":
            required = quantity
        else:
            notional = command["limit_price_ticks"] * assets[command["asset_id"]]["price_tick_minor"] * quantity
            commission = ledger["fees"]["commission"]
            required = notional + notional * commission["numerator"] // commission["denominator"]
        assert row["amount"] == required
        reserved[(command["account_id"], row["resource_id"])] += required
        reserved_by_resource[row["resource_id"]] += required
    available = {(actor, resource): accounts[actor]["balances"][resource]
                 - accounts[actor]["locked"].get(resource, 0) - reserved[(actor, resource)]
                 for actor in accounts for resource in resources}
    assert min(available.values()) >= 0
    return {"initial_totals": initial_totals, "terminal_totals": totals,
            "resource_residuals": {key: 0 for key in resources}, "cash_residual_minor": 0,
            "asset_residual_units": {key: 0 for key in resources if key != currency},
            "cash_scale": ledger["cash_scale"], "journal_transactions": len(ledger["journal"]),
            "fees": {"journal_fee_tax_total_minor": fees, "fee_account_change_minor": fee_change, "residual_minor": 0},
            "reservations": {"records": len(ledger["reservations"]),
                             "nonzero_amount_records": sum(row["amount"] > 0 for row in ledger["reservations"]),
                             "reserved_by_resource": reserved_by_resource},
            "minimum_available_by_resource": {resource: min(available[(actor, resource)] for actor in actors)
                                               for resource in resources},
            "independent_checks": {"journal_postings_reconstruct_balances": True,
                                   "historical_fee_integer_arithmetic": True,
                                   "reservation_amount_integer_arithmetic": True,
                                   "available_balances_nonnegative": True}}


def optional_number(value):
    return None if value == "" else float(value)


def check_tables(table_root, plan, projected, admitted):
    rows = read_csv(Path(table_root) / "run.csv")
    paired = read_csv(Path(table_root) / "paired.csv")
    descriptive = read_csv(Path(table_root) / "descriptive.csv")
    assert len(rows) == len({row["label"] for row in rows}) == 9
    assert [row["label"] for row in rows] == [job["label"] for job in plan["jobs"]]
    lookup = {}
    for row in rows:
        label = row["label"]
        assert (row["complete_health_outcomes"] == "True") == admitted[label]
        for name in METRICS:
            expected = projected[label]["metrics"][name] if admitted[label] else None
            assert optional_number(row[name]) == expected, (label, name, row[name], expected)
        if admitted[label]:
            assert row["higher_moment_status"] == "UNDEFINED_ONE_MODEL_DAY"
        else:
            assert row["higher_moment_status"] == "INCOMPLETE_OR_UNANALYZED_WORLD"
        assert all(row[name] == "" for name in ("excess_kurtosis", "raw_return_acf", "absolute_return_acf"))
        lookup[(int(row["seed"]), row["baseline"])] = row
    assert len(lookup) == 9
    expected_pair_keys = {(seed, "full-minus-" + baseline, name)
                          for seed in plan["seed_order"] for baseline in plan["baseline_order"][1:]
                          for name in METRICS}
    assert len(paired) == len(expected_pair_keys) == 42
    assert {(int(row["seed"]), row["comparison"], row["metric"]) for row in paired} == expected_pair_keys
    pair_denominators = Counter()
    for row in paired:
        seed, name = int(row["seed"]), row["metric"]
        baseline = row["comparison"].removeprefix("full-minus-")
        full = optional_number(lookup[(seed, "full")][name])
        control = optional_number(lookup[(seed, baseline)][name])
        difference = None if full is None or control is None else full - control
        assert optional_number(row["full"]) == full and optional_number(row["control"]) == control
        assert optional_number(row["difference"]) == difference
        assert (row["both_complete"] == "True") == (difference is not None)
        if difference is not None:
            pair_denominators[(row["comparison"], name)] += 1
    assert len(descriptive) == len(plan["baseline_order"]) * len(METRICS) == 21
    expected_summary_keys = {(baseline, name) for baseline in plan["baseline_order"] for name in METRICS}
    assert {(row["baseline"], row["metric"]) for row in descriptive} == expected_summary_keys
    for row in descriptive:
        values = [optional_number(lookup[(seed, row["baseline"])][row["metric"]]) for seed in plan["seed_order"]]
        available = [value for value in values if value is not None]
        assert int(row["planned_worlds"]) == 3 and int(row["observed_worlds"]) == len(available)
        expected = {"mean": math.fsum(available) / len(available) if available else None,
                    "sample_sd": statistics.stdev(available) if len(available) > 1 else None,
                    "minimum": min(available) if available else None,
                    "maximum": max(available) if available else None}
        for key, value in expected.items():
            assert optional_number(row[key]) == value, (row["baseline"], row["metric"], key)
    return {"run_rows": 9, "paired_rows": 42, "descriptive_rows": 21,
            "planned_independent_seed_units": 3, "trajectories_per_seed": 3,
            "complete_rows": sum(admitted.values()), "zero_filled": False,
            "pair_observed_seed_counts": {comparison + "/" + name: pair_denominators[(comparison, name)]
                                          for comparison, name in sorted({(r["comparison"], r["metric"]) for r in paired})},
            "formal_H1_H2_H3_completed": False}
