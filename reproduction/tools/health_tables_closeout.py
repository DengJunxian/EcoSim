"""Retain every planned world and form descriptive paired health tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


METRICS = ("no_trade_ratio", "one_sided_ratio", "two_sided_coverage", "trade_count",
           "fees_minor", "daily_log_return", "traded_quantity")
READ_ERRORS = {}


def descriptive(values):
    observed = [value for value in values if value is not None]
    return {"observed_worlds": len(observed), "planned_worlds": len(values),
            "mean": statistics.fmean(observed) if observed else None,
            "sample_sd": statistics.stdev(observed) if len(observed) > 1 else None,
            "minimum": min(observed) if observed else None,
            "maximum": max(observed) if observed else None}


def difference(full, control):
    return None if full is None or control is None else full - control


def load_if(path):
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        READ_ERRORS[str(path)] = f"{type(exc).__name__}: {exc}"
        return None


def tables(root, plan, results_root=None, stage_controls=None):
    results_root = results_root or root
    stage_controls = stage_controls or root
    rows = []
    for job in plan["jobs"]:
        READ_ERRORS.clear()
        label = job["label"]
        archive = root / label / "run"
        identity = load_if(archive / "identity.json")
        controller = load_if(root / (label + ".json"))
        measurement = load_if(root / label / "measurement.json")
        analysis = load_if(results_root / (label + "-analysis.json"))
        audit = load_if(results_root / (label + "-audit-result.json"))
        audit_controller = load_if(stage_controls / ("closeout-audit-" + label + ".json"))
        analysis_controller = load_if(stage_controls / ("closeout-analysis-" + label + ".json"))
        finance = audit.get("financial_checks", {}) if audit else {}
        chain = audit.get("commit_chain", {}) if audit else {}
        model = audit.get("model_call_proof", {}) if audit else {}
        attempts = [value for path in sorted((archive / "attempts").glob("*.json")) if (value := load_if(path))]
        terminal = next((row for row in reversed(attempts) if "status" in row), {})
        markers = [value for path in (archive / "commits").glob("*.json") if (value := load_if(path))]
        row = {"label": label, "seed": job["seed"], "baseline": job["baseline"],
               "run_id": identity["run_id"] if identity else None,
               "config_hash": job["config_hash"], "runtime_protocol_id": plan["runtime_protocol_id"],
               "campaign_protocol_id": plan["campaign_protocol_id"],
               "controller_status": controller["status"] if controller else "NOT_RUN" if not identity else "NO_TERMINAL_RECEIPT",
               "runtime_status": terminal.get("status", "NOT_RUN"),
               "raw_status": terminal.get("status", "NOT_RUN"),
               "stop_reason": controller["stop_reason"] if controller else None,
               "runtime_error": terminal.get("error") or terminal.get("export_error"),
               "original_asus_postprocess_status": "NOT_RUN_PRESERVED",
               "postprocessing_phase": "post_window_readonly_closeout",
               "audit_controller_status": audit_controller["status"] if audit_controller else "NOT_RUN",
               "audit_stop_reason": audit_controller.get("stop_reason") if audit_controller else None,
               "analysis_controller_status": analysis_controller["status"] if analysis_controller else "NOT_RUN",
               "analysis_stop_reason": analysis_controller.get("stop_reason") if analysis_controller else None,
               "financial_audit_status": finance.get("status", "NOT_AUDITED_OR_FAILED"),
               "cash_residual_minor": finance.get("cash_residual_minor"),
               "asset_residual_units": json.dumps(finance["asset_residual_units"], sort_keys=True) if "asset_residual_units" in finance else None,
               "fee_residual_minor": finance.get("fees", {}).get("residual_minor"),
               "reservation_validation": finance.get("reservation_validation"),
               "commit_chain_status": chain.get("status", "NOT_AUDITED_OR_FAILED"),
               "head_state_hash": chain.get("head_state_hash"),
               "restore_status": audit.get("restoration", {}).get("status") if audit else None,
               "model_call_proof_status": model.get("status", "NOT_AUDITED_OR_FAILED"),
               "audited_model_calls": model.get("model_calls"),
               "original_network_guard_status": model.get("network_guard_status"),
               "requested_ticks": 100, "committed_ticks": max((marker["tick"] for marker in markers), default=0),
               "elapsed_seconds": controller["elapsed_seconds"] if controller else None,
               "peak_rss_bytes": max(measurement["peak_rss_bytes"] if measurement else 0,
                                     controller["sampled_peak_rss_bytes"] if controller else 0) if controller else None,
               "artifact_errors": json.dumps(READ_ERRORS, sort_keys=True) if READ_ERRORS else None,
               "analysis_status": "passed" if analysis and analysis.get("analysis_receipt", {}).get("status") == "passed" else "FAILED_ARTIFACT" if READ_ERRORS else "NOT_ANALYZED"}
        eligible = (row["controller_status"] == "completed" and row["runtime_status"] == "completed"
                    and row["committed_ticks"] == 100 and row["analysis_status"] == "passed" and not READ_ERRORS
                    and row["financial_audit_status"] == "passed_existing_runtime_validation"
                    and row["audit_controller_status"] == "completed" and row["audit_stop_reason"] is None
                    and row["analysis_controller_status"] == "completed" and row["analysis_stop_reason"] is None)
        row["complete_health_outcomes"] = eligible
        row.update({metric: None for metric in METRICS})
        row.update(excess_kurtosis=None, raw_return_acf=None, absolute_return_acf=None,
                   higher_moment_status="UNDEFINED_ONE_MODEL_DAY" if eligible else "INCOMPLETE_OR_UNANALYZED_WORLD")
        if eligible:
            assert analysis["run"]["run_id"] == row["run_id"]
            assert analysis["run"]["resolved_config_hash"] == job["config_hash"]
            assert analysis["analysis_receipt"]["campaign_protocol_id"] == plan["campaign_protocol_id"]
            for metric in METRICS:
                if metric == "trade_count":
                    value = sum(fact["trade_count"] for fact in analysis["facts"])
                elif metric == "daily_log_return":
                    returns = analysis["summary"]["daily_statistics"]["returns"]
                    assert len(returns) == 1
                    value = returns[0]
                else:
                    value = analysis["summary"]["metrics"][metric]["value"]
                row[metric] = value
            assert analysis["summary"]["metrics"]["excess_kurtosis"]["value"] is None
        rows.append(row)
    assert len(rows) == 9 and len({row["label"] for row in rows}) == 9
    lookup = {(row["seed"], row["baseline"]): row for row in rows}
    paired = []
    for seed in plan["seed_order"]:
        for baseline in plan["baseline_order"][1:]:
            for metric in METRICS:
                full, control = lookup[(seed, "full")][metric], lookup[(seed, baseline)][metric]
                paired.append({"seed": seed, "comparison": "full-minus-" + baseline, "metric": metric,
                               "full": full, "control": control, "difference": difference(full, control),
                               "both_complete": full is not None and control is not None})
    summaries = [{"baseline": baseline, "metric": metric,
                  **descriptive([lookup[(seed, baseline)][metric] for seed in plan["seed_order"]])}
                 for baseline in plan["baseline_order"] for metric in METRICS]
    return rows, paired, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stage-controls", type=Path, required=True)
    args = parser.parse_args()
    assert descriptive([0, 2, None]) == {"observed_worlds": 2, "planned_worlds": 3, "mean": 1.0,
                                       "sample_sd": 2**.5, "minimum": 0, "maximum": 2}
    assert difference(0, 2) == -2 and difference(0, None) is None
    plan_raw = (args.root / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    assert hashlib.sha256(plan_raw).hexdigest() == (args.root / "plan.sha256").read_text().strip()
    rows, paired, summaries = tables(args.root, plan, args.results, args.stage_controls)
    args.output.mkdir(exist_ok=False)
    finance_fields = ("label", "seed", "baseline", "run_id", "controller_status", "raw_status", "committed_ticks",
                      "audit_controller_status", "audit_stop_reason", "analysis_controller_status", "analysis_stop_reason",
                      "financial_audit_status", "cash_residual_minor", "asset_residual_units", "fee_residual_minor",
                      "reservation_validation", "commit_chain_status", "head_state_hash", "restore_status",
                      "model_call_proof_status", "audited_model_calls", "original_network_guard_status", "artifact_errors")
    financial = [{key: row[key] for key in finance_fields} for row in rows]
    for name, records in (("run", rows), ("paired", paired), ("descriptive", summaries), ("financial", financial)):
        with (args.output / (name + ".csv")).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    receipt = {"status": "passed", "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
               "postprocessing_phase": "post_window_readonly_closeout",
               "closeout_plan_sha256": hashlib.sha256((args.results / "plan.json").read_bytes()).hexdigest(),
               "original_asus_postprocess_status": "NOT_RUN_PRESERVED",
               "planned_worlds": 9, "complete_health_outcomes": sum(row["complete_health_outcomes"] for row in rows),
               "independent_seeds": plan["seed_order"], "same_seed_baselines_are_paired": True,
               "independent_seed_units": len(plan["seed_order"]), "trajectories_per_seed": 3,
               "zero_fill": False, "retries": 0, "new_simulations": 0,
               "scope": "Three raw world points per baseline; descriptive variation and paired differences only. No power confirmation, inference, or H1 claim.",
               "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.output.glob("*.csv")}}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
