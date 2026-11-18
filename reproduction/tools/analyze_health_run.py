"""Use the frozen KDD27 projections for one complete initial-transient health pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import signal
import time

from civitas.analytics.kdd27_data import analyze_run
from civitas.application.run import implementation_identity
from civitas.storage.commit_store import CommitStore


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("label")
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    assert not args.output.exists()
    plan_raw = args.plan.read_bytes()
    assert hashlib.sha256(plan_raw).hexdigest() == args.plan.with_suffix(".sha256").read_text().strip()
    plan = json.loads(plan_raw)
    assert implementation_identity(financial=True)["source_hash"] == plan["source_hash"]
    job = next(job for job in plan["jobs"] if job["label"] == args.label)
    store = CommitStore(args.archive)
    assert store.identity.config_hash == job["config_hash"]
    assert store.identity.code_hash == plan["source_hash"]
    before = hashes(args.archive)
    start, cpu = time.perf_counter(), time.process_time()
    result = analyze_run(args.archive, experiment_id=plan["campaign_protocol_id"],
                         condition_id=job["baseline"], protocol_version=plan["runtime_protocol_id"])
    result["run"]["execution_phase"] = "pilot"
    assert result["run"]["status"] == "completed" and result["run"]["world_seed"] == job["seed"]
    assert [row["tick"] for row in result["facts"]] == list(range(1, 101))
    assert len(result["decisions"]) == 10000
    assert len(result["summary"]["daily_statistics"]["returns"]) == 1
    for name in ("excess_kurtosis", "mean_abs_return_acf_lags_1_20",
                 "maximum_abs_raw_return_acf_lags_1_20", "volume_abs_return_correlation"):
        assert result["summary"]["metrics"][name]["status"] == "UNDEFINED"
        assert result["summary"]["metrics"][name]["value"] is None
    assert len(result["trades"]) == sum(row["trade_count"] for row in result["facts"])
    assert sum(row["fees_step_minor"] for row in result["facts"]) == result["facts"][-1]["fees_cumulative_minor"]
    assert result["model_cost"]["owned_attempts"] == 0
    assert hashes(args.archive) == before
    result["analysis_receipt"] = {
        "status": "passed", "campaign_protocol_id": plan["campaign_protocol_id"],
        "runtime_protocol_id": plan["runtime_protocol_id"],
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(), "frozen_source_commit": plan["base_commit"],
        "analysis_source_hash": plan["source_hash"], "runtime_dependency_hash": store.identity.metadata["implementation"]["dependency_hash"],
        "wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "archive_files_unchanged": len(before), "archive_hashes": before,
        "archive_bytes_written": 0, "forward_steps": 0, "new_paid_calls": 0,
        "scope": "One model day from initialization; engineering health description, not H1 or power confirmation."}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "passed", "label": args.label, "run_id": result["run"]["run_id"],
                      "metrics": result["summary"]["metrics"], "daily": result["summary"]["daily_statistics"]}))


if __name__ == "__main__":
    main()
