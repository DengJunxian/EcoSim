#!/usr/bin/env python3
"""Verify the bundle and rebuild results from archived data. No simulation or provider calls."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(TOOLS))
from health_final_arithmetic import check_tables, ledger_projection, project, read_csv

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")

def same(actual, expected, label):
    a = json.dumps(actual, sort_keys=True, allow_nan=False)
    b = json.dumps(expected, sort_keys=True, allow_nan=False)
    assert a == b, label

def verify_bundle():
    manifest = read(BUNDLE / "MANIFEST.json")
    for name, ref in manifest["files"].items():
        path = BUNDLE / name
        assert path.resolve().is_relative_to(BUNDLE) and not path.is_symlink(), name
        assert path.stat().st_size == ref["bytes"] and sha(path) == ref["sha256"], name
    for name in manifest["empty_directories"]:
        assert (BUNDLE / name).is_dir(), name
    return len(manifest["files"])

def offline(script, args, output, *, plot=False):
    source = HERE / "runtime/source"
    python_path = os.pathsep.join(map(str, (HERE / "runtime/offline", source / "src", source)))
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
           if key in os.environ}
    log = output / "offline-network.log"
    env.update(PYTHONPATH=python_path, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", WP00_NET_GUARD_LOG=str(log))
    if plot:
        env["MPLCONFIGDIR"] = str(output / "matplotlib-config")
    command = [sys.executable, str(script), *map(str, args)]
    subprocess.run(command, cwd=output, env=env, check=True)
    guard = log.read_text().splitlines()
    assert guard and set(guard) == {"guard-installed"}, log

def check_archive(root, audit, expected_ticks, *, initial_tick=0, financial=True):
    identity = read(root / "identity.json")
    assert identity["config"]["cognition_mode"] == "rule"
    markers = [read(path) for path in sorted((root / "commits").glob("*.json"))]
    assert [m["tick"] for m in markers] == list(range(initial_tick, expected_ticks + 1))
    assert markers[0]["kind"] == "initial" and markers[0]["parent_commit_hash"] is None
    for previous, current in zip(markers, markers[1:]):
        assert previous["commit_hash"] == current["parent_commit_hash"]
        assert current["identity_hash"] == markers[0]["identity_hash"] and current["kind"] == "tick"
    assert all(m["model_facts"] is None and not m["invocation_receipt_refs"] for m in markers)
    head = markers[-1]
    assert head["state_hash"] == audit["state_hash"] and identity["run_id"] == audit["run_id"]
    checkpoints = []
    for path in (root / "checkpoints").glob("*.json.gz"):
        assert sha(path) == path.name.removesuffix(".json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            checkpoint = json.load(f)
        marker = markers[checkpoint["tick"] - initial_tick]
        assert all(checkpoint[key] == marker[key] for key in ("tick", "commit_hash", "identity_hash", "state_hash"))
        if checkpoint["tick"] == expected_ticks:
            checkpoints.append(checkpoint["snapshot"])
    assert len(checkpoints) == 1 and checkpoints[0]["tick"] == expected_ticks, root
    ledger = None
    if financial:
        snapshot = checkpoints[0]
        ledger = ledger_projection(snapshot["data"]["multiasset_market"]["ledger"],
                                   snapshot["actors"], identity["config"]["market"]["currency"])
    return markers, ledger

def check_health(out):
    root = HERE / "health"
    projections = root / "projections"
    plan = read(root / "plan.json")
    assert sha(root / "plan.json") == (root / "plan.sha256").read_text().strip()
    projected, admitted = {}, {}
    activations = defaultdict(list)
    for job in plan["jobs"]:
        label = job["label"]
        p = projections / (label + "-analysis.json")
        analysis = read(p)
        audit = read(projections / (label + "-audit-result.json"))
        config = root / "configs" / Path(job["config_file"]).name
        assert sha(config) == job["config_file_sha256"]
        marker, ledger = check_archive(root / label / "run", audit, 100)
        expected = [{"tick": m["tick"], "state_hash": m["state_hash"], "commit_hash": m["commit_hash"]}
                    for m in marker[1:]]
        assert analysis["committed_hashes"] == expected
        finance = audit["financial_checks"]
        for key in ("cash_residual_minor", "asset_residual_units", "fees", "reservations", "minimum_available_by_resource"):
            assert ledger[key] == finance[key], (label, key)
        projected[label] = project(analysis, read(config))
        admitted[label] = finance["status"] == "passed_existing_runtime_validation"
        activations[job["seed"]].append({(d["tick"], d["actor_id"]): d["active"] for d in analysis["decisions"]})
    assert all(len(values) == 3 and values[0] == values[1] == values[2] for values in activations.values())
    offline(TOOLS / "health_tables_closeout.py", [root, out / "health", "--results", projections,
                                                   "--stage-controls", projections / "controllers"], out)
    for name in ("run.csv", "paired.csv", "descriptive.csv", "financial.csv"):
        assert sha(out / "health" / name) == sha(root / "reference" / name), name
    receipt = check_tables(out / "health", plan, projected, admitted)
    assert receipt["complete_rows"] == 9
    with zipfile.ZipFile(BUNDLE / "paper/Civitas_KDD27_Overleaf.zip") as z:
        manuscript_values = json.loads(z.read("derived/health_outcomes.json"))
    for row in manuscript_values["rows"]:
        label = f"seed-{row['seed']}-{row['baseline']}"
        for key in ("no_trade_ratio", "two_sided_coverage"):
            assert row[key] == projected[label]["metrics"][key]
    receipt["terminal_ledger_checks"] = 9
    return receipt

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

def check_capacity():
    root = HERE / "capacity"
    runs = read(root / "capacity-results.json")["capacity_runs"]
    for run in runs:
        src = root / run["label"]
        audit = read(src / "audit-result.json")
        controller = read(src / "controller.json")
        markers, ledger = check_archive(src / "run", audit, run["committed_ticks"])
        assert controller["status"] == run["controller_status"]
        assert controller["elapsed_seconds"] == run["total_seconds"]
        assert len(markers) == audit["verified_markers"]
        assert sum(p.stat().st_size for p in (src / "run").rglob("*") if p.is_file()) == run["archive_bytes"]
        measurement = src / "measurement.json"
        if measurement.is_file():
            measured = read(measurement)
            assert measured["total"]["wall_seconds"] == run["full_profile_seconds"]
            assert measured["peak_rss_bytes"] == run["peak_rss_bytes"]
            assert measured["checkpoint"]["wall_seconds"] == run["checkpoint_wall_seconds"]
        else:
            assert run["measurement_json_present"] is False and controller["exit_code"] == -2
            assert controller["sampled_peak_rss_bytes"] == run["peak_rss_bytes"]
        assert ledger["cash_residual_minor"] == ledger["fees"]["residual_minor"] == 0
    return {r["label"]: {"steps": r["committed_ticks"], "controller_status": r["controller_status"]} for r in runs}

def check_historical():
    source = HERE / "historical_replay"
    receipt = read(source / "reproduction-check.json")
    terminals = set()
    for label in ("continuous", "split", "none"):
        expected = receipt["states_by_run"][label]
        audit = {"run_id": receipt["run_ids"][label], "state_hash": expected[-1]["state_hash"]}
        markers, _ = check_archive(source / "runs" / label, audit, 12, initial_tick=expected[0]["tick"])
        assert [{key: marker[key] for key in ("tick", "state_hash", "commit_hash")}
                for marker in markers] == expected
        terminals.add(markers[-1]["state_hash"])
    assert len(terminals) == 1
    tape = receipt["real_tape"]
    assert tape["cases"] == 20 and tape["all_semantic_requests_responses_and_parsed_equal"]
    assert tape["all_case_results_equal"] is False
    return {"rule_terminal_states_matched": 3, "terminal_state_hash": terminals.pop(),
            "recorded_model_response_replay": "20/20",
            "recorded_final_decision_exact_match": "19/20",
            "raw_model_tape_reexecuted": False}

def portable_selection(root, out):
    frozen = root / "selection.frozen.json"
    selection = read(frozen)
    old_scope = Path(selection["supersedes"]["path"]).parent.parent
    replacements = {}

    def new_path(old):
        path = root / "inputs" / Path(old).relative_to(old_scope)
        assert path.exists(), path
        replacements[old] = str(path)
        return str(path)

    def remap(ref):
        ref["path"] = new_path(ref["path"])

    for entry in selection["runs"]:
        entry["record"]["raw_path"] = new_path(entry["record"]["raw_path"])
        remap(entry["registry_source"])
        for ref in entry["archive_files"]:
            remap(ref)
    for entry in selection["planned_families"]:
        remap(entry["source"])
    for entry in selection["performance"]:
        remap(entry["measurement"])
    for group in ("historical_sources", "method_sources", "paper_sources", "analysis_sources"):
        for ref in selection[group]:
            remap(ref)
    save(out, selection)
    return replacements

def rebuild_legacy(out):
    root = HERE / "legacy"
    selection = out / "selection.relocated.json"
    replacements = portable_selection(root, selection)
    results = out / "legacy"
    offline(TOOLS / "kdd27_measured.py",
            ["--selection", selection, "--selection-sha256", sha(selection), "--out", results, "--scope", "engineering"], out)
    receipt = read(results / "BUILD.json")
    reference = root / "reference"
    reference_build = read(reference / "BUILD.json")
    assert set(receipt["files"]) == set(reference_build["files"])
    assert receipt["analysis_script_sha256"] == reference_build["analysis_script_sha256"]
    assert all(receipt[key] == reference_build[key] for key in
               ("recomputed_runs", "registered_runs", "mock_register_only", "planned_rows",
                "new_steps", "new_model_calls", "formal_figures_built", "archive_files_unchanged"))
    assert receipt["planned_rows"] == 920
    rows = read(results / "planned_rows.json")
    assert len(rows) == 920 and all(not row["attempts"] for row in rows)
    path_columns = {"failure.csv": ("source",), "run_level.csv": ("raw_path", "raw_manifest_hash"),
                    "claim_evidence.csv": ("selection_sha256",)}
    recorded_hashes = {}
    exact, relocated = [], []
    for name in sorted(receipt["files"]):
        original, rebuilt = reference / name, results / name
        if name.startswith("projections/"):
            a, b = read(original), read(rebuilt)
            a["run"]["raw_path"] = replacements[a["run"]["raw_path"]]
            h0, h1 = a.pop("source_archive_manifest_hash"), b.pop("source_archive_manifest_hash")
            recorded_hashes[a["run"]["run_id"]] = (h0, h1)
            same(a, b, name)
            relocated.append(name)
        elif name == "selected_runs.json":
            a = read(original)
            for row in a:
                row["raw_path"] = replacements[row["raw_path"]]
            same(a, read(rebuilt), name)
            relocated.append(name)
        elif name == "traces.json":
            a = read(original)
            for item in a:
                for row in item.get("source_measurements", []):
                    ref = row["measurement"]
                    ref["path"] = replacements[ref["path"]]
            same(a, read(rebuilt), name)
            relocated.append(name)
        elif name not in path_columns:
            assert sha(original) == sha(rebuilt), name
            exact.append(name)
    for name, fields in path_columns.items():
        a, b = read_csv(reference / name), read_csv(results / name)
        assert len(a) == len(b)
        for old, new in zip(a, b):
            for field in fields:
                if field == "raw_manifest_hash":
                    pair = recorded_hashes[old["run_id"]]
                    assert (old[field], new[field]) == pair
                    old[field] = pair[1]
                elif field == "selection_sha256":
                    assert old[field] == sha(root / "selection.frozen.json") and new[field] == sha(selection)
                    old[field] = new[field]
                else:
                    old[field] = replacements[old[field]]
            assert old == new, name
        relocated.append(name)
    assert len(recorded_hashes) == 18
    return {"archives_recomputed": 18, "mock_registry_only": 4, "formal_rows_not_run": 920,
            "byte_identical_files": exact, "files_equal_after_path_or_derived_hash_remap": relocated}

def rebuild_figures(out):
    root = out / "health-figure"
    root.mkdir()
    for name in ("draw_health_figure.py", "health_figure_contract.json"):
        (root / name).write_bytes((TOOLS / name).read_bytes())
    paper = root / "overleaf_draft"
    (paper / "derived").mkdir(parents=True)
    (paper / "figures").mkdir()
    plotted = paper / "derived/health_outcomes.json"
    with zipfile.ZipFile(BUNDLE / "paper/Civitas_KDD27_Overleaf.zip") as z:
        plotted.write_bytes(z.read("derived/health_outcomes.json"))
        offline(root / "draw_health_figure.py", [plotted], out, plot=True)
        health_matches = {}
        for ext in ("pdf", "svg", "png"):
            name = f"figures/health_diagnostics.{ext}"
            health_matches[ext] = (paper / name).read_bytes() == z.read(name)
        legacy = out / "legacy"
        offline(TOOLS / "render_kdd27_measured.py",
                ["--build", legacy, "--build-sha256", sha(legacy / "BUILD.json"), "--out", out / "legacy-figures"],
                out, plot=True)
        archive_matches = {}
        for stem in ("engineering_liquidity", "engineering_branches", "engineering_timing"):
            archive_matches[stem] = {ext: (out / "legacy-figures" / f"{stem}.{ext}").read_bytes() ==
                                     z.read(f"figures/{stem}.{ext}") for ext in ("pdf", "svg")}
    return {"health_matches_source_bytes": health_matches, "legacy_matches_source_bytes": archive_matches,
            "method_diagram_editable_svg_provided": True}

def main():
    global BUNDLE, HERE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Extracted historical Civitas_KDD27 bundle")
    parser.add_argument("--output", type=Path, required=True, help="New folder outside the extracted bundle")
    parser.add_argument("--full", action="store_true", help="Also re-analyze 18 immutable archives and rebuild measured plots")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run without Python -O so the verification assertions are active")
    BUNDLE = args.bundle.resolve()
    HERE = BUNDLE / "reproduction"
    out = args.output.resolve()
    if out.is_relative_to(BUNDLE):
        parser.error("Use an output folder outside the extracted bundle")
    verified = verify_bundle()
    out.mkdir(parents=True, exist_ok=False)
    print(f"Verified {verified} bundled artifacts", flush=True)
    evidence = {"verified_input_files": verified, "health": check_health(out),
                "model_pilot": check_model_pilot(out), "capacity": check_capacity(),
                "historical_replay": check_historical(),
                "formal_worlds_added": 0, "new_model_calls": 0,
                "route": "archived computation" if args.full else "existing projections and terminal archives"}
    if args.full:
        evidence["legacy"] = rebuild_legacy(out)
        evidence["plots"] = rebuild_figures(out)
    save(out / "reproduction-receipt.json", evidence)
    print(f"PASS. Receipt: {out / 'reproduction-receipt.json'}")

if __name__ == "__main__":
    main()
