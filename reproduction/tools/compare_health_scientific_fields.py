#!/usr/bin/env python3
"""Compare existing health analysis JSON only; never import or run a simulator.

Scientific equality means exact parsed JSON values (including numeric types and
the sign of floating zero), not identical source-file bytes. No tolerance is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCIENTIFIC_KEYS = (
    "facts", "decisions", "trades", "committed_hashes", "summary", "model_cost",
    "analysis_is_new_simulation", "formal_eligible",
)
EXPECTED_TOP = set(SCIENTIFIC_KEYS) | {"run", "analysis_receipt"}
OPERATIONAL_RUN = {"raw_path"}
OPERATIONAL_RECEIPT = {"wall_seconds", "cpu_seconds", "peak_rss_bytes"}
MISSING = object()


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant is not accepted: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def read_json(path):
    raw = path.read_bytes()
    obj = json.loads(raw, object_pairs_hook=unique_object,
                     parse_constant=reject_constant)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a top-level object: {path}")
    return obj, hashlib.sha256(raw).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def pointer(path, key):
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def value_record(value):
    if value is MISSING:
        return {"present": False}
    out = {"present": True, "json_type": type(value).__name__, "value": value}
    if type(value) is float:
        out["float_hex"] = value.hex()
    return out


def differences(left, right, path=""):
    if left is MISSING or right is MISSING:
        yield {"path": path, "reason": "missing", "left": value_record(left),
               "right": value_record(right)}
        return
    if type(left) is not type(right):
        yield {"path": path, "reason": "type", "left": value_record(left),
               "right": value_record(right)}
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            yield from differences(left.get(key, MISSING), right.get(key, MISSING),
                                   pointer(path, key))
    elif isinstance(left, list):
        for i in range(max(len(left), len(right))):
            yield from differences(left[i] if i < len(left) else MISSING,
                                   right[i] if i < len(right) else MISSING,
                                   pointer(path, i))
    elif (left.hex() != right.hex() if type(left) is float else left != right):
        yield {"path": path, "reason": "value", "left": value_record(left),
               "right": value_record(right)}


def sections(obj):
    absent = sorted(EXPECTED_TOP - set(obj))
    if absent:
        raise ValueError(f"Missing required analysis sections: {absent}")
    if not isinstance(obj["run"], dict) or not isinstance(obj["analysis_receipt"], dict):
        raise ValueError("run and analysis_receipt must be JSON objects")
    return {
        "scientific": {key: obj[key] for key in SCIENTIFIC_KEYS},
        "run_identity": {key: val for key, val in obj["run"].items()
                         if key not in OPERATIONAL_RUN},
        "analysis_contract": {key: val for key, val in obj["analysis_receipt"].items()
                              if key not in OPERATIONAL_RECEIPT},
        "operational": {
            "run": {key: obj["run"][key] for key in OPERATIONAL_RUN if key in obj["run"]},
            "analysis_receipt": {key: obj["analysis_receipt"][key]
                                 for key in OPERATIONAL_RECEIPT if key in obj["analysis_receipt"]},
        },
        # New top-level fields are checked, never silently dropped.
        "other_top_level": {key: val for key, val in obj.items() if key not in EXPECTED_TOP},
    }


def compare_objects(left, right):
    a, b = sections(left), sections(right)
    output = []
    for category in a:
        output.extend({"category": category, **row}
                      for row in differences(a[category], b[category]))
    counts = Counter(row["category"] for row in output)
    all_counts = {key: counts[key] for key in a}
    nonoperational = sum(value for key, value in all_counts.items() if key != "operational")
    return {
        "status": ("EXACT_SCIENTIFIC_AND_CONTRACT_MATCH" if nonoperational == 0
                   else "EXACT_DIFFERENCES_RETAINED"),
        "difference_counts": all_counts,
        "difference_count_total": len(output),
        "science_exact_equal": counts["scientific"] == 0,
        "run_identity_exact_equal": counts["run_identity"] == 0,
        "analysis_contract_exact_equal": counts["analysis_contract"] == 0,
        "left_scientific_projection_sha256": digest(a["scientific"]),
        "right_scientific_projection_sha256": digest(b["scientific"]),
    }, output


def selfcheck():
    cases = {
        "tiny_float_difference": ({"v": 1.0}, {"v": 1.0000000000000002}, 1),
        "negative_zero": ({"v": 0.0}, {"v": -0.0}, 1),
        "integer_float_type": ({"v": 1}, {"v": 1.0}, 1),
        "bool_integer_type": ({"v": True}, {"v": 1}, 1),
        "missing_is_not_null": ({}, {"v": None}, 1),
        "list_order_matters": ([1, 2], [2, 1], 2),
        "object_key_order_irrelevant": ({"a": 1, "b": 2}, {"b": 2, "a": 1}, 0),
        "no_missing_tail_truncation": ([1, 2, 3], [1], 2),
    }
    results = {}
    for name, (a, b, count) in cases.items():
        actual = len(list(differences(a, b)))
        assert actual == count, (name, actual, count)
        results[name] = {"difference_count": actual, "expected": count, "passed": True}
    dummy = {key: [] for key in SCIENTIFIC_KEYS}
    dummy.update(run={"run_id": "test", "raw_path": "left"},
                 analysis_receipt={"wall_seconds": 1.0, "status": "passed"})
    other = copy.deepcopy(dummy)
    other["run"]["raw_path"] = "right"
    other["analysis_receipt"]["wall_seconds"] = 2.0
    result, _ = compare_objects(dummy, other)
    assert result["status"] == "EXACT_SCIENTIFIC_AND_CONTRACT_MATCH"
    assert result["difference_counts"]["operational"] == 2
    results["operational_separation"] = {"passed": True, "differences": 2}
    other["run"]["run_id"] = "changed"
    result, _ = compare_objects(dummy, other)
    assert result["difference_counts"]["run_identity"] == 1
    results["identity_drift_detected"] = {"passed": True}
    for value, name in [(' {"a":1,"a":2}', "duplicate_key_rejected"),
                        (' {"a":NaN}', "nonfinite_rejected")]:
        try:
            json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except ValueError:
            results[name] = {"passed": True}
        else:
            raise AssertionError(name)
    return {"status": "PASS", "checks": results, "check_count": len(results),
            "actual_analysis_comparisons": 0, "world_constructions": 0,
            "forward_steps": 0, "provider_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    audit_directory = Path(__file__).resolve().parent
    if output.parent != audit_directory:
        parser.error("Outputs must be new files in this review directory")
    difference_path = output.with_suffix(".differences.jsonl")
    if output.exists() or difference_path.exists():
        parser.error("Refusing to overwrite existing review evidence")
    result = {"reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
              "comparator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.selfcheck:
        if args.left or args.right:
            parser.error("Selfcheck does not accept actual analysis inputs")
        result.update(selfcheck())
    else:
        if args.left is None or args.right is None:
            parser.error("Both --left and --right are required")
        result["inputs"] = {"left": str(args.left.resolve()), "right": str(args.right.resolve())}
        if not args.left.is_file() or not args.right.is_file():
            result.update(status="PENDING_INPUTS",
                          missing=[key for key, path in (("left", args.left), ("right", args.right))
                                   if not path.is_file()])
        else:
            left, left_hash = read_json(args.left)
            right, right_hash = read_json(args.right)
            comparison, all_differences = compare_objects(left, right)
            result.update(comparison, left_file_sha256=left_hash, right_file_sha256=right_hash,
                          source_files_byte_identical=left_hash == right_hash)
            with difference_path.open("x", encoding="utf-8") as handle:
                for item in all_differences:
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            result["differences_file"] = str(difference_path)
            result["differences_sha256"] = hashlib.sha256(difference_path.read_bytes()).hexdigest()
            assert hashlib.sha256(args.left.read_bytes()).hexdigest() == left_hash
            assert hashlib.sha256(args.right.read_bytes()).hexdigest() == right_hash
            result["input_files_unchanged"] = True
        result.update(financial_admission="NOT_DECIDED_BY_THIS_COMPARISON",
                      tolerance_applied=False, world_constructions=0, analyze_run_calls=0,
                      forward_steps=0, provider_calls=0, new_independent_seeds=0,
                      actual_analysis_dependency_identity="Use bound environment/launch sidecars; runtime_dependency_hash belongs to the archive")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
