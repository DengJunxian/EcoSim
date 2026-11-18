"""Structured-output engineering rates; these are not behavioral realism scores."""
import math


def invocation_metrics(records):
    groups = {}
    for row in records:
        groups.setdefault(row['request_id'], []).append(row)
    latencies = sorted(row['latency_ms'] for row in records)
    def quantile(p):
        return latencies[math.ceil(p * len(latencies)) - 1] if latencies else None
    logical = len(groups)
    return {'logical_requests_with_attempts': logical, 'attempts': len(records),
        'first_valid_rate': sum(min(g, key=lambda r: r['number'])['parsed'] is not None for g in groups.values()) / logical if logical else None,
        'final_valid_rate': sum(any(r['parsed'] is not None for r in g) for g in groups.values()) / logical if logical else None,
        'attempt_failure_rate': sum(r['error'] is not None for r in records) / len(records) if records else None,
        'input_tokens_recorded': sum(r['incoming'] or 0 for r in records),
        'output_tokens_recorded': sum(r['outgoing'] or 0 for r in records),
        'unknown_usage_attempts': sum(r['incoming'] is None for r in records),
        'live_attempts_in_records': sum(r['request']['configuration']['source'] == 'live' for r in records),
        'latency_p50_ms': quantile(0.50), 'latency_p95_ms': quantile(0.95),
        'quantile_definition': 'nearest_rank_per_attempt', 'behavioral_realism': 'not_evaluated'}
