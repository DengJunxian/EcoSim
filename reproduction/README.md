# Experimental artifacts

| Directory | Contents |
|---|---|
| `health/` | Configuration, per-step data, receipts, terminal ledgers and reference tables for nine trajectories across three seeds and three baselines. |
| `legacy/reference/` | Projections and analysis tables for 18 rule archives, including the policy branches and paired validator timings. The status tables retain four mock records and the unexecuted formal plans. |
| `model_pilot/` | Case- and pair-level action records, reported usage and rate-based accounting, input-consistency diagnostics and event-step exposure counts. |
| `capacity/` | Recorded timings, status, memory measurements and terminal ledgers for three capacity units. |
| `historical_replay/` | Rule-replay configuration and audit records for the historical state and model response comparisons. |
| `runtime/` | Frozen simulator source snapshot and dependency versions. |
| `plans/` | Original run plans and metric specifications. |

`verify.py` recomputes health outcomes and pair comparisons, validates integer ledger postings and fees, and recomputes the fixed-observation action contrasts and usage totals. Runtime timing measurements are retained as observations. The plotting script uses the inputs in `paper/source/source_data`.

The public data consist of scientific projections, simulated terminal ledgers, configurations and supporting receipts. Historic paths and archive digests identify the source evidence. Raw provider messages, controlled logs, market-vendor files and path-bound state archives remain with the original project. Rebuilding the figures and recalculating the published comparisons use the files here. Resuming the original worlds directly requires the retained source archives and matching runtime.
