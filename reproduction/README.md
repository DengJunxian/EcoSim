# Reproduce EcoSim

Run the commands from the repository root. Each entry point takes a new output directory.

## Verify the published data

Python 3.11 or later is sufficient.

```sh
python3 reproduction/verify.py --output ../ecosim-verified
```

The script recomputes health outcomes and paired comparisons, checks integer ledger postings and fees, and verifies action signatures, token counts and schedule exposures. Outputs include a JSON summary and the rebuilt model pairs.

## Rebuild the figures and tables

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r paper/source/scripts/requirements-figures.txt
python reproduction/rebuild.py --output ../ecosim-rebuilt
```

The build runs the data verifier, rebuilds the manuscript charts and result tables in a copy of the LaTeX source, checks the generated tables against the recorded hashes, and redraws the homepage figures. It writes the manuscript copy to `paper/`, the homepage art to `homepage/`, and numeric checks to `verification/` inside the output directory. The [matplotlib figure specifications](../docs/figures/design.md) describe the displayed samples and units. The original [paper renderer](../paper/source/scripts/render_real_figures.py) and [homepage renderer](render_readme.py) can also be run individually.

## Replay the historical state archives

Use [`replay_archives.py`](replay_archives.py) with the extracted **Civitas_KDD27** historical bundle delivered with the 15 November methods-and-pilot package. It expects the bundle root with its original `MANIFEST.json`, `paper/` and `reproduction/`. The delivered wheelhouses target Apple Silicon macOS and Python 3.12. Set `BUNDLE` to the extracted root and install the bundled dependencies in a separate environment.

```sh
BUNDLE=/absolute/path/to/Civitas_KDD27
python3.12 -m venv ../ecosim-archives-venv
PY=../ecosim-archives-venv/bin/python
"$PY" -m pip install --no-index \
  --find-links "$BUNDLE/reproduction/runtime/wheelhouse" \
  --find-links "$BUNDLE/reproduction/runtime/render-wheelhouse" \
  -r reproduction/runtime/requirements-core-build.lock \
  -r reproduction/runtime/requirements-render.lock
"$PY" reproduction/replay_archives.py --bundle "$BUNDLE" \
  --full --output ../ecosim-historical-replay
```

This route verifies the archive files, recomputes the 18 historical rule projections, checks market-health outcomes, and regenerates the earlier diagnostic plots. It uses a Python network guard and preserves the original archive. The [published analysis tools](tools/) include raw-state extraction, health archive auditing, scientific-field comparison, closeout tables and historical plot generation. The [frozen simulator source](runtime/source/) and [dependency lock](runtime/requirements-core-build.lock) document the executed runtime.

## Data map

| Directory | Contents |
| --- | --- |
| [`health/`](health) | Nine trajectory configurations, per-step projections, ledgers and matched-seed reference tables |
| [`legacy/reference/`](legacy/reference) | Projections for 18 rule runs, including the policy branches and timing pairs, plus status records for fixtures and planned runs |
| [`model_pilot/`](model_pilot) | Case and pair actions, reported usage, input consistency and event scheduling |
| [`capacity/`](capacity) | Timings, status, memory measurements and ledgers for three capacity units |
| [`historical_replay/`](historical_replay) | Rule replay configuration and historical audit receipts |
| [`plans/`](plans) | Run plans and metric specifications |
| [`paper/source/source_data/`](../paper/source/source_data) | Paper figure inputs, variable definitions and provenance |

Rebuilding the current paper uses the public projections and ledgers above. Raw model requests and responses, vendor market data and path-bound state archives stay with the original project. Resuming the original worlds requires the corresponding controlled archives and matching runtime.
