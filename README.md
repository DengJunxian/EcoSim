# EcoSim

This repository accompanies *Civitas Economica: An Interdisciplinary LLM Agent-Based Market Simulation Architecture for Multi-Scale Socio-Economic Dynamics*. It contains the bilingual manuscript, plotting inputs, and offline materials for the experiments reported in the paper.

- [English manuscript](paper/Civitas_KDD27_EN.pdf) and [Chinese manuscript](paper/Civitas_KDD27_ZH.pdf)
- [Overleaf ZIP](paper/Overleaf.zip) and [editable LaTeX files](paper/source)
- [Figure data and provenance](paper/source/source_data/README.md)
- [Experiment data and verification guide](reproduction/README.md)

The paper reports nine short market-health trajectories, eleven branches from a shared parent, 54 fixed-observation model calls, and capacity and timing diagnostics. The formal three-RQ experiment matrix remains planned.

## Build and verify

In Overleaf, use `main.tex` with pdfLaTeX for English or `main_zh.tex` with XeLaTeX for Chinese. The source directory includes the vector figures and generated result tables.

Run the numeric checks with Python 3.11 or later. Use a new output directory.

```sh
python3 reproduction/verify.py --output ../ecosim-verified
```

To regenerate the figures and result tables, install the NumPy and Matplotlib requirements and run the renderer in a working copy.

```sh
python3 -m pip install -r paper/source/scripts/requirements-figures.txt
python3 paper/source/scripts/render_real_figures.py
```
