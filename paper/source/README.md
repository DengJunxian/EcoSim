# Civitas Economica

The English manuscript and its Chinese counterpart share one bibliography, one set of data tables, and the same empirical figures.

## Compile in Overleaf

Upload this ZIP as a new project. Select TeX Live 2026 where available. Change the main document and compiler together when switching languages.

| Manuscript | Main document | Compiler |
| --- | --- | --- |
| English | main.tex | pdfLaTeX |
| Chinese | main_zh.tex | XeLaTeX |

The Chinese manuscript uses Fandol fonts from TeX Live. The project includes all vector PDFs and data tables required to compile either manuscript.

## Reproduce the figures

With NumPy and Matplotlib available, run

    python scripts/render_real_figures.py

The script reads the packaged archive projections and rebuilds the empirical figures and generated result tables. It produces vector PDF and SVG figures and 300 dpi PNG previews. The architecture figure is supplied separately as an editable SVG, a vector PDF, and a PNG preview. See source_data/README.md for variable definitions and archive provenance.
