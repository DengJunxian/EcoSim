"""Rebuild the published paper figures, result tables and homepage from this checkout."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for rebuilt artifacts")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run with assertions enabled.")
    out = args.output.resolve()
    manuscript = ROOT / "paper/source"
    if out.is_relative_to(manuscript):
        parser.error("Choose an output directory outside paper/source.")
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(ROOT / "reproduction/verify.py"),
                    "--output", str(out / "verification")], check=True)
    rebuilt = shutil.copytree(manuscript, out / "paper")
    subprocess.run([sys.executable, str(rebuilt / "scripts/render_real_figures.py")], check=True)
    expected = json.loads((manuscript / "source_data/render_receipt.json").read_text())
    actual = json.loads((rebuilt / "source_data/render_receipt.json").read_text())
    for key in ("source_hashes", "generated_table_sha256", "figure1_unchanged_sha256"):
        if actual[key] != expected[key]:
            raise ValueError(f"Rebuild diverged from the manuscript receipt at {key}")
    subprocess.run([sys.executable, str(ROOT / "reproduction/render_readme.py"),
                    "--output", str(out / "homepage")], check=True)
    print(f"Rebuilt the paper figures and tables and the homepage figures in {out}")


if __name__ == "__main__":
    main()
