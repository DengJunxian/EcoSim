"""Render the homepage figures from the versioned manuscript data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper/source/source_data"
BG, INK, MUTED = "#F5F8FC", "#132641", "#486078"
BLUE, TEAL, AMBER = "#265EAC", "#008273", "#AE5819"
COLORS = [BLUE, TEAL, AMBER]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "svg.hashsalt": "ecosim-readme", "pdf.fonttype": 42,
})


def rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(figure, target):
    for suffix, metadata in [
        ("svg", {"Date": None}),
        ("pdf", {"CreationDate": None, "ModDate": None}),
        ("png", {}),
    ]:
        path = target.with_suffix("." + suffix)
        figure.savefig(path, dpi=300, facecolor=BG, metadata=metadata)
        if suffix == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)


def architecture(out):
    figure, ax = plt.subplots(figsize=(11.4, 3.0), facecolor=BG)
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 114), ylim=(0, 30))
    ax.axis("off")
    ax.text(4, 25.8, "One causal clock across investor, group and market",
            size=17, weight="bold", va="center")
    cards = [
        ("01", "Prepare", "Process boundaries\nand scheduled events", BLUE),
        ("02", "Observe & update", "Filter peer signals\nUpdate panic and beliefs", TEAL),
        ("03", "Decide", "Rule, persona or LLM\nproposes orders", AMBER),
        ("04", "Settle & commit", "Match orders and update\nthe integer ledger", "#554F92"),
    ]
    for index, (number, title, story, color) in enumerate(cards):
        x = 4 + 27 * index
        ax.add_patch(FancyBboxPatch((x, 9.2), 24.1, 11.8,
                     boxstyle="round,pad=0.25,rounding_size=1.25",
                     facecolor="white", edgecolor="#D2DDEB", linewidth=1))
        ax.text(x+1.6, 18.7, number, color=color, size=9.2, weight="bold")
        ax.text(x+1.6, 16.0, title, size=11.7, weight="bold")
        ax.text(x+1.6, 13.5, story, va="top", color=MUTED, size=9.7, linespacing=1.35)
        if index < 3:
            ax.annotate("", xy=(x+26.7, 15.2), xytext=(x+24.5, 15.2),
                        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.1})
    ax.plot([97, 97, 17], [8.8, 5.9, 5.9], color=TEAL, lw=1.4)
    ax.add_patch(FancyArrowPatch((17, 5.9), (17, 8.9), arrowstyle="-|>",
                                mutation_scale=10, color=TEAL, lw=1.4))
    ax.text(57, 3.4, "Committed prices and portfolios enter the next step",
            ha="center", va="center", color=TEAL, size=10)
    save(figure, out / "architecture")


def results(out):
    health = rows("health.csv")
    branches = rows("branch_step_metrics.csv")
    with (DATA / "snapshot_summary.json").open(encoding="utf-8") as stream:
        backends = json.load(stream)["backend_summaries"]
    keys = ["full", "zero_intelligence", "homogeneous_linear_value"]
    seeds = sorted({row["seed"] for row in health})
    assert len(seeds) == 3 and len(health) == 9
    assert {(row["seed"], row["baseline"]) for row in health} == {
        (seed, key) for seed in seeds for key in keys
    }
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.96), facecolor=BG)
    figure.subplots_adjust(left=.075, bottom=.20, right=.974, top=.70, wspace=.49)
    figure.text(.04, .92, "What the archived experiments show",
                size=17, weight="bold", va="center")
    for ax, title, detail in zip(axes,
            ["Market health", "Contagion branch", "Action sensitivity"],
            ["9 trajectories · 3 matched seeds", "Shared parent · infection share", "Fixed observations · 27 pairs per backend"]):
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.text(0, 1.23, title, transform=ax.transAxes, weight="bold", size=12)
        ax.text(0, 1.12, detail, transform=ax.transAxes, color=MUTED, size=8.6)
        ax.tick_params(length=0)

    ax = axes[0]
    for index, key in enumerate(keys):
        for row in (r for r in health if r["baseline"] == key):
            seed = seeds.index(row["seed"])
            ax.scatter(float(row["two_sided_coverage"])*100, index+(seed-1)*.15,
                       color=COLORS[seed], marker=("o", "s", "^")[seed], s=41)
    ax.set_yticks(range(3), ["Full", "Zero intel.", "Linear"])
    ax.invert_yaxis()
    ax.set(xlim=(0, 30), xticks=(0, 10, 20, 30), xlabel="Two-sided quote coverage")
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="x", color="#E0E7F1")

    ax = axes[1]
    for key, label, color, linestyle in [
        ("full_control", "Full control", BLUE, "-"),
        ("no_secondary_contagion", "Secondary off", TEAL, "--"),
    ]:
        points = sorted((int(row["event_step"]), float(row["infected_percent"]))
                        for row in branches if row["condition"] == key)
        assert [step for step, _ in points] == list(range(28))
        ax.plot(*zip(*points), lw=2.1, label=label, color=color, linestyle=linestyle)
    ax.set(xlim=(0, 27), ylim=(0, 52), xticks=(0, 9, 18, 27), xlabel="Event step")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color="#E0E7F1")
    ax.legend(loc="upper left", frameon=False, fontsize=8.2)

    ax = axes[2]
    for index, (key, label, color) in enumerate([
        ("rule", "Rule", BLUE), ("heuristic", "Persona", TEAL), ("gpt6_luna_live", "LLM", AMBER),
    ]):
        row = backends[key]
        value, total = row["changed_economic_actions"], row["fact_pairs"]
        assert total == 27 and 0 <= value <= total
        ax.barh(index, total, height=.48, color="#E8EDF5")
        ax.barh(index, value, height=.48, color=color)
        ax.text(value+.55, index, f"{value}/{total}", va="center", size=9, weight="bold")
    ax.set_yticks(range(3), ["Rule", "Persona", "LLM"])
    ax.invert_yaxis()
    ax.set(xlim=(0, 29), xticks=(0, 9, 18, 27), xlabel="Changed action signatures")
    ax.grid(axis="x", color="#E0E7F1")
    save(figure, out / "results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    architecture(args.output)
    results(args.output)
