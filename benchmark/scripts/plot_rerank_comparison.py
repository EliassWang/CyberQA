"""Render the rerank on/off comparison at the deployed default (k=10, alpha=0.5)
as a grouped bar chart: hit rate and MRR side by side, each split into a
without-rerank / with-rerank pair.

Reads the retrieval-ablation result JSON, pulls the two configs matching
k=10, alpha=0.5, and draws a single panel. Color is categorical by rerank
status, reusing the same two slots (blue, orange) as the alpha-sweep figure.
Output is a vector PDF for direct \\includegraphics in the report.

Run with:
  uv run --with matplotlib python -m benchmark.scripts.plot_rerank_comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# Categorical slots 1 (blue, no rerank) and 2 (orange, rerank) from
# references/palette.md, matching plot_retrieval_ablation.py's palette.
COLOR_OFF = "#5e7da2"
COLOR_ON = "#b4806b"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY = "#52514e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("benchmark/result/retrieval_ablation_n1000-seed42_20260909T154503Z.json"))
    parser.add_argument("--output", type=Path, default=Path("report/figures/rerank_comparison.pdf"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    all_configs = json.loads(args.result.read_text())["configs"]
    cfg_off = next(c for c in all_configs if c["k"] == args.k and c["alpha"] == args.alpha and c["rerank"] is False)
    cfg_on = next(c for c in all_configs if c["k"] == args.k and c["alpha"] == args.alpha and c["rerank"] is True)

    metrics = ["hit_rate", "mrr"]
    labels = ["Hit rate", "MRR"]
    off_vals = [cfg_off[m] for m in metrics]
    on_vals = [cfg_on[m] for m in metrics]

    x = range(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(4.4, 3.3))

    x_off = [i - width / 2 - 0.01 for i in x]
    x_on = [i + width / 2 + 0.01 for i in x]
    bars_off = ax.bar(x_off, off_vals, width=width, color=COLOR_OFF, zorder=3)
    bars_on = ax.bar(x_on, on_vals, width=width, color=COLOR_ON, zorder=3)
    for bar, v in zip(list(bars_off) + list(bars_on), off_vals + on_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10.5, color=SECONDARY)
    ax.set_ylim(0.75, 1.0)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ax.tick_params(axis="y", labelsize=10, colors=SECONDARY, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.subplots_adjust(bottom=0.24)
    handles = [Patch(facecolor=COLOR_OFF, label="Without reranking"),
               Patch(facecolor=COLOR_ON, label="With reranking")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=False, fontsize=9.5, handlelength=1.6, columnspacing=1.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
