"""Render the alpha ablation as grouped hit-rate bars: hit@5 vs hit@10, with
and without reranking.

Reads the retrieval-ablation result JSON and draws two panels, rerank off and
rerank on, sharing a y-axis; within each panel, for every alpha, hit@5 and
hit@10 are plotted as a pair of side-by-side bars. Color is categorical by k
(k=5 vs k=10), desaturated so the bars read as data rather than a bold UI
accent. Output is a vector PDF for direct \\includegraphics in the report.

Run with:
  uv run --with matplotlib python -m benchmark.scripts.plot_retrieval_ablation
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# Categorical slots 1 (blue, k=5) and 2 (orange, k=10) from references/palette.md,
# desaturated ~60% toward gray so the bars stay low-key on the page.
COLOR_K5 = "#5e7da2"
COLOR_K10 = "#b4806b"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY = "#52514e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("benchmark/result/retrieval_ablation_n1000-seed42_20260909T154503Z.json"))
    parser.add_argument("--output", type=Path, default=Path("report/figures/retrieval_ablation.pdf"))
    args = parser.parse_args()

    all_configs = json.loads(args.result.read_text())["configs"]
    by_rerank = {
        rerank: {(c["k"], c["alpha"]): c for c in all_configs if c["rerank"] == rerank}
        for rerank in (False, True)
    }
    alphas = sorted({a for (_, a) in by_rerank[True]})
    x = range(len(alphas))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)

    bars5 = bars10 = None
    for ax, rerank, title in zip(axes, (False, True), ("Without reranking", "With reranking")):
        configs = by_rerank[rerank]
        hit5 = [configs[(5, a)]["hit_rate"] for a in alphas]
        hit10 = [configs[(10, a)]["hit_rate"] for a in alphas]

        x5 = [i - width / 2 - 0.01 for i in x]
        x10 = [i + width / 2 + 0.01 for i in x]
        bars5 = ax.bar(x5, hit5, width=width, color=COLOR_K5, zorder=3)
        bars10 = ax.bar(x10, hit10, width=width, color=COLOR_K10, zorder=3)
        for bar, v in zip(list(bars5) + list(bars10), hit5 + hit10):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8.5, color=INK)

        ax.set_title(title, fontsize=10, color=INK, pad=8)
        ax.set_xticks(list(x))
        labels = [f"{a:g}" for a in alphas]
        ax.set_xticklabels(labels, fontsize=10, color=SECONDARY)
        ax.set_xlabel(r"$\alpha$ (lexical-only $\rightarrow$ dense-only)",
                      fontsize=9.5, color=SECONDARY, labelpad=10)
        ax.set_ylim(0.85, 1.0)
        ax.set_yticks([0.85, 0.90, 0.95, 1.0])
        ax.tick_params(axis="y", labelsize=10, colors=SECONDARY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.spines["bottom"].set_linewidth(0.8)

    # Reserve a fixed figure-fraction band below the axes for the xlabel
    # (just under the axes) and the legend (near the figure's bottom edge),
    # rather than placing the legend via an axes-fraction bbox_to_anchor --
    # that couples its offset to the axes' own (font-size-dependent) height
    # and silently drifts into the xlabel whenever tick/label font sizes
    # change.
    fig.subplots_adjust(bottom=0.28, wspace=0.08)
    handles = [Patch(facecolor=COLOR_K5, label="hit@5"), Patch(facecolor=COLOR_K10, label="hit@10")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=2, frameon=False, fontsize=10, handlelength=1.6, columnspacing=1.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
