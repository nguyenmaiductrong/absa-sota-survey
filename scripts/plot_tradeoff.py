"""
Trade-off plot: Response Latency (ms) vs Sentiment Accuracy.

Reads all metrics JSON files under results/metrics/ and draws a scatter plot
where each point is one (model, dataset) run.

Usage:
  python scripts/plot_tradeoff.py                          # reads results/metrics/
  python scripts/plot_tradeoff.py --metrics-dir my/path   # custom dir
  python scripts/plot_tradeoff.py --out tradeoff.png       # save instead of show
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Colour / marker scheme ────────────────────────────────────────────────────

# One colour per model method (auto-assigned if unseen)
_PALETTE = [
    "#2196F3",  # blue
    "#E91E63",  # pink
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#F44336",  # red
    "#8BC34A",  # light-green
]

# One marker per dataset
_DATASET_MARKERS = {
    "SemEval-2014-Restaurant": "o",
    "SemEval-2014-Laptop":     "s",
    "UIT-VSFC":                "^",
}
_DEFAULT_MARKER = "D"


# ── Load metrics ──────────────────────────────────────────────────────────────

def load_metrics(metrics_dir: Path) -> list[dict]:
    """Return a list of metric dicts from all *.json files in metrics_dir."""
    records: list[dict] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                rec = json.load(fh)
            records.append(rec)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[skip] {path}: {exc}")
    return records


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_tradeoff(records: list[dict], out_path: Path | None = None) -> None:
    if not records:
        print("No metrics found. Run evaluate.py for at least one model first.")
        return

    # Collect unique methods → colour
    methods = list(dict.fromkeys(r.get("method", "?") for r in records))
    method_colour = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(9, 6))

    for rec in records:
        method  = rec.get("method", "?")
        dataset = rec.get("dataset", "?")

        x = rec.get("avg_latency_ms")
        y = rec.get("sentiment_accuracy")

        if x is None or y is None:
            continue

        colour = method_colour.get(method, "#607D8B")
        marker = _DATASET_MARKERS.get(dataset, _DEFAULT_MARKER)

        ax.scatter(x, y, s=180, color=colour, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=3)

        # Annotate with method name slightly offset
        ax.annotate(
            f"{method}\n({dataset.replace('SemEval-2014-', 'SE14-')})",
            xy=(x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=colour,
        )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xlabel("Average Latency per Sample (ms)", fontsize=12)
    ax.set_ylabel("Sentiment Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Latency Trade-off — ABSA Models", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Legend: models (colours) ──────────────────────────────────────────────
    model_handles = [
        mpatches.Patch(color=method_colour[m], label=m)
        for m in methods if m in method_colour
    ]

    # ── Legend: datasets (markers) ────────────────────────────────────────────
    dataset_handles = [
        plt.Line2D([0], [0], marker=mk, color="grey", linestyle="None",
                   markersize=9, label=ds)
        for ds, mk in _DATASET_MARKERS.items()
        if any(r.get("dataset") == ds for r in records)
    ]

    leg1 = ax.legend(handles=model_handles,  title="Model",   loc="lower right",
                     fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=dataset_handles, title="Dataset", loc="lower left",
              fontsize=8, title_fontsize=9)

    # ── Ideal corner annotation ───────────────────────────────────────────────
    ax.annotate("← faster\n↑ more accurate",
                xy=(0.01, 0.97), xycoords="axes fraction",
                fontsize=8, color="grey", va="top")

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
    else:
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot latency vs accuracy trade-off for all evaluated models."
    )
    p.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"),
                   help="Directory with *.json metric files (default: results/metrics)")
    p.add_argument("--out", type=Path, default=None,
                   help="Save figure to this path instead of displaying it.")
    args = p.parse_args()

    if not args.metrics_dir.exists():
        print(f"Metrics directory not found: {args.metrics_dir}")
        print("Run evaluate.py first to generate metrics.")
        return

    records = load_metrics(args.metrics_dir)
    print(f"Loaded {len(records)} metric file(s) from {args.metrics_dir}")
    plot_tradeoff(records, out_path=args.out)


if __name__ == "__main__":
    main()
