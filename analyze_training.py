"""
Training History Analysis & Visualization
==========================================
Analyzes RL training results from training_history.csv and produces
a multi-panel dashboard saved as training_analysis.png.

Usage:
    python analyze_training.py
    python analyze_training.py --csv path/to/training_history.csv
    python analyze_training.py --csv training_history.csv --out results.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings

warnings.filterwarnings("ignore")

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"
GREY   = "#8C8C8C"
BG     = "#F8F9FA"

# ── helpers ───────────────────────────────────────────────────────────────────

def smooth(series: pd.Series, window: int = 10) -> pd.Series:
    """Rolling mean with min_periods=1 so early episodes are still shown."""
    return series.rolling(window, min_periods=1).mean()


def annotate_best(ax, x, y, label: str, color: str = RED) -> None:
    """Mark the best point on a curve."""
    idx = y.idxmax()
    ax.axvline(x[idx], color=color, lw=0.8, ls="--", alpha=0.6)
    ax.scatter(x[idx], y[idx], color=color, zorder=5, s=60)
    ax.annotate(
        f"{label}\n ep {x[idx]}",
        xy=(x[idx], y[idx]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise text summary to stdout."""
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total episodes          : {len(df)}")
    print(f"  Total reward  — final   : {df['total_reward'].iloc[-1]:.2f}")
    print(f"  Total reward  — best    : {df['total_reward'].max():.2f}  (ep {df['total_reward'].idxmax() + 1})")
    print(f"  Normalised reward— best : {df['normalised_reward'].max():.4f}")
    print(f"  Mean loss     — final   : {df['mean_loss'].iloc[-1]:.6f}")
    print(f"  Epsilon       — final   : {df['epsilon'].iloc[-1]:.4f}")

    if df["eval_eval_total_reward"].notna().any():
        eval_col = df["eval_eval_total_reward"].dropna()
        print(f"  Eval reward   — best    : {eval_col.max():.2f}  (ep {eval_col.idxmax() + 1})")

    print()
    print("  Completed requests (train) ─ final ep :", df["completed_requests"].iloc[-1])
    print("  Picked-up  requests (train) ─ final ep :", df["picked_up_requests"].iloc[-1])
    print(f"  Avg wait until pickup       ─ final ep : {df['avg_wait_until_pickup'].iloc[-1]:.2f}")
    print(f"  Avg excess ride time        ─ final ep : {df['avg_excess_ride_time'].iloc[-1]:.2f}")
    print("=" * 60)


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_training(df: pd.DataFrame, out_path: str = "training_analysis.png") -> None:

    ep = df["episode"]
    W  = 15   # smoothing window

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle(
        "RL Training Analysis Dashboard",
        fontsize=17, fontweight="bold", y=0.98, color="#222222"
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45,
        wspace=0.38,
        left=0.06, right=0.97,
        top=0.93, bottom=0.06,
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    def style(ax, title: str, xlabel: str = "Episode", ylabel: str = "") -> None:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.set_facecolor(BG)
        ax.grid(True, lw=0.4, alpha=0.6, color="#CCCCCC")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # ── 1. Total reward ───────────────────────────────────────────────────────
    ax = axes[0]
    raw  = df["total_reward"]
    sraw = smooth(raw, W)
    ax.fill_between(ep, raw, alpha=0.18, color=BLUE)
    ax.plot(ep, raw,  color=BLUE,   lw=0.8, alpha=0.5, label="Raw")
    ax.plot(ep, sraw, color=BLUE,   lw=2.0, label=f"Smoothed (w={W})")

    # overlay eval reward if available
    eval_mask = df["eval_eval_total_reward"].notna()
    if eval_mask.any():
        ax.plot(
            ep[eval_mask], df["eval_eval_total_reward"][eval_mask],
            "o--", color=ORANGE, lw=1.2, ms=4, label="Eval reward"
        )

    annotate_best(ax, ep, sraw, "Best\nsmoothed")
    ax.legend(fontsize=7, loc="lower right")
    style(ax, "1 · Total Reward", ylabel="Reward")

    # ── 2. Normalised reward ──────────────────────────────────────────────────
    ax = axes[1]
    nr = df["normalised_reward"]
    ax.fill_between(ep, nr, alpha=0.18, color=GREEN)
    ax.plot(ep, nr,          color=GREEN, lw=0.8, alpha=0.5)
    ax.plot(ep, smooth(nr, W), color=GREEN, lw=2.0)
    annotate_best(ax, ep, smooth(nr, W), "Best")
    style(ax, "2 · Normalised Reward", ylabel="Normalised Reward")

    # ── 3. Mean loss ──────────────────────────────────────────────────────────
    ax = axes[2]
    loss = df["mean_loss"]
    ax.semilogy(ep, loss,            color=RED, lw=0.8, alpha=0.5)
    ax.semilogy(ep, smooth(loss, W), color=RED, lw=2.0, label=f"Smoothed (w={W})")
    ax.legend(fontsize=7)
    style(ax, "3 · Mean Loss (log scale)", ylabel="Loss")

    # ── 4. Epsilon (exploration) ──────────────────────────────────────────────
    ax = axes[3]
    ax.plot(ep, df["epsilon"], color=PURPLE, lw=2.0)
    ax.fill_between(ep, df["epsilon"], alpha=0.15, color=PURPLE)
    style(ax, "4 · Epsilon (Exploration Rate)", ylabel="ε")

    # ── 5. Learning rate ──────────────────────────────────────────────────────
    ax = axes[4]
    ax.plot(ep, df["lr"], color=ORANGE, lw=2.0)
    ax.fill_between(ep, df["lr"], alpha=0.15, color=ORANGE)
    style(ax, "5 · Learning Rate", ylabel="LR")

    # ── 6. Completed vs picked-up requests ───────────────────────────────────
    ax = axes[5]
    ax.plot(ep, smooth(df["completed_requests"],  W), color=BLUE,  lw=2.0, label="Completed")
    ax.plot(ep, smooth(df["picked_up_requests"],  W), color=GREEN, lw=2.0, label="Picked up")
    ax.fill_between(ep,
                    smooth(df["completed_requests"], W),
                    smooth(df["picked_up_requests"], W),
                    alpha=0.12, color=RED, label="Gap")
    ax.legend(fontsize=7)
    style(ax, "6 · Requests: Completed vs Picked-Up", ylabel="Count")

    # ── 7. Avg wait until pickup ──────────────────────────────────────────────
    ax = axes[6]
    wait = df["avg_wait_until_pickup"]
    ax.fill_between(ep, wait, alpha=0.18, color=RED)
    ax.plot(ep, wait,            color=RED, lw=0.8, alpha=0.5)
    ax.plot(ep, smooth(wait, W), color=RED, lw=2.0, label=f"Smoothed (w={W})")

    if "eval_avg_wait_until_pickup" in df.columns and eval_mask.any():
        ax.plot(ep[eval_mask], df["eval_avg_wait_until_pickup"][eval_mask],
                "o--", color=ORANGE, lw=1.2, ms=4, label="Eval")

    ax.invert_yaxis()   # lower wait = better; put best at top
    ax.legend(fontsize=7)
    style(ax, "7 · Avg Wait Until Pickup ↓ better", ylabel="Time steps")

    # ── 8. Avg excess ride time ───────────────────────────────────────────────
    ax = axes[7]
    ert = df["avg_excess_ride_time"]
    ax.fill_between(ep, ert, alpha=0.18, color=ORANGE)
    ax.plot(ep, ert,            color=ORANGE, lw=0.8, alpha=0.5)
    ax.plot(ep, smooth(ert, W), color=ORANGE, lw=2.0, label=f"Smoothed (w={W})")

    if "eval_avg_excess_ride_time" in df.columns and eval_mask.any():
        ax.plot(ep[eval_mask], df["eval_avg_excess_ride_time"][eval_mask],
                "o--", color=BLUE, lw=1.2, ms=4, label="Eval")

    ax.invert_yaxis()   # lower = better
    ax.legend(fontsize=7)
    style(ax, "8 · Avg Excess Ride Time ↓ better", ylabel="Time steps")

    # ── 9. Train vs Eval reward scatter ──────────────────────────────────────
    ax = axes[8]
    if eval_mask.any():
        tr = df.loc[eval_mask, "total_reward"]
        ev = df.loc[eval_mask, "eval_eval_total_reward"]
        sc = ax.scatter(tr, ev, c=ep[eval_mask], cmap="viridis", s=40, zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Episode", fontsize=7)
        cb.ax.tick_params(labelsize=7)

        # perfect-correlation line
        mn = min(tr.min(), ev.min()) - 2
        mx = max(tr.max(), ev.max()) + 2
        ax.plot([mn, mx], [mn, mx], "--", color=GREY, lw=1.0, alpha=0.6, label="y=x")
        ax.legend(fontsize=7)
        style(ax, "9 · Train vs Eval Reward", xlabel="Train reward", ylabel="Eval reward")
    else:
        ax.text(0.5, 0.5, "No eval data available",
                ha="center", va="center", transform=ax.transAxes, color=GREY)
        style(ax, "9 · Train vs Eval Reward")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n✅  Plot saved → {out_path}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse RL training history CSV.")
    parser.add_argument("--csv", default="training_history.csv",
                        help="Path to the training_history.csv file")
    parser.add_argument("--out", default="training_analysis.png",
                        help="Output image path (default: training_analysis.png)")
    parser.add_argument("--smooth-window", type=int, default=10,
                        help="Rolling-mean window size (default: 10)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} episodes from '{args.csv}'")

    print_summary(df)
    plot_training(df, out_path=args.out)


if __name__ == "__main__":
    main()