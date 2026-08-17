"""Plots train and test accuracy vs. training step for each weight-decay
value in the multi-seed rigor sweep, parsing the per-step rows straight out
of `results/rigor.log` rather than recomputing anything.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RIGOR_LOG = Path(__file__).resolve().parent.parent / "results" / "rigor.log"
OUTPUT = Path(__file__).resolve().parent.parent / "results" / "headline.png"

BLOCK_RE = re.compile(r"^=== seed=(\d+) weight_decay=([\d.]+) ===$")
STEP_RE = re.compile(
    r"^step\s+(\d+)/\d+\s+wd=[\d.]+\s+batch_loss=[\d.]+\s+"
    r"train_acc=([\d.]+)\s+test_acc=([\d.]+)$"
)

WEIGHT_DECAYS = [0.0, 0.3, 1.0]
COLORS = {0.0: "#c44e52", 0.3: "#4c72b0", 1.0: "#55a868"}


def parse_rigor_log(path):
    """Returns {weight_decay: {seed: [(step, train_acc, test_acc), ...]}}."""
    runs = {wd: {} for wd in WEIGHT_DECAYS}
    seed = wd = None
    for line in path.read_text().splitlines():
        block = BLOCK_RE.match(line)
        if block:
            seed, wd = int(block.group(1)), float(block.group(2))
            runs[wd][seed] = []
            continue
        step_match = STEP_RE.match(line)
        if step_match and seed is not None:
            step = int(step_match.group(1))
            train_acc = float(step_match.group(2))
            test_acc = float(step_match.group(3))
            runs[wd][seed].append((step, train_acc, test_acc))
    return runs


def mean_std_by_step(seed_runs, col):
    """col: 1 for train_acc, 2 for test_acc. Averages over whichever seeds
    have a logged row at each step -- early-stopped seeds simply stop
    contributing past their last logged step, rather than being padded.
    """
    by_step = {}
    for rows in seed_runs.values():
        for row in rows:
            by_step.setdefault(row[0], []).append(row[col])
    steps = sorted(by_step)
    means = np.array([np.mean(by_step[s]) for s in steps])
    stds = np.array([np.std(by_step[s]) for s in steps])
    return np.array(steps), means, stds


def main():
    runs = parse_rigor_log(RIGOR_LOG)

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(16, 9))
    fig.set_dpi(150)

    for ax, col, title in ((ax_train, 1, "train accuracy"), (ax_test, 2, "test accuracy")):
        for wd in WEIGHT_DECAYS:
            steps, means, stds = mean_std_by_step(runs[wd], col)
            label = f"weight decay = {wd:g}" + (" (baseline)" if wd == 0.0 else "")
            ax.plot(steps, means, label=label, color=COLORS[wd], linewidth=2)
            ax.fill_between(steps, means - stds, means + stds, color=COLORS[wd], alpha=0.2)
        ax.set_xlabel("training step", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=12)
        ax.grid(axis="y", alpha=0.25)

    ax_train.legend(fontsize=12, loc="lower right")
    fig.suptitle(
        "weight decay lifts test accuracy far past the zero-decay plateau, in every seed",
        fontsize=14,
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT, facecolor="white")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
