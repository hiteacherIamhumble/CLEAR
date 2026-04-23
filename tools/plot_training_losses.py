#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects


TRAIN_KEYS = [
    "train/box_loss",
    "train/pose_loss",
    "train/kobj_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/rle_loss",
]

VAL_KEYS = [
    "val/box_loss",
    "val/pose_loss",
    "val/kobj_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/rle_loss",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot polished training/validation loss curves from YOLO results.csv.")
    parser.add_argument("--csv", required=True, help="Path to results.csv")
    parser.add_argument("--out", required=True, help="Path to output image")
    parser.add_argument("--x", choices=["epoch", "step"], default="epoch", help="X-axis mode")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Required when --x step.",
    )
    return parser.parse_args()


def load_csv(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def to_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def moving_average(values, window=3):
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        valid = arr[lo:hi]
        valid = valid[~np.isnan(valid)]
        if valid.size:
            out[i] = valid.mean()
    return out


def infer_convergence_epoch(val_total):
    smooth = moving_average(val_total, window=5)
    best_idx = int(np.nanargmin(smooth))
    best_val = float(smooth[best_idx])
    threshold = max(0.01 * best_val, 0.003)
    for i in range(len(smooth)):
        tail = smooth[i:]
        if np.all(np.abs(tail - best_val) <= threshold):
            return i + 1, best_idx + 1
    return best_idx + 1, best_idx + 1


def prettify_label(name: str) -> str:
    label = name.split("/", 1)[1].replace("_loss", "").replace("_", " ")
    return label.upper() if label == "dfl" else label.title()


def build_series(rows, keys):
    return {k: np.array([to_float(row.get(k)) for row in rows], dtype=float) for k in keys}


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    rows = load_csv(csv_path)

    epochs = np.array([int(float(row["epoch"])) for row in rows], dtype=float)
    if args.x == "step":
        if not args.steps_per_epoch:
            raise ValueError("--steps-per-epoch is required when --x step")
        x = epochs * int(args.steps_per_epoch)
        x_label = "Training Step"
        convergence_label = f"step {int(x[-1] if len(x) == 0 else x[0])}"
    else:
        x = epochs
        x_label = "Epoch"
        convergence_label = f"epoch {int(epochs[-1])}"

    train = build_series(rows, TRAIN_KEYS)
    val = build_series(rows, VAL_KEYS)
    train_total = np.nansum(np.column_stack([train[k] for k in TRAIN_KEYS]), axis=1)
    val_total = np.nansum(np.column_stack([val[k] for k in VAL_KEYS]), axis=1)
    train_total_s = moving_average(train_total, window=3)
    val_total_s = moving_average(val_total, window=3)

    converge_epoch, best_epoch = infer_convergence_epoch(val_total)
    best_idx = best_epoch - 1

    plt.style.use("default")
    fig = plt.figure(figsize=(18, 12), dpi=180, facecolor="#f6f3ee")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.22, wspace=0.12)
    ax_main = fig.add_subplot(gs[0, :])
    ax_train = fig.add_subplot(gs[1, 0])
    ax_val = fig.add_subplot(gs[1, 1])

    for ax in (ax_main, ax_train, ax_val):
        ax.set_facecolor("#fffdf8")
        ax.grid(True, color="#d8d2c8", alpha=0.45, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#8c7f70")
            spine.set_linewidth(1.2)

    c_train = "#0f766e"
    c_train_s = "#14b8a6"
    c_val = "#9a3412"
    c_val_s = "#f97316"
    c_best = "#7c3aed"
    shadow = [patheffects.Stroke(linewidth=5, foreground="white", alpha=0.9), patheffects.Normal()]

    ax_main.plot(x, train_total, color=c_train, alpha=0.22, linewidth=2.0)
    ax_main.plot(x, train_total_s, color=c_train_s, linewidth=3.4, label="Train Total Loss", path_effects=shadow)
    ax_main.plot(x, val_total, color=c_val, alpha=0.18, linewidth=2.0)
    ax_main.plot(x, val_total_s, color=c_val_s, linewidth=3.4, label="Val Total Loss", path_effects=shadow)

    best_x = x[best_idx]
    best_y = val_total_s[best_idx]
    ax_main.axvspan(
        x[max(converge_epoch - 1, 0)],
        x[-1],
        color="#fde68a",
        alpha=0.2,
        label=f"Convergence Zone (>= {x_label.lower()} {int(x[max(converge_epoch - 1, 0)])})",
    )
    ax_main.axvline(best_x, color=c_best, linestyle="--", linewidth=2.0, alpha=0.95)
    ax_main.scatter([best_x], [best_y], s=120, color=c_best, edgecolor="white", linewidth=1.4, zorder=5)
    ann = ax_main.annotate(
        f"Best val total\n{x_label} {best_x:.0f}\n{best_y:.3f}",
        xy=(best_x, best_y),
        xytext=(16, 18),
        textcoords="offset points",
        fontsize=11,
        color="#2f1b4e",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f5edff", ec="#c4b5fd", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.5),
    )
    ann.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white", alpha=0.7)])

    component_colors = ["#1d4ed8", "#db2777", "#ea580c", "#16a34a", "#7c2d12", "#6d28d9"]
    for color, key in zip(component_colors, TRAIN_KEYS):
        ax_train.plot(x, train[key], label=prettify_label(key), linewidth=2.3, color=color)
    for color, key in zip(component_colors, VAL_KEYS):
        ax_val.plot(x, val[key], label=prettify_label(key), linewidth=2.3, color=color)

    ax_main.set_title("Training vs Validation Loss Landscape", fontsize=24, fontweight="bold", color="#2f241c", pad=14)
    ax_main.text(
        0.0,
        1.02,
        "Smoothed total losses show where learning slows and validation levels off.",
        transform=ax_main.transAxes,
        fontsize=12,
        color="#5f5448",
    )

    for ax, title in ((ax_train, "Train Loss Components"), (ax_val, "Validation Loss Components")):
        ax.set_title(title, fontsize=16, fontweight="bold", color="#2f241c", pad=10)
        ax.set_xlabel(x_label, fontsize=12, color="#3f362f")
        ax.set_ylabel("Loss", fontsize=12, color="#3f362f")
        ax.tick_params(labelsize=11, colors="#4b4138")
        ax.legend(frameon=True, facecolor="white", edgecolor="#d6cec3", fontsize=10, ncol=2)

    ax_main.set_xlabel(x_label, fontsize=13, color="#3f362f")
    ax_main.set_ylabel("Total Loss", fontsize=13, color="#3f362f")
    ax_main.tick_params(labelsize=11, colors="#4b4138")
    ax_main.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#d6cec3", fontsize=11)

    fig.text(
        0.015,
        0.015,
        f"Source: {csv_path} | Best epoch: {best_epoch} | Convergence starts around epoch {converge_epoch}",
        fontsize=10,
        color="#6a5f54",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"best_epoch={best_epoch}")
    print(f"converge_epoch={converge_epoch}")


if __name__ == "__main__":
    main()
