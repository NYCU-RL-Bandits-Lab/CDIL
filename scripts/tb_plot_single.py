#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_file(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        # pick the newest tfevents file in the directory
        candidates = sorted(p.glob("events.out.tfevents.*"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
        if not candidates:
            # also allow any tfevents.* (some writers omit "events.out.")
            candidates = sorted(p.glob("tfevents.*"),
                                key=lambda x: x.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"No TensorBoard event files found under: {p}")
        return candidates[0]
    raise FileNotFoundError(f"Path not found: {path}")


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling standard deviation with simple rectangular window.
    Pads the edges by reflecting to keep length consistent.
    """
    if window <= 1:
        return np.zeros_like(x)
    # reflect-pad to avoid shrinking edges
    pad = window // 2
    xr = np.pad(x, pad_width=pad, mode="reflect")
    # rolling mean E[x]
    kernel = np.ones(window) / window
    mu = np.convolve(xr, kernel, mode="valid")
    # rolling E[x^2]
    ex2 = np.convolve(xr**2, kernel, mode="valid")
    var = np.maximum(ex2 - mu**2, 0.0)
    return np.sqrt(var)


def load_scalar(event_file: Path, tag: str):
    """Load steps and values for a scalar tag from a TensorBoard event file."""
    # Keep a reasonable size guidance for scalars; set to None for unlimited
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(
            f"Tag '{tag}' not found in {event_file.name}.\nAvailable scalar tags:\n  " +
            "\n  ".join(tags)
        )
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.int64)
    values = np.array([e.value for e in events], dtype=np.float64)

    # Deduplicate steps (keep the last occurrence) and ensure strictly increasing
    uniq_steps, idx = np.unique(
        steps, return_index=False, return_inverse=False, return_counts=False), None
    # Simpler: sort by step then unique
    order = np.argsort(steps, kind="stable")
    steps = steps[order]
    values = values[order]
    # drop consecutive duplicates in steps keeping the last value
    if len(steps) > 1:
        keep = np.ones_like(steps, dtype=bool)
        keep[:-1] = steps[:-1] != steps[1:]
        # keep only positions where next step differs; also keep the last
        steps = steps[keep]
        values = values[keep]
    return steps, values, tags


def make_plot(steps, values, title, xlabel, ylabel, out, smooth_win, std_win, alpha_std, line_width, markersize, markevery):
    # Styling similar to your ipynb sample
    plt.rcParams.update({
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })

    # Smoothing
    smoothed = values.copy()
    if smooth_win and smooth_win > 1:
        smoothed = uniform_filter1d(values, size=smooth_win)

    # Rolling std for a single run (visual noise band)
    std_band = np.zeros_like(smoothed)
    if std_win and std_win > 1:
        std_band = rolling_std(values, std_win)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Main line
    (line,) = ax.plot(
        steps, smoothed, label=title,
        marker='o', markersize=markersize, markevery=markevery,
        linewidth=line_width
    )

    # Shaded band (± 0.5 * rolling std), like your sample
    if std_win and std_win > 1:
        ax.fill_between(
            steps, smoothed - 0.5 * std_band, smoothed + 0.5 * std_band,
            alpha=alpha_std
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which="major",
            color="lightgray", linestyle="--", linewidth=2)
    ax.set_facecolor("white")

    # Thicker black spines
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2.5)

    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format='png', bbox_inches='tight', pad_inches=0.3)
    print(f"Saved plot -> {out}")
    # plt.show()  # uncomment if you want to display
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a single scalar from a TensorBoard event file.")
    parser.add_argument(
        "path", help="Path to an event file OR a directory containing one (will pick the newest).")
    parser.add_argument("--tag", default="Test average return",
                        help="Scalar tag to read (default: 'Test average return').")
    parser.add_argument("--title", default=None,
                        help="Plot title (default: directory or file stem).")
    parser.add_argument("--xlabel", default="Steps", help="X-axis label.")
    parser.add_argument(
        "--ylabel", default="Test Average Return", help="Y-axis label.")
    parser.add_argument("--out", default="tb_single_plot.png",
                        help="Output PNG path.")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Uniform smoothing window size (>=1).")
    parser.add_argument("--std-window", type=int, default=0,
                        help="Rolling std window size for shading (0 to disable).")
    parser.add_argument("--std-alpha", type=float,
                        default=0.20, help="Alpha for std shading.")
    parser.add_argument("--linewidth", type=float,
                        default=3.5, help="Main line width.")
    parser.add_argument("--markersize", type=float,
                        default=6, help="Marker size.")
    parser.add_argument("--markevery", type=int, default=10,
                        help="Mark every N points.")
    args = parser.parse_args()

    try:
        event_file = find_event_file(args.path)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        steps, values, tags = load_scalar(event_file, args.tag)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    title = args.title or event_file.parent.name
    make_plot(
        steps=steps,
        values=values,
        title=title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        out=args.out,
        smooth_win=max(1, args.smooth_window),
        std_win=max(0, args.std_window),
        alpha_std=args.std_alpha,
        line_width=args.linewidth,
        markersize=args.markersize,
        markevery=max(1, args.markevery),
    )


if __name__ == "__main__":
    main()
