#!/usr/bin/env python3

# from pathlib import Path
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# tag = "Test average return"
# for p in [
#     " tfboard/avatardice/hopper_set1_seed0",
#     " tfboard/avatardice/wipe_set1_seed0",
#     " tfboard/avatardice/ant_set1_seed0",
#     " tfboard/avatardice/cheetah_set1_seed0",
#     " tfboard/avatardice/door_set1_seed0",
#     " tfboard/avatardice/lift_set1_seed0",
# ]:
#     # pass the directory to include all its event files
#     ea = EventAccumulator(p)
#     ea.Reload()
#     steps = [e.step for e in ea.Scalars(tag)]
#     print(Path(p).name, "last_step=", max(steps) if steps else None)


import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_file(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        # choose newest TensorBoard events file in directory
        cands = sorted(
            p.glob("events.out.tfevents.*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            cands = sorted(
                p.glob("tfevents.*"), key=lambda x: x.stat().st_mtime, reverse=True
            )
        if not cands:
            raise FileNotFoundError(f"No tfevents file found under: {p}")
        return cands[0]
    raise FileNotFoundError(f"Path not found: {path}")


def load_scalar(event_file: Path, tag: str):
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(
            f"Tag '{tag}' not found in {event_file}.\nAvailable scalars: {tags}"
        )
    scal = ea.Scalars(tag)
    steps = np.array([s.step for s in scal], dtype=np.int64)
    vals = np.array([s.value for s in scal], dtype=np.float64)

    # sort by step & drop duplicate steps keeping the last occurrence
    order = np.argsort(steps, kind="stable")
    steps, vals = steps[order], vals[order]
    if len(steps) > 1:
        keep = np.ones_like(steps, dtype=bool)
        keep[:-1] = steps[:-1] != steps[1:]
        steps, vals = steps[keep], vals[keep]
    return steps, vals


def align_by_common_steps(runs):
    """
    runs: list of (steps ndarray, values ndarray)
    returns: common_steps, stacked_values (n_runs, n_steps)
    """
    common = set(runs[0][0])
    for st, _ in runs[1:]:
        common &= set(st)
    common_steps = np.array(sorted(common), dtype=np.int64)

    if common_steps.size == 0:
        # fallback: truncate to shortest length by index
        min_len = min(len(st) for st, _ in runs)
        xs = runs[0][0][:min_len]
        stacked = np.stack([v[:min_len] for _, v in runs], axis=0)
        return xs, stacked, "index"
    # map each run to dict(step->value) and collect
    stacked = np.stack(
        [
            np.array([dict(zip(st, v))[s] for s in common_steps], dtype=np.float64)
            for st, v in runs
        ],
        axis=0,
    )
    return common_steps, stacked, "step"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot single mean line with across-seed deviation shading from 5"
            " TensorBoard runs."
        )
    )
    ap.add_argument(
        "paths",
        nargs="+",
        help="Paths to 5 run directories OR tfevents files (seed0..seed4).",
    )
    ap.add_argument("--tag", default="Test average return", help="Scalar tag to read.")
    ap.add_argument(
        "--title",
        default=None,
        help="Plot title (default: inferred from parent dir of first run).",
    )
    ap.add_argument("--xlabel", default="Steps", help="X-axis label.")
    ap.add_argument("--ylabel", default="Test Average Return", help="Y-axis label.")
    ap.add_argument("--out", default="tb_5seeds_mean_band.png", help="Output PNG path.")
    ap.add_argument(
        "--smooth", type=int, default=5, help="Uniform smoothing window for mean (>=1)."
    )
    ap.add_argument(
        "--smooth-std",
        type=int,
        default=5,
        help="Uniform smoothing window for std band (>=1; 1 disables smoothing).",
    )
    ap.add_argument(
        "--std-scale",
        type=float,
        default=0.5,
        help="Scale factor for std band (e.g., 0.5 for ±0.5σ).",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.20, help="Alpha for the shaded deviation band."
    )
    ap.add_argument("--linewidth", type=float, default=3.5, help="Mean line width.")
    ap.add_argument("--markersize", type=float, default=6, help="Marker size.")
    ap.add_argument("--markevery", type=int, default=10, help="Mark every N points.")
    ap.add_argument(
        "--color",
        default=None,
        help="Hex or name for the line/shade. Defaults to Matplotlib cycle.",
    )
    args = ap.parse_args()

    if len(args.paths) < 2:
        print("Provide at least 2 paths (ideally 5: seed0..seed4).", file=sys.stderr)

    # resolve files
    files = []
    for p in args.paths:
        try:
            files.append(find_event_file(p))
        except Exception as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
    if len(files) < 2:
        print("Not enough valid TensorBoard event files.", file=sys.stderr)
        sys.exit(1)

    # load scalars
    runs = []
    for f in files:
        try:
            st, v = load_scalar(f, args.tag)
            runs.append((st, v))
        except KeyError as e:
            print(f"[skip] {f}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[skip] {f}: {e}", file=sys.stderr)
    if len(runs) < 2:
        print("No usable runs for the requested tag.", file=sys.stderr)
        sys.exit(2)

    x, stacked, mode = align_by_common_steps(runs)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    # smoothing (uniform)
    smooth_n = max(1, args.smooth)
    if smooth_n > 1:
        mean = uniform_filter1d(mean, size=smooth_n)
    smooth_std_n = max(1, args.smooth_std)
    if smooth_std_n > 1:
        std = uniform_filter1d(std, size=smooth_std_n)

    # figure style similar to your notebook aesthetic
    plt.rcParams.update(
        {
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 8))

    # choose color
    color = args.color
    if color is None:
        # let Matplotlib choose first cycle color
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    # draw shading first so it appears behind the line
    band_low = mean - args.std_scale * std
    band_high = mean + args.std_scale * std
    ax.fill_between(x, band_low, band_high, alpha=args.alpha, color=color, zorder=1)

    # draw mean line
    ax.plot(
        x,
        mean,
        label=args.title or Path(files[0]).parent.name,
        marker="o",
        markersize=args.markersize,
        markevery=max(1, args.markevery),
        linewidth=args.linewidth,
        color=color,
        zorder=2,
    )

    ax.set_title(args.title or Path(files[0]).parent.name)
    ax.set_xlabel(args.xlabel + ("" if mode == "step" else " (index-aligned)"))
    ax.set_ylabel(args.ylabel)
    ax.grid(visible=True, which="major", color="lightgray", linestyle="--", linewidth=2)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2.5)
    ax.legend(loc="best", frameon=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(args.out, format="png", bbox_inches="tight", pad_inches=0.3)
    print(f"Saved plot -> {args.out}")


if __name__ == "__main__":
    main()
