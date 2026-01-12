#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, EngFormatter
from scipy.ndimage import uniform_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_file(path: Path) -> Path:
    """Return an event file for a run directory or a file path itself."""
    if path.is_file():
        return path
    if path.is_dir():
        cands = sorted(path.glob("events.out.tfevents.*"),
                       key=lambda x: x.stat().st_mtime, reverse=True)
        if not cands:
            cands = sorted(path.glob("tfevents.*"),
                           key=lambda x: x.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"No tfevents file found under: {path}")
        return cands[0]
    raise FileNotFoundError(f"Path not found: {path}")


def load_scalar(event_file: Path, tag: str):
    """Load (steps, values) for a scalar tag from a TB event file."""
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(
            f"Tag '{tag}' not found in {event_file}.\nAvailable scalars: {tags}")
    scal = ea.Scalars(tag)
    steps = np.array([s.step for s in scal], dtype=np.int64)
    vals = np.array([s.value for s in scal], dtype=np.float64)
    if steps.size:
        order = np.argsort(steps, kind="stable")
        steps, vals = steps[order], vals[order]
        keep = np.ones_like(steps, dtype=bool)
        keep[:-1] = steps[:-1] != steps[1:]  # drop duplicate steps, keep last
        steps, vals = steps[keep], vals[keep]
    return steps, vals


def align_by_common_steps(runs):
    """
    runs: list[(steps ndarray, values ndarray)]
    returns: x, stacked (n_runs, n_steps), mode
    """
    if not runs:
        return np.array([]), np.empty((0, 0)), "empty"
    common = set(runs[0][0])
    for st, _ in runs[1:]:
        common &= set(st)
    common_steps = np.array(sorted(common), dtype=np.int64)
    if common_steps.size == 0:
        # fallback: shortest-by-index
        min_len = min(len(st) for st, _ in runs)
        xs = runs[0][0][:min_len]
        stacked = np.stack([v[:min_len] for _, v in runs], axis=0)
        return xs, stacked, "index"
    stacked = np.stack([
        np.array([dict(zip(st, v))[s] for s in common_steps], dtype=np.float64)
        for st, v in runs
    ], axis=0)
    return common_steps, stacked, "step"

# ---------- Discovery helpers ----------


PATTERN = re.compile(r"(.+?)_set(\d+)_seed(\d+)$")


def scan_layout(base: Path):
    """Scan base dir and return sorted env list, set list, and seeds per (env,set)."""
    envs, sets = set(), set()
    exists = {}
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = PATTERN.match(d.name)
        if not m:
            continue
        env, s, seed = m.group(1), int(m.group(2)), int(m.group(3))
        envs.add(env)
        sets.add(s)
        exists.setdefault((env, s), set()).add(seed)
    envs = sorted(envs, key=str.lower)
    sets = sorted(sets)
    return envs, sets, exists


def collect_runs(base: Path, env: str, set_id: int, seeds: list[int], tag: str, verbose=True):
    runs = []
    for seed in seeds:
        run_dir = base / f"{env}_set{set_id}_seed{seed}"
        try:
            ev = find_event_file(run_dir)
            st, v = load_scalar(ev, tag)
            if st.size and v.size:
                runs.append((st, v))
            elif verbose:
                print(f"[skip] empty scalars: {run_dir}", file=sys.stderr)
        except Exception as e:
            if verbose:
                print(f"[skip] {run_dir}: {e}", file=sys.stderr)
    return runs

# ---------- Plotting ----------


def plot_grid(
    base: Path,
    envs: list[str],
    sets: list[int],
    tag: str,
    seeds_per_set: int,
    out_path: Path,
    title: str | None,
    smooth: int,
    smooth_std: int,
    std_scale: float,
    alpha: float,
    linewidth: float,
    markersize: float,
    markevery: int,
):
    mpl.rcParams['figure.dpi'] = 600
    # Figure + 2x3 grid
    plt.rcParams.update({
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    })
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
    axes = axes.flatten()
    # Format x-axis: at most ~6 ticks, show 100k-style labels
    eng_fmt = EngFormatter(unit="")
    for ax in axes:
        # ~6 ticks: 0, 100k, 200k, ...
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(eng_fmt)

    # Consistent colors PER SET across all subplots
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    set_colors = {s: palette[i % len(palette)] for i, s in enumerate(sets)}

    # Prepare legend proxies (one per set)
    legend_handles = [
        Line2D([0], [0], color=set_colors[s], lw=3.5, label=f"set{s}") for s in sets]

    for ax, env in zip(axes, envs):
        handles_this_ax = []  # not used, but keep for clarity
        # plot each set for this env
        for s in sets:
            seeds = list(range(seeds_per_set))
            runs = collect_runs(base, env, s, seeds, tag, verbose=False)
            if not runs:
                continue
            x, stacked, mode = align_by_common_steps(runs)
            if x.size == 0:
                continue
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            if smooth and smooth > 1:
                mean = uniform_filter1d(mean, size=smooth)
            if smooth_std and smooth_std > 1:
                std = uniform_filter1d(std, size=smooth_std)

            color = set_colors[s]
            # shaded deviation (±k·std) behind line
            low, high = mean - std_scale*std, mean + std_scale*std
            ax.fill_between(x, low, high, color=color, alpha=alpha, zorder=1)
            ax.plot(
                x, mean, color=color, linewidth=linewidth,
                marker='o', markersize=markersize, markevery=max(1, markevery),
                label=f"set{s}", zorder=2
            )

        # per-axes cosmetics
        ax.set_title(env)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Test Average Return")
        ax.grid(visible=True, which="major", color="lightgray",
                linestyle="--", linewidth=1.5)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.8)

    # If fewer than 6 envs, hide extra axes
    for j in range(len(envs), 6):
        axes[j].axis("off")

    # Optional centered figure title
    if title:
        # keep the title centered near the top edge
        # centered; adjust y if needed
        fig.suptitle(title, y=0.985, fontsize=24)#.set_in_layout(False) 
        # suptitle docs: center via x=0.5 / y param
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html

    # ONE legend at the top-right, outside the axes area
    # Use fig.legend with an outside anchor so it doesn't steal bottom space
    leg = fig.legend(
        handles=legend_handles,
        loc="outside upper right",
        # keep it inside the top-right corner of the figure
        bbox_to_anchor=(0.995, 0.995),
        ncol=len(sets),
        frameon=True,
        fontsize=18
    )
    # Exclude legend from tight_layout calculations so we can shrink bottom margins
    leg.set_in_layout(False)

    # Leave room for suptitle + top-right legend; tighten bottom aggressively
    plt.tight_layout(rect=[0.02, 0.04, 1.00, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[ok] saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="2x3 grid: one subplot per env; within each subplot, lines = sets (mean across seeds) + ±std shading."
    )
    ap.add_argument("--base", default="tfboard/smodice",
                    help="Root containing <env>_set<k>_seed<i> folders.")
    ap.add_argument("--envs", default="hopper,ant,cheetah,door,lift,wipe",
                    help="Comma-separated list of 6 env names (e.g., hopper,ant,cheetah,door,lift,wipe). Auto-discover if empty.")
    ap.add_argument("--sets", type=int, nargs="+",
                    default=[1], help="Set IDs to include (default: 1).")
    ap.add_argument("--seeds-per-set", type=int, default=5,
                    help="Seeds per (env,set) to aggregate (default: 5).")
    ap.add_argument("--tag", default="Test average return",
                    help="Scalar tag to read.")
    ap.add_argument("--out", default="plots/plot_grid/avatardice.png", help="Output image path.")
    ap.add_argument("--title", default="avatardice set1", help="Figure title.")
    ap.add_argument("--smooth", type=int, default=5,
                    help="Uniform smoothing window for mean (>=1).")
    ap.add_argument("--smooth-std", type=int, default=5,
                    help="Uniform smoothing window for std band (>=1).")
    ap.add_argument("--std-scale", type=float, default=0.5,
                    help="Shaded band size in std units (e.g., 0.5 => ±0.5 sigma).")
    ap.add_argument("--alpha", type=float, default=0.20,
                    help="Alpha for shaded std band.")
    ap.add_argument("--linewidth", type=float,
                    default=3.5, help="Mean line width.")
    ap.add_argument("--markersize", type=float, default=5, help="Marker size.")
    ap.add_argument("--markevery", type=int, default=20,
                    help="Mark every N points.")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"[error] base dir not found: {base}", file=sys.stderr)
        sys.exit(1)

    # discovery
    if args.envs:
        envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    else:
        discovered_envs, discovered_sets, _exists = scan_layout(base)
        if not discovered_envs:
            print("[error] could not discover any envs under base", file=sys.stderr)
            sys.exit(2)
        # Take first 6 distinct envs by name
        envs = discovered_envs[:6]
        # If sets wasn't provided, keep defaults; else ensure subset exists
        if not args.sets:
            args.sets = discovered_sets

    # enforce exactly 6 envs for a 2x3 grid (pad/truncate)
    if len(envs) < 6:
        envs = envs + [""] * (6 - len(envs))
    elif len(envs) > 6:
        envs = envs[:6]

    plot_grid(
        base=base,
        envs=envs,
        sets=args.sets,
        tag=args.tag,
        seeds_per_set=args.seeds_per_set,
        out_path=Path(args.out),
        title=args.title,
        smooth=max(1, args.smooth),
        smooth_std=max(1, args.smooth_std),
        std_scale=args.std_scale,
        alpha=args.alpha,
        linewidth=args.linewidth,
        markersize=args.markersize,
        markevery=max(1, args.markevery),
    )


if __name__ == "__main__":
    main()
