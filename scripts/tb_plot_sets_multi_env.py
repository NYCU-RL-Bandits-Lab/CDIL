#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_file(path: Path) -> Path:
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
        keep[:-1] = steps[:-1] != steps[1:]
        steps, vals = steps[keep], vals[keep]
    return steps, vals


def align_by_common_steps(runs):
    """
    runs: list of (steps ndarray, values ndarray)
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


def discover_envs(base: Path, set_id: int):
    # look for dirs like "<env>_set{set_id}_seed0"
    pat = re.compile(rf"(.+?)_set{set_id}_seed0$")
    envs = []
    for d in base.iterdir():
        if d.is_dir():
            m = pat.match(d.name)
            if m:
                envs.append(m.group(1))
    envs = sorted(set(envs))
    return envs


def collect_runs_for_env(base: Path, env: str, set_id: int, seeds: int, tag: str, verbose=True):
    runs = []
    for s in range(seeds):
        run_dir = base / f"{env}_set{set_id}_seed{s}"
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


def plot_one_set(base: Path, set_id: int, envs, tag: str, outdir: Path,
                 smooth: int = 5, smooth_std: int = 5, std_scale: float = 0.5, alpha: float = 0.20,
                 linewidth: float = 3.5, markersize: float = 5, markevery: int = 20,
                 title_prefix: str = None):
    # Matplotlib style like your notebook
    plt.rcParams.update({
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })
    fig, ax = plt.subplots(figsize=(10, 8))
    handles, labels = [], []

    # color per env (consistent across sets): use tab10 cycle deterministically
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {env: palette[i % len(palette)] for i, env in enumerate(envs)}

    for env in envs:
        runs = collect_runs_for_env(base, env, set_id, seeds=5, tag=tag)
        if len(runs) < 2:
            print(
                f"[warn] {env} set{set_id}: only {len(runs)} run(s); plotting mean=that run, std=0", file=sys.stderr)
        x, stacked, mode = align_by_common_steps(runs)
        if x.size == 0:
            print(f"[skip] {env} set{set_id}: no aligned data",
                  file=sys.stderr)
            continue
        mean = stacked.mean(
            axis=0) if stacked.ndim == 2 and stacked.shape[0] else stacked
        std = stacked.std(
            axis=0) if stacked.ndim == 2 and stacked.shape[0] else np.zeros_like(mean)

        if smooth and smooth > 1:
            mean = uniform_filter1d(mean, size=smooth)
        if smooth_std and smooth_std > 1:
            std = uniform_filter1d(std, size=smooth_std)

        color = color_map[env]
        # draw band behind
        low = mean - std_scale*std
        high = mean + std_scale*std
        # :contentReference[oaicite:4]{index=4}
        ax.fill_between(x, low, high, alpha=alpha, color=color, zorder=1)
        # draw line
        (line,) = ax.plot(x, mean, label=env, color=color,
                          marker='o', markersize=markersize, markevery=max(1, markevery),
                          linewidth=linewidth, zorder=2)
        handles.append(line)
        labels.append(env)

    # axes cosmetics
    title = f"{title_prefix + ' - ' if title_prefix else ''}Set {set_id}"
    ax.set_title(title)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Test Average Return")
    ax.grid(visible=True, which="major",
            color="lightgray", linestyle="--", linewidth=2)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2.5)

    # legend OUTSIDE at bottom, centered
    # using fig.legend keeps it global and easy to center under the axes
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
                     ncol=max(1, len(labels)), frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.5)  # :contentReference[oaicite:5]{index=5}

    # give legend some space at bottom
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"set{set_id}_multi_env.png"
    plt.savefig(outfile, format="png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[ok] saved -> {outfile}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot 3 figures (sets 1–3). Each figure has six envs (mean across seeds) with seed deviation shaded.")
    ap.add_argument("--base", default="sensitivity_scan_tfboard",
                    help="Base directory containing <env>_set<k>_seed<i> runs.")
    ap.add_argument("--sets", type=int, nargs="+",
                    default=[1, 2, 3], help="Set ids to plot (default: 1 2 3).")
    ap.add_argument("--envs", type=str, default="",
                    help="Comma-separated env list (e.g., ant,cheetah,door,hopper,lift,wipe). If empty, auto-discover per set.")
    ap.add_argument("--tag", default="Test average return",
                    help="Scalar tag to read.")
    ap.add_argument("--outdir", default="plots_multi",
                    help="Output directory for figures.")
    ap.add_argument("--smooth", type=int, default=5,
                    help="Uniform smoothing window for mean (>=1).")
    ap.add_argument("--smooth-std", type=int, default=5,
                    help="Uniform smoothing window for std band (>=1).")
    ap.add_argument("--std-scale", type=float, default=0.5,
                    help="Band size in std units (e.g., 0.5 => ±0.5σ).")
    ap.add_argument("--alpha", type=float, default=0.20,
                    help="Alpha for shaded deviation band.")
    ap.add_argument("--linewidth", type=float, default=3.5, help="Line width.")
    ap.add_argument("--markersize", type=float, default=5, help="Marker size.")
    ap.add_argument("--markevery", type=int, default=20,
                    help="Mark every N points.")
    ap.add_argument("--title-prefix", default="",
                    help="Optional title prefix for figures.")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"[error] base dir not found: {base}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    envs_cli = [e.strip() for e in args.envs.split(
        ",") if e.strip()] if args.envs else None

    for set_id in args.sets:
        envs = envs_cli or discover_envs(base, set_id)
        if not envs:
            print(
                f"[warn] no envs discovered for set {set_id}. Skipping.", file=sys.stderr)
            continue
        # keep a stable order if auto-discovered
        envs = sorted(envs, key=lambda s: s.lower())
        plot_one_set(
            base=base, set_id=set_id, envs=envs, tag=args.tag, outdir=outdir,
            smooth=args.smooth, smooth_std=args.smooth_std, std_scale=args.std_scale,
            alpha=args.alpha, linewidth=args.linewidth, markersize=args.markersize,
            markevery=args.markevery, title_prefix=args.title_prefix or None
        )


if __name__ == "__main__":
    main()
