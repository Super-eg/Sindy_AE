"""PCA visualization of SINDy-AE input data.

Run from examples/:
    python3 show_input.py --data lorenz/delay_xcoordinate_d20_5.npz
    python3 show_input.py --data lorenz/delay_xcoordinate_d20_5.npz --ic 3
    python3 show_input.py --data pendulum/pendulum_data.npz --n_ics 20

Outputs PNGs to the same folder as the .npz file:
    Single IC : pca2d_<stem>_ic<N>.png / pca3d_<stem>_ic<N>.png
    Multi  IC : pca2d_<stem>_<K>ics.png / pca3d_<stem>_<K>ics.png
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from sklearn.decomposition import PCA


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_npz(path):
    npz   = np.load(path, allow_pickle=False)
    X     = npz["train_x"]
    t_arr = npz["train_t"] if "train_t" in npz.files else None

    meta = {}
    for k in ("dt", "t_end", "noise_strength",
              "n_train_ics", "n_val_ics",
              "delay_dim", "delay_steps", "tau", "normalization"):
        if k in npz.files:
            meta[k] = npz[k].item()

    return X, t_arr, meta


def _get_n_steps(X, t_arr, meta):
    N = X.shape[0]
    if t_arr is not None:
        return t_arr.size
    if "n_train_ics" in meta:
        return N // int(meta["n_train_ics"])
    raise ValueError("Cannot determine n_steps: no train_t or n_train_ics in npz.")


def _extract_ic(X, n_steps, ic_idx):
    n_ics = X.shape[0] // n_steps
    if ic_idx >= n_ics:
        raise ValueError(f"--ic {ic_idx} out of range (only {n_ics} ICs available).")
    start = ic_idx * n_steps
    return X[start : start + n_steps]   # (n_steps, D)


def _meta_title(meta, stem):
    parts = [stem]
    if "delay_dim" in meta:
        tau = meta.get("tau", "?")
        parts.append(f"d={meta['delay_dim']}  τ={tau:.3f}s"
                     if isinstance(tau, float) else f"d={meta['delay_dim']}")
    if "noise_strength" in meta:
        parts.append(f"noise={meta['noise_strength']}")
    return "  |  ".join(parts)


def _time_color(n):
    """Return normalised time array [0, 1] of length n."""
    return np.linspace(0.0, 1.0, n)


# ── plots ─────────────────────────────────────────────────────────────────────

# Distinct colours for multi-IC overlay (cycles if more ICs than colours)
_IC_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#8dd3c7", "#fb8072",
]


def plot_pca2d_single(pcs, t, var_ratio, title, out_path):
    """Single IC: 2-D trajectory coloured by time."""
    c   = _time_color(len(t))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pcs[:, 0], pcs[:, 1], lw=0.4, color="lightgray", zorder=1)
    sc = ax.scatter(pcs[:, 0], pcs[:, 1],
                    c=c, cmap="plasma", s=6, alpha=0.8, linewidths=0, zorder=2)
    ax.scatter(*pcs[0,  :2], color="green", s=60, zorder=3, label="start")
    ax.scatter(*pcs[-1, :2], color="red",   s=60, zorder=3, label="end")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("time (normalised)")
    ax.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% var)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pca3d_single(pcs, t, var_ratio, title, out_path):
    """Single IC: 3-D trajectory coloured by time."""
    c   = _time_color(len(t))
    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2],
            lw=0.3, color="lightgray", alpha=0.6, zorder=1)
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2],
                    c=c, cmap="plasma", s=3, alpha=0.7, zorder=2)
    ax.scatter(*pcs[0,  :3], color="green", s=60, zorder=3)
    ax.scatter(*pcs[-1, :3], color="red",   s=60, zorder=3)
    fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6).set_label("time (normalised)")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pca2d_multi(pcs_list, var_ratio, title, out_path):
    """Multi-IC overlay: each IC gets its own colour, alpha fades over time."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, pcs in enumerate(pcs_list):
        col = _IC_COLORS[i % len(_IC_COLORS)]
        n   = len(pcs)
        # alpha gradient: early = transparent, late = opaque
        alphas = np.linspace(0.2, 0.9, n)
        ax.plot(pcs[:, 0], pcs[:, 1], lw=0.5, color=col, alpha=0.3, zorder=1)
        # scatter every few points to keep the plot light
        step = max(1, n // 150)
        idx  = np.arange(0, n, step)
        ax.scatter(pcs[idx, 0], pcs[idx, 1],
                   color=col, s=4, alpha=alphas[idx], linewidths=0, zorder=2)
    ax.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% var)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pca3d_multi(pcs_list, var_ratio, title, out_path):
    """Multi-IC overlay: each IC gets its own colour."""
    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")
    for i, pcs in enumerate(pcs_list):
        col  = _IC_COLORS[i % len(_IC_COLORS)]
        n    = len(pcs)
        step = max(1, n // 150)
        idx  = np.arange(0, n, step)
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2],
                lw=0.4, color=col, alpha=0.3, zorder=1)
        ax.scatter(pcs[idx, 0], pcs[idx, 1], pcs[idx, 2],
                   color=col, s=3, alpha=0.6, zorder=2)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PCA of IC trajectories — verify delay-embedding attractor.")
    parser.add_argument("--data",    required=True,
                        help="Path to .npz file")
    parser.add_argument("--ic",      type=int, default=0,
                        help="Single IC to visualise (0-indexed, default: 0). "
                             "Ignored when --n_ics is set.")
    parser.add_argument("--n_ics",   type=int, default=None,
                        help="Overlay this many ICs (starting from IC 0). "
                             "When set, --ic is ignored.")
    parser.add_argument("--n_comps", type=int, default=3,
                        help="Number of PCA components (default: 3)")
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    out_dir = os.path.dirname(os.path.abspath(data_path))
    stem    = os.path.splitext(os.path.basename(data_path))[0]

    print(f"Loading {data_path} ...")
    X, t_arr, meta = _load_npz(data_path)
    N, D = X.shape

    n_steps_raw = _get_n_steps(X, t_arr, meta)
    n_ics_total_raw = N // n_steps_raw

    print()
    print("=" * 60)
    print("DATA SUMMARY")
    print(f"  train samples : {N:,}  (shape: {X.shape})")
    print(f"  input_dim     : {D}")
    if t_arr is not None:
        print(f"  time range    : {t_arr[0]:.3f} → {t_arr[-1]:.3f}  (dt={meta.get('dt', '?')})")
    elif "dt" in meta and "t_end" in meta:
        print(f"  time range    : 0.0 → {meta['t_end']}  (dt={meta['dt']})")
    if "delay_dim" in meta:
        tau = meta.get("tau", "?")
        steps = meta.get("delay_steps", "?")
        tau_str = f"τ={tau:.3f}s  ({int(steps)} steps)" if isinstance(tau, float) else f"steps={steps}"
        print(f"  delay embed   : d={meta['delay_dim']},  {tau_str}")
    if "noise_strength" in meta:
        print(f"  noise_strength: {meta['noise_strength']}")
    if "n_train_ics" in meta:
        print(f"  n_train_ics   : {meta['n_train_ics']}")
    if "n_val_ics" in meta:
        print(f"  n_val_ics     : {meta['n_val_ics']}")
    print(f"  steps / IC    : {n_steps_raw}  ({n_ics_total_raw} ICs detected)")
    print("=" * 60)
    print()

    n_steps = _get_n_steps(X, t_arr, meta)
    n_ics_total = N // n_steps
    t_ic = t_arr if t_arr is not None else np.arange(n_steps)

    multi_mode = args.n_ics is not None
    n_comps = min(max(args.n_comps, 3), D, n_steps)

    if multi_mode:
        k = min(args.n_ics, n_ics_total)
        print(f"\nOverlay mode: using {k} ICs (out of {n_ics_total} available)")

        # Fit PCA on all k ICs stacked together for a consistent projection
        X_stack = np.vstack([_extract_ic(X, n_steps, i) for i in range(k)])
        print(f"Fitting PCA ({n_comps} components) on {len(X_stack):,} points ...")
        pca = PCA(n_components=n_comps)
        pca.fit(X_stack)

        var_ratio = pca.explained_variance_ratio_
        cumvar    = np.cumsum(var_ratio)
        print("  Explained variance:")
        for i, (v, cv) in enumerate(zip(var_ratio, cumvar)):
            print(f"    PC{i+1}: {v*100:.2f}%   (cumulative {cv*100:.2f}%)")

        # Project each IC separately for plotting
        pcs_list = [pca.transform(_extract_ic(X, n_steps, i)) for i in range(k)]

        title   = _meta_title(meta, stem) + f"  |  {k} ICs overlaid"
        tag     = f"{k}ics"
        out2d   = os.path.join(out_dir, f"pca2d_{stem}_{tag}.png")
        out3d   = os.path.join(out_dir, f"pca3d_{stem}_{tag}.png")
        plot_pca2d_multi(pcs_list, var_ratio, title, out2d)
        if n_comps >= 3:
            plot_pca3d_multi(pcs_list, var_ratio, title, out3d)

    else:
        ic_idx = args.ic
        print(f"\nSingle IC mode: IC {ic_idx} / {n_ics_total}")
        X_ic = _extract_ic(X, n_steps, ic_idx)
        print(f"Fitting PCA ({n_comps} components) on {n_steps} points ...")
        pca  = PCA(n_components=n_comps)
        pcs  = pca.fit_transform(X_ic)

        var_ratio = pca.explained_variance_ratio_
        cumvar    = np.cumsum(var_ratio)
        print("  Explained variance:")
        for i, (v, cv) in enumerate(zip(var_ratio, cumvar)):
            print(f"    PC{i+1}: {v*100:.2f}%   (cumulative {cv*100:.2f}%)")

        title  = _meta_title(meta, stem) + f"  |  IC {ic_idx}/{n_ics_total}"
        ic_tag = f"ic{ic_idx}"
        out2d  = os.path.join(out_dir, f"pca2d_{stem}_{ic_tag}.png")
        out3d  = os.path.join(out_dir, f"pca3d_{stem}_{ic_tag}.png")
        plot_pca2d_single(pcs, t_ic, var_ratio, title, out2d)
        if n_comps >= 3:
            plot_pca3d_single(pcs, t_ic, var_ratio, title, out3d)

    print("\nDone.")


if __name__ == "__main__":
    main()
