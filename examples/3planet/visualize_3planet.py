"""Visualize trained SINDy-AE model for the 3-Body (BHH) problem.

Reads precomputed data from `eval_<stamp>.npz` (produced by ../analyze.py) and
generates figures next to it:
    fig0_1_train_reconstruction.png    - [Training] Delay vector x and dx reconstruction
    fig0_2_train_sindy_simulation.png  - [Training] SINDy ODE forward vs encoder trajectory
    fig0_3_train_x_comparison.png      - [Training] Observable x_1(t): true vs SINDy-decoded
    fig1_phase_portrait.png   - 2D phase portrait: True (x1, v_x1) vs latent (xi_0, xi_1)
    fig2_reconstruction.png   - Input delay vector x and dx reconstruction quality
    fig3_time_series.png      - Time series: true (x1, v_x1) vs latent (xi_0, xi_1)
    fig4_sindy_simulation.png - SINDy ODE forward simulation vs encoder trajectory
    fig4b_sindy_2d.png        - 2D pairwise latent panels: encoder vs SINDy propagated
    fig5_long_term_phase.png  - Long-term continuous phase portrait comparison
    fig6_x_comparison.png     - Observable x_1(t): true vs SINDy-decoded

Usage (run from examples/3planet/):
    python3 visualize_3planet.py --eval checkpoints/<dir>/eval_<stamp>.npz
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Loader ─────────────────────────────────────────────────────────────────

def _load_eval(path):
    """Load eval_<stamp>.npz; unbox 0-d numeric arrays into Python scalars."""
    npz = np.load(path, allow_pickle=False)
    out = {}
    for k in npz.files:
        v = npz[k]
        if v.ndim == 0 and v.dtype.kind not in ("O", "U", "S"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _check_sindy(ev, split):
    """Return (z_sindy, x_sindy, converged_flag) or (None, None, False)."""
    z_key  = f"{split}_z_sindy"
    x_key  = f"{split}_x_sindy"
    ok_key = f"{split}_sindy_ok"
    if z_key not in ev:
        return None, None, False
    z = ev[z_key]
    x = ev[x_key]
    ok = bool(ev[ok_key][0]) if ok_key in ev and ev[ok_key].size > 0 else not np.any(np.isnan(z))
    return z, x, ok


# ─── Figure 0_1: Training reconstruction ────────────────────────────────────

def fig0_1_train_reconstruction(ev, out_dir):
    """[Training] Delay-vector x and dx reconstruction quality."""
    if "train_x" not in ev:
        print("  Skipping fig0_1: no train data in eval npz.")
        return

    t      = ev["train_t"]
    x_np   = ev["train_x"]
    dx_np  = ev["train_dx"]
    x_dec  = ev["train_x_dec"]
    dx_dec = ev["train_dx_dec"]

    n_dims = x_np.shape[1]
    dims   = [0, n_dims // 2, n_dims - 1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for row, d in enumerate(dims):
        axes[row, 0].plot(t, x_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 0].plot(t, x_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 0].set_ylabel(f"Delay $x_{{{d}}}$")
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, dx_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 1].plot(t, dx_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 1].set_ylabel(f"Delay $dx_{{{d}}}$")
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title("Input Reconstruction: Delay Vector $\\mathbf{x}$", fontsize=11)
    axes[0, 1].set_title("Derivative Reconstruction: $\\mathbf{dx}$", fontsize=11)
    axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)")

    rel_x  = np.mean((x_dec  - x_np ) ** 2) / np.mean(x_np  ** 2)
    rel_dx = np.mean((dx_dec - dx_np) ** 2) / np.mean(dx_np ** 2)
    fig.suptitle(
        f"[Training] Delay Embedding Reconstruction Quality\n"
        f"Relative Error x: {rel_x:.2e}   Relative Error dx: {rel_dx:.2e}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig0_1_train_reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 0_2: Training SINDy forward simulation ──────────────────────────

def fig0_2_train_sindy_simulation(ev, out_dir):
    """[Training] Encoder trajectory vs SINDy ODE forward simulation (main plot only)."""
    if "train_x" not in ev:
        print("  Skipping fig0_2: no train data in eval npz.")
        return

    t       = ev["train_t"]
    z_enc   = ev["train_z_enc"]
    z_sindy, _, sim_ok = _check_sindy(ev, "train")
    n_lat   = z_enc.shape[1]

    fig, axes = plt.subplots(n_lat, 1, figsize=(12, 2.5 * n_lat + 1), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(n_lat):
        axes[i].plot(t, z_enc[:, i], "steelblue", lw=1.4, label=f"Encoder $\\xi_{i}$")
        if sim_ok:
            axes[i].plot(t, z_sindy[:, i], "tomato", lw=1.2, ls="--",
                         label=f"SINDy ODE $\\xi_{i}$")
        axes[i].set_ylabel(f"$\\xi_{i}$")
        axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    status = "Converged" if sim_ok else "DIVERGED"
    axes[0].set_title(
        f"[Training] SINDy Forward Simulation vs Encoder Trajectory [{status}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig0_2_train_sindy_simulation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 0_3: Training observable dimension comparison ───────────────────

def fig0_3_train_x_comparison(ev, out_dir):
    """[Training] Compare true x_1(t) vs SINDy-decoded x_1(t)."""
    if "train_x" not in ev:
        print("  Skipping fig0_3: no train data in eval npz.")
        return

    t       = ev["train_t"]
    x_np    = ev["train_x"]
    _, x_sindy, sim_ok = _check_sindy(ev, "train")

    if not sim_ok:
        print("  Skipping fig0_3: training SINDy integration diverged.")
        return

    x0_true   = x_np[:, 0]
    x0_sindy  = x_sindy[:, 0]
    err_sindy = (x0_sindy - x0_true) ** 2

    fig, (ax_ts, ax_err) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.35},
    )

    ax_ts.plot(t, x0_true,  color="#4D94D1", lw=2.0, label="True $x_1(t)$")
    ax_ts.plot(t, x0_sindy, color="tomato",  lw=1.2, ls="--", label="SINDy -> Decoder")
    ax_ts.set_ylabel("Position $x_1(t)$", fontsize=14)
    ax_ts.set_title("[Training] SINDy Prediction", fontsize=15)
    ax_ts.legend(fontsize=10); ax_ts.grid(True, alpha=0.3)

    ax_err.semilogy(t, err_sindy, color="tomato", lw=1.4, label="Err²")
    ax_err.set_xlabel("Time (s)", fontsize=14)
    ax_err.set_ylabel("Squared Error", fontsize=14)
    ax_err.set_title("[Training] Pointwise Squared Error", fontsize=15)
    ax_err.legend(fontsize=10, loc="lower right")
    ax_err.grid(True, which="both", alpha=0.3)

    fig.suptitle("[Training] Planet 1 Position $x_1(t)$ Comparison",
                 fontsize=13, fontweight="bold")
    path = os.path.join(out_dir, "fig0_3_train_x_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Figure 1: 2D Phase Portrait ────────────────────────────────────────────

def fig1_phase_portrait(ev, out_dir):
    """True (x1, v_x1) phase portrait vs learned latent space."""
    has_z  = "test_z" in ev
    z_true = ev["test_z"] if has_z else ev["test_x"][:, :2]   # fallback to delay coords
    z_lat  = ev["test_z_enc"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(z_true[:, 0], z_true[:, 1], color="steelblue", lw=0.5, alpha=0.8)
    if has_z:
        ax1.set_title("True 3-Body BHH Orbit\n(Planet 1: $x_1$ vs $v_{x1}$)", fontsize=12)
        ax1.set_xlabel("Position $x_1$"); ax1.set_ylabel("Velocity $v_{x1}$")
    else:
        ax1.set_title("Observed delay coordinates\n($x_1(t)$ vs $x_1(t-\\tau)$)", fontsize=12)
        ax1.set_xlabel("$x_1(t)$"); ax1.set_ylabel("$x_1(t-\\tau)$")
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_lat[:, 0], z_lat[:, 1], color="tomato", lw=0.5, alpha=0.8)
    ax2.set_title("Learned Latent Phase Portrait\n(Encoder output $\\xi_0, \\xi_1$)", fontsize=12)
    ax2.set_xlabel("$\\xi_0$"); ax2.set_ylabel("$\\xi_1$")
    ax2.grid(True, alpha=0.3)

    for ax, data in ((ax1, z_true), (ax2, z_lat)):
        cx = (data[:, 0].max() + data[:, 0].min()) / 2
        cy = (data[:, 1].max() + data[:, 1].min()) / 2
        r  = max(data[:, 0].max() - data[:, 0].min(),
                 data[:, 1].max() - data[:, 1].min()) / 2 * 1.1
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)

    fig.suptitle("Phase Portrait Comparison — Test Set (Continuous Trajectory)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_phase_portrait.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 2: Reconstruction Quality ───────────────────────────────────────

def fig2_reconstruction(ev, out_dir):
    """Delay-vector x and dx reconstruction quality."""
    t        = ev["test_t"]
    x_np     = ev["test_x"]
    dx_np    = ev["test_dx"]
    x_dec    = ev["test_x_dec"]
    dx_dec   = ev["test_dx_dec"]

    n_dims = x_np.shape[1]
    dims   = [0, n_dims // 2, n_dims - 1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for row, d in enumerate(dims):
        axes[row, 0].plot(t, x_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 0].plot(t, x_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 0].set_ylabel(f"Delay $x_{{{d}}}$")
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, dx_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 1].plot(t, dx_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 1].set_ylabel(f"Delay $dx_{{{d}}}$")
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title("Input Reconstruction: Delay Vector $\\mathbf{x}$", fontsize=11)
    axes[0, 1].set_title("Derivative Reconstruction: $\\mathbf{dx}$", fontsize=11)
    axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)")

    rel_x  = np.mean((x_dec  - x_np ) ** 2) / np.mean(x_np  ** 2)
    rel_dx = np.mean((dx_dec - dx_np) ** 2) / np.mean(dx_np ** 2)
    fig.suptitle(
        f"Delay Embedding Reconstruction Quality\n"
        f"Relative Error x: {rel_x:.2e}   Relative Error dx: {rel_dx:.2e}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 3: Time Series ──────────────────────────────────────────────────

def fig3_time_series(ev, out_dir):
    """True (x1, v_x1) vs latent (xi_0, xi_1) over time."""
    t      = ev["test_t"]
    has_z  = "test_z" in ev
    z_true = ev["test_z"] if has_z else ev["test_x"][:, :2]   # fallback to delay coords

    z_lat  = ev["test_z_enc"]

    colors_true = ["steelblue", "darkorange"]
    colors_lat  = ["tomato",    "orchid"]
    ylabels_true = (["$x_1(t)$", "$v_{x1}(t)$"] if has_z
                    else ["$x_1(t)$", "$x_1(t-\\tau)$"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
    for i in range(2):
        axes[i, 0].plot(t, z_true[:, i], color=colors_true[i], lw=1.2)
        axes[i, 0].set_ylabel(ylabels_true[i], fontsize=11)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(t, z_lat[:, i], color=colors_lat[i], lw=1.2)
        axes[i, 1].set_ylabel(f"$\\xi_{i}(t)$ (Learned)", fontsize=11)
        axes[i, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title("True 3-Body State (Planet 1)" if has_z
                         else "Observed delay coordinates", fontsize=11)
    axes[0, 1].set_title("Learned Latent Variables", fontsize=11)
    axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)")

    fig.suptitle("Time Series Comparison: True vs Latent Space",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_time_series.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 4: SINDy Forward Simulation (test split) ────────────────────────

def fig4_sindy_simulation(ev, out_dir):
    """Compare encoder trajectory vs SINDy ODE forward simulation."""
    t       = ev["test_t"]
    z_enc   = ev["test_z_enc"]
    z_sindy, _, sim_ok = _check_sindy(ev, "test")
    n_lat   = z_enc.shape[1]

    fig, axes = plt.subplots(n_lat, 1, figsize=(12, 2.5 * n_lat + 1), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(n_lat):
        axes[i].plot(t, z_enc[:, i], "steelblue", lw=1.4, label=f"Encoder $\\xi_{i}$")
        if sim_ok:
            axes[i].plot(t, z_sindy[:, i], "tomato", lw=1.2, ls="--",
                         label=f"SINDy ODE $\\xi_{i}$")
        axes[i].set_ylabel(f"$\\xi_{i}$")
        axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    status = "Converged" if sim_ok else "DIVERGED"
    axes[0].set_title(f"SINDy Forward Simulation vs Encoder Trajectory [{status}]",
                      fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_sindy_simulation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    if not sim_ok:
        return

    # 2D pairwise panels
    from itertools import combinations
    pairs = list(combinations(range(n_lat), 2))
    n_pairs = len(pairs)

    fig2d, axes2d = plt.subplots(n_pairs, 2, figsize=(12, 4.5 * n_pairs))
    axes2d = np.atleast_2d(axes2d)
    for row, (i, j) in enumerate(pairs):
        for col, (traj, color, title) in enumerate([
            (z_enc,   "steelblue", "Encoder trajectory\n(Data -> Encoder)"),
            (z_sindy, "tomato",    "SINDy propagated trajectory\n(Learned ODE)"),
        ]):
            ax = axes2d[row, col]
            ax.plot(traj[:, i], traj[:, j], color=color, lw=0.5, alpha=0.85)
            ax.scatter(traj[0,  i], traj[0,  j], color="lime",  s=60, zorder=6, label="Start")
            ax.scatter(traj[-1, i], traj[-1, j], color="black", s=60, zorder=6,
                       marker="s", label="End")
            ax.set_xlabel(f"$\\xi_{i}$"); ax.set_ylabel(f"$\\xi_{j}$")
            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig2d.suptitle("2D Latent Space: Encoder vs Learned SINDy ODE",
                   fontsize=13, fontweight="bold")
    fig2d.tight_layout()
    path2d = os.path.join(out_dir, "fig4b_sindy_2d.png")
    plt.savefig(path2d, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2d}")


# ─── Figure 5: Long-Term Phase Portrait ─────────────────────────────────────

def fig5_long_term_phase(ev, out_dir):
    """Continuous latent phase portrait — does SINDy preserve the rosette topology?"""
    z_enc  = ev["test_z_enc"]
    z_sin, _, sim_ok = _check_sindy(ev, "test")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(z_enc[:, 0], z_enc[:, 1], color="lightgray", lw=1.5, alpha=0.6,
            label="Encoder Reference")

    if sim_ok:
        ax.plot(z_sin[:, 0], z_sin[:, 1], color="tomato", lw=0.5, alpha=0.9,
                label="SINDy ODE")
        ax.scatter(z_sin[0, 0], z_sin[0, 1], color="lime", s=50, zorder=5, label="Start")
        status = "Converged"
    else:
        ax.scatter(z_enc[0, 0], z_enc[0, 1], color="red", s=100, marker="X",
                   zorder=5, label="SINDy Diverged")
        status = "Diverged"

    ax.set_aspect("equal", "box")
    ax.set_xlabel("$\\xi_0$"); ax.set_ylabel("$\\xi_1$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(f"Long-term Latent Phase Portrait [{status}]\n"
                 f"Does SINDy preserve the Rosette Topology?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_long_term_phase.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 6: Observable Dimension x_1(t) ──────────────────────────────────

def fig6_x_comparison(ev, out_dir):
    """Compare true x_1(t) vs SINDy-decoded x_1(t)."""
    t       = ev["test_t"]
    x_np    = ev["test_x"]
    _, x_sindy, sim_ok = _check_sindy(ev, "test")

    if not sim_ok:
        print("  Skipping fig6: SINDy integration diverged.")
        return

    x0_true  = x_np[:, 0]
    x0_sindy = x_sindy[:, 0]
    err_sindy = (x0_sindy - x0_true) ** 2

    fig, (ax_ts, ax_err) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.35},
    )

    ax_ts.plot(t, x0_true,  color="#4D94D1", lw=2.0, label="True $x_1(t)$")
    ax_ts.plot(t, x0_sindy, color="tomato",  lw=1.2, ls="--", label="SINDy -> Decoder")
    ax_ts.set_ylabel("Position $x_1(t)$", fontsize=14)
    ax_ts.set_title("SINDy Prediction", fontsize=15)
    ax_ts.legend(fontsize=10); ax_ts.grid(True, alpha=0.3)

    ax_err.semilogy(t, err_sindy, color="tomato", lw=1.4, label="Err²")
    ax_err.set_xlabel("Time (s)", fontsize=14)
    ax_err.set_ylabel("Squared Error", fontsize=14)
    ax_err.set_title("Pointwise Squared Error", fontsize=15)
    ax_err.legend(fontsize=10, loc="lower right")
    ax_err.grid(True, which="both", alpha=0.3)

    fig.suptitle("Planet 1 Position $x_1(t)$ Comparison",
                 fontsize=13, fontweight="bold")
    path = os.path.join(out_dir, "fig6_x_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize SINDy-AE results for the 3-Body problem from precomputed eval data."
    )
    parser.add_argument("--eval", required=True,
                        help="Path to eval_<stamp>.npz produced by analyze.py")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for figures (default: same dir as --eval)")
    args = parser.parse_args()

    print(f"Loading eval data from {args.eval} ...")
    ev = _load_eval(args.eval)
    print(f"  latent_dim={ev['latent_dim']}, library_dim={ev['library_dim']}, "
          f"model_stamp={ev['model_stamp']}")
    print(f"  test samples : {ev['test_x'].shape[0]:,}  shape={tuple(ev['test_x'].shape)}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.eval))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating figures into: {out_dir}/")

    def _safe(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")

    _safe(fig0_1_train_reconstruction,   ev, out_dir)
    _safe(fig0_2_train_sindy_simulation, ev, out_dir)
    _safe(fig0_3_train_x_comparison,     ev, out_dir)
    _safe(fig1_phase_portrait,   ev, out_dir)
    _safe(fig2_reconstruction,   ev, out_dir)
    _safe(fig3_time_series,      ev, out_dir)
    _safe(fig4_sindy_simulation, ev, out_dir)
    _safe(fig5_long_term_phase,  ev, out_dir)
    _safe(fig6_x_comparison,     ev, out_dir)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
