"""Visualize trained Van der Pol model: learned latent dynamics vs true state.

Reads precomputed data from `eval_<stamp>.npz` (produced by ../analyze.py).

Generates figures saved next to the eval npz:
    fig0_1_train_reconstruction.png    - [Training] x and dx reconstruction (1 IC)
    fig0_2_train_sindy_simulation.png  - [Training] SINDy ODE forward vs encoder (1 IC)
    fig0_3_train_x_comparison.png      - [Training] x(t): true vs encoder/SINDy decoded (1 IC)
    fig1_phase_portrait.png     - 2D limit cycle: true z vs learned latent xi (all test ICs)
    fig2_reconstruction.png     - Input x and dx reconstruction quality (1 IC)
    fig3_time_series.png        - Time series: true z0z1 vs latent xi0xi1 (1 IC)
    fig4_sindy_simulation.png   - SINDy ODE forward simulation vs encoder trajectory
    fig4b_sindy_2d.png          - 2D pairwise latent panels: encoder vs SINDy propagated
    fig4c_sindy_overlay.gif     - 2D animated overlay
    fig5_long_term_phase.png    - Long-term continuous 2D phase portrait comparison (topology check)
    fig6_x_comparison.png       - Observable x(t): true vs encoder-decoded vs SINDy-decoded

Usage (run from examples/vander/):
    python3 visualize_vander.py --eval checkpoints/<dir>/eval_<stamp>.npz
    python3 visualize_vander.py --eval ... --ic 3
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Loader & helpers ───────────────────────────────────────────────────────

def _load_eval(path):
    npz = np.load(path, allow_pickle=False)
    out = {}
    for k in npz.files:
        v = npz[k]
        if v.ndim == 0 and v.dtype.kind not in ("O", "U", "S"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _ic_slice(ic_idx, n_steps):
    return slice(ic_idx * n_steps, (ic_idx + 1) * n_steps)


def _ic_ok(ev, split, ic):
    key = f"{split}_sindy_ok"
    if key not in ev:
        return False
    flags = ev[key]
    return bool(flags[ic]) if ic < len(flags) else False


# ─── figure 0_1: training reconstruction ─────────────────────────────────────

def fig0_1_train_reconstruction(ev, out_dir, ic_idx):
    """[Training] x and dx reconstruction quality for a single IC."""
    if "train_x" not in ev:
        print("  Skipping fig0_1: no train data in eval npz.")
        return

    t       = ev["train_t"]
    n_steps = len(t)
    n_ics   = ev["train_x"].shape[0] // n_steps
    if ic_idx >= n_ics:
        print(f"  Skipping fig0_1: train has {n_ics} ICs, requested #{ic_idx}.")
        return
    sl      = _ic_slice(ic_idx, n_steps)

    x_np  = ev["train_x"][sl]
    x_dec = ev["train_x_dec"][sl]

    order2 = ev.get("model_order", 1) == 2
    if order2 and "train_ddx" in ev and "train_ddx_dec" in ev:
        deriv_np    = ev["train_ddx"][sl]
        deriv_dec   = ev["train_ddx_dec"][sl]
        deriv_label = "ddx"
    else:
        deriv_np    = ev["train_dx"][sl]
        deriv_dec   = ev["train_dx_dec"][sl]
        deriv_label = "dx"

    n_dims = x_np.shape[1]
    dims = [0, n_dims // 2, n_dims - 1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for row, d in enumerate(dims):
        axes[row, 0].plot(t, x_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 0].plot(t, x_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 0].set_ylabel(f"x[{d}]")
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, deriv_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 1].plot(t, deriv_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 1].set_ylabel(f"{deriv_label}[{d}]")
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title("Input reconstruction  x", fontsize=11)
    axes[0, 1].set_title(f"Derivative reconstruction  {deriv_label}", fontsize=11)
    axes[-1, 0].set_xlabel("t"); axes[-1, 1].set_xlabel("t")

    rel_x     = np.mean((x_dec     - x_np    ) ** 2) / np.mean(x_np     ** 2)
    rel_deriv = np.mean((deriv_dec - deriv_np) ** 2) / np.mean(deriv_np ** 2)
    fig.suptitle(
        f"[Training] Reconstruction Quality — IC #{ic_idx}\n"
        f"relative err x: {rel_x:.2e}   relative err {deriv_label}: {rel_deriv:.2e}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig0_1_train_reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 0_2: training SINDy forward simulation ───────────────────────────

def fig0_2_train_sindy_simulation(ev, out_dir, ic_idx):
    """[Training] Encoder trajectory vs SINDy ODE forward simulation (single IC, main plot only)."""
    if "train_x" not in ev:
        print("  Skipping fig0_2: no train data in eval npz.")
        return

    t       = ev["train_t"]
    n_steps = len(t)
    n_ics   = ev["train_x"].shape[0] // n_steps
    if ic_idx >= n_ics:
        print(f"  Skipping fig0_2: train has {n_ics} ICs, requested #{ic_idx}.")
        return
    sl      = _ic_slice(ic_idx, n_steps)

    z_enc   = ev["train_z_enc"][sl]
    sim_ok  = _ic_ok(ev, "train", ic_idx)
    z_sindy = ev["train_z_sindy"][sl] if "train_z_sindy" in ev else None
    if z_sindy is None or np.any(np.isnan(z_sindy)):
        sim_ok = False

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i in range(2):
        axes[i].plot(t, z_enc[:, i], "steelblue", lw=1.4, label=f"Encoder xi{i}")
        if sim_ok:
            axes[i].plot(t, z_sindy[:, i], "tomato", lw=1.2, ls="--",
                         label=f"SINDy ODE xi{i}")
        axes[i].set_ylabel(f"xi{i}")
        axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("t")
    status = "converged" if sim_ok else "DIVERGED"
    axes[0].set_title(
        f"[Training] SINDy Forward Simulation vs Encoder Trajectory — IC #{ic_idx}  [{status}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig0_2_train_sindy_simulation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 0_3: training observable dimension comparison ────────────────────

def fig0_3_train_x_comparison(ev, out_dir, ic_idx):
    """[Training] Compare true x(t) vs encoder->decoder x(t) vs SINDy->decoder x(t) on dim 0."""
    if "train_x" not in ev:
        print("  Skipping fig0_3: no train data in eval npz.")
        return

    t       = ev["train_t"]
    n_steps = len(t)
    n_ics   = ev["train_x"].shape[0] // n_steps
    if ic_idx >= n_ics:
        print(f"  Skipping fig0_3: train has {n_ics} ICs, requested #{ic_idx}.")
        return
    sl      = _ic_slice(ic_idx, n_steps)

    x_np    = ev["train_x"][sl]
    x_enc   = ev["train_x_dec"][sl]
    sim_ok  = _ic_ok(ev, "train", ic_idx)
    x_sindy = ev["train_x_sindy"][sl] if "train_x_sindy" in ev else None
    if x_sindy is None or np.any(np.isnan(x_sindy)):
        sim_ok = False

    if not sim_ok:
        print("  Skipping fig0_3: training SINDy integration diverged.")
        return

    x0_true  = x_np[:, 0]
    x0_enc   = x_enc[:, 0]
    x0_sindy = x_sindy[:, 0]

    err_enc   = (x0_enc   - x0_true) ** 2
    err_sindy = (x0_sindy - x0_true) ** 2

    fig, (ax_ts, ax_err) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.35},
    )

    ax_ts.plot(t, x0_true,  color="#333333",   lw=1.5, label="True x(t)")
    ax_ts.plot(t, x0_enc,   color="steelblue", lw=1.2, ls="--", label="Encoder->Decoder")
    ax_ts.plot(t, x0_sindy, color="tomato",    lw=1.2, ls=":",  label="SINDy->Decoder")
    ax_ts.set_ylabel("x(t)  [observable dim]", fontsize=11)
    ax_ts.set_title("Observable Dimension: True x(t) vs Reconstructions", fontsize=12)
    ax_ts.legend(fontsize=10); ax_ts.grid(True, alpha=0.3)

    ax_err.semilogy(t, err_enc,   color="steelblue", lw=1.4, label="Encoder err²")
    ax_err.semilogy(t, err_sindy, color="tomato",    lw=1.4, ls="--", label="SINDy err²")
    ax_err.set_xlabel("t", fontsize=11)
    ax_err.set_ylabel("Squared error  x(t)", fontsize=11)
    ax_err.set_title("Pointwise squared error on observable dimension", fontsize=12)
    ax_err.legend(fontsize=10); ax_err.grid(True, which="both", alpha=0.3)

    rmse_enc   = float(np.sqrt(np.mean(err_enc)))
    rmse_sindy = float(np.sqrt(np.mean(err_sindy)))
    fig.suptitle(
        f"[Training] Observable Dimension Comparison — IC #{ic_idx}\n"
        f"RMSE  Encoder: {rmse_enc:.4f}   SINDy: {rmse_sindy:.4f}",
        fontsize=13, fontweight="bold",
    )
    path = os.path.join(out_dir, "fig0_3_train_x_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── figure 1: 2D phase portrait ─────────────────────────────────────────────

def fig1_phase_portrait(ev, out_dir):
    """True Van der Pol limit cycle vs learned latent phase portrait (all ICs)."""
    has_z  = "test_z" in ev
    z_true = ev["test_z"].reshape(-1, 2) if has_z else ev["test_x"][:, :2]
    z_lat  = ev["test_z_enc"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(z_true[:, 0], z_true[:, 1], color="steelblue", lw=0.25, alpha=0.5)
    ax1.set_title("True Van der Pol limit cycle\n(normalized z0, z1)", fontsize=12)
    ax1.set_xlabel("z0"); ax1.set_ylabel("z1")
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_lat[:, 0], z_lat[:, 1], color="tomato", lw=0.25, alpha=0.5)
    ax2.set_title("Learned latent phase portrait\n(encoder output xi0, xi1)", fontsize=12)
    ax2.set_xlabel("xi0"); ax2.set_ylabel("xi1")
    ax2.grid(True, alpha=0.3)

    for ax, data in ((ax1, z_true), (ax2, z_lat)):
        cx = (data[:, 0].max() + data[:, 0].min()) / 2
        cy = (data[:, 1].max() + data[:, 1].min()) / 2
        r  = max(data[:, 0].max() - data[:, 0].min(),
                 data[:, 1].max() - data[:, 1].min()) / 2 * 1.1
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)

    fig.suptitle("Phase Portrait Comparison — All Test ICs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_phase_portrait.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 2: reconstruction quality ───────────────────────────────────────

def fig2_reconstruction(ev, out_dir, ic_idx):
    """x and dx reconstruction quality for a single IC."""
    t       = ev["test_t"]
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)

    x_np  = ev["test_x"][sl]
    x_dec = ev["test_x_dec"][sl]

    order2 = ev.get("model_order", 1) == 2
    if order2 and "test_ddx" in ev and "test_ddx_dec" in ev:
        deriv_np   = ev["test_ddx"][sl]
        deriv_dec  = ev["test_ddx_dec"][sl]
        deriv_label = "ddx"
    else:
        deriv_np   = ev["test_dx"][sl]
        deriv_dec  = ev["test_dx_dec"][sl]
        deriv_label = "dx"

    n_dims = x_np.shape[1]
    dims = [0, n_dims // 2, n_dims - 1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for row, d in enumerate(dims):
        axes[row, 0].plot(t, x_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 0].plot(t, x_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 0].set_ylabel(f"x[{d}]")
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, deriv_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 1].plot(t, deriv_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 1].set_ylabel(f"{deriv_label}[{d}]")
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title("Input reconstruction  x", fontsize=11)
    axes[0, 1].set_title(f"Derivative reconstruction  {deriv_label}", fontsize=11)
    axes[-1, 0].set_xlabel("t"); axes[-1, 1].set_xlabel("t")

    rel_x     = np.mean((x_dec     - x_np    ) ** 2) / np.mean(x_np     ** 2)
    rel_deriv = np.mean((deriv_dec - deriv_np) ** 2) / np.mean(deriv_np ** 2)
    fig.suptitle(
        f"Reconstruction Quality — IC #{ic_idx}\n"
        f"relative err x: {rel_x:.2e}   relative err {deriv_label}: {rel_deriv:.2e}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 3: per-dimension time series ─────────────────────────────────────

def fig3_time_series(ev, out_dir, ic_idx):
    """True z0z1 vs latent xi0xi1 over time (single IC)."""
    t       = ev["test_t"]
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)
    has_z   = "test_z" in ev
    z_true  = ev["test_z"][sl] if has_z else ev["test_x"][sl, :2]
    z_lat   = ev["test_z_enc"][sl]

    colors_true = ["steelblue", "darkorange"]
    colors_lat  = ["tomato",    "orchid"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
    for i in range(2):
        axes[i, 0].plot(t, z_true[:, i], color=colors_true[i], lw=1.2)
        axes[i, 0].set_ylabel(f"z{i} (true)" if has_z else f"x_delay[{i}]", fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(t, z_lat[:, i], color=colors_lat[i], lw=1.2)
        axes[i, 1].set_ylabel(f"xi{i} (learned)", fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title("True Van der Pol state" if has_z else "Observed coordinates",
                          fontsize=11)
    axes[0, 1].set_title("Learned latent variables", fontsize=11)
    axes[-1, 0].set_xlabel("t"); axes[-1, 1].set_xlabel("t")

    fig.suptitle(f"Per-dimension Time Series — IC #{ic_idx}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_time_series.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 4: SINDy forward simulation ──────────────────────────────────────

def fig4_sindy_simulation(ev, out_dir, ic_idx):
    """Encoder trajectory vs SINDy ODE forward simulation (single IC)."""
    t       = ev["test_t"]
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)

    z_enc   = ev["test_z_enc"][sl]
    sim_ok  = _ic_ok(ev, "test", ic_idx)
    z_sindy = ev["test_z_sindy"][sl] if "test_z_sindy" in ev else None
    if z_sindy is None or np.any(np.isnan(z_sindy)):
        sim_ok = False

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i in range(2):
        axes[i].plot(t, z_enc[:, i], "steelblue", lw=1.4, label=f"Encoder xi{i}")
        if sim_ok:
            axes[i].plot(t, z_sindy[:, i], "tomato", lw=1.2, ls="--",
                         label=f"SINDy ODE xi{i}")
        axes[i].set_ylabel(f"xi{i}")
        axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("t")
    status = "converged" if sim_ok else "DIVERGED"
    axes[0].set_title(
        f"SINDy Forward Simulation vs Encoder Trajectory — IC #{ic_idx}  [{status}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_sindy_simulation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    if not sim_ok:
        return

    # ── fig4b: pairwise 2D latent panels (encoder vs SINDy) ──
    from itertools import combinations
    latent_dim = z_enc.shape[1]
    pairs = list(combinations(range(latent_dim), 2))
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    fig2d, axes2d = plt.subplots(n_pairs, 2, figsize=(12, 4.5 * n_pairs))
    axes2d = np.atleast_2d(axes2d)
    for row, (i, j) in enumerate(pairs):
        for col, (traj, color, title) in enumerate([
            (z_enc,   "steelblue", "Encoder trajectory\n(true data -> encoder)"),
            (z_sindy, "tomato",    "SINDy propagated trajectory\n(learned ODE)"),
        ]):
            ax = axes2d[row, col]
            ax.plot(traj[:, i], traj[:, j], color=color, lw=1.0, alpha=0.85)
            ax.scatter(traj[0,  i], traj[0,  j], color="lime",  s=60, zorder=6, label="start")
            ax.scatter(traj[-1, i], traj[-1, j], color="black", s=60, zorder=6,
                       marker="s", label="end")
            ax.set_xlabel(f"xi{i}"); ax.set_ylabel(f"xi{j}")
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

    # ── fig4c: animated 2D overlay ──
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    n_frames  = min(200, len(t))
    frame_idx = np.linspace(0, len(t) - 1, n_frames, dtype=int)

    all_pts = np.vstack([z_enc, z_sindy])
    cx = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
    cy = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
    r  = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
             all_pts[:, 1].max() - all_pts[:, 1].min()) / 2 * 1.15

    figc, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_xlabel("xi0"); ax.set_ylabel("xi1")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"2D Overlay: Encoder vs SINDy ODE — IC #{ic_idx}",
                 fontsize=12, fontweight="bold")

    ax.plot(z_enc[:, 0],   z_enc[:, 1],   color="steelblue", lw=0.5, alpha=0.2)
    ax.plot(z_sindy[:, 0], z_sindy[:, 1], color="tomato",    lw=0.5, alpha=0.2)
    ax.scatter(z_enc[0, 0],   z_enc[0, 1],   color="steelblue", s=80, zorder=6, marker="o")
    ax.scatter(z_sindy[0, 0], z_sindy[0, 1], color="tomato",    s=80, zorder=6, marker="o")

    line_enc,   = ax.plot([], [], color="steelblue", lw=1.4, label="Encoder")
    line_sindy, = ax.plot([], [], color="tomato",    lw=1.4, ls="--", label="SINDy ODE")
    dot_enc,    = ax.plot([], [], "o", color="steelblue", ms=7, zorder=7)
    dot_sindy,  = ax.plot([], [], "o", color="tomato",    ms=7, zorder=7)
    time_text   = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, va="top")

    legend_handles = [
        Line2D([0], [0], color="steelblue", lw=1.4,              label="Encoder"),
        Line2D([0], [0], color="tomato",    lw=1.4, ls="--",     label="SINDy ODE"),
        Line2D([0], [0], marker="o", color="gray", linestyle="none", ms=7, label="start"),
        Line2D([0], [0], marker="s", color="gray", linestyle="none", ms=7, label="end"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
    figc.tight_layout()

    def _init():
        line_enc.set_data([], [])
        line_sindy.set_data([], [])
        dot_enc.set_data([], [])
        dot_sindy.set_data([], [])
        time_text.set_text("")
        return line_enc, line_sindy, dot_enc, dot_sindy, time_text

    def _update(frame):
        k = frame_idx[frame]
        line_enc.set_data(z_enc[:k+1, 0],   z_enc[:k+1, 1])
        line_sindy.set_data(z_sindy[:k+1, 0], z_sindy[:k+1, 1])
        dot_enc.set_data([z_enc[k, 0]],   [z_enc[k, 1]])
        dot_sindy.set_data([z_sindy[k, 0]], [z_sindy[k, 1]])
        time_text.set_text(f"t = {t[k]:.2f}")
        if frame == n_frames - 1:
            ax.scatter(z_enc[-1, 0],   z_enc[-1, 1],   color="steelblue", s=80, zorder=8, marker="s")
            ax.scatter(z_sindy[-1, 0], z_sindy[-1, 1], color="tomato",    s=80, zorder=8, marker="s")
        return line_enc, line_sindy, dot_enc, dot_sindy, time_text

    anim = FuncAnimation(figc, _update, frames=n_frames,
                         init_func=_init, blit=True, interval=50)
    pathc = os.path.join(out_dir, "fig4c_sindy_overlay.gif")
    anim.save(pathc, writer=PillowWriter(fps=20))
    plt.close()
    print(f"  Saved: {pathc}")


# ─── figure 5: long-term phase portrait ──────────────────────────────────────

def fig5_long_term_phase(ev, out_dir, ic_idx):
    """Continuous latent 2D phase portrait — does SINDy preserve the limit-cycle topology?"""
    t       = ev["test_t"]
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)

    z_enc = ev["test_z_enc"][sl]
    sim_ok  = _ic_ok(ev, "test", ic_idx)
    z_sin = ev["test_z_sindy"][sl] if "test_z_sindy" in ev else None
    if z_sin is None or np.any(np.isnan(z_sin)):
        sim_ok = False

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(z_enc[:, 0], z_enc[:, 1], color="lightgray", lw=1.5, alpha=0.6,
            label="Encoder Reference")

    if sim_ok:
        ax.plot(z_sin[:, 0], z_sin[:, 1], color="tomato", lw=0.6, alpha=0.9,
                label="SINDy ODE")
        ax.scatter(z_sin[0, 0], z_sin[0, 1], color="lime", s=50, zorder=5, label="Start")
        status = "Converged"
    else:
        ax.scatter(z_enc[0, 0], z_enc[0, 1], color="red", s=100, marker="X",
                   zorder=5, label="SINDy Diverged")
        status = "Diverged"

    ax.set_aspect("equal", "box")
    ax.set_xlabel("xi0"); ax.set_ylabel("xi1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(f"Long-term Latent Phase Portrait — IC #{ic_idx} [{status}]\n"
                 f"Does SINDy preserve the Limit-Cycle Topology?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_long_term_phase.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 6: observable dimension comparison ───────────────────────────────

def fig6_x_comparison(ev, out_dir, ic_idx):
    """Compare true x(t) vs encoder->decoder x(t) vs SINDy->decoder x(t) on dim 0."""
    t       = ev["test_t"]
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)

    x_np    = ev["test_x"][sl]
    x_enc   = ev["test_x_dec"][sl]
    sim_ok  = _ic_ok(ev, "test", ic_idx)
    x_sindy = ev["test_x_sindy"][sl] if "test_x_sindy" in ev else None
    if x_sindy is None or np.any(np.isnan(x_sindy)):
        sim_ok = False

    if not sim_ok:
        print("  Skipping fig6: SINDy integration diverged.")
        return

    x0_true  = x_np[:, 0]
    x0_enc   = x_enc[:, 0]
    x0_sindy = x_sindy[:, 0]

    err_enc   = (x0_enc   - x0_true) ** 2
    err_sindy = (x0_sindy - x0_true) ** 2

    fig, (ax_ts, ax_err) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.35},
    )

    ax_ts.plot(t, x0_true,  color="#333333",   lw=1.5, label="True x(t)")
    ax_ts.plot(t, x0_enc,   color="steelblue", lw=1.2, ls="--", label="Encoder->Decoder")
    ax_ts.plot(t, x0_sindy, color="tomato",    lw=1.2, ls=":",  label="SINDy->Decoder")
    ax_ts.set_ylabel("x(t)  [observable dim]", fontsize=11)
    ax_ts.set_title("Observable Dimension: True x(t) vs Reconstructions", fontsize=12)
    ax_ts.legend(fontsize=10); ax_ts.grid(True, alpha=0.3)

    ax_err.semilogy(t, err_enc,   color="steelblue", lw=1.4, label="Encoder err²")
    ax_err.semilogy(t, err_sindy, color="tomato",    lw=1.4, ls="--", label="SINDy err²")
    ax_err.set_xlabel("t", fontsize=11)
    ax_err.set_ylabel("Squared error  x(t)", fontsize=11)
    ax_err.set_title("Pointwise squared error on observable dimension", fontsize=12)
    ax_err.legend(fontsize=10); ax_err.grid(True, which="both", alpha=0.3)

    rmse_enc   = float(np.sqrt(np.mean(err_enc)))
    rmse_sindy = float(np.sqrt(np.mean(err_sindy)))
    fig.suptitle(
        f"Observable Dimension Comparison — IC #{ic_idx}\n"
        f"RMSE  Encoder: {rmse_enc:.4f}   SINDy: {rmse_sindy:.4f}",
        fontsize=13, fontweight="bold",
    )
    path = os.path.join(out_dir, "fig6_x_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Van der Pol SINDy-AE results from precomputed eval data."
    )
    parser.add_argument("--eval", required=True,
                        help="Path to eval_<stamp>.npz produced by analyze.py")
    parser.add_argument("--ic", type=int, default=0,
                        help="Which test IC to use for time-series / reconstruction plots (default: 0)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for figures (default: same dir as --eval)")
    args = parser.parse_args()

    print(f"Loading eval data from {args.eval} ...")
    ev = _load_eval(args.eval)
    print(f"  latent_dim={ev['latent_dim']}, library_dim={ev['library_dim']}, "
          f"model_stamp={ev['model_stamp']}")

    n_steps = len(ev["test_t"])
    n_ics   = ev["test_x"].shape[0] // n_steps
    print(f"  {n_ics} ICs x {n_steps} steps = {ev['test_x'].shape[0]} samples")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.eval))
    os.makedirs(out_dir, exist_ok=True)
    ic = args.ic
    print(f"\nGenerating figures (IC #{ic} for time-series plots) into: {out_dir}/")

    def _safe(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")

    _safe(fig0_1_train_reconstruction,   ev, out_dir, ic)
    _safe(fig0_2_train_sindy_simulation, ev, out_dir, ic)
    _safe(fig0_3_train_x_comparison,     ev, out_dir, ic)
    _safe(fig1_phase_portrait,     ev, out_dir)
    _safe(fig2_reconstruction,     ev, out_dir, ic)
    _safe(fig3_time_series,        ev, out_dir, ic)
    _safe(fig4_sindy_simulation,   ev, out_dir, ic)
    _safe(fig5_long_term_phase,    ev, out_dir, ic)
    _safe(fig6_x_comparison,       ev, out_dir, ic)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
