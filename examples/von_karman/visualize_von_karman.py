"""Visualize trained Von Kármán model: learned latent dynamics vs true state.

Auto-dispatches between 2D rendering (latent_dim=2, Hopf normal form) and
3D rendering (latent_dim=3, Noack mean-field).

Generates figures saved next to the .mat checkpoint:
    fig1_phase_portrait.png     - latent_dim=2: 2D limit cycle, true z vs latent ξ
    fig1_3d_attractor.png       - latent_dim=3: 3D attractor, true z vs latent ξ
    fig2_reconstruction.png     - Input x and dx reconstruction quality (1 IC)
    fig3_time_series.png        - Time series: true z vs latent ξ (1 IC)
    fig4_sindy_simulation.png   - SINDy ODE forward simulation vs encoder
    fig4b_sindy_2d/3d.png       - latent space: encoder vs SINDy propagated
    fig4c_sindy_overlay.gif     - animated overlay
    fig5_multi_ic_attractor.png - Multi-IC: encoder vs SINDy attractor coverage
    fig6_x_comparison.png       - Observable x(t): true vs encoder/SINDy reconstruction

Usage (run from examples/von_karman/):
    python3 visualize_von_karman.py --mat checkpoints/model_<TS>.mat \\
                                     --data delay_xcoordinate_hopf_d10_5.npz
    python3 visualize_von_karman.py --mat checkpoints/model_<TS>.mat \\
                                     --data delay_xcoordinate_noack_d10_5.npz --ic 3
"""
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import solve_ivp

from sindyae import load_model_mat, load_model_json
from sindyae.sindy_library import sindy_library_torch


# ─── helpers ────────────────────────────────────────────────────────────────

def _encode(model, x_np, device, bs=4096):
    out = []
    for i in range(0, len(x_np), bs):
        xb = torch.tensor(x_np[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            out.append(model.encoder(xb).cpu().numpy())
    return np.concatenate(out)


def _decode(model, z_np, device, bs=4096):
    out = []
    for i in range(0, len(z_np), bs):
        zb = torch.tensor(z_np[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            out.append(model.decoder(zb).cpu().numpy())
    return np.concatenate(out)


def _sindy_rhs(Xi, poly_order, include_sine):
    def rhs(t, z):
        zt = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            Theta = sindy_library_torch(zt, poly_order, include_sine)
        return (Theta.numpy() @ Xi)[0]
    return rhs


def _ic_slice(ic_idx, n_steps):
    return slice(ic_idx * n_steps, (ic_idx + 1) * n_steps)


def _model_label(model_name):
    if model_name == "hopf":
        return "Von Kármán (Hopf normal form)"
    if model_name == "noack":
        return "Von Kármán (Noack mean-field)"
    return f"Von Kármán ({model_name})"


# ─── figure 1: phase portrait (2D) or 3D attractor ────────────────────────────

def fig1_phase_portrait_2d(model, test_data, device, out_dir, label):
    """2D limit-cycle comparison: true state vs learned latent (all ICs)."""
    has_z = 'z' in test_data
    z_true = test_data['z'].reshape(-1, 2) if has_z else test_data['x'][:, :2]
    z_lat  = _encode(model, test_data['x'], device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(z_true[:, 0], z_true[:, 1], color='steelblue', lw=0.25, alpha=0.5)
    ax1.set_title('True limit cycle\n(normalized z₀, z₁)' if has_z
                  else 'Observed delay coordinates\n(x[t], x[t-τ])', fontsize=12)
    ax1.set_xlabel('z₀' if has_z else 'x[t]')
    ax1.set_ylabel('z₁' if has_z else 'x[t-τ]')
    ax1.grid(True, alpha=0.3); ax1.set_aspect('equal', 'box')

    ax2.plot(z_lat[:, 0], z_lat[:, 1], color='tomato', lw=0.25, alpha=0.5)
    ax2.set_title('Learned latent phase portrait\n(encoder ξ₀, ξ₁)', fontsize=12)
    ax2.set_xlabel('ξ₀'); ax2.set_ylabel('ξ₁')
    ax2.grid(True, alpha=0.3); ax2.set_aspect('equal', 'box')

    fig.suptitle(f'{label} — Phase Portrait (All Test ICs)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1_phase_portrait.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig1_3d_attractor(model, test_data, device, out_dir, label):
    """3D attractor comparison: true state vs learned latent (all ICs)."""
    has_z = 'z' in test_data
    z_true = test_data['z'].reshape(-1, 3) if has_z else test_data['x'][:, :3]
    z_lat  = _encode(model, test_data['x'], device)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2],
             color='steelblue', lw=0.25, alpha=0.5)
    if has_z:
        ax1.set_title('True attractor\n(normalized z₀, z₁, z₂)', fontsize=12)
        ax1.set_xlabel('z₀'); ax1.set_ylabel('z₁'); ax1.set_zlabel('z₂')
    else:
        ax1.set_title('Observed delay coordinates\n(x[t], x[t-τ], x[t-2τ])', fontsize=12)
        ax1.set_xlabel('x[t]'); ax1.set_ylabel('x[t-τ]'); ax1.set_zlabel('x[t-2τ]')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(z_lat[:, 0], z_lat[:, 1], z_lat[:, 2],
             color='tomato', lw=0.25, alpha=0.5)
    ax2.set_title('Learned latent attractor\n(encoder ξ₀, ξ₁, ξ₂)', fontsize=12)
    ax2.set_xlabel('ξ₀'); ax2.set_ylabel('ξ₁'); ax2.set_zlabel('ξ₂')

    fig.suptitle(f'{label} — 3D Attractor (All Test ICs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_3d_attractor.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 2: reconstruction quality (dim-agnostic) ─────────────────────────

def fig2_reconstruction(model, test_data, device, out_dir, ic_idx, label):
    t       = test_data['t']
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)

    x_np  = test_data['x'][sl]
    dx_np = test_data['dx'][sl]

    x_t  = torch.tensor(x_np,  dtype=torch.float32, device=device)
    dx_t = torch.tensor(dx_np, dtype=torch.float32, device=device)
    ddx_t = None
    if 'ddx' in test_data:
        ddx_t = torch.tensor(test_data['ddx'][sl], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(x_t, dx_t, ddx_t)
    x_dec = outputs['x_decode'].cpu().numpy()

    order2 = model.model_order == 2
    if order2:
        deriv_dec   = outputs['ddx_decode'].cpu().numpy()
        deriv_np    = test_data['ddx'][sl]
        deriv_label = 'ddx'
    else:
        deriv_dec   = outputs['dx_decode'].cpu().numpy()
        deriv_np    = dx_np
        deriv_label = 'dx'

    n_dims = x_np.shape[1]
    dims = [0, n_dims // 2, n_dims - 1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    for row, d in enumerate(dims):
        axes[row, 0].plot(t, x_np[:, d],  'steelblue', lw=1.5, label='True')
        axes[row, 0].plot(t, x_dec[:, d], 'tomato',    lw=1.2, ls='--', label='Decoded')
        axes[row, 0].set_ylabel(f'x[{d}]')
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, deriv_np[:, d],  'steelblue', lw=1.5, label='True')
        axes[row, 1].plot(t, deriv_dec[:, d], 'tomato',    lw=1.2, ls='--', label='Decoded')
        axes[row, 1].set_ylabel(f'{deriv_label}[{d}]')
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title('Input reconstruction  x', fontsize=11)
    axes[0, 1].set_title(f'Derivative reconstruction  {deriv_label}', fontsize=11)
    axes[-1, 0].set_xlabel('t'); axes[-1, 1].set_xlabel('t')

    rel_x     = np.mean((x_dec     - x_np    ) ** 2) / np.mean(x_np     ** 2)
    rel_deriv = np.mean((deriv_dec - deriv_np) ** 2) / np.mean(deriv_np ** 2)
    fig.suptitle(
        f'{label} — Reconstruction Quality (IC #{ic_idx})\n'
        f'relative err x: {rel_x:.2e}   relative err {deriv_label}: {rel_deriv:.2e}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_reconstruction.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 3: per-dim time series (latent_dim 2 or 3) ───────────────────────

def fig3_time_series(model, test_data, device, out_dir, ic_idx, label):
    t       = test_data['t']
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)
    has_z   = 'z' in test_data

    latent_dim = model.latent_dim
    state_dim  = test_data['z'].shape[1] if has_z else latent_dim
    n_rows     = max(latent_dim, state_dim) if has_z else latent_dim

    if has_z:
        z_true = test_data['z'][sl]
    else:
        z_true = test_data['x'][sl, :latent_dim]
    z_lat = _encode(model, test_data['x'][sl], device)

    colors_true = ['steelblue', 'darkorange', 'seagreen']
    colors_lat  = ['tomato',    'orchid',     'goldenrod']

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows), sharex=True,
                              squeeze=False)

    for i in range(n_rows):
        if has_z and i < state_dim:
            axes[i, 0].plot(t, z_true[:, i], color=colors_true[i % 3], lw=1.2)
            axes[i, 0].set_ylabel(f'z{i} (true)', fontsize=10)
        elif not has_z and i < latent_dim:
            axes[i, 0].plot(t, z_true[:, i], color=colors_true[i % 3], lw=1.2)
            axes[i, 0].set_ylabel(f'x_delay[{i}]', fontsize=10)
        else:
            axes[i, 0].axis('off')
        axes[i, 0].grid(True, alpha=0.3)

        if i < latent_dim:
            axes[i, 1].plot(t, z_lat[:, i], color=colors_lat[i % 3], lw=1.2)
            axes[i, 1].set_ylabel(f'ξ{i} (learned)', fontsize=10)
            axes[i, 1].grid(True, alpha=0.3)
        else:
            axes[i, 1].axis('off')

    axes[0, 0].set_title('True state' if has_z else 'Observed delay coords', fontsize=11)
    axes[0, 1].set_title('Learned latent variables', fontsize=11)
    axes[-1, 0].set_xlabel('t'); axes[-1, 1].set_xlabel('t')

    fig.suptitle(f'{label} — Per-dimension Time Series (IC #{ic_idx})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_time_series.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 4: SINDy forward simulation (per-dim time series + 2D/3D side-by-side + animation) ─

def _integrate_sindy(model, params, x0_np, t):
    """Integrate the learned SINDy ODE from a single x0 row. Returns (z_sindy, sim_ok)."""
    x0 = torch.tensor(x0_np[None, :], dtype=torch.float32)
    with torch.no_grad():
        z0 = model.encoder(x0).cpu().numpy()[0]
    Xi           = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order   = params['poly_order']
    include_sine = params.get('include_sine', False)
    rhs = _sindy_rhs(Xi, poly_order, include_sine)
    try:
        sol = solve_ivp(rhs, [t[0], t[-1]], z0, t_eval=t,
                        method='RK45', rtol=1e-6, atol=1e-9, max_step=0.02)
        sim_ok = sol.success and not np.any(np.isnan(sol.y))
        z_sindy = sol.y[:model.latent_dim, :].T if sim_ok else None
    except Exception as e:
        print(f"  Warning: SINDy integration failed ({e})")
        sim_ok, z_sindy = False, None
    return z_sindy, sim_ok, z0


def fig4_sindy_simulation(model, params, test_data, device, out_dir, ic_idx, label):
    t       = test_data['t']
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)
    x_np    = test_data['x'][sl]
    latent_dim = model.latent_dim

    z_sindy, sim_ok, _ = _integrate_sindy(model, params, x_np[0], t)
    z_enc = _encode(model, x_np, device)

    fig, axes = plt.subplots(latent_dim, 1, figsize=(12, 3 * latent_dim),
                              sharex=True, squeeze=False)
    for i in range(latent_dim):
        ax = axes[i, 0]
        ax.plot(t, z_enc[:, i], 'steelblue', lw=1.4, label=f'Encoder ξ{i}')
        if sim_ok:
            ax.plot(t, z_sindy[:, i], 'tomato', lw=1.2, ls='--',
                    label=f'SINDy ODE ξ{i}')
        ax.set_ylabel(f'ξ{i}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('t')
    status = 'converged' if sim_ok else '⚠ DIVERGED'
    axes[0, 0].set_title(
        f'{label} — SINDy Forward Simulation (IC #{ic_idx})  [{status}]',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_sindy_simulation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    if not sim_ok:
        return

    if latent_dim == 2:
        _fig4b_2d(z_enc, z_sindy, ic_idx, out_dir, label)
        _fig4c_2d_overlay(z_enc, z_sindy, t, ic_idx, out_dir, label)
    else:
        _fig4b_3d(z_enc, z_sindy, ic_idx, out_dir, label)
        _fig4c_3d_overlay(z_enc, z_sindy, t, ic_idx, out_dir, label)


def _fig4b_2d(z_enc, z_sindy, ic_idx, out_dir, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, traj, color, title in (
        (ax1, z_enc,   'steelblue', 'Encoder trajectory\n(true data → encoder)'),
        (ax2, z_sindy, 'tomato',    'SINDy propagated trajectory\n(learned ODE)'),
    ):
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.0, alpha=0.85)
        ax.scatter(traj[0, 0],  traj[0, 1],  color='lime',  s=80, zorder=6, marker='o', label='start')
        ax.scatter(traj[-1, 0], traj[-1, 1], color='black', s=80, zorder=6, marker='s', label='end')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('ξ₀'); ax.set_ylabel('ξ₁')
        ax.set_aspect('equal', 'box')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f'{label} — 2D Latent Space (IC #{ic_idx})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig4b_sindy_2d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _fig4b_3d(z_enc, z_sindy, ic_idx, out_dir, label):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    for ax, traj, color, title in (
        (ax1, z_enc,   'steelblue', 'Encoder trajectory\n(true data → encoder)'),
        (ax2, z_sindy, 'tomato',    'SINDy propagated trajectory\n(learned ODE)'),
    ):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=1.0, alpha=0.85)
        ax.scatter(*traj[0],  color='lime',  s=80, zorder=6, marker='o', label='start')
        ax.scatter(*traj[-1], color='black', s=80, zorder=6, marker='s', label='end')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('ξ₀'); ax.set_ylabel('ξ₁'); ax.set_zlabel('ξ₂')
        ax.legend(fontsize=8)

    fig.suptitle(f'{label} — 3D Latent Space (IC #{ic_idx})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig4b_sindy_3d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _fig4c_2d_overlay(z_enc, z_sindy, t, ic_idx, out_dir, label):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    n_frames = min(200, len(t))
    frame_idx = np.linspace(0, len(t) - 1, n_frames, dtype=int)

    all_pts = np.vstack([z_enc, z_sindy])
    cx = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
    cy = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
    r  = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
             all_pts[:, 1].max() - all_pts[:, 1].min()) / 2 * 1.15

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_xlabel('ξ₀'); ax.set_ylabel('ξ₁')
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{label} — 2D Overlay (IC #{ic_idx})',
                 fontsize=12, fontweight='bold')

    ax.plot(z_enc[:, 0],   z_enc[:, 1],   color='steelblue', lw=0.5, alpha=0.2)
    ax.plot(z_sindy[:, 0], z_sindy[:, 1], color='tomato',    lw=0.5, alpha=0.2)
    ax.scatter(z_enc[0, 0],   z_enc[0, 1],   color='steelblue', s=80, zorder=6, marker='o')
    ax.scatter(z_sindy[0, 0], z_sindy[0, 1], color='tomato',    s=80, zorder=6, marker='o')

    line_enc,   = ax.plot([], [], color='steelblue', lw=1.4)
    line_sindy, = ax.plot([], [], color='tomato',    lw=1.4, ls='--')
    dot_enc,    = ax.plot([], [], 'o', color='steelblue', ms=7, zorder=7)
    dot_sindy,  = ax.plot([], [], 'o', color='tomato',    ms=7, zorder=7)
    time_text   = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                          fontsize=9, va='top')

    legend_handles = [
        Line2D([0], [0], color='steelblue', lw=1.4,              label='Encoder'),
        Line2D([0], [0], color='tomato',    lw=1.4, ls='--',     label='SINDy ODE'),
        Line2D([0], [0], marker='o', color='gray', linestyle='none', ms=7, label='start'),
        Line2D([0], [0], marker='s', color='gray', linestyle='none', ms=7, label='end'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper right')
    fig.tight_layout()

    def _init():
        line_enc.set_data([], []); line_sindy.set_data([], [])
        dot_enc.set_data([], []);  dot_sindy.set_data([], [])
        time_text.set_text('')
        return line_enc, line_sindy, dot_enc, dot_sindy, time_text

    def _update(frame):
        k = frame_idx[frame]
        line_enc.set_data(z_enc[:k+1, 0],   z_enc[:k+1, 1])
        line_sindy.set_data(z_sindy[:k+1, 0], z_sindy[:k+1, 1])
        dot_enc.set_data([z_enc[k, 0]],   [z_enc[k, 1]])
        dot_sindy.set_data([z_sindy[k, 0]], [z_sindy[k, 1]])
        time_text.set_text(f't = {t[k]:.2f}')
        if frame == n_frames - 1:
            ax.scatter(z_enc[-1, 0],   z_enc[-1, 1],   color='steelblue', s=80, zorder=8, marker='s')
            ax.scatter(z_sindy[-1, 0], z_sindy[-1, 1], color='tomato',    s=80, zorder=8, marker='s')
        return line_enc, line_sindy, dot_enc, dot_sindy, time_text

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         init_func=_init, blit=True, interval=50)
    path = os.path.join(out_dir, 'fig4c_sindy_overlay.gif')
    anim.save(path, writer=PillowWriter(fps=20))
    plt.close()
    print(f"  Saved: {path}")


def _fig4c_3d_overlay(z_enc, z_sindy, t, ic_idx, out_dir, label):
    from matplotlib.animation import FuncAnimation, PillowWriter

    n_frames  = min(200, len(t))
    frame_idx = np.linspace(0, len(t) - 1, n_frames, dtype=int)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(z_enc[:, 0],   z_enc[:, 1],   z_enc[:, 2],
            color='steelblue', lw=0.5, alpha=0.2)
    ax.plot(z_sindy[:, 0], z_sindy[:, 1], z_sindy[:, 2],
            color='tomato',    lw=0.5, alpha=0.2)
    ax.scatter(*z_enc[0],   color='lime', s=80, zorder=6, marker='o')

    line_enc,   = ax.plot([], [], [], color='steelblue', lw=1.4, label='Encoder')
    line_sindy, = ax.plot([], [], [], color='tomato',    lw=1.4, ls='--', label='SINDy ODE')
    dot_enc,    = ax.plot([], [], [], 'o', color='steelblue', ms=7, zorder=7)
    dot_sindy,  = ax.plot([], [], [], 'o', color='tomato',    ms=7, zorder=7)
    time_text   = ax.text2D(0.02, 0.96, '', transform=ax.transAxes,
                            fontsize=9, va='top')

    ax.set_xlabel('ξ₀'); ax.set_ylabel('ξ₁'); ax.set_zlabel('ξ₂')
    ax.set_title(f'{label} — 3D Overlay (IC #{ic_idx})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    def _init():
        line_enc.set_data_3d([], [], []); line_sindy.set_data_3d([], [], [])
        dot_enc.set_data_3d([], [], []);  dot_sindy.set_data_3d([], [], [])
        time_text.set_text('')
        return line_enc, line_sindy, dot_enc, dot_sindy, time_text

    def _update(frame):
        k = frame_idx[frame]
        line_enc.set_data_3d(z_enc[:k+1, 0],   z_enc[:k+1, 1],   z_enc[:k+1, 2])
        line_sindy.set_data_3d(z_sindy[:k+1, 0], z_sindy[:k+1, 1], z_sindy[:k+1, 2])
        dot_enc.set_data_3d([z_enc[k, 0]],   [z_enc[k, 1]],   [z_enc[k, 2]])
        dot_sindy.set_data_3d([z_sindy[k, 0]], [z_sindy[k, 1]], [z_sindy[k, 2]])
        time_text.set_text(f't = {t[k]:.2f}')
        if frame == n_frames - 1:
            ax.scatter(*z_enc[-1],   color='steelblue', s=80, zorder=8, marker='s')
            ax.scatter(*z_sindy[-1], color='tomato',    s=80, zorder=8, marker='s')
        return line_enc, line_sindy, dot_enc, dot_sindy, time_text

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         init_func=_init, blit=False, interval=50)
    path = os.path.join(out_dir, 'fig4c_sindy_overlay.gif')
    anim.save(path, writer=PillowWriter(fps=20))
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 5: multi-IC attractor coverage ──────────────────────────────────

def fig5_multi_ic_attractor(model, params, test_data, device, out_dir, n_ics_plot, label):
    t       = test_data['t']
    n_steps = len(t)
    n_ics_total = test_data['x'].shape[0] // n_steps
    n_show = min(n_ics_plot, n_ics_total)
    ic_indices = np.linspace(0, n_ics_total - 1, n_show, dtype=int)

    Xi          = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order  = params['poly_order']
    include_sine = params.get('include_sine', False)
    latent_dim  = model.latent_dim
    colors      = plt.cm.tab10(np.arange(n_show) % 10)

    fig = plt.figure(figsize=(12, 5))
    if latent_dim == 2:
        ax_enc = fig.add_subplot(121)
        ax_sin = fig.add_subplot(122)
    else:
        ax_enc = fig.add_subplot(121, projection='3d')
        ax_sin = fig.add_subplot(122, projection='3d')

    n_converged = 0
    for ci, ic_idx in enumerate(ic_indices):
        sl   = _ic_slice(ic_idx, n_steps)
        x_np = test_data['x'][sl]
        c    = colors[ci]

        z_enc = _encode(model, x_np, device)
        x0_t  = torch.tensor(x_np[0:1], dtype=torch.float32, device=device)
        with torch.no_grad():
            z0_np = model.encoder(x0_t).cpu().numpy()[0]

        if latent_dim == 2:
            ax_enc.plot(z_enc[:, 0], z_enc[:, 1], color=c, lw=0.8, alpha=0.75)
            ax_enc.scatter(*z_enc[0], color=c, s=25, zorder=5)
        else:
            ax_enc.plot(z_enc[:, 0], z_enc[:, 1], z_enc[:, 2],
                        color=c, lw=0.7, alpha=0.75)
            ax_enc.scatter(*z_enc[0], color=c, s=25, zorder=5)

        rhs = _sindy_rhs(Xi, poly_order, include_sine)
        try:
            sol = solve_ivp(rhs, [t[0], t[-1]], z0_np, t_eval=t,
                            method='RK45', rtol=1e-6, atol=1e-9, max_step=0.02)
            if sol.success and not np.any(np.isnan(sol.y)):
                z_sin = sol.y[:latent_dim, :].T
                if latent_dim == 2:
                    ax_sin.plot(z_sin[:, 0], z_sin[:, 1], color=c, lw=0.8, alpha=0.75)
                    ax_sin.scatter(*z0_np, color=c, s=25, zorder=5)
                else:
                    ax_sin.plot(z_sin[:, 0], z_sin[:, 1], z_sin[:, 2],
                                color=c, lw=0.7, alpha=0.75)
                    ax_sin.scatter(*z0_np, color=c, s=25, zorder=5)
                n_converged += 1
            else:
                ax_sin.scatter(*z0_np, color=c, s=60, marker='x', zorder=5)
        except Exception:
            ax_sin.scatter(*z0_np, color=c, s=60, marker='x', zorder=5)

    if latent_dim == 2:
        for ax in (ax_enc, ax_sin):
            ax.set_aspect('equal', 'box')
            ax.set_xlabel('ξ₀'); ax.set_ylabel('ξ₁'); ax.grid(True, alpha=0.3)
    else:
        for ax in (ax_enc, ax_sin):
            ax.set_xlabel('ξ₀'); ax.set_ylabel('ξ₁'); ax.set_zlabel('ξ₂')

    ax_enc.set_title(f'Encoder trajectories ({n_show} ICs)', fontsize=11)
    ax_sin.set_title(f'SINDy ODE trajectories\n({n_converged}/{n_show} converged)',
                     fontsize=11)

    fig.suptitle(f'{label} — Multi-IC Attractor', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_multi_ic_attractor.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}  ({n_converged}/{n_show} SINDy ICs converged)")


# ─── figure 6: observable dimension comparison ──────────────────────────────

def fig6_x_comparison(model, params, test_data, device, out_dir, ic_idx, label):
    t       = test_data['t']
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)
    x_np    = test_data['x'][sl]

    z_enc = _encode(model, x_np, device)
    x_enc = _decode(model, z_enc, device)

    z_sindy, sim_ok, _ = _integrate_sindy(model, params, x_np[0], t)
    if not sim_ok:
        print("  Skipping fig6: SINDy integration diverged.")
        return
    x_sindy = _decode(model, z_sindy, device)

    x0_true  = x_np[:, 0]
    x0_enc   = x_enc[:, 0]
    x0_sindy = x_sindy[:, 0]

    err_enc   = (x0_enc   - x0_true) ** 2
    err_sindy = (x0_sindy - x0_true) ** 2

    fig, (ax_ts, ax_err) = plt.subplots(2, 1, figsize=(12, 8),
                                         sharex=True,
                                         gridspec_kw={'height_ratios': [1.4, 1],
                                                      'hspace': 0.35})

    ax_ts.plot(t, x0_true,  color='#333333',   lw=1.5, label='True x(t)')
    ax_ts.plot(t, x0_enc,   color='steelblue', lw=1.2, ls='--', label='Encoder→Decoder')
    ax_ts.plot(t, x0_sindy, color='tomato',    lw=1.2, ls=':',  label='SINDy→Decoder')
    ax_ts.set_ylabel('x(t)  [observable dim]', fontsize=11)
    ax_ts.set_title('Observable Dimension: True x(t) vs Reconstructions', fontsize=12)
    ax_ts.legend(fontsize=10); ax_ts.grid(True, alpha=0.3)

    ax_err.semilogy(t, err_enc,   color='steelblue', lw=1.4, label='Encoder err²')
    ax_err.semilogy(t, err_sindy, color='tomato',    lw=1.4, ls='--', label='SINDy err²')
    ax_err.set_xlabel('t', fontsize=11)
    ax_err.set_ylabel('Squared error  x(t)', fontsize=11)
    ax_err.set_title('Pointwise squared error on observable dimension', fontsize=12)
    ax_err.legend(fontsize=10); ax_err.grid(True, which='both', alpha=0.3)

    rmse_enc   = float(np.sqrt(np.mean(err_enc)))
    rmse_sindy = float(np.sqrt(np.mean(err_sindy)))
    fig.suptitle(
        f'{label} — Observable Dim (IC #{ic_idx})\n'
        f'RMSE  Encoder: {rmse_enc:.4f}   SINDy: {rmse_sindy:.4f}',
        fontsize=13, fontweight='bold'
    )
    path = os.path.join(out_dir, 'fig6_x_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', required=True,
                        help='Path to checkpoint .mat (or .json) file')
    parser.add_argument('--data', required=True,
                        help='Path to .npz data file; uses test split (falls back to val)')
    parser.add_argument('--ic', type=int, default=0,
                        help='Test IC for time-series plots (default: 0)')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory for figures (default: same as .mat)')
    parser.add_argument('--n_ics_plot', type=int, default=10,
                        help='Number of ICs in fig5 (default: 10)')
    args = parser.parse_args()

    device = torch.device('cpu')

    if args.mat.endswith('.json'):
        mat_prefix = args.mat[:-5]; fmt = 'json'
    else:
        mat_prefix = args.mat[:-4] if args.mat.endswith('.mat') else args.mat
        fmt = 'mat'

    print(f"Loading model from {mat_prefix} ({fmt})...")
    if fmt == 'mat':
        model, params = load_model_mat(mat_prefix, device=device)
    else:
        model, params = load_model_json(mat_prefix, device=device)
    model.eval()
    latent_dim = params['latent_dim']
    print(f"  latent_dim={latent_dim}, library_dim={params['library_dim']}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(mat_prefix))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from {args.data} ...")
    npz = np.load(args.data, allow_pickle=False)
    split = 'test' if 'test_x' in npz.files else 'val'
    print(f"  Using '{split}' split.")
    t_key = f'{split}_t' if f'{split}_t' in npz.files else 'train_t'
    test_data = {
        'x':  npz[f'{split}_x'],
        'dx': npz[f'{split}_dx'],
        't':  npz[t_key],
    }
    if f'{split}_ddx' in npz.files:
        test_data['ddx'] = npz[f'{split}_ddx']
    if f'{split}_z' in npz.files:
        test_data['z'] = npz[f'{split}_z']

    model_name = str(npz['model']) if 'model' in npz.files else 'unknown'
    label = _model_label(model_name)
    n_steps = len(test_data['t'])
    n_ics   = test_data['x'].shape[0] // n_steps
    print(f"  {n_ics} ICs × {n_steps} steps = {test_data['x'].shape[0]} samples")
    print(f"  Model: {label}")

    ic = args.ic
    print(f"\nGenerating figures (IC #{ic} for time-series plots)...")
    print(f"Figures will be saved to: {out_dir}/")

    if latent_dim == 2:
        fig1_phase_portrait_2d(model, test_data, device, out_dir, label)
    elif latent_dim == 3:
        fig1_3d_attractor(model, test_data, device, out_dir, label)
    else:
        print(f"  Skipping fig1: latent_dim={latent_dim} not supported (expected 2 or 3)")

    fig2_reconstruction(model, test_data, device, out_dir, ic, label)
    fig3_time_series(model, test_data, device, out_dir, ic, label)
    fig4_sindy_simulation(model, params, test_data, device, out_dir, ic, label)
    fig5_multi_ic_attractor(model, params, test_data, device, out_dir,
                             args.n_ics_plot, label)
    fig6_x_comparison(model, params, test_data, device, out_dir, ic, label)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
