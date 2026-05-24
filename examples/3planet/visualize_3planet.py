"""Visualize trained SINDy-AE model for 3-Body (BHH) Problem.

Generates figures saved next to the .mat checkpoint:
    fig1_phase_portrait.png   - 2D Phase Portrait: True (x1, v_x1) vs Learned latent (ξ₀, ξ₁)
    fig2_reconstruction.png   - Input delay vector x and dx reconstruction quality
    fig3_time_series.png      - Time series: true (x1, v_x1) vs latent (ξ₀, ξ₁)
    fig4_sindy_simulation.png - SINDy ODE forward simulation vs encoder trajectory
    fig4b_sindy_2d.png        - 2D: encoder vs SINDy propagated, with start/end markers
    fig4c_sindy_overlay.gif   - Animated 2D overlay: encoder and SINDy on the same axes
    fig5_long_term_phase.png  - Long-term continuous phase portrait comparison
    fig6_x_comparison.png     - Observable dim x1(t): true vs encoder-decoded vs SINDy-decoded

Usage:
    python3 visualize_3planet.py --mat checkpoints/model_YYYYMMDD_HHMMSS.mat --data delay_3body_x1_d5_10.npz
"""
import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ensure SINDy-AE modules can be imported
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sindyae import load_model_mat, load_model_json
from sindyae.sindy_library import sindy_library_torch


# ─── Helpers ────────────────────────────────────────────────────────────────

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


# ─── Figure 1: 2D Phase Portrait ─────────────────────────────────────────────

def fig1_phase_portrait(model, test_data, device, out_dir):
    """True (x1, v_x1) phase portrait vs learned latent space."""
    z_true = test_data['z']  # Contains [x_1, v_x1] from ground truth
    z_lat  = _encode(model, test_data['x'], device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # True Phase Portrait (Rosette pattern)
    ax1.plot(z_true[:, 0], z_true[:, 1], color='steelblue', lw=0.5, alpha=0.8)
    ax1.set_title('True 3-Body BHH Orbit\n(Planet 1: $x_1$ vs $v_{x1}$)', fontsize=12)
    ax1.set_xlabel('Position $x_1$'); ax1.set_ylabel('Velocity $v_{x1}$')
    ax1.grid(True, alpha=0.3)

    # Latent Phase Portrait
    ax2.plot(z_lat[:, 0], z_lat[:, 1], color='tomato', lw=0.5, alpha=0.8)
    ax2.set_title('Learned Latent Phase Portrait\n(Encoder output $\\xi_0, \\xi_1$)', fontsize=12)
    ax2.set_xlabel('$\\xi_0$'); ax2.set_ylabel('$\\xi_1$')
    ax2.grid(True, alpha=0.3)

    for ax, data in ((ax1, z_true), (ax2, z_lat)):
        cx = (data[:, 0].max() + data[:, 0].min()) / 2
        cy = (data[:, 1].max() + data[:, 1].min()) / 2
        r  = max(data[:, 0].max() - data[:, 0].min(),
                 data[:, 1].max() - data[:, 1].min()) / 2 * 1.1
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)

    fig.suptitle('Phase Portrait Comparison — Test Set (Continuous Trajectory)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1_phase_portrait.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 2: Reconstruction Quality ───────────────────────────────────────

def fig2_reconstruction(model, test_data, device, out_dir):
    """x and dx delay vector reconstruction quality."""
    t     = test_data['t']
    x_np  = test_data['x']
    dx_np = test_data['dx']

    x_t  = torch.tensor(x_np,  dtype=torch.float32, device=device)
    dx_t = torch.tensor(dx_np, dtype=torch.float32, device=device)
    ddx_t = None

    with torch.no_grad():
        outputs = model(x_t, dx_t, ddx_t)
    x_dec = outputs['x_decode'].cpu().numpy()
    deriv_dec = outputs['dx_decode'].cpu().numpy()
    deriv_np  = dx_np
    deriv_label = 'dx'

    n_dims = x_np.shape[1]
    # Check 3 dimensions of the delay vector: current, middle delay, max delay
    dims = [0, n_dims // 2, n_dims - 1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    for row, d in enumerate(dims):
        axes[row, 0].plot(t, x_np[:, d],    'steelblue', lw=1.5, label='True')
        axes[row, 0].plot(t, x_dec[:, d],   'tomato',    lw=1.2, ls='--', label='Decoded')
        axes[row, 0].set_ylabel(f'Delay $x_{{{d}}}$')
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, deriv_np[:, d],  'steelblue', lw=1.5, label='True')
        axes[row, 1].plot(t, deriv_dec[:, d], 'tomato',    lw=1.2, ls='--', label='Decoded')
        axes[row, 1].set_ylabel(f'Delay ${deriv_label}_{{{d}}}$')
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title('Input Reconstruction: Delay Vector $\\mathbf{x}$', fontsize=11)
    axes[0, 1].set_title(f'Derivative Reconstruction: $\\mathbf{{{deriv_label}}}$', fontsize=11)
    axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')

    rel_x     = np.mean((x_dec     - x_np    ) ** 2) / np.mean(x_np     ** 2)
    rel_deriv = np.mean((deriv_dec - deriv_np) ** 2) / np.mean(deriv_np ** 2)
    fig.suptitle(
        f'Delay Embedding Reconstruction Quality\n'
        f'Relative Error x: {rel_x:.2e}   Relative Error {deriv_label}: {rel_deriv:.2e}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_reconstruction.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 3: Time Series ───────────────────────────────────────────────────

def fig3_time_series(model, test_data, device, out_dir):
    """True (x1, v_x1) vs latent (ξ₀, ξ₁) over time."""
    t      = test_data['t']
    z_true = test_data['z']
    z_lat  = _encode(model, test_data['x'], device)

    colors_true = ['steelblue', 'darkorange']
    colors_lat  = ['tomato',    'orchid']
    ylabels_true = ['$x_1(t)$', '$v_{x1}(t)$']

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)

    for i in range(2):
        axes[i, 0].plot(t, z_true[:, i], color=colors_true[i], lw=1.2)
        axes[i, 0].set_ylabel(ylabels_true[i], fontsize=11)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(t, z_lat[:, i], color=colors_lat[i], lw=1.2)
        axes[i, 1].set_ylabel(f'$\\xi_{i}(t)$ (Learned)', fontsize=11)
        axes[i, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title('True 3-Body State (Planet 1)', fontsize=11)
    axes[0, 1].set_title('Learned Latent Variables', fontsize=11)
    axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')

    fig.suptitle('Time Series Comparison: True vs Latent Space', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_time_series.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 4: SINDy Forward Simulation ──────────────────────────────────────

def fig4_sindy_simulation(model, params, test_data, device, out_dir):
    """Integrate the learned SINDy ODE; compare with encoder trajectory."""
    t    = test_data['t']
    x_np = test_data['x']

    x0 = torch.tensor(x_np[0:1], dtype=torch.float32, device=device)
    with torch.no_grad():
        z0 = model.encoder(x0).cpu().numpy()[0]

    Xi           = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order   = params['poly_order']
    include_sine = params.get('include_sine', False)

    rhs = _sindy_rhs(Xi, poly_order, include_sine)
    try:
        sol = solve_ivp(rhs, [t[0], t[-1]], z0, t_eval=t,
                        method='DOP853', rtol=1e-8, atol=1e-10) # High precision for orbital dynamics
        sim_ok  = sol.success and not np.any(np.isnan(sol.y))
        z_sindy = sol.y.T if sim_ok else None
    except Exception as e:
        print(f"  Warning: SINDy integration failed ({e})")
        sim_ok, z_sindy = False, None

    z_enc = _encode(model, x_np, device)
    n_lat = z_enc.shape[1]

    # ── Time Series ──
    fig, axes = plt.subplots(n_lat, 1, figsize=(12, 2.5 * n_lat + 1), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(n_lat):
        axes[i].plot(t, z_enc[:, i], 'steelblue', lw=1.4, label='Encoder $\\xi_%d$' % i)
        if sim_ok:
            axes[i].plot(t, z_sindy[:, i], 'tomato', lw=1.2, ls='--', label='SINDy ODE $\\xi_%d$' % i)
        axes[i].set_ylabel(f'$\\xi_{i}$')
        axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    status = 'Converged' if sim_ok else '⚠ DIVERGED'
    axes[0].set_title(f'SINDy Forward Simulation vs Encoder Trajectory [{status}]', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_sindy_simulation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    if sim_ok:
        # ── 2D Pairwise Phase Portraits ──
        from itertools import combinations
        pairs = list(combinations(range(n_lat), 2))
        n_pairs = len(pairs)

        fig2d, axes2d = plt.subplots(n_pairs, 2, figsize=(12, 4.5 * n_pairs))
        axes2d = np.atleast_2d(axes2d)

        for row, (i, j) in enumerate(pairs):
            for col, (traj, color, title) in enumerate([
                (z_enc,   'steelblue', 'Encoder trajectory\n(Data → Encoder)'),
                (z_sindy, 'tomato',    'SINDy propagated trajectory\n(Learned ODE)'),
            ]):
                ax = axes2d[row, col]
                ax.plot(traj[:, i], traj[:, j], color=color, lw=0.5, alpha=0.85)
                ax.scatter(traj[0,  i], traj[0,  j], color='lime',  s=60, zorder=6, label='Start')
                ax.scatter(traj[-1, i], traj[-1, j], color='black', s=60, zorder=6, marker='s', label='End')
                ax.set_xlabel(f'$\\xi_{i}$'); ax.set_ylabel(f'$\\xi_{j}$')
                if row == 0:
                    ax.set_title(title, fontsize=11)
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig2d.suptitle('2D Latent Space: Encoder vs Learned SINDy ODE', fontsize=13, fontweight='bold')
        fig2d.tight_layout()
        path2d = os.path.join(out_dir, 'fig4b_sindy_2d.png')
        plt.savefig(path2d, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path2d}")


# ─── Figure 5: Long Term Phase Portrait (Replacing Multi-IC) ─────────────────

def fig5_long_term_phase(model, params, test_data, device, out_dir):
    """Simulate SINDy ODE and plot continuous Phase Portrait to check structure preservation."""
    t    = test_data['t']
    x_np = test_data['x']

    x0_t = torch.tensor(x_np[0:1], dtype=torch.float32, device=device)
    with torch.no_grad():
        z0_np = model.encoder(x0_t).cpu().numpy()[0]

    Xi          = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order  = params['poly_order']
    include_sine = params.get('include_sine', False)

    rhs = _sindy_rhs(Xi, poly_order, include_sine)
    try:
        sol = solve_ivp(rhs, [t[0], t[-1]], z0_np, t_eval=t,
                        method='DOP853', rtol=1e-8, atol=1e-10)
        sim_ok = sol.success and not np.any(np.isnan(sol.y))
        z_sin = sol.y[:2, :].T if sim_ok else None
    except Exception:
        sim_ok = False

    z_enc = _encode(model, x_np, device)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot Encoder as background
    ax.plot(z_enc[:, 0], z_enc[:, 1], color='lightgray', lw=1.5, alpha=0.6, label='Encoder Reference')
    
    if sim_ok:
        # Plot SINDy trajectory over it
        ax.plot(z_sin[:, 0], z_sin[:, 1], color='tomato', lw=0.5, alpha=0.9, label='SINDy ODE')
        ax.scatter(z_sin[0, 0], z_sin[0, 1], color='lime', s=50, zorder=5, label='Start')
        status = 'Converged'
    else:
        ax.scatter(z0_np[0], z0_np[1], color='red', s=100, marker='X', zorder=5, label='SINDy Diverged')
        status = 'Diverged'

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$\\xi_0$'); ax.set_ylabel('$\\xi_1$')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    ax.set_title(f'Long-term Latent Phase Portrait [{status}]\nDoes SINDy preserve the Rosette Topology?', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_long_term_phase.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 6: Observable Dimension x1(t) ────────────────────────────────────

def fig6_x_comparison(model, params, test_data, device, out_dir):
    """Compare true x1(t) vs decoder output from encoder and SINDy ODE."""
    t    = test_data['t']
    x_np = test_data['x']

    # ── SINDy Path ──
    x0 = torch.tensor(x_np[0:1], dtype=torch.float32, device=device)
    with torch.no_grad():
        z0 = model.encoder(x0).cpu().numpy()[0]

    Xi           = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order   = params['poly_order']
    include_sine = params.get('include_sine', False)

    rhs = _sindy_rhs(Xi, poly_order, include_sine)
    try:
        sol = solve_ivp(rhs, [t[0], t[-1]], z0, t_eval=t,
                        method='DOP853', rtol=1e-8, atol=1e-10)
        sim_ok = sol.success and not np.any(np.isnan(sol.y))
        z_sindy = sol.y.T if sim_ok else None
    except Exception:
        sim_ok = False

    if not sim_ok:
        print("  Skipping fig6: SINDy integration diverged.")
        return

    x_sindy = _decode(model, z_sindy, device)

    # Observable dimension (x_1 is the 0-th dimension of delay vector)
    x0_true  = x_np[:, 0]
    x0_sindy = x_sindy[:, 0]

    err_sindy = (x0_sindy - x0_true) ** 2

    fig, (ax_ts, ax_err) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                         gridspec_kw={'height_ratios': [1.4, 1], 'hspace': 0.35})

    # Row 1: Time Series
    ax_ts.plot(t, x0_true,  color="#4D94D1", lw=2.0, label='True $x_1(t)$')
    ax_ts.plot(t, x0_sindy, color='tomato',  lw=1.2, ls='--', label='SINDy → Decoder')
    ax_ts.set_ylabel('Position $x_1(t)$', fontsize=14)
    ax_ts.set_title('SINDy Prediction', fontsize=15)
    ax_ts.legend(fontsize=10); ax_ts.grid(True, alpha=0.3)

    # Row 2: Squared Error
    ax_err.semilogy(t, err_sindy, color='tomato', lw=1.4, label='Err²')
    ax_err.set_xlabel('Time (s)', fontsize=14)
    ax_err.set_ylabel('Squared Error', fontsize=14)
    ax_err.set_title('Pointwise Squared Error', fontsize=15)
    ax_err.legend(fontsize=10, loc='lower right'); ax_err.grid(True, which='both', alpha=0.3)

    rmse_sindy = float(np.sqrt(np.mean(err_sindy)))
    fig.suptitle(
        f'Planet 1 Position $x_1(t)$ Comparison',
        fontsize=13, fontweight='bold'
    )
    path = os.path.join(out_dir, 'fig6_x_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize SINDy-AE results for 3-Body (BHH) problem.")
    parser.add_argument('--mat', required=True, help='Path to checkpoint .mat (or .json) file')
    parser.add_argument('--data', required=True, help='Path to the delay embedding .npz data file')
    parser.add_argument('--out_dir', default=None, help='Output directory for figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    fmt = 'json' if args.mat.endswith('.json') else 'mat'
    mat_prefix = args.mat[:-5] if fmt == 'json' else (args.mat[:-4] if args.mat.endswith('.mat') else args.mat)

    print(f"Loading model from {mat_prefix} ({fmt}) on {device}...")
    if fmt == 'mat':
        model, params = load_model_mat(mat_prefix, device=device)
    else:
        model, params = load_model_json(mat_prefix, device=device)
    model.eval()
    print(f"  latent_dim={params['latent_dim']}, library_dim={params['library_dim']}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(mat_prefix))
    os.makedirs(out_dir, exist_ok=True)

    # Load Data
    print(f"Loading data from {args.data} ...")
    npz = np.load(args.data, allow_pickle=False)
    
    # We use the test split for visualization
    split = 'test' if 'test_x' in npz.files else 'val'
    print(f"  Using '{split}' split for visualization.")
    
    t_key = f'{split}_t' if f'{split}_t' in npz.files else ('t' if 't' in npz.files else 'train_t')
    test_data = {
        'x':  npz[f'{split}_x'],
        'dx': npz[f'{split}_dx'],
        't':  npz[t_key] if t_key in npz.files else np.arange(npz[f'{split}_x'].shape[0]) * npz['dt'].item(),
    }
    
    # Load ground truth if available
    if f'{split}_z' in npz.files:
        test_data['z'] = npz[f'{split}_z']
    
    print(f"  Total {test_data['x'].shape[0]} samples in continuous trajectory.")
    print(f"\nGenerating figures...")

    # Generate all plots
    # fig1_phase_portrait(model, test_data, device, out_dir)
    # fig2_reconstruction(model, test_data, device, out_dir)
    # fig3_time_series(model, test_data, device, out_dir)
    # fig4_sindy_simulation(model, params, test_data, device, out_dir)
    # fig5_long_term_phase(model, params, test_data, device, out_dir)
    fig6_x_comparison(model, params, test_data, device, out_dir)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == '__main__':
    main()