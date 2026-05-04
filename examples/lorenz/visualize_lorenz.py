"""Visualize trained Lorenz model: learned latent dynamics vs true Lorenz.

Generates 6 figures saved next to the .mat checkpoint:
    fig1_3d_attractor.png     - 3D attractor: true z vs learned latent ξ (all test ICs)
    fig2_time_series.png      - Time series: true z₀z₁z₂ vs latent ξ₀ξ₁ξ₂ (1 IC)
    fig3_reconstruction.png   - Input x and dx reconstruction quality (1 IC)
    fig4_sindy_simulation.png - SINDy ODE forward simulation vs encoder trajectory
    fig4b_sindy_3d.png        - 3D: encoder trajectory vs SINDy propagated trajectory
    fig5_multi_ic_attractor.png - Multi-IC 3D: encoder vs SINDy attractor coverage

Usage (run from examples/lorenz/):
    python3 visualize_lorenz.py --mat model_YYYYMMDD_HHMMSS.mat --data legendre_full.npz
    python3 visualize_lorenz.py --mat model_YYYYMMDD_HHMMSS.mat --data legendre_full.npz --ic 3
    python3 visualize_lorenz.py --mat model_YYYYMMDD_HHMMSS.mat --data legendre_full.npz --n_ics_plot 20
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
matplotlib.use('Agg')  # non-interactive; works on server without display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import solve_ivp

from sindyae import load_model_mat, load_model_json
from sindyae.sindy_library import sindy_library_torch, sindy_library_torch_order2
from sindyae.autoencoder import _seq_jvp


# ─── helpers ────────────────────────────────────────────────────────────────

def _encode(model, x_np, device, bs=4096):
    """numpy [N,128] → numpy [N,3]."""
    out = []
    for i in range(0, len(x_np), bs):
        xb = torch.tensor(x_np[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            out.append(model.encoder(xb).cpu().numpy())
    return np.concatenate(out)


def _decode(model, z_np, device, bs=4096):
    """numpy [N,3] → numpy [N,128]."""
    out = []
    for i in range(0, len(z_np), bs):
        zb = torch.tensor(z_np[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            out.append(model.decoder(zb).cpu().numpy())
    return np.concatenate(out)


def _sindy_rhs(Xi, poly_order, include_sine):
    """Return a scipy-compatible RHS function for the learned SINDy ODE (order 1)."""
    def rhs(t, z):
        zt = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            Theta = sindy_library_torch(zt, poly_order, include_sine)
        return (Theta.numpy() @ Xi)[0]
    return rhs


def _sindy_rhs_order2(Xi, latent_dim, poly_order, include_sine):
    """Return a scipy-compatible RHS for a 2nd-order SINDy ODE.

    State vector: [z (latent_dim), dz (latent_dim)].
    Returns [dz, ddz] where ddz = Theta(z, dz) @ Xi.
    """
    def rhs(t, state):
        z  = state[:latent_dim]
        dz = state[latent_dim:]
        zt  = torch.tensor(z,  dtype=torch.float32).unsqueeze(0)
        dzt = torch.tensor(dz, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            Theta = sindy_library_torch_order2(zt, dzt, poly_order, include_sine)
        ddz = (Theta.numpy() @ Xi)[0]
        return np.concatenate([dz, ddz])
    return rhs


def _ic_slice(ic_idx, n_steps):
    return slice(ic_idx * n_steps, (ic_idx + 1) * n_steps)


# ─── figure 1: 3D attractor comparison ──────────────────────────────────────

def fig1_3d_attractor(model, test_data, device, out_dir):
    """True Lorenz z-space vs learned latent ξ-space (all ICs)."""
    has_z = 'z' in test_data
    if has_z:
        z_true = test_data['z'].reshape(-1, 3)       # [N*T, 3] normalized
    else:
        z_true = test_data['x'][:, :3]               # first 3 delay dims as proxy
    z_lat  = _encode(model, test_data['x'], device)  # [N*T, 3]

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2],
             color='steelblue', lw=0.25, alpha=0.5)
    if has_z:
        ax1.set_title('True Lorenz attractor\n(normalized z₀, z₁, z₂)', fontsize=12)
        ax1.set_xlabel('z₀'); ax1.set_ylabel('z₁'); ax1.set_zlabel('z₂')
    else:
        ax1.set_title('Observed delay coordinates\n(x[t], x[t-τ], x[t-2τ])', fontsize=12)
        ax1.set_xlabel('x[t]'); ax1.set_ylabel('x[t-τ]'); ax1.set_zlabel('x[t-2τ]')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(z_lat[:, 0], z_lat[:, 1], z_lat[:, 2],
             color='tomato', lw=0.25, alpha=0.5)
    ax2.set_title('Learned latent attractor\n(encoder output ξ₀, ξ₁, ξ₂)', fontsize=12)
    ax2.set_xlabel('ξ₀'); ax2.set_ylabel('ξ₁'); ax2.set_zlabel('ξ₂')

    fig.suptitle('3D Attractor Comparison — All Test ICs', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_3d_attractor.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 2: per-dimension time series ─────────────────────────────────────

def fig2_time_series(model, test_data, device, out_dir, ic_idx):
    """True z₀z₁z₂ vs latent ξ₀ξ₁ξ₂ over time (single IC)."""
    t       = test_data['t']                          # [T]
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)
    has_z   = 'z' in test_data
    if has_z:
        z_true = test_data['z'][sl]                   # [T, 3]
    else:
        z_true = test_data['x'][sl, :3]               # first 3 delay dims as proxy [T, 3]
    z_lat   = _encode(model, test_data['x'][sl], device)  # [T, 3]

    colors_true = ['steelblue', 'darkorange', 'seagreen']
    colors_lat  = ['tomato',    'orchid',     'goldenrod']

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    for i in range(3):
        axes[i, 0].plot(t, z_true[:, i], color=colors_true[i], lw=1.2)
        if has_z:
            axes[i, 0].set_ylabel(f'z{i} (true)', fontsize=10)
        else:
            axes[i, 0].set_ylabel(f'x_delay[{i}] (observed)', fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(t, z_lat[:, i], color=colors_lat[i], lw=1.2)
        axes[i, 1].set_ylabel(f'ξ{i} (learned)', fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title('True Lorenz state' if has_z else 'Observed delay coordinates', fontsize=11)
    axes[0, 1].set_title('Learned latent variables', fontsize=11)
    axes[-1, 0].set_xlabel('t'); axes[-1, 1].set_xlabel('t')

    fig.suptitle(f'Per-dimension Time Series — IC #{ic_idx}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_time_series.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 3: reconstruction quality ────────────────────────────────────────

def fig3_reconstruction(model, test_data, device, out_dir, ic_idx):
    """x and dx reconstruction quality for a single IC."""
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
        deriv_dec = outputs['ddx_decode'].cpu().numpy()
        deriv_np  = test_data['ddx'][sl]
        deriv_label = 'ddx'
    else:
        deriv_dec = outputs['dx_decode'].cpu().numpy()
        deriv_np  = dx_np
        deriv_label = 'dx'

    n_dims = x_np.shape[1]
    dims = [0, n_dims // 2, n_dims - 1]  # 3 representative input dimensions

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    for row, d in enumerate(dims):
        axes[row, 0].plot(t, x_np[:, d],    'steelblue', lw=1.5, label='True')
        axes[row, 0].plot(t, x_dec[:, d],   'tomato',    lw=1.2, ls='--', label='Decoded')
        axes[row, 0].set_ylabel(f'x[{d}]')
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, deriv_np[:, d],  'steelblue', lw=1.5, label='True')
        axes[row, 1].plot(t, deriv_dec[:, d], 'tomato',    lw=1.2, ls='--', label='Decoded')
        axes[row, 1].set_ylabel(f'{deriv_label}[{d}]')
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title('Input reconstruction  x', fontsize=11)
    axes[0, 1].set_title(f'Derivative reconstruction  {deriv_label}', fontsize=11)
    axes[-1, 0].set_xlabel('t'); axes[-1, 1].set_xlabel('t')

    # Relative MSE annotations
    rel_x     = np.mean((x_dec     - x_np    )**2) / np.mean(x_np    **2)
    rel_deriv = np.mean((deriv_dec - deriv_np)**2) / np.mean(deriv_np**2)
    fig.suptitle(
        f'Reconstruction Quality — IC #{ic_idx}\n'
        f'relative err x: {rel_x:.2e}   relative err {deriv_label}: {rel_deriv:.2e}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_reconstruction.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 4: SINDy forward simulation ──────────────────────────────────────

def fig4_sindy_simulation(model, params, test_data, device, out_dir, ic_idx):
    """Integrate the learned SINDy ODE; compare with encoder trajectory."""
    t       = test_data['t']
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)
    x_np    = test_data['x'][sl]

    # Initial latent state from encoder
    x0  = torch.tensor(x_np[0:1],                      dtype=torch.float32, device=device)
    dx0 = torch.tensor(test_data['dx'][sl][0:1],        dtype=torch.float32, device=device)
    with torch.no_grad():
        z0 = model.encoder(x0).cpu().numpy()[0]

    Xi           = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order   = params['poly_order']
    include_sine = params.get('include_sine', False)
    order2       = model.model_order == 2

    # Integrate SINDy ODE
    if order2:
        _, dz0 = _seq_jvp(model.encoder, x0, dx0)
        dz0 = dz0.detach().cpu().numpy()[0]
        state0 = np.concatenate([z0, dz0])
        rhs = _sindy_rhs_order2(Xi, model.latent_dim, poly_order, include_sine)
    else:
        state0 = z0
        rhs = _sindy_rhs(Xi, poly_order, include_sine)

    try:
        sol = solve_ivp(rhs, [t[0], t[-1]], state0, t_eval=t,
                        method='RK45', rtol=1e-6, atol=1e-9, max_step=0.02)
        sim_ok  = sol.success and not np.any(np.isnan(sol.y))
        z_sindy = sol.y[:model.latent_dim, :].T if sim_ok else None
    except Exception as e:
        print(f"  Warning: SINDy integration failed ({e})")
        sim_ok, z_sindy = False, None

    # True latent trajectory (encoder)
    z_enc = _encode(model, x_np, device)  # [T, 3]

    # ── time-series panel ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 13))

    for i in range(3):
        axes[i].plot(t, z_enc[:, i], 'steelblue', lw=1.4, label='Encoder ξ%d' % i)
        if sim_ok:
            axes[i].plot(t, z_sindy[:, i], 'tomato', lw=1.2, ls='--',
                         label='SINDy ODE ξ%d' % i)
        axes[i].set_ylabel(f'ξ{i}')
        axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3)

    # Reconstruction MSE over time
    x_from_enc = _decode(model, z_enc, device)
    err_enc = np.mean((x_from_enc - x_np)**2, axis=1)
    axes[3].semilogy(t, err_enc, 'steelblue', lw=1.4, label='Encoder → Decoder')
    if sim_ok:
        x_from_sindy = _decode(model, z_sindy, device)
        err_sindy = np.mean((x_from_sindy - x_np)**2, axis=1)
        axes[3].semilogy(t, err_sindy, 'tomato', lw=1.2, ls='--',
                         label='SINDy ODE → Decoder')
    axes[3].set_ylabel('MSE vs true x'); axes[3].set_xlabel('t')
    axes[3].legend(fontsize=9); axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Pointwise reconstruction error over time')

    status = 'converged' if sim_ok else '⚠ DIVERGED'
    axes[0].set_title(
        f'SINDy Forward Simulation vs Encoder Trajectory — IC #{ic_idx}  [{status}]',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_sindy_simulation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── 3D comparison panel ───────────────────────────────────────────────────
    if sim_ok:
        fig3d = plt.figure(figsize=(14, 6))

        ax1 = fig3d.add_subplot(121, projection='3d')
        ax1.plot(z_enc[:, 0], z_enc[:, 1], z_enc[:, 2],
                 color='steelblue', lw=1.0, alpha=0.85)
        ax1.scatter(*z0, color='lime', s=60, zorder=5, label='IC')
        ax1.set_title('Encoder trajectory\n(true data → encoder)', fontsize=11)
        ax1.set_xlabel('ξ₀'); ax1.set_ylabel('ξ₁'); ax1.set_zlabel('ξ₂')
        ax1.legend(fontsize=8)

        ax2 = fig3d.add_subplot(122, projection='3d')
        ax2.plot(z_sindy[:, 0], z_sindy[:, 1], z_sindy[:, 2],
                 color='tomato', lw=1.0, alpha=0.85)
        ax2.scatter(*z0, color='lime', s=60, zorder=5, label='IC')
        ax2.set_title('SINDy propagated trajectory\n(learned ODE)', fontsize=11)
        ax2.set_xlabel('ξ₀'); ax2.set_ylabel('ξ₁'); ax2.set_zlabel('ξ₂')
        ax2.legend(fontsize=8)

        fig3d.suptitle('3D Latent Space: Encoder vs SINDy ODE', fontsize=13, fontweight='bold')
        plt.tight_layout()
        path3d = os.path.join(out_dir, 'fig4b_sindy_3d.png')
        plt.savefig(path3d, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path3d}")


# ─── figure 5: multi-IC attractor coverage ────────────────────────────────────

def fig5_multi_ic_attractor(model, params, test_data, device, out_dir, n_ics_plot=10):
    """Multi-IC 3D attractor: encoder trajectories vs SINDy ODE (side-by-side).

    Left panel : trajectories produced by the encoder on real data.
    Right panel: trajectories produced by integrating the learned SINDy ODE
                 from the same initial latent states.
    Each IC gets a distinct colour; diverged SINDy runs are marked with ×.
    """
    t = test_data['t']
    n_steps = len(t)
    n_ics_total = test_data['x'].shape[0] // n_steps
    n_show = min(n_ics_plot, n_ics_total)

    # Evenly spaced IC indices across the test set
    ic_indices = np.linspace(0, n_ics_total - 1, n_show, dtype=int)

    Xi          = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order  = params['poly_order']
    include_sine = params.get('include_sine', False)
    order2      = model.model_order == 2

    # Cycle through tab10; works for ≤10 ICs and wraps gracefully for more
    colors = plt.cm.tab10(np.arange(n_show) % 10)

    fig = plt.figure(figsize=(14, 6))
    ax_enc = fig.add_subplot(121, projection='3d')
    ax_sin = fig.add_subplot(122, projection='3d')

    n_converged = 0
    for ci, ic_idx in enumerate(ic_indices):
        sl   = _ic_slice(ic_idx, n_steps)
        x_np = test_data['x'][sl]
        c    = colors[ci]

        # ── encoder trajectory ──
        z_enc = _encode(model, x_np, device)
        ax_enc.plot(z_enc[:, 0], z_enc[:, 1], z_enc[:, 2],
                    color=c, lw=0.7, alpha=0.75)
        ax_enc.scatter(*z_enc[0], color=c, s=25, zorder=5)

        # ── SINDy ODE trajectory ──
        x0_t  = torch.tensor(x_np[0:1], dtype=torch.float32, device=device)
        dx0_t = torch.tensor(test_data['dx'][sl][0:1], dtype=torch.float32, device=device)
        with torch.no_grad():
            z0_np = model.encoder(x0_t).cpu().numpy()[0]

        if order2:
            _, dz0_t = _seq_jvp(model.encoder, x0_t, dx0_t)
            state0 = np.concatenate([z0_np, dz0_t.detach().cpu().numpy()[0]])
            rhs = _sindy_rhs_order2(Xi, model.latent_dim, poly_order, include_sine)
        else:
            state0 = z0_np
            rhs    = _sindy_rhs(Xi, poly_order, include_sine)

        try:
            sol = solve_ivp(rhs, [t[0], t[-1]], state0, t_eval=t,
                            method='RK45', rtol=1e-6, atol=1e-9, max_step=0.02)
            if sol.success and not np.any(np.isnan(sol.y)):
                z_sin = sol.y[:model.latent_dim, :].T
                ax_sin.plot(z_sin[:, 0], z_sin[:, 1], z_sin[:, 2],
                            color=c, lw=0.7, alpha=0.75)
                ax_sin.scatter(*z0_np, color=c, s=25, zorder=5)
                n_converged += 1
            else:
                ax_sin.scatter(*z0_np, color=c, s=60, marker='x', zorder=5)
        except Exception:
            ax_sin.scatter(*z0_np, color=c, s=60, marker='x', zorder=5)

    ax_enc.set_title(f'Encoder trajectories ({n_show} ICs)', fontsize=11)
    ax_enc.set_xlabel('ξ₀'); ax_enc.set_ylabel('ξ₁'); ax_enc.set_zlabel('ξ₂')

    ax_sin.set_title(f'SINDy ODE trajectories\n({n_converged}/{n_show} converged)',
                     fontsize=11)
    ax_sin.set_xlabel('ξ₀'); ax_sin.set_ylabel('ξ₁'); ax_sin.set_zlabel('ξ₂')

    fig.suptitle('Multi-IC Attractor — Encoder vs Learned SINDy Dynamics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_multi_ic_attractor.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}  ({n_converged}/{n_show} SINDy ICs converged)")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', required=True,
                        help='Path to checkpoint .mat (or .json) file, '
                             'e.g. model_20260501_150000.mat')
    parser.add_argument('--data', required=True,
                        help='Path to .npz data file; uses test_x/dx/t (falls back to val split)')
    parser.add_argument('--ic', type=int, default=0,
                        help='Which test IC to use for time-series / reconstruction plots (default: 0)')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory for figures (default: same directory as .mat file)')
    parser.add_argument('--n_ics_plot', type=int, default=10,
                        help='Number of ICs to show in fig5 multi-IC attractor (default: 10)')
    args = parser.parse_args()

    device = torch.device('cpu')  # visualization is CPU; keeps output deterministic

    # Strip extension to get prefix; infer format from extension
    if args.mat.endswith('.json'):
        mat_prefix = args.mat[:-5]
        fmt = 'json'
    else:
        mat_prefix = args.mat[:-4] if args.mat.endswith('.mat') else args.mat
        fmt = 'mat'

    print(f"Loading model from {mat_prefix} ({fmt})...")
    if fmt == 'mat':
        model, params = load_model_mat(mat_prefix, device=device)
    else:
        model, params = load_model_json(mat_prefix, device=device)
    model.eval()
    print(f"  latent_dim={params['latent_dim']}, library_dim={params['library_dim']}")

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
    ddx_key = f'{split}_ddx'
    if ddx_key in npz.files:
        test_data['ddx'] = npz[ddx_key]
    z_key = f'{split}_z'
    if z_key in npz.files:
        test_data['z'] = npz[z_key]  # [N*T, 3] normalized true Lorenz state
    n_steps = len(test_data['t'])
    n_ics   = test_data['x'].shape[0] // n_steps
    print(f"  {n_ics} ICs × {n_steps} steps = {test_data['x'].shape[0]} samples")

    ic = args.ic
    print(f"\nGenerating figures (IC #{ic} for time-series plots)...")
    print(f"Figures will be saved to: {out_dir}/")

    fig1_3d_attractor(model, test_data, device, out_dir)
    fig2_time_series(model, test_data, device, out_dir, ic)
    fig3_reconstruction(model, test_data, device, out_dir, ic)
    fig4_sindy_simulation(model, params, test_data, device, out_dir, ic)
    fig5_multi_ic_attractor(model, params, test_data, device, out_dir, n_ics_plot=args.n_ics_plot)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
