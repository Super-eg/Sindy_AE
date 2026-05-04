"""Visualize trained Van der Pol model: learned latent dynamics vs true state.

Generates 4-5 figures saved next to the .mat checkpoint:
    fig1_phase_portrait.png   - 2D limit cycle: true z vs learned latent ξ (all test ICs)
    fig2_time_series.png      - Time series: true z₀z₁ vs latent ξ₀ξ₁ (1 IC)
    fig3_reconstruction.png   - Input x and dx reconstruction quality (1 IC)
    fig4_sindy_simulation.png - SINDy ODE forward simulation vs encoder trajectory
    fig4b_sindy_2d.png        - 2D: encoder trajectory vs SINDy propagated (if converged)

Usage (run from examples/vander/):
    python3 visualize_vander.py --mat checkpoints/model_YYYYMMDD_HHMMSS.mat --data vander_legendre.npz
    python3 visualize_vander.py --mat checkpoints/model_YYYYMMDD_HHMMSS.mat --data vander_legendre.npz --ic 3
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


# ─── figure 1: 2D phase portrait ─────────────────────────────────────────────

def fig1_phase_portrait(model, test_data, device, out_dir):
    """True Van der Pol limit cycle vs learned latent phase portrait (all ICs)."""
    has_z = 'z' in test_data
    if has_z:
        z_true = test_data['z'].reshape(-1, 2)
    else:
        z_true = test_data['x'][:, :2]
    z_lat = _encode(model, test_data['x'], device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(z_true[:, 0], z_true[:, 1], color='steelblue', lw=0.25, alpha=0.5)
    ax1.set_title('True Van der Pol limit cycle\n(normalized z₀, z₁)', fontsize=12)
    ax1.set_xlabel('z₀'); ax1.set_ylabel('z₁')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_lat[:, 0], z_lat[:, 1], color='tomato', lw=0.25, alpha=0.5)
    ax2.set_title('Learned latent phase portrait\n(encoder output ξ₀, ξ₁)', fontsize=12)
    ax2.set_xlabel('ξ₀'); ax2.set_ylabel('ξ₁')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Phase Portrait Comparison — All Test ICs', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_phase_portrait.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 2: per-dimension time series ─────────────────────────────────────

def fig2_time_series(model, test_data, device, out_dir, ic_idx):
    """True z₀z₁ vs latent ξ₀ξ₁ over time (single IC)."""
    t       = test_data['t']
    n_steps = len(t)
    sl      = _ic_slice(ic_idx, n_steps)
    has_z   = 'z' in test_data
    z_true  = test_data['z'][sl] if has_z else test_data['x'][sl, :2]
    z_lat   = _encode(model, test_data['x'][sl], device)

    colors_true = ['steelblue', 'darkorange']
    colors_lat  = ['tomato',    'orchid']

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)

    for i in range(2):
        axes[i, 0].plot(t, z_true[:, i], color=colors_true[i], lw=1.2)
        axes[i, 0].set_ylabel(f'z{i} (true)' if has_z else f'x_delay[{i}]', fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(t, z_lat[:, i], color=colors_lat[i], lw=1.2)
        axes[i, 1].set_ylabel(f'ξ{i} (learned)', fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title('True Van der Pol state' if has_z else 'Observed coordinates', fontsize=11)
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

    rel_x     = np.mean((x_dec     - x_np    ) ** 2) / np.mean(x_np     ** 2)
    rel_deriv = np.mean((deriv_dec - deriv_np) ** 2) / np.mean(deriv_np ** 2)
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

    x0 = torch.tensor(x_np[0:1], dtype=torch.float32, device=device)
    with torch.no_grad():
        z0 = model.encoder(x0).cpu().numpy()[0]

    Xi           = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order   = params['poly_order']
    include_sine = params.get('include_sine', False)

    rhs = _sindy_rhs(Xi, poly_order, include_sine)
    try:
        sol = solve_ivp(rhs, [t[0], t[-1]], z0, t_eval=t,
                        method='RK45', rtol=1e-6, atol=1e-9, max_step=0.02)
        sim_ok  = sol.success and not np.any(np.isnan(sol.y))
        z_sindy = sol.y[:2, :].T if sim_ok else None
    except Exception as e:
        print(f"  Warning: SINDy integration failed ({e})")
        sim_ok, z_sindy = False, None

    z_enc = _encode(model, x_np, device)   # [T, 2]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for i in range(2):
        axes[i].plot(t, z_enc[:, i], 'steelblue', lw=1.4, label='Encoder ξ%d' % i)
        if sim_ok:
            axes[i].plot(t, z_sindy[:, i], 'tomato', lw=1.2, ls='--',
                         label='SINDy ODE ξ%d' % i)
        axes[i].set_ylabel(f'ξ{i}')
        axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3)

    x_from_enc = _decode(model, z_enc, device)
    err_enc = np.mean((x_from_enc - x_np) ** 2, axis=1)
    axes[2].semilogy(t, err_enc, 'steelblue', lw=1.4, label='Encoder → Decoder')
    if sim_ok:
        x_from_sindy = _decode(model, z_sindy, device)
        err_sindy = np.mean((x_from_sindy - x_np) ** 2, axis=1)
        axes[2].semilogy(t, err_sindy, 'tomato', lw=1.2, ls='--',
                         label='SINDy ODE → Decoder')
    axes[2].set_ylabel('MSE vs true x'); axes[2].set_xlabel('t')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Pointwise reconstruction error over time')

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

    if sim_ok:
        fig2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(z_enc[:, 0], z_enc[:, 1], color='steelblue', lw=1.0, alpha=0.85)
        ax1.scatter(*z0, color='lime', s=60, zorder=5, label='IC')
        ax1.set_title('Encoder trajectory\n(true data → encoder)', fontsize=11)
        ax1.set_xlabel('ξ₀'); ax1.set_ylabel('ξ₁')
        ax1.set_aspect('equal', 'box')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2.plot(z_sindy[:, 0], z_sindy[:, 1], color='tomato', lw=1.0, alpha=0.85)
        ax2.scatter(*z0, color='lime', s=60, zorder=5, label='IC')
        ax2.set_title('SINDy propagated trajectory\n(learned ODE)', fontsize=11)
        ax2.set_xlabel('ξ₀'); ax2.set_ylabel('ξ₁')
        ax2.set_aspect('equal', 'box')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        fig2d.suptitle('2D Latent Space: Encoder vs SINDy ODE', fontsize=13, fontweight='bold')
        plt.tight_layout()
        path2d = os.path.join(out_dir, 'fig4b_sindy_2d.png')
        plt.savefig(path2d, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path2d}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', required=True,
                        help='Path to checkpoint .mat (or .json) file')
    parser.add_argument('--data', required=True,
                        help='Path to .npz data file; uses test split (falls back to val)')
    parser.add_argument('--ic', type=int, default=0,
                        help='Which test IC to use for time-series / reconstruction plots (default: 0)')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory for figures (default: same directory as .mat file)')
    args = parser.parse_args()

    device = torch.device('cpu')

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
    if f'{split}_ddx' in npz.files:
        test_data['ddx'] = npz[f'{split}_ddx']
    if f'{split}_z' in npz.files:
        test_data['z'] = npz[f'{split}_z']   # [N*T, 2] normalized true state
    n_steps = len(test_data['t'])
    n_ics   = test_data['x'].shape[0] // n_steps
    print(f"  {n_ics} ICs × {n_steps} steps = {test_data['x'].shape[0]} samples")

    ic = args.ic
    print(f"\nGenerating figures (IC #{ic} for time-series plots)...")
    print(f"Figures will be saved to: {out_dir}/")

    fig1_phase_portrait(model, test_data, device, out_dir)
    fig2_time_series(model, test_data, device, out_dir, ic)
    fig3_reconstruction(model, test_data, device, out_dir, ic)
    fig4_sindy_simulation(model, params, test_data, device, out_dir, ic)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
