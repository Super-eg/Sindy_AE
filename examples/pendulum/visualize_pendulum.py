"""Visualize trained Pendulum model: learned latent dynamics vs true pendulum.

Generates 5 figures saved next to the .mat checkpoint:
    fig1_phase_portrait.png    - True (θ, θ̇) vs learned (ξ, dξ/dt) phase portrait
    fig2_time_series.png       - True θ(t), θ̇(t) vs learned ξ(t), dξ/dt(t)  (1 IC)
    fig3_reconstruction.png    - Image x and d²x reconstruction quality          (1 IC)
    fig4_sindy_simulation.png  - SINDy 2nd-order ODE forward simulation vs encoder
    fig5_images.png            - Visual: true vs decoded 51×51 images at snapshots (1 IC)

Usage (run from examples/pendulum/):
    python3 visualize_pendulum.py --mat model_YYYYMMDD_HHMMSS.mat --data pendulum_data.npz
    python3 visualize_pendulum.py --mat model_YYYYMMDD_HHMMSS.mat --data pendulum_data.npz --ic 3
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
matplotlib.use("Agg")   # non-interactive; works on server without display
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sindyae import load_model_mat, load_model_json
from sindyae.autoencoder import _seq_jvp
from sindyae.sindy_library import sindy_library_torch_order2


# ─── helpers ────────────────────────────────────────────────────────────────

def _encode_with_dz(model, x_np, dx_np, device, bs=4096):
    """Return (z, dz) from encoder JVP.  z, dz: numpy [N, latent_dim]."""
    z_out, dz_out = [], []
    for i in range(0, len(x_np), bs):
        xb  = torch.tensor(x_np[i:i+bs],  dtype=torch.float32, device=device)
        dxb = torch.tensor(dx_np[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            z_b, dz_b = _seq_jvp(model.encoder, xb, dxb)
        z_out.append(z_b.cpu().numpy())
        dz_out.append(dz_b.cpu().numpy())
    return np.concatenate(z_out), np.concatenate(dz_out)


def _decode(model, z_np, device, bs=4096):
    """numpy [N, latent_dim] → numpy [N, input_dim]."""
    out = []
    for i in range(0, len(z_np), bs):
        zb = torch.tensor(z_np[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            out.append(model.decoder(zb).cpu().numpy())
    return np.concatenate(out)


def _sindy_rhs_order2(Xi, poly_order, include_sine, latent_dim):
    """Return a scipy-compatible RHS for the 2nd-order learned SINDy ODE.

    State = [z₀, ..., z_{n-1}, dz₀, ..., dz_{n-1}]  (2n-dimensional)
    d/dt state = [dz; Θ([z,dz]) @ Xi]
    """
    n = latent_dim

    def rhs(t, state):
        z_arr  = np.array(state[:n],  dtype=np.float32)
        dz_arr = np.array(state[n:],  dtype=np.float32)
        zt  = torch.tensor(z_arr[None],  dtype=torch.float32)
        dzt = torch.tensor(dz_arr[None], dtype=torch.float32)
        with torch.no_grad():
            Theta = sindy_library_torch_order2(zt, dzt, poly_order, include_sine)
        ddz = (Theta.numpy() @ Xi)[0]    # [latent_dim]
        return list(dz_arr) + list(ddz)

    return rhs


def _ic_slice(ic_idx, n_steps):
    return slice(ic_idx * n_steps, (ic_idx + 1) * n_steps)


# ─── figure 1: phase portrait ─────────────────────────────────────────────────

def fig1_phase_portrait(model, test_data, device, out_dir):
    """True (θ, θ̇) vs learned (ξ, dξ/dt) phase portrait — all test ICs."""
    z_true = test_data["z"]          # [N, 2]:  col0=θ, col1=θ̇
    z_enc, dz_enc = _encode_with_dz(model, test_data["x"], test_data["dx"], device)
    # z_enc: [N, 1]  dz_enc: [N, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(z_true[:, 0], z_true[:, 1], "steelblue", lw=0.3, alpha=0.5)
    axes[0].set_xlabel("θ (rad)", fontsize=11)
    axes[0].set_ylabel("θ̇ (rad/s)", fontsize=11)
    axes[0].set_title("True phase portrait  (θ, θ̇)", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z_enc[:, 0], dz_enc[:, 0], "tomato", lw=0.3, alpha=0.5)
    axes[1].set_xlabel("ξ  (latent)", fontsize=11)
    axes[1].set_ylabel("dξ/dt  (latent velocity)", fontsize=11)
    axes[1].set_title("Learned phase portrait  (ξ, dξ/dt)", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Phase Portrait Comparison — All Test ICs", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_phase_portrait.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 2: time series ────────────────────────────────────────────────────

def fig2_time_series(model, test_data, device, out_dir, ic_idx):
    """True θ(t), θ̇(t) vs learned ξ(t), dξ/dt(t) — single IC."""
    t      = test_data["t"]
    n_steps = len(t)
    sl     = _ic_slice(ic_idx, n_steps)

    z_true = test_data["z"][sl]   # [T, 2]
    z_enc, dz_enc = _encode_with_dz(
        model, test_data["x"][sl], test_data["dx"][sl], device
    )   # [T, 1], [T, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)

    axes[0, 0].plot(t, z_true[:, 0], "steelblue", lw=1.3)
    axes[0, 0].set_ylabel("θ (rad)", fontsize=10)
    axes[0, 0].set_title("True state", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(t, z_true[:, 1], "darkorange", lw=1.3)
    axes[1, 0].set_ylabel("θ̇ (rad/s)", fontsize=10)
    axes[1, 0].set_xlabel("t (s)", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, z_enc[:, 0], "tomato", lw=1.3)
    axes[0, 1].set_ylabel("ξ  (latent)", fontsize=10)
    axes[0, 1].set_title("Learned latent", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(t, dz_enc[:, 0], "orchid", lw=1.3)
    axes[1, 1].set_ylabel("dξ/dt  (latent vel.)", fontsize=10)
    axes[1, 1].set_xlabel("t (s)", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Time Series — IC #{ic_idx}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_time_series.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 3: reconstruction quality ────────────────────────────────────────

def fig3_reconstruction(model, test_data, device, out_dir, ic_idx):
    """Pixel-level x and ddx reconstruction for a single IC."""
    t      = test_data["t"]
    n_steps = len(t)
    sl     = _ic_slice(ic_idx, n_steps)

    x_np   = test_data["x"][sl]
    dx_np  = test_data["dx"][sl]
    ddx_np = test_data["ddx"][sl]

    x_t   = torch.tensor(x_np,   dtype=torch.float32, device=device)
    dx_t  = torch.tensor(dx_np,  dtype=torch.float32, device=device)
    ddx_t = torch.tensor(ddx_np, dtype=torch.float32, device=device)

    outputs = model(x_t, dx_t, ddx_t)
    x_dec   = outputs["x_decode"].detach().cpu().numpy()
    ddx_dec = outputs["ddx_decode"].detach().cpu().numpy()

    # Show 3 representative pixel indices
    total_pixels = x_np.shape[1]   # 2601
    dims = [0, total_pixels // 2, total_pixels - 1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    for row, d in enumerate(dims):
        axes[row, 0].plot(t, x_np[:, d],   "steelblue", lw=1.5, label="True")
        axes[row, 0].plot(t, x_dec[:, d],  "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 0].set_ylabel(f"x[{d}]", fontsize=9)
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, ddx_np[:, d],  "steelblue", lw=1.5, label="True")
        axes[row, 1].plot(t, ddx_dec[:, d], "tomato",    lw=1.2, ls="--", label="Decoded")
        axes[row, 1].set_ylabel(f"ddx[{d}]", fontsize=9)
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title("Image reconstruction  x", fontsize=11)
    axes[0, 1].set_title("2nd-derivative reconstruction  d²x/dt²", fontsize=11)
    axes[-1, 0].set_xlabel("t (s)"); axes[-1, 1].set_xlabel("t (s)")

    rel_x   = np.mean((x_dec   - x_np  )**2) / np.mean(x_np  **2)
    rel_ddx = np.mean((ddx_dec - ddx_np)**2) / np.mean(ddx_np**2)
    fig.suptitle(
        f"Reconstruction Quality — IC #{ic_idx}\n"
        f"relative err x: {rel_x:.2e}   relative err d²x: {rel_ddx:.2e}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 4: SINDy 2nd-order ODE simulation ────────────────────────────────

def fig4_sindy_simulation(model, params, test_data, device, out_dir, ic_idx):
    """Integrate the learned 2nd-order SINDy ODE; compare with encoder trajectory."""
    t      = test_data["t"]
    n_steps = len(t)
    sl     = _ic_slice(ic_idx, n_steps)

    x_np  = test_data["x"][sl]
    dx_np = test_data["dx"][sl]

    # Encode initial condition
    z_enc, dz_enc = _encode_with_dz(model, x_np, dx_np, device)  # [T, 1]

    Xi          = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order  = params["poly_order"]
    include_sine = params.get("include_sine", False)
    latent_dim  = params["latent_dim"]

    # Initial state for 2nd-order ODE: [ξ₀, dξ₀/dt]
    ic_state = list(z_enc[0]) + list(dz_enc[0])    # length 2*latent_dim

    rhs = _sindy_rhs_order2(Xi, poly_order, include_sine, latent_dim)
    try:
        sol = solve_ivp(
            rhs, [t[0], t[-1]], ic_state, t_eval=t,
            method="RK45", rtol=1e-6, atol=1e-9, max_step=0.02
        )
        sim_ok  = sol.success and not np.any(np.isnan(sol.y))
        if sim_ok:
            z_sindy  = sol.y[:latent_dim].T   # [T, latent_dim]
            dz_sindy = sol.y[latent_dim:].T   # [T, latent_dim]
        else:
            z_sindy = dz_sindy = None
    except Exception as e:
        print(f"  Warning: SINDy integration failed ({e})")
        sim_ok, z_sindy, dz_sindy = False, None, None

    # ── time-series panel ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # ξ(t)
    axes[0].plot(t, z_enc[:, 0],  "steelblue", lw=1.4, label="Encoder ξ")
    if sim_ok:
        axes[0].plot(t, z_sindy[:, 0], "tomato", lw=1.2, ls="--", label="SINDy ODE ξ")
    axes[0].set_ylabel("ξ  (latent)", fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # dξ/dt(t)
    axes[1].plot(t, dz_enc[:, 0],  "steelblue", lw=1.4, label="Encoder dξ/dt")
    if sim_ok:
        axes[1].plot(t, dz_sindy[:, 0], "tomato", lw=1.2, ls="--", label="SINDy ODE dξ/dt")
    axes[1].set_ylabel("dξ/dt", fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # Reconstruction MSE over time
    x_from_enc = _decode(model, z_enc, device)
    err_enc = np.mean((x_from_enc - x_np)**2, axis=1)
    axes[2].semilogy(t, err_enc, "steelblue", lw=1.4, label="Encoder → Decoder")
    if sim_ok:
        x_from_sindy = _decode(model, z_sindy, device)
        err_sindy = np.mean((x_from_sindy - x_np)**2, axis=1)
        axes[2].semilogy(t, err_sindy, "tomato", lw=1.2, ls="--",
                         label="SINDy ODE → Decoder")
    axes[2].set_ylabel("MSE vs true x", fontsize=10)
    axes[2].set_xlabel("t (s)", fontsize=10)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Pointwise reconstruction error over time")

    status = "converged" if sim_ok else "⚠ DIVERGED"
    axes[0].set_title(
        f"SINDy 2nd-Order ODE vs Encoder Trajectory — IC #{ic_idx}  [{status}]",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_sindy_simulation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── figure 5: visual image snapshots ────────────────────────────────────────

def fig5_images(model, test_data, device, out_dir, ic_idx, n_snapshots=8):
    """True vs decoded 51×51 images at evenly spaced time snapshots."""
    t      = test_data["t"]
    n_steps = len(t)
    sl     = _ic_slice(ic_idx, n_steps)

    x_np  = test_data["x"][sl]    # [T, 2601]
    dx_np = test_data["dx"][sl]
    ddx_np = test_data["ddx"][sl]

    x_t   = torch.tensor(x_np,   dtype=torch.float32, device=device)
    dx_t  = torch.tensor(dx_np,  dtype=torch.float32, device=device)
    ddx_t = torch.tensor(ddx_np, dtype=torch.float32, device=device)

    outputs = model(x_t, dx_t, ddx_t)
    x_dec = outputs["x_decode"].detach().cpu().numpy()   # [T, 2601]

    step_indices = np.linspace(0, n_steps - 1, n_snapshots, dtype=int)
    n = 51   # image size

    fig, axes = plt.subplots(2, n_snapshots, figsize=(2.5 * n_snapshots, 5))

    vmin = x_np[step_indices].min()
    vmax = x_np[step_indices].max()

    for col, idx in enumerate(step_indices):
        img_true = x_np[idx].reshape(n, n)
        img_dec  = x_dec[idx].reshape(n, n)

        axes[0, col].imshow(img_true, cmap="hot", vmin=vmin, vmax=vmax, aspect="equal")
        axes[0, col].set_title(f"t={t[idx]:.2f}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_dec,  cmap="hot", vmin=vmin, vmax=vmax, aspect="equal")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("True", fontsize=10, labelpad=4)
    axes[1, 0].set_ylabel("Decoded", fontsize=10, labelpad=4)
    # Re-enable left axis labels without ticks
    for row in range(2):
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Pendulum Image Snapshots — IC #{ic_idx}\n(True vs Decoded)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_images.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True,
                        help="Path to checkpoint .mat (or .json) file, "
                             "e.g. model_20260501_150000.mat")
    parser.add_argument("--data", required=True,
                        help="Path to .npz data file; uses test_x/dx/ddx/z/t (falls back to val split)")
    parser.add_argument("--ic", type=int, default=0,
                        help="Which test IC to use for per-IC plots (default: 0)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for figures (default: same directory as .mat file)")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Strip extension to get prefix; infer format from extension
    if args.mat.endswith(".json"):
        mat_prefix = args.mat[:-5]
        fmt = "json"
    else:
        mat_prefix = args.mat[:-4] if args.mat.endswith(".mat") else args.mat
        fmt = "mat"

    print(f"Loading model from {mat_prefix} ({fmt})...")
    if fmt == "mat":
        model, params = load_model_mat(mat_prefix, device=device)
    else:
        model, params = load_model_json(mat_prefix, device=device)
    model.eval()
    print(f"  latent_dim={params['latent_dim']}, library_dim={params['library_dim']}, "
          f"model_order={params.get('model_order', 1)}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(mat_prefix))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from {args.data} ...")
    npz = np.load(args.data, allow_pickle=False)
    split = "test" if "test_x" in npz.files else "val"
    print(f"  Using '{split}' split.")
    test_data = {
        "x":   npz[f"{split}_x"],
        "dx":  npz[f"{split}_dx"],
        "ddx": npz[f"{split}_ddx"],
        "t":   npz[f"{split}_t"],
        "z":   npz[f"{split}_z"],
    }
    n_steps = len(test_data["t"])
    n_ics   = test_data["x"].shape[0] // n_steps
    print(f"  {n_ics} ICs × {n_steps} steps = {test_data['x'].shape[0]} samples")

    ic = args.ic
    print(f"\nGenerating figures (IC #{ic} for per-IC plots)...")
    print(f"Figures will be saved to: {out_dir}/")

    fig1_phase_portrait(model, test_data, device, out_dir)
    fig2_time_series(model, test_data, device, out_dir, ic)
    fig3_reconstruction(model, test_data, device, out_dir, ic)
    fig4_sindy_simulation(model, params, test_data, device, out_dir, ic)
    fig5_images(model, test_data, device, out_dir, ic)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
