"""Unified SINDy-Autoencoder analysis script.

Loads a trained model and produces two artefacts in the same directory as the
.mat checkpoint (timestamp inherited from the model filename):

    result_<stamp>.log   - Loss metrics + learned SINDy equations (human-readable)
    eval_<stamp>.npz     - All data needed by the visualize_*.py scripts:
        * Pass-through delay-embedding data (train, val, test)
        * Encoder latent representations             (train, val, test)
        * Encoder -> Decoder reconstructions         (train, val, test)
        * SINDy ODE integration from each IC's IC    (train, test)
        * SINDy -> Decoder reconstructions           (train, test)
        * SINDy convergence flags                    (train, test)
        * Learned SINDy coefficient matrix + metadata

Run from examples/:
    python3 analyze.py --mat lorenz/checkpoints/<dir>/model_<stamp>.mat \
                        --data lorenz/delay_xcoordinate_d20_5.npz

Visualization scripts then consume eval_<stamp>.npz directly.
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from scipy.integrate import solve_ivp

from sindyae import load_model_mat, compute_losses
from sindyae.sindy_library import sindy_library_torch


# ─── Logging ────────────────────────────────────────────────────────────────

class _Tee:
    """Mirror every write() to both the original stream and a log file."""
    def __init__(self, stream, log_path):
        self._stream = stream
        self._log = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def close(self):
        self._log.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _extract_stamp(mat_prefix):
    """Pull the timestamp portion out of a 'model_<stamp>' prefix.

    Supports both 'model_YYYYMMDD_HHMMSS' (legacy) and 'model_MMDD_HHMMSS'.
    Falls back to the full basename if it doesn't start with 'model_'.
    """
    basename = os.path.basename(mat_prefix)
    if basename.startswith("model_"):
        return basename[len("model_"):]
    return basename


def _make_batch(data, device, model_order):
    batch = {
        "x":  torch.as_tensor(data["x"],  dtype=torch.float32, device=device),
        "dx": torch.as_tensor(data["dx"], dtype=torch.float32, device=device),
    }
    if model_order == 2 and "ddx" in data:
        batch["ddx"] = torch.as_tensor(data["ddx"], dtype=torch.float32, device=device)
    return batch


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


def _full_forward(model, x_np, dx_np, ddx_np, device, model_order, bs=4096):
    """Run model forward pass on (x, dx[, ddx]); return reconstructions.

    Returns (x_dec, dx_dec, ddx_dec). ddx_dec is None when model_order == 1.
    """
    x_dec_list, dx_dec_list, ddx_dec_list = [], [], []
    for i in range(0, len(x_np), bs):
        xb  = torch.tensor(x_np[i:i+bs],  dtype=torch.float32, device=device)
        dxb = torch.tensor(dx_np[i:i+bs], dtype=torch.float32, device=device)
        ddxb = None
        if model_order == 2 and ddx_np is not None:
            ddxb = torch.tensor(ddx_np[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = model(xb, dxb, ddxb)
        x_dec_list.append(outputs["x_decode"].cpu().numpy())
        dx_dec_list.append(outputs["dx_decode"].cpu().numpy())
        if model_order == 2 and "ddx_decode" in outputs:
            ddx_dec_list.append(outputs["ddx_decode"].cpu().numpy())
    x_dec  = np.concatenate(x_dec_list)
    dx_dec = np.concatenate(dx_dec_list)
    ddx_dec = np.concatenate(ddx_dec_list) if ddx_dec_list else None
    return x_dec, dx_dec, ddx_dec


def _sindy_rhs(Xi, poly_order, include_sine):
    """SciPy-compatible RHS for the learned SINDy ODE (order 1)."""
    def rhs(t, z):
        zt = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            Theta = sindy_library_torch(zt, poly_order, include_sine)
        return (Theta.numpy() @ Xi)[0]
    return rhs


def _get_t(npz, split):
    """Resolve the time array for a split, falling back as needed."""
    for key in (f"{split}_t", "t", "train_t"):
        if key in npz.files:
            return np.asarray(npz[key])
    # Synthesize from dt + length
    if "dt" in npz.files:
        n = npz[f"{split}_x"].shape[0]
        return np.arange(n) * float(npz["dt"])
    raise KeyError(f"Cannot resolve time array for split '{split}'.")


def _integrate_per_ic(model, params, x_np, t, n_steps, device,
                      solver, rtol, atol, max_step):
    """For each IC (chunk of n_steps in x_np), integrate the learned SINDy ODE
    from the encoder's output at that IC's first sample.

    Returns:
        z_sindy: [N_total, latent_dim] -- NaN rows for diverged ICs.
        x_sindy: [N_total, input_dim]  -- NaN rows for diverged ICs.
        ok     : [n_ics,] bool         -- True for ICs that converged.
    """
    n_total = x_np.shape[0]
    n_ics   = n_total // n_steps
    if n_ics == 0:
        # Treat the whole thing as a single trajectory
        n_ics   = 1
        n_steps = n_total

    Xi           = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    poly_order   = params["poly_order"]
    include_sine = params.get("include_sine", False)
    latent_dim   = params["latent_dim"]
    input_dim    = x_np.shape[1]

    rhs = _sindy_rhs(Xi, poly_order, include_sine)

    z_sindy = np.full((n_total, latent_dim), np.nan, dtype=np.float64)
    x_sindy = np.full((n_total, input_dim),  np.nan, dtype=np.float64)
    ok      = np.zeros(n_ics, dtype=bool)

    solver_kwargs = dict(method=solver, rtol=rtol, atol=atol)
    if max_step is not None:
        solver_kwargs["max_step"] = max_step

    for ic in range(n_ics):
        sl   = slice(ic * n_steps, (ic + 1) * n_steps)
        x_seg = x_np[sl]
        x0_t = torch.tensor(x_seg[0:1], dtype=torch.float32, device=device)
        with torch.no_grad():
            z0 = model.encoder(x0_t).cpu().numpy()[0]

        try:
            sol = solve_ivp(rhs, [t[0], t[-1]], z0, t_eval=t, **solver_kwargs)
            if sol.success and not np.any(np.isnan(sol.y)):
                z_traj = sol.y[:latent_dim, :].T   # [n_steps, latent_dim]
                z_sindy[sl] = z_traj
                x_sindy[sl] = _decode(model, z_traj.astype(np.float32), device)
                ok[ic] = True
            else:
                print(f"    IC {ic}: integration did not converge (success={sol.success})")
        except Exception as e:
            print(f"    IC {ic}: integration failed ({e})")

    return z_sindy, x_sindy, ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SINDy-AE checkpoint and dump data for visualization."
    )
    parser.add_argument("--mat",  required=True,
                        help="Path to checkpoint .mat file (with or without .mat extension)")
    parser.add_argument("--data", required=True,
                        help="Path to .npz data file used for training")
    parser.add_argument("--solver", default="DOP853",
                        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
                        help="ODE solver for SINDy integration (default: DOP853)")
    parser.add_argument("--rtol", type=float, default=1e-8,
                        help="ODE solver relative tolerance (default: 1e-8)")
    parser.add_argument("--atol", type=float, default=1e-10,
                        help="ODE solver absolute tolerance (default: 1e-10)")
    parser.add_argument("--max_step", type=float, default=None,
                        help="ODE solver max step size (default: None = unlimited)")
    parser.add_argument("--skip_sindy", action="store_true",
                        help="Skip the SINDy integration block (encoder/decoder data only)")
    parser.add_argument("--max_ics", type=int, default=0,
                        help="Cap the number of ICs processed per split (train/val/test). "
                             "0 = no cap (default). Truncation applied uniformly to x/dx/[ddx]/z arrays.")
    args = parser.parse_args()

    # ---- Resolve paths and timestamps ----
    mat_prefix  = args.mat[:-4] if args.mat.endswith(".mat") else args.mat
    out_dir     = os.path.dirname(os.path.abspath(mat_prefix))
    model_stamp = _extract_stamp(mat_prefix)
    log_path    = os.path.join(out_dir, f"result_{model_stamp}.log")
    npz_path    = os.path.join(out_dir, f"eval_{model_stamp}.npz")

    sys.stdout = _Tee(sys.stdout, log_path)
    print(f"Log        : {log_path}")
    print(f"NPZ output : {npz_path}")
    print(f"Checkpoint : {mat_prefix}.mat")
    print(f"Data file  : {args.data}")

    # ---- Load model ----
    device = torch.device("cpu")   # analysis is light, CPU keeps things deterministic
    model, params = load_model_mat(mat_prefix, device=device)
    model.eval()

    model_order = params.get("model_order", 1)
    print(f"\nLoaded model -- latent_dim={params['latent_dim']}, "
          f"library_dim={params['library_dim']}, model_order={model_order}")

    # ---- Load data ----
    npz = np.load(args.data, allow_pickle=False)
    has_test  = "test_x" in npz.files
    eval_split = "test" if has_test else "val"
    available_splits = [s for s in ("train", "val", "test") if f"{s}_x" in npz.files]
    print(f"Using '{eval_split}' split for loss evaluation. Splits in data: {available_splits}")

    # ---- Loss metrics on eval split (preserves prior behavior) ----
    eval_data = {
        "x":  npz[f"{eval_split}_x"],
        "dx": npz[f"{eval_split}_dx"],
    }
    if model_order == 2:
        key = f"{eval_split}_ddx"
        if key in npz.files:
            eval_data["ddx"] = npz[key]

    n = eval_data["x"].shape[0]
    print(f"  Evaluation samples: {n:,}  (shape: {eval_data['x'].shape})")

    batch   = _make_batch(eval_data, device, model_order)
    outputs = model(batch["x"], batch["dx"], batch.get("ddx"))
    losses  = compute_losses(outputs, batch, model, params)

    x_t      = batch["x"].detach()
    x_dec_t  = outputs["x_decode"].detach()
    rel_err_x = (x_t - x_dec_t).pow(2).mean() / x_t.pow(2).mean()

    if model_order == 1:
        d_t      = batch["dx"].detach()
        d_dec_t  = outputs["dx_decode"].detach()
        d_label  = "dx"
    else:
        d_t      = batch["ddx"].detach()
        d_dec_t  = outputs["ddx_decode"].detach()
        d_label  = "ddx"
    rel_err_d = (d_t - d_dec_t).pow(2).mean() / d_t.pow(2).mean()

    print("\n--- Test set metrics ---")
    print(f"  decoder MSE:            {losses['decoder'].item():.3e}")
    print(f"  sindy_z MSE:            {losses['sindy_z'].item():.3e}")
    print(f"  sindy_x MSE:            {losses['sindy_x'].item():.3e}")
    if "coord" in losses:
        print(f"  coord MSE:              {losses['coord'].item():.3e}")
    print(f"  relative error (x):     {rel_err_x.item():.3e}")
    print(f"  relative error ({d_label}):   {rel_err_d.item():.3e}")

    Xi     = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()
    active = int((np.abs(Xi) > 0).sum())
    print(f"\n--- Learned SINDy coefficients ({active} active) ---")
    np.set_printoptions(precision=4, suppress=True)
    print(Xi)

    # ---- Build eval data dictionary ----
    print(f"\n=== Building {os.path.basename(npz_path)} ===")
    eval_dict = {}

    # Metadata (1-D arrays/scalars only — npz friendly)
    eval_dict["model_stamp"]  = model_stamp
    eval_dict["data_path"]    = args.data
    eval_dict["latent_dim"]   = params["latent_dim"]
    eval_dict["library_dim"]  = params["library_dim"]
    eval_dict["poly_order"]   = params["poly_order"]
    eval_dict["include_sine"] = bool(params.get("include_sine", False))
    eval_dict["model_order"]  = model_order
    eval_dict["Xi"]           = Xi

    # Pass scalar metadata through from data file
    for k in ("dt", "t_end", "t_start", "delay_dim", "delay_steps", "tau",
              "noise_strength", "n_train_ics", "n_val_ics", "n_test_ics"):
        if k in npz.files:
            v = npz[k]
            eval_dict[k] = v.item() if np.ndim(v) == 0 else v

    # SINDy integration settings (for reproducibility)
    eval_dict["sindy_solver"] = args.solver
    eval_dict["sindy_rtol"]   = args.rtol
    eval_dict["sindy_atol"]   = args.atol
    if args.max_step is not None:
        eval_dict["sindy_max_step"] = args.max_step

    # ---- Per-split: pass-through + encoder + encoder->decoder ----
    for split in available_splits:
        t_split = _get_t(npz, split)
        n_steps = len(t_split)

        x_full     = np.asarray(npz[f"{split}_x"])
        n_ics_orig = max(1, x_full.shape[0] // n_steps)
        if args.max_ics > 0 and args.max_ics < n_ics_orig:
            n_ics_keep = args.max_ics
            n_rows     = n_ics_keep * n_steps
            print(f"  [{split}] --max_ics={args.max_ics}: truncating from {n_ics_orig} ICs "
                  f"to {n_ics_keep} ICs ({n_rows} samples)")
        else:
            n_ics_keep = n_ics_orig
            n_rows     = x_full.shape[0]

        x_np  = x_full[:n_rows]
        dx_np = np.asarray(npz[f"{split}_dx"])[:n_rows]
        eval_dict[f"{split}_x"]  = x_np
        eval_dict[f"{split}_dx"] = dx_np
        eval_dict[f"{split}_t"]  = t_split

        # Optional pass-throughs: z (ground truth), ddx
        if f"{split}_z" in npz.files:
            eval_dict[f"{split}_z"] = np.asarray(npz[f"{split}_z"])[:n_rows]
        ddx_np = None
        if f"{split}_ddx" in npz.files:
            ddx_np = np.asarray(npz[f"{split}_ddx"])[:n_rows]
            eval_dict[f"{split}_ddx"] = ddx_np

        # Encoder
        z_enc = _encode(model, x_np, device)
        eval_dict[f"{split}_z_enc"] = z_enc

        # Encoder -> Decoder reconstructions (full forward pass)
        x_dec, dx_dec, ddx_dec = _full_forward(
            model, x_np, dx_np, ddx_np, device, model_order,
        )
        eval_dict[f"{split}_x_dec"]  = x_dec
        eval_dict[f"{split}_dx_dec"] = dx_dec
        if ddx_dec is not None:
            eval_dict[f"{split}_ddx_dec"] = ddx_dec

        # Override n_{split}_ics metadata to reflect what we actually kept.
        eval_dict[f"n_{split}_ics"] = n_ics_keep

        print(f"  [{split}] x: {x_np.shape}  ->  z_enc: {z_enc.shape}  x_dec: {x_dec.shape}")

    # ---- Per-split: SINDy integration (train + test) ----
    if not args.skip_sindy and model_order == 1:
        sindy_splits = [s for s in ("train", "test") if s in available_splits]
        for split in sindy_splits:
            x_np    = eval_dict[f"{split}_x"]   # already truncated
            t       = eval_dict[f"{split}_t"]
            n_steps = len(t)
            n_ics   = max(1, x_np.shape[0] // n_steps)
            print(f"  [{split}] SINDy ODE integration: {n_ics} IC(s) x {n_steps} steps "
                  f"(solver={args.solver}, rtol={args.rtol}, atol={args.atol})")

            z_sindy, x_sindy, ok = _integrate_per_ic(
                model, params, x_np, t, n_steps, device,
                args.solver, args.rtol, args.atol, args.max_step,
            )
            eval_dict[f"{split}_z_sindy"]  = z_sindy
            eval_dict[f"{split}_x_sindy"]  = x_sindy
            eval_dict[f"{split}_sindy_ok"] = ok
            print(f"    Converged: {int(ok.sum())}/{n_ics}")
    elif model_order == 2:
        print("  (SINDy integration skipped for model_order=2; not implemented in analyze.)")

    # ---- Save ----
    np.savez(npz_path, **eval_dict)
    print(f"\nSaved eval data: {npz_path}")


if __name__ == "__main__":
    main()
