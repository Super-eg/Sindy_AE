"""Unified SINDy-Autoencoder analysis script.

Run from examples/:
    python3 analyze.py --mat lorenz/model_20260501_150000.mat --data lorenz/delay_xcoordinate_d20_5.npz

Loads the trained model and evaluates on test data (falls back to val data if test
split is not present). Prints loss metrics and learned SINDy coefficients.
Result log is written to the same directory as --mat.
"""
import argparse
import datetime
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from sindyae import load_model_mat, compute_losses


class _Tee:
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


def _make_batch(data, device, model_order):
    batch = {
        "x":  torch.as_tensor(data["x"],  dtype=torch.float32, device=device),
        "dx": torch.as_tensor(data["dx"], dtype=torch.float32, device=device),
    }
    if model_order == 2 and "ddx" in data:
        batch["ddx"] = torch.as_tensor(data["ddx"], dtype=torch.float32, device=device)
    return batch


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SINDy-AE checkpoint."
    )
    parser.add_argument("--mat",  required=True,
                        help="Path to checkpoint .mat file (without .mat extension), "
                             "e.g. lorenz/model_20260501_150000")
    parser.add_argument("--data", required=True,
                        help="Path to .npz data file used for training")
    args = parser.parse_args()

    # Normalise mat prefix (strip .mat if user included it)
    mat_prefix = args.mat[:-4] if args.mat.endswith(".mat") else args.mat

    out_dir  = os.path.dirname(os.path.abspath(mat_prefix))
    stamp    = datetime.datetime.now().strftime("%m%d_%H%M")
    log_path = os.path.join(out_dir, f"result{stamp}.log")
    sys.stdout = _Tee(sys.stdout, log_path)
    print(f"Results will be saved to: {log_path}")
    print(f"Checkpoint : {mat_prefix}.mat")
    print(f"Data file  : {args.data}")

    device = torch.device("cpu")   # analysis is cheap; CPU keeps output deterministic
    model, params = load_model_mat(mat_prefix, device=device)
    model.eval()

    model_order = params.get("model_order", 1)
    print(f"\nLoaded model — latent_dim={params['latent_dim']},  "
          f"library_dim={params['library_dim']},  model_order={model_order}")

    # ---- Load test data (fall back to val if test split absent) ----
    npz = np.load(args.data, allow_pickle=False)

    has_test = "test_x" in npz.files
    split    = "test" if has_test else "val"
    print(f"Using {'test' if has_test else 'validation'} split for evaluation.")

    data = {
        "x":  npz[f"{split}_x"],
        "dx": npz[f"{split}_dx"],
    }
    if model_order == 2:
        key = f"{split}_ddx"
        if key in npz.files:
            data["ddx"] = npz[key]
        else:
            print(f"WARNING: model_order=2 but {key} not found in data; ddx will be missing.")

    n = data["x"].shape[0]
    print(f"  Evaluation samples: {n:,}  (shape: {data['x'].shape})")

    batch   = _make_batch(data, device, model_order)
    outputs = model(batch["x"], batch["dx"], batch.get("ddx"))
    losses  = compute_losses(outputs, batch, model, params)

    # Relative reconstruction errors
    x     = batch["x"].detach()
    x_dec = outputs["x_decode"].detach()
    rel_err_x = (x - x_dec).pow(2).mean() / x.pow(2).mean()

    if model_order == 1:
        dx     = batch["dx"].detach()
        dx_dec = outputs["dx_decode"].detach()
        rel_err_d = (dx - dx_dec).pow(2).mean() / dx.pow(2).mean()
        d_label   = "dx"
    else:
        ddx     = batch["ddx"].detach()
        ddx_dec = outputs["ddx_decode"].detach()
        rel_err_d = (ddx - ddx_dec).pow(2).mean() / ddx.pow(2).mean()
        d_label   = "ddx"

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


if __name__ == "__main__":
    main()
