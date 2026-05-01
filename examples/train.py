"""Unified SINDy-Autoencoder training script.

Run from examples/:
    python3 train.py --data lorenz/delay_xcoordinate_d20_5.npz
    python3 train.py --data pendulum/pendulum_data.npz --latent_dim 1 --include_sine

All outputs (.mat model, log, loss plot) are written to the same directory as --data.

Model order is auto-detected: if the .npz contains train_ddx → order 2 (pendulum);
otherwise order 1 (Lorenz / delay embedding).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sindyae import library_size, train_network, save_model_mat


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


def _load_split(npz, prefix):
    """Extract keys with given prefix, strip prefix from key names."""
    return {k[len(prefix):]: npz[k] for k in npz.files if k.startswith(prefix)}


def _load_meta(npz):
    """Read scalar metadata fields from npz."""
    meta = {}
    for k in ("t_end", "dt", "noise_strength", "n_train_ics", "n_val_ics", "n_test_ics",
              "delay_dim", "delay_steps", "tau", "t_start"):
        if k in npz.files:
            meta[k] = npz[k].item()
    return meta


def _save_loss_plot(validation_losses, out_path):
    """Save a semilogy plot of each loss component and total loss over epochs."""
    if not validation_losses:
        return

    epochs = [d["epoch"] for d in validation_losses]
    phases = [d.get("phase", "main") for d in validation_losses]

    # Determine which loss keys are present
    sample = validation_losses[0]
    loss_keys = [k for k in ("decoder", "sindy_z", "sindy_x", "sindy_regularization", "coord")
                 if k in sample]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "total":                "#1f77b4",
        "decoder":              "#ff7f0e",
        "sindy_z":              "#2ca02c",
        "sindy_x":              "#d62728",
        "sindy_regularization": "#9467bd",
        "coord":                "#8c564b",
    }
    labels = {
        "total":                "total",
        "decoder":              "decoder",
        "sindy_z":              "sindy_z",
        "sindy_x":              "sindy_x",
        "sindy_regularization": "reg",
        "coord":                "coord",
    }

    ax.semilogy(epochs, [d["total"] for d in validation_losses],
                color=colors["total"], lw=2, label="total")
    for key in loss_keys:
        vals = [d.get(key, float("nan")) for d in validation_losses]
        ax.semilogy(epochs, vals, lw=1.2, linestyle="--",
                    color=colors.get(key, "gray"), label=labels.get(key, key))

    # Shade refinement region if present
    refine_start = next((d["epoch"] for d in validation_losses if d.get("phase") == "refine"), None)
    if refine_start is not None:
        ax.axvline(refine_start, color="gray", linestyle=":", lw=1.0, label="refinement start")

    ax.set_xlabel("Epoch (validation eval)")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Validation losses over training")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified SINDy-AE training. Outputs go to same directory as --data."
    )

    # --- Data ---
    parser.add_argument("--data", required=True,
                        help="Path to .npz file with train_x, train_dx (and optionally train_ddx)")

    # --- Architecture ---
    parser.add_argument("--latent_dim",  type=int,   default=3,
                        help="Latent space dimension (default: 3)")
    parser.add_argument("--poly_order",  type=int,   default=3,
                        help="SINDy library polynomial order (default: 3)")
    parser.add_argument("--include_sine", action="store_true",
                        help="Include sin(z_i) terms in SINDy library (default: off)")
    parser.add_argument("--widths",      default="64,32",
                        help="Comma-separated encoder hidden widths (default: 64,32)")
    parser.add_argument("--activation",  default="tanh",
                        choices=["sigmoid", "tanh", "elu", "relu", "linear"],
                        help="Hidden-layer activation (default: tanh)")

    # --- Loss weights ---
    parser.add_argument("--loss_weight_decoder",   type=float, default=1.0)
    parser.add_argument("--loss_weight_sindy_z",   type=float, default=0.01)
    parser.add_argument("--loss_weight_sindy_x",   type=float, default=0.01)
    parser.add_argument("--loss_weight_sindy_reg", type=float, default=1e-4,
                        help="L1 regularization weight on SINDy coefficients (default: 1e-4)")
    parser.add_argument("--loss_weight_coord",     type=float, default=0.0,
                        help="Weight for coord loss z[:,0] ≈ x[:,0]; useful for delay embedding "
                             "(default: 0.0 = disabled)")

    # --- Thresholding ---
    parser.add_argument("--coefficient_threshold", type=float, default=0.2)
    parser.add_argument("--threshold_frequency",   type=int,   default=500)

    # --- Training ---
    parser.add_argument("--max_epochs",        type=int,   default=15001)
    parser.add_argument("--refinement_epochs", type=int,   default=3001)
    parser.add_argument("--batch_size",        type=int,   default=1024)
    parser.add_argument("--learning_rate",     type=float, default=1e-3)
    parser.add_argument("--patience",          type=int,   default=5,
                        help="Early-stop: consecutive validation evals with no improvement "
                             "(default: 5; 0 = disabled)")
    parser.add_argument("--print_frequency",   type=int,   default=100)

    # --- LR scheduler ---
    parser.add_argument("--scheduler_type",    default="plateau",
                        choices=["plateau", "cosine", "none"])
    parser.add_argument("--scheduler_patience", type=int,   default=500)
    parser.add_argument("--scheduler_factor",   type=float, default=0.5)
    parser.add_argument("--scheduler_min_lr",   type=float, default=1e-5)

    args = parser.parse_args()

    # ---- Output directory = same folder as input data ----
    out_dir = os.path.dirname(os.path.abspath(args.data))
    stamp   = datetime.datetime.now().strftime("%m%d_%H%M")
    stamp_long = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(out_dir, f"output{stamp}.log")
    sys.stdout = _Tee(sys.stdout, log_path)
    print(f"Log: {log_path}")

    torch.manual_seed(0)
    np.random.seed(0)

    # ---- Load data ----
    print(f"Loading data from {args.data} …")
    npz  = np.load(args.data, allow_pickle=False)
    meta = _load_meta(npz)

    training_data   = _load_split(npz, "train_")
    validation_data = _load_split(npz, "val_")

    # Auto-detect model order
    model_order = 2 if "train_ddx" in npz.files else 1

    # ---- DATA SUMMARY ----
    print()
    print("=" * 60)
    print("DATA SUMMARY")
    print(f"  train samples : {training_data['x'].shape[0]:,}  (shape: {training_data['x'].shape})")
    print(f"  val   samples : {validation_data['x'].shape[0]:,}  (shape: {validation_data['x'].shape})")
    print(f"  input_dim     : {training_data['x'].shape[1]}")
    print(f"  model_order   : {model_order}  (auto-detected from {'train_ddx' if model_order == 2 else 'no ddx'})")
    if "dt" in meta and "t_end" in meta:
        print(f"  time range    : {meta.get('t_start', 0.0)} → {meta['t_end']}  (dt={meta['dt']})")
    if "delay_dim" in meta:
        print(f"  delay embed   : d={meta['delay_dim']},  "
              f"τ={meta.get('tau', '?'):.3f}s  ({int(meta.get('delay_steps', 0))} steps)")
    if "noise_strength" in meta:
        print(f"  noise_strength: {meta['noise_strength']}")
    if "n_train_ics" in meta:
        print(f"  n_train_ics   : {meta['n_train_ics']}")
    if "n_val_ics" in meta:
        print(f"  n_val_ics     : {meta['n_val_ics']}")
    if "n_test_ics" in meta:
        print(f"  n_test_ics    : {meta['n_test_ics']}")
    print("=" * 60)
    print()

    # ---- Build params ----
    input_dim  = training_data["x"].shape[1]
    latent_dim = args.latent_dim
    lib_n      = 2 * latent_dim if model_order == 2 else latent_dim
    widths     = [int(w) for w in args.widths.split(",")]

    params = {
        "input_dim":    input_dim,
        "latent_dim":   latent_dim,
        "model_order":  model_order,
        "poly_order":   args.poly_order,
        "include_sine": args.include_sine,
        "library_dim":  library_size(lib_n, args.poly_order, args.include_sine, True),
        "activation":   args.activation,
        "widths":       widths,

        # Thresholding
        "sequential_thresholding":    True,
        "coefficient_threshold":      args.coefficient_threshold,
        "threshold_frequency":        args.threshold_frequency,
        "coefficient_initialization": "constant",
        "coefficient_mask":           np.ones((
            library_size(lib_n, args.poly_order, args.include_sine, True),
            latent_dim,
        )),

        # Loss weights
        "loss_weight_decoder":              args.loss_weight_decoder,
        "loss_weight_sindy_z":              args.loss_weight_sindy_z,
        "loss_weight_sindy_x":              args.loss_weight_sindy_x,
        "loss_weight_sindy_regularization": args.loss_weight_sindy_reg,
        "loss_weight_coord":                args.loss_weight_coord,

        # LR scheduler
        "scheduler_type":    None if args.scheduler_type == "none" else args.scheduler_type,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor":   args.scheduler_factor,
        "scheduler_min_lr":   args.scheduler_min_lr,

        # Training
        "epoch_size":               training_data["x"].shape[0],
        "batch_size":               args.batch_size,
        "learning_rate":            args.learning_rate,
        "max_epochs":               args.max_epochs,
        "refinement_epochs":        args.refinement_epochs,
        "print_progress":           True,
        "print_frequency":          args.print_frequency,
        "early_stopping_patience":  args.patience,
        "early_stopping_min_delta": 1e-6,
    }

    # Persist metadata so checkpoint is self-contained
    for k in ("delay_dim", "delay_steps", "dt", "t_end", "t_start", "tau"):
        if k in meta:
            params[k] = meta[k]

    # ---- HYPERPARAMETER SUMMARY ----
    enc_arch = (f"{input_dim} → "
                + " → ".join(str(w) for w in widths)
                + f" → {latent_dim}")
    dec_arch = (f"{latent_dim} → "
                + " → ".join(str(w) for w in reversed(widths))
                + f" → {input_dim}")
    sched_str = (
        f"{params['scheduler_type']}  "
        f"(factor={params['scheduler_factor']}, "
        f"patience={params['scheduler_patience']} ep, "
        f"min_lr={params['scheduler_min_lr']:.0e})"
        if params["scheduler_type"] else "none"
    )
    coord_str = (f"{args.loss_weight_coord:.0e}"
                 if args.loss_weight_coord > 0 else "disabled")

    print("=" * 60)
    print("HYPERPARAMETERS")
    print(f"  Encoder       : {enc_arch}")
    print(f"  Decoder       : {dec_arch}")
    print(f"  Activation    : {args.activation}")
    print(f"  poly_order    : {args.poly_order},  include_sine={args.include_sine}")
    print(f"  library_dim   : {params['library_dim']}")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  learning_rate : {args.learning_rate:.0e}")
    print(f"  Loss weights  : decoder={args.loss_weight_decoder},  "
          f"sindy_z={args.loss_weight_sindy_z},  "
          f"sindy_x={args.loss_weight_sindy_x},  "
          f"reg={args.loss_weight_sindy_reg:.0e},  "
          f"coord={coord_str}")
    print(f"  Thresholding  : freq={args.threshold_frequency} ep,  "
          f"threshold={args.coefficient_threshold}")
    print(f"  LR scheduler  : {sched_str}")
    print(f"  Early stop    : patience={args.patience} evals  "
          f"(each eval = {args.print_frequency} epochs)")
    print(f"  max_epochs    : {args.max_epochs}  +  refinement={args.refinement_epochs}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device        : {device}")
    print("=" * 60)
    print()

    save_path = os.path.join(out_dir, f"model_{stamp_long}")
    print(f"Outputs will be written to: {out_dir}/")
    print()

    # ---- Train ----
    results = train_network(training_data, validation_data, params, device=device)
    model   = results["model"]

    # ---- Save model (.mat only) ----
    mat_path = save_model_mat(model, params, save_path)
    print(f"\nSaved .mat: {mat_path}")

    # ---- Save loss plot ----
    loss_plot_path = os.path.join(out_dir, f"loss{stamp}.png")
    _save_loss_plot(results["losses"], loss_plot_path)

    # ---- Print final coefficients ----
    final_coeffs = results["final_sindy_coefficients"]
    print(f"\nFinal active coefficients: {int((np.abs(final_coeffs) > 0).sum())}")
    print("Learned SINDy coefficient matrix (masked):")
    np.set_printoptions(precision=4, suppress=True)
    print(final_coeffs)


if __name__ == "__main__":
    main()
