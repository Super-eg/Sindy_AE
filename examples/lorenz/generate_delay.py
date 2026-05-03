"""Generate Lorenz data using delay embedding of the x-coordinate and save to .npz.

Instead of projecting all 3 Lorenz coordinates to high-dim space, we observe
only x(t) and apply Takens-style delay embedding:

    input[t] = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

With embedding dimension d ≥ 2·dim(attractor)+1 ≈ 7, the delay vector is
diffeomorphic to the original 3-D Lorenz attractor (Takens' theorem).

Run from examples/lorenz/:
    python3 generate_delay.py

Output .npz is passed to examples/train.py via --data <path>.
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from example_lorenz import simulate_lorenz


def get_lorenz_data_delay_x(n_ics, t, delay_dim=20, delay_steps=5,
                             noise_strength=0):
    """Generate delay-embedded Lorenz data from the x-coordinate only.

    For each IC and each valid time step t_j, the input vector is:
        x[j]  = [z_x(t_j), z_x(t_j - τ), ..., z_x(t_j - (d-1)τ)]
        dx[j] = [ż_x(t_j), ż_x(t_j - τ), ..., ż_x(t_j - (d-1)τ)]

    where τ = delay_steps × dt and z_x is the normalized x-coordinate (×1/40).

    Parameters
    ----------
    n_ics        : number of initial conditions (trajectories)
    t            : time array (np.arange)
    delay_dim    : embedding dimension d  (default 20)
    delay_steps  : gap between delays in time-steps, τ = delay_steps × dt
    noise_strength: Gaussian noise added to x and dx

    Returns
    -------
    dict with keys:
        x   : (n_ics * n_valid, delay_dim)  float32
        dx  : (n_ics * n_valid, delay_dim)  float32
        z   : (n_ics * n_valid, 3)          float32  — true normalized Lorenz state
        t   : (n_valid,)   valid time points (first (d-1)*delay_steps dropped)
    """
    normalization = 1.0 / 40.0          # same scale as example_lorenz.py

    ic_means  = np.array([0.0,  0.0, 25.0])
    ic_widths = 2 * np.array([36.0, 48.0, 41.0])
    ics = ic_widths * (np.random.rand(n_ics, 3) - 0.5) + ic_means

    n_steps = t.size
    min_idx = (delay_dim - 1) * delay_steps   # first valid time index
    n_valid = n_steps - min_idx

    if n_valid <= 0:
        raise ValueError(
            f"Time series too short for delay_dim={delay_dim}, "
            f"delay_steps={delay_steps}. Need at least "
            f"{min_idx + 1} steps, got {n_steps}."
        )

    x_all  = np.zeros((n_ics, n_valid, delay_dim), dtype=np.float32)
    dx_all = np.zeros_like(x_all)
    z_all  = np.zeros((n_ics, n_valid, 3), dtype=np.float32)

    for i in range(n_ics):
        z, dz, _ = simulate_lorenz(ics[i], t)
        z_x  = z[:, 0]  * normalization   # shape (n_steps,)
        dz_x = dz[:, 0] * normalization

        for k in range(delay_dim):
            s = min_idx - k * delay_steps
            e = s + n_valid
            x_all[i, :, k]  = z_x[s:e]
            dx_all[i, :, k] = dz_x[s:e]

        # True normalized Lorenz state at valid time points
        z_all[i] = (z[min_idx:min_idx + n_valid] * normalization).astype(np.float32)

    # Flatten: (n_ics, n_valid, delay_dim) → (n_ics*n_valid, delay_dim)
    x_flat  = x_all.reshape(-1, delay_dim)
    dx_flat = dx_all.reshape(-1, delay_dim)
    z_flat  = z_all.reshape(-1, 3)

    if noise_strength > 0:
        x_flat  = x_flat  + (noise_strength * np.random.randn(*x_flat.shape)).astype(np.float32)
        dx_flat = dx_flat + (noise_strength * np.random.randn(*dx_flat.shape)).astype(np.float32)

    return {'x': x_flat, 'dx': dx_flat, 'z': z_flat, 't': t[min_idx:]}


def main():
    parser = argparse.ArgumentParser(
        description="Generate Lorenz data via delay embedding of x-coordinate and save to .npz"
    )
    parser.add_argument("--n_train_ics",    type=int,   default=1024)
    parser.add_argument("--n_val_ics",      type=int,   default=20)
    parser.add_argument("--n_test_ics",     type=int,   default=20)
    parser.add_argument("--noise_strength", type=float, default=1e-6)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--t_start",        type=float, default=0.0)
    parser.add_argument("--t_end",          type=float, default=10.0)
    parser.add_argument("--dt",             type=float, default=0.02)
    parser.add_argument("--delay_dim",      type=int,   default=20,
                        help="Embedding dimension d (default: 20)")
    parser.add_argument("--delay_steps",    type=int,   default=5,
                        help="Gap between delays in time-steps; "
                             "τ = delay_steps × dt (default: 5 → τ=0.1s)")
    parser.add_argument(
        "--out", default=None,
        help="Output .npz file path (default: delay_xcoordinate_d<dim>_<steps>.npz)",
    )
    args = parser.parse_args()

    # Build default filename from actual dim/steps after parsing.
    if args.out is None:
        args.out = f"delay_xcoordinate_d{args.delay_dim}_{args.delay_steps}.npz"

    np.random.seed(args.seed)

    t   = np.arange(args.t_start, args.t_end, args.dt)
    tau = args.delay_steps * args.dt

    print("=" * 60)
    print("Lorenz data generation — delay embedding of x-coordinate")
    print(f"  Embedding dim  : {args.delay_dim}  (d)")
    print(f"  Delay step     : {args.delay_steps} steps  (τ = {tau:.3f} s)")
    print(f"  Time range     : {args.t_start} → {args.t_end}  (dt={args.dt})")
    print(f"  Min valid steps: {t.size} − {(args.delay_dim-1)*args.delay_steps} = "
          f"{t.size - (args.delay_dim-1)*args.delay_steps}")
    print(f"  Train ICs      : {args.n_train_ics}")
    print(f"  Val ICs        : {args.n_val_ics}")
    print(f"  Test ICs       : {args.n_test_ics}")
    print(f"  Noise          : {args.noise_strength}")
    print(f"  Output         : {args.out}")
    print("=" * 60)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Generating training data ({args.n_train_ics} ICs) …")
    train = get_lorenz_data_delay_x(
        args.n_train_ics, t,
        delay_dim=args.delay_dim, delay_steps=args.delay_steps,
        noise_strength=args.noise_strength,
    )
    print(f"  → train_x shape: {train['x'].shape}")

    print(f"Generating validation data ({args.n_val_ics} ICs) …")
    val = get_lorenz_data_delay_x(
        args.n_val_ics, t,
        delay_dim=args.delay_dim, delay_steps=args.delay_steps,
        noise_strength=args.noise_strength,
    )
    print(f"  → val_x shape: {val['x'].shape}")

    print(f"Generating test data ({args.n_test_ics} ICs) …")
    test = get_lorenz_data_delay_x(
        args.n_test_ics, t,
        delay_dim=args.delay_dim, delay_steps=args.delay_steps,
        noise_strength=args.noise_strength,
    )
    print(f"  → test_x shape: {test['x'].shape}")

    save_dict = {f"train_{k}": v for k, v in train.items()}
    save_dict.update({f"val_{k}": v for k, v in val.items()})
    save_dict.update({f"test_{k}": v for k, v in test.items()})
    save_dict["t_start"]        = np.float64(args.t_start)
    save_dict["t_end"]          = np.float64(args.t_end)
    save_dict["dt"]             = np.float64(args.dt)
    save_dict["delay_dim"]      = np.int64(args.delay_dim)
    save_dict["delay_steps"]    = np.int64(args.delay_steps)
    save_dict["tau"]            = np.float64(tau)
    save_dict["noise_strength"] = np.float64(args.noise_strength)
    save_dict["n_train_ics"]    = np.int64(args.n_train_ics)
    save_dict["n_val_ics"]      = np.int64(args.n_val_ics)
    save_dict["n_test_ics"]     = np.int64(args.n_test_ics)

    np.savez(args.out, **save_dict)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
