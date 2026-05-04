"""Generate Van der Pol data using delay embedding of the x-coordinate and save to .npz.

Instead of projecting both state variables to high-dim space, we observe
only x(t) and apply Takens-style delay embedding:

    input[t] = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]
    dx[t]    = [ẋ(t), ẋ(t-τ), ẋ(t-2τ), ..., ẋ(t-(d-1)τ)]

where τ = delay_steps × dt and x is the normalized position (×1/2).

By Takens' theorem, for d ≥ 2·dim(attractor)+1, the delay vector is
diffeomorphic to the original attractor.  For the Van der Pol limit cycle
(dimension 1), d ≥ 3 suffices; the default d=10 gives plenty of margin.

Run from examples/vander/:
    python3 generate_delay.py

Output .npz is passed to examples/train.py via
    --data vander/delay_xcoordinate_d<dim>_<steps>.npz
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from example_vander import simulate_vanderpol


def get_vanderpol_data_delay_x(n_ics, t, delay_dim=10, delay_steps=5,
                                noise_strength=0, mu=2.0):
    """Generate delay-embedded Van der Pol data from the x-coordinate only.

    For each IC and each valid time step t_j, the input vector is:
        x[j]  = [x_norm(t_j), x_norm(t_j - τ), ..., x_norm(t_j - (d-1)τ)]
        dx[j] = [ẋ_norm(t_j), ẋ_norm(t_j - τ), ..., ẋ_norm(t_j - (d-1)τ)]

    where τ = delay_steps × dt and x_norm = x × (1/2).

    ICs are sampled uniformly in [-3, 3] × [-3, 3] with the near-origin
    disk (||z|| < 0.1) excluded to avoid the unstable fixed point.

    Parameters
    ----------
    n_ics         : number of initial conditions (trajectories)
    t             : time array (np.arange)
    delay_dim     : embedding dimension d  (default 10)
    delay_steps   : gap between delays in time-steps; τ = delay_steps × dt
    noise_strength: Gaussian noise added to x and dx
    mu            : Van der Pol damping parameter (default 2.0)

    Returns
    -------
    dict with keys:
        x   : (n_ics * n_valid, delay_dim)  float32
        dx  : (n_ics * n_valid, delay_dim)  float32
        z   : (n_ics * n_valid, 2)          float32 — true normalized state [x, v]×(1/2)
        t   : (n_valid,) valid time points (first (d-1)*delay_steps dropped)
    """
    normalization = 1.0 / 2.0   # x limit cycle amplitude ≈ 2 → normalize to [-1, 1]

    n_steps = t.size
    min_idx = (delay_dim - 1) * delay_steps     # first valid time index
    n_valid = n_steps - min_idx

    if n_valid <= 0:
        raise ValueError(
            f"Time series too short for delay_dim={delay_dim}, "
            f"delay_steps={delay_steps}. Need at least "
            f"{min_idx + 1} steps, got {n_steps}."
        )

    # Sample ICs uniformly in [-3,3]×[-3,3], exclude near-origin disk
    ics = np.zeros((n_ics, 2))
    count = 0
    rng = np.random.RandomState()   # uses current global seed
    while count < n_ics:
        batch = rng.uniform(-3.0, 3.0, size=(n_ics * 4, 2))
        valid = batch[np.linalg.norm(batch, axis=1) >= 0.1]
        take = min(len(valid), n_ics - count)
        ics[count:count + take] = valid[:take]
        count += take

    x_all  = np.zeros((n_ics, n_valid, delay_dim), dtype=np.float32)
    dx_all = np.zeros_like(x_all)
    z_all  = np.zeros((n_ics, n_valid, 2), dtype=np.float32)

    for i in range(n_ics):
        z, dz = simulate_vanderpol(ics[i], t, mu=mu)

        x_norm  = z[:, 0]  * normalization   # shape (n_steps,)
        dx_norm = dz[:, 0] * normalization   # ẋ / 2

        for k in range(delay_dim):
            s = min_idx - k * delay_steps
            e = s + n_valid
            x_all[i, :, k]  = x_norm[s:e]
            dx_all[i, :, k] = dx_norm[s:e]

        # True normalized state [x, v] at valid time points
        z_all[i] = (z[min_idx:min_idx + n_valid] * normalization).astype(np.float32)

    x_flat  = x_all.reshape(-1, delay_dim)
    dx_flat = dx_all.reshape(-1, delay_dim)
    z_flat  = z_all.reshape(-1, 2)

    if noise_strength > 0:
        x_flat  = x_flat  + (noise_strength * np.random.randn(*x_flat.shape)).astype(np.float32)
        dx_flat = dx_flat + (noise_strength * np.random.randn(*dx_flat.shape)).astype(np.float32)

    return {'x': x_flat, 'dx': dx_flat, 'z': z_flat, 't': t[min_idx:]}


def main():
    parser = argparse.ArgumentParser(
        description="Generate Van der Pol data via delay embedding of x-coordinate and save to .npz"
    )
    parser.add_argument("--n_train_ics",    type=int,   default=1024)
    parser.add_argument("--n_val_ics",      type=int,   default=20)
    parser.add_argument("--n_test_ics",     type=int,   default=20)
    parser.add_argument("--noise_strength", type=float, default=1e-6)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--t_start",        type=float, default=0.0)
    parser.add_argument("--t_end",          type=float, default=20.0)
    parser.add_argument("--dt",             type=float, default=0.02)
    parser.add_argument("--delay_dim",      type=int,   default=10,
                        help="Embedding dimension d (default: 10)")
    parser.add_argument("--delay_steps",    type=int,   default=5,
                        help="Gap between delays in time-steps; "
                             "τ = delay_steps × dt (default: 5 → τ=0.1s)")
    parser.add_argument("--mu",             type=float, default=2.0,
                        help="Van der Pol damping parameter (default: 2.0)")
    parser.add_argument("--out",            default=None,
                        help="Output .npz file path "
                             "(default: delay_xcoordinate_d<dim>_<steps>.npz)")
    args = parser.parse_args()

    if args.out is None:
        args.out = f"delay_xcoordinate_d{args.delay_dim}_{args.delay_steps}.npz"

    np.random.seed(args.seed)

    t   = np.arange(args.t_start, args.t_end, args.dt)
    tau = args.delay_steps * args.dt

    min_idx = (args.delay_dim - 1) * args.delay_steps
    n_valid = t.size - min_idx

    print("=" * 60)
    print("Van der Pol data generation — delay embedding of x-coordinate")
    print(f"  mu             : {args.mu}")
    print(f"  Embedding dim  : {args.delay_dim}  (d)")
    print(f"  Delay step     : {args.delay_steps} steps  (τ = {tau:.3f} s)")
    print(f"  Time range     : {args.t_start} → {args.t_end}  (dt={args.dt})")
    print(f"  Steps dropped  : {min_idx}  (first (d-1)×delay_steps)")
    print(f"  Valid steps/IC : {n_valid}")
    print(f"  Normalization  : x × 1/2")
    print(f"  Train ICs      : {args.n_train_ics}  → {args.n_train_ics * n_valid:,} samples")
    print(f"  Val ICs        : {args.n_val_ics}")
    print(f"  Test ICs       : {args.n_test_ics}")
    print(f"  Noise          : {args.noise_strength}")
    print(f"  Output         : {args.out}")
    print("=" * 60)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Generating training data ({args.n_train_ics} ICs) …")
    train = get_vanderpol_data_delay_x(
        args.n_train_ics, t,
        delay_dim=args.delay_dim, delay_steps=args.delay_steps,
        noise_strength=args.noise_strength, mu=args.mu,
    )
    print(f"  → train_x shape: {train['x'].shape}")

    print(f"Generating validation data ({args.n_val_ics} ICs) …")
    val = get_vanderpol_data_delay_x(
        args.n_val_ics, t,
        delay_dim=args.delay_dim, delay_steps=args.delay_steps,
        noise_strength=args.noise_strength, mu=args.mu,
    )
    print(f"  → val_x shape: {val['x'].shape}")

    print(f"Generating test data ({args.n_test_ics} ICs) …")
    test = get_vanderpol_data_delay_x(
        args.n_test_ics, t,
        delay_dim=args.delay_dim, delay_steps=args.delay_steps,
        noise_strength=args.noise_strength, mu=args.mu,
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
    save_dict["mu"]             = np.float64(args.mu)

    np.savez(args.out, **save_dict)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
