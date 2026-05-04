"""Generate Van der Pol data via Legendre projection and save to .npz.

The 2D Van der Pol state (x, v) is embedded into an n_points-dimensional
observation via:

    y[t] = P0 * x + P1 * v + P2 * x³ + P3 * v³   (nonlinear, default)

where P_i = P_i(y_spatial) are Legendre polynomials on a grid of n_points
points in [-1, 1].  Analytic first derivatives (dx, no ddx) are computed
from the Van der Pol vector field.  The absence of train_ddx causes train.py
to auto-detect model_order=1.

Run from examples/vander/:
    python3 generate_vander.py

Output .npz is passed to examples/train.py via --data vander/vander_legendre.npz.
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from example_vander import generate_vander_data


def get_split(n_ics, t, input_dim, linear, normalization, noise_strength, rng, mu):
    """Simulate n_ics trajectories and return flat (n_ics*n_steps, input_dim) arrays.

    ICs are sampled uniformly in [-3, 3] × [-3, 3] with the near-origin
    disk (||z|| < 0.1) excluded to avoid the unstable fixed point.
    """
    ics = np.zeros((n_ics, 2))
    count = 0
    while count < n_ics:
        batch = rng.uniform(-3.0, 3.0, size=(n_ics * 4, 2))
        valid = batch[np.linalg.norm(batch, axis=1) >= 0.1]
        take = min(len(valid), n_ics - count)
        ics[count:count + take] = valid[:take]
        count += take

    data = generate_vander_data(ics, t, input_dim, linear=linear,
                                normalization=normalization, mu=mu)

    x  = data['x'].reshape(-1, input_dim).astype(np.float32)
    dx = data['dx'].reshape(-1, input_dim).astype(np.float32)
    z  = data['z'].reshape(-1, 2).astype(np.float32)

    if noise_strength > 0:
        x  += (noise_strength * rng.randn(*x.shape)).astype(np.float32)
        dx += (noise_strength * rng.randn(*dx.shape)).astype(np.float32)

    return x, dx, z, data['sindy_coefficients'], data['modes'], data['y_spatial']


def main():
    parser = argparse.ArgumentParser(
        description="Generate Van der Pol data via Legendre projection and save to .npz"
    )
    parser.add_argument("--n_train_ics",    type=int,   default=1024)
    parser.add_argument("--n_val_ics",      type=int,   default=20)
    parser.add_argument("--n_test_ics",     type=int,   default=20)
    parser.add_argument("--noise_strength", type=float, default=1e-6)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--t_start",        type=float, default=0.0)
    parser.add_argument("--t_end",          type=float, default=20.0)
    parser.add_argument("--dt",             type=float, default=0.02)
    parser.add_argument("--input_dim",      type=int,   default=128,
                        help="Observation dimension (Legendre grid size, default: 128)")
    parser.add_argument("--linear",         action="store_true",
                        help="Use linear-only projection (default: nonlinear cubic terms)")
    parser.add_argument("--mu",             type=float, default=2.0,
                        help="Van der Pol damping parameter (default: 2.0)")
    parser.add_argument("--out",            default=None,
                        help="Output .npz file path (default: vander_legendre.npz)")
    args = parser.parse_args()

    if args.out is None:
        mode = "linear" if args.linear else "legendre"
        args.out = f"vander_{mode}.npz"

    rng = np.random.RandomState(args.seed)

    t             = np.arange(args.t_start, args.t_end, args.dt)
    normalization = np.array([1.0 / 3.0, 1.0 / 3.0])
    linear        = args.linear

    print("=" * 60)
    print("Van der Pol data generation — Legendre polynomial projection")
    print(f"  mu                   : {args.mu}")
    print(f"  Input dim (n_points) : {args.input_dim}")
    print(f"  Projection           : {'linear (P0-P1)' if linear else 'nonlinear (P0-P3, cubic)'}")
    print(f"  Time range           : {args.t_start} → {args.t_end}  (dt={args.dt})")
    print(f"  Steps per trajectory : {t.size}")
    print(f"  Normalization        : {normalization}")
    print(f"  IC range             : [-3, 3] × [-3, 3]  (||z|| ≥ 0.1)")
    print(f"  Train ICs            : {args.n_train_ics}  → {args.n_train_ics * t.size:,} samples")
    print(f"  Val ICs              : {args.n_val_ics}")
    print(f"  Test ICs             : {args.n_test_ics}")
    print(f"  Noise                : {args.noise_strength}")
    print(f"  Output               : {args.out}")
    print("=" * 60)

    print(f"Generating training data ({args.n_train_ics} ICs) …")
    train_x, train_dx, train_z, sindy_coefficients, modes, y_spatial = get_split(
        args.n_train_ics, t, args.input_dim, linear, normalization,
        args.noise_strength, rng, args.mu,
    )
    print(f"  → train_x shape: {train_x.shape}")

    print(f"Generating validation data ({args.n_val_ics} ICs) …")
    val_x, val_dx, val_z, _, _, _ = get_split(
        args.n_val_ics, t, args.input_dim, linear, normalization,
        args.noise_strength, rng, args.mu,
    )
    print(f"  → val_x shape: {val_x.shape}")

    print(f"Generating test data ({args.n_test_ics} ICs) …")
    test_x, test_dx, test_z, _, _, _ = get_split(
        args.n_test_ics, t, args.input_dim, linear, normalization,
        args.noise_strength, rng, args.mu,
    )
    print(f"  → test_x shape: {test_x.shape}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.out,
        # ── data splits (no ddx → model_order=1 auto-detected) ───────────
        train_x=train_x,   train_dx=train_dx,   train_z=train_z,
        val_x=val_x,       val_dx=val_dx,       val_z=val_z,
        test_x=test_x,     test_dx=test_dx,     test_z=test_z,
        # ── time grids ────────────────────────────────────────────────────
        train_t=t,         val_t=t,             test_t=t,
        # ── Van der Pol metadata ──────────────────────────────────────────
        sindy_coefficients=sindy_coefficients,
        modes=modes,
        y_spatial=y_spatial,
        normalization=normalization,
        # ── scalar metadata ───────────────────────────────────────────────
        t_start=np.float64(args.t_start),
        t_end=np.float64(args.t_end),
        dt=np.float64(args.dt),
        input_dim=np.int64(args.input_dim),
        noise_strength=np.float64(args.noise_strength),
        n_train_ics=np.int64(args.n_train_ics),
        n_val_ics=np.int64(args.n_val_ics),
        n_test_ics=np.int64(args.n_test_ics),
        linear=np.bool_(linear),
        mu=np.float64(args.mu),
    )
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
