"""Generate pendulum training/validation/test data and save to .npz.

Run from examples/pendulum/:
    python3 generate_pendulum.py

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

from example_pendulum import get_pendulum_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate pendulum data (51×51 Gaussian images) and save to .npz"
    )
    parser.add_argument("--n_train_ics",    type=int,   default=100)
    parser.add_argument("--n_val_ics",      type=int,   default=20)
    parser.add_argument("--n_test_ics",     type=int,   default=20)
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--t_end",          type=float, default=10.0,
                        help="Integration end time (default: 10.0 s)")
    parser.add_argument("--dt",             type=float, default=0.02,
                        help="Time step (default: 0.02 s)")
    parser.add_argument(
        "--out",
        default="pendulum_data.npz",
        help="Output .npz file path (default: pendulum_data.npz)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    t = np.arange(0, args.t_end, args.dt)

    print("=" * 60)
    print("Pendulum data generation — 51×51 Gaussian image (model_order=2)")
    print(f"  Time range : 0 → {args.t_end}  (dt={args.dt}, {t.size} steps)")
    print(f"  Train ICs  : {args.n_train_ics}  →  {args.n_train_ics * t.size:,} samples")
    print(f"  Val ICs    : {args.n_val_ics}  →  {args.n_val_ics * t.size:,} samples")
    print(f"  Test ICs   : {args.n_test_ics}  →  {args.n_test_ics * t.size:,} samples")
    print(f"  Noise      : {args.noise_strength}")
    print(f"  Output     : {args.out}")
    print("=" * 60)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Generating training data ({args.n_train_ics} ICs) …")
    train = get_pendulum_data(args.n_train_ics, t=t, noise_strength=args.noise_strength)
    print(f"  → train_x shape: {train['x'].shape}")

    print(f"Generating validation data ({args.n_val_ics} ICs) …")
    val = get_pendulum_data(args.n_val_ics, t=t, noise_strength=args.noise_strength)
    print(f"  → val_x shape: {val['x'].shape}")

    print(f"Generating test data ({args.n_test_ics} ICs) …")
    test = get_pendulum_data(args.n_test_ics, t=t, noise_strength=args.noise_strength)
    print(f"  → test_x shape: {test['x'].shape}")

    save_dict = {f"train_{k}": v for k, v in train.items()}
    save_dict.update({f"val_{k}": v for k, v in val.items()})
    save_dict.update({f"test_{k}": v for k, v in test.items()})
    save_dict["t_end"]          = np.float64(args.t_end)
    save_dict["dt"]             = np.float64(args.dt)
    save_dict["noise_strength"] = np.float64(args.noise_strength)
    save_dict["n_train_ics"]    = np.int64(args.n_train_ics)
    save_dict["n_val_ics"]      = np.int64(args.n_val_ics)
    save_dict["n_test_ics"]     = np.int64(args.n_test_ics)

    np.savez(args.out, **save_dict)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
