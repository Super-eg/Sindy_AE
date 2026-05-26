"""Generate BHH 3-body data using delay embedding of Planet 1's x-coordinate.

This script simulates the Broucke-Hadjidemetriou-Henon (BHH) quasi-periodic orbit,
extracts the x_1 position and velocity of the first planet, and applies Takens-style 
delay embedding to create the input format required by train.py.

    input[t] = [x_1(t), x_1(t-τ), ..., x_1(t-(d-1)τ)]
    dx[t]    = [v_{x1}(t), v_{x1}(t-τ), ..., v_{x1}(t-(d-1)τ)]

Run:
    python3 generate_3body_delay.py --delay_dim 5 --delay_steps 10
"""
import argparse
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp

# ── Physical Parameters ───────────────────────────────────────────────────────
G  = 1.0
m1 = 1.0
m2 = 1.0
m3 = 1.0

T_PERIOD = 9.1993  # Internal Period

_R2_0 = np.array([-1.21708465, 0.0])
_V2_0 = np.array([ 0.0, -0.89338778])
_R1_0 = np.array([ 1.10854233, 0.0])
_V1_0 = np.array([ 0.0, -0.28857029])
_R3_0 = np.array([ 0.10854233, 0.0])
_V3_0 = np.array([ 0.0,  1.18195807])

# ── RHS for 3-Body ────────────────────────────────────────────────────────────
def _rhs(t, s):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    r12_3 = ((x2-x1)**2 + (y2-y1)**2) ** 1.5
    r13_3 = ((x3-x1)**2 + (y3-y1)**2) ** 1.5
    r23_3 = ((x3-x2)**2 + (y3-y2)**2) ** 1.5
    return [vx1, vy1, vx2, vy2, vx3, vy3,
            G*m2*(x2-x1)/r12_3 + G*m3*(x3-x1)/r13_3,
            G*m2*(y2-y1)/r12_3 + G*m3*(y3-y1)/r13_3,
            G*m1*(x1-x2)/r12_3 + G*m3*(x3-x2)/r23_3,
            G*m1*(y1-y2)/r12_3 + G*m3*(y3-y2)/r23_3,
            G*m1*(x1-x3)/r13_3 + G*m2*(x2-x3)/r23_3,
            G*m1*(y1-y3)/r13_3 + G*m2*(y2-y3)/r23_3]

def build_delay_matrices(x_raw, v_raw, delay_dim, delay_steps, noise_strength=0.0):
    """
    Constructs the delay embedded matrices from 1D raw coordinate sequences.
    """
    n_steps = len(x_raw)
    min_idx = (delay_dim - 1) * delay_steps
    n_valid = n_steps - min_idx
    
    if n_valid <= 0:
        raise ValueError(f"Time series too short. Need at least {min_idx + 1} steps, got {n_steps}.")

    x_delay = np.zeros((n_valid, delay_dim), dtype=np.float32)
    dx_delay = np.zeros((n_valid, delay_dim), dtype=np.float32)
    z_true = np.zeros((n_valid, 2), dtype=np.float32)

    for k in range(delay_dim):
        s = min_idx - k * delay_steps
        e = s + n_valid
        x_delay[:, k]  = x_raw[s:e]
        dx_delay[:, k] = v_raw[s:e]

    # Ground truth reference (x_1, v_x1) at the valid time steps
    z_true[:, 0] = x_raw[min_idx : min_idx + n_valid]
    z_true[:, 1] = v_raw[min_idx : min_idx + n_valid]

    if noise_strength > 0:
        x_delay  += (noise_strength * np.random.randn(*x_delay.shape)).astype(np.float32)
        dx_delay += (noise_strength * np.random.randn(*dx_delay.shape)).astype(np.float32)

    return {'x': x_delay, 'dx': dx_delay, 'z': z_true}

def main():
    parser = argparse.ArgumentParser(description="Generate 3-Body BHH delay embedding data.")
    parser.add_argument("--train_periods",  type=int,   default=50)
    parser.add_argument("--val_periods",    type=int,   default=10)
    parser.add_argument("--test_periods",   type=int,   default=10)
    parser.add_argument("--dt",             type=float, default=0.01)
    parser.add_argument("--delay_dim",      type=int,   default=5,
                        help="Embedding dimension d (default: 5)")
    parser.add_argument("--delay_steps",    type=int,   default=5,
                        help="Gap between delays in time-steps (default: 5)")
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--out",            default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"delay_3body_x1_d{args.delay_dim}_{args.delay_steps}.npz"

    # 1. Total simulation time
    n_total_periods = args.train_periods + args.val_periods + args.test_periods
    t_total = n_total_periods * T_PERIOD
    t_eval = np.arange(0, t_total, args.dt)

    print(f"Simulating BHH Orbit for {n_total_periods} periods (Total t={t_total:.2f})...")
    s0 = np.concatenate([_R1_0, _R2_0, _R3_0, _V1_0, _V2_0, _V3_0])
    sol = solve_ivp(_rhs, [0, t_total], s0, method='DOP853', t_eval=t_eval, rtol=1e-12, atol=1e-14)
    
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Extract ONLY Planet 1's x coordinate and velocity
    x1_all = sol.y[0]  # x1
    v1_all = sol.y[6]  # vx1

    # 2. Compute slice indices based on periods
    idx_train = int((args.train_periods * T_PERIOD) / args.dt)
    idx_val   = idx_train + int((args.val_periods * T_PERIOD) / args.dt)

    # To ensure delay_dim doesn't drop the first few points of val/test sets,
    # we overlap the raw array slices by the exact margin required.
    margin = (args.delay_dim - 1) * args.delay_steps

    train_raw_x = x1_all[:idx_train]
    train_raw_v = v1_all[:idx_train]

    val_raw_x = x1_all[idx_train - margin : idx_val]
    val_raw_v = v1_all[idx_train - margin : idx_val]

    test_raw_x = x1_all[idx_val - margin :]
    test_raw_v = v1_all[idx_val - margin :]

    # 3. Build Delay Matrices
    print("Building delay embedding matrices...")
    train = build_delay_matrices(train_raw_x, train_raw_v, args.delay_dim, args.delay_steps, args.noise_strength)
    val   = build_delay_matrices(val_raw_x,   val_raw_v,   args.delay_dim, args.delay_steps, args.noise_strength)
    test  = build_delay_matrices(test_raw_x,  test_raw_v,  args.delay_dim, args.delay_steps, args.noise_strength)

    print("=" * 60)
    print("3-Body (BHH) Data Generation — Delay embedding of x_1 coordinate")
    print(f"  Embedding dim  : {args.delay_dim}  (d)")
    print(f"  Delay step     : {args.delay_steps} steps  (τ = {args.delay_steps*args.dt:.3f} s)")
    t0_train = 0.0
    t1_train = args.train_periods * T_PERIOD
    t0_val   = t1_train
    t1_val   = t0_val + args.val_periods * T_PERIOD
    t0_test  = t1_val
    t1_test  = t0_test + args.test_periods * T_PERIOD
    print(f"  train          : {t0_train:.1f} s ~ {t1_train:.1f} s  ({train['x'].shape[0]:,} samples)")
    print(f"  val            : {t0_val:.1f} s ~ {t1_val:.1f} s  ({val['x'].shape[0]:,} samples)")
    print(f"  test           : {t0_test:.1f} s ~ {t1_test:.1f} s  ({test['x'].shape[0]:,} samples)")
    print(f"  Output         : {args.out}")
    print("=" * 60)

    # 4. Save to format matching train.py expectations
    save_dict = {f"train_{k}": v for k, v in train.items()}
    save_dict.update({f"val_{k}": v for k, v in val.items()})
    save_dict.update({f"test_{k}": v for k, v in test.items()})
    
    save_dict["t_start"]        = np.float64(0.0)
    save_dict["t_end"]          = np.float64(t_total)
    save_dict["dt"]             = np.float64(args.dt)
    save_dict["delay_dim"]      = np.int64(args.delay_dim)
    save_dict["delay_steps"]    = np.int64(args.delay_steps)
    save_dict["tau"]            = np.float64(args.delay_steps * args.dt)
    save_dict["noise_strength"] = np.float64(args.noise_strength)

    # 由於是沿著單一軌道做時間切割，這裡將 n_ics 標記為 1
    save_dict["n_train_ics"]    = np.int64(1)
    save_dict["n_val_ics"]      = np.int64(1)
    save_dict["n_test_ics"]     = np.int64(1)

    # Per-split relative time arrays (starting at 0 within each split) and
    # absolute offsets so downstream scripts can recover global time:
    #   global_t_split = t_<split>_offset + <split>_t
    save_dict["train_t"] = np.arange(train["x"].shape[0], dtype=np.float64) * args.dt
    save_dict["val_t"]   = np.arange(val["x"].shape[0],   dtype=np.float64) * args.dt
    save_dict["test_t"]  = np.arange(test["x"].shape[0],  dtype=np.float64) * args.dt
    save_dict["t_train_offset"] = np.float64(0.0)
    save_dict["t_val_offset"]   = np.float64(args.train_periods * T_PERIOD)
    save_dict["t_test_offset"]  = np.float64((args.train_periods + args.val_periods) * T_PERIOD)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.savez(args.out, **save_dict)
    print(f"\nSuccessfully saved to {args.out}")

if __name__ == '__main__':
    main()