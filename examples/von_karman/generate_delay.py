"""Generate Von Kármán vortex shedding data via Takens delay embedding.

Two underlying low-D models are supported via --model:
    - hopf  : 2D Hopf normal form (Stuart-Landau), latent_dim=2
    - noack : 3D Noack mean-field (a₁, a₂, a₃), latent_dim=3

We observe only the first physical coordinate (z₀ for hopf, a₁ for noack)
and apply Takens-style delay embedding:

    input[t] = [y(t), y(t-τ), y(t-2τ), ..., y(t-(d-1)τ)]
    dx[t]    = [ẏ(t), ẏ(t-τ), ẏ(t-2τ), ..., ẏ(t-(d-1)τ)]

where τ = delay_steps × dt and y is the normalized observation.

By Takens' theorem, for d ≥ 2·dim(attractor)+1 the delay vector is
diffeomorphic to the original attractor. The default d=10 gives generous
margin (5 for hopf, 7 for noack).

Run from examples/von_karman/:
    python3 generate_delay.py                         # hopf default
    python3 generate_delay.py --model noack           # Noack mean-field

Output .npz is passed to examples/train.py via
    --data von_karman/delay_xcoordinate_<model>_d<dim>_<steps>.npz
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from example_von_karman import simulate_hopf, simulate_noack


def _sample_ics_hopf(n_ics, mu, rng):
    """Sample ICs in an annulus around the limit-cycle radius √mu, avoid origin."""
    r_lc = float(np.sqrt(mu))
    r_outer = 2.0 * r_lc
    ics = np.zeros((n_ics, 2))
    count = 0
    while count < n_ics:
        batch = rng.uniform(-r_outer, r_outer, size=(n_ics * 4, 2))
        radii = np.linalg.norm(batch, axis=1)
        valid = batch[(radii >= 0.1) & (radii <= r_outer)]
        take = min(len(valid), n_ics - count)
        ics[count:count + take] = valid[:take]
        count += take
    return ics


def _sample_ics_noack(n_ics, mu, alpha, rng):
    """Sample ICs in a 3D box covering both transient and limit cycle, avoid origin."""
    r_lc = float(np.sqrt(mu / alpha))
    extent = max(1.5, 2.0 * r_lc)
    ics = np.zeros((n_ics, 3))
    count = 0
    while count < n_ics:
        batch = rng.uniform(-extent, extent, size=(n_ics * 4, 3))
        radii = np.linalg.norm(batch, axis=1)
        valid = batch[radii >= 0.1]
        take = min(len(valid), n_ics - count)
        ics[count:count + take] = valid[:take]
        count += take
    return ics


def _biot_savart_obs(z_traj, t, omega,
                     x_probe=10.0, y_probe=0.0,
                     x_shed=0.60, u_x=1.5, cy_half=0.65,
                     gamma=1.0, delta2=0.09, x_max=13.5):
    """Transverse velocity at (x_probe, y_probe) from the Kármán vortex street.

    Phase θ(t) = atan2(z₁, z₀) increases at rate ω for both Hopf and Noack.
    Each upward crossing of k·π sheds a vortex:
        k even  → upper (CCW, sign=+1, cy=+cy_half)
        k odd   → lower (CW,  sign=−1, cy=−cy_half)
    Vortices advect at u_x; Biot-Savart gives vy at the probe.
    """
    phase = np.unwrap(np.arctan2(z_traj[:, 1], z_traj[:, 0]))
    dt = float(t[1] - t[0])

    births, cys, signs = [], [], []
    for i in range(1, len(t)):
        if phase[i] <= phase[i - 1]:
            continue
        k_lo = int(np.floor(phase[i - 1] / np.pi))
        k_hi = int(np.floor(phase[i]     / np.pi))
        for k in range(k_lo + 1, k_hi + 1):
            frac = (k * np.pi - phase[i - 1]) / (phase[i] - phase[i - 1])
            births.append(t[i - 1] + frac * dt)
            sign = 1 if k % 2 == 0 else -1
            signs.append(sign)
            cys.append(cy_half * sign)

    if not births:
        return np.zeros(len(t))

    t_shed = np.array(births, dtype=np.float64)
    sign_a = np.array(signs,  dtype=np.float64)
    cy_a   = np.array(cys,    dtype=np.float64)

    # Vectorised Biot-Savart: tau[i,j] = t[i] - t_shed[j]
    tau    = t[:, None] - t_shed[None, :]       # (n_steps, n_vortices)
    cx     = x_shed + u_x * tau
    active = (tau >= 0) & (cx <= x_max)

    dx      = x_probe - cx
    dy      = y_probe - cy_a[None, :]
    r2      = dx ** 2 + dy ** 2 + delta2
    contrib = -sign_a[None, :] * (gamma / (2 * np.pi)) * dx / r2
    contrib[~active] = 0.0

    return contrib.sum(axis=1)                  # (n_steps,)


def get_data_delay(model, n_ics, t, delay_dim, delay_steps, normalization,
                   noise_strength, rng, mu, omega, lam=10.0, alpha=1.0,
                   x_probe=None, y_probe=0.0, obs_norm=1.0):
    """Generate delay-embedded Von Kármán data from the y-coordinate only.

    For each IC and each valid time step t_j, the input vector is:
        x[j]  = [y_norm(t_j), y_norm(t_j - τ), ..., y_norm(t_j - (d-1)τ)]
        dx[j] = [ẏ_norm(t_j), ẏ_norm(t_j - τ), ..., ẏ_norm(t_j - (d-1)τ)]

    where y is the first physical coordinate (z₀ or a₁) and y_norm = y * norm0.

    Returns
    -------
    dict with keys:
        x   : (n_ics * n_valid, delay_dim)        float32
        dx  : (n_ics * n_valid, delay_dim)        float32
        z   : (n_ics * n_valid, state_dim)        float32 — true normalized state
        t   : (n_valid,) valid time points (first (d-1)*delay_steps dropped)
    """
    n_steps = t.size
    min_idx = (delay_dim - 1) * delay_steps
    n_valid = n_steps - min_idx

    if n_valid <= 0:
        raise ValueError(
            f"Time series too short for delay_dim={delay_dim}, "
            f"delay_steps={delay_steps}. Need at least "
            f"{min_idx + 1} steps, got {n_steps}."
        )

    if model == "hopf":
        ics = _sample_ics_hopf(n_ics, mu, rng)
        state_dim = 2
        sim = lambda z0_: simulate_hopf(z0_, t, mu=mu, omega=omega)
    else:  # noack
        ics = _sample_ics_noack(n_ics, mu, alpha, rng)
        state_dim = 3
        sim = lambda a0_: simulate_noack(a0_, t, mu=mu, omega=omega,
                                          lam=lam, alpha=alpha)

    x_all  = np.zeros((n_ics, n_valid, delay_dim), dtype=np.float32)
    dx_all = np.zeros_like(x_all)
    z_all  = np.zeros((n_ics, n_valid, state_dim), dtype=np.float32)

    for i in range(n_ics):
        z_phys, dz_phys = sim(ics[i])
        z_norm  = z_phys  * normalization        # broadcasts (n_steps, d) * (d,)
        dz_norm = dz_phys * normalization

        if x_probe is not None:
            y_raw = _biot_savart_obs(z_phys, t, omega,
                                     x_probe=x_probe, y_probe=y_probe)
            y     = (y_raw / obs_norm).astype(np.float32)
            y_dot = np.gradient(y, t[1] - t[0]).astype(np.float32)
        else:
            y     = z_norm[:, 0]    # observe first coordinate
            y_dot = dz_norm[:, 0]

        for k in range(delay_dim):
            s = min_idx - k * delay_steps
            e = s + n_valid
            x_all[i, :, k]  = y[s:e]
            dx_all[i, :, k] = y_dot[s:e]

        z_all[i] = z_norm[min_idx:min_idx + n_valid].astype(np.float32)

    x_flat  = x_all.reshape(-1, delay_dim)
    dx_flat = dx_all.reshape(-1, delay_dim)
    z_flat  = z_all.reshape(-1, state_dim)

    if noise_strength > 0:
        x_flat  = x_flat  + (noise_strength * rng.randn(*x_flat.shape)).astype(np.float32)
        dx_flat = dx_flat + (noise_strength * rng.randn(*dx_flat.shape)).astype(np.float32)

    return {'x': x_flat, 'dx': dx_flat, 'z': z_flat, 't': t[min_idx:]}


def main():
    parser = argparse.ArgumentParser(
        description="Generate Von Kármán vortex shedding data via delay embedding"
    )
    parser.add_argument("--model",          choices=("hopf", "noack"),
                        default="hopf",
                        help="Underlying low-D model (default: hopf)")
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
    parser.add_argument("--mu",             type=float, default=None,
                        help="Oscillation growth rate. Hopf default 1.0, Noack default 0.1.")
    parser.add_argument("--omega",          type=float, default=None,
                        help="Angular frequency. Hopf default 2π, Noack default 1.0.")
    parser.add_argument("--lam",            type=float, default=10.0,
                        help="Noack shift-mode time scale (default: 10.0)")
    parser.add_argument("--alpha",          type=float, default=1.0,
                        help="Noack nonlinear coupling (default: 1.0)")
    parser.add_argument("--x_probe",  type=float, default=10.0,
                        help="Probe x-coordinate for Biot-Savart observation "
                             "(default: 10.0; set to -1 to use z₀ directly)")
    parser.add_argument("--y_probe",  type=float, default=0.0,
                        help="Probe y-coordinate (default: 0.0, centre-line)")
    parser.add_argument("--out",            default=None,
                        help="Output .npz file path "
                             "(default: delay_xcoordinate_<model>_d<dim>_<steps>.npz)")
    args = parser.parse_args()

    if args.mu is None:
        args.mu = 1.0 if args.model == "hopf" else 0.1
    if args.omega is None:
        args.omega = 2 * np.pi if args.model == "hopf" else 1.0

    use_biot = args.x_probe >= 0
    obs_norm = 1.0
    if use_biot:
        # Calibrate: run one limit-cycle trajectory and measure peak |vy|
        t_cal = np.arange(args.t_start, args.t_end, args.dt)
        if args.model == "hopf":
            z0_cal = np.array([float(np.sqrt(args.mu)), 0.0])
            z_cal, _ = simulate_hopf(z0_cal, t_cal, mu=args.mu, omega=args.omega)
        else:
            r_lc_cal = float(np.sqrt(args.mu / args.alpha))
            z0_cal = np.array([r_lc_cal, 0.0, 0.0])
            z_cal, _ = simulate_noack(z0_cal, t_cal, mu=args.mu, omega=args.omega,
                                      lam=args.lam, alpha=args.alpha)
        y_cal    = _biot_savart_obs(z_cal, t_cal, args.omega,
                                    x_probe=args.x_probe, y_probe=args.y_probe)
        obs_norm = max(float(np.max(np.abs(y_cal[len(t_cal) // 2:]))), 1e-6)

    probe_tag = f"_x{args.x_probe:.0f}" if use_biot else ""
    if args.out is None:
        args.out = (f"delay_xcoordinate_{args.model}"
                    f"_d{args.delay_dim}_{args.delay_steps}{probe_tag}.npz")

    if args.model == "hopf":
        norm_scalar = 1.0 / float(np.sqrt(args.mu))
        normalization = np.array([norm_scalar, norm_scalar])
        state_dim = 2
        r_lc = float(np.sqrt(args.mu))
    else:
        norm_scalar = 1.0 / float(np.sqrt(args.mu / args.alpha))
        normalization = np.array([norm_scalar, norm_scalar, norm_scalar])
        state_dim = 3
        r_lc = float(np.sqrt(args.mu / args.alpha))

    rng = np.random.RandomState(args.seed)
    t   = np.arange(args.t_start, args.t_end, args.dt)
    tau = args.delay_steps * args.dt

    min_idx = (args.delay_dim - 1) * args.delay_steps
    n_valid = t.size - min_idx

    print("=" * 60)
    print(f"Von Kármán data generation — delay embedding ({args.model})")
    print(f"  Model           : {args.model}")
    print(f"  mu              : {args.mu}")
    print(f"  omega           : {args.omega}")
    if args.model == "noack":
        print(f"  lam             : {args.lam}")
        print(f"  alpha           : {args.alpha}")
    print(f"  Limit-cycle r   : {r_lc:.4f}")
    print(f"  State dim       : {state_dim}")
    print(f"  Normalization   : {normalization}")
    if use_biot:
        print(f"  Probe (x, y)    : ({args.x_probe}, {args.y_probe})  [Biot-Savart]")
        print(f"  Obs. norm       : {obs_norm:.4f}  (limit-cycle peak vy)")
    else:
        print(f"  Observation     : z₀  (first oscillator coordinate)")
    print(f"  Embedding dim   : {args.delay_dim}  (d)")
    print(f"  Delay step      : {args.delay_steps} steps  (τ = {tau:.3f} s)")
    print(f"  Time range      : {args.t_start} → {args.t_end}  (dt={args.dt})")
    print(f"  Steps dropped   : {min_idx}  (first (d-1)×delay_steps)")
    print(f"  Valid steps/IC  : {n_valid}")
    print(f"  Train ICs       : {args.n_train_ics}  → {args.n_train_ics * n_valid:,} samples")
    print(f"  Val ICs         : {args.n_val_ics}")
    print(f"  Test ICs        : {args.n_test_ics}")
    print(f"  Noise           : {args.noise_strength}")
    print(f"  Output          : {args.out}")
    print("=" * 60)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _probe_kw = dict(x_probe=args.x_probe if use_biot else None,
                     y_probe=args.y_probe, obs_norm=obs_norm)

    print(f"Generating training data ({args.n_train_ics} ICs) …")
    train = get_data_delay(
        args.model, args.n_train_ics, t,
        args.delay_dim, args.delay_steps, normalization,
        args.noise_strength, rng, args.mu, args.omega,
        lam=args.lam, alpha=args.alpha, **_probe_kw,
    )
    print(f"  → train_x shape: {train['x'].shape}")

    print(f"Generating validation data ({args.n_val_ics} ICs) …")
    val = get_data_delay(
        args.model, args.n_val_ics, t,
        args.delay_dim, args.delay_steps, normalization,
        args.noise_strength, rng, args.mu, args.omega,
        lam=args.lam, alpha=args.alpha, **_probe_kw,
    )
    print(f"  → val_x shape: {val['x'].shape}")

    print(f"Generating test data ({args.n_test_ics} ICs) …")
    test = get_data_delay(
        args.model, args.n_test_ics, t,
        args.delay_dim, args.delay_steps, normalization,
        args.noise_strength, rng, args.mu, args.omega,
        lam=args.lam, alpha=args.alpha, **_probe_kw,
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
    save_dict["omega"]          = np.float64(args.omega)
    save_dict["model"]          = np.array(args.model)
    if use_biot:
        save_dict["x_probe"]   = np.float64(args.x_probe)
        save_dict["y_probe"]   = np.float64(args.y_probe)
        save_dict["obs_norm"]  = np.float64(obs_norm)
    if args.model == "noack":
        save_dict["lam"]   = np.float64(args.lam)
        save_dict["alpha"] = np.float64(args.alpha)

    np.savez(args.out, **save_dict)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
