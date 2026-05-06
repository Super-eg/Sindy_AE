"""Von Kármán vortex shedding data generation — pure numpy + scipy.

Two canonical low-dimensional ODE models for vortex shedding behind a bluff body:

1. Hopf normal form (Stuart-Landau), 2D 1st-order:
       ż₀ = mu·z₀ - omega·z₁ - (z₀² + z₁²)·z₀
       ż₁ = omega·z₀ + mu·z₁ - (z₀² + z₁²)·z₁
   Stable circular limit cycle of radius √mu (mu>0), angular speed omega.

2. Noack mean-field model (Noack et al. 2003), 3D 1st-order:
       ȧ₁ = mu·a₁ - omega·a₂ - alpha·a₁·a₃
       ȧ₂ = omega·a₁ + mu·a₂ - alpha·a₂·a₃
       ȧ₃ = -lam·(a₃ - (a₁² + a₂²))
   Includes a slow shift mode a₃ representing mean-flow distortion.
   On the limit cycle, a₃ → mu/alpha and (a₁,a₂) trace a circle of radius √(mu/alpha).
"""
import numpy as np
from scipy.integrate import odeint

from sindyae.sindy_library import library_size


# ─── Hopf normal form (2D) ────────────────────────────────────────────────────

def simulate_hopf(z0, t, mu=1.0, omega=2 * np.pi):
    """Integrate the Hopf normal form from z0 = [z₀_init, z₁_init].

    Returns
    -------
    z  : (n_steps, 2)
    dz : (n_steps, 2)
    """
    def f(z, _t):
        r2 = z[0] ** 2 + z[1] ** 2
        return [mu * z[0] - omega * z[1] - r2 * z[0],
                omega * z[0] + mu * z[1] - r2 * z[1]]
    z = odeint(f, z0, t)
    dz = np.zeros_like(z)
    for i in range(t.size):
        dz[i] = f(z[i], 0)
    return z, dz


def hopf_coefficients(normalization, poly_order=3, mu=1.0, omega=2 * np.pi):
    """Ground-truth SINDy coefficient matrix Xi for the Hopf normal form.

    With z_norm = z_phys * normalization (assumed isotropic n0 == n1 == n):
        ż₀_norm =  mu·z₀_norm - omega·z₁_norm - (1/n²)·(z₀_norm³ + z₀_norm·z₁_norm²)
        ż₁_norm =  omega·z₀_norm + mu·z₁_norm - (1/n²)·(z₀_norm²·z₁_norm + z₁_norm³)

    Library index layout for n=2, poly_order=3 (10 terms):
        0:1  1:z₀  2:z₁  3:z₀²  4:z₀z₁  5:z₁²  6:z₀³  7:z₀²z₁  8:z₀z₁²  9:z₁³
    """
    n0, n1 = float(normalization[0]), float(normalization[1])
    Xi = np.zeros((library_size(2, poly_order), 2))
    # ż₀
    Xi[1, 0] = mu                       # z₀
    Xi[2, 0] = -omega * (n0 / n1)       # z₁  (anisotropic scale)
    Xi[6, 0] = -1.0 / (n0 ** 2)         # z₀³
    Xi[8, 0] = -1.0 / (n1 ** 2)         # z₀z₁²
    # ż₁
    Xi[1, 1] = omega * (n1 / n0)        # z₀
    Xi[2, 1] = mu                       # z₁
    Xi[7, 1] = -1.0 / (n0 ** 2)         # z₀²z₁
    Xi[9, 1] = -1.0 / (n1 ** 2)         # z₁³
    return Xi


# ─── Noack mean-field (3D) ────────────────────────────────────────────────────

def simulate_noack(a0, t, mu=0.1, omega=1.0, lam=10.0, alpha=1.0):
    """Integrate the Noack mean-field model from a0 = [a₁, a₂, a₃].

    Returns
    -------
    a  : (n_steps, 3)
    da : (n_steps, 3)
    """
    def f(a, _t):
        return [mu * a[0] - omega * a[1] - alpha * a[0] * a[2],
                omega * a[0] + mu * a[1] - alpha * a[1] * a[2],
                -lam * (a[2] - (a[0] ** 2 + a[1] ** 2))]
    a = odeint(f, a0, t)
    da = np.zeros_like(a)
    for i in range(t.size):
        da[i] = f(a[i], 0)
    return a, da


def noack_coefficients(normalization, poly_order=2, mu=0.1, omega=1.0,
                       lam=10.0, alpha=1.0):
    """Ground-truth SINDy coefficient matrix Xi for the Noack model.

    With a_norm = a_phys * normalization (assumed isotropic n0 == n1 == n2 == n):
        ȧ₁_norm =  mu·a₁ - omega·a₂ - (alpha/n)·a₁·a₃
        ȧ₂_norm =  omega·a₁ + mu·a₂ - (alpha/n)·a₂·a₃
        ȧ₃_norm = -lam·a₃ + (lam/n)·(a₁² + a₂²)

    Library index layout for n=3, poly_order=2 (10 terms):
        0:1  1:a₁  2:a₂  3:a₃  4:a₁²  5:a₁a₂  6:a₁a₃  7:a₂²  8:a₂a₃  9:a₃²
    """
    n0 = float(normalization[0])
    n1 = float(normalization[1])
    n2 = float(normalization[2])
    Xi = np.zeros((library_size(3, poly_order), 3))
    # ȧ₁
    Xi[1, 0] = mu
    Xi[2, 0] = -omega * (n0 / n1)
    Xi[6, 0] = -alpha / n2                   # a₁a₃
    # ȧ₂
    Xi[1, 1] = omega * (n1 / n0)
    Xi[2, 1] = mu
    Xi[8, 1] = -alpha / n2                   # a₂a₃
    # ȧ₃
    Xi[3, 2] = -lam
    Xi[4, 2] = lam * n2 / (n0 ** 2)          # a₁²
    Xi[7, 2] = lam * n2 / (n1 ** 2)          # a₂²
    return Xi


# ─── unified data generation ──────────────────────────────────────────────────

def generate_data(model, ics, t, normalization=None,
                  mu=None, omega=None, lam=10.0, alpha=1.0):
    """Simulate Von Kármán trajectories under the chosen low-D model.

    Parameters
    ----------
    model         : "hopf" or "noack"
    ics           : (n_ics, d) initial conditions in physical coordinates (d=2 or 3)
    t             : (n_steps,) time grid
    normalization : (d,) per-dim scale factors; z_norm = z_phys * normalization
    mu, omega     : oscillation parameters
    lam, alpha    : Noack-only parameters (ignored for hopf)

    Returns
    -------
    dict with keys: t, x, dx, z, dz, sindy_coefficients, normalization
        x, dx : normalized state trajectories, shape (n_ics, n_steps, d), float32
        z, dz : same as x, dx (kept for API parity with vander/lorenz)
    """
    if model == "hopf":
        d = 2
        if mu is None:    mu = 1.0
        if omega is None: omega = 2 * np.pi
        sim = lambda z0_: simulate_hopf(z0_, t, mu=mu, omega=omega)
        coeffs_fn = lambda norm: hopf_coefficients(norm, poly_order=3,
                                                    mu=mu, omega=omega)
    elif model == "noack":
        d = 3
        if mu is None:    mu = 0.1
        if omega is None: omega = 1.0
        sim = lambda a0_: simulate_noack(a0_, t, mu=mu, omega=omega,
                                          lam=lam, alpha=alpha)
        coeffs_fn = lambda norm: noack_coefficients(norm, poly_order=2,
                                                     mu=mu, omega=omega,
                                                     lam=lam, alpha=alpha)
    else:
        raise ValueError(f"Unknown model '{model}', expected 'hopf' or 'noack'")

    if normalization is None:
        normalization = np.ones(d)
    normalization = np.asarray(normalization, dtype=np.float64)
    if normalization.shape != (d,):
        raise ValueError(
            f"normalization must have shape ({d},) for model '{model}', "
            f"got {normalization.shape}"
        )

    n_ics = ics.shape[0]
    n_steps = t.size

    z_phys  = np.zeros((n_ics, n_steps, d))
    dz_phys = np.zeros_like(z_phys)
    for i in range(n_ics):
        z_phys[i], dz_phys[i] = sim(ics[i])

    z  = z_phys  * normalization
    dz = dz_phys * normalization

    sindy_coefficients = coeffs_fn(normalization)

    return {
        't': t,
        'x':  z.astype(np.float32),
        'dx': dz.astype(np.float32),
        'z':  z,
        'dz': dz,
        'sindy_coefficients': sindy_coefficients.astype(np.float32),
        'normalization': normalization.astype(np.float32),
    }
