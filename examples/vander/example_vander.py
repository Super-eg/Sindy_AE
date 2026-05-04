"""Van der Pol data generation — pure numpy + scipy, no TensorFlow/PyTorch.

ODE:  d²x/dt² = μ(1-x²)(dx/dt) - x
State space (1st-order 2D):
    dz0/dt = z1
    dz1/dt = μ(1-z0²)*z1 - z0
"""
import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre

from sindyae.sindy_library import library_size


def simulate_vanderpol(z0, t, mu=2.0):
    """Integrate the Van der Pol ODE from z0 = [x0, v0].

    Returns
    -------
    z  : (n_steps, 2)  [x, v]
    dz : (n_steps, 2)  [v, μ(1-x²)v - x]
    """
    f = lambda z, t: [z[1], mu * (1 - z[0] ** 2) * z[1] - z[0]]
    z = odeint(f, z0, t)
    dz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(z[i], 0)
    return z, dz


def vander_coefficients(normalization, poly_order=3, mu=2.0):
    """Ground-truth SINDy coefficient matrix Xi in normalized coordinates.

    With z_norm = z_phys * normalization (element-wise):
        dz0_norm/dt = (n0/n1) * z1_norm
        dz1_norm/dt = -(n1/n0)*z0_norm + mu*z1_norm - (mu/n0²)*z0_norm²*z1_norm

    Library index layout for n=2, poly_order=3:
        0:1  1:z0  2:z1  3:z0²  4:z0z1  5:z1²  6:z0³  7:z0²z1  8:z0z1²  9:z1³
    """
    n0, n1 = float(normalization[0]), float(normalization[1])
    Xi = np.zeros((library_size(2, poly_order), 2))
    Xi[2, 0] = n0 / n1          # z1 → dz0
    Xi[1, 1] = -n1 / n0         # z0 → dz1
    Xi[2, 1] = mu                # z1 → dz1
    Xi[7, 1] = -mu / n0 ** 2    # z0²z1 → dz1
    return Xi


def generate_vander_data(ics, t, n_points, linear=False, normalization=None, mu=2.0):
    """Simulate Van der Pol trajectories and project to n_points-dimensional
    observation space via Legendre polynomial modes.

    Parameters
    ----------
    ics          : (n_ics, 2) initial conditions [x0, v0]
    t            : (n_steps,) time grid
    n_points     : observation dimension
    linear       : if True, use P0*z0 + P1*z1 only (no cubic terms)
    normalization: (2,) scale factors applied as z_norm = z_phys * norm
    mu           : Van der Pol damping parameter

    Returns
    -------
    dict with keys: t, y_spatial, modes, x, dx, z, dz, sindy_coefficients
    """
    n_ics = ics.shape[0]
    n_steps = t.size
    d = 2

    z = np.zeros((n_ics, n_steps, d))
    dz = np.zeros(z.shape)
    for i in range(n_ics):
        z[i], dz[i] = simulate_vanderpol(ics[i], t, mu=mu)

    if normalization is not None:
        z *= normalization   # broadcasts [n_ics, n_steps, 2] * [2]
        dz *= normalization

    n = n_points
    y_spatial = np.linspace(-1, 1, n)

    # 2*d = 4 Legendre modes: P0, P1, P2, P3
    modes = np.zeros((2 * d, n))
    for i in range(2 * d):
        modes[i] = legendre(i)(y_spatial)

    x = np.zeros((n_ics, n_steps, n))
    dx = np.zeros(x.shape)

    for i in range(n_ics):
        for j in range(n_steps):
            x[i, j] = modes[0] * z[i, j, 0] + modes[1] * z[i, j, 1]
            if not linear:
                x[i, j] += (modes[2] * z[i, j, 0] ** 3
                             + modes[3] * z[i, j, 1] ** 3)

            dx[i, j] = modes[0] * dz[i, j, 0] + modes[1] * dz[i, j, 1]
            if not linear:
                dx[i, j] += (modes[2] * 3 * z[i, j, 0] ** 2 * dz[i, j, 0]
                              + modes[3] * 3 * z[i, j, 1] ** 2 * dz[i, j, 1])

    norm = normalization if normalization is not None else np.array([1.0, 1.0])
    sindy_coefficients = vander_coefficients(norm, poly_order=3, mu=mu)

    return {
        't': t, 'y_spatial': y_spatial, 'modes': modes,
        'x': x.astype(np.float32), 'dx': dx.astype(np.float32),
        'z': z, 'dz': dz,
        'sindy_coefficients': sindy_coefficients.astype(np.float32),
    }
