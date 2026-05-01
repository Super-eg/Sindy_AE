"""Pendulum data generation for SINDy-AE (PyTorch port).

Simulates the nonlinear pendulum  d²θ/dt² = -sin(θ)  and maps the angle
to 51×51 Gaussian-blob images (input_dim = 2601, model_order = 2).

Nothing here depends on TensorFlow; it is pure numpy + scipy.
"""
import numpy as np
from scipy.integrate import odeint


def simulate_pendulum(z0, t):
    """Integrate the nonlinear pendulum ODE from z0 = [θ₀, θ̇₀].

    Returns
    -------
    z  : (n_steps, 2)  [θ, θ̇]
    dz : (n_steps, 2)  [θ̇, -sin(θ)]   (= ż, i.e. d/dt of state vector)
    """
    f = lambda z, t: [z[1], -np.sin(z[0])]
    z = odeint(f, z0, t)
    dz = np.array([f(z[j], t[j]) for j in range(len(t))])
    return z, dz


def wrap_to_pi(z):
    """Wrap angle to the interval (−π, π]."""
    z_mod = z % (2 * np.pi)
    return z_mod + (z_mod > np.pi) * (-2 * np.pi)


def pendulum_to_movie(z, dz, n=51):
    """Map pendulum angle trajectories to sequences of n×n Gaussian-blob images.

    The pendulum bob is placed at (cos(θ−π/2), sin(θ−π/2)) on the unit circle,
    and a Gaussian with σ² = 0.05 is drawn on an n×n grid in [−1.5, 1.5]².

    Parameters
    ----------
    z  : (n_ics, n_steps, 2)  [θ, θ̇]
    dz : (n_ics, n_steps, 2)  [θ̇, −sin(θ)]

    Returns
    -------
    x   : (n_ics, n_steps, n, n)  image observations
    dx  : (n_ics, n_steps, n, n)  d(image)/dt
    ddx : (n_ics, n_steps, n, n)  d²(image)/dt²
    """
    n_ics, n_steps, _ = z.shape
    y1, y2 = np.meshgrid(np.linspace(-1.5, 1.5, n), np.linspace(1.5, -1.5, n))

    x   = np.zeros((n_ics, n_steps, n, n))
    dx  = np.zeros_like(x)
    ddx = np.zeros_like(x)

    # Gaussian blob centred at (cos(θ−π/2), sin(θ−π/2))
    create_image = lambda theta: np.exp(
        -((y1 - np.cos(theta - np.pi / 2)) ** 2
          + (y2 - np.sin(theta - np.pi / 2)) ** 2) / 0.05
    )

    # d/dt of the Gaussian exponent  (= d/dt of [-r²/0.05])
    arg_deriv = lambda theta, dtheta: (-1.0 / 0.05) * (
        2 * (y1 - np.cos(theta - np.pi / 2)) * np.sin(theta - np.pi / 2) * dtheta
        + 2 * (y2 - np.sin(theta - np.pi / 2)) * (-np.cos(theta - np.pi / 2)) * dtheta
    )

    # d²/dt² of the Gaussian exponent  (= d/dt of arg_deriv, divided by I)
    arg_deriv2 = lambda theta, dtheta, ddtheta: (-2.0 / 0.05) * (
        np.sin(theta - np.pi / 2) ** 2 * dtheta ** 2
        + (y1 - np.cos(theta - np.pi / 2)) * np.cos(theta - np.pi / 2) * dtheta ** 2
        + (y1 - np.cos(theta - np.pi / 2)) * np.sin(theta - np.pi / 2) * ddtheta
        + np.cos(theta - np.pi / 2) ** 2 * dtheta ** 2
        + (y2 - np.sin(theta - np.pi / 2)) * np.sin(theta - np.pi / 2) * dtheta ** 2
        + (y2 - np.sin(theta - np.pi / 2)) * (-np.cos(theta - np.pi / 2)) * ddtheta
    )

    for i in range(n_ics):
        for j in range(n_steps):
            theta   = z[i, j, 0]   # angle θ
            dtheta  = dz[i, j, 0]  # θ̇
            ddtheta = dz[i, j, 1]  # θ̈ = −sin(θ)

            img = create_image(theta)
            ad  = arg_deriv(theta, dtheta)

            x[i, j]   = img
            dx[i, j]  = img * ad
            ddx[i, j] = img * (ad ** 2 + arg_deriv2(theta, dtheta, ddtheta))

    return x, dx, ddx


def get_pendulum_data(n_ics, t=None, noise_strength=0):
    """Generate n_ics pendulum trajectories mapped to 51×51 image space.

    Initial conditions are sampled uniformly from θ ∈ [−π, π], θ̇ ∈ [−2.1, 2.1]
    and filtered to keep only oscillating (non-spinning) trajectories via the
    energy criterion |E| ≤ 0.99 where E = θ̇²/2 − cos(θ).

    Returns a dict with keys: x, dx, ddx (float32, shape [n_ics*n_steps, 2601])
                          and t (float64), z (float32, shape [n_ics*n_steps, 2]).
    """
    if t is None:
        t = np.arange(0, 10, 0.02)
    n_steps = t.size
    n_pixels = 51

    z1range = np.array([-np.pi, np.pi])
    z2range = np.array([-2.1, 2.1])

    z_all  = np.zeros((n_ics, n_steps, 2))
    dz_all = np.zeros_like(z_all)

    ic_count = 0
    while ic_count < n_ics:
        z0 = np.array([
            z1range[0] + (z1range[1] - z1range[0]) * np.random.rand(),
            z2range[0] + (z2range[1] - z2range[0]) * np.random.rand(),
        ])
        # Keep only oscillating trajectories (not spinning over the top)
        if np.abs(z0[1] ** 2 / 2.0 - np.cos(z0[0])) > 0.99:
            continue
        z, dz = simulate_pendulum(z0, t)
        z[:, 0] = wrap_to_pi(z[:, 0])
        z_all[ic_count]  = z
        dz_all[ic_count] = dz
        ic_count += 1

    x, dx, ddx = pendulum_to_movie(z_all, dz_all, n=n_pixels)

    N = n_ics * n_steps
    x   = x.reshape(N, -1).astype(np.float32)
    dx  = dx.reshape(N, -1).astype(np.float32)
    ddx = ddx.reshape(N, -1).astype(np.float32)
    z_flat = z_all.reshape(N, 2).astype(np.float32)

    if noise_strength > 0:
        x   += noise_strength * np.random.randn(*x.shape).astype(np.float32)
        dx  += noise_strength * np.random.randn(*dx.shape).astype(np.float32)
        ddx += noise_strength * np.random.randn(*ddx.shape).astype(np.float32)

    return {'x': x, 'dx': dx, 'ddx': ddx, 't': t, 'z': z_flat}
