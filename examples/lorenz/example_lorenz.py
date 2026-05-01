"""Lorenz data generation — copied from examples/lorenz/example_lorenz.py with
import paths adjusted so it works standalone (no sys.path hacks).

Nothing here depends on TensorFlow; it is pure numpy + scipy.
"""
import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre

from sindyae.sindy_library import library_size


def get_lorenz_data(n_ics, noise_strength=0, input_dim=128):
    t = np.arange(0, 5, .02)
    n_steps = t.size

    ic_means = np.array([0, 0, 25])
    ic_widths = 2 * np.array([36, 48, 41])

    ics = ic_widths * (np.random.rand(n_ics, 3) - .5) + ic_means
    data = generate_lorenz_data(
        ics, t, input_dim, linear=False,
        normalization=np.array([1 / 40, 1 / 40, 1 / 40]),
    )
    data['x'] = data['x'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['dx'] = data['dx'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['ddx'] = data['ddx'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)

    return data


def lorenz_coefficients(normalization, poly_order=3, sigma=10., beta=8 / 3, rho=28.):
    Xi = np.zeros((library_size(3, poly_order), 3))
    Xi[1, 0] = -sigma
    Xi[2, 0] = sigma * normalization[0] / normalization[1]
    Xi[1, 1] = rho * normalization[1] / normalization[0]
    Xi[2, 1] = -1
    Xi[6, 1] = -normalization[1] / (normalization[0] * normalization[2])
    Xi[3, 2] = -beta
    Xi[5, 2] = normalization[2] / (normalization[0] * normalization[1])
    return Xi


def simulate_lorenz(z0, t, sigma=10., beta=8 / 3, rho=28.):
    f = lambda z, t: [sigma * (z[1] - z[0]),
                      z[0] * (rho - z[2]) - z[1],
                      z[0] * z[1] - beta * z[2]]
    df = lambda z, dz, t: [sigma * (dz[1] - dz[0]),
                           dz[0] * (rho - z[2]) + z[0] * (-dz[2]) - dz[1],
                           dz[0] * z[1] + z[0] * dz[1] - beta * dz[2]]

    z = odeint(f, z0, t)

    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(z[i], 0)
        ddz[i] = df(z[i], dz[i], 0)
    return z, dz, ddz


def generate_lorenz_data(ics, t, n_points, linear=True, normalization=None,
                         sigma=10, beta=8 / 3, rho=28):
    n_ics = ics.shape[0]
    n_steps = t.size

    d = 3
    z = np.zeros((n_ics, n_steps, d))
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(n_ics):
        z[i], dz[i], ddz[i] = simulate_lorenz(ics[i], t, sigma=sigma, beta=beta, rho=rho)

    if normalization is not None:
        z *= normalization
        dz *= normalization
        ddz *= normalization

    n = n_points
    L = 1
    y_spatial = np.linspace(-L, L, n)

    modes = np.zeros((2 * d, n))
    for i in range(2 * d):
        modes[i] = legendre(i)(y_spatial)

    x = np.zeros((n_ics, n_steps, n))
    dx = np.zeros(x.shape)
    ddx = np.zeros(x.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            x[i, j] = modes[0] * z[i, j, 0] + modes[1] * z[i, j, 1] + modes[2] * z[i, j, 2]
            if not linear:
                x[i, j] += (modes[3] * z[i, j, 0] ** 3
                            + modes[4] * z[i, j, 1] ** 3
                            + modes[5] * z[i, j, 2] ** 3)

            dx[i, j] = modes[0] * dz[i, j, 0] + modes[1] * dz[i, j, 1] + modes[2] * dz[i, j, 2]
            if not linear:
                dx[i, j] += (modes[3] * 3 * (z[i, j, 0] ** 2) * dz[i, j, 0]
                             + modes[4] * 3 * (z[i, j, 1] ** 2) * dz[i, j, 1]
                             + modes[5] * 3 * (z[i, j, 2] ** 2) * dz[i, j, 2])

            ddx[i, j] = modes[0] * ddz[i, j, 0] + modes[1] * ddz[i, j, 1] + modes[2] * ddz[i, j, 2]
            if not linear:
                ddx[i, j] += (modes[3] * (6 * z[i, j, 0] * dz[i, j, 0] ** 2 + 3 * (z[i, j, 0] ** 2) * ddz[i, j, 0])
                              + modes[4] * (6 * z[i, j, 1] * dz[i, j, 1] ** 2 + 3 * (z[i, j, 1] ** 2) * ddz[i, j, 1])
                              + modes[5] * (6 * z[i, j, 2] * dz[i, j, 2] ** 2 + 3 * (z[i, j, 2] ** 2) * ddz[i, j, 2]))

    if normalization is None:
        sindy_coefficients = lorenz_coefficients([1, 1, 1], sigma=sigma, beta=beta, rho=rho)
    else:
        sindy_coefficients = lorenz_coefficients(normalization, sigma=sigma, beta=beta, rho=rho)

    return {
        't': t, 'y_spatial': y_spatial, 'modes': modes,
        'x': x.astype(np.float32), 'dx': dx.astype(np.float32), 'ddx': ddx.astype(np.float32),
        'z': z, 'dz': dz, 'ddz': ddz,
        'sindy_coefficients': sindy_coefficients.astype(np.float32),
    }
