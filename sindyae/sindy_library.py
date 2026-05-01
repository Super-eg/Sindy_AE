import torch
from scipy.special import binom


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_library_torch(z, poly_order, include_sine=False):
    """Build SINDy feature library on z with shape [batch, latent_dim].

    Term ordering matches sindyae/sindy_utils.py::sindy_library:
      [1, z_i, z_i*z_j (i<=j), z_i*z_j*z_k (i<=j<=k), ..., up to poly_order,
       sin(z_i) (optional)].
    """
    m, n = z.shape
    cols = [torch.ones(m, device=z.device, dtype=z.dtype)]

    for i in range(n):
        cols.append(z[:, i])

    if poly_order > 1:
        for i in range(n):
            for j in range(i, n):
                cols.append(z[:, i] * z[:, j])

    if poly_order > 2:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    cols.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        cols.append(z[:, i] * z[:, j] * z[:, k] * z[:, q])

    if poly_order > 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        for r in range(q, n):
                            cols.append(z[:, i] * z[:, j] * z[:, k] * z[:, q] * z[:, r])

    if include_sine:
        for i in range(n):
            cols.append(torch.sin(z[:, i]))

    return torch.stack(cols, dim=1)


def sindy_library_torch_order2(z, dz, poly_order, include_sine=False):
    """Order-2 library: treat [z, dz] as a 2n-dim vector and build the library."""
    z_combined = torch.cat([z, dz], dim=1)
    return sindy_library_torch(z_combined, poly_order, include_sine)
