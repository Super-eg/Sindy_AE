import torch
import torch.nn as nn
import torch.nn.functional as F

from .sindy_library import sindy_library_torch, sindy_library_torch_order2


_ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "linear": nn.Identity,
}


def _build_mlp(input_dim, output_dim, widths, activation):
    """Stack Linear + activation for each width; final layer is linear only."""
    layers = []
    last = input_dim
    act_cls = _ACTIVATIONS[activation]
    for w in widths:
        lin = nn.Linear(last, w)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if activation != "linear":
            layers.append(act_cls())
        last = w
    final = nn.Linear(last, output_dim)
    nn.init.xavier_uniform_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


def _seq_jvp(seq, h, dh):
    """Analytical forward-mode JVP through an nn.Sequential of Linear + pointwise activations.

    Propagates (h, dh) jointly: h is the primal, dh is the tangent (J @ input_tangent).
    Returns (output, d_output). Avoids second-order computation graphs entirely.
    """
    for module in seq:
        if isinstance(module, nn.Linear):
            dh = dh @ module.weight.T
            h = module(h)
        elif isinstance(module, nn.Sigmoid):
            h_new = torch.sigmoid(h)
            dh = dh * (h_new * (1.0 - h_new))
            h = h_new
        elif isinstance(module, nn.Tanh):
            h_new = torch.tanh(h)
            dh = dh * (1.0 - h_new ** 2)
            h = h_new
        elif isinstance(module, nn.ELU):
            h_new = F.elu(h)
            dh = dh * torch.where(h > 0, torch.ones_like(h), torch.exp(h))
            h = h_new
        elif isinstance(module, nn.ReLU):
            h_new = F.relu(h)
            dh = dh * (h > 0).to(dh.dtype)
            h = h_new
        elif isinstance(module, nn.Identity):
            pass
        else:
            raise ValueError(f"_seq_jvp: unsupported layer type {type(module).__name__}")
    return h, dh


class SindyAutoencoder(nn.Module):
    """Autoencoder + SINDy model using torch.autograd for derivatives.

    params keys:
        input_dim, latent_dim, widths, activation,
        poly_order, include_sine, library_dim, model_order,
        coefficient_initialization ('xavier'|'constant'|'normal'|'specified'),
        init_coefficients (required if coefficient_initialization == 'specified').
    """

    def __init__(self, params):
        super().__init__()
        self.params = dict(params)
        self.input_dim = params["input_dim"]
        self.latent_dim = params["latent_dim"]
        self.poly_order = params["poly_order"]
        self.include_sine = params.get("include_sine", False)
        self.model_order = params.get("model_order", 1)
        self.library_dim = params["library_dim"]
        self.activation = params.get("activation", "sigmoid")
        self.widths = list(params.get("widths", []))

        self.encoder = _build_mlp(self.input_dim, self.latent_dim, self.widths, self.activation)
        self.decoder = _build_mlp(self.latent_dim, self.input_dim, list(reversed(self.widths)), self.activation)

        coeffs = torch.empty(self.library_dim, self.latent_dim)
        init = params.get("coefficient_initialization", "constant")
        if init == "xavier":
            nn.init.xavier_uniform_(coeffs)
        elif init == "constant":
            nn.init.ones_(coeffs)
        elif init == "normal":
            nn.init.normal_(coeffs)
        elif init == "specified":
            coeffs = torch.as_tensor(params["init_coefficients"], dtype=torch.float32).clone()
        else:
            raise ValueError(f"Unknown coefficient_initialization: {init}")
        self.sindy_coefficients = nn.Parameter(coeffs)

        mask_init = params.get("coefficient_mask")
        if mask_init is None:
            mask = torch.ones(self.library_dim, self.latent_dim)
        else:
            mask = torch.as_tensor(mask_init, dtype=torch.float32).clone()
        self.register_buffer("coefficient_mask", mask)

    def _masked_coefficients(self):
        return self.coefficient_mask * self.sindy_coefficients

    def forward(self, x, dx, ddx=None):
        """Forward pass.

        x, dx: [batch, input_dim]. ddx required if model_order == 2.
        Returns dict with:
            z, x_decode, dz, dz_predict, dx_decode, Theta,
            and (if order 2) ddz, ddz_predict, ddx_decode.
        """
        Xi = self._masked_coefficients()

        if self.model_order == 1:
            # Analytical JVP through encoder: no second-order graph needed.
            z, dz = _seq_jvp(self.encoder, x, dx)
            Theta = sindy_library_torch(z, self.poly_order, self.include_sine)
            dz_predict = Theta @ Xi
            # Pull dz_predict back to input space through decoder JVP.
            x_decode, dx_decode = _seq_jvp(self.decoder, z, dz_predict)
            return {
                "z": z, "x_decode": x_decode,
                "dz": dz, "dz_predict": dz_predict,
                "dx_decode": dx_decode, "Theta": Theta,
            }
        else:
            if ddx is None:
                raise ValueError("ddx is required when model_order == 2")
            # First-order JVP through encoder.
            z, dz = _seq_jvp(self.encoder, x, dx)
            # Second-order: d²z/dt² = J_enc(x) @ ddx + H_enc(x)[dx, dx].
            # J_enc(x) @ ddx is the first-order JVP with tangent ddx.
            _, ddz_linear = _seq_jvp(self.encoder, x, ddx)
            # H_enc(x)[dx, dx]: differentiate dz (which already tracks x) w.r.t. x
            # contracted with dx. Since dz from _seq_jvp depends on x through the
            # activation derivatives, we can use autograd for the Hessian term.
            # H_enc(x)[dx, dx]: Hessian-vector product for each latent dimension.
            # dz_req[i,j] = (J_enc(x_req[i]) @ dx[i])[j] depends on x_req through
            # the activation derivatives in _seq_jvp.
            # We want hess_term[i,j] = (∂ dz_req[i,j] / ∂ x_req[i,:]) · dx[i,:]
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                _, dz_req = _seq_jvp(self.encoder, x_req, dx)
                hess_term = torch.zeros(
                    x.shape[0], self.latent_dim, device=x.device, dtype=x.dtype
                )
                for j in range(self.latent_dim):
                    grad_j = torch.autograd.grad(
                        dz_req[:, j].sum(), x_req,
                        retain_graph=True,
                        create_graph=self.training,
                    )[0]  # [batch, input_dim]
                    hess_term[:, j] = (grad_j * dx).sum(dim=1)  # [batch]
            ddz = ddz_linear + hess_term  # [batch, latent_dim]

            Theta = sindy_library_torch_order2(z, dz, self.poly_order, self.include_sine)
            ddz_predict = Theta @ Xi
            # Full 2nd-order decoder chain rule: ddx = J_ψ(z)·z̈ + H_ψ(z)[ż, ż].
            # _seq_jvp gives only the first term; add the Hessian term explicitly.
            # Mirrors the encoder Hessian above (lines ~156-169) but loops over
            # output dims of the decoder (= input_dim) instead of latent_dim.
            x_decode, ddx_decode_linear = _seq_jvp(self.decoder, z, ddz_predict)
            with torch.enable_grad():
                z_req = z.detach().requires_grad_(True)
                _, dx_req = _seq_jvp(self.decoder, z_req, dz)
                hess_dec = torch.zeros_like(x_decode)
                for i in range(self.input_dim):
                    grad_i = torch.autograd.grad(
                        dx_req[:, i].sum(), z_req,
                        retain_graph=(i < self.input_dim - 1),
                        create_graph=self.training,
                    )[0]  # [batch, latent_dim]
                    hess_dec[:, i] = (grad_i * dz).sum(dim=1)
            ddx_decode = ddx_decode_linear + hess_dec
            return {
                "z": z, "x_decode": x_decode,
                "dz": dz, "ddz": ddz,
                "ddz_predict": ddz_predict, "ddx_decode": ddx_decode,
                "Theta": Theta,
            }
