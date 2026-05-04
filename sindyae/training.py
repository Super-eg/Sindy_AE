import time

import numpy as np
import torch
import torch.nn.functional as F

from .sindy_library import sindy_library_torch


def _to_tensor(arr, device, dtype=torch.float32):
    return torch.as_tensor(arr, dtype=dtype, device=device)


def _rk4_step(z, f, dt):
    """Single RK4 integration step: z_{n+1} = RK4(z_n, f, dt)."""
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _sindy_cons_loss(model, z0, x, params):
    """SINDy consistency loss (eq 1.11, Bakarji et al. 2023).

    Integrates the latent SINDy dynamics from z0 = encoder(x) and
    checks that z[:,0] reproduces each element of the delay embedding x.

    Our delay convention is backward: x[k] = y(t - k*tau), so we
    integrate with dt = -tau to step backward in time.
    """
    delay_dim = x.shape[1]
    if delay_dim < 2:
        return torch.zeros([], device=x.device, dtype=x.dtype)

    tau = params.get("tau")
    if tau is None:
        tau = params.get("dt", 1.0) * params.get("delay_steps", 1)
    dt = -float(tau)  # negative: x[k] = y(t - k*tau)

    Xi = model._masked_coefficients()

    def sindy_rhs(z):
        return sindy_library_torch(z, model.poly_order, model.include_sine) @ Xi

    loss = torch.zeros([], device=x.device, dtype=x.dtype)
    z = z0
    for j in range(1, delay_dim):
        z = _rk4_step(z, sindy_rhs, dt)
        if not torch.isfinite(z).all():
            return torch.zeros([], device=x.device, dtype=x.dtype)
        loss = loss + F.mse_loss(z[:, 0], x[:, j])

    return loss / (delay_dim - 1)


def compute_losses(outputs, batch, model, params):
    """Return dict of individual loss terms (all scalar tensors)."""
    losses = {}
    x = batch["x"]
    losses["decoder"] = F.mse_loss(outputs["x_decode"], x)

    if params["model_order"] == 1:
        dx = batch["dx"]
        losses["sindy_z"] = F.mse_loss(outputs["dz_predict"], outputs["dz"])
        losses["sindy_x"] = F.mse_loss(outputs["dx_decode"], dx)
    else:
        ddx = batch["ddx"]
        losses["sindy_z"] = F.mse_loss(outputs["ddz_predict"], outputs["ddz"])
        losses["sindy_x"] = F.mse_loss(outputs["ddx_decode"], ddx)

    masked = model.coefficient_mask * model.sindy_coefficients
    losses["sindy_regularization"] = masked.abs().mean()

    # coord loss: first latent variable ≈ first input element (useful for delay embedding)
    if params.get("loss_weight_coord", 0.0) > 0:
        losses["coord"] = F.mse_loss(outputs["z"][:, 0], x[:, 0])

    # SINDy consistency loss (Bakarji et al. 2023, eq 1.11):
    # integrate latent SINDy dynamics and verify z[:,0] reproduces delay entries
    if params.get("loss_weight_sindy_cons", 0.0) > 0 and params.get("tau") is not None:
        losses["sindy_cons"] = _sindy_cons_loss(model, outputs["z"], x, params)

    return losses


def _weighted_total(losses, params, include_reg=True):
    total = (
        params["loss_weight_decoder"] * losses["decoder"]
        + params["loss_weight_sindy_z"] * losses["sindy_z"]
        + params["loss_weight_sindy_x"] * losses["sindy_x"]
    )
    if include_reg:
        total = total + params["loss_weight_sindy_regularization"] * losses["sindy_regularization"]
    if "coord" in losses:
        total = total + params.get("loss_weight_coord", 0.0) * losses["coord"]
    if "sindy_cons" in losses:
        total = total + params.get("loss_weight_sindy_cons", 0.0) * losses["sindy_cons"]
    return total


def _iter_batches(n_samples, batch_size):
    """Yield shuffled random batch index arrays."""
    idxs = np.random.permutation(n_samples)
    n_batches = n_samples // batch_size
    for j in range(n_batches):
        yield idxs[j * batch_size:(j + 1) * batch_size]


def _make_batch(data, idxs, device, model_order):
    batch = {
        "x": _to_tensor(data["x"][idxs], device),
        "dx": _to_tensor(data["dx"][idxs], device),
    }
    if model_order == 2:
        batch["ddx"] = _to_tensor(data["ddx"][idxs], device)
    return batch


def _eval_validation(model, validation_data, params, device):
    """Run validation on the full set (in one pass) and return loss dict."""
    model.eval()
    # Need grads enabled for autograd-based derivative computation, so no no_grad().
    idxs = np.arange(validation_data["x"].shape[0])
    batch = _make_batch(validation_data, idxs, device, params["model_order"])
    outputs = model(batch["x"], batch["dx"], batch.get("ddx"))
    losses = compute_losses(outputs, batch, model, params)
    total = _weighted_total(losses, params, include_reg=True).item()
    model.train()
    return {k: float(v.item()) for k, v in losses.items()}, total


def _make_scheduler(optimizer, params, print_frequency):
    """Build an LR scheduler (or return None) from params.

    Supported scheduler_type values:
        'plateau'  — ReduceLROnPlateau (default, recommended)
        'cosine'   — CosineAnnealingLR
        None / ''  — no scheduler
    """
    sched_type = params.get("scheduler_type", "plateau")
    if not sched_type:
        return None, None

    factor  = params.get("scheduler_factor",  0.5)
    min_lr  = params.get("scheduler_min_lr",  1e-5)

    if sched_type == "plateau":
        # patience is given in epochs; convert to validation-evaluation steps
        patience_epochs = params.get("scheduler_patience", 500)
        patience_evals  = max(1, patience_epochs // print_frequency)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor,
            patience=patience_evals, threshold=1e-4,
            min_lr=min_lr,
        )
    elif sched_type == "cosine":
        T_max   = params.get("scheduler_T_max", params.get("max_epochs", 15001))
        eta_min = min_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {sched_type!r}. Use 'plateau', 'cosine', or None.")

    return scheduler, sched_type


def train_network(training_data, validation_data, params, model=None, device=None):
    """Train a SindyAutoencoder.

    Returns dict with:
        model: trained SindyAutoencoder
        losses: list of validation-loss dicts (sampled every print_frequency epochs)
        sindy_model_terms: active-coefficient count after each thresholding step
        final_coefficient_mask: numpy array
        final_sindy_coefficients: numpy array (masked)

    Scheduler notes
    ---------------
    Set params['scheduler_type'] = 'plateau' (default) to use ReduceLROnPlateau.
    The scheduler steps on validation loss every print_frequency epochs.
    When the LR drops the early-stop counter resets, giving the model a fresh
    window to improve at the new learning rate.

    Early stopping
    --------------
    params['early_stopping_patience'] — number of consecutive validation evaluations
    (each separated by print_frequency epochs) without improvement before stopping.
    Default: 5. After each LR reduction the counter resets automatically.
    """
    from .autoencoder import SindyAutoencoder

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = SindyAutoencoder(params)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    n_samples = training_data["x"].shape[0]
    batch_size = params["batch_size"]
    max_epochs = params["max_epochs"]
    refinement_epochs = params.get("refinement_epochs", 0)
    print_progress = params.get("print_progress", True)
    print_frequency = params.get("print_frequency", 100)
    seq_thresh = params.get("sequential_thresholding", False)
    threshold_frequency = params.get("threshold_frequency", 500)
    coef_threshold = params.get("coefficient_threshold", 0.0)
    # patience = number of consecutive validation evaluations with no improvement
    patience  = params.get("early_stopping_patience", 5)
    min_delta = params.get("early_stopping_min_delta", 1e-6)

    scheduler, sched_type = _make_scheduler(optimizer, params, print_frequency)

    validation_losses = []
    sindy_model_terms = [int(model.coefficient_mask.sum().item())]
    best_loss  = float("inf")
    best_epoch = 0
    es_count   = 0      # consecutive evaluations without improvement
    last_lr    = params["learning_rate"]   # track for LR-drop detection

    t0 = time.time()

    # ---- Main training phase ----
    for epoch in range(max_epochs):
        for idxs in _iter_batches(n_samples, batch_size):
            batch = _make_batch(training_data, idxs, device, params["model_order"])
            optimizer.zero_grad()
            outputs = model(batch["x"], batch["dx"], batch.get("ddx"))
            losses = compute_losses(outputs, batch, model, params)
            total = _weighted_total(losses, params, include_reg=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        # CosineAnnealingLR steps every epoch (not tied to validation)
        if scheduler is not None and sched_type == "cosine":
            scheduler.step()

        if epoch % print_frequency == 0:
            val_losses, val_total = _eval_validation(model, validation_data, params, device)
            validation_losses.append({"epoch": epoch, "phase": "main", "total": val_total, **val_losses})

            # ReduceLROnPlateau steps on validation loss
            if scheduler is not None and sched_type == "plateau":
                scheduler.step(val_total)

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            coord_str = (f"  coord={val_losses['coord']:.3e}"
                         if "coord" in val_losses else "")
            cons_str = (f"  cons={val_losses['sindy_cons']:.3e}"
                        if "sindy_cons" in val_losses else "")
            if print_progress:
                print(
                    f"[main {epoch:5d}] total={val_total:.3e}  "
                    f"dec={val_losses['decoder']:.3e}  sindy_z={val_losses['sindy_z']:.3e}  "
                    f"sindy_x={val_losses['sindy_x']:.3e}  reg={val_losses['sindy_regularization']:.3e}"
                    f"{coord_str}{cons_str}  lr={current_lr:.2e}  ({elapsed:.1f}s)"
                )

            # Detect LR drop → reset early-stop counter so the model gets a
            # fresh window at the new (lower) learning rate.
            if current_lr < last_lr - 1e-14:
                print(f"  [scheduler] LR reduced: {last_lr:.2e} → {current_lr:.2e}  "
                      f"(early-stop counter reset)")
                best_loss  = val_total
                best_epoch = epoch
                es_count   = 0
                last_lr    = current_lr

            if patience > 0:
                if val_total < best_loss - min_delta:
                    best_loss, best_epoch = val_total, epoch
                    es_count = 0
                else:
                    es_count += 1
                    if es_count >= patience:
                        print(
                            f"Early stopping at epoch {epoch}: "
                            f"{patience} consecutive evaluations with no improvement "
                            f">{min_delta:.0e}  (best={best_loss:.3e} @ epoch {best_epoch})"
                        )
                        break

        if seq_thresh and (epoch % threshold_frequency == 0) and (epoch > 0):
            with torch.no_grad():
                new_mask = (model.sindy_coefficients.abs() > coef_threshold).float()
                model.coefficient_mask.copy_(new_mask)
            active = int(model.coefficient_mask.sum().item())
            sindy_model_terms.append(active)
            print(f"THRESHOLDING: {active} active coefficients")

    # ---- Refinement phase (no L1 regularization) ----
    if refinement_epochs > 0:
        print("REFINEMENT")
        # Use a fresh optimizer at the final main-phase LR so the scheduler
        # state doesn't carry over; refinement does its own simple fine-tuning.
        refine_lr = optimizer.param_groups[0]["lr"]
        refine_optimizer = torch.optim.Adam(model.parameters(), lr=refine_lr)
        best_loss  = float("inf")
        best_epoch = 0
        es_count   = 0
        for epoch in range(refinement_epochs):
            for idxs in _iter_batches(n_samples, batch_size):
                batch = _make_batch(training_data, idxs, device, params["model_order"])
                refine_optimizer.zero_grad()
                outputs = model(batch["x"], batch["dx"], batch.get("ddx"))
                losses = compute_losses(outputs, batch, model, params)
                total = _weighted_total(losses, params, include_reg=False)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                refine_optimizer.step()

            if epoch % print_frequency == 0:
                val_losses, val_total = _eval_validation(model, validation_data, params, device)
                validation_losses.append({"epoch": epoch, "phase": "refine", "total": val_total, **val_losses})
                elapsed = time.time() - t0
                coord_str = (f"  coord={val_losses['coord']:.3e}"
                             if "coord" in val_losses else "")
                cons_str = (f"  cons={val_losses['sindy_cons']:.3e}"
                            if "sindy_cons" in val_losses else "")
                if print_progress:
                    print(
                        f"[refine {epoch:5d}] total={val_total:.3e}  "
                        f"dec={val_losses['decoder']:.3e}  sindy_z={val_losses['sindy_z']:.3e}  "
                        f"sindy_x={val_losses['sindy_x']:.3e}{coord_str}{cons_str}  lr={refine_lr:.2e}  ({elapsed:.1f}s)"
                    )
                if patience > 0:
                    if val_total < best_loss - min_delta:
                        best_loss, best_epoch = val_total, epoch
                        es_count = 0
                    else:
                        es_count += 1
                        if es_count >= patience:
                            print(
                                f"Early stopping (refine) at epoch {epoch}: "
                                f"{patience} consecutive evaluations with no improvement "
                                f">{min_delta:.0e}  (best={best_loss:.3e} @ epoch {best_epoch})"
                            )
                            break

    with torch.no_grad():
        final_mask = model.coefficient_mask.detach().cpu().numpy()
        final_coeffs = (model.coefficient_mask * model.sindy_coefficients).detach().cpu().numpy()

    return {
        "model": model,
        "losses": validation_losses,
        "sindy_model_terms": sindy_model_terms,
        "final_coefficient_mask": final_mask,
        "final_sindy_coefficients": final_coeffs,
    }
