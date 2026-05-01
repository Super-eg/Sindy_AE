import json
import os

import numpy as np
import torch
from scipy.io import savemat, loadmat


_NON_PERSISTED_KEYS = {"coefficient_mask"}  # stored separately via state_dict


def _params_to_jsonable(params):
    """Convert a params dict into JSON-safe types (numpy -> list, scalars preserved)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = {"__ndarray__": True, "data": v.tolist(), "shape": list(v.shape), "dtype": str(v.dtype)}
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _params_from_jsonable(obj):
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict) and v.get("__ndarray__"):
            out[k] = np.array(v["data"], dtype=v["dtype"]).reshape(v["shape"])
        else:
            out[k] = v
    return out


# ---------------- JSON ----------------

def save_model_json(model, params, path_prefix):
    """Write `{path_prefix}_params.json` and `{path_prefix}_weights.json`."""
    params_path = f"{path_prefix}_params.json"
    weights_path = f"{path_prefix}_weights.json"

    with open(params_path, "w") as f:
        json.dump(_params_to_jsonable(params), f, indent=2)

    state = model.state_dict()
    weights_obj = {}
    for k, v in state.items():
        arr = v.detach().cpu().numpy()
        weights_obj[k] = {"data": arr.tolist(), "shape": list(arr.shape), "dtype": str(arr.dtype)}
    with open(weights_path, "w") as f:
        json.dump(weights_obj, f)

    return params_path, weights_path


def load_model_json(path_prefix, device=None):
    """Rebuild a SindyAutoencoder from `{path_prefix}_params.json` + `_weights.json`."""
    from .autoencoder import SindyAutoencoder

    params_path = f"{path_prefix}_params.json"
    weights_path = f"{path_prefix}_weights.json"

    with open(params_path) as f:
        params = _params_from_jsonable(json.load(f))
    with open(weights_path) as f:
        weights_obj = json.load(f)

    model = SindyAutoencoder(params)
    state = {}
    for k, v in weights_obj.items():
        arr = np.array(v["data"], dtype=v["dtype"]).reshape(v["shape"])
        state[k] = torch.from_numpy(arr)
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    return model, params


# ---------------- MAT ----------------

def _flat_params_for_mat(params):
    """scipy.io.savemat can't handle None or nested dicts with mixed types well;
    coerce to a flat dict of numpy-friendly values."""
    safe = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            safe[k] = np.array(int(v))
        elif isinstance(v, (int, float, str)):
            safe[k] = v
        elif isinstance(v, np.ndarray):
            safe[k] = v
        elif isinstance(v, (list, tuple)):
            safe[k] = np.array(v)
        else:
            safe[k] = str(v)
    return safe


def save_model_mat(model, params, path_prefix):
    """Write a single `{path_prefix}.mat` containing weights, mask, coefficients, and params."""
    path = f"{path_prefix}.mat"

    mat_dict = {}
    # State dict: all weights, biases, sindy_coefficients, coefficient_mask
    for k, v in model.state_dict().items():
        # mat keys can't contain '.', replace with '__'
        mat_key = k.replace(".", "__")
        mat_dict[mat_key] = v.detach().cpu().numpy()

    # params under a struct-like key
    mat_dict["params"] = _flat_params_for_mat(params)

    savemat(path, mat_dict, long_field_names=True)
    return path


def load_model_mat(path_prefix, device=None, params_override=None):
    """Rebuild a SindyAutoencoder from `{path_prefix}.mat`.

    `params_override` can be supplied when the saved params don't round-trip
    cleanly through .mat (savemat mangles some types). Otherwise we attempt to
    reconstruct them from the 'params' struct in the file.
    """
    from .autoencoder import SindyAutoencoder

    if path_prefix.endswith('.mat'):
        path_prefix = path_prefix[:-4]
    path = f"{path_prefix}.mat"
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    if params_override is not None:
        params = dict(params_override)
    else:
        raw = mat["params"]
        params = {}
        for name in raw._fieldnames:
            val = getattr(raw, name)
            if isinstance(val, np.ndarray) and val.dtype.kind in ("U", "S"):
                params[name] = str(val)
            else:
                params[name] = val
        # coerce known ints
        for k in ("input_dim", "latent_dim", "poly_order", "library_dim",
                  "model_order", "batch_size", "max_epochs", "refinement_epochs",
                  "threshold_frequency", "print_frequency", "epoch_size",
                  "delay_dim", "delay_steps"):
            if k in params:
                params[k] = int(np.asarray(params[k]).item())
        # coerce known floats
        for k in ("tau", "dt"):
            if k in params:
                params[k] = float(np.asarray(params[k]).item())
        # coerce widths to list
        if "widths" in params:
            w = np.atleast_1d(params["widths"]).astype(int).tolist()
            params["widths"] = w
        # coerce bools
        for k in ("include_sine", "sequential_thresholding", "print_progress"):
            if k in params:
                params[k] = bool(np.asarray(params[k]).item())

    # squeeze_me=True collapses size-1 dims, so coefficient_mask [L,1] becomes [L].
    # Restore it to [library_dim, latent_dim] before the model is constructed so
    # that the buffer is registered with the correct shape from the start.
    if "coefficient_mask" in params:
        params["coefficient_mask"] = np.asarray(params["coefficient_mask"]).reshape(
            params["library_dim"], params["latent_dim"]
        )

    model = SindyAutoencoder(params)
    state = {}
    for k, v_ref in model.state_dict().items():
        mat_key = k.replace(".", "__")
        arr = np.asarray(mat[mat_key])
        # Reshape to the model's expected shape in case squeeze_me collapsed dims.
        state[k] = torch.from_numpy(arr).to(torch.float32).reshape(v_ref.shape)
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    return model, params
