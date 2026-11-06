from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .pgccpi_observation_encoding import PGCCPI_DATA_TERM_ORDER, normalize_pgccpi_data_term_name


PGCCPI_REGULARIZER_ORDER = ("TV", "L1", "TGV", "MCP", "LP", "L0", "DIFFPIR", "CBM3D", "NLM")
PGCCPI_REGULARIZER_TO_ID = {
    name: index for index, name in enumerate(PGCCPI_REGULARIZER_ORDER)
}
PGCCPI_PARAM_FAMILY_ORDER = (
    "TV",
    "TV_WITH_HUBER_DATA",
    "L1",
    "L0",
    "TGV",
    "TGV_WITH_HUBER_DATA",
    "MCP",
    "LP",
    "DIFFPIR",
    "CBM3D",
    "NLM",
)
PGCCPI_PARAM_FAMILY_TO_ID = {
    name: index for index, name in enumerate(PGCCPI_PARAM_FAMILY_ORDER)
}
PGCCPI_PARAM_ROLE_ORDER = (
    "pad",
    "lambda",
    "alpha",
    "gamma",
    "beta",
    "p",
    "delta",
    "zeta",
    "unknown",
)
PGCCPI_PARAM_ROLE_TO_ID = {
    name: index for index, name in enumerate(PGCCPI_PARAM_ROLE_ORDER)
}
PGCCPI_PARAM_SCALE_ORDER = ("none", "log10", "linear")
PGCCPI_PARAM_SCALE_TO_ID = {
    name: index for index, name in enumerate(PGCCPI_PARAM_SCALE_ORDER)
}

_PGCCPI_PARAM_SPEC_TABLE: Dict[str, List[Dict[str, str]]] = {
    "TV": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        }
    ],
    "TV_WITH_HUBER_DATA": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        },
        {
            "encoded_name": "log10_delta",
            "raw_name": "delta",
            "role": "delta",
            "scale": "log10",
        },
    ],
    "L1": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        }
    ],
    "L0": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        }
    ],
    "TGV": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        },
        {
            "encoded_name": "gamma",
            "raw_name": "gamma",
            "role": "gamma",
            "scale": "linear",
        },
    ],
    "TGV_WITH_HUBER_DATA": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        },
        {
            "encoded_name": "gamma",
            "raw_name": "gamma",
            "role": "gamma",
            "scale": "linear",
        },
        {
            "encoded_name": "log10_delta",
            "raw_name": "delta",
            "role": "delta",
            "scale": "log10",
        },
    ],
    "MCP": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        },
        {
            "encoded_name": "log10_beta",
            "raw_name": "beta",
            "role": "beta",
            "scale": "log10",
        },
    ],
    "LP": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        },
        {
            "encoded_name": "p",
            "raw_name": "p",
            "role": "p",
            "scale": "linear",
        },
    ],
    "DIFFPIR": [
        {
            "encoded_name": "log10_lambda",
            "raw_name": "lambda",
            "role": "lambda",
            "scale": "log10",
        },
        {
            "encoded_name": "zeta",
            "raw_name": "zeta",
            "role": "zeta",
            "scale": "linear",
        },
    ],
    "CBM3D": [
        {
            "encoded_name": "log10_sigma_psd",
            "raw_name": "sigma_psd",
            "role": "unknown",
            "scale": "log10",
        }
    ],
    "NLM": [
        {
            "encoded_name": "log10_h",
            "raw_name": "h",
            "role": "unknown",
            "scale": "log10",
        }
    ],
}

MAX_PGCCPI_PARAM_DIM = max(len(specs) for specs in _PGCCPI_PARAM_SPEC_TABLE.values())


def _resolve_param_family(regularizer_key: str, data_term_key: str | None = None) -> str:
    reg_key = normalize_pgccpi_regularizer_name(regularizer_key)
    if data_term_key is None:
        return reg_key
    term_key = normalize_pgccpi_data_term_name(data_term_key)
    if reg_key == "TV" and term_key == "HUBER":
        return "TV_WITH_HUBER_DATA"
    if reg_key == "TGV" and term_key == "HUBER":
        return "TGV_WITH_HUBER_DATA"
    return reg_key


def pgccpi_param_family_id(regularizer: str, data_term: str | None = None) -> int:
    family = _resolve_param_family(regularizer, data_term)
    return int(PGCCPI_PARAM_FAMILY_TO_ID[family])


def _param_specs_for_family(param_family: str) -> List[Dict[str, str]]:
    try:
        return [dict(spec) for spec in _PGCCPI_PARAM_SPEC_TABLE[str(param_family).upper()]]
    except KeyError as exc:
        raise ValueError(f"Unsupported PG/CCPi parameter family for encoding: {param_family}.") from exc


def pgccpi_param_specs(regularizer: str, data_term: str | None = None) -> List[Dict[str, str]]:
    return _param_specs_for_family(_resolve_param_family(regularizer, data_term))


def pgccpi_param_dim(regularizer: str, data_term: str | None = None) -> int:
    return int(len(pgccpi_param_specs(regularizer, data_term)))


def _encode_scalar(raw_value: float, scale: str) -> float:
    if scale == "log10":
        return float(np.log10(float(raw_value)))
    if scale in {"linear", "none"}:
        return float(raw_value)
    raise ValueError(f"Unsupported parameter scale: {scale}")


def _decode_scalar(encoded_value: torch.Tensor, scale: str) -> torch.Tensor:
    if scale == "log10":
        return 10 ** encoded_value
    if scale in {"linear", "none"}:
        return encoded_value
    raise ValueError(f"Unsupported parameter scale: {scale}")


def _encode_param_value(
    *,
    param_family: str,
    spec: Dict[str, str],
    optimal_param: Dict[str, Any],
    noise_sigma: float | None,
) -> float:
    if param_family == "DIFFPIR" and spec["raw_name"] == "lambda":
        # Oracle search stores the DiffPIR-original lambda; the unified lambda
        # absorbs sigma^2 so that the solver needs no noise level at inference.
        if noise_sigma is None:
            raise ValueError("DiffPIR target encoding requires noise_sigma to compute lambda * sigma^2.")
        return float(optimal_param["lambda"]) * float(noise_sigma) ** 2
    return float(optimal_param[spec["raw_name"]])


def padded_pgccpi_param_role_ids(
    regularizer: str,
    data_term: str | None = None,
    *,
    pad_to: int = MAX_PGCCPI_PARAM_DIM,
) -> np.ndarray:
    if int(pad_to) <= 0:
        raise ValueError(f"pad_to must be > 0, got {pad_to}.")
    specs = pgccpi_param_specs(regularizer, data_term)
    role_ids = np.full(int(pad_to), PGCCPI_PARAM_ROLE_TO_ID["pad"], dtype=np.int64)
    for index, spec in enumerate(specs):
        if index >= int(pad_to):
            raise ValueError(f"Cannot pad role metadata with dim={len(specs)} into pad_to={pad_to}.")
        role_ids[index] = int(PGCCPI_PARAM_ROLE_TO_ID[spec["role"]])
    return role_ids


def padded_pgccpi_param_scale_ids(
    regularizer: str,
    data_term: str | None = None,
    *,
    pad_to: int = MAX_PGCCPI_PARAM_DIM,
) -> np.ndarray:
    if int(pad_to) <= 0:
        raise ValueError(f"pad_to must be > 0, got {pad_to}.")
    specs = pgccpi_param_specs(regularizer, data_term)
    scale_ids = np.full(int(pad_to), PGCCPI_PARAM_SCALE_TO_ID["none"], dtype=np.int64)
    for index, spec in enumerate(specs):
        if index >= int(pad_to):
            raise ValueError(f"Cannot pad scale metadata with dim={len(specs)} into pad_to={pad_to}.")
        scale_ids[index] = int(PGCCPI_PARAM_SCALE_TO_ID[spec["scale"]])
    return scale_ids


def build_pgccpi_param_metadata_tensors(
    regularizer_ids: torch.Tensor,
    *,
    data_term_ids: torch.Tensor | None = None,
    pad_to: int = MAX_PGCCPI_PARAM_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    role_rows = []
    scale_rows = []
    data_term_list = None
    if data_term_ids is not None:
        data_term_list = [int(v) for v in data_term_ids.detach().cpu().tolist()]
    for index, regularizer_id in enumerate(regularizer_ids.detach().cpu().tolist()):
        regularizer_name = PGCCPI_REGULARIZER_ORDER[int(regularizer_id)]
        data_term_name = None if data_term_list is None else PGCCPI_DATA_TERM_ORDER[int(data_term_list[index])]
        role_rows.append(padded_pgccpi_param_role_ids(regularizer_name, data_term_name, pad_to=pad_to))
        scale_rows.append(padded_pgccpi_param_scale_ids(regularizer_name, data_term_name, pad_to=pad_to))
    role_tensor = torch.as_tensor(np.stack(role_rows, axis=0), dtype=torch.long, device=regularizer_ids.device)
    scale_tensor = torch.as_tensor(np.stack(scale_rows, axis=0), dtype=torch.long, device=regularizer_ids.device)
    return role_tensor, scale_tensor


def normalize_pgccpi_regularizer_name(regularizer: str) -> str:
    regularizer_key = str(regularizer).strip().upper().replace("-", "_").replace(" ", "_")
    parts = tuple(part for part in regularizer_key.split("_") if part)
    huber_parts = tuple(part for part in parts if part in {"HUBER", "HUBERDATA"})
    base_parts = tuple(part for part in parts if part not in {"HUBER", "HUBERDATA"})
    if huber_parts and base_parts == ("TV",):
        raise ValueError(
            "Legacy Huber-data regularizer labels are no longer supported. "
            "Use regularizer='TV' and data_term='HUBER'."
        )
    if huber_parts and base_parts == ("TGV",):
        raise ValueError(
            "Legacy Huber-data regularizer labels are no longer supported. "
            "Use regularizer='TGV' and data_term='HUBER'."
        )
    if regularizer_key not in PGCCPI_REGULARIZER_TO_ID:
        raise ValueError(
            f"Unsupported PG/CCPi regularizer: {regularizer}. "
            f"Expected one of {list(PGCCPI_REGULARIZER_ORDER)}."
        )
    return regularizer_key


def pgccpi_regularizer_id(regularizer: str) -> int:
    return int(PGCCPI_REGULARIZER_TO_ID[normalize_pgccpi_regularizer_name(regularizer)])


def encode_pgccpi_target(
    regularizer: str,
    optimal_param: Dict[str, Any],
    *,
    data_term: str | None = None,
    noise_sigma: float | None = None,
    pad_to: int | None = None,
) -> Dict[str, Any]:
    regularizer_key = normalize_pgccpi_regularizer_name(regularizer)
    param_family = _resolve_param_family(regularizer_key, data_term)
    specs = _param_specs_for_family(param_family)
    vector = np.asarray(
        [
            _encode_scalar(
                _encode_param_value(
                    param_family=param_family,
                    spec=spec,
                    optimal_param=optimal_param,
                    noise_sigma=noise_sigma,
                ),
                spec["scale"],
            )
            for spec in specs
        ],
        dtype=np.float32,
    )
    names = [spec["encoded_name"] for spec in specs]
    raw_names = [spec["raw_name"] for spec in specs]
    role_names = [spec["role"] for spec in specs]
    scale_names = [spec["scale"] for spec in specs]

    mask = np.ones_like(vector, dtype=np.float32)
    payload: Dict[str, Any] = {
        "vector": vector,
        "mask": mask,
        "names": list(names),
        "raw_names": list(raw_names),
        "role_names": list(role_names),
        "scale_names": list(scale_names),
    }
    if pad_to is not None:
        padded, padded_mask = pad_pgccpi_target_vector(vector, pad_to=pad_to)
        payload["padded_vector"] = padded
        payload["padded_mask"] = padded_mask
        payload["padded_names"] = list(names) + [f"pad_{idx}" for idx in range(len(names), int(pad_to))]
        payload["padded_role_ids"] = padded_pgccpi_param_role_ids(regularizer_key, data_term, pad_to=int(pad_to))
        payload["padded_scale_ids"] = padded_pgccpi_param_scale_ids(regularizer_key, data_term, pad_to=int(pad_to))
    return payload


def pad_pgccpi_target_vector(vector: np.ndarray | torch.Tensor, *, pad_to: int = MAX_PGCCPI_PARAM_DIM):
    if int(pad_to) <= 0:
        raise ValueError(f"pad_to must be > 0, got {pad_to}.")
    if isinstance(vector, torch.Tensor):
        values = vector.detach().cpu().flatten().to(dtype=torch.float32)
        if values.numel() > int(pad_to):
            raise ValueError(f"Cannot pad tensor with dim={values.numel()} into pad_to={pad_to}.")
        padded = torch.zeros(int(pad_to), dtype=values.dtype)
        mask = torch.zeros(int(pad_to), dtype=values.dtype)
        padded[: values.numel()] = values
        mask[: values.numel()] = 1.0
        return padded, mask

    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size > int(pad_to):
        raise ValueError(f"Cannot pad vector with dim={values.size} into pad_to={pad_to}.")
    padded = np.zeros(int(pad_to), dtype=np.float32)
    mask = np.zeros(int(pad_to), dtype=np.float32)
    padded[: values.size] = values
    mask[: values.size] = 1.0
    return padded, mask


def decode_pgccpi_target_tensor(
    targets: torch.Tensor,
    regularizer: str,
    data_term: str | None = None,
) -> torch.Tensor:
    regularizer_key = normalize_pgccpi_regularizer_name(regularizer)
    specs = _param_specs_for_family(_resolve_param_family(regularizer_key, data_term))
    input_was_vector = targets.ndim == 1
    decoded = targets.unsqueeze(0).clone() if input_was_vector else targets.clone()
    for index, spec in enumerate(specs):
        if index >= int(decoded.shape[1]):
            break
        decoded[:, index] = _decode_scalar(decoded[:, index], spec["scale"])
    usable_dim = min(len(specs), int(decoded.shape[1]))
    return decoded[0, :usable_dim] if input_was_vector else decoded[:, :usable_dim]


def decode_mixed_pgccpi_targets(
    targets: torch.Tensor,
    regularizer_ids: torch.Tensor,
    data_term_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if targets.ndim != 2:
        raise ValueError(f"decode_mixed_pgccpi_targets expects rank-2 targets, got {targets.shape}.")
    if regularizer_ids.ndim != 1 or int(regularizer_ids.shape[0]) != int(targets.shape[0]):
        raise ValueError(
            "decode_mixed_pgccpi_targets expects regularizer_ids to be rank-1 and aligned with targets, "
            f"got targets={targets.shape} regularizer_ids={regularizer_ids.shape}."
        )
    data_term_list = None
    if data_term_ids is not None:
        if data_term_ids.ndim != 1 or int(data_term_ids.shape[0]) != int(targets.shape[0]):
            raise ValueError(
                "decode_mixed_pgccpi_targets expects data_term_ids to be rank-1 and aligned with targets, "
                f"got targets={targets.shape} data_term_ids={data_term_ids.shape}."
            )
        data_term_list = [int(v) for v in data_term_ids.detach().cpu().tolist()]
    decoded = targets.clone()
    decoded.zero_()
    for index in range(int(targets.shape[0])):
        regularizer_name = PGCCPI_REGULARIZER_ORDER[int(regularizer_ids[index].item())]
        data_term_name = None if data_term_list is None else PGCCPI_DATA_TERM_ORDER[int(data_term_list[index])]
        row = decode_pgccpi_target_tensor(targets[index], regularizer_name, data_term_name)
        decoded[index, : int(row.shape[0])] = row.to(device=decoded.device, dtype=decoded.dtype)
    return decoded


def pgccpi_param_dict_from_tensor(
    values: torch.Tensor,
    regularizer: str,
    *,
    data_term: str | None = None,
    encoded: bool = False,
) -> Dict[str, float]:
    specs = pgccpi_param_specs(regularizer, data_term)
    row = values.detach()
    if row.ndim != 1:
        row = row.reshape(-1)
    if encoded:
        row = decode_pgccpi_target_tensor(row, regularizer, data_term)
    else:
        row = row[: len(specs)]
    param_dict: Dict[str, float] = {}
    for index, spec in enumerate(specs):
        if index >= int(row.numel()):
            if _resolve_param_family(regularizer, data_term) == "DIFFPIR" and spec["raw_name"] == "zeta":
                continue
            raise ValueError(
                f"Parameter tensor for {regularizer}/{data_term} has dim={int(row.numel())}, "
                f"but spec requires {len(specs)} values."
            )
        param_dict[str(spec["raw_name"])] = float(row[index].item())
    return param_dict


def pgccpi_solver_param_dict_from_tensor(
    values: torch.Tensor,
    regularizer: str,
    *,
    data_term: str | None = None,
    encoded: bool = False,
    noise_sigma: float | None = None,
) -> Dict[str, float]:
    param_dict = pgccpi_param_dict_from_tensor(
        values,
        regularizer,
        data_term=data_term,
        encoded=encoded,
    )
    if _resolve_param_family(regularizer, data_term) == "DIFFPIR":
        param_dict.setdefault("zeta", 1.0)
    return param_dict
