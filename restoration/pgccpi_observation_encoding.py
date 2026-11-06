from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


PGCCPI_DATA_TERM_ORDER = ("NLL", "L2", "HUBER")
PGCCPI_DATA_TERM_TO_ID = {
    name: index for index, name in enumerate(PGCCPI_DATA_TERM_ORDER)
}
PGCCPI_NOISE_MODEL_ORDER = ("UNKNOWN", "GAUSSIAN", "POISSON", "POISSON_GAUSSIAN", "IMPULSE", "IMPULSE_GAUSSIAN")
PGCCPI_NOISE_MODEL_TO_ID = {
    name: index for index, name in enumerate(PGCCPI_NOISE_MODEL_ORDER)
}
PGCCPI_OPERATOR_ORDER = ("KNOWN_PSF_DEBLUR", "IDENTITY_DENOISE", "SUPER_RESOLUTION")
PGCCPI_OPERATOR_TO_ID = {
    name: index for index, name in enumerate(PGCCPI_OPERATOR_ORDER)
}
PGCCPI_NOISE_ATTRIBUTE_NAMES = (
    "noise_is_unknown",
    "has_gaussian",
    "has_poisson",
    "has_impulse",
    "impulse_is_bw",
    "impulse_is_rgb",
)
PGCCPI_OBSERVATION_META_NAMES = (
    "meta_applicable",
    "meta_observed",
    "log10_alpha",
    "log10_sigma_read",
    "background",
)


def _normalize_key(
    value: str,
    valid_map: Dict[str, int],
    label: str,
    aliases: Mapping[str, str] | None = None,
) -> str:
    key = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if aliases is not None:
        key = str(aliases.get(key, key))
    if key not in valid_map:
        raise ValueError(f"Unsupported PG/CCPi {label}: {value}. Expected one of {list(valid_map)}.")
    return key


def normalize_pgccpi_data_term_name(data_term: str) -> str:
    return _normalize_key(
        data_term,
        PGCCPI_DATA_TERM_TO_ID,
        "data term",
        aliases={
            "POISSON_GAUSSIAN": "NLL",
            "POISSON": "NLL",
            "NEGATIVE_LOG_LIKELIHOOD": "NLL",
            "NEGATIVE_LOG_LIKEHOOD": "NLL",
            "HUBER_LOSS": "HUBER",
            "HUBER_FIDELITY": "HUBER",
        },
    )


def pgccpi_data_term_id(data_term: str) -> int:
    return int(PGCCPI_DATA_TERM_TO_ID[normalize_pgccpi_data_term_name(data_term)])


def normalize_pgccpi_noise_model_name(noise_model: str) -> str:
    return _normalize_key(
        noise_model,
        PGCCPI_NOISE_MODEL_TO_ID,
        "noise model",
        aliases={
            "NONE": "UNKNOWN",
            "BW_IMPULSE": "IMPULSE",
            "BLACK_WHITE_IMPULSE": "IMPULSE",
            "BLACKWHITE_IMPULSE": "IMPULSE",
            "RGB_IMPULSE": "IMPULSE",
            "COLOR_IMPULSE": "IMPULSE",
            "COLOUR_IMPULSE": "IMPULSE",
            "BW_IMPULSE_GAUSSIAN": "IMPULSE_GAUSSIAN",
            "BLACK_WHITE_IMPULSE_GAUSSIAN": "IMPULSE_GAUSSIAN",
            "BLACKWHITE_IMPULSE_GAUSSIAN": "IMPULSE_GAUSSIAN",
            "RGB_IMPULSE_GAUSSIAN": "IMPULSE_GAUSSIAN",
            "COLOR_IMPULSE_GAUSSIAN": "IMPULSE_GAUSSIAN",
            "COLOUR_IMPULSE_GAUSSIAN": "IMPULSE_GAUSSIAN",
        },
    )


def pgccpi_noise_model_id(noise_model: str) -> int:
    return int(PGCCPI_NOISE_MODEL_TO_ID[normalize_pgccpi_noise_model_name(noise_model)])


def encode_pgccpi_noise_attributes(
    noise_model: str,
    *,
    channel_independent: float | bool | None = None,
) -> np.ndarray:
    raw_key = str(noise_model).strip().upper().replace("-", "_").replace(" ", "_")
    model_key = normalize_pgccpi_noise_model_name(noise_model)
    has_gaussian = model_key in {"GAUSSIAN", "POISSON_GAUSSIAN", "IMPULSE_GAUSSIAN"}
    has_poisson = model_key in {"POISSON", "POISSON_GAUSSIAN"}
    has_impulse = model_key in {"IMPULSE", "IMPULSE_GAUSSIAN"}
    noise_is_unknown = model_key == "UNKNOWN"

    impulse_is_bw = False
    impulse_is_rgb = False
    if has_impulse:
        if "RGB" in raw_key or "COLOR" in raw_key or "COLOUR" in raw_key:
            impulse_is_rgb = True
        elif "BW" in raw_key or "BLACK_WHITE" in raw_key or "BLACKWHITE" in raw_key:
            impulse_is_bw = True
        elif channel_independent is not None:
            impulse_is_rgb = bool(float(channel_independent))
            impulse_is_bw = not impulse_is_rgb
        else:
            impulse_is_bw = True

    return np.asarray(
        [
            float(noise_is_unknown),
            float(has_gaussian),
            float(has_poisson),
            float(has_impulse),
            float(impulse_is_bw),
            float(impulse_is_rgb),
        ],
        dtype=np.float32,
    )


def normalize_pgccpi_operator_name(operator_kind: str) -> str:
    return _normalize_key(
        operator_kind,
        PGCCPI_OPERATOR_TO_ID,
        "operator kind",
        aliases={
            "IDENTITY": "IDENTITY_DENOISE",
            "DENOISE": "IDENTITY_DENOISE",
            "DENOISING": "IDENTITY_DENOISE",
            "SR": "SUPER_RESOLUTION",
            "SISR": "SUPER_RESOLUTION",
            "SUPER_RES": "SUPER_RESOLUTION",
            "SUPERRESOLUTION": "SUPER_RESOLUTION",
        },
    )


def pgccpi_operator_id(operator_kind: str) -> int:
    return int(PGCCPI_OPERATOR_TO_ID[normalize_pgccpi_operator_name(operator_kind)])


def encode_pgccpi_observation_metadata(
    *,
    alpha: float | None = None,
    sigma_read: float | None = None,
    background: float | None = None,
    applicable: float | bool = False,
    observed: float | bool = False,
) -> np.ndarray:
    meta_applicable = bool(float(applicable))
    meta_observed = bool(float(observed)) and meta_applicable
    if meta_observed:
        alpha_value = float(np.log10(max(float(alpha if alpha is not None else 0.0), 1e-8)))
        sigma_read_value = float(np.log10(max(float(sigma_read if sigma_read is not None else 0.0), 1e-8)))
        background_value = float(background if background is not None else 0.0)
    else:
        alpha_value = 0.0
        sigma_read_value = 0.0
        background_value = 0.0
    return np.asarray(
        [
            float(meta_applicable),
            float(meta_observed),
            alpha_value,
            sigma_read_value,
            background_value,
        ],
        dtype=np.float32,
    )
