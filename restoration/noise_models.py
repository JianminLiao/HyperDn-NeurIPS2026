from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


DEFAULT_NOISE_FOR_DATA_TERM: Dict[str, str] = {
    "L2": "GAUSSIAN",
    "L1": "LAPLACE",
    "POISSON": "POISSON",
    "POISSON_GAUSSIAN": "POISSON_GAUSSIAN",
    "IMPULSE": "IMPULSE",
    "IMPULSE_GAUSSIAN": "IMPULSE_GAUSSIAN",
}


@dataclass(frozen=True)
class NoiseRealization:
    kind: str
    params: Dict[str, float]
    noisy: np.ndarray


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def apply_noise(
    image: np.ndarray,
    noise_kind: str,
    rng: Optional[np.random.Generator] = None,
    clip: bool = True,
    **params,
) -> NoiseRealization:
    kind = str(noise_kind).strip().upper()
    generator = rng if rng is not None else make_rng()
    image_arr = np.asarray(image, dtype=np.float32)

    if kind == "GAUSSIAN":
        sigma = float(params.get("sigma", 0.0))
        noisy = image_arr + generator.normal(loc=0.0, scale=sigma, size=image_arr.shape).astype(np.float32)
        metadata = {"sigma": sigma}
    elif kind == "LAPLACE":
        scale = float(params.get("scale", 0.0))
        noisy = image_arr + generator.laplace(loc=0.0, scale=scale, size=image_arr.shape).astype(np.float32)
        metadata = {"scale": scale}
    elif kind == "POISSON":
        p_max = float(params.get("p_max", 0.0))
        alpha = float(params.get("alpha", 1.0 / p_max if p_max > 0.0 else 1.0))
        background = float(params.get("background", 0.0))
        noisy = _apply_poisson_gaussian_noise(
            image=image_arr,
            alpha=alpha,
            sigma_read=0.0,
            background=background,
            rng=generator,
        )
        metadata = {"alpha": alpha, "p_max": float(1.0 / alpha) if alpha > 0.0 else 0.0, "background": background}
    elif kind == "POISSON_GAUSSIAN":
        sigma_read = float(params.get("sigma_read", 0.0))
        p_max = float(params.get("p_max", 0.0))
        alpha = float(params.get("alpha", 1.0 / p_max if p_max > 0.0 else 1.0))
        background = float(params.get("background", 0.0))
        noisy = _apply_poisson_gaussian_noise(
            image=image_arr,
            alpha=alpha,
            sigma_read=sigma_read,
            background=background,
            rng=generator,
        )
        metadata = {
            "alpha": alpha,
            "p_max": float(1.0 / alpha) if alpha > 0.0 else 0.0,
            "sigma_read": sigma_read,
            "background": background,
        }
    elif kind == "IMPULSE":
        amount = float(params.get("amount", 0.0))
        salt_vs_pepper = float(params.get("salt_vs_pepper", 0.5))
        channel_independent = bool(float(params.get("channel_independent", 0.0)))
        noisy = image_arr.copy()
        if amount > 0.0:
            mask_shape = image_arr.shape if channel_independent else image_arr.shape[-2:]
            mask = generator.random(size=mask_shape).astype(np.float32)
            salt_mask = mask < (amount * salt_vs_pepper)
            pepper_mask = (mask >= (amount * salt_vs_pepper)) & (mask < amount)
            if channel_independent:
                noisy[salt_mask] = 1.0
                noisy[pepper_mask] = 0.0
            else:
                noisy[:, salt_mask] = 1.0
                noisy[:, pepper_mask] = 0.0
        metadata = {
            "amount": amount,
            "salt_vs_pepper": salt_vs_pepper,
            "channel_independent": float(channel_independent),
        }
    elif kind == "IMPULSE_GAUSSIAN":
        amount = float(params.get("amount", 0.0))
        salt_vs_pepper = float(params.get("salt_vs_pepper", 0.5))
        channel_independent = bool(float(params.get("channel_independent", 0.0)))
        sigma = float(params.get("sigma", 0.0))
        noisy = image_arr.copy()
        if amount > 0.0:
            mask_shape = image_arr.shape if channel_independent else image_arr.shape[-2:]
            mask = generator.random(size=mask_shape).astype(np.float32)
            salt_mask = mask < (amount * salt_vs_pepper)
            pepper_mask = (mask >= (amount * salt_vs_pepper)) & (mask < amount)
            if channel_independent:
                noisy[salt_mask] = 1.0
                noisy[pepper_mask] = 0.0
            else:
                noisy[:, salt_mask] = 1.0
                noisy[:, pepper_mask] = 0.0
        if sigma > 0.0:
            noisy = noisy + generator.normal(loc=0.0, scale=sigma, size=image_arr.shape).astype(np.float32)
        metadata = {
            "amount": amount,
            "salt_vs_pepper": salt_vs_pepper,
            "channel_independent": float(channel_independent),
            "sigma": sigma,
        }
    else:
        raise ValueError(
            f"Unsupported noise_kind={noise_kind}. Expected one of "
            f"{sorted(DEFAULT_NOISE_FOR_DATA_TERM.values())}."
        )

    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)
    return NoiseRealization(kind=kind, params=metadata, noisy=np.asarray(noisy, dtype=np.float32))


def _apply_poisson_gaussian_noise(
    image: np.ndarray,
    alpha: float,
    sigma_read: float,
    background: float,
    rng: np.random.Generator,
) -> np.ndarray:
    image_arr = np.asarray(image, dtype=np.float32)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0 for Poisson-Gaussian noise, got {alpha}.")
    if sigma_read < 0.0:
        raise ValueError(f"sigma_read must be >= 0, got {sigma_read}.")

    rate = np.maximum((image_arr + float(background)) / float(alpha), 0.0)
    shot = float(alpha) * rng.poisson(rate).astype(np.float32)
    if sigma_read > 0.0:
        shot = shot + rng.normal(loc=0.0, scale=float(sigma_read), size=image_arr.shape).astype(np.float32)
    return shot.astype(np.float32, copy=False)
