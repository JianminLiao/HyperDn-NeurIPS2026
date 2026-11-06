from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import convolve

try:
    from cil.framework import ImageData, ImageGeometry
    from cil.optimisation.functions import Function
except ImportError:
    ImageData = None
    ImageGeometry = None
    Function = None

_FunctionBase = Function if Function is not None else object


def require_cil_core():
    if ImageData is None or ImageGeometry is None or Function is None:
        raise ImportError(
            "Missing dependency `cil`. Install the Core Imaging Library to use the CIL-based "
            "Poisson-Gaussian deblurring components."
        )


def forward_convolve2d(image_2d: np.ndarray, psf: np.ndarray) -> np.ndarray:
    return convolve(np.asarray(image_2d, dtype=np.float32), np.asarray(psf, dtype=np.float32), mode="same").astype(
        np.float32,
        copy=False,
    )


def adjoint_convolve2d(image_2d: np.ndarray, psf_mirror: np.ndarray) -> np.ndarray:
    return convolve(
        np.asarray(image_2d, dtype=np.float32),
        np.asarray(psf_mirror, dtype=np.float32),
        mode="same",
    ).astype(np.float32, copy=False)


def estimate_psf_operator_norm_sq(psf: np.ndarray, shape, num_iterations: int = 25) -> float:
    psf_arr = np.asarray(psf, dtype=np.float32)
    psf_mirror = np.flip(psf_arr)
    rng = np.random.RandomState(0)
    vec = rng.standard_normal(shape).astype(np.float32)
    vec /= max(np.linalg.norm(vec.ravel()), 1e-8)

    rayleigh = 1.0
    for _ in range(max(1, int(num_iterations))):
        vec = adjoint_convolve2d(forward_convolve2d(vec, psf_arr), psf_mirror)
        norm = float(np.linalg.norm(vec.ravel()))
        if norm <= 1e-12:
            return 1.0
        vec /= norm
        rayleigh = float(
            np.sum(vec * adjoint_convolve2d(forward_convolve2d(vec, psf_arr), psf_mirror))
        )
    return float(max(rayleigh, 1e-8))


def poisson_gaussian_scalar_curvature(
    observed: np.ndarray,
    model_value: np.ndarray | float,
    alpha: float,
    sigma_read_sq: float,
    variance_epsilon: float,
) -> np.ndarray:
    u = np.asarray(model_value, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    variance = np.maximum(alpha * u + sigma_read_sq, float(variance_epsilon))
    residual = y - u
    return (
        (1.0 / variance)
        + (2.0 * alpha * residual) / (variance * variance)
        - (0.5 * alpha * alpha) / (variance * variance)
        + (alpha * alpha * residual * residual) / (variance * variance * variance)
    )


def estimate_poisson_gaussian_lipschitz(
    observed: np.ndarray,
    psf: np.ndarray,
    alpha: float,
    sigma_read: float,
    background: np.ndarray | float,
    variance_epsilon: float = 1e-8,
    operator_norm_iterations: int = 25,
    curvature_grid_points: int = 33,
    safety_scale: float = 1.25,
) -> float:
    observed_arr = np.asarray(observed, dtype=np.float32)
    background_arr = np.asarray(background, dtype=np.float32)
    sigma_read_sq = float(sigma_read * sigma_read)
    operator_norm_sq = estimate_psf_operator_norm_sq(
        psf=np.asarray(psf, dtype=np.float32),
        shape=observed_arr.shape,
        num_iterations=operator_norm_iterations,
    )

    background_min = float(np.min(background_arr)) if background_arr.size else float(background_arr)
    background_max = float(np.max(background_arr)) if background_arr.size else float(background_arr)
    # signal_floor caps how small the curvature-grid model values go. The NLL
    # curvature 1/(alpha*u + sigma_read^2) blows up as u -> 0, so the floor sets
    # the worst-case Lipschitz: L ~ 1/(alpha * signal_floor).
    #
    # The previous formula `max(alpha, sigma_read, 1e-6)` gave floor=alpha, so
    # L ~ 1/alpha^2. For bright Poisson images (large p_max => small alpha, e.g.
    # alpha=0.008), this yielded L ~ 4e8 and FISTA step ~ 2.5e-9 — so microscopic
    # that TGV's PDHG proximal could not converge in 200 iterations, producing
    # the FM16 bright-image TGV<TV gap. FGP-TV's proximal is less sensitive to
    # outer step size, which is why TV still worked.
    #
    # Fix: use a fixed absolute minimum (0.1) so the worst-case curvature is
    # bounded by ~1/(alpha * 0.1) = 10/alpha. For bright alpha=0.008 this gives
    # L ~ 1500 (vs. 4e8) and step ~ 6e-4 — FISTA can actually converge. For dim
    # alpha=0.18 the max(alpha, 0.1) = 0.18 is unchanged, so dim behaviour is
    # preserved.
    signal_floor = max(0.1, float(alpha), float(sigma_read), 1e-6)
    model_min = max(background_min, signal_floor)
    model_max = max(1.0 + background_max, float(np.max(observed_arr)) + 0.5)
    model_grid = np.linspace(model_min, model_max, max(3, int(curvature_grid_points)), dtype=np.float64)

    max_curvature = 0.0
    for model_value in model_grid:
        curvature = poisson_gaussian_scalar_curvature(
            observed=observed_arr,
            model_value=model_value,
            alpha=float(alpha),
            sigma_read_sq=sigma_read_sq,
            variance_epsilon=float(variance_epsilon),
        )
        max_curvature = max(max_curvature, float(np.max(np.abs(curvature))))

    return float(max(operator_norm_sq * max_curvature * float(safety_scale), 1e-6))


class ExactPoissonGaussianFunction(_FunctionBase):
    def __init__(
        self,
        observed: np.ndarray,
        psf: np.ndarray,
        alpha: float,
        sigma_read: float,
        geometry,
        background: np.ndarray | float = 0.0,
        variance_epsilon: float = 1e-8,
        operator_norm_iterations: int = 25,
        curvature_grid_points: int = 33,
        lipschitz_safety_scale: float = 1.25,
    ):
        require_cil_core()

        self.geometry = geometry
        self.observed = np.asarray(observed, dtype=np.float32)
        self.psf = np.asarray(psf, dtype=np.float32)
        self.psf_mirror = np.flip(self.psf)
        self.alpha = float(alpha)
        self.sigma_read = float(sigma_read)
        self.sigma_read_sq = float(self.sigma_read * self.sigma_read)
        self.background = np.asarray(background, dtype=np.float32)
        self.variance_epsilon = float(max(variance_epsilon, 1e-12))

        lipschitz = estimate_poisson_gaussian_lipschitz(
            observed=self.observed,
            psf=self.psf,
            alpha=self.alpha,
            sigma_read=self.sigma_read,
            background=self.background,
            variance_epsilon=self.variance_epsilon,
            operator_norm_iterations=operator_norm_iterations,
            curvature_grid_points=curvature_grid_points,
            safety_scale=lipschitz_safety_scale,
        )
        super().__init__(L=lipschitz)

    def __call__(self, x):
        image = np.asarray(x.as_array(), dtype=np.float32)
        forward_model = forward_convolve2d(image, self.psf) + self._background_for(image)
        variance = np.maximum(self.alpha * forward_model + self.sigma_read_sq, self.variance_epsilon)
        residual = self.observed - forward_model
        return float(0.5 * np.sum((residual * residual) / variance + np.log(variance)))

    def gradient(self, x, out: Optional[ImageData] = None):
        image = np.asarray(x.as_array(), dtype=np.float32)
        forward_model = forward_convolve2d(image, self.psf) + self._background_for(image)
        variance = np.maximum(self.alpha * forward_model + self.sigma_read_sq, self.variance_epsilon)
        residual = self.observed - forward_model
        grad_forward = ((forward_model - self.observed) / variance) + (
            0.5 * self.alpha * (1.0 / variance - (residual * residual) / (variance * variance))
        )
        grad = adjoint_convolve2d(grad_forward, self.psf_mirror).astype(np.float32, copy=False)
        if out is None:
            return ImageData(array=grad, geometry=self.geometry)
        out.fill(grad)
        return out

    def _background_for(self, image: np.ndarray) -> np.ndarray:
        if self.background.ndim == 0:
            return np.full_like(image, float(self.background))
        if self.background.shape == image.shape:
            return self.background.astype(np.float32, copy=False)
        raise ValueError(
            f"Background shape {self.background.shape} is incompatible with image shape {image.shape}."
        )
