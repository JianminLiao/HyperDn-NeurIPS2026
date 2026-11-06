from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:
    from cil.framework import ImageData
    from cil.optimisation.functions import Function, KullbackLeibler, L1Norm, LeastSquares, OperatorCompositionFunction
except ImportError:
    Function = None
    ImageData = None
    KullbackLeibler = None
    L1Norm = None
    LeastSquares = None
    OperatorCompositionFunction = None

from .cil_ops import KnownPSFConvolutionOperator
from .cil_pg import ExactPoissonGaussianFunction, require_cil_core
from .noise_models import DEFAULT_NOISE_FOR_DATA_TERM

_FunctionBase = Function if Function is not None else object


@dataclass
class CILDataTermSpec:
    kind: str
    noise_model: str
    function: Any = None
    algorithm_family: str = "smooth"
    pdhg_function: Any = None
    operator: Any = None
    is_convex: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def require_cil_data_terms():
    require_cil_core()
    if (
        ImageData is None
        or Function is None
        or KullbackLeibler is None
        or L1Norm is None
        or LeastSquares is None
        or OperatorCompositionFunction is None
    ):
        raise ImportError(
            "Missing CIL optimisation functions. Install `cil` with optimisation support "
            "to use the packaged data-term factories."
        )


def build_known_psf_operator(geometry, psf: np.ndarray, norm_iterations: int = 25):
    require_cil_data_terms()
    return KnownPSFConvolutionOperator(geometry=geometry, psf=psf, norm_iterations=norm_iterations)


def build_cil_data_term(
    data_term_kind: str,
    geometry,
    operator,
    observed: np.ndarray,
    background: float | np.ndarray = 0.0,
    alpha: Optional[float] = None,
    sigma_read: Optional[float] = None,
    delta: Optional[float] = None,
    variance_epsilon: float = 1e-8,
    operator_norm_iterations: int = 25,
    curvature_grid_points: int = 33,
    lipschitz_safety_scale: float = 1.25,
    poisson_backend: str = "numba",
) -> CILDataTermSpec:
    require_cil_data_terms()

    kind = str(data_term_kind).strip().upper()
    observed_arr = np.asarray(observed, dtype=np.float32)
    background_arr = _background_for_shape(background, observed_arr.shape)
    observed_data = ImageData(array=observed_arr, geometry=geometry)

    if kind == "L2":
        centered_data = ImageData(array=(observed_arr - background_arr).astype(np.float32), geometry=geometry)
        return CILDataTermSpec(
            kind=kind,
            noise_model=DEFAULT_NOISE_FOR_DATA_TERM[kind],
            function=LeastSquares(A=operator, b=centered_data, c=0.5),
            algorithm_family="smooth",
            operator=operator,
            metadata={"background_mode": "subtract_observed"},
        )

    if kind == "L1":
        centered_data = ImageData(array=(observed_arr - background_arr).astype(np.float32), geometry=geometry)
        return CILDataTermSpec(
            kind=kind,
            noise_model=DEFAULT_NOISE_FOR_DATA_TERM[kind],
            algorithm_family="pdhg",
            pdhg_function=L1Norm(b=centered_data),
            operator=operator,
            metadata={"background_mode": "subtract_observed"},
        )

    if kind == "HUBER":
        centered_data = ImageData(array=(observed_arr - background_arr).astype(np.float32), geometry=geometry)
        huber_delta = float(1.0 / 255.0 if delta is None else delta)
        return CILDataTermSpec(
            kind=kind,
            noise_model="UNKNOWN",
            function=OperatorCompositionFunction(
                HuberLoss(geometry=geometry, b=centered_data, delta=huber_delta),
                operator,
            ),
            algorithm_family="smooth",
            operator=operator,
            metadata={
                "background_mode": "subtract_observed",
                "delta": huber_delta,
            },
        )

    if kind == "POISSON":
        eta = ImageData(array=np.maximum(background_arr, 0.0).astype(np.float32), geometry=geometry)
        fidelity = KullbackLeibler(b=observed_data, eta=eta, backend=poisson_backend)
        return CILDataTermSpec(
            kind=kind,
            noise_model=DEFAULT_NOISE_FOR_DATA_TERM[kind],
            function=OperatorCompositionFunction(fidelity, operator),
            algorithm_family="smooth",
            operator=operator,
            metadata={"background_mode": "eta"},
        )

    if kind == "POISSON_GAUSSIAN":
        if alpha is None or sigma_read is None:
            raise ValueError("POISSON_GAUSSIAN data term requires both alpha and sigma_read.")
        return CILDataTermSpec(
            kind=kind,
            noise_model=DEFAULT_NOISE_FOR_DATA_TERM[kind],
            function=ExactPoissonGaussianFunction(
                observed=observed_arr,
                psf=np.asarray(operator.psf, dtype=np.float32),
                alpha=float(alpha),
                sigma_read=float(sigma_read),
                geometry=geometry,
                background=background_arr,
                variance_epsilon=variance_epsilon,
                operator_norm_iterations=operator_norm_iterations,
                curvature_grid_points=curvature_grid_points,
                lipschitz_safety_scale=lipschitz_safety_scale,
            ),
            algorithm_family="smooth",
            operator=operator,
            metadata={"background_mode": "explicit_pg"},
        )

    raise ValueError(
        f"Unsupported data_term_kind={data_term_kind}. Expected one of "
        f"{sorted(list(DEFAULT_NOISE_FOR_DATA_TERM) + ['HUBER'])}."
    )


def _background_for_shape(background: float | np.ndarray, shape) -> np.ndarray:
    background_arr = np.asarray(background, dtype=np.float32)
    if background_arr.ndim == 0:
        return np.full(shape, float(background_arr), dtype=np.float32)
    if background_arr.shape == shape:
        return background_arr.astype(np.float32, copy=False)
    raise ValueError(f"Background shape {background_arr.shape} is incompatible with image shape {shape}.")


class HuberLoss(_FunctionBase):
    def __init__(self, geometry, b, delta: float):
        require_cil_data_terms()
        self.geometry = geometry
        self.b = b
        self.delta = float(max(float(delta), 1e-8))
        super().__init__(L=1.0)

    def __call__(self, x):
        residual = np.asarray(x.as_array(), dtype=np.float32) - np.asarray(self.b.as_array(), dtype=np.float32)
        abs_residual = np.abs(residual)
        loss = np.where(
            abs_residual <= self.delta,
            0.5 * residual * residual,
            self.delta * (abs_residual - (0.5 * self.delta)),
        )
        return float(np.sum(loss, dtype=np.float64))

    def gradient(self, x, out=None):
        residual = np.asarray(x.as_array(), dtype=np.float32) - np.asarray(self.b.as_array(), dtype=np.float32)
        grad = np.clip(residual, -self.delta, self.delta).astype(np.float32, copy=False)
        if out is None:
            return ImageData(array=grad, geometry=self.geometry)
        out.fill(grad)
        return out
