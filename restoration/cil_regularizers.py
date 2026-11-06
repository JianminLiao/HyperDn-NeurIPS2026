from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np

try:
    from cil.framework import ImageData
    from cil.optimisation.functions import Function, L1Norm
    from cil.plugins.ccpi_regularisation.functions import FGP_TV, TGV as CCPiTGV
except ImportError:
    ImageData = None
    Function = None
    L1Norm = None
    FGP_TV = None
    CCPiTGV = None

try:
    import proxop
except ImportError:
    proxop = None

from .cil_pg import require_cil_core

_FunctionBase = Function if Function is not None else object


@dataclass
class CILRegularizerSpec:
    kind: str
    function: Any
    is_convex: bool
    supports_objective_value: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def require_cil_regularizers(needs_ccpi: bool = False, needs_proxop: bool = False):
    require_cil_core()
    if ImageData is None or Function is None or L1Norm is None:
        raise ImportError(
            "Missing CIL optimisation components. Install `cil` with optimisation support "
            "to use the packaged regularizer factories."
        )
    if needs_ccpi and (FGP_TV is None or CCPiTGV is None):
        raise ImportError(
            "Missing dependency `ccpi-regulariser`. Install the CCPi regularisation plugin "
            "to use TV/TGV regularizers."
        )
    if needs_proxop and proxop is None:
        raise ImportError(
            "Missing dependency `proxop`. Install it to use MCP/L^p proximal regularizers."
        )


class _NonNegClippedFunction(_FunctionBase):
    """Wraps a CIL regulariser proximal with a [0, upper] box projection.

    CCPi's TGV has no ``nonnegativity`` flag (unlike FGP_TV).  For Poisson /
    Poisson-Gaussian NLL data terms the iterates *must* stay non-negative,
    otherwise the variance model breaks and PSNR collapses.  This wrapper
    applies ``np.clip(result, 0, upper)`` after every proximal evaluation.
    """

    def __init__(self, inner, upper_bound: float = 1.0):
        super().__init__(L=None)
        self._inner = inner
        self._upper = float(upper_bound) if upper_bound is not None else None

    def __call__(self, x):
        return self._inner(x)

    def proximal(self, x, tau, out=None):
        result = self._inner.proximal(x, tau, out=out)
        target = out if out is not None else result
        arr = target.as_array()
        if self._upper is not None:
            np.clip(arr, 0.0, self._upper, out=arr)
        else:
            np.maximum(arr, 0.0, out=arr)
        target.fill(arr)
        return result  # None when out was provided, DataContainer otherwise

    def convex_conjugate(self, x):
        return self._inner.convex_conjugate(x)


def build_cil_regularizer(
    regularizer_kind: str,
    geometry,
    param: Any,
    clip: bool = True,
    regularizer_max_iter: int = 100,
    regularizer_tolerance: float = 0.0,
    ccpi_device: str = "cpu",
) -> CILRegularizerSpec:
    kind = str(regularizer_kind).strip().upper()

    if kind == "TV":
        require_cil_regularizers(needs_ccpi=True)
        reg_param = _normalize_tv_param(param)
        return CILRegularizerSpec(
            kind=kind,
            function=FGP_TV(
                alpha=float(reg_param["lambda"]),
                max_iteration=int(max(1, regularizer_max_iter)),
                tolerance=float(max(0.0, regularizer_tolerance)),
                isotropic=True,
                nonnegativity=bool(clip),
                device=str(ccpi_device).strip().lower() or "cpu",
            ),
            is_convex=True,
            supports_objective_value=True,
            metadata=reg_param,
        )

    if kind == "TGV":
        require_cil_regularizers(needs_ccpi=True)
        reg_param = _normalize_tgv_param(param)
        tgv_fn = CCPiTGV(
            alpha=float(reg_param["alpha"]),
            gamma=float(reg_param["gamma"]),
            max_iteration=int(max(1, regularizer_max_iter)),
            tolerance=float(max(0.0, regularizer_tolerance)),
            device=str(ccpi_device).strip().lower() or "cpu",
        )
        # CCPi TGV has no nonnegativity flag (unlike FGP_TV).
        # Without box-clipping the proximal output, FISTA iterates go negative
        # and Poisson/PG NLL gradients explode, collapsing PSNR.
        if clip:
            tgv_fn = _NonNegClippedFunction(tgv_fn, upper_bound=1.0)
        return CILRegularizerSpec(
            kind=kind,
            function=tgv_fn,
            is_convex=True,
            supports_objective_value=False,
            metadata=reg_param,
        )

    if kind == "L1":
        require_cil_regularizers()
        reg_param = _normalize_scalar_lambda_param(param, name="L1 lambda")
        return CILRegularizerSpec(
            kind=kind,
            function=float(reg_param["lambda"]) * L1Norm(),
            is_convex=True,
            supports_objective_value=True,
            metadata=reg_param,
        )

    if kind == "L0":
        require_cil_regularizers(needs_proxop=True)
        reg_param = _normalize_scalar_lambda_param(param, name="L0 lambda")
        function = ProxOpElementwiseFunction(
            geometry=geometry,
            prox=_build_l0_prox(weight=float(reg_param["lambda"])),
            value_fn=_l0_value(float(reg_param["lambda"])),
            name="L0",
        )
        return CILRegularizerSpec(
            kind=kind,
            function=function,
            is_convex=False,
            supports_objective_value=True,
            metadata=reg_param,
        )

    if kind == "MCP":
        require_cil_regularizers(needs_proxop=True)
        reg_param = _normalize_mcp_param(param)
        function = ProxOpElementwiseFunction(
            geometry=geometry,
            prox=_build_mcp_prox(lambda_value=float(reg_param["lambda"]), beta=float(reg_param["beta"])),
            value_fn=_mcp_value(float(reg_param["lambda"]), float(reg_param["beta"])),
            name="MCP",
        )
        return CILRegularizerSpec(
            kind=kind,
            function=function,
            is_convex=False,
            supports_objective_value=True,
            metadata=reg_param,
        )

    if kind == "LP":
        require_cil_regularizers(needs_proxop=True)
        reg_param = _normalize_lp_param(param)
        function = ProxOpElementwiseFunction(
            geometry=geometry,
            prox=_build_lp_prox(weight=float(reg_param["lambda"]), power=float(reg_param["p"])),
            value_fn=_lp_value(float(reg_param["lambda"]), float(reg_param["p"])),
            name="Lp",
        )
        return CILRegularizerSpec(
            kind=kind,
            function=function,
            is_convex=False,
            supports_objective_value=True,
            metadata=reg_param,
        )

    raise ValueError(
        f"Unsupported regularizer_kind={regularizer_kind}. Expected one of ['TV', 'TGV', 'L1', 'MCP', 'LP', 'L0']."
    )


class ProxOpElementwiseFunction(_FunctionBase):
    def __init__(
        self,
        geometry,
        prox: Callable[[np.ndarray, float], np.ndarray],
        value_fn: Callable[[np.ndarray], float],
        name: str,
    ):
        require_cil_regularizers()
        super().__init__(L=None)
        self.geometry = geometry
        self._prox = prox
        self._value_fn = value_fn
        self.name = str(name)

    def __call__(self, x):
        arr = np.asarray(x.as_array(), dtype=np.float32)
        return float(self._value_fn(arr))

    def proximal(self, x, tau, out=None):
        arr = np.asarray(x.as_array(), dtype=np.float32)
        prox_arr = np.asarray(self._prox(arr, float(tau)), dtype=np.float32)
        if out is None:
            return ImageData(array=prox_arr, geometry=self.geometry)
        out.fill(prox_arr)
        return out


def _normalize_tv_param(param: Any) -> Dict[str, float]:
    return _normalize_scalar_lambda_param(param, name="TV lambda")


def _normalize_tgv_param(param: Any) -> Dict[str, float]:
    if isinstance(param, dict):
        alpha = float(param["alpha"] if "alpha" in param else param["lambda"])
        gamma = float(param["gamma"])
    elif isinstance(param, (tuple, list)) and len(param) == 2:
        alpha = float(param[0])
        gamma = float(param[1])
    else:
        raise ValueError(
            "TGV parameter must be a dict with keys {'lambda','gamma'} / {'alpha','gamma'} or a 2-tuple."
        )
    if alpha < 0.0:
        raise ValueError(f"TGV alpha must be >= 0, got {alpha}.")
    if gamma <= 0.0:
        raise ValueError(f"TGV gamma must be > 0, got {gamma}.")
    if gamma < 1.0 or gamma > 2.0:
        import warnings
        warnings.warn(
            f"CCPi TGV gamma {gamma:.6f} is outside [1, 2]; clamping to valid range.",
            RuntimeWarning, stacklevel=2,
        )
        gamma = max(1.0, min(2.0, gamma))
    return {"alpha": alpha, "gamma": gamma}


def _normalize_mcp_param(param: Any) -> Dict[str, float]:
    if isinstance(param, dict):
        lambda_value = float(param["lambda"])
        beta = float(param["beta"])
    elif isinstance(param, (tuple, list)) and len(param) == 2:
        lambda_value = float(param[0])
        beta = float(param[1])
    else:
        raise ValueError("MCP parameter must be a dict {'lambda','beta'} or a 2-tuple (lambda, beta).")
    if lambda_value < 0.0:
        raise ValueError(f"MCP lambda must be >= 0, got {lambda_value}.")
    if beta <= 0.0:
        raise ValueError(f"MCP beta must be > 0, got {beta}.")
    return {"lambda": lambda_value, "beta": beta}


def _normalize_lp_param(param: Any) -> Dict[str, float]:
    if isinstance(param, dict):
        lambda_value = float(param["lambda"])
        power = float(param["p"])
    elif isinstance(param, (tuple, list)) and len(param) == 2:
        lambda_value = float(param[0])
        power = float(param[1])
    else:
        raise ValueError("LP parameter must be a dict {'lambda','p'} or a 2-tuple (lambda, p).")
    if lambda_value < 0.0:
        raise ValueError(f"LP lambda must be >= 0, got {lambda_value}.")
    if not (0.0 < power < 1.0):
        raise ValueError(f"LP power must satisfy 0 < p < 1, got {power}.")
    return {"lambda": lambda_value, "p": power}


def _normalize_scalar_lambda_param(param: Any, name: str) -> Dict[str, float]:
    value = float(param["lambda"] if isinstance(param, dict) else param)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return {"lambda": value}


def _build_mcp_prox(lambda_value: float, beta: float) -> Callable[[np.ndarray, float], np.ndarray]:
    prox_class = _resolve_proxop_class("MCP")
    prox_instance = _instantiate_proxop(
        prox_class,
        {
            "lambda_": lambda_value,
            "lamb": lambda_value,
            "lam": lambda_value,
            "beta": beta,
            "a": beta,
        },
    )
    return lambda x, tau: _call_prox(prox_instance, x, gamma=tau)


def _build_l0_prox(weight: float) -> Callable[[np.ndarray, float], np.ndarray]:
    prox_class = _resolve_proxop_class("L0Norm")
    prox_instance = _instantiate_proxop(prox_class, {"tol": 1e-10})
    return lambda x, tau: _call_prox(prox_instance, x, gamma=tau * weight)


def _build_lp_prox(weight: float, power: float) -> Callable[[np.ndarray, float], np.ndarray]:
    prox_class = _resolve_proxop_class("Root")
    prox_instance = _instantiate_proxop(
        prox_class,
        {
            "q": power,
            "p": power,
            "power": power,
        },
    )
    return lambda x, tau: _call_prox(prox_instance, x, gamma=tau * weight)


def _mcp_value(lambda_value: float, beta: float) -> Callable[[np.ndarray], float]:
    def evaluate(x: np.ndarray) -> float:
        abs_x = np.abs(np.asarray(x, dtype=np.float64))
        threshold = float(beta * lambda_value)
        quadratic = lambda_value * abs_x - (abs_x * abs_x) / (2.0 * beta)
        saturated = 0.5 * beta * lambda_value * lambda_value
        value = np.where(abs_x <= threshold, quadratic, saturated)
        return float(np.sum(value))

    return evaluate


def _l0_value(weight: float, tol: float = 1e-10) -> Callable[[np.ndarray], float]:
    def evaluate(x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=np.float64)
        return float(weight * np.count_nonzero(np.abs(arr) > float(tol)))

    return evaluate


def _lp_value(weight: float, power: float) -> Callable[[np.ndarray], float]:
    def evaluate(x: np.ndarray) -> float:
        return float(weight * np.sum(np.abs(np.asarray(x, dtype=np.float64)) ** power))

    return evaluate


def _resolve_proxop_class(name: str):
    require_cil_regularizers(needs_proxop=True)
    try:
        return getattr(proxop, name)
    except AttributeError as exc:
        raise ImportError(f"proxop is installed but does not expose class {name}.") from exc


def _instantiate_proxop(cls, candidate_kwargs: Dict[str, Any]):
    signature = inspect.signature(cls)
    kwargs = {key: value for key, value in candidate_kwargs.items() if key in signature.parameters}
    try:
        return cls(**kwargs)
    except TypeError:
        filtered = {key: value for key, value in kwargs.items() if key != "lambda_"}
        if filtered != kwargs:
            return cls(**filtered)
        raise


def _call_prox(instance, x: np.ndarray, gamma: float) -> np.ndarray:
    return np.asarray(instance.prox(np.asarray(x, dtype=np.float64), gamma=float(gamma)), dtype=np.float32)
