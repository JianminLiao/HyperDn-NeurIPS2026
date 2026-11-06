from abc import ABC, abstractmethod
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import convolve
from skimage.restoration import denoise_tv_bregman

try:
    from cil.framework import ImageData, ImageGeometry
    from cil.optimisation.algorithms import FISTA, ISTA, PDHG
    from cil.optimisation.utilities.callbacks import EarlyStoppingObjectiveValue
except ImportError:
    ImageData = None
    ImageGeometry = None
    FISTA = None
    ISTA = None
    PDHG = None
    EarlyStoppingObjectiveValue = None

from .cil_pg import require_cil_core
from .cil_regularizers import build_cil_regularizer
from .cil_terms import build_cil_data_term, build_known_psf_operator


def _clamp_tv_grid_lambda(value: float) -> float:
    """Clamp TV regularization weight to the problem grid used for labels/training."""
    from .config import ProblemConfig

    return float(np.clip(float(value), float(ProblemConfig.LAM_MIN), float(ProblemConfig.LAM_MAX)))


def _require_cil_algorithms():
    if (
        ImageData is None
        or ImageGeometry is None
        or ISTA is None
        or FISTA is None
        or PDHG is None
        or EarlyStoppingObjectiveValue is None
    ):
        raise ImportError(
            "Missing dependency `cil` with optimisation utilities. "
            "Install the Core Imaging Library to use the CIL-based deblurring solvers."
        )


class EarlyStoppingRelativeUpdate:
    """
    Objective-free stopping rule based on relative iterate change.

    This is used for CCPi TGV because the packaged regularizer exposes a proximal
    operator but does not implement a finite objective value.
    """

    def __init__(self, threshold: float, min_iterations: int = 5, epsilon: float = 1e-8):
        self.threshold = float(max(threshold, 0.0))
        self.min_iterations = int(max(0, min_iterations))
        self.epsilon = float(max(epsilon, 1e-12))

    def __call__(self, algorithm) -> None:
        if self.threshold <= 0.0:
            return
        if int(getattr(algorithm, "iteration", -1)) < self.min_iterations:
            return
        current = getattr(algorithm, "x_old", None)
        previous = getattr(algorithm, "x", None)
        if current is None or previous is None:
            return
        current_arr = np.asarray(current.as_array(), dtype=np.float32)
        previous_arr = np.asarray(previous.as_array(), dtype=np.float32)
        diff_norm = float(np.linalg.norm((current_arr - previous_arr).ravel()))
        current_norm = float(np.linalg.norm(current_arr.ravel()))
        relative_change = diff_norm / max(current_norm, self.epsilon)
        if np.isfinite(relative_change) and relative_change <= self.threshold:
            raise StopIteration


class AbstractSolver(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def solve(self, noisy_image: np.ndarray, param: Any) -> np.ndarray:
        pass


class NormalizedSplitBregmanTVSolver(AbstractSolver):
    """
    Normalized TV Denoising Solver.
    Energy: 0.5 * ||u - f||^2 + lambda * TV(u)
    Mapping: weight = 1 / (2 * lambda)
    """

    def get_name(self) -> str:
        return "NormalizedSplitBregmanTV"

    def param_to_internal(self, user_param: float) -> float:
        if user_param < 1e-9:
            return 1e9
        return 1.0 / (2.0 * user_param)

    def solve(self, noisy_image: np.ndarray, param: float) -> np.ndarray:
        param = _clamp_tv_grid_lambda(float(param))
        weight = self.param_to_internal(param)
        return denoise_tv_bregman(noisy_image, weight=weight, isotropic=True, channel_axis=0)


class L1ADMMTVDenoiseSolver(AbstractSolver):
    """
    Periodic ADMM solver for TV-L1 denoising.
    Energy: ||u - f||_1 + lambda * TV(u)
    """

    def __init__(self, max_iter: int = 100, rho_tv: float = 2.0, rho_data: float = 5.0, clip: bool = True):
        self.max_iter = int(max_iter)
        self.rho_tv = float(rho_tv)
        self.rho_data = float(rho_data)
        self.clip = bool(clip)

    def get_name(self) -> str:
        return "L1ADMMTVDenoise"

    @staticmethod
    def _grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = np.roll(x, -1, axis=1) - x
        gy = np.roll(x, -1, axis=0) - x
        return gx, gy

    @staticmethod
    def _div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))

    @staticmethod
    def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - float(thresh), 0.0)

    @staticmethod
    def _vector_shrink(gx: np.ndarray, gy: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray]:
        norm = np.sqrt(gx * gx + gy * gy)
        scale = np.maximum(1.0 - float(thresh) / np.maximum(norm, 1e-8), 0.0)
        return gx * scale, gy * scale

    def _solve_channel(self, noisy: np.ndarray, lam: float) -> np.ndarray:
        noisy = np.asarray(noisy, dtype=np.float32)
        height, width = noisy.shape
        wx = 2.0 * np.pi * np.arange(width, dtype=np.float32) / float(width)
        wy = 2.0 * np.pi * np.arange(height, dtype=np.float32) / float(height)
        eig = (
            4.0
            - 2.0 * np.cos(wx)[None, :]
            - 2.0 * np.cos(wy)[:, None]
        ).astype(np.float32)
        denom = (self.rho_data + self.rho_tv * eig).astype(np.float32)

        u = noisy.copy()
        dx = np.zeros_like(noisy, dtype=np.float32)
        dy = np.zeros_like(noisy, dtype=np.float32)
        vx = np.zeros_like(noisy, dtype=np.float32)
        bx = np.zeros_like(noisy, dtype=np.float32)
        by = np.zeros_like(noisy, dtype=np.float32)
        bv = np.zeros_like(noisy, dtype=np.float32)

        for _ in range(self.max_iter):
            rhs = self.rho_data * (noisy + vx - bv) - self.rho_tv * self._div(dx - bx, dy - by)
            u = np.fft.ifft2(np.fft.fft2(rhs) / denom).real.astype(np.float32)
            if self.clip:
                u = np.clip(u, 0.0, 1.0)

            gx, gy = self._grad(u)
            dx, dy = self._vector_shrink(gx + bx, gy + by, lam / self.rho_tv)
            vx = self._soft_threshold(u - noisy + bv, 1.0 / self.rho_data)

            bx = bx + gx - dx
            by = by + gy - dy
            bv = bv + u - noisy - vx

        if self.clip:
            u = np.clip(u, 0.0, 1.0)
        return u.astype(np.float32, copy=False)

    def solve(self, noisy_image: np.ndarray, param: float) -> np.ndarray:
        noisy = np.asarray(noisy_image, dtype=np.float32)
        lam = max(float(param), 0.0)
        if noisy.ndim == 2:
            return self._solve_channel(noisy, lam)
        if noisy.ndim == 3:
            restored = np.empty_like(noisy, dtype=np.float32)
            for channel_idx in range(noisy.shape[0]):
                restored[channel_idx] = self._solve_channel(noisy[channel_idx], lam)
            return restored
        raise ValueError(f"Expected [H,W] or [C,H,W], got {noisy.shape}.")


class HuberADMMTVDenoiseSolver(AbstractSolver):
    """
    Periodic ADMM solver for Huber-data TV denoising.
    Energy: sum huber_delta(u - f) + lambda * TV(u)
    Delta can either be set explicitly or derived from an external Gaussian noise estimate.
    """

    def __init__(
        self,
        max_iter: int = 100,
        rho_tv: float = 2.0,
        rho_data: float = 5.0,
        delta_scale: float = 1.0,
        min_delta: float = 1.0 / 255.0,
        clip: bool = True,
    ):
        self.max_iter = int(max_iter)
        self.rho_tv = float(rho_tv)
        self.rho_data = float(rho_data)
        self.delta_scale = float(delta_scale)
        self.min_delta = float(min_delta)
        self.clip = bool(clip)
        self._sigma_estimate = float(self.min_delta)

    def get_name(self) -> str:
        return "HuberADMMTVDenoise"

    def set_sigma_estimate(self, sigma_est: float) -> None:
        self._sigma_estimate = float(max(float(sigma_est), self.min_delta))

    @staticmethod
    def _grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = np.roll(x, -1, axis=1) - x
        gy = np.roll(x, -1, axis=0) - x
        return gx, gy

    @staticmethod
    def _div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))

    @staticmethod
    def _vector_shrink(gx: np.ndarray, gy: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray]:
        norm = np.sqrt(gx * gx + gy * gy)
        scale = np.maximum(1.0 - float(thresh) / np.maximum(norm, 1e-8), 0.0)
        return gx * scale, gy * scale

    @staticmethod
    def _prox_huber(y: np.ndarray, tau: float, delta: float) -> np.ndarray:
        abs_y = np.abs(y)
        threshold = delta * (1.0 + tau)
        quadratic = y / (1.0 + tau)
        linear = y - tau * delta * np.sign(y)
        return np.where(abs_y <= threshold, quadratic, linear).astype(np.float32, copy=False)

    def _solve_channel(self, noisy: np.ndarray, lam: float, delta: float) -> np.ndarray:
        noisy = np.asarray(noisy, dtype=np.float32)
        height, width = noisy.shape
        wx = 2.0 * np.pi * np.arange(width, dtype=np.float32) / float(width)
        wy = 2.0 * np.pi * np.arange(height, dtype=np.float32) / float(height)
        eig = (
            4.0
            - 2.0 * np.cos(wx)[None, :]
            - 2.0 * np.cos(wy)[:, None]
        ).astype(np.float32)
        denom = (self.rho_data + self.rho_tv * eig).astype(np.float32)

        u = noisy.copy()
        dx = np.zeros_like(noisy, dtype=np.float32)
        dy = np.zeros_like(noisy, dtype=np.float32)
        vx = np.zeros_like(noisy, dtype=np.float32)
        bx = np.zeros_like(noisy, dtype=np.float32)
        by = np.zeros_like(noisy, dtype=np.float32)
        bv = np.zeros_like(noisy, dtype=np.float32)

        for _ in range(self.max_iter):
            rhs = self.rho_data * (noisy + vx - bv) - self.rho_tv * self._div(dx - bx, dy - by)
            u = np.fft.ifft2(np.fft.fft2(rhs) / denom).real.astype(np.float32)
            if self.clip:
                u = np.clip(u, 0.0, 1.0)

            gx, gy = self._grad(u)
            dx, dy = self._vector_shrink(gx + bx, gy + by, lam / self.rho_tv)
            vx = self._prox_huber(u - noisy + bv, 1.0 / self.rho_data, delta)

            bx = bx + gx - dx
            by = by + gy - dy
            bv = bv + u - noisy - vx

        if self.clip:
            u = np.clip(u, 0.0, 1.0)
        return u.astype(np.float32, copy=False)

    def solve(self, noisy_image: np.ndarray, param: Any) -> np.ndarray:
        noisy = np.asarray(noisy_image, dtype=np.float32)
        explicit_delta = None
        if isinstance(param, dict):
            lam = max(float(param["lambda"]), 0.0)
            if "delta" in param:
                explicit_delta = max(float(param["delta"]), 1e-8)
        else:
            lam = max(float(param), 0.0)
        delta = (
            float(explicit_delta)
            if explicit_delta is not None
            else max(float(self._sigma_estimate) * self.delta_scale, self.min_delta)
        )
        if noisy.ndim == 2:
            return self._solve_channel(noisy, lam, delta)
        if noisy.ndim == 3:
            restored = np.empty_like(noisy, dtype=np.float32)
            for channel_idx in range(noisy.shape[0]):
                restored[channel_idx] = self._solve_channel(noisy[channel_idx], lam, delta)
            return restored
        raise ValueError(f"Expected [H,W] or [C,H,W], got {noisy.shape}.")


class CCPiTGVExternalDenoiseSolver(AbstractSolver):
    """
    Thin wrapper around the CCPi TGV denoiser, invoked as an out-of-process helper.
    Energy is controlled by lambda plus a ratio-like gamma term where alpha0 = gamma * lambda.
    The interpreter used to run the helper is resolved from the ``TGV_HELPER_PYTHON``
    environment variable, falling back to ``sys.executable``.
    """

    def __init__(
        self,
        python_executable: Optional[str] = None,
        helper_script: Optional[str] = None,
        alpha0_ratio: float = 2.0,
        iterations: int = 120,
        l2: float = 1.0,
        eps: float = 1e-6,
    ):
        if python_executable is None:
            python_executable = os.environ.get("TGV_HELPER_PYTHON", sys.executable)
        if helper_script is None:
            helper_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "ccpi_tgv_denoise_helper.py",
            )
        self.python_executable = str(python_executable)
        self.helper_script = str(helper_script)
        self.alpha0_ratio = float(alpha0_ratio)
        self.iterations = int(iterations)
        self.l2 = float(l2)
        self.eps = float(eps)

    def get_name(self) -> str:
        return "CCPiTGVExternalDenoise"

    def solve(self, noisy_image: np.ndarray, param: Any) -> np.ndarray:
        noisy = np.asarray(noisy_image, dtype=np.float32)
        if isinstance(param, dict):
            lam = float(param["lambda"] if "lambda" in param else param["alpha"])
            gamma = float(param.get("gamma", self.alpha0_ratio))
        else:
            lam = float(param)
            gamma = float(self.alpha0_ratio)
        alpha1 = lam
        alpha0 = max(gamma * lam, 1e-8)
        with tempfile.TemporaryDirectory(prefix="ccpi_tgv_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_path = tmp_root / "input.npy"
            output_path = tmp_root / "output.npy"
            np.save(input_path, noisy)
            command = [
                self.python_executable,
                self.helper_script,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--lambda",
                str(lam),
                "--alpha1",
                str(alpha1),
                "--alpha0",
                str(alpha0),
                "--iterations",
                str(self.iterations),
                "--l2",
                str(self.l2),
                "--eps",
                str(self.eps),
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            restored = np.asarray(np.load(output_path), dtype=np.float32)
        if noisy.ndim == 2 and restored.ndim == 3 and restored.shape[0] == 1:
            return restored[0]
        return restored


class CILKnownPSFDeblurSolver(AbstractSolver):
    """
    Known-PSF deblurring with CIL as the optimisation framework.

    Data-fidelity terms are assembled through packaged CIL functions when available,
    with the exact Poisson-Gaussian term provided by the local adapter. Regularizers
    are delegated to CCPi/CIL/proxop backends.
    """

    _DEFAULT_OPERATOR_NORM_ITERATIONS = 25
    _DEFAULT_CURVATURE_GRID_POINTS = 33
    _DEFAULT_LIPSCHITZ_SAFETY_SCALE = 1.25
    _DEFAULT_OBJECTIVE_UPDATE_INTERVAL = 1

    def __init__(
        self,
        psf: np.ndarray,
        data_term_kind: str,
        regularizer_kind: str,
        data_term_options: Optional[Dict[str, Any]] = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        clip: bool = True,
        variance_epsilon: float = 1e-8,
        regularizer_max_iter: int = 100,
        regularizer_tolerance: float = 0.0,
        ccpi_device: str = "cpu",
    ):
        _require_cil_algorithms()
        psf_arr = np.asarray(psf, dtype=np.float32)
        if psf_arr.ndim != 2:
            raise ValueError(f"CILKnownPSFDeblurSolver expects a 2D PSF, got {psf_arr.shape}.")
        psf_arr = np.maximum(psf_arr, 0.0)
        psf_sum = float(np.sum(psf_arr))
        if not np.isfinite(psf_sum) or psf_sum <= 0.0:
            raise ValueError("PSF must have a finite positive sum.")

        self.psf = (psf_arr / psf_sum).astype(np.float32, copy=False)
        self.data_term_kind = str(data_term_kind).strip().upper()
        self.data_term_options = dict(data_term_options or {})
        self.background = np.asarray(self.data_term_options.get("background", 0.0), dtype=np.float32)
        self.regularizer_kind = str(regularizer_kind).strip().upper()
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.operator_norm_iterations = self._DEFAULT_OPERATOR_NORM_ITERATIONS
        self.curvature_grid_points = self._DEFAULT_CURVATURE_GRID_POINTS
        self.lipschitz_safety_scale = self._DEFAULT_LIPSCHITZ_SAFETY_SCALE
        self.clip = bool(clip)
        self.variance_epsilon = float(max(variance_epsilon, 1e-12))
        self.regularizer_max_iter = int(max(1, regularizer_max_iter))
        self.regularizer_tolerance = float(max(0.0, regularizer_tolerance))
        self.ccpi_device = str(ccpi_device).strip().lower() or "cpu"

    def get_name(self) -> str:
        return f"CIL{self.data_term_kind}{self.regularizer_kind}KnownPSFDeblur"

    def solve(self, noisy_image: np.ndarray, param: float) -> np.ndarray:
        return self.solve_with_param(noisy_image, param=param, initial_estimate=None)

    def solve_with_param(
        self,
        noisy_image: np.ndarray,
        param: Any,
        initial_estimate: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        observed = self._to_numpy_chw(noisy_image)
        regularizer_param = self._normalize_regularizer_param(param)
        if initial_estimate is None:
            x = np.clip(observed - self._background_for(observed), 0.0, 1.0).astype(np.float32, copy=False)
        else:
            x = self._to_numpy_chw(initial_estimate)
            if x.shape != observed.shape:
                raise ValueError(
                    f"initial_estimate shape {x.shape} does not match noisy image shape {observed.shape}."
                )
            if self.clip:
                x = np.clip(x, 0.0, 1.0)

        restored = np.empty_like(x)
        data_term_options = dict(self.data_term_options)
        if self.data_term_kind == "HUBER" and isinstance(param, dict) and "delta" in param:
            data_term_options["delta"] = float(param["delta"])
        for channel_index in range(observed.shape[0]):
            channel_initial = x[channel_index]
            channel_observed = observed[channel_index]
            channel_background = self._background_for(observed)[channel_index]
            restored[channel_index] = self._solve_single_channel(
                observed=channel_observed,
                initial=channel_initial,
                background=channel_background,
                regularizer_param=regularizer_param,
                data_term_options=data_term_options,
            )
        return restored.astype(np.float32, copy=False)

    def _background_for(self, image: np.ndarray) -> np.ndarray:
        if self.background.ndim == 0:
            return np.full_like(image, float(self.background))
        if self.background.shape == image.shape:
            return self.background.astype(np.float32, copy=False)
        if self.background.shape == image.shape[1:]:
            return np.broadcast_to(self.background, image.shape).astype(np.float32, copy=False)
        raise ValueError(
            f"Background shape {self.background.shape} is incompatible with image shape {image.shape}."
        )

    @staticmethod
    def _to_numpy_chw(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected image shape [C, H, W] or [H, W], got {arr.shape}.")
        return arr

    def _solve_single_channel(
        self,
        observed: np.ndarray,
        initial: np.ndarray,
        background: np.ndarray,
        regularizer_param: Dict[str, float],
        data_term_options: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        require_cil_core()

        x = np.asarray(initial, dtype=np.float32)
        if self.clip:
            x = np.clip(x, 0.0, 1.0)

        geometry = ImageGeometry(voxel_num_x=int(x.shape[1]), voxel_num_y=int(x.shape[0]))
        initial_data = ImageData(array=x, geometry=geometry)
        operator = build_known_psf_operator(
            geometry=geometry,
            psf=self.psf,
            norm_iterations=self.operator_norm_iterations,
        )
        fidelity = build_cil_data_term(
            data_term_kind=self.data_term_kind,
            geometry=geometry,
            operator=operator,
            observed=np.asarray(observed, dtype=np.float32),
            background=np.asarray(background, dtype=np.float32),
            alpha=(data_term_options or self.data_term_options).get("alpha"),
            sigma_read=(data_term_options or self.data_term_options).get("sigma_read"),
            delta=(data_term_options or self.data_term_options).get("delta"),
            poisson_backend=str((data_term_options or self.data_term_options).get("poisson_backend", "numba")),
            variance_epsilon=self.variance_epsilon,
            operator_norm_iterations=self.operator_norm_iterations,
            curvature_grid_points=self.curvature_grid_points,
            lipschitz_safety_scale=self.lipschitz_safety_scale,
        )
        regularizer = build_cil_regularizer(
            regularizer_kind=self.regularizer_kind,
            geometry=geometry,
            param=regularizer_param,
            clip=self.clip,
            regularizer_max_iter=self.regularizer_max_iter,
            regularizer_tolerance=self.regularizer_tolerance,
            ccpi_device=self.ccpi_device,
        )

        algorithm = self._build_algorithm(initial_data, fidelity, regularizer)
        callbacks = None
        if self.tol > 0.0:
            if regularizer.supports_objective_value:
                callbacks = [EarlyStoppingObjectiveValue(threshold=self.tol)]
            else:
                callbacks = [EarlyStoppingRelativeUpdate(threshold=self.tol)]
        algorithm.run(
            iterations=self.max_iter,
            callbacks=callbacks,
            verbose=0,
        )

        result = np.asarray(algorithm.get_output().as_array(), dtype=np.float32)
        if self.clip:
            result = np.clip(result, 0.0, 1.0)
        return result.astype(np.float32, copy=False)

    def _normalize_regularizer_param(self, param: Any) -> Dict[str, float]:
        if self.regularizer_kind in {"TV", "L1", "L0"}:
            value = float(param["lambda"] if isinstance(param, dict) else param)
            if value < 0.0:
                raise ValueError(f"{self.regularizer_kind} lambda must be >= 0, got {value}.")
            return {"lambda": value}

        if self.regularizer_kind == "TGV":
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
            return {"alpha": alpha, "gamma": gamma}

        if self.regularizer_kind == "MCP":
            if isinstance(param, dict):
                lambda_value = float(param["lambda"])
                beta = float(param["beta"])
            elif isinstance(param, (tuple, list)) and len(param) == 2:
                lambda_value = float(param[0])
                beta = float(param[1])
            else:
                raise ValueError(
                    "MCP parameter must be a dict with keys {'lambda','beta'} or a 2-tuple (lambda, beta)."
                )
            return {"lambda": lambda_value, "beta": beta}

        if self.regularizer_kind == "LP":
            if isinstance(param, dict):
                lambda_value = float(param["lambda"])
                power = float(param["p"])
            elif isinstance(param, (tuple, list)) and len(param) == 2:
                lambda_value = float(param[0])
                power = float(param[1])
            else:
                raise ValueError(
                    "LP parameter must be a dict with keys {'lambda','p'} or a 2-tuple (lambda, p)."
                )
            return {"lambda": lambda_value, "p": power}

        raise ValueError(
            f"Unsupported regularizer_kind={self.regularizer_kind}. "
            "Expected one of ['TV', 'TGV', 'L1', 'MCP', 'LP', 'L0']."
        )

    def _build_algorithm(self, initial_data, fidelity, regularizer):
        objective_update_interval = (
            self._DEFAULT_OBJECTIVE_UPDATE_INTERVAL if regularizer.supports_objective_value else 0
        )
        if fidelity.algorithm_family == "pdhg":
            if not regularizer.is_convex:
                raise ValueError(
                    "The CIL PDHG backend currently supports only convex regularizers with L1 data fidelity."
                )
            return PDHG(
                f=fidelity.pdhg_function,
                g=regularizer.function,
                operator=fidelity.operator,
                tau=None,
                sigma=None,
                initial=initial_data,
                max_iteration=self.max_iter,
                update_objective_interval=objective_update_interval,
                check_convergence=True,
            )

        try:
            lipschitz = float(getattr(fidelity.function, "L", 1.0) or 1.0)
        except (TypeError, ValueError):
            lipschitz = 1.0
        step_size = float(1.0 / max(lipschitz, 1e-8))
        algorithm_cls = FISTA if regularizer.is_convex else ISTA
        return algorithm_cls(
            initial=initial_data,
            f=fidelity.function,
            g=regularizer.function,
            step_size=step_size,
            max_iteration=self.max_iter,
            update_objective_interval=objective_update_interval,
        )


class CCPiPoissonGaussianDeblurSolver(CILKnownPSFDeblurSolver):
    """
    Backwards-compatible wrapper for the existing exact Poisson-Gaussian TV/TGV route.
    """

    def __init__(
        self,
        psf: np.ndarray,
        alpha: float,
        sigma_read: float,
        regularizer_kind: str,
        background: float | np.ndarray = 0.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        clip: bool = True,
        variance_epsilon: float = 1e-8,
        regularizer_max_iter: int = 100,
        regularizer_tolerance: float = 0.0,
        ccpi_device: str = "cpu",
    ):
        if float(alpha) <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}.")
        if float(sigma_read) < 0.0:
            raise ValueError(f"sigma_read must be >= 0, got {sigma_read}.")
        super().__init__(
            psf=psf,
            data_term_kind="POISSON_GAUSSIAN",
            regularizer_kind=regularizer_kind,
            data_term_options={
                "alpha": float(alpha),
                "sigma_read": float(sigma_read),
                "background": background,
            },
            max_iter=max_iter,
            tol=tol,
            clip=clip,
            variance_epsilon=variance_epsilon,
            regularizer_max_iter=regularizer_max_iter,
            regularizer_tolerance=regularizer_tolerance,
            ccpi_device=ccpi_device,
        )

    def get_name(self) -> str:
        return f"CCPiPoissonGaussian{self.regularizer_kind}Deblur"

