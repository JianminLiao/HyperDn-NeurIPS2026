from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from cil.optimisation.operators import LinearOperator
    from cil.framework import ImageData
except ImportError:
    LinearOperator = None
    ImageData = None

from .cil_pg import (
    adjoint_convolve2d,
    estimate_psf_operator_norm_sq,
    forward_convolve2d,
    require_cil_core,
)

_LinearOperatorBase = LinearOperator if LinearOperator is not None else object


class KnownPSFConvolutionOperator(_LinearOperatorBase):
    def __init__(
        self,
        geometry,
        psf: np.ndarray,
        norm_iterations: int = 25,
    ):
        require_cil_core()
        psf_arr = np.asarray(psf, dtype=np.float32)
        if psf_arr.ndim != 2:
            raise ValueError(f"KnownPSFConvolutionOperator expects a 2D PSF, got {psf_arr.shape}.")
        psf_arr = np.maximum(psf_arr, 0.0)
        psf_sum = float(np.sum(psf_arr))
        if not np.isfinite(psf_sum) or psf_sum <= 0.0:
            raise ValueError("PSF must have a finite positive sum.")

        self.psf = (psf_arr / psf_sum).astype(np.float32, copy=False)
        self.psf_mirror = np.flip(self.psf)
        self._norm_iterations = int(max(1, norm_iterations))
        self._cached_norm: Optional[float] = None
        super().__init__(domain_geometry=geometry, range_geometry=geometry)

    def direct(self, x, out=None):
        arr = np.asarray(x.as_array(), dtype=np.float32)
        result = forward_convolve2d(arr, self.psf)
        if out is None:
            return ImageData(array=result, geometry=self.range_geometry())
        out.fill(result)
        return out

    def adjoint(self, x, out=None):
        arr = np.asarray(x.as_array(), dtype=np.float32)
        result = adjoint_convolve2d(arr, self.psf_mirror)
        if out is None:
            return ImageData(array=result, geometry=self.domain_geometry())
        out.fill(result)
        return out

    def norm(self, **kwargs):
        if self._cached_norm is None:
            probe_shape = self.domain_geometry().shape
            self._cached_norm = float(
                np.sqrt(
                    estimate_psf_operator_norm_sq(
                        psf=self.psf,
                        shape=probe_shape,
                        num_iterations=self._norm_iterations,
                    )
                )
            )
        return float(self._cached_norm)

    def is_linear(self):
        return True
