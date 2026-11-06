from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


DEFAULT_QUALITY_OBJECTIVE_METRICS = ("PSNR", "SSIM")


def metric_value_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def canonical_quality_metric_names(extra_names: Iterable[str] | None = None) -> List[str]:
    ordered: List[str] = []
    for name in DEFAULT_QUALITY_OBJECTIVE_METRICS:
        ordered.append(str(name).upper())
    for name in extra_names or ():
        key = str(name).strip().upper()
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def quality_pairs_path(main_path: str | Path, suffix: str = "_quality_pairs.pt") -> str:
    base = Path(main_path).expanduser().resolve()
    return str(base.with_suffix("").as_posix() + suffix)


def as_storage_chw(image_chw: Any, dtype=np.float16) -> np.ndarray:
    if isinstance(image_chw, torch.Tensor):
        image_chw = image_chw.detach().cpu().numpy()
    image = np.asarray(image_chw, dtype=np.float32)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image for export, got shape={image.shape}.")
    image = np.clip(image, 0.0, 1.0)
    return image.astype(dtype, copy=False)


def initialize_quality_results(config: Dict[str, Any], objective_metric_names: Sequence[str]) -> Dict[str, Any]:
    return {
        "config": dict(config),
        "objective_metric_names": [str(name).upper() for name in objective_metric_names],
        "records": [],
    }
