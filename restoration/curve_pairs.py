from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def canonical_curve_metric_names(extra_names: Iterable[str] | None = None) -> List[str]:
    ordered: List[str] = []
    for name in extra_names or ():
        key = str(name).strip().upper()
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def curve_pairs_path(main_path: str | Path, suffix: str = "_curve_pairs.pt") -> str:
    base = Path(main_path).expanduser().resolve()
    return str(base.with_suffix("").as_posix() + suffix)


def as_curve_storage_array(values: Any, dtype=np.float32) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(dtype, copy=False)


def initialize_curve_results(
    config: Dict[str, Any],
    metric_names: Sequence[str],
    *,
    max_param_dim: int,
) -> Dict[str, Any]:
    return {
        "config": dict(config),
        "metric_names": [str(name).upper() for name in metric_names],
        "max_param_dim": int(max_param_dim),
        "records": [],
    }
