from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_LEVIN_KERNEL_ROOT = Path("data") / "Levin8" / "kernels"
DEFAULT_LEVIN09_MAT_PATH = Path("data") / "Levin09.mat"
DEFAULT_LEVIN09_NPY_PATH = Path("data") / "Levin09.npy"

ORIGINAL_LEVIN_BENCHMARK_URL = (
    "https://billf.mit.edu/publication/understanding-and-evaluating-blind-deconvolution-algorithms/"
)
DEEPINV_LEVIN09_EXAMPLE_URL = (
    "https://deepinv.github.io/deepinv/auto_examples/plug-and-play/demo_PnP_DPIR_deblur.html"
)

_EXPECTED_KERNEL_COUNT = 8
_EXPECTED_SHAPES: Tuple[Tuple[int, int], ...] = (
    (19, 19),
    (17, 17),
    (15, 15),
    (27, 27),
    (13, 13),
    (21, 21),
    (23, 23),
    (23, 23),
)


def _normalize_psf(psf: np.ndarray) -> np.ndarray:
    array = np.asarray(psf, dtype=np.float32)
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("Levin kernel has non-positive sum and cannot be normalized.")
    return array / total


def _kernel_filename(kernel_index: int) -> str:
    return f"kernel_{int(kernel_index) + 1:02d}.npy"


def _resolve_kernel_root(root: Optional[object] = None) -> Path:
    if root is None or str(root).strip() == "":
        return DEFAULT_LEVIN_KERNEL_ROOT
    return Path(root)


def _validate_kernel_sequence(kernels: Sequence[np.ndarray], source_name: str) -> List[np.ndarray]:
    normalized = [_normalize_psf(kernel) for kernel in kernels]
    if len(normalized) != _EXPECTED_KERNEL_COUNT:
        raise ValueError(
            f"{source_name} should contain {_EXPECTED_KERNEL_COUNT} kernels, got {len(normalized)}."
        )
    shapes = tuple(tuple(int(v) for v in kernel.shape) for kernel in normalized)
    if shapes != _EXPECTED_SHAPES:
        raise ValueError(f"{source_name} has unexpected Levin09 shapes {shapes}; expected {_EXPECTED_SHAPES}.")
    return normalized


def _load_levin09_mat(path: Path) -> List[np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading Levin09.mat requires h5py.") from exc

    with h5py.File(path, "r") as handle:
        refs = handle["kernels"][()]
        kernels = [np.array(handle[ref], dtype=np.float32) for ref in refs.ravel()]
    return _validate_kernel_sequence(kernels, str(path))


def _load_levin09_npy(path: Path) -> List[np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        kernels = [np.asarray(item, dtype=np.float32) for item in payload.ravel()]
    elif isinstance(payload, np.ndarray) and payload.ndim == 3:
        kernels = [np.asarray(payload[idx], dtype=np.float32) for idx in range(payload.shape[0])]
    else:
        raise ValueError(
            f"Unsupported Levin09.npy layout in {path}: shape={getattr(payload, 'shape', None)}, "
            f"dtype={getattr(payload, 'dtype', None)}."
        )
    return _validate_kernel_sequence(kernels, str(path))


def _iter_standard_sources() -> List[Tuple[str, Path, List[np.ndarray]]]:
    sources: List[Tuple[str, Path, List[np.ndarray]]] = []
    if DEFAULT_LEVIN09_MAT_PATH.exists():
        sources.append(("Levin09.mat", DEFAULT_LEVIN09_MAT_PATH, _load_levin09_mat(DEFAULT_LEVIN09_MAT_PATH)))
    if DEFAULT_LEVIN09_NPY_PATH.exists():
        sources.append(("Levin09.npy", DEFAULT_LEVIN09_NPY_PATH, _load_levin09_npy(DEFAULT_LEVIN09_NPY_PATH)))
    return sources


def _assert_cross_source_consistency(sources: Sequence[Tuple[str, Path, List[np.ndarray]]]) -> None:
    if len(sources) < 2:
        return
    _, reference_path, reference_kernels = sources[0]
    for _, candidate_path, candidate_kernels in sources[1:]:
        for kernel_index, (reference, candidate) in enumerate(zip(reference_kernels, candidate_kernels)):
            if reference.shape != candidate.shape or not np.allclose(reference, candidate, atol=1e-8):
                raise ValueError(
                    "Levin kernel mismatch across sources: "
                    f"{reference_path} vs {candidate_path} at kernel {kernel_index + 1}."
                )


def _load_canonical_levin_kernels() -> Tuple[List[np.ndarray], str]:
    sources = _iter_standard_sources()
    if not sources:
        raise FileNotFoundError(
            "No canonical Levin09 source found. Expected one of:\n"
            f"  - {DEFAULT_LEVIN09_MAT_PATH}\n"
            f"  - {DEFAULT_LEVIN09_NPY_PATH}\n"
            "The old embedded square kernels have been removed because they were not the standard Levin09 kernels."
        )
    _assert_cross_source_consistency(sources)
    return sources[0][2], ", ".join(str(path) for _, path, _ in sources)


def _save_kernel_files(root: Path, kernels: Sequence[np.ndarray]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for kernel_index, kernel in enumerate(kernels):
        np.save(root / _kernel_filename(kernel_index), _normalize_psf(kernel).astype(np.float32))


def _load_saved_kernel_files(root: Path) -> List[np.ndarray]:
    paths = [root / _kernel_filename(idx) for idx in range(_EXPECTED_KERNEL_COUNT)]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(
            f"Missing Levin kernel files under {root}. Expected kernel_01.npy ... kernel_08.npy."
        )
    return [_normalize_psf(np.load(path)) for path in paths]


def ensure_levin_kernel_files(root: Optional[object] = None) -> Path:
    kernel_root = _resolve_kernel_root(root)
    canonical_kernels, _ = _load_canonical_levin_kernels()
    kernel_root.mkdir(parents=True, exist_ok=True)

    rewrite_needed = True
    try:
        current_kernels = _load_saved_kernel_files(kernel_root)
        rewrite_needed = False
        for current, canonical in zip(current_kernels, canonical_kernels):
            if current.shape != canonical.shape or not np.allclose(current, canonical, atol=1e-8):
                rewrite_needed = True
                break
    except FileNotFoundError:
        rewrite_needed = True

    if rewrite_needed:
        _save_kernel_files(kernel_root, canonical_kernels)
    return kernel_root


def load_levin_kernels(root: Optional[object] = None) -> List[Dict[str, object]]:
    kernel_root = ensure_levin_kernel_files(root)
    canonical_kernels, source_description = _load_canonical_levin_kernels()
    kernels: List[Dict[str, object]] = []
    for kernel_index, psf in enumerate(canonical_kernels):
        kernels.append(
            {
                "kernel_index": int(kernel_index),
                "kernel_one_based": int(kernel_index + 1),
                "name": f"Levin09 kernel {kernel_index + 1}",
                "path": str(kernel_root / _kernel_filename(kernel_index)),
                "psf": _normalize_psf(psf),
                "shape": tuple(int(v) for v in psf.shape),
                "source": source_description,
            }
        )
    return kernels


def describe_kernel_subset(
    kernels: Sequence[Dict[str, object]],
    kernel_indices: Iterable[int],
) -> List[Dict[str, object]]:
    descriptions: List[Dict[str, object]] = []
    for kernel_index in kernel_indices:
        kernel_info = kernels[int(kernel_index)]
        descriptions.append(
            {
                "kernel_index": int(kernel_info["kernel_index"]),
                "kernel_one_based": int(kernel_info["kernel_one_based"]),
                "name": str(kernel_info["name"]),
                "path": str(kernel_info["path"]),
                "shape": list(kernel_info["shape"]),
                "source": str(kernel_info.get("source", "")),
            }
        )
    return descriptions


def resolve_leave_one_kernel_out_split(
    num_kernels: int,
    held_out_kernel: int,
    split: str,
) -> List[int]:
    if int(num_kernels) <= 0:
        raise ValueError("num_kernels must be positive.")
    held_out = int(held_out_kernel)
    if held_out < 0 or held_out >= int(num_kernels):
        raise ValueError(f"held_out_kernel must be in [0, {int(num_kernels) - 1}].")

    split_name = str(split).strip().lower()
    if split_name == "train":
        return [idx for idx in range(int(num_kernels)) if idx != held_out]
    if split_name in {"test", "val", "validation"}:
        return [held_out]
    raise ValueError("split must be one of: train, test, val, validation.")


def get_levin_kernels(as_tensor: bool = True):
    kernels, _ = _load_canonical_levin_kernels()
    if not as_tensor:
        return kernels
    return [torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0) for kernel in kernels]


if __name__ == "__main__":
    kernels, source_description = _load_canonical_levin_kernels()
    print(f"\nLoaded {len(kernels)} Levin kernels from: {source_description}")
    print("-" * 30)
    for idx, kernel in enumerate(kernels, start=1):
        print(f"Kernel {idx} shape: {kernel.shape}, sum={float(kernel.sum()):.8f}")
