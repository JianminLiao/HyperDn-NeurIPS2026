from PIL import Image as _PILImage  # noqa: F401 — preload Pillow before torch (Windows DLL load order).
import zlib
import torch
from torch.utils.data import Dataset
try:
    import torchvision
except Exception:
    torchvision = None
import numpy as np
import os
from typing import Dict, Optional
from restoration import ProblemConfig
from restoration.levin import load_levin_kernels
from restoration.noise_models import apply_noise, make_rng
from restoration.pgccpi_observation_encoding import (
    normalize_pgccpi_data_term_name,
    normalize_pgccpi_noise_model_name,
    normalize_pgccpi_operator_name,
)
from restoration.pgccpi_param_encoding import (
    decode_pgccpi_target_tensor as _decode_pgccpi_target_tensor_impl,
    encode_pgccpi_target as _encode_pgccpi_target_impl,
    normalize_pgccpi_regularizer_name,
    pgccpi_param_specs,
)


_RGB_CHANNEL_PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _require_torchvision():
    if torchvision is None:
        raise ImportError(
            "torchvision is unavailable in the current environment. "
            "Use a dataset path that does not depend on torchvision datasets/transforms, "
            "or install a working torchvision build."
        )
    return torchvision


class _Normalizer:
    """Picklable normalizer — avoids closure issues with Windows multiprocessing spawn."""

    def __init__(self, mean, std):
        self.mean_tensor = torch.tensor(list(mean), dtype=torch.float32).view(-1, 1, 1)
        self.std_tensor = torch.tensor(list(std), dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean_tensor.to(device=tensor.device, dtype=tensor.dtype)) / self.std_tensor.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )


def _build_normalizer(mean, std):
    return _Normalizer(mean, std)


def _torch_load_compat(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _to_float_hwc(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if np.issubdtype(array.dtype, np.floating):
        array = np.asarray(array, dtype=np.float32)
        if np.max(array) > 1.0 or np.min(array) < 0.0:
            array = array / 255.0
    else:
        array = np.asarray(array, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0), dtype=np.float32)


def _safe_scalar_float(value, default: float = 0.0) -> float:
    return float(default if value is None else value)


def _compact_target_names_for_regularizer(regularizer: str, data_term: str | None = None):
    return [str(spec["encoded_name"]) for spec in pgccpi_param_specs(regularizer, data_term)]


def _normalize_compact_pgccpi_record(
    record: Dict,
    *,
    default_regularizer: str,
    default_data_term: str,
    default_noise_model: str,
    default_operator_kind: str,
    default_tgv_parameterization=None,
) -> Dict:
    regularizer = normalize_pgccpi_regularizer_name(record.get("regularizer", default_regularizer))
    data_term = normalize_pgccpi_data_term_name(record.get("data_term", default_data_term))
    noise_model = normalize_pgccpi_noise_model_name(record.get("noise_model", default_noise_model))
    operator_kind = normalize_pgccpi_operator_name(record.get("operator_kind", default_operator_kind))
    tgv_parameterization = record.get("tgv_parameterization", default_tgv_parameterization)

    noise_params_raw = record.get("noise_params", {})
    noise_params = dict(noise_params_raw) if isinstance(noise_params_raw, dict) else {}
    noise_sigma = float(
        record.get(
            "noise_sigma",
            noise_params.get("sigma", record.get("sigma", 0.0)),
        )
    )
    target_vector = [float(v) for v in list(record.get("target_vector", []))]
    if not target_vector:
        raise ValueError("Compact PG/CCPi record is missing target_vector.")
    target_names_raw = record.get("target_names", None)
    target_names = (
        [str(name) for name in list(target_names_raw)]
        if target_names_raw is not None
        else _compact_target_names_for_regularizer(regularizer, data_term)
    )
    optimal_param = {
        str(k): float(v)
        for k, v in dict(record.get("optimal_param", {})).items()
    }
    if regularizer == "DIFFPIR":
        if not optimal_param:
            raise ValueError("DIFFPIR record is missing optimal_param. Re-generate labels.")
        target_info = _encode_pgccpi_target(
            regularizer,
            dict(optimal_param),
            data_term=data_term,
            noise_sigma=noise_sigma,
        )
        target_vector = target_info["vector"].tolist()
        target_names = list(target_info["names"])
        # Keep the lambda label; target_dim=2 preserves the historical padded loss slot.
        # The executor always uses zeta=1, independently of output slot 1.
        target_vector = target_vector[:1]
        target_names = target_names[:1]
    target_dim = int(record.get("target_dim", len(target_vector)))
    if target_dim <= 0:
        target_dim = int(len(target_vector))

    image_index = int(record["image_index"])
    sample_index = int(record.get("sample_index", record.get("noise_index", 0)))
    kernel_index = int(record.get("kernel_index", 0))
    noise_seed = int(record.get("noise_seed", 0))
    augmentation_seed = int(
        record.get(
            "augmentation_seed",
            _derive_augmentation_seed(
                image_index=image_index,
                sample_index=sample_index,
                kernel_index=kernel_index,
                noise_seed=noise_seed,
                operator_kind=operator_kind,
            ),
        )
    )

    return {
        "task_key": str(
            record.get(
                "task_key",
                _pgccpi_task_key(
                    image_index,
                    sample_index,
                    kernel_index,
                    operator_kind=operator_kind,
                ),
            )
        ),
        "image_index": image_index,
        "sample_index": sample_index,
        "kernel_index": kernel_index,
        "kernel_one_based": int(record.get("kernel_one_based", 0)),
        "kernel_name": str(record.get("kernel_name", "identity")),
        "alpha": _safe_scalar_float(record.get("alpha")),
        "sigma_read": _safe_scalar_float(record.get("sigma_read")),
        "background": _safe_scalar_float(record.get("background")),
        "noise_sigma": noise_sigma,
        "noise_seed": noise_seed,
        "augmentation_seed": augmentation_seed,
        "noise_params": {str(k): _safe_scalar_float(v) for k, v in noise_params.items()},
        "optimal_param": optimal_param,
        "target_vector": list(target_vector),
        "target_names": list(target_names),
        "target_dim": target_dim,
        "best_score": float(record["best_score"]),
        "regularizer": regularizer,
        "data_term": data_term,
        "noise_model": noise_model,
        "operator_kind": operator_kind,
        "tgv_parameterization": tgv_parameterization,
    }


def _build_compact_pgccpi_bundle(
    compact_records,
    *,
    kernel_root=None,
    default_tgv_parameterization=None,
):
    if not compact_records:
        raise ValueError("No usable PG/CCPi records found for compact bundle.")
    regularizers = []
    data_terms = []
    noise_models = []
    operator_kinds = []
    for record in compact_records:
        regularizer = normalize_pgccpi_regularizer_name(record["regularizer"])
        data_term = normalize_pgccpi_data_term_name(record["data_term"])
        noise_model = normalize_pgccpi_noise_model_name(record["noise_model"])
        operator_kind = normalize_pgccpi_operator_name(record["operator_kind"])
        if regularizer not in regularizers:
            regularizers.append(regularizer)
        if data_term not in data_terms:
            data_terms.append(data_term)
        if noise_model not in noise_models:
            noise_models.append(noise_model)
        if operator_kind not in operator_kinds:
            operator_kinds.append(operator_kind)
    target_dim = max(int(record.get("target_dim", len(record["target_vector"]))) for record in compact_records)
    target_name_candidates = [tuple(record.get("target_names", [])) for record in compact_records]
    if target_name_candidates and len({candidate for candidate in target_name_candidates}) == 1:
        target_names = list(target_name_candidates[0])
    else:
        target_names = [f"param_{index}" for index in range(target_dim)]

    tgv_parameterization = default_tgv_parameterization
    for record in compact_records:
        if record.get("tgv_parameterization", None):
            tgv_parameterization = record.get("tgv_parameterization")
            break

    return {
        "records": compact_records,
        "regularizer": str(regularizers[0]),
        "target_dim": int(target_dim),
        "target_names": list(target_names),
        "kernel_root": kernel_root,
        "data_term": str(data_terms[0]),
        "noise_model": str(noise_models[0]),
        "operator_kind": str(operator_kinds[0]),
        "tgv_parameterization": tgv_parameterization,
        "regularizers": list(regularizers),
        "data_terms": list(data_terms),
        "noise_models": list(noise_models),
        "operator_kinds": list(operator_kinds),
        "has_mixed_regularizers": bool(len(regularizers) > 1),
        "has_mixed_data_terms": bool(len(data_terms) > 1),
        "has_mixed_noise_models": bool(len(noise_models) > 1),
        "has_mixed_operator_kinds": bool(len(operator_kinds) > 1),
    }


def _normalize_seed(seed: int) -> int:
    max_seed = 2**31 - 1
    normalized = int(seed) % max_seed
    return normalized if normalized != 0 else 1


def _derive_augmentation_seed(
    *,
    image_index: int,
    sample_index: int,
    kernel_index: int,
    noise_seed: int,
    operator_kind: str,
    epoch: int = 0,
) -> int:
    operator_key = normalize_pgccpi_operator_name(operator_kind)
    max_seed = 2**31 - 1
    mixed = (
        146959810
        + (int(image_index) + 1) * 1000003
        + (int(sample_index) + 1) * 9176
        + (int(noise_seed) + 1) * 170003
    ) % max_seed
    if operator_key != "IDENTITY_DENOISE":
        mixed = (mixed + (int(kernel_index) + 1) * 2000003) % max_seed
    epoch_int = int(epoch)
    if epoch_int != 0:
        # Mix the epoch into the base seed using a stable, non-randomized hash
        # (zlib.crc32 is deterministic across Python runs, unlike builtin hash()).
        epoch_bytes = int(epoch_int & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
        base_bytes = int(mixed & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
        mixed = int(zlib.crc32(base_bytes + epoch_bytes)) % max_seed
    return mixed if mixed != 0 else 1


def _resolve_record_augmentation_seed(
    record: Dict,
    *,
    operator_kind: str,
    epoch: int = 0,
) -> int:
    # When epoch > 0 we deliberately bypass any cached record["augmentation_seed"]
    # so that augmentation plans vary from epoch to epoch. With epoch == 0 this
    # preserves the original deterministic behavior used by val / eval paths.
    if int(epoch) == 0:
        stored = record.get("augmentation_seed", None)
        if stored is not None:
            return _normalize_seed(int(stored))
    return _derive_augmentation_seed(
        image_index=int(record.get("image_index", 0)),
        sample_index=int(record.get("sample_index", 0)),
        kernel_index=int(record.get("kernel_index", 0)),
        noise_seed=int(record.get("noise_seed", 0)),
        operator_kind=operator_kind,
        epoch=int(epoch),
    )


def _sample_training_augmentation_plan(
    *,
    augmentation_seed: int,
    apply_augmentations: bool,
    enable_rotate_90_270: bool,
    enable_pixel_dithering: bool,
    pixel_dithering_prob: float,
    enable_rgb_permutation: bool,
    rgb_permutation_prob: float,
    channel_count: int,
) -> Dict[str, object]:
    plan = {
        "hflip": False,
        "vflip": False,
        "rotation_k": 0,
        "pixel_delta_step": 0,
        "rgb_permutation": tuple(range(int(channel_count))),
    }
    if not apply_augmentations:
        return plan

    rng = np.random.RandomState(_normalize_seed(int(augmentation_seed)))
    rotation_choices = [0, 2]
    if enable_rotate_90_270:
        rotation_choices.extend([1, 3])

    plan["hflip"] = bool(rng.rand() < 0.5)
    plan["vflip"] = bool(rng.rand() < 0.5)
    plan["rotation_k"] = int(rotation_choices[int(rng.randint(0, len(rotation_choices)))])

    if enable_pixel_dithering and float(pixel_dithering_prob) > 0.0 and rng.rand() < float(pixel_dithering_prob):
        plan["pixel_delta_step"] = int(rng.randint(0, 3)) - 1

    if (
        enable_rgb_permutation
        and float(rgb_permutation_prob) > 0.0
        and int(channel_count) == 3
        and rng.rand() < float(rgb_permutation_prob)
    ):
        perm_idx = int(rng.randint(0, len(_RGB_CHANNEL_PERMUTATIONS)))
        plan["rgb_permutation"] = tuple(int(v) for v in _RGB_CHANNEL_PERMUTATIONS[perm_idx])

    return plan


def _apply_training_augmentation_plan(image_hwc: np.ndarray, plan: Dict[str, object]) -> np.ndarray:
    image = np.asarray(image_hwc, dtype=np.float32)
    if bool(plan.get("hflip", False)):
        image = np.flip(image, axis=1)
    if bool(plan.get("vflip", False)):
        image = np.flip(image, axis=0)

    rotation_k = int(plan.get("rotation_k", 0))
    if rotation_k % 4 != 0:
        image = np.rot90(image, k=rotation_k, axes=(0, 1))

    delta_step = int(plan.get("pixel_delta_step", 0))
    if delta_step != 0:
        image = np.clip(image + (delta_step / 255.0), 0.0, 1.0)

    permutation = tuple(int(v) for v in plan.get("rgb_permutation", tuple(range(image.shape[2]))))
    if image.ndim == 3 and image.shape[2] == len(permutation):
        image = image[..., permutation]

    return np.ascontiguousarray(image, dtype=np.float32)


def _apply_joint_training_augmentations(
    clean_chw: np.ndarray,
    noisy_chw: np.ndarray,
    *,
    augmentation_seed: int,
    apply_augmentations: bool,
    enable_rotate_90_270: bool,
    enable_pixel_dithering: bool,
    pixel_dithering_prob: float,
    enable_rgb_permutation: bool,
    rgb_permutation_prob: float,
):
    clean = np.asarray(clean_chw, dtype=np.float32)
    noisy = np.asarray(noisy_chw, dtype=np.float32)
    if clean.shape != noisy.shape:
        raise ValueError(f"Expected clean/noisy pair to share shape, got {clean.shape} vs {noisy.shape}.")

    plan = _sample_training_augmentation_plan(
        augmentation_seed=int(augmentation_seed),
        apply_augmentations=apply_augmentations,
        enable_rotate_90_270=enable_rotate_90_270,
        enable_pixel_dithering=enable_pixel_dithering,
        pixel_dithering_prob=pixel_dithering_prob,
        enable_rgb_permutation=enable_rgb_permutation,
        rgb_permutation_prob=rgb_permutation_prob,
        channel_count=int(clean.shape[0]),
    )

    clean_hwc = clean.transpose((1, 2, 0))
    noisy_hwc = noisy.transpose((1, 2, 0))
    clean_aug = _apply_training_augmentation_plan(clean_hwc, plan)
    noisy_aug = _apply_training_augmentation_plan(noisy_hwc, plan)
    return (
        np.ascontiguousarray(clean_aug.transpose((2, 0, 1)), dtype=np.float32),
        np.ascontiguousarray(noisy_aug.transpose((2, 0, 1)), dtype=np.float32),
    )


def _apply_known_kernel_blur(clean_chw: np.ndarray, psf: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve

    clean = np.asarray(clean_chw, dtype=np.float32)
    out = np.empty_like(clean)
    for channel_index in range(clean.shape[0]):
        out[channel_index] = convolve(clean[channel_index], np.asarray(psf, dtype=np.float32), mode='same')
    return out.astype(np.float32, copy=False)


def _poisson_gaussian_degrade_known_kernel(
    clean_chw: np.ndarray,
    psf: np.ndarray,
    alpha: float,
    sigma_read: float,
    background: float,
    seed: int,
) -> np.ndarray:
    alpha = max(float(alpha), 1e-12)
    sigma_read = float(sigma_read)
    background = float(background)

    mean_signal = _apply_known_kernel_blur(clean_chw, psf) + background
    mean_signal = np.maximum(mean_signal, 0.0)
    rng = np.random.RandomState(int(seed))
    poisson_rate = np.clip(mean_signal / alpha, 0.0, None)
    shot = rng.poisson(poisson_rate).astype(np.float32)
    read = rng.normal(0.0, sigma_read, size=mean_signal.shape).astype(np.float32)
    return (alpha * shot + read).astype(np.float32, copy=False)


def _pgccpi_task_key(
    image_index: int,
    sample_index: int,
    kernel_index: Optional[int] = None,
    *,
    operator_kind: str = "KNOWN_PSF_DEBLUR",
) -> str:
    operator_key = normalize_pgccpi_operator_name(operator_kind)
    base = f"img{int(image_index):05d}_sample{int(sample_index):03d}"
    if operator_key == "IDENTITY_DENOISE" or kernel_index is None:
        return base
    return f"{base}_kernel{int(kernel_index):02d}"


def _encode_pgccpi_target(
    regularizer: str,
    optimal_param: dict,
    data_term: str | None = None,
    noise_sigma: float | None = None,
):
    return _encode_pgccpi_target_impl(
        regularizer,
        optimal_param,
        data_term=data_term,
        noise_sigma=noise_sigma,
    )


def _decode_pgccpi_target_tensor(
    targets: torch.Tensor,
    regularizer: str,
    data_term: str | None = None,
) -> torch.Tensor:
    return _decode_pgccpi_target_tensor_impl(targets, regularizer, data_term)


def _resolve_compact_sequence(payload, names, total: int, default_factory):
    for name in names:
        if name not in payload:
            continue
        values = payload[name]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        elif isinstance(values, np.ndarray):
            values = values.tolist()
        elif not isinstance(values, (list, tuple)):
            raise TypeError(f"Expected {name} in compact PG/CCPi payload to be list-like, got {type(values)!r}.")
        if len(values) != int(total):
            raise ValueError(
                f"Inconsistent PG/CCPi label lengths for {name}: expected {total}, got {len(values)}."
            )
        return list(values)
    return [default_factory(index) for index in range(int(total))]


def _load_compact_pgccpi_records(ground_truth_path: str, metric: str):
    if not ground_truth_path or not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    payload = _torch_load_compat(ground_truth_path)
    config = payload.get("config", {})
    regularizer_raw = config.get("regularizer", payload.get("regularizer", None))
    data_term_raw = config.get("data_term", payload.get("data_term", "NLL"))
    noise_model_raw = config.get("noise_model", payload.get("noise_model", "POISSON_GAUSSIAN"))
    operator_kind_raw = config.get(
        "operator_kind",
        config.get("operator", payload.get("operator_kind", "KNOWN_PSF_DEBLUR")),
    )
    default_regularizer = (
        normalize_pgccpi_regularizer_name(str(regularizer_raw).strip())
        if regularizer_raw is not None and str(regularizer_raw).strip()
        else "TV"
    )
    default_data_term = normalize_pgccpi_data_term_name(data_term_raw)
    default_noise_model = normalize_pgccpi_noise_model_name(noise_model_raw)
    default_operator_kind = normalize_pgccpi_operator_name(operator_kind_raw)
    default_tgv_parameterization = config.get("tgv_parameterization", payload.get("tgv_parameterization"))

    param_key = f"{metric}_optimal_params"
    score_key = f"{metric}_best_scores"
    records = payload.get("records", None)
    if records:
        first_record = records[0]
        if "target_vector" in first_record and "best_score" in first_record:
            compact_records = [
                _normalize_compact_pgccpi_record(
                    record,
                    default_regularizer=default_regularizer,
                    default_data_term=default_data_term,
                    default_noise_model=default_noise_model,
                    default_operator_kind=default_operator_kind,
                    default_tgv_parameterization=default_tgv_parameterization,
                )
                for record in records
            ]
            return _build_compact_pgccpi_bundle(
                compact_records,
                kernel_root=config.get("kernel_root", payload.get("kernel_root")),
                default_tgv_parameterization=default_tgv_parameterization,
            )

    if param_key not in payload or score_key not in payload:
        if not records:
            raise ValueError(f"Ground truth file missing metric payload for {metric}: {ground_truth_path}")
        compact_records = []
        for record_index, record in enumerate(records):
            metric_result = record.get("metric_results", {}).get(metric, None)
            if metric_result is None:
                continue
            image_index = int(record["image_index"])
            sample_index = int(record.get("sample_index", record.get("noise_index", record_index)))
            kernel_index = int(record.get("kernel_index", 0))
            noise_params_raw = record.get("noise_params", {})
            noise_params = dict(noise_params_raw) if isinstance(noise_params_raw, dict) else {}
            noise_sigma = float(
                record.get(
                    "noise_sigma",
                    noise_params.get("sigma", record.get("sigma", 0.0)),
                )
            )
            record_regularizer = normalize_pgccpi_regularizer_name(record.get("regularizer", default_regularizer))
            record_data_term = normalize_pgccpi_data_term_name(record.get("data_term", default_data_term))
            record_noise_model = normalize_pgccpi_noise_model_name(record.get("noise_model", default_noise_model))
            record_operator_kind = normalize_pgccpi_operator_name(record.get("operator_kind", default_operator_kind))
            record_tgv_parameterization = record.get("tgv_parameterization", default_tgv_parameterization)
            target_info = _encode_pgccpi_target(
                record_regularizer,
                dict(metric_result["optimal_param"]),
                record_data_term,
                noise_sigma=noise_sigma,
            )
            compact_records.append(
                {
                    "task_key": str(
                        record.get(
                            "task_key",
                            _pgccpi_task_key(
                                image_index,
                                sample_index,
                                kernel_index,
                                operator_kind=record_operator_kind,
                            ),
                        )
                    ),
                    "image_index": image_index,
                    "sample_index": sample_index,
                    "kernel_index": kernel_index,
                    "kernel_one_based": int(record.get("kernel_one_based", 0)),
                    "kernel_name": str(record.get("kernel_name", "identity")),
                    "alpha": _safe_scalar_float(record.get("alpha")),
                    "sigma_read": _safe_scalar_float(record.get("sigma_read")),
                    "background": _safe_scalar_float(record.get("background")),
                    "noise_sigma": noise_sigma,
                    "noise_seed": int(record.get("noise_seed", 0)),
                    "augmentation_seed": int(
                        record.get(
                            "augmentation_seed",
                            _derive_augmentation_seed(
                                image_index=image_index,
                                sample_index=sample_index,
                                kernel_index=kernel_index,
                                noise_seed=int(record.get("noise_seed", 0)),
                                operator_kind=record_operator_kind,
                            ),
                        )
                    ),
                    "noise_params": {str(k): float(v) for k, v in noise_params.items()},
                    "optimal_param": {str(k): float(v) for k, v in dict(metric_result["optimal_param"]).items()},
                    "target_vector": target_info["vector"].tolist(),
                    "target_names": list(target_info["names"]),
                    "target_dim": int(len(target_info["vector"])),
                    "best_score": float(metric_result["best_score"]),
                    "regularizer": record_regularizer,
                    "data_term": record_data_term,
                    "noise_model": record_noise_model,
                    "operator_kind": record_operator_kind,
                    "tgv_parameterization": record_tgv_parameterization,
                }
            )
        return _build_compact_pgccpi_bundle(
            compact_records,
            kernel_root=config.get("kernel_root"),
            default_tgv_parameterization=default_tgv_parameterization,
        )

    raw_image_ids = payload.get("image_ids", payload.get("image_indices", []))
    total = len(raw_image_ids)
    image_ids = _resolve_compact_sequence(payload, ("image_ids", "image_indices"), total, lambda idx: idx)
    params = payload.get(param_key, [])
    scores = payload.get(score_key, [])

    if len(params) != total or len(scores) != total:
        raise ValueError(f"Inconsistent PG/CCPi label lengths in {ground_truth_path}.")
    sample_indices = _resolve_compact_sequence(payload, ("sample_indices", "noise_indices"), total, lambda idx: idx)
    kernel_indices = _resolve_compact_sequence(payload, ("kernel_indices",), total, lambda _: 0)
    kernel_one_based = _resolve_compact_sequence(payload, ("kernel_one_based",), total, lambda _: 0)
    kernel_names = _resolve_compact_sequence(payload, ("kernel_names",), total, lambda _: "identity")
    alphas = _resolve_compact_sequence(payload, ("alphas", "alpha"), total, lambda _: 0.0)
    sigma_reads = _resolve_compact_sequence(payload, ("sigma_reads", "sigma_read"), total, lambda _: 0.0)
    backgrounds = _resolve_compact_sequence(payload, ("backgrounds", "background"), total, lambda _: 0.0)
    noise_sigmas = _resolve_compact_sequence(payload, ("noise_sigmas", "sigmas", "noise_sigma"), total, lambda _: 0.0)
    noise_seeds = _resolve_compact_sequence(payload, ("noise_seeds",), total, lambda _: 0)
    augmentation_seeds = _resolve_compact_sequence(payload, ("augmentation_seeds",), total, lambda _: None)
    task_keys = _resolve_compact_sequence(payload, ("task_keys",), total, lambda _: "")
    noise_params = _resolve_compact_sequence(payload, ("noise_params",), total, lambda _: {})

    compact_records = []
    for idx in range(total):
        target_info = _encode_pgccpi_target(
            default_regularizer,
            dict(params[idx]),
            default_data_term,
            noise_sigma=_safe_scalar_float(noise_sigmas[idx]),
        )
        compact_records.append(
            {
                "task_key": (
                    str(task_keys[idx])
                    if str(task_keys[idx]).strip()
                    else _pgccpi_task_key(
                        image_ids[idx],
                        sample_indices[idx],
                        kernel_indices[idx],
                        operator_kind=default_operator_kind,
                    )
                ),
                "image_index": int(image_ids[idx]),
                "sample_index": int(sample_indices[idx]),
                "kernel_index": int(kernel_indices[idx]),
                "kernel_one_based": int(kernel_one_based[idx]),
                "kernel_name": str(kernel_names[idx]),
                "alpha": _safe_scalar_float(alphas[idx]),
                "sigma_read": _safe_scalar_float(sigma_reads[idx]),
                "background": _safe_scalar_float(backgrounds[idx]),
                "noise_sigma": _safe_scalar_float(noise_sigmas[idx]),
                "noise_seed": int(noise_seeds[idx]),
                "augmentation_seed": int(
                    augmentation_seeds[idx]
                    if augmentation_seeds[idx] is not None
                    else _derive_augmentation_seed(
                        image_index=int(image_ids[idx]),
                        sample_index=int(sample_indices[idx]),
                        kernel_index=int(kernel_indices[idx]),
                        noise_seed=int(noise_seeds[idx]),
                        operator_kind=default_operator_kind,
                    )
                ),
                "noise_params": (
                    {str(k): _safe_scalar_float(v) for k, v in dict(noise_params[idx]).items()}
                    if isinstance(noise_params[idx], dict)
                    else {}
                ),
                "optimal_param": {str(k): float(v) for k, v in dict(params[idx]).items()},
                "target_vector": target_info["vector"].tolist(),
                "target_names": list(target_info["names"]),
                "target_dim": int(len(target_info["vector"])),
                "best_score": float(scores[idx]),
                "regularizer": default_regularizer,
                "data_term": default_data_term,
                "noise_model": default_noise_model,
                "operator_kind": default_operator_kind,
                "tgv_parameterization": default_tgv_parameterization,
            }
        )

    return _build_compact_pgccpi_bundle(
        compact_records,
        kernel_root=config.get("kernel_root"),
        default_tgv_parameterization=default_tgv_parameterization,
    )


def _filter_pgccpi_records_by_kernel_indices(compact_bundle, include_kernel_indices=None, exclude_kernel_indices=None):
    if include_kernel_indices is None and exclude_kernel_indices is None:
        return compact_bundle

    include_set = None if include_kernel_indices is None else {int(v) for v in include_kernel_indices}
    exclude_set = None if exclude_kernel_indices is None else {int(v) for v in exclude_kernel_indices}

    filtered_records = []
    for record in compact_bundle["records"]:
        kernel_index = int(record["kernel_index"])
        if include_set is not None and kernel_index not in include_set:
            continue
        if exclude_set is not None and kernel_index in exclude_set:
            continue
        filtered_records.append(record)

    if not filtered_records:
        raise ValueError("Kernel filtering removed all PG/CCPi records.")

    filtered_bundle = dict(compact_bundle)
    filtered_bundle["records"] = filtered_records
    filtered_bundle["filtered_kernel_indices"] = sorted(dict.fromkeys(int(r["kernel_index"]) for r in filtered_records))
    return filtered_bundle


class _BasePGCCPIDataset(Dataset):
    def __init__(
        self,
        ground_truth_path: str,
        metric: str,
        kernel_root: Optional[str],
        apply_augmentations: bool,
        augment_before_degrade: bool,
        resample_noise_realization: bool,
        enable_rotate_90_270: bool,
        enable_pixel_dithering: bool,
        pixel_dithering_prob: float,
        enable_rgb_permutation: bool,
        rgb_permutation_prob: float,
        include_kernel_indices=None,
        exclude_kernel_indices=None,
        compact_bundle=None,
    ):
        self.metric = metric
        self.apply_augmentations = bool(apply_augmentations)
        self.augment_before_degrade = bool(augment_before_degrade)
        self.resample_noise_realization = bool(resample_noise_realization)
        self.enable_rotate_90_270 = bool(enable_rotate_90_270)
        self.enable_pixel_dithering = bool(enable_pixel_dithering)
        self.pixel_dithering_prob = float(pixel_dithering_prob)
        self.enable_rgb_permutation = bool(enable_rgb_permutation)
        self.rgb_permutation_prob = float(rgb_permutation_prob)

        if compact_bundle is None:
            print(f"Loading PG/CCPi Ground Truth from {ground_truth_path}...")
            compact_bundle = _load_compact_pgccpi_records(ground_truth_path, metric)
        else:
            print("Reusing preloaded PG/CCPi compact records...")

        compact_bundle = _filter_pgccpi_records_by_kernel_indices(
            compact_bundle,
            include_kernel_indices=include_kernel_indices,
            exclude_kernel_indices=exclude_kernel_indices,
        )

        resolved_kernel_root = str(kernel_root or compact_bundle.get("kernel_root") or "").strip()

        self.compact_bundle = compact_bundle
        self.records = list(compact_bundle["records"])
        self.record_regularizers = list(
            compact_bundle.get(
                "regularizers",
                sorted(dict.fromkeys(normalize_pgccpi_regularizer_name(record["regularizer"]) for record in self.records)),
            )
        )
        self.record_data_terms = list(
            compact_bundle.get(
                "data_terms",
                sorted(dict.fromkeys(normalize_pgccpi_data_term_name(record["data_term"]) for record in self.records)),
            )
        )
        self.record_noise_models = list(
            compact_bundle.get(
                "noise_models",
                sorted(dict.fromkeys(normalize_pgccpi_noise_model_name(record["noise_model"]) for record in self.records)),
            )
        )
        self.record_operator_kinds = list(
            compact_bundle.get(
                "operator_kinds",
                sorted(dict.fromkeys(normalize_pgccpi_operator_name(record["operator_kind"]) for record in self.records)),
            )
        )
        self.has_mixed_regularizers = bool(len(self.record_regularizers) > 1)
        self.has_mixed_data_terms = bool(len(self.record_data_terms) > 1)
        self.has_mixed_noise_models = bool(len(self.record_noise_models) > 1)
        self.has_mixed_operator_kinds = bool(len(self.record_operator_kinds) > 1)
        self.regularizer = str(self.record_regularizers[0]).upper()
        self.target_dim = int(
            max(int(record.get("target_dim", len(record["target_vector"]))) for record in self.records)
        )
        if not self.has_mixed_regularizers:
            target_names = list(compact_bundle.get("target_names", []))
            if not target_names:
                target_names = list(self.records[0].get("target_names", []))
            self.target_names = list(target_names)
        else:
            self.target_names = [f"param_{index}" for index in range(self.target_dim)]
        self.data_term = str(self.record_data_terms[0])
        self.noise_model = str(self.record_noise_models[0])
        self.operator_kind = str(self.record_operator_kinds[0])
        self.tgv_parameterization = compact_bundle.get("tgv_parameterization")
        self.kernel_root = resolved_kernel_root
        if "KNOWN_PSF_DEBLUR" in self.record_operator_kinds:
            if not self.kernel_root:
                raise ValueError("PG/CCPi deblur dataset requires kernel_root either from GT config or constructor.")
            self.kernels = load_levin_kernels(self.kernel_root)
        else:
            self.kernels = []
        self.filtered_kernel_indices = list(compact_bundle.get("filtered_kernel_indices", []))
        self.length = len(self.records)
        self._epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self):
        return self.length

    def decode_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.has_mixed_regularizers or self.has_mixed_data_terms:
            raise ValueError(
                "decode_targets is not defined for mixed-family PG/CCPi bundles. "
                "Use decode_mixed_pgccpi_targets with per-sample regularizer/data-term ids."
            )
        return _decode_pgccpi_target_tensor(targets, self.regularizer, self.data_term)

    def _build_sample(self, clean_img, record):
        clean_np = _to_float_hwc(clean_img)
        clean_chw = clean_np.transpose((2, 0, 1))
        record_operator_kind = normalize_pgccpi_operator_name(record.get("operator_kind", self.operator_kind))
        record_noise_model = normalize_pgccpi_noise_model_name(record.get("noise_model", self.noise_model))
        # Only mix the current epoch into the augmentation seed when augmentations
        # are actually applied. This keeps val / eval (apply_augmentations=False)
        # fully deterministic while letting training see a fresh augmentation plan
        # every epoch.
        epoch_for_seed = int(self._epoch) if self.apply_augmentations else 0
        augmentation_seed = _resolve_record_augmentation_seed(
            record,
            operator_kind=record_operator_kind,
            epoch=epoch_for_seed,
        )
        if epoch_for_seed == 0:
            record.setdefault("augmentation_seed", int(augmentation_seed))
        apply_post_degrade_augment = True
        if record_operator_kind == "IDENTITY_DENOISE" and self.apply_augmentations and self.augment_before_degrade:
            clean_chw, _ = _apply_joint_training_augmentations(
                clean_chw,
                clean_chw,
                augmentation_seed=augmentation_seed,
                apply_augmentations=True,
                enable_rotate_90_270=self.enable_rotate_90_270,
                enable_pixel_dithering=self.enable_pixel_dithering,
                pixel_dithering_prob=self.pixel_dithering_prob,
                enable_rgb_permutation=self.enable_rgb_permutation,
                rgb_permutation_prob=self.rgb_permutation_prob,
            )
            apply_post_degrade_augment = False
        if record_operator_kind == "KNOWN_PSF_DEBLUR":
            kernel_info = self.kernels[int(record["kernel_index"])]
            noisy_chw = _poisson_gaussian_degrade_known_kernel(
                clean_chw=clean_chw,
                psf=np.asarray(kernel_info["psf"], dtype=np.float32),
                alpha=float(record["alpha"]),
                sigma_read=float(record["sigma_read"]),
                background=float(record["background"]),
                seed=int(record["noise_seed"]),
            )
        elif record_operator_kind == "IDENTITY_DENOISE":
            noise_params_raw = record.get("noise_params", {})
            noise_params = dict(noise_params_raw) if isinstance(noise_params_raw, dict) else {}
            if record_noise_model == "GAUSSIAN":
                noise_params.setdefault("sigma", float(record.get("noise_sigma", 0.0)))
            elif record_noise_model == "POISSON":
                noise_params.setdefault("alpha", float(record.get("alpha", 0.0)))
                noise_params.setdefault("background", float(record.get("background", 0.0)))
            elif record_noise_model == "POISSON_GAUSSIAN":
                noise_params.setdefault("alpha", float(record.get("alpha", 0.0)))
                noise_params.setdefault("sigma_read", float(record.get("sigma_read", 0.0)))
                noise_params.setdefault("background", float(record.get("background", 0.0)))
            elif record_noise_model == "IMPULSE":
                noise_params.setdefault("amount", float(noise_params.get("amount", 0.0)))
                noise_params.setdefault("salt_vs_pepper", float(noise_params.get("salt_vs_pepper", 0.5)))
                noise_params.setdefault("channel_independent", float(noise_params.get("channel_independent", 0.0)))
            elif record_noise_model == "IMPULSE_GAUSSIAN":
                noise_params.setdefault("amount", float(noise_params.get("amount", 0.0)))
                noise_params.setdefault("salt_vs_pepper", float(noise_params.get("salt_vs_pepper", 0.5)))
                noise_params.setdefault("channel_independent", float(noise_params.get("channel_independent", 0.0)))
                noise_params.setdefault("sigma", float(record.get("noise_sigma", noise_params.get("sigma", 0.0))))
            noise_rng = make_rng()
            if not self.resample_noise_realization:
                noise_rng = make_rng(int(record["noise_seed"]))
            noisy_chw = apply_noise(
                clean_chw,
                record_noise_model,
                rng=noise_rng,
                clip=False,
                **noise_params,
            ).noisy
        else:
            raise ValueError(f"Unsupported PG/CCPi operator_kind={record_operator_kind!r}.")
        if apply_post_degrade_augment:
            clean_chw, noisy_chw = _apply_joint_training_augmentations(
                clean_chw,
                noisy_chw,
                augmentation_seed=augmentation_seed,
                apply_augmentations=self.apply_augmentations,
                enable_rotate_90_270=self.enable_rotate_90_270,
                enable_pixel_dithering=self.enable_pixel_dithering,
                pixel_dithering_prob=self.pixel_dithering_prob,
                enable_rgb_permutation=self.enable_rgb_permutation,
                rgb_permutation_prob=self.rgb_permutation_prob,
            )
        clean_tensor = torch.from_numpy(clean_chw).float()
        noisy_raw_tensor = torch.from_numpy(noisy_chw).float()
        noisy_tensor = self.normalize(noisy_raw_tensor)
        raw_target = torch.tensor(record["target_vector"], dtype=torch.float32)
        if raw_target.numel() < self.target_dim:
            target_tensor = torch.zeros(self.target_dim, dtype=torch.float32)
            target_tensor[: raw_target.numel()] = raw_target
            target_mask = torch.zeros(self.target_dim, dtype=torch.float32)
            target_mask[: raw_target.numel()] = 1.0
        else:
            target_tensor = raw_target
            target_mask = torch.ones(self.target_dim, dtype=torch.float32)
        score_tensor = torch.tensor(float(record["best_score"]), dtype=torch.float32)
        return noisy_tensor, target_tensor, target_mask, score_tensor, clean_tensor, noisy_raw_tensor


class STL10PGCCPIDataset(_BasePGCCPIDataset):
    def __init__(
        self,
        stl10_root='./data',
        ground_truth_path=None,
        metric='PSNR',
        split='train',
        kernel_root=None,
        apply_augmentations=True,
        augment_before_degrade=False,
        resample_noise_realization=False,
        enable_rotate_90_270=True,
        enable_pixel_dithering=False,
        pixel_dithering_prob=0.0,
        enable_rgb_permutation=False,
        rgb_permutation_prob=0.0,
        include_kernel_indices=None,
        exclude_kernel_indices=None,
        compact_bundle=None,
    ):
        self.stl10 = torchvision.datasets.STL10(root=stl10_root, split=split, download=True)
        self.split = split
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.4467, 0.4398, 0.4066],
            std=[0.2603, 0.2566, 0.2713],
        )
        super().__init__(
            ground_truth_path=ground_truth_path,
            metric=metric,
            kernel_root=kernel_root,
            apply_augmentations=apply_augmentations,
            augment_before_degrade=augment_before_degrade,
            resample_noise_realization=resample_noise_realization,
            enable_rotate_90_270=enable_rotate_90_270,
            enable_pixel_dithering=enable_pixel_dithering,
            pixel_dithering_prob=pixel_dithering_prob,
            enable_rgb_permutation=enable_rgb_permutation,
            rgb_permutation_prob=rgb_permutation_prob,
            include_kernel_indices=include_kernel_indices,
            exclude_kernel_indices=exclude_kernel_indices,
            compact_bundle=compact_bundle,
        )
        print(
            f"STL10 PG/CCPi dataset loaded. Total samples: {self.length}. "
            f"Metric: {metric}. Regularizer: {self.regularizer}."
        )

    def __getitem__(self, idx):
        record = self.records[idx]
        clean_img, _ = self.stl10[int(record["image_index"])]
        return self._build_sample(clean_img, record)


class WEDPGCCPIDataset(_BasePGCCPIDataset):
    def __init__(
        self,
        wed_root='./data/WED_crop',
        ground_truth_path=None,
        metric='PSNR',
        split='train',
        kernel_root=None,
        apply_augmentations=True,
        augment_before_degrade=False,
        resample_noise_realization=False,
        enable_rotate_90_270=True,
        enable_pixel_dithering=False,
        pixel_dithering_prob=0.0,
        enable_rgb_permutation=False,
        rgb_permutation_prob=0.0,
        include_kernel_indices=None,
        exclude_kernel_indices=None,
        compact_bundle=None,
    ):
        from glob import glob

        if split == 'train':
            subdir = 'train'
        elif split in ('val', 'test'):
            subdir = 'val'
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        full_root = os.path.join(wed_root, subdir)
        self.image_paths = sorted(glob(os.path.join(full_root, '*.png')))
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {full_root}")

        self.split = split
        self.normalize = _build_normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset_name = os.path.basename(os.path.normpath(wed_root)) or "WED_crop"
        super().__init__(
            ground_truth_path=ground_truth_path,
            metric=metric,
            kernel_root=kernel_root,
            apply_augmentations=apply_augmentations,
            augment_before_degrade=augment_before_degrade,
            resample_noise_realization=resample_noise_realization,
            enable_rotate_90_270=enable_rotate_90_270,
            enable_pixel_dithering=enable_pixel_dithering,
            pixel_dithering_prob=pixel_dithering_prob,
            enable_rgb_permutation=enable_rgb_permutation,
            rgb_permutation_prob=rgb_permutation_prob,
            include_kernel_indices=include_kernel_indices,
            exclude_kernel_indices=exclude_kernel_indices,
            compact_bundle=compact_bundle,
        )
        print(
            f"{self.dataset_name} PG/CCPi dataset loaded. Total samples: {self.length}. "
            f"Metric: {metric}. Regularizer: {self.regularizer}."
        )

    def __getitem__(self, idx):
        from PIL import Image

        record = self.records[idx]
        clean_img = Image.open(self.image_paths[int(record["image_index"])]).convert('RGB')
        return self._build_sample(clean_img, record)


class Kodak24PGCCPIDataset(_BasePGCCPIDataset):
    def __init__(
        self,
        kodak24_root='./data/Kodak24',
        ground_truth_path=None,
        metric='PSNR',
        split='test',
        kernel_root=None,
        apply_augmentations=False,
        augment_before_degrade=False,
        resample_noise_realization=False,
        enable_rotate_90_270=False,
        enable_pixel_dithering=False,
        pixel_dithering_prob=0.0,
        enable_rgb_permutation=False,
        rgb_permutation_prob=0.0,
        include_kernel_indices=None,
        exclude_kernel_indices=None,
        compact_bundle=None,
    ):
        from glob import glob

        self.image_paths = sorted(
            glob(os.path.join(kodak24_root, '*.png')) + glob(os.path.join(kodak24_root, '*.bmp'))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {kodak24_root}")

        self.split = split
        self.normalize = _build_normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        super().__init__(
            ground_truth_path=ground_truth_path,
            metric=metric,
            kernel_root=kernel_root,
            apply_augmentations=apply_augmentations,
            augment_before_degrade=augment_before_degrade,
            resample_noise_realization=resample_noise_realization,
            enable_rotate_90_270=enable_rotate_90_270,
            enable_pixel_dithering=enable_pixel_dithering,
            pixel_dithering_prob=pixel_dithering_prob,
            enable_rgb_permutation=enable_rgb_permutation,
            rgb_permutation_prob=rgb_permutation_prob,
            include_kernel_indices=include_kernel_indices,
            exclude_kernel_indices=exclude_kernel_indices,
            compact_bundle=compact_bundle,
        )
        print(
            f"Kodak24 PG/CCPi dataset loaded. Total samples: {self.length}. "
            f"Metric: {metric}. Regularizer: {self.regularizer}."
        )

    def __getitem__(self, idx):
        from PIL import Image

        record = self.records[idx]
        clean_img = Image.open(self.image_paths[int(record["image_index"])]).convert('RGB')
        return self._build_sample(clean_img, record)


class CBSD68PGCCPIDataset(_BasePGCCPIDataset):
    def __init__(
        self,
        cbsd68_root='./data/CBSD68',
        ground_truth_path=None,
        metric='PSNR',
        split='test',
        kernel_root=None,
        apply_augmentations=False,
        augment_before_degrade=False,
        resample_noise_realization=False,
        enable_rotate_90_270=False,
        enable_pixel_dithering=False,
        pixel_dithering_prob=0.0,
        enable_rgb_permutation=False,
        rgb_permutation_prob=0.0,
        include_kernel_indices=None,
        exclude_kernel_indices=None,
        compact_bundle=None,
    ):
        from glob import glob
        self.image_paths = sorted(
            glob(os.path.join(cbsd68_root, '*.png')) + glob(os.path.join(cbsd68_root, '*.bmp'))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {cbsd68_root}")
        self.split = split
        self.normalize = _build_normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        super().__init__(
            ground_truth_path=ground_truth_path, metric=metric, kernel_root=kernel_root,
            apply_augmentations=apply_augmentations, augment_before_degrade=augment_before_degrade,
            resample_noise_realization=resample_noise_realization,
            enable_rotate_90_270=enable_rotate_90_270, enable_pixel_dithering=enable_pixel_dithering,
            pixel_dithering_prob=pixel_dithering_prob, enable_rgb_permutation=enable_rgb_permutation,
            rgb_permutation_prob=rgb_permutation_prob,
            include_kernel_indices=include_kernel_indices, exclude_kernel_indices=exclude_kernel_indices,
            compact_bundle=compact_bundle,
        )
        print(f"CBSD68 PG/CCPi dataset loaded. Total samples: {self.length}. "
              f"Metric: {metric}. Regularizer: {self.regularizer}.")

    def __getitem__(self, idx):
        from PIL import Image
        record = self.records[idx]
        clean_img = Image.open(self.image_paths[int(record["image_index"])]).convert('RGB')
        return self._build_sample(clean_img, record)


class Urban100PGCCPIDataset(_BasePGCCPIDataset):
    def __init__(
        self,
        urban100_root='./data/Urban100',
        ground_truth_path=None,
        metric='PSNR',
        split='test',
        kernel_root=None,
        apply_augmentations=False,
        augment_before_degrade=False,
        resample_noise_realization=False,
        enable_rotate_90_270=False,
        enable_pixel_dithering=False,
        pixel_dithering_prob=0.0,
        enable_rgb_permutation=False,
        rgb_permutation_prob=0.0,
        include_kernel_indices=None,
        exclude_kernel_indices=None,
        compact_bundle=None,
    ):
        from glob import glob
        self.image_paths = sorted(
            glob(os.path.join(urban100_root, '*.png')) + glob(os.path.join(urban100_root, '*.bmp'))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {urban100_root}")
        self.split = split
        self.normalize = _build_normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        super().__init__(
            ground_truth_path=ground_truth_path, metric=metric, kernel_root=kernel_root,
            apply_augmentations=apply_augmentations, augment_before_degrade=augment_before_degrade,
            resample_noise_realization=resample_noise_realization,
            enable_rotate_90_270=enable_rotate_90_270, enable_pixel_dithering=enable_pixel_dithering,
            pixel_dithering_prob=pixel_dithering_prob, enable_rgb_permutation=enable_rgb_permutation,
            rgb_permutation_prob=rgb_permutation_prob,
            include_kernel_indices=include_kernel_indices, exclude_kernel_indices=exclude_kernel_indices,
            compact_bundle=compact_bundle,
        )
        print(f"Urban100 PG/CCPi dataset loaded. Total samples: {self.length}. "
              f"Metric: {metric}. Regularizer: {self.regularizer}.")

    def __getitem__(self, idx):
        from PIL import Image
        record = self.records[idx]
        clean_img = Image.open(self.image_paths[int(record["image_index"])]).convert('RGB')
        return self._build_sample(clean_img, record)
