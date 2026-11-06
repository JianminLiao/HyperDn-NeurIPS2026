import json
import os
import signal
import tempfile
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from restoration.curve_pairs import (
    as_curve_storage_array,
    curve_pairs_path,
    initialize_curve_results,
)
from restoration.diffpir_official import OfficialDiffPIRDenoiser
from restoration.loaders import WEDLoader
from restoration.noise_models import apply_noise, make_rng
from restoration.pgccpi_param_encoding import (
    MAX_PGCCPI_PARAM_DIM,
    encode_pgccpi_target,
)


RUN_NAME = "auto"

DATASET = "WED_256"
SPLIT = "train"
IMAGE_INDICES = None

OPERATOR_KIND = "IDENTITY_DENOISE"
NOISE_MODEL = "GAUSSIAN"
PSF_FAMILY = "IDENTITY"
GAUSSIAN_KERNEL_SIGMA_MIN = 1.0
GAUSSIAN_KERNEL_SIGMA_MAX = 1.0
GAUSSIAN_KERNEL_SIGMA_STEPS = 1
GAUSSIAN_KERNEL_TRUNCATION = 3.0
SR_SCALE = 4
SR_OPERATOR = "bicubic"
LAMBDA_ENCODING_SIGMA = 0.0

DIFFPIR_REPO_ROOT = ""
DIFFPIR_DEVICE = "cuda"
DIFFPIR_MODEL_NAME = "256x256_diffusion_uncond"
DIFFPIR_MODEL_PATH = ""
DIFFPIR_CANDIDATE_BATCH_SIZE = 4

ITER_NUM = 100
ITER_NUM_U = 1
SKIP_TYPE = "quad"
ETA = 0.0
GUIDANCE_SCALE = 1.0
BETA_START = 1e-4
BETA_END = 2e-2
NUM_TRAIN_TIMESTEPS = 1000
DDIM_SAMPLE = False
NOISE_INIT_IMG = "max"

SAMPLES_PER_IMAGE_TRAIN = 1
SAMPLES_PER_IMAGE_TEST = 1
GAUSSIAN_NOISE_SIGMA_RANGE = (5.0 / 255.0, 15.0 / 255.0)

LAMBDA_MIN = 1e-8
LAMBDA_MAX = 1_000_000.0
LAMBDA_COARSE_STEPS = 13
LAMBDA_FINE_STEPS = 17
LAMBDA_REFINE_LEVELS = 3
ZETA_MIN = 0.0
ZETA_MAX = 1.0
ZETA_COARSE_STEPS = 7
ZETA_MID1_STEPS = 13
ZETA_MID2_STEPS = 13
ZETA_FINE_STEPS = 5

METRIC_NAMES = ["PSNR", "SSIM"]
PRIMARY_METRIC = "PSNR"

RUN_SEED = 20260331
CHECKPOINT_EVERY = 5
SAVE_DIR = "dataset_labels"

_STOP_AFTER_CURRENT_TASK = False
_RECEIVED_SIGNAL: Optional[int] = None


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def _task_key(image_index: int, sample_index: int) -> str:
    return f"img{int(image_index):05d}_sample{int(sample_index):03d}"


def _normalize_seed(seed: int) -> int:
    max_seed = 2**31 - 1
    value = int(seed) % max_seed
    return value if value != 0 else 1


def _load_torch_payload(path: str) -> Dict:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _request_graceful_stop(signum, _frame) -> None:
    global _STOP_AFTER_CURRENT_TASK, _RECEIVED_SIGNAL
    if _RECEIVED_SIGNAL is None:
        print(f"Received signal {int(signum)}; will checkpoint after the current task.", flush=True)
    _STOP_AFTER_CURRENT_TASK = True
    _RECEIVED_SIGNAL = int(signum)


def _install_signal_handlers() -> None:
    for signame in ("SIGTERM", "SIGINT"):
        if hasattr(signal, signame):
            signal.signal(getattr(signal, signame), _request_graceful_stop)


def _atomic_torch_save(payload: Dict, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(fd)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _parse_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else float(default)


def _parse_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else int(default)


def _parse_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "y", "on")


def _parse_env_optional_int_list(name: str, default):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    if ":" in raw and "," not in raw:
        start_str, end_str = raw.split(":", 1)
        start = int(start_str.strip())
        end = int(end_str.strip())
        return list(range(start, end))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_env_optional_float_tuple(name: str, default):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    parts = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two comma-separated floats.")
    return (float(parts[0]), float(parts[1]))


def _normalize_operator_kind(value: str) -> str:
    key = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "IDENTITY": "IDENTITY_DENOISE",
        "DENOISE": "IDENTITY_DENOISE",
        "DENOISING": "IDENTITY_DENOISE",
        "KNOWN_PSF": "KNOWN_PSF_DEBLUR",
        "KNOWN_PSF_DEBLURRING": "KNOWN_PSF_DEBLUR",
        "DEBLUR": "KNOWN_PSF_DEBLUR",
        "DEBLURRING": "KNOWN_PSF_DEBLUR",
        "SR": "SUPER_RESOLUTION",
        "SISR": "SUPER_RESOLUTION",
        "SUPER_RES": "SUPER_RESOLUTION",
        "SUPERRESOLUTION": "SUPER_RESOLUTION",
    }
    key = aliases.get(key, key)
    if key not in {"IDENTITY_DENOISE", "KNOWN_PSF_DEBLUR", "SUPER_RESOLUTION"}:
        raise ValueError(
            "OPERATOR_KIND must be IDENTITY_DENOISE, KNOWN_PSF_DEBLUR, or SUPER_RESOLUTION."
        )
    return key


def _apply_runtime_env_overrides() -> None:
    global RUN_NAME, DATASET, SPLIT, IMAGE_INDICES
    global OPERATOR_KIND, NOISE_MODEL, PSF_FAMILY
    global GAUSSIAN_KERNEL_SIGMA_MIN, GAUSSIAN_KERNEL_SIGMA_MAX, GAUSSIAN_KERNEL_SIGMA_STEPS, GAUSSIAN_KERNEL_TRUNCATION
    global SR_SCALE, SR_OPERATOR, LAMBDA_ENCODING_SIGMA
    global DIFFPIR_REPO_ROOT, DIFFPIR_DEVICE, DIFFPIR_MODEL_NAME, DIFFPIR_MODEL_PATH, DIFFPIR_CANDIDATE_BATCH_SIZE
    global ITER_NUM, ITER_NUM_U, SKIP_TYPE, ETA, GUIDANCE_SCALE, BETA_START, BETA_END, NUM_TRAIN_TIMESTEPS
    global DDIM_SAMPLE, NOISE_INIT_IMG
    global SAMPLES_PER_IMAGE_TRAIN, SAMPLES_PER_IMAGE_TEST, GAUSSIAN_NOISE_SIGMA_RANGE
    global LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COARSE_STEPS, LAMBDA_FINE_STEPS, LAMBDA_REFINE_LEVELS
    global ZETA_MIN, ZETA_MAX, ZETA_COARSE_STEPS, ZETA_MID1_STEPS, ZETA_MID2_STEPS, ZETA_FINE_STEPS
    global METRIC_NAMES, PRIMARY_METRIC, RUN_SEED, CHECKPOINT_EVERY, SAVE_DIR

    if os.getenv("RUN_NAME", "").strip():
        RUN_NAME = os.getenv("RUN_NAME", "").strip()
    if os.getenv("DATASET", "").strip():
        DATASET = os.getenv("DATASET", "").strip()
    if os.getenv("SPLIT", "").strip():
        SPLIT = os.getenv("SPLIT", "").strip()
    IMAGE_INDICES = _parse_env_optional_int_list("IMAGE_INDICES", IMAGE_INDICES)

    if os.getenv("OPERATOR_KIND", "").strip():
        OPERATOR_KIND = _normalize_operator_kind(os.getenv("OPERATOR_KIND", "").strip())
    if os.getenv("NOISE_MODEL", "").strip():
        NOISE_MODEL = str(os.getenv("NOISE_MODEL", "").strip()).upper().replace("-", "_").replace(" ", "_")
    if os.getenv("PSF_FAMILY", "").strip():
        PSF_FAMILY = str(os.getenv("PSF_FAMILY", "").strip()).upper().replace("-", "_").replace(" ", "_")
    GAUSSIAN_KERNEL_SIGMA_MIN = _parse_env_float("GAUSSIAN_KERNEL_SIGMA_MIN", GAUSSIAN_KERNEL_SIGMA_MIN)
    GAUSSIAN_KERNEL_SIGMA_MAX = _parse_env_float("GAUSSIAN_KERNEL_SIGMA_MAX", GAUSSIAN_KERNEL_SIGMA_MAX)
    GAUSSIAN_KERNEL_SIGMA_STEPS = _parse_env_int("GAUSSIAN_KERNEL_SIGMA_STEPS", GAUSSIAN_KERNEL_SIGMA_STEPS)
    GAUSSIAN_KERNEL_TRUNCATION = _parse_env_float("GAUSSIAN_KERNEL_TRUNCATION", GAUSSIAN_KERNEL_TRUNCATION)
    SR_SCALE = _parse_env_int("SR_SCALE", SR_SCALE)
    if os.getenv("SR_OPERATOR", "").strip():
        SR_OPERATOR = os.getenv("SR_OPERATOR", "").strip()
    LAMBDA_ENCODING_SIGMA = _parse_env_float("LAMBDA_ENCODING_SIGMA", LAMBDA_ENCODING_SIGMA)

    if os.getenv("DIFFPIR_REPO_ROOT", "").strip():
        DIFFPIR_REPO_ROOT = os.getenv("DIFFPIR_REPO_ROOT", "").strip()
    if os.getenv("DIFFPIR_DEVICE", "").strip():
        DIFFPIR_DEVICE = os.getenv("DIFFPIR_DEVICE", "").strip()
    if os.getenv("DIFFPIR_MODEL_NAME", "").strip():
        DIFFPIR_MODEL_NAME = os.getenv("DIFFPIR_MODEL_NAME", "").strip()
    if os.getenv("DIFFPIR_MODEL_PATH", "").strip():
        DIFFPIR_MODEL_PATH = os.getenv("DIFFPIR_MODEL_PATH", "").strip()
    DIFFPIR_CANDIDATE_BATCH_SIZE = _parse_env_int("DIFFPIR_CANDIDATE_BATCH_SIZE", DIFFPIR_CANDIDATE_BATCH_SIZE)

    ITER_NUM = _parse_env_int("ITER_NUM", ITER_NUM)
    ITER_NUM_U = _parse_env_int("ITER_NUM_U", ITER_NUM_U)
    if os.getenv("SKIP_TYPE", "").strip():
        SKIP_TYPE = os.getenv("SKIP_TYPE", "").strip()
    ETA = _parse_env_float("ETA", ETA)
    GUIDANCE_SCALE = _parse_env_float("GUIDANCE_SCALE", GUIDANCE_SCALE)
    BETA_START = _parse_env_float("BETA_START", BETA_START)
    BETA_END = _parse_env_float("BETA_END", BETA_END)
    NUM_TRAIN_TIMESTEPS = _parse_env_int("NUM_TRAIN_TIMESTEPS", NUM_TRAIN_TIMESTEPS)
    DDIM_SAMPLE = _parse_env_bool("DDIM_SAMPLE", DDIM_SAMPLE)
    if os.getenv("NOISE_INIT_IMG", "").strip():
        NOISE_INIT_IMG = os.getenv("NOISE_INIT_IMG", "").strip()

    SAMPLES_PER_IMAGE_TRAIN = _parse_env_int("SAMPLES_PER_IMAGE_TRAIN", SAMPLES_PER_IMAGE_TRAIN)
    SAMPLES_PER_IMAGE_TEST = _parse_env_int("SAMPLES_PER_IMAGE_TEST", SAMPLES_PER_IMAGE_TEST)
    GAUSSIAN_NOISE_SIGMA_RANGE = _parse_env_optional_float_tuple("GAUSSIAN_NOISE_SIGMA_RANGE", GAUSSIAN_NOISE_SIGMA_RANGE)

    LAMBDA_MIN = _parse_env_float("LAMBDA_MIN", LAMBDA_MIN)
    LAMBDA_MAX = _parse_env_float("LAMBDA_MAX", LAMBDA_MAX)
    LAMBDA_COARSE_STEPS = _parse_env_int("LAMBDA_COARSE_STEPS", LAMBDA_COARSE_STEPS)
    LAMBDA_FINE_STEPS = _parse_env_int("LAMBDA_FINE_STEPS", LAMBDA_FINE_STEPS)
    LAMBDA_REFINE_LEVELS = _parse_env_int("LAMBDA_REFINE_LEVELS", LAMBDA_REFINE_LEVELS)
    ZETA_MIN = _parse_env_float("ZETA_MIN", ZETA_MIN)
    ZETA_MAX = _parse_env_float("ZETA_MAX", ZETA_MAX)
    ZETA_COARSE_STEPS = _parse_env_int("ZETA_COARSE_STEPS", ZETA_COARSE_STEPS)
    ZETA_MID1_STEPS = _parse_env_int("ZETA_MID1_STEPS", ZETA_MID1_STEPS)
    ZETA_MID2_STEPS = _parse_env_int("ZETA_MID2_STEPS", ZETA_MID2_STEPS)
    ZETA_FINE_STEPS = _parse_env_int("ZETA_FINE_STEPS", ZETA_FINE_STEPS)

    if os.getenv("METRIC_NAMES", "").strip():
        METRIC_NAMES = [str(part).strip().upper() for part in os.getenv("METRIC_NAMES", "").split(",") if str(part).strip()]
    if os.getenv("PRIMARY_METRIC", "").strip():
        PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "").strip().upper()
    if str(PRIMARY_METRIC).upper() not in [str(name).upper() for name in METRIC_NAMES]:
        METRIC_NAMES = list(METRIC_NAMES) + [str(PRIMARY_METRIC).upper()]
    RUN_SEED = _parse_env_int("RUN_SEED", RUN_SEED)
    CHECKPOINT_EVERY = _parse_env_int("CHECKPOINT_EVERY", CHECKPOINT_EVERY)
    if os.getenv("SAVE_DIR", "").strip():
        SAVE_DIR = os.getenv("SAVE_DIR", "").strip()


def _build_loader(dataset_name: str, split: str):
    if dataset_name == "WED_256":
        return WEDLoader(root="./data/WED_256", split=split)
    if dataset_name == "WED_crop":
        return WEDLoader(root="./data/WED_crop", split=split)
    # 256-crop cross-dataset benchmarks; WEDLoader is layout-generic.
    if dataset_name == "CBSD68_256":
        return WEDLoader(root="./data/CBSD68_256", split=split)
    if dataset_name == "URBAN100_256":
        return WEDLoader(root="./data/URBAN100_256", split=split)
    if dataset_name == "FFHQ_256":
        return WEDLoader(root="./data/FFHQ_256", split=split)
    raise ValueError(
        f"Unsupported DATASET={dataset_name}. Expected WED_256, WED_crop, CBSD68_256, URBAN100_256, or FFHQ_256."
    )


def _normalize_image_indices(image_indices, dataset_len: int) -> List[int]:
    if image_indices is None:
        return list(range(dataset_len))
    if isinstance(image_indices, int):
        indices = [int(image_indices)]
    else:
        indices = [int(v) for v in image_indices]
    if not indices:
        raise ValueError("IMAGE_INDICES cannot be empty.")
    for index in indices:
        if index < 0 or index >= dataset_len:
            raise ValueError(f"Image index {index} out of range for dataset_len={dataset_len}.")
    return sorted(dict.fromkeys(indices))


def _samples_per_image(split: str) -> int:
    return int(SAMPLES_PER_IMAGE_TRAIN if str(split).strip().lower() == "train" else SAMPLES_PER_IMAGE_TEST)


def _build_gaussian_psf(sigma: float, truncation: float = 3.0) -> np.ndarray:
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        raise ValueError(f"Gaussian blur sigma must be positive, got {sigma}.")
    radius = max(1, int(np.ceil(float(truncation) * sigma_value)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    psf = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma_value * sigma_value)).astype(np.float32)
    psf_sum = float(psf.sum())
    if not np.isfinite(psf_sum) or psf_sum <= 0.0:
        raise ValueError(f"Invalid Gaussian PSF for sigma={sigma}.")
    return (psf / psf_sum).astype(np.float32, copy=False)


def _resolved_gaussian_kernel_sigma() -> float:
    if int(GAUSSIAN_KERNEL_SIGMA_STEPS) != 1:
        raise ValueError("DiffPIR transfer label search expects one fixed Gaussian blur kernel per run.")
    if not np.isclose(float(GAUSSIAN_KERNEL_SIGMA_MIN), float(GAUSSIAN_KERNEL_SIGMA_MAX)):
        raise ValueError("Set GAUSSIAN_KERNEL_SIGMA_MIN == GAUSSIAN_KERNEL_SIGMA_MAX for fixed-kernel DiffPIR runs.")
    return float(GAUSSIAN_KERNEL_SIGMA_MIN)


def _apply_known_kernel_blur(clean_chw: np.ndarray, psf: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve

    clean = np.asarray(clean_chw, dtype=np.float32)
    kernel = np.asarray(psf, dtype=np.float32)
    out = np.empty_like(clean, dtype=np.float32)
    for channel_index in range(clean.shape[0]):
        out[channel_index] = convolve(clean[channel_index], kernel, mode="same")
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _super_resolution_degrade(clean_chw: np.ndarray, scale: int, operator: str) -> np.ndarray:
    scale_int = int(scale)
    if scale_int <= 1:
        raise ValueError(f"SUPER_RESOLUTION requires SR_SCALE >= 2, got {scale}.")
    mode = str(operator).strip().lower()
    if mode not in {"bicubic", "area"}:
        raise ValueError(f"Unsupported SR_OPERATOR={operator!r}; expected bicubic or area.")
    clean = torch.from_numpy(np.asarray(clean_chw, dtype=np.float32)).unsqueeze(0)
    h, w = int(clean.shape[-2]), int(clean.shape[-1])
    lr_size = (max(1, h // scale_int), max(1, w // scale_int))
    if mode == "bicubic":
        lr = F.interpolate(clean, size=lr_size, mode="bicubic", align_corners=False)
    else:
        lr = F.interpolate(clean, size=lr_size, mode="area")
    return lr.squeeze(0).clamp(0.0, 1.0).numpy().astype(np.float32, copy=False)


def _sample_noise_sigma(seed: int) -> float:
    noise_key = str(NOISE_MODEL).strip().upper().replace("-", "_").replace(" ", "_")
    if noise_key == "UNKNOWN":
        return 0.0
    if noise_key != "GAUSSIAN":
        raise ValueError("DiffPIR transfer label search supports NOISE_MODEL=GAUSSIAN or UNKNOWN.")
    return _sample_gaussian_sigma(seed)


def _lambda_encoding_sigma(noise_sigma: float) -> float:
    if float(LAMBDA_ENCODING_SIGMA) > 0.0:
        return float(LAMBDA_ENCODING_SIGMA)
    if float(noise_sigma) > 0.0:
        return float(noise_sigma)
    return 1.0


def _build_observation(clean_chw: np.ndarray, seed: int) -> Dict[str, object]:
    operator_key = _normalize_operator_kind(OPERATOR_KIND)
    noise_key = str(NOISE_MODEL).strip().upper().replace("-", "_").replace(" ", "_")
    noise_sigma = _sample_noise_sigma(seed)
    rng = make_rng(seed)
    psf = None
    kernel_name = "identity"
    kernel_description = None

    if operator_key == "IDENTITY_DENOISE":
        if noise_key != "GAUSSIAN":
            raise ValueError("IDENTITY_DENOISE DiffPIR labels expect NOISE_MODEL=GAUSSIAN.")
        observed = apply_noise(
            clean_chw,
            "GAUSSIAN",
            rng=rng,
            clip=False,
            sigma=float(noise_sigma),
        ).noisy
    elif operator_key == "KNOWN_PSF_DEBLUR":
        psf_key = str(PSF_FAMILY).strip().upper().replace("-", "_").replace(" ", "_")
        if psf_key != "GAUSSIAN":
            raise ValueError("DiffPIR deblur transfer labels currently support PSF_FAMILY=GAUSSIAN.")
        sigma = _resolved_gaussian_kernel_sigma()
        psf = _build_gaussian_psf(sigma, truncation=float(GAUSSIAN_KERNEL_TRUNCATION))
        kernel_name = f"Gaussian blur sigma={sigma:.4f}"
        kernel_description = {
            "index": 0,
            "one_based": 1,
            "name": kernel_name,
            "sigma": float(sigma),
            "shape": [int(v) for v in psf.shape],
            "source": "synthetic_gaussian_kernel_bank",
            "psf_family": "GAUSSIAN",
        }
        observed = _apply_known_kernel_blur(clean_chw, psf)
        if noise_key == "GAUSSIAN":
            observed = apply_noise(
                observed,
                "GAUSSIAN",
                rng=rng,
                clip=False,
                sigma=float(noise_sigma),
            ).noisy
        elif noise_key != "UNKNOWN":
            raise ValueError("KNOWN_PSF_DEBLUR DiffPIR labels support NOISE_MODEL=GAUSSIAN or UNKNOWN.")
    elif operator_key == "SUPER_RESOLUTION":
        observed = _super_resolution_degrade(clean_chw, int(SR_SCALE), str(SR_OPERATOR))
        if noise_key == "GAUSSIAN":
            observed = apply_noise(
                observed,
                "GAUSSIAN",
                rng=rng,
                clip=False,
                sigma=float(noise_sigma),
            ).noisy
        elif noise_key != "UNKNOWN":
            raise ValueError("SUPER_RESOLUTION DiffPIR labels support NOISE_MODEL=GAUSSIAN or UNKNOWN.")
    else:
        raise ValueError(f"Unsupported OPERATOR_KIND={OPERATOR_KIND}.")

    noise_params = {"sigma": float(noise_sigma)} if noise_key == "GAUSSIAN" else {}
    return {
        "observed_chw": np.asarray(observed, dtype=np.float32),
        "psf": psf,
        "kernel_name": kernel_name,
        "kernel_description": kernel_description,
        "noise_sigma": float(noise_sigma),
        "noise_params": noise_params,
        "lambda_encoding_sigma": float(_lambda_encoding_sigma(noise_sigma)),
    }


def _seed_for_task(run_seed: int, image_index: int, sample_index: int) -> int:
    mixed = (
        2166136261
        + (int(run_seed) + 1) * 1000003
        + (int(image_index) + 1) * 9176
        + (int(sample_index) + 1) * 170003
    )
    return _normalize_seed(mixed)


def _sample_gaussian_sigma(seed: int) -> float:
    rng = np.random.RandomState(_normalize_seed(seed))
    return float(rng.uniform(GAUSSIAN_NOISE_SIGMA_RANGE[0], GAUSSIAN_NOISE_SIGMA_RANGE[1]))


def _search_spec() -> Dict[str, object]:
    coarse_lambda_grid = np.logspace(np.log10(LAMBDA_MIN), np.log10(LAMBDA_MAX), int(LAMBDA_COARSE_STEPS))
    coarse_zeta_grid = np.linspace(ZETA_MIN, ZETA_MAX, int(ZETA_COARSE_STEPS))
    return {
        "regularizer": "DIFFPIR",
        "search_mode": "coordinate_hierarchical_uniform4",
        "lambda_min": float(LAMBDA_MIN),
        "lambda_max": float(LAMBDA_MAX),
        "zeta_min": float(ZETA_MIN),
        "zeta_max": float(ZETA_MAX),
        "coarse_lambda_grid": [float(v) for v in coarse_lambda_grid.tolist()],
        "coarse_zeta_grid": [float(v) for v in coarse_zeta_grid.tolist()],
        "fine_lambda_steps": int(LAMBDA_FINE_STEPS),
        "lambda_refine_levels": int(LAMBDA_REFINE_LEVELS),
        "zeta_mid1_steps": int(ZETA_MID1_STEPS),
        "zeta_mid2_steps": int(ZETA_MID2_STEPS),
        "fine_zeta_steps": int(ZETA_FINE_STEPS),
    }


def _parameter_key(param: Dict[str, float]):
    return (
        f"{float(param['lambda']):.16e}",
        f"{float(param['zeta']):.16e}",
    )


def _unique_sorted_params(params: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    unique = {}
    for param in params:
        unique[_parameter_key(param)] = {"lambda": float(param["lambda"]), "zeta": float(param["zeta"])}
    return sorted(unique.values(), key=lambda item: (float(item["zeta"]), float(item["lambda"])))


def _local_log_grid(global_min: float, global_max: float, coarse_grid: Sequence[float], center_value: float, fine_steps: int) -> np.ndarray:
    coarse = np.asarray(coarse_grid, dtype=np.float64)
    center_idx = int(np.argmin(np.abs(coarse - float(center_value))))
    if coarse.size == 1 or int(fine_steps) <= 1:
        return np.asarray([float(coarse[center_idx])], dtype=np.float64)
    low_idx = max(0, center_idx - 1)
    high_idx = min(coarse.size - 1, center_idx + 1)
    low = max(float(global_min), float(coarse[low_idx]))
    high = min(float(global_max), float(coarse[high_idx]))
    fine = np.logspace(np.log10(low), np.log10(high), int(fine_steps))
    fine = np.concatenate([fine, np.asarray([float(coarse[center_idx])], dtype=np.float64)])
    return np.unique(np.asarray(fine, dtype=np.float64))


def _local_linear_grid(global_min: float, global_max: float, coarse_grid: Sequence[float], center_value: float, fine_steps: int) -> np.ndarray:
    coarse = np.asarray(coarse_grid, dtype=np.float64)
    center_idx = int(np.argmin(np.abs(coarse - float(center_value))))
    if coarse.size == 1 or int(fine_steps) <= 1:
        return np.asarray([float(coarse[center_idx])], dtype=np.float64)
    low_idx = max(0, center_idx - 1)
    high_idx = min(coarse.size - 1, center_idx + 1)
    low = max(float(global_min), float(coarse[low_idx]))
    high = min(float(global_max), float(coarse[high_idx]))
    fine = np.linspace(low, high, int(fine_steps))
    fine = np.concatenate([fine, np.asarray([float(coarse[center_idx])], dtype=np.float64)])
    return np.unique(np.asarray(fine, dtype=np.float64))


def _build_coarse_params(search_spec: Dict[str, object]) -> List[Dict[str, float]]:
    coarse_lambda_grid = np.asarray(search_spec["coarse_lambda_grid"], dtype=np.float64)
    coarse_zeta_grid = np.asarray(search_spec["coarse_zeta_grid"], dtype=np.float64)
    params = []
    for zeta in coarse_zeta_grid.tolist():
        for lambda_value in coarse_lambda_grid.tolist():
            params.append({"lambda": float(lambda_value), "zeta": float(zeta)})
    return params


def _build_fine_params(search_spec: Dict[str, object], coarse_best_param: Dict[str, float]) -> List[Dict[str, float]]:
    fine_lambda_grid = _local_log_grid(
        global_min=float(search_spec["lambda_min"]),
        global_max=float(search_spec["lambda_max"]),
        coarse_grid=search_spec["coarse_lambda_grid"],
        center_value=float(coarse_best_param["lambda"]),
        fine_steps=int(search_spec["fine_lambda_steps"]),
    )
    params = []
    for lambda_value in fine_lambda_grid.tolist():
        params.append({"lambda": float(lambda_value), "zeta": float(coarse_best_param["zeta"])})
    return params


def _build_refined_zeta_grid(
    *,
    global_min: float,
    global_max: float,
    parent_grid: Sequence[float],
    center_value: float,
    stage_steps: int,
) -> np.ndarray:
    return _local_linear_grid(
        global_min=float(global_min),
        global_max=float(global_max),
        coarse_grid=parent_grid,
        center_value=float(center_value),
        fine_steps=int(stage_steps),
    )


def _build_zeta_stage_params(lambda_value: float, zeta_grid: Sequence[float]) -> List[Dict[str, float]]:
    return [
        {"lambda": float(lambda_value), "zeta": float(zeta)}
        for zeta in np.asarray(zeta_grid, dtype=np.float64).tolist()
    ]


def _evaluate_uncached_params(
    *,
    denoiser: OfficialDiffPIRDenoiser,
    clean_chw: np.ndarray,
    noisy_chw: np.ndarray,
    lambda_encoding_sigma: float,
    params: Sequence[Dict[str, float]],
    metric_names: Sequence[str],
    seed: int,
    candidate_batch_size: int,
    evaluation_cache: Dict,
    operator_kind: str = "IDENTITY_DENOISE",
    psf: np.ndarray | None = None,
    sr_scale: int = 1,
) -> None:
    pending_params = [dict(param) for param in params if _parameter_key(param) not in evaluation_cache]
    if not pending_params:
        return
    # Convert oracle-search lambda (DiffPIR-original) to unified lambda = lambda_old * sigma^2
    sigma2 = float(lambda_encoding_sigma) ** 2
    solver_params = [
        {"lambda": float(p["lambda"]) * sigma2, "zeta": float(p["zeta"])}
        for p in pending_params
    ]
    stage_results = denoiser.evaluate_candidate_scores(
        clean_chw=clean_chw,
        noisy_chw=noisy_chw,
        params=solver_params,
        metric_names=metric_names,
        seed=seed,
        candidate_batch_size=candidate_batch_size,
        operator_kind=operator_kind,
        psf=psf,
        sr_scale=int(sr_scale),
    )
    # Cache with original (lambda_old, zeta) keys so downstream lookups match
    for original_param, result in zip(pending_params, stage_results):
        evaluation_cache[_parameter_key(original_param)] = {
            "param": dict(original_param),
            "scores": {str(k).upper(): float(v) for k, v in result.scores.items()},
        }


def _best_by_metric_from_cache(
    metric_names: Sequence[str],
    params: Sequence[Dict[str, float]],
    evaluation_cache: Dict,
) -> Dict[str, Dict[str, object]]:
    best = {}
    for name in metric_names:
        metric_name = str(name).upper()
        best[metric_name] = {
            "best_param": None,
            "best_score": (np.inf if metric_name == "LPIPS" else -np.inf),
        }
    for param in params:
        entry = evaluation_cache[_parameter_key(param)]
        for metric_name in best.keys():
            score = float(entry["scores"][metric_name])
            if (
                (metric_name == "LPIPS" and score < best[metric_name]["best_score"])
                or (metric_name != "LPIPS" and score > best[metric_name]["best_score"])
            ):
                best[metric_name]["best_score"] = score
                best[metric_name]["best_param"] = dict(param)
    return best


def _build_curve_trace(
    metric_names: Sequence[str],
    params: Sequence[Dict[str, float]],
    evaluation_cache: Dict,
    *,
    lambda_encoding_sigma: float,
) -> Dict[str, object]:
    curve_metric_names = [str(name).upper() for name in metric_names]
    param_vectors = []
    param_masks = []
    score_rows = []
    for param in params:
        entry = evaluation_cache[_parameter_key(param)]
        encoded = encode_pgccpi_target(
            "DIFFPIR",
            dict(param),
            data_term="L2",
            noise_sigma=float(lambda_encoding_sigma),
            pad_to=MAX_PGCCPI_PARAM_DIM,
        )
        param_vectors.append(np.asarray(encoded["padded_vector"], dtype=np.float32))
        param_masks.append(np.asarray(encoded["padded_mask"], dtype=np.float32))
        score_rows.append(
            np.asarray([float(entry["scores"][metric_name]) for metric_name in curve_metric_names], dtype=np.float32)
        )

    score_matrix = np.asarray(score_rows, dtype=np.float32)
    best_index_by_metric = {}
    for metric_index, metric_name in enumerate(curve_metric_names):
        if str(metric_name).upper() == "LPIPS":
            best_index_by_metric[str(metric_name)] = int(np.argmin(score_matrix[:, metric_index]))
        else:
            best_index_by_metric[str(metric_name)] = int(np.argmax(score_matrix[:, metric_index]))

    return {
        "param_vectors": np.asarray(param_vectors, dtype=np.float32),
        "param_masks": np.asarray(param_masks, dtype=np.float32),
        "metric_scores": score_matrix,
        "metric_names": list(curve_metric_names),
        "best_index_by_metric": best_index_by_metric,
    }


def _resolve_run_name(run_name: str, dataset: str, split: str, search_spec: Dict[str, object], run_seed: int) -> str:
    run_name_str = str(run_name).strip()
    if run_name_str and run_name_str.lower() != "auto":
        return _sanitize_name(run_name_str)
    lambda_min_token = f"{float(search_spec['lambda_min']):.0e}" if float(search_spec["lambda_min"]) < 1.0 else f"{float(search_spec['lambda_min']):.0f}"
    lambda_max_token = f"{float(search_spec['lambda_max']):.0e}" if float(search_spec["lambda_max"]) < 1.0 else f"{float(search_spec['lambda_max']):.0f}"
    search_token = (
        f"L{lambda_min_token}-{lambda_max_token}_"
        f"C{len(search_spec['coarse_lambda_grid'])}_F{int(search_spec['fine_lambda_steps'])}_"
        f"Z{float(search_spec['zeta_min']):.1f}-{float(search_spec['zeta_max']):.1f}_"
        f"C{len(search_spec['coarse_zeta_grid'])}_"
        f"M{int(search_spec['zeta_mid1_steps'])}_"
        f"M{int(search_spec['zeta_mid2_steps'])}_"
        f"F{int(search_spec['fine_zeta_steps'])}"
    )
    operator_token = _normalize_operator_kind(OPERATOR_KIND).lower()
    if operator_token == "known_psf_deblur":
        operator_token = f"gblur_s{_resolved_gaussian_kernel_sigma():g}"
    elif operator_token == "super_resolution":
        operator_token = f"sr_x{int(SR_SCALE)}_{str(SR_OPERATOR).lower()}"
    return _sanitize_name(
        f"diffpir_{operator_token}_{dataset}_{split}_{search_token}_NFE{int(ITER_NUM)}_seed{int(run_seed)}"
    )


def _write_summary_files(base_path: str, summary: Dict[str, object]) -> None:
    json_path = f"{base_path}_summary.json"
    txt_path = f"{base_path}_summary.txt"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    lines = [
        f"Run name: {summary.get('run_name')}",
        f"Dataset/split: {summary.get('dataset')}/{summary.get('split')}",
        f"Regularizer: {summary.get('regularizer')}",
        f"Noise model/operator: {summary.get('noise_model')}/{summary.get('operator_kind')}",
        f"Primary metric: {summary.get('primary_metric')}",
        f"Images processed: {summary.get('completed_tasks')}",
        f"Output: {summary.get('output_path')}",
        f"Curve sidecar: {summary.get('curve_pairs_path')}",
    ]
    metric_summary = dict(summary.get("metrics", {})).get(str(summary.get("primary_metric", "PSNR")).upper(), {})
    if metric_summary:
        lines.extend(
            [
                (
                    f"[{str(summary.get('primary_metric', 'PSNR')).upper()}] "
                    f"score_mean={float(metric_summary['score_mean']):.6f}, "
                    f"score_std={float(metric_summary['score_std']):.6f}"
                ),
                (
                    f"[{str(summary.get('primary_metric', 'PSNR')).upper()}] "
                    f"lambda={float(metric_summary['lambda_min']):.6g}.."
                    f"{float(metric_summary['lambda_max']):.6g}, "
                    f"lambda_boundary_ratio={float(metric_summary['lambda_boundary_ratio']):.2%}, "
                    f"zeta_boundary_ratio={float(metric_summary['zeta_boundary_ratio']):.2%}"
                ),
            ]
        )
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _summarize_records(
    records: Sequence[Dict[str, object]],
    *,
    config: Dict[str, object],
    primary_metric: str,
) -> Dict[str, object]:
    if not records:
        return {}
    lambdas = np.asarray(
        [float(record["optimal_param"]["lambda"]) for record in records],
        dtype=np.float64,
    )
    zetas = np.asarray(
        [float(record["optimal_param"]["zeta"]) for record in records],
        dtype=np.float64,
    )
    scores = np.asarray([float(record["best_score"]) for record in records], dtype=np.float64)
    image_indices = np.asarray([int(record["image_index"]) for record in records], dtype=np.int64)
    search_spec = dict(config.get("search_spec", {}))
    lambda_min = float(search_spec.get("lambda_min", np.nan))
    lambda_max = float(search_spec.get("lambda_max", np.nan))
    zeta_min = float(search_spec.get("zeta_min", np.nan))
    zeta_max = float(search_spec.get("zeta_max", np.nan))
    eps = 1e-12
    lambda_boundary = np.logical_or(
        np.isclose(lambdas, lambda_min, atol=eps),
        np.isclose(lambdas, lambda_max, atol=eps),
    )
    zeta_boundary = np.logical_or(
        np.isclose(zetas, zeta_min, atol=eps),
        np.isclose(zetas, zeta_max, atol=eps),
    )
    metric_name = str(primary_metric).upper()
    return {
        metric_name: {
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "lambda_mean": float(np.mean(lambdas)),
            "lambda_std": float(np.std(lambdas)),
            "lambda_min": float(np.min(lambdas)),
            "lambda_max": float(np.max(lambdas)),
            "lambda_unique_count": int(np.unique(lambdas).size),
            "lambda_low_boundary_count": int(np.sum(np.isclose(lambdas, lambda_min, atol=eps))),
            "lambda_high_boundary_count": int(np.sum(np.isclose(lambdas, lambda_max, atol=eps))),
            "lambda_boundary_ratio": float(np.mean(lambda_boundary)),
            "zeta_mean": float(np.mean(zetas)),
            "zeta_std": float(np.std(zetas)),
            "zeta_min": float(np.min(zetas)),
            "zeta_max": float(np.max(zetas)),
            "zeta_unique_count": int(np.unique(zetas).size),
            "zeta_low_boundary_count": int(np.sum(np.isclose(zetas, zeta_min, atol=eps))),
            "zeta_high_boundary_count": int(np.sum(np.isclose(zetas, zeta_max, atol=eps))),
            "zeta_boundary_ratio": float(np.mean(zeta_boundary)),
            "image_index_min": int(np.min(image_indices)),
            "image_index_max": int(np.max(image_indices)),
            "image_unique_count": int(np.unique(image_indices).size),
        }
    }


def _save_result_payloads(
    *,
    results: Dict,
    curve_results: Dict,
    out_path: str,
    curve_out_path: str,
) -> None:
    _atomic_torch_save(results, out_path)
    _atomic_torch_save(curve_results, curve_out_path)


def main():
    _apply_runtime_env_overrides()
    _install_signal_handlers()
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not str(DIFFPIR_REPO_ROOT).strip():
        raise ValueError("DIFFPIR_REPO_ROOT must point to the official DiffPIR checkout.")

    operator_key = _normalize_operator_kind(OPERATOR_KIND)
    noise_key = str(NOISE_MODEL).strip().upper().replace("-", "_").replace(" ", "_")
    psf_family_key = str(PSF_FAMILY).strip().upper().replace("-", "_").replace(" ", "_")
    if operator_key == "IDENTITY_DENOISE" and noise_key != "GAUSSIAN":
        raise ValueError("IDENTITY_DENOISE DiffPIR labels expect NOISE_MODEL=GAUSSIAN.")
    if operator_key == "KNOWN_PSF_DEBLUR" and psf_family_key != "GAUSSIAN":
        raise ValueError("KNOWN_PSF_DEBLUR DiffPIR labels currently expect PSF_FAMILY=GAUSSIAN.")
    if operator_key == "SUPER_RESOLUTION" and int(SR_SCALE) <= 1:
        raise ValueError("SUPER_RESOLUTION DiffPIR labels require SR_SCALE >= 2.")
    if noise_key not in {"GAUSSIAN", "UNKNOWN"}:
        raise ValueError("DiffPIR labels support NOISE_MODEL=GAUSSIAN or UNKNOWN.")

    loader = _build_loader(DATASET, SPLIT)
    image_indices = _normalize_image_indices(IMAGE_INDICES, len(loader))
    samples_per_image = _samples_per_image(SPLIT)
    search_spec = _search_spec()
    resolved_run_name = _resolve_run_name(RUN_NAME, DATASET, SPLIT, search_spec, RUN_SEED)

    out_path = os.path.join(SAVE_DIR, f"{resolved_run_name}.pt")
    curve_out_path = curve_pairs_path(out_path)

    current_config = {
        "run_name": resolved_run_name,
        "image_indices": [int(value) for value in image_indices],
        "dataset": DATASET,
        "split": SPLIT,
        "regularizer": "DIFFPIR",
        "data_term": "L2",
        "noise_model": noise_key,
        "operator_kind": operator_key,
        "psf_family": psf_family_key if operator_key == "KNOWN_PSF_DEBLUR" else "IDENTITY",
        "gaussian_kernel_sigma_range": [float(GAUSSIAN_KERNEL_SIGMA_MIN), float(GAUSSIAN_KERNEL_SIGMA_MAX)],
        "gaussian_kernel_sigma_steps": int(GAUSSIAN_KERNEL_SIGMA_STEPS),
        "gaussian_kernel_truncation": float(GAUSSIAN_KERNEL_TRUNCATION),
        "sr_scale": int(SR_SCALE) if operator_key == "SUPER_RESOLUTION" else 1,
        "sr_operator": str(SR_OPERATOR) if operator_key == "SUPER_RESOLUTION" else "",
        "lambda_encoding_sigma": float(LAMBDA_ENCODING_SIGMA),
        "metric_names": [str(name).upper() for name in METRIC_NAMES],
        "primary_metric": str(PRIMARY_METRIC).upper(),
        "samples_per_image": int(samples_per_image),
        "gaussian_noise_sigma_range": [float(v) for v in GAUSSIAN_NOISE_SIGMA_RANGE] if noise_key == "GAUSSIAN" else None,
        "search_spec": dict(search_spec),
        "iter_num": int(ITER_NUM),
        "iter_num_u": int(ITER_NUM_U),
        "skip_type": str(SKIP_TYPE),
        "eta": float(ETA),
        "guidance_scale": float(GUIDANCE_SCALE),
        "beta_start": float(BETA_START),
        "beta_end": float(BETA_END),
        "num_train_timesteps": int(NUM_TRAIN_TIMESTEPS),
        "ddim_sample": bool(DDIM_SAMPLE),
        "noise_init_img": str(NOISE_INIT_IMG),
        "diffpir_repo_root": os.path.abspath(str(DIFFPIR_REPO_ROOT)),
        "diffpir_model_name": str(DIFFPIR_MODEL_NAME),
        "diffpir_model_path": (
            os.path.abspath(str(DIFFPIR_MODEL_PATH))
            if str(DIFFPIR_MODEL_PATH).strip()
            else ""
        ),
        "run_seed": int(RUN_SEED),
    }
    curve_config = {
        "ground_truth_path": out_path,
        **current_config,
    }

    results = {
        "config": dict(current_config),
        "records": [],
        "regularizer": "DIFFPIR",
        "target_dim": 2,
        "target_names": ["log10_lambda", "zeta"],
        "data_term": "L2",
        "noise_model": noise_key,
        "operator_kind": operator_key,
        "regularizers": ["DIFFPIR"],
        "data_terms": ["L2"],
        "noise_models": [noise_key],
        "operator_kinds": [operator_key],
        "has_mixed_regularizers": False,
        "has_mixed_data_terms": False,
        "has_mixed_noise_models": False,
        "has_mixed_operator_kinds": False,
        "psf_family": psf_family_key if operator_key == "KNOWN_PSF_DEBLUR" else "IDENTITY",
        "kernel_descriptions": [],
        "gaussian_kernel_truncation": float(GAUSSIAN_KERNEL_TRUNCATION),
    }
    curve_results = initialize_curve_results(curve_config, current_config["metric_names"], max_param_dim=MAX_PGCCPI_PARAM_DIM)
    completed_keys = set()

    if os.path.exists(out_path):
        checkpoint = _load_torch_payload(out_path)
        if checkpoint.get("config", {}) != current_config:
            raise RuntimeError("Existing DiffPIR checkpoint config mismatch. Remove the old file or change RUN_NAME.")
        results = checkpoint
        completed_keys = {
            str(record["task_key"])
            for record in list(results.get("records", []))
        }
        if os.path.exists(curve_out_path):
            curve_checkpoint = _load_torch_payload(curve_out_path)
            if curve_checkpoint.get("config", {}) != curve_config:
                raise RuntimeError("Existing DiffPIR curve sidecar config mismatch. Remove the old file or change RUN_NAME.")
            curve_results = curve_checkpoint
            if len(curve_results.get("records", [])) != len(completed_keys):
                raise RuntimeError(
                    f"Curve sidecar record count mismatch for {curve_out_path}: "
                    f"main={len(completed_keys)} sidecar={len(curve_results.get('records', []))}"
                )
        elif completed_keys:
            raise RuntimeError(f"Main checkpoint exists but curve sidecar is missing: {curve_out_path}")

    denoiser = OfficialDiffPIRDenoiser(
        repo_root=DIFFPIR_REPO_ROOT,
        device=DIFFPIR_DEVICE,
        model_name=DIFFPIR_MODEL_NAME,
        model_path=DIFFPIR_MODEL_PATH,
        iter_num=ITER_NUM,
        iter_num_u=ITER_NUM_U,
        skip_type=SKIP_TYPE,
        eta=ETA,
        guidance_scale=GUIDANCE_SCALE,
        beta_start=BETA_START,
        beta_end=BETA_END,
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        ddim_sample=DDIM_SAMPLE,
        noise_init_img=NOISE_INIT_IMG,
    )

    task_items = []
    for image_index in image_indices:
        for sample_index in range(samples_per_image):
            key = _task_key(image_index, sample_index)
            task_items.append((int(image_index), int(sample_index), key))

    print(f"Run name: {resolved_run_name}")
    print(f"Output path: {out_path}")
    print(f"Curve sidecar path: {curve_out_path}")
    print(f"Tasks: {len(task_items)} total | {len(completed_keys)} already completed")

    for task_idx, (image_index, sample_index, key) in enumerate(tqdm(task_items, desc="DiffPIR teacher search"), start=1):
        if key in completed_keys:
            continue

        clean_chw = np.asarray(loader.get_image(image_index), dtype=np.float32)
        noise_seed = _seed_for_task(RUN_SEED, image_index, sample_index)
        observation = _build_observation(clean_chw, noise_seed)
        noisy_chw = np.asarray(observation["observed_chw"], dtype=np.float32)
        noise_sigma = float(observation["noise_sigma"])
        lambda_sigma = float(observation["lambda_encoding_sigma"])
        psf = observation["psf"]
        kernel_description = observation.get("kernel_description")
        if kernel_description is not None and not results.get("kernel_descriptions"):
            results["kernel_descriptions"] = [dict(kernel_description)]

        coarse_params = _build_coarse_params(search_spec)
        evaluation_cache = {}
        _evaluate_uncached_params(
            denoiser=denoiser,
            clean_chw=clean_chw,
            noisy_chw=noisy_chw,
            lambda_encoding_sigma=lambda_sigma,
            params=coarse_params,
            metric_names=METRIC_NAMES,
            seed=noise_seed,
            candidate_batch_size=DIFFPIR_CANDIDATE_BATCH_SIZE,
            evaluation_cache=evaluation_cache,
            operator_kind=operator_key,
            psf=psf,
            sr_scale=int(SR_SCALE),
        )
        coarse_best = _best_by_metric_from_cache(
            [PRIMARY_METRIC],
            coarse_params,
            evaluation_cache,
        )[str(PRIMARY_METRIC).upper()]

        current_best = coarse_best
        current_lambda_grid = np.asarray(search_spec["coarse_lambda_grid"], dtype=np.float64)
        refine_levels = int(search_spec.get("lambda_refine_levels", 1))
        for _lambda_level in range(refine_levels):
            fine_lambda_grid = _local_log_grid(
                global_min=float(search_spec["lambda_min"]),
                global_max=float(search_spec["lambda_max"]),
                coarse_grid=current_lambda_grid,
                center_value=float(current_best["best_param"]["lambda"]),
                fine_steps=int(search_spec["fine_lambda_steps"]),
            )
            lambda_refine_params = [
                {"lambda": float(v), "zeta": float(current_best["best_param"]["zeta"])}
                for v in fine_lambda_grid.tolist()
            ]
            _evaluate_uncached_params(
                denoiser=denoiser,
                clean_chw=clean_chw,
                noisy_chw=noisy_chw,
                lambda_encoding_sigma=lambda_sigma,
                params=lambda_refine_params,
                metric_names=METRIC_NAMES,
                seed=noise_seed,
                candidate_batch_size=DIFFPIR_CANDIDATE_BATCH_SIZE,
                evaluation_cache=evaluation_cache,
                operator_kind=operator_key,
                psf=psf,
                sr_scale=int(SR_SCALE),
            )
            current_best = _best_by_metric_from_cache(
                [PRIMARY_METRIC],
                _unique_sorted_params([entry["param"] for entry in evaluation_cache.values()]),
                evaluation_cache,
            )[str(PRIMARY_METRIC).upper()]
            current_lambda_grid = np.unique(np.concatenate([current_lambda_grid, fine_lambda_grid]))

        current_best = _best_by_metric_from_cache(
            [PRIMARY_METRIC],
            _unique_sorted_params([entry["param"] for entry in evaluation_cache.values()]),
            evaluation_cache,
        )[str(PRIMARY_METRIC).upper()]
        current_zeta_grid = np.asarray(search_spec["coarse_zeta_grid"], dtype=np.float64)
        zeta_stage_specs = [
            ("mid1", int(search_spec["zeta_mid1_steps"])),
            ("mid2", int(search_spec["zeta_mid2_steps"])),
            ("fine", int(search_spec["fine_zeta_steps"])),
        ]
        for _stage_name, stage_steps in zeta_stage_specs:
            current_zeta_grid = _build_refined_zeta_grid(
                global_min=float(search_spec["zeta_min"]),
                global_max=float(search_spec["zeta_max"]),
                parent_grid=current_zeta_grid,
                center_value=float(current_best["best_param"]["zeta"]),
                stage_steps=int(stage_steps),
            )
            zeta_stage_params = _build_zeta_stage_params(
                float(current_best["best_param"]["lambda"]),
                current_zeta_grid,
            )
            _evaluate_uncached_params(
                denoiser=denoiser,
                clean_chw=clean_chw,
                noisy_chw=noisy_chw,
                lambda_encoding_sigma=lambda_sigma,
                params=zeta_stage_params,
                metric_names=METRIC_NAMES,
                seed=noise_seed,
                candidate_batch_size=DIFFPIR_CANDIDATE_BATCH_SIZE,
                evaluation_cache=evaluation_cache,
                operator_kind=operator_key,
                psf=psf,
                sr_scale=int(SR_SCALE),
            )
            current_best = _best_by_metric_from_cache(
                [PRIMARY_METRIC],
                _unique_sorted_params([entry["param"] for entry in evaluation_cache.values()]),
                evaluation_cache,
            )[str(PRIMARY_METRIC).upper()]

        all_params = _unique_sorted_params([entry["param"] for entry in evaluation_cache.values()])
        best_by_metric = _best_by_metric_from_cache(METRIC_NAMES, all_params, evaluation_cache)
        primary_best = best_by_metric[str(PRIMARY_METRIC).upper()]
        target_info = encode_pgccpi_target(
            "DIFFPIR",
            dict(primary_best["best_param"]),
            data_term="L2",
            noise_sigma=float(lambda_sigma),
        )
        curve_trace = _build_curve_trace(
            METRIC_NAMES,
            all_params,
            evaluation_cache,
            lambda_encoding_sigma=float(lambda_sigma),
        )

        results["records"].append(
            {
                "task_key": str(key),
                "image_index": int(image_index),
                "sample_index": int(sample_index),
                "kernel_index": 0,
                "kernel_one_based": 1 if operator_key == "KNOWN_PSF_DEBLUR" else 0,
                "kernel_name": str(observation["kernel_name"]),
                "alpha": 0.0,
                "sigma_read": 0.0,
                "background": 0.0,
                "noise_sigma": float(noise_sigma),
                "lambda_encoding_sigma": float(lambda_sigma),
                "noise_seed": int(noise_seed),
                "noise_params": dict(observation["noise_params"]),
                "optimal_param": {
                    "lambda": float(primary_best["best_param"]["lambda"]),
                    "zeta": float(primary_best["best_param"]["zeta"]),
                },
                "target_vector": target_info["vector"].tolist(),
                "target_names": list(target_info["names"]),
                "target_dim": int(len(target_info["vector"])),
                "best_score": float(primary_best["best_score"]),
                "regularizer": "DIFFPIR",
                "data_term": "L2",
                "noise_model": noise_key,
                "operator_kind": operator_key,
                "sr_scale": int(SR_SCALE) if operator_key == "SUPER_RESOLUTION" else 1,
                "sr_operator": str(SR_OPERATOR) if operator_key == "SUPER_RESOLUTION" else "",
            }
        )
        curve_results["records"].append(
            {
                "task_key": str(key),
                "image_index": int(image_index),
                "sample_index": int(sample_index),
                "kernel_index": 0,
                "kernel_one_based": 1 if operator_key == "KNOWN_PSF_DEBLUR" else 0,
                "kernel_name": str(observation["kernel_name"]),
                "alpha": 0.0,
                "sigma_read": 0.0,
                "background": 0.0,
                "noise_sigma": float(noise_sigma),
                "lambda_encoding_sigma": float(lambda_sigma),
                "noise_seed": int(noise_seed),
                "noise_params": dict(observation["noise_params"]),
                "curve_param_vectors": as_curve_storage_array(curve_trace["param_vectors"]),
                "curve_param_masks": as_curve_storage_array(curve_trace["param_masks"]),
                "curve_metric_scores": as_curve_storage_array(curve_trace["metric_scores"]),
                "curve_metric_names": list(curve_trace["metric_names"]),
                "curve_best_index_by_metric": dict(curve_trace["best_index_by_metric"]),
            }
        )
        completed_keys.add(str(key))

        if task_idx % int(CHECKPOINT_EVERY) == 0 or _STOP_AFTER_CURRENT_TASK:
            _save_result_payloads(
                results=results,
                curve_results=curve_results,
                out_path=out_path,
                curve_out_path=curve_out_path,
            )
        if _STOP_AFTER_CURRENT_TASK:
            break

    _save_result_payloads(
        results=results,
        curve_results=curve_results,
        out_path=out_path,
        curve_out_path=curve_out_path,
    )
    summary = {
        "run_name": resolved_run_name,
        "dataset": DATASET,
        "split": SPLIT,
        "regularizer": "DIFFPIR",
        "data_term": "L2",
        "noise_model": noise_key,
        "operator_kind": operator_key,
        "psf_family": psf_family_key if operator_key == "KNOWN_PSF_DEBLUR" else "IDENTITY",
        "sr_scale": int(SR_SCALE) if operator_key == "SUPER_RESOLUTION" else 1,
        "sr_operator": str(SR_OPERATOR) if operator_key == "SUPER_RESOLUTION" else "",
        "lambda_encoding_sigma": float(LAMBDA_ENCODING_SIGMA),
        "primary_metric": str(PRIMARY_METRIC).upper(),
        "metric_names": [str(name).upper() for name in METRIC_NAMES],
        "completed_tasks": int(len(results.get("records", []))),
        "metrics": _summarize_records(
            list(results.get("records", [])),
            config=current_config,
            primary_metric=str(PRIMARY_METRIC).upper(),
        ),
        "output_path": out_path,
        "curve_pairs_path": curve_out_path,
        "search_spec": dict(search_spec),
    }
    _write_summary_files(os.path.splitext(out_path)[0], summary)
    print(f"Saved main bundle to {out_path}")
    print(f"Saved curve sidecar to {curve_out_path}")
    return out_path, summary


if __name__ == "__main__":
    main()
