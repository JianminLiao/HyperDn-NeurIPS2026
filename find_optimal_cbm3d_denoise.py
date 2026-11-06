"""PSNR-oracle search for the single CBM3D ``sigma_psd`` parameter on WED256.

The corruption protocol matches the paper's DiffPIR transfer
experiment: one deterministic Gaussian realization per image, RUN_SEED=20260331,
and sigma_true sampled uniformly from [5/255, 50/255].  Only ``sigma_psd`` is
passed to ``bm3d_rgb``; every other CBM3D argument uses the package default.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from restoration.loaders import WEDLoader
from restoration.metrics import PSNRMetric
from restoration.noise_models import apply_noise, make_rng


RUN_MODE = os.getenv("RUN_MODE", "search").strip().lower()
RUN_NAME = os.getenv("RUN_NAME", "cbm3d_wed256").strip()
SPLIT = os.getenv("SPLIT", "train").strip().lower()
SAVE_DIR = Path(os.getenv("SAVE_DIR", "dataset_labels/cbm3d_wed256"))
RUN_SEED = int(os.getenv("RUN_SEED", "20260331"))
NUM_SHARDS = int(os.getenv("NUM_SHARDS", "1"))
SHARD_ID = int(os.getenv("SHARD_ID", "0"))
NOISE_SIGMA_MIN = float(os.getenv("GAUSSIAN_NOISE_SIGMA_MIN", str(5.0 / 255.0)))
NOISE_SIGMA_MAX = float(os.getenv("GAUSSIAN_NOISE_SIGMA_MAX", str(50.0 / 255.0)))
SIGMA_PSD_MIN = float(os.getenv("SIGMA_PSD_MIN", "0.003"))
SIGMA_PSD_MAX = float(os.getenv("SIGMA_PSD_MAX", "0.30"))
COARSE_STEPS = int(os.getenv("SIGMA_PSD_COARSE_STEPS", "16"))
REFINE_STEPS = int(os.getenv("SIGMA_PSD_REFINE_STEPS", "16"))
REFINE_LEVELS = int(os.getenv("SIGMA_PSD_REFINE_LEVELS", "2"))
DELETE_SHARDS_AFTER_MERGE = os.getenv("DELETE_SHARDS_AFTER_MERGE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _parse_indices(raw: str, default_stop: int) -> List[int]:
    raw = raw.strip()
    if not raw:
        return list(range(default_stop))
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if "-" in token:
            left, right = token.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(token))
    return values


def _normalize_seed(seed: int) -> int:
    return int(seed) % (2**31 - 1)


def _seed_for_task(run_seed: int, image_index: int, sample_index: int = 0) -> int:
    mixed = (
        2166136261
        + (int(run_seed) + 1) * 1000003
        + (int(image_index) + 1) * 9176
        + (int(sample_index) + 1) * 170003
    )
    return _normalize_seed(mixed)


def _sample_noise_sigma(seed: int) -> float:
    rng = np.random.RandomState(_normalize_seed(seed))
    return float(rng.uniform(NOISE_SIGMA_MIN, NOISE_SIGMA_MAX))


def _cbm3d(noisy_chw: np.ndarray, sigma_psd: float) -> np.ndarray:
    from bm3d import bm3d_rgb

    restored_hwc = bm3d_rgb(
        np.transpose(np.asarray(noisy_chw, dtype=np.float32), (1, 2, 0)),
        sigma_psd=float(sigma_psd),
    )
    return np.transpose(np.asarray(restored_hwc, dtype=np.float32), (2, 0, 1))


def _local_log_grid(grid: np.ndarray, best_value: float) -> np.ndarray:
    grid = np.unique(np.asarray(grid, dtype=np.float64))
    best_index = int(np.argmin(np.abs(grid - float(best_value))))
    left = SIGMA_PSD_MIN if best_index == 0 else float(grid[best_index - 1])
    right = SIGMA_PSD_MAX if best_index == len(grid) - 1 else float(grid[best_index + 1])
    return np.geomspace(left, right, REFINE_STEPS)


def _search_one(clean_chw: np.ndarray, image_index: int) -> Dict:
    noise_seed = _seed_for_task(RUN_SEED, image_index)
    noise_sigma = _sample_noise_sigma(noise_seed)
    noisy_chw = apply_noise(
        clean_chw,
        "GAUSSIAN",
        rng=make_rng(noise_seed),
        clip=False,
        sigma=noise_sigma,
    ).noisy
    metric = PSNRMetric()
    cache: Dict[float, float] = {}

    def score(sigma_psd: float) -> float:
        key = float(np.round(float(sigma_psd), 12))
        if key not in cache:
            value = metric.calculate(clean_chw, _cbm3d(noisy_chw, key))
            cache[key] = float(value.item() if torch.is_tensor(value) else value)
        return cache[key]

    current_grid = np.geomspace(SIGMA_PSD_MIN, SIGMA_PSD_MAX, COARSE_STEPS)
    for value in current_grid:
        score(float(value))
    for _ in range(REFINE_LEVELS):
        best_value = max(cache, key=cache.get)
        current_grid = np.unique(np.concatenate([current_grid, _local_log_grid(current_grid, best_value)]))
        for value in current_grid:
            score(float(value))

    best_sigma = float(max(cache, key=cache.get))
    best_score = float(cache[best_sigma])
    sorted_curve = sorted(cache.items())
    return {
        "task_key": f"img{image_index:05d}_sample000",
        "image_index": int(image_index),
        "sample_index": 0,
        "kernel_index": 0,
        "kernel_one_based": 0,
        "kernel_name": "identity",
        "alpha": 0.0,
        "sigma_read": 0.0,
        "background": 0.0,
        "noise_sigma": float(noise_sigma),
        "noise_seed": int(noise_seed),
        "noise_params": {"sigma": float(noise_sigma)},
        "optimal_param": {"sigma_psd": best_sigma},
        "target_vector": [float(np.log10(best_sigma))],
        "target_names": ["log10_sigma_psd"],
        "target_role_names": ["unknown"],
        "target_dim": 1,
        "best_score": best_score,
        "regularizer": "CBM3D",
        "data_term": "L2",
        "noise_model": "GAUSSIAN",
        "operator_kind": "IDENTITY_DENOISE",
        "search_info": {
            "evaluated_sigma_psd_count": len(cache),
            "sigma_psd_values": [float(v) for v, _ in sorted_curve],
            "psnr_values": [float(v) for _, v in sorted_curve],
        },
    }


def _base_config(indices: Iterable[int]) -> Dict:
    return {
        "run_name": RUN_NAME,
        "dataset": "WED_256",
        "split": SPLIT,
        "image_indices": [int(v) for v in indices],
        "samples_per_image": 1,
        "run_seed": RUN_SEED,
        "regularizer": "CBM3D",
        "data_term": "L2",
        "noise_model": "GAUSSIAN",
        "operator_kind": "IDENTITY_DENOISE",
        "parameter_semantics": "unknown",
        "target_names": ["log10_sigma_psd"],
        "target_role_names": ["unknown"],
        "gaussian_noise_sigma_range": [NOISE_SIGMA_MIN, NOISE_SIGMA_MAX],
        "search_spec": {
            "sigma_psd_min": SIGMA_PSD_MIN,
            "sigma_psd_max": SIGMA_PSD_MAX,
            "coarse_steps": COARSE_STEPS,
            "refine_steps": REFINE_STEPS,
            "refine_levels": REFINE_LEVELS,
            "scale": "log10",
        },
        "cbm3d_call": {
            "function": "bm3d.bm3d_rgb",
            "explicit_arguments": ["noisy_image", "sigma_psd"],
            "all_other_arguments": "package_defaults",
        },
        "num_shards": NUM_SHARDS,
    }


def _bundle(config: Dict, records: List[Dict]) -> Dict:
    return {
        "config": dict(config),
        "regularizer": "CBM3D",
        "target_dim": 1,
        "target_names": ["log10_sigma_psd"],
        "target_role_names": ["unknown"],
        "data_term": "L2",
        "noise_model": "GAUSSIAN",
        "operator_kind": "IDENTITY_DENOISE",
        "regularizers": ["CBM3D"],
        "data_terms": ["L2"],
        "noise_models": ["GAUSSIAN"],
        "operator_kinds": ["IDENTITY_DENOISE"],
        "has_mixed_regularizers": False,
        "has_mixed_data_terms": False,
        "has_mixed_noise_models": False,
        "has_mixed_operator_kinds": False,
        "psf_family": "IDENTITY",
        "kernel_descriptions": [],
        "records": records,
    }


def _write_summary(path: Path, records: List[Dict], config: Dict) -> None:
    sigmas = np.asarray([r["optimal_param"]["sigma_psd"] for r in records], dtype=np.float64)
    scores = np.asarray([r["best_score"] for r in records], dtype=np.float64)
    summary = {
        "run_name": RUN_NAME,
        "dataset": "WED_256",
        "split": SPLIT,
        "completed_tasks": len(records),
        "output_path": str(path),
        "parameter_semantics": "unknown",
        "sigma_psd_min": float(sigmas.min()),
        "sigma_psd_max": float(sigmas.max()),
        "sigma_psd_mean": float(sigmas.mean()),
        "sigma_psd_std": float(sigmas.std()),
        "sigma_psd_low_boundary_count": int(np.isclose(sigmas, SIGMA_PSD_MIN).sum()),
        "sigma_psd_high_boundary_count": int(np.isclose(sigmas, SIGMA_PSD_MAX).sum()),
        "oracle_psnr_mean": float(scores.mean()),
        "oracle_psnr_std": float(scores.std()),
        "search_spec": config["search_spec"],
        "cbm3d_call": config["cbm3d_call"],
    }
    path.with_name(path.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def run_search() -> None:
    default_stop = 128 if SPLIT == "train" else 50
    indices = _parse_indices(os.getenv("IMAGE_INDICES", ""), default_stop)
    shard_indices = [value for ordinal, value in enumerate(indices) if ordinal % NUM_SHARDS == SHARD_ID]
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output = SAVE_DIR / f"{RUN_NAME}_shard_{SHARD_ID:03d}of{NUM_SHARDS:03d}.pt"
    config = _base_config(indices)
    records: List[Dict] = []
    if output.exists():
        records = list(torch.load(output, map_location="cpu", weights_only=False).get("records", []))
    completed = {int(record["image_index"]) for record in records}
    loader = WEDLoader(root="./data/WED_256", split=SPLIT)
    for image_index in shard_indices:
        if image_index in completed:
            continue
        clean_chw = np.asarray(loader.get_image(image_index), dtype=np.float32)
        record = _search_one(clean_chw, image_index)
        records.append(record)
        records.sort(key=lambda item: int(item["image_index"]))
        torch.save(_bundle(config, records), output)
        print(
            f"[{SHARD_ID}/{NUM_SHARDS}] image={image_index} "
            f"sigma_psd={record['optimal_param']['sigma_psd']:.6f} "
            f"PSNR={record['best_score']:.4f}",
            flush=True,
        )
    _write_summary(output, records, config)


def run_merge() -> None:
    default_stop = 128 if SPLIT == "train" else 50
    indices = _parse_indices(os.getenv("IMAGE_INDICES", ""), default_stop)
    config = _base_config(indices)
    shard_paths = sorted(SAVE_DIR.glob(f"{RUN_NAME}_shard_*of{NUM_SHARDS:03d}.pt"))
    if len(shard_paths) != NUM_SHARDS:
        raise RuntimeError(f"Expected {NUM_SHARDS} shards, found {len(shard_paths)}")
    records: List[Dict] = []
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload.get("records", []))
    records.sort(key=lambda item: int(item["image_index"]))
    if [int(record["image_index"]) for record in records] != indices:
        raise RuntimeError("Merged CBM3D records do not match the requested image indices")
    output = SAVE_DIR / f"{RUN_NAME}_merged.pt"
    merged_config = dict(config)
    merged_config["merged_from_shards"] = NUM_SHARDS
    torch.save(_bundle(merged_config, records), output)
    _write_summary(output, records, merged_config)
    if DELETE_SHARDS_AFTER_MERGE:
        for path in shard_paths:
            path.unlink()
            summary_path = path.with_name(path.stem + "_summary.json")
            if summary_path.exists():
                summary_path.unlink()
    print(f"Merged {len(records)} records -> {output}", flush=True)


def main() -> None:
    if RUN_MODE == "search":
        run_search()
    elif RUN_MODE == "merge":
        run_merge()
    else:
        raise ValueError(f"RUN_MODE must be search or merge, got {RUN_MODE!r}")


if __name__ == "__main__":
    main()
