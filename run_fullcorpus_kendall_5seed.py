"""Source pretraining on the 13-bank input inventory.

Default: seed 43. Set FC_SEEDS=42,43,44,45,46 for the paper's five seeds.
Set FC_OUTPUT_ROOT to isolate a new protocol or smoke run.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List

from run_diffpir_transfer_experiments import (
    ROOT,
    train_run,
    preserve_contract,
)


OUTPUT_ROOT = Path(os.getenv("FC_OUTPUT_ROOT", str(ROOT / "experiment_runs" / "fullcorpus_kendall_bilinear_5seed"))).resolve()

_DEFAULT_SEEDS = [43]
_env_seeds = os.getenv("FC_SEEDS", "").strip()
SEEDS: List[int] = (
    [int(s) for s in _env_seeds.split(",") if s.strip()]
    if _env_seeds
    else _DEFAULT_SEEDS
)


# Original 13-bank source input inventory.
TRAIN_SOURCES: List[Dict[str, str]] = [
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_bw_impulse_gaussian_tv_huberdata_stl10_train_s1_20260413_impulse_fix_s1_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_bw_impulse_huber_tgv_stl10_train_s1_20260413_tgv_v3_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_bw_impulse_tv_huberdata_stl10_train_s1_20260413_impulse_fix_s1_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_gaussian_huber_tgv_stl10_train_s1_20260413_tgv_v3_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_gaussian_l2_tgv_stl10_train_s1_20260413_tgv_v3_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_gaussian_l2_tv_stl10_train_s4_20260411_v2_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "WED_256",
        "ground_truth_path": "dataset_labels/foundation_gaussian_l2_tv_wed256_train_s1_20260409_tv_uniform_v1_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_gaussian_tv_huberdata_stl10_train_s4_20260411_v2_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_poisson_gaussian_nll_tgv_stl10_train_s1_20260415_131606_nll100_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_poisson_gaussian_nll_tv_stl10_train_s4_20260411_v2_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_poisson_nll_tv_stl10_train_s4_20260411_v2_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_rgb_impulse_gaussian_huber_tgv_stl10_train_s1_20260409_tgv_uniform_v1_merged.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/foundation_rgb_impulse_huber_tgv_stl10_train_s1_20260409_tgv_uniform_v1_merged.pt",
        "split": "train",
    },
]


_DEFAULTS: Dict[str, str] = {
    "SPLIT": "train",
    "VAL_RATIO": "0.05",
    "BACKBONE_PRETRAINED": "1",
    "INIT_CHECKPOINT": "",
    "HEAD_TYPE": "bilinear",
    "EPOCHS": "80",
    # BATCH_SIZE=128 on 4090 24GB: WED_256 source #7 has 256x256 images (7x larger than
    # STL10 96x96), at BS=256 this OOMs at step ~136; 128 fits with headroom.
    "BATCH_SIZE": "128",
    "WORKERS": "4",
    "USE_AMP": "1",
    "LEARNING_RATE_HEAD": "2e-4",
    "LLRD_DECAY": "0.8",
    "WARMUP_RATIO": "0.1",
    "WEIGHT_DECAY": "1e-5",
    "USE_UNCERTAINTY_LOSS": "1",
}


def _train_env(seed: int) -> Dict[str, str]:
    """Full-corpus Kendall defaults, with per-key shell-env override.

    Any key in _DEFAULTS present in os.environ wins, so smoke tests can do
    `EPOCHS=2 BATCH_SIZE=64 python run_fullcorpus_kendall_5seed.py`.
    """
    env = {key: os.environ.get(key, val) for key, val in _DEFAULTS.items()}
    env["SEED"] = str(seed)
    return env


def main() -> None:
    ckpt_dir = OUTPUT_ROOT / "checkpoints"
    runs_dir = OUTPUT_ROOT / "runs"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    manifest = {
        "run_type": "fullcorpus_kendall_bilinear_5seed",
        "seeds": SEEDS,
        "train_sources": TRAIN_SOURCES,
        "train_env_representative": _train_env(SEEDS[0]),
        "downstream_consumers": [
            "DiffPIR few-shot (§3.2)",
            "CBM3D few-shot",
        ],
    }
    preserve_contract(OUTPUT_ROOT / "manifest.json", manifest)

    for seed in SEEDS:
        run_name = f"fullcorpus_kendall_s{seed}"
        print(f"\n{'='*60}\n  SEED {seed}  |  {run_name}\n{'='*60}")
        ckpt = train_run(
            train_sources=TRAIN_SOURCES,
            run_name=run_name,
            checkpoint_dir=ckpt_dir,
            runs_dir=runs_dir,
            extra_env=_train_env(seed),
        )
        print(f"[ok] saved {ckpt}")

    print(f"\n========== MAIN RUN SUMMARY ==========")
    print(f"  seeds:  {SEEDS}")
    print(f"  ckpts:  {ckpt_dir}")
    print(f"  runs:   {runs_dir}")


if __name__ == "__main__":
    main()
