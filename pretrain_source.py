"""Pretrain HyperDn on the 13 pooled TV/TGV source label banks.

Runs seed 43 by default. PRETRAIN_SEEDS=42,43,44,45,46 runs the paper's five seeds.
PRETRAIN_OUTPUT_ROOT sets the output directory (default experiment_runs/source_pretraining).
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List

from experiment_runner import (
    ROOT,
    train_run,
    preserve_contract,
)


OUTPUT_ROOT = Path(os.getenv("PRETRAIN_OUTPUT_ROOT", str(ROOT / "experiment_runs" / "source_pretraining"))).resolve()

_DEFAULT_SEEDS = [43]
_env_seeds = os.getenv("PRETRAIN_SEEDS", "").strip()
SEEDS: List[int] = (
    [int(s) for s in _env_seeds.split(",") if s.strip()]
    if _env_seeds
    else _DEFAULT_SEEDS
)


# The 13 source label banks: TV/TGV oracle labels under L2, Huber and NLL data terms.
TRAIN_SOURCES: List[Dict[str, str]] = [
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tv_huber_bw-impulse-gaussian_stl10.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tgv_huber_bw-impulse_stl10.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tv_huber_bw-impulse_stl10.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tgv_huber_gaussian_stl10.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tgv_l2_gaussian_stl10.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tv_l2_gaussian_stl10_x4.pt",
        "split": "train",
    },
    {
        "dataset_name": "WED_256",
        "ground_truth_path": "dataset_labels/source/tv_l2_gaussian_wed256.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tv_huber_gaussian_stl10_x4.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tgv_nll_poisson-gaussian_stl10-first100.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tv_nll_poisson-gaussian_stl10_x4.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tv_nll_poisson_stl10_x4.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tgv_huber_rgb-impulse-gaussian_stl10.pt",
        "split": "train",
    },
    {
        "dataset_name": "STL10",
        "ground_truth_path": "dataset_labels/source/tgv_huber_rgb-impulse_stl10.pt",
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
    """Paper defaults for source pretraining; any key can be overridden from the shell.

    Any key in _DEFAULTS present in os.environ wins, so smoke tests can do
    `EPOCHS=2 BATCH_SIZE=64 python pretrain_source.py`.
    """
    env = {key: os.environ.get(key, val) for key, val in _DEFAULTS.items()}
    env["SEED"] = str(seed)
    return env


def main() -> None:
    ckpt_dir = OUTPUT_ROOT / "checkpoints"
    runs_dir = OUTPUT_ROOT / "runs"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    manifest = {
        "run_type": "source_pretraining",
        "seeds": SEEDS,
        "train_sources": TRAIN_SOURCES,
        "train_env_representative": _train_env(SEEDS[0]),
        "downstream_consumers": [
            "DiffPIR few-shot",
            "CBM3D few-shot",
        ],
    }
    preserve_contract(OUTPUT_ROOT / "manifest.json", manifest)

    for seed in SEEDS:
        run_name = f"source_s{seed}"
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
