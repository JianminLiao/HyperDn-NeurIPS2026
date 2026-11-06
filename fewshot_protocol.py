"""Paper few-shot schedules for WED/PSNR DiffPIR and CBM3D."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import mean, stdev, pstdev

from create_imagenet_only_foundation_checkpoint import save_imagenet_only_checkpoint
from run_diffpir_transfer_experiments import (
    ROOT, PYTHON, train_run, eval_run, extract_eval_metrics, run_with_log,
    preserve_contract, implementation_signature, source_signatures, write_csv, environment_contract,
)

SHOTS = [1, 2, 4, 16, 32, 64, 128]
METHODS = ["ours", "imagenet_only", "afkham"]
SOURCES = {
    "diffpir": {
        "train": {"dataset_name": "WED_256", "ground_truth_path": "dataset_labels/diffpir_wed256_train_psnr_iter100_v2_og/diffpir_wed256_train_psnr_iter100_v2_73shards_20260412_193726_merged.pt", "split": "train"},
        "eval": {"dataset_name": "WED_256", "ground_truth_path": "dataset_labels/diffpir_wed256_val_psnr_iter100_v2_og/diffpir_wed256_val_psnr_iter100_v2_8shards_resub2_merged.pt", "split": "val"},
    },
    "cbm3d": {
        "train": {"dataset_name": "WED_256", "ground_truth_path": "dataset_labels/cbm3d_wed256_train_unknown_og/cbm3d_wed256_train_first128_unknown_merged.pt", "split": "train"},
        "eval": {"dataset_name": "WED_256", "ground_truth_path": "dataset_labels/cbm3d_wed256_val_unknown_og/cbm3d_wed256_val_first50_unknown_merged.pt", "split": "val"},
    },
}


def parse_list(name, default):
    raw = os.getenv(name, "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else list(default)


def source_checkpoint(seed: int) -> Path:
    custom = os.getenv("SOURCE_CHECKPOINT_DIR", "").strip()
    directory = Path(custom) if custom else ROOT / "experiment_runs/fullcorpus_kendall_bilinear_5seed/checkpoints"
    trained = directory / f"fullcorpus_kendall_s{seed}_best.pth"
    if trained.exists():
        complete = directory / f"fullcorpus_kendall_s{seed}_complete.json"
        if not custom and not complete.exists():
            raise RuntimeError(f"Source pretraining has no completion marker: {complete}. Do not use a partial validation-best checkpoint.")
        return trained
    bundled = ROOT / "checkpoints" / f"hyperdn_source_s{seed}.pt"
    if not custom and bundled.exists():
        return bundled
    raise FileNotFoundError(f"No source model for seed {seed}. Train it with FC_SEEDS={seed} python run_fullcorpus_kendall_5seed.py, or set SOURCE_CHECKPOINT_DIR.")


def foundation_env(shot: int, seed: int) -> dict:
    return {
        "SPLIT": "train", "VAL_RATIO": "0", "WORKERS": "0", "USE_AMP": "0",
        "WEIGHT_DECAY": "1e-5", "MAX_TRAIN_SAMPLES": str(shot), "SEED": str(seed),
        "HEAD_TYPE": "bilinear", "USE_UNCERTAINTY_LOSS": "1", "EPOCHS": "150",
        "BATCH_SIZE": str(min(shot, 8)), "LEARNING_RATE_HEAD": "1e-4",
        "LLRD_DECAY": "0.1", "WARMUP_RATIO": "0.3", "DIFFPIR_LAMBDA_ONLY_LOSS": "0",
        "INIT_BACKBONE_CHECKPOINT": "", "RESUME_CHECKPOINT": "",
    }


def afkham_env(shot: int, seed: int) -> dict:
    return {
        "DATASET": "WED_256", "MAX_TRAIN_SAMPLES": str(shot), "SEED": str(seed),
        "EPOCHS": "500" if shot <= 2 else "100", "BATCH_SIZE": str(min(shot, 8)),
        "WORKERS": "0", "LEARNING_RATE": "1e-3", "WEIGHT_DECAY": "1e-3",
        "WARMUP_RATIO": "0", "DIFFPIR_LAMBDA_ONLY_LOSS": "0",
    }


def protocol_summary(target: str) -> dict:
    return {
        "target": target, "metric": "PSNR", "train_selection": "numpy.default_rng(seed).shuffle, prefix N",
        "evaluation_images": 50, "checkpoint_selection": "final epoch, no target validation selection",
        "foundation_epochs": 150, "foundation_amp": False,
        "afkham_epochs": {"N<=2": 500, "N>2": 100},
        "afkham_cuda_amp": True, "afkham_warmup_epochs": 1,
        "afkham_note": "Historical USE_AMP=0 was ignored; WARMUP_RATIO=0 still produced one warmup epoch.",
        "diffpir_loss": "historical two-slot label (second slot padded zero)",
        "diffpir_executor_zeta": 1.0, "diffpir_iterations": 100,
        "cbm3d_sigma_psd_clip": [0.003, 0.30],
        "seed_sd_ddof": 1 if target == "diffpir" else 0,
    }


def run_cell(target, method, shot, seed, output, only_eval=False):
    sources = SOURCES[target]
    prefix = "expc" if target == "diffpir" else "expc_cbm3d_unknown"
    run_name = f"{prefix}_{method}_n{shot}_s{seed}"
    checkpoint_dir, eval_dir = output / "checkpoints", output / "eval" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if only_eval:
        suffix = "last" if method == "afkham" else "best"
        checkpoint = checkpoint_dir / f"{run_name}_{suffix}.pth"
        if not checkpoint.exists() and method == "ours" and shot == 2 and seed == 43:
            checkpoint = ROOT / "checkpoints" / f"hyperdn_{target}_2shot_s43.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    elif method == "afkham":
        checkpoint = checkpoint_dir / f"{run_name}_last.pth"
        settings = {**afkham_env(shot, seed), "TRAIN_GT_PATH": sources["train"]["ground_truth_path"],
                    "RUN_NAME": run_name, "CHECKPOINT_DIR": str(checkpoint_dir)}
        preserve_contract(checkpoint_dir / f"{run_name}_contract.json", {
            "env": environment_contract({**os.environ, **settings}), "effective_cuda_amp": True, "effective_warmup_epochs": 1,
            "sources": source_signatures([sources["train"]]), "implementation": implementation_signature(),
        }, [checkpoint])
        if not checkpoint.exists():
            run_with_log([str(PYTHON), "-u", "train_afkham_baseline.py"],
                         env={**os.environ, **settings}, log_path=checkpoint_dir / f"{run_name}.log")
    else:
        if method == "ours":
            init = source_checkpoint(seed)
        else:
            init = checkpoint_dir / f"imagenet_only_s{seed}.pth"
            preserve_contract(checkpoint_dir / f"imagenet_only_s{seed}_contract.json", {
                "seed": seed, "initialization": "ImageNet backbone, random head", "implementation": implementation_signature(),
            }, [init])
            if not init.exists():
                save_imagenet_only_checkpoint(str(init), run_name=f"imagenet_only_s{seed}", seed=seed)
        settings = {**foundation_env(shot, seed), "INIT_CHECKPOINT": str(init),
                    # Match original initialization order; the initialized weights are then loaded.
                    "BACKBONE_PRETRAINED": "0" if method == "ours" else "1"}
        checkpoint = train_run(train_sources=[sources["train"]], run_name=run_name,
                               checkpoint_dir=checkpoint_dir, runs_dir=output / "runs", extra_env=settings)
    evaluation_settings = {"MAX_RECON_SAMPLES_PER_SOURCE": "50", "USE_AMP": "0"}
    if target == "diffpir":
        evaluation_settings.update({"DIFFPIR_ITER_NUM": "100", "DIFFPIR_FIXED_ZETA": "1.0"})
    summary = eval_run(checkpoint_path=checkpoint, eval_sources=[sources["eval"]], eval_dir=eval_dir,
                       split="val", extra_env=evaluation_settings)
    return {"method": method, "shot": shot, "seed": seed, "run_name": run_name, **extract_eval_metrics(summary)}


def write_reports(directory: Path, rows: list, target: str):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(directory / "raw_rows.csv", rows)
    aggregate = []
    deviation = stdev if target == "diffpir" else pstdev
    for method, shot in sorted({(row["method"], row["shot"]) for row in rows}):
        group = [row for row in rows if row["method"] == method and row["shot"] == shot]
        values = [row["recon_psnr_mean"] for row in group]
        aggregate.append({"method": method, "shot": shot, "seeds": [row["seed"] for row in group],
                          "n_seeds": len(group), "psnr_mean": mean(values),
                          "psnr_std": deviation(values) if len(values) > 1 else 0.0,
                          "std_ddof": 1 if target == "diffpir" else 0})
    (directory / "aggregated.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    write_csv(directory / "aggregated.csv", aggregate)


def main(target: str):
    env_prefix = "EXPC" if target == "diffpir" else "CBM3D"
    methods = parse_list(f"{env_prefix}_METHODS", METHODS)
    shots = [int(value) for value in parse_list(f"{env_prefix}_SHOTS", SHOTS)]
    seeds = [int(value) for value in parse_list(f"{env_prefix}_SEEDS", [43])]
    if not set(methods) <= set(METHODS) or not set(shots) <= set(SHOTS):
        raise ValueError(f"Use methods {METHODS} and paper budgets {SHOTS}.")
    default = "expc_fullcorpus_kendall" if target == "diffpir" else "expc_cbm3d_unknown"
    output = Path(os.getenv(f"{env_prefix}_OUTPUT_ROOT", str(ROOT / "experiment_runs" / default))).resolve()
    only_eval = os.getenv(f"{env_prefix}_ONLY_EVAL", "0").lower() in {"1", "true", "yes"}
    rank, world = int(os.getenv(f"{env_prefix}_RANK", "0")), int(os.getenv(f"{env_prefix}_WORLD_SIZE", "1"))
    reports = output / (f"reports_worker{rank:02d}" if world > 1 else "reports")
    preserve_contract(reports / "manifest.json", {
        **protocol_summary(target), "methods": methods, "shots": shots, "seeds": seeds,
        "rank": rank, "world_size": world,
        "sources": SOURCES[target], "implementation": implementation_signature(),
    }, [reports / "raw_rows.json"])
    rows = []
    grid = [(method, shot, seed) for seed in seeds for shot in shots for method in methods]
    for index, (method, shot, seed) in enumerate(grid):
        if index % world != rank:
            continue
        started = time.time()
        # Always visit the cell so its input/checkpoint contract is checked on rerun.
        row = run_cell(target, method, shot, seed, output, only_eval)
        row["elapsed_sec"] = round(time.time() - started, 1)
        rows.append(row)
        write_reports(reports, rows, target)
        print(f"[{target}] {method} N={shot} seed={seed}: {row['recon_psnr_mean']:.4f} dB", flush=True)
