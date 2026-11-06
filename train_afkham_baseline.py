"""Train the per-configuration Afkham CNN on a specified oracle bank.

Use run_expc_fullcorpus_kendall.py or run_expc_cbm3d_unknown.py for the
public few-shot experiments. Direct use requires TRAIN_GT_PATH.
"""
from __future__ import annotations

import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from afkham_baseline import AfkhamCNN
from dataset import STL10PGCCPIDataset, WEDPGCCPIDataset


class Config:
    DATASET = "WED_256"
    TRAIN_GT_PATH = ""
    VAL_RATIO = 0.05
    MAX_TRAIN_SAMPLES = 0
    SEED = 43
    EPOCHS = 60
    BATCH_SIZE = 64
    WORKERS = 4
    USE_AMP = True
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    MIN_LR = 1e-6
    WARMUP_RATIO = 0.2
    EARLY_STOPPING_PATIENCE = 10
    RUN_NAME = ""
    CHECKPOINT_DIR = "checkpoints"
    DIFFPIR_LAMBDA_ONLY_LOSS = False


def _parse_env(name, default, type_fn):
    raw = os.getenv(name, "").strip()
    return type_fn(raw) if raw else default


def _apply_env():
    Config.DATASET = _parse_env("DATASET", Config.DATASET, str)
    Config.TRAIN_GT_PATH = os.getenv("TRAIN_GT_PATH", Config.TRAIN_GT_PATH).strip()
    if not Config.TRAIN_GT_PATH:
        raise ValueError("Set TRAIN_GT_PATH, or use run_expc_fullcorpus_kendall.py / run_expc_cbm3d_unknown.py.")
    Config.MAX_TRAIN_SAMPLES = _parse_env("MAX_TRAIN_SAMPLES", Config.MAX_TRAIN_SAMPLES, int)
    Config.SEED = _parse_env("SEED", Config.SEED, int)
    Config.EPOCHS = _parse_env("EPOCHS", Config.EPOCHS, int)
    Config.BATCH_SIZE = _parse_env("BATCH_SIZE", Config.BATCH_SIZE, int)
    Config.WORKERS = _parse_env("WORKERS", Config.WORKERS, int)
    Config.LEARNING_RATE = _parse_env("LEARNING_RATE", Config.LEARNING_RATE, float)
    Config.WEIGHT_DECAY = _parse_env("WEIGHT_DECAY", Config.WEIGHT_DECAY, float)
    Config.WARMUP_RATIO = _parse_env("WARMUP_RATIO", Config.WARMUP_RATIO, float)
    Config.CHECKPOINT_DIR = _parse_env("CHECKPOINT_DIR", Config.CHECKPOINT_DIR, str)
    Config.RUN_NAME = _parse_env("RUN_NAME", Config.RUN_NAME, str)
    # Historical runs always enabled CUDA AMP; USE_AMP was not parsed.
    # Keep the original effective protocol rather than changing those runs.


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataset(ground_truth_path: str, dataset_name: str, apply_augmentations: bool):
    kwargs = dict(ground_truth_path=ground_truth_path, metric="PSNR", split="train",
                  apply_augmentations=apply_augmentations, resample_noise_realization=False)
    if dataset_name == "STL10":
        return STL10PGCCPIDataset(stl10_root="./data", **kwargs)
    if dataset_name == "WED_256":
        return WEDPGCCPIDataset(wed_root="./data/WED_256", **kwargs)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main():
    _apply_env()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = _build_dataset(Config.TRAIN_GT_PATH, Config.DATASET, apply_augmentations=True)
    target_dim = dataset.target_dim

    # No val split for few-shot: all N samples used for training.
    # Evaluation done on separate test set after training.
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(Config.SEED)
    rng.shuffle(indices)
    train_indices = indices.tolist()

    if Config.MAX_TRAIN_SAMPLES > 0:
        train_indices = train_indices[:Config.MAX_TRAIN_SAMPLES]

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=Config.WORKERS, pin_memory=True, drop_last=False)

    model = AfkhamCNN(in_channels=3, out_dim=target_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"AfkhamCNN: out_dim={target_dim}, params={total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    warmup_epochs = max(1, int(round(Config.EPOCHS * Config.WARMUP_RATIO)))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, Config.EPOCHS - warmup_epochs), eta_min=Config.MIN_LR)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and Config.USE_AMP))
    criterion = nn.MSELoss()

    run_name = Config.RUN_NAME or f"afkham_{Config.DATASET}_n{Config.MAX_TRAIN_SAMPLES or 'full'}_{int(time.time())}"
    save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_last.pth")
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    import json
    with open(os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_selection.json"), "w") as f:
        json.dump({"seed": Config.SEED, "train_indices": train_indices,
                   "validation_indices": [], "checkpoint_selection": "final_epoch",
                   "effective_cuda_amp": True, "effective_warmup_epochs": warmup_epochs}, f, indent=2)
    loss_slots = "lambda only" if Config.DIFFPIR_LAMBDA_ONLY_LOSS else "all target slots"
    print(f"Run: {run_name} | train={len(train_indices)} | fixed {Config.EPOCHS} epochs (no val) | loss={loss_slots}")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False):
            if len(batch) == 6:
                _, targets_tensor, _, _, _, noisy_raw = batch
            else:
                _, targets_tensor, _, _, noisy_raw = batch
            inputs = noisy_raw.to(device, non_blocking=True)
            targets_tensor = targets_tensor.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and Config.USE_AMP)):
                preds = model(inputs)
                loss = criterion(preds, targets_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = inputs.size(0)
            train_loss += loss.item() * bs
            train_count += bs

        scheduler.step()
        train_loss /= max(train_count, 1)
        if (epoch + 1) % 10 == 0 or epoch == Config.EPOCHS - 1:
            print(f"  Epoch {epoch+1}/{Config.EPOCHS} | train_loss={train_loss:.6f}")

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": Config.EPOCHS - 1,
                "config": {k: getattr(Config, k) for k in dir(Config) if k.isupper()},
                "target_dim": target_dim, "run_name": run_name}, save_path)
    print(f"Done. Saved: {save_path}")


if __name__ == "__main__":
    main()
