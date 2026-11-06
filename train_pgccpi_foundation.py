from __future__ import annotations

import bisect
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
import shutil
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from pgccpi_checkpoint_utils import (
    load_backbone_checkpoint,
    load_checkpoint_payload,
    load_model_checkpoint_forgiving,
    validate_foundation_observation_contract,
)
from pgccpi_foundation_dataset import PGCCPiFoundationDataset, foundation_collate_fn
from pgccpi_foundation_model import PGCCPiFoundationModel
from restoration.pgccpi_param_encoding import PGCCPI_PARAM_ROLE_ORDER
from summary_writer import SummaryWriter
from restoration.pgccpi_observation_encoding import (
    PGCCPI_DATA_TERM_ORDER,
)
from restoration.pgccpi_param_encoding import (
    MAX_PGCCPI_PARAM_DIM,
    PGCCPI_PARAM_FAMILY_ORDER,
    PGCCPI_PARAM_SCALE_ORDER,
    PGCCPI_REGULARIZER_ORDER,
    decode_mixed_pgccpi_targets,
)


@dataclass
class SourceSpec:
    dataset_name: str
    ground_truth_path: str
    split: str | None = None


class Config:
    TRAIN_SOURCES = ""
    VAL_SOURCES = ""
    DATASET = "STL10"
    REGULARIZER_HINT = ""
    TRAIN_GT_PATH = None
    VAL_GT_PATH = None
    SPLIT = "train"
    METRIC = "PSNR"
    KERNEL_ROOT = "./data/Levin8/kernels"

    VAL_RATIO = 0.05
    MAX_TRAIN_SAMPLES = 0
    MAX_VAL_SAMPLES = 0
    TRAIN_SHUFFLE = True
    WORKERS = 8
    BATCH_SIZE = 48
    TRAIN_BATCHES_PER_EPOCH = 0
    EPOCHS = 30
    SEED = 43
    USE_AMP = True

    BACKBONE_NAME = "convnextv2_base.fcmae_ft_in22k_in1k"
    BACKBONE_PRETRAINED = True
    STAGE_INDICES = (0, 1)
    HIDDEN_DIM = 512
    TASK_EMBED_DIM = 32
    OBSERVATION_EMBED_DIM = 16
    SLOT_EMBED_DIM = 32
    BILINEAR_DIM = 128
    DROPOUT = 0.1
    LLRD_DECAY = 0.65

    LEARNING_RATE_HEAD = 1e-3
    WEIGHT_DECAY = 1e-5
    MIN_LR = 1e-6
    WARMUP_RATIO = 0.3
    GRAD_CLIP_NORM = 1.0

    LOSS_WEIGHT_PARAM = 1.0
    USE_UNCERTAINTY_LOSS = True  # Kendall et al. (2018) homoscedastic uncertainty weighting
    USE_SIMPLE_HEAD = False  # legacy alias for HEAD_TYPE="simple"
    HEAD_TYPE = "bilinear"  # "bilinear", "slot_aware", "simple"

    INIT_BACKBONE_CHECKPOINT = ""
    INIT_CHECKPOINT = ""
    RESUME_CHECKPOINT = ""
    RUN_NAME = ""
    CHECKPOINT_DIR = "checkpoints"
    RUNS_DIR = "runs"

    ENABLE_ROTATE_90_270 = True
    ENABLE_PIXEL_DITHERING = False
    PIXEL_DITHERING_PROB = 0.0
    ENABLE_RGB_PERMUTATION = False
    RGB_PERMUTATION_PROB = 0.0
    NOISE_ATTRIBUTE_DROPOUT_PROB = 0.0
    OBSERVATION_META_DROPOUT_PROB = 0.0
    JOINT_CONDITION_DROPOUT_PROB = 0.0
    FORCE_UNKNOWN_CONDITION = False


def _parse_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "y", "on")


def _parse_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else int(default)


def _parse_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else float(default)


def _parse_env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "")
    return raw if raw.strip() else str(default)


def _apply_runtime_env_overrides():
    Config.TRAIN_SOURCES = _parse_env_str("TRAIN_SOURCES", Config.TRAIN_SOURCES)
    Config.VAL_SOURCES = _parse_env_str("VAL_SOURCES", Config.VAL_SOURCES)
    Config.DATASET = _parse_env_str("DATASET", Config.DATASET)
    Config.REGULARIZER_HINT = _parse_env_str("REGULARIZER_HINT", Config.REGULARIZER_HINT)
    Config.TRAIN_GT_PATH = os.getenv("TRAIN_GT_PATH", Config.TRAIN_GT_PATH)
    Config.VAL_GT_PATH = os.getenv("VAL_GT_PATH", Config.VAL_GT_PATH)
    Config.SPLIT = _parse_env_str("SPLIT", Config.SPLIT)
    Config.METRIC = _parse_env_str("METRIC", Config.METRIC)

    Config.KERNEL_ROOT = _parse_env_str("KERNEL_ROOT", Config.KERNEL_ROOT)

    Config.VAL_RATIO = _parse_env_float("VAL_RATIO", Config.VAL_RATIO)
    Config.MAX_TRAIN_SAMPLES = _parse_env_int("MAX_TRAIN_SAMPLES", Config.MAX_TRAIN_SAMPLES)
    Config.MAX_VAL_SAMPLES = _parse_env_int("MAX_VAL_SAMPLES", Config.MAX_VAL_SAMPLES)
    Config.TRAIN_SHUFFLE = _parse_env_bool("TRAIN_SHUFFLE", Config.TRAIN_SHUFFLE)
    Config.WORKERS = _parse_env_int("WORKERS", Config.WORKERS)
    Config.BATCH_SIZE = _parse_env_int("BATCH_SIZE", Config.BATCH_SIZE)
    Config.TRAIN_BATCHES_PER_EPOCH = _parse_env_int(
        "TRAIN_BATCHES_PER_EPOCH", Config.TRAIN_BATCHES_PER_EPOCH
    )
    Config.EPOCHS = _parse_env_int("EPOCHS", Config.EPOCHS)
    Config.SEED = _parse_env_int("SEED", Config.SEED)
    Config.USE_AMP = _parse_env_bool("USE_AMP", Config.USE_AMP)

    Config.BACKBONE_NAME = _parse_env_str("BACKBONE_NAME", Config.BACKBONE_NAME)
    Config.BACKBONE_PRETRAINED = _parse_env_bool("BACKBONE_PRETRAINED", Config.BACKBONE_PRETRAINED)
    _stage_raw = os.getenv("STAGE_INDICES", "").strip()
    if _stage_raw:
        Config.STAGE_INDICES = tuple(int(v.strip()) for v in _stage_raw.split(",") if v.strip())
    Config.HIDDEN_DIM = _parse_env_int("HIDDEN_DIM", Config.HIDDEN_DIM)
    Config.TASK_EMBED_DIM = _parse_env_int("TASK_EMBED_DIM", Config.TASK_EMBED_DIM)
    Config.OBSERVATION_EMBED_DIM = _parse_env_int("OBSERVATION_EMBED_DIM", Config.OBSERVATION_EMBED_DIM)
    Config.SLOT_EMBED_DIM = _parse_env_int("SLOT_EMBED_DIM", Config.SLOT_EMBED_DIM)
    Config.BILINEAR_DIM = _parse_env_int("BILINEAR_DIM", Config.BILINEAR_DIM)
    Config.DROPOUT = _parse_env_float("DROPOUT", Config.DROPOUT)

    Config.LEARNING_RATE_HEAD = _parse_env_float("LEARNING_RATE_HEAD", Config.LEARNING_RATE_HEAD)
    Config.LLRD_DECAY = _parse_env_float("LLRD_DECAY", Config.LLRD_DECAY)
    Config.WEIGHT_DECAY = _parse_env_float("WEIGHT_DECAY", Config.WEIGHT_DECAY)
    Config.MIN_LR = _parse_env_float("MIN_LR", Config.MIN_LR)
    Config.WARMUP_RATIO = _parse_env_float("WARMUP_RATIO", Config.WARMUP_RATIO)
    Config.GRAD_CLIP_NORM = _parse_env_float("GRAD_CLIP_NORM", Config.GRAD_CLIP_NORM)

    Config.LOSS_WEIGHT_PARAM = _parse_env_float("LOSS_WEIGHT_PARAM", Config.LOSS_WEIGHT_PARAM)
    Config.USE_UNCERTAINTY_LOSS = _parse_env_bool("USE_UNCERTAINTY_LOSS", Config.USE_UNCERTAINTY_LOSS)
    Config.USE_SIMPLE_HEAD = _parse_env_bool("USE_SIMPLE_HEAD", Config.USE_SIMPLE_HEAD)
    Config.HEAD_TYPE = _parse_env_str("HEAD_TYPE", Config.HEAD_TYPE)

    Config.INIT_BACKBONE_CHECKPOINT = _parse_env_str("INIT_BACKBONE_CHECKPOINT", Config.INIT_BACKBONE_CHECKPOINT)
    Config.INIT_CHECKPOINT = _parse_env_str("INIT_CHECKPOINT", Config.INIT_CHECKPOINT)
    Config.RESUME_CHECKPOINT = _parse_env_str("RESUME_CHECKPOINT", Config.RESUME_CHECKPOINT)
    Config.RUN_NAME = _parse_env_str("RUN_NAME", Config.RUN_NAME)
    Config.CHECKPOINT_DIR = _parse_env_str("CHECKPOINT_DIR", Config.CHECKPOINT_DIR)
    Config.RUNS_DIR = _parse_env_str("RUNS_DIR", Config.RUNS_DIR)

    Config.ENABLE_ROTATE_90_270 = _parse_env_bool("ENABLE_ROTATE_90_270", Config.ENABLE_ROTATE_90_270)
    Config.ENABLE_PIXEL_DITHERING = _parse_env_bool("ENABLE_PIXEL_DITHERING", Config.ENABLE_PIXEL_DITHERING)
    Config.PIXEL_DITHERING_PROB = _parse_env_float("PIXEL_DITHERING_PROB", Config.PIXEL_DITHERING_PROB)
    Config.ENABLE_RGB_PERMUTATION = _parse_env_bool("ENABLE_RGB_PERMUTATION", Config.ENABLE_RGB_PERMUTATION)
    Config.RGB_PERMUTATION_PROB = _parse_env_float("RGB_PERMUTATION_PROB", Config.RGB_PERMUTATION_PROB)
    Config.NOISE_ATTRIBUTE_DROPOUT_PROB = _parse_env_float(
        "NOISE_ATTRIBUTE_DROPOUT_PROB",
        Config.NOISE_ATTRIBUTE_DROPOUT_PROB,
    )
    Config.OBSERVATION_META_DROPOUT_PROB = _parse_env_float(
        "OBSERVATION_META_DROPOUT_PROB",
        Config.OBSERVATION_META_DROPOUT_PROB,
    )
    Config.JOINT_CONDITION_DROPOUT_PROB = _parse_env_float(
        "JOINT_CONDITION_DROPOUT_PROB",
        Config.JOINT_CONDITION_DROPOUT_PROB,
    )
    Config.FORCE_UNKNOWN_CONDITION = _parse_env_bool(
        "FORCE_UNKNOWN_CONDITION",
        Config.FORCE_UNKNOWN_CONDITION,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sanitize_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def find_latest_pgccpi_gt(dataset: str, split: str, regularizer_hint: str = "") -> str:
    regularizer_token = f"{regularizer_hint.lower()}_" if str(regularizer_hint).strip() else "*"
    pattern = os.path.join("dataset_labels", f"pgccpi*_{regularizer_token}{dataset}_{split}_*.pt")
    candidates = glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No PG/CCPi GT matched pattern: {pattern}")
    latest = max(candidates, key=os.path.getmtime)
    print(f">> Auto-detected latest GT: {latest}")
    return latest


def _parse_source_specs(raw: str, default_dataset: str) -> List[SourceSpec]:
    specs: List[SourceSpec] = []
    for chunk in str(raw).split(";"):
        token = chunk.strip()
        if not token:
            continue
        if "|" in token:
            dataset_token, ground_truth_path = token.split("|", 1)
        else:
            dataset_token, ground_truth_path = default_dataset, token
        dataset_token = str(dataset_token).strip()
        split = None
        if "@" in dataset_token:
            dataset_name, split = dataset_token.split("@", 1)
            dataset_name = str(dataset_name).strip()
            split = str(split).strip()
        else:
            dataset_name = dataset_token
        specs.append(
            SourceSpec(
                dataset_name=str(dataset_name).strip(),
                ground_truth_path=str(ground_truth_path).strip(),
                split=(str(split).strip() if split else None),
            )
        )
    return specs


def _resolve_train_sources() -> List[SourceSpec]:
    if str(Config.TRAIN_SOURCES).strip():
        return _parse_source_specs(Config.TRAIN_SOURCES, Config.DATASET)
    if Config.TRAIN_GT_PATH:
        return [SourceSpec(dataset_name=Config.DATASET, ground_truth_path=str(Config.TRAIN_GT_PATH), split=Config.SPLIT)]
    return [
        SourceSpec(
            dataset_name=Config.DATASET,
            ground_truth_path=find_latest_pgccpi_gt(Config.DATASET, Config.SPLIT, Config.REGULARIZER_HINT),
            split=Config.SPLIT,
        )
    ]


def _resolve_val_sources(train_sources: Sequence[SourceSpec]) -> List[SourceSpec]:
    if str(Config.VAL_SOURCES).strip():
        return _parse_source_specs(Config.VAL_SOURCES, Config.DATASET)
    if Config.VAL_GT_PATH:
        return [SourceSpec(dataset_name=Config.DATASET, ground_truth_path=str(Config.VAL_GT_PATH), split=Config.SPLIT)]
    return []


def _build_dataset_from_spec(spec: SourceSpec, *, apply_augmentations: bool) -> PGCCPiFoundationDataset:
    enable_condition_dropout = bool(apply_augmentations)
    return PGCCPiFoundationDataset(
        dataset_name=spec.dataset_name,
        ground_truth_path=spec.ground_truth_path,
        metric=Config.METRIC,
        split=(spec.split or Config.SPLIT),
        kernel_root=Config.KERNEL_ROOT,
        apply_augmentations=apply_augmentations,
        enable_rotate_90_270=Config.ENABLE_ROTATE_90_270,
        enable_pixel_dithering=Config.ENABLE_PIXEL_DITHERING,
        pixel_dithering_prob=Config.PIXEL_DITHERING_PROB,
        enable_rgb_permutation=Config.ENABLE_RGB_PERMUTATION,
        rgb_permutation_prob=Config.RGB_PERMUTATION_PROB,
        noise_attribute_dropout_prob=Config.NOISE_ATTRIBUTE_DROPOUT_PROB if enable_condition_dropout else 0.0,
        observation_meta_dropout_prob=Config.OBSERVATION_META_DROPOUT_PROB if enable_condition_dropout else 0.0,
        joint_condition_dropout_prob=Config.JOINT_CONDITION_DROPOUT_PROB if enable_condition_dropout else 0.0,
        max_param_dim=MAX_PGCCPI_PARAM_DIM,
    )


def _subset_indices(length: int, limit: int, seed: int) -> List[int]:
    if limit <= 0 or limit >= length:
        return list(range(length))
    generator = np.random.default_rng(seed)
    indices = np.arange(length)
    generator.shuffle(indices)
    return indices[:limit].tolist()


def _split_indices(length: int, val_ratio: float, seed: int):
    if length <= 1:
        return [0], []
    if float(val_ratio) <= 0:
        return list(range(length)), []
    val_size = max(1, int(round(length * float(val_ratio))))
    val_size = min(val_size, length - 1)
    indices = np.arange(length)
    generator = np.random.default_rng(seed)
    generator.shuffle(indices)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return train_indices, val_indices


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weight = mask.float()
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(-1)
    diff = (pred - target).pow(2) * weight
    return diff.sum() / weight.sum().clamp_min(eps)


def _decode_padded_params(
    values: torch.Tensor,
    regularizer_ids: torch.Tensor,
    data_term_ids: torch.Tensor,
) -> torch.Tensor:
    return decode_mixed_pgccpi_targets(values, regularizer_ids, data_term_ids)



def _maybe_override_noise_attr(tensor: torch.Tensor) -> torch.Tensor:
    """If OVERRIDE_NOISE_ATTR env is set, replace the `noise_attributes` tensor
    with the specified vector (broadcast across the batch dim). Used by
    the compositional-encoding ablation (tab:ablation_composition). Accepts
    a comma-separated float vector whose length must match
    `noise_attributes.shape[-1]` (6 for PG/CCPi). Empty env = no-op.
    """
    raw = os.environ.get("OVERRIDE_NOISE_ATTR", "").strip()
    if not raw:
        return tensor
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return tensor
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"OVERRIDE_NOISE_ATTR must be comma-separated floats, got {raw!r}"
        ) from exc
    if tensor.shape[-1] != len(vals):
        raise ValueError(
            f"OVERRIDE_NOISE_ATTR length {len(vals)} != noise_attributes last "
            f"dim {tensor.shape[-1]}"
        )
    vec = torch.tensor(vals, dtype=tensor.dtype, device=tensor.device)
    return vec.expand_as(tensor).clone()


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    prepared = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            prepared[key] = value.to(device, non_blocking=True)
        else:
            prepared[key] = value
    if "noise_attributes" in prepared and torch.is_tensor(prepared["noise_attributes"]):
        prepared["noise_attributes"] = _maybe_override_noise_attr(prepared["noise_attributes"])
    if Config.FORCE_UNKNOWN_CONDITION:
        prepared["noise_attributes"] = torch.zeros_like(prepared["noise_attributes"])
        prepared["noise_attributes"][:, 0] = 1.0
        prepared["observation_meta"] = torch.zeros_like(prepared["observation_meta"])
    return prepared


def _compute_losses(model_outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], model: "PGCCPiFoundationModel"):
    pred = model_outputs["param_pred"]        # [B, D]
    target = batch["target_param"]             # [B, D]
    mask = batch["target_param_mask"]           # [B, D]

    per_slot_mse = (pred - target).pow(2)
    active = mask.sum().clamp_min(1e-8)
    raw_mse = (per_slot_mse * mask).sum() / active

    if Config.USE_UNCERTAINTY_LOSS:
        role_ids = model_outputs["role_ids"]        # [B, D]
        log_var = model.role_log_var[role_ids].clamp(-10, 10)  # [B, D]
        # Kendall et al. (2018): L = 1/(2σ²) · MSE + ½·log(σ²)
        precision = torch.exp(-log_var)
        weighted = 0.5 * precision * per_slot_mse + 0.5 * log_var
        loss = (weighted * mask).sum() / active
    else:
        loss = raw_mse

    return {
        "loss": Config.LOSS_WEIGHT_PARAM * loss,
        "param_loss": raw_mse,
    }


def _update_metric_sums(metric_sums: Dict[str, float], losses: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], model_outputs):
    batch_size = int(batch["target_param"].shape[0])
    metric_sums["loss"] += float(losses["loss"].item()) * batch_size
    metric_sums["param_loss"] += float(losses["param_loss"].item()) * batch_size

    pred_decoded = _decode_padded_params(
        model_outputs["param_pred"].detach(),
        batch["regularizer_id"],
        batch["data_term_id"],
    )
    target_decoded = _decode_padded_params(
        batch["target_param"],
        batch["regularizer_id"],
        batch["data_term_id"],
    )
    rel_err = (pred_decoded - target_decoded).abs() / target_decoded.abs().clamp_min(1e-8)
    rel_err = (rel_err * batch["target_param_mask"]).sum(dim=1) / batch["target_param_mask"].sum(dim=1).clamp_min(1e-8)
    metric_sums["param_rel_err"] += float(rel_err.mean().item()) * batch_size

    metric_sums["count"] += batch_size


def _finalize_metric_sums(metric_sums: Dict[str, float]) -> Dict[str, float]:
    count = max(1.0, float(metric_sums["count"]))
    return {
        key: (value / count)
        for key, value in metric_sums.items()
        if key != "count"
    }


def _evaluate(
    model: PGCCPiFoundationModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metric_sums = {
        "loss": 0.0,
        "param_loss": 0.0,
        "param_rel_err": 0.0,
        "count": 0.0,
    }
    with torch.no_grad():
        for batch in loader:
            batch = _prepare_batch(batch, device)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and Config.USE_AMP)):
                outputs = model(
                    batch["noisy_raw"],
                    regularizer_ids=batch["regularizer_id"],
                    data_term_ids=batch["data_term_id"],
                    noise_attributes=batch["noise_attributes"],
                    observation_meta=batch["observation_meta"],
                )
                losses = _compute_losses(outputs, batch, model)
            _update_metric_sums(metric_sums, losses, batch, outputs)
    return _finalize_metric_sums(metric_sums)


def _group_named_parameters(module: nn.Module):
    decay = []
    no_decay = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def _build_llrd_optimizer(model: PGCCPiFoundationModel) -> optim.AdamW:
    """Per-block layer-wise LR decay. LR = LEARNING_RATE_HEAD * LLRD_DECAY^level.

    Level 0:  prediction head (simple MLP or bilinear trunk + slot readout)
    Level 1:  stage_projs, condition embeddings/encoders (parallel)
    Level 2+: backbone blocks from output to input, then stem
    """
    param_groups: list = []

    def _lr(level: int) -> float:
        return Config.LEARNING_RATE_HEAD * (Config.LLRD_DECAY ** level)

    def _add(module: nn.Module, level: int):
        decay_p, no_decay_p = _group_named_parameters(module)
        if decay_p:
            param_groups.append({"params": decay_p, "lr": _lr(level), "weight_decay": Config.WEIGHT_DECAY})
        if no_decay_p:
            param_groups.append({"params": no_decay_p, "lr": _lr(level), "weight_decay": 0.0})

    # Level 0 — prediction head
    if model.head_type == "simple":
        _add(model.param_head, 0)
    elif model.head_type == "slot_aware":
        for mod in [model.param_head, model.role_embedding, model.scale_embedding, model.slot_affine]:
            _add(mod, 0)
    else:  # bilinear
        for mod in [
            model.trunk,
            model.role_embedding,
            model.scale_embedding,
            model.q_proj,
            model.slot_proj,
            model.h_out,
            model.slot_out,
            model.family_gate,
            model.family_bias,
        ]:
            _add(mod, 0)
    # role_log_var is a bare nn.Parameter (not a sub-module), add to level 0 with no weight decay
    param_groups.append({"params": [model.role_log_var], "lr": _lr(0), "weight_decay": 0.0})

    # Level 1 — parallel branches
    for mod in [
        model.stage_projs,
        model.regularizer_embedding,
        model.data_term_embedding,
        model.noise_attribute_encoder,
        model.observation_meta_encoder,
    ]:
        _add(mod, 1)

    # Level 2+ — backbone blocks from output to input, then stem
    backbone = model.backbone

    def _get_bb(attr: str, idx: int | None = None):
        flat = f"{attr}_{idx}" if idx is not None else attr
        if idx is not None and hasattr(backbone, flat):
            return getattr(backbone, flat)
        if hasattr(backbone, attr):
            obj = getattr(backbone, attr)
            return obj[idx] if idx is not None else obj
        if hasattr(backbone, "model"):
            inner = backbone.model
            obj = getattr(inner, attr)
            return obj[idx] if idx is not None else obj
        raise AttributeError(f"Cannot locate backbone.{attr}[{idx}]")

    level = 2
    for stage_idx in reversed(model.stage_indices):
        stage = _get_bb("stages", stage_idx)
        blocks = list(stage.blocks) if hasattr(stage, "blocks") else list(stage.children())
        for block in reversed(blocks):
            _add(block, level)
            level += 1
        if stage_idx > 0:
            try:
                _add(_get_bb("downsample_layers", stage_idx), level)
                level += 1
            except AttributeError:
                pass

    for stem_candidate in ["stem", ("downsample_layers", 0)]:
        try:
            if isinstance(stem_candidate, tuple):
                _add(_get_bb(*stem_candidate), level)
            else:
                _add(_get_bb(stem_candidate), level)
            break
        except AttributeError:
            continue

    assigned_ids = {id(p) for g in param_groups for p in g["params"]}
    leftover_decay, leftover_no_decay = [], []
    for name, param in backbone.named_parameters():
        if id(param) not in assigned_ids and param.requires_grad:
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                leftover_no_decay.append(param)
            else:
                leftover_decay.append(param)
    if leftover_decay:
        param_groups.append({"params": leftover_decay, "lr": _lr(level), "weight_decay": Config.WEIGHT_DECAY})
    if leftover_no_decay:
        param_groups.append({"params": leftover_no_decay, "lr": _lr(level), "weight_decay": 0.0})
    if leftover_decay or leftover_no_decay:
        n = len(leftover_decay) + len(leftover_no_decay)
        print(
            f"[LLRD] {n} backbone params unmatched by block traversal assigned to level {level} "
            f"(lr={_lr(level):.2e})",
            flush=True,
        )

    return optim.AdamW(param_groups)


def _build_run_name(train_sources: Sequence[SourceSpec]) -> str:
    if str(Config.RUN_NAME).strip():
        return _sanitize_tag(Config.RUN_NAME)
    dataset_token = _sanitize_tag("-".join(sorted({spec.dataset_name for spec in train_sources})))
    return _sanitize_tag(
        f"pgccpi_foundation_{dataset_token}_{Config.METRIC}_{Config.BACKBONE_NAME}_{int(time.time())}"
    )


def _get_git_commit() -> str:
    """Return short SHA of current HEAD; empty string on failure."""
    import subprocess as _sp
    try:
        out = _sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)) or None,
            stderr=_sp.DEVNULL,
            timeout=2,
        )
        return out.decode("ascii", errors="replace").strip()
    except Exception:
        return ""


def _save_checkpoint(
    *,
    path: str,
    model: PGCCPiFoundationModel,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    run_name: str,
    train_sources: Sequence[SourceSpec],
    val_sources: Sequence[SourceSpec],
    noise_attribute_names: Sequence[str],
    observation_meta_names: Sequence[str],
    data_terms: Sequence[str],
    metrics: Dict[str, float],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "run_name": run_name,
        "config": {key: getattr(Config, key) for key in dir(Config) if key.isupper()},
        "train_sources": [spec.__dict__ for spec in train_sources],
        "val_sources": [spec.__dict__ for spec in val_sources],
        "noise_attribute_names": [str(name) for name in noise_attribute_names],
        "observation_meta_names": [str(name) for name in observation_meta_names],
        "data_terms": [str(name) for name in data_terms],
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "git_commit": _get_git_commit(),
        "encoding_contract": {
            "regularizer_order": list(PGCCPI_REGULARIZER_ORDER),
            "data_term_order": list(PGCCPI_DATA_TERM_ORDER),
            "param_role_order": list(PGCCPI_PARAM_ROLE_ORDER),
            "param_scale_order": list(PGCCPI_PARAM_SCALE_ORDER),
            "param_family_order": list(PGCCPI_PARAM_FAMILY_ORDER),
        },
    }
    # Retry on Windows OneDrive file locks — broadened to any "cannot be opened"
    # / sharing / PermissionError pattern.
    import time as _time
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            torch.save(payload, path)
            return
        except (OSError, RuntimeError, PermissionError) as e:
            last_err = e
            msg = str(e).lower()
            if any(s in msg for s in ("error code: 32", "sharing", "cannot be opened",
                                       "permission denied", "being used by another")):
                _time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"torch.save failed after 5 retries: {last_err}")


def _propagate_epoch_to_datasets(dataset, epoch: int) -> None:
    """Recursively call ``set_epoch`` on any wrapped dataset that exposes one.

    Handles arbitrary nesting of ``Subset`` and ``ConcatDataset`` so that the
    current training epoch reaches the underlying ``PGCCPiFoundationDataset``
    / ``_BasePGCCPIDataset`` instances, where it is mixed into the
    augmentation seed so the augmentation plan varies from epoch to epoch.
    """
    if dataset is None:
        return
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        try:
            setter(int(epoch))
        except TypeError:
            # Some samplers take no epoch kwarg; fall through to recursion.
            pass
    if isinstance(dataset, Subset):
        _propagate_epoch_to_datasets(dataset.dataset, epoch)
    elif isinstance(dataset, ConcatDataset):
        for sub in dataset.datasets:
            _propagate_epoch_to_datasets(sub, epoch)


def _get_source_ids(dataset) -> List[int]:
    """For each effective index i in `dataset`, return which sub-dataset it came from.

    Handles arbitrary nesting of Subset and ConcatDataset.
    """
    indices: Optional[List[int]] = None
    base = dataset
    while isinstance(base, Subset):
        cur = list(base.indices)
        indices = cur if indices is None else [cur[i] for i in indices]
        base = base.dataset
    if isinstance(base, ConcatDataset):
        cum = base.cumulative_sizes  # e.g. [5000, 6000] for sizes [5000, 1000]
        global_indices = indices if indices is not None else list(range(len(dataset)))
        return [bisect.bisect_right(cum, gi) for gi in global_indices]
    return [0] * len(dataset)


class DatasetHomogeneousBatchSampler(torch.utils.data.Sampler):
    """Yields batches where every index in a batch comes from the same source dataset.

    This allows joint training on datasets with different spatial dimensions
    (e.g. STL10 96×96 and WED_256 256×256) without cropping, because
    ``foundation_collate_fn`` stacks tensors with ``torch.stack`` which
    requires all tensors in a batch to have identical shape.
    """

    def __init__(
        self,
        source_ids: List[int],
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        batches_per_epoch: int = 0,
    ):
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batches_per_epoch = int(batches_per_epoch)
        self._epoch = 0
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, s in enumerate(source_ids):
            groups[s].append(i)
        self.groups: List[List[int]] = [idxs for _, idxs in sorted(groups.items())]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        all_batches: List[List[int]] = []
        for group in self.groups:
            if not group:
                continue
            if self.shuffle:
                perm = rng.permutation(len(group)).tolist()
                ordered = [group[i] for i in perm]
            else:
                ordered = list(group)
            # Drop partial batches within each source.
            for start in range(0, len(ordered) - self.batch_size + 1, self.batch_size):
                all_batches.append(ordered[start : start + self.batch_size])
        if self.shuffle:
            order = rng.permutation(len(all_batches)).tolist()
            all_batches = [all_batches[i] for i in order]
        if self.batches_per_epoch > 0:
            if self.batches_per_epoch > len(all_batches):
                extra = rng.integers(0, len(all_batches), size=self.batches_per_epoch - len(all_batches))
                all_batches.extend([list(all_batches[i]) for i in extra])
            else:
                all_batches = all_batches[: self.batches_per_epoch]
            if self.shuffle:
                order = rng.permutation(len(all_batches)).tolist()
                all_batches = [all_batches[i] for i in order]
        yield from all_batches

    def __len__(self) -> int:
        natural = sum(len(g) // self.batch_size for g in self.groups)
        return self.batches_per_epoch if self.batches_per_epoch > 0 else natural


def main():
    _apply_runtime_env_overrides()
    set_seed(Config.SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_sources = _resolve_train_sources()
    val_sources = _resolve_val_sources(train_sources)
    run_name = _build_run_name(train_sources)

    train_datasets = [_build_dataset_from_spec(spec, apply_augmentations=True) for spec in train_sources]
    val_datasets = [_build_dataset_from_spec(spec, apply_augmentations=False) for spec in val_sources]
    reference_dataset = train_datasets[0]
    noise_attribute_names = list(reference_dataset.noise_attribute_names)
    observation_meta_names = list(reference_dataset.observation_meta_names)
    data_terms = sorted(
        {
            str(name)
            for dataset in train_datasets
            for name in getattr(dataset, "data_term_names", [dataset.data_term])
        }
    )

    val_dataset: Optional[Dataset] = None
    if val_sources:
        # Explicit val set (VAL_SOURCES / VAL_GT_PATH)
        train_dataset: Dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        _val_raw: Dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        if Config.MAX_VAL_SAMPLES > 0:
            _val_raw = Subset(_val_raw, _subset_indices(len(_val_raw), Config.MAX_VAL_SAMPLES, Config.SEED + 1))
        val_dataset = _val_raw
    elif Config.VAL_RATIO > 0:
        # Split training data into train/val via VAL_RATIO (used for pretraining)
        _train_base: Dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        train_indices, val_indices = _split_indices(len(_train_base), Config.VAL_RATIO, Config.SEED)
        train_dataset = Subset(_train_base, train_indices)
        _val_split: Dataset = Subset(_train_base, val_indices)
        if Config.MAX_VAL_SAMPLES > 0:
            _val_split = Subset(_val_split, _subset_indices(len(_val_split), Config.MAX_VAL_SAMPLES, Config.SEED + 1))
        val_dataset = _val_split
    else:
        # No val — training only (e.g. few-shot DiffPIR with VAL_RATIO=0)
        train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    if Config.MAX_TRAIN_SAMPLES > 0:
        train_dataset = Subset(train_dataset, _subset_indices(len(train_dataset), Config.MAX_TRAIN_SAMPLES, Config.SEED))

    _dl_kwargs = dict(
        num_workers=Config.WORKERS,
        pin_memory=True,
        persistent_workers=(Config.WORKERS > 0),
        collate_fn=foundation_collate_fn,
    )
    if len(train_datasets) > 1:
        # Multi-source: use homogeneous sampler so each batch has same image size.
        train_src_ids = _get_source_ids(train_dataset)
        train_batch_sampler: Optional[DatasetHomogeneousBatchSampler] = DatasetHomogeneousBatchSampler(
            train_src_ids,
            batch_size=Config.BATCH_SIZE,
            seed=Config.SEED,
            shuffle=Config.TRAIN_SHUFFLE,
            batches_per_epoch=Config.TRAIN_BATCHES_PER_EPOCH,
        )
        if len(train_batch_sampler) <= 0:
            raise ValueError(
                f"Multi-source train batch_size={Config.BATCH_SIZE} is too large for grouped source sizes "
                f"{[len(g) for g in train_batch_sampler.groups]}"
            )
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **_dl_kwargs)
        if val_dataset is not None:
            val_src_ids = _get_source_ids(val_dataset)
            val_batch_sampler: Optional[DatasetHomogeneousBatchSampler] = DatasetHomogeneousBatchSampler(
                val_src_ids, batch_size=Config.BATCH_SIZE, seed=Config.SEED, shuffle=False
            )
            if len(val_batch_sampler) > 0:
                val_loader: Optional[DataLoader] = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **_dl_kwargs)
            else:
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **_dl_kwargs)
                print(
                    f"Val fallback: grouped val source sizes {[len(g) for g in val_batch_sampler.groups]} "
                    f"cannot form homogeneous batches at batch_size={Config.BATCH_SIZE}; using batch_size=1.",
                    flush=True,
                )
        else:
            val_loader = None
        print(
            f"Using DatasetHomogeneousBatchSampler: "
            f"{len(train_batch_sampler.groups)} source(s), "
            f"sizes={[len(g) for g in train_batch_sampler.groups]}, "
            f"batches_per_epoch={len(train_batch_sampler)}",
            flush=True,
        )
    else:
        train_batch_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=Config.TRAIN_SHUFFLE, **_dl_kwargs
        )
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, **_dl_kwargs) if val_dataset is not None else None

    model = PGCCPiFoundationModel(
        backbone_name=Config.BACKBONE_NAME,
        pretrained=Config.BACKBONE_PRETRAINED,
        stage_indices=Config.STAGE_INDICES,
        hidden_dim=Config.HIDDEN_DIM,
        task_embed_dim=Config.TASK_EMBED_DIM,
        observation_embed_dim=Config.OBSERVATION_EMBED_DIM,
        slot_embed_dim=Config.SLOT_EMBED_DIM,
        bilinear_dim=Config.BILINEAR_DIM,
        max_param_dim=MAX_PGCCPI_PARAM_DIM,
        num_regularizers=len(PGCCPI_REGULARIZER_ORDER),
        num_data_terms=len(PGCCPI_DATA_TERM_ORDER),
        head_type=Config.HEAD_TYPE,
        use_simple_head=Config.USE_SIMPLE_HEAD,
        dropout_p=Config.DROPOUT,
    ).to(device)

    if str(Config.INIT_BACKBONE_CHECKPOINT).strip():
        backbone_load = load_backbone_checkpoint(model.backbone, Config.INIT_BACKBONE_CHECKPOINT)
        print(
            "Loaded INIT_BACKBONE_CHECKPOINT="
            f"{Config.INIT_BACKBONE_CHECKPOINT} "
            f"(missing={len(backbone_load['missing_keys'])}, unexpected={len(backbone_load['unexpected_keys'])})"
        )

    if str(Config.INIT_CHECKPOINT).strip():
        init_state = load_checkpoint_payload(Config.INIT_CHECKPOINT)
        validate_foundation_observation_contract(init_state, path=Config.INIT_CHECKPOINT)
        load_result = load_model_checkpoint_forgiving(model, init_state)
        print(
            "Loaded INIT_CHECKPOINT="
            f"{Config.INIT_CHECKPOINT} "
            f"(missing={len(load_result['missing_keys'])}, "
            f"unexpected={len(load_result['unexpected_keys'])}, "
            f"skipped_shape={len(load_result['skipped_shape_mismatch'])})"
        )

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    def selected_indices(dataset):
        indices = list(range(len(dataset)))
        while isinstance(dataset, Subset):
            indices = [int(dataset.indices[index]) for index in indices]
            dataset = dataset.dataset
        return indices
    selection = {"seed": Config.SEED, "train_indices": selected_indices(train_dataset),
                 "validation_indices": selected_indices(val_dataset) if val_dataset is not None else [],
                 "train_sources": [vars(source) for source in train_sources],
                 "checkpoint_selection": "minimum_validation_loss" if val_dataset is not None else "final_epoch"}
    with open(os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_selection.json"), "w") as handle:
        json.dump(selection, handle, indent=2)

    optimizer = _build_llrd_optimizer(model)
    warmup_epochs = max(1, int(round(Config.EPOCHS * Config.WARMUP_RATIO)))
    cosine_epochs = max(1, Config.EPOCHS - warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=Config.MIN_LR,
    )
    if warmup_epochs > 0 and Config.WARMUP_RATIO > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and Config.USE_AMP))

    start_epoch = 0
    best_val_loss = float("inf")
    if str(Config.RESUME_CHECKPOINT).strip():
        resume_state = load_checkpoint_payload(Config.RESUME_CHECKPOINT)
        validate_foundation_observation_contract(resume_state, path=Config.RESUME_CHECKPOINT)
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if scheduler is not None and resume_state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_val_loss = float(resume_state.get("best_val_loss", best_val_loss))
        print(f"Resumed from {Config.RESUME_CHECKPOINT} at epoch {start_epoch}")

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.RUNS_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(Config.RUNS_DIR, run_name))

    print(f"Using device: {device}")
    print(f"Run name: {run_name}")
    print(f"Train sources: {[spec.__dict__ for spec in train_sources]}")
    print(f"Val sources: {[spec.__dict__ for spec in val_sources]}")
    print(
        "Observation context: "
        f"data_terms={data_terms} noise={noise_attribute_names} meta={observation_meta_names}"
    )
    val_size_str = str(len(val_dataset)) if val_dataset is not None else "disabled"
    print(f"Train size: {len(train_dataset)} | Val size: {val_size_str}", flush=True)

    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_best.pth")
    last_model_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_last.pth")

    for epoch in range(start_epoch, Config.EPOCHS):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        # Propagate epoch into the underlying datasets so that the
        # augmentation seed varies across epochs (val/eval datasets are
        # unaffected because their _build_sample uses epoch=0 when
        # apply_augmentations=False).
        _propagate_epoch_to_datasets(train_dataset, epoch)
        if val_dataset is not None:
            _propagate_epoch_to_datasets(val_dataset, epoch)
        model.train()
        metric_sums = {
            "loss": 0.0,
            "param_loss": 0.0,
            "param_rel_err": 0.0,
            "count": 0.0,
        }

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}", leave=False)
        for batch in progress:
            batch = _prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and Config.USE_AMP)):
                outputs = model(
                    batch["noisy_raw"],
                    regularizer_ids=batch["regularizer_id"],
                    data_term_ids=batch["data_term_id"],
                    noise_attributes=batch["noise_attributes"],
                    observation_meta=batch["observation_meta"],
                )
                losses = _compute_losses(outputs, batch, model)

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            _update_metric_sums(metric_sums, losses, batch, outputs)
            train_metrics_live = _finalize_metric_sums(metric_sums)
            progress.set_postfix(
                loss=f"{train_metrics_live['loss']:.4f}",
                param=f"{train_metrics_live['param_rel_err']:.4f}",
            )

        scheduler.step()
        train_metrics = _finalize_metric_sums(metric_sums)

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        writer.add_scalar("train/lr_level0", optimizer.param_groups[-1]["lr"], epoch)
        writer.add_scalar("train/lr_stem", optimizer.param_groups[0]["lr"], epoch)
        for role_idx, role_name in enumerate(PGCCPI_PARAM_ROLE_ORDER):
            if role_name != "pad":
                sigma2 = torch.exp(model.role_log_var[role_idx]).item()
                writer.add_scalar(f"uncertainty/{role_name}_sigma2", sigma2, epoch)

        summary: Dict[str, Any] = {"epoch": epoch, "train": train_metrics}
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, device)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            summary["val"] = val_metrics
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                _save_checkpoint(
                    path=best_model_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    run_name=run_name,
                    train_sources=train_sources,
                    val_sources=val_sources,
                    noise_attribute_names=noise_attribute_names,
                    observation_meta_names=observation_meta_names,
                    data_terms=data_terms,
                    metrics=summary,
                )
                print(f"Saved new best model to {best_model_path}", flush=True)

        print(json.dumps(summary, indent=2), flush=True)

        _save_checkpoint(
            path=last_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            run_name=run_name,
            train_sources=train_sources,
            val_sources=val_sources,
            noise_attribute_names=noise_attribute_names,
            observation_meta_names=observation_meta_names,
            data_terms=data_terms,
            metrics=summary,
        )

    writer.close()
    if not os.path.exists(best_model_path):
        shutil.copy2(last_model_path, best_model_path)
        print(f"No val split: copied last checkpoint to {best_model_path}", flush=True)
    print(f"Training complete. Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
