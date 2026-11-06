from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch

from pgccpi_foundation_model import PGCCPiFoundationModel
from restoration.pgccpi_observation_encoding import (
    PGCCPI_DATA_TERM_ORDER,
    PGCCPI_NOISE_ATTRIBUTE_NAMES,
    PGCCPI_OBSERVATION_META_NAMES,
)
from restoration.pgccpi_param_encoding import (
    MAX_PGCCPI_PARAM_DIM,
    PGCCPI_REGULARIZER_ORDER,
)


def build_imagenet_only_payload(
    *,
    run_name: str,
    backbone_name: str,
    pretrained: bool,
    hidden_dim: int,
    task_embed_dim: int,
    observation_embed_dim: int,
    slot_embed_dim: int,
    bilinear_dim: int,
    dropout: float,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    model = PGCCPiFoundationModel(
        backbone_name=backbone_name,
        pretrained=bool(pretrained),
        hidden_dim=hidden_dim,
        task_embed_dim=task_embed_dim,
        observation_embed_dim=observation_embed_dim,
        slot_embed_dim=slot_embed_dim,
        bilinear_dim=bilinear_dim,
        max_param_dim=MAX_PGCCPI_PARAM_DIM,
        num_regularizers=len(PGCCPI_REGULARIZER_ORDER),
        num_data_terms=len(PGCCPI_DATA_TERM_ORDER),
        dropout_p=dropout,
    )
    config = {
        "BACKBONE_NAME": str(backbone_name),
        "BACKBONE_PRETRAINED": bool(pretrained),
        "HIDDEN_DIM": int(hidden_dim),
        "TASK_EMBED_DIM": int(task_embed_dim),
        "OBSERVATION_EMBED_DIM": int(observation_embed_dim),
        "SLOT_EMBED_DIM": int(slot_embed_dim),
        "BILINEAR_DIM": int(bilinear_dim),
        "DROPOUT": float(dropout),
        "METRIC": "PSNR",
        "SPLIT": "train",
        "SEED": int(seed),
    }
    return {
        "epoch": -1,
        "best_val_loss": float("inf"),
        "run_name": str(run_name),
        "config": config,
        "train_sources": [],
        "val_sources": [],
        "noise_attribute_names": [str(name) for name in PGCCPI_NOISE_ATTRIBUTE_NAMES],
        "observation_meta_names": [str(name) for name in PGCCPI_OBSERVATION_META_NAMES],
        "data_terms": [str(name) for name in PGCCPI_DATA_TERM_ORDER],
        "metrics": {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
    }


def save_imagenet_only_checkpoint(
    output_path: str,
    *,
    run_name: str,
    backbone_name: str = "convnextv2_base.fcmae_ft_in22k_in1k",
    pretrained: bool = True,
    hidden_dim: int = 512,
    task_embed_dim: int = 32,
    observation_embed_dim: int = 16,
    slot_embed_dim: int = 32,
    bilinear_dim: int = 128,
    dropout: float = 0.1,
    seed: int = 43,
) -> str:
    payload = build_imagenet_only_payload(
        run_name=run_name,
        backbone_name=backbone_name,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        task_embed_dim=task_embed_dim,
        observation_embed_dim=observation_embed_dim,
        slot_embed_dim=slot_embed_dim,
        bilinear_dim=bilinear_dim,
        dropout=dropout,
        seed=seed,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    parser.add_argument("--run-name", default="imagenet_only_foundation")
    parser.add_argument("--backbone-name", default="convnextv2_base.fcmae_ft_in22k_in1k")
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--task-embed-dim", type=int, default=32)
    parser.add_argument("--observation-embed-dim", type=int, default=16)
    parser.add_argument("--slot-embed-dim", type=int, default=32)
    parser.add_argument("--bilinear-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    path = save_imagenet_only_checkpoint(
        args.output_path,
        run_name=args.run_name,
        backbone_name=args.backbone_name,
        pretrained=bool(args.pretrained),
        hidden_dim=args.hidden_dim,
        task_embed_dim=args.task_embed_dim,
        observation_embed_dim=args.observation_embed_dim,
        slot_embed_dim=args.slot_embed_dim,
        bilinear_dim=args.bilinear_dim,
        dropout=args.dropout,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    main()
