from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import (
    STL10PGCCPIDataset,
    WEDPGCCPIDataset,
    Kodak24PGCCPIDataset,
    CBSD68PGCCPIDataset,
    Urban100PGCCPIDataset,
)
from restoration.pgccpi_observation_encoding import (
    PGCCPI_NOISE_ATTRIBUTE_NAMES,
    PGCCPI_OBSERVATION_META_NAMES,
    encode_pgccpi_noise_attributes,
    encode_pgccpi_observation_metadata,
    normalize_pgccpi_data_term_name,
    normalize_pgccpi_noise_model_name,
    normalize_pgccpi_operator_name,
    pgccpi_data_term_id,
    pgccpi_operator_id,
)
from restoration.pgccpi_param_encoding import (
    MAX_PGCCPI_PARAM_DIM,
    decode_pgccpi_target_tensor,
    pad_pgccpi_target_vector,
    pgccpi_regularizer_id,
    normalize_pgccpi_regularizer_name,
)


def _load_payload(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _observation_meta_applicable(noise_model_name: str) -> bool:
    # Foundation observation metadata currently encodes the Poisson-Gaussian
    # case only. Pure Poisson can be added later as a separate contract change.
    return normalize_pgccpi_noise_model_name(noise_model_name) == "POISSON_GAUSSIAN"


def _build_foundation_observation_meta(
    base_record: Dict[str, Any],
    *,
    applicable: bool,
    observed: bool,
) -> np.ndarray:
    return encode_pgccpi_observation_metadata(
        alpha=float(base_record.get("alpha", 0.0)),
        sigma_read=float(base_record.get("sigma_read", 0.0)),
        background=float(base_record.get("background", 0.0)),
        applicable=bool(applicable),
        observed=bool(observed),
    )


def _unknown_noise_attributes() -> np.ndarray:
    return encode_pgccpi_noise_attributes("UNKNOWN")


def _build_base_dataset(
    dataset_name: str,
    *,
    ground_truth_path: str,
    metric: str,
    split: str,
    kernel_root: Optional[str],
    apply_augmentations: bool,
    enable_rotate_90_270: bool,
    enable_pixel_dithering: bool,
    pixel_dithering_prob: float,
    enable_rgb_permutation: bool,
    rgb_permutation_prob: float,
    include_kernel_indices=None,
    exclude_kernel_indices=None,
):
    kwargs = dict(
        ground_truth_path=ground_truth_path,
        metric=metric,
        split=split,
        kernel_root=kernel_root,
        apply_augmentations=apply_augmentations,
        enable_rotate_90_270=enable_rotate_90_270,
        enable_pixel_dithering=enable_pixel_dithering,
        pixel_dithering_prob=pixel_dithering_prob,
        enable_rgb_permutation=enable_rgb_permutation,
        rgb_permutation_prob=rgb_permutation_prob,
        include_kernel_indices=include_kernel_indices,
        exclude_kernel_indices=exclude_kernel_indices,
    )
    if dataset_name == "STL10":
        return STL10PGCCPIDataset(stl10_root="./data", **kwargs)
    if dataset_name == "WED_crop":
        return WEDPGCCPIDataset(wed_root="./data/WED_crop", **kwargs)
    if dataset_name == "WED_256":
        return WEDPGCCPIDataset(wed_root="./data/WED_256", **kwargs)
    if dataset_name == "Kodak24":
        return Kodak24PGCCPIDataset(kodak24_root="./data/Kodak24", **kwargs)
    if dataset_name == "CBSD68":
        return CBSD68PGCCPIDataset(cbsd68_root="./data/CBSD68", **kwargs)
    if dataset_name == "Urban100":
        return Urban100PGCCPIDataset(urban100_root="./data/Urban100", **kwargs)
    # 256-crop cross-dataset benchmarks; WED-like folder layout (val/ subdir).
    if dataset_name == "CBSD68_256":
        return WEDPGCCPIDataset(wed_root="./data/CBSD68_256", **kwargs)
    if dataset_name == "URBAN100_256":
        return WEDPGCCPIDataset(wed_root="./data/URBAN100_256", **kwargs)
    if dataset_name == "FFHQ_256":
        return WEDPGCCPIDataset(wed_root="./data/FFHQ_256", **kwargs)
    raise ValueError(f"Unsupported foundation PG/CCPi dataset: {dataset_name}")


class PGCCPiFoundationDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_name: str,
        ground_truth_path: str,
        metric_name: str = "PSNR",
        metric: str = "",
        split: str = "train",
        kernel_root: Optional[str] = None,
        apply_augmentations: bool = False,
        enable_rotate_90_270: bool = False,
        enable_pixel_dithering: bool = False,
        pixel_dithering_prob: float = 0.0,
        enable_rgb_permutation: bool = False,
        rgb_permutation_prob: float = 0.0,
        noise_attribute_dropout_prob: float = 0.0,
        observation_meta_dropout_prob: float = 0.0,
        joint_condition_dropout_prob: float = 0.0,
        include_kernel_indices=None,
        exclude_kernel_indices=None,
        max_param_dim: int = MAX_PGCCPI_PARAM_DIM,
        # Legacy kwargs (ignored)
        curve_pairs_override: Optional[str] = None,
        curve_metric: str = "",
        require_curves: bool = False,
    ):
        if not ground_truth_path or not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

        resolved_metric = str(metric_name or metric or "PSNR").upper()

        self.base_dataset = _build_base_dataset(
            dataset_name,
            ground_truth_path=ground_truth_path,
            metric=resolved_metric,
            split=split,
            kernel_root=kernel_root,
            apply_augmentations=apply_augmentations,
            enable_rotate_90_270=enable_rotate_90_270,
            enable_pixel_dithering=enable_pixel_dithering,
            pixel_dithering_prob=pixel_dithering_prob,
            enable_rgb_permutation=enable_rgb_permutation,
            rgb_permutation_prob=rgb_permutation_prob,
            include_kernel_indices=include_kernel_indices,
            exclude_kernel_indices=exclude_kernel_indices,
        )
        self.dataset_name = str(dataset_name)
        self.ground_truth_path = str(ground_truth_path)
        self.metric = resolved_metric
        self.max_param_dim = int(max_param_dim)
        self.noise_attribute_dropout_prob = float(max(0.0, noise_attribute_dropout_prob))
        self.observation_meta_dropout_prob = float(max(0.0, observation_meta_dropout_prob))
        self.joint_condition_dropout_prob = float(max(0.0, joint_condition_dropout_prob))
        self.regularizer_names = list(getattr(self.base_dataset, "record_regularizers", [str(self.base_dataset.regularizer).upper()]))
        self.data_term_names = list(getattr(self.base_dataset, "record_data_terms", [normalize_pgccpi_data_term_name(getattr(self.base_dataset, "data_term", "NLL"))]))
        resolved_noise_models = list(
            getattr(
                self.base_dataset,
                "record_noise_models",
                [normalize_pgccpi_noise_model_name(getattr(self.base_dataset, "noise_model", "UNKNOWN"))],
            )
        )
        self.operator_kind_names = list(getattr(self.base_dataset, "record_operator_kinds", [normalize_pgccpi_operator_name(getattr(self.base_dataset, "operator_kind", "KNOWN_PSF_DEBLUR"))]))
        self.regularizer = str(self.regularizer_names[0]).upper()
        self.regularizer_id = int(pgccpi_regularizer_id(self.regularizer)) if len(self.regularizer_names) == 1 else -1
        self.data_term = str(self.data_term_names[0]).upper()
        self.data_term_id = int(pgccpi_data_term_id(self.data_term)) if len(self.data_term_names) == 1 else -1
        self.noise_model = str(resolved_noise_models[0]).upper()
        self.operator_kind = str(self.operator_kind_names[0]).upper()
        self.operator_id = int(pgccpi_operator_id(self.operator_kind)) if len(self.operator_kind_names) == 1 else -1
        self.noise_attribute_names = list(PGCCPI_NOISE_ATTRIBUTE_NAMES)
        self.noise_attribute_dim = int(len(self.noise_attribute_names))
        self.observation_meta_names = list(PGCCPI_OBSERVATION_META_NAMES)
        self.observation_meta_dim = int(len(self.observation_meta_names))
        self.target_names = list(getattr(self.base_dataset, "target_names", []))
        self.target_dim = int(getattr(self.base_dataset, "target_dim", self.max_param_dim))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        # Forward epoch to the underlying base dataset so that augmentation
        # seeds vary across epochs. Mirrors the DistributedSampler API.
        setter = getattr(self.base_dataset, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))

    def decode_param_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        usable = predictions[..., : self.target_dim]
        if len(self.regularizer_names) > 1 or len(self.data_term_names) > 1:
            raise ValueError(
                "decode_param_predictions is ambiguous for mixed-family foundation datasets. "
                "Use decode_mixed_pgccpi_targets with per-sample regularizer/data-term ids."
            )
        return decode_pgccpi_target_tensor(usable, self.regularizer, self.data_term)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_items = self.base_dataset[int(index)]
        if len(base_items) == 6:
            noisy_tensor, target_tensor, _target_mask, best_score_tensor, clean_tensor, noisy_raw_tensor = base_items
        else:
            noisy_tensor, target_tensor, best_score_tensor, clean_tensor, noisy_raw_tensor = base_items
        base_record = self.base_dataset.records[int(index)]
        task_key = str(base_record["task_key"])
        padded_target, target_mask = pad_pgccpi_target_vector(target_tensor, pad_to=self.max_param_dim)
        regularizer_name = normalize_pgccpi_regularizer_name(base_record.get("regularizer", self.regularizer))
        # Retain the historical padded second DiffPIR loss slot.
        data_term_name = normalize_pgccpi_data_term_name(base_record.get("data_term", self.data_term))
        noise_model_name = normalize_pgccpi_noise_model_name(base_record.get("noise_model", self.noise_model))
        operator_kind_name = normalize_pgccpi_operator_name(base_record.get("operator_kind", self.operator_kind))
        regularizer_id = int(pgccpi_regularizer_id(regularizer_name))
        data_term_id = int(pgccpi_data_term_id(data_term_name))
        operator_id = int(pgccpi_operator_id(operator_kind_name))
        noise_params_raw = base_record.get("noise_params", {})
        noise_params = dict(noise_params_raw) if isinstance(noise_params_raw, dict) else {}
        observation_meta_applicable = _observation_meta_applicable(noise_model_name)
        observation_meta = _build_foundation_observation_meta(
            base_record,
            applicable=observation_meta_applicable,
            observed=observation_meta_applicable,
        )
        noise_attributes = encode_pgccpi_noise_attributes(
            noise_model_name,
            channel_independent=noise_params.get("channel_independent", None),
        )

        if self.joint_condition_dropout_prob > 0.0 and float(np.random.random()) < self.joint_condition_dropout_prob:
            noise_attributes = _unknown_noise_attributes()
            observation_meta = np.zeros(len(self.observation_meta_names), dtype=np.float32)
        else:
            if self.noise_attribute_dropout_prob > 0.0 and float(np.random.random()) < self.noise_attribute_dropout_prob:
                noise_attributes = _unknown_noise_attributes()
            if (
                observation_meta_applicable
                and self.observation_meta_dropout_prob > 0.0
                and float(np.random.random()) < self.observation_meta_dropout_prob
            ):
                observation_meta = _build_foundation_observation_meta(
                    base_record,
                    applicable=True,
                    observed=False,
                )

        return {
            "noisy_raw": noisy_raw_tensor.float(),
            "clean": clean_tensor.float(),
            "target_param": padded_target.float(),
            "target_param_mask": target_mask.float(),
            "oracle_metric_score": torch.tensor(float(best_score_tensor.item()), dtype=torch.float32),
            "regularizer_id": torch.tensor(regularizer_id, dtype=torch.long),
            "regularizer_name": regularizer_name,
            "data_term_id": torch.tensor(data_term_id, dtype=torch.long),
            "data_term_name": data_term_name,
            "noise_attributes": torch.from_numpy(noise_attributes).float(),
            "noise_attribute_names": list(self.noise_attribute_names),
            "operator_id": torch.tensor(operator_id, dtype=torch.long),
            "operator_name": operator_kind_name,
            "observation_meta": torch.from_numpy(observation_meta).float(),
            "observation_meta_names": list(self.observation_meta_names),
            "noise_seed": torch.tensor(int(base_record.get("noise_seed", 0)), dtype=torch.long),
            "augmentation_seed": torch.tensor(int(base_record.get("augmentation_seed", 0)), dtype=torch.long),
            "dataset_name": self.dataset_name,
            "task_key": task_key,
        }


def foundation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("foundation_collate_fn received an empty batch.")
    reference_noise_attribute_names = list(batch[0]["noise_attribute_names"])
    reference_observation_meta_names = list(batch[0]["observation_meta_names"])
    for item in batch[1:]:
        if list(item["noise_attribute_names"]) != reference_noise_attribute_names:
            raise ValueError(
                "foundation_collate_fn requires a consistent noise_attribute_names ordering within each batch."
            )
        if list(item["observation_meta_names"]) != reference_observation_meta_names:
            raise ValueError(
                "foundation_collate_fn requires a consistent observation_meta_names ordering within each batch."
            )

    return {
        "noisy_raw": torch.stack([item["noisy_raw"] for item in batch], dim=0),
        "clean": torch.stack([item["clean"] for item in batch], dim=0),
        "target_param": torch.stack([item["target_param"] for item in batch], dim=0),
        "target_param_mask": torch.stack([item["target_param_mask"] for item in batch], dim=0),
        "oracle_metric_score": torch.stack([item["oracle_metric_score"] for item in batch], dim=0),
        "regularizer_id": torch.stack([item["regularizer_id"] for item in batch], dim=0),
        "regularizer_name": [item["regularizer_name"] for item in batch],
        "data_term_id": torch.stack([item["data_term_id"] for item in batch], dim=0),
        "data_term_name": [item["data_term_name"] for item in batch],
        "noise_attributes": torch.stack([item["noise_attributes"] for item in batch], dim=0),
        "noise_attribute_names": reference_noise_attribute_names,
        "operator_id": torch.stack([item["operator_id"] for item in batch], dim=0),
        "operator_name": [item["operator_name"] for item in batch],
        "observation_meta": torch.stack([item["observation_meta"] for item in batch], dim=0),
        "observation_meta_names": reference_observation_meta_names,
        "noise_seed": torch.stack([item["noise_seed"] for item in batch], dim=0),
        "augmentation_seed": torch.stack([item["augmentation_seed"] for item in batch], dim=0),
        "dataset_name": [item["dataset_name"] for item in batch],
        "task_key": [item["task_key"] for item in batch],
    }
