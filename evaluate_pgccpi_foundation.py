from __future__ import annotations

import hashlib
import json
import os
import sys
import time

# On Windows with conda envs, cv2/Qt/etc. DLLs live in <env>/Library/bin which is
# NOT on the PATH that CreateProcess uses for dependent-DLL lookup when the parent
# shell exports a POSIX-style PATH (e.g. Git Bash). Register explicitly via
# os.add_dll_directory so cv2 import works in subprocess.
if sys.platform == "win32":
    for _rel in ["Library/bin", "Library/mingw-w64/bin", "Scripts", ""]:
        _p = os.path.join(sys.prefix, *_rel.split("/")) if _rel else sys.prefix
        if os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
            except (OSError, AttributeError):
                pass
from dataclasses import dataclass
from functools import lru_cache
from glob import glob
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pgccpi_checkpoint_utils import (
    load_checkpoint_payload,
    load_model_checkpoint_forgiving,
    validate_foundation_observation_contract,
)
from pgccpi_foundation_dataset import PGCCPiFoundationDataset, foundation_collate_fn
from afkham_baseline import AfkhamCNN, AfkhamCNNFoundationWrapper
from pgccpi_foundation_model import PGCCPiFoundationModel
from restoration import (
    CILKnownPSFDeblurSolver,
    CCPiPoissonGaussianDeblurSolver,
    HuberADMMTVDenoiseSolver,
    L1ADMMTVDenoiseSolver,
    MAEMetric,
    NormalizedSplitBregmanTVSolver,
    PSNRMetric,
    SSIMMetric,
)
from restoration.metrics import LPIPSMetric
from restoration.diffpir_official import OfficialDiffPIRDenoiser
from restoration.pgccpi_observation_encoding import (
    PGCCPI_DATA_TERM_ORDER,
)
from restoration.pgccpi_param_encoding import (
    MAX_PGCCPI_PARAM_DIM,
    PGCCPI_REGULARIZER_ORDER,
    decode_mixed_pgccpi_targets,
    pgccpi_solver_param_dict_from_tensor,
)


_DIFFPIR_REPO_ROOT = os.getenv("DIFFPIR_REPO_ROOT", "").strip() or os.path.join(
    os.path.dirname(__file__), "external", "DiffPIR_official"
)

_diffpir_denoiser_cache: Dict[str, OfficialDiffPIRDenoiser] = {}


@lru_cache(maxsize=1)
def _diffpir_executor_spec() -> Dict[str, Any]:
    model_name = os.getenv("DIFFPIR_MODEL_NAME", "256x256_diffusion_uncond").strip()
    model_path = os.getenv("DIFFPIR_MODEL_PATH", "").strip()
    if not model_path:
        model_path = os.path.join(_DIFFPIR_REPO_ROOT, "model_zoo", f"{model_name}.pt")
    model_path = os.path.abspath(model_path)
    sha256 = None
    if os.path.isfile(model_path):
        digest = hashlib.sha256()
        with open(model_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
    return {
        "repo_root": os.path.abspath(_DIFFPIR_REPO_ROOT),
        "model_name": model_name,
        "model_path": model_path,
        "model_sha256": sha256,
        "iter_num": int(os.getenv("DIFFPIR_ITER_NUM", "100") or 100),
        "fixed_zeta": 1.0,
    }


def _get_diffpir_denoiser(device: torch.device) -> OfficialDiffPIRDenoiser:
    spec = _diffpir_executor_spec()
    key = f"{device}:iter={spec['iter_num']}:model={spec['model_name']}:path={spec['model_path']}"
    if key not in _diffpir_denoiser_cache:
        _diffpir_denoiser_cache[key] = OfficialDiffPIRDenoiser(
            repo_root=spec["repo_root"],
            device=str(device),
            model_name=spec["model_name"],
            model_path=spec["model_path"],
            iter_num=spec["iter_num"],
        )
    return _diffpir_denoiser_cache[key]


@dataclass(frozen=True)
class SourceSpec:
    dataset_name: str
    ground_truth_path: str
    split: str | None = None


class Config:
    CHECKPOINT_PATH = ""
    EVAL_SOURCES = ""
    SOURCE_SET = "val"  # val / train / all / explicit
    SPLIT = "train"
    METRIC = ""
    KERNEL_ROOT = "./data/Levin8/kernels"
    LOSS_WEIGHT_PARAM = 1.0

    MAX_SAMPLES_PER_SOURCE = 0
    MAX_RECON_SAMPLES_PER_SOURCE = 0  # 0 => full reconstruction eval, <0 => disable
    WORKERS = 4
    BATCH_SIZE = 32
    USE_AMP = True
    SAVE_DIR = "eval_reports"
    SAVE_PREDICTIONS = True
    SAVE_RECON_DIR = ""


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
    Config.CHECKPOINT_PATH = _parse_env_str("CHECKPOINT_PATH", Config.CHECKPOINT_PATH)
    Config.EVAL_SOURCES = _parse_env_str("EVAL_SOURCES", Config.EVAL_SOURCES)
    Config.SOURCE_SET = _parse_env_str("SOURCE_SET", Config.SOURCE_SET)
    Config.SPLIT = _parse_env_str("SPLIT", Config.SPLIT)
    Config.METRIC = _parse_env_str("METRIC", Config.METRIC)
    Config.KERNEL_ROOT = _parse_env_str("KERNEL_ROOT", Config.KERNEL_ROOT)
    Config.LOSS_WEIGHT_PARAM = _parse_env_float("LOSS_WEIGHT_PARAM", Config.LOSS_WEIGHT_PARAM)

    Config.MAX_SAMPLES_PER_SOURCE = _parse_env_int("MAX_SAMPLES_PER_SOURCE", Config.MAX_SAMPLES_PER_SOURCE)
    Config.MAX_RECON_SAMPLES_PER_SOURCE = _parse_env_int(
        "MAX_RECON_SAMPLES_PER_SOURCE",
        Config.MAX_RECON_SAMPLES_PER_SOURCE,
    )
    Config.WORKERS = _parse_env_int("WORKERS", Config.WORKERS)
    Config.BATCH_SIZE = _parse_env_int("BATCH_SIZE", Config.BATCH_SIZE)
    Config.USE_AMP = _parse_env_bool("USE_AMP", Config.USE_AMP)
    Config.SAVE_DIR = _parse_env_str("SAVE_DIR", Config.SAVE_DIR)
    Config.SAVE_PREDICTIONS = _parse_env_bool("SAVE_PREDICTIONS", Config.SAVE_PREDICTIONS)
    Config.SAVE_RECON_DIR = os.getenv("SAVE_RECON_DIR", "").strip()


def _sanitize_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def _parse_source_specs(raw: str) -> List[SourceSpec]:
    specs: List[SourceSpec] = []
    for chunk in str(raw).split(";"):
        token = chunk.strip()
        if not token:
            continue
        if "|" not in token:
            raise ValueError(
                "EVAL_SOURCES entries must use DATASET|PATH syntax, "
                f"got {token!r}."
            )
        dataset_token, ground_truth_path = token.split("|", 1)
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


def _is_afkham_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    return "target_dim" in checkpoint and "BACKBONE_NAME" not in checkpoint.get("config", {})


def _build_afkham_model_from_checkpoint(
    checkpoint: Dict[str, Any], device: torch.device
) -> AfkhamCNNFoundationWrapper:
    target_dim = int(checkpoint["target_dim"])
    afkham = AfkhamCNN(in_channels=3, out_dim=target_dim)
    load_model_checkpoint_forgiving(afkham, checkpoint)
    wrapper = AfkhamCNNFoundationWrapper(afkham, max_param_dim=MAX_PGCCPI_PARAM_DIM)
    wrapper.eval()
    return wrapper.to(device)


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = load_checkpoint_payload(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported foundation checkpoint format: {path}")
    if not _is_afkham_checkpoint(checkpoint):
        validate_foundation_observation_contract(checkpoint, path=path)
    return checkpoint


def _find_latest_foundation_checkpoint() -> str:
    pattern = os.path.join("checkpoints", "pgccpi_foundation_*_best.pth")
    candidates = glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No foundation checkpoints matched pattern: {pattern}")
    latest = max(candidates, key=os.path.getmtime)
    print(f">> Auto-detected latest foundation checkpoint: {latest}")
    return latest


def _source_spec_from_dict(item: Dict[str, Any]) -> SourceSpec:
    if "dataset_name" not in item or "ground_truth_path" not in item:
        raise KeyError(f"Malformed source spec in checkpoint: {item}")
    return SourceSpec(
        dataset_name=str(item["dataset_name"]),
        ground_truth_path=str(item["ground_truth_path"]),
        split=(str(item["split"]).strip() if item.get("split") else None),
    )


def _unique_source_specs(specs: Sequence[SourceSpec]) -> List[SourceSpec]:
    deduped: List[SourceSpec] = []
    seen = set()
    for spec in specs:
        key = (spec.dataset_name, str(spec.split or ""), os.path.abspath(spec.ground_truth_path))
        if key in seen:
            continue
        deduped.append(spec)
        seen.add(key)
    return deduped


def _resolve_eval_sources(checkpoint: Dict[str, Any]) -> List[SourceSpec]:
    if str(Config.EVAL_SOURCES).strip():
        return _parse_source_specs(Config.EVAL_SOURCES)

    source_set = str(Config.SOURCE_SET).strip().lower() or "val"
    checkpoint_val = [_source_spec_from_dict(item) for item in checkpoint.get("val_sources", [])]
    checkpoint_train = [_source_spec_from_dict(item) for item in checkpoint.get("train_sources", [])]

    if source_set == "val":
        return _unique_source_specs(checkpoint_val or checkpoint_train)
    if source_set == "train":
        return _unique_source_specs(checkpoint_train or checkpoint_val)
    if source_set == "all":
        return _unique_source_specs(list(checkpoint_val) + list(checkpoint_train))
    if source_set == "explicit":
        raise ValueError("SOURCE_SET=explicit requires EVAL_SOURCES to be set.")
    raise ValueError(f"Unsupported SOURCE_SET={Config.SOURCE_SET!r}.")


def _resolve_metric_name(config_dict: Dict[str, Any], key: str, fallback: str) -> str:
    env_value = str(getattr(Config, key, "")).strip()
    if env_value:
        return env_value
    checkpoint_value = str(config_dict.get(key, "")).strip()
    if checkpoint_value:
        return checkpoint_value
    return str(fallback)


def _build_dataset(
    source: SourceSpec,
    *,
    metric_name: str,
) -> PGCCPiFoundationDataset:
    return PGCCPiFoundationDataset(
        dataset_name=source.dataset_name,
        ground_truth_path=source.ground_truth_path,
        metric_name=metric_name,
        split=(source.split or Config.SPLIT),
        kernel_root=Config.KERNEL_ROOT,
        apply_augmentations=False,
        enable_rotate_90_270=False,
        enable_pixel_dithering=False,
        pixel_dithering_prob=0.0,
        enable_rgb_permutation=False,
        rgb_permutation_prob=0.0,
        max_param_dim=MAX_PGCCPI_PARAM_DIM,
    )


def _build_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> PGCCPiFoundationModel:
    config_dict = dict(checkpoint.get("config", checkpoint.get("model_config", {})))
    state_dict = checkpoint.get("model_state_dict", {})

    def _embedding_rows(key: str, default: int) -> int:
        weight = state_dict.get(key)
        if torch.is_tensor(weight):
            return int(weight.shape[0])
        return int(default)

    def _infer_stage_indices() -> tuple:
        cfg_val = config_dict.get("STAGE_INDICES")
        if cfg_val:
            if isinstance(cfg_val, str):
                return tuple(int(v.strip()) for v in cfg_val.split(",") if v.strip())
            return tuple(int(v) for v in cfg_val)
        n_stages = sum(1 for k in state_dict.keys() if k.startswith("stage_projs.") and k.endswith(".0.weight"))
        if n_stages <= 0:
            return (0, 1)
        if n_stages == 1:
            return (0,)
        return tuple(range(n_stages))

    model = PGCCPiFoundationModel(
        backbone_name=str(config_dict.get("BACKBONE_NAME", "convnextv2_base.fcmae_ft_in22k_in1k")),
        pretrained=False,
        stage_indices=_infer_stage_indices(),
        hidden_dim=int(config_dict.get("HIDDEN_DIM", 512)),
        task_embed_dim=int(config_dict.get("TASK_EMBED_DIM", 32)),
        observation_embed_dim=int(config_dict.get("OBSERVATION_EMBED_DIM", 16)),
        slot_embed_dim=int(config_dict.get("SLOT_EMBED_DIM", 32)),
        bilinear_dim=int(config_dict.get("BILINEAR_DIM", 128)),
        max_param_dim=MAX_PGCCPI_PARAM_DIM,
        num_regularizers=_embedding_rows("regularizer_embedding.weight", len(PGCCPI_REGULARIZER_ORDER)),
        num_data_terms=_embedding_rows("data_term_embedding.weight", len(PGCCPI_DATA_TERM_ORDER)),
        head_type=str(config_dict.get("HEAD_TYPE", "bilinear")),
        use_simple_head=bool(config_dict.get("USE_SIMPLE_HEAD", False)),
        dropout_p=float(config_dict.get("DROPOUT", 0.1)),
    ).to(device)
    load_info = load_model_checkpoint_forgiving(model, checkpoint)
    skipped_shape_mismatch = list(load_info.get("skipped_shape_mismatch", []))
    if skipped_shape_mismatch:
        raise RuntimeError(
            "Checkpoint is incompatible with the current foundation model architecture: "
            + ", ".join(skipped_shape_mismatch)
        )
    model.eval()
    return model


def _maybe_override_noise_attr(tensor: torch.Tensor) -> torch.Tensor:
    """If OVERRIDE_NOISE_ATTR env is set, replace the `noise_attributes` tensor
    with the specified vector (broadcast across the batch dim). Used by
    the compositional-encoding ablation (tab:ablation_composition). Empty env
    = no-op. See train_pgccpi_foundation.py for the matching helper.
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


def _prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    prepared: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            prepared[key] = value.to(device, non_blocking=True)
        else:
            prepared[key] = value
    if "noise_attributes" in prepared and torch.is_tensor(prepared["noise_attributes"]):
        prepared["noise_attributes"] = _maybe_override_noise_attr(prepared["noise_attributes"])
    return prepared


def _foundation_condition_kwargs(model: PGCCPiFoundationModel, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "regularizer_ids": payload["regularizer_id"],
        "data_term_ids": payload["data_term_id"],
        "noise_attributes": payload["noise_attributes"],
        "observation_meta": payload["observation_meta"],
    }


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
    return decode_mixed_pgccpi_targets(
        values.detach().cpu(),
        regularizer_ids.detach().cpu(),
        data_term_ids.detach().cpu(),
    )


def _decode_param_dict(
    regularizer: str,
    data_term: str,
    row: torch.Tensor,
    *,
    noise_sigma: float | None = None,
) -> Dict[str, float]:
    return pgccpi_solver_param_dict_from_tensor(
        row,
        regularizer,
        data_term=data_term,
        encoded=False,
        noise_sigma=noise_sigma,
    )


def _metric_objects(primary_metric_name: str, device: torch.device) -> List[Any]:
    metrics: List[Any] = [PSNRMetric(), SSIMMetric(), MAEMetric()]
    if str(primary_metric_name).upper() == "LPIPS":
        metrics.append(LPIPSMetric(net="vgg", device=str(device)))
    return metrics


def _effective_reconstruction_data_term(data_term: str, noise_model: str) -> str:
    from restoration.pgccpi_observation_encoding import (
        normalize_pgccpi_data_term_name, normalize_pgccpi_noise_model_name,
    )
    data_term_key = normalize_pgccpi_data_term_name(data_term)
    if data_term_key != "NLL":
        return data_term_key
    if normalize_pgccpi_noise_model_name(noise_model) in {"POISSON", "POISSON_GAUSSIAN"}:
        return "NLL_EXACT_PG"
    raise ValueError(f"NLL reconstruction requires Poisson-family noise, got {noise_model}.")


def _metric_state_template() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "param_loss": 0.0,
        "param_rel_err": 0.0,
        "param_abs_err": 0.0,
        "count": 0.0,
    }


def _accumulate_metric_state(
    state: Dict[str, float],
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
):
    batch_size = int(batch["target_param"].shape[0])

    param_loss = _masked_mse(outputs["param_pred"], batch["target_param"], batch["target_param_mask"])
    total_loss = Config.LOSS_WEIGHT_PARAM * param_loss

    pred_decoded = _decode_padded_params(
        outputs["param_pred"].detach(),
        batch["regularizer_id"],
        batch["data_term_id"],
    )
    target_decoded = _decode_padded_params(
        batch["target_param"],
        batch["regularizer_id"],
        batch["data_term_id"],
    )
    abs_err = torch.abs(pred_decoded - target_decoded)
    rel_err = abs_err / target_decoded.abs().clamp_min(1e-8)
    target_param_mask_cpu = batch["target_param_mask"].detach().cpu()
    abs_err = (abs_err * target_param_mask_cpu).sum(dim=1) / target_param_mask_cpu.sum(dim=1).clamp_min(1e-8)
    rel_err = (rel_err * target_param_mask_cpu).sum(dim=1) / target_param_mask_cpu.sum(dim=1).clamp_min(1e-8)

    state["loss"] += float(total_loss.item()) * batch_size
    state["param_loss"] += float(param_loss.item()) * batch_size
    state["param_rel_err"] += float(rel_err.mean().item()) * batch_size
    state["param_abs_err"] += float(abs_err.mean().item()) * batch_size
    state["count"] += batch_size


def _finalize_metric_state(state: Dict[str, float]) -> Dict[str, float]:
    count = max(float(state["count"]), 1.0)
    return {key: (value / count) for key, value in state.items() if key != "count"}


def _source_report_tag(source: SourceSpec) -> str:
    stem = os.path.splitext(os.path.basename(source.ground_truth_path))[0]
    split_token = f"_{source.split}" if source.split else ""
    return _sanitize_tag(f"{source.dataset_name}{split_token}_{stem}")


def _weighted_average_reports(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not items:
        return {}
    totals: Dict[str, float] = {}
    total_count = 0.0
    for item in items:
        count = float(item["sample_count"])
        total_count += count
        for key, value in item["metrics"].items():
            totals[key] = totals.get(key, 0.0) + (float(value) * count)
    if total_count <= 0.0:
        return {key: 0.0 for key in totals}
    return {key: value / total_count for key, value in totals.items()}


def _select_indices(length: int, limit: int) -> List[int]:
    if limit <= 0 or limit >= length:
        return list(range(length))
    return list(range(limit))


def _save_chw_png(path: str, image: np.ndarray) -> None:
    hwc = np.transpose(np.asarray(image, dtype=np.float32), (1, 2, 0))
    uint8 = (np.clip(hwc, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(uint8).save(path)


def _evaluate_reconstruction(
    model: PGCCPiFoundationModel,
    dataset: PGCCPiFoundationDataset,
    indices: Sequence[int],
    device: torch.device,
    primary_metric_name: str,
) -> Dict[str, Any] | None:
    if not indices:
        return None

    metrics = _metric_objects(primary_metric_name, device)
    primary_metric_key = str(primary_metric_name).upper()
    metric_values = {metric.get_name(): [] for metric in metrics}
    oracle_scores: List[float] = []
    pred_psnrs: List[float] = []
    save_root = str(Config.SAVE_RECON_DIR).strip()
    if save_root:
        for kind in ("clean", "noisy", "recon"):
            os.makedirs(os.path.join(save_root, kind), exist_ok=True)

    model.eval()
    with torch.no_grad():
        for index in tqdm(indices, desc=f"Recon eval {dataset.regularizer}/{dataset.dataset_name}", leave=False):
            sample = dataset[int(index)]
            base_record = dataset.base_dataset.records[int(index)]
            noisy_raw = sample["noisy_raw"].unsqueeze(0).to(device)

            condition_kwargs = {
                "regularizer_ids": sample["regularizer_id"].view(1).to(device),
                "data_term_ids": sample["data_term_id"].view(1).to(device),
                "noise_attributes": _maybe_override_noise_attr(
                    sample["noise_attributes"].view(1, -1).to(device)
                ),
                "observation_meta": sample["observation_meta"].view(1, -1).to(device),
            }
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and Config.USE_AMP)):
                outputs = model(noisy_raw, **condition_kwargs)

            pred_decoded = _decode_padded_params(
                outputs["param_pred"].detach().cpu(),
                sample["regularizer_id"].view(1),
                sample["data_term_id"].view(1),
            )[0]
            param_dict = _decode_param_dict(
                dataset.regularizer,
                dataset.data_term,
                pred_decoded,
            )
            if dataset.regularizer == "DIFFPIR":
                # Paper contract: output slot 1 is a historical padding target.
                param_dict["zeta"] = 1.0
            solver_data_term = _effective_reconstruction_data_term(dataset.data_term, dataset.noise_model)
            if dataset.operator_kind == "KNOWN_PSF_DEBLUR":
                kernel_info = dataset.base_dataset.kernels[int(base_record["kernel_index"])]
                if dataset.regularizer == "DIFFPIR":
                    raise ValueError(
                        "The portable release evaluates DiffPIR only for identity denoising."
                    )
                elif solver_data_term == "NLL_EXACT_PG":
                    solver = CCPiPoissonGaussianDeblurSolver(
                        psf=np.asarray(kernel_info["psf"], dtype=np.float32),
                        alpha=float(base_record["alpha"]),
                        sigma_read=float(base_record.get("sigma_read", 0.0) or 0.0),
                        regularizer_kind=dataset.regularizer,
                        background=float(base_record.get("background", 0.0) or 0.0),
                    )
                else:
                    solver = CILKnownPSFDeblurSolver(
                        psf=np.asarray(kernel_info["psf"], dtype=np.float32),
                        data_term_kind=solver_data_term,
                        regularizer_kind=dataset.regularizer,
                        data_term_options={"background": float(base_record.get("background", 0.0) or 0.0)},
                    )
                if dataset.regularizer != "DIFFPIR":
                    restored = solver.solve_with_param(
                        sample["noisy_raw"].detach().cpu().numpy(),
                        param=param_dict,
                    )
            elif dataset.operator_kind == "IDENTITY_DENOISE":
                if dataset.regularizer == "TV" and dataset.data_term == "HUBER":
                    solver = HuberADMMTVDenoiseSolver()
                    solver_param = param_dict
                elif dataset.regularizer == "L1":
                    solver = L1ADMMTVDenoiseSolver()
                    solver_param = float(param_dict["lambda"])
                elif dataset.regularizer == "TV":
                    if dataset.data_term == "L2":
                        solver = NormalizedSplitBregmanTVSolver()
                        solver_param = float(param_dict["lambda"])
                    elif solver_data_term == "NLL_EXACT_PG":
                        solver = CCPiPoissonGaussianDeblurSolver(
                            psf=np.asarray([[1.0]], dtype=np.float32),
                            alpha=float(base_record["alpha"]),
                            sigma_read=float(base_record.get("sigma_read", 0.0) or 0.0),
                            regularizer_kind="TV",
                            background=float(base_record.get("background", 0.0) or 0.0),
                        )
                        solver_param = param_dict
                    else:
                        solver = CILKnownPSFDeblurSolver(
                            psf=np.asarray([[1.0]], dtype=np.float32),
                            data_term_kind=solver_data_term,
                            regularizer_kind="TV",
                            data_term_options={"background": float(base_record.get("background", 0.0) or 0.0)},
                        )
                        solver_param = param_dict
                elif dataset.regularizer == "TGV":
                    if solver_data_term == "NLL_EXACT_PG":
                        solver = CCPiPoissonGaussianDeblurSolver(
                            psf=np.asarray([[1.0]], dtype=np.float32),
                            alpha=float(base_record["alpha"]),
                            sigma_read=float(base_record.get("sigma_read", 0.0) or 0.0),
                            regularizer_kind="TGV",
                            background=float(base_record.get("background", 0.0) or 0.0),
                        )
                    else:
                        solver = CILKnownPSFDeblurSolver(
                            psf=np.asarray([[1.0]], dtype=np.float32),
                            data_term_kind=solver_data_term,
                            regularizer_kind="TGV",
                            data_term_options={"background": float(base_record.get("background", 0.0) or 0.0)},
                        )
                    solver_param = param_dict
                elif dataset.regularizer == "DIFFPIR":
                    denoiser = _get_diffpir_denoiser(device)
                    noise_seed = int(base_record.get("noise_seed", 0))
                    pred_lambda = float(param_dict["lambda"])
                    pred_zeta = float(param_dict["zeta"])
                    restored_batch = denoiser._run_candidate_batch(
                        noisy_chw=sample["noisy_raw"].detach().cpu().numpy(),
                        params=[{"lambda": pred_lambda, "zeta": pred_zeta}],
                        seed=noise_seed,
                    )
                    restored = restored_batch[0].detach().cpu().numpy()
                elif dataset.regularizer == "CBM3D":
                    from bm3d import bm3d_rgb

                    noisy_chw = sample["noisy_raw"].detach().cpu().numpy()
                    sigma_psd = float(
                        np.clip(
                            param_dict["sigma_psd"],
                            float(os.getenv("CBM3D_SIGMA_PSD_MIN", "0.003")),
                            float(os.getenv("CBM3D_SIGMA_PSD_MAX", "0.30")),
                        )
                    )
                    restored_hwc = bm3d_rgb(
                        np.transpose(noisy_chw, (1, 2, 0)),
                        sigma_psd=sigma_psd,
                    )
                    restored = np.transpose(
                        np.asarray(restored_hwc, dtype=np.float32),
                        (2, 0, 1),
                    )
                elif dataset.regularizer == "NLM":
                    from skimage.restoration import denoise_nl_means

                    noisy_chw = sample["noisy_raw"].detach().cpu().numpy()
                    h = float(
                        np.clip(
                            param_dict["h"],
                            float(os.getenv("NLM_H_MIN", "0.003")),
                            float(os.getenv("NLM_H_MAX", "0.30")),
                        )
                    )
                    restored_hwc = denoise_nl_means(
                        np.transpose(noisy_chw, (1, 2, 0)),
                        h=h,
                        preserve_range=True,
                        channel_axis=-1,
                    )
                    restored = np.transpose(
                        np.asarray(restored_hwc, dtype=np.float32),
                        (2, 0, 1),
                    )
                else:
                    raise ValueError(
                        f"Unsupported denoising regularizer={dataset.regularizer!r} for reconstruction eval."
                    )
                if dataset.regularizer not in {"DIFFPIR", "CBM3D", "NLM"}:
                    restored = solver.solve(
                        sample["noisy_raw"].detach().cpu().numpy(),
                        param=solver_param,
                    )
            else:
                raise ValueError(f"Unsupported reconstruction operator_kind={dataset.operator_kind!r}.")
            clean = sample["clean"].detach().cpu().numpy()
            if save_root:
                filename = f"{int(index):04d}.png"
                _save_chw_png(os.path.join(save_root, "clean", filename), clean)
                _save_chw_png(
                    os.path.join(save_root, "noisy", filename),
                    sample["noisy_raw"].detach().cpu().numpy(),
                )
                _save_chw_png(os.path.join(save_root, "recon", filename), restored)
            for metric in metrics:
                metric_values[metric.get_name()].append(float(metric.calculate(clean, restored)))
            oracle_score = float(sample["oracle_metric_score"].item())
            oracle_scores.append(oracle_score)
            if primary_metric_key in metric_values and metric_values[primary_metric_key]:
                pred_psnrs.append(metric_values[primary_metric_key][-1])

    score_gaps = [p - o for p, o in zip(pred_psnrs, oracle_scores)] if pred_psnrs else []
    return {
        "reconstruction_sample_count": int(len(indices)),
        "oracle_metric_name": primary_metric_key,
        "oracle_score_mean": float(np.mean(np.asarray(oracle_scores, dtype=np.float64))),
        "oracle_score_std": float(np.std(np.asarray(oracle_scores, dtype=np.float64))),
        "primary_metric_gap_mean": float(np.mean(np.asarray(score_gaps, dtype=np.float64))) if score_gaps else 0.0,
        "primary_metric_gap_std": float(np.std(np.asarray(score_gaps, dtype=np.float64))) if score_gaps else 0.0,
        "metrics": {
            name: {
                "mean": float(np.mean(np.asarray(values, dtype=np.float64))),
                "std": float(np.std(np.asarray(values, dtype=np.float64))),
            }
            for name, values in metric_values.items()
        },
        # Per-sample dump (additive; enables paired significance tests, E5).
        "per_sample": {
            "indices": [int(i) for i in indices],
            "oracle_scores": [float(v) for v in oracle_scores],
            "metrics": {name: [float(v) for v in values] for name, values in metric_values.items()},
        },
    }


def _evaluate_source(
    model: PGCCPiFoundationModel,
    dataset: PGCCPiFoundationDataset,
    source: SourceSpec,
    device: torch.device,
    *,
    primary_metric_name: str,
) -> Dict[str, Any]:
    indices = _select_indices(len(dataset), Config.MAX_SAMPLES_PER_SOURCE)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.WORKERS,
        pin_memory=True,
        persistent_workers=(Config.WORKERS > 0),
        collate_fn=foundation_collate_fn,
    )

    metric_state = _metric_state_template()
    prediction_records: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _prepare_batch(batch, device)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and Config.USE_AMP)):
                outputs = model(
                    batch["noisy_raw"],
                    **_foundation_condition_kwargs(model, batch),
                )
            _accumulate_metric_state(metric_state, batch, outputs)

            if Config.SAVE_PREDICTIONS:
                pred_decoded = _decode_padded_params(
                    outputs["param_pred"].detach().cpu(),
                    batch["regularizer_id"].cpu(),
                    batch["data_term_id"].detach().cpu(),
                )
                target_decoded = _decode_padded_params(
                    batch["target_param"].detach().cpu(),
                    batch["regularizer_id"].cpu(),
                    batch["data_term_id"].detach().cpu(),
                )
                for row_index in range(batch["target_param"].shape[0]):
                    prediction_records.append(
                        {
                            "task_key": str(batch["task_key"][row_index]),
                            "dataset_name": str(batch["dataset_name"][row_index]),
                            "regularizer_name": str(batch["regularizer_name"][row_index]),
                            "data_term_name": str(batch["data_term_name"][row_index]),
                            "operator_name": str(batch["operator_name"][row_index]),
                            "noise_attribute_names": list(batch["noise_attribute_names"]),
                            "noise_attributes": batch["noise_attributes"][row_index].detach().cpu().tolist(),
                            "observation_meta_names": list(batch["observation_meta_names"]),
                            "observation_meta": batch["observation_meta"][row_index].detach().cpu().tolist(),
                            "pred_param_encoded": outputs["param_pred"][row_index].detach().cpu().tolist(),
                            "target_param_encoded": batch["target_param"][row_index].detach().cpu().tolist(),
                            "pred_param_decoded": pred_decoded[row_index].tolist(),
                            "target_param_decoded": target_decoded[row_index].tolist(),
                            "oracle_metric_score": float(batch["oracle_metric_score"][row_index].item()),
                        }
                    )

    reconstruction_summary = None
    recon_limit = int(Config.MAX_RECON_SAMPLES_PER_SOURCE)
    if recon_limit >= 0:
        recon_indices = _select_indices(len(dataset), len(dataset) if recon_limit == 0 else recon_limit)
        reconstruction_summary = _evaluate_reconstruction(
            model=model,
            dataset=dataset,
            indices=recon_indices,
            device=device,
            primary_metric_name=primary_metric_name,
        )

    report = {
        "source_tag": _source_report_tag(source),
        "dataset_name": dataset.dataset_name,
        "ground_truth_path": source.ground_truth_path,
        "split": (source.split or Config.SPLIT),
        "regularizer": dataset.regularizer,
        "data_term": dataset.data_term,
        "noise_model": dataset.noise_model,
        "operator_kind": dataset.operator_kind,
        "observation_meta_names": list(dataset.observation_meta_names),
        "sample_count": len(indices),
        "metrics": _finalize_metric_state(metric_state),
        "reconstruction_summary": reconstruction_summary,
    }
    if Config.SAVE_PREDICTIONS:
        report["prediction_records"] = prediction_records
    return report


def main():
    _apply_runtime_env_overrides()

    device_override = os.getenv("DEVICE", "").strip().lower()
    if device_override:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    checkpoint_path = str(Config.CHECKPOINT_PATH).strip() or _find_latest_foundation_checkpoint()
    checkpoint = _load_checkpoint(checkpoint_path, device)
    checkpoint_config = dict(checkpoint.get("config", {}))

    metric_name = _resolve_metric_name(checkpoint_config, "METRIC", "PSNR")
    eval_sources = _resolve_eval_sources(checkpoint)
    if not eval_sources:
        raise ValueError("No evaluation sources resolved from config/checkpoint.")

    if _is_afkham_checkpoint(checkpoint):
        model = _build_afkham_model_from_checkpoint(checkpoint, device)
    else:
        model = _build_model_from_checkpoint(checkpoint, device)
    source_reports: List[Dict[str, Any]] = []
    for source in eval_sources:
        print(f"Evaluating source: {source}")
        dataset = _build_dataset(
            source,
            metric_name=metric_name,
        )
        source_reports.append(
            _evaluate_source(
                model=model,
                dataset=dataset,
                source=source,
                device=device,
                primary_metric_name=metric_name,
            )
        )

    overall = _weighted_average_reports(source_reports)
    per_regularizer: Dict[str, Dict[str, Any]] = {}
    for regularizer in sorted({report["regularizer"] for report in source_reports}):
        items = [report for report in source_reports if report["regularizer"] == regularizer]
        per_regularizer[regularizer] = {
            "source_count": len(items),
            "sample_count": int(sum(int(item["sample_count"]) for item in items)),
            "metrics": _weighted_average_reports(items),
        }

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_tag = _sanitize_tag(f"{checkpoint.get('run_name', 'foundation')}_{Config.SOURCE_SET}_{timestamp}")

    prediction_path = None
    if Config.SAVE_PREDICTIONS:
        prediction_path = os.path.join(Config.SAVE_DIR, f"{report_tag}_predictions.jsonl")
        with open(prediction_path, "w", encoding="utf-8") as f:
            for report in source_reports:
                for record in report.pop("prediction_records", []):
                    payload = {
                        "source_tag": report["source_tag"],
                        **record,
                    }
                    f.write(json.dumps(payload) + "\n")

    summary = {
        "checkpoint_path": checkpoint_path,
        "run_name": checkpoint.get("run_name"),
        "metric": metric_name,
        "source_set": Config.SOURCE_SET,
        "diffpir_executor": _diffpir_executor_spec()
        if any(report["regularizer"] == "DIFFPIR" for report in source_reports)
        else None,
        "override_noise_attr_env": os.environ.get("OVERRIDE_NOISE_ATTR", "") or None,
        "sources": [
            {
                "dataset_name": report["dataset_name"],
                "ground_truth_path": report["ground_truth_path"],
                "split": report["split"],
                "regularizer": report["regularizer"],
                "data_term": report["data_term"],
                "noise_model": report["noise_model"],
                "operator_kind": report["operator_kind"],
                "observation_meta_names": report["observation_meta_names"],
                "sample_count": report["sample_count"],
                "metrics": report["metrics"],
                "reconstruction_summary": report["reconstruction_summary"],
            }
            for report in source_reports
        ],
        "overall": overall,
        "per_regularizer": per_regularizer,
        "prediction_path": prediction_path,
    }
    summary_path = os.path.join(Config.SAVE_DIR, f"{report_tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("==============================================")
    print("        PG/CCPi Foundation Eval Report        ")
    print("==============================================")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Source set:      {Config.SOURCE_SET}")
    print(f"Metric:          {metric_name}")
    print(f"Sources:         {len(source_reports)}")
    for report in source_reports:
        metrics = report["metrics"]
        line = (
            f"- {report['source_tag']}: n={report['sample_count']} "
            f"{report['data_term']}/{report['noise_model']}/{report['operator_kind']} "
            f"loss={metrics['loss']:.4f} "
            f"param_rel={metrics['param_rel_err']:.4f}"
        )
        print(line)
        reconstruction_summary = report.get("reconstruction_summary")
        if reconstruction_summary:
            recon_metric = reconstruction_summary.get("metrics", {}).get(str(metric_name).upper(), {})
            print(
                "  "
                f"recon_{str(metric_name).lower()}={float(recon_metric.get('mean', 0.0)):.4f} "
                f"oracle_{str(metric_name).lower()}={float(reconstruction_summary.get('oracle_score_mean', 0.0)):.4f} "
                f"gap={float(reconstruction_summary.get('primary_metric_gap_mean', 0.0)):+.4f}"
            )
    if overall:
        overall_line = (
            "Overall: "
            f"loss={overall.get('loss', 0.0):.4f} "
            f"param_rel={overall.get('param_rel_err', 0.0):.4f}"
        )
        print(overall_line)
    print(f"Saved summary to {summary_path}")
    if prediction_path:
        print(f"Saved predictions to {prediction_path}")


if __name__ == "__main__":
    main()
