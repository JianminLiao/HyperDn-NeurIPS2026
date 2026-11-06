from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from restoration.pgccpi_observation_encoding import (
    PGCCPI_DATA_TERM_ORDER,
    PGCCPI_NOISE_ATTRIBUTE_NAMES,
    PGCCPI_OBSERVATION_META_NAMES,
)
from restoration.pgccpi_param_encoding import (
    MAX_PGCCPI_PARAM_DIM,
    PGCCPI_PARAM_FAMILY_ORDER,
    PGCCPI_PARAM_ROLE_ORDER,
    PGCCPI_PARAM_SCALE_ORDER,
    PGCCPI_REGULARIZER_ORDER,
    padded_pgccpi_param_role_ids,
    padded_pgccpi_param_scale_ids,
    pgccpi_param_family_id,
)


def _as_image_stats(value, fallback):
    sequence = list(value if value is not None else fallback)
    if len(sequence) != 3:
        raise ValueError(f"Expected 3-channel image statistics, got {sequence}.")
    return torch.tensor([float(v) for v in sequence], dtype=torch.float32).view(1, 3, 1, 1)


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(int(input_dim)),
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Dropout(float(dropout_p)),
        nn.Linear(int(hidden_dim), int(output_dim)),
    )


def _gn_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0 and channels // g >= 1:
            return g
    return 1


def _parse_stage_indices(value):
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


class PGCCPiFoundationModel(nn.Module):
    """Foundation model for hyperparameter prediction across restoration configurations."""

    def __init__(
        self,
        *,
        backbone_name: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        stage_indices=(0, 1),
        stage_out_dim: int = 128,
        fixed_grid: int = 12,
        hidden_dim: int = 512,
        task_embed_dim: int = 32,
        observation_embed_dim: int = 16,
        slot_embed_dim: int = 32,
        bilinear_dim: int = 128,
        max_param_dim: int = MAX_PGCCPI_PARAM_DIM,
        num_regularizers: int = len(PGCCPI_REGULARIZER_ORDER),
        num_data_terms: int = len(PGCCPI_DATA_TERM_ORDER),
        noise_attribute_dim: int = len(PGCCPI_NOISE_ATTRIBUTE_NAMES),
        observation_meta_dim: int = len(PGCCPI_OBSERVATION_META_NAMES),
        use_hw_log: bool = True,
        head_type: str = "bilinear",
        use_simple_head: bool = False,
        dropout_p: float = 0.1,
        image_mean=None,
        image_std=None,
    ):
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.max_param_dim = int(max_param_dim)
        self.task_embed_dim = int(task_embed_dim)
        self.observation_embed_dim = int(observation_embed_dim)
        self.slot_embed_dim = int(slot_embed_dim)
        self.bilinear_dim = int(bilinear_dim)
        self.noise_attribute_dim = int(noise_attribute_dim)
        self.observation_meta_dim = int(observation_meta_dim)
        self.stage_out_dim = int(stage_out_dim)
        self.fixed_grid = int(fixed_grid)
        self.use_hw_log = bool(use_hw_log)
        # head_type: "bilinear" (default), "slot_aware", "simple"
        # use_simple_head is a legacy alias for head_type="simple"
        if use_simple_head and head_type == "bilinear":
            head_type = "simple"
        self.head_type = str(head_type).strip().lower()
        self.use_simple_head = self.head_type == "simple"  # backward compat
        self.stage_indices = tuple(int(v) for v in _parse_stage_indices(stage_indices))

        import timm
        from timm.data import resolve_model_data_config

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=bool(pretrained),
            features_only=True,
            out_indices=self.stage_indices,
            in_chans=3,
        )
        data_config = resolve_model_data_config(self.backbone)
        self.register_buffer(
            "input_mean",
            _as_image_stats(image_mean, data_config.get("mean", (0.485, 0.456, 0.406))),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            _as_image_stats(image_std, data_config.get("std", (0.229, 0.224, 0.225))),
            persistent=False,
        )

        self.stage_projs = nn.ModuleList()
        for channels in self.backbone.feature_info.channels():
            self.stage_projs.append(
                nn.Sequential(
                    nn.Conv2d(int(channels), self.stage_out_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_gn_groups(self.stage_out_dim), self.stage_out_dim),
                    nn.GELU(),
                )
            )
        self.fixed_pool = nn.AdaptiveAvgPool2d((self.fixed_grid, self.fixed_grid))

        n_stages = len(self.stage_projs)
        backbone_feat_dim = n_stages * self.stage_out_dim
        backbone_feat_dim += 2 if self.use_hw_log else 0

        self.regularizer_embedding = nn.Embedding(int(num_regularizers), self.task_embed_dim)
        self.data_term_embedding = nn.Embedding(int(num_data_terms), self.observation_embed_dim)
        self.noise_attribute_encoder = _build_mlp(
            input_dim=self.noise_attribute_dim,
            hidden_dim=32,
            output_dim=self.observation_embed_dim,
            dropout_p=dropout_p,
        )
        self.observation_meta_encoder = _build_mlp(
            input_dim=self.observation_meta_dim,
            hidden_dim=32,
            output_dim=self.observation_embed_dim,
            dropout_p=dropout_p,
        )
        self.condition_dim = self.task_embed_dim + (3 * self.observation_embed_dim)

        head_input_dim = backbone_feat_dim + self.condition_dim
        if self.head_type == "simple":
            self.param_head = _build_mlp(
                input_dim=head_input_dim,
                hidden_dim=int(hidden_dim),
                output_dim=self.max_param_dim,
                dropout_p=dropout_p,
            )
        elif self.head_type == "slot_aware":
            self.param_head = _build_mlp(
                input_dim=head_input_dim,
                hidden_dim=int(hidden_dim),
                output_dim=self.max_param_dim,
                dropout_p=dropout_p,
            )
            self.role_embedding = nn.Embedding(len(PGCCPI_PARAM_ROLE_ORDER), self.slot_embed_dim)
            self.scale_embedding = nn.Embedding(len(PGCCPI_PARAM_SCALE_ORDER), self.slot_embed_dim)
            self.slot_affine = nn.Linear(self.slot_embed_dim, 2)  # per-slot (scale, bias)
            nn.init.constant_(self.slot_affine.weight[:1], 0.0)  # scale init → 1 after exp
            nn.init.zeros_(self.slot_affine.bias)
        else:  # bilinear
            self.trunk = nn.Sequential(
                nn.LayerNorm(int(head_input_dim)),
                nn.Linear(int(head_input_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout_p)),
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout_p)),
            )
            self.role_embedding = nn.Embedding(len(PGCCPI_PARAM_ROLE_ORDER), self.slot_embed_dim)
            self.scale_embedding = nn.Embedding(len(PGCCPI_PARAM_SCALE_ORDER), self.slot_embed_dim)
            self.q_proj = nn.Linear(int(hidden_dim), self.bilinear_dim)
            self.slot_proj = nn.Linear(self.slot_embed_dim, self.bilinear_dim)
            self.h_out = nn.Linear(int(hidden_dim), self.max_param_dim)
            self.slot_out = nn.Linear(self.slot_embed_dim, 1)
            self.family_gate = nn.Embedding(len(PGCCPI_PARAM_FAMILY_ORDER), self.max_param_dim)
            self.family_bias = nn.Embedding(len(PGCCPI_PARAM_FAMILY_ORDER), self.max_param_dim)
            nn.init.ones_(self.family_gate.weight)
            nn.init.zeros_(self.family_bias.weight)

        # Homoscedastic uncertainty: one log(σ²) per parameter role (Kendall et al., 2018)
        self.role_log_var = nn.Parameter(torch.zeros(len(PGCCPI_PARAM_ROLE_ORDER)))

        family_lookup_rows = []
        role_lookup_rows = []
        scale_lookup_rows = []
        for regularizer_name in PGCCPI_REGULARIZER_ORDER[: int(num_regularizers)]:
            family_row = []
            role_row = []
            scale_row = []
            for data_term_name in PGCCPI_DATA_TERM_ORDER[: int(num_data_terms)]:
                family_row.append(pgccpi_param_family_id(regularizer_name, data_term_name))
                role_row.append(
                    padded_pgccpi_param_role_ids(
                        regularizer_name,
                        data_term_name,
                        pad_to=self.max_param_dim,
                    )
                )
                scale_row.append(
                    padded_pgccpi_param_scale_ids(
                        regularizer_name,
                        data_term_name,
                        pad_to=self.max_param_dim,
                    )
                )
            family_lookup_rows.append(family_row)
            role_lookup_rows.append(role_row)
            scale_lookup_rows.append(scale_row)

        self.register_buffer(
            "family_lookup",
            torch.as_tensor(family_lookup_rows, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "role_lookup",
            torch.as_tensor(np.asarray(role_lookup_rows), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "scale_lookup",
            torch.as_tensor(np.asarray(scale_lookup_rows), dtype=torch.long),
            persistent=False,
        )

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.input_mean) / self.input_std

    def _pooled_stage_mean(self, grid_feat: torch.Tensor) -> torch.Tensor:
        return grid_feat.mean(dim=(2, 3))

    def _extract_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        features = self.backbone(self.normalize_inputs(images))
        parts = []
        for feat, proj in zip(features, self.stage_projs):
            reduced = proj(feat)
            grid_feat = self.fixed_pool(reduced)
            parts.append(self._pooled_stage_mean(grid_feat))
        feat = torch.cat(parts, dim=1)
        if self.use_hw_log:
            hw = feat.new_tensor([H, W]).float().clamp_min(1.0)
            hw_log = torch.log(hw).unsqueeze(0).expand(B, -1)
            feat = torch.cat([feat, hw_log], dim=1)
        return feat

    def _default_observation_context(
        self,
        *,
        regularizer_ids: torch.Tensor,
        images: torch.Tensor,
        data_term_ids: Optional[torch.Tensor],
        noise_attributes: Optional[torch.Tensor],
        observation_meta: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(regularizer_ids.shape[0])
        if data_term_ids is None:
            data_term_ids = regularizer_ids.new_zeros((batch_size,))
        if noise_attributes is None:
            noise_attributes = images.new_zeros((batch_size, self.noise_attribute_dim))
            if self.noise_attribute_dim > 0:
                noise_attributes[:, 0] = 1.0
        if observation_meta is None:
            observation_meta = images.new_zeros((batch_size, self.observation_meta_dim))
        return (
            data_term_ids.long(),
            noise_attributes.float(),
            observation_meta.float(),
        )

    def _build_condition_embedding(
        self,
        *,
        regularizer_ids: torch.Tensor,
        data_term_ids: torch.Tensor,
        noise_attributes: torch.Tensor,
        observation_meta: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self.regularizer_embedding(regularizer_ids.long()),
                self.data_term_embedding(data_term_ids.long()),
                self.noise_attribute_encoder(noise_attributes.float()),
                self.observation_meta_encoder(observation_meta.float()),
            ],
            dim=-1,
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        regularizer_ids: torch.Tensor,
        data_term_ids: Optional[torch.Tensor] = None,
        noise_attributes: Optional[torch.Tensor] = None,
        observation_meta: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        data_term_ids, noise_attributes, observation_meta = self._default_observation_context(
            regularizer_ids=regularizer_ids,
            images=images,
            data_term_ids=data_term_ids,
            noise_attributes=noise_attributes,
            observation_meta=observation_meta,
        )
        backbone_features = self._extract_backbone_features(images)
        condition_embedding = self._build_condition_embedding(
            regularizer_ids=regularizer_ids,
            data_term_ids=data_term_ids,
            noise_attributes=noise_attributes,
            observation_meta=observation_meta,
        )
        head_input = torch.cat([backbone_features, condition_embedding], dim=-1)

        regularizer_ids = regularizer_ids.long()
        data_term_ids = data_term_ids.long()
        role_ids = self.role_lookup[regularizer_ids, data_term_ids]

        if self.head_type == "simple":
            param_pred = self.param_head(head_input)
        elif self.head_type == "slot_aware":
            scale_ids = self.scale_lookup[regularizer_ids, data_term_ids]
            slot_semantics = self.role_embedding(role_ids) + self.scale_embedding(scale_ids)
            base = self.param_head(head_input)
            affine = self.slot_affine(slot_semantics)  # [B, D, 2]
            slot_scale = torch.exp(affine[..., 0])   # [B, D]; matches init comment "scale init → 1 after exp"
            slot_bias = affine[..., 1]    # [B, D]
            param_pred = slot_scale * base + slot_bias
        else:  # bilinear
            h = self.trunk(head_input)
            family_ids = self.family_lookup[regularizer_ids, data_term_ids]
            scale_ids = self.scale_lookup[regularizer_ids, data_term_ids]

            slot_semantics = self.role_embedding(role_ids) + self.scale_embedding(scale_ids)
            q = self.q_proj(h)
            k = self.slot_proj(slot_semantics)
            interaction = torch.einsum("bd,bld->bl", q, k)
            slot_term = self.slot_out(slot_semantics).squeeze(-1)
            head_term = self.h_out(h)
            family_gate = self.family_gate(family_ids)
            family_bias = self.family_bias(family_ids)
            param_pred = family_gate * (interaction + slot_term) + head_term + family_bias

        return {"param_pred": param_pred, "role_ids": role_ids}
