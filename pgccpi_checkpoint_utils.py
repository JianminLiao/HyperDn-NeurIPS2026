from __future__ import annotations

from typing import Any, Dict

import torch

from restoration.pgccpi_observation_encoding import (
    PGCCPI_NOISE_ATTRIBUTE_NAMES,
    PGCCPI_OBSERVATION_META_NAMES,
)


def load_checkpoint_payload(path: str, *, map_location: Any = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint at {path} must be a dict-like payload, got {type(payload)!r}.")
    return payload


def unwrap_model_state_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"model_state_dict must be dict-like, got {type(state_dict)!r}.")
    return state_dict


def extract_backbone_state_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    backbone_state = payload.get("backbone_state_dict", None)
    if isinstance(backbone_state, dict):
        return backbone_state

    state_dict = unwrap_model_state_dict(payload)
    prefixed = {
        str(key)[len("backbone.") :]: value
        for key, value in state_dict.items()
        if str(key).startswith("backbone.")
    }
    if prefixed:
        return prefixed
    return state_dict


def _remap_backbone_keys(src_state: Dict[str, Any], target_keys: set) -> Dict[str, Any]:
    """Remap backbone keys between naming conventions.

    Foundation model uses sequential indexing: stem.0., stages.0., stages.1.
    Ablation model uses underscore indexing: stem_0., stages_0., stages_1.
    This function tries both directions to find matches.
    """
    if set(src_state.keys()) & target_keys:
        return src_state  # already matching
    import re
    # Try: stem.X. -> stem_X., stages.X. -> stages_X.
    dot_to_underscore = {}
    for key, value in src_state.items():
        new_key = re.sub(r'^(stem|stages|downsample_layers)\.(\d+)\.', r'\1_\2.', key)
        dot_to_underscore[new_key] = value
    if dot_to_underscore.keys() & target_keys:
        return dot_to_underscore
    # Try reverse: stem_X. -> stem.X., stages_X. -> stages.X.
    underscore_to_dot = {}
    for key, value in src_state.items():
        new_key = re.sub(r'^(stem|stages|downsample_layers)_(\d+)\.', r'\1.\2.', key)
        underscore_to_dot[new_key] = value
    if underscore_to_dot.keys() & target_keys:
        return underscore_to_dot
    return src_state  # no remapping helped


def load_backbone_checkpoint(backbone: torch.nn.Module, path: str) -> Dict[str, Any]:
    payload = load_checkpoint_payload(path)
    backbone_state = extract_backbone_state_dict(payload)
    target_keys = set(backbone.state_dict().keys())
    backbone_state = _remap_backbone_keys(backbone_state, target_keys)
    incompatible = backbone.load_state_dict(backbone_state, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_model_checkpoint_forgiving(
    model: torch.nn.Module,
    payload_or_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Load every tensor, allowing only an explicit vocabulary-prefix extension.

    The historical name is retained for callers. Missing, unexpected, and other
    incompatible tensors are errors; they must never leave random model weights.
    """
    contract = payload_or_state.get("encoding_contract", {})
    if contract:
        from restoration.pgccpi_param_encoding import (
            PGCCPI_REGULARIZER_ORDER, PGCCPI_PARAM_ROLE_ORDER,
            PGCCPI_PARAM_SCALE_ORDER, PGCCPI_PARAM_FAMILY_ORDER,
        )
        from restoration.pgccpi_observation_encoding import PGCCPI_DATA_TERM_ORDER
        current = {"regularizer_order": PGCCPI_REGULARIZER_ORDER,
                   "data_term_order": PGCCPI_DATA_TERM_ORDER,
                   "param_role_order": PGCCPI_PARAM_ROLE_ORDER,
                   "param_scale_order": PGCCPI_PARAM_SCALE_ORDER,
                   "param_family_order": PGCCPI_PARAM_FAMILY_ORDER}
        for field, names in current.items():
            if field in contract and list(contract[field]) != list(names[:len(contract[field])]):
                raise RuntimeError(f"Checkpoint vocabulary is not a compatible prefix: {field}")
    state_dict = unwrap_model_state_dict(payload_or_state)
    model_state = model.state_dict()
    missing = set(model_state) - set(state_dict)
    unexpected = set(state_dict) - set(model_state)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    prefix_extend_keys = {
        "regularizer_embedding.weight", "role_embedding.weight", "role_log_var",
        "family_gate.weight", "family_bias.weight",
    }
    loaded = {}
    prefix_extended = []
    for key, value in state_dict.items():
        target = model_state[key]
        if value.shape == target.shape:
            loaded[key] = value
        elif (key in prefix_extend_keys and value.ndim == target.ndim
              and value.shape[1:] == target.shape[1:] and value.shape[0] < target.shape[0]):
            extended = target.clone()
            extended[:value.shape[0]].copy_(value)
            loaded[key] = extended
            prefix_extended.append(key)
        else:
            raise RuntimeError(f"Checkpoint tensor shape mismatch for {key}: {tuple(value.shape)} vs {tuple(target.shape)}")
    model.load_state_dict(loaded, strict=True)
    return {"missing_keys": [], "unexpected_keys": [], "skipped_shape_mismatch": [],
            "prefix_extended": prefix_extended}


def _normalize_name_sequence(payload: Dict[str, Any], field_name: str):
    values = payload.get(field_name, None)
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be list-like in a foundation checkpoint, got {type(values)!r}.")
    return [str(value) for value in values]


def validate_foundation_observation_contract(payload: Dict[str, Any], *, path: str = "") -> None:
    checkpoint_label = path or "<checkpoint>"
    expected_fields = {
        "noise_attribute_names": [str(name) for name in PGCCPI_NOISE_ATTRIBUTE_NAMES],
        "observation_meta_names": [str(name) for name in PGCCPI_OBSERVATION_META_NAMES],
    }
    for field_name, expected_names in expected_fields.items():
        actual_names = _normalize_name_sequence(payload, field_name)
        if actual_names is None:
            continue
        if list(actual_names) != list(expected_names):
            raise ValueError(
                f"Foundation checkpoint {checkpoint_label} is incompatible with the current observation contract: "
                f"{field_name}={actual_names}, expected={expected_names}. "
                "Retrain the checkpoint or migrate it explicitly."
            )
