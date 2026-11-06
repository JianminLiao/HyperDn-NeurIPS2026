import glob
import json
import os
import re
from typing import Dict, List

import torch

from find_optimal_diffpir_denoise import _write_summary_files
from restoration.curve_pairs import curve_pairs_path


OUTPUT_DIR = os.getenv("JOB_OUTPUT_DIR", os.getenv("SAVE_DIR", "dataset_labels"))
RUN_NAME = os.getenv("RUN_NAME", "").strip()
DELETE_SHARDS_AFTER_MERGE = os.getenv("DELETE_SHARDS_AFTER_MERGE", "").strip().lower() in {"1", "true", "yes", "on"}
SHARD_NAME_RE = re.compile(r"_shard(\d+)of(\d+)\.pt$")
SHARD_RANGE_RE = re.compile(r"_shard_(\d+)_(\d+)\.pt$")


def _torch_load(path: str) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _require_run_name() -> str:
    if not RUN_NAME:
        raise ValueError("RUN_NAME must be provided.")
    return str(RUN_NAME)


def _list_shard_paths(output_dir: str, run_name: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(output_dir, f"{run_name}_shard*of*.pt")))
    range_paths = sorted(glob.glob(os.path.join(output_dir, f"{run_name}_shard_*.pt")))
    shard_paths = [
        path for path in sorted(set(paths + range_paths))
        if SHARD_NAME_RE.search(os.path.basename(path)) or SHARD_RANGE_RE.search(os.path.basename(path))
    ]
    if not shard_paths:
        raise FileNotFoundError(f"No DiffPIR shard files found for run {run_name} in {output_dir}")

    indexed_paths = []
    range_spans = []
    for path in shard_paths:
        basename = os.path.basename(path)
        indexed_match = SHARD_NAME_RE.search(basename)
        if indexed_match:
            indexed_paths.append((int(indexed_match.group(1)), int(indexed_match.group(2)), path))
            continue
        range_match = SHARD_RANGE_RE.search(basename)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                raise ValueError(f"Invalid DiffPIR shard range: {basename}")
            range_spans.append((start, end, path))

    if indexed_paths and range_spans:
        raise ValueError("Mixed indexed/range-named DiffPIR shards are not supported in one merge.")

    if indexed_paths:
        total_counts = {total for _, total, _ in indexed_paths}
        if len(total_counts) != 1:
            raise ValueError(f"Shard total mismatch: {sorted(total_counts)}")
        total_count = next(iter(total_counts))
        shard_ids = [shard_id for shard_id, _, _ in indexed_paths]
        missing = sorted(set(range(total_count)) - set(shard_ids))
        if missing:
            raise ValueError(f"Missing shard ids before merge: {missing}")
        return [path for _, _, path in sorted(indexed_paths, key=lambda item: item[0])]

    previous_end = None
    ordered_paths = []
    for start, end, path in sorted(range_spans, key=lambda item: (item[0], item[1], item[2])):
        if previous_end is not None and start != previous_end + 1:
            raise ValueError(
                f"Non-contiguous DiffPIR shard ranges before merge: previous_end={previous_end}, next_start={start}"
            )
        previous_end = end
        ordered_paths.append(path)
    return ordered_paths


def _canonicalize(value):
    if isinstance(value, float):
        return round(value, 15)
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize(v) for v in value)
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in value.items()}
    return value


def _config_signature(config: Dict) -> Dict:
    signature = dict(config)
    for key in (
        "run_name",
        "image_indices",
        "shard_id",
        "num_shards",
        "shard_image_indices",
        "base_run_name",
        "ground_truth_path",
    ):
        signature.pop(key, None)
    return _canonicalize(signature)


def _merge_main_payloads(payloads: List[Dict], run_name: str) -> Dict:
    first_config = dict(payloads[0].get("config", {}))
    signature = _config_signature(first_config)
    merged = dict(payloads[0])
    merged["config"] = dict(signature)
    merged["config"]["run_name"] = run_name
    merged["config"]["merged_from_shards"] = len(payloads)
    merged["records"] = []
    merged_image_indices = []
    for payload in payloads:
        config = dict(payload.get("config", {}))
        if _config_signature(config) != signature:
            raise ValueError("DiffPIR shard config mismatch.")
        merged["records"].extend(list(payload.get("records", [])))
        merged_image_indices.extend(int(v) for v in config.get("image_indices", []))
    merged["records"] = sorted(merged["records"], key=lambda record: str(record["task_key"]))
    merged["config"]["image_indices"] = sorted(dict.fromkeys(merged_image_indices))
    summary = {
        "run_name": run_name,
        "dataset": merged["config"].get("dataset"),
        "split": merged["config"].get("split"),
        "regularizer": merged.get("regularizer", "DIFFPIR"),
        "data_term": merged.get("data_term", "L2"),
        "noise_model": merged.get("noise_model", "GAUSSIAN"),
        "operator_kind": merged.get("operator_kind", "IDENTITY_DENOISE"),
        "primary_metric": merged["config"].get("primary_metric", "PSNR"),
        "metric_names": list(merged["config"].get("metric_names", ["PSNR", "SSIM"])),
        "completed_tasks": int(len(merged["records"])),
    }
    merged["summary"] = summary
    return merged


def _merge_curve_payloads(payloads: List[Dict], run_name: str, merged_main_path: str) -> Dict:
    first = payloads[0]
    first_config = dict(first.get("config", {}))
    signature = _config_signature(first_config)
    merged = dict(first)
    merged["config"] = dict(signature)
    merged["config"]["run_name"] = run_name
    merged["config"]["merged_from_shards"] = len(payloads)
    merged["config"]["ground_truth_path"] = str(merged_main_path)
    records = []
    for payload in payloads:
        config = dict(payload.get("config", {}))
        if _config_signature(config) != signature:
            raise ValueError("DiffPIR curve shard config mismatch.")
        records.extend(list(payload.get("records", [])))
    merged["records"] = sorted(records, key=lambda record: str(record["task_key"]))
    return merged


def _validate_main_payloads(payloads: List[Dict]) -> None:
    seen_image_indices = set()
    seen_task_keys = set()
    for payload in payloads:
        config = dict(payload.get("config", {}))
        image_indices = [int(v) for v in config.get("image_indices", [])]
        records = list(payload.get("records", []))
        if image_indices and len(records) != len(image_indices):
            raise ValueError(
                f"DiffPIR shard record count mismatch for {config.get('run_name')}: "
                f"records={len(records)} image_indices={len(image_indices)}"
            )
        record_image_indices = sorted({int(record["image_index"]) for record in records})
        if image_indices and record_image_indices != sorted(image_indices):
            raise ValueError(f"DiffPIR shard image index mismatch for {config.get('run_name')}.")
        for image_index in image_indices:
            if image_index in seen_image_indices:
                raise ValueError(f"Duplicate DiffPIR image index across shards: {image_index}")
            seen_image_indices.add(image_index)
        for record in records:
            task_key = str(record["task_key"])
            if task_key in seen_task_keys:
                raise ValueError(f"Duplicate DiffPIR task key across shards: {task_key}")
            seen_task_keys.add(task_key)


def _validate_curve_payloads(payloads: List[Dict], main_payloads: List[Dict]) -> None:
    expected_task_keys = {
        str(record["task_key"])
        for payload in main_payloads
        for record in list(payload.get("records", []))
    }
    seen_task_keys = set()
    for payload in payloads:
        for record in list(payload.get("records", [])):
            task_key = str(record["task_key"])
            if task_key in seen_task_keys:
                raise ValueError(f"Duplicate DiffPIR curve task key across shards: {task_key}")
            seen_task_keys.add(task_key)
    if seen_task_keys != expected_task_keys:
        missing = sorted(expected_task_keys - seen_task_keys)[:5]
        extra = sorted(seen_task_keys - expected_task_keys)[:5]
        raise ValueError(
            "DiffPIR curve shard task keys do not match main payloads. "
            f"missing={missing} extra={extra}"
        )


def _summary_paths(path: str) -> List[str]:
    base, _ = os.path.splitext(path)
    return [f"{base}_summary.json", f"{base}_summary.txt"]


def main() -> None:
    run_name = _require_run_name()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shard_paths = _list_shard_paths(OUTPUT_DIR, run_name)
    main_payloads = [_torch_load(path) for path in shard_paths]
    _validate_main_payloads(main_payloads)
    merged = _merge_main_payloads(main_payloads, run_name)
    merged_path = os.path.join(OUTPUT_DIR, f"{run_name}_merged.pt")
    merged["summary"]["output_path"] = merged_path
    merged["summary"]["curve_pairs_path"] = curve_pairs_path(merged_path)
    torch.save(merged, merged_path)
    _write_summary_files(os.path.splitext(merged_path)[0], merged["summary"])

    curve_shard_paths = [curve_pairs_path(path) for path in shard_paths]
    if not all(os.path.exists(path) for path in curve_shard_paths):
        raise FileNotFoundError("Missing DiffPIR curve_pairs shard sidecar.")
    curve_payloads = [_torch_load(path) for path in curve_shard_paths]
    _validate_curve_payloads(curve_payloads, main_payloads)
    merged_curve = _merge_curve_payloads(curve_payloads, run_name, merged_path)
    merged_curve_path = curve_pairs_path(merged_path)
    torch.save(merged_curve, merged_curve_path)

    merged_json_path = os.path.join(OUTPUT_DIR, f"{run_name}_merged_manifest.json")
    with open(merged_json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": run_name,
                "merged_path": merged_path,
                "curve_pairs_path": merged_curve_path,
                "num_shards": len(shard_paths),
                "shard_paths": shard_paths,
            },
            handle,
            indent=2,
        )

    if DELETE_SHARDS_AFTER_MERGE:
        for path in shard_paths:
            for target in [path, curve_pairs_path(path), *_summary_paths(path)]:
                if os.path.exists(target):
                    try:
                        os.remove(target)
                    except OSError:
                        pass

    print(f"Merged {len(shard_paths)} DiffPIR shards into {merged_path}")


if __name__ == "__main__":
    main()
