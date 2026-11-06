"""Check the narrow public package and every distributed predictor checkpoint."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    errors = []
    manifest_path = ROOT / "release_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Missing release_manifest.json: the release inventory is not finalized.")
    manifest = json.loads(manifest_path.read_text())
    paths = manifest["files"]
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Missing release file: {name}")
        elif path.suffix == ".py":
            try:
                ast.parse(path.read_text(), filename=name)
            except SyntaxError as exc:
                errors.append(str(exc))
    sums = ROOT / "SHA256SUMS"
    if not sums.exists():
        errors.append("Missing SHA256SUMS")
    else:
        entries = dict(line.split("  ", 1)[::-1] for line in sums.read_text().splitlines() if line)
        expected = set(paths) - {"SHA256SUMS"}
        if set(entries) != expected:
            errors.append("SHA256SUMS does not cover exactly the release inventory")
        for name, expected_hash in entries.items():
            path = ROOT / name
            if path.is_file() and sha256(path) != expected_hash:
                errors.append(f"Checksum mismatch: {name}")

    weights = json.loads((ROOT / "weight_manifest.json").read_text())
    expected_weights = {"source_s43", "diffpir_2shot_s43", "cbm3d_2shot_s43"}
    if {item["id"] for item in weights["weights"]} != expected_weights:
        errors.append("Expected source, DiffPIR and CBM3D seed43 checkpoints")
    from infer import load_model
    import torch

    torch.set_num_threads(2)
    checked = []
    for weight in weights["weights"]:
        if weight["seed"] != 43:
            errors.append(f"Unexpected seed: {weight['id']}")
        path = ROOT / weight["path"]
        if not path.is_file():
            errors.append(f"Missing checkpoint: {weight['path']}")
            continue
        if path.stat().st_size != weight["bytes"] or sha256(path) != weight["sha256"]:
            errors.append(f"Weight provenance mismatch: {weight['path']}")
            continue
        model = load_model("cpu", path)  # Safe deserialization and strict tensor loading.
        if len(model.state_dict()) != weight["tensor_count"]:
            errors.append(f"Tensor count mismatch: {weight['path']}")
        checked.append(weight["id"])
        del model

    labels = json.loads((ROOT / "configs/label_banks.json").read_text())
    banks = labels["banks"]
    source_banks = [item for item in banks if item["id"].startswith("source_")]
    if len(banks) != 17 or len(source_banks) != 13:
        errors.append("Expected 13 source banks and four target banks")
    archive_info = labels["archive"]
    archive_path = ROOT / archive_info["path"]
    if archive_path.stat().st_size != archive_info["bytes"] or sha256(archive_path) != archive_info["sha256"]:
        errors.append("Oracle archive checksum mismatch")
    else:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != {item["path"] for item in banks} or len(members) != len(banks):
                errors.append("Oracle archive files differ from the label inventory")
            else:
                for bank in banks:
                    stream = archive.extractfile(bank["path"])
                    payload = stream.read() if stream is not None else b""
                    if len(payload) != bank["bytes"] or hashlib.sha256(payload).hexdigest() != bank["sha256"]:
                        errors.append(f"Oracle bank checksum mismatch: {bank['id']}")
    recipes = json.loads((ROOT / "configs/oracles.json").read_text())["banks"]
    if len(recipes) != 4 or any(item["group"] not in {"diffpir", "cbm3d"} for item in recipes):
        errors.append("Expected four target-oracle recipes")
    from fewshot_protocol import SHOTS, METHODS, SOURCES
    if SHOTS != [1, 2, 4, 16, 32, 64, 128] or set(METHODS) != {"ours", "imagenet_only", "afkham"}:
        errors.append("Few-shot grid differs from the selected release")
    outputs = {item["output"] for item in recipes}
    from run_fullcorpus_kendall_5seed import TRAIN_SOURCES
    if [item["ground_truth_path"] for item in TRAIN_SOURCES] != [
        item["path"] for item in source_banks
    ]:
        errors.append("Source pretraining and label inventory use different ordered banks")
    target_bank_paths = {item["path"] for item in banks if not item["id"].startswith("source_")}
    if outputs != target_bank_paths:
        errors.append("Target recipes and label inventory use different banks")
    for target in SOURCES.values():
        for source in target.values():
            if source["ground_truth_path"] not in outputs:
                errors.append("Few-shot label path has no corresponding oracle recipe")
    if errors:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"status": "PASS", "files": len(paths), "weights": checked,
                      "label_banks": len(banks), "target_recipes": len(recipes)}, indent=2))


if __name__ == "__main__":
    main()
