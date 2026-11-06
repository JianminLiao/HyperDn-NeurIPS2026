from __future__ import annotations

import argparse
import ast
import re
import importlib.metadata
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
DATASET_ROOT = ROOT / "dataset_labels"
DEFAULT_OUTPUT_ROOT = ROOT / "experiment_runs"
DEFAULT_BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k"


def source_entry(source: Dict[str, str]) -> str:
    split = str(source.get("split", "")).strip()
    dataset_token = str(source["dataset_name"])
    if split:
        dataset_token = f"{dataset_token}@{split}"
    return f"{dataset_token}|{source['ground_truth_path']}"


def run_with_log(cmd: List[str], *, env: Dict[str, str], log_path: Path) -> None:
    os.makedirs(log_path.parent, exist_ok=True)
    run_env = {**env, "PYTHONIOENCODING": "utf-8"}
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def find_single_file(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one *{suffix} under {directory}, found {len(matches)}")
    return matches[0]


def train_run(
    *,
    train_sources: List[Dict[str, str]],
    run_name: str,
    checkpoint_dir: Path,
    runs_dir: Path,
    extra_env: Dict[str, Any],
    val_sources: List[Dict[str, str]] | None = None,
) -> Path:
    best_path = checkpoint_dir / f"{run_name}_best.pth"
    env = {
        **os.environ,
        "TRAIN_SOURCES": ";".join(source_entry(item) for item in train_sources),
        "VAL_SOURCES": ";".join(source_entry(item) for item in (val_sources or [])),
        "RUN_NAME": run_name,
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "RUNS_DIR": str(runs_dir),
    }
    for key, value in extra_env.items():
        env[str(key)] = str(value)
    contract = {
        "entry": "train_pgccpi_foundation.py", "train_sources": source_signatures(train_sources),
        "val_sources": source_signatures(val_sources or []), "env": environment_contract(env),
        "implementation": implementation_signature(),
    }
    init_path = str(extra_env.get("INIT_CHECKPOINT", ""))
    if init_path:
        contract["initial_checkpoint"] = file_signature(Path(init_path))
    preserve_contract(checkpoint_dir / f"{run_name}_contract.json", contract, [best_path])
    complete_path = checkpoint_dir / f"{run_name}_complete.json"
    if best_path.exists():
        if not complete_path.exists():
            raise RuntimeError(f"Interrupted run has a partial best checkpoint: {best_path}. Resume explicitly or use a fresh output directory.")
        print(f"[reuse verified completed run] {best_path}")
        return best_path
    log_path = checkpoint_dir / f"{run_name}.log"
    run_with_log([str(PYTHON), "-u", "train_pgccpi_foundation.py"], env=env, log_path=log_path)
    if not best_path.exists():
        raise FileNotFoundError(f"Expected best checkpoint at {best_path}")
    complete_path.write_text(json.dumps({"epochs": int(extra_env["EPOCHS"]),
                                        "checkpoint": file_signature(best_path)}, indent=2) + "\n")
    return best_path


def eval_run(
    *,
    checkpoint_path: Path,
    eval_sources: List[Dict[str, str]],
    eval_dir: Path,
    split: str,
    extra_env: Dict[str, str] | None = None,
) -> Path:
    os.makedirs(eval_dir, exist_ok=True)
    env = {
        **os.environ,
        "CHECKPOINT_PATH": str(checkpoint_path),
        "SOURCE_SET": "explicit",
        "EVAL_SOURCES": ";".join(source_entry(item) for item in eval_sources),
        "SPLIT": str(split),
        "METRIC": "PSNR",
        "MAX_RECON_SAMPLES_PER_SOURCE": "0",
        "MAX_SAMPLES_PER_SOURCE": "0",
        "SAVE_PREDICTIONS": "0",
        "SAVE_DIR": str(eval_dir),
        "BATCH_SIZE": "8",
        "WORKERS": "0",
        "USE_AMP": "0",
    }
    if extra_env:
        env.update(extra_env)
    existing = list(eval_dir.glob("*_summary.json"))
    preserve_contract(eval_dir / "evaluation_contract.json", {
        "checkpoint": file_signature(checkpoint_path), "sources": source_signatures(eval_sources),
        "split": split, "eval_env": environment_contract(env),
        "external_executor": external_executor_signature(env, eval_sources),
        "extra_env": extra_env or {}, "implementation": implementation_signature(),
    }, existing)
    if existing:
        return find_single_file(eval_dir, "_summary.json")
    log_path = eval_dir / "eval.log"
    run_with_log([str(PYTHON), "-u", "evaluate_pgccpi_foundation.py"], env=env, log_path=log_path)
    return find_single_file(eval_dir, "_summary.json")


def extract_eval_metrics(summary_path: Path) -> Dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    overall = dict(summary.get("overall", {}))
    source = dict(summary.get("sources", [{}])[0])
    recon = dict(source.get("reconstruction_summary") or {})
    recon_metrics = dict(recon.get("metrics") or {}).get("PSNR", {})
    return {
        "summary_path": str(summary_path),
        "checkpoint_path": str(summary.get("checkpoint_path", "")),
        "run_name": summary.get("run_name"),
        "param_rel_err": float(overall["param_rel_err"]),
        "recon_psnr_mean": float(recon_metrics["mean"]),
        "oracle_psnr_mean": float(recon["oracle_score_mean"]),
        "primary_metric_gap_mean": float(recon["primary_metric_gap_mean"]),
    }


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = ["param_rel_err", "recon_psnr_mean", "oracle_psnr_mean", "primary_metric_gap_mean"]
    summary = {"n_runs": len(rows)}
    n = len(rows)
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        mean = sum(values) / n
        summary[key] = mean
        if n <= 1:
            summary[f"{key}_std"] = 0.0
        else:
            var = sum((value - mean) ** 2 for value in values) / (n - 1)
            summary[f"{key}_std"] = var ** 0.5
    return summary


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(path.parent, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)



def file_signature(path: Path) -> Dict[str, Any]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def source_signatures(sources: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    return [{**source, "artifact": file_signature(ROOT / source["ground_truth_path"])} for source in sources]


def implementation_signature() -> Dict[str, Any]:
    names = ["train_pgccpi_foundation.py", "train_afkham_baseline.py", "dataset.py",
             "pgccpi_foundation_dataset.py", "pgccpi_foundation_model.py", "pgccpi_checkpoint_utils.py",
             "evaluate_pgccpi_foundation.py", "fewshot_protocol.py", "run_diffpir_transfer_experiments.py",
             "afkham_baseline.py", "create_imagenet_only_foundation_checkpoint.py"]
    names += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "restoration").glob("*.py"))]
    versions = {}
    for package in ["torch", "torchvision", "timm", "numpy", "scipy", "bm3d", "scikit-image"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"files": {name: file_signature(ROOT / name)["sha256"] for name in names}, "packages": versions}


def environment_contract(env: Dict[str, str]) -> Dict[str, Any]:
    """Record inherited scientific overrides as well as explicit launcher settings."""
    names = set()
    for path in list(ROOT.glob("*.py")) + list((ROOT / "restoration").glob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Config":
                for assignment in node.body:
                    if isinstance(assignment, ast.Assign):
                        names.update(target.id for target in assignment.targets if isinstance(target, ast.Name) and target.id.isupper())
        names.update(re.findall(r'(?:getenv|environ\.get|_parse_env\w*)\(\s*["\']([A-Z][A-Z0-9_]*)["\']', source))
    return {name: env.get(name) for name in sorted(names)}


def external_executor_signature(env: Dict[str, str], sources: List[Dict[str, str]]) -> Dict[str, Any]:
    if not any("diffpir" in source["ground_truth_path"].lower() for source in sources):
        return {}
    repo = Path(env.get("DIFFPIR_REPO_ROOT") or ROOT / "external/DiffPIR_official")
    name = env.get("DIFFPIR_MODEL_NAME") or "256x256_diffusion_uncond"
    prior = Path(env.get("DIFFPIR_MODEL_PATH") or repo / "model_zoo" / f"{name}.pt")
    return {"prior": file_signature(prior), "implementation": {
        str(path.relative_to(repo)): file_signature(path)["sha256"]
        for path in sorted(repo.rglob("*.py"))
    }}


def preserve_contract(path: Path, contract: Dict[str, Any], outputs=()) -> None:
    """Only reuse outputs created with the same recorded inputs and settings."""
    normalized = json.loads(json.dumps(contract))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise RuntimeError(f"Run contract changed: {path}. Choose a fresh output directory.")
    else:
        if any(Path(output).exists() for output in outputs):
            raise RuntimeError(f"Existing output has no run contract: {path}. Choose a fresh output directory.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, indent=2) + "\n")
