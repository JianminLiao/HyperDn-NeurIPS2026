"""Predict HyperDn hyperparameters, optionally restoring a noisy RGB image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pgccpi_foundation_model import PGCCPiFoundationModel
from restoration.pgccpi_observation_encoding import PGCCPI_DATA_TERM_ORDER
from restoration.pgccpi_param_encoding import PGCCPI_REGULARIZER_ORDER

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "diffpir": ROOT / "checkpoints/hyperdn_diffpir_2shot_s43.pt",
    "cbm3d": ROOT / "checkpoints/hyperdn_cbm3d_2shot_s43.pt",
}
CBM3D_SIGMA_RANGE = (0.003, 0.30)


def load_model(device="cpu", checkpoint=None):
    """Load every tensor strictly, preserving the selected checkpoint vocabulary."""
    payload = torch.load(checkpoint or CHECKPOINTS["diffpir"], map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    model = PGCCPiFoundationModel(
        pretrained=False,
        stage_indices=(0, 1),
        num_regularizers=state["regularizer_embedding.weight"].shape[0],
        num_data_terms=state["data_term_embedding.weight"].shape[0],
    )
    # Source/DiffPIR weights predate the additional CBM3D/UNKNOWN entries.
    for name in ("role_embedding", "scale_embedding", "family_gate", "family_bias"):
        rows, cols = state[name + ".weight"].shape
        setattr(model, name, torch.nn.Embedding(rows, cols))
    model.role_log_var = torch.nn.Parameter(torch.zeros_like(state["role_log_var"]))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_hyperparameter(model, rgb, algorithm):
    """BCHW RGB in [0,1] -> unified DiffPIR lambda or CBM3D sigma_psd.

    The predictor receives the Gaussian category, with zero observation metadata.
    It does not receive the noise standard deviation or a clean target image.
    """
    regularizer_id = PGCCPI_REGULARIZER_ORDER.index(algorithm.upper())
    if regularizer_id >= model.regularizer_embedding.num_embeddings:
        raise ValueError(f"This checkpoint has no {algorithm.upper()} entry; use its adapted checkpoint.")
    n = rgb.shape[0]
    output = model(
        rgb,
        regularizer_ids=torch.full((n,), regularizer_id, device=rgb.device, dtype=torch.long),
        data_term_ids=torch.full(
            (n,), PGCCPI_DATA_TERM_ORDER.index("L2"), device=rgb.device, dtype=torch.long
        ),
        noise_attributes=rgb.new_tensor([0, 1, 0, 0, 0, 0]).repeat(n, 1),
        observation_meta=rgb.new_zeros(n, 5),
    )
    return 10.0 ** output["param_pred"][:, 0]


def predict_lambda(model, rgb):
    """Return unified lambda = original DiffPIR lambda * sigma^2."""
    return predict_hyperparameter(model, rgb, "diffpir")


def solver_parameters(value, algorithm):
    """Use exactly the fixed-zeta / sigma clipping conventions of evaluation."""
    if algorithm == "diffpir":
        # Already unified: do not multiply by sigma^2 again.
        return {"lambda": float(value), "zeta": 1.0}
    return {"sigma_psd": float(np.clip(value, *CBM3D_SIGMA_RANGE))}


def restore_cbm3d(noisy_chw, sigma_psd):
    """Run the external CBM3D executor with the evaluation defaults."""
    from bm3d import bm3d_rgb

    restored = bm3d_rgb(
        np.transpose(noisy_chw, (1, 2, 0)),
        sigma_psd=float(np.clip(sigma_psd, *CBM3D_SIGMA_RANGE)),
    )
    return np.transpose(np.asarray(restored, dtype=np.float32), (2, 0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--image", type=Path, required=True, help="Noisy RGB image")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the algorithm's two-shot seed-43 weight")
    parser.add_argument("--device", default="cpu", help="HyperDn device, e.g. cpu or cuda")
    parser.add_argument("--output", type=Path, help="Save the restored image; omit for parameter prediction only")
    parser.add_argument("--json", type=Path, help="Also save the JSON printed to stdout")
    parser.add_argument("--diffpir-root", type=Path, default=ROOT / "external/DiffPIR_official")
    parser.add_argument("--diffpir-prior", type=Path, help="External 256x256_diffusion_uncond.pt")
    parser.add_argument("--diffpir-device", help="Diffusion-prior device; defaults to --device")
    parser.add_argument("--diffpir-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43, help="DiffPIR sampling seed")
    args = parser.parse_args()
    if args.diffpir_iterations < 2 or args.diffpir_iterations > 1000:
        parser.error("--diffpir-iterations must be between 2 and 1000")
    with Image.open(args.image) as image:
        noisy = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    if args.output and args.algorithm == "diffpir" and noisy.shape[1:] != (256, 256):
        parser.error("The released DiffPIR setting uses 256x256 RGB inputs; supply an image of that size.")
    checkpoint = args.checkpoint or CHECKPOINTS[args.algorithm]
    model = load_model(args.device, checkpoint)
    rgb = torch.from_numpy(noisy.copy())[None].to(args.device)
    value = float(predict_hyperparameter(model, rgb, args.algorithm)[0].cpu())
    if not np.isfinite(value):
        raise RuntimeError("The model produced a non-finite hyperparameter.")
    parameters = solver_parameters(value, args.algorithm)
    result = {
        "algorithm": args.algorithm,
        "checkpoint": str(checkpoint),
        "image": str(args.image),
        "prediction_before_solver_clip": value,
        "parameters": parameters,
        "parameter_units": (
            {"lambda": "original_lambda * sigma^2, sigma in RGB [0,1] units", "zeta": "dimensionless (fixed)"}
            if args.algorithm == "diffpir"
            else {"sigma_psd": "noise standard deviation in RGB [0,1] units"}
        ),
    }
    if args.output:
        # Free the predictor before loading the much larger diffusion prior.
        del model, rgb
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        if args.algorithm == "cbm3d":
            restored = restore_cbm3d(noisy, parameters["sigma_psd"])
        else:
            from restoration.diffpir_official import OfficialDiffPIRDenoiser

            denoiser = OfficialDiffPIRDenoiser(
                repo_root=str(args.diffpir_root),
                model_path=str(args.diffpir_prior) if args.diffpir_prior else "",
                device=args.diffpir_device or args.device,
                iter_num=args.diffpir_iterations,
            )
            restored = denoiser.denoise(
                noisy_chw=noisy, unified_lambda=parameters["lambda"], zeta=1.0, seed=args.seed
            )
            result["sampling_seed"] = args.seed
            result["diffpir_iterations"] = args.diffpir_iterations
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image = np.rint(np.clip(restored.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image).save(args.output)
        result["output"] = str(args.output)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
