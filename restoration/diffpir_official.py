from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Sequence

from PIL import Image as _PILImage  # noqa: F401 — before torch: DiffPIR utils_image imports torchvision (Windows DLL order).
import numpy as np
import torch
from skimage.metrics import structural_similarity as skimage_ssim


@contextmanager
def _prepend_sys_path(path: str):
    path_str = str(path)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _normalize_seed(seed: int) -> int:
    max_seed = 2**31 - 1
    value = int(seed) % max_seed
    return value if value != 0 else 1


_LPIPS_MODEL_CACHE: Dict[tuple[str, str], object] = {}


def _lpips_model(device: torch.device, net: str = "vgg"):
    cache_key = (str(device), str(net))
    if cache_key not in _LPIPS_MODEL_CACHE:
        import lpips

        model = lpips.LPIPS(net=str(net)).to(device)
        model.eval()
        _LPIPS_MODEL_CACHE[cache_key] = model
    return _LPIPS_MODEL_CACHE[cache_key]


def _score_restored_batch(
    clean_chw: np.ndarray,
    restored_batch: torch.Tensor,
    metric_names: Sequence[str],
) -> List[Dict[str, float]]:
    clean_np = np.asarray(clean_chw, dtype=np.float32)
    clean_batch = torch.from_numpy(clean_np[None, ...]).to(restored_batch.device, dtype=restored_batch.dtype)
    mse = torch.mean((restored_batch - clean_batch) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-8)
    psnr_values = (10.0 * torch.log10(1.0 / mse)).detach().cpu().numpy()
    metric_keys = [str(name).upper() for name in metric_names]
    lpips_values = None
    if "LPIPS" in metric_keys:
        clean_lpips = clean_batch * 2.0 - 1.0
        restored_lpips = restored_batch * 2.0 - 1.0
        with torch.no_grad():
            lpips_values = (
                _lpips_model(restored_batch.device, net="vgg")(restored_lpips, clean_lpips)
                .view(-1)
                .detach()
                .cpu()
                .numpy()
            )
    restored_np = restored_batch.detach().cpu().numpy()

    scores: List[Dict[str, float]] = []
    clean_hwc = np.transpose(clean_np, (1, 2, 0))
    for index, restored in enumerate(restored_np):
        row: Dict[str, float] = {}
        restored_hwc = np.transpose(restored, (1, 2, 0))
        for metric_name in metric_keys:
            if metric_name == "PSNR":
                row[metric_name] = float(psnr_values[index])
            elif metric_name == "SSIM":
                row[metric_name] = float(
                    skimage_ssim(clean_hwc, restored_hwc, data_range=1.0, channel_axis=2)
                )
            elif metric_name == "LPIPS":
                row[metric_name] = float(lpips_values[index])
            else:
                raise ValueError(f"Unsupported DiffPIR metric: {metric_name}.")
        scores.append(row)
    return scores


@dataclass
class DiffPIRScoreResult:
    param: Dict[str, float]
    scores: Dict[str, float]


class OfficialDiffPIRDenoiser:
    def __init__(
        self,
        *,
        repo_root: str,
        device: str = "cuda",
        model_name: str = "256x256_diffusion_uncond",
        model_path: str = "",
        iter_num: int = 20,
        iter_num_u: int = 1,
        skip_type: str = "quad",
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        num_train_timesteps: int = 1000,
        ddim_sample: bool = False,
        noise_init_img: str = "max",
        n_channels: int = 3,
    ):
        self.repo_root = os.path.abspath(str(repo_root))
        if not os.path.isdir(self.repo_root):
            raise FileNotFoundError(f"Official DiffPIR repo not found: {self.repo_root}")
        self.device = torch.device(device if str(device).strip() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = str(model_name)
        self.model_path = (
            os.path.abspath(str(model_path))
            if str(model_path).strip()
            else os.path.join(self.repo_root, "model_zoo", f"{self.model_name}.pt")
        )
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Official DiffPIR checkpoint not found: {self.model_path}")

        self.iter_num = int(iter_num)
        self.iter_num_u = int(iter_num_u)
        self.skip_type = str(skip_type).strip().lower()
        self.eta = float(eta)
        self.guidance_scale = float(guidance_scale)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.num_train_timesteps = int(num_train_timesteps)
        self.ddim_sample = bool(ddim_sample)
        self.noise_init_img = str(noise_init_img)
        self.n_channels = int(n_channels)

        self._utils_model = None
        self._create_model_and_diffusion = None
        self._model_and_diffusion_defaults = None
        self._args_to_dict = None
        self._load_official_modules()
        self._build_model()
        self._build_schedule()

    def _load_official_modules(self) -> None:
        with _prepend_sys_path(self.repo_root):
            utils_model = importlib.import_module("utils.utils_model")
            script_util = importlib.import_module("guided_diffusion.script_util")
        self._utils_model = utils_model
        self._create_model_and_diffusion = getattr(script_util, "create_model_and_diffusion")
        self._model_and_diffusion_defaults = getattr(script_util, "model_and_diffusion_defaults")
        self._args_to_dict = getattr(script_util, "args_to_dict")

    def _default_model_config(self) -> Dict[str, object]:
        if self.model_name == "diffusion_ffhq_10m":
            return {
                "model_path": self.model_path,
                "num_channels": 128,
                "num_res_blocks": 1,
                "attention_resolutions": "16",
            }
        return {
            "model_path": self.model_path,
            "num_channels": 256,
            "num_res_blocks": 2,
            "attention_resolutions": "8,16,32",
        }

    def _build_model(self) -> None:
        model_config = self._default_model_config()
        args = self._utils_model.create_argparser(model_config).parse_args([])
        model, diffusion = self._create_model_and_diffusion(
            **self._args_to_dict(args, self._model_and_diffusion_defaults().keys())
        )
        state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        for _, parameter in model.named_parameters():
            parameter.requires_grad = False
        self.model = model.to(self.device)
        self.diffusion = diffusion

    def _build_schedule(self) -> None:
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_train_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        self.alphas = 1.0 - self.betas
        alphas_cumprod_np = np.cumprod(self.alphas.detach().cpu().numpy(), axis=0).astype(np.float32)
        self.alphas_cumprod_cpu = torch.from_numpy(alphas_cumprod_np).float()
        self.alphas_cumprod = self.alphas_cumprod_cpu.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.reduced_alpha_cumprod = (
            self.sqrt_1m_alphas_cumprod / self.sqrt_alphas_cumprod
        ).detach().cpu().numpy()
        self.sigmas = torch.flip(
            self.sqrt_1m_alphas_cumprod / self.sqrt_alphas_cumprod,
            dims=[0],
        )

        skip = self.num_train_timesteps // self.iter_num
        if self.skip_type == "uniform":
            seq = [int(i * skip) for i in range(self.iter_num)]
            if skip > 1:
                seq.append(self.num_train_timesteps - 1)
        else:
            seq = np.sqrt(np.linspace(0, self.num_train_timesteps**2, self.iter_num))
            seq = [int(value) for value in list(seq)]
            seq[-1] = seq[-1] - 1
        self.seq = list(seq)

    def _sample_shared_noise(
        self,
        shape: Sequence[int],
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
        return noise.to(self.device)

    def _run_candidate_batch(
        self,
        *,
        noisy_chw: np.ndarray,
        params: Sequence[Dict[str, float]],
        seed: int,
    ) -> torch.Tensor:
        batch_size = int(len(params))
        noisy = np.asarray(noisy_chw, dtype=np.float32)
        y = torch.from_numpy(noisy[None, ...]).to(self.device, dtype=torch.float32).repeat(batch_size, 1, 1, 1)
        y_condition = 2.0 * y - 1.0

        lambdas = torch.tensor([float(item["lambda"]) for item in params], dtype=torch.float32, device=self.device)
        lambdas = torch.clamp(lambdas, min=0.0)
        zetas = torch.tensor([float(item["zeta"]) for item in params], dtype=torch.float32, device=self.device)
        zetas = torch.clamp(zetas, min=0.0, max=1.0).view(batch_size, 1, 1, 1)
        sigma_ks = self.sqrt_1m_alphas_cumprod / self.sqrt_alphas_cumprod
        rhos = lambdas[:, None] / torch.clamp(sigma_ks[None, :] ** 2, min=1e-12)

        if self.noise_init_img.lower() == "max":
            t_start = self.num_train_timesteps - 1
        else:
            t_start = int(self._utils_model.find_nearest(self.reduced_alpha_cumprod, 2.0 * float(self.noise_init_img) / 255.0))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(_normalize_seed(seed))
        init_noise = self._sample_shared_noise((1,) + tuple(y.shape[1:]), generator=generator).expand(batch_size, -1, -1, -1)
        x = self.sqrt_alphas_cumprod[t_start] * y_condition + self.sqrt_1m_alphas_cumprod[t_start] * init_noise

        with torch.no_grad():
            for step_index, seq_index in enumerate(self.seq):
                curr_sigma = float(self.sigmas[int(seq_index)].detach().cpu().item())
                t_i = int(self._utils_model.find_nearest(self.reduced_alpha_cumprod, curr_sigma))
                if t_i > t_start:
                    continue

                x0 = self._utils_model.model_fn(
                    x,
                    noise_level=curr_sigma * 255.0,
                    model_out_type="pred_xstart",
                    model_diffusion=self.model,
                    diffusion=self.diffusion,
                    ddim_sample=self.ddim_sample,
                    alphas_cumprod=self.alphas_cumprod_cpu,
                )
                if step_index == len(self.seq) - 1:
                    x = x0
                    break

                tau = rhos[:, t_i].view(batch_size, 1, 1, 1)
                x0_projected = (y_condition + tau * x0) / (1.0 + tau)
                x0 = x0 + self.guidance_scale * (x0_projected - x0)

                next_sigma = float(self.sigmas[int(self.seq[step_index + 1])].detach().cpu().item())
                t_im1 = int(self._utils_model.find_nearest(self.reduced_alpha_cumprod, next_sigma))
                eps = (x - self.sqrt_alphas_cumprod[t_i] * x0) / self.sqrt_1m_alphas_cumprod[t_i]
                eta_sigma = (
                    self.eta
                    * self.sqrt_1m_alphas_cumprod[t_im1]
                    / self.sqrt_1m_alphas_cumprod[t_i]
                    * torch.sqrt(self.betas[t_i])
                )
                shared_eps_noise = self._sample_shared_noise((1,) + tuple(y.shape[1:]), generator=generator).expand(
                    batch_size, -1, -1, -1
                )
                shared_zeta_noise = self._sample_shared_noise((1,) + tuple(y.shape[1:]), generator=generator).expand(
                    batch_size, -1, -1, -1
                )
                residual_scale = torch.sqrt(
                    torch.clamp(
                        self.sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma**2,
                        min=0.0,
                    )
                )
                x = self.sqrt_alphas_cumprod[t_im1] * x0
                x = x + torch.sqrt(torch.clamp(1.0 - zetas, min=0.0)) * (
                    residual_scale * eps + eta_sigma * shared_eps_noise
                )
                x = x + torch.sqrt(zetas) * self.sqrt_1m_alphas_cumprod[t_im1] * shared_zeta_noise

        return torch.clamp(x / 2.0 + 0.5, min=0.0, max=1.0)

    def denoise(
        self,
        *,
        noisy_chw: np.ndarray,
        unified_lambda: float,
        zeta: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Restore one RGB image without a clean target or noise-level input.

        ``unified_lambda`` already includes the original DiffPIR sigma-squared
        factor. This uses exactly the candidate path used for oracle evaluation.
        """
        restored = self._run_candidate_batch(
            noisy_chw=noisy_chw,
            params=[{"lambda": float(unified_lambda), "zeta": float(zeta)}],
            seed=int(seed),
        )
        return restored[0].detach().cpu().numpy()

    def evaluate_candidate_scores(
        self,
        *,
        clean_chw: np.ndarray,
        noisy_chw: np.ndarray,
        params: Sequence[Dict[str, float]],
        metric_names: Sequence[str] = ("PSNR", "SSIM"),
        seed: int = 0,
        candidate_batch_size: int = 4,
    ) -> List[DiffPIRScoreResult]:
        if not params:
            return []
        batch_limit = max(1, int(candidate_batch_size))
        results: List[DiffPIRScoreResult] = []
        for start in range(0, len(params), batch_limit):
            batch_params = [dict(item) for item in params[start : start + batch_limit]]
            restored_batch = self._run_candidate_batch(
                noisy_chw=noisy_chw,
                params=batch_params,
                seed=int(seed),
            )
            batch_scores = _score_restored_batch(
                np.asarray(clean_chw, dtype=np.float32),
                restored_batch,
                metric_names,
            )
            for param, scores in zip(batch_params, batch_scores):
                results.append(DiffPIRScoreResult(param=dict(param), scores=dict(scores)))
        return results
