
import numpy as np
import torch
from abc import ABC, abstractmethod
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def to_numpy(x):
    """Convert Tensor (CPU/CUDA) or Array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def ensure_4d(x):
    """Ensure tensor is (N, C, H, W) for torchmetrics."""
    if x.ndim == 3:
        return x.unsqueeze(0) # (C,H,W) -> (1,C,H,W)
    return x

class AbstractMetric(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass
    @abstractmethod
    def calculate(self, clean, denoised) -> float:
        pass

class SSIMMetric(AbstractMetric):
    def get_name(self) -> str:
        return "SSIM"
    def calculate(self, clean, denoised):
        if isinstance(clean, torch.Tensor):
            from torchmetrics.functional.image import structural_similarity_index_measure
            # GPU/Torch Path
            clean = ensure_4d(clean)
            denoised = ensure_4d(denoised)
            # reduction='none' returns [B]
            val = structural_similarity_index_measure(denoised, clean, data_range=1.0, reduction='none')
            return val.detach() # Keep on device
        
        # CPU/Numpy Path
        clean = to_numpy(clean)
        denoised = to_numpy(denoised)
        return ssim(clean, denoised, data_range=1.0, channel_axis=0)

class PSNRMetric(AbstractMetric):
    def get_name(self) -> str:
        return "PSNR"
    def calculate(self, clean, denoised):
        if isinstance(clean, torch.Tensor):
            clean = ensure_4d(clean)
            denoised = ensure_4d(denoised)
            # data_range=1.0
            # PSNR = 10 * log10(MAX^2 / MSE)
            # We calculate MSE per sample [B]
            mse = torch.mean((clean - denoised) ** 2, dim=[1, 2, 3])
            mse = torch.clamp(mse, min=1e-8)
            psnr_val = 10 * torch.log10(1.0 / mse)
            return psnr_val.detach() # Keep on device
            
        clean = to_numpy(clean)
        denoised = to_numpy(denoised)
        
        # Handle exact match for numpy path
        mse_np = np.mean((clean - denoised) ** 2)
        if mse_np <= 1e-8:
            return float(10 * np.log10(1.0 / 1e-8))
            
        return psnr(clean, denoised, data_range=1.0)

class MAEMetric(AbstractMetric):
    def get_name(self) -> str:
        return "MAE"
    def calculate(self, clean, denoised):
        if isinstance(clean, torch.Tensor):
            clean = ensure_4d(clean)
            denoised = ensure_4d(denoised)
            # [B]
            err = torch.mean(torch.abs(clean - denoised), dim=[1, 2, 3])
            return -err.detach() # Keep on device

        clean = to_numpy(clean)
        denoised = to_numpy(denoised)
        err = np.mean(np.abs(clean - denoised))
        return -err


class LPIPSMetric(AbstractMetric):
    """NEGATED LPIPS (lpips package, net='vgg'), following the MAEMetric
    convention so that every metric in this module is higher-is-better and
    selection logic (`score > best`) needs no lower-is-better branch.

    Stored scores are -LPIPS; argmax(-LPIPS) = argmin(LPIPS). NOTE: the
    DiffPIR sweep path (find_optimal_diffpir_denoise.py) stores POSITIVE
    LPIPS with its own lower-is-better handling -- do not mix the two score
    conventions within one bundle.

    Inputs are CHW (numpy or torch) in [0, 1]; rescaled to [-1, 1] as the
    lpips package expects. Model is lazily built and cached per (net, device).
    """

    _models = {}

    def __init__(self, net: str = "vgg", device: str = "cpu"):
        self._net = str(net)
        self._device = torch.device(device)

    def get_name(self) -> str:
        return "LPIPS"

    def _model(self):
        key = (self._net, str(self._device))
        model = LPIPSMetric._models.get(key)
        if model is None:
            import lpips  # deferred: only needed when LPIPS is requested
            model = lpips.LPIPS(net=self._net, verbose=False).to(self._device)
            model.eval()
            LPIPSMetric._models[key] = model
        return model

    def _prepare(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.ascontiguousarray(to_numpy(x)))
        x = x.float()
        x = ensure_4d(x)
        return (x * 2.0 - 1.0).to(self._device)

    def calculate(self, clean, denoised):
        was_tensor = isinstance(clean, torch.Tensor)
        model = self._model()
        with torch.no_grad():
            dist = model(self._prepare(clean), self._prepare(denoised))
        dist = dist.reshape(-1)
        if not was_tensor or dist.numel() == 1:
            return float(-dist[0].item()) if dist.numel() == 1 else -dist.detach()
        return -dist.detach()  # torch batch path, [B], negated like MAE
