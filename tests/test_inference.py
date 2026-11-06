"""Deployment contracts: no noise-level input, fixed zeta, and executor parity."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from infer import solver_parameters
from restoration.diffpir_official import OfficialDiffPIRDenoiser


def small_executor():
    # Exercise the real schedule/reconstruction code with a cheap deterministic
    # prior; downloading the 2 GB external prior is unnecessary for this test.
    denoiser = OfficialDiffPIRDenoiser.__new__(OfficialDiffPIRDenoiser)
    denoiser.device = torch.device("cpu")
    denoiser.iter_num = 4
    denoiser.num_train_timesteps = 1000
    denoiser.beta_start = 1e-4
    denoiser.beta_end = 2e-2
    denoiser.skip_type = "quad"
    denoiser.noise_init_img = "max"
    denoiser.guidance_scale = 1.0
    denoiser.eta = 0.0
    denoiser.ddim_sample = False
    denoiser.model = None
    denoiser.diffusion = None
    denoiser._utils_model = SimpleNamespace(
        find_nearest=lambda values, value: np.abs(values - value).argmin(),
        model_fn=lambda image, **kwargs: image * 0.75,
    )
    denoiser._build_schedule()
    return denoiser


class InferenceContracts(unittest.TestCase):
    def test_diffpir_unified_lambda_passes_unchanged_and_fixes_zeta(self):
        self.assertEqual(solver_parameters(0.025, "diffpir"), {"lambda": 0.025, "zeta": 1.0})

    def test_cbm3d_uses_paper_executor_limits(self):
        self.assertEqual(solver_parameters(0.0001, "cbm3d"), {"sigma_psd": 0.003})
        self.assertEqual(solver_parameters(0.6, "cbm3d"), {"sigma_psd": 0.3})
        self.assertEqual(solver_parameters(0.04, "cbm3d"), {"sigma_psd": 0.04})

    def test_diffpir_deployment_matches_evaluation_and_repeats_seed(self):
        denoiser = small_executor()
        noisy = np.random.default_rng(8).random((3, 16, 16), dtype=np.float32)
        parameters = [{"lambda": 0.025, "zeta": 1.0}, {"lambda": 0.08, "zeta": 1.0}]
        batch = denoiser._run_candidate_batch(noisy_chw=noisy, params=parameters, seed=42)
        restored = denoiser.denoise(noisy_chw=noisy, unified_lambda=0.025, seed=42)
        np.testing.assert_array_equal(restored, batch[0].numpy())
        np.testing.assert_array_equal(
            restored, denoiser.denoise(noisy_chw=noisy, unified_lambda=0.025, seed=42)
        )
        self.assertFalse(np.array_equal(restored, denoiser.denoise(noisy_chw=noisy, unified_lambda=0.025, seed=43)))
        clean = np.full_like(noisy, 0.5)
        scores = denoiser.evaluate_candidate_scores(
            clean_chw=clean, noisy_chw=noisy, params=parameters,
            metric_names=("PSNR",), seed=42,
        )
        expected = 10.0 * np.log10(1.0 / np.mean((clean - restored) ** 2))
        self.assertAlmostEqual(scores[0].scores["PSNR"], expected, places=5)
        self.assertEqual(scores[0].param, parameters[0])


if __name__ == "__main__":
    unittest.main()
