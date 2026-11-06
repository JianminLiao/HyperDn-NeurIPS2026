"""CPU checks for the training objective, recorded protocols and strict loading."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from fewshot_protocol import afkham_env, foundation_env, protocol_summary, write_reports
from pgccpi_checkpoint_utils import load_checkpoint_payload, load_model_checkpoint_forgiving
from pgccpi_foundation_dataset import PGCCPiFoundationDataset, foundation_collate_fn
from run_diffpir_transfer_experiments import preserve_contract, environment_contract, train_run, eval_run
from train_pgccpi_foundation import _compute_losses, _subset_indices, _build_llrd_optimizer, DatasetHomogeneousBatchSampler
from evaluate_pgccpi_foundation import _build_model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class TrainingContractTests(unittest.TestCase):
    def test_baseline_effective_schedule(self):
        self.assertEqual(afkham_env(2, 43)["EPOCHS"], "500")
        self.assertEqual(afkham_env(64, 43)["EPOCHS"], "100")
        self.assertEqual(foundation_env(2, 43)["EPOCHS"], "150")
        for target in ["diffpir", "cbm3d"]:
            self.assertTrue(protocol_summary(target)["afkham_cuda_amp"])
            self.assertEqual(protocol_summary(target)["afkham_warmup_epochs"], 1)
        expected = np.arange(128)
        np.random.default_rng(43).shuffle(expected)
        self.assertEqual(_subset_indices(128, 2, 43), expected[:2].tolist())
        self.assertEqual(_subset_indices(128, 4, 43)[:2], expected[:2].tolist())

    def test_source_batches_are_full_without_resampling(self):
        # Three sources: one remainder, two complete batches, and one source
        # smaller than a batch. No duplicated records or cross-source mixing.
        source_ids = [0] * 5 + [1] * 8 + [2] * 3
        for shuffle in [False, True]:
            sampler = DatasetHomogeneousBatchSampler(source_ids, batch_size=4, seed=43, shuffle=shuffle)
            batches = list(sampler)
            self.assertEqual(len(batches), 3)
            self.assertEqual(len(sampler), 3)
            selected = [index for batch in batches for index in batch]
            self.assertEqual(len(selected), len(set(selected)))
            self.assertTrue(all(len(batch) == 4 for batch in batches))
            self.assertTrue(all(len({source_ids[index] for index in batch}) == 1 for batch in batches))
            self.assertTrue(all(source_ids[index] != 2 for index in selected))
            self.assertEqual(sum(source_ids[index] == 0 for index in selected), 4)
            self.assertEqual(sum(source_ids[index] == 1 for index in selected), 8)

    def test_reuse_requires_same_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            output = Path(directory) / "checkpoint.pt"
            preserve_contract(path, {"epochs": 150}, [output])
            output.touch()
            preserve_contract(path, {"epochs": 150}, [output])
            with self.assertRaises(RuntimeError):
                preserve_contract(path, {"epochs": 100}, [output])
            with self.assertRaises(RuntimeError):
                preserve_contract(path.with_name("missing.json"), {}, [output])

    def test_inherited_settings_and_partial_training(self):
        before = environment_contract({"CBM3D_SIGMA_PSD_MAX": "0.30", "DROPOUT": "0.1"})
        after = environment_contract({"CBM3D_SIGMA_PSD_MAX": "0.20", "DROPOUT": "0.2"})
        self.assertNotEqual(before, after)
        self.assertEqual(before["CBM3D_SIGMA_PSD_MAX"], "0.30")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bank = directory / "bank.pt"
            bank.write_bytes(b"small fixture")
            checkpoint = directory / "partial_best.pth"
            def interrupted(*args, **kwargs):
                checkpoint.write_bytes(b"partial epoch")
                raise RuntimeError("simulated interrupted source training")
            kwargs = dict(train_sources=[{"dataset_name": "WED_256", "ground_truth_path": str(bank), "split": "train"}],
                          run_name="partial", checkpoint_dir=directory, runs_dir=directory / "runs", extra_env={"EPOCHS": "80"})
            with patch("run_diffpir_transfer_experiments.run_with_log", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    train_run(**kwargs)
                with self.assertRaisesRegex(RuntimeError, "partial best"):
                    train_run(**kwargs)

    def test_table_standard_deviations(self):
        rows = [{"method": "ours", "shot": 2, "seed": i,
                 "recon_psnr_mean": value} for i, value in enumerate([20.0, 22.0])]
        with tempfile.TemporaryDirectory() as directory:
            for target, expected in [("diffpir", 2 ** 0.5), ("cbm3d", 1.0)]:
                write_reports(Path(directory), rows, target)
                result = json.loads((Path(directory) / "aggregated.json").read_text())[0]
                self.assertAlmostEqual(result["psnr_std"], expected)

    def test_two_slot_labels_checkpoint_and_backward(self):
        torch.set_num_threads(2)
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            images = directory / "data/WED_256/train"
            images.mkdir(parents=True)
            Image.fromarray(np.full((32, 32, 3), 127, np.uint8)).save(images / "image.png")
            bank = directory / "diffpir.pt"
            torch.save({"config": {"regularizer": "DIFFPIR", "data_term": "L2", "noise_model": "GAUSSIAN", "operator_kind": "IDENTITY"}, "records": [{
                "image_index": 0, "noise_seed": 20260331, "noise_sigma": 0.05,
                "noise_params": {"sigma": 0.05}, "target_vector": [-2.0, 1.0],
                "target_dim": 2, "best_score": 30.0, "optimal_param": {"lambda": 4.0, "zeta": 1.0},
            }]}, bank)
            try:
                os.chdir(directory)
                data = PGCCPiFoundationDataset(dataset_name="WED_256", ground_truth_path=str(bank))
                first = data[0]
                batch = foundation_collate_fn([first, first])
            finally:
                os.chdir(original_cwd)
        self.assertEqual(batch["target_param_mask"][0, :2].tolist(), [1.0, 1.0])
        self.assertEqual(float(batch["target_param"][0, 1]), 0.0)
        checkpoint = load_checkpoint_payload(str(ROOT / "checkpoints/hyperdn_source_s43.pt"))
        model = _build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        broken = dict(checkpoint["model_state_dict"])
        broken.pop(next(iter(broken)))
        with self.assertRaises(RuntimeError):
            load_model_checkpoint_forgiving(model, broken)
        model.train()
        outputs = model(batch["noisy_raw"], regularizer_ids=batch["regularizer_id"],
                        data_term_ids=batch["data_term_id"], noise_attributes=batch["noise_attributes"],
                        observation_meta=batch["observation_meta"])
        loss = _compute_losses(outputs, batch, model)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.h_out.weight.grad).all())
        self.assertGreater(float(model.h_out.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(model.role_log_var.grad).all())
        optimizer = _build_llrd_optimizer(model)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            result = model(batch["noisy_raw"], regularizer_ids=batch["regularizer_id"],
                           data_term_ids=batch["data_term_id"], noise_attributes=batch["noise_attributes"],
                           observation_meta=batch["observation_meta"])["param_pred"]
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
