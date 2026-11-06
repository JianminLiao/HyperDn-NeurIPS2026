# HyperDn

### Oracle Supervision Transfers for Hyperparameter Prediction in Model-Based Image Denoising

**NeurIPS 2026** · Jianmin Liao, Lixin Shen, and Yuesheng Xu

[Paper](https://openreview.net/forum?id=vXGWG7Z20k) · [Quick start](#quick-start) · [Training](#training-and-few-shot-transfer) · [Data](docs/data.md) · [Citation](#citation)

**Reuse oracle supervision from related denoising problems to predict hyperparameters with fewer target labels.**

HyperDn takes a noisy image and information about the denoising task, then predicts its hyperparameters. It pools oracle supervision across TV/TGV source configurations and transfers the learned predictor to new denoisers. This release includes source pretraining, DiffPIR WED/PSNR and CBM3D few-shot training and evaluation, both comparison methods, and pretrained seed43 models.

![HyperDn architecture: image and task information feed a shared hyperparameter predictor.](docs/figures/architecture.png)

## Quick start

Use Python 3.12 and run commands from the repository root.

```bash
git clone https://github.com/JianminLiao/HyperDn-NeurIPS2026.git
cd HyperDn-NeurIPS2026
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU inference is supported. CUDA is recommended for training and DiffPIR.

Predict a CBM3D parameter and denoise the included synthetic example on CPU:

```bash
python infer.py --algorithm cbm3d --image examples/noisy.png \
  --output outputs/cbm3d.png --json outputs/cbm3d.json
```

The command prints the predicted parameter and saves the restored image. The example is a generated test input, not a paper benchmark. Replace `examples/noisy.png` with your own noisy RGB image.

Predict the DiffPIR parameter without downloading its diffusion prior:

```bash
python infer.py --algorithm diffpir --image examples/noisy.png
```

The predictor uses the Gaussian noise category, with no noise-level input or clean reference image.

## Pretrained models

All three models use seed43. The target models use two training labels.
The checkpoints are included in the repository; their provenance and hashes are recorded in [weight_manifest.json](weight_manifest.json).

| Model | Checkpoint |
|---|---|
| Source pretrained | `checkpoints/hyperdn_source_s43.pt` |
| DiffPIR, WED/PSNR | `checkpoints/hyperdn_diffpir_2shot_s43.pt` |
| CBM3D | `checkpoints/hyperdn_cbm3d_2shot_s43.pt` |

## Inference

Predict a parameter from a noisy RGB image:

```bash
python infer.py --algorithm diffpir --image noisy.png
python infer.py --algorithm cbm3d --image noisy.png
```

Add `--output` to save the denoised image and `--json` to save its parameters:

```bash
python infer.py --algorithm cbm3d --image noisy.png --output restored.png --json parameters.json
```

Use `--checkpoint PATH` for another model. DiffPIR outputs the effective coefficient
`lambda_eff = lambda_cache * sigma^2`, which is passed directly to the executor;
`zeta=1`. CBM3D outputs `sigma_psd` in `[0,1]` image units, clipped to `[0.003,0.30]`.

For DiffPIR denoising, install the external solver and download its prior:

```bash
git clone https://github.com/yuanzhi-zhu/DiffPIR.git external/DiffPIR_official
git -C external/DiffPIR_official checkout 2a9898129a1b274131b98746e5b364bc20adc1e1
mkdir -p external/DiffPIR_official/model_zoo
curl -fL https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt \
  -o external/DiffPIR_official/model_zoo/256x256_diffusion_uncond.pt
python infer.py --algorithm diffpir --image noisy256.png --device cuda --output restored.png
```

The prior is about 2.1 GiB; SHA-256:
`a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1`.
DiffPIR uses 256×256 RGB inputs and 100 iterations. Optional flags:
`--diffpir-root`, `--diffpir-prior`, `--diffpir-device`, `--seed`.

## Training and few-shot transfer

| Task | Entry point | Inputs |
|---|---|---|
| Source pretraining | `run_fullcorpus_kendall_5seed.py` | STL10/WED images and source label banks |
| DiffPIR WED/PSNR transfer | `run_expc_fullcorpus_kendall.py` | WED images, DiffPIR labels, source checkpoint, external solver/prior |
| CBM3D transfer | `run_expc_cbm3d_unknown.py` | WED images, CBM3D labels, source checkpoint |
| Optional target oracle search | `scripts/generate_oracles.py` | Images and the corresponding denoiser; see [data instructions](docs/data.md#target-oracle-generation) |

Prepare the STL-10/WED images using [docs/data.md](docs/data.md), then extract the
included oracle labels:

```bash
python scripts/prepare_labels.py
```

Source pretraining reads the supplied labels. To train the source model:

```bash
python run_fullcorpus_kendall_5seed.py
```

Source training runs for 80 epochs with configuration-homogeneous batches of 128;
incomplete batches are dropped. The released source checkpoint can also be used
directly for target adaptation.

Run the DiffPIR WED/PSNR and CBM3D experiments:

```bash
python run_expc_fullcorpus_kendall.py
python run_expc_cbm3d_unknown.py
```

Both default to seed43, methods `ours,imagenet_only,afkham`, and label budgets
`1,2,4,16,32,64,128`. To run only two-shot HyperDn:

```bash
EXPC_METHODS=ours EXPC_SHOTS=2 python run_expc_fullcorpus_kendall.py
CBM3D_METHODS=ours CBM3D_SHOTS=2 python run_expc_cbm3d_unknown.py
```

Set `EXPC_ONLY_EVAL=1` or `CBM3D_ONLY_EVAL=1` with these commands to evaluate the
released two-shot checkpoints. Select other seeds with `FC_SEEDS`, `EXPC_SEEDS`,
or `CBM3D_SEEDS` (comma-separated); adaptation requires matching source models.
Set `FC_OUTPUT_ROOT`, `EXPC_OUTPUT_ROOT`, or `CBM3D_OUTPUT_ROOT` to change output
locations. Use a new output directory when changing training settings.

Despite its filename, `run_fullcorpus_kendall_5seed.py` also defaults to seed43. For example, set `FC_SEEDS=42,43,44,45,46` to train five source models.

HyperDn and ImageNet init use 150 target epochs and the final checkpoint. Afkham
uses 500 epochs for budgets 1–2 and 100 otherwise. Evaluation uses 50 WED
validation images. DiffPIR summaries use sample SD; CBM3D uses population SD.

## Checks

```bash
python -m unittest discover -s tests -v
python tools/validate_release.py
```

These commands run the existing CPU tests and verify the release files, checkpoint loading, and oracle-label checksums. [release_manifest.json](release_manifest.json) lists the files; [SHA256SUMS](SHA256SUMS) records their hashes.

The Python code, configurations, label banks, and weights are identical to the selected NeurIPS supplementary package. This repository adds the paper homepage, citation metadata, architecture figure, synthetic example, and automated checks. See [release provenance](docs/release.md).

## Citation

```bibtex
@inproceedings{liao2026oracle,
  title = {Oracle Supervision Transfers for Hyperparameter Prediction in Model-Based Image Denoising},
  author = {Liao, Jianmin and Shen, Lixin and Xu, Yuesheng},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://openreview.net/forum?id=vXGWG7Z20k}
}
```

## License and acknowledgments

Code: [MIT](LICENSE). Pretrained weights: CC BY-NC 4.0.
See [third-party notices](THIRD_PARTY_NOTICES.md).

We thank the authors of [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2), [timm](https://github.com/huggingface/pytorch-image-models), [DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR), and the CBM3D software for their released tools and models.

For questions about the implementation, please open a GitHub issue.
