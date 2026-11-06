# Data and oracle labels

Run these commands from the repository root. Images and the external DiffPIR prior are downloaded separately.

## Label banks

Unpack the supplied 17 label banks:

```sh
python scripts/prepare_labels.py
```

The script checks the archive and every extracted file against `configs/label_banks.json`. The 13 source banks contain 119,444 records. DiffPIR uses 4,344 training and 400 validation records; evaluation uses the first 50 validation records. CBM3D uses 128 training and 50 validation records. Labels include the image indices, noise parameters, and noise seeds used by the data loaders. The archive contains no images.

## Images

Download the official [STL-10](https://cs.stanford.edu/~acoates/stl10/) training split:

```sh
python -c "from torchvision.datasets import STL10; STL10(root='data', split='train', download=True)"
```

Use its 5,000 RGB images at their original 96×96 resolution and in binary-file order. The four `s4` source banks contain four noise realizations per image; the NLL–TGV bank uses the first 100 images.

Download the 4,744 pristine BMP images from the [Waterloo Exploration Database](https://kedema.org/project/exploration/index.html) into `data/WED/`, retaining their filenames. Create the 256×256 center crops:

```sh
python scripts/generate_wed_256.py --source-root data/WED --output-root data/WED_256
```

`configs/wed_split.json` fixes the 4,344 training and 400 validation filenames. The loaders read each split in lexicographic filename order; keep this order because labels refer to image indices. CBM3D uses the first 128 training and first 50 validation images. The WED source bank and DiffPIR training bank use the training split.

```text
data/
  stl10_binary/train_X.bin
  stl10_binary/train_y.bin
  WED/<original filename>.bmp
  WED_256/train/<original filename>.png
  WED_256/val/<original filename>.png
dataset_labels/
  <files listed in configs/label_banks.json>
```

## Target oracle generation

The supplied banks can be used directly for training. Optional DiffPIR and CBM3D label searches are configured in `configs/oracles.json`. For DiffPIR, download the official code and ImageNet prior using the [README](../README.md), then set:

```sh
export DIFFPIR_REPO_ROOT="$PWD/external/DiffPIR_official"
export DIFFPIR_MODEL_PATH="$DIFFPIR_REPO_ROOT/model_zoo/256x256_diffusion_uncond.pt"
```

Preview commands, then add `--execute` to run them:

```sh
python scripts/generate_oracles.py --group diffpir
python scripts/generate_oracles.py --group cbm3d
python scripts/generate_oracles.py --group diffpir --execute
python scripts/generate_oracles.py --group cbm3d --execute
```

Outputs go to `dataset_labels_regenerated/`. To train on newly generated target banks, add `--output-root dataset_labels` in a fresh checkout where the supplied banks have not been unpacked.

Both targets use one Gaussian observation per image, sigma uniformly sampled from `[5/255,50/255]`, and seed 20260331. Clean RGB images are in `[0,1]`; noisy observations are not clipped.

DiffPIR uses 100 iterations, fixed zeta=1, and lambda-cache in `[2,20]`: eight logarithmic coarse points followed by two eight-point refinements. The predictor's training target is `log10(lambda_cache * sigma²)`; the bank stores lambda-cache.

CBM3D requires `bm3d==4.0.3`. It searches sigma_psd in `[0.003,0.3]` with 16 logarithmic coarse points and two 16-point refinements, then merges the output automatically. Its target has the UNKNOWN parameter role and is encoded as `log10(sigma_psd)`.
