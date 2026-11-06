# Release provenance

The initial official repository uses the selected seed43 NeurIPS supplementary code package.

| Original supplementary archive | Value |
|---|---|
| Filename | `neurips_2026_camera_ready_supplement.zip` |
| Size | 47,220,127 bytes |
| Files | 62 |
| SHA-256 | `ad5d8a68f61e02eb0b85a0909fc63094282fe8db5d750ea0a16bdd7d58419203` |

All original Python source files, training configurations, checkpoint tensors, oracle-label banks, dependency lists, data instructions, and license files are preserved byte-for-byte. The repository updates `README.md` and the release inventory/checksums, and adds citation metadata, the paper architecture figure, a synthetic inference example, this provenance note, and a CPU-check workflow.

Download the code, checkpoints, and label banks through the repository or GitHub's source archives. The archive hash above identifies the original conference supplementary package used for comparison.

## Included experimental scope

- Source pretraining using the 13 supplied source label banks.
- DiffPIR WED/PSNR and CBM3D few-shot training and evaluation, with HyperDn, ImageNet initialization, and Afkham controls.
- Source, DiffPIR two-shot, and CBM3D two-shot predictor checkpoints, all seed43.
- Seventeen oracle-label banks and four target-oracle generation recipes.
- Parameter prediction and denoising interfaces.

The default runs use seed43. Other seeds can be selected in the experiment entry points, with matching source checkpoints. Raw images and the external DiffPIR solver/prior are obtained using the README and data instructions.

## Synthetic example

`examples/noisy.png` is a generated 256×256 RGB test input with Gaussian noise. It provides an immediately runnable inference example without redistributing dataset images. It is not used in the paper's experimental results.
