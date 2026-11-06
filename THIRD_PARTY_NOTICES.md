# Third-party notices

HyperDn project code uses the [MIT license](LICENSE). Third-party software,
datasets, and pretrained weights retain their own terms.

## Included predictor weights

The included weights derive from **ConvNeXt V2: Co-designing and Scaling ConvNets
with Masked Autoencoders**, Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen,
Zhuang Liu, In So Kweon, and Saining Xie (2023), KAIST, Meta AI, and New York
University. The backbone is `convnextv2_base.fcmae_ft_in22k_in1k` through timm.

The [upstream project](https://github.com/facebookresearch/ConvNeXt-V2) releases
its code under MIT and pretrained weights under **CC BY-NC 4.0**. The included
checkpoints contain weights modified by HyperDn source training and, for target
models, few-shot adaptation. They retain the upstream noncommercial weight terms;
they are not covered solely by the HyperDn code license. The complete upstream
license is retained in [THIRD_PARTY_LICENSE.txt](THIRD_PARTY_LICENSE.txt).
See also [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
No endorsement by the upstream authors is implied.

## External dependencies

- [DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR) is MIT-licensed and obtained
  separately. Its ImageNet prior comes from
  [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion).
  Diffusion-prior weights are not included in this repository.
- [bm3d 4.0.3](https://pypi.org/project/bm3d/4.0.3/), used for CBM3D, is available
  for noncommercial use under its upstream license. It and its `bm4d` dependency
  are installed separately.
- PyTorch, torchvision, timm, and the remaining Python dependencies are installed
  separately under their upstream licenses.

## Data

STL10 and WED images are not redistributed. Obtain them from their original
sources and follow their terms. The split manifest contains filenames only.
