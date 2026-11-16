"""Few-shot transfer to CBM3D (predicting sigma_psd without the noise level).

Trains and evaluates HyperDn, ImageNet initialization and Afkham et al. with
1, 2, 4, 16, 32, 64 and 128 target labels, seed 43 by default.

  CBM3D_SEEDS=42,43,44,45,46   all five paper seeds (needs a source model per seed)
  CBM3D_METHODS, CBM3D_SHOTS   run a subset, e.g. CBM3D_METHODS=ours CBM3D_SHOTS=2
  CBM3D_OUTPUT_ROOT            output directory (default experiment_runs/fewshot_cbm3d)
  CBM3D_ONLY_EVAL=1            evaluate saved checkpoints, including the bundled 2-label model
  CBM3D_RANK, CBM3D_WORLD_SIZE split the runs across workers
"""
from fewshot_protocol import main

if __name__ == "__main__":
    main("cbm3d")
