"""Few-shot transfer to DiffPIR on WED 256x256 with PSNR oracle labels.

Trains and evaluates HyperDn, ImageNet initialization and Afkham et al. with
1, 2, 4, 16, 32, 64 and 128 target labels, seed 43 by default.

  DIFFPIR_SEEDS=42,43,44,45,46     all five paper seeds (needs a source model per seed)
  DIFFPIR_METHODS, DIFFPIR_SHOTS   run a subset, e.g. DIFFPIR_METHODS=ours DIFFPIR_SHOTS=2
  DIFFPIR_OUTPUT_ROOT              output directory (default experiment_runs/fewshot_diffpir)
  DIFFPIR_ONLY_EVAL=1              evaluate saved checkpoints, including the bundled 2-label model
  DIFFPIR_RANK, DIFFPIR_WORLD_SIZE split the runs across workers
"""
from fewshot_protocol import main

if __name__ == "__main__":
    main("diffpir")
