"""CBM3D few-shot: three arms, seven budgets; seed 43 by default.

CBM3D_SEEDS=42,43,44,45,46 selects all paper seeds after source pretraining.
CBM3D_SHOTS and CBM3D_METHODS select cells. CBM3D_OUTPUT_ROOT separates runs.
CBM3D_ONLY_EVAL=1 uses saved checkpoints (including the bundled 2-shot model).
CBM3D_RANK / CBM3D_WORLD_SIZE partition cells across workers.
"""
from fewshot_protocol import main

if __name__ == "__main__":
    main("cbm3d")
