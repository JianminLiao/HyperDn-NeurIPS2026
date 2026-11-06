"""WED/PSNR DiffPIR: three arms, seven budgets; seed 43 by default.

EXPC_SEEDS=42,43,44,45,46 selects all paper seeds after source pretraining.
EXPC_SHOTS and EXPC_METHODS select cells. EXPC_OUTPUT_ROOT separates runs.
EXPC_ONLY_EVAL=1 uses saved checkpoints (including the bundled 2-shot model).
"""
from fewshot_protocol import main

if __name__ == "__main__":
    main("diffpir")
