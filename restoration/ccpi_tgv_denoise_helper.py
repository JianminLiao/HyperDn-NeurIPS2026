from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ccpi.filters.regularisers import TGV


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CCPi TGV denoising on a saved CHW array.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda", dest="lambda_value", required=True, type=float)
    parser.add_argument("--alpha1", required=True, type=float)
    parser.add_argument("--alpha0", required=True, type=float)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--eps", required=True, type=float)
    parser.add_argument("--l2", required=True, type=float)
    return parser.parse_args()


def _run_channel(
    channel: np.ndarray,
    lambda_value: float,
    alpha1: float,
    alpha0: float,
    iterations: int,
    l2: float,
    eps: float,
) -> np.ndarray:
    restored = TGV(
        np.asarray(channel, dtype=np.float32),
        float(lambda_value),
        float(alpha1),
        float(alpha0),
        int(iterations),
        float(l2),
        float(eps),
    )
    return np.clip(np.asarray(restored, dtype=np.float32), 0.0, 1.0)


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    noisy = np.asarray(np.load(input_path), dtype=np.float32)
    if noisy.ndim == 2:
        noisy = noisy[np.newaxis, ...]
    restored = np.empty_like(noisy, dtype=np.float32)
    for channel_idx in range(noisy.shape[0]):
        restored[channel_idx] = _run_channel(
            noisy[channel_idx],
            lambda_value=args.lambda_value,
            alpha1=args.alpha1,
            alpha0=args.alpha0,
            iterations=args.iterations,
            l2=args.l2,
            eps=args.eps,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, restored)


if __name__ == "__main__":
    main()
