from __future__ import annotations

import os


class _NullSummaryWriter:
    def __init__(self, *args, **kwargs):
        self.log_dir = kwargs.get("log_dir")

    def add_scalar(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def close(self):
        return None


if os.getenv("ENABLE_TENSORBOARD", "").strip().lower() in {"1", "true", "yes", "on"}:
    try:
        from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
    except Exception as exc:
        print(f"Warning: tensorboard SummaryWriter unavailable, disabling logs. ({exc})")
        SummaryWriter = _NullSummaryWriter
    else:
        SummaryWriter = _TorchSummaryWriter
else:
    SummaryWriter = _NullSummaryWriter
