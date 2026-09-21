"""The decision model backends.

A backend is anything that can answer typed questions about a state:

    answer(state, questions) -> {"answers": {...}, "usage": {...}}

Two ship here:

* `laya_mlx`   — Apple Silicon (MLX). Fastest on a Mac; `pip install laya-mlx`.
* `laya_torch` — any platform with PyTorch; `pip install laya`.

Both run the same open-weight Laya checkpoints, so a decision made on a Mac and a
decision made on a Linux box are the same decision.
"""

from .base import Backend, BackendError, resolve_backend

__all__ = ["Backend", "BackendError", "resolve_backend"]
